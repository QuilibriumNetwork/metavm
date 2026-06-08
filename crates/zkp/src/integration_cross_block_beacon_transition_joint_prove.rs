//! Cross-AIR joint-prove integration test composing
//! [`crate::multi_block_proof_air`] + [`crate::beacon_state_transition_air`] +
//! [`crate::casper_ffg_chain_air`] (task #244).
//!
//! This integration weaves the **Layer-A execution-block chain** through the
//! **beacon state-transition chain** (slot-level state-root evolution) and
//! up into the **Casper FFG justification / finalization chain**. The
//! witness is a 3-consecutive-block chain that straddles an epoch boundary
//! (slot 31 — the last slot of epoch 0) so that:
//!
//!   * `multi_block_proof_air` enforces `parent_hash(ω·X) = block_hash(X)`
//!     and `block_number(ω·X) = block_number(X) + 1` on the 3 blocks.
//!   * `beacon_state_transition_air` enforces `slot(ω·X) = slot(X) + 1`
//!     plus the 32-byte `prev_state_root(ω·X) = post_state_root(X)` chain,
//!     with one row whose `slot mod 32 == 31` flips `is_epoch_boundary`.
//!   * `casper_ffg_chain_air` enforces the 2/3-supermajority justification
//!     gate and the two-epoch finalization rule on
//!     `(source=0, target=1)`, `(source=1, target=2)`. Row 0 finalizes
//!     epoch 0 (its `target_epoch == 1`).
//!
//! ## Cross-AIR descriptors (single-column tuples — joint_prove v0)
//!
//! - **D0** — per-block index ↔ slot:
//!   `multi_block.COL_BLOCK_NUMBER` (gated by `IS_REAL`) ↔
//!   `state_transition.COL_SLOT` (gated by `IS_REAL`). Honest witness:
//!   both sides project the same `{slot0, slot0+1, slot0+2}` multiset.
//! - **D1** — block-hash first-byte binding:
//!   `multi_block.COL_BLOCK_HASH_OFFSET` ↔
//!   `state_transition.COL_BLOCK_ROOT_OFFSET`. Single-byte tuple gated by
//!   `IS_REAL` on both sides. Honest witness uses the same per-row 32-byte
//!   hash for `block_hash` and `block_root` (the EL block hash = the BBH
//!   block-root for post-merge slots), so the byte-0 projections match
//!   as a 3-element multiset.
//! - **D2** — finalized epoch ↔ epoch-boundary epoch:
//!   `casper_ffg.COL_TARGET_EPOCH` (gated by `IS_FINALIZED`) ↔
//!   `state_transition.COL_EPOCH` (gated by `IS_EPOCH_BOUNDARY`). Honest
//!   witness: FFG row 0 finalizes with `target_epoch = 1`; the lone
//!   epoch-boundary row (slot 31) has `epoch = 0` — so we instead build
//!   the multi-block chain at slots `(61, 62, 63)` where slot 63 is the
//!   epoch-1 boundary and `epoch = 1`, matching FFG row 0's
//!   `target_epoch = 1` (the canonical "epoch 0 finalized by epoch 1
//!   target" pattern).
//!
//! ## Topology
//!
//! - layer 0: `multi_block_proof_air` (3 active rows, 148 cols).
//! - layer 1: `beacon_state_transition_air` (3 active rows, 117 cols).
//! - layer 2: `casper_ffg_chain_air` (2 active rows, 32 cols).
//!
//! All three traces pad to the BLS48-581 minimum FFT width of 16.
//!
//! ## Tests
//!
//! - `descriptor_consistency` (fast) — descriptors are well-formed,
//!   witness shapes and on-row multisets match.
//! - `honest_round_trip` (#[ignore], slow) — full `joint_prove` +
//!   `joint_verify` round-trip; expects all 3 closures to match.
//! - `tampered_closure_rejected` (#[ignore], slow) — mutates one closure
//!   so `joint_verify`'s scalar equality check fires.

#[cfg(test)]
mod tests {
    use crate::beacon_state_transition_air as bst;
    use crate::casper_ffg_chain_air as ffg;
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::multi_block_proof_air as mb;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    /// First slot of the 3-block chain — chosen so that slot 63 is the
    /// last slot of epoch 1 (`63 mod 32 == 31`) and `epoch = 1`, which
    /// matches Casper FFG row 0's `target_epoch = 1` (the
    /// epoch-0-finalized-by-epoch-1-target classical FFG pattern).
    const FIRST_SLOT: u64 = 61;
    const CHAIN_LENGTH: usize = 3;

    fn h(seed: u8) -> [u8; mb::HASH_LEN] {
        let mut x = [0u8; mb::HASH_LEN];
        for k in 0..mb::HASH_LEN {
            x[k] = seed.wrapping_add(k as u8);
        }
        x
    }

    // ─── Witness builders ─────────────────────────────────────────────

    /// Build a 3-block multi-block witness rooted at a synthetic
    /// parent. `block_number[i] = FIRST_SLOT + i`. The block hashes
    /// `BH[i]` are seeded so that `BH[i].0` (first byte) is distinct
    /// per row; we then reuse `BH[i]` as the per-row `block_root` on
    /// the state-transition side so D1 closes.
    fn block_hashes() -> Vec<[u8; mb::HASH_LEN]> {
        (0..CHAIN_LENGTH).map(|i| h(0x10u8 + i as u8)).collect()
    }

    fn multi_block_witness() -> mb::MultiBlockProofWitness {
        let bhs = block_hashes();
        let mut out: Vec<(u64, [u8; mb::HASH_LEN], [u8; mb::HASH_LEN],
                          [u8; mb::HASH_LEN], [u8; mb::HASH_LEN])> =
            Vec::with_capacity(CHAIN_LENGTH);
        let mut parent = h(0xee);
        for i in 0..CHAIN_LENGTH {
            let sr = h(0x40u8 + i as u8);
            let psr = h(0x80u8 + i as u8);
            out.push((FIRST_SLOT + i as u64, bhs[i], parent, sr, psr));
            parent = bhs[i];
        }
        mb::MultiBlockProofWitness::from_chain(&out)
    }

    /// Build a 3-slot beacon state-transition witness at slots
    /// `(FIRST_SLOT, +1, +2)`. The lone epoch-boundary row is slot 63
    /// (slot_mod_32 == 31). Block roots reuse the multi-block chain's
    /// block hashes so descriptor D1's per-row byte-0 projection
    /// matches.
    fn state_transition_witness() -> bst::BeaconStateTransitionWitness {
        let bhs = block_hashes();
        let initial = h(0xc0);
        let mut blocks: Vec<(u64, [u8; 32], [u8; 32])> =
            Vec::with_capacity(CHAIN_LENGTH);
        for i in 0..CHAIN_LENGTH {
            let post_sr = h(0xa0u8 + i as u8);
            blocks.push((FIRST_SLOT + i as u64, bhs[i], post_sr));
        }
        bst::BeaconStateTransitionWitness::from_chain(initial, &blocks)
    }

    /// Casper FFG 2-epoch witness: `(source=0,target=1)` and
    /// `(source=1,target=2)`, both well above 2/3 supermajority. Row 0
    /// finalizes (`is_finalized = true`) with `target_epoch = 1`,
    /// which matches the epoch-boundary row's `epoch = 1` projection
    /// on D2.
    fn ffg_witness() -> ffg::CasperFfgChainWitness {
        // total_active = 100, vote_count = 70 → 3·70 = 210 ≥ 2·100 = 200.
        let tallies = vec![
            (0u64, 1u64, 70u64, 100u64),
            (1u64, 2u64, 70u64, 100u64),
        ];
        ffg::CasperFfgChainWitness::from_tallies(&tallies)
    }

    // ─── Descriptors (single-column tuples) ───────────────────────────

    /// D0: `multi_block.COL_BLOCK_NUMBER` ↔ `state_transition.COL_SLOT`.
    fn d0_block_number_to_slot() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "cross_block_to_beacon_state_block_number_slot_v1".into(),
            a_layer_index: 0,
            a_columns: vec![mb::COL_BLOCK_NUMBER],
            a_selector_column: Some(mb::COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![bst::COL_SLOT],
            b_selector_column: Some(bst::COL_IS_REAL),
        }
    }

    /// D1: `multi_block.COL_BLOCK_HASH_OFFSET[0]` ↔
    /// `state_transition.COL_BLOCK_ROOT_OFFSET[0]` (byte-0 binding).
    fn d1_block_hash_to_block_root_byte0() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "cross_block_to_beacon_state_block_hash_byte0_v1".into(),
            a_layer_index: 0,
            a_columns: vec![mb::COL_BLOCK_HASH_OFFSET],
            a_selector_column: Some(mb::COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![bst::COL_BLOCK_ROOT_OFFSET],
            b_selector_column: Some(bst::COL_IS_REAL),
        }
    }

    /// D2: `casper_ffg.COL_TARGET_EPOCH` (gated by `IS_FINALIZED`) ↔
    /// `state_transition.COL_EPOCH` (gated by `IS_EPOCH_BOUNDARY`).
    fn d2_finalized_target_to_boundary_epoch() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "casper_ffg_finalized_to_beacon_boundary_epoch_v1".into(),
            a_layer_index: 2,
            a_columns: vec![ffg::COL_TARGET_EPOCH],
            a_selector_column: Some(ffg::COL_IS_FINALIZED),
            b_layer_index: 1,
            b_columns: vec![bst::COL_EPOCH],
            b_selector_column: Some(bst::COL_IS_EPOCH_BOUNDARY),
        }
    }

    // ─── Fast static check ────────────────────────────────────────────

    #[test]
    fn descriptor_consistency() {
        let curve = CurveType::Bls48581;

        // Witness shapes.
        let mb_w = multi_block_witness();
        let bst_w = state_transition_witness();
        let ffg_w = ffg_witness();

        assert_eq!(mb_w.rows.len(), CHAIN_LENGTH);
        assert_eq!(bst_w.rows.len(), CHAIN_LENGTH);
        assert_eq!(ffg_w.rows.len(), 2);

        // Multi-block chain numbers line up with state-transition slots.
        for i in 0..CHAIN_LENGTH {
            assert_eq!(mb_w.rows[i].block_number, FIRST_SLOT + i as u64);
            assert_eq!(bst_w.rows[i].slot, FIRST_SLOT + i as u64);
        }

        // Block-hash byte-0 projection matches block-root byte-0
        // projection element-by-element.
        let bhs = block_hashes();
        for i in 0..CHAIN_LENGTH {
            assert_eq!(mb_w.rows[i].block_hash[0], bhs[i][0]);
            assert_eq!(bst_w.rows[i].block_root[0], bhs[i][0]);
        }

        // Slot 63 (i=2) is the lone epoch boundary; epoch = 1.
        assert!(!bst_w.rows[0].is_epoch_boundary);
        assert!(!bst_w.rows[1].is_epoch_boundary);
        assert!(bst_w.rows[2].is_epoch_boundary, "slot 63 must be epoch boundary");
        assert_eq!(bst_w.rows[2].epoch, 1, "slot 63 sits in epoch 1");

        // FFG row 0 finalizes with target_epoch = 1.
        assert!(ffg_w.rows[0].is_justified);
        assert!(ffg_w.rows[1].is_justified);
        assert!(ffg_w.rows[0].is_finalized, "two-epoch rule finalises FFG row 0");
        assert!(!ffg_w.rows[1].is_finalized);
        assert_eq!(ffg_w.rows[0].target_epoch, 1);

        // Trace shapes.
        let mb_trace = mb::build_trace_polynomials(&mb_w, curve);
        let bst_trace = bst::build_trace_polynomials(&bst_w, curve);
        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);

        assert_eq!(mb_trace.num_rows, CHAIN_LENGTH);
        assert_eq!(bst_trace.num_rows, CHAIN_LENGTH);
        assert_eq!(ffg_trace.num_rows, 2);
        // Min FFT width on BLS48-581 = 16.
        assert_eq!(mb_trace.padded_size, 16);
        assert_eq!(bst_trace.padded_size, 16);
        assert_eq!(ffg_trace.padded_size, 16);

        // Descriptor well-formedness.
        let d0 = d0_block_number_to_slot();
        let d1 = d1_block_hash_to_block_root_byte0();
        let d2 = d2_finalized_target_to_boundary_epoch();

        for d in [&d0, &d1, &d2] {
            assert_eq!(d.a_columns.len(), 1, "joint_prove v0 requires single-column tuples");
            assert_eq!(d.b_columns.len(), 1, "joint_prove v0 requires single-column tuples");
            assert_ne!(d.a_layer_index, d.b_layer_index);
            assert!(d.a_layer_index < 3);
            assert!(d.b_layer_index < 3);
        }

        // Column bounds.
        assert!(d0.a_columns[0] < mb::NUM_COLUMNS);
        assert!(d0.b_columns[0] < bst::NUM_COLUMNS);
        assert!(d1.a_columns[0] < mb::NUM_COLUMNS);
        assert!(d1.b_columns[0] < bst::NUM_COLUMNS);
        assert!(d2.a_columns[0] < ffg::NUM_COLUMNS);
        assert!(d2.b_columns[0] < bst::NUM_COLUMNS);

        // Selectors.
        assert_eq!(d0.a_selector_column, Some(mb::COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(bst::COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(mb::COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(bst::COL_IS_REAL));
        assert_eq!(d2.a_selector_column, Some(ffg::COL_IS_FINALIZED));
        assert_eq!(d2.b_selector_column, Some(bst::COL_IS_EPOCH_BOUNDARY));

        // On-row multiset projections.
        // D0: multi_block COL_BLOCK_NUMBER (rows 0..3) vs state_trans COL_SLOT.
        for i in 0..CHAIN_LENGTH {
            let a = mb_trace.columns[mb::COL_BLOCK_NUMBER].evaluations[i].to_u64();
            let b = bst_trace.columns[bst::COL_SLOT].evaluations[i].to_u64();
            assert_eq!(a, b, "D0 row {} projection mismatch", i);
            assert_eq!(a, FIRST_SLOT + i as u64);
        }
        // D1: block_hash byte 0 == block_root byte 0 per row.
        for i in 0..CHAIN_LENGTH {
            let a = mb_trace.columns[mb::COL_BLOCK_HASH_OFFSET].evaluations[i].to_u64();
            let b = bst_trace.columns[bst::COL_BLOCK_ROOT_OFFSET].evaluations[i].to_u64();
            assert_eq!(a, b, "D1 row {} byte-0 projection mismatch", i);
            assert_eq!(a, bhs[i][0] as u64);
        }
        // D2: A side single element = FFG row 0 TARGET_EPOCH (= 1);
        //     B side single element = state_trans row 2 EPOCH (= 1).
        let a_target =
            ffg_trace.columns[ffg::COL_TARGET_EPOCH].evaluations[0].to_u64();
        let b_epoch =
            bst_trace.columns[bst::COL_EPOCH].evaluations[2].to_u64();
        assert_eq!(a_target, 1);
        assert_eq!(b_epoch, 1);
        assert_eq!(a_target, b_epoch, "D2 finalized target ↔ boundary epoch");

        // Confirm the joint-prove input shape (3 traces + 3 linkages).
        let mb_cs = mb::MultiBlockProofConstraintSystem::new(mb_trace.num_rows);
        let bst_cs = bst::BeaconStateTransitionConstraintSystem::new(bst_trace.num_rows);
        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mb_trace, &mb_cs), (&bst_trace, &bst_cs), (&ffg_trace, &ffg_cs)];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone()];
        assert_eq!(traces.len(), 3);
        assert_eq!(linkages.len(), 3);
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip.
    /// Expected runtime under BLS48-581 is in the high hundreds of
    /// seconds on release builds (3 commitments + 3 per-linkage cross
    /// SNARKs + closure equality checks).
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_round_trip() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let mb_w = multi_block_witness();
        let bst_w = state_transition_witness();
        let ffg_w = ffg_witness();

        let mb_trace = mb::build_trace_polynomials(&mb_w, curve);
        let bst_trace = bst::build_trace_polynomials(&bst_w, curve);
        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);

        let mb_cs = mb::MultiBlockProofConstraintSystem::new(mb_trace.num_rows);
        let bst_cs = bst::BeaconStateTransitionConstraintSystem::new(bst_trace.num_rows);
        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mb_trace, &mb_cs), (&bst_trace, &bst_cs), (&ffg_trace, &ffg_cs)];
        let linkages = vec![
            d0_block_number_to_slot(),
            d1_block_hash_to_block_root_byte0(),
            d2_finalized_target_to_boundary_epoch(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 3);
        assert_eq!(ext.linkage_proofs.len(), 3);

        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mb_cs, &bst_cs, &ffg_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the cross-block ↔ beacon ↔ FFG chain",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper `closure_a` on the D2 finalized-target linkage and confirm
    /// `joint_verify` rejects.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_round_trip"]
    fn tampered_closure_rejected() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let mb_w = multi_block_witness();
        let bst_w = state_transition_witness();
        let ffg_w = ffg_witness();

        let mb_trace = mb::build_trace_polynomials(&mb_w, curve);
        let bst_trace = bst::build_trace_polynomials(&bst_w, curve);
        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);

        let mb_cs = mb::MultiBlockProofConstraintSystem::new(mb_trace.num_rows);
        let bst_cs = bst::BeaconStateTransitionConstraintSystem::new(bst_trace.num_rows);
        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mb_trace, &mb_cs), (&bst_trace, &bst_cs), (&ffg_trace, &ffg_cs)];
        let linkages = vec![
            d0_block_number_to_slot(),
            d1_block_hash_to_block_root_byte0(),
            d2_finalized_target_to_boundary_epoch(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed");

        // Mutate the D2 (finalized target ↔ boundary epoch) closure_a.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[2].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mb_cs, &bst_cs, &ffg_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject a tampered D2 closure",
        );
    }
}
