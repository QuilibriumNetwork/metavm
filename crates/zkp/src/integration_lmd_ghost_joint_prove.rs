//! LMD-GHOST fork choice ↔ attestation aggregate ↔ block header
//! 3-AIR joint-prove integration test.
//!
//! Mirrors the `integration_joint_prove_three_air` template (3 AIRs / 2
//! descriptors) but with the LMD-GHOST fork-choice AIR as the shared
//! "middle" layer of a head-block binding chain:
//!
//! - layer 0: [`crate::lmd_ghost_fork_choice_air`] (99 cols), gated by
//!   its [`crate::lmd_ghost_fork_choice_air::COL_IS_HEAD`] selector on
//!   the A side of both descriptors so only the greedy head path
//!   contributes tuples,
//! - layer 1: [`crate::attestation_aggregate_air`] (≈266 cols), gated
//!   by its [`crate::attestation_aggregate_air::COL_IS_REAL`] selector
//!   on the per-attestation rows,
//! - layer 2: [`crate::block_header_air`] (979 cols), gated by its
//!   [`crate::block_header_air::COL_IS_REAL`] selector on the per-row
//!   block-header window.
//!
//! ## Descriptors
//!
//! - **D0 — head ↔ attestation signing_root**: re-exports
//!   [`crate::lmd_ghost_fork_choice_air::make_fork_choice_to_attestation_aggregate_descriptor`].
//!   A side = lmd_ghost `block_hash[0..32]` gated by `COL_IS_HEAD`.
//!   B side = attestation `signing_root[0..32]` gated by `COL_IS_REAL`.
//!   32-column tuple (β-RLC over the 32 hash bytes).
//!
//! - **D1 — head ↔ block-header block_hash**: re-exports
//!   [`crate::lmd_ghost_fork_choice_air::make_fork_choice_to_block_header_descriptor`].
//!   A side = lmd_ghost `block_hash[0..32]` gated by `COL_IS_HEAD`.
//!   B side = block-header `block_hash[0..32]` gated by `COL_IS_REAL`.
//!   32-column tuple.
//!
//! ## Witness shape
//!
//! 3-block fork choice tree at BLS48-581:
//!
//!   - block 0: root (parent = `u64::MAX` sentinel), weight 200,
//!   - block 1: child of 0, weight 50  (lighter sibling — NOT head),
//!   - block 2: child of 0, weight 150 (heavier sibling — HEAD).
//!
//! Greedy descent: `0 → 2`. Head path set = `{block 0, block 2}`
//! (since [`crate::lmd_ghost_fork_choice_air::LmdGhostForkChoiceWitness::from_blocks`]
//! marks the root as `is_head` along with each subsequent
//! max-weight pick). The lighter block 1 stays off the head path.
//!
//! For the slow honest round-trip to produce matching closures, the
//! attestation aggregate and block-header traces commit *the same two
//! head block_hashes* (block 0 + block 2) — one entry per hash on each
//! side — so the per-tuple multisets gated by their respective
//! selectors coincide between A (head rows on lmd_ghost) and B
//! (attestation rows / block-header rows). The fast
//! `descriptor_consistency` test does NOT invoke `joint_prove`; it
//! validates wiring, layer indices, tuple shape and column-index
//! bounds, then constructs the orchestrator input vector.
//!
//! ## Curve
//!
//! BLS48-581. All three AIRs declare 8-bit byte range lookups → LogUp
//! auto-inflation pads each per-AIR domain to the 256-row range table
//! inside `joint_prove`. The slow tests are `#[ignore]` because the
//! block-header AIR alone runs ~974 columns × 256 rows under
//! `prove_with_scheme`.
//!
//! ## Cross-references
//!
//! - `integration_joint_prove_three_air.rs` — the heterogeneous 3-AIR
//!   smoke-test template this mirrors (small AIRs, 2 single-column
//!   descriptors).
//! - `lmd_ghost_fork_choice_air::make_fork_choice_to_*_descriptor` —
//!   the pre-built 32-column hash-tuple descriptors consumed here.
//! - `integration_casper_ffg_joint_prove.rs` — neighbouring 3-AIR
//!   integration using attestation_aggregate_air on BLS12-381 (this
//!   file mirrors the structure under BLS48-581).

#[cfg(test)]
mod tests {
    use crate::attestation_aggregate_air as att_air;
    use crate::block_header_air as bh_air;
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::lmd_ghost_fork_choice_air as fc_air;
    use crate::nonnative_fp::Fp;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    // ─── Witness topology ──────────────────────────────────────────────

    /// Block hash for `block_index = i`. Distinct 32-byte pattern per
    /// block.
    fn h(seed: u8) -> [u8; 32] {
        let mut x = [0u8; 32];
        for k in 0..32 {
            x[k] = seed.wrapping_add(k as u8);
        }
        x
    }

    /// 3-block fork choice tree:
    ///   - block 0: root (parent = u64::MAX), weight 200,
    ///   - block 1: child of 0, weight 50  (lighter),
    ///   - block 2: child of 0, weight 150 (heavier → head pick).
    ///
    /// Greedy descent: `0 → 2`. Head set = `{0, 2}`.
    fn build_fork_choice_witness() -> fc_air::LmdGhostForkChoiceWitness {
        let blocks = vec![
            (0u64, u64::MAX, h(0x10), h(0x00), 200u64),
            (1u64, 0u64,      h(0x20), h(0x10),  50u64),
            (2u64, 0u64,      h(0x30), h(0x10), 150u64),
        ];
        fc_air::LmdGhostForkChoiceWitness::from_blocks(blocks)
            .expect("3-block fork choice witness must build")
    }

    /// The two block_hashes the greedy head path commits (block 0 + block
    /// 2). These are the A-side multiset gated by `COL_IS_HEAD` and must
    /// match the B-side multisets on both descriptors for honest
    /// `joint_prove` closures.
    fn head_path_block_hashes() -> [[u8; 32]; 2] {
        [h(0x10), h(0x30)]
    }

    // ─── Synthetic attestation rows ────────────────────────────────────

    /// Build a synthetic attestation-aggregate witness whose per-row
    /// `signing_root` values exactly match the two head-path block
    /// hashes (block 0 + block 2). Bypasses the realistic BLS witness
    /// builder ([`att_air::AttestationAggregateWitness::from_attestation`])
    /// because that would require constructing an `AttestationData`
    /// whose `hash_tree_root()` coincides with each block hash — a
    /// pre-image we cannot synthesise directly. This is faithful for
    /// the head-binding LogUp descriptor: only the `signing_root`
    /// column participates in D0; the other columns are filled with
    /// zero-shape values to keep the trace builder honest about
    /// `IS_REAL` gating.
    fn build_attestation_witness() -> att_air::AttestationAggregateWitness {
        let head_hashes = head_path_block_hashes();
        let rows: Vec<att_air::AttestationAggregateRow> = head_hashes
            .iter()
            .enumerate()
            .map(|(i, signing_root)| att_air::AttestationAggregateRow {
                filtered_pubkey: [0u8; att_air::PK_BYTES],
                bit_index: i as u64,
                signing_root: *signing_root,
                aggregate_sig: [0u8; att_air::SIG_BYTES],
                aggregate_pubkey_x_bytes: [0u8; att_air::PK_BYTES],
                agg_pk_x: Fp::zero(),
                agg_pk_y: Fp::zero(),
                msg_g2_x_c0: Fp::zero(),
                msg_g2_x_c1: Fp::zero(),
                msg_g2_y_c0: Fp::zero(),
                msg_g2_y_c1: Fp::zero(),
                slot: 0,
                committee_index: 0,
                source_epoch: 0,
                target_epoch: 0,
            })
            .collect();
        att_air::AttestationAggregateWitness {
            rows,
            source: Default::default(),
            target: Default::default(),
        }
    }

    // ─── Synthetic block-header rows ───────────────────────────────────

    /// Block-header witness containing one row per head-path block,
    /// with `block_hash` set directly to each head block's 32-byte
    /// value. All other fields are zero so the trace stays minimal.
    /// The `BlockHeaderRow` is filled directly (not via
    /// [`bh_air::from_block_header`]) so we can pin `block_hash` to
    /// the head-path values without computing a real keccak preimage.
    fn build_block_header_witness() -> bh_air::BlockHeaderWitness {
        let head_hashes = head_path_block_hashes();
        let headers: Vec<bh_air::BlockHeaderRow> = head_hashes
            .iter()
            .enumerate()
            .map(|(i, block_hash)| bh_air::BlockHeaderRow {
                header_rlp: [0u8; bh_air::HEADER_RLP_MAX_LEN],
                header_rlp_len: 0,
                state_root: [0u8; 32],
                transactions_root: [0u8; 32],
                receipts_root: [0u8; 32],
                withdrawals_root: [0u8; 32],
                block_hash: *block_hash,
                parent_hash: [0u8; 32],
                number: i as u64,
                timestamp: 0,
                gas_limit: 0,
                gas_used: 0,
                base_fee_per_gas: [0u64; 4],
                beneficiary: [0u64; 4],
                prev_randao: [0u64; 4],
                chain_id: 0,
            })
            .collect();
        bh_air::BlockHeaderWitness::from_headers(headers)
    }

    // ─── Descriptor builders ──────────────────────────────────────────

    /// D0: lmd_ghost `block_hash[0..32]` (gated by `COL_IS_HEAD`)
    /// ↔ attestation aggregate `signing_root[0..32]` (gated by
    /// `COL_IS_REAL`). Re-exports
    /// [`fc_air::make_fork_choice_to_attestation_aggregate_descriptor`]
    /// with layer indices wired for the (fc=0, att=1, bh=2) layout.
    fn d0_head_to_attestation_signing_root() -> CrossAirLogUpDescriptor {
        fc_air::make_fork_choice_to_attestation_aggregate_descriptor(0, 1)
    }

    /// D1: lmd_ghost `block_hash[0..32]` (gated by `COL_IS_HEAD`)
    /// ↔ block-header `block_hash[0..32]` (gated by `COL_IS_REAL`).
    /// Re-exports [`fc_air::make_fork_choice_to_block_header_descriptor`].
    fn d1_head_to_block_header_block_hash() -> CrossAirLogUpDescriptor {
        fc_air::make_fork_choice_to_block_header_descriptor(0, 2)
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Validate the 3-AIR / 2-descriptor wiring without invoking
    /// `joint_prove`. Confirms:
    ///   1. each descriptor is a 32-column tuple binding the 32-byte
    ///      head block hash to its B-side counterpart,
    ///   2. layer indices (0 = fork choice, 1 = attestation, 2 = block
    ///      header) are correctly threaded into both descriptors,
    ///   3. selector columns are wired per spec,
    ///   4. column-index bounds are in range,
    ///   5. the fork-choice greedy head set is `{block 0, block 2}`,
    ///   6. the attestation + block-header traces both expose 2 active
    ///      rows committing the same 2 head block_hashes (honest
    ///      multiset match for the slow tests),
    ///   7. the 3-trace orchestrator input vector + descriptor list
    ///      are constructible without panic.
    #[test]
    fn descriptor_consistency() {
        let curve = CurveType::Bls48581;

        let d0 = d0_head_to_attestation_signing_root();
        let d1 = d1_head_to_block_header_block_hash();

        // Tuple shape: both descriptors are 32-column hash bindings.
        assert_eq!(d0.a_columns.len(), fc_air::HASH_LEN);
        assert_eq!(d0.b_columns.len(), fc_air::HASH_LEN);
        assert_eq!(d1.a_columns.len(), fc_air::HASH_LEN);
        assert_eq!(d1.b_columns.len(), fc_air::HASH_LEN);

        // Layer indices: D0 wires 0↔1 (fc ↔ attestation), D1 wires 0↔2
        // (fc ↔ block-header). Layer 0 (lmd_ghost) is the shared
        // "head" trace participating in both descriptors.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_ne!(d0.a_layer_index, d0.b_layer_index);
        assert_ne!(d1.a_layer_index, d1.b_layer_index);

        // Selectors: A side gated by lmd_ghost `COL_IS_HEAD` on both
        // descriptors; B side gated by each layer's `COL_IS_REAL`.
        assert_eq!(d0.a_selector_column, Some(fc_air::COL_IS_HEAD));
        assert_eq!(d0.b_selector_column, Some(att_air::COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(fc_air::COL_IS_HEAD));
        assert_eq!(d1.b_selector_column, Some(bh_air::COL_IS_REAL));

        // Column-index references: every A column on both descriptors
        // is a contiguous slice of lmd_ghost's `COL_BLOCK_HASH_OFFSET..
        // +HASH_LEN` window. B columns target each AIR's hash-window
        // base.
        for k in 0..fc_air::HASH_LEN {
            assert_eq!(d0.a_columns[k], fc_air::COL_BLOCK_HASH_OFFSET + k);
            assert_eq!(d0.b_columns[k], att_air::COL_SIGNING_ROOT_OFFSET + k);
            assert_eq!(d1.a_columns[k], fc_air::COL_BLOCK_HASH_OFFSET + k);
            assert_eq!(d1.b_columns[k], bh_air::COL_BLOCK_HASH_OFFSET + k);

            // Bounds check inside each AIR's column count.
            assert!(d0.a_columns[k] < fc_air::NUM_COLUMNS);
            assert!(d0.b_columns[k] < att_air::NUM_COLUMNS);
            assert!(d1.a_columns[k] < fc_air::NUM_COLUMNS);
            assert!(d1.b_columns[k] < bh_air::NUM_COLUMNS);
        }

        // ─── Greedy head set ───────────────────────────────────────
        let fc_w = build_fork_choice_witness();
        // Three rows, one per block.
        assert_eq!(fc_w.rows.len(), 3);
        // Head set must be {block 0, block 2}.
        let mut head_indices: Vec<u64> = fc_w
            .rows
            .iter()
            .filter(|r| r.is_head)
            .map(|r| r.block_index)
            .collect();
        head_indices.sort();
        assert_eq!(head_indices, vec![0, 2], "greedy head path must be 0 → 2");
        // Block 1 (the lighter sibling) is OFF the head path.
        assert!(
            !fc_w.rows.iter().any(|r| r.is_head && r.block_index == 1),
            "lighter sibling block 1 must not be on the head path",
        );

        // ─── Honest multiset match check ──────────────────────────
        let att_w = build_attestation_witness();
        let bh_w = build_block_header_witness();
        let head_hashes = head_path_block_hashes();

        assert_eq!(att_w.rows.len(), 2);
        assert_eq!(bh_w.headers.len(), 2);
        for k in 0..2 {
            assert_eq!(att_w.rows[k].signing_root, head_hashes[k]);
            assert_eq!(bh_w.headers[k].block_hash, head_hashes[k]);
        }
        // Confirm both head-path block hashes appear in the
        // fork-choice trace under `is_head = true`.
        for hh in head_hashes.iter() {
            assert!(
                fc_w.rows.iter().any(|r| r.is_head && r.block_hash == *hh),
                "head hash must appear on a head row",
            );
        }

        // ─── Orchestrator input shape ─────────────────────────────
        let fc_trace = fc_air::build_trace_polynomials(&fc_w, curve);
        let att_trace = att_air::build_trace_polynomials(&att_w, curve);
        let bh_trace = bh_air::build_trace_polynomials(&bh_w, curve);

        let fc_cs = fc_air::LmdGhostForkChoiceConstraintSystem::new(fc_trace.num_rows);
        let att_cs = att_air::AttestationAggregateConstraintSystem::new(att_trace.num_rows);
        let bh_cs = bh_air::BlockHeaderConstraintSystem::new(bh_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&fc_trace, &fc_cs), (&att_trace, &att_cs), (&bh_trace, &bh_cs)];
        let linkages = vec![d0.clone(), d1.clone()];

        assert_eq!(traces.len(), 3, "3-AIR joint_prove input must have 3 traces");
        assert_eq!(linkages.len(), 2, "must wire exactly 2 descriptors");
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip with
    /// the 2 head-binding descriptors on BLS48-581.
    ///
    /// Marked `#[ignore]` because, even on a 3-block witness,
    /// `joint_prove` runs:
    ///   - 3× per-AIR `prove_with_scheme` (block_header_air is ~974
    ///     cols × 256-row range table → dominates the wall-clock),
    ///   - 2× per-linkage `prove_with_scheme` on the inner
    ///     `LinkageConstraintSystem` (each over a 32-column hash
    ///     tuple under joint γ),
    ///   - cross-trace + closure-wrap KZG opens for each descriptor.
    /// Expected release runtime: high hundreds of seconds.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify under BLS48-581 (5 inner prove calls; block_header_air dominates)"]
    fn honest_three_air_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let fc_w = build_fork_choice_witness();
        let att_w = build_attestation_witness();
        let bh_w = build_block_header_witness();

        let fc_trace = fc_air::build_trace_polynomials(&fc_w, curve);
        let att_trace = att_air::build_trace_polynomials(&att_w, curve);
        let bh_trace = bh_air::build_trace_polynomials(&bh_w, curve);

        let fc_cs = fc_air::LmdGhostForkChoiceConstraintSystem::new(fc_trace.num_rows);
        let att_cs = att_air::AttestationAggregateConstraintSystem::new(att_trace.num_rows);
        let bh_cs = bh_air::BlockHeaderConstraintSystem::new(bh_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&fc_trace, &fc_cs), (&att_trace, &att_cs), (&bh_trace, &bh_cs)];
        let linkages = vec![
            d0_head_to_attestation_signing_root(),
            d1_head_to_block_header_block_hash(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 3, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            2,
            "expected one CrossAirLogUpProof per descriptor",
        );

        // Honest closure equality: A side commits the 32-byte head
        // block_hashes for the 2 is_head rows under joint γ; B side
        // commits the matching 32-byte signing_root / block_hash
        // tuples on 2 IS_REAL rows. Multiset cardinalities match
        // 1-for-1 → per-descriptor closures coincide.
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&fc_cs, &att_cs, &bh_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 3-AIR joint_verify must accept the LMD-GHOST head-binding chain",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper `closure_a` on descriptor D1 (the head ↔ block-header
    /// linkage). The verifier's `closure_a == closure_b` scalar
    /// equality must reject.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_three_air_joint_verify_true"]
    fn tampered_three_air_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let fc_w = build_fork_choice_witness();
        let att_w = build_attestation_witness();
        let bh_w = build_block_header_witness();

        let fc_trace = fc_air::build_trace_polynomials(&fc_w, curve);
        let att_trace = att_air::build_trace_polynomials(&att_w, curve);
        let bh_trace = bh_air::build_trace_polynomials(&bh_w, curve);

        let fc_cs = fc_air::LmdGhostForkChoiceConstraintSystem::new(fc_trace.num_rows);
        let att_cs = att_air::AttestationAggregateConstraintSystem::new(att_trace.num_rows);
        let bh_cs = bh_air::BlockHeaderConstraintSystem::new(bh_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&fc_trace, &fc_cs), (&att_trace, &att_cs), (&bh_trace, &bh_cs)];
        let linkages = vec![
            d0_head_to_attestation_signing_root(),
            d1_head_to_block_header_block_hash(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed");

        // Mutate closure_a on the SECOND descriptor (head ↔
        // block_header_air). joint_verify's per-descriptor
        // closure-equality must fire.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&fc_cs, &att_cs, &bh_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject a tampered head ↔ block-header closure",
        );
    }
}
