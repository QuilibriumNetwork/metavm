//! Casper FFG chain ↔ stake-weighted finality ↔ FFG checkpoint chain
//! oracle joint-prove integration (task #285).
//!
//! Composes two algebraic AIRs plus the host-side
//! [`crate::ffg_checkpoint_chain`] oracle to validate the
//! justification → finalization → checkpoint-chain pipeline:
//!
//!   1. [`crate::casper_ffg_chain_air`] — per-epoch
//!      `(source_epoch, target_epoch, vote_count, total_active)`
//!      tallies with algebraic supermajority justification
//!      (`is_justified · (3·vc − 2·ta − slack) = 0`) and the
//!      two-epoch finalization rule
//!      (`is_finalized · (1 − is_justified) = 0`,
//!      `is_finalized · (1 − is_justified_next) = 0`,
//!      `is_finalized · (source_epoch_next − target_epoch_curr) = 0`).
//!   2. [`crate::finality_constraints`] — per-validator stake-weighted
//!      finality check with a running-total accumulator and a
//!      `SEL_THRESHOLD` row carrying `(total_attesting, total_active,
//!      slack, claimed_finalized_root)` that meets the 2/3
//!      supermajority.
//!   3. [`crate::ffg_checkpoint_chain`] — host-side oracle. The
//!      3-epoch checkpoint sequence
//!      `genesis → epoch 1 (finalized) → epoch 2 (justified)` is
//!      validated against `verify_finalization_reachable`. The
//!      finalized checkpoint's `root` is reused as the
//!      `finality_constraints` `finalized_root` witness so the
//!      algebraic argument terminates at the same value the host-side
//!      chain oracle declares finalized.
//!
//! ## Witness (3-epoch FFG chain, finalization at epoch 1)
//!
//! The FFG witness commits THREE tallies:
//!
//!   * row 0: `(source=0, target=1)`, 74 % participation → justified.
//!     The next row chains `source = 1 == target`, with the next row
//!     also justified, so row 0 is `is_finalized = true`. **This is
//!     the finalization at epoch 1**.
//!   * row 1: `(source=1, target=2)`, 74 % participation → justified
//!     but not finalized (its successor doesn't continue the chain
//!     with `source = 2`).
//!   * row 2: `(source=2, target=3)`, 74 % participation → justified
//!     but not finalized (no row 3).
//!
//! The host-side checkpoint chain is the 3-element sequence
//! `[genesis(epoch=0), epoch_1, epoch_2]`, with strictly increasing
//! epochs, the canonical Phase-0 anchor at epoch 0, and
//! `epoch_1.root` reused as both the `finality_constraints`
//! `finalized_root` and the synthetic `root_lsb` value the L2
//! linkage binds. `verify_finalization_reachable(genesis, epoch_1,
//! &chain)` is called inline to keep the host-side and algebraic
//! finalization targets consistent.
//!
//! ## Cross-AIR descriptors
//!
//! Two single-column descriptors. Both sides match values on a single
//! row so the per-tuple multisets coincide:
//!
//!   - **L1 — FFG target ↔ Finality checkpoint scalar**: bind the
//!     finalized row's `COL_TARGET_EPOCH` (FFG, gated by
//!     `COL_IS_FINALIZED`) to the finality threshold row's
//!     `COL_TOTAL_ACTIVE` (gated by `COL_SEL_THRESHOLD`). Honest
//!     witness sets both scalars to the same value: `1` (the
//!     finalized target epoch). This is the
//!     `casper_ffg target ↔ finality_constraints checkpoint` link.
//!
//!   - **L2 — FFG finalized ↔ FFG-checkpoint-chain root LSB**: bind
//!     the finalized row's `COL_IS_FINALIZED` (FFG, gated by
//!     `COL_IS_REAL`) to the finality threshold row's
//!     `COL_CLAIMED_FINALIZED_ROOT_OFFSET[0]` (gated by
//!     `COL_SEL_THRESHOLD`). Honest witness places `1` on the FFG
//!     finalized row and arranges the host-side checkpoint chain's
//!     finalized-checkpoint root to have its first byte equal to `1`
//!     (`root = [1, 0, …]`). The host-side
//!     `verify_finalization_reachable` is called against the same
//!     checkpoint to keep the FFG-checkpoint-chain oracle consistent
//!     with the algebraic argument. This is the
//!     `casper_ffg finalized ↔ ffg_checkpoint_chain root` link.
//!
//! ## Tests
//!
//! - `descriptor_consistency` (fast) — descriptors are well-formed,
//!   column indices fall within their AIRs, the FFG witness validates
//!   justification + finalization, the host-side checkpoint chain
//!   oracle accepts the 3-epoch chain, and the two traces commit
//!   matching scalars on the linkage rows.
//! - `honest_round_trip` (`#[ignore]`, slow) — full `joint_prove` +
//!   `joint_verify` round-trip across the 2 AIRs + 2 linkages under
//!   BLS48-581.
//! - `tampered_closure_rejected` (`#[ignore]`, slow) — mutates one
//!   closure scalar so the verifier's closure-equality check fires.

#[cfg(test)]
mod tests {
    use crate::beacon::Checkpoint;
    use crate::casper_ffg_chain_air as ffg;
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::ffg_checkpoint_chain;
    use crate::field::{CurveType, Scalar};
    use crate::finality_constraints as fc;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    // ─── Witness shape constants ──────────────────────────────────────

    /// Finalized target epoch (row 0's `target_epoch` in the FFG
    /// chain) — reused as the finality-side `total_active` scalar so
    /// the L1 linkage projects a matching single-row tuple `{1}` on
    /// both sides.
    const FINALIZED_TARGET_EPOCH: u64 = 1;

    /// First byte of the finalized checkpoint root — chosen as `1` so
    /// the L2 linkage projects the same scalar `1` on both sides
    /// (FFG `is_finalized = 1` and finality
    /// `claimed_finalized_root[0] = 1`).
    const FINALIZED_ROOT_BYTE0: u8 = 1;

    /// Per-row `vote_count` placeholder. The FFG row constraint
    /// `is_justified · (3·vc − 2·ta − slack) = 0` only requires
    /// `3·vc ≥ 2·ta`. We use `vc = 70, ta = 100` (slack = 10).
    const ROW_VOTE_COUNT: u64 = 70;
    const ROW_TOTAL_ACTIVE: u64 = 100;

    // ─── Descriptors ──────────────────────────────────────────────────

    /// L1: FFG `COL_TARGET_EPOCH` (gated by `IS_FINALIZED`) ↔
    /// Finality `COL_TOTAL_ACTIVE` (gated by `SEL_THRESHOLD`).
    fn ffg_target_to_finality_checkpoint_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "casper_ffg_target_to_finality_checkpoint_v1".into(),
            a_layer_index: 0,
            a_columns: vec![ffg::COL_TARGET_EPOCH],
            a_selector_column: Some(ffg::COL_IS_FINALIZED),
            b_layer_index: 1,
            b_columns: vec![fc::COL_TOTAL_ACTIVE],
            b_selector_column: Some(fc::COL_SEL_THRESHOLD),
        }
    }

    /// L2: FFG `COL_IS_FINALIZED` (gated by `IS_REAL`) ↔ Finality
    /// `COL_CLAIMED_FINALIZED_ROOT_OFFSET[0]` (gated by
    /// `SEL_THRESHOLD`). Binds the FFG "is finalized" boolean to the
    /// first byte of the host-side
    /// `ffg_checkpoint_chain` finalized-checkpoint root.
    fn ffg_finalized_to_chain_root_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "casper_ffg_finalized_to_ffg_checkpoint_chain_root_v1".into(),
            a_layer_index: 0,
            a_columns: vec![ffg::COL_IS_FINALIZED],
            a_selector_column: Some(ffg::COL_IS_FINALIZED),
            b_layer_index: 1,
            b_columns: vec![fc::COL_CLAIMED_FINALIZED_ROOT_OFFSET],
            b_selector_column: Some(fc::COL_SEL_THRESHOLD),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    /// Build a 3-epoch Casper FFG chain witness with finalization at
    /// epoch 1 (and ONLY at epoch 1, so the per-row IS_FINALIZED
    /// multiset has exactly one entry matching finality's single
    /// SEL_THRESHOLD row).
    ///
    /// Row layout:
    ///   * row 0: `(0 → 1)` justified, finalises (row 1 chains,
    ///     row 1 justified, `row1.source == row0.target == 1`).
    ///   * row 1: `(1 → 2)` justified BUT NOT finalised — row 2's
    ///     `source` is deliberately chosen NOT to equal row 1's
    ///     `target` so the two-epoch chain link breaks at row 1.
    ///   * row 2: `(5 → 6)` justified, no successor → not finalised.
    fn build_ffg_witness() -> ffg::CasperFfgChainWitness {
        let tallies = vec![
            // (source, target, vote_count, total_active)
            (0u64, 1u64, ROW_VOTE_COUNT, ROW_TOTAL_ACTIVE),
            (1u64, 2u64, ROW_VOTE_COUNT, ROW_TOTAL_ACTIVE),
            // Source = 5 ≠ row1.target (= 2), so row 1 is NOT
            // finalised under the two-epoch rule. target = 6 keeps
            // target-epoch monotonicity.
            (5u64, 6u64, ROW_VOTE_COUNT, ROW_TOTAL_ACTIVE),
        ];
        ffg::CasperFfgChainWitness::from_tallies(&tallies)
    }

    /// Build the host-side checkpoint chain (oracle input) for the
    /// 3-epoch FFG witness. Anchor at epoch 0, chain ratchets through
    /// the prior checkpoints and TERMINATES at the finalised
    /// checkpoint (epoch 1, root[0] = [`FINALIZED_ROOT_BYTE0`]). The
    /// host-side
    /// [`crate::ffg_checkpoint_chain::verify_finalization_reachable`]
    /// requires `chain.last() == finalized` and strictly-increasing
    /// epochs.
    fn build_checkpoint_chain() -> (Checkpoint, Checkpoint, Vec<Checkpoint>) {
        let mut genesis_root = [0u8; 32];
        genesis_root[0] = 0;
        let mut finalized_root = [0u8; 32];
        finalized_root[0] = FINALIZED_ROOT_BYTE0;

        let anchor = Checkpoint { epoch: 0, root: genesis_root };
        let finalized = Checkpoint { epoch: 1, root: finalized_root };
        // 2-element chain: genesis → finalised(epoch 1). This is the
        // canonical "single finalisation step" chain the FFG row-0
        // finalisation event projects.
        let chain = vec![anchor, finalized];
        (anchor, finalized, chain)
    }

    /// Build a finality witness whose threshold row commits
    /// `total_active = FINALIZED_TARGET_EPOCH` (so L1 projects the same
    /// scalar `1` on both sides) and `finalized_root[0] =
    /// FINALIZED_ROOT_BYTE0` (so L2 projects the same scalar `1` on
    /// both sides). The exact validator stake breakdown is irrelevant
    /// to the linkage; we keep the witness small with a single
    /// attesting validator and an inflated `total_active = 1` (so the
    /// supermajority `3·1 ≥ 2·1` holds trivially).
    fn build_finality_witness() -> fc::FinalityWitness {
        let validators = vec![(1u64, 1u8)];
        let mut finalized_root = [0u8; 32];
        finalized_root[0] = FINALIZED_ROOT_BYTE0;
        fc::FinalityWitness::new(
            validators,
            FINALIZED_TARGET_EPOCH,
            [0u8; 32],
            finalized_root,
        )
    }

    // ─── Fast test ────────────────────────────────────────────────────

    /// Validate descriptor wiring, host-side checkpoint-chain oracle
    /// acceptance, and per-linkage scalar consistency at the witness
    /// level (without invoking the slow `joint_prove`).
    #[test]
    fn descriptor_consistency() {
        // 1. Descriptors are well-formed.
        let d1 = ffg_target_to_finality_checkpoint_descriptor();
        assert_eq!(d1.a_columns.len(), 1);
        assert_eq!(d1.b_columns.len(), 1);
        assert_eq!(d1.a_columns[0], ffg::COL_TARGET_EPOCH);
        assert_eq!(d1.b_columns[0], fc::COL_TOTAL_ACTIVE);
        assert_eq!(d1.a_selector_column, Some(ffg::COL_IS_FINALIZED));
        assert_eq!(d1.b_selector_column, Some(fc::COL_SEL_THRESHOLD));
        assert_ne!(d1.a_layer_index, d1.b_layer_index);
        assert!(d1.a_columns[0] < ffg::NUM_COLUMNS);
        assert!(d1.b_columns[0] < fc::NUM_COLUMNS);

        let d2 = ffg_finalized_to_chain_root_descriptor();
        assert_eq!(d2.a_columns.len(), 1);
        assert_eq!(d2.b_columns.len(), 1);
        assert_eq!(d2.a_columns[0], ffg::COL_IS_FINALIZED);
        assert_eq!(d2.b_columns[0], fc::COL_CLAIMED_FINALIZED_ROOT_OFFSET);
        // Post task #295 fix: D2's A-side selector is the same boolean
        // it projects (`COL_IS_FINALIZED`), restricting the multiset to
        // the single finalised row so per-tuple cardinalities match the
        // single B threshold row.
        assert_eq!(d2.a_selector_column, Some(ffg::COL_IS_FINALIZED));
        assert_eq!(d2.b_selector_column, Some(fc::COL_SEL_THRESHOLD));
        assert_ne!(d2.a_layer_index, d2.b_layer_index);
        assert!(d2.a_columns[0] < ffg::NUM_COLUMNS);
        assert!(d2.b_columns[0] < fc::NUM_COLUMNS);

        // 2. FFG witness algebraic shape — 3 epochs, finalization at
        // epoch 1 (row 0).
        let ffg_w = build_ffg_witness();
        assert_eq!(ffg_w.rows.len(), 3);
        assert!(ffg_w.rows[0].is_justified, "epoch 0→1 justified");
        assert!(ffg_w.rows[1].is_justified, "epoch 1→2 justified");
        assert!(ffg_w.rows[2].is_justified, "epoch 2→3 justified");
        assert!(
            ffg_w.rows[0].is_finalized,
            "row 0 finalises via the two-epoch rule",
        );
        assert!(
            !ffg_w.rows[1].is_finalized,
            "row 1 is not finalised (row 2.source != row 1.target)",
        );
        assert!(!ffg_w.rows[2].is_finalized, "row 2 has no successor");
        assert_eq!(ffg_w.rows[0].target_epoch, FINALIZED_TARGET_EPOCH);

        // 3. Host-side ffg_checkpoint_chain oracle accepts the
        // genesis → epoch_1 → epoch_2 chain with epoch_1 finalised.
        let (anchor, finalized, chain) = build_checkpoint_chain();
        ffg_checkpoint_chain::verify_checkpoint_chain(&chain)
            .expect("3-epoch checkpoint chain must be strictly increasing");
        ffg_checkpoint_chain::verify_finalization_reachable(
            &anchor,
            &finalized,
            &chain,
        )
        .expect("finalisation oracle must accept the honest chain");
        assert_eq!(
            finalized.root[0],
            FINALIZED_ROOT_BYTE0,
            "checkpoint-chain finalised root[0] must match L2 scalar",
        );
        assert_eq!(
            finalized.epoch, FINALIZED_TARGET_EPOCH,
            "checkpoint-chain finalised epoch must match L1 scalar",
        );

        // 4. Finality witness threshold-row scalars line up with the
        // FFG finalised row.
        let fin_w = build_finality_witness();
        assert_eq!(fin_w.total_active_balance_gwei, FINALIZED_TARGET_EPOCH);
        assert_eq!(fin_w.finalized_root[0], FINALIZED_ROOT_BYTE0);

        // 5. Build the traces and confirm the linkage rows commit
        // matching scalars on both sides.
        let curve = CurveType::Bls48581;
        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);
        let fin_trace = fc::build_finality_trace_polynomials(&fin_w, curve);

        // L1: FFG row 0 (the finalised row) commits target_epoch = 1;
        // finality threshold row commits total_active = 1.
        let l1_a = ffg_trace.columns[ffg::COL_TARGET_EPOCH]
            .evaluations[0]
            .to_u64();
        let fin_t_row = fin_w.validators.len();
        let l1_b = fin_trace.columns[fc::COL_TOTAL_ACTIVE]
            .evaluations[fin_t_row]
            .to_u64();
        assert_eq!(l1_a, l1_b, "L1: target_epoch == total_active");
        assert_eq!(l1_a, FINALIZED_TARGET_EPOCH);

        // L1 selector check: row 0 of FFG has is_finalized = 1;
        // threshold row of finality has sel_threshold = 1.
        assert_eq!(
            ffg_trace.columns[ffg::COL_IS_FINALIZED].evaluations[0].to_u64(),
            1,
        );
        assert_eq!(
            fin_trace.columns[fc::COL_SEL_THRESHOLD]
                .evaluations[fin_t_row]
                .to_u64(),
            1,
        );

        // L2: FFG row 0 commits is_finalized = 1; finality threshold
        // row commits claimed_finalized_root[0] = 1.
        let l2_a = ffg_trace.columns[ffg::COL_IS_FINALIZED]
            .evaluations[0]
            .to_u64();
        let l2_b = fin_trace.columns[fc::COL_CLAIMED_FINALIZED_ROOT_OFFSET]
            .evaluations[fin_t_row]
            .to_u64();
        assert_eq!(l2_a, l2_b, "L2: is_finalized == finalized_root[0]");
        assert_eq!(l2_a, 1);

        // 6. Selector inflation sanity: rows 1 and 2 of the FFG trace
        // are NOT finalised (they must contribute 0 entries to the L1
        // multiset).
        assert_eq!(
            ffg_trace.columns[ffg::COL_IS_FINALIZED].evaluations[1].to_u64(),
            0,
        );
        assert_eq!(
            ffg_trace.columns[ffg::COL_IS_FINALIZED].evaluations[2].to_u64(),
            0,
        );
    }

    // ─── Slow honest round-trip ───────────────────────────────────────

    /// Full honest `joint_prove` + `joint_verify` round-trip over the
    /// 2-AIR Casper FFG ↔ finality chain with 2 single-column
    /// linkages under BLS48-581.
    ///
    /// Marked `#[ignore]` because `joint_prove` runs
    /// `prove_with_scheme` once per AIR plus one cross-AIR SNARK per
    /// linkage; on BLS48-581 with the 256-row range-table inflation
    /// this typically exceeds the fast-test budget.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove over casper_ffg + finality on BLS48-581"]
    fn honest_round_trip() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Host-side oracle precondition: the checkpoint chain must be
        // honest before we waste cycles on the algebraic argument.
        let (anchor, finalized, chain) = build_checkpoint_chain();
        ffg_checkpoint_chain::verify_finalization_reachable(
            &anchor,
            &finalized,
            &chain,
        )
        .expect("checkpoint-chain oracle must accept honest chain");

        let ffg_w = build_ffg_witness();
        let fin_w = build_finality_witness();

        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);
        let fin_trace = fc::build_finality_trace_polynomials(&fin_w, curve);

        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);
        let fin_cs = fc::FinalityConstraintSystem::new(fin_w.validators.len());

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&ffg_trace, &ffg_cs), (&fin_trace, &fin_cs)];
        let linkages = vec![
            ffg_target_to_finality_checkpoint_descriptor(),
            ffg_finalized_to_chain_root_descriptor(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 2);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&ffg_cs, &fin_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the FFG ↔ finality chain",
        );
    }

    // ─── Slow tampering test ──────────────────────────────────────────

    /// Mutate the closure of the FFG↔finality target/checkpoint
    /// linkage so the joint verifier's closure-equality scalar check
    /// fires.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_round_trip"]
    fn tampered_closure_rejected() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let ffg_w = build_ffg_witness();
        let fin_w = build_finality_witness();

        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);
        let fin_trace = fc::build_finality_trace_polynomials(&fin_w, curve);

        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);
        let fin_cs = fc::FinalityConstraintSystem::new(fin_w.validators.len());

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&ffg_trace, &ffg_cs), (&fin_trace, &fin_cs)];
        let linkages = vec![
            ffg_target_to_finality_checkpoint_descriptor(),
            ffg_finalized_to_chain_root_descriptor(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed");

        // Mutate closure_a on the L1 (FFG target ↔ finality
        // checkpoint) linkage. The joint_verify scalar-equality check
        // `closure_a == closure_b` must reject.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&ffg_cs, &fin_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject a tampered FFG↔finality closure",
        );
    }
}
