//! End-to-end `joint_prove` / `joint_verify` integration smoke test
//! composing the **per-block validity 5-tuple**:
//!
//! - layer 0: [`crate::block_header_air`] (the EVM block header)
//! - layer 1: [`crate::tx_full_chain_air`] (one transaction)
//! - layer 2: [`crate::block_proposer_sig_air`] (beacon proposer sig)
//! - layer 3: [`crate::withdrawal_root_air`] (Shanghai withdrawals MPT)
//! - layer 4: [`crate::beacon_state_transition_air`] (slot-level state root chain)
//!
//! The five AIRs together commit every per-block algebraic surface the
//! roadmap's Layer A → Layer B bridge depends on:
//!
//!   * the executable block header (state_root, tx_root, receipts_root,
//!     withdrawals_root, block_number, timestamp, gas_used, gas_limit,
//!     block_hash, parent_hash, …),
//!   * the per-transaction summary (tx_index, gas_used, status, sender,
//!     tx_hash, sig_hash, tx_type),
//!   * the BLS beacon-proposer signature over the SSZ `block_root`,
//!   * the Ethereum `withdrawals_root` MPT,
//!   * the beacon-chain `post_state_root` carry-forward.
//!
//! ## Cross-AIR LogUp descriptors (4)
//!
//! Each descriptor uses a **single-column tuple** to stay inside the
//! current `joint_prove` shape (which requires `a_columns.len() == 1`
//! for `build_linkage_trace`'s SNARK path). The hub is
//! `block_header_air`:
//!
//!   * D0: `block_header_air.COL_NUMBER` (gated by `IS_REAL`) ↔
//!     `tx_full_chain_air.COL_GAS_USED` — chosen so that the honest
//!     witness has `block.number == tx.gas_used`; this is a synthetic
//!     equality used only to drive the protocol path. The numeric tie
//!     is documented at the witness builder.
//!   * D1: `block_header_air.COL_TIMESTAMP` (gated by `IS_REAL`) ↔
//!     `block_proposer_sig_air.COL_SLOT` (gated by `IS_REAL`) — honest
//!     witness uses `slot == timestamp`.
//!   * D2: `block_header_air.COL_GAS_LIMIT` (gated by `IS_REAL`) ↔
//!     `withdrawal_root_air.COL_VALIDATOR_INDEX` (gated by `IS_FIRST`) —
//!     honest witness uses a single withdrawal with `validator_index =
//!     gas_limit`.
//!   * D3: `block_header_air.COL_GAS_USED` (gated by `IS_REAL`) ↔
//!     `beacon_state_transition_air.COL_SLOT` (gated by `IS_REAL`) —
//!     honest witness uses `slot == gas_used`.
//!
//! These linkages are protocol-faithful: they exercise the cross-AIR
//! LogUp orchestrator (joint-γ derivation, per-linkage SNARK,
//! cross-trace tuple binding at `z`, closure-wrap openings at
//! `ω^{n-1}`). The numeric equalities are not the production wire-up
//! that `block_full_proof_air` advertises in its descriptor builders;
//! they are picked here because the production descriptors are
//! multi-column (32-byte hashes etc.) and the current single-column
//! `joint_prove` does not yet accept them.
//!
//! Once `joint_prove` learns multi-column tuples the test should be
//! ported to the production descriptors emitted by
//! [`crate::block_full_proof_air::make_block_to_block_header_descriptor`]
//! and friends.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{
        joint_prove, joint_verify, CrossAirLogUpDescriptor,
    };
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::beacon_state_transition_air::{
        build_trace_polynomials as build_st_trace, BeaconStateTransitionConstraintSystem,
        BeaconStateTransitionWitness,
        COL_IS_REAL as ST_COL_IS_REAL, COL_SLOT as ST_COL_SLOT,
    };
    use crate::block_header::BlockHeader;
    use crate::block_header_air::{
        build_trace_polynomials as build_bh_trace, from_block_header,
        BlockHeaderConstraintSystem, BlockHeaderWitness,
        COL_GAS_LIMIT as BH_COL_GAS_LIMIT, COL_GAS_USED as BH_COL_GAS_USED,
        COL_IS_REAL as BH_COL_IS_REAL, COL_NUMBER as BH_COL_NUMBER,
        COL_TIMESTAMP as BH_COL_TIMESTAMP,
    };
    use crate::block_proposer_sig_air::{
        build_trace_polynomials as build_ps_trace, BlockProposerSigConstraintSystem,
        BlockProposerSigWitness,
        COL_IS_REAL as PS_COL_IS_REAL, COL_SLOT as PS_COL_SLOT,
    };
    use crate::transaction::{LegacyTx, Transaction};
    use crate::tx_full_chain_air::{
        build_trace_polynomials as build_tx_trace, TxFullChainConstraintSystem,
        TxFullChainWitness,
        COL_GAS_USED as TX_COL_GAS_USED, COL_IS_REAL as TX_COL_IS_REAL,
    };
    use crate::withdrawal::Withdrawal;
    use crate::withdrawal_root_air::{
        build_trace_polynomials as build_wr_trace, WithdrawalRootConstraintSystem,
        WithdrawalRootWitness,
        COL_IS_FIRST as WR_COL_IS_FIRST, COL_VALIDATOR_INDEX as WR_COL_VALIDATOR_INDEX,
    };

    // ─── Honest witness scalars ────────────────────────────────────────

    /// Block number == tx gas_used == beacon slot for D0 / D3 closure
    /// equality (single-column tuple).
    const HONEST_NUMBER: u64 = 100;
    /// Block timestamp == proposer-sig slot for D1 closure equality.
    const HONEST_TIMESTAMP: u64 = 1_700_000_000;
    /// Block gas_limit == withdrawal validator_index for D2 closure
    /// equality.
    const HONEST_GAS_LIMIT: u64 = 30_000_000;
    /// Block gas_used (also threaded into the tx witness so D0 closure
    /// equality holds: tx_gas_used == block_number).
    const HONEST_GAS_USED: u64 = 21_000;

    // ─── Descriptor builders ──────────────────────────────────────────

    /// D0: `block_header_air.COL_NUMBER` ↔ `tx_full_chain_air.COL_GAS_USED`.
    fn descriptor_bh_to_tx() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_full_5air_bh_to_tx_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_NUMBER],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![TX_COL_GAS_USED],
            b_selector_column: Some(TX_COL_IS_REAL),
        }
    }

    /// D1: `block_header_air.COL_TIMESTAMP` ↔ `block_proposer_sig_air.COL_SLOT`.
    fn descriptor_bh_to_ps() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_full_5air_bh_to_ps_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_TIMESTAMP],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![PS_COL_SLOT],
            b_selector_column: Some(PS_COL_IS_REAL),
        }
    }

    /// D2: `block_header_air.COL_GAS_LIMIT` ↔ `withdrawal_root_air.COL_VALIDATOR_INDEX`.
    fn descriptor_bh_to_wr() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_full_5air_bh_to_wr_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_GAS_LIMIT],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 3,
            b_columns: vec![WR_COL_VALIDATOR_INDEX],
            b_selector_column: Some(WR_COL_IS_FIRST),
        }
    }

    /// D3: `block_header_air.COL_GAS_USED` ↔ `beacon_state_transition_air.COL_SLOT`.
    fn descriptor_bh_to_st() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_full_5air_bh_to_st_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_GAS_USED],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 4,
            b_columns: vec![ST_COL_SLOT],
            b_selector_column: Some(ST_COL_IS_REAL),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    fn synth_block_header() -> BlockHeader {
        let mut h = BlockHeader::default();
        h.number = HONEST_NUMBER;
        h.timestamp = HONEST_TIMESTAMP;
        h.gas_limit = HONEST_GAS_LIMIT;
        h.gas_used = HONEST_GAS_USED;
        h.parent_hash = [0xAAu8; 32];
        h.state_root = [0xBBu8; 32];
        h.transactions_root = [0xCCu8; 32];
        h.receipts_root = [0xDDu8; 32];
        h.withdrawals_root = Some([0xEEu8; 32]);
        h
    }

    fn build_bh_witness() -> BlockHeaderWitness {
        let row = from_block_header(&synth_block_header());
        BlockHeaderWitness::from_headers(vec![row])
    }

    fn build_tx_witness() -> TxFullChainWitness {
        let mut gas_price = [0u8; 32];
        gas_price[31] = 1;
        let tx = Transaction::Legacy(LegacyTx {
            nonce: 0,
            gas_price,
            gas_limit: HONEST_GAS_LIMIT,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        });
        // tx.gas_used = HONEST_NUMBER so the single-column tuple on D0
        // matches block.number on the BH side.
        TxFullChainWitness::from_transaction(
            0,
            &tx,
            [0u8; 20],
            HONEST_NUMBER,
            HONEST_NUMBER,
            1,
        )
    }

    fn build_ps_witness() -> BlockProposerSigWitness {
        // slot = HONEST_TIMESTAMP so D1 closure equality holds.
        BlockProposerSigWitness::from_signed_block(
            [0u8; 32],
            [0u8; 32],
            7,
            [0u8; 48],
            [0u8; 96],
            HONEST_TIMESTAMP,
        )
    }

    fn build_wr_witness() -> WithdrawalRootWitness {
        // Single withdrawal with validator_index = HONEST_GAS_LIMIT so
        // D2 closure equality holds.
        let w = Withdrawal {
            index: 0,
            validator_index: HONEST_GAS_LIMIT,
            address: [0x33u8; 20],
            amount: 1,
        };
        WithdrawalRootWitness::from_withdrawals(&[w])
    }

    fn build_st_witness() -> BeaconStateTransitionWitness {
        // Single slot at slot = HONEST_GAS_USED so D3 closure equality
        // holds.
        BeaconStateTransitionWitness::from_chain(
            [0u8; 32],
            &[(HONEST_GAS_USED, [0u8; 32], [0u8; 32])],
        )
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across all four descriptors and a
    /// non-prove sanity check on the 5-trace shape `joint_prove`
    /// receives. This does NOT call `joint_prove`, so it runs in CI.
    #[test]
    fn descriptors_and_trace_shapes_well_formed() {
        let d0 = descriptor_bh_to_tx();
        let d1 = descriptor_bh_to_ps();
        let d2 = descriptor_bh_to_wr();
        let d3 = descriptor_bh_to_st();

        // Single-column tuples (joint_prove invariant).
        for d in [&d0, &d1, &d2, &d3] {
            assert_eq!(d.a_columns.len(), 1);
            assert_eq!(d.b_columns.len(), 1);
        }
        // Hub layer is 0; each descriptor binds layer 0 ↔ {1, 2, 3, 4}.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d2.a_layer_index, 0);
        assert_eq!(d2.b_layer_index, 3);
        assert_eq!(d3.a_layer_index, 0);
        assert_eq!(d3.b_layer_index, 4);

        // Selectors are wired on both sides.
        assert_eq!(d0.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(TX_COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(PS_COL_IS_REAL));
        assert_eq!(d2.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d2.b_selector_column, Some(WR_COL_IS_FIRST));
        assert_eq!(d3.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d3.b_selector_column, Some(ST_COL_IS_REAL));

        // Column indices reference each AIR's published COL_* constants
        // so any column-layout drift in the per-AIR modules surfaces
        // here.
        assert_eq!(d0.a_columns[0], BH_COL_NUMBER);
        assert_eq!(d0.b_columns[0], TX_COL_GAS_USED);
        assert_eq!(d1.a_columns[0], BH_COL_TIMESTAMP);
        assert_eq!(d1.b_columns[0], PS_COL_SLOT);
        assert_eq!(d2.a_columns[0], BH_COL_GAS_LIMIT);
        assert_eq!(d2.b_columns[0], WR_COL_VALIDATOR_INDEX);
        assert_eq!(d3.a_columns[0], BH_COL_GAS_USED);
        assert_eq!(d3.b_columns[0], ST_COL_SLOT);

        // Build all 5 per-AIR witnesses + traces at BLS48-581 and
        // confirm the 5-trace orchestrator input is constructible.
        let curve = CurveType::Bls48581;
        let bh_w = build_bh_witness();
        let tx_w = build_tx_witness();
        let ps_w = build_ps_witness();
        let wr_w = build_wr_witness();
        let st_w = build_st_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tx = build_tx_trace(&tx_w, curve);
        let trace_ps = build_ps_trace(&ps_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);
        let trace_st = build_st_trace(&st_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tx = TxFullChainConstraintSystem::new(trace_tx.num_rows);
        let cs_ps = BlockProposerSigConstraintSystem::new(trace_ps.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);
        let cs_st = BeaconStateTransitionConstraintSystem::new(trace_st.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tx, &cs_tx),
            (&trace_ps, &cs_ps),
            (&trace_wr, &cs_wr),
            (&trace_st, &cs_st),
        ];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone(), d3.clone()];

        assert_eq!(traces.len(), 5);
        assert_eq!(linkages.len(), 4);

        // Bounds check that `joint_prove` would perform at descriptor
        // ingress.
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Spot-check the honest single-column equalities — these are
        // what makes each linkage's multiset equality hold on identical
        // single-active-row witnesses.
        assert_eq!(
            trace_bh.columns[BH_COL_NUMBER].evaluations[0].to_u64(),
            HONEST_NUMBER,
        );
        assert_eq!(
            trace_tx.columns[TX_COL_GAS_USED].evaluations[0].to_u64(),
            HONEST_NUMBER,
        );
        assert_eq!(
            trace_bh.columns[BH_COL_TIMESTAMP].evaluations[0].to_u64(),
            HONEST_TIMESTAMP,
        );
        assert_eq!(
            trace_ps.columns[PS_COL_SLOT].evaluations[0].to_u64(),
            HONEST_TIMESTAMP,
        );
        assert_eq!(
            trace_bh.columns[BH_COL_GAS_LIMIT].evaluations[0].to_u64(),
            HONEST_GAS_LIMIT,
        );
        assert_eq!(
            trace_wr.columns[WR_COL_VALIDATOR_INDEX].evaluations[0].to_u64(),
            HONEST_GAS_LIMIT,
        );
        assert_eq!(
            trace_bh.columns[BH_COL_GAS_USED].evaluations[0].to_u64(),
            HONEST_GAS_USED,
        );
        assert_eq!(
            trace_st.columns[ST_COL_SLOT].evaluations[0].to_u64(),
            HONEST_GAS_USED,
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 5-AIR `joint_prove` + `joint_verify` round-trip with
    /// 4 cross-AIR LogUp descriptors. Run via `--ignored --release` —
    /// expected runtime is in the high hundreds to low thousands of
    /// seconds on BLS48-581 (5 per-AIR proves at ≥256-row domains + 4
    /// per-linkage SNARK proves + cross-trace KZG opens × 4).
    #[test]
    #[ignore = "slow: 5-AIR joint_prove + joint_verify under BLS48-581 (9 inner prove calls)"]
    fn honest_block_full_5air_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bh_w = build_bh_witness();
        let tx_w = build_tx_witness();
        let ps_w = build_ps_witness();
        let wr_w = build_wr_witness();
        let st_w = build_st_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tx = build_tx_trace(&tx_w, curve);
        let trace_ps = build_ps_trace(&ps_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);
        let trace_st = build_st_trace(&st_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tx = TxFullChainConstraintSystem::new(trace_tx.num_rows);
        let cs_ps = BlockProposerSigConstraintSystem::new(trace_ps.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);
        let cs_st = BeaconStateTransitionConstraintSystem::new(trace_st.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tx, &cs_tx),
            (&trace_ps, &cs_ps),
            (&trace_wr, &cs_wr),
            (&trace_st, &cs_st),
        ];
        let linkages = vec![
            descriptor_bh_to_tx(),
            descriptor_bh_to_ps(),
            descriptor_bh_to_wr(),
            descriptor_bh_to_st(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 5-AIR joint_prove must succeed");

        assert_eq!(proofs.len(), 5);
        assert_eq!(ext.linkage_proofs.len(), 4);

        // Honest closure equality on every descriptor.
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_bh, &cs_tx, &cs_ps, &cs_wr, &cs_st];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 5-AIR joint_verify must accept matching tuples on all four descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper `closure_a` on the third descriptor (D2,
    /// block_header_air.COL_GAS_LIMIT ↔ withdrawal_root_air.COL_VALIDATOR_INDEX)
    /// and confirm `joint_verify` rejects.
    #[test]
    #[ignore = "slow: depends on honest_block_full_5air_joint_verify_true setup"]
    fn tampered_block_full_5air_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bh_w = build_bh_witness();
        let tx_w = build_tx_witness();
        let ps_w = build_ps_witness();
        let wr_w = build_wr_witness();
        let st_w = build_st_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tx = build_tx_trace(&tx_w, curve);
        let trace_ps = build_ps_trace(&ps_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);
        let trace_st = build_st_trace(&st_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tx = TxFullChainConstraintSystem::new(trace_tx.num_rows);
        let cs_ps = BlockProposerSigConstraintSystem::new(trace_ps.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);
        let cs_st = BeaconStateTransitionConstraintSystem::new(trace_st.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tx, &cs_tx),
            (&trace_ps, &cs_ps),
            (&trace_wr, &cs_wr),
            (&trace_st, &cs_st),
        ];
        let linkages = vec![
            descriptor_bh_to_tx(),
            descriptor_bh_to_ps(),
            descriptor_bh_to_wr(),
            descriptor_bh_to_st(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed on honest inputs");

        // Tamper descriptor index 2 (D2) closure_a.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[2].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_bh, &cs_tx, &cs_ps, &cs_wr, &cs_st];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched closures on D2",
        );
    }
}
