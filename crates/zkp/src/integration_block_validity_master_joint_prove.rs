//! Task #251 — master per-block validity 5-AIR `joint_prove` /
//! `joint_verify` integration scaffold composing the **executable block
//! header** with the **transaction / receipt / withdrawal** Layer-A
//! surfaces and the **beacon state transition** Layer-B bridge:
//!
//! - layer 0: [`crate::block_header_air`] — the Ethereum block header
//!   (state_root, tx_root, receipts_root, withdrawals_root, number,
//!   timestamp, …),
//! - layer 1: [`crate::tx_full_chain_air`] — per-transaction summary
//!   (tx_index, tx_hash, sender, nonce, gas_used, cumulative_gas,
//!   status, tx_type, sig_hash),
//! - layer 2: [`crate::receipt_status_air`] — per-receipt status and
//!   cumulative-gas chain,
//! - layer 3: [`crate::withdrawal_root_air`] — Shanghai withdrawals
//!   MPT,
//! - layer 4: [`crate::beacon_state_transition_air`] — beacon-chain
//!   slot-level state-root carry-forward.
//!
//! ## Cross-AIR LogUp descriptors (4)
//!
//! All four descriptors are **single-column tuples** (the
//! `cross_air_logup::joint_prove` API supports
//! `a_columns.len() == 1` only — see
//! `cross_air_logup::build_linkage_trace`'s assertion). The hub is
//! layer 0 (`block_header_air`):
//!
//!   * D0 (`tx_root` ↔ `tx_hash` byte 0):
//!     `block_header_air.COL_TRANSACTIONS_ROOT_OFFSET` (byte 0, gated
//!     by `IS_REAL`) ↔ `tx_full_chain_air.COL_TX_HASH_OFFSET` (byte 0,
//!     gated by `IS_REAL`). Honest witness sets
//!     `header.transactions_root[0] = tx0.hash()[0]`.
//!
//!   * D1 (`receipts_root` ↔ first-receipt `cumulative_gas`):
//!     `block_header_air.COL_RECEIPTS_ROOT_OFFSET` (byte 0, gated by
//!     `IS_REAL`) ↔ `receipt_status_air.COL_CUMULATIVE_GAS` (gated by
//!     `IS_FIRST`). Honest witness sets `header.receipts_root[0] =
//!     receipt[0].cumulative_gas_used` (with `cumulative_gas_used ≤
//!     255` so the single-byte tuple matches the full u64 column).
//!
//!   * D2 (`withdrawals_root` byte 0):
//!     `block_header_air.COL_WITHDRAWALS_ROOT_OFFSET` (byte 0, gated
//!     by `IS_REAL`) ↔ `withdrawal_root_air.COL_WITHDRAWALS_ROOT_OFFSET`
//!     (byte 0, gated by `IS_FIRST`). Honest witness sets
//!     `header.withdrawals_root = withdrawal_witness.withdrawals_root`.
//!
//!   * D3 (`block_number` ↔ beacon `slot`):
//!     `block_header_air.COL_NUMBER` (gated by `IS_REAL`) ↔
//!     `beacon_state_transition_air.COL_SLOT` (gated by `IS_REAL`).
//!     Honest witness sets `header.number = state_transition.slot`
//!     (synthetic "modulo conversion": both AIRs commit the raw u64
//!     scalar, so the binding is a numeric identity rather than a
//!     deep slot-↔-block-number derivation).
//!
//! ## Witness shape
//!
//! Per task #251 the block in this harness has **2 transactions** and
//! **2 withdrawals**:
//!
//!   - `block_header_air`: 1 row (the single executable header).
//!   - `tx_full_chain_air`: 1 active row (the first transaction). The
//!     second transaction is conceptually part of the block; because
//!     `tx_full_chain_air` does not publish an `IS_FIRST` selector
//!     column, the cross-AIR LogUp single-column tuple on D0 can only
//!     anchor the first transaction. The second tx is omitted from
//!     the AIR witness here. Multi-row tx joint-prove with first-tx
//!     anchoring is a follow-up once tx_full_chain_air gains
//!     `IS_FIRST` or the joint-prove protocol gains multi-column
//!     tuples.
//!   - `receipt_status_air`: 2 rows (both receipts). D1 gates the
//!     first receipt via `IS_FIRST`.
//!   - `withdrawal_root_air`: 2 rows (both withdrawals). D2 gates the
//!     first withdrawal via `IS_FIRST`.
//!   - `beacon_state_transition_air`: 1 row.
//!
//! ## Cross-references
//!
//!   - [`crate::integration_block_full_proof_joint_prove`] — the
//!     existing 5-AIR per-block validity harness this module mirrors;
//!     uses the same hub layer and single-column tuple protocol.
//!   - [`crate::integration_multi_tx_full_chain_joint_prove`] —
//!     existing multi-row `tx_full_chain_air` harness.

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
        BeaconStateTransitionWitness, COL_IS_REAL as ST_COL_IS_REAL, COL_SLOT as ST_COL_SLOT,
    };
    use crate::block_header::BlockHeader;
    use crate::block_header_air::{
        build_trace_polynomials as build_bh_trace, from_block_header,
        BlockHeaderConstraintSystem, BlockHeaderWitness,
        COL_IS_REAL as BH_COL_IS_REAL, COL_NUMBER as BH_COL_NUMBER,
        COL_RECEIPTS_ROOT_OFFSET as BH_COL_RECEIPTS_ROOT_OFFSET,
        COL_TRANSACTIONS_ROOT_OFFSET as BH_COL_TRANSACTIONS_ROOT_OFFSET,
        COL_WITHDRAWALS_ROOT_OFFSET as BH_COL_WITHDRAWALS_ROOT_OFFSET,
    };
    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air::{
        build_trace_polynomials as build_rs_trace, ReceiptStatusConstraintSystem,
        ReceiptStatusWitness, COL_CUMULATIVE_GAS as RS_COL_CUMULATIVE_GAS,
        COL_IS_FIRST as RS_COL_IS_FIRST,
    };
    use crate::transaction::{LegacyTx, Transaction};
    use crate::tx_full_chain_air::{
        build_trace_polynomials as build_tf_trace, TxFullChainConstraintSystem,
        TxFullChainWitness, COL_IS_REAL as TF_COL_IS_REAL,
        COL_TX_HASH_OFFSET as TF_COL_TX_HASH_OFFSET,
    };
    use crate::withdrawal::Withdrawal;
    use crate::withdrawal_root_air::{
        build_trace_polynomials as build_wr_trace, WithdrawalRootConstraintSystem,
        WithdrawalRootWitness, COL_IS_FIRST as WR_COL_IS_FIRST,
        COL_WITHDRAWALS_ROOT_OFFSET as WR_COL_WITHDRAWALS_ROOT_OFFSET,
    };

    // ─── Honest witness scalars ────────────────────────────────────────

    /// D1 honest equality: `receipts_root[0] = cumulative_gas_used`
    /// where the latter is a full u64 column. Keep it ≤ 255 so the
    /// single byte on the block_header side actually matches the u64
    /// scalar value.
    const HONEST_FIRST_CUMUL_GAS: u64 = 42;

    /// D1 second-receipt cumulative gas (must be ≥ HONEST_FIRST so the
    /// receipt_status_air cumulative-chain constraint holds). Only the
    /// first row is gated through D1 by `IS_FIRST`.
    const HONEST_SECOND_CUMUL_GAS: u64 = 84;

    /// D3 honest equality: `block_number = state_transition.slot`.
    const HONEST_SLOT: u64 = 100;

    // ─── Honest transactions ──────────────────────────────────────────

    fn tx0() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: 0,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        })
    }

    /// Conceptual second transaction in the block (see module docs).
    /// Currently unused by the AIR witness because `tx_full_chain_air`
    /// does not publish an `IS_FIRST` selector; the second tx will be
    /// added to the witness once first-tx anchoring is available.
    #[allow(dead_code)]
    fn tx1() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: 1,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x43u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x33u8; 32],
            s: [0x44u8; 32],
        })
    }

    fn honest_withdrawals() -> [Withdrawal; 2] {
        [
            Withdrawal {
                index: 0,
                validator_index: 7,
                address: [0x77u8; 20],
                amount: 1_000,
            },
            Withdrawal {
                index: 1,
                validator_index: 8,
                address: [0x88u8; 20],
                amount: 2_000,
            },
        ]
    }

    fn honest_receipts() -> [Receipt; 2] {
        [
            Receipt {
                ty: ReceiptType::Legacy,
                status: 1,
                cumulative_gas_used: HONEST_FIRST_CUMUL_GAS,
                logs_bloom: [0u8; 256],
                logs: Vec::new(),
            },
            Receipt {
                ty: ReceiptType::Legacy,
                status: 1,
                cumulative_gas_used: HONEST_SECOND_CUMUL_GAS,
                logs_bloom: [0u8; 256],
                logs: Vec::new(),
            },
        ]
    }

    // ─── Witness builders ─────────────────────────────────────────────

    /// Block header with the four cross-AIR single-byte / single-u64
    /// columns synthesised to match the downstream AIR witnesses
    /// (so each honest single-column tuple closure is equal across
    /// the linkage).
    fn synth_block_header() -> BlockHeader {
        // D0: transactions_root[0] = tx0.hash()[0].
        let tx0_hash = tx0().hash();
        let mut transactions_root = [0u8; 32];
        transactions_root[0] = tx0_hash[0];

        // D1: receipts_root[0] = first receipt cumulative_gas (≤ 255).
        let mut receipts_root = [0u8; 32];
        receipts_root[0] = HONEST_FIRST_CUMUL_GAS as u8;

        // D2: withdrawals_root = withdrawal_root_air's witness root
        // (so byte 0 of the column matches on both sides).
        let withdrawals = honest_withdrawals();
        let wr_root =
            crate::withdrawal_root_air::compute_withdrawals_root(&withdrawals);

        let mut h = BlockHeader::default();
        h.number = HONEST_SLOT; // D3
        h.timestamp = 1_700_000_000;
        h.gas_limit = 30_000_000;
        h.gas_used = HONEST_FIRST_CUMUL_GAS + (HONEST_SECOND_CUMUL_GAS - HONEST_FIRST_CUMUL_GAS);
        h.parent_hash = [0xAAu8; 32];
        h.state_root = [0xBBu8; 32];
        h.transactions_root = transactions_root;
        h.receipts_root = receipts_root;
        h.withdrawals_root = Some(wr_root);
        h
    }

    fn build_bh_witness() -> BlockHeaderWitness {
        let row = from_block_header(&synth_block_header());
        BlockHeaderWitness::from_headers(vec![row])
    }

    /// One-row tx_full_chain witness for the first transaction. The
    /// second tx is conceptually in the block but not present in this
    /// AIR witness (see module docs).
    fn build_tf_witness() -> TxFullChainWitness {
        TxFullChainWitness::from_transaction(
            0,
            &tx0(),
            [0u8; 20],
            HONEST_FIRST_CUMUL_GAS,
            HONEST_FIRST_CUMUL_GAS,
            1,
        )
    }

    fn build_rs_witness() -> ReceiptStatusWitness {
        ReceiptStatusWitness::from_receipts(&honest_receipts())
    }

    fn build_wr_witness() -> WithdrawalRootWitness {
        WithdrawalRootWitness::from_withdrawals(&honest_withdrawals())
    }

    fn build_st_witness() -> BeaconStateTransitionWitness {
        BeaconStateTransitionWitness::from_chain(
            [0u8; 32],
            &[(HONEST_SLOT, [0u8; 32], [0u8; 32])],
        )
    }

    // ─── Descriptor builders ──────────────────────────────────────────

    /// D0: `block_header_air.TRANSACTIONS_ROOT[0]` ↔
    /// `tx_full_chain_air.TX_HASH[0]`.
    fn descriptor_bh_to_tx() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_validity_master_bh_to_tx_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_TRANSACTIONS_ROOT_OFFSET],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![TF_COL_TX_HASH_OFFSET],
            b_selector_column: Some(TF_COL_IS_REAL),
        }
    }

    /// D1: `block_header_air.RECEIPTS_ROOT[0]` ↔
    /// `receipt_status_air.CUMULATIVE_GAS` (first row, gated by
    /// `IS_FIRST`).
    fn descriptor_bh_to_rs() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_validity_master_bh_to_rs_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_RECEIPTS_ROOT_OFFSET],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![RS_COL_CUMULATIVE_GAS],
            b_selector_column: Some(RS_COL_IS_FIRST),
        }
    }

    /// D2: `block_header_air.WITHDRAWALS_ROOT[0]` ↔
    /// `withdrawal_root_air.WITHDRAWALS_ROOT[0]` (first row, gated by
    /// `IS_FIRST`).
    fn descriptor_bh_to_wr() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_validity_master_bh_to_wr_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_WITHDRAWALS_ROOT_OFFSET],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 3,
            b_columns: vec![WR_COL_WITHDRAWALS_ROOT_OFFSET],
            b_selector_column: Some(WR_COL_IS_FIRST),
        }
    }

    /// D3: `block_header_air.NUMBER` ↔
    /// `beacon_state_transition_air.SLOT` (modulo conversion: identity
    /// on u64).
    fn descriptor_bh_to_st() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_validity_master_bh_to_st_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_NUMBER],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 4,
            b_columns: vec![ST_COL_SLOT],
            b_selector_column: Some(ST_COL_IS_REAL),
        }
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static descriptor + 5-trace shape sanity check. Validates every
    /// published descriptor's column constants, selectors, layer
    /// indices, and the honest single-column tuple equalities on the
    /// active rows. Does **not** call `joint_prove`, so runs in CI.
    #[test]
    fn descriptor_consistency_block_validity_master() {
        let d0 = descriptor_bh_to_tx();
        let d1 = descriptor_bh_to_rs();
        let d2 = descriptor_bh_to_wr();
        let d3 = descriptor_bh_to_st();

        // Single-column tuples (joint_prove invariant).
        for d in [&d0, &d1, &d2, &d3] {
            assert_eq!(d.a_columns.len(), 1);
            assert_eq!(d.b_columns.len(), 1);
        }
        // Hub layer is 0; descriptors bind layer 0 ↔ {1, 2, 3, 4}.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d2.a_layer_index, 0);
        assert_eq!(d2.b_layer_index, 3);
        assert_eq!(d3.a_layer_index, 0);
        assert_eq!(d3.b_layer_index, 4);

        // Selectors are wired on both sides per task spec.
        assert_eq!(d0.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(TF_COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(RS_COL_IS_FIRST));
        assert_eq!(d2.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d2.b_selector_column, Some(WR_COL_IS_FIRST));
        assert_eq!(d3.a_selector_column, Some(BH_COL_IS_REAL));
        assert_eq!(d3.b_selector_column, Some(ST_COL_IS_REAL));

        // Column indices reference each AIR's published COL_* constants,
        // so any column-layout drift in a sub-AIR surfaces here.
        assert_eq!(d0.a_columns[0], BH_COL_TRANSACTIONS_ROOT_OFFSET);
        assert_eq!(d0.b_columns[0], TF_COL_TX_HASH_OFFSET);
        assert_eq!(d1.a_columns[0], BH_COL_RECEIPTS_ROOT_OFFSET);
        assert_eq!(d1.b_columns[0], RS_COL_CUMULATIVE_GAS);
        assert_eq!(d2.a_columns[0], BH_COL_WITHDRAWALS_ROOT_OFFSET);
        assert_eq!(d2.b_columns[0], WR_COL_WITHDRAWALS_ROOT_OFFSET);
        assert_eq!(d3.a_columns[0], BH_COL_NUMBER);
        assert_eq!(d3.b_columns[0], ST_COL_SLOT);

        // Build all 5 per-AIR witnesses + traces at BLS48-581 and
        // confirm the 5-trace orchestrator input is constructible.
        let curve = CurveType::Bls48581;
        let bh_w = build_bh_witness();
        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let wr_w = build_wr_witness();
        let st_w = build_st_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tf = build_tf_trace(&tf_w, curve);
        let trace_rs = build_rs_trace(&rs_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);
        let trace_st = build_st_trace(&st_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tf = TxFullChainConstraintSystem::new(trace_tf.num_rows);
        let cs_rs = ReceiptStatusConstraintSystem::new(trace_rs.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);
        let cs_st = BeaconStateTransitionConstraintSystem::new(trace_st.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tf, &cs_tf),
            (&trace_rs, &cs_rs),
            (&trace_wr, &cs_wr),
            (&trace_st, &cs_st),
        ];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone(), d3.clone()];

        assert_eq!(traces.len(), 5);
        assert_eq!(linkages.len(), 4);

        // `joint_prove` ingress bounds checks.
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Witness shape: 2 txs, 2 withdrawals per task spec.
        assert_eq!(wr_w.rows.len(), 2);
        assert_eq!(rs_w.rows.len(), 2);
        // tx_full_chain holds the first tx only (see module docs).
        assert_eq!(tf_w.rows.len(), 1);

        // ─── Honest single-column tuple equalities ──────────────────

        // D0: header.transactions_root[0] == tx0.hash()[0].
        let tx0_hash_byte0 = tx0().hash()[0] as u64;
        assert_eq!(
            trace_bh.columns[BH_COL_TRANSACTIONS_ROOT_OFFSET].evaluations[0].to_u64(),
            tx0_hash_byte0,
        );
        assert_eq!(
            trace_tf.columns[TF_COL_TX_HASH_OFFSET].evaluations[0].to_u64(),
            tx0_hash_byte0,
        );

        // D1: header.receipts_root[0] == receipt[0].cumulative_gas_used.
        assert_eq!(
            trace_bh.columns[BH_COL_RECEIPTS_ROOT_OFFSET].evaluations[0].to_u64(),
            HONEST_FIRST_CUMUL_GAS,
        );
        assert_eq!(
            trace_rs.columns[RS_COL_CUMULATIVE_GAS].evaluations[0].to_u64(),
            HONEST_FIRST_CUMUL_GAS,
        );
        // IS_FIRST=1 on row 0, 0 on row 1 (gating the first receipt).
        assert_eq!(
            trace_rs.columns[RS_COL_IS_FIRST].evaluations[0].to_u64(),
            1,
        );
        assert_eq!(
            trace_rs.columns[RS_COL_IS_FIRST].evaluations[1].to_u64(),
            0,
        );

        // D2: header.withdrawals_root[0] == wr_witness.withdrawals_root[0].
        let wr_root_byte0 = wr_w.withdrawals_root[0] as u64;
        assert_eq!(
            trace_bh.columns[BH_COL_WITHDRAWALS_ROOT_OFFSET].evaluations[0].to_u64(),
            wr_root_byte0,
        );
        assert_eq!(
            trace_wr.columns[WR_COL_WITHDRAWALS_ROOT_OFFSET].evaluations[0].to_u64(),
            wr_root_byte0,
        );
        // IS_FIRST=1 on row 0, 0 on row 1 (gating the first withdrawal).
        assert_eq!(
            trace_wr.columns[WR_COL_IS_FIRST].evaluations[0].to_u64(),
            1,
        );
        assert_eq!(
            trace_wr.columns[WR_COL_IS_FIRST].evaluations[1].to_u64(),
            0,
        );

        // D3: header.number == state_transition.slot.
        assert_eq!(
            trace_bh.columns[BH_COL_NUMBER].evaluations[0].to_u64(),
            HONEST_SLOT,
        );
        assert_eq!(
            trace_st.columns[ST_COL_SLOT].evaluations[0].to_u64(),
            HONEST_SLOT,
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Honest 5-AIR `joint_prove` + `joint_verify` round-trip with all
    /// 4 cross-AIR LogUp descriptors. Run via `cargo test --release
    /// --ignored honest_block_validity_master_joint_verify_true`.
    /// Expected runtime is in the high hundreds to low thousands of
    /// seconds on BLS48-581 (5 per-AIR proves + 4 per-linkage SNARK
    /// proves + 4 cross-trace KZG opens).
    #[test]
    #[ignore = "slow: 5-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_block_validity_master_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bh_w = build_bh_witness();
        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let wr_w = build_wr_witness();
        let st_w = build_st_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tf = build_tf_trace(&tf_w, curve);
        let trace_rs = build_rs_trace(&rs_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);
        let trace_st = build_st_trace(&st_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tf = TxFullChainConstraintSystem::new(trace_tf.num_rows);
        let cs_rs = ReceiptStatusConstraintSystem::new(trace_rs.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);
        let cs_st = BeaconStateTransitionConstraintSystem::new(trace_st.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tf, &cs_tf),
            (&trace_rs, &cs_rs),
            (&trace_wr, &cs_wr),
            (&trace_st, &cs_st),
        ];
        let linkages = vec![
            descriptor_bh_to_tx(),
            descriptor_bh_to_rs(),
            descriptor_bh_to_wr(),
            descriptor_bh_to_st(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 5-AIR joint_prove must succeed");

        assert_eq!(proofs.len(), 5);
        assert_eq!(ext.linkage_proofs.len(), 4);

        // Honest closure equality across every descriptor.
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_bh, &cs_tf, &cs_rs, &cs_wr, &cs_st];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 5-AIR joint_verify must accept matching tuples on all four descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper `closure_a` on D2 (the withdrawals-root linkage) and
    /// confirm `joint_verify` rejects.
    #[test]
    #[ignore = "slow: depends on honest_block_validity_master_joint_verify_true setup"]
    fn tampered_block_validity_master_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bh_w = build_bh_witness();
        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let wr_w = build_wr_witness();
        let st_w = build_st_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tf = build_tf_trace(&tf_w, curve);
        let trace_rs = build_rs_trace(&rs_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);
        let trace_st = build_st_trace(&st_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tf = TxFullChainConstraintSystem::new(trace_tf.num_rows);
        let cs_rs = ReceiptStatusConstraintSystem::new(trace_rs.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);
        let cs_st = BeaconStateTransitionConstraintSystem::new(trace_st.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tf, &cs_tf),
            (&trace_rs, &cs_rs),
            (&trace_wr, &cs_wr),
            (&trace_st, &cs_st),
        ];
        let linkages = vec![
            descriptor_bh_to_tx(),
            descriptor_bh_to_rs(),
            descriptor_bh_to_wr(),
            descriptor_bh_to_st(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed on honest inputs");

        // Tamper descriptor index 2 (D2 — withdrawals-root) closure_a.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[2].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_bh, &cs_tf, &cs_rs, &cs_wr, &cs_st];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched closures on D2",
        );
    }
}
