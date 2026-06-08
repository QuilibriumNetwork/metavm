//! Multi-transaction `joint_prove` / `joint_verify` integration scaffold
//! for the [`crate::tx_full_chain_air`] composer.
//!
//! Task #242: extend [`crate::integration_tx_full_chain_joint_prove`]
//! (the single-tx 3-AIR scaffold) to a **3-transaction block** that
//! composes three `tx_full_chain_air` instances against one shared
//! [`crate::receipt_status_air`] (with the cross-tx **cumulative gas**
//! chain) and one shared [`crate::tx_nonce_air`] (with the per-sender
//! **nonce monotonicity** chain).
//!
//! ## AIR layer layout (5 traces)
//!
//!   - Layer 0: `tx_full_chain[0]` — composer witness for `tx[0]`
//!     (`tx_index = 0`, `nonce = 10`, `gas_used = 21_000`,
//!     `cumulative_gas_used = 21_000`).
//!   - Layer 1: `tx_full_chain[1]` — composer witness for `tx[1]`
//!     (`tx_index = 1`, `nonce = 11`, `gas_used = 30_000`,
//!     `cumulative_gas_used = 51_000`).
//!   - Layer 2: `tx_full_chain[2]` — composer witness for `tx[2]`
//!     (`tx_index = 2`, `nonce = 12`, `gas_used = 25_000`,
//!     `cumulative_gas_used = 76_000`).
//!   - Layer 3: `receipt_status` — 3-row witness built from the full
//!     `[Receipt; 3]` list. The AIR's `NUM_SHIFTED = 1` cross-row
//!     constraint algebraically pins
//!     `prev_cumulative_gas_used[i+1] = cumulative_gas_used[i]`, so
//!     the **cumulative gas chain** is enforced row-locally /
//!     row-shifted inside this one AIR.
//!   - Layer 4: `tx_nonce` — 3-row witness from the **same sender** with
//!     `pre_account_nonce = 10, 11, 12` and `tx_nonce = pre` on every
//!     row. Per-row `tx_nonce == pre` + `post = pre + 1` are enforced
//!     by the AIR's in-row constraints. Host-side, the witness encodes
//!     the **monotone nonce chain** by stepping `pre` by `+1` per row.
//!
//! ## Cross-AIR LogUp descriptors (4 single-column bindings)
//!
//!   - **D0** (`tx_full_chain[0].tx_index ↔ receipt_status.tx_index`)
//!     — gated by `IS_REAL` on both sides.
//!   - **D1** (`tx_full_chain[1].tx_index ↔ receipt_status.tx_index`)
//!     — gated by `IS_REAL` on both sides.
//!   - **D2** (`tx_full_chain[2].tx_index ↔ receipt_status.tx_index`)
//!     — gated by `IS_REAL` on both sides.
//!     Together D0+D1+D2 force the receipt-status `tx_index` multiset
//!     to contain `{0, 1, 2}`, anchoring the cumulative gas chain to
//!     the per-tx composer witnesses.
//!   - **D3** (`tx_full_chain[0].nonce ↔ tx_nonce.tx_nonce`) — gated by
//!     `IS_REAL` on both sides. Single-column anchor of the tx_nonce
//!     monotone chain. The 3-row tx_nonce witness encodes
//!     `nonce[i] = base + i` host-side; the AIR's `nonce_matches_pre`
//!     constraint enforces `tx_nonce == pre` per row, and the
//!     descriptor multiset only needs to contain D3's anchor value
//!     (`nonce[0] = 10`).
//!
//! ## Why single-column descriptors
//!
//! The current `cross_air_logup::joint_prove` API algebraically
//! supports only single-column tuple descriptors (see the assertion
//! documented at `cross_air_logup::build_linkage_trace`). The composer's
//! own multi-column descriptors (4-col, 22-col, …) are validated
//! statically in the fast `descriptor_consistency_multi_tx_full_chain`
//! test but are **not** wired through `joint_prove` here. Mirrors the
//! pattern from [`crate::integration_tx_full_chain_joint_prove`].
//!
//! ## Test coverage
//!
//!   - **Fast** `descriptor_consistency_multi_tx_full_chain` — verifies
//!     witness shapes, column constants, and the cumulative + nonce
//!     monotone chains in the assembled (un-proven) 5-AIR input.
//!   - **Slow (ignored)** `honest_multi_tx_full_chain_joint_verify_true`
//!     — 5-AIR `joint_prove` + `joint_verify` round-trip on BLS48-581.
//!   - **Slow (ignored)** `tampered_multi_tx_full_chain_joint_verify_false`
//!     — overwrites `closure_a` on the D3 nonce linkage; verifier must
//!     reject.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air::{
        build_trace_polynomials as build_rs_trace, ReceiptStatusConstraintSystem,
        ReceiptStatusWitness, COL_CUMULATIVE_GAS as RS_COL_CUMULATIVE_GAS,
        COL_IS_REAL as RS_COL_IS_REAL, COL_PREV_CUMULATIVE_GAS as RS_COL_PREV_CUMULATIVE_GAS,
        COL_TX_INDEX as RS_COL_TX_INDEX,
    };
    use crate::transaction::{LegacyTx, Transaction};
    use crate::tx_full_chain_air::{
        build_trace_polynomials as build_tf_trace, TxFullChainConstraintSystem,
        TxFullChainWitness, ADDR_LEN, COL_IS_REAL as TF_COL_IS_REAL, COL_NONCE as TF_COL_NONCE,
        COL_TX_INDEX as TF_COL_TX_INDEX,
    };
    use crate::tx_nonce_air::{
        build_trace_polynomials as build_tn_trace, TxNonceConstraintSystem, TxNonceRow,
        TxNonceWitness, COL_IS_REAL as TN_COL_IS_REAL, COL_TX_NONCE as TN_COL_TX_NONCE,
    };

    // ─── Honest 3-tx witness parameters ──────────────────────────────────

    /// Shared sender for the per-sender nonce monotonicity chain.
    const HONEST_SENDER: [u8; ADDR_LEN] = [0xabu8; ADDR_LEN];

    /// Per-tx parameters: (tx_index, nonce, gas_used, cumulative_gas).
    ///
    /// Cumulative chain (algebraically enforced by receipt_status_air):
    ///   cum[0] = gas[0]                      = 21_000
    ///   cum[1] = cum[0] + gas[1] = 21k + 30k = 51_000
    ///   cum[2] = cum[1] + gas[2] = 51k + 25k = 76_000
    ///
    /// Nonce monotone chain (host-side ordering + tx_nonce_air per-row
    /// `tx_nonce == pre` + `post = pre + 1` constraints):
    ///   pre[0] = 10, pre[1] = 11, pre[2] = 12.
    const TX_PARAMS: [(u64, u64, u64, u64); 3] = [
        (0, 10, 21_000, 21_000),
        (1, 11, 30_000, 51_000),
        (2, 12, 25_000, 76_000),
    ];

    fn legacy_tx_for(nonce: u64) -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 27,
            r: [1u8; 32],
            s: [2u8; 32],
        })
    }

    fn build_tf_witness(i: usize) -> TxFullChainWitness {
        let (tx_index, nonce, gas_used, cumulative) = TX_PARAMS[i];
        TxFullChainWitness::from_transaction(
            tx_index,
            &legacy_tx_for(nonce),
            HONEST_SENDER,
            gas_used,
            cumulative,
            1,
        )
    }

    fn build_rs_witness() -> ReceiptStatusWitness {
        // The 3 receipts mirror TX_PARAMS' cumulative chain. The
        // `from_receipts` builder derives `prev_cumulative_gas_used`
        // from row position, so the AIR's shifted constraint
        // `prev[i+1] = cumulative[i]` is honestly satisfied.
        let receipts: Vec<Receipt> = TX_PARAMS
            .iter()
            .map(|(_, _, _, cum)| Receipt {
                ty: ReceiptType::Legacy,
                status: 1,
                cumulative_gas_used: *cum,
                logs_bloom: [0u8; 256],
                logs: Vec::new(),
            })
            .collect();
        ReceiptStatusWitness::from_receipts(&receipts)
    }

    fn build_tn_witness() -> TxNonceWitness {
        // 3-row tx_nonce witness from the same sender with monotone
        // pre-account-nonces 10, 11, 12 (each row also satisfies
        // `tx_nonce == pre` and `post = pre + 1`).
        let rows: Vec<TxNonceRow> = TX_PARAMS
            .iter()
            .map(|(tx_index, nonce, _, _)| TxNonceRow {
                sender_address: HONEST_SENDER,
                tx_nonce: *nonce,
                pre_account_nonce: *nonce,
                post_account_nonce: *nonce + 1,
                tx_index: *tx_index,
            })
            .collect();
        TxNonceWitness::from_rows(rows)
    }

    // ─── Descriptors used by the joint-prove scaffold ───────────────────

    /// Single-column descriptor binding
    /// `tx_full_chain[layer_a].COL_TX_INDEX` (gated by `IS_REAL`)
    /// to `receipt_status.COL_TX_INDEX` (layer 3, gated by `IS_REAL`).
    fn tx_index_descriptor(layer_a: usize, label: &str) -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: label.into(),
            a_layer_index: layer_a,
            a_columns: vec![TF_COL_TX_INDEX],
            a_selector_column: Some(TF_COL_IS_REAL),
            b_layer_index: 3,
            b_columns: vec![RS_COL_TX_INDEX],
            b_selector_column: Some(RS_COL_IS_REAL),
        }
    }

    /// D0: `tx_full_chain[0].tx_index ↔ receipt_status.tx_index`.
    fn d0_descriptor() -> CrossAirLogUpDescriptor {
        tx_index_descriptor(0, "multi_tx_full_chain_to_receipt_status_tx_index_0_v1")
    }
    /// D1: `tx_full_chain[1].tx_index ↔ receipt_status.tx_index`.
    fn d1_descriptor() -> CrossAirLogUpDescriptor {
        tx_index_descriptor(1, "multi_tx_full_chain_to_receipt_status_tx_index_1_v1")
    }
    /// D2: `tx_full_chain[2].tx_index ↔ receipt_status.tx_index`.
    fn d2_descriptor() -> CrossAirLogUpDescriptor {
        tx_index_descriptor(2, "multi_tx_full_chain_to_receipt_status_tx_index_2_v1")
    }

    /// D3: `tx_full_chain[0].nonce ↔ tx_nonce.tx_nonce`. Single-column
    /// anchor of the tx_nonce monotone chain (the 3-row tx_nonce
    /// witness encodes the chain; the AIR's per-row `nonce_matches_pre`
    /// constraint pins `tx_nonce == pre` on every row).
    fn d3_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "multi_tx_full_chain_to_tx_nonce_monotone_chain_v1".into(),
            a_layer_index: 0,
            a_columns: vec![TF_COL_NONCE],
            a_selector_column: Some(TF_COL_IS_REAL),
            b_layer_index: 4,
            b_columns: vec![TN_COL_TX_NONCE],
            b_selector_column: Some(TN_COL_IS_REAL),
        }
    }

    // ─── Fast static checks (un-ignored) ─────────────────────────────────

    /// Validates the 5-AIR witness shapes + descriptor wiring + the
    /// cumulative-gas and nonce-monotone chains in the assembled (but
    /// un-proven) joint input. Catches column-layout drift in any of
    /// the 3 composing AIRs. Does NOT run `joint_prove`, so well under
    /// the CI budget.
    #[test]
    fn descriptor_consistency_multi_tx_full_chain() {
        // ─── 1. Build the 5-AIR traces ───────────────────────────────
        let curve = CurveType::Bls48581;
        let tf_w0 = build_tf_witness(0);
        let tf_w1 = build_tf_witness(1);
        let tf_w2 = build_tf_witness(2);
        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness();

        let trace_0 = build_tf_trace(&tf_w0, curve);
        let trace_1 = build_tf_trace(&tf_w1, curve);
        let trace_2 = build_tf_trace(&tf_w2, curve);
        let trace_3 = build_rs_trace(&rs_w, curve);
        let trace_4 = build_tn_trace(&tn_w, curve);

        let cs_0 = TxFullChainConstraintSystem::new(trace_0.num_rows);
        let cs_1 = TxFullChainConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxFullChainConstraintSystem::new(trace_2.num_rows);
        let cs_3 = ReceiptStatusConstraintSystem::new(trace_3.num_rows);
        let cs_4 = TxNonceConstraintSystem::new(trace_4.num_rows);

        // ─── 2. Per-tx tx_full_chain witnesses encode the expected
        //     (tx_index, nonce, gas_used, cumulative_gas). ──────────
        for (i, w) in [&tf_w0, &tf_w1, &tf_w2].iter().enumerate() {
            let (expected_tx_index, expected_nonce, expected_gas, expected_cum) = TX_PARAMS[i];
            assert_eq!(w.rows.len(), 1);
            let row = &w.rows[0];
            assert_eq!(row.tx_index, expected_tx_index);
            assert_eq!(row.nonce, expected_nonce);
            assert_eq!(row.gas_used, expected_gas);
            assert_eq!(row.cumulative_gas_used, expected_cum);
            assert_eq!(row.sender_address, HONEST_SENDER);
        }

        // ─── 3. receipt_status witness encodes the cumulative-gas
        //     chain `cum[i] = cum[i-1] + gas[i]`. ────────────────────
        assert_eq!(rs_w.rows.len(), 3);
        let mut running = 0u64;
        for (i, row) in rs_w.rows.iter().enumerate() {
            let (expected_tx_index, _, expected_gas, expected_cum) = TX_PARAMS[i];
            assert_eq!(row.tx_index, expected_tx_index);
            assert_eq!(row.gas_used, expected_gas);
            assert_eq!(row.prev_cumulative_gas_used, running);
            running = running.checked_add(expected_gas).expect("gas sum fits in u64");
            assert_eq!(row.cumulative_gas_used, expected_cum);
            assert_eq!(row.cumulative_gas_used, running);
        }
        // Final cumulative matches the published last-tx cumulative.
        assert_eq!(running, TX_PARAMS[2].3);

        // ─── 4. tx_nonce witness encodes the per-sender monotone
        //     nonce chain. ──────────────────────────────────────────
        assert_eq!(tn_w.rows.len(), 3);
        for (i, row) in tn_w.rows.iter().enumerate() {
            let (_, expected_nonce, _, _) = TX_PARAMS[i];
            assert_eq!(row.sender_address, HONEST_SENDER);
            assert_eq!(row.tx_nonce, expected_nonce);
            assert_eq!(row.pre_account_nonce, expected_nonce);
            assert_eq!(row.post_account_nonce, expected_nonce + 1);
            if i > 0 {
                // Monotone chain: pre[i] = pre[i-1] + 1.
                assert_eq!(row.pre_account_nonce, tn_w.rows[i - 1].pre_account_nonce + 1);
            }
        }

        // ─── 5. Descriptors land on the expected layers / columns. ──
        let d0 = d0_descriptor();
        let d1 = d1_descriptor();
        let d2 = d2_descriptor();
        let d3 = d3_descriptor();

        for d in [&d0, &d1, &d2] {
            assert_eq!(d.a_columns.len(), 1);
            assert_eq!(d.b_columns.len(), 1);
            assert_eq!(d.a_columns[0], TF_COL_TX_INDEX);
            assert_eq!(d.b_columns[0], RS_COL_TX_INDEX);
            assert_eq!(d.b_layer_index, 3);
            assert_eq!(d.a_selector_column, Some(TF_COL_IS_REAL));
            assert_eq!(d.b_selector_column, Some(RS_COL_IS_REAL));
            assert_ne!(d.a_layer_index, d.b_layer_index);
        }
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d1.a_layer_index, 1);
        assert_eq!(d2.a_layer_index, 2);

        assert_eq!(d3.a_columns.len(), 1);
        assert_eq!(d3.b_columns.len(), 1);
        assert_eq!(d3.a_columns[0], TF_COL_NONCE);
        assert_eq!(d3.b_columns[0], TN_COL_TX_NONCE);
        assert_eq!(d3.a_layer_index, 0);
        assert_eq!(d3.b_layer_index, 4);
        assert_eq!(d3.a_selector_column, Some(TF_COL_IS_REAL));
        assert_eq!(d3.b_selector_column, Some(TN_COL_IS_REAL));

        // ─── 6. Assemble the 5-AIR joint input (without proving). ───
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
            (&trace_4, &cs_4),
        ];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone(), d3.clone()];

        assert_eq!(traces.len(), 5);
        assert_eq!(linkages.len(), 4);
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // ─── 7. Cross-layer alignment between tx_full_chain rows
        //     and the receipt-status / tx_nonce shared traces. ─────
        // tx_index alignment: tf[i].tx_index == rs.row[i].tx_index.
        for i in 0..3 {
            let tf_trace = [&trace_0, &trace_1, &trace_2][i];
            assert_eq!(
                tf_trace.columns[TF_COL_TX_INDEX].evaluations[0].to_u64(),
                TX_PARAMS[i].0,
            );
            assert_eq!(
                trace_3.columns[RS_COL_TX_INDEX].evaluations[i].to_u64(),
                TX_PARAMS[i].0,
            );
        }

        // Cumulative-gas chain visible in the receipt-status trace:
        // prev[i+1] = cumulative[i].
        for i in 0..2 {
            let cum_i = trace_3.columns[RS_COL_CUMULATIVE_GAS].evaluations[i].to_u64();
            let prev_i1 = trace_3.columns[RS_COL_PREV_CUMULATIVE_GAS].evaluations[i + 1].to_u64();
            assert_eq!(
                cum_i, prev_i1,
                "receipt_status_air shifted constraint requires cum[i] = prev[i+1]",
            );
        }

        // D3 anchor: tf[0].nonce == tn.row[0].tx_nonce.
        assert_eq!(
            trace_0.columns[TF_COL_NONCE].evaluations[0].to_u64(),
            TX_PARAMS[0].1,
        );
        assert_eq!(
            trace_4.columns[TN_COL_TX_NONCE].evaluations[0].to_u64(),
            TX_PARAMS[0].1,
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 5-AIR `joint_prove` + `joint_verify` round-trip for
    /// the multi-tx tx_full_chain ↔ receipt_status ↔ tx_nonce chain.
    ///
    /// `#[ignore]` because `joint_prove` runs 5× per-AIR
    /// `prove_with_scheme` + 4× per-linkage SNARK + KZG opens on
    /// BLS48-581 — comfortably > 120 s in release on most hardware.
    /// Run via `cargo test --release --ignored
    /// honest_multi_tx_full_chain_joint_verify_true`.
    #[test]
    #[ignore = "slow: 5-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_multi_tx_full_chain_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let tf_w0 = build_tf_witness(0);
        let tf_w1 = build_tf_witness(1);
        let tf_w2 = build_tf_witness(2);
        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness();

        let trace_0 = build_tf_trace(&tf_w0, curve);
        let trace_1 = build_tf_trace(&tf_w1, curve);
        let trace_2 = build_tf_trace(&tf_w2, curve);
        let trace_3 = build_rs_trace(&rs_w, curve);
        let trace_4 = build_tn_trace(&tn_w, curve);

        let cs_0 = TxFullChainConstraintSystem::new(trace_0.num_rows);
        let cs_1 = TxFullChainConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxFullChainConstraintSystem::new(trace_2.num_rows);
        let cs_3 = ReceiptStatusConstraintSystem::new(trace_3.num_rows);
        let cs_4 = TxNonceConstraintSystem::new(trace_4.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
            (&trace_4, &cs_4),
        ];
        let linkages = vec![
            d0_descriptor(),
            d1_descriptor(),
            d2_descriptor(),
            d3_descriptor(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 5-AIR joint_prove must succeed for multi-tx tx_full_chain");

        assert_eq!(proofs.len(), 5);
        assert_eq!(ext.linkage_proofs.len(), 4);

        // Honest closure-equality across all 4 descriptors.
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3, &cs_4];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 5-AIR joint_verify must accept matching tuples on all 4 descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tampered closure: overwrite `closure_a` on descriptor D3 (the
    /// nonce monotone-chain linkage) after a successful `joint_prove`.
    /// Verifier must reject.
    ///
    /// `#[ignore]` because the setup half is the same `joint_prove`
    /// invocation as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_multi_tx_full_chain_joint_verify_true"]
    fn tampered_multi_tx_full_chain_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let tf_w0 = build_tf_witness(0);
        let tf_w1 = build_tf_witness(1);
        let tf_w2 = build_tf_witness(2);
        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness();

        let trace_0 = build_tf_trace(&tf_w0, curve);
        let trace_1 = build_tf_trace(&tf_w1, curve);
        let trace_2 = build_tf_trace(&tf_w2, curve);
        let trace_3 = build_rs_trace(&rs_w, curve);
        let trace_4 = build_tn_trace(&tn_w, curve);

        let cs_0 = TxFullChainConstraintSystem::new(trace_0.num_rows);
        let cs_1 = TxFullChainConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxFullChainConstraintSystem::new(trace_2.num_rows);
        let cs_3 = ReceiptStatusConstraintSystem::new(trace_3.num_rows);
        let cs_4 = TxNonceConstraintSystem::new(trace_4.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
            (&trace_4, &cs_4),
        ];
        let linkages = vec![
            d0_descriptor(),
            d1_descriptor(),
            d2_descriptor(),
            d3_descriptor(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 5-AIR joint_prove must succeed for multi-tx tx_full_chain");

        // Tamper with descriptor D3's closure_a (the nonce linkage).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[3].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3, &cs_4];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the D3 nonce descriptor",
        );
    }
}
