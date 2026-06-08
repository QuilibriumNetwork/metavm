//! Integration `joint_prove` / `joint_verify` test for the
//! [`crate::tx_full_chain_air`] composer.
//!
//! Task #151: scaffold a runnable joint-prove harness that composes a
//! single transaction's full validity bundle across the composer + at
//! least one downstream sub-AIR via the cross-AIR LogUp protocol.
//!
//! ## Scope
//!
//! The composer's full sub-AIR set is (see `tx_full_chain_air`):
//!
//!   - [`crate::tx_sender_recovery_air`] — ECDSA sender recovery,
//!   - [`crate::tx_nonce_air`] — nonce vs pre-account-nonce binding,
//!   - [`crate::access_list_air`] — EIP-2930/1559 access list,
//!   - [`crate::receipt_status_air`] — receipt status + gas accounting,
//!   - [`crate::tx_rlp_air`] — canonical RLP / wire encoding.
//!
//! The composer publishes 5 multi-column tuple descriptors (21 / 22 /
//! 20 / 4 / 1 columns wide). The current `cross_air_logup::joint_prove`
//! API algebraically supports only **single-column tuple descriptors**
//! (see the assertion documented at
//! `cross_air_logup::build_linkage_trace`). Wiring the full multi-column
//! descriptors is deferred to the multi-column tuple follow-up.
//!
//! ## What this file does today
//!
//! 1. The fast `descriptor_consistency` test validates every published
//!    composer descriptor (shape, label, column constants, selectors)
//!    plus the two single-column descriptors used by the joint-prove
//!    scaffold.
//!
//! 2. The `#[ignore]`'d slow tests scaffold a **3-AIR chain**:
//!    `tx_full_chain (layer 0) ↔ receipt_status (layer 1) ↔ tx_nonce
//!    (layer 2)` with two single-column descriptors:
//!      - D0 (`tx_index`): `tx_full_chain.COL_TX_INDEX` ↔
//!        `receipt_status.COL_TX_INDEX`,
//!      - D1 (`nonce`): `tx_full_chain.COL_NONCE` ↔
//!        `tx_nonce.COL_TX_NONCE`.
//!    Mirrors the canonical 3-AIR template
//!    [`crate::integration_joint_prove_three_air`].
//!
//! ## Witness alignment
//!
//!   - `receipt_status` builds `tx_index` from row position, so a
//!     single-row witness forces `tx_index = 0`. Composer and tx_nonce
//!     traces are constructed with `tx_index = 0` to match.
//!   - `tx_nonce_air::from_inputs` requires `tx_nonce == pre`, so we
//!     use a single nonce value on both the composer and tx_nonce sides.
//!
//! ## Deferred (follow-up tasks)
//!
//!   - Full 21-col composer ↔ tx_sender_recovery joint-prove pipeline
//!     (needs multi-column tuple support).
//!   - Full 22-col composer ↔ tx_nonce joint-prove (ditto).
//!   - Full 20-col composer ↔ access_list joint-prove (ditto).
//!   - Full 4-col composer ↔ receipt_status joint-prove (ditto).
//!   - 6-AIR end-to-end joint-prove with all 5 composer descriptors.
//!
//! ## Cross-references
//!
//!   - [`crate::integration_joint_prove_three_air`] for the 3-AIR
//!     single-column template this module mirrors.
//!   - [`crate::tx_full_chain_air`] for the composer module.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air::{
        build_trace_polynomials as build_rs_trace, ReceiptStatusConstraintSystem,
        ReceiptStatusWitness, COL_IS_REAL as RS_COL_IS_REAL,
        COL_TX_INDEX as RS_COL_TX_INDEX,
    };
    use crate::transaction::{LegacyTx, Transaction};
    use crate::tx_full_chain_air::{
        build_trace_polynomials as build_tf_trace, make_tx_full_to_access_list_descriptor,
        make_tx_full_to_receipt_status_descriptor, make_tx_full_to_tx_nonce_descriptor,
        make_tx_full_to_tx_rlp_descriptor, make_tx_full_to_tx_sender_recovery_descriptor,
        TxFullChainConstraintSystem, TxFullChainWitness, ADDR_LEN, COL_IS_REAL as TF_COL_IS_REAL,
        COL_NONCE as TF_COL_NONCE, COL_TX_INDEX as TF_COL_TX_INDEX,
    };
    use crate::tx_nonce_air::{
        build_trace_polynomials as build_tn_trace, TxNonceConstraintSystem, TxNonceWitness,
        COL_IS_REAL as TN_COL_IS_REAL, COL_TX_NONCE as TN_COL_TX_NONCE,
    };

    // ─── Honest test inputs ──────────────────────────────────────────────

    /// `receipt_status_air` populates `COL_TX_INDEX` from row position,
    /// so a single-row receipt witness forces `tx_index = 0`. Both
    /// composer and tx_nonce sides are built with `tx_index = 0` to
    /// match the receipt-status anchor.
    const HONEST_TX_INDEX: u64 = 0;

    /// Honest nonce threaded through both composer and tx_nonce_air.
    /// `tx_nonce_air::from_inputs` requires `tx_nonce == pre`, so this
    /// single value seeds both fields.
    const HONEST_NONCE: u64 = 11;

    /// Honest sender address (composer commits 20 bytes; tx_nonce_air
    /// also commits 20 bytes but here the joint-prove descriptors only
    /// bind tx_index / nonce so the per-side sender bytes are
    /// independent).
    const HONEST_SENDER: [u8; ADDR_LEN] = [0xabu8; ADDR_LEN];

    fn legacy_tx() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: HONEST_NONCE,
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

    fn build_tf_witness() -> TxFullChainWitness {
        TxFullChainWitness::from_transaction(
            HONEST_TX_INDEX,
            &legacy_tx(),
            HONEST_SENDER,
            21_000,
            21_000,
            1,
        )
    }

    fn build_rs_witness() -> ReceiptStatusWitness {
        // Single-row receipt list → row position 0 → tx_index = 0.
        let receipts = vec![Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }];
        ReceiptStatusWitness::from_receipts(&receipts)
    }

    fn build_tn_witness() -> TxNonceWitness {
        // tx_nonce_air enforces tx_nonce == pre.
        TxNonceWitness::from_inputs([0u8; ADDR_LEN], HONEST_NONCE, HONEST_NONCE, HONEST_TX_INDEX)
    }

    // ─── Descriptors used by the joint-prove scaffold ───────────────────

    /// D0 (`tx_index`): `tx_full_chain.COL_TX_INDEX` (layer 0, gated by
    /// `TF_COL_IS_REAL`) ↔ `receipt_status.COL_TX_INDEX` (layer 1, gated
    /// by `RS_COL_IS_REAL`).
    fn tx_index_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "tx_full_chain_to_receipt_status_tx_index_v1".into(),
            a_layer_index: 0,
            a_columns: vec![TF_COL_TX_INDEX],
            a_selector_column: Some(TF_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![RS_COL_TX_INDEX],
            b_selector_column: Some(RS_COL_IS_REAL),
        }
    }

    /// D1 (`nonce`): `tx_full_chain.COL_NONCE` (layer 0, gated by
    /// `TF_COL_IS_REAL`) ↔ `tx_nonce.COL_TX_NONCE` (layer 2, gated by
    /// `TN_COL_IS_REAL`).
    fn nonce_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "tx_full_chain_to_tx_nonce_nonce_v1".into(),
            a_layer_index: 0,
            a_columns: vec![TF_COL_NONCE],
            a_selector_column: Some(TF_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![TN_COL_TX_NONCE],
            b_selector_column: Some(TN_COL_IS_REAL),
        }
    }

    // ─── Fast static checks (un-ignored) ─────────────────────────────────

    /// Verifies every published composer descriptor + the two
    /// single-column descriptors used by the slow joint-prove tests.
    /// Catches column-layout drift in the composer or any downstream
    /// sub-AIR. Does NOT run `joint_prove`, so well under the CI budget.
    #[test]
    fn descriptor_consistency_tx_full_chain() {
        use crate::access_list_air as al;
        use crate::receipt_status_air as rs;
        use crate::tx_nonce_air as tn;
        use crate::tx_rlp_air as txr;
        use crate::tx_sender_recovery_air as sr;

        // ─── 1. Composer's full multi-column descriptors ─────────────
        let d_sr = make_tx_full_to_tx_sender_recovery_descriptor(0, 1);
        assert_eq!(d_sr.label, "tx_full_to_tx_sender_recovery_v1");
        assert_eq!(d_sr.a_columns.len(), 1 + ADDR_LEN);
        assert_eq!(d_sr.b_columns.len(), 1 + ADDR_LEN);
        assert_eq!(d_sr.b_columns[0], sr::COL_TX_INDEX);
        assert_eq!(d_sr.b_selector_column, Some(sr::COL_IS_REAL));

        let d_tn = make_tx_full_to_tx_nonce_descriptor(0, 2);
        assert_eq!(d_tn.label, "tx_full_to_tx_nonce_v1");
        assert_eq!(d_tn.a_columns.len(), 1 + ADDR_LEN + 1);
        assert_eq!(d_tn.b_columns.len(), 1 + ADDR_LEN + 1);
        assert_eq!(d_tn.b_columns[0], tn::COL_TX_INDEX);
        assert_eq!(d_tn.b_columns[1 + ADDR_LEN], tn::COL_TX_NONCE);

        let d_al = make_tx_full_to_access_list_descriptor(0, 3);
        assert_eq!(d_al.label, "tx_full_to_access_list_v1");
        assert_eq!(d_al.a_columns.len(), ADDR_LEN);
        assert_eq!(d_al.b_columns.len(), ADDR_LEN);
        assert_eq!(d_al.b_columns[0], al::COL_ADDRESS_OFFSET);

        let d_rs = make_tx_full_to_receipt_status_descriptor(0, 4);
        assert_eq!(d_rs.label, "tx_full_to_receipt_status_v1");
        assert_eq!(d_rs.a_columns.len(), 4);
        assert_eq!(d_rs.b_columns.len(), 4);
        assert_eq!(d_rs.b_columns[0], rs::COL_TX_INDEX);

        let d_rlp = make_tx_full_to_tx_rlp_descriptor(0, 5);
        assert_eq!(d_rlp.label, "tx_full_to_tx_rlp_v1");
        assert_eq!(d_rlp.a_columns, vec![TF_COL_NONCE]);
        assert_eq!(d_rlp.b_columns, vec![txr::COL_NONCE]);

        // ─── 2. Single-column descriptors used by joint_prove ───────
        let d0 = tx_index_descriptor();
        let d1 = nonce_descriptor();

        assert_eq!(d0.a_columns.len(), 1);
        assert_eq!(d0.b_columns.len(), 1);
        assert_eq!(d1.a_columns.len(), 1);
        assert_eq!(d1.b_columns.len(), 1);

        // 3-AIR chain layer indices: D0 wires 0↔1, D1 wires 0↔2.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_ne!(d0.a_layer_index, d0.b_layer_index);
        assert_ne!(d1.a_layer_index, d1.b_layer_index);

        assert_eq!(d0.a_columns[0], TF_COL_TX_INDEX);
        assert_eq!(d0.b_columns[0], RS_COL_TX_INDEX);
        assert_eq!(d1.a_columns[0], TF_COL_NONCE);
        assert_eq!(d1.b_columns[0], TN_COL_TX_NONCE);

        assert_eq!(d0.a_selector_column, Some(TF_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(RS_COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(TF_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(TN_COL_IS_REAL));

        // ─── 3. Assemble (without proving) the 3-AIR joint input ────
        let curve = CurveType::Bls48581;
        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness();
        let trace_0 = build_tf_trace(&tf_w, curve);
        let trace_1 = build_rs_trace(&rs_w, curve);
        let trace_2 = build_tn_trace(&tn_w, curve);
        let cs_0 = TxFullChainConstraintSystem::new(trace_0.num_rows);
        let cs_1 = ReceiptStatusConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxNonceConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![d0.clone(), d1.clone()];

        assert_eq!(traces.len(), 3);
        assert_eq!(linkages.len(), 2);
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Witness alignment: tx_index is honest 0 on all three sides.
        assert_eq!(
            trace_0.columns[TF_COL_TX_INDEX].evaluations[0].to_u64(),
            HONEST_TX_INDEX,
        );
        assert_eq!(
            trace_1.columns[RS_COL_TX_INDEX].evaluations[0].to_u64(),
            HONEST_TX_INDEX,
        );
        // Nonce is honest HONEST_NONCE on both composer and tx_nonce
        // sides.
        assert_eq!(
            trace_0.columns[TF_COL_NONCE].evaluations[0].to_u64(),
            HONEST_NONCE,
        );
        assert_eq!(
            trace_2.columns[TN_COL_TX_NONCE].evaluations[0].to_u64(),
            HONEST_NONCE,
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip for
    /// the tx_full_chain composer ↔ receipt_status ↔ tx_nonce chain
    /// (single-column bindings on tx_index and nonce).
    ///
    /// `#[ignore]` because `joint_prove` runs 3× per-AIR
    /// `prove_with_scheme` + 2× per-linkage SNARK + KZG opens on
    /// BLS48-581 — comfortably > 120 s in release on most hardware.
    /// Run via `cargo test --release --ignored
    /// honest_tx_full_chain_joint_verify_true`.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_tx_full_chain_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness();
        let trace_0 = build_tf_trace(&tf_w, curve);
        let trace_1 = build_rs_trace(&rs_w, curve);
        let trace_2 = build_tn_trace(&tn_w, curve);
        let cs_0 = TxFullChainConstraintSystem::new(trace_0.num_rows);
        let cs_1 = ReceiptStatusConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxNonceConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![tx_index_descriptor(), nonce_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed for tx_full_chain");

        assert_eq!(proofs.len(), 3);
        assert_eq!(ext.linkage_proofs.len(), 2);

        // Honest closure-equality across both descriptors.
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 3-AIR joint_verify must accept matching tuples on both descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tampered closure: overwrite `closure_a` on descriptor D1 (the
    /// nonce linkage) after a successful `joint_prove`. Verifier must
    /// reject.
    ///
    /// `#[ignore]` because the setup half is the same `joint_prove`
    /// invocation as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_tx_full_chain_joint_verify_true"]
    fn tampered_tx_full_chain_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness();
        let trace_0 = build_tf_trace(&tf_w, curve);
        let trace_1 = build_rs_trace(&rs_w, curve);
        let trace_2 = build_tn_trace(&tn_w, curve);
        let cs_0 = TxFullChainConstraintSystem::new(trace_0.num_rows);
        let cs_1 = ReceiptStatusConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxNonceConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![tx_index_descriptor(), nonce_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed for tx_full_chain");

        // Tamper with descriptor D1's closure_a (the nonce linkage).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the nonce descriptor",
        );
    }
}
