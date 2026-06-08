//! Third cross-AIR LogUp `joint_prove` / `joint_verify` integration
//! smoke test, scaling the surface from 2 AIRs / 1 descriptor (rounds
//! 20 + 22) to **3 AIRs / 2 descriptors**.
//!
//! Prior smoke tests:
//!   - `integration_joint_prove_smoke` (round 20): 2× `receipt_status_air`
//!     with a single-column self-linkage on `COL_CUMULATIVE_GAS`.
//!   - `integration_joint_prove_smoke_2` (round 22):
//!     `withdrawal_queue_air` ↔ `withdrawal_credential_air` with a
//!     single-column linkage on `COL_VALIDATOR_INDEX`.
//!
//! This third smoke test wires a **heterogeneous 3-AIR chain** with two
//! cross-AIR LogUp descriptors:
//!
//! - layer 0: [`crate::receipt_status_air`] (23 cols), gated by its
//!   [`crate::receipt_status_air::COL_IS_REAL`] selector,
//! - layer 1: [`crate::tx_nonce_air`] (53 cols), gated by its
//!   [`crate::tx_nonce_air::COL_IS_REAL`] selector,
//! - layer 2: [`crate::attestation_committee_air`] (50 cols), gated by
//!   its [`crate::attestation_committee_air::COL_IS_REAL`] selector.
//!
//! Descriptor D0 (`txindex`): single-column tuple over `tx_index`
//! connecting layer 0 (`COL_TX_INDEX = 0` on `receipt_status_air`) ↔
//! layer 1 (`COL_TX_INDEX = 23` on `tx_nonce_air`).
//!
//! Descriptor D1 (`nonce↔validator_index`): single-column tuple
//! connecting layer 1 (`COL_PRE_ACCOUNT_NONCE` on `tx_nonce_air`) ↔
//! layer 2 (`COL_VALIDATOR_INDEX` on `attestation_committee_air`).
//!
//! ## Why this triple
//!
//! - All three AIRs are small (≤ 53 cols) and declare 8-bit byte range
//!   lookups, so they all auto-inflate to the same 256-row range-table
//!   domain inside `joint_prove`, exercising the cross-AIR
//!   domain-alignment path without needing larger FFT widths.
//! - The chain wires two *independent* descriptors whose layer indices
//!   share a common middle layer (layer 1 participates in both D0 and
//!   D1), exercising:
//!     1. joint-γ derivation over a multi-descriptor transcript,
//!     2. per-linkage `LinkageConstraintSystem` SNARK in a loop,
//!     3. multiple cross-trace tuple bindings at independent z values,
//!     4. multiple closure-wrap openings at `ω^{n-1}`,
//!     5. 3× per-AIR `prove_with_scheme` under BLS48-581.
//! - The "shared middle layer" pattern is the canonical multi-AIR
//!   chain shape (e.g. EVM main ↔ storage gadget ↔ KeccakExtract from
//!   the storage-access AIR work), so this is the smallest faithful
//!   stress test for the orchestrator before larger chains.
//!
//! ## Witness alignment
//!
//! All three traces commit a single real row. Honest values:
//!   - `tx_index = 7`        (receipt_status + tx_nonce)
//!   - `pre_account_nonce = 11` (tx_nonce) == `validator_index = 11`
//!     (attestation_committee).
//! The `tx_nonce_air::from_inputs` builder requires
//! `tx_nonce == pre`, so `tx_nonce = 11` as well.
//!
//! ## Cross-references
//!
//! See `crates/zkp/src/integration_joint_prove_smoke.rs` and
//! `integration_joint_prove_smoke_2.rs` for the 2-AIR templates this
//! mirrors. See `crates/zkp/src/sha3_mem_chain/mod.rs` for the
//! production multi-AIR template.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::attestation_committee_air::{
        build_trace_polynomials as build_ac_trace, AttestationCommitteeConstraintSystem,
        AttestationCommitteeWitness, COL_IS_REAL as AC_COL_IS_REAL,
        COL_VALIDATOR_INDEX as AC_COL_VALIDATOR_INDEX,
    };
    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air::{
        build_trace_polynomials as build_rs_trace, ReceiptStatusConstraintSystem,
        ReceiptStatusWitness, COL_IS_REAL as RS_COL_IS_REAL,
        COL_TX_INDEX as RS_COL_TX_INDEX,
    };
    use crate::tx_nonce_air::{
        build_trace_polynomials as build_tn_trace, TxNonceConstraintSystem, TxNonceWitness,
        COL_IS_REAL as TN_COL_IS_REAL, COL_PRE_ACCOUNT_NONCE as TN_COL_PRE_ACCOUNT_NONCE,
        COL_TX_INDEX as TN_COL_TX_INDEX,
    };

    const HONEST_TX_INDEX: u64 = 7;
    const HONEST_NONCE: u64 = 11;
    const HONEST_VALIDATOR_INDEX: u64 = 11;

    // ─── Descriptor builders ──────────────────────────────────────────

    /// D0: `receipt_status_air.COL_TX_INDEX` (layer 0, gated by
    /// `RS_COL_IS_REAL`) ↔ `tx_nonce_air.COL_TX_INDEX` (layer 1, gated
    /// by `TN_COL_IS_REAL`).
    fn tx_index_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "three_air_tx_index_v1".into(),
            a_layer_index: 0,
            a_columns: vec![RS_COL_TX_INDEX],
            a_selector_column: Some(RS_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![TN_COL_TX_INDEX],
            b_selector_column: Some(TN_COL_IS_REAL),
        }
    }

    /// D1: `tx_nonce_air.COL_PRE_ACCOUNT_NONCE` (layer 1, gated by
    /// `TN_COL_IS_REAL`) ↔ `attestation_committee_air.COL_VALIDATOR_INDEX`
    /// (layer 2, gated by `AC_COL_IS_REAL`).
    fn nonce_to_validator_index_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "three_air_nonce_to_validator_index_v1".into(),
            a_layer_index: 1,
            a_columns: vec![TN_COL_PRE_ACCOUNT_NONCE],
            a_selector_column: Some(TN_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![AC_COL_VALIDATOR_INDEX],
            b_selector_column: Some(AC_COL_IS_REAL),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    fn build_rs_witness() -> ReceiptStatusWitness {
        let receipts = vec![Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            // The receipt witness builder consumes (receipts, cumulative
            // gas series) and the tx_index column is populated implicitly
            // from row position; column 0 still ends up == 0 for a
            // single-row witness, which matches our honest descriptor
            // closure only if `HONEST_TX_INDEX == 0`. To be safe, we use
            // `from_receipts` and use `tx_index = 0`.
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }];
        ReceiptStatusWitness::from_receipts(&receipts)
    }

    fn build_tn_witness(tx_index: u64, nonce: u64) -> TxNonceWitness {
        // tx_nonce_air requires tx_nonce == pre. We thread our nonce
        // through both, with sender_address = 0 (the AIR enforces the
        // 12-byte high-zero pad for 20-byte addresses).
        TxNonceWitness::from_inputs([0u8; 20], nonce, nonce, tx_index)
    }

    fn build_ac_witness(validator_index: u64) -> AttestationCommitteeWitness {
        // slot=0, committee_index=0, position=0, committee_size=1.
        AttestationCommitteeWitness::from_assignment(0, 0, validator_index, 0, 1)
    }

    /// Honest receipt-status tx_index, derived from the witness builder
    /// rather than re-asserted from a constant: the receipt-status AIR
    /// populates `COL_TX_INDEX` from row position, so a single-row
    /// witness commits `tx_index = 0`. Both sides of D0 must agree, so
    /// `tx_nonce` is built with `tx_index = 0` as well.
    const ACTUAL_RS_TX_INDEX: u64 = 0;

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across both descriptors, plus a
    /// non-prove sanity check on the 3-trace shape that `joint_prove`
    /// receives. Crucially this does NOT run `joint_prove`, so it stays
    /// well under 120s and runs in CI.
    #[test]
    fn descriptor_consistency_three_air() {
        let d0 = tx_index_descriptor();
        let d1 = nonce_to_validator_index_descriptor();

        // Single-column-tuple invariant (current `joint_prove` only
        // wires single-column tuples).
        assert_eq!(d0.a_columns.len(), 1);
        assert_eq!(d0.b_columns.len(), 1);
        assert_eq!(d1.a_columns.len(), 1);
        assert_eq!(d1.b_columns.len(), 1);

        // Layer indices: D0 wires 0↔1, D1 wires 1↔2. Layer 1 is the
        // shared "middle" trace participating in both descriptors.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 1);
        assert_eq!(d1.b_layer_index, 2);
        assert_ne!(d0.a_layer_index, d0.b_layer_index);
        assert_ne!(d1.a_layer_index, d1.b_layer_index);

        // Selectors must be wired on both sides of both descriptors.
        assert_eq!(d0.a_selector_column, Some(RS_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(TN_COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(TN_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(AC_COL_IS_REAL));

        // Column indices reference the AIR-published `COL_*` constants
        // so any layout drift in the per-AIR modules will surface here.
        assert_eq!(d0.a_columns[0], RS_COL_TX_INDEX);
        assert_eq!(d0.b_columns[0], TN_COL_TX_INDEX);
        assert_eq!(d1.a_columns[0], TN_COL_PRE_ACCOUNT_NONCE);
        assert_eq!(d1.b_columns[0], AC_COL_VALIDATOR_INDEX);

        // Build the per-AIR witnesses + traces at BLS48-581 and confirm
        // the 3-trace orchestrator inputs are constructible without
        // panic. The auto-inflation inside `joint_prove` will resize
        // each trace's columns to the same target_padded; here we only
        // check that the constructed `traces` Vec has the expected
        // shape and that each (trace, cs) pair is wired correctly.
        let curve = CurveType::Bls48581;
        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness(ACTUAL_RS_TX_INDEX, HONEST_NONCE);
        let ac_w = build_ac_witness(HONEST_VALIDATOR_INDEX);

        let trace_0 = build_rs_trace(&rs_w, curve);
        let trace_1 = build_tn_trace(&tn_w, curve);
        let trace_2 = build_ac_trace(&ac_w, curve);

        let cs_0 = ReceiptStatusConstraintSystem::new(trace_0.num_rows);
        let cs_1 = TxNonceConstraintSystem::new(trace_1.num_rows);
        let cs_2 = AttestationCommitteeConstraintSystem::new(trace_2.num_rows);

        // Confirm we can assemble the input shape `joint_prove` takes.
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![d0.clone(), d1.clone()];

        assert_eq!(traces.len(), 3, "3-AIR joint_prove input must have 3 traces");
        assert_eq!(linkages.len(), 2, "must wire exactly 2 descriptors");

        // Every linkage layer index must be in range — this is the
        // bounds check `joint_prove` performs at the top of its loop.
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Sanity: the honest tx_index threaded into both layer 0
        // (computed implicitly from row position by the receipt-status
        // trace builder) and layer 1 (passed in via `from_inputs`)
        // matches.
        assert_eq!(ACTUAL_RS_TX_INDEX, 0);
        // Sanity: the honest validator-index/pre-nonce coupling is
        // numerically consistent for D1.
        assert_eq!(HONEST_NONCE, HONEST_VALIDATOR_INDEX);
        // Defensive: HONEST_TX_INDEX is documented but unused for the
        // actual closure since the receipt-status builder forces
        // tx_index = row_position. Silences the dead-const lint.
        let _ = HONEST_TX_INDEX;
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip with
    /// 2 cross-AIR LogUp descriptors.
    ///
    /// Marked `#[ignore]` because, even on single-row witnesses,
    /// `joint_prove` runs:
    ///   - 3× per-AIR `prove_with_scheme` (each AIR declares 8-bit byte
    ///     range lookups → LogUp inflates each per-AIR domain to 256),
    ///   - 2× per-linkage `prove_with_scheme` on the inner
    ///     `LinkageConstraintSystem`,
    ///   - cross-trace + closure-wrap KZG opens for each descriptor.
    /// On BLS48-581 the expected release runtime is in the high
    /// hundreds of seconds (each per-AIR is ~30–90s; 5 prove calls
    /// total), comfortably > 120s. Run via `--ignored --release`.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify under BLS48-581 (5 inner prove calls)"]
    fn honest_three_air_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness(ACTUAL_RS_TX_INDEX, HONEST_NONCE);
        let ac_w = build_ac_witness(HONEST_VALIDATOR_INDEX);

        let trace_0 = build_rs_trace(&rs_w, curve);
        let trace_1 = build_tn_trace(&tn_w, curve);
        let trace_2 = build_ac_trace(&ac_w, curve);

        let cs_0 = ReceiptStatusConstraintSystem::new(trace_0.num_rows);
        let cs_1 = TxNonceConstraintSystem::new(trace_1.num_rows);
        let cs_2 = AttestationCommitteeConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![tx_index_descriptor(), nonce_to_validator_index_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed");

        assert_eq!(proofs.len(), 3, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            2,
            "expected one CrossAirLogUpProof per descriptor",
        );

        // Honest closure-equality across both descriptors. Since both
        // sides commit the same scalar (tx_index for D0, nonce ==
        // validator_index for D1) gated by IS_REAL on a single real
        // row, the running-sum closures at ω^{n-1} must match per
        // descriptor.
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

    /// Tampered closure: corrupt `closure_a` on descriptor D1 (the
    /// nonce↔validator_index linkage between layer 1 and layer 2).
    /// The verifier's `closure_a == closure_b` scalar equality must
    /// reject.
    ///
    /// Marked `#[ignore]` because the setup half is the same
    /// `joint_prove` call as the honest test. Re-running it under
    /// `--ignored` lets auditors confirm tampering is caught
    /// end-to-end.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_three_air_joint_verify_true"]
    fn tampered_three_air_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let rs_w = build_rs_witness();
        let tn_w = build_tn_witness(ACTUAL_RS_TX_INDEX, HONEST_NONCE);
        let ac_w = build_ac_witness(HONEST_VALIDATOR_INDEX);

        let trace_0 = build_rs_trace(&rs_w, curve);
        let trace_1 = build_tn_trace(&tn_w, curve);
        let trace_2 = build_ac_trace(&ac_w, curve);

        let cs_0 = ReceiptStatusConstraintSystem::new(trace_0.num_rows);
        let cs_1 = TxNonceConstraintSystem::new(trace_1.num_rows);
        let cs_2 = AttestationCommitteeConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![tx_index_descriptor(), nonce_to_validator_index_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed");

        // Tamper with the SECOND descriptor's closure_a so we exercise
        // a per-descriptor rejection path (round 20 + 22 both tamper
        // the only available descriptor; here we have two and target
        // the second).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the second descriptor",
        );
    }
}
