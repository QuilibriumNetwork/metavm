//! Second cross-AIR LogUp `joint_prove` / `joint_verify` integration
//! smoke test, covering DIFFERENT AIRs than
//! [`crate::integration_joint_prove_smoke`].
//!
//! Round 20's smoke test pairs two instances of `receipt_status_air`
//! (23 cols) with a single-column self-linkage on `COL_CUMULATIVE_GAS`.
//! This second smoke test exercises a *heterogeneous* cross-AIR linkage:
//! a single-column tuple over `COL_VALIDATOR_INDEX` shared between
//!
//! - layer A: [`crate::withdrawal_queue_air`] (59 cols), gated by its
//!   [`crate::withdrawal_queue_air::COL_IS_REAL`] selector, and
//! - layer B: [`crate::withdrawal_credential_air`] (66 cols), gated by
//!   its [`crate::withdrawal_credential_air::COL_IS_REAL`] selector.
//!
//! Both AIRs already declare an 8-byte LE decomposition of
//! `validator_index`, so the column is genuinely shared semantics across
//! two independent constraint systems. With matching honest witnesses
//! (same validator index on both sides) the multiset-equality check is
//! honest by construction, and the test still drives every cross-AIR
//! LogUp protocol step:
//!
//! 1. joint-γ derivation in the Fiat-Shamir transcript,
//! 2. per-linkage `LinkageConstraintSystem` SNARK,
//! 3. cross-trace tuple binding at the joint challenge `z`,
//! 4. closure-wrap openings at `ω^{n-1}`, and
//! 5. heterogeneous per-AIR `prove_with_scheme` calls under BLS48-581.
//!
//! ## Why this pair
//!
//! - Both AIRs are small (~60 cols).
//! - Both expose `COL_VALIDATOR_INDEX` as the "u64 view" of the LE
//!   byte decomp column block, with a binary `COL_IS_REAL` selector.
//! - The pair is heterogeneous enough that any column-index drift or
//!   per-AIR domain mismatch will surface as a tuple-binding failure
//!   at the joint challenge, exercising the auto-inflation path in
//!   [`crate::cross_air_logup::joint_prove`] that round 20 (matched
//!   per-AIR shapes) does not stress as hard.
//!
//! ## Cross-references
//!
//! See `crates/zkp/src/integration_joint_prove_smoke.rs` for the round
//! 20 self-linkage template that this module mirrors, and
//! `crates/zkp/src/sha3_mem_chain/mod.rs` for the multi-AIR template.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::withdrawal_credential_air::{
        build_trace_polynomials as build_wc_trace, from_credentials,
        WithdrawalCredentialConstraintSystem, COL_IS_REAL as WC_COL_IS_REAL,
        COL_VALIDATOR_INDEX as WC_COL_VALIDATOR_INDEX, ETH1_PREFIX_BYTE,
    };
    use crate::withdrawal_queue_air::{
        build_trace_polynomials as build_wq_trace, from_withdrawals,
        WithdrawalQueueConstraintSystem, COL_IS_REAL as WQ_COL_IS_REAL,
        COL_VALIDATOR_INDEX as WQ_COL_VALIDATOR_INDEX,
    };

    /// Build the cross-AIR linkage descriptor that ties
    /// `withdrawal_queue_air.COL_VALIDATOR_INDEX` (layer 0, gated by
    /// `WQ_COL_IS_REAL`) to
    /// `withdrawal_credential_air.COL_VALIDATOR_INDEX` (layer 1, gated by
    /// `WC_COL_IS_REAL`).
    fn validator_index_linkage_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "wq_wc_validator_index_v1".into(),
            a_layer_index: 0,
            a_columns: vec![WQ_COL_VALIDATOR_INDEX],
            a_selector_column: Some(WQ_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![WC_COL_VALIDATOR_INDEX],
            b_selector_column: Some(WC_COL_IS_REAL),
        }
    }

    /// A single-row withdrawal-queue witness referencing validator index
    /// `VI`. The pre-balance comfortably covers the amount.
    fn build_wq_witness(vi: u64) -> crate::withdrawal_queue_air::WithdrawalQueueWitness {
        let addr = [0u8; 20];
        from_withdrawals(&[(0u64, vi, addr, 1_000)], &[2_000])
    }

    /// A single-row execution-shaped withdrawal-credential witness with
    /// validator index `VI` and an all-zero execution address. The
    /// chosen prefix byte (`ETH1_PREFIX_BYTE = 0x01`) and the zero
    /// payload satisfy every row-local constraint in
    /// `withdrawal_credential_air`.
    fn build_wc_witness(vi: u64) -> crate::withdrawal_credential_air::WithdrawalCredentialWitness
    {
        let mut creds = [0u8; 32];
        creds[0] = ETH1_PREFIX_BYTE;
        // bytes [1..12] = 0 (zero pad) and bytes [12..32] = 0 (exec addr).
        from_credentials(creds, vi)
    }

    /// Fast static check: descriptor is well-formed and matches the
    /// single-column-tuple invariant required by the current
    /// `joint_prove` implementation.
    #[test]
    fn descriptor_consistency() {
        let d = validator_index_linkage_descriptor();
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert_eq!(
            d.a_columns.len(),
            1,
            "joint_prove currently requires single-column tuples",
        );
        assert_ne!(d.a_layer_index, d.b_layer_index, "linkage layers must differ");
        assert_eq!(d.a_columns[0], WQ_COL_VALIDATOR_INDEX);
        assert_eq!(d.b_columns[0], WC_COL_VALIDATOR_INDEX);
        assert_eq!(d.a_selector_column, Some(WQ_COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(WC_COL_IS_REAL));
        assert_ne!(
            d.a_columns[0], d.b_columns[0],
            "this round wires a heterogeneous cross-AIR linkage; the two \
             column indices live in different AIRs and need not coincide",
        );
    }

    /// Full honest `joint_prove` + `joint_verify` round-trip across
    /// `withdrawal_queue_air` (layer 0) and `withdrawal_credential_air`
    /// (layer 1) with a 1-column validator-index linkage.
    ///
    /// Marked `#[ignore]` because, even on a 1-row witness on each side,
    /// `joint_prove` runs:
    ///   - 2× per-AIR `prove_with_scheme` (both AIRs declare 8-bit byte
    ///     range lookups → LogUp inflates each per-AIR domain to 256),
    ///   - 1× per-linkage `prove_with_scheme` on the inner
    ///     `LinkageConstraintSystem`,
    ///   - 6 KZG opens against the per-AIR / per-linkage commitments.
    /// On BLS48-581 this routinely exceeds 60s in debug and may exceed
    /// 60s in release on modest hardware. Keeping it `#[ignore]`
    /// preserves CI green while still letting auditors run the actual
    /// algebraic check via `--ignored`.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify across wq_air + wc_air (BLS48-581)"]
    fn honest_round_trip_smoke_2() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let vi: u64 = 42;
        let wq_w = build_wq_witness(vi);
        let wc_w = build_wc_witness(vi);
        let trace_a = build_wq_trace(&wq_w, curve);
        let trace_b = build_wc_trace(&wc_w, curve);
        let cs_a = WithdrawalQueueConstraintSystem::new(trace_a.num_rows);
        let cs_b = WithdrawalCredentialConstraintSystem::new(trace_b.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![validator_index_linkage_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme).expect(
            "honest joint_prove must succeed for wq↔wc validator_index linkage",
        );
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);

        // Both AIRs commit the same validator-index value, so the
        // per-side closure scalars must match.
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest matching validator-index witnesses must yield equal closures",
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_a, &cs_b];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept matching wq↔wc validator-index linkage",
        );
    }

    /// Tampered closure: corrupt `closure_a` in the extension envelope
    /// after a successful `joint_prove`. The verifier's
    /// `closure_a == closure_b` scalar equality check must reject.
    /// Re-uses the (slow) prover step from the honest test pattern.
    ///
    /// Marked `#[ignore]` because the setup half is the same
    /// `joint_prove` call as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_round_trip_smoke_2"]
    fn tampered_closure_is_rejected_2() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let vi: u64 = 7;
        let wq_w = build_wq_witness(vi);
        let wc_w = build_wc_witness(vi);
        let trace_a = build_wq_trace(&wq_w, curve);
        let trace_b = build_wc_trace(&wc_w, curve);
        let cs_a = WithdrawalQueueConstraintSystem::new(trace_a.num_rows);
        let cs_b = WithdrawalCredentialConstraintSystem::new(trace_b.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![validator_index_linkage_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme).expect(
            "honest joint_prove must succeed for wq↔wc validator_index linkage",
        );

        // Force a closure mismatch by overwriting `closure_a` with `1`.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_a, &cs_b];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures across wq↔wc",
        );
    }
}
