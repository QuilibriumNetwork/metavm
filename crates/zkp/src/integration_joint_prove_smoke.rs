//! Cross-AIR LogUp `joint_prove`/`joint_verify` integration smoke test.
//!
//! This module exists to actually exercise the cross-AIR LogUp prover +
//! verifier wiring end-to-end on the lightest pair of AIRs we can find.
//! Static well-formedness tests against `CrossAirLogUpDescriptor`s are
//! plentiful across the crate, but they don't catch protocol-level bugs
//! in [`crate::cross_air_logup::joint_prove`] /
//! [`crate::cross_air_logup::joint_verify`] (e.g. transcript mismatches,
//! domain-inflation bugs, closure-binding regressions). This file picks
//! a single tiny AIR (`receipt_status_air`, 23 columns) and runs an
//! honest 2-trace `joint_prove` + `joint_verify` round-trip across two
//! independent instances of the same AIR with a single-column self
//! linkage on `COL_CUMULATIVE_GAS`.
//!
//! ## Why this pair
//!
//! - `receipt_status_air` has the smallest column count (23) of the
//!   wired Phase A2/A3 AIRs and has both row-local and shifted
//!   constraints, exercising the full per-AIR pipeline.
//! - Using two instances of the same AIR with a single-column tuple
//!   keeps the linkage SNARK trivial: identical traces ⇒ multiset
//!   equality is automatic, and the test still drives the joint-γ
//!   derivation, the per-linkage SNARK, cross-trace openings, and
//!   closure-wrap openings.
//! - The witness is a single receipt, padded to a 1-row trace
//!   (auto-inflated to the LogUp range-table domain inside
//!   `joint_prove`). This minimises FFT widths and SRS-MSM work.
//!
//! ## Cross-references
//!
//! See `crates/zkp/src/sha3_mem_chain/mod.rs` for the multi-AIR
//! template and `crates/zkp/src/mpt_extension_rlp_air.rs` for the
//! standard 2-AIR joint_prove pattern that this test mirrors.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air::{
        build_trace_polynomials, ReceiptStatusConstraintSystem, ReceiptStatusWitness,
        COL_CUMULATIVE_GAS, COL_IS_REAL,
    };
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    fn make_receipt(status: u8, cumulative_gas_used: u64) -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status,
            cumulative_gas_used,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    /// Construct a self-linkage descriptor: trace A and trace B both
    /// look at `COL_CUMULATIVE_GAS` gated by `COL_IS_REAL`. With
    /// identical witnesses on both sides, the multiset of A's active
    /// cumulative-gas values equals B's, so the multiset-equality check
    /// is honest by construction.
    fn cumulative_gas_self_linkage_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "receipt_status_self_link_cumul_v1".into(),
            a_layer_index: 0,
            a_columns: vec![COL_CUMULATIVE_GAS],
            a_selector_column: Some(COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![COL_CUMULATIVE_GAS],
            b_selector_column: Some(COL_IS_REAL),
        }
    }

    /// Build a minimal single-receipt witness shared between both sides.
    fn build_smoke_witness() -> ReceiptStatusWitness {
        let receipts = vec![make_receipt(1, 21_000)];
        ReceiptStatusWitness::from_receipts(&receipts)
    }

    /// Static check: descriptor is well-formed and matches the
    /// single-column-tuple invariant required by the current
    /// `joint_prove` implementation.
    #[test]
    fn descriptor_consistency() {
        let d = cumulative_gas_self_linkage_descriptor();
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert_eq!(d.a_columns.len(), 1, "joint_prove currently requires single-column tuples");
        assert_ne!(d.a_layer_index, d.b_layer_index, "linkage layers must differ");
        assert_eq!(d.a_columns[0], COL_CUMULATIVE_GAS);
        assert_eq!(d.b_columns[0], COL_CUMULATIVE_GAS);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));
    }

    /// Full honest `joint_prove` + `joint_verify` round-trip across two
    /// `receipt_status_air` instances with a 1-column self-linkage.
    ///
    /// Marked `#[ignore]` because, even on the smallest possible
    /// witness, `joint_prove` runs:
    ///   - 2× per-AIR `prove_with_scheme` (range-table domain = 256
    ///     because `ReceiptStatusConstraintSystem` declares 16
    ///     8-bit byte lookups → LogUp inflates the domain to 256),
    ///   - 1× per-linkage `prove_with_scheme` on the inner
    ///     `LinkageConstraintSystem`,
    ///   - 6 KZG opens against the per-AIR / per-linkage commitments.
    /// On BLS48-581 this routinely sits >60s in debug and may also
    /// exceed 60s in release on modest hardware. Keeping it `#[ignore]`
    /// preserves CI green while still letting auditors run the actual
    /// algebraic check via `--ignored`.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify over 2 receipt_status_air AIRs (BLS48-581)"]
    fn honest_round_trip_smoke() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = build_smoke_witness();
        let trace_a = build_trace_polynomials(&w, curve);
        let trace_b = build_trace_polynomials(&w, curve);
        let cs_a = ReceiptStatusConstraintSystem::new(trace_a.num_rows);
        let cs_b = ReceiptStatusConstraintSystem::new(trace_b.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![cumulative_gas_self_linkage_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed for receipt_status self-linkage");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        // Closures must match for a single shared witness.
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "self-linkage closures must match for identical traces",
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> = vec![&cs_a, &cs_b];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept matching self-linkage witnesses",
        );
    }

    /// Tampered witness: corrupt one closure scalar in the extension
    /// envelope so `closure_a != closure_b`. The verifier's closure
    /// equality check must reject. Re-uses the honest proofs from a
    /// fresh `joint_prove` to avoid double-spending the (slow) prover
    /// step.
    ///
    /// Also marked `#[ignore]` because the setup half is the same
    /// `joint_prove` call as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_round_trip_smoke"]
    fn tampered_closure_is_rejected() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = build_smoke_witness();
        let trace_a = build_trace_polynomials(&w, curve);
        let trace_b = build_trace_polynomials(&w, curve);
        let cs_a = ReceiptStatusConstraintSystem::new(trace_a.num_rows);
        let cs_b = ReceiptStatusConstraintSystem::new(trace_b.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![cumulative_gas_self_linkage_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed for receipt_status self-linkage");

        // Mutate closure_a so that the joint verifier's
        // `closure_a == closure_b` scalar equality check fires.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> = vec![&cs_a, &cs_b];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures",
        );
    }
}
