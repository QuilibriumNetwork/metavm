//! Soundness sanity tests for cross-AIR LogUp `joint_prove` /
//! `joint_verify`: tamper with **row data inside the trace** (not the
//! closure scalars in the extension envelope) and confirm the verifier
//! still rejects.
//!
//! ## Motivation
//!
//! [`crate::integration_joint_prove_smoke`] and
//! [`crate::integration_joint_prove_smoke_2`] only tamper with the
//! per-linkage `closure_a` scalar in the extension envelope after a
//! successful prove. That exercises exactly one of the verifier's
//! rejection paths (`closure_a == closure_b` scalar equality). It does
//! NOT exercise the much more interesting cases where the trace itself
//! is wrong:
//!
//! 1. A LogUp-side value is changed (multiset equality breaks → either
//!    `joint_prove` fails internally building the consistent extension,
//!    or `joint_verify` rejects via tuple-binding / closure-wrap.)
//! 2. A binary selector column is set to a non-binary value (binarity
//!    constraint must fire inside the per-AIR `prove_with_scheme`.)
//! 3. The two sides of a self-linkage are swapped on one column so the
//!    multisets no longer match.
//!
//! Each test below uses the cheapest available 2-AIR self-linkage
//! pair (`receipt_status_air` × 2, mirroring
//! [`crate::integration_joint_prove_smoke`]) and mutates the per-side
//! trace **before** the call to `joint_prove`. The expected outcome is
//! that either `joint_prove` returns `Err` (witness inconsistency
//! detected at proving time) or `joint_verify` returns `false`. A pass
//! through both stages would be a soundness break — the assertion
//! messages call that out explicitly.
//!
//! ## Marked `#[ignore]`
//!
//! Each individual test calls `joint_prove` once and is therefore as
//! slow as `honest_round_trip_smoke` (see the doc on that test).
//! Marked `#[ignore]` to preserve CI green; auditors can run them via
//! `cargo test ... -- --ignored`.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air::{
        build_trace_polynomials, ReceiptStatusConstraintSystem, ReceiptStatusWitness,
        COL_CUMULATIVE_GAS, COL_GAS_USED, COL_IS_REAL,
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

    /// Same self-linkage as `integration_joint_prove_smoke`: both
    /// sides view `COL_CUMULATIVE_GAS` gated by `COL_IS_REAL`.
    fn cumulative_gas_self_linkage_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "receipt_status_tampered_rows_v1".into(),
            a_layer_index: 0,
            a_columns: vec![COL_CUMULATIVE_GAS],
            a_selector_column: Some(COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![COL_CUMULATIVE_GAS],
            b_selector_column: Some(COL_IS_REAL),
        }
    }

    fn build_smoke_witness() -> ReceiptStatusWitness {
        let receipts = vec![make_receipt(1, 21_000)];
        ReceiptStatusWitness::from_receipts(&receipts)
    }

    /// Run `joint_prove` + `joint_verify` and assert the overall result
    /// is rejection (either via `Err` from the prover or `false` from
    /// the verifier). Panics LOUDLY with the supplied `breakage_label`
    /// if BOTH stages succeed — that would be a soundness break.
    fn assert_chain_rejects(
        traces: &[(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )],
        linkages: &[CrossAirLogUpDescriptor],
        cs_refs: &[&dyn crate::vm_constraints::VmConstraintSystem],
        scheme: &dyn CommitmentScheme,
        curve: CurveType,
        breakage_label: &str,
    ) {
        match joint_prove(traces, linkages, scheme) {
            Err(_) => {
                // Prover detected inconsistency. This is acceptable.
            }
            Ok((proofs, ext)) => {
                let accepted =
                    joint_verify(&proofs, cs_refs, linkages, &ext, scheme, curve);
                assert!(
                    !accepted,
                    "SOUNDNESS BREAK: joint_prove succeeded AND joint_verify accepted \
                     a tampered trace ({breakage_label}). This means a malicious \
                     prover could swap row data without detection — investigate \
                     before deploying.",
                );
            }
        }
    }

    /// Tamper #1: mutate `COL_GAS_USED` on trace A's row 0 only.
    ///
    /// `COL_GAS_USED` is constrained by **two** row-local constraints:
    ///   * `cumulative_chain` — `IS_REAL * (cumul - prev - gas_used) = 0`
    ///   * `gas_used_le_decomp` — LE byte decomp residual against
    ///     `COL_GAS_USED_BYTE_OFFSET..+8`.
    /// Bumping just the scalar value (without rebuilding the byte
    /// columns) violates BOTH. The per-AIR `prove_with_scheme` inside
    /// `joint_prove` therefore either fails to build a valid quotient
    /// (returning `Err`) or produces a proof that the verifier rejects.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify; tampered gas_used row data (BLS48-581)"]
    fn tampered_row_data_rejects_joint_verify() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = build_smoke_witness();
        let mut trace_a = build_trace_polynomials(&w, curve);
        let trace_b = build_trace_polynomials(&w, curve);

        // Mutate row 0 of trace A's COL_GAS_USED: honest is 21_000, we
        // bump to 21_001. This breaks the cumulative chain
        // (`cumul=21_000 != prev=0 + gas_used=21_001`) AND the LE byte
        // decomp (low byte is now 0x29, decomp says 0x28).
        let tampered = Scalar::from_u64(21_001, curve);
        trace_a.columns[COL_GAS_USED].evaluations[0] = tampered;

        let cs_a = ReceiptStatusConstraintSystem::new(trace_a.num_rows);
        let cs_b = ReceiptStatusConstraintSystem::new(trace_b.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![cumulative_gas_self_linkage_descriptor()];
        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_a, &cs_b];

        assert_chain_rejects(
            &traces,
            &linkages,
            &cs_refs,
            &scheme,
            curve,
            "COL_GAS_USED row 0 value flipped",
        );
    }

    /// Tamper #2: set `COL_IS_REAL` on trace A's row 0 from 1 to 2.
    ///
    /// This attacks the binarity surface for selector columns. The
    /// per-AIR constraint system declares `COL_IS_REAL` binary; the
    /// per-AIR `prove_with_scheme` step inside `joint_prove` should
    /// reject. Even if the per-AIR slip passes, the LogUp side uses
    /// this column as a gating selector, so a non-binary value
    /// inflates the multiset count and the linkage closure mismatches.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify; tampered selector non-binary (BLS48-581)"]
    fn tampered_selector_value_rejects_joint_verify() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = build_smoke_witness();
        let mut trace_a = build_trace_polynomials(&w, curve);
        let trace_b = build_trace_polynomials(&w, curve);

        // Force COL_IS_REAL[0] = 2, breaking binarity.
        let two = Scalar::from_u64(2, curve);
        trace_a.columns[COL_IS_REAL].evaluations[0] = two;

        let cs_a = ReceiptStatusConstraintSystem::new(trace_a.num_rows);
        let cs_b = ReceiptStatusConstraintSystem::new(trace_b.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![cumulative_gas_self_linkage_descriptor()];
        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_a, &cs_b];

        assert_chain_rejects(
            &traces,
            &linkages,
            &cs_refs,
            &scheme,
            curve,
            "COL_IS_REAL row 0 set to non-binary value 2",
        );
    }

    /// Tamper #3: corrupt the linkage column on trace A's row 0 ONLY.
    ///
    /// `COL_CUMULATIVE_GAS` participates in both per-AIR constraints
    /// (cumulative-gas chain) and the LogUp linkage tuple. Changing
    /// only A's value while leaving B intact creates a multiset
    /// mismatch on the linked column. The per-AIR cumulative-gas
    /// chain on side A should also reject, since the byte-decomp of
    /// `COL_CUMULATIVE_GAS` (`COL_CUMUL_BYTE_OFFSET..`) no longer
    /// matches the scalar value.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify; tampered linkage column on A only (BLS48-581)"]
    fn tampered_a_columns_swap_rejects_joint_verify() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = build_smoke_witness();
        let mut trace_a = build_trace_polynomials(&w, curve);
        let trace_b = build_trace_polynomials(&w, curve);

        // Replace A's row-0 cumulative gas with a different value.
        // B keeps the honest 21_000. The two multisets — gated by
        // COL_IS_REAL — are now {99_999} on A vs {21_000} on B, so
        // closure_a ≠ closure_b and `joint_verify` rejects via the
        // closure equality check; AND the per-AIR cumulative-gas
        // chain on A fails its byte-decomp residual.
        let bad = Scalar::from_u64(99_999, curve);
        trace_a.columns[COL_CUMULATIVE_GAS].evaluations[0] = bad;

        let cs_a = ReceiptStatusConstraintSystem::new(trace_a.num_rows);
        let cs_b = ReceiptStatusConstraintSystem::new(trace_b.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![cumulative_gas_self_linkage_descriptor()];
        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_a, &cs_b];

        assert_chain_rejects(
            &traces,
            &linkages,
            &cs_refs,
            &scheme,
            curve,
            "COL_CUMULATIVE_GAS row 0 mutated on A but not B",
        );
    }
}
