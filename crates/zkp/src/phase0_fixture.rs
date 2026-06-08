//! Minimal prove/verify fixture for iterating on the extended LogUp
//! wiring. Uses a trivial constraint system with only a single byte-range
//! declaration so the prover/verifier pipeline is exercised end-to-end
//! without VM-scale selector polynomials slowing the debug cycle.

#[cfg(test)]
mod tests {
    use crate::field::{CurveType, Scalar};
    use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;
    use crate::trace::{Polynomial, TracePolynomials};
    use crate::vm_constraints::VmConstraintSystem;

    /// A trivial VM whose only declared constraint is "column 0 is an 8-bit
    /// value when column 1 (selector) is 1." No algebraic VM constraints,
    /// no permutations, no cross-row. Exercises just the LogUp machinery.
    struct ByteRangeOnly;

    impl VmConstraintSystem for ByteRangeOnly {
        fn num_constraints(&self) -> usize {
            0
        }

        fn constraint_labels(&self) -> Vec<String> {
            Vec::new()
        }

        fn evaluate_on_domain(
            &self,
            _columns: &[&Vec<Scalar>],
            _num_rows: usize,
        ) -> Vec<Vec<Scalar>> {
            Vec::new()
        }

        fn evaluate_at_point(
            &self,
            _col_evals_at_z: &[Scalar],
            alpha: &Scalar,
        ) -> Scalar {
            Scalar::zero(alpha.curve_type())
        }

        /// Declare column 1 as a selector so the prover takes the `build_poly`
        /// path (selector-based), which is the path where the extended LogUp
        /// constraints are actually wired. Without this, the prover takes the
        /// `evaluate_on_domain + IFFT` fallback which skips LogUp entirely.
        fn selector_column_indices(&self) -> Vec<usize> {
            vec![1]
        }

        /// No padding selector — the fixture's selector column is 0 on padding
        /// rows, matching the "lookup inactive on padding" semantics.
        fn padding_selector_column(&self) -> Option<usize> {
            None
        }

        fn lookup_declarations(&self) -> LookupRequirements {
            LookupRequirements {
                tables: vec![LookupTable::range(8)],
                declarations: vec![(
                    LookupDeclaration {
                        label: "value_range".to_string(),
                        column_index: 0,
                        max_bits: 8,
                        selector_column: Some(1),
                    },
                    0,
                )],
            }
        }
    }

    /// Build a 2-column trace of length `domain_size` with `num_real` active
    /// rows. Column 0 (value) holds the given byte values on active rows;
    /// column 1 (selector) is 1 on active rows, 0 elsewhere.
    fn build_trace(
        values: &[u64],
        domain_size: usize,
        curve: CurveType,
    ) -> TracePolynomials {
        assert!(values.len() <= domain_size);
        let mut value_evals = vec![0u64; domain_size];
        let mut sel_evals = vec![0u64; domain_size];
        for (i, &v) in values.iter().enumerate() {
            assert!(v < 256, "value must fit in a byte");
            value_evals[i] = v;
            sel_evals[i] = 1;
        }
        let value_poly = Polynomial::from_u64_vec_with_curve(&value_evals, curve);
        let sel_poly = Polynomial::from_u64_vec_with_curve(&sel_evals, curve);
        TracePolynomials::from_polynomials(vec![value_poly, sel_poly], values.len(), curve)
    }

    #[test]
    fn test_baseline_prove_verify_tiny_trace() {
        // Baseline sanity: with the currently wired (partial) LogUp
        // enforcement, the minimal fixture should already prove/verify.
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let trace = build_trace(&[7, 12, 3], 256, curve);
        let cs = ByteRangeOnly;

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "minimal fixture must prove/verify under baseline");
    }

    /// Mirrors SBF's pattern: a declaration with `selector_column: None`
    /// (always-active range check on a column that is zero on padding rows).
    struct ByteRangeAlwaysActive;

    impl VmConstraintSystem for ByteRangeAlwaysActive {
        fn num_constraints(&self) -> usize { 0 }
        fn constraint_labels(&self) -> Vec<String> { Vec::new() }
        fn evaluate_on_domain(&self, _: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
            Vec::new()
        }
        fn evaluate_at_point(&self, _: &[Scalar], alpha: &Scalar) -> Scalar {
            Scalar::zero(alpha.curve_type())
        }
        // Take the build_poly path even though this fixture has no algebraic
        // selector constraints — column 1 exists only to keep the trace shape
        // identical to `ByteRangeOnly` and force the build_poly branch.
        fn selector_column_indices(&self) -> Vec<usize> {
            vec![1]
        }
        fn padding_selector_column(&self) -> Option<usize> {
            None
        }
        fn lookup_declarations(&self) -> LookupRequirements {
            LookupRequirements {
                tables: vec![LookupTable::range(8)],
                declarations: vec![(
                    LookupDeclaration {
                        label: "always_active_range".to_string(),
                        column_index: 0,
                        max_bits: 8,
                        selector_column: None, // ← the SBF pattern
                    },
                    0,
                )],
            }
        }
    }

    /// Two independent groups in a single trace:
    /// - column 0 gated by selector (column 2)
    /// - column 1 always-active (no selector)
    /// Exercises the multi-group transition / aggregation path.
    struct TwoGroups;

    impl VmConstraintSystem for TwoGroups {
        fn num_constraints(&self) -> usize { 0 }
        fn constraint_labels(&self) -> Vec<String> { Vec::new() }
        fn evaluate_on_domain(&self, _: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
            Vec::new()
        }
        fn evaluate_at_point(&self, _: &[Scalar], alpha: &Scalar) -> Scalar {
            Scalar::zero(alpha.curve_type())
        }
        fn selector_column_indices(&self) -> Vec<usize> {
            vec![2]
        }
        fn padding_selector_column(&self) -> Option<usize> {
            None
        }
        fn lookup_declarations(&self) -> LookupRequirements {
            LookupRequirements {
                tables: vec![LookupTable::range(8)],
                declarations: vec![
                    (
                        LookupDeclaration {
                            label: "gated_range".to_string(),
                            column_index: 0,
                            max_bits: 8,
                            selector_column: Some(2),
                        },
                        0,
                    ),
                    (
                        LookupDeclaration {
                            label: "always_range".to_string(),
                            column_index: 1,
                            max_bits: 8,
                            selector_column: None,
                        },
                        0,
                    ),
                ],
            }
        }
    }

    /// Soundness end-to-end: feed the prover an 8-bit range-check fixture
    /// whose "value" column contains an out-of-range value. The prover still
    /// constructs a proof (it doesn't validate the trace), but the verifier
    /// must reject because the byte decomposition `sel · (v - limb) = 0`
    /// cannot hold with v ≥ 256 and limb a single byte.
    ///
    /// Wraps `ByteRangeOnly` directly since its declaration pins column 0 to
    /// 8 bits via LogUp; a value of 300 forces a mismatch.
    #[test]
    fn test_prove_verify_rejects_out_of_range_value() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Deliberately illegal: value 300 is outside the 8-bit range.
        let domain_size = 256usize;
        let mut value_evals = vec![0u64; domain_size];
        let mut sel_evals = vec![0u64; domain_size];
        value_evals[0] = 300; // > 255 — invalid for 8-bit check
        sel_evals[0] = 1;

        let value_poly = Polynomial::from_u64_vec_with_curve(&value_evals, curve);
        let sel_poly = Polynomial::from_u64_vec_with_curve(&sel_evals, curve);
        let trace = TracePolynomials::from_polynomials(
            vec![value_poly, sel_poly], 1, curve,
        );
        let cs = ByteRangeOnly;

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(
            !valid,
            "out-of-range value must be rejected end-to-end"
        );
    }

    /// Minimal bitwise-lookup fixture: a 4-bit AND gate gated by a selector.
    /// Exercises the full bitwise LogUp wiring — nibble decomp, result
    /// derivation, per-nibble inverse, table inverse, transition, canonical
    /// t_a/t_b/t_c(z) binding.
    struct ByteAndOnly;

    impl VmConstraintSystem for ByteAndOnly {
        fn num_constraints(&self) -> usize { 0 }
        fn constraint_labels(&self) -> Vec<String> { Vec::new() }
        fn evaluate_on_domain(&self, _: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
            Vec::new()
        }
        fn evaluate_at_point(&self, _: &[Scalar], alpha: &Scalar) -> Scalar {
            Scalar::zero(alpha.curve_type())
        }
        fn selector_column_indices(&self) -> Vec<usize> {
            vec![3]
        }
        fn padding_selector_column(&self) -> Option<usize> {
            None
        }
        fn bitwise_lookup_declarations(
            &self,
        ) -> Vec<crate::lookup::BitwiseLookupDeclaration> {
            vec![crate::lookup::BitwiseLookupDeclaration {
                label: "a_and_b".to_string(),
                operand_a_column: 0,
                operand_b_column: 1,
                result_column: 2,
                width_bits: 4,
                op: crate::lookup::BitwiseOp::And,
                selectors: vec![3],
            }]
        }
    }

    #[test]
    fn test_prove_verify_bitwise_and_fixture() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 3 active rows of 4-bit AND: (5,3)→1, (0xF,0xA)→0xA, (0,0xF)→0.
        let domain_size = 256usize;
        let mut col_a = vec![0u64; domain_size];
        let mut col_b = vec![0u64; domain_size];
        let mut col_r = vec![0u64; domain_size];
        let mut col_sel = vec![0u64; domain_size];
        let samples: [(u64, u64); 3] = [(5, 3), (0xF, 0xA), (0, 0xF)];
        for (i, &(a, b)) in samples.iter().enumerate() {
            col_a[i] = a;
            col_b[i] = b;
            col_r[i] = a & b;
            col_sel[i] = 1;
        }
        let a_poly = Polynomial::from_u64_vec_with_curve(&col_a, curve);
        let b_poly = Polynomial::from_u64_vec_with_curve(&col_b, curve);
        let r_poly = Polynomial::from_u64_vec_with_curve(&col_r, curve);
        let sel_poly = Polynomial::from_u64_vec_with_curve(&col_sel, curve);
        let trace = TracePolynomials::from_polynomials(
            vec![a_poly, b_poly, r_poly, sel_poly], 3, curve,
        );

        let cs = ByteAndOnly;
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "bitwise AND fixture must prove/verify");
    }

    /// End-to-end bitwise soundness: supply a result column with a *wrong*
    /// AND value (e.g. `0xF` instead of `0x5 & 0x3 = 0x1`) and confirm the
    /// verifier rejects the resulting proof. Exercises the
    /// `sel · (result - c_recomp) = 0` result-derivation constraint.
    #[test]
    fn test_prove_verify_rejects_wrong_bitwise_result() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let domain_size = 256usize;
        let mut col_a = vec![0u64; domain_size];
        let mut col_b = vec![0u64; domain_size];
        let mut col_r = vec![0u64; domain_size];
        let mut col_sel = vec![0u64; domain_size];
        // Correct: 0x5 & 0x3 = 0x1. We claim 0xF instead — invalid.
        col_a[0] = 0x5;
        col_b[0] = 0x3;
        col_r[0] = 0xF; // <-- wrong (correct would be 0x1)
        col_sel[0] = 1;
        let a_poly = Polynomial::from_u64_vec_with_curve(&col_a, curve);
        let b_poly = Polynomial::from_u64_vec_with_curve(&col_b, curve);
        let r_poly = Polynomial::from_u64_vec_with_curve(&col_r, curve);
        let sel_poly = Polynomial::from_u64_vec_with_curve(&col_sel, curve);
        let trace = TracePolynomials::from_polynomials(
            vec![a_poly, b_poly, r_poly, sel_poly], 1, curve,
        );

        let cs = ByteAndOnly;
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(
            !valid,
            "wrong AND result must be rejected end-to-end"
        );
    }

    #[test]
    fn test_prove_verify_two_groups() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let domain_size = 256usize;
        let num_real = 4usize;
        let mut col0 = vec![0u64; domain_size];
        let mut col1 = vec![0u64; domain_size];
        let mut sel = vec![0u64; domain_size];
        for (i, (a, b)) in [(5u64, 11u64), (42, 7), (100, 200), (0, 0)].iter().enumerate() {
            col0[i] = *a;
            col1[i] = *b;
            sel[i] = 1;
        }
        let _ = num_real;

        let value_poly = Polynomial::from_u64_vec_with_curve(&col0, curve);
        let col1_poly = Polynomial::from_u64_vec_with_curve(&col1, curve);
        let sel_poly = Polynomial::from_u64_vec_with_curve(&sel, curve);
        let trace = TracePolynomials::from_polynomials(
            vec![value_poly, col1_poly, sel_poly], 4, curve,
        );

        let cs = TwoGroups;
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "two-group fixture must prove/verify");
    }

    #[test]
    fn test_prove_verify_always_active_range_check() {
        // Regression fixture for the padding-row accounting bug:
        // with `selector_column: None`, the polynomial `active(X) = 1`
        // on every row including padding, so the witness MUST count
        // padding-row byte contributions in its multiplicities. This
        // test fails loudly if the witness skips padding rows.
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let trace = build_trace(&[5, 42, 100], 256, curve);
        let cs = ByteRangeAlwaysActive;

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(
            valid,
            "empty-selector (always-active) range check must prove/verify"
        );
    }
}
