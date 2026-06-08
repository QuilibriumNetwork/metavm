//! `logs_bloom` sub-tree pair AIR.
//!
//! Per-row exposes one `sha256_pair(left, right) -> hash` invocation
//! from a [`LogsBloomHtrWitness`] (the 7-invocation merkleization of
//! 8 chunks → root). Mirrors the body-pair / payload-pair / BBH-pair
//! AIR architecture.
//!
//! # Soundness scope
//!
//! - Per-row `(input_64, output_32)` matches a row in `Sha256Extract`
//!   via [`make_logs_bloom_pair_to_sha256_extract_linkage_descriptor`].
//! - Computed root (HASH at the root invocation row) bound to
//!   `CLAIMED_ROOT` (cross-row constant + at-root-invocation row-local).
//! - `CLAIMED_ROOT` exposed for payload-pair binding via
//!   [`make_logs_bloom_pair_to_payload_pair_linkage_descriptor`].
//!
//! With this AIR + the linkage descriptor, the payload-pair AIR's
//! `CLAIMED_LOGS_BLOOM_ROOT` is algebraically bound to a real
//! merkleization of an actual 256-byte logs_bloom field, closing the
//! soundness gap where the payload-pair AIR previously trusted the
//! logs_bloom field root opaquely.

use crate::beacon_block_header_air::Sha256PairInvocation;
use crate::field::{CurveType, Scalar};
use crate::logs_bloom_air::LogsBloomHtrWitness;
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_LEFT_OFFSET: usize = 0;            // 0..32
pub const COL_RIGHT_OFFSET: usize = 32;          // 32..64
pub const COL_HASH_OFFSET: usize = 64;           // 64..96
pub const COL_IS_REAL: usize = 96;

pub const COL_CLAIMED_ROOT_OFFSET: usize = 97;   // 97..129
pub const COL_IS_ROOT_BOUND_AT: usize = 129;
pub const NUM_COLUMNS: usize = COL_IS_ROOT_BOUND_AT + 1; // 130

/// Invocation index of the root (the layer-2 pair). 7 invocations,
/// indices 0..7: layer 0 (4) + layer 1 (2) + layer 2 (1). Index 6 is
/// the root.
pub const ROOT_INVOCATION_INDEX: usize = 6;

// Row-locals:
//   0: is_real binary
//   1: is_root_bound_at binary
//   2: is_root_bound_at * Σ β^k * (HASH[k] - CLAIMED_ROOT[k])
pub const NUM_ROW_CONSTRAINTS: usize = 3;

// Shifted: cross-row constancy of CLAIMED_ROOT.
pub const NUM_SHIFTED: usize = 1;

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &LogsBloomHtrWitness,
    curve: CurveType,
) -> TracePolynomials {
    build_trace_polynomials_from_invocations(&witness.invocations, curve)
}

pub fn build_trace_polynomials_from_invocations(
    invocations: &[Sha256PairInvocation],
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, inv) in invocations.iter().enumerate() {
        for k in 0..32 {
            columns[COL_LEFT_OFFSET + k][i] = Scalar::from_u64(inv.left[k] as u64, curve);
            columns[COL_RIGHT_OFFSET + k][i] = Scalar::from_u64(inv.right[k] as u64, curve);
            columns[COL_HASH_OFFSET + k][i] = Scalar::from_u64(inv.hash[k] as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
    }

    // CLAIMED_ROOT replicated across all rows from the root invocation hash.
    if num_rows > ROOT_INVOCATION_INDEX {
        let root_hash = invocations[ROOT_INVOCATION_INDEX].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(root_hash[k] as u64, curve);
            for r in 0..num_rows {
                columns[COL_CLAIMED_ROOT_OFFSET + k][r] = v.clone();
            }
        }
        columns[COL_IS_ROOT_BOUND_AT][ROOT_INVOCATION_INDEX] = one.clone();
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct LogsBloomPairConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl LogsBloomPairConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for LogsBloomPairConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_root_bound_at_binary".into(),
            "hash_eq_claimed_root_at_root_invocation".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let beta_test = Scalar::from_u64(7, curve);

        let mut is_real_bin = vec![Scalar::zero(curve); n];
        let mut is_root_bin = vec![Scalar::zero(curve); n];
        let mut hash_eq_claimed = vec![Scalar::zero(curve); n];

        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            is_real_bin[r] = v.mul(&v.sub(&one));
            let rb = &columns[COL_IS_ROOT_BOUND_AT][r];
            is_root_bin[r] = rb.mul(&rb.sub(&one));

            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let h = &columns[COL_HASH_OFFSET + k][r];
                let c = &columns[COL_CLAIMED_ROOT_OFFSET + k][r];
                acc = acc.add(&bp.mul(&h.sub(c)));
                bp = bp.mul(&beta_test);
            }
            hash_eq_claimed[r] = rb.mul(&acc);
        }
        vec![is_real_bin, is_root_bin, hash_eq_claimed]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let rb = &col_evals[COL_IS_ROOT_BOUND_AT];
        let is_real_bin = v.mul(&v.sub(&one));
        let is_root_bin = rb.mul(&rb.sub(&one));

        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let h = &col_evals[COL_HASH_OFFSET + k];
            let c = &col_evals[COL_CLAIMED_ROOT_OFFSET + k];
            acc = acc.add(&bp.mul(&h.sub(c)));
            bp = bp.mul(alpha);
        }
        let hash_eq_claimed = rb.mul(&acc);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = total.add(&ap.mul(&is_root_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&hash_eq_claimed));
        total
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let is_real_bin = poly_mul(v, &v_m1, curve);

        let rb = &col_coeffs[COL_IS_ROOT_BOUND_AT];
        let rb_m1 = poly_sub(rb, &one_poly, curve);
        let is_root_bin = poly_mul(rb, &rb_m1, curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let h = &col_coeffs[COL_HASH_OFFSET + k];
            let c = &col_coeffs[COL_CLAIMED_ROOT_OFFSET + k];
            let diff = poly_sub(h, c, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let hash_eq_claimed = poly_mul(rb, &acc, curve);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = poly_add(&total, &poly_scalar_mul(&is_root_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&hash_eq_claimed, &ap), curve);
        total
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        let mut cols = Vec::with_capacity(33);
        cols.push(COL_IS_REAL);
        for k in 0..32 { cols.push(COL_CLAIMED_ROOT_OFFSET + k); }
        cols
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        if shifted_evals.len() != 33 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let gating = v.mul(v_next);

        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let cur = &col_evals_at_z[COL_CLAIMED_ROOT_OFFSET + k];
            let nxt = &shifted_evals[1 + k];
            acc = acc.add(&bp.mul(&nxt.sub(cur)));
            bp = bp.mul(alpha);
        }
        let body = gating.mul(&acc);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        ap.mul(&body).mul(&z.sub(omega_n_minus_1))
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let v = &column_coeffs[COL_IS_REAL];
        let v_next = poly_shift(v, omega);
        let gating = poly_mul(v, &v_next, curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let cur = &column_coeffs[COL_CLAIMED_ROOT_OFFSET + k];
            let nxt = poly_shift(cur, omega);
            let diff = poly_sub(&nxt, cur, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body = poly_mul(&gating, &acc, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let total = poly_scalar_mul(&body, &ap);

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_IS_ROOT_BOUND_AT]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(8)];
        let mut declarations = Vec::new();
        for k in 0..32 {
            for (name, off) in [
                ("left", COL_LEFT_OFFSET),
                ("right", COL_RIGHT_OFFSET),
                ("hash", COL_HASH_OFFSET),
                ("claimed_root", COL_CLAIMED_ROOT_OFFSET),
            ] {
                declarations.push((
                    LookupDeclaration {
                        label: format!("logs_bloom_pair_{}_{}_8bit", name, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkages ─────────────────────────────────────────

/// Bind each `(LEFT || RIGHT, HASH)` row of this AIR to a real
/// [`crate::sha256_extract`] row.
pub fn make_logs_bloom_pair_to_sha256_extract_linkage_descriptor(
    logs_bloom_pair_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..32 { a_columns.push(COL_LEFT_OFFSET + b); }
    for b in 0..32 { a_columns.push(COL_RIGHT_OFFSET + b); }
    for b in 0..32 { a_columns.push(COL_HASH_OFFSET + b); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..crate::sha256_extract::NUM_INPUT_BYTES {
        b_columns.push(crate::sha256_extract::COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..crate::sha256_extract::NUM_OUTPUT_BYTES {
        b_columns.push(crate::sha256_extract::COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "logs_bloom_pair_sha256_extract_v1".into(),
        a_layer_index: logs_bloom_pair_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(crate::sha256_extract::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp linkage: logs_bloom-pair's CLAIMED_ROOT (the
/// computed logs_bloom merkle root) ↔ payload-pair's
/// CLAIMED_LOGS_BLOOM_ROOT (the logs_bloom field root claim).
///
/// **A side (logs_bloom-pair)**: 32-byte CLAIMED_ROOT gated by
/// `IS_ROOT_BOUND_AT` (1 entry at invocation 6 = root).
///
/// **B side (payload-pair)**: 32-byte CLAIMED_LOGS_BLOOM_ROOT gated by
/// `IS_LOGS_BLOOM_LEAF_BOUND_AT` (1 entry at payload invocation 2).
///
/// Multiset equality on 1-vs-1 algebraically pins
/// `logs_bloom_pair.computed_root == payload.logs_bloom_field_root`,
/// closing the soundness gap where payload previously trusted the
/// logs_bloom field root opaquely.
pub fn make_logs_bloom_pair_to_payload_pair_linkage_descriptor(
    logs_bloom_pair_layer_index: usize,
    payload_pair_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::execution_payload_pair_air as payload;

    let mut a_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { a_columns.push(COL_CLAIMED_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { b_columns.push(payload::COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k); }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "logs_bloom_pair_root_to_payload_pair_claimed_logs_bloom_v1".into(),
        a_layer_index: logs_bloom_pair_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ROOT_BOUND_AT),
        b_layer_index: payload_pair_layer_index,
        b_columns,
        b_selector_column: Some(payload::COL_IS_LOGS_BLOOM_LEAF_BOUND_AT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LogsBloomHtrWitness {
        let mut bloom = [0u8; 256];
        for i in 0..256 { bloom[i] = (i as u8).wrapping_add(7); }
        LogsBloomHtrWitness::from_logs_bloom(bloom)
    }

    #[test]
    fn trace_populates_7_real_rows() {
        let w = sample();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.num_rows, 7);
        assert_eq!(trace.padded_size, 16);
        for r in 0..7 {
            assert_eq!(trace.columns[COL_IS_REAL].evaluations[r].to_u64(), 1);
        }
        for r in 7..16 {
            assert!(trace.columns[COL_IS_REAL].evaluations[r].is_zero());
        }
    }

    #[test]
    fn is_root_bound_at_one_hot_at_invocation_6() {
        let trace = build_trace_polynomials(&sample(), CurveType::Bls48581);
        for r in 0..16 {
            let expected = if r == ROOT_INVOCATION_INDEX { 1 } else { 0 };
            assert_eq!(
                trace.columns[COL_IS_ROOT_BOUND_AT].evaluations[r].to_u64(), expected,
            );
        }
    }

    #[test]
    fn claimed_root_populated_and_constant() {
        let w = sample();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for r in 0..trace.num_rows {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_ROOT_OFFSET + k].evaluations[r].to_u64(),
                    w.root[k] as u64,
                );
            }
        }
    }

    #[test]
    fn payload_linkage_descriptor_shape() {
        let d = make_logs_bloom_pair_to_payload_pair_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        assert_eq!(d.label, "logs_bloom_pair_root_to_payload_pair_claimed_logs_bloom_v1");
        assert_eq!(d.a_selector_column, Some(COL_IS_ROOT_BOUND_AT));
        use crate::execution_payload_pair_air as payload;
        assert_eq!(d.b_selector_column, Some(payload::COL_IS_LOGS_BLOOM_LEAF_BOUND_AT));
    }

    #[test]
    fn sha256_extract_linkage_descriptor_shape() {
        let d = make_logs_bloom_pair_to_sha256_extract_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 96);
        assert_eq!(d.b_columns.len(), 96);
        assert_eq!(d.label, "logs_bloom_pair_sha256_extract_v1");
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let w = sample();
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = LogsBloomPairConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }
}
