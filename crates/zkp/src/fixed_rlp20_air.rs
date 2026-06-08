//! Fixed-width 20-byte RLP encoding gadget AIR.
//!
//! Companion to [`crate::fixed_rlp_air`] (which handles 32-byte fields).
//! This variant covers 20-byte fields: the Ethereum `beneficiary`
//! (coinbase) address in the block header.
//!
//! RLP encoding of a 20-byte string: always `0x94 || 20 bytes` = 21
//! bytes. Fixed-width, no leading-zero stripping.
//!
//! Constraint layout mirrors fixed_rlp_air exactly:
//!   - `is_real` binary
//!   - `encoded[0] = 0x94` (= `0x80 + 20`, gated by is_real)
//!   - `encoded[k+1] = field[k]` for k ∈ 0..20 (β-RLC)

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const FIELD_LEN: usize = 20;
pub const ENCODED_LEN: usize = 21;
pub const RLP20_PREFIX: u8 = 0x94;

pub const COL_FIELD_BYTE_OFFSET: usize = 0;
pub const COL_ENCODED_BYTE_OFFSET: usize = FIELD_LEN;
pub const COL_IS_REAL: usize = COL_ENCODED_BYTE_OFFSET + ENCODED_LEN;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 42

pub const NUM_ROW_CONSTRAINTS: usize = 3;
pub const NUM_SHIFTED: usize = 0;

#[derive(Clone, Debug, Default)]
pub struct FixedRlp20Witness {
    pub fields: Vec<[u8; FIELD_LEN]>,
}

pub fn build_trace_polynomials(
    witness: &FixedRlp20Witness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.fields.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, field) in witness.fields.iter().enumerate() {
        for k in 0..FIELD_LEN {
            columns[COL_FIELD_BYTE_OFFSET + k][i] =
                Scalar::from_u64(field[k] as u64, curve);
        }
        columns[COL_ENCODED_BYTE_OFFSET][i] = Scalar::from_u64(RLP20_PREFIX as u64, curve);
        for k in 0..FIELD_LEN {
            columns[COL_ENCODED_BYTE_OFFSET + 1 + k][i] =
                Scalar::from_u64(field[k] as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
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

pub struct FixedRlp20ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl FixedRlp20ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for FixedRlp20ConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "encoded_byte_0_eq_0x94".into(),
            "encoded_bytes_match_field_bytes_rlc".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let prefix = Scalar::from_u64(RLP20_PREFIX as u64, curve);
        let beta_test = Scalar::from_u64(7, curve);
        let mut bin = vec![Scalar::zero(curve); n];
        let mut pre = vec![Scalar::zero(curve); n];
        let mut bm = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin[r] = v.mul(&v.sub(&one));
            pre[r] = v.mul(&columns[COL_ENCODED_BYTE_OFFSET][r].sub(&prefix));
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..FIELD_LEN {
                let e = &columns[COL_ENCODED_BYTE_OFFSET + 1 + k][r];
                let f = &columns[COL_FIELD_BYTE_OFFSET + k][r];
                acc = acc.add(&bp.mul(&e.sub(f)));
                bp = bp.mul(&beta_test);
            }
            bm[r] = v.mul(&acc);
        }
        vec![bin, pre, bm]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let prefix = Scalar::from_u64(RLP20_PREFIX as u64, curve);
        let v = &col_evals[COL_IS_REAL];
        let bin = v.mul(&v.sub(&one));
        let pre = v.mul(&col_evals[COL_ENCODED_BYTE_OFFSET].sub(&prefix));
        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN {
            let e = &col_evals[COL_ENCODED_BYTE_OFFSET + 1 + k];
            let f = &col_evals[COL_FIELD_BYTE_OFFSET + k];
            acc = acc.add(&bp.mul(&e.sub(f)));
            bp = bp.mul(alpha);
        }
        let bm = v.mul(&acc);
        let mut total = bin;
        let mut ap = alpha.clone();
        total = total.add(&ap.mul(&pre));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&bm));
        total
    }

    fn build_constraint_polynomial(&self, col_coeffs: &[Vec<Scalar>], alpha: &Scalar, _domain_size: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let prefix_poly = vec![Scalar::from_u64(RLP20_PREFIX as u64, curve)];
        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let bin = poly_mul(v, &v_m1, curve);
        let e0 = &col_coeffs[COL_ENCODED_BYTE_OFFSET];
        let pre = poly_mul(v, &poly_sub(e0, &prefix_poly, curve), curve);
        let mut acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN {
            let e = &col_coeffs[COL_ENCODED_BYTE_OFFSET + 1 + k];
            let f = &col_coeffs[COL_FIELD_BYTE_OFFSET + k];
            let diff = poly_sub(e, f, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let bm = poly_mul(v, &acc, curve);
        let mut total = bin;
        let mut ap = alpha.clone();
        total = poly_add(&total, &poly_scalar_mul(&pre, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&bm, &ap), curve);
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
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
        for k in 0..FIELD_LEN {
            declarations.push((LookupDeclaration { label: format!("rlp20_field_{}_8bit", k), column_index: COL_FIELD_BYTE_OFFSET + k, max_bits: 8, selector_column: None }, 0));
        }
        for k in 0..ENCODED_LEN {
            declarations.push((LookupDeclaration { label: format!("rlp20_enc_{}_8bit", k), column_index: COL_ENCODED_BYTE_OFFSET + k, max_bits: 8, selector_column: None }, 0));
        }
        LookupRequirements { tables, declarations }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = FixedRlp20Witness { fields: vec![[0xab; 20]] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = FixedRlp20ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} at row {}", i, r);
            }
        }
    }

    #[test]
    fn prefix_fires_on_tamper() {
        let w = FixedRlp20Witness { fields: vec![[0; 20]] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_ENCODED_BYTE_OFFSET][0] = Scalar::from_u64(0xb8, CurveType::Bls48581);
        let cs = FixedRlp20ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[1][0].is_zero());
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = FixedRlp20Witness { fields: vec![[0xcd; 20]] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = FixedRlp20ConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581));
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 42);
        assert_eq!(RLP20_PREFIX, 0x94);
    }
}
