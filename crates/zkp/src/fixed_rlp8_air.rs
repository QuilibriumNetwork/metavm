//! Fixed-width 8-byte RLP encoding gadget (nonce field).
//!
//! RLP: `0x88 || 8 bytes` = 9 bytes. Same pattern as fixed_rlp_air /
//! fixed_rlp20_air.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const FIELD_LEN: usize = 8;
pub const ENCODED_LEN: usize = 9;
pub const RLP8_PREFIX: u8 = 0x88;

pub const COL_FIELD_BYTE_OFFSET: usize = 0;
pub const COL_ENCODED_BYTE_OFFSET: usize = FIELD_LEN;
pub const COL_IS_REAL: usize = COL_ENCODED_BYTE_OFFSET + ENCODED_LEN;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 18

pub const NUM_ROW_CONSTRAINTS: usize = 3;
pub const NUM_SHIFTED: usize = 0;

#[derive(Clone, Debug, Default)]
pub struct FixedRlp8Witness { pub fields: Vec<[u8; FIELD_LEN]> }

pub fn build_trace_polynomials(w: &FixedRlp8Witness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.fields.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve); let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (i, field) in w.fields.iter().enumerate() {
        for k in 0..FIELD_LEN { columns[COL_FIELD_BYTE_OFFSET+k][i] = Scalar::from_u64(field[k] as u64, curve); }
        columns[COL_ENCODED_BYTE_OFFSET][i] = Scalar::from_u64(RLP8_PREFIX as u64, curve);
        for k in 0..FIELD_LEN { columns[COL_ENCODED_BYTE_OFFSET+1+k][i] = Scalar::from_u64(field[k] as u64, curve); }
        columns[COL_IS_REAL][i] = one.clone();
    }
    let polys: Vec<Polynomial> = columns.into_iter().map(|e| Polynomial { evaluations: e, degree: num_rows }).collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

pub struct FixedRlp8ConstraintSystem { pub num_rows: usize, pub omega: Option<Scalar>, pub domain_size: Option<u64> }
impl FixedRlp8ConstraintSystem {
    pub fn new(num_rows: usize) -> Self { Self { num_rows, omega: None, domain_size: None } }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self { self.omega = Some(omega); self.domain_size = Some(domain_size); self }
}

impl VmConstraintSystem for FixedRlp8ConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
    fn constraint_labels(&self) -> Vec<String> { vec!["is_real_binary".into(),"encoded_0_eq_prefix".into(),"bytes_match_rlc".into()] }
    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type(); let one = Scalar::one(curve); let n = columns[0].len();
        let pf = Scalar::from_u64(RLP8_PREFIX as u64, curve); let bt = Scalar::from_u64(7, curve);
        let mut b = vec![Scalar::zero(curve);n]; let mut p = vec![Scalar::zero(curve);n]; let mut m = vec![Scalar::zero(curve);n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r]; b[r] = v.mul(&v.sub(&one));
            p[r] = v.mul(&columns[COL_ENCODED_BYTE_OFFSET][r].sub(&pf));
            let mut acc = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for k in 0..FIELD_LEN { acc = acc.add(&bp.mul(&columns[COL_ENCODED_BYTE_OFFSET+1+k][r].sub(&columns[COL_FIELD_BYTE_OFFSET+k][r]))); bp = bp.mul(&bt); }
            m[r] = v.mul(&acc);
        }
        vec![b, p, m]
    }
    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type(); let one = Scalar::one(curve); let pf = Scalar::from_u64(RLP8_PREFIX as u64, curve);
        let v = &ce[COL_IS_REAL]; let b = v.mul(&v.sub(&one)); let p = v.mul(&ce[COL_ENCODED_BYTE_OFFSET].sub(&pf));
        let mut acc = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN { acc = acc.add(&bp.mul(&ce[COL_ENCODED_BYTE_OFFSET+1+k].sub(&ce[COL_FIELD_BYTE_OFFSET+k]))); bp = bp.mul(alpha); }
        let m = v.mul(&acc);
        let mut t = b; let mut ap = alpha.clone(); t = t.add(&ap.mul(&p)); ap = ap.mul(alpha); t = t.add(&ap.mul(&m)); t
    }
    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type(); let one_p = vec![Scalar::one(curve)]; let pf_p = vec![Scalar::from_u64(RLP8_PREFIX as u64, curve)];
        let v = &cc[COL_IS_REAL]; let b = poly_mul(v, &poly_sub(v, &one_p, curve), curve);
        let p = poly_mul(v, &poly_sub(&cc[COL_ENCODED_BYTE_OFFSET], &pf_p, curve), curve);
        let mut acc = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN { let d = poly_sub(&cc[COL_ENCODED_BYTE_OFFSET+1+k], &cc[COL_FIELD_BYTE_OFFSET+k], curve); acc = poly_add(&acc, &poly_scalar_mul(&d, &bp), curve); bp = bp.mul(alpha); }
        let m = poly_mul(v, &acc, curve);
        let mut t = b; let mut ap = alpha.clone(); t = poly_add(&t, &poly_scalar_mul(&p, &ap), curve); ap = ap.mul(alpha); t = poly_add(&t, &poly_scalar_mul(&m, &ap), curve); t
    }
    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, cols: &mut [Vec<Scalar>], nr: usize, ps: usize) {
        if nr == 0 || nr >= ps || cols.len() < NUM_COLUMNS { return; }
        let z = Scalar::zero(cols[0][0].curve_type());
        for c in cols.iter_mut().take(NUM_COLUMNS) { for cell in c.iter_mut().skip(nr).take(ps-nr) { *cell = z.clone(); } }
    }
    fn lookup_declarations(&self) -> LookupRequirements {
        let t = vec![LookupTable::range(8)]; let mut d = Vec::new();
        for k in 0..FIELD_LEN { d.push((LookupDeclaration{label:format!("rlp8_f_{}", k),column_index:COL_FIELD_BYTE_OFFSET+k,max_bits:8,selector_column:None},0)); }
        for k in 0..ENCODED_LEN { d.push((LookupDeclaration{label:format!("rlp8_e_{}", k),column_index:COL_ENCODED_BYTE_OFFSET+k,max_bits:8,selector_column:None},0)); }
        LookupRequirements{tables:t,declarations:d}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn constraints_zero() {
        let w = FixedRlp8Witness{fields:vec![[0xab;8]]}; let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = FixedRlp8ConstraintSystem::new(t.num_rows); let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p|&p.evaluations).collect();
        for (i,b) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() { for (r,v) in b.iter().enumerate() { assert!(v.is_zero(), "c{} r{}", i, r); } }
    }
    #[test] fn standalone_prove() {
        use crate::prover::prove_with_scheme; use crate::scheme::bls48581_scheme::Bls48581Scheme; use crate::scheme::CommitmentScheme; use crate::verifier::verify_with_scheme;
        let s = Bls48581Scheme::new(); s.init(); let w = FixedRlp8Witness{fields:vec![[0xcd;8]]};
        let t = build_trace_polynomials(&w, CurveType::Bls48581); let o = s.domain_generator(t.padded_size);
        let cs = FixedRlp8ConstraintSystem::new(t.num_rows).with_omega_and_domain(o, t.padded_size);
        assert!(verify_with_scheme(&prove_with_scheme(&t, &cs, &s), &cs, &s, CurveType::Bls48581));
    }
}
