//! Fixed-width 256-byte logs_bloom RLP encoding gadget.
//!
//! RLP of a 256-byte string uses the "long string" prefix:
//! `0xb9 0x01 0x00 || 256 bytes` = 259 bytes.
//!
//! - `0xb7 + 2` = `0xb9` (length-of-length is 2 since 256 = 0x0100
//!   needs 2 BE bytes)
//! - `0x01 0x00` = 256 in BE
//! - Then the 256 data bytes

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const FIELD_LEN: usize = 256;
pub const PREFIX_LEN: usize = 3; // 0xb9, 0x01, 0x00
pub const ENCODED_LEN: usize = PREFIX_LEN + FIELD_LEN; // 259
pub const PREFIX_BYTES: [u8; 3] = [0xb9, 0x01, 0x00];

pub const COL_FIELD_BYTE_OFFSET: usize = 0;           // 0..256
pub const COL_ENCODED_BYTE_OFFSET: usize = FIELD_LEN;  // 256..515
pub const COL_IS_REAL: usize = COL_ENCODED_BYTE_OFFSET + ENCODED_LEN; // 515
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 516

// Constraints:
//   0: is_real binary
//   1: encoded[0]=0xb9, encoded[1]=0x01, encoded[2]=0x00 (β-RLC)
//   2: encoded[3+k]=field[k] for k=0..256 (β-RLC)
pub const NUM_ROW_CONSTRAINTS: usize = 3;
pub const NUM_SHIFTED: usize = 0;

#[derive(Clone, Debug)]
pub struct RlpLogsBloomWitness { pub blooms: Vec<[u8; FIELD_LEN]> }

pub fn build_trace_polynomials(w: &RlpLogsBloomWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.blooms.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve); let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (i, bloom) in w.blooms.iter().enumerate() {
        for k in 0..FIELD_LEN { columns[COL_FIELD_BYTE_OFFSET+k][i] = Scalar::from_u64(bloom[k] as u64, curve); }
        for (j, &pb) in PREFIX_BYTES.iter().enumerate() { columns[COL_ENCODED_BYTE_OFFSET+j][i] = Scalar::from_u64(pb as u64, curve); }
        for k in 0..FIELD_LEN { columns[COL_ENCODED_BYTE_OFFSET+PREFIX_LEN+k][i] = Scalar::from_u64(bloom[k] as u64, curve); }
        columns[COL_IS_REAL][i] = one.clone();
    }
    let polys: Vec<Polynomial> = columns.into_iter().map(|e| Polynomial{evaluations:e,degree:num_rows}).collect();
    TracePolynomials{columns:polys,num_rows,padded_size:padded as u64,curve}
}

pub struct RlpLogsBloomConstraintSystem { pub num_rows: usize, pub omega: Option<Scalar>, pub domain_size: Option<u64> }
impl RlpLogsBloomConstraintSystem {
    pub fn new(n: usize) -> Self { Self{num_rows:n,omega:None,domain_size:None} }
    pub fn with_omega_and_domain(mut self, o: Scalar, d: u64) -> Self { self.omega=Some(o); self.domain_size=Some(d); self }
}

impl VmConstraintSystem for RlpLogsBloomConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
    fn constraint_labels(&self) -> Vec<String> { vec!["is_real_binary".into(),"prefix_eq_b9_01_00_rlc".into(),"body_bytes_match_rlc".into()] }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type(); let one = Scalar::one(curve); let n = columns[0].len();
        let bt = Scalar::from_u64(7, curve);
        let mut c0 = vec![Scalar::zero(curve);n]; let mut c1 = vec![Scalar::zero(curve);n]; let mut c2 = vec![Scalar::zero(curve);n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r]; c0[r] = v.mul(&v.sub(&one));
            let mut pa = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for j in 0..PREFIX_LEN {
                let expected = Scalar::from_u64(PREFIX_BYTES[j] as u64, curve);
                pa = pa.add(&bp.mul(&columns[COL_ENCODED_BYTE_OFFSET+j][r].sub(&expected)));
                bp = bp.mul(&bt);
            }
            c1[r] = v.mul(&pa);
            let mut ba = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for k in 0..FIELD_LEN {
                ba = ba.add(&bp.mul(&columns[COL_ENCODED_BYTE_OFFSET+PREFIX_LEN+k][r].sub(&columns[COL_FIELD_BYTE_OFFSET+k][r])));
                bp = bp.mul(&bt);
            }
            c2[r] = v.mul(&ba);
        }
        vec![c0,c1,c2]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type(); let one = Scalar::one(curve);
        let v = &ce[COL_IS_REAL]; let c0 = v.mul(&v.sub(&one));
        let mut pa = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for j in 0..PREFIX_LEN {
            let exp = Scalar::from_u64(PREFIX_BYTES[j] as u64, curve);
            pa = pa.add(&bp.mul(&ce[COL_ENCODED_BYTE_OFFSET+j].sub(&exp)));
            bp = bp.mul(alpha);
        }
        let c1 = v.mul(&pa);
        let mut ba = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN {
            ba = ba.add(&bp.mul(&ce[COL_ENCODED_BYTE_OFFSET+PREFIX_LEN+k].sub(&ce[COL_FIELD_BYTE_OFFSET+k])));
            bp = bp.mul(alpha);
        }
        let c2 = v.mul(&ba);
        let mut t = c0; let mut ap = alpha.clone(); t = t.add(&ap.mul(&c1)); ap = ap.mul(alpha); t = t.add(&ap.mul(&c2)); t
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type(); let one_p = vec![Scalar::one(curve)];
        let v = &cc[COL_IS_REAL]; let c0 = poly_mul(v, &poly_sub(v, &one_p, curve), curve);
        let mut pa = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for j in 0..PREFIX_LEN {
            let exp = vec![Scalar::from_u64(PREFIX_BYTES[j] as u64, curve)];
            pa = poly_add(&pa, &poly_scalar_mul(&poly_sub(&cc[COL_ENCODED_BYTE_OFFSET+j], &exp, curve), &bp), curve);
            bp = bp.mul(alpha);
        }
        let c1 = poly_mul(v, &pa, curve);
        let mut ba = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN {
            let d = poly_sub(&cc[COL_ENCODED_BYTE_OFFSET+PREFIX_LEN+k], &cc[COL_FIELD_BYTE_OFFSET+k], curve);
            ba = poly_add(&ba, &poly_scalar_mul(&d, &bp), curve); bp = bp.mul(alpha);
        }
        let c2 = poly_mul(v, &ba, curve);
        let mut t = c0; let mut ap = alpha.clone(); t = poly_add(&t, &poly_scalar_mul(&c1, &ap), curve); ap = ap.mul(alpha); t = poly_add(&t, &poly_scalar_mul(&c2, &ap), curve); t
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, cols: &mut [Vec<Scalar>], nr: usize, ps: usize) {
        if nr == 0 || nr >= ps || cols.len() < NUM_COLUMNS { return; }
        let z = Scalar::zero(cols[0][0].curve_type());
        for c in cols.iter_mut().take(NUM_COLUMNS) { for cell in c.iter_mut().skip(nr).take(ps-nr) { *cell = z.clone(); } }
    }
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn constraints_zero_on_honest() {
        let w = RlpLogsBloomWitness{blooms:vec![[0xAB;256]]}; let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = RlpLogsBloomConstraintSystem::new(t.num_rows); let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p|&p.evaluations).collect();
        for (i,b) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() { for (r,v) in b.iter().enumerate() { assert!(v.is_zero(), "c{} r{}", i, r); } }
    }
    #[test] fn prefix_fires_on_tamper() {
        let w = RlpLogsBloomWitness{blooms:vec![[0;256]]}; let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p|p.evaluations.clone()).collect();
        cols[COL_ENCODED_BYTE_OFFSET][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = RlpLogsBloomConstraintSystem::new(t.num_rows); let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        assert!(!cs.evaluate_on_domain(&cr, t.num_rows)[1][0].is_zero());
    }
    #[test] fn encoded_matches_rlp_spec() {
        let bloom = [0x42u8; 256]; let w = RlpLogsBloomWitness{blooms:vec![bloom]};
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(t.columns[COL_ENCODED_BYTE_OFFSET].evaluations[0].to_u64(), 0xb9);
        assert_eq!(t.columns[COL_ENCODED_BYTE_OFFSET+1].evaluations[0].to_u64(), 0x01);
        assert_eq!(t.columns[COL_ENCODED_BYTE_OFFSET+2].evaluations[0].to_u64(), 0x00);
        assert_eq!(t.columns[COL_ENCODED_BYTE_OFFSET+3].evaluations[0].to_u64(), 0x42);
    }
    #[test] fn num_columns_pinned() { assert_eq!(NUM_COLUMNS, 516); assert_eq!(ENCODED_LEN, 259); }
}
