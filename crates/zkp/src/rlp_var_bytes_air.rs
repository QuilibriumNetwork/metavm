//! Variable-length byte string RLP encoding gadget (step 0+1a).
//!
//! Handles `extra_data: ByteList[32]` and similar short variable-byte
//! fields. RLP encoding for byte strings of length 0..=55:
//!
//! ```text
//! length == 0                    → [0x80]                   len 1
//! length == 1 AND byte[0] < 0x80 → [byte[0]]               len 1
//! 1 ≤ length ≤ 55 (otherwise)   → [0x80 + length, data…]   len 1+length
//! ```
//!
//! Column layout (~100 cols): data[0..32] + is_active[0..32] +
//! data_len + encoded[0..33] + encoded_len + case selectors +
//! data0_high_bit + is_real.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

const MAX_DATA: usize = 32;
const MAX_ENC: usize = 33; // 1 prefix + 32 data

pub const COL_DATA_OFFSET: usize = 0;               // 0..32
pub const COL_IS_ACTIVE_OFFSET: usize = MAX_DATA;   // 32..64
pub const COL_DATA_LEN: usize = 2 * MAX_DATA;       // 64
pub const COL_DATA0_HIGH_BIT: usize = COL_DATA_LEN + 1; // 65
pub const COL_DATA0_LOW7: usize = COL_DATA0_HIGH_BIT + 1; // 66
pub const COL_IS_EMPTY: usize = COL_DATA0_LOW7 + 1; // 67
pub const COL_IS_SINGLE_SHORT: usize = COL_IS_EMPTY + 1; // 68
pub const COL_IS_MULTI: usize = COL_IS_SINGLE_SHORT + 1; // 69
pub const COL_ENCODED_OFFSET: usize = COL_IS_MULTI + 1; // 70..103
pub const COL_ENCODED_LEN: usize = COL_ENCODED_OFFSET + MAX_ENC; // 103
pub const COL_IS_REAL: usize = COL_ENCODED_LEN + 1; // 104
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 105

pub const NUM_ROW_CONSTRAINTS: usize = 12;
pub const NUM_SHIFTED: usize = 0;

pub fn rlp_encode_bytes(data: &[u8]) -> Vec<u8> {
    if data.is_empty() { return vec![0x80]; }
    if data.len() == 1 && data[0] < 0x80 { return vec![data[0]]; }
    let mut out = Vec::with_capacity(1 + data.len());
    out.push(0x80 + data.len() as u8);
    out.extend_from_slice(data);
    out
}

#[derive(Clone, Debug)]
pub struct VarBytesRlpWitness { pub entries: Vec<Vec<u8>> }

pub fn build_trace_polynomials(w: &VarBytesRlpWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.entries.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve); let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, data) in w.entries.iter().enumerate() {
        assert!(data.len() <= MAX_DATA);
        for (k, &b) in data.iter().enumerate() { columns[COL_DATA_OFFSET+k][r] = Scalar::from_u64(b as u64, curve); }
        for k in 0..MAX_DATA { columns[COL_IS_ACTIVE_OFFSET+k][r] = Scalar::from_u64(if k < data.len() { 1 } else { 0 }, curve); }
        columns[COL_DATA_LEN][r] = Scalar::from_u64(data.len() as u64, curve);
        let d0 = if data.is_empty() { 0u8 } else { data[0] };
        columns[COL_DATA0_HIGH_BIT][r] = Scalar::from_u64((d0 >> 7) as u64, curve);
        columns[COL_DATA0_LOW7][r] = Scalar::from_u64((d0 & 0x7f) as u64, curve);

        let is_empty = data.is_empty();
        let is_single_short = data.len() == 1 && data[0] < 0x80;
        let is_multi = !is_empty && !is_single_short;
        columns[COL_IS_EMPTY][r] = Scalar::from_u64(is_empty as u64, curve);
        columns[COL_IS_SINGLE_SHORT][r] = Scalar::from_u64(is_single_short as u64, curve);
        columns[COL_IS_MULTI][r] = Scalar::from_u64(is_multi as u64, curve);

        let encoded = rlp_encode_bytes(data);
        for (i, &eb) in encoded.iter().enumerate() { columns[COL_ENCODED_OFFSET+i][r] = Scalar::from_u64(eb as u64, curve); }
        columns[COL_ENCODED_LEN][r] = Scalar::from_u64(encoded.len() as u64, curve);
        columns[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = columns.into_iter().map(|e| Polynomial{evaluations:e,degree:num_rows}).collect();
    TracePolynomials{columns:polys,num_rows,padded_size:padded as u64,curve}
}

pub struct VarBytesRlpConstraintSystem { pub num_rows: usize, pub omega: Option<Scalar>, pub domain_size: Option<u64> }
impl VarBytesRlpConstraintSystem {
    pub fn new(n: usize) -> Self { Self{num_rows:n,omega:None,domain_size:None} }
    pub fn with_omega_and_domain(mut self, o: Scalar, d: u64) -> Self { self.omega=Some(o); self.domain_size=Some(d); self }
}

impl VmConstraintSystem for VarBytesRlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
    fn constraint_labels(&self) -> Vec<String> {
        vec!["is_real_binary".into(),"is_empty_binary".into(),"is_single_short_binary".into(),
             "is_multi_binary".into(),"case_exclusivity".into(),
             "is_active_binary_rlc".into(),"is_active_monotonic_rlc".into(),
             "is_active_sum_eq_data_len".into(),
             "data0_decomp".into(),"data0_hb_binary".into(),
             "encoded_len_formula".into(),"encoded_0_formula".into()]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type(); let one = Scalar::one(curve); let n = columns[0].len();
        let bt = Scalar::from_u64(7, curve);
        let mk = || vec![Scalar::zero(curve);n];
        let (mut c0,mut c1,mut c2,mut c3,mut c4) = (mk(),mk(),mk(),mk(),mk());
        let (mut c5,mut c6,mut c7,mut c8,mut c9) = (mk(),mk(),mk(),mk(),mk());
        let (mut c10,mut c11) = (mk(),mk());

        for r in 0..n {
            let ir = &columns[COL_IS_REAL][r]; let ie = &columns[COL_IS_EMPTY][r];
            let iss = &columns[COL_IS_SINGLE_SHORT][r]; let im = &columns[COL_IS_MULTI][r];
            let dl = &columns[COL_DATA_LEN][r];
            let d0 = &columns[COL_DATA_OFFSET][r];
            let hb = &columns[COL_DATA0_HIGH_BIT][r]; let lo = &columns[COL_DATA0_LOW7][r];
            let el = &columns[COL_ENCODED_LEN][r]; let e0 = &columns[COL_ENCODED_OFFSET][r];

            c0[r] = ir.mul(&ir.sub(&one));
            c1[r] = ie.mul(&ie.sub(&one));
            c2[r] = iss.mul(&iss.sub(&one));
            c3[r] = im.mul(&im.sub(&one));
            c4[r] = ir.mul(&ie.add(iss).add(im).sub(ir));

            let mut a5 = Scalar::zero(curve); let mut a6 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for k in 0..MAX_DATA {
                let ia = &columns[COL_IS_ACTIVE_OFFSET+k][r];
                a5 = a5.add(&bp.mul(&ia.mul(&ia.sub(&one))));
                if k < MAX_DATA - 1 {
                    let ia1 = &columns[COL_IS_ACTIVE_OFFSET+k+1][r];
                    a6 = a6.add(&bp.mul(&ia1.mul(&one.sub(ia))));
                }
                bp = bp.mul(&bt);
            }
            c5[r] = a5; c6[r] = a6;

            let mut ia_sum = Scalar::zero(curve);
            for k in 0..MAX_DATA { ia_sum = ia_sum.add(&columns[COL_IS_ACTIVE_OFFSET+k][r]); }
            c7[r] = ia_sum.sub(dl);

            let one28 = Scalar::from_u64(128, curve);
            c8[r] = d0.sub(&hb.mul(&one28).add(lo));
            c9[r] = hb.mul(&hb.sub(&one));

            // encoded_len: empty→1, single_short→1, multi→1+data_len
            let exp_el = ir.add(&im.mul(dl));
            c10[r] = el.sub(&exp_el);

            // encoded[0]: empty→0x80, single_short→data[0], multi→0x80+data_len
            let pf = Scalar::from_u64(0x80, curve);
            let exp_e0 = pf.mul(&ie.add(im)).add(&iss.mul(d0)).add(&im.mul(dl));
            c11[r] = e0.sub(&exp_e0);
        }
        vec![c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type(); let one = Scalar::one(curve);
        let ir = &ce[COL_IS_REAL]; let ie = &ce[COL_IS_EMPTY]; let iss = &ce[COL_IS_SINGLE_SHORT]; let im = &ce[COL_IS_MULTI];
        let dl = &ce[COL_DATA_LEN]; let d0 = &ce[COL_DATA_OFFSET]; let hb = &ce[COL_DATA0_HIGH_BIT]; let lo = &ce[COL_DATA0_LOW7];
        let el = &ce[COL_ENCODED_LEN]; let e0 = &ce[COL_ENCODED_OFFSET];

        let mut bodies = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(ir.mul(&ir.sub(&one)));
        bodies.push(ie.mul(&ie.sub(&one)));
        bodies.push(iss.mul(&iss.sub(&one)));
        bodies.push(im.mul(&im.sub(&one)));
        bodies.push(ir.mul(&ie.add(iss).add(im).sub(ir)));

        let mut a5 = Scalar::zero(curve); let mut a6 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for k in 0..MAX_DATA {
            let ia = &ce[COL_IS_ACTIVE_OFFSET+k];
            a5 = a5.add(&bp.mul(&ia.mul(&ia.sub(&one))));
            if k < MAX_DATA-1 { let ia1 = &ce[COL_IS_ACTIVE_OFFSET+k+1]; a6 = a6.add(&bp.mul(&ia1.mul(&one.sub(ia)))); }
            bp = bp.mul(alpha);
        }
        bodies.push(a5); bodies.push(a6);

        let mut ia_sum = Scalar::zero(curve);
        for k in 0..MAX_DATA { ia_sum = ia_sum.add(&ce[COL_IS_ACTIVE_OFFSET+k]); }
        bodies.push(ia_sum.sub(dl));

        let one28 = Scalar::from_u64(128, curve);
        bodies.push(d0.sub(&hb.mul(&one28).add(lo)));
        bodies.push(hb.mul(&hb.sub(&one)));
        bodies.push(el.sub(&ir.add(&im.mul(dl))));
        let pf = Scalar::from_u64(0x80, curve);
        bodies.push(e0.sub(&pf.mul(&ie.add(im)).add(&iss.mul(d0)).add(&im.mul(dl))));

        let mut total = bodies[0].clone(); let mut ap = alpha.clone();
        for b in &bodies[1..] { total = total.add(&ap.mul(b)); ap = ap.mul(alpha); }
        total
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type(); let one_p = vec![Scalar::one(curve)];
        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_p, curve), curve);
        let ir = &cc[COL_IS_REAL]; let ie = &cc[COL_IS_EMPTY]; let iss = &cc[COL_IS_SINGLE_SHORT]; let im = &cc[COL_IS_MULTI];
        let dl = &cc[COL_DATA_LEN]; let d0 = &cc[COL_DATA_OFFSET]; let hb = &cc[COL_DATA0_HIGH_BIT]; let lo = &cc[COL_DATA0_LOW7];
        let el = &cc[COL_ENCODED_LEN]; let e0 = &cc[COL_ENCODED_OFFSET];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(bin(ir)); bodies.push(bin(ie)); bodies.push(bin(iss)); bodies.push(bin(im));
        let sum_case = poly_add(&poly_add(ie, iss, curve), im, curve);
        bodies.push(poly_mul(ir, &poly_sub(&sum_case, ir, curve), curve));

        let mut a5 = vec![Scalar::zero(curve)]; let mut a6 = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for k in 0..MAX_DATA {
            let ia = &cc[COL_IS_ACTIVE_OFFSET+k];
            a5 = poly_add(&a5, &poly_scalar_mul(&bin(ia), &bp), curve);
            if k < MAX_DATA-1 { let ia1 = &cc[COL_IS_ACTIVE_OFFSET+k+1]; let t = poly_mul(ia1, &poly_sub(&one_p, ia, curve), curve); a6 = poly_add(&a6, &poly_scalar_mul(&t, &bp), curve); }
            bp = bp.mul(alpha);
        }
        bodies.push(a5); bodies.push(a6);
        let mut ia_sum = vec![Scalar::zero(curve)];
        for k in 0..MAX_DATA { ia_sum = poly_add(&ia_sum, &cc[COL_IS_ACTIVE_OFFSET+k], curve); }
        bodies.push(poly_sub(&ia_sum, dl, curve));

        let one28p = vec![Scalar::from_u64(128, curve)];
        bodies.push(poly_sub(d0, &poly_add(&poly_scalar_mul(hb, &one28p[0]), lo, curve), curve));
        bodies.push(bin(hb));
        bodies.push(poly_sub(el, &poly_add(ir, &poly_mul(im, dl, curve), curve), curve));
        let pfp = vec![Scalar::from_u64(0x80, curve)];
        let ie_im = poly_add(ie, im, curve);
        let exp = poly_add(&poly_add(&poly_scalar_mul(&ie_im, &pfp[0]), &poly_mul(iss, d0, curve), curve), &poly_mul(im, dl, curve), curve);
        bodies.push(poly_sub(e0, &exp, curve));

        let mut total = bodies[0].clone(); let mut ap = alpha.clone();
        for b in &bodies[1..] { total = poly_add(&total, &poly_scalar_mul(b, &ap), curve); ap = ap.mul(alpha); }
        total
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
        for k in 0..MAX_DATA { d.push((LookupDeclaration{label:format!("vb_data_{}",k),column_index:COL_DATA_OFFSET+k,max_bits:8,selector_column:None},0)); }
        for k in 0..MAX_ENC { d.push((LookupDeclaration{label:format!("vb_enc_{}",k),column_index:COL_ENCODED_OFFSET+k,max_bits:8,selector_column:None},0)); }
        d.push((LookupDeclaration{label:"vb_d0lo7".into(),column_index:COL_DATA0_LOW7,max_bits:8,selector_column:None},0));
        LookupRequirements{tables:t,declarations:d}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_for(entries: Vec<Vec<u8>>) -> TracePolynomials {
        let w = VarBytesRlpWitness { entries };
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    #[test]
    fn rlp_encode_covers_all_cases() {
        assert_eq!(rlp_encode_bytes(&[]), vec![0x80]);
        assert_eq!(rlp_encode_bytes(&[0x42]), vec![0x42]);
        assert_eq!(rlp_encode_bytes(&[0x80]), vec![0x81, 0x80]);
        assert_eq!(rlp_encode_bytes(&[0xDE, 0xAD]), vec![0x82, 0xDE, 0xAD]);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let trace = build_for(vec![vec![], vec![0x42], vec![0x80], vec![0xDE, 0xAD, 0xBE, 0xEF]]);
        let cs = VarBytesRlpConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} at row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn encoded_matches_canonical() {
        let entries = vec![vec![], vec![0x42], vec![0x80], vec![0xDE, 0xAD, 0xBE, 0xEF]];
        let trace = build_for(entries.clone());
        for (r, data) in entries.iter().enumerate() {
            let expected = rlp_encode_bytes(data);
            let len = trace.columns[COL_ENCODED_LEN].evaluations[r].to_u64() as usize;
            assert_eq!(len, expected.len(), "row {} encoded_len", r);
            for i in 0..expected.len() {
                assert_eq!(trace.columns[COL_ENCODED_OFFSET+i].evaluations[r].to_u64(), expected[i] as u64, "row {} enc[{}]", r, i);
            }
        }
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;
        let s = Bls48581Scheme::new(); s.init();
        let w = VarBytesRlpWitness { entries: vec![vec![], vec![0x42], vec![0x80], vec![0xDE, 0xAD]] };
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let o = s.domain_generator(t.padded_size);
        let cs = VarBytesRlpConstraintSystem::new(t.num_rows).with_omega_and_domain(o, t.padded_size);
        assert!(verify_with_scheme(&prove_with_scheme(&t, &cs, &s), &cs, &s, CurveType::Bls48581));
    }

    #[test] fn num_columns_pinned() { assert_eq!(NUM_COLUMNS, 105); }
}
