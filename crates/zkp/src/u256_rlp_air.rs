//! Variable-length u256 RLP encoding AIR.
//!
//! Extension of [`crate::u64_rlp_air`] to 32-byte values. Covers
//! `difficulty` (usually 0 post-Merge) and `base_fee_per_gas` in
//! Ethereum block headers. Same 3-case structure:
//!
//! ```text
//! value == 0      → [0x80]                       len 1
//! 1 ≤ v ≤ 0x7f    → [v as u8]                     len 1
//! v ≥ 0x80        → [0x80 + n, b_{n-1}, …, b_0]   len 1+n (n ≤ 32)
//! ```
//!
//! Note: n ≤ 32 ≤ 55, so we always use the "short string" RLP prefix
//! (no length-of-length).
//!
//! Column count: 170. Constraint count: 20. Same patterns as u64_rlp_air.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::vm_constraints::VmConstraintSystem;

const N: usize = 32; // byte width

pub const COL_BYTE_OFFSET: usize = 0;               // 0..32
pub const COL_LZ_MASK_OFFSET: usize = N;            // 32..64
pub const COL_BYTE_LAST_HIGH_BIT: usize = 2 * N;    // 64
pub const COL_BYTE_LAST_LOW7: usize = 2 * N + 1;    // 65
pub const COL_N_SIGNIFICANT: usize = 2 * N + 2;     // 66
pub const COL_N_EQ_OFFSET: usize = 2 * N + 3;       // 67..100 (33 one-hot, 0..=32)
pub const COL_IS_ZERO: usize = COL_N_EQ_OFFSET + N + 1;      // 100
pub const COL_IS_SHORT: usize = COL_IS_ZERO + 1;              // 101
pub const COL_IS_LONG: usize = COL_IS_SHORT + 1;              // 102
pub const COL_ENCODED_OFFSET: usize = COL_IS_LONG + 1;        // 103..136 (33 bytes max)
pub const COL_ENCODED_LEN: usize = COL_ENCODED_OFFSET + N + 1; // 136
pub const COL_IS_REAL: usize = COL_ENCODED_LEN + 1;           // 137
pub const COL_SHIFTED_BYTE_OFFSET: usize = COL_IS_REAL + 1;   // 138..170 (32 aux)
pub const NUM_COLUMNS: usize = COL_SHIFTED_BYTE_OFFSET + N;   // 170
pub const MAX_ENCODED_LEN: usize = N + 1; // 33

pub const NUM_ROW_CONSTRAINTS: usize = 20;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness + trace builder ──────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct U256RlpRow {
    pub value_be: [u8; N],
}

#[derive(Clone, Debug, Default)]
pub struct U256RlpWitness {
    pub rows: Vec<U256RlpRow>,
}

impl U256RlpWitness {
    pub fn from_be_values(values: &[[u8; N]]) -> Self {
        Self { rows: values.iter().map(|&value_be| U256RlpRow { value_be }).collect() }
    }
}

pub fn rlp_encode_u256_be(be: &[u8; N]) -> Vec<u8> {
    let first_nz = be.iter().position(|&b| b != 0);
    match first_nz {
        None => vec![0x80],
        Some(pos) => {
            let significant = &be[pos..];
            if significant.len() == 1 && significant[0] < 0x80 {
                vec![significant[0]]
            } else {
                let mut out = Vec::with_capacity(1 + significant.len());
                out.push(0x80 + significant.len() as u8);
                out.extend_from_slice(significant);
                out
            }
        }
    }
}

pub fn build_trace_polynomials(
    witness: &U256RlpWitness,
    curve: CurveType,
) -> crate::trace::TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        let be = &row.value_be;
        for i in 0..N { columns[COL_BYTE_OFFSET + i][r] = Scalar::from_u64(be[i] as u64, curve); }

        let mut prefix_all_zero = true;
        for i in 0..N {
            if be[i] != 0 { prefix_all_zero = false; }
            columns[COL_LZ_MASK_OFFSET + i][r] = Scalar::from_u64(prefix_all_zero as u64, curve);
        }

        let last = be[N - 1];
        columns[COL_BYTE_LAST_HIGH_BIT][r] = Scalar::from_u64((last >> 7) as u64, curve);
        columns[COL_BYTE_LAST_LOW7][r] = Scalar::from_u64((last & 0x7f) as u64, curve);

        let n: u64 = match be.iter().position(|&b| b != 0) {
            None => 0,
            Some(pos) => (N - pos) as u64,
        };
        columns[COL_N_SIGNIFICANT][r] = Scalar::from_u64(n, curve);
        for k in 0..=(N as u64) {
            columns[COL_N_EQ_OFFSET + k as usize][r] = Scalar::from_u64(if k == n { 1 } else { 0 }, curve);
        }

        let is_zero = n == 0;
        let is_short = n == 1 && last < 0x80;
        let is_long = !is_zero && !is_short;
        columns[COL_IS_ZERO][r] = Scalar::from_u64(is_zero as u64, curve);
        columns[COL_IS_SHORT][r] = Scalar::from_u64(is_short as u64, curve);
        columns[COL_IS_LONG][r] = Scalar::from_u64(is_long as u64, curve);

        let encoded = rlp_encode_u256_be(be);
        for (i, &eb) in encoded.iter().enumerate() {
            columns[COL_ENCODED_OFFSET + i][r] = Scalar::from_u64(eb as u64, curve);
        }
        columns[COL_ENCODED_LEN][r] = Scalar::from_u64(encoded.len() as u64, curve);
        columns[COL_IS_REAL][r] = one.clone();

        for k in 1..=N {
            let mut sb = 0u64;
            if n > 0 && (k as u64) <= n {
                sb = be[N - 1 + k - n as usize] as u64;
            }
            columns[COL_SHIFTED_BYTE_OFFSET + (k - 1)][r] = Scalar::from_u64(sb, curve);
        }
    }

    let polys: Vec<crate::trace::Polynomial> = columns
        .into_iter()
        .map(|evals| crate::trace::Polynomial { evaluations: evals, degree: num_rows })
        .collect();
    crate::trace::TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct U256RlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl U256RlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self { Self { num_rows, omega: None, domain_size: None } }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega); self.domain_size = Some(domain_size); self
    }
}

impl VmConstraintSystem for U256RlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(), "is_zero_binary".into(), "is_short_binary".into(),
            "is_long_binary".into(), "case_exclusivity".into(), "lz_mask_binary_rlc".into(),
            "lz_mask_monotonic_rlc".into(), "leading_zeros_rlc".into(),
            "byte_last_decomp".into(), "byte_last_hb_binary".into(),
            "is_zero_eq_mask_last".into(), "is_short_formula".into(),
            "n_eq_binary_rlc".into(), "n_eq_one_hot_sum".into(),
            "n_significant_linear_combo".into(), "encoded_len_formula".into(),
            "encoded_0_formula".into(),
            "shifted_byte_from_neq_byte_rlc".into(), "encoded_k_eq_long_times_shifted_rlc".into(),
            "unused_placeholder".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let bt = Scalar::from_u64(7, curve);
        let nn = columns[0].len();
        let mk = || vec![Scalar::zero(curve); nn];

        let (mut c0,mut c1,mut c2,mut c3,mut c4) = (mk(),mk(),mk(),mk(),mk());
        let (mut c5,mut c6,mut c7,mut c8,mut c9) = (mk(),mk(),mk(),mk(),mk());
        let (mut c10,mut c11,mut c12,mut c13,mut c14) = (mk(),mk(),mk(),mk(),mk());
        let (mut c15,mut c16,mut c17,mut c18,mut c19) = (mk(),mk(),mk(),mk(),mk());

        for r in 0..nn {
            let ir = &columns[COL_IS_REAL][r]; let iz = &columns[COL_IS_ZERO][r];
            let is = &columns[COL_IS_SHORT][r]; let il = &columns[COL_IS_LONG][r];
            let blh = &columns[COL_BYTE_LAST_HIGH_BIT][r];
            let bll = &columns[COL_BYTE_LAST_LOW7][r];
            let lz_last = &columns[COL_LZ_MASK_OFFSET + N - 1][r];
            let lz_prev = &columns[COL_LZ_MASK_OFFSET + N - 2][r];
            let bl = &columns[COL_BYTE_OFFSET + N - 1][r];
            let ns = &columns[COL_N_SIGNIFICANT][r];
            let el = &columns[COL_ENCODED_LEN][r];
            let e0 = &columns[COL_ENCODED_OFFSET][r];

            c0[r] = ir.mul(&ir.sub(&one));
            c1[r] = iz.mul(&iz.sub(&one));
            c2[r] = is.mul(&is.sub(&one));
            c3[r] = il.mul(&il.sub(&one));
            c4[r] = ir.mul(&iz.add(is).add(il).sub(ir));

            let mut a5 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for i in 0..N { let m = &columns[COL_LZ_MASK_OFFSET+i][r]; a5 = a5.add(&bp.mul(&m.mul(&m.sub(&one)))); bp = bp.mul(&bt); }
            c5[r] = a5;
            let mut a6 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for i in 0..N-1 { let mi = &columns[COL_LZ_MASK_OFFSET+i][r]; let mi1 = &columns[COL_LZ_MASK_OFFSET+i+1][r]; a6 = a6.add(&bp.mul(&mi1.mul(&one.sub(mi)))); bp = bp.mul(&bt); }
            c6[r] = a6;
            let mut a7 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for i in 0..N { let bi = &columns[COL_BYTE_OFFSET+i][r]; let mi = &columns[COL_LZ_MASK_OFFSET+i][r]; a7 = a7.add(&bp.mul(&bi.mul(mi))); bp = bp.mul(&bt); }
            c7[r] = a7;

            let one28 = Scalar::from_u64(128, curve);
            c8[r] = bl.sub(&blh.mul(&one28).add(bll));
            c9[r] = blh.mul(&blh.sub(&one));
            c10[r] = iz.sub(lz_last);
            c11[r] = is.sub(&lz_prev.sub(lz_last).mul(&one.sub(blh)));

            let mut a12 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for k in 0..=N { let neq = &columns[COL_N_EQ_OFFSET+k][r]; a12 = a12.add(&bp.mul(&neq.mul(&neq.sub(&one)))); bp = bp.mul(&bt); }
            c12[r] = a12;
            let mut neq_sum = Scalar::zero(curve);
            for k in 0..=N { neq_sum = neq_sum.add(&columns[COL_N_EQ_OFFSET+k][r]); }
            c13[r] = neq_sum.sub(ir);
            let mut nl = Scalar::zero(curve);
            for k in 0..=N { nl = nl.add(&Scalar::from_u64(k as u64, curve).mul(&columns[COL_N_EQ_OFFSET+k][r])); }
            c14[r] = nl.sub(ns);

            c15[r] = el.sub(&ir.add(&il.mul(ns)));
            let pf = Scalar::from_u64(0x80, curve);
            c16[r] = e0.sub(&pf.mul(&iz.add(il)).add(&is.mul(bl)).add(&il.mul(ns)));

            let mut a17 = Scalar::zero(curve); let mut a18 = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 1..=N {
                let sb = &columns[COL_SHIFTED_BYTE_OFFSET+(k-1)][r];
                let mut snb = Scalar::zero(curve);
                for t in k..=N { snb = snb.add(&columns[COL_N_EQ_OFFSET+t][r].mul(&columns[COL_BYTE_OFFSET+N-1+k-t][r])); }
                a17 = a17.add(&bp.mul(&sb.sub(&snb)));
                a18 = a18.add(&bp.mul(&columns[COL_ENCODED_OFFSET+k][r].sub(&il.mul(sb))));
                bp = bp.mul(&bt);
            }
            c17[r] = a17; c18[r] = a18;
            c19[r] = Scalar::zero(curve); // placeholder
        }
        vec![c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let ir = &ce[COL_IS_REAL]; let iz = &ce[COL_IS_ZERO];
        let is = &ce[COL_IS_SHORT]; let il = &ce[COL_IS_LONG];
        let blh = &ce[COL_BYTE_LAST_HIGH_BIT]; let bll = &ce[COL_BYTE_LAST_LOW7];
        let bl = &ce[COL_BYTE_OFFSET + N - 1];
        let lz_last = &ce[COL_LZ_MASK_OFFSET + N - 1];
        let lz_prev = &ce[COL_LZ_MASK_OFFSET + N - 2];
        let ns = &ce[COL_N_SIGNIFICANT]; let el = &ce[COL_ENCODED_LEN]; let e0 = &ce[COL_ENCODED_OFFSET];

        let c0 = ir.mul(&ir.sub(&one));
        let c1 = iz.mul(&iz.sub(&one));
        let c2 = is.mul(&is.sub(&one));
        let c3 = il.mul(&il.sub(&one));
        let c4 = ir.mul(&iz.add(is).add(il).sub(ir));

        let mut c5 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for i in 0..N { let m = &ce[COL_LZ_MASK_OFFSET+i]; c5 = c5.add(&bp.mul(&m.mul(&m.sub(&one)))); bp = bp.mul(alpha); }

        let mut c6 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for i in 0..N-1 { let mi = &ce[COL_LZ_MASK_OFFSET+i]; let mi1 = &ce[COL_LZ_MASK_OFFSET+i+1]; c6 = c6.add(&bp.mul(&mi1.mul(&one.sub(mi)))); bp = bp.mul(alpha); }

        let mut c7 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for i in 0..N { let bi = &ce[COL_BYTE_OFFSET+i]; let mi = &ce[COL_LZ_MASK_OFFSET+i]; c7 = c7.add(&bp.mul(&bi.mul(mi))); bp = bp.mul(alpha); }

        let one28 = Scalar::from_u64(128, curve);
        let c8 = bl.sub(&blh.mul(&one28).add(bll));
        let c9 = blh.mul(&blh.sub(&one));
        let c10 = iz.sub(lz_last);
        let c11 = is.sub(&lz_prev.sub(lz_last).mul(&one.sub(blh)));

        let mut c12 = Scalar::zero(curve); let mut bp = Scalar::one(curve);
        for k in 0..=N { let neq = &ce[COL_N_EQ_OFFSET+k]; c12 = c12.add(&bp.mul(&neq.mul(&neq.sub(&one)))); bp = bp.mul(alpha); }

        let mut neq_sum = Scalar::zero(curve);
        for k in 0..=N { neq_sum = neq_sum.add(&ce[COL_N_EQ_OFFSET+k]); }
        let c13 = neq_sum.sub(ir);

        let mut nl = Scalar::zero(curve);
        for k in 0..=N { nl = nl.add(&Scalar::from_u64(k as u64, curve).mul(&ce[COL_N_EQ_OFFSET+k])); }
        let c14 = nl.sub(ns);

        let c15 = el.sub(&ir.add(&il.mul(ns)));
        let pf = Scalar::from_u64(0x80, curve);
        let c16 = e0.sub(&pf.mul(&iz.add(il)).add(&is.mul(bl)).add(&il.mul(ns)));

        let mut c17 = Scalar::zero(curve); let mut c18 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 1..=N {
            let sb = &ce[COL_SHIFTED_BYTE_OFFSET+(k-1)];
            let mut snb = Scalar::zero(curve);
            for t in k..=N { snb = snb.add(&ce[COL_N_EQ_OFFSET+t].mul(&ce[COL_BYTE_OFFSET+N-1+k-t])); }
            c17 = c17.add(&bp.mul(&sb.sub(&snb)));
            c18 = c18.add(&bp.mul(&ce[COL_ENCODED_OFFSET+k].sub(&il.mul(sb))));
            bp = bp.mul(alpha);
        }
        let c19 = Scalar::zero(curve);

        let bodies = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19];
        let mut total = bodies[0].clone(); let mut ap = alpha.clone();
        for b in &bodies[1..] { total = total.add(&ap.mul(b)); ap = ap.mul(alpha); }
        total
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _domain_size: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_p, curve), curve);

        let ir = &cc[COL_IS_REAL]; let iz = &cc[COL_IS_ZERO];
        let is = &cc[COL_IS_SHORT]; let il = &cc[COL_IS_LONG];
        let blh = &cc[COL_BYTE_LAST_HIGH_BIT]; let bll = &cc[COL_BYTE_LAST_LOW7];
        let bl = &cc[COL_BYTE_OFFSET + N - 1];
        let lz_last = &cc[COL_LZ_MASK_OFFSET + N - 1];
        let lz_prev = &cc[COL_LZ_MASK_OFFSET + N - 2];
        let ns = &cc[COL_N_SIGNIFICANT]; let el = &cc[COL_ENCODED_LEN]; let e0 = &cc[COL_ENCODED_OFFSET];

        let c0 = bin(ir); let c1 = bin(iz); let c2 = bin(is); let c3 = bin(il);
        let sum_cases = poly_add(&poly_add(iz, is, curve), il, curve);
        let c4 = poly_mul(ir, &poly_sub(&sum_cases, ir, curve), curve);

        let mut c5 = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for i in 0..N { let m = &cc[COL_LZ_MASK_OFFSET+i]; c5 = poly_add(&c5, &poly_scalar_mul(&bin(m), &bp), curve); bp = bp.mul(alpha); }

        let mut c6 = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for i in 0..N-1 { let mi = &cc[COL_LZ_MASK_OFFSET+i]; let mi1 = &cc[COL_LZ_MASK_OFFSET+i+1]; let t = poly_mul(mi1, &poly_sub(&one_p, mi, curve), curve); c6 = poly_add(&c6, &poly_scalar_mul(&t, &bp), curve); bp = bp.mul(alpha); }

        let mut c7 = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for i in 0..N { let bi = &cc[COL_BYTE_OFFSET+i]; let mi = &cc[COL_LZ_MASK_OFFSET+i]; c7 = poly_add(&c7, &poly_scalar_mul(&poly_mul(bi, mi, curve), &bp), curve); bp = bp.mul(alpha); }

        let one28 = Scalar::from_u64(128, curve);
        let c8 = poly_sub(bl, &poly_add(&poly_scalar_mul(blh, &one28), bll, curve), curve);
        let c9 = bin(blh);
        let c10 = poly_sub(iz, lz_last, curve);
        let single = poly_sub(lz_prev, lz_last, curve);
        let lo_bit = poly_sub(&one_p, blh, curve);
        let c11 = poly_sub(is, &poly_mul(&single, &lo_bit, curve), curve);

        let mut c12 = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
        for k in 0..=N { let neq = &cc[COL_N_EQ_OFFSET+k]; c12 = poly_add(&c12, &poly_scalar_mul(&bin(neq), &bp), curve); bp = bp.mul(alpha); }

        let mut neq_sum = vec![Scalar::zero(curve)];
        for k in 0..=N { neq_sum = poly_add(&neq_sum, &cc[COL_N_EQ_OFFSET+k], curve); }
        let c13 = poly_sub(&neq_sum, ir, curve);

        let mut n_lin = vec![Scalar::zero(curve)];
        for k in 0..=N { let coef = Scalar::from_u64(k as u64, curve); n_lin = poly_add(&n_lin, &poly_scalar_mul(&cc[COL_N_EQ_OFFSET+k], &coef), curve); }
        let c14 = poly_sub(&n_lin, ns, curve);

        let c15 = poly_sub(el, &poly_add(ir, &poly_mul(il, ns, curve), curve), curve);
        let pf = Scalar::from_u64(0x80, curve);
        let zol = poly_add(iz, il, curve);
        let e0_exp = poly_add(&poly_add(&poly_scalar_mul(&zol, &pf), &poly_mul(is, bl, curve), curve), &poly_mul(il, ns, curve), curve);
        let c16 = poly_sub(e0, &e0_exp, curve);

        let mut c17 = vec![Scalar::zero(curve)]; let mut c18 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 1..=N {
            let sb = &cc[COL_SHIFTED_BYTE_OFFSET+(k-1)];
            let mut snb = vec![Scalar::zero(curve)];
            for t in k..=N { snb = poly_add(&snb, &poly_mul(&cc[COL_N_EQ_OFFSET+t], &cc[COL_BYTE_OFFSET+N-1+k-t], curve), curve); }
            c17 = poly_add(&c17, &poly_scalar_mul(&poly_sub(sb, &snb, curve), &bp), curve);
            c18 = poly_add(&c18, &poly_scalar_mul(&poly_sub(&cc[COL_ENCODED_OFFSET+k], &poly_mul(il, sb, curve), curve), &bp), curve);
            bp = bp.mul(alpha);
        }
        let c19 = vec![Scalar::zero(curve)];

        let bodies = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19];
        let mut total = bodies[0].clone(); let mut ap = alpha.clone();
        for b in &bodies[1..] { total = poly_add(&total, &poly_scalar_mul(b, &ap), curve); ap = ap.mul(alpha); }
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
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) { *cell = zero.clone(); }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(8)];
        let mut decls = Vec::new();
        for i in 0..N { decls.push((LookupDeclaration { label: format!("u256_rlp_byte_{}", i), column_index: COL_BYTE_OFFSET+i, max_bits: 8, selector_column: None }, 0)); }
        for i in 0..MAX_ENCODED_LEN { decls.push((LookupDeclaration { label: format!("u256_rlp_enc_{}", i), column_index: COL_ENCODED_OFFSET+i, max_bits: 8, selector_column: None }, 0)); }
        for i in 0..N { decls.push((LookupDeclaration { label: format!("u256_rlp_sb_{}", i), column_index: COL_SHIFTED_BYTE_OFFSET+i, max_bits: 8, selector_column: None }, 0)); }
        decls.push((LookupDeclaration { label: "u256_rlp_lo7".into(), column_index: COL_BYTE_LAST_LOW7, max_bits: 8, selector_column: None }, 0));
        LookupRequirements { tables, declarations: decls }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_for(values: &[[u8; N]]) -> crate::trace::TracePolynomials {
        let w = U256RlpWitness::from_be_values(values);
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    #[test]
    fn rlp_encode_covers_all_cases() {
        assert_eq!(rlp_encode_u256_be(&[0; 32]), vec![0x80]);
        let mut v1 = [0u8; 32]; v1[31] = 0x42;
        assert_eq!(rlp_encode_u256_be(&v1), vec![0x42]);
        let mut v2 = [0u8; 32]; v2[31] = 0x80;
        assert_eq!(rlp_encode_u256_be(&v2), vec![0x81, 0x80]);
        let mut v3 = [0u8; 32]; v3[30] = 0x01; v3[31] = 0x00;
        assert_eq!(rlp_encode_u256_be(&v3), vec![0x82, 0x01, 0x00]);
        assert_eq!(rlp_encode_u256_be(&[0xff; 32]).len(), 33);
        assert_eq!(rlp_encode_u256_be(&[0xff; 32])[0], 0x80 + 32);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let zero = [0u8; 32];
        let mut short = [0u8; 32]; short[31] = 0x42;
        let mut long1 = [0u8; 32]; long1[31] = 0x80;
        let mut long2 = [0u8; 32]; long2[27] = 0x01; long2[28] = 0x23; long2[29] = 0x45; long2[30] = 0x67; long2[31] = 0x89;
        let full = [0xff; 32];

        let trace = build_for(&[zero, short, long1, long2, full]);
        let cs = U256RlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} at row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 170);
        assert_eq!(MAX_ENCODED_LEN, 33);
    }

    #[test]
    fn evaluate_at_point_consistent_with_domain() {
        let zero = [0u8; 32];
        let mut short = [0u8; 32]; short[31] = 0x42;
        let mut long1 = [0u8; 32]; long1[31] = 0x80;
        let full = [0xff; 32];

        let trace = build_for(&[zero, short, long1, full]);
        let cs = U256RlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(0x1234, CurveType::Bls48581);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();

        // Evaluate at each row using point evaluation.
        for r in 0..trace.num_rows {
            let row_evals: Vec<Scalar> = col_refs.iter().map(|c| c[r].clone()).collect();
            let pt = cs.evaluate_at_point(&row_evals, &alpha);
            // On an honest witness, all constraint bodies are zero → point eval is zero.
            assert!(pt.is_zero(), "evaluate_at_point nonzero at row {}", r);
        }
    }

    #[test]
    #[ignore = "slow: standalone prove + verify (~60s release)"]
    fn u256_rlp_cs_prove_verify() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::verifier::verify_with_scheme;

        let zero = [0u8; 32];
        let mut short = [0u8; 32]; short[31] = 0x42;
        let mut long1 = [0u8; 32]; long1[31] = 0x80;
        let mut long5 = [0u8; 32]; long5[27] = 0x01; long5[28] = 0x23; long5[29] = 0x45; long5[30] = 0x67; long5[31] = 0x89;
        let full = [0xff; 32];

        let trace = build_for(&[zero, short, long1, long5, full]);
        let cs = U256RlpConstraintSystem::new(trace.num_rows);
        let scheme = Bls48581Scheme::new();
        let curve = CurveType::Bls48581;
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        let ok = verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(ok, "u256_rlp_air prove+verify failed");
    }

    #[test]
    fn tampered_is_zero_detected() {
        let mut v = [0u8; 32]; v[31] = 0x42; // short case
        let trace = build_for(&[v]);
        let cs = U256RlpConstraintSystem::new(trace.num_rows);
        let mut cols: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Force is_zero=1 on a non-zero value — should break constraint 4 (case exclusivity).
        cols[COL_IS_ZERO][0] = Scalar::one(CurveType::Bls48581);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let any_nonzero = res.iter().any(|body| body[0..trace.num_rows].iter().any(|v| !v.is_zero()));
        assert!(any_nonzero, "tampered is_zero should be detected");
    }

    #[test]
    fn tampered_encoded_len_detected() {
        let mut v = [0u8; 32]; v[30] = 0x01; v[31] = 0x00; // 2-byte value
        let trace = build_for(&[v]);
        let cs = U256RlpConstraintSystem::new(trace.num_rows);
        let mut cols: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper encoded_len to claim it's 1 instead of 3.
        cols[COL_ENCODED_LEN][0] = Scalar::from_u64(1, CurveType::Bls48581);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let any_nonzero = res.iter().any(|body| body[0..trace.num_rows].iter().any(|v| !v.is_zero()));
        assert!(any_nonzero, "tampered encoded_len should be detected");
    }

    #[test]
    fn encoded_bytes_match_canonical() {
        let mut v = [0u8; 32];
        // base_fee_per_gas ~ 15 gwei
        let bfpg = 15_000_000_000u64;
        v[24..32].copy_from_slice(&bfpg.to_be_bytes());
        let trace = build_for(&[v]);
        let expected = rlp_encode_u256_be(&v);
        let enc_len = trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64() as usize;
        assert_eq!(enc_len, expected.len());
        for i in 0..expected.len() {
            assert_eq!(
                trace.columns[COL_ENCODED_OFFSET + i].evaluations[0].to_u64(),
                expected[i] as u64,
            );
        }
    }
}
