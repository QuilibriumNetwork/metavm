//! Widened variable-length byte-string RLP encoding gadget (up to 1024 bytes).
//!
//! Wider variant of `rlp_var_bytes_air` covering data lengths 0..=1024,
//! needed for `tx.data` and `log.data` payloads that exceed the 32-byte
//! cap of the original gadget.
//!
//! RLP encoding rules covered:
//!
//! ```text
//! length == 0                       → [0x80]                          len 1
//! length == 1 AND byte[0] < 0x80    → [byte[0]]                       len 1
//! 1 ≤ length ≤ 55  (otherwise)      → [0x80 + length, data…]          len 1+length
//! 56 ≤ length ≤ 255                 → [0xb8, length, data…]           len 2+length
//! 256 ≤ length ≤ 1024               → [0xb9, len_hi, len_lo, data…]   len 3+length
//! ```
//!
//! Column layout (≈3087 cols):
//!
//! ```text
//! [0..1024)       data bytes
//! [1024..2048)    is_active flags (monotonic 1...10...0)
//! 2048            data_len
//! 2049            data[0] high bit (used to distinguish single_short ↔ short)
//! 2050            data[0] low 7 bits
//! 2051            len_hi  (=> data_len / 256, only meaningful for is_long_2)
//! 2052            len_lo  (=> data_len % 256, only meaningful for is_long_2)
//! 2053            is_empty
//! 2054            is_single_short
//! 2055            is_short
//! 2056            is_long_1
//! 2057            is_long_2
//! [2058..3085)    encoded bytes (1 prefix + 2 len + 1024 data = 1027)
//! 3085            encoded_len
//! 3086            is_real
//! ```

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const MAX_DATA: usize = 1024;
pub const MAX_ENC: usize = 1027; // 1 prefix + 2 len + 1024 data

pub const COL_DATA_OFFSET: usize = 0; // 0..1024
pub const COL_IS_ACTIVE_OFFSET: usize = MAX_DATA; // 1024..2048
pub const COL_DATA_LEN: usize = 2 * MAX_DATA; // 2048
pub const COL_DATA0_HIGH_BIT: usize = COL_DATA_LEN + 1; // 2049
pub const COL_DATA0_LOW7: usize = COL_DATA0_HIGH_BIT + 1; // 2050
pub const COL_LEN_HI: usize = COL_DATA0_LOW7 + 1; // 2051
pub const COL_LEN_LO: usize = COL_LEN_HI + 1; // 2052
pub const COL_IS_EMPTY: usize = COL_LEN_LO + 1; // 2053
pub const COL_IS_SINGLE_SHORT: usize = COL_IS_EMPTY + 1; // 2054
pub const COL_IS_SHORT: usize = COL_IS_SINGLE_SHORT + 1; // 2055
pub const COL_IS_LONG_1: usize = COL_IS_SHORT + 1; // 2056
pub const COL_IS_LONG_2: usize = COL_IS_LONG_1 + 1; // 2057
pub const COL_ENCODED_OFFSET: usize = COL_IS_LONG_2 + 1; // 2058..3085
pub const COL_ENCODED_LEN: usize = COL_ENCODED_OFFSET + MAX_ENC; // 3085
pub const COL_IS_REAL: usize = COL_ENCODED_LEN + 1; // 3086
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 3087

pub const NUM_ROW_CONSTRAINTS: usize = 22;
pub const NUM_SHIFTED: usize = 0;

/// Canonical RLP byte-string encoder (host-side reference, also used to
/// populate the witness).
pub fn rlp_encode_bytes_wide(data: &[u8]) -> Vec<u8> {
    let len = data.len();
    assert!(len <= MAX_DATA, "data length {} exceeds MAX_DATA={}", len, MAX_DATA);
    if len == 0 {
        return vec![0x80];
    }
    if len == 1 && data[0] < 0x80 {
        return vec![data[0]];
    }
    if len <= 55 {
        let mut out = Vec::with_capacity(1 + len);
        out.push(0x80 + len as u8);
        out.extend_from_slice(data);
        return out;
    }
    if len <= 255 {
        let mut out = Vec::with_capacity(2 + len);
        out.push(0xb8);
        out.push(len as u8);
        out.extend_from_slice(data);
        return out;
    }
    // 256..=1024 → 2-byte length
    let mut out = Vec::with_capacity(3 + len);
    out.push(0xb9);
    out.push(((len >> 8) & 0xff) as u8);
    out.push((len & 0xff) as u8);
    out.extend_from_slice(data);
    out
}

#[derive(Clone, Debug)]
pub struct WideVarBytesWitness {
    pub entries: Vec<Vec<u8>>,
}

impl WideVarBytesWitness {
    /// Build a single-row witness for the given byte slice.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() <= MAX_DATA);
        Self { entries: vec![data.to_vec()] }
    }
}

pub fn build_trace_polynomials(w: &WideVarBytesWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.entries.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, data) in w.entries.iter().enumerate() {
        let len = data.len();
        assert!(len <= MAX_DATA, "row {} data len {} > {}", r, len, MAX_DATA);

        // Data bytes + is_active.
        for (k, &b) in data.iter().enumerate() {
            columns[COL_DATA_OFFSET + k][r] = Scalar::from_u64(b as u64, curve);
        }
        for k in 0..MAX_DATA {
            columns[COL_IS_ACTIVE_OFFSET + k][r] = Scalar::from_u64(if k < len { 1 } else { 0 }, curve);
        }

        // data_len & data[0] decomposition.
        columns[COL_DATA_LEN][r] = Scalar::from_u64(len as u64, curve);
        let d0 = if data.is_empty() { 0u8 } else { data[0] };
        columns[COL_DATA0_HIGH_BIT][r] = Scalar::from_u64((d0 >> 7) as u64, curve);
        columns[COL_DATA0_LOW7][r] = Scalar::from_u64((d0 & 0x7f) as u64, curve);

        // Length high/low bytes (always populated; binding constraint
        // simply enforces data_len = len_hi*256 + len_lo).
        columns[COL_LEN_HI][r] = Scalar::from_u64(((len >> 8) & 0xff) as u64, curve);
        columns[COL_LEN_LO][r] = Scalar::from_u64((len & 0xff) as u64, curve);

        // Case selectors.
        let is_empty = len == 0;
        let is_single_short = len == 1 && data[0] < 0x80;
        let is_short = !is_empty && !is_single_short && len <= 55;
        let is_long_1 = (56..=255).contains(&len);
        let is_long_2 = (256..=MAX_DATA).contains(&len);
        columns[COL_IS_EMPTY][r] = Scalar::from_u64(is_empty as u64, curve);
        columns[COL_IS_SINGLE_SHORT][r] = Scalar::from_u64(is_single_short as u64, curve);
        columns[COL_IS_SHORT][r] = Scalar::from_u64(is_short as u64, curve);
        columns[COL_IS_LONG_1][r] = Scalar::from_u64(is_long_1 as u64, curve);
        columns[COL_IS_LONG_2][r] = Scalar::from_u64(is_long_2 as u64, curve);

        // Encoded bytes.
        let encoded = rlp_encode_bytes_wide(data);
        for (i, &eb) in encoded.iter().enumerate() {
            columns[COL_ENCODED_OFFSET + i][r] = Scalar::from_u64(eb as u64, curve);
        }
        columns[COL_ENCODED_LEN][r] = Scalar::from_u64(encoded.len() as u64, curve);

        columns[COL_IS_REAL][r] = one.clone();
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

pub struct WideVarBytesConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl WideVarBytesConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain);
        self
    }
}

impl VmConstraintSystem for WideVarBytesConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),                  // 0
            "is_empty_binary".into(),                 // 1
            "is_single_short_binary".into(),          // 2
            "is_short_binary".into(),                 // 3
            "is_long_1_binary".into(),                // 4
            "is_long_2_binary".into(),                // 5
            "case_exclusivity".into(),                // 6
            "is_active_binary_rlc".into(),            // 7
            "is_active_monotonic_rlc".into(),         // 8
            "is_active_sum_eq_data_len".into(),       // 9
            "data0_decomp".into(),                    // 10
            "data0_hb_binary".into(),                 // 11
            "single_short_data0_below_0x80".into(),   // 12
            "len_hi_lo_eq_data_len".into(),           // 13
            "encoded_len_formula".into(),             // 14
            "encoded_0_formula".into(),               // 15
            "long1_encoded_1_eq_len".into(),          // 16
            "long2_encoded_1_eq_len_hi".into(),       // 17
            "long2_encoded_2_eq_len_lo".into(),       // 18
            "short_body_match_rlc".into(),            // 19
            "long1_body_match_rlc".into(),            // 20
            "long2_body_match_rlc".into(),            // 21
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let bt = Scalar::from_u64(7, curve);
        let mk = || vec![Scalar::zero(curve); n];
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS).map(|_| mk()).collect();

        let pf_short = Scalar::from_u64(0x80, curve);
        let pf_long1 = Scalar::from_u64(0xb8, curve);
        let pf_long2 = Scalar::from_u64(0xb9, curve);
        let one28 = Scalar::from_u64(128, curve);
        let two56 = Scalar::from_u64(256, curve);
        let two_p = Scalar::from_u64(2, curve);
        let three_p = Scalar::from_u64(3, curve);

        for r in 0..n {
            let ir = &columns[COL_IS_REAL][r];
            let ie = &columns[COL_IS_EMPTY][r];
            let iss = &columns[COL_IS_SINGLE_SHORT][r];
            let is = &columns[COL_IS_SHORT][r];
            let il1 = &columns[COL_IS_LONG_1][r];
            let il2 = &columns[COL_IS_LONG_2][r];
            let dl = &columns[COL_DATA_LEN][r];
            let d0 = &columns[COL_DATA_OFFSET][r];
            let hb = &columns[COL_DATA0_HIGH_BIT][r];
            let lo = &columns[COL_DATA0_LOW7][r];
            let lh = &columns[COL_LEN_HI][r];
            let ll = &columns[COL_LEN_LO][r];
            let el = &columns[COL_ENCODED_LEN][r];
            let e0 = &columns[COL_ENCODED_OFFSET][r];
            let e1 = &columns[COL_ENCODED_OFFSET + 1][r];
            let e2 = &columns[COL_ENCODED_OFFSET + 2][r];

            // 0..5: binary checks.
            bodies[0][r] = ir.mul(&ir.sub(&one));
            bodies[1][r] = ie.mul(&ie.sub(&one));
            bodies[2][r] = iss.mul(&iss.sub(&one));
            bodies[3][r] = is.mul(&is.sub(&one));
            bodies[4][r] = il1.mul(&il1.sub(&one));
            bodies[5][r] = il2.mul(&il2.sub(&one));

            // 6: case exclusivity (ir = sum of cases).
            let case_sum = ie.add(iss).add(is).add(il1).add(il2);
            bodies[6][r] = ir.mul(&case_sum.sub(ir));

            // 7,8: is_active flag binary + monotonic (β-RLC).
            let mut a7 = Scalar::zero(curve);
            let mut a8 = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..MAX_DATA {
                let ia = &columns[COL_IS_ACTIVE_OFFSET + k][r];
                a7 = a7.add(&bp.mul(&ia.mul(&ia.sub(&one))));
                if k + 1 < MAX_DATA {
                    let ia1 = &columns[COL_IS_ACTIVE_OFFSET + k + 1][r];
                    a8 = a8.add(&bp.mul(&ia1.mul(&one.sub(ia))));
                }
                bp = bp.mul(&bt);
            }
            bodies[7][r] = a7;
            bodies[8][r] = a8;

            // 9: sum is_active = data_len.
            let mut sum_ia = Scalar::zero(curve);
            for k in 0..MAX_DATA {
                sum_ia = sum_ia.add(&columns[COL_IS_ACTIVE_OFFSET + k][r]);
            }
            bodies[9][r] = sum_ia.sub(dl);

            // 10,11: data[0] decomp.
            bodies[10][r] = d0.sub(&hb.mul(&one28).add(lo));
            bodies[11][r] = hb.mul(&hb.sub(&one));

            // 12: single_short → data[0] < 0x80 (hb = 0).
            bodies[12][r] = iss.mul(hb);

            // 13: data_len = len_hi*256 + len_lo.
            bodies[13][r] = dl.sub(&lh.mul(&two56).add(ll));

            // 14: encoded_len.
            // empty→1, single_short→1, short→1+len, long_1→2+len, long_2→3+len
            let exp_el = ie.add(iss)
                .add(&is.mul(&one.add(dl)))
                .add(&il1.mul(&two_p.add(dl)))
                .add(&il2.mul(&three_p.add(dl)));
            bodies[14][r] = el.sub(&exp_el);

            // 15: encoded[0].
            // empty→0x80, single_short→data[0], short→0x80+len, long_1→0xb8, long_2→0xb9
            let exp_e0 = pf_short.mul(ie)
                .add(&iss.mul(d0))
                .add(&is.mul(&pf_short.add(dl)))
                .add(&il1.mul(&pf_long1))
                .add(&il2.mul(&pf_long2));
            bodies[15][r] = e0.sub(&exp_e0);

            // 16: long_1 → encoded[1] = data_len.
            bodies[16][r] = il1.mul(&e1.sub(dl));

            // 17: long_2 → encoded[1] = len_hi.
            bodies[17][r] = il2.mul(&e1.sub(lh));

            // 18: long_2 → encoded[2] = len_lo.
            bodies[18][r] = il2.mul(&e2.sub(ll));

            // 19,20,21: body byte matches (β-RLC, gated by case).
            // is_short: encoded[1+k] = data[k] for k < data_len (i.e. is_active[k]=1).
            // is_long_1: encoded[2+k] = data[k].
            // is_long_2: encoded[3+k] = data[k].
            let mut bs = Scalar::zero(curve);
            let mut bl1 = Scalar::zero(curve);
            let mut bl2 = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..MAX_DATA {
                let ia = &columns[COL_IS_ACTIVE_OFFSET + k][r];
                let dk = &columns[COL_DATA_OFFSET + k][r];
                let es = &columns[COL_ENCODED_OFFSET + 1 + k][r];
                let el1 = &columns[COL_ENCODED_OFFSET + 2 + k][r];
                let el2 = &columns[COL_ENCODED_OFFSET + 3 + k][r];
                bs = bs.add(&bp.mul(&ia.mul(&es.sub(dk))));
                bl1 = bl1.add(&bp.mul(&ia.mul(&el1.sub(dk))));
                bl2 = bl2.add(&bp.mul(&ia.mul(&el2.sub(dk))));
                bp = bp.mul(&bt);
            }
            bodies[19][r] = is.mul(&bs);
            bodies[20][r] = il1.mul(&bl1);
            bodies[21][r] = il2.mul(&bl2);
        }

        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let pf_short = Scalar::from_u64(0x80, curve);
        let pf_long1 = Scalar::from_u64(0xb8, curve);
        let pf_long2 = Scalar::from_u64(0xb9, curve);
        let one28 = Scalar::from_u64(128, curve);
        let two56 = Scalar::from_u64(256, curve);
        let two_p = Scalar::from_u64(2, curve);
        let three_p = Scalar::from_u64(3, curve);

        let ir = &ce[COL_IS_REAL];
        let ie = &ce[COL_IS_EMPTY];
        let iss = &ce[COL_IS_SINGLE_SHORT];
        let is = &ce[COL_IS_SHORT];
        let il1 = &ce[COL_IS_LONG_1];
        let il2 = &ce[COL_IS_LONG_2];
        let dl = &ce[COL_DATA_LEN];
        let d0 = &ce[COL_DATA_OFFSET];
        let hb = &ce[COL_DATA0_HIGH_BIT];
        let lo = &ce[COL_DATA0_LOW7];
        let lh = &ce[COL_LEN_HI];
        let ll = &ce[COL_LEN_LO];
        let el = &ce[COL_ENCODED_LEN];
        let e0 = &ce[COL_ENCODED_OFFSET];
        let e1 = &ce[COL_ENCODED_OFFSET + 1];
        let e2 = &ce[COL_ENCODED_OFFSET + 2];

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(ir.mul(&ir.sub(&one)));
        bodies.push(ie.mul(&ie.sub(&one)));
        bodies.push(iss.mul(&iss.sub(&one)));
        bodies.push(is.mul(&is.sub(&one)));
        bodies.push(il1.mul(&il1.sub(&one)));
        bodies.push(il2.mul(&il2.sub(&one)));
        let case_sum = ie.add(iss).add(is).add(il1).add(il2);
        bodies.push(ir.mul(&case_sum.sub(ir)));

        // is_active binary + monotonic (β = alpha).
        let mut a7 = Scalar::zero(curve);
        let mut a8 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_DATA {
            let ia = &ce[COL_IS_ACTIVE_OFFSET + k];
            a7 = a7.add(&bp.mul(&ia.mul(&ia.sub(&one))));
            if k + 1 < MAX_DATA {
                let ia1 = &ce[COL_IS_ACTIVE_OFFSET + k + 1];
                a8 = a8.add(&bp.mul(&ia1.mul(&one.sub(ia))));
            }
            bp = bp.mul(alpha);
        }
        bodies.push(a7);
        bodies.push(a8);

        let mut sum_ia = Scalar::zero(curve);
        for k in 0..MAX_DATA {
            sum_ia = sum_ia.add(&ce[COL_IS_ACTIVE_OFFSET + k]);
        }
        bodies.push(sum_ia.sub(dl));

        bodies.push(d0.sub(&hb.mul(&one28).add(lo)));
        bodies.push(hb.mul(&hb.sub(&one)));
        bodies.push(iss.mul(hb));
        bodies.push(dl.sub(&lh.mul(&two56).add(ll)));

        let exp_el = ie.add(iss)
            .add(&is.mul(&one.add(dl)))
            .add(&il1.mul(&two_p.add(dl)))
            .add(&il2.mul(&three_p.add(dl)));
        bodies.push(el.sub(&exp_el));

        let exp_e0 = pf_short.mul(ie)
            .add(&iss.mul(d0))
            .add(&is.mul(&pf_short.add(dl)))
            .add(&il1.mul(&pf_long1))
            .add(&il2.mul(&pf_long2));
        bodies.push(e0.sub(&exp_e0));

        bodies.push(il1.mul(&e1.sub(dl)));
        bodies.push(il2.mul(&e1.sub(lh)));
        bodies.push(il2.mul(&e2.sub(ll)));

        let mut bs = Scalar::zero(curve);
        let mut bl1 = Scalar::zero(curve);
        let mut bl2 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_DATA {
            let ia = &ce[COL_IS_ACTIVE_OFFSET + k];
            let dk = &ce[COL_DATA_OFFSET + k];
            let es = &ce[COL_ENCODED_OFFSET + 1 + k];
            let el1 = &ce[COL_ENCODED_OFFSET + 2 + k];
            let el2 = &ce[COL_ENCODED_OFFSET + 3 + k];
            bs = bs.add(&bp.mul(&ia.mul(&es.sub(dk))));
            bl1 = bl1.add(&bp.mul(&ia.mul(&el1.sub(dk))));
            bl2 = bl2.add(&bp.mul(&ia.mul(&el2.sub(dk))));
            bp = bp.mul(alpha);
        }
        bodies.push(is.mul(&bs));
        bodies.push(il1.mul(&bl1));
        bodies.push(il2.mul(&bl2));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _domain: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_p, curve), curve);

        let ir = &cc[COL_IS_REAL];
        let ie = &cc[COL_IS_EMPTY];
        let iss = &cc[COL_IS_SINGLE_SHORT];
        let is = &cc[COL_IS_SHORT];
        let il1 = &cc[COL_IS_LONG_1];
        let il2 = &cc[COL_IS_LONG_2];
        let dl = &cc[COL_DATA_LEN];
        let d0 = &cc[COL_DATA_OFFSET];
        let hb = &cc[COL_DATA0_HIGH_BIT];
        let lo = &cc[COL_DATA0_LOW7];
        let lh = &cc[COL_LEN_HI];
        let ll = &cc[COL_LEN_LO];
        let el = &cc[COL_ENCODED_LEN];
        let e0 = &cc[COL_ENCODED_OFFSET];
        let e1 = &cc[COL_ENCODED_OFFSET + 1];
        let e2 = &cc[COL_ENCODED_OFFSET + 2];

        let pf_short = Scalar::from_u64(0x80, curve);
        let pf_long1 = Scalar::from_u64(0xb8, curve);
        let pf_long2 = Scalar::from_u64(0xb9, curve);
        let one28 = Scalar::from_u64(128, curve);
        let two56 = Scalar::from_u64(256, curve);
        let two_p = Scalar::from_u64(2, curve);
        let three_p = Scalar::from_u64(3, curve);

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(bin(ir));
        bodies.push(bin(ie));
        bodies.push(bin(iss));
        bodies.push(bin(is));
        bodies.push(bin(il1));
        bodies.push(bin(il2));
        let case_sum = poly_add(&poly_add(&poly_add(&poly_add(ie, iss, curve), is, curve), il1, curve), il2, curve);
        bodies.push(poly_mul(ir, &poly_sub(&case_sum, ir, curve), curve));

        // is_active binary + monotonic.
        let mut a7: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut a8: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_DATA {
            let ia = &cc[COL_IS_ACTIVE_OFFSET + k];
            a7 = poly_add(&a7, &poly_scalar_mul(&bin(ia), &bp), curve);
            if k + 1 < MAX_DATA {
                let ia1 = &cc[COL_IS_ACTIVE_OFFSET + k + 1];
                let t = poly_mul(ia1, &poly_sub(&one_p, ia, curve), curve);
                a8 = poly_add(&a8, &poly_scalar_mul(&t, &bp), curve);
            }
            bp = bp.mul(alpha);
        }
        bodies.push(a7);
        bodies.push(a8);

        let mut sum_ia: Vec<Scalar> = vec![Scalar::zero(curve)];
        for k in 0..MAX_DATA {
            sum_ia = poly_add(&sum_ia, &cc[COL_IS_ACTIVE_OFFSET + k], curve);
        }
        bodies.push(poly_sub(&sum_ia, dl, curve));

        // data[0] decomp + hb binary.
        bodies.push(poly_sub(d0, &poly_add(&poly_scalar_mul(hb, &one28), lo, curve), curve));
        bodies.push(bin(hb));

        // single_short → hb = 0.
        bodies.push(poly_mul(iss, hb, curve));

        // data_len = len_hi*256 + len_lo.
        bodies.push(poly_sub(dl, &poly_add(&poly_scalar_mul(lh, &two56), ll, curve), curve));

        // encoded_len formula.
        let one_plus_dl = poly_add(&one_p, dl, curve);
        let two_plus_dl = poly_add(&vec![two_p.clone()], dl, curve);
        let three_plus_dl = poly_add(&vec![three_p.clone()], dl, curve);
        let exp_el = poly_add(
            &poly_add(
                &poly_add(
                    &poly_add(ie, iss, curve),
                    &poly_mul(is, &one_plus_dl, curve),
                    curve,
                ),
                &poly_mul(il1, &two_plus_dl, curve),
                curve,
            ),
            &poly_mul(il2, &three_plus_dl, curve),
            curve,
        );
        bodies.push(poly_sub(el, &exp_el, curve));

        // encoded[0] formula.
        let pf_short_plus_dl = poly_add(&vec![pf_short.clone()], dl, curve);
        let exp_e0 = poly_add(
            &poly_add(
                &poly_add(
                    &poly_add(
                        &poly_scalar_mul(ie, &pf_short),
                        &poly_mul(iss, d0, curve),
                        curve,
                    ),
                    &poly_mul(is, &pf_short_plus_dl, curve),
                    curve,
                ),
                &poly_scalar_mul(il1, &pf_long1),
                curve,
            ),
            &poly_scalar_mul(il2, &pf_long2),
            curve,
        );
        bodies.push(poly_sub(e0, &exp_e0, curve));

        // long_1 → encoded[1] = data_len.
        bodies.push(poly_mul(il1, &poly_sub(e1, dl, curve), curve));
        // long_2 → encoded[1] = len_hi.
        bodies.push(poly_mul(il2, &poly_sub(e1, lh, curve), curve));
        // long_2 → encoded[2] = len_lo.
        bodies.push(poly_mul(il2, &poly_sub(e2, ll, curve), curve));

        // Body byte matches per case.
        let mut bs: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bl1: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bl2: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_DATA {
            let ia = &cc[COL_IS_ACTIVE_OFFSET + k];
            let dk = &cc[COL_DATA_OFFSET + k];
            let es = &cc[COL_ENCODED_OFFSET + 1 + k];
            let el1 = &cc[COL_ENCODED_OFFSET + 2 + k];
            let el2 = &cc[COL_ENCODED_OFFSET + 3 + k];
            let ts = poly_mul(ia, &poly_sub(es, dk, curve), curve);
            let tl1 = poly_mul(ia, &poly_sub(el1, dk, curve), curve);
            let tl2 = poly_mul(ia, &poly_sub(el2, dk, curve), curve);
            bs = poly_add(&bs, &poly_scalar_mul(&ts, &bp), curve);
            bl1 = poly_add(&bl1, &poly_scalar_mul(&tl1, &bp), curve);
            bl2 = poly_add(&bl2, &poly_scalar_mul(&tl2, &bp), curve);
            bp = bp.mul(alpha);
        }
        bodies.push(poly_mul(is, &bs, curve));
        bodies.push(poly_mul(il1, &bl1, curve));
        bodies.push(poly_mul(il2, &bl2, curve));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(&self, cols: &mut [Vec<Scalar>], nr: usize, ps: usize) {
        if nr == 0 || nr >= ps || cols.len() < NUM_COLUMNS { return; }
        let z = Scalar::zero(cols[0][0].curve_type());
        for c in cols.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(nr).take(ps - nr) { *cell = z.clone(); }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // Single shared 8-bit range table across all byte-valued columns
        // (data, encoded, data0_low7, len_hi, len_lo).
        let tables = vec![LookupTable::range(8)];
        let mut decls = Vec::new();
        for k in 0..MAX_DATA {
            decls.push((
                LookupDeclaration {
                    label: format!("wvb_data_{}", k),
                    column_index: COL_DATA_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MAX_ENC {
            decls.push((
                LookupDeclaration {
                    label: format!("wvb_enc_{}", k),
                    column_index: COL_ENCODED_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        decls.push((
            LookupDeclaration {
                label: "wvb_data0_low7".into(),
                column_index: COL_DATA0_LOW7,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        decls.push((
            LookupDeclaration {
                label: "wvb_len_hi".into(),
                column_index: COL_LEN_HI,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        decls.push((
            LookupDeclaration {
                label: "wvb_len_lo".into(),
                column_index: COL_LEN_LO,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        LookupRequirements { tables, declarations: decls }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_for(entries: Vec<Vec<u8>>) -> TracePolynomials {
        let w = WideVarBytesWitness { entries };
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    fn evaluate(trace: &TracePolynomials) -> Vec<Vec<Scalar>> {
        let cs = WideVarBytesConstraintSystem::new(trace.num_rows);
        let refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&refs, trace.num_rows)
    }

    #[test]
    fn rlp_encode_covers_all_cases() {
        assert_eq!(rlp_encode_bytes_wide(&[]), vec![0x80]);
        assert_eq!(rlp_encode_bytes_wide(&[0x42]), vec![0x42]);
        assert_eq!(rlp_encode_bytes_wide(&[0x80]), vec![0x81, 0x80]);
        let short55: Vec<u8> = (0..55u8).collect();
        let enc55 = rlp_encode_bytes_wide(&short55);
        assert_eq!(enc55.len(), 56);
        assert_eq!(enc55[0], 0x80 + 55);
        assert_eq!(&enc55[1..], &short55[..]);

        let long100: Vec<u8> = (0..100u8).collect();
        let enc100 = rlp_encode_bytes_wide(&long100);
        assert_eq!(enc100[0], 0xb8);
        assert_eq!(enc100[1], 100);
        assert_eq!(&enc100[2..], &long100[..]);

        let long500: Vec<u8> = (0..500).map(|i| (i & 0xff) as u8).collect();
        let enc500 = rlp_encode_bytes_wide(&long500);
        assert_eq!(enc500[0], 0xb9);
        assert_eq!(enc500[1], 0x01);
        assert_eq!(enc500[2], 0xf4);
        assert_eq!(&enc500[3..], &long500[..]);

        let long1024 = vec![0xCDu8; 1024];
        let enc1024 = rlp_encode_bytes_wide(&long1024);
        assert_eq!(enc1024[0], 0xb9);
        assert_eq!(enc1024[1], 0x04);
        assert_eq!(enc1024[2], 0x00);
    }

    #[test]
    fn empty_row_constraints_pass() {
        let trace = build_for(vec![vec![]]);
        for (i, body) in evaluate(&trace).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "empty: constraint {} row {} nonzero", i, r);
            }
        }
        assert_eq!(trace.columns[COL_ENCODED_OFFSET].evaluations[0].to_u64(), 0x80);
        assert_eq!(trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn single_short_row_constraints_pass() {
        let trace = build_for(vec![vec![0x42]]);
        for (i, body) in evaluate(&trace).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "single_short: constraint {} row {} nonzero", i, r);
            }
        }
        assert_eq!(trace.columns[COL_ENCODED_OFFSET].evaluations[0].to_u64(), 0x42);
        assert_eq!(trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn short_55_byte_row_constraints_pass() {
        let data: Vec<u8> = (0..55u8).collect();
        let trace = build_for(vec![data.clone()]);
        for (i, body) in evaluate(&trace).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "short55: constraint {} row {} nonzero", i, r);
            }
        }
        assert_eq!(trace.columns[COL_ENCODED_OFFSET].evaluations[0].to_u64(), 0x80 + 55);
        assert_eq!(trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64(), 56);
    }

    #[test]
    fn long1_100_byte_row_constraints_pass() {
        let data: Vec<u8> = (0..100u8).collect();
        let trace = build_for(vec![data.clone()]);
        for (i, body) in evaluate(&trace).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "long1_100: constraint {} row {} nonzero", i, r);
            }
        }
        assert_eq!(trace.columns[COL_ENCODED_OFFSET].evaluations[0].to_u64(), 0xb8);
        assert_eq!(trace.columns[COL_ENCODED_OFFSET + 1].evaluations[0].to_u64(), 100);
        assert_eq!(trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64(), 102);
    }

    #[test]
    fn long2_500_byte_row_constraints_pass() {
        let data: Vec<u8> = (0..500).map(|i| (i & 0xff) as u8).collect();
        let trace = build_for(vec![data.clone()]);
        for (i, body) in evaluate(&trace).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "long2_500: constraint {} row {} nonzero", i, r);
            }
        }
        assert_eq!(trace.columns[COL_ENCODED_OFFSET].evaluations[0].to_u64(), 0xb9);
        assert_eq!(trace.columns[COL_ENCODED_OFFSET + 1].evaluations[0].to_u64(), 0x01);
        assert_eq!(trace.columns[COL_ENCODED_OFFSET + 2].evaluations[0].to_u64(), 0xf4);
        assert_eq!(trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64(), 503);
    }

    #[test]
    fn long2_1024_max_size_row_constraints_pass() {
        let data = vec![0xABu8; MAX_DATA];
        let trace = build_for(vec![data]);
        for (i, body) in evaluate(&trace).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "long2_1024: constraint {} row {} nonzero", i, r);
            }
        }
        assert_eq!(trace.columns[COL_ENCODED_OFFSET].evaluations[0].to_u64(), 0xb9);
        assert_eq!(trace.columns[COL_ENCODED_OFFSET + 1].evaluations[0].to_u64(), 0x04);
        assert_eq!(trace.columns[COL_ENCODED_OFFSET + 2].evaluations[0].to_u64(), 0x00);
        assert_eq!(trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64(), 1027);
    }

    #[test]
    fn tampered_body_byte_detected() {
        let data: Vec<u8> = (0..100u8).collect();
        let trace = build_for(vec![data]);
        let mut cols: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Corrupt encoded[5] (which lives in the long_1 body region).
        cols[COL_ENCODED_OFFSET + 5][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = WideVarBytesConstraintSystem::new(trace.num_rows);
        let refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&refs, trace.num_rows);
        // Body 20 = long1_body_match_rlc.
        assert!(!bodies[20][0].is_zero(), "tampered body byte not detected");
    }

    #[test]
    fn tampered_prefix_byte_detected() {
        let data: Vec<u8> = (0..500).map(|i| (i & 0xff) as u8).collect();
        let trace = build_for(vec![data]);
        let mut cols: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_ENCODED_OFFSET][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = WideVarBytesConstraintSystem::new(trace.num_rows);
        let refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&refs, trace.num_rows);
        // Body 15 = encoded_0_formula.
        assert!(!bodies[15][0].is_zero(), "tampered prefix byte not detected");
    }

    #[test]
    fn descriptor_well_formed() {
        let cs = WideVarBytesConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        let labels = cs.constraint_labels();
        assert_eq!(labels.len(), NUM_ROW_CONSTRAINTS);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 1024 data + 1027 encoded + (data0_low7, len_hi, len_lo) = 2054
        assert_eq!(reqs.declarations.len(), MAX_DATA + MAX_ENC + 3);
        for (decl, _) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
        }
        assert_eq!(cs.selector_column_indices(), vec![COL_IS_REAL]);
    }

    #[test]
    fn from_bytes_builder_round_trip() {
        let data: Vec<u8> = (0..200u8).collect();
        let w = WideVarBytesWitness::from_bytes(&data);
        assert_eq!(w.entries.len(), 1);
        assert_eq!(w.entries[0], data);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for (i, body) in evaluate(&trace).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "from_bytes: constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 3087);
        assert_eq!(MAX_DATA, 1024);
        assert_eq!(MAX_ENC, 1027);
    }
}
