//! Algebraic limb-level AIR for **BN254** non-native `Fp` arithmetic.
//!
//! Mirrors [`crate::secp256k1_fp_air`] (and ultimately
//! [`crate::nonnative_fp_air`]) but specialised to the BN254 base prime
//!
//! ```text
//! p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//!   = 0x30644E72E131A029B85045B68181585D97816A916871CA8D3C208C16D87CFD47
//! ```
//!
//! All field elements are 254-bit (top two bits of the high limb are
//! zero), held as **4 big-endian u64 limbs**.
//!
//! # Why this module
//!
//! [`crate::bn254_curve_ops_air`] commits the IO of one G1/G2 curve
//! operation per row but **does not algebraically enforce** the curve
//! law or any underlying `Fp` arithmetic — see the parent module's
//! scope note. This submodule provides the **Fp Add / Sub / Mul / Inv**
//! AIR that the deferred curve-law decomposition will route into.
//!
//! ## Deferred
//!
//! - Fp2 / Fp6 / Fp12 algebraic decomposition (Miller loop, final exp).
//!   Fp2 ops reduce to a constant-coefficient combination of Fp ops
//!   which can re-use this AIR row-wise; the row-sequencing required to
//!   prove `R = P + Q` over `G1Affine` algebraically (rather than via
//!   the curve-ops scaffold's host-side commitment) is a follow-up
//!   tracked under the parent module's "Once the deferred phase lands"
//!   note.
//! - Byte-level Rx/Ry binding to [`crate::bn254_precompile_air`] (which
//!   commits the precompile's 32-byte output). The descriptor in this
//!   module is at **limb level** (4 BE u64s); the byte alignment to
//!   the precompile AIR requires adding byte-decomposition columns on
//!   either side and is part of the deferred phase.
//!
//! # Row shape
//!
//! Identical algebraic shape to [`crate::secp256k1_fp_air`]'s **102
//! columns** with `LIMBS_PER_FP = 4`, `LIMBS_PER_PROD = 8`. Only the
//! modulus constants change (BN254 `p` vs secp256k1 `p`).
//!
//! ```text
//! offset  name              size  notes
//! 0       a_limbs            4    input A (BE)
//! 4       b_limbs            4    input B (BE)
//! 8       c_limbs            4    output = op(A, B) (BE)
//!
//! -- Add aux --
//! 12      add_unred_sum      4
//! 16      add_carries        4    LSB-first
//! 20      add_reduce_flag    1
//! 21      add_trial_diff     4
//! 25      add_sub_borrows    4    LSB-first
//!
//! -- Sub aux --
//! 29      sub_unred_diff     4
//! 33      sub_borrows        4    LSB-first
//! 37      sub_add_back_flag  1
//! 38      sub_trial_sum      4
//! 42      sub_add_carries    4    LSB-first
//!
//! -- Mul aux (Barrett-style q·p + r = a·b) --
//! 46      prod_limbs         8    8-limb product a·b (BE)
//! 54      quotient_limbs     4
//! 58      qp_limbs           8    8-limb product q·p (BE)
//! 66      mul_carry_ab       8    LSB-first
//! 74      mul_carry_qp       8    LSB-first
//! 82      mul_sum_carry      8    LSB-first binary
//!
//! -- Slack (c < p) --
//! 90      slack_limbs        4    p - 1 - c (BE)
//! 94      slack_borrows      4    LSB-first
//!
//! -- Selectors --
//! 98      sel_add            1
//! 99      sel_sub            1
//! 100     sel_mul            1
//! 101     sel_inv            1   reuses Mul aux + asserts c = 1
//! ```
//!
//! # Constraints — 19 categories
//!
//! Identical shape to [`crate::secp256k1_fp_air`] — see that module
//! for the catalogue. Inv reuses the entire Mul body (host-side
//! witness `b = a^{-1}`, `c = 1`) and only adds a `c == 1` constraint.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupDeclaration;

// ─── Modulus ─────────────────────────────────────────────────────────────

/// BN254 base-field prime `p`, big-endian 4 × u64 limbs.
///
/// `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`.
pub const P_LIMBS: [u64; 4] = super::BN254_P_LIMBS;

// ─── Column layout constants ─────────────────────────────────────────────

pub const LIMBS_PER_FP: usize = 4;
pub const LIMBS_PER_PROD: usize = 8;

pub const COL_A_OFFSET:              usize = 0;
pub const COL_B_OFFSET:              usize = COL_A_OFFSET + LIMBS_PER_FP;
pub const COL_C_OFFSET:              usize = COL_B_OFFSET + LIMBS_PER_FP;

// Add aux
pub const COL_ADD_UNRED_SUM_OFFSET:  usize = COL_C_OFFSET + LIMBS_PER_FP;
pub const COL_ADD_CARRIES_OFFSET:    usize = COL_ADD_UNRED_SUM_OFFSET + LIMBS_PER_FP;
pub const COL_ADD_REDUCE_FLAG:       usize = COL_ADD_CARRIES_OFFSET + LIMBS_PER_FP;
pub const COL_ADD_TRIAL_DIFF_OFFSET: usize = COL_ADD_REDUCE_FLAG + 1;
pub const COL_ADD_SUB_BORROWS_OFFSET:usize = COL_ADD_TRIAL_DIFF_OFFSET + LIMBS_PER_FP;

// Sub aux
pub const COL_SUB_UNRED_DIFF_OFFSET: usize = COL_ADD_SUB_BORROWS_OFFSET + LIMBS_PER_FP;
pub const COL_SUB_BORROWS_OFFSET:    usize = COL_SUB_UNRED_DIFF_OFFSET + LIMBS_PER_FP;
pub const COL_SUB_ADD_BACK_FLAG:     usize = COL_SUB_BORROWS_OFFSET + LIMBS_PER_FP;
pub const COL_SUB_TRIAL_SUM_OFFSET:  usize = COL_SUB_ADD_BACK_FLAG + 1;
pub const COL_SUB_ADD_CARRIES_OFFSET:usize = COL_SUB_TRIAL_SUM_OFFSET + LIMBS_PER_FP;

// Mul aux
pub const COL_PROD_OFFSET:           usize = COL_SUB_ADD_CARRIES_OFFSET + LIMBS_PER_FP;
pub const COL_QUOTIENT_OFFSET:       usize = COL_PROD_OFFSET + LIMBS_PER_PROD;
pub const COL_QP_OFFSET:             usize = COL_QUOTIENT_OFFSET + LIMBS_PER_FP;
pub const COL_MUL_CARRY_AB_OFFSET:   usize = COL_QP_OFFSET + LIMBS_PER_PROD;
pub const COL_MUL_CARRY_QP_OFFSET:   usize = COL_MUL_CARRY_AB_OFFSET + LIMBS_PER_PROD;
pub const COL_MUL_SUM_CARRY_OFFSET:  usize = COL_MUL_CARRY_QP_OFFSET + LIMBS_PER_PROD;

// Slack
pub const COL_SLACK_OFFSET:          usize = COL_MUL_SUM_CARRY_OFFSET + LIMBS_PER_PROD;
pub const COL_SLACK_BORROWS_OFFSET:  usize = COL_SLACK_OFFSET + LIMBS_PER_FP;

pub const NUM_DATA_COLUMNS: usize = COL_SLACK_BORROWS_OFFSET + LIMBS_PER_FP;

pub const COL_SEL_ADD: usize = NUM_DATA_COLUMNS;
pub const COL_SEL_SUB: usize = COL_SEL_ADD + 1;
pub const COL_SEL_MUL: usize = COL_SEL_SUB + 1;
pub const COL_SEL_INV: usize = COL_SEL_MUL + 1;

pub const NUM_BN254_FP_COLUMNS: usize = COL_SEL_INV + 1;

// ─── Column index helpers ────────────────────────────────────────────────

#[inline] pub fn a_limb(i: usize) -> usize { debug_assert!(i < LIMBS_PER_FP); COL_A_OFFSET + i }
#[inline] pub fn b_limb(i: usize) -> usize { debug_assert!(i < LIMBS_PER_FP); COL_B_OFFSET + i }
#[inline] pub fn c_limb(i: usize) -> usize { debug_assert!(i < LIMBS_PER_FP); COL_C_OFFSET + i }
#[inline] pub fn add_unred_sum(i: usize) -> usize { COL_ADD_UNRED_SUM_OFFSET + i }
#[inline] pub fn add_carry(i: usize) -> usize { COL_ADD_CARRIES_OFFSET + i }
#[inline] pub fn add_trial_diff(i: usize) -> usize { COL_ADD_TRIAL_DIFF_OFFSET + i }
#[inline] pub fn add_sub_borrow(i: usize) -> usize { COL_ADD_SUB_BORROWS_OFFSET + i }
#[inline] pub fn sub_unred_diff(i: usize) -> usize { COL_SUB_UNRED_DIFF_OFFSET + i }
#[inline] pub fn sub_borrow(i: usize) -> usize { COL_SUB_BORROWS_OFFSET + i }
#[inline] pub fn sub_trial_sum(i: usize) -> usize { COL_SUB_TRIAL_SUM_OFFSET + i }
#[inline] pub fn sub_add_carry(i: usize) -> usize { COL_SUB_ADD_CARRIES_OFFSET + i }
#[inline] pub fn prod_limb(i: usize) -> usize { debug_assert!(i < LIMBS_PER_PROD); COL_PROD_OFFSET + i }
#[inline] pub fn quotient_limb(i: usize) -> usize { debug_assert!(i < LIMBS_PER_FP); COL_QUOTIENT_OFFSET + i }
#[inline] pub fn qp_limb(i: usize) -> usize { debug_assert!(i < LIMBS_PER_PROD); COL_QP_OFFSET + i }
#[inline] pub fn mul_carry_ab(i: usize) -> usize { debug_assert!(i < LIMBS_PER_PROD); COL_MUL_CARRY_AB_OFFSET + i }
#[inline] pub fn mul_carry_qp(i: usize) -> usize { debug_assert!(i < LIMBS_PER_PROD); COL_MUL_CARRY_QP_OFFSET + i }
#[inline] pub fn mul_sum_carry(i: usize) -> usize { debug_assert!(i < LIMBS_PER_PROD); COL_MUL_SUM_CARRY_OFFSET + i }
#[inline] pub fn slack_limb(i: usize) -> usize { debug_assert!(i < LIMBS_PER_FP); COL_SLACK_OFFSET + i }
#[inline] pub fn slack_borrow(i: usize) -> usize { debug_assert!(i < LIMBS_PER_FP); COL_SLACK_BORROWS_OFFSET + i }

// ─── Host-side Fp arithmetic ─────────────────────────────────────────────

/// BN254 base-field element. Big-endian 4-limb representation; reduced
/// modulo `P_LIMBS` whenever it's the output of an op constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp {
    pub limbs: [u64; LIMBS_PER_FP],
}

impl Fp {
    #[inline] pub fn zero() -> Self { Fp { limbs: [0; 4] } }
    #[inline] pub fn one()  -> Self { Fp { limbs: [0, 0, 0, 1] } }
    #[inline] pub fn from_u64(v: u64) -> Self { Fp { limbs: [0, 0, 0, v] } }
    #[inline] pub fn is_zero(&self) -> bool { self.limbs.iter().all(|l| *l == 0) }
}

// Internal: compare BE slice (Greater iff a > b).
fn cmp_be(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        match a[i].cmp(&b[i]) { std::cmp::Ordering::Equal => continue, o => return o }
    }
    std::cmp::Ordering::Equal
}

#[inline]
fn mul_add_u64(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let wide = (a as u128) * (b as u128) + (c as u128) + (carry as u128);
    (wide as u64, (wide >> 64) as u64)
}

/// Big-endian 4×4 → 8-limb multiplication.
fn mul_4_be(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
    let a_le: [u64; 4] = [a[3], a[2], a[1], a[0]];
    let b_le: [u64; 4] = [b[3], b[2], b[1], b[0]];
    let mut p_le = [0u64; 8];
    for i in 0..4 {
        let mut carry: u64 = 0;
        for j in 0..4 {
            let (lo, hi) = mul_add_u64(a_le[i], b_le[j], p_le[i + j], carry);
            p_le[i + j] = lo;
            carry = hi;
        }
        p_le[i + 4] = carry;
    }
    let mut out = [0u64; 8];
    for i in 0..8 { out[i] = p_le[7 - i]; }
    out
}

/// `(a - b) mod 2^256` (BE), returns (diff, final_borrow).
fn sub4(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], u64) {
    let mut d = [0u64; 4];
    let mut borrow: u64 = 0;
    for i in (0..4).rev() {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        d[i] = d2;
        borrow = (b1 as u64) + (b2 as u64);
    }
    (d, borrow)
}

/// Shift the 4-limb BE modulus left by `s` bits into an 8-limb BE buffer.
fn shift_left_4_into_8(m: &[u64; 4], s: usize) -> [u64; 8] {
    let mut m_le_ext = [0u64; 8];
    for k in 0..4 { m_le_ext[k] = m[3 - k]; }
    let limb_shift = s / 64;
    let bit_shift = s % 64;
    let mut out_le = [0u64; 8];
    for k in 0..8 {
        let src_lo = if k >= limb_shift && (k - limb_shift) < 8 { m_le_ext[k - limb_shift] } else { 0 };
        let src_hi = if bit_shift > 0 && k >= limb_shift + 1 && (k - limb_shift - 1) < 8 { m_le_ext[k - limb_shift - 1] } else { 0 };
        let lo_part = src_lo.wrapping_shl(bit_shift as u32);
        let hi_part = if bit_shift == 0 { 0 } else { src_hi.wrapping_shr((64 - bit_shift) as u32) };
        out_le[k] = lo_part | hi_part;
    }
    let mut out_be = [0u64; 8];
    for k in 0..8 { out_be[k] = out_le[7 - k]; }
    out_be
}

/// Long-division of 8-limb BE numerator by 4-limb BE modulus.
/// Returns `(q[4], r[4])` (BE), both < 2^256.
fn divmod_8_by_4(num: &[u64; 8], modulus: &[u64; 4]) -> ([u64; 4], [u64; 4]) {
    let mut rem: [u64; 8] = *num;
    let mut quot_bits = [0u8; 256];
    for s in (0..256).rev() {
        let shifted = shift_left_4_into_8(modulus, s);
        if cmp_be(&rem, &shifted) != std::cmp::Ordering::Less {
            let mut borrow: u64 = 0;
            for i in (0..8).rev() {
                let (d1, b1) = rem[i].overflowing_sub(shifted[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                rem[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
            }
            quot_bits[255 - s] = 1;
        }
    }
    let mut q = [0u64; 4];
    for i in 0..256 {
        if quot_bits[i] == 1 {
            let limb_idx = i / 64;
            let bit_in = 63 - (i % 64);
            q[limb_idx] |= 1u64 << bit_in;
        }
    }
    let mut r = [0u64; 4];
    r.copy_from_slice(&rem[4..8]);
    debug_assert!(rem[..4].iter().all(|x| *x == 0));
    (q, r)
}

// ─── Witness structs ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AddWitness {
    pub sum: [u64; 4],
    pub add_carries: [u64; 5],
    pub trial_diff: [u64; 4],
    pub sub_borrows: [u64; 5],
    pub reduce_flag: u64,
    pub result: [u64; 4],
}

#[derive(Debug, Clone)]
pub struct SubWitness {
    pub diff: [u64; 4],
    pub sub_borrows: [u64; 5],
    pub trial_sum: [u64; 4],
    pub add_carries: [u64; 5],
    pub add_back_flag: u64,
    pub result: [u64; 4],
}

#[derive(Debug, Clone)]
pub struct MulWitness {
    pub product: [u64; 8],
    pub q: [u64; 4],
    pub r: [u64; 4],
    pub q_times_p: [u64; 8],
}

impl Fp {
    pub fn add_witness(&self, other: &Self) -> AddWitness {
        let a = self.limbs;
        let b = other.limbs;
        let mut sum = [0u64; 4];
        let mut add_carries = [0u64; 5];
        let mut carry: u64 = 0;
        for i in (0..4).rev() {
            let (s1, c1) = a[i].overflowing_add(b[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            sum[i] = s2;
            let cout = (c1 as u64) + (c2 as u64);
            add_carries[3 - i] = cout;
            carry = cout;
        }
        add_carries[4] = carry;
        let overflow = carry;
        let mut trial_diff = [0u64; 4];
        let mut sub_borrows = [0u64; 5];
        let mut borrow: u64 = 0;
        for i in (0..4).rev() {
            let (d1, b1) = sum[i].overflowing_sub(P_LIMBS[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            trial_diff[i] = d2;
            let bout = (b1 as u64) + (b2 as u64);
            sub_borrows[3 - i] = bout;
            borrow = bout;
        }
        sub_borrows[4] = borrow;
        let reduce_flag = if overflow == 1 || borrow == 0 { 1 } else { 0 };
        let result = if reduce_flag == 1 { trial_diff } else { sum };
        AddWitness { sum, add_carries, trial_diff, sub_borrows, reduce_flag, result }
    }

    pub fn sub_witness(&self, other: &Self) -> SubWitness {
        let a = self.limbs;
        let b = other.limbs;
        let mut diff = [0u64; 4];
        let mut sub_borrows = [0u64; 5];
        let mut borrow: u64 = 0;
        for i in (0..4).rev() {
            let (d1, b1) = a[i].overflowing_sub(b[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            diff[i] = d2;
            let bout = (b1 as u64) + (b2 as u64);
            sub_borrows[3 - i] = bout;
            borrow = bout;
        }
        sub_borrows[4] = borrow;
        let underflow = borrow;
        let mut trial_sum = [0u64; 4];
        let mut add_carries = [0u64; 5];
        let mut carry: u64 = 0;
        for i in (0..4).rev() {
            let (s1, c1) = diff[i].overflowing_add(P_LIMBS[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            trial_sum[i] = s2;
            let cout = (c1 as u64) + (c2 as u64);
            add_carries[3 - i] = cout;
            carry = cout;
        }
        add_carries[4] = carry;
        let add_back_flag = underflow;
        let result = if add_back_flag == 1 { trial_sum } else { diff };
        SubWitness { diff, sub_borrows, trial_sum, add_carries, add_back_flag, result }
    }

    pub fn mul_witness(&self, other: &Self) -> MulWitness {
        let product = mul_4_be(&self.limbs, &other.limbs);
        let (q, r) = divmod_8_by_4(&product, &P_LIMBS);
        let q_times_p = mul_4_be(&q, &P_LIMBS);
        MulWitness { product, q, r, q_times_p }
    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self { Fp { limbs: self.add_witness(other).result } }
    #[inline]
    pub fn sub(&self, other: &Self) -> Self { Fp { limbs: self.sub_witness(other).result } }
    #[inline]
    pub fn mul(&self, other: &Self) -> Self { Fp { limbs: self.mul_witness(other).r } }
    #[inline]
    pub fn square(&self) -> Self { self.mul(self) }

    /// Multiplicative inverse via Fermat's little theorem: `a^{p-2} mod p`.
    pub fn invert(&self) -> Option<Self> {
        if self.is_zero() { return None; }
        let two = Fp { limbs: [0, 0, 0, 2] };
        let (p_minus_2, _b) = sub4(&P_LIMBS, &two.limbs);
        Some(self.pow(&p_minus_2))
    }

    pub fn pow(&self, exp: &[u64; 4]) -> Self {
        let mut result = Fp::one();
        let mut started = false;
        for i in 0..4 {
            let limb = exp[i];
            for bit in (0..64).rev() {
                if started { result = result.square(); }
                if ((limb >> bit) & 1) == 1 {
                    if !started { result = *self; started = true; }
                    else        { result = result.mul(self); }
                }
            }
        }
        if !started { return Fp::one(); }
        result
    }
}

// ─── Op enum ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpOp {
    Add { a: Fp, b: Fp },
    Sub { a: Fp, b: Fp },
    Mul { a: Fp, b: Fp },
    Inv { a: Fp },
}

// ─── Small scalar helpers ────────────────────────────────────────────────

/// `2^64` as a Scalar (matches `nonnative_fp_air::two_pow_64`).
pub fn two_pow_64(curve: CurveType) -> Scalar {
    use bls48581::bls48581::big;
    match curve {
        CurveType::Bls48581 => {
            let mut buf = [0u8; big::MODBYTES];
            buf[big::MODBYTES - 9] = 1;
            Scalar::Bls48581(big::BIG::frombytes(&buf))
        }
        CurveType::Bls12381 => {
            let mut scalar = blst::blst_scalar::default();
            scalar.b[8] = 1;
            let mut fr = blst::blst_fr::default();
            unsafe { blst::blst_fr_from_scalar(&mut fr, &scalar); }
            Scalar::Bls12381(fr)
        }
    }
}

#[inline]
fn sc(v: u64, curve: CurveType) -> Scalar { Scalar::from_u64(v, curve) }

fn sc128(v: u128, curve: CurveType) -> Scalar {
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    let lo_s = Scalar::from_u64(lo, curve);
    if hi == 0 { return lo_s; }
    let hi_s = Scalar::from_u64(hi, curve);
    let two64 = two_pow_64(curve);
    lo_s.add(&hi_s.mul(&two64))
}

#[inline]
fn le_to_be_idx(lsb_first_index: usize, n_limbs: usize) -> usize {
    n_limbs - 1 - lsb_first_index
}

pub fn alloc_trace(num_rows: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..NUM_BN254_FP_COLUMNS).map(|_| vec![zero.clone(); num_rows]).collect()
}

// ─── Schoolbook for the AIR's exact 4×4 → 8 chain ────────────────────────

fn schoolbook_4x4_le(a_be: &[u64; 4], b_be: &[u64; 4]) -> ([u64; 8], [u128; 8]) {
    let a_le: [u64; 4] = [a_be[3], a_be[2], a_be[1], a_be[0]];
    let b_le: [u64; 4] = [b_be[3], b_be[2], b_be[1], b_be[0]];
    let mut prod = [0u64; 8];
    let mut carry_out = [0u128; 8];
    let mut carry_in: u128 = 0;
    for k in 0..8 {
        let mut lo: u128 = 0;
        let mut hi: u64 = 0;
        for i in 0..4 {
            let j = k as isize - i as isize;
            if j < 0 || j >= 4 { continue; }
            let p = (a_le[i] as u128) * (b_le[j as usize] as u128);
            let (new_lo, ov) = lo.overflowing_add(p);
            lo = new_lo;
            if ov { hi += 1; }
        }
        let (new_lo, ov) = lo.overflowing_add(carry_in);
        lo = new_lo;
        if ov { hi += 1; }
        prod[k] = lo as u64;
        let cout: u128 = ((hi as u128) << 64) + (lo >> 64);
        carry_out[k] = cout;
        carry_in = cout;
    }
    debug_assert_eq!(carry_in, 0, "4x4 schoolbook carry overflowed at top");
    (prod, carry_out)
}

fn sum_carries_le(qp_be: &[u64; 8], r_be: &[u64; 4]) -> ([u64; 8], [u64; 8]) {
    let mut qp_le = [0u64; 8];
    for k in 0..8 { qp_le[k] = qp_be[7 - k]; }
    let mut r_le = [0u64; 8];
    for k in 0..4 { r_le[k] = r_be[3 - k]; }
    let mut sum = [0u64; 8];
    let mut carries = [0u64; 8];
    let mut carry: u64 = 0;
    for k in 0..8 {
        let t = (qp_le[k] as u128) + (r_le[k] as u128) + (carry as u128);
        sum[k] = t as u64;
        carry = (t >> 64) as u64;
        carries[k] = carry;
    }
    (sum, carries)
}

fn slack_witness(r_be: &[u64; 4]) -> ([u64; 4], [u64; 4]) {
    let mut pm1 = P_LIMBS;
    pm1[3] = pm1[3].wrapping_sub(1);
    let mut diff = [0u64; 4];
    let mut borrows = [0u64; 4];
    let mut borrow: u64 = 0;
    for i in (0..4).rev() {
        let (d1, b1) = pm1[i].overflowing_sub(r_be[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        diff[i] = d2;
        let bout = (b1 as u64) + (b2 as u64);
        borrows[3 - i] = bout;
        borrow = bout;
    }
    (diff, borrows)
}

// ─── Row population ──────────────────────────────────────────────────────

pub fn populate_row(
    columns: &mut [Vec<Scalar>],
    row: usize,
    op: &FpOp,
    curve: CurveType,
) {
    assert_eq!(columns.len(), NUM_BN254_FP_COLUMNS, "columns shape");
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);

    for col in columns.iter_mut() { col[row] = zero.clone(); }

    let write_be4 = |columns: &mut [Vec<Scalar>], offset: usize, limbs: &[u64; 4], row: usize| {
        for i in 0..4 { columns[offset + i][row] = sc(limbs[i], curve); }
    };
    let write_be8 = |columns: &mut [Vec<Scalar>], offset: usize, limbs: &[u64; 8], row: usize| {
        for i in 0..8 { columns[offset + i][row] = sc(limbs[i], curve); }
    };
    let write_le_carries8 = |columns: &mut [Vec<Scalar>], offset: usize, c: &[u64; 8], row: usize| {
        for k in 0..8 { columns[offset + k][row] = sc(c[k], curve); }
    };
    let write_le_carries8_u128 = |columns: &mut [Vec<Scalar>], offset: usize, c: &[u128; 8], row: usize| {
        for k in 0..8 { columns[offset + k][row] = sc128(c[k], curve); }
    };

    match op {
        FpOp::Add { a, b } => {
            let w = a.add_witness(b);
            write_be4(columns, COL_A_OFFSET, &a.limbs, row);
            write_be4(columns, COL_B_OFFSET, &b.limbs, row);
            write_be4(columns, COL_C_OFFSET, &w.result, row);
            write_be4(columns, COL_ADD_UNRED_SUM_OFFSET, &w.sum, row);
            for i in 0..4 { columns[COL_ADD_CARRIES_OFFSET + i][row] = sc(w.add_carries[i], curve); }
            columns[COL_ADD_REDUCE_FLAG][row] = sc(w.reduce_flag, curve);
            write_be4(columns, COL_ADD_TRIAL_DIFF_OFFSET, &w.trial_diff, row);
            for i in 0..4 { columns[COL_ADD_SUB_BORROWS_OFFSET + i][row] = sc(w.sub_borrows[i], curve); }
            let (slack, sb) = slack_witness(&w.result);
            write_be4(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..4 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }
            columns[COL_SEL_ADD][row] = one.clone();
        }
        FpOp::Sub { a, b } => {
            let w = a.sub_witness(b);
            write_be4(columns, COL_A_OFFSET, &a.limbs, row);
            write_be4(columns, COL_B_OFFSET, &b.limbs, row);
            write_be4(columns, COL_C_OFFSET, &w.result, row);
            write_be4(columns, COL_SUB_UNRED_DIFF_OFFSET, &w.diff, row);
            for i in 0..4 { columns[COL_SUB_BORROWS_OFFSET + i][row] = sc(w.sub_borrows[i], curve); }
            columns[COL_SUB_ADD_BACK_FLAG][row] = sc(w.add_back_flag, curve);
            write_be4(columns, COL_SUB_TRIAL_SUM_OFFSET, &w.trial_sum, row);
            for i in 0..4 { columns[COL_SUB_ADD_CARRIES_OFFSET + i][row] = sc(w.add_carries[i], curve); }
            let (slack, sb) = slack_witness(&w.result);
            write_be4(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..4 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }
            columns[COL_SEL_SUB][row] = one.clone();
        }
        FpOp::Mul { a, b } => {
            let w = a.mul_witness(b);
            write_be4(columns, COL_A_OFFSET, &a.limbs, row);
            write_be4(columns, COL_B_OFFSET, &b.limbs, row);
            write_be4(columns, COL_C_OFFSET, &w.r, row);
            let (prod_le, carry_ab_le) = schoolbook_4x4_le(&a.limbs, &b.limbs);
            let mut prod_be = [0u64; 8];
            for k in 0..8 { prod_be[k] = prod_le[7 - k]; }
            write_be8(columns, COL_PROD_OFFSET, &prod_be, row);
            write_le_carries8_u128(columns, COL_MUL_CARRY_AB_OFFSET, &carry_ab_le, row);
            write_be4(columns, COL_QUOTIENT_OFFSET, &w.q, row);
            let (qp_le, carry_qp_le) = schoolbook_4x4_le(&w.q, &P_LIMBS);
            let mut qp_be = [0u64; 8];
            for k in 0..8 { qp_be[k] = qp_le[7 - k]; }
            debug_assert_eq!(qp_be, w.q_times_p, "qp schoolbook mismatch");
            write_be8(columns, COL_QP_OFFSET, &qp_be, row);
            write_le_carries8_u128(columns, COL_MUL_CARRY_QP_OFFSET, &carry_qp_le, row);
            let (sum_check, sum_carries_vals) = sum_carries_le(&qp_be, &w.r);
            debug_assert_eq!(sum_check, prod_le, "qp + r reconstruction mismatch");
            write_le_carries8(columns, COL_MUL_SUM_CARRY_OFFSET, &sum_carries_vals, row);
            let (slack, sb) = slack_witness(&w.r);
            write_be4(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..4 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }
            columns[COL_SEL_MUL][row] = one.clone();
        }
        FpOp::Inv { a } => {
            let inv = a.invert().expect("Inv on zero Fp");
            let w = a.mul_witness(&inv);
            let one_limbs: [u64; 4] = [0, 0, 0, 1];
            debug_assert_eq!(w.r, one_limbs, "a · a^-1 must reduce to 1");
            write_be4(columns, COL_A_OFFSET, &a.limbs, row);
            write_be4(columns, COL_B_OFFSET, &inv.limbs, row);
            write_be4(columns, COL_C_OFFSET, &one_limbs, row);
            let (prod_le, carry_ab_le) = schoolbook_4x4_le(&a.limbs, &inv.limbs);
            let mut prod_be = [0u64; 8];
            for k in 0..8 { prod_be[k] = prod_le[7 - k]; }
            write_be8(columns, COL_PROD_OFFSET, &prod_be, row);
            write_le_carries8_u128(columns, COL_MUL_CARRY_AB_OFFSET, &carry_ab_le, row);
            write_be4(columns, COL_QUOTIENT_OFFSET, &w.q, row);
            let (qp_le, carry_qp_le) = schoolbook_4x4_le(&w.q, &P_LIMBS);
            let mut qp_be = [0u64; 8];
            for k in 0..8 { qp_be[k] = qp_le[7 - k]; }
            debug_assert_eq!(qp_be, w.q_times_p);
            write_be8(columns, COL_QP_OFFSET, &qp_be, row);
            write_le_carries8_u128(columns, COL_MUL_CARRY_QP_OFFSET, &carry_qp_le, row);
            let (sum_check, sum_carries_vals) = sum_carries_le(&qp_be, &one_limbs);
            debug_assert_eq!(sum_check, prod_le);
            write_le_carries8(columns, COL_MUL_SUM_CARRY_OFFSET, &sum_carries_vals, row);
            let (slack, sb) = slack_witness(&one_limbs);
            write_be4(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..4 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }
            columns[COL_SEL_INV][row] = one.clone();
        }
    }
}

pub fn populate_trace(ops: &[FpOp], curve: CurveType, num_rows: Option<usize>) -> Vec<Vec<Scalar>> {
    let rows = num_rows.unwrap_or(ops.len()).max(ops.len());
    let mut columns = alloc_trace(rows, curve);
    for (row, op) in ops.iter().enumerate() { populate_row(&mut columns, row, op, curve); }
    columns
}

// ─── Constraint evaluation ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ConstraintEval {
    pub label: String,
    pub values: Vec<Scalar>,
}

pub fn evaluate_constraints(columns: &[&Vec<Scalar>], beta: &Scalar) -> Vec<ConstraintEval> {
    assert_eq!(columns.len(), NUM_BN254_FP_COLUMNS, "expected {} columns", NUM_BN254_FP_COLUMNS);
    let num_rows = columns[0].len();
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);
    let two64 = two_pow_64(curve);

    let p_le: [Scalar; 4] = [
        sc(P_LIMBS[3], curve),
        sc(P_LIMBS[2], curve),
        sc(P_LIMBS[1], curve),
        sc(P_LIMBS[0], curve),
    ];

    let a_at = |i: usize, row: usize| columns[a_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let b_at = |i: usize, row: usize| columns[b_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let c_at = |i: usize, row: usize| columns[c_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let add_unred_at = |i: usize, row: usize| columns[add_unred_sum(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let add_trial_at = |i: usize, row: usize| columns[add_trial_diff(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let sub_diff_at  = |i: usize, row: usize| columns[sub_unred_diff(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let sub_trial_at = |i: usize, row: usize| columns[sub_trial_sum(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let quot_at      = |i: usize, row: usize| columns[quotient_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let prod_at      = |k: usize, row: usize| columns[prod_limb(le_to_be_idx(k, LIMBS_PER_PROD))][row].clone();
    let qp_at        = |k: usize, row: usize| columns[qp_limb(le_to_be_idx(k, LIMBS_PER_PROD))][row].clone();

    let sel_add_col = columns[COL_SEL_ADD];
    let sel_sub_col = columns[COL_SEL_SUB];
    let sel_mul_col = columns[COL_SEL_MUL];
    let sel_inv_col = columns[COL_SEL_INV];
    let sel_mul_or_inv: Vec<Scalar> =
        (0..num_rows).map(|row| sel_mul_col[row].add(&sel_inv_col[row])).collect();

    let mut result: Vec<ConstraintEval> = Vec::new();

    // CAT 1: Add limb chain.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let c_in = if i == 0 { zero.clone() } else { columns[COL_ADD_CARRIES_OFFSET + i - 1][row].clone() };
                let c_out = columns[COL_ADD_CARRIES_OFFSET + i][row].clone();
                let lhs = a_at(i, row).add(&b_at(i, row)).add(&c_in);
                let rhs = add_unred_at(i, row).add(&c_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_add_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "add_limb_chain".into(), values: acc });
    }

    // CAT 2: Add trial diff chain.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let b_in = if i == 0 { zero.clone() } else { columns[COL_ADD_SUB_BORROWS_OFFSET + i - 1][row].clone() };
                let b_out = columns[COL_ADD_SUB_BORROWS_OFFSET + i][row].clone();
                let lhs = add_unred_at(i, row).sub(&p_le[i]).sub(&b_in);
                let rhs = add_trial_at(i, row).sub(&b_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_add_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "add_trial_diff".into(), values: acc });
    }

    // CAT 3: Add reduce flag = overflow OR (NOT top-borrow).
    {
        let top = LIMBS_PER_FP - 1;
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let x = columns[COL_ADD_CARRIES_OFFSET + top][row].clone();
            let y = one.sub(&columns[COL_ADD_SUB_BORROWS_OFFSET + top][row]);
            let xy = x.mul(&y);
            let or_xy = x.add(&y).sub(&xy);
            let flag = columns[COL_ADD_REDUCE_FLAG][row].clone();
            let body = flag.sub(&or_xy);
            acc[row] = sel_add_col[row].mul(&body);
        }
        result.push(ConstraintEval { label: "add_reduce_flag_def".into(), values: acc });
    }

    // CAT 4: Add result selection.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let flag = columns[COL_ADD_REDUCE_FLAG][row].clone();
                let one_m_flag = one.sub(&flag);
                let rhs = one_m_flag.mul(&add_unred_at(i, row)).add(&flag.mul(&add_trial_at(i, row)));
                let body = c_at(i, row).sub(&rhs);
                let gated = sel_add_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "add_result_select".into(), values: acc });
    }

    // CAT 5: Sub limb chain.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let b_in = if i == 0 { zero.clone() } else { columns[COL_SUB_BORROWS_OFFSET + i - 1][row].clone() };
                let b_out = columns[COL_SUB_BORROWS_OFFSET + i][row].clone();
                let lhs = a_at(i, row).sub(&b_at(i, row)).sub(&b_in);
                let rhs = sub_diff_at(i, row).sub(&b_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_sub_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "sub_limb_chain".into(), values: acc });
    }

    // CAT 6: Sub trial add-back.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let c_in = if i == 0 { zero.clone() } else { columns[COL_SUB_ADD_CARRIES_OFFSET + i - 1][row].clone() };
                let c_out = columns[COL_SUB_ADD_CARRIES_OFFSET + i][row].clone();
                let lhs = sub_diff_at(i, row).add(&p_le[i]).add(&c_in);
                let rhs = sub_trial_at(i, row).add(&c_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_sub_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "sub_trial_sum".into(), values: acc });
    }

    // CAT 7: Sub add-back flag = top borrow.
    {
        let top = LIMBS_PER_FP - 1;
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let flag = columns[COL_SUB_ADD_BACK_FLAG][row].clone();
            let top_borrow = columns[COL_SUB_BORROWS_OFFSET + top][row].clone();
            let body = flag.sub(&top_borrow);
            acc[row] = sel_sub_col[row].mul(&body);
        }
        result.push(ConstraintEval { label: "sub_add_back_flag_def".into(), values: acc });
    }

    // CAT 8: Sub result selection.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let flag = columns[COL_SUB_ADD_BACK_FLAG][row].clone();
                let one_m_flag = one.sub(&flag);
                let rhs = one_m_flag.mul(&sub_diff_at(i, row)).add(&flag.mul(&sub_trial_at(i, row)));
                let body = c_at(i, row).sub(&rhs);
                let gated = sel_sub_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "sub_result_select".into(), values: acc });
    }

    // CAT 9: Mul schoolbook a·b.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for k in 0..LIMBS_PER_PROD {
            for row in 0..num_rows {
                let mut col_sum = zero.clone();
                for i in 0..LIMBS_PER_FP {
                    let j = k as isize - i as isize;
                    if j < 0 || j >= LIMBS_PER_FP as isize { continue; }
                    col_sum = col_sum.add(&a_at(i, row).mul(&b_at(j as usize, row)));
                }
                let c_in = if k == 0 { zero.clone() } else { columns[COL_MUL_CARRY_AB_OFFSET + k - 1][row].clone() };
                let c_out = columns[COL_MUL_CARRY_AB_OFFSET + k][row].clone();
                let lhs = col_sum.add(&c_in);
                let rhs = prod_at(k, row).add(&c_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_mul_or_inv[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "mul_schoolbook_ab".into(), values: acc });
    }

    // CAT 10: Mul a·b top carry zero.
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let top = columns[COL_MUL_CARRY_AB_OFFSET + LIMBS_PER_PROD - 1][row].clone();
            acc[row] = sel_mul_or_inv[row].mul(&top);
        }
        result.push(ConstraintEval { label: "mul_ab_top_carry_zero".into(), values: acc });
    }

    // CAT 11: Mul q·p schoolbook.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for k in 0..LIMBS_PER_PROD {
            for row in 0..num_rows {
                let mut col_sum = zero.clone();
                for i in 0..LIMBS_PER_FP {
                    let j = k as isize - i as isize;
                    if j < 0 || j >= LIMBS_PER_FP as isize { continue; }
                    col_sum = col_sum.add(&quot_at(i, row).mul(&p_le[j as usize]));
                }
                let c_in = if k == 0 { zero.clone() } else { columns[COL_MUL_CARRY_QP_OFFSET + k - 1][row].clone() };
                let c_out = columns[COL_MUL_CARRY_QP_OFFSET + k][row].clone();
                let lhs = col_sum.add(&c_in);
                let rhs = qp_at(k, row).add(&c_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_mul_or_inv[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "mul_schoolbook_qp".into(), values: acc });
    }

    // CAT 12: Mul q·p top carry zero.
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let top = columns[COL_MUL_CARRY_QP_OFFSET + LIMBS_PER_PROD - 1][row].clone();
            acc[row] = sel_mul_or_inv[row].mul(&top);
        }
        result.push(ConstraintEval { label: "mul_qp_top_carry_zero".into(), values: acc });
    }

    // CAT 13: qp + r_padded = prod chain.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for k in 0..LIMBS_PER_PROD {
            for row in 0..num_rows {
                let r_term = if k < LIMBS_PER_FP { c_at(k, row) } else { zero.clone() };
                let c_in = if k == 0 { zero.clone() } else { columns[COL_MUL_SUM_CARRY_OFFSET + k - 1][row].clone() };
                let c_out = columns[COL_MUL_SUM_CARRY_OFFSET + k][row].clone();
                let lhs = qp_at(k, row).add(&r_term).add(&c_in);
                let rhs = prod_at(k, row).add(&c_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_mul_or_inv[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "mul_sum_equals_prod".into(), values: acc });
    }

    // CAT 14: top sum-carry zero.
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let top = columns[COL_MUL_SUM_CARRY_OFFSET + LIMBS_PER_PROD - 1][row].clone();
            acc[row] = sel_mul_or_inv[row].mul(&top);
        }
        result.push(ConstraintEval { label: "mul_sum_top_carry_zero".into(), values: acc });
    }

    // CAT 15: Slack chain (p - 1 - c) with no final borrow.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        let sel_any = |row: usize| -> Scalar {
            sel_add_col[row].add(&sel_sub_col[row]).add(&sel_mul_or_inv[row])
        };
        for i in 0..LIMBS_PER_FP {
            let p_minus_1_at_i = if i == 0 { p_le[0].sub(&one) } else { p_le[i].clone() };
            for row in 0..num_rows {
                let b_in = if i == 0 { zero.clone() } else { columns[COL_SLACK_BORROWS_OFFSET + i - 1][row].clone() };
                let b_out = columns[COL_SLACK_BORROWS_OFFSET + i][row].clone();
                let slack_i = columns[slack_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
                let lhs = p_minus_1_at_i.sub(&c_at(i, row)).sub(&b_in);
                let rhs = slack_i.sub(&b_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_any(row).mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "slack_chain".into(), values: acc });
    }

    // CAT 16: top slack borrow = 0.
    {
        let top = LIMBS_PER_FP - 1;
        let mut acc = vec![zero.clone(); num_rows];
        let sel_any = |row: usize| -> Scalar {
            sel_add_col[row].add(&sel_sub_col[row]).add(&sel_mul_or_inv[row])
        };
        for row in 0..num_rows {
            let top_borrow = columns[COL_SLACK_BORROWS_OFFSET + top][row].clone();
            acc[row] = sel_any(row).mul(&top_borrow);
        }
        result.push(ConstraintEval { label: "slack_top_borrow_zero".into(), values: acc });
    }

    // CAT 17: binary flags / carries / selectors.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        let mut binary_cols: Vec<usize> = Vec::new();
        for i in 0..LIMBS_PER_FP { binary_cols.push(COL_ADD_CARRIES_OFFSET + i); }
        for i in 0..LIMBS_PER_FP { binary_cols.push(COL_ADD_SUB_BORROWS_OFFSET + i); }
        binary_cols.push(COL_ADD_REDUCE_FLAG);
        for i in 0..LIMBS_PER_FP { binary_cols.push(COL_SUB_BORROWS_OFFSET + i); }
        for i in 0..LIMBS_PER_FP { binary_cols.push(COL_SUB_ADD_CARRIES_OFFSET + i); }
        binary_cols.push(COL_SUB_ADD_BACK_FLAG);
        for k in 0..LIMBS_PER_PROD { binary_cols.push(COL_MUL_SUM_CARRY_OFFSET + k); }
        for i in 0..LIMBS_PER_FP { binary_cols.push(COL_SLACK_BORROWS_OFFSET + i); }
        binary_cols.push(COL_SEL_ADD);
        binary_cols.push(COL_SEL_SUB);
        binary_cols.push(COL_SEL_MUL);
        binary_cols.push(COL_SEL_INV);
        for col_idx in binary_cols {
            for row in 0..num_rows {
                let v = columns[col_idx][row].clone();
                let body = v.mul(&v.sub(&one));
                acc[row] = acc[row].add(&beta_pow.mul(&body));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "binary_flags".into(), values: acc });
    }

    // CAT 18: selector sum ∈ {0, 1}.
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let s = sel_add_col[row].add(&sel_sub_col[row]).add(&sel_mul_col[row]).add(&sel_inv_col[row]);
            acc[row] = s.mul(&s.sub(&one));
        }
        result.push(ConstraintEval { label: "selector_sum_01".into(), values: acc });
    }

    // CAT 19: Inv c = 1.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for be in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let c_be = columns[COL_C_OFFSET + be][row].clone();
                let target = if be == LIMBS_PER_FP - 1 { c_be.sub(&one) } else { c_be };
                let body = sel_inv_col[row].mul(&target);
                acc[row] = acc[row].add(&beta_pow.mul(&body));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "inv_result_is_one".into(), values: acc });
    }

    result
}

pub const NUM_CONSTRAINT_CATEGORIES: usize = 19;

// ─── Lookup declarations ─────────────────────────────────────────────────

pub fn lookup_declarations() -> Vec<LookupDeclaration> {
    let mut decls = Vec::new();
    let range_64 = |col: usize, label: &str| LookupDeclaration {
        label: label.into(), column_index: col, max_bits: 64, selector_column: None,
    };
    let range_72 = |col: usize, label: &str| LookupDeclaration {
        label: label.into(), column_index: col, max_bits: 72, selector_column: None,
    };
    let binary = |col: usize, label: &str| LookupDeclaration {
        label: label.into(), column_index: col, max_bits: 1, selector_column: None,
    };

    for i in 0..LIMBS_PER_FP {
        decls.push(range_64(a_limb(i),         &format!("a_limb_{}_range", i)));
        decls.push(range_64(b_limb(i),         &format!("b_limb_{}_range", i)));
        decls.push(range_64(c_limb(i),         &format!("c_limb_{}_range", i)));
        decls.push(range_64(add_unred_sum(i),  &format!("add_unred_sum_{}_range", i)));
        decls.push(range_64(add_trial_diff(i), &format!("add_trial_diff_{}_range", i)));
        decls.push(range_64(sub_unred_diff(i), &format!("sub_unred_diff_{}_range", i)));
        decls.push(range_64(sub_trial_sum(i),  &format!("sub_trial_sum_{}_range", i)));
        decls.push(range_64(quotient_limb(i),  &format!("quotient_limb_{}_range", i)));
        decls.push(range_64(slack_limb(i),     &format!("slack_limb_{}_range", i)));
    }
    for k in 0..LIMBS_PER_PROD {
        decls.push(range_64(prod_limb(k),    &format!("prod_limb_{}_range", k)));
        decls.push(range_64(qp_limb(k),      &format!("qp_limb_{}_range", k)));
        decls.push(range_72(mul_carry_ab(k), &format!("mul_carry_ab_{}_range", k)));
        decls.push(range_72(mul_carry_qp(k), &format!("mul_carry_qp_{}_range", k)));
    }
    for i in 0..LIMBS_PER_FP {
        decls.push(binary(add_carry(i),      &format!("add_carry_{}_bin", i)));
        decls.push(binary(add_sub_borrow(i), &format!("add_sub_borrow_{}_bin", i)));
        decls.push(binary(sub_borrow(i),     &format!("sub_borrow_{}_bin", i)));
        decls.push(binary(sub_add_carry(i),  &format!("sub_add_carry_{}_bin", i)));
        decls.push(binary(slack_borrow(i),   &format!("slack_borrow_{}_bin", i)));
    }
    decls.push(binary(COL_ADD_REDUCE_FLAG,   "add_reduce_flag_bin"));
    decls.push(binary(COL_SUB_ADD_BACK_FLAG, "sub_add_back_flag_bin"));
    for k in 0..LIMBS_PER_PROD {
        decls.push(binary(mul_sum_carry(k), &format!("mul_sum_carry_{}_bin", k)));
    }
    decls.push(binary(COL_SEL_ADD, "sel_add_bin"));
    decls.push(binary(COL_SEL_SUB, "sel_sub_bin"));
    decls.push(binary(COL_SEL_MUL, "sel_mul_bin"));
    decls.push(binary(COL_SEL_INV, "sel_inv_bin"));

    decls
}

// ─── Cross-AIR LogUp linkage (Rx/Ry binding, limb level) ─────────────────

/// Build a cross-AIR LogUp descriptor that binds the **Mul output**
/// (the 4 BE u64 limbs in `c_limbs` on this Fp-arithmetic AIR) to the
/// 4 BE u64 limbs of the curve-op output `r.x` (or `r.y`) in
/// [`crate::bn254_curve_ops_air`].
///
/// `which_coord` selects which curve-op coordinate to bind: `0` =
/// `r.x` limbs (cols
/// `COL_R_OFFSET .. COL_R_OFFSET + LIMBS_PER_BN254_FP`), `1` = `r.y`
/// limbs (the next 4 cols).
///
/// # Soundness scope (deferred)
///
/// This descriptor is a **limb-level shape stub**: it asserts that, when
/// the curve-op AIR's `R = P + Q` row is paired with an Fp-Mul row whose
/// `c_limbs` match a particular coordinate of `R`, the multiset binding
/// holds under the LogUp γ challenge. Full algebraic closure of the
/// curve law (i.e. that `c_limbs` is literally `λ² - x_p - x_q` over
/// `Fp`, for affine add) requires several Mul / Sub / Inv rows per
/// curve-op row plus inter-row sequencing — that's the deferred
/// per-curve-op-decomposition phase.
///
/// The 32-byte Rx / Ry binding to
/// [`crate::bn254_precompile_air`] (the EIP-196/197 precompile witness
/// rows) requires byte-decomposition columns on either side; see the
/// module doc for the deferred plan.
pub fn make_bn254_curve_ops_to_fp_air_linkage_descriptor(
    curve_ops_layer_index: usize,
    fp_layer_index: usize,
    which_coord: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert!(which_coord < 2, "which_coord must be 0 (Rx) or 1 (Ry)");
    let r_offset = super::COL_R_OFFSET + which_coord * super::LIMBS_PER_BN254_FP;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("bn254_curve_ops_r{}_to_fp_mul_v1", if which_coord == 0 { "x" } else { "y" }),
        a_layer_index: curve_ops_layer_index,
        a_columns: (0..LIMBS_PER_FP).map(|i| r_offset + i).collect(),
        a_selector_column: Some(super::COL_IS_REAL),
        b_layer_index: fp_layer_index,
        b_columns: (0..LIMBS_PER_FP).map(|i| c_limb(i)).collect(),
        b_selector_column: Some(COL_SEL_MUL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn beta_challenge() -> Scalar { Scalar::from_u64(31337, CurveType::Bls48581) }
    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> { columns.iter().collect() }
    fn assert_all_zero(evals: &[ConstraintEval]) {
        for ce in evals {
            for (row, v) in ce.values.iter().enumerate() {
                assert!(v.is_zero(), "constraint `{}` fired at row {} (valid witness)", ce.label, row);
            }
        }
    }
    fn sample_fps() -> Vec<Fp> {
        vec![
            Fp::zero(),
            Fp::one(),
            Fp::from_u64(2),
            Fp::from_u64(0xffff_ffff_ffff_ffff),
            // Random-ish element clearly < p (top limb's MS byte < 0x30).
            Fp { limbs: [0x0123456789abcdef, 0xfedcba9876543210, 0xdeadbeefcafebabe, 0x0f0e0d0c0b0a0908] },
            // p - 1
            Fp { limbs: [P_LIMBS[0], P_LIMBS[1], P_LIMBS[2], P_LIMBS[3] - 1] },
        ]
    }

    #[test]
    fn bn254_fp_air_column_layout_is_consistent() {
        assert_eq!(LIMBS_PER_FP, 4);
        assert_eq!(LIMBS_PER_PROD, 8);
        assert_eq!(COL_A_OFFSET, 0);
        assert_eq!(COL_B_OFFSET, 4);
        assert_eq!(COL_C_OFFSET, 8);
        assert_eq!(NUM_DATA_COLUMNS, 98);
        assert_eq!(COL_SEL_ADD, 98);
        assert_eq!(COL_SEL_SUB, 99);
        assert_eq!(COL_SEL_MUL, 100);
        assert_eq!(COL_SEL_INV, 101);
        assert_eq!(NUM_BN254_FP_COLUMNS, 102);
    }

    #[test]
    fn bn254_fp_modulus_is_canonical() {
        // BN254 p has 254 bits — top limb's high 2 bits must be zero.
        assert_eq!(P_LIMBS[0] >> 62, 0);
        // Matches the BN254 RFC value at MSB and LSB limbs.
        assert_eq!(P_LIMBS[0], 0x30644E72E131A029);
        assert_eq!(P_LIMBS[3], 0x3C208C16D87CFD47);
    }

    #[test]
    fn bn254_fp_air_known_add_vector() {
        // (p - 1) + 1 ≡ 0 (mod p) — exercises the reduce path.
        let a = Fp { limbs: [P_LIMBS[0], P_LIMBS[1], P_LIMBS[2], P_LIMBS[3] - 1] };
        let b = Fp::one();
        let r = a.add(&b);
        assert!(r.is_zero(), "(p - 1) + 1 must reduce to 0");

        // 2 + 3 = 5 — exercises the non-reduce path.
        let s = Fp::from_u64(2).add(&Fp::from_u64(3));
        assert_eq!(s, Fp::from_u64(5));

        let ops = vec![FpOp::Add { a, b }, FpOp::Add { a: Fp::from_u64(2), b: Fp::from_u64(3) }];
        let columns = populate_trace(&ops, CurveType::Bls48581, None);
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn bn254_fp_air_known_mul_vector() {
        // 7 * 11 = 77, no reduction.
        let a = Fp::from_u64(7);
        let b = Fp::from_u64(11);
        let r = a.mul(&b);
        assert_eq!(r, Fp::from_u64(77));

        // (p - 1) * (p - 1) ≡ 1 (mod p).
        let pm1 = Fp { limbs: [P_LIMBS[0], P_LIMBS[1], P_LIMBS[2], P_LIMBS[3] - 1] };
        let r2 = pm1.mul(&pm1);
        assert_eq!(r2, Fp::one(), "(p - 1)^2 must reduce to 1 mod p");

        // 1 * x = x identity for a non-trivial x.
        let big = Fp { limbs: [0x0123456789abcdef, 0xfedcba9876543210, 0xdeadbeefcafebabe, 0x0f0e0d0c0b0a0908] };
        assert_eq!(Fp::one().mul(&big), big, "Fp::one() must be the mul identity");

        let samples = sample_fps();
        let ops: Vec<FpOp> = samples.iter().flat_map(|a| samples.iter().map(move |b| FpOp::Mul { a: *a, b: *b })).collect();
        let columns = populate_trace(&ops, CurveType::Bls48581, None);
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn bn254_fp_air_known_inv_vector() {
        // 2 * 2^{-1} = 1 mod p.
        let two = Fp::from_u64(2);
        let inv = two.invert().expect("nonzero");
        let prod = two.mul(&inv);
        assert_eq!(prod, Fp::one(), "2 * inv(2) must equal 1 mod p");

        let samples = sample_fps();
        let ops: Vec<FpOp> = samples.iter()
            .filter(|x| !x.is_zero())
            .map(|x| FpOp::Inv { a: *x })
            .collect();
        let columns = populate_trace(&ops, CurveType::Bls48581, None);
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn bn254_fp_air_mixed_ops_in_one_trace() {
        let a = Fp::from_u64(7);
        let b = Fp::from_u64(11);
        // Big sample with top byte < 0x30 so it's clearly < p.
        let big = Fp { limbs: [0x0123456789abcdef, 0xfedcba9876543210, 0xdeadbeefcafebabe, 0x0080808080808080] };
        let ops = vec![
            FpOp::Add { a, b },
            FpOp::Sub { a, b },
            FpOp::Mul { a, b },
            FpOp::Add { a: big, b: big },
            FpOp::Mul { a: big, b },
            FpOp::Sub { a: b, b: a },
            FpOp::Inv { a },
        ];
        let columns = populate_trace(&ops, CurveType::Bls48581, None);
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn bn254_fp_air_rejects_tampered_c_limb() {
        let a = Fp::from_u64(123_456_789);
        let b = Fp::from_u64(987_654_321);
        let ops = vec![FpOp::Mul { a, b }];
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&ops, curve, None);
        let one = Scalar::one(curve);
        columns[c_limb(3)][0] = columns[c_limb(3)][0].add(&one);
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let fired = evals.iter().any(|ce| !ce.values[0].is_zero());
        assert!(fired, "tampered c_limb must trigger some constraint");
    }

    #[test]
    fn bn254_fp_air_rejects_double_selector() {
        let ops = vec![FpOp::Add { a: Fp::from_u64(1), b: Fp::from_u64(2) }];
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&ops, curve, None);
        let one = Scalar::one(curve);
        columns[COL_SEL_MUL][0] = one;
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let cat = evals.iter().find(|ce| ce.label == "selector_sum_01").unwrap();
        assert!(!cat.values[0].is_zero(), "double selector must fire selector_sum_01");
    }

    #[test]
    fn bn254_fp_air_padding_row_is_valid() {
        let ops = vec![FpOp::Add { a: Fp::from_u64(5), b: Fp::from_u64(7) }];
        let columns = populate_trace(&ops, CurveType::Bls48581, Some(4));
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn bn254_fp_air_lookup_declarations_are_well_formed() {
        let decls = lookup_declarations();
        assert!(!decls.is_empty());
        for d in &decls {
            assert!(d.column_index < NUM_BN254_FP_COLUMNS,
                    "decl `{}` references out-of-range column {}", d.label, d.column_index);
            assert!(d.max_bits == 64 || d.max_bits == 72 || d.max_bits == 1,
                    "unexpected max_bits {} for decl `{}`", d.max_bits, d.label);
        }
    }

    #[test]
    fn bn254_curve_ops_to_fp_air_linkage_descriptor_well_formed() {
        let d_rx = make_bn254_curve_ops_to_fp_air_linkage_descriptor(0, 1, 0);
        assert_eq!(d_rx.a_layer_index, 0);
        assert_eq!(d_rx.b_layer_index, 1);
        assert_eq!(d_rx.a_columns.len(), LIMBS_PER_FP);
        assert_eq!(d_rx.b_columns.len(), LIMBS_PER_FP);
        // a side: r.x limbs in curve_ops.
        assert_eq!(d_rx.a_columns[0], crate::bn254_curve_ops_air::COL_R_OFFSET);
        assert_eq!(d_rx.a_columns[LIMBS_PER_FP - 1], crate::bn254_curve_ops_air::COL_R_OFFSET + LIMBS_PER_FP - 1);
        // b side: c_limbs in Fp AIR.
        assert_eq!(d_rx.b_columns[0], COL_C_OFFSET);
        assert_eq!(d_rx.a_selector_column, Some(crate::bn254_curve_ops_air::COL_IS_REAL));
        assert_eq!(d_rx.b_selector_column, Some(COL_SEL_MUL));
        // Ry version.
        let d_ry = make_bn254_curve_ops_to_fp_air_linkage_descriptor(0, 1, 1);
        assert_eq!(d_ry.a_columns[0], crate::bn254_curve_ops_air::COL_R_OFFSET + LIMBS_PER_FP);
        assert!(d_ry.label.contains("ry"));
    }
}
