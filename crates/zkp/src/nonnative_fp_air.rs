//! Limb-level AIR for BLS12-381 non-native `Fp` arithmetic.
//!
//! Foundation for the in-circuit BLS12-381 pairing: expresses [`Fp::add`],
//! [`Fp::sub`] and [`Fp::mul`] as per-row algebraic constraints over the
//! native proving field. Each row encodes exactly one Fp operation,
//! discriminated by a one-hot `sel_*` selector.
//!
//! # Limb-level, not bit-level
//!
//! `Fp` is 381 bits held as 6 u64 limbs (big-endian). A bit-level AIR would
//! need 6·64 = 384 bit columns *per value* (1152 for (a, b, r)); instead we
//! keep limbs as u64-valued columns and rely on 64-bit range-check LogUp
//! declarations to bind each limb to `[0, 2^64)`. Carries are kept as small
//! single-limb columns. The practical column count is ~95 per row — two
//! orders of magnitude smaller than the bit-level Keccak AIR.
//!
//! # Row shape
//!
//! One row per Fp operation. Operation selectors are mutually exclusive and
//! sum to ≤ 1 per row (so padding rows can have all selectors zero).
//!
//! # Column layout (big-endian limb order)
//!
//! All limb columns use big-endian indexing (index 0 = most significant
//! u64). Carry/borrow columns use *LSB-first* indexing (index 0 = carry-out
//! of the least significant limb) to match the natural chain direction.
//!
//! ```text
//! offset  name              size  notes
//! 0       a_limbs            6    input A (BE)
//! 6       b_limbs            6    input B (BE)
//! 12      r_limbs            6    output = op(A, B) (BE)
//!
//! -- Add aux --
//! 18      add_unred_sum      6    unreduced a + b (BE)
//! 24      add_carries        6    LSB-first carry-out, [5] = final overflow
//! 30      add_reduce_flag    1    1 iff unred_sum ≥ p (take trial_diff)
//! 31      add_trial_diff     6    unred_sum - p (BE)
//! 37      add_sub_borrows    6    LSB-first borrow-out for trial subtract
//!
//! -- Sub aux --
//! 43      sub_unred_diff     6    unreduced a - b (BE)
//! 49      sub_borrows        6    LSB-first borrow-out, [5] = final underflow
//! 55      sub_add_back_flag  1    1 iff a < b (take trial_sum)
//! 56      sub_trial_sum      6    unred_diff + p (BE)
//! 62      sub_add_carries    6    LSB-first carry-out for trial add
//!
//! -- Mul aux --
//! 68      prod_limbs        12    12-limb product a·b (BE)
//! 80      quotient_limbs     6    q such that a·b = q·p + r (BE)
//! 86      qp_limbs          12    12-limb product q·p (BE)
//! 98      mul_carry_ab      12    carry chain for a·b schoolbook (LSB-first)
//! 110     mul_carry_qp      12    carry chain for q·p schoolbook (LSB-first)
//! 122     mul_sum_carry     12    carry chain for qp + r_padded = prod (binary)
//!
//! -- Slack (r < p) --
//! 134     slack_limbs        6    p - 1 - r (BE), always ≥ 0 iff r < p
//! 140     slack_borrows      6    LSB-first borrow-out for (p - 1) - r
//!
//! -- Selectors --
//! 146     sel_add            1
//! 147     sel_sub            1
//! 148     sel_mul            1
//! 149     sel_inv            1   reuses Mul aux + asserts r = 1
//! ```
//!
//! Total: **150 columns** per row.
//!
//! # Constraints (all gated by the row's op selector where relevant)
//!
//! ## Add (gated by `sel_add`)
//!
//! For limbs indexed LSB-first (i.e. i = 0 is BE limb 5, i = 5 is BE limb 0):
//!
//!  1. Limb addition chain:
//!     `a[i] + b[i] + c_in[i] = unred_sum[i] + c_out[i]·2^64`
//!     where `c_in[0] = 0`, `c_in[i] = add_carries[i-1]` for i ≥ 1,
//!     and `c_out[i] = add_carries[i]`.
//!  2. Each `add_carries[i]` is binary: `c·(c−1) = 0`.
//!  3. Trial subtraction `unred_sum − p = add_trial_diff`:
//!     `unred_sum[i] − p[i] − borrow_in[i] = trial_diff[i] − borrow_out[i]·2^64`
//!  4. Each `add_sub_borrows[i]` is binary.
//!  5. `add_reduce_flag` is binary.
//!  6. `reduce_flag = add_carries[5] OR (1 − add_sub_borrows[5])`
//!     expressed as `reduce_flag − add_carries[5] − (1 − add_sub_borrows[5]) +
//!     add_carries[5]·(1 − add_sub_borrows[5]) = 0`
//!     (i.e. OR: x + y − xy).
//!  7. Result selection:
//!     `r[i] = (1 − flag)·unred_sum[i] + flag·trial_diff[i]`
//!
//! ## Sub (gated by `sel_sub`)
//!
//!  1. Limb subtraction chain:
//!     `a[i] − b[i] − borrow_in[i] = unred_diff[i] − borrow_out[i]·2^64`
//!  2. Each `sub_borrows[i]` binary.
//!  3. Trial add-back `unred_diff + p = trial_sum`:
//!     `unred_diff[i] + p[i] + carry_in[i] = trial_sum[i] + carry_out[i]·2^64`
//!  4. Each `sub_add_carries[i]` binary.
//!  5. `sub_add_back_flag` is binary.
//!  6. `flag = sub_borrows[5]` (1 iff underflow).
//!  7. Result selection:
//!     `r[i] = (1 − flag)·unred_diff[i] + flag·trial_sum[i]`
//!
//! ## Mul (gated by `sel_mul`)
//!
//!  1. 6×6 schoolbook for `a·b = prod`: for each LSB-position k ∈ [0, 12):
//!     `Σ_{i+j=k; i,j<6} a_le[i]·b_le[j] + carry_in[k] − prod_le[k] −
//!      mul_carry_ab[k]·2^64 = 0`
//!     (where a_le = reversed a_limbs, carry_in[0] = 0, and the last
//!     carry-out at k=11 must be zero — enforced implicitly by not having
//!     a carry sink; equivalently we bound `mul_carry_ab[11] = 0`).
//!  2. Same schoolbook for `q·p = qp`, in `mul_carry_qp` and `qp_limbs`.
//!  3. `qp + r_padded = prod` via 12-limb add with binary `mul_sum_carry`:
//!     `qp_le[k] + r_padded_le[k] + c_in[k] = prod_le[k] + c_out[k]·2^64`
//!     where `r_padded_le` is r (6 limbs) in the low 6 LSB positions,
//!     zeroed above.
//!  4. Each `mul_sum_carry[k]` is binary.
//!  5. `r < p` via slack: for each LSB-position i,
//!     `(p[i] − 1·[i==0]) − r[i] − borrow_in[i] + borrow_out[i]·2^64 −
//!      slack[i] = 0` — i.e. we witness `p − 1 − r` and require no final
//!     borrow. Equivalent algebraic statement: `slack_borrows[5] = 0`.
//!
//! ## Inv (gated by `sel_inv`)
//!
//! Reuses the entire Mul body (CAT 9–14) — Inv rows populate the row
//! exactly like Mul with `b = a^{-1}` (host-side witness) and `r = 1`.
//! All Mul gating constraints are evaluated under `sel_mul + sel_inv`.
//! The only Inv-specific constraint is:
//!
//!  1. `r = 1` on Inv rows: `sel_inv · r[i] = 0` for high BE limbs i = 0..4
//!     and `sel_inv · (r[5] − 1) = 0` for the low BE limb. Together with
//!     the Mul body this proves `a · b ≡ 1 (mod p)`, i.e. `b = a^{-1}`.
//!
//! ## Shared
//!
//!  - Each `sel_*` is binary.
//!  - `sel_add + sel_sub + sel_mul + sel_inv ∈ {0, 1}` enforced as
//!    `s·(s − 1) = 0` where `s = Σ sel_*`.
//!
//! # Range-check widths
//!
//! - `mul_carry_ab[k]` and `mul_carry_qp[k]` fit in roughly 67 bits (sum of
//!   6 partial products at limb k is up to `6·(2^64−1)^2`, whose top-64-bit
//!   part is up to `6·(2^64−1) + 5` ≈ 2^66.6). Range-checked at 72 bits
//!   (9-byte byte-decomposition) to comfortably cover that worst case.
//! - `mul_sum_carry[k]` is exactly 1-bit: 2 inputs (qp limb + r limb) plus
//!   previous carry ≤ 2, so carry-out ≤ 1. Declared binary.
//! - `slack_borrows[5] = 0` is enforced directly (no range gap).
//!
//! # Scope
//!
//! - This module only provides: column layout, witness population, per-row
//!   constraint evaluation (`evaluate_constraints`) and lookup declarations
//!   (`lookup_declarations`). It does **not** implement the full
//!   `VmConstraintSystem` trait nor hook into prove/verify — that is
//!   follow-up work.
//! - Sub is fully implemented (`Fp::sub_witness` was added alongside
//!   `add_witness` in `nonnative_fp.rs`).

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupDeclaration;
use crate::nonnative_fp::{Fp, P_LIMBS};

// ─── Column layout constants ──────────────────────────────────────────────

pub const LIMBS_PER_FP: usize = 6;
pub const LIMBS_PER_PROD: usize = 12;

pub const COL_A_OFFSET:             usize = 0;
pub const COL_B_OFFSET:             usize = COL_A_OFFSET + LIMBS_PER_FP;
pub const COL_R_OFFSET:             usize = COL_B_OFFSET + LIMBS_PER_FP;

// Add aux
pub const COL_ADD_UNRED_SUM_OFFSET: usize = COL_R_OFFSET + LIMBS_PER_FP;
pub const COL_ADD_CARRIES_OFFSET:   usize = COL_ADD_UNRED_SUM_OFFSET + LIMBS_PER_FP;
pub const COL_ADD_REDUCE_FLAG:      usize = COL_ADD_CARRIES_OFFSET + LIMBS_PER_FP;
pub const COL_ADD_TRIAL_DIFF_OFFSET:usize = COL_ADD_REDUCE_FLAG + 1;
pub const COL_ADD_SUB_BORROWS_OFFSET:usize = COL_ADD_TRIAL_DIFF_OFFSET + LIMBS_PER_FP;

// Sub aux
pub const COL_SUB_UNRED_DIFF_OFFSET:usize = COL_ADD_SUB_BORROWS_OFFSET + LIMBS_PER_FP;
pub const COL_SUB_BORROWS_OFFSET:   usize = COL_SUB_UNRED_DIFF_OFFSET + LIMBS_PER_FP;
pub const COL_SUB_ADD_BACK_FLAG:    usize = COL_SUB_BORROWS_OFFSET + LIMBS_PER_FP;
pub const COL_SUB_TRIAL_SUM_OFFSET: usize = COL_SUB_ADD_BACK_FLAG + 1;
pub const COL_SUB_ADD_CARRIES_OFFSET:usize = COL_SUB_TRIAL_SUM_OFFSET + LIMBS_PER_FP;

// Mul aux
pub const COL_PROD_OFFSET:          usize = COL_SUB_ADD_CARRIES_OFFSET + LIMBS_PER_FP;
pub const COL_QUOTIENT_OFFSET:      usize = COL_PROD_OFFSET + LIMBS_PER_PROD;
pub const COL_QP_OFFSET:            usize = COL_QUOTIENT_OFFSET + LIMBS_PER_FP;
pub const COL_MUL_CARRY_AB_OFFSET:  usize = COL_QP_OFFSET + LIMBS_PER_PROD;
pub const COL_MUL_CARRY_QP_OFFSET:  usize = COL_MUL_CARRY_AB_OFFSET + LIMBS_PER_PROD;
pub const COL_MUL_SUM_CARRY_OFFSET: usize = COL_MUL_CARRY_QP_OFFSET + LIMBS_PER_PROD;

// Slack
pub const COL_SLACK_OFFSET:         usize = COL_MUL_SUM_CARRY_OFFSET + LIMBS_PER_PROD;
pub const COL_SLACK_BORROWS_OFFSET: usize = COL_SLACK_OFFSET + LIMBS_PER_FP;

// Data-column count (before selectors).
pub const NUM_DATA_COLUMNS: usize = COL_SLACK_BORROWS_OFFSET + LIMBS_PER_FP;

// Selectors
pub const COL_SEL_ADD: usize = NUM_DATA_COLUMNS;
pub const COL_SEL_SUB: usize = COL_SEL_ADD + 1;
pub const COL_SEL_MUL: usize = COL_SEL_SUB + 1;
// Inv reuses the entire Mul aux layout (a, b=witness inverse, r forced to 1
// plus prod/quotient/qp/carries/slack columns). The only new gating it adds
// is `r = 1`. So no new data columns — only this selector.
pub const COL_SEL_INV: usize = COL_SEL_MUL + 1;

pub const NUM_NONNATIVE_FP_COLUMNS: usize = COL_SEL_INV + 1;

// ─── Column index helpers ────────────────────────────────────────────────

#[inline]
pub fn a_limb(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_FP);
    COL_A_OFFSET + i
}
#[inline]
pub fn b_limb(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_FP);
    COL_B_OFFSET + i
}
#[inline]
pub fn r_limb(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_FP);
    COL_R_OFFSET + i
}
#[inline]
pub fn add_unred_sum(i: usize) -> usize {
    COL_ADD_UNRED_SUM_OFFSET + i
}
#[inline]
pub fn add_carry(i: usize) -> usize {
    COL_ADD_CARRIES_OFFSET + i
}
#[inline]
pub fn add_trial_diff(i: usize) -> usize {
    COL_ADD_TRIAL_DIFF_OFFSET + i
}
#[inline]
pub fn add_sub_borrow(i: usize) -> usize {
    COL_ADD_SUB_BORROWS_OFFSET + i
}
#[inline]
pub fn sub_unred_diff(i: usize) -> usize {
    COL_SUB_UNRED_DIFF_OFFSET + i
}
#[inline]
pub fn sub_borrow(i: usize) -> usize {
    COL_SUB_BORROWS_OFFSET + i
}
#[inline]
pub fn sub_trial_sum(i: usize) -> usize {
    COL_SUB_TRIAL_SUM_OFFSET + i
}
#[inline]
pub fn sub_add_carry(i: usize) -> usize {
    COL_SUB_ADD_CARRIES_OFFSET + i
}
#[inline]
pub fn prod_limb(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_PROD);
    COL_PROD_OFFSET + i
}
#[inline]
pub fn quotient_limb(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_FP);
    COL_QUOTIENT_OFFSET + i
}
#[inline]
pub fn qp_limb(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_PROD);
    COL_QP_OFFSET + i
}
#[inline]
pub fn mul_carry_ab(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_PROD);
    COL_MUL_CARRY_AB_OFFSET + i
}
#[inline]
pub fn mul_carry_qp(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_PROD);
    COL_MUL_CARRY_QP_OFFSET + i
}
#[inline]
pub fn mul_sum_carry(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_PROD);
    COL_MUL_SUM_CARRY_OFFSET + i
}
#[inline]
pub fn slack_limb(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_FP);
    COL_SLACK_OFFSET + i
}
#[inline]
pub fn slack_borrow(i: usize) -> usize {
    debug_assert!(i < LIMBS_PER_FP);
    COL_SLACK_BORROWS_OFFSET + i
}

// ─── Operation description ───────────────────────────────────────────────

/// A single Fp operation for one trace row.
///
/// `Inv { a }` proves `a · a^{-1} = 1` by populating the row with `b = a^{-1}`
/// (host-side witness) and forcing `r = 1`. It reuses every Mul aux column
/// (prod/qp/quotient/carries/slack) and gating, plus a small extra constraint
/// that asserts `r = 1` on Inv rows. `a` must be nonzero; calling `Inv`
/// on `a = 0` panics during witness population.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpOp {
    Add { a: Fp, b: Fp },
    Sub { a: Fp, b: Fp },
    Mul { a: Fp, b: Fp },
    Inv { a: Fp },
}

// ─── Small helpers ───────────────────────────────────────────────────────

/// `2^64` as a field scalar. Matches the SBF `two_pow_64` helper.
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

/// Convert a u64 to `Scalar`.
#[inline]
fn sc(v: u64, curve: CurveType) -> Scalar {
    Scalar::from_u64(v, curve)
}

/// Convert a u128 to `Scalar`. Used for schoolbook mul carries which can
/// exceed 64 bits (~67 bits in practice).
fn sc128(v: u128, curve: CurveType) -> Scalar {
    // Decompose as `lo + hi * 2^64` where both are u64, using field
    // arithmetic to combine. This is curve-agnostic.
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    let lo_s = Scalar::from_u64(lo, curve);
    if hi == 0 {
        return lo_s;
    }
    let hi_s = Scalar::from_u64(hi, curve);
    let two64 = two_pow_64(curve);
    lo_s.add(&hi_s.mul(&two64))
}

/// Allocate `NUM_NONNATIVE_FP_COLUMNS` columns each of `num_rows` zeros.
pub fn alloc_trace(num_rows: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..NUM_NONNATIVE_FP_COLUMNS)
        .map(|_| vec![zero.clone(); num_rows])
        .collect()
}

#[inline]
fn le_to_be_idx(lsb_first_index: usize, n_limbs: usize) -> usize {
    n_limbs - 1 - lsb_first_index
}

// ─── Witness population ──────────────────────────────────────────────────

/// Compute the 12-limb product as a column-by-column 6×6 schoolbook, matching
/// the AIR's constraint shape exactly. Returns LSB-first product limbs and
/// LSB-first carry-out chain (`carry_out[k]` = carry out of column k).
///
/// At column k, accumulator = Σ_{i+j=k} a_le[i]·b_le[j] + carry_in_from_k-1.
/// Since this accumulator can exceed u128 (6 · (2^64−1)^2 ≈ 2^130.58), we
/// track (hi: u64, lo: u128) — low 128 bits and overflow count of 2^128.
/// Carry out of column k = (hi·2^128 + lo) >> 64, which fits in u128.
fn schoolbook_6x6_le(a_be: &[u64; 6], b_be: &[u64; 6]) -> ([u64; 12], [u128; 12]) {
    let a_le: [u64; 6] = [a_be[5], a_be[4], a_be[3], a_be[2], a_be[1], a_be[0]];
    let b_le: [u64; 6] = [b_be[5], b_be[4], b_be[3], b_be[2], b_be[1], b_be[0]];
    let mut prod = [0u64; 12];
    let mut carry_out = [0u128; 12];
    // Column carry fits in u128 comfortably (empirical max ~2^67).
    let mut carry_in: u128 = 0;
    for k in 0..12 {
        // Accumulate partial products with overflow tracking.
        // lo: u128 low 128 bits; hi: u64 overflow count.
        let mut lo: u128 = 0;
        let mut hi: u64 = 0;
        for i in 0..6 {
            let j = k as isize - i as isize;
            if j < 0 || j >= 6 { continue; }
            let p = (a_le[i] as u128) * (b_le[j as usize] as u128);
            let (new_lo, ov) = lo.overflowing_add(p);
            lo = new_lo;
            if ov { hi += 1; }
        }
        // Add carry_in.
        let (new_lo, ov) = lo.overflowing_add(carry_in);
        lo = new_lo;
        if ov { hi += 1; }

        // Extract low 64 bits for prod[k]; the remainder forms carry_out[k].
        prod[k] = lo as u64;
        // Next carry = ((hi << 128) + lo) >> 64 = (hi << 64) + (lo >> 64).
        let cout: u128 = ((hi as u128) << 64) + (lo >> 64);
        carry_out[k] = cout;
        carry_in = cout;
    }
    debug_assert_eq!(carry_in, 0, "6x6 schoolbook carry overflowed at top");
    (prod, carry_out)
}

/// Compute `qp + r_padded = prod` 12-limb add; return LSB-first carry chain.
fn sum_carries_le(qp_be: &[u64; 12], r_be: &[u64; 6]) -> ([u64; 12], [u64; 12]) {
    // LE: qp_le[k] = qp_be[11 - k], r padded into low 6 LSB positions.
    let mut qp_le = [0u64; 12];
    for k in 0..12 { qp_le[k] = qp_be[11 - k]; }
    let mut r_le = [0u64; 12];
    for k in 0..6  { r_le[k]  = r_be[5 - k]; }
    let mut sum = [0u64; 12];
    let mut carries = [0u64; 12];
    let mut carry: u64 = 0;
    for k in 0..12 {
        let t = (qp_le[k] as u128) + (r_le[k] as u128) + (carry as u128);
        sum[k] = t as u64;
        carry = (t >> 64) as u64;
        carries[k] = carry;
    }
    (sum, carries)
}

/// Witness the slack `p - 1 - r` with borrow chain. Returns
/// `(slack_be, borrows_le)`. Caller must check `borrows_le[5] == 0`.
fn slack_witness(r_be: &[u64; 6]) -> ([u64; 6], [u64; 6]) {
    // Compute `p_minus_1 - r` as 6-limb BE; borrow chain LSB-first.
    // p_minus_1 = P - 1.
    let mut pm1 = P_LIMBS;
    // BE index 5 is LSB.
    pm1[5] = pm1[5].wrapping_sub(1);
    let mut diff = [0u64; 6];
    let mut borrows = [0u64; 6];
    let mut borrow: u64 = 0;
    for i in (0..6).rev() {
        let (d1, b1) = pm1[i].overflowing_sub(r_be[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        diff[i] = d2;
        let bout = (b1 as u64) + (b2 as u64);
        borrows[5 - i] = bout;
        borrow = bout;
    }
    (diff, borrows)
}

/// Populate a single row with an [`FpOp`].
pub fn populate_row(
    columns: &mut [Vec<Scalar>],
    row: usize,
    op: &FpOp,
    curve: CurveType,
) {
    assert_eq!(columns.len(), NUM_NONNATIVE_FP_COLUMNS, "columns shape");
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);

    // Reset the row to zero first (so fields not touched by this op are
    // cleanly zero).
    for col in columns.iter_mut() {
        col[row] = zero.clone();
    }

    // Helpers to write be limbs.
    let write_be = |columns: &mut [Vec<Scalar>], offset: usize, limbs: &[u64; 6], row: usize| {
        for i in 0..6 {
            columns[offset + i][row] = sc(limbs[i], curve);
        }
    };
    let write_be_12 = |columns: &mut [Vec<Scalar>], offset: usize, limbs: &[u64; 12], row: usize| {
        for i in 0..12 {
            columns[offset + i][row] = sc(limbs[i], curve);
        }
    };
    let write_le_carries_12 = |columns: &mut [Vec<Scalar>], offset: usize, carries: &[u64; 12], row: usize| {
        for k in 0..12 {
            columns[offset + k][row] = sc(carries[k], curve);
        }
    };
    let write_le_carries_12_u128 = |columns: &mut [Vec<Scalar>], offset: usize, carries: &[u128; 12], row: usize| {
        for k in 0..12 {
            columns[offset + k][row] = sc128(carries[k], curve);
        }
    };

    match op {
        FpOp::Add { a, b } => {
            let w = a.add_witness(b);

            write_be(columns, COL_A_OFFSET, &a.limbs, row);
            write_be(columns, COL_B_OFFSET, &b.limbs, row);
            write_be(columns, COL_R_OFFSET, &w.result, row);

            write_be(columns, COL_ADD_UNRED_SUM_OFFSET, &w.sum, row);
            // add_carries is LSB-first; we only store indices 0..6.
            for i in 0..6 {
                columns[COL_ADD_CARRIES_OFFSET + i][row] = sc(w.add_carries[i], curve);
            }
            columns[COL_ADD_REDUCE_FLAG][row] = sc(w.reduce_flag, curve);
            write_be(columns, COL_ADD_TRIAL_DIFF_OFFSET, &w.trial_diff, row);
            for i in 0..6 {
                columns[COL_ADD_SUB_BORROWS_OFFSET + i][row] = sc(w.sub_borrows[i], curve);
            }

            // Slack: r < p
            let (slack, sb) = slack_witness(&w.result);
            write_be(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..6 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }

            columns[COL_SEL_ADD][row] = one.clone();
        }
        FpOp::Sub { a, b } => {
            let w = a.sub_witness(b);

            write_be(columns, COL_A_OFFSET, &a.limbs, row);
            write_be(columns, COL_B_OFFSET, &b.limbs, row);
            write_be(columns, COL_R_OFFSET, &w.result, row);

            write_be(columns, COL_SUB_UNRED_DIFF_OFFSET, &w.diff, row);
            for i in 0..6 {
                columns[COL_SUB_BORROWS_OFFSET + i][row] = sc(w.sub_borrows[i], curve);
            }
            columns[COL_SUB_ADD_BACK_FLAG][row] = sc(w.add_back_flag, curve);
            write_be(columns, COL_SUB_TRIAL_SUM_OFFSET, &w.trial_sum, row);
            for i in 0..6 {
                columns[COL_SUB_ADD_CARRIES_OFFSET + i][row] = sc(w.add_carries[i], curve);
            }

            let (slack, sb) = slack_witness(&w.result);
            write_be(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..6 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }

            columns[COL_SEL_SUB][row] = one.clone();
        }
        FpOp::Mul { a, b } => {
            let w = a.mul_witness(b);

            write_be(columns, COL_A_OFFSET, &a.limbs, row);
            write_be(columns, COL_B_OFFSET, &b.limbs, row);
            write_be(columns, COL_R_OFFSET, &w.r, row);

            // a*b schoolbook: recompute LE chain.
            let (prod_le, carry_ab_le) = schoolbook_6x6_le(&a.limbs, &b.limbs);
            // Convert prod_le to BE for column storage.
            let mut prod_be = [0u64; 12];
            for k in 0..12 { prod_be[k] = prod_le[11 - k]; }
            write_be_12(columns, COL_PROD_OFFSET, &prod_be, row);
            write_le_carries_12_u128(columns, COL_MUL_CARRY_AB_OFFSET, &carry_ab_le, row);

            // Quotient q (BE).
            write_be(columns, COL_QUOTIENT_OFFSET, &w.q, row);

            // q*p schoolbook.
            let (qp_le, carry_qp_le) = schoolbook_6x6_le(&w.q, &P_LIMBS);
            let mut qp_be = [0u64; 12];
            for k in 0..12 { qp_be[k] = qp_le[11 - k]; }
            // Cross-check with the witness:
            debug_assert_eq!(qp_be, w.q_times_p, "qp schoolbook mismatch");
            write_be_12(columns, COL_QP_OFFSET, &qp_be, row);
            write_le_carries_12_u128(columns, COL_MUL_CARRY_QP_OFFSET, &carry_qp_le, row);

            // qp + r = prod additive chain (LSB-first carries).
            let (sum_check, sum_carries_le_vals) = sum_carries_le(&qp_be, &w.r);
            debug_assert_eq!(sum_check, prod_le, "qp + r reconstruction mismatch");
            write_le_carries_12(columns, COL_MUL_SUM_CARRY_OFFSET, &sum_carries_le_vals, row);

            // Slack: r < p.
            let (slack, sb) = slack_witness(&w.r);
            write_be(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..6 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }

            columns[COL_SEL_MUL][row] = one.clone();
        }
        FpOp::Inv { a } => {
            let inv = a.invert().expect("Inv on zero Fp");
            // Populate as a Mul row would for (a, inv) with r forced to 1.
            let w = a.mul_witness(&inv);
            // Fp::one()'s big-endian limbs.
            let one_limbs: [u64; 6] = [0, 0, 0, 0, 0, 1];
            debug_assert_eq!(w.r, one_limbs, "a · a^-1 must reduce to 1");

            write_be(columns, COL_A_OFFSET, &a.limbs, row);
            write_be(columns, COL_B_OFFSET, &inv.limbs, row);
            write_be(columns, COL_R_OFFSET, &one_limbs, row);

            let (prod_le, carry_ab_le) = schoolbook_6x6_le(&a.limbs, &inv.limbs);
            let mut prod_be = [0u64; 12];
            for k in 0..12 { prod_be[k] = prod_le[11 - k]; }
            write_be_12(columns, COL_PROD_OFFSET, &prod_be, row);
            write_le_carries_12_u128(columns, COL_MUL_CARRY_AB_OFFSET, &carry_ab_le, row);

            write_be(columns, COL_QUOTIENT_OFFSET, &w.q, row);

            let (qp_le, carry_qp_le) = schoolbook_6x6_le(&w.q, &P_LIMBS);
            let mut qp_be = [0u64; 12];
            for k in 0..12 { qp_be[k] = qp_le[11 - k]; }
            debug_assert_eq!(qp_be, w.q_times_p, "qp schoolbook mismatch");
            write_be_12(columns, COL_QP_OFFSET, &qp_be, row);
            write_le_carries_12_u128(columns, COL_MUL_CARRY_QP_OFFSET, &carry_qp_le, row);

            let (sum_check, sum_carries_le_vals) = sum_carries_le(&qp_be, &one_limbs);
            debug_assert_eq!(sum_check, prod_le, "qp + 1 reconstruction mismatch");
            write_le_carries_12(columns, COL_MUL_SUM_CARRY_OFFSET, &sum_carries_le_vals, row);

            // Slack: r=1 < p. Compute exactly as the Mul branch.
            let (slack, sb) = slack_witness(&one_limbs);
            write_be(columns, COL_SLACK_OFFSET, &slack, row);
            for i in 0..6 { columns[COL_SLACK_BORROWS_OFFSET + i][row] = sc(sb[i], curve); }

            columns[COL_SEL_INV][row] = one.clone();
        }
    }
}

/// Populate an entire trace from a sequence of operations. Pads the trace
/// with all-zero rows (no selector set) to `num_rows` if provided.
pub fn populate_trace(
    ops: &[FpOp],
    curve: CurveType,
    num_rows: Option<usize>,
) -> Vec<Vec<Scalar>> {
    let rows = num_rows.unwrap_or(ops.len()).max(ops.len());
    let mut columns = alloc_trace(rows, curve);
    for (row, op) in ops.iter().enumerate() {
        populate_row(&mut columns, row, op, curve);
    }
    columns
}

// ─── Constraint evaluation ───────────────────────────────────────────────

/// A single constraint body's per-row evaluations.
#[derive(Debug, Clone)]
pub struct ConstraintEval {
    pub label: String,
    pub values: Vec<Scalar>,
}

/// Evaluate the full nonnative-Fp AIR on a populated trace, returning one
/// `ConstraintEval` per category. Each category aggregates many limb-level
/// sub-constraints via a random linear combination powered by `beta`.
pub fn evaluate_constraints(
    columns: &[&Vec<Scalar>],
    beta: &Scalar,
) -> Vec<ConstraintEval> {
    assert_eq!(
        columns.len(),
        NUM_NONNATIVE_FP_COLUMNS,
        "expected {} columns, got {}",
        NUM_NONNATIVE_FP_COLUMNS,
        columns.len()
    );
    let num_rows = columns[0].len();
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);
    let two64 = two_pow_64(curve);

    // Scalar versions of the p limbs, LSB-first (index i = big-endian limb 5-i).
    let p_le: [Scalar; 6] = [
        sc(P_LIMBS[5], curve),
        sc(P_LIMBS[4], curve),
        sc(P_LIMBS[3], curve),
        sc(P_LIMBS[2], curve),
        sc(P_LIMBS[1], curve),
        sc(P_LIMBS[0], curve),
    ];

    // LSB-first helpers for a[i], b[i] from big-endian column layout.
    let a_at = |i: usize, row: usize| columns[a_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let b_at = |i: usize, row: usize| columns[b_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let r_at = |i: usize, row: usize| columns[r_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let add_unred_at = |i: usize, row: usize|
        columns[add_unred_sum(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let add_trial_at = |i: usize, row: usize|
        columns[add_trial_diff(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let sub_diff_at = |i: usize, row: usize|
        columns[sub_unred_diff(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let sub_trial_at = |i: usize, row: usize|
        columns[sub_trial_sum(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let quot_at = |i: usize, row: usize|
        columns[quotient_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
    let prod_at = |k: usize, row: usize|
        columns[prod_limb(le_to_be_idx(k, LIMBS_PER_PROD))][row].clone();
    let qp_at = |k: usize, row: usize|
        columns[qp_limb(le_to_be_idx(k, LIMBS_PER_PROD))][row].clone();

    let sel_add_col = columns[COL_SEL_ADD];
    let sel_sub_col = columns[COL_SEL_SUB];
    let sel_mul_col = columns[COL_SEL_MUL];
    let sel_inv_col = columns[COL_SEL_INV];
    // Inv reuses every Mul gating: build a per-row "is mul-shaped" mask once.
    let sel_mul_or_inv: Vec<Scalar> = (0..num_rows)
        .map(|row| sel_mul_col[row].add(&sel_inv_col[row]))
        .collect();

    let mut result: Vec<ConstraintEval> = Vec::new();

    // ── CAT 1: Add limb chain ──
    // For LSB i in 0..6:
    //   a[i] + b[i] + c_in[i] − unred_sum[i] − add_carries[i]·2^64 = 0
    // where c_in[0] = 0 and c_in[i] = add_carries[i-1] for i ≥ 1.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..6 {
            for row in 0..num_rows {
                let c_in = if i == 0 { zero.clone() }
                    else { columns[COL_ADD_CARRIES_OFFSET + i - 1][row].clone() };
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

    // ── CAT 2: Add trial diff chain: unred_sum[i] − p[i] − bin[i] −
    //   trial_diff[i] + bout[i]·2^64 = 0
    // Note: borrow chain is *subtraction*, so `unred_sum − p − borrow_in = trial_diff − borrow_out·2^64`.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..6 {
            for row in 0..num_rows {
                let b_in = if i == 0 { zero.clone() }
                    else { columns[COL_ADD_SUB_BORROWS_OFFSET + i - 1][row].clone() };
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

    // ── CAT 3: Add reduce flag definition:
    // flag == overflow OR (NOT borrow)
    //   let x = add_carries[5], y = (1 − add_sub_borrows[5]).
    //   flag = x + y − x·y.
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let x = columns[COL_ADD_CARRIES_OFFSET + 5][row].clone();
            let y = one.sub(&columns[COL_ADD_SUB_BORROWS_OFFSET + 5][row]);
            let xy = x.mul(&y);
            let or = x.add(&y).sub(&xy);
            let flag = columns[COL_ADD_REDUCE_FLAG][row].clone();
            let body = flag.sub(&or);
            let gated = sel_add_col[row].mul(&body);
            acc[row] = gated;
        }
        result.push(ConstraintEval { label: "add_reduce_flag_def".into(), values: acc });
    }

    // ── CAT 4: Add result selection (one body per limb):
    //   r[i] − (1 − flag)·unred_sum[i] − flag·trial_diff[i] = 0
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..6 {
            for row in 0..num_rows {
                let flag = columns[COL_ADD_REDUCE_FLAG][row].clone();
                let one_m_flag = one.sub(&flag);
                let rhs = one_m_flag.mul(&add_unred_at(i, row))
                    .add(&flag.mul(&add_trial_at(i, row)));
                let body = r_at(i, row).sub(&rhs);
                let gated = sel_add_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "add_result_select".into(), values: acc });
    }

    // ── CAT 5: Sub limb chain (borrow):
    //   a[i] − b[i] − borrow_in[i] = unred_diff[i] − borrow_out[i]·2^64
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..6 {
            for row in 0..num_rows {
                let b_in = if i == 0 { zero.clone() }
                    else { columns[COL_SUB_BORROWS_OFFSET + i - 1][row].clone() };
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

    // ── CAT 6: Sub trial add-back chain (unred_diff + p = trial_sum + c·2^64) ──
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..6 {
            for row in 0..num_rows {
                let c_in = if i == 0 { zero.clone() }
                    else { columns[COL_SUB_ADD_CARRIES_OFFSET + i - 1][row].clone() };
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

    // ── CAT 7: Sub add-back flag = sub_borrows[5] ──
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let flag = columns[COL_SUB_ADD_BACK_FLAG][row].clone();
            let top_borrow = columns[COL_SUB_BORROWS_OFFSET + 5][row].clone();
            let body = flag.sub(&top_borrow);
            let gated = sel_sub_col[row].mul(&body);
            acc[row] = gated;
        }
        result.push(ConstraintEval { label: "sub_add_back_flag_def".into(), values: acc });
    }

    // ── CAT 8: Sub result selection (one body per limb) ──
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for i in 0..6 {
            for row in 0..num_rows {
                let flag = columns[COL_SUB_ADD_BACK_FLAG][row].clone();
                let one_m_flag = one.sub(&flag);
                let rhs = one_m_flag.mul(&sub_diff_at(i, row))
                    .add(&flag.mul(&sub_trial_at(i, row)));
                let body = r_at(i, row).sub(&rhs);
                let gated = sel_sub_col[row].mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "sub_result_select".into(), values: acc });
    }

    // ── CAT 9: Mul — 6×6 schoolbook for a·b = prod ──
    // For LSB k in 0..12:
    //   Σ_{i+j=k; i,j<6} a[i]·b[j] + carry_in[k] − prod[k] − carry_out[k]·2^64 = 0
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for k in 0..LIMBS_PER_PROD {
            for row in 0..num_rows {
                let mut col_sum = zero.clone();
                for i in 0..6 {
                    let j = k as isize - i as isize;
                    if j < 0 || j >= 6 { continue; }
                    col_sum = col_sum.add(&a_at(i, row).mul(&b_at(j as usize, row)));
                }
                let c_in = if k == 0 { zero.clone() }
                    else { columns[COL_MUL_CARRY_AB_OFFSET + k - 1][row].clone() };
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

    // ── CAT 10: Mul — final top carry of a·b schoolbook must be zero ──
    // For a, b < 2^384, a·b < 2^768, which fits in 12 u64 limbs with zero
    // final carry. Enforce mul_carry_ab[11] = 0 (gated by sel_mul + sel_inv).
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let top = columns[COL_MUL_CARRY_AB_OFFSET + 11][row].clone();
            let gated = sel_mul_or_inv[row].mul(&top);
            acc[row] = gated;
        }
        result.push(ConstraintEval { label: "mul_ab_top_carry_zero".into(), values: acc });
    }

    // ── CAT 11: Mul — schoolbook for q·p = qp ──
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for k in 0..LIMBS_PER_PROD {
            for row in 0..num_rows {
                let mut col_sum = zero.clone();
                for i in 0..6 {
                    let j = k as isize - i as isize;
                    if j < 0 || j >= 6 { continue; }
                    col_sum = col_sum.add(&quot_at(i, row).mul(&p_le[j as usize]));
                }
                let c_in = if k == 0 { zero.clone() }
                    else { columns[COL_MUL_CARRY_QP_OFFSET + k - 1][row].clone() };
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

    // ── CAT 12: Mul — q·p top carry zero ──
    // q < 2^384, p < 2^381 so q·p < 2^765, fits in 12 limbs with zero top carry.
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let top = columns[COL_MUL_CARRY_QP_OFFSET + 11][row].clone();
            let gated = sel_mul_or_inv[row].mul(&top);
            acc[row] = gated;
        }
        result.push(ConstraintEval { label: "mul_qp_top_carry_zero".into(), values: acc });
    }

    // ── CAT 13: Mul — qp + r_padded = prod (12-limb add with binary carry) ──
    // For LSB k in 0..12:
    //   qp[k] + r_pad[k] + c_in[k] − prod[k] − c_out[k]·2^64 = 0
    // where r_pad[k] = r[k] for k < 6 else 0.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for k in 0..LIMBS_PER_PROD {
            for row in 0..num_rows {
                let r_term = if k < 6 { r_at(k, row) } else { zero.clone() };
                let c_in = if k == 0 { zero.clone() }
                    else { columns[COL_MUL_SUM_CARRY_OFFSET + k - 1][row].clone() };
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

    // ── CAT 14: Mul — top sum carry zero (qp + r fits in 12 limbs since
    // result equals a*b < 2^768). ──
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let top = columns[COL_MUL_SUM_CARRY_OFFSET + 11][row].clone();
            let gated = sel_mul_or_inv[row].mul(&top);
            acc[row] = gated;
        }
        result.push(ConstraintEval { label: "mul_sum_top_carry_zero".into(), values: acc });
    }

    // ── CAT 15: Slack chain — witness (p − 1 − r) and require no final borrow.
    // For LSB i: p[i] − [i==0]·1 − r[i] − borrow_in[i] − slack[i] + borrow_out[i]·2^64 = 0
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        let sel_any = |row: usize| -> Scalar {
            sel_add_col[row].add(&sel_sub_col[row]).add(&sel_mul_or_inv[row])
        };
        for i in 0..6 {
            let p_minus_1_at_i = if i == 0 { p_le[0].sub(&one) } else { p_le[i].clone() };
            for row in 0..num_rows {
                let b_in = if i == 0 { zero.clone() }
                    else { columns[COL_SLACK_BORROWS_OFFSET + i - 1][row].clone() };
                let b_out = columns[COL_SLACK_BORROWS_OFFSET + i][row].clone();
                let slack_i = columns[slack_limb(le_to_be_idx(i, LIMBS_PER_FP))][row].clone();
                let lhs = p_minus_1_at_i.sub(&r_at(i, row)).sub(&b_in);
                let rhs = slack_i.sub(&b_out.mul(&two64));
                let body = lhs.sub(&rhs);
                let gated = sel_any(row).mul(&body);
                acc[row] = acc[row].add(&beta_pow.mul(&gated));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "slack_chain".into(), values: acc });
    }

    // ── CAT 16: Slack top borrow = 0 (guarantees r ≤ p − 1, i.e. r < p).
    {
        let mut acc = vec![zero.clone(); num_rows];
        let sel_any = |row: usize| -> Scalar {
            sel_add_col[row].add(&sel_sub_col[row]).add(&sel_mul_or_inv[row])
        };
        for row in 0..num_rows {
            let top = columns[COL_SLACK_BORROWS_OFFSET + 5][row].clone();
            let gated = sel_any(row).mul(&top);
            acc[row] = gated;
        }
        result.push(ConstraintEval { label: "slack_top_borrow_zero".into(), values: acc });
    }

    // ── CAT 17: Binary flags and carries ──
    // All the following columns are binary: c·(c − 1) = 0.
    // Categories aggregated via β powers:
    //   add_carries[0..6], add_sub_borrows[0..6], add_reduce_flag,
    //   sub_borrows[0..6], sub_add_carries[0..6], sub_add_back_flag,
    //   mul_sum_carry[0..12], sel_add, sel_sub, sel_mul.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();

        let binary_cols: Vec<usize> = {
            let mut v = Vec::new();
            for i in 0..6 { v.push(COL_ADD_CARRIES_OFFSET + i); }
            for i in 0..6 { v.push(COL_ADD_SUB_BORROWS_OFFSET + i); }
            v.push(COL_ADD_REDUCE_FLAG);
            for i in 0..6 { v.push(COL_SUB_BORROWS_OFFSET + i); }
            for i in 0..6 { v.push(COL_SUB_ADD_CARRIES_OFFSET + i); }
            v.push(COL_SUB_ADD_BACK_FLAG);
            for k in 0..LIMBS_PER_PROD { v.push(COL_MUL_SUM_CARRY_OFFSET + k); }
            v.push(COL_SEL_ADD);
            v.push(COL_SEL_SUB);
            v.push(COL_SEL_MUL);
            v.push(COL_SEL_INV);
            v
        };

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

    // ── CAT 18: Selector sum ∈ {0, 1} ──
    //   s = sel_add + sel_sub + sel_mul + sel_inv; s·(s − 1) = 0.
    {
        let mut acc = vec![zero.clone(); num_rows];
        for row in 0..num_rows {
            let s = sel_add_col[row]
                .add(&sel_sub_col[row])
                .add(&sel_mul_col[row])
                .add(&sel_inv_col[row]);
            let body = s.mul(&s.sub(&one));
            acc[row] = body;
        }
        result.push(ConstraintEval { label: "selector_sum_01".into(), values: acc });
    }

    // ── CAT 19: Inv result is 1 ──
    //   sel_inv · r[i]       = 0   for big-endian limbs i = 0..5    (high limbs)
    //   sel_inv · (r[5] − 1) = 0                                    (low limb)
    // Together with the Mul body these prove a · inv ≡ 1 (mod p),
    // i.e. inv = a^{-1} on Inv rows.
    {
        let mut acc = vec![zero.clone(); num_rows];
        let mut beta_pow = one.clone();
        for be in 0..LIMBS_PER_FP {
            for row in 0..num_rows {
                let r_be = columns[COL_R_OFFSET + be][row].clone();
                let target = if be == LIMBS_PER_FP - 1 {
                    r_be.sub(&one)
                } else {
                    r_be
                };
                let body = sel_inv_col[row].mul(&target);
                acc[row] = acc[row].add(&beta_pow.mul(&body));
            }
            beta_pow = beta_pow.mul(beta);
        }
        result.push(ConstraintEval { label: "inv_result_is_one".into(), values: acc });
    }

    result
}

/// The total number of distinct constraint categories returned by
/// [`evaluate_constraints`].
pub const NUM_CONSTRAINT_CATEGORIES: usize = 19;

// ─── Lookup declarations ─────────────────────────────────────────────────

/// Declare all range-check lookups needed for a populated trace. Callers
/// merge these into the global `LookupRequirements` for a whole-proof
/// constraint system.
///
/// **Range widths**
/// - 64-bit: every limb column (a, b, r, sums, differences, prod, qp,
///   quotient, slack). On padding rows (all selectors zero) these columns
///   are all zero per `populate_row`, which trivially lies in `[0, 2^64)`.
/// - 1-bit: carry/borrow/flag columns. Sound: declared as `max_bits = 1`.
/// - 72-bit: Mul schoolbook carries (`mul_carry_ab`, `mul_carry_qp`).
///   For `Mul(a, b)` with both operands being full 6-limb Fp values, the
///   per-column carry can reach ~2^67 (six u64×u64 partial products plus
///   carry-in). Declaring at 72 bits (9-byte byte-decomposition) covers
///   that range; existing nonnative_fp tests with smaller operands stay
///   within the wider window trivially.
pub fn lookup_declarations() -> Vec<LookupDeclaration> {
    let mut decls = Vec::new();

    let range_64 = |col: usize, label: &str| LookupDeclaration {
        label: label.into(),
        column_index: col,
        max_bits: 64,
        selector_column: None,
    };
    let range_72 = |col: usize, label: &str| LookupDeclaration {
        label: label.into(),
        column_index: col,
        max_bits: 72,
        selector_column: None,
    };
    let binary = |col: usize, label: &str| LookupDeclaration {
        label: label.into(),
        column_index: col,
        max_bits: 1,
        selector_column: None,
    };

    // 64-bit limbs.
    for i in 0..LIMBS_PER_FP {
        decls.push(range_64(a_limb(i),           &format!("a_limb_{}_range", i)));
        decls.push(range_64(b_limb(i),           &format!("b_limb_{}_range", i)));
        decls.push(range_64(r_limb(i),           &format!("r_limb_{}_range", i)));
        decls.push(range_64(add_unred_sum(i),    &format!("add_unred_sum_{}_range", i)));
        decls.push(range_64(add_trial_diff(i),   &format!("add_trial_diff_{}_range", i)));
        decls.push(range_64(sub_unred_diff(i),   &format!("sub_unred_diff_{}_range", i)));
        decls.push(range_64(sub_trial_sum(i),    &format!("sub_trial_sum_{}_range", i)));
        decls.push(range_64(quotient_limb(i),    &format!("quotient_limb_{}_range", i)));
        decls.push(range_64(slack_limb(i),       &format!("slack_limb_{}_range", i)));
    }
    for k in 0..LIMBS_PER_PROD {
        decls.push(range_64(prod_limb(k),        &format!("prod_limb_{}_range", k)));
        decls.push(range_64(qp_limb(k),          &format!("qp_limb_{}_range", k)));
        // Mul schoolbook carries: 72-bit (covers the ~2^67 worst case of
        // full Fp · Fp products).
        decls.push(range_72(mul_carry_ab(k),     &format!("mul_carry_ab_{}_range", k)));
        decls.push(range_72(mul_carry_qp(k),     &format!("mul_carry_qp_{}_range", k)));
    }

    // Binary columns (range [0, 1]).
    for i in 0..LIMBS_PER_FP {
        decls.push(binary(add_carry(i),          &format!("add_carry_{}_bin", i)));
        decls.push(binary(add_sub_borrow(i),     &format!("add_sub_borrow_{}_bin", i)));
        decls.push(binary(sub_borrow(i),         &format!("sub_borrow_{}_bin", i)));
        decls.push(binary(sub_add_carry(i),      &format!("sub_add_carry_{}_bin", i)));
        decls.push(binary(slack_borrow(i),       &format!("slack_borrow_{}_bin", i)));
    }
    decls.push(binary(COL_ADD_REDUCE_FLAG,      "add_reduce_flag_bin"));
    decls.push(binary(COL_SUB_ADD_BACK_FLAG,    "sub_add_back_flag_bin"));
    for k in 0..LIMBS_PER_PROD {
        decls.push(binary(mul_sum_carry(k),     &format!("mul_sum_carry_{}_bin", k)));
    }
    decls.push(binary(COL_SEL_ADD, "sel_add_bin"));
    decls.push(binary(COL_SEL_SUB, "sel_sub_bin"));
    decls.push(binary(COL_SEL_MUL, "sel_mul_bin"));
    decls.push(binary(COL_SEL_INV, "sel_inv_bin"));

    decls
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;

    fn beta_challenge() -> Scalar {
        Scalar::from_u64(31337, CurveType::Bls48581)
    }

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    fn assert_all_zero(evals: &[ConstraintEval]) {
        for ce in evals {
            for (row, v) in ce.values.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint `{}` fired at row {} (valid witness)",
                    ce.label, row
                );
            }
        }
    }

    fn sample_fps() -> Vec<Fp> {
        vec![
            Fp::zero(),
            Fp::one(),
            Fp::from_u64(2),
            Fp::from_u64(0xffff_ffff_ffff_ffff),
            Fp { limbs: [
                0x0123456789abcdef,
                0xfedcba9876543210,
                0xdeadbeefcafebabe,
                0x0f0e0d0c0b0a0908,
                0x0706050403020100,
                0x8080808080808080,
            ]},
            // p - 1
            Fp { limbs: [
                P_LIMBS[0], P_LIMBS[1], P_LIMBS[2],
                P_LIMBS[3], P_LIMBS[4], P_LIMBS[5] - 1,
            ]},
        ]
    }

    #[test]
    fn nonnative_fp_air_column_layout_is_consistent() {
        // Selectors end exactly at the total count.
        assert_eq!(COL_SEL_ADD + 4, NUM_NONNATIVE_FP_COLUMNS);

        // Individual offsets land at the right places.
        assert_eq!(COL_A_OFFSET,               0);
        assert_eq!(COL_B_OFFSET,               6);
        assert_eq!(COL_R_OFFSET,               12);
        assert_eq!(COL_ADD_UNRED_SUM_OFFSET,   18);
        assert_eq!(COL_ADD_CARRIES_OFFSET,     24);
        assert_eq!(COL_ADD_REDUCE_FLAG,        30);
        assert_eq!(COL_ADD_TRIAL_DIFF_OFFSET,  31);
        assert_eq!(COL_ADD_SUB_BORROWS_OFFSET, 37);
        assert_eq!(COL_SUB_UNRED_DIFF_OFFSET,  43);
        assert_eq!(COL_SUB_BORROWS_OFFSET,     49);
        assert_eq!(COL_SUB_ADD_BACK_FLAG,      55);
        assert_eq!(COL_SUB_TRIAL_SUM_OFFSET,   56);
        assert_eq!(COL_SUB_ADD_CARRIES_OFFSET, 62);
        assert_eq!(COL_PROD_OFFSET,            68);
        assert_eq!(COL_QUOTIENT_OFFSET,        80);
        assert_eq!(COL_QP_OFFSET,              86);
        assert_eq!(COL_MUL_CARRY_AB_OFFSET,    98);
        assert_eq!(COL_MUL_CARRY_QP_OFFSET,   110);
        assert_eq!(COL_MUL_SUM_CARRY_OFFSET,  122);
        assert_eq!(COL_SLACK_OFFSET,          134);
        assert_eq!(COL_SLACK_BORROWS_OFFSET,  140);
        assert_eq!(NUM_DATA_COLUMNS,          146);
        assert_eq!(COL_SEL_ADD,               146);
        assert_eq!(COL_SEL_SUB,               147);
        assert_eq!(COL_SEL_MUL,               148);
        assert_eq!(COL_SEL_INV,               149);
        assert_eq!(NUM_NONNATIVE_FP_COLUMNS,  150);
    }

    #[test]
    fn nonnative_fp_air_add_vanishes_on_valid() {
        let samples = sample_fps();
        let ops: Vec<FpOp> = samples.iter().flat_map(|a| {
            samples.iter().map(move |b| FpOp::Add { a: *a, b: *b })
        }).collect();
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&ops, curve, None);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn nonnative_fp_air_sub_vanishes_on_valid() {
        let samples = sample_fps();
        let ops: Vec<FpOp> = samples.iter().flat_map(|a| {
            samples.iter().map(move |b| FpOp::Sub { a: *a, b: *b })
        }).collect();
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&ops, curve, None);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn nonnative_fp_air_mul_vanishes_on_valid() {
        let samples = sample_fps();
        let ops: Vec<FpOp> = samples.iter().flat_map(|a| {
            samples.iter().map(move |b| FpOp::Mul { a: *a, b: *b })
        }).collect();
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&ops, curve, None);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn nonnative_fp_air_mixed_ops_in_one_trace() {
        // Mix all three operations in one trace.
        let a = Fp::from_u64(7);
        let b = Fp::from_u64(11);
        let big = Fp { limbs: [
            0x0123456789abcdef,
            0xfedcba9876543210,
            0xdeadbeefcafebabe,
            0x0f0e0d0c0b0a0908,
            0x0706050403020100,
            0x0080808080808080,
        ]};
        let ops = vec![
            FpOp::Add { a: a, b: b },
            FpOp::Sub { a: a, b: b },
            FpOp::Mul { a: a, b: b },
            FpOp::Add { a: big, b: big },
            FpOp::Mul { a: big, b: b },
            FpOp::Sub { a: b, b: a },
        ];
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&ops, curve, None);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn nonnative_fp_air_constraints_reject_tampered_r_limb() {
        // Populate a valid mul trace, then corrupt a single r limb.
        let a = Fp::from_u64(123456789);
        let b = Fp::from_u64(987654321);
        let ops = vec![FpOp::Mul { a, b }];
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&ops, curve, None);

        // Tamper: flip r_limbs[5] (the LSB BE limb) by adding 1.
        let one = Scalar::one(curve);
        columns[r_limb(5)][0] = columns[r_limb(5)][0].add(&one);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);

        let fired = evals.iter().any(|ce| !ce.values[0].is_zero());
        assert!(fired, "tampered r_limb must trigger some constraint");
    }

    #[test]
    fn nonnative_fp_air_constraints_reject_tampered_q_limb() {
        // Populate a valid mul trace, then corrupt a quotient limb.
        let a = Fp { limbs: [
            0x0123456789abcdef,
            0xfedcba9876543210,
            0xdeadbeefcafebabe,
            0x0f0e0d0c0b0a0908,
            0x0706050403020100,
            0x0080808080808080,
        ]};
        let b = Fp::from_u64(999_999_999);
        let ops = vec![FpOp::Mul { a, b }];
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&ops, curve, None);

        // Tamper: flip q_limbs[5] (LSB BE).
        let one = Scalar::one(curve);
        columns[quotient_limb(5)][0] = columns[quotient_limb(5)][0].add(&one);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);

        let fired = evals.iter().any(|ce| !ce.values[0].is_zero());
        assert!(fired, "tampered q limb must trigger some constraint");
    }

    #[test]
    fn nonnative_fp_air_constraints_reject_flipped_selector() {
        // Set an extra selector — violates selector_sum_01 (sum == 2).
        let ops = vec![FpOp::Add { a: Fp::from_u64(1), b: Fp::from_u64(2) }];
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&ops, curve, None);

        let one = Scalar::one(curve);
        columns[COL_SEL_MUL][0] = one.clone();  // already have sel_add=1

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);

        let cat = evals.iter().find(|ce| ce.label == "selector_sum_01").unwrap();
        assert!(!cat.values[0].is_zero(),
                "double-selector must fire selector_sum_01");
    }

    #[test]
    fn nonnative_fp_air_mul_agrees_with_reference() {
        // Cross-check: the `r_limbs` columns of a mul row equal the reference
        // Fp::mul output.
        let test_cases = vec![
            (Fp::zero(),       Fp::zero()),
            (Fp::zero(),       Fp::from_u64(42)),
            (Fp::from_u64(1),  Fp::from_u64(1)),
            (Fp::from_u64(2),  Fp::from_u64(3)),
            (Fp::from_u64(0xffff_ffff_ffff_ffff),
             Fp::from_u64(0xffff_ffff_ffff_ffff)),
            // p - 1 times p - 1 = 1 (mod p).
            (Fp { limbs: [
                P_LIMBS[0], P_LIMBS[1], P_LIMBS[2],
                P_LIMBS[3], P_LIMBS[4], P_LIMBS[5] - 1,
            ]},
             Fp { limbs: [
                P_LIMBS[0], P_LIMBS[1], P_LIMBS[2],
                P_LIMBS[3], P_LIMBS[4], P_LIMBS[5] - 1,
            ]}),
            // Arbitrary large × small.
            (Fp { limbs: [
                0x0123456789abcdef,
                0xfedcba9876543210,
                0xdeadbeefcafebabe,
                0x0f0e0d0c0b0a0908,
                0x0706050403020100,
                0x0080808080808080,
            ]}, Fp::from_u64(0x100_0001)),
        ];

        let curve = CurveType::Bls48581;
        for (a, b) in test_cases {
            let ops = vec![FpOp::Mul { a, b }];
            let columns = populate_trace(&ops, curve, None);
            let expected = a.mul(&b);
            // Read back r_limbs at row 0.
            for i in 0..6 {
                let cell_u64 = columns[r_limb(i)][0].to_u64();
                assert_eq!(
                    cell_u64, expected.limbs[i],
                    "mul r_limb[{}] mismatch for a={:?} b={:?} expected={:?}",
                    i, a, b, expected
                );
            }

            // And the AIR constraints vanish.
            let beta = beta_challenge();
            let refs = col_refs(&columns);
            let evals = evaluate_constraints(&refs, &beta);
            for ce in &evals {
                for (row, v) in ce.values.iter().enumerate() {
                    assert!(
                        v.is_zero(),
                        "cross-check constraint `{}` fired at row {} for a={:?} b={:?}",
                        ce.label, row, a, b
                    );
                }
            }
        }
    }

    #[test]
    fn nonnative_fp_air_lookup_declarations_are_well_formed() {
        let decls = lookup_declarations();
        assert!(!decls.is_empty(), "expected non-empty lookup declarations");
        // Every declaration must reference a valid column index.
        for d in &decls {
            assert!(
                d.column_index < NUM_NONNATIVE_FP_COLUMNS,
                "declaration `{}` references out-of-range column {}",
                d.label, d.column_index
            );
            assert!(
                d.max_bits == 64 || d.max_bits == 72 || d.max_bits == 1,
                "unexpected max_bits {} for declaration `{}`",
                d.max_bits, d.label,
            );
        }
    }

    #[test]
    fn nonnative_fp_air_padding_row_is_valid() {
        // A trace with one real op and one padding row (all-zero, no selector).
        let a = Fp::from_u64(5);
        let b = Fp::from_u64(7);
        let ops = vec![FpOp::Add { a, b }];
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&ops, curve, Some(2)); // 2 rows, second is padding

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn nonnative_fp_air_inv_vanishes_on_valid() {
        // Every nonzero sample value must produce a satisfied Inv row.
        let samples = sample_fps();
        let ops: Vec<FpOp> = samples.iter()
            .filter(|a| !a.limbs.iter().all(|l| *l == 0))
            .map(|a| FpOp::Inv { a: *a })
            .collect();
        assert!(!ops.is_empty(), "need at least one nonzero sample");
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&ops, curve, None);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }

    #[test]
    fn nonnative_fp_air_inv_writes_b_equal_to_inverse() {
        // The b column on an Inv row must equal a^{-1} as computed by the
        // reference Fp::invert.
        let a = Fp::from_u64(0x1234_5678);
        let inv = a.invert().expect("nonzero");
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&[FpOp::Inv { a }], curve, None);

        for i in 0..LIMBS_PER_FP {
            let want = Scalar::from_u64(inv.limbs[i], curve);
            assert!(
                columns[b_limb(i)][0].sub(&want).is_zero(),
                "b limb {} must equal a^-1's limb",
                i,
            );
        }
        // r is the constant 1.
        for i in 0..LIMBS_PER_FP - 1 {
            assert!(columns[r_limb(i)][0].is_zero(), "r high limb {} must be 0", i);
        }
        assert!(
            columns[r_limb(LIMBS_PER_FP - 1)][0].sub(&Scalar::one(curve)).is_zero(),
            "r low limb must be 1",
        );
        // sel_inv is set, sel_mul is not.
        assert!(columns[COL_SEL_INV][0].sub(&Scalar::one(curve)).is_zero());
        assert!(columns[COL_SEL_MUL][0].is_zero());
    }

    #[test]
    fn nonnative_fp_air_inv_constraints_reject_wrong_witness() {
        // Populate a valid Inv trace, then corrupt the inverse witness in b.
        let a = Fp::from_u64(13);
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&[FpOp::Inv { a }], curve, None);

        // Tamper b's low limb so a · b ≠ 1 mod p.
        let one = Scalar::one(curve);
        columns[b_limb(5)][0] = columns[b_limb(5)][0].add(&one);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert!(
            evals.iter().any(|ce| !ce.values[0].is_zero()),
            "tampered inverse witness must trigger some constraint",
        );
    }

    #[test]
    fn nonnative_fp_air_inv_constraints_reject_nonone_r() {
        // Populate a valid Inv trace and tamper r so it's no longer 1.
        // The Inv-result-is-one constraint (CAT 19) must catch this even if
        // the prover also re-balances q to make the Mul body satisfied.
        let a = Fp::from_u64(17);
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&[FpOp::Inv { a }], curve, None);

        // Tamper: set r low limb to 2 instead of 1. (Mul body will fire too,
        // but CAT 19 is the dedicated Inv-only check.)
        let two = Scalar::from_u64(2, curve);
        columns[r_limb(LIMBS_PER_FP - 1)][0] = two;

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let inv_one = evals.iter()
            .find(|ce| ce.label == "inv_result_is_one")
            .expect("inv_result_is_one constraint must exist");
        assert!(!inv_one.values[0].is_zero(), "CAT 19 must fire for r ≠ 1 on Inv row");
    }

    #[test]
    fn nonnative_fp_air_inv_alongside_other_ops() {
        // Trace mixing Add/Sub/Mul/Inv must remain satisfied.
        let a = Fp::from_u64(7);
        let b = Fp::from_u64(11);
        let ops = vec![
            FpOp::Add { a, b },
            FpOp::Inv { a },
            FpOp::Mul { a, b },
            FpOp::Inv { a: b },
            FpOp::Sub { a, b },
        ];
        let curve = CurveType::Bls48581;
        let columns = populate_trace(&ops, curve, None);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        assert_all_zero(&evals);
    }
}
