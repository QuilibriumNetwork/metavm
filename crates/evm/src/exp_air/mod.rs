//! 256-bit EXP gadget AIR for EVM.
//!
//! Proves `result = base ^ exponent mod 2^256` via the standard left-to-right
//! square-and-multiply algorithm over the exponent's 256 bits. Each row records
//! one `square -> conditional multiply` step. The first row initializes
//! `result = 1`; the last row's `result_out` is the EXP opcode's output.
//!
//! ## Decoupling from the main EVM constraint module
//!
//! This module is intentionally **standalone** from
//! [`crate::constraints::EvmConstraintSystem`]. The main EVM constraint module
//! still treats EXP as an oracle (selector body identically zero — see
//! `crates/evm/src/constraints/mod.rs:389`). The cross-AIR linkage that would
//! bind the EVM trace's `(base, exponent, output)` columns at an EXP-active
//! row to the FIRST and LAST rows of a `EvmExpAir` instance is a separate
//! follow-up and is **not** implemented here.
//!
//! What this module does deliver, end-to-end and provably:
//!
//! - A 256-row witness layout for the square-and-multiply algorithm.
//! - Algebraic constraints (degree ≤ 2, all gated to vanish on padding rows)
//!   that pin every row to schoolbook 4-limb modular arithmetic.
//! - A cross-row transition that chains row `r`'s computed result into row
//!   `r+1`'s starting result.
//! - Counter / base / exponent-bit invariants enforced row-locally.
//!
//! ## Algorithm
//!
//! ```text
//! result = 1
//! for bit in (255..=0).rev():        // r = 0..255, bit_index = 255 - r
//!     squared = result * result mod 2^256
//!     mul     = squared * base       mod 2^256
//!     result  = if exp_bit == 1 { mul } else { squared }
//! ```
//!
//! ## Per-row layout (data columns, one row per algorithm step)
//!
//! ```text
//! offset       size  meaning
//! 0..4         4     result_in            current accumulated result (4 LE u64 limbs)
//! 4..8         4     base                 base (constant across all rows)
//! 8..12        4     squared              = result_in * result_in mod 2^256
//! 12..16       4     squared_carry        carry chain for the squaring multiply
//! 16..20       4     mul                  = squared * base mod 2^256
//! 20..24       4     mul_carry            carry chain for the squared*base multiply
//! 24..25       1     exp_bit              current bit of exponent (0 or 1)
//! 25..26       1     bit_index            decremented counter (255 down to 0)
//! 26..27       1     active               1 on real rows, 0 on padding
//! ```
//!
//! Total: **27 data columns**, no auxiliary selectors — the `active` column
//! plays the role of a selector that gates every row-local body so padding
//! rows trivially satisfy all constraints.
//!
//! ## Constraint catalog
//!
//! Row-local (each gated by `active`):
//!
//!  1. `exp_bit_binary`         — `exp_bit · (exp_bit − 1) = 0`
//!  2. `active_binary`          — `active · (active − 1) = 0`
//!  3. `squared_limb0..3`       — schoolbook 4-limb MUL with carry chain:
//!     `result_in × result_in = squared (mod 2^256)`
//!  4. `mul_limb0..3`           — schoolbook 4-limb MUL with carry chain:
//!     `squared × base = mul (mod 2^256)`
//!
//! Cross-row (multiplied by `(X − ω^{n−1})` to skip wrap-around):
//!
//!  5. `result_chain_limb0..3`  — `result_in(ω·X) = exp_bit · mul + (1 − exp_bit) · squared`
//!  6. `base_invariant_limb0..3` — `base(ω·X) = base(X)`
//!  7. `bit_index_decrement`    — `bit_index(ω·X) = bit_index(X) − active(ω·X)`
//!
//! The `active`-gated boundary between real and padding rows is handled by
//! `bit_index_decrement` cleanly: on the last real → first padding step the
//! delta is `0 − 0 = 0` because `active(ω·X) = 0` and `bit_index = 0` at the
//! end of the algorithm. (For traces with no padding — the natural 256-row
//! case — every `active` is 1 and the constraint is `bit_index(ω·X) =
//! bit_index(X) − 1`.)

use metavm_zkp::field::{CurveType, Scalar};

use revm::primitives::U256;

// ──── Column offsets ─────────────────────────────────────────────────────

pub const NUM_LIMBS: usize = 4;

pub const COL_RESULT_IN_OFFSET: usize = 0;
pub const COL_BASE_OFFSET: usize = COL_RESULT_IN_OFFSET + NUM_LIMBS; // 4
pub const COL_SQUARED_OFFSET: usize = COL_BASE_OFFSET + NUM_LIMBS; // 8
pub const COL_SQUARED_CARRY_OFFSET: usize = COL_SQUARED_OFFSET + NUM_LIMBS; // 12
pub const COL_MUL_OFFSET: usize = COL_SQUARED_CARRY_OFFSET + NUM_LIMBS; // 16
pub const COL_MUL_CARRY_OFFSET: usize = COL_MUL_OFFSET + NUM_LIMBS; // 20

pub const COL_EXP_BIT: usize = COL_MUL_CARRY_OFFSET + NUM_LIMBS; // 24
pub const COL_BIT_INDEX: usize = COL_EXP_BIT + 1; // 25
pub const COL_ACTIVE: usize = COL_BIT_INDEX + 1; // 26

// ──── Cross-AIR linkage columns (added for #94 EVM-EXP linkage) ────────
//
// `EXPONENT` and `FINAL_OUTPUT` hold the full exponent and the final
// `base^exponent` value of the EXP invocation. Both are constant
// across all 256 rows of an invocation (enforced by the shifted
// `_invariant` constraints). These columns let the cross-AIR LogUp
// build a tuple `(base, exponent, final_output)` exposed on the
// "anchor row" of each invocation (where `IS_FIRST_ROW = 1`),
// matching against the EVM main trace's `(INPUT0, INPUT1, OUTPUT0)`
// tuple at SEL_EXP rows.
pub const COL_EXPONENT_OFFSET: usize = COL_ACTIVE + 1; // 27
pub const COL_FINAL_OUTPUT_OFFSET: usize = COL_EXPONENT_OFFSET + NUM_LIMBS; // 31

/// Selector that's `1` only on the first row of each EXP invocation
/// (where `bit_index = NUM_STEPS − 1 = 255`). On padding rows this
/// is 0. The cross-AIR LogUp uses this as the source-side selector
/// so exactly one tuple is exposed per invocation.
pub const COL_IS_FIRST_ROW: usize = COL_FINAL_OUTPUT_OFFSET + NUM_LIMBS; // 35

pub const NUM_EXP_AIR_COLUMNS: usize = COL_IS_FIRST_ROW + 1; // 36

/// Number of real algorithm steps (always 256: one per exponent bit).
pub const NUM_STEPS: usize = 256;

// ──── Column index helpers ───────────────────────────────────────────────

#[inline]
pub fn result_in_limb(i: usize) -> usize {
    debug_assert!(i < NUM_LIMBS);
    COL_RESULT_IN_OFFSET + i
}

#[inline]
pub fn base_limb(i: usize) -> usize {
    debug_assert!(i < NUM_LIMBS);
    COL_BASE_OFFSET + i
}

#[inline]
pub fn squared_limb(i: usize) -> usize {
    debug_assert!(i < NUM_LIMBS);
    COL_SQUARED_OFFSET + i
}

#[inline]
pub fn squared_carry_limb(i: usize) -> usize {
    debug_assert!(i < NUM_LIMBS);
    COL_SQUARED_CARRY_OFFSET + i
}

#[inline]
pub fn mul_limb(i: usize) -> usize {
    debug_assert!(i < NUM_LIMBS);
    COL_MUL_OFFSET + i
}

#[inline]
pub fn mul_carry_limb(i: usize) -> usize {
    debug_assert!(i < NUM_LIMBS);
    COL_MUL_CARRY_OFFSET + i
}

// ──── Witness ────────────────────────────────────────────────────────────

/// One row of the EXP square-and-multiply trace.
///
/// All 4-limb arrays are little-endian: `arr[0]` is the least significant
/// 64 bits. Carry arrays hold the inter-limb carry chain produced by the
/// schoolbook multiply that collapses an 8-limb product down to a 4-limb
/// `mod 2^256` result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpRow {
    /// Result entering this row (= result after step `r-1`, with row 0 = 1).
    pub result_in: [u64; 4],
    /// Base of the exponentiation (constant across all rows).
    pub base: [u64; 4],
    /// `result_in * result_in mod 2^256`.
    pub squared: [u64; 4],
    /// Per-limb carry chain for the squaring multiply (witnesses the
    /// schoolbook collapse from 8 limbs back down to 4). Each entry is the
    /// 64-bit carry-out of the corresponding limb-sum.
    pub squared_carry: [u64; 4],
    /// `squared * base mod 2^256`.
    pub mul: [u64; 4],
    /// Per-limb carry chain for the `squared * base` multiply.
    pub mul_carry: [u64; 4],
    /// Current bit of the exponent: `(exponent >> bit_index) & 1`.
    pub exp_bit: u8,
    /// Decremented counter: 255 at row 0, 0 at row 255.
    pub bit_index: u8,
}

/// Compute the 4-limb little-endian schoolbook product `a * b mod 2^256`
/// alongside the per-limb carry chain. Returns `(out, carry)` where
/// `out[k] = limb_k_of(a*b)` and `carry[k] = (limb_k_partial_sum >> 64)`.
///
/// Mirrors the EVM main-trace MUL witness shape, so the same algebraic
/// constraint (`Σ aᵢ·bⱼ + carry_in = carry_out·2^64 + out`) applies here.
pub fn schoolbook_mul_mod_2_256(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    // Use u128 accumulators to capture each limb's full 128-bit partial sum.
    let mut out = [0u64; 4];
    let mut carry = [0u64; 4];

    // limb 0: a0*b0 = c0*2^64 + out0
    let s0: u128 = (a[0] as u128) * (b[0] as u128);
    out[0] = s0 as u64;
    carry[0] = (s0 >> 64) as u64;

    // limb 1: a0*b1 + a1*b0 + c0 = c1*2^64 + out1
    let s1: u128 = (a[0] as u128) * (b[1] as u128)
        + (a[1] as u128) * (b[0] as u128)
        + (carry[0] as u128);
    out[1] = s1 as u64;
    carry[1] = (s1 >> 64) as u64;

    // limb 2: a0*b2 + a1*b1 + a2*b0 + c1 = c2*2^64 + out2
    let s2: u128 = (a[0] as u128) * (b[2] as u128)
        + (a[1] as u128) * (b[1] as u128)
        + (a[2] as u128) * (b[0] as u128)
        + (carry[1] as u128);
    out[2] = s2 as u64;
    carry[2] = (s2 >> 64) as u64;

    // limb 3: a0*b3 + a1*b2 + a2*b1 + a3*b0 + c2 = c3*2^64 + out3
    let s3: u128 = (a[0] as u128) * (b[3] as u128)
        + (a[1] as u128) * (b[2] as u128)
        + (a[2] as u128) * (b[1] as u128)
        + (a[3] as u128) * (b[0] as u128)
        + (carry[2] as u128);
    out[3] = s3 as u64;
    carry[3] = (s3 >> 64) as u64;

    (out, carry)
}

/// Convert a `revm::U256` into 4 little-endian u64 limbs.
fn u256_limbs(u: &U256) -> [u64; 4] {
    *u.as_limbs()
}

/// Generate the 256-row witness for `base ^ exponent mod 2^256`.
///
/// The witness is laid out row-by-row in algorithm order: row 0 inspects
/// the high bit of the exponent, row 255 inspects the low bit.
pub fn exp_witness(base: U256, exponent: U256) -> Vec<ExpRow> {
    let base_limbs = u256_limbs(&base);
    let mut result: [u64; 4] = [1, 0, 0, 0];
    let mut rows = Vec::with_capacity(NUM_STEPS);

    for r in 0..NUM_STEPS {
        let bit_index = (NUM_STEPS - 1 - r) as u8;
        let exp_bit = ((exponent >> bit_index as usize) & U256::from(1u64))
            .as_limbs()[0] as u8;

        let result_in = result;
        let (squared, squared_carry) = schoolbook_mul_mod_2_256(result_in, result_in);
        let (mul, mul_carry) = schoolbook_mul_mod_2_256(squared, base_limbs);

        let result_out: [u64; 4] = if exp_bit == 1 { mul } else { squared };

        rows.push(ExpRow {
            result_in,
            base: base_limbs,
            squared,
            squared_carry,
            mul,
            mul_carry,
            exp_bit,
            bit_index,
        });

        result = result_out;
    }

    rows
}

/// Reference implementation of `base ^ exponent mod 2^256`, useful for
/// cross-checking witness output without the AIR machinery.
pub fn pow_mod_2_256(base: U256, exponent: U256) -> [u64; 4] {
    let mut result: [u64; 4] = [1, 0, 0, 0];
    let base_limbs = u256_limbs(&base);
    for r in 0..NUM_STEPS {
        let bit_index = (NUM_STEPS - 1 - r) as u8;
        let exp_bit = ((exponent >> bit_index as usize) & U256::from(1u64))
            .as_limbs()[0] as u8;
        let (squared, _) = schoolbook_mul_mod_2_256(result, result);
        if exp_bit == 1 {
            let (mul, _) = schoolbook_mul_mod_2_256(squared, base_limbs);
            result = mul;
        } else {
            result = squared;
        }
    }
    result
}

/// Allocate a column set of `num_rows` zero scalars per column.
pub fn alloc_trace(num_rows: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..NUM_EXP_AIR_COLUMNS)
        .map(|_| vec![zero.clone(); num_rows])
        .collect()
}

/// Populate row `row` of `columns` from an [`ExpRow`].
pub fn populate_row(
    columns: &mut [Vec<Scalar>],
    row: usize,
    er: &ExpRow,
    curve: CurveType,
) {
    assert_eq!(columns.len(), NUM_EXP_AIR_COLUMNS, "wrong column shape");
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    for i in 0..NUM_LIMBS {
        columns[result_in_limb(i)][row] = Scalar::from_u64(er.result_in[i], curve);
        columns[base_limb(i)][row] = Scalar::from_u64(er.base[i], curve);
        columns[squared_limb(i)][row] = Scalar::from_u64(er.squared[i], curve);
        columns[squared_carry_limb(i)][row] =
            Scalar::from_u64(er.squared_carry[i], curve);
        columns[mul_limb(i)][row] = Scalar::from_u64(er.mul[i], curve);
        columns[mul_carry_limb(i)][row] = Scalar::from_u64(er.mul_carry[i], curve);
    }

    columns[COL_EXP_BIT][row] = if er.exp_bit == 1 { one.clone() } else { zero.clone() };
    columns[COL_BIT_INDEX][row] = Scalar::from_u64(er.bit_index as u64, curve);
    columns[COL_ACTIVE][row] = one;
}

/// Populate every row of a freshly allocated 256-row trace from `rows`.
///
/// `rows` must have exactly [`NUM_STEPS`] entries. Returns the column set.
///
/// In addition to per-row population, this fills in the cross-AIR
/// linkage columns:
/// - `EXPONENT`: reconstructed from `exp_bit` values
///   (`Σ_k exp_bit[k] · 2^(NUM_STEPS−1−k)`) and replicated across all
///   256 rows.
/// - `FINAL_OUTPUT`: the EXP result (`base^exponent mod 2^256`).
///   Computed as the result-out of the last row (`exp_bit = 1 ⇒ mul`,
///   else `squared`) and replicated across all 256 rows.
/// - `IS_FIRST_ROW`: 1 on row 0 only, 0 elsewhere.
pub fn populate_trace(rows: &[ExpRow], curve: CurveType) -> Vec<Vec<Scalar>> {
    assert_eq!(rows.len(), NUM_STEPS, "expected {} rows", NUM_STEPS);
    let mut columns = alloc_trace(NUM_STEPS, curve);
    for (r, er) in rows.iter().enumerate() {
        populate_row(&mut columns, r, er, curve);
    }

    // Reconstruct the full exponent from the per-row exp_bit values.
    // exp_bit[r] is the bit at position `NUM_STEPS - 1 - r` (row 0 = high
    // bit, row 255 = low bit).
    let mut exponent_limbs: [u64; 4] = [0; 4];
    for (r, er) in rows.iter().enumerate() {
        if er.exp_bit == 1 {
            let bit_pos = NUM_STEPS - 1 - r;
            let limb = bit_pos / 64;
            let off = bit_pos % 64;
            exponent_limbs[limb] |= 1u64 << off;
        }
    }

    // FINAL_OUTPUT is the algorithm's result after the last row. Per
    // exp_witness's recurrence, after row 255 `result = if exp_bit==1
    // {mul} else {squared}`, so the final output is the LAST row's
    // result_out.
    let last = &rows[NUM_STEPS - 1];
    let final_output: [u64; 4] = if last.exp_bit == 1 { last.mul } else { last.squared };

    // Replicate across all 256 rows so the cross-AIR linkage tuple is
    // accessible at the anchor (first) row, and a constancy constraint
    // can pin the column shape end-to-end.
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    for r in 0..NUM_STEPS {
        for i in 0..NUM_LIMBS {
            columns[COL_EXPONENT_OFFSET + i][r] = Scalar::from_u64(exponent_limbs[i], curve);
            columns[COL_FINAL_OUTPUT_OFFSET + i][r] = Scalar::from_u64(final_output[i], curve);
        }
        columns[COL_IS_FIRST_ROW][r] = if r == 0 { one.clone() } else { zero.clone() };
    }

    columns
}

/// **Multi-invocation** populate: builds N×NUM_STEPS rows from `invocations`,
/// one block of `NUM_STEPS` rows per `(base, exponent)` pair. Closes #94's
/// "production EVM use" gap by letting a single EXP-gadget AIR layer prove
/// multiple EVM EXP opcodes at once.
///
/// Layout:
///   - rows `k·NUM_STEPS .. (k+1)·NUM_STEPS` carry invocation `k`'s
///     square-and-multiply trace (identical structure to single-invocation
///     populate_trace).
///   - `IS_FIRST_ROW` is set on rows `0, NUM_STEPS, 2·NUM_STEPS, ...` —
///     i.e. the first row of each invocation.
///   - `EXPONENT` and `FINAL_OUTPUT` are replicated WITHIN each invocation
///     block but DIFFER between blocks.
///
/// The cross-row constraints (gated by `(1 − IS_FIRST_ROW(ω·X))` since
/// Phase 1 of #94) automatically vanish on transitions between
/// invocations, so internal continuity violations don't fire. Each
/// invocation is independently `final_output_at_last_row`-pinned via
/// the `(active · (is_first_shift + (1 − act_shift)))` indicator.
pub fn populate_multi_trace(
    invocations: &[(U256, U256)],
    curve: CurveType,
) -> Vec<Vec<Scalar>> {
    let n = invocations.len();
    assert!(n > 0, "populate_multi_trace requires at least one invocation");
    let total_rows = n * NUM_STEPS;
    let mut columns = alloc_trace(total_rows, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    for (k, &(base, exponent)) in invocations.iter().enumerate() {
        let rows = exp_witness(base, exponent);
        debug_assert_eq!(rows.len(), NUM_STEPS);

        for (r, er) in rows.iter().enumerate() {
            let row_idx = k * NUM_STEPS + r;
            populate_row(&mut columns, row_idx, er, curve);
        }

        // Per-invocation EXPONENT and FINAL_OUTPUT replicated in this
        // block of NUM_STEPS rows.
        let mut exponent_limbs: [u64; 4] = [0; 4];
        for (r, er) in rows.iter().enumerate() {
            if er.exp_bit == 1 {
                let bit_pos = NUM_STEPS - 1 - r;
                let limb = bit_pos / 64;
                let off = bit_pos % 64;
                exponent_limbs[limb] |= 1u64 << off;
            }
        }
        let last = &rows[NUM_STEPS - 1];
        let final_output: [u64; 4] = if last.exp_bit == 1 {
            last.mul
        } else {
            last.squared
        };
        let block_start = k * NUM_STEPS;
        for r in 0..NUM_STEPS {
            let row_idx = block_start + r;
            for i in 0..NUM_LIMBS {
                columns[COL_EXPONENT_OFFSET + i][row_idx] =
                    Scalar::from_u64(exponent_limbs[i], curve);
                columns[COL_FINAL_OUTPUT_OFFSET + i][row_idx] =
                    Scalar::from_u64(final_output[i], curve);
            }
            // IS_FIRST_ROW on the first row of THIS invocation only.
            columns[COL_IS_FIRST_ROW][row_idx] =
                if r == 0 { one.clone() } else { zero.clone() };
        }
    }

    columns
}

/// Pre-computed `2^64` as a curve-typed scalar.
#[inline]
pub(crate) fn two_pow_64(curve: CurveType) -> Scalar {
    // 2^64 = 1<<63 + 1<<63
    let half = Scalar::from_u64(1u64 << 63, curve);
    half.add(&half)
}

// ──── Constraint evaluation (per-row scalar evaluator) ────────────────

/// One named constraint and its per-row evaluation values.
pub struct ConstraintEval {
    pub label: String,
    pub values: Vec<Scalar>,
}

/// Evaluate every row-local AIR body on `columns` with a Fiat-Shamir
/// challenge `beta` (used to RLC the four limb-level sub-bodies of each
/// MUL into a single scalar per row).
///
/// Returns one [`ConstraintEval`] per consolidated category. A valid
/// witness produces zero in every cell of every category.
///
/// Categories returned (in order):
///   0. `exp_bit_binary`
///   1. `active_binary`
///   2. `squared_mul_chain`   (β-RLC over 4 squared-MUL limbs)
///   3. `mul_mul_chain`       (β-RLC over 4 squared*base limbs)
pub fn evaluate_row_local(
    columns: &[&Vec<Scalar>],
    beta: &Scalar,
) -> Vec<ConstraintEval> {
    assert_eq!(columns.len(), NUM_EXP_AIR_COLUMNS);
    let num_rows = columns[0].len();
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let two_64 = two_pow_64(curve);

    let mut out: Vec<ConstraintEval> = Vec::with_capacity(4);

    // 1. exp_bit_binary: active · exp_bit · (exp_bit − 1) = 0.
    let mut bin_eb = vec![zero.clone(); num_rows];
    for r in 0..num_rows {
        let active = &columns[COL_ACTIVE][r];
        let eb = &columns[COL_EXP_BIT][r];
        bin_eb[r] = active.mul(&eb.mul(&eb.sub(&one)));
    }
    out.push(ConstraintEval { label: "exp_bit_binary".into(), values: bin_eb });

    // 2. active_binary: active · (active − 1) = 0  (ungated; structural).
    let mut bin_act = vec![zero.clone(); num_rows];
    for r in 0..num_rows {
        let active = &columns[COL_ACTIVE][r];
        bin_act[r] = active.mul(&active.sub(&one));
    }
    out.push(ConstraintEval { label: "active_binary".into(), values: bin_act });

    // 3. squared_mul_chain: 4 limb-level bodies aggregated via β powers,
    //    each gated by `active` so padding rows trivially vanish.
    //
    //    limb k: a*a accumulator at limb k − carry_k·2^64 − squared_k = 0
    let mut sq_chain = vec![zero.clone(); num_rows];
    for r in 0..num_rows {
        let active = &columns[COL_ACTIVE][r];
        let a0 = &columns[result_in_limb(0)][r];
        let a1 = &columns[result_in_limb(1)][r];
        let a2 = &columns[result_in_limb(2)][r];
        let a3 = &columns[result_in_limb(3)][r];
        let s0 = &columns[squared_limb(0)][r];
        let s1 = &columns[squared_limb(1)][r];
        let s2 = &columns[squared_limb(2)][r];
        let s3 = &columns[squared_limb(3)][r];
        let c0 = &columns[squared_carry_limb(0)][r];
        let c1 = &columns[squared_carry_limb(1)][r];
        let c2 = &columns[squared_carry_limb(2)][r];
        let c3 = &columns[squared_carry_limb(3)][r];

        // Schoolbook squaring `a * a`:
        //   limb 0: a0·a0
        //   limb 1: a0·a1 + a1·a0          (= 2·a0·a1)
        //   limb 2: a0·a2 + a1·a1 + a2·a0  (= 2·a0·a2 + a1·a1)
        //   limb 3: a0·a3 + a1·a2 + a2·a1 + a3·a0
        let bodies = [
            mul_limb_body(a0, a0, &zero, &zero, &zero, &zero, &zero, &zero, &zero, s0, c0, &two_64, 0),
            mul_limb_body(a0, a1, a1, a0, &zero, &zero, &zero, &zero, c0, s1, c1, &two_64, 1),
            mul_limb_body(a0, a2, a1, a1, a2, a0, &zero, &zero, c1, s2, c2, &two_64, 2),
            mul_limb_body(a0, a3, a1, a2, a2, a1, a3, a0, c2, s3, c3, &two_64, 3),
        ];
        // β-RLC the four limb bodies.
        let mut acc = zero.clone();
        let mut bp = one.clone();
        for body in &bodies {
            acc = acc.add(&bp.mul(body));
            bp = bp.mul(beta);
        }
        sq_chain[r] = active.mul(&acc);
    }
    out.push(ConstraintEval { label: "squared_mul_chain".into(), values: sq_chain });

    // 4. mul_mul_chain: squared * base = mul (mod 2^256), 4 limb bodies.
    let mut sm_chain = vec![zero.clone(); num_rows];
    for r in 0..num_rows {
        let active = &columns[COL_ACTIVE][r];
        let s0 = &columns[squared_limb(0)][r];
        let s1 = &columns[squared_limb(1)][r];
        let s2 = &columns[squared_limb(2)][r];
        let s3 = &columns[squared_limb(3)][r];
        let b0 = &columns[base_limb(0)][r];
        let b1 = &columns[base_limb(1)][r];
        let b2 = &columns[base_limb(2)][r];
        let b3 = &columns[base_limb(3)][r];
        let m0 = &columns[mul_limb(0)][r];
        let m1 = &columns[mul_limb(1)][r];
        let m2 = &columns[mul_limb(2)][r];
        let m3 = &columns[mul_limb(3)][r];
        let c0 = &columns[mul_carry_limb(0)][r];
        let c1 = &columns[mul_carry_limb(1)][r];
        let c2 = &columns[mul_carry_limb(2)][r];
        let c3 = &columns[mul_carry_limb(3)][r];

        let bodies = [
            mul_limb_body(s0, b0, &zero, &zero, &zero, &zero, &zero, &zero, &zero, m0, c0, &two_64, 0),
            mul_limb_body(s0, b1, s1, b0, &zero, &zero, &zero, &zero, c0, m1, c1, &two_64, 1),
            mul_limb_body(s0, b2, s1, b1, s2, b0, &zero, &zero, c1, m2, c2, &two_64, 2),
            mul_limb_body(s0, b3, s1, b2, s2, b1, s3, b0, c2, m3, c3, &two_64, 3),
        ];
        let mut acc = zero.clone();
        let mut bp = one.clone();
        for body in &bodies {
            acc = acc.add(&bp.mul(body));
            bp = bp.mul(beta);
        }
        sm_chain[r] = active.mul(&acc);
    }
    out.push(ConstraintEval { label: "mul_mul_chain".into(), values: sm_chain });

    // 5. is_first_row_binary: IS_FIRST_ROW · (IS_FIRST_ROW − 1) = 0.
    let mut bin_first = vec![zero.clone(); num_rows];
    for r in 0..num_rows {
        let v = &columns[COL_IS_FIRST_ROW][r];
        bin_first[r] = v.mul(&v.sub(&one));
    }
    out.push(ConstraintEval { label: "is_first_row_binary".into(), values: bin_first });

    // 6. is_first_row_pins_bit_index: IS_FIRST_ROW · (bit_index − (NUM_STEPS − 1)) = 0.
    let last_idx = Scalar::from_u64((NUM_STEPS - 1) as u64, curve);
    let mut pin_first = vec![zero.clone(); num_rows];
    for r in 0..num_rows {
        let v = &columns[COL_IS_FIRST_ROW][r];
        let bi = &columns[COL_BIT_INDEX][r];
        pin_first[r] = v.mul(&bi.sub(&last_idx));
    }
    out.push(ConstraintEval { label: "is_first_row_pins_bit_index".into(), values: pin_first });

    // 7. is_first_row_pins_result_in (#94 Phase 3 — closes the residual
    //    soundness gap left by Phase 1's boundary gating). β-RLC over
    //    4 limb bodies, gated by `IS_FIRST_ROW`:
    //      limb 0: result_in[0] − 1
    //      limb k>0: result_in[k]
    //    Forces every invocation's first row to start with
    //    `result_in = (1, 0, 0, 0) = 1`, which is the algorithm's
    //    initializer. Without this, a malicious prover could supply
    //    arbitrary result_in at any invocation start because the
    //    cross-row `result_chain` is suppressed at boundaries.
    let mut pin_init = vec![zero.clone(); num_rows];
    for r in 0..num_rows {
        let v = &columns[COL_IS_FIRST_ROW][r];
        let mut acc = zero.clone();
        let mut bp = one.clone();
        for i in 0..NUM_LIMBS {
            let ri = &columns[result_in_limb(i)][r];
            let body = if i == 0 { ri.sub(&one) } else { ri.clone() };
            acc = acc.add(&bp.mul(&body));
            bp = bp.mul(beta);
        }
        pin_init[r] = v.mul(&acc);
    }
    out.push(ConstraintEval {
        label: "is_first_row_pins_result_in".into(),
        values: pin_init,
    });

    out
}

/// Schoolbook MUL limb body for limb `k`.
///
/// Computes the constraint
///   `Σ aᵢ·bⱼ (i+j=k) + carry_in − carry_out·2^64 − out_k = 0`.
///
/// Pass `&zero` for cross-product slots that do not exist for this limb
/// (e.g. limb 0 only has `a0·b0`). `carry_in` is `&zero` for limb 0.
#[allow(clippy::too_many_arguments)]
fn mul_limb_body(
    a_i0: &Scalar,
    b_j0: &Scalar,
    a_i1: &Scalar,
    b_j1: &Scalar,
    a_i2: &Scalar,
    b_j2: &Scalar,
    a_i3: &Scalar,
    b_j3: &Scalar,
    carry_in: &Scalar,
    out_k: &Scalar,
    carry_out: &Scalar,
    two_64: &Scalar,
    limb: usize,
) -> Scalar {
    // limb k's product term count
    let mut sum = a_i0.mul(b_j0);
    if limb >= 1 {
        sum = sum.add(&a_i1.mul(b_j1));
    }
    if limb >= 2 {
        sum = sum.add(&a_i2.mul(b_j2));
    }
    if limb >= 3 {
        sum = sum.add(&a_i3.mul(b_j3));
    }
    sum = sum.add(carry_in);
    let hi = carry_out.mul(two_64);
    sum.sub(&hi).sub(out_k)
}

/// Evaluate row-local bodies at a single point (`evaluate_at_point` style).
/// Returns 7 scalar values, one per category, in the same order as
/// [`evaluate_row_local`]:
///   0. `exp_bit_binary`
///   1. `active_binary`
///   2. `squared_mul_chain`
///   3. `mul_mul_chain`
///   4. `is_first_row_binary`
///   5. `is_first_row_pins_bit_index`
///   6. `is_first_row_pins_result_in`  (#94 Phase 3)
///
/// `col_evals_at_z` is the value of every column at the challenge point z.
pub fn evaluate_row_local_at_point(col_evals_at_z: &[Scalar], beta: &Scalar) -> [Scalar; 7] {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let two_64 = two_pow_64(curve);

    let active = &col_evals_at_z[COL_ACTIVE];
    let eb = &col_evals_at_z[COL_EXP_BIT];

    // 1. exp_bit_binary
    let body_eb = active.mul(&eb.mul(&eb.sub(&one)));
    // 2. active_binary
    let body_act = active.mul(&active.sub(&one));

    // 3. squared MUL chain
    let a = [
        col_evals_at_z[result_in_limb(0)].clone(),
        col_evals_at_z[result_in_limb(1)].clone(),
        col_evals_at_z[result_in_limb(2)].clone(),
        col_evals_at_z[result_in_limb(3)].clone(),
    ];
    let s = [
        col_evals_at_z[squared_limb(0)].clone(),
        col_evals_at_z[squared_limb(1)].clone(),
        col_evals_at_z[squared_limb(2)].clone(),
        col_evals_at_z[squared_limb(3)].clone(),
    ];
    let cs = [
        col_evals_at_z[squared_carry_limb(0)].clone(),
        col_evals_at_z[squared_carry_limb(1)].clone(),
        col_evals_at_z[squared_carry_limb(2)].clone(),
        col_evals_at_z[squared_carry_limb(3)].clone(),
    ];
    // Schoolbook squaring `a * a` — see `evaluate_row_local`.
    let bodies_sq = [
        mul_limb_body(&a[0], &a[0], &zero, &zero, &zero, &zero, &zero, &zero, &zero, &s[0], &cs[0], &two_64, 0),
        mul_limb_body(&a[0], &a[1], &a[1], &a[0], &zero, &zero, &zero, &zero, &cs[0], &s[1], &cs[1], &two_64, 1),
        mul_limb_body(&a[0], &a[2], &a[1], &a[1], &a[2], &a[0], &zero, &zero, &cs[1], &s[2], &cs[2], &two_64, 2),
        mul_limb_body(&a[0], &a[3], &a[1], &a[2], &a[2], &a[1], &a[3], &a[0], &cs[2], &s[3], &cs[3], &two_64, 3),
    ];
    let mut acc_sq = zero.clone();
    let mut bp = one.clone();
    for body in &bodies_sq {
        acc_sq = acc_sq.add(&bp.mul(body));
        bp = bp.mul(beta);
    }
    let body_sq = active.mul(&acc_sq);

    // 4. mul (squared * base) MUL chain
    let b = [
        col_evals_at_z[base_limb(0)].clone(),
        col_evals_at_z[base_limb(1)].clone(),
        col_evals_at_z[base_limb(2)].clone(),
        col_evals_at_z[base_limb(3)].clone(),
    ];
    let m = [
        col_evals_at_z[mul_limb(0)].clone(),
        col_evals_at_z[mul_limb(1)].clone(),
        col_evals_at_z[mul_limb(2)].clone(),
        col_evals_at_z[mul_limb(3)].clone(),
    ];
    let cm = [
        col_evals_at_z[mul_carry_limb(0)].clone(),
        col_evals_at_z[mul_carry_limb(1)].clone(),
        col_evals_at_z[mul_carry_limb(2)].clone(),
        col_evals_at_z[mul_carry_limb(3)].clone(),
    ];
    let bodies_m = [
        mul_limb_body(&s[0], &b[0], &zero, &zero, &zero, &zero, &zero, &zero, &zero, &m[0], &cm[0], &two_64, 0),
        mul_limb_body(&s[0], &b[1], &s[1], &b[0], &zero, &zero, &zero, &zero, &cm[0], &m[1], &cm[1], &two_64, 1),
        mul_limb_body(&s[0], &b[2], &s[1], &b[1], &s[2], &b[0], &zero, &zero, &cm[1], &m[2], &cm[2], &two_64, 2),
        mul_limb_body(&s[0], &b[3], &s[1], &b[2], &s[2], &b[1], &s[3], &b[0], &cm[2], &m[3], &cm[3], &two_64, 3),
    ];
    let mut acc_m = zero.clone();
    let mut bp = one.clone();
    for body in &bodies_m {
        acc_m = acc_m.add(&bp.mul(body));
        bp = bp.mul(beta);
    }
    let body_m = active.mul(&acc_m);

    // 5. is_first_row_binary: IS_FIRST_ROW · (IS_FIRST_ROW − 1) = 0
    let is_first = &col_evals_at_z[COL_IS_FIRST_ROW];
    let body_first_bin = is_first.mul(&is_first.sub(&one));

    // 6. is_first_row_pins_bit_index: IS_FIRST_ROW · (bit_index − (NUM_STEPS − 1)) = 0
    let bit_idx = &col_evals_at_z[COL_BIT_INDEX];
    let last_idx = Scalar::from_u64((NUM_STEPS - 1) as u64, curve);
    let body_first_pin = is_first.mul(&bit_idx.sub(&last_idx));

    // 7. is_first_row_pins_result_in (#94 Phase 3):
    //    IS_FIRST_ROW · β-RLC_i (result_in[i] − initializer[i]) = 0,
    //    initializer = (1, 0, 0, 0).
    let mut init_acc = zero.clone();
    let mut bp = one.clone();
    for i in 0..NUM_LIMBS {
        let ri = &col_evals_at_z[result_in_limb(i)];
        let body = if i == 0 { ri.sub(&one) } else { ri.clone() };
        init_acc = init_acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    let body_first_init = is_first.mul(&init_acc);

    [body_eb, body_act, body_sq, body_m, body_first_bin, body_first_pin, body_first_init]
}

/// Evaluate cross-row bodies at point `z`, given column evaluations at z and
/// shifted column evaluations at ω·z. Cross-row evaluations are split into
/// three β-RLC'd categories (each group of 4 limb bodies is folded with β
/// the same way the row-local MUL bodies are):
///
///   0. `result_chain`  (β-RLC over 4 limbs)
///   1. `base_invariant` (β-RLC over 4 limbs)
///   2. `bit_index_decrement` (single body)
///
/// All shifted evaluations are passed in `shifted_evals` ordered by
/// [`shifted_column_indices`].
pub fn evaluate_cross_row_at_point(
    col_evals_at_z: &[Scalar],
    shifted_evals: &[Scalar],
    beta: &Scalar,
) -> [Scalar; 6] {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    // shifted_evals layout: 4 result_in limbs, 4 base limbs, 1 bit_index, 1 active
    let res_shift: [Scalar; 4] = [
        shifted_evals[0].clone(),
        shifted_evals[1].clone(),
        shifted_evals[2].clone(),
        shifted_evals[3].clone(),
    ];
    let base_shift: [Scalar; 4] = [
        shifted_evals[4].clone(),
        shifted_evals[5].clone(),
        shifted_evals[6].clone(),
        shifted_evals[7].clone(),
    ];
    let bi_shift = &shifted_evals[8];
    let act_shift = &shifted_evals[9];
    // shifted_evals[18] = IS_FIRST_ROW(ω·X) for #94 multi-invocation
    // boundary gating. Multiplying every cross-row body by
    // `(1 − is_first_shift)` makes transitions INTO an
    // invocation-start row (where IS_FIRST_ROW(ω·X) = 1) vanish
    // naturally, so internal invocation boundaries don't violate
    // any continuity constraint. For single-invocation traces, the
    // only row with IS_FIRST_ROW = 1 is row 0, and the transition
    // INTO row 0 is the wrap (already excluded by
    // `(z − ω^{n−1})`); the new factor is 1 on every other row, so
    // single-invocation behavior is unchanged.
    let is_first_shift = &shifted_evals[18];
    let inv_boundary_gate = one.sub(is_first_shift);

    let eb = &col_evals_at_z[COL_EXP_BIT];
    let one_minus_eb = one.sub(eb);

    // 1. result_chain: result_in(ω·X)[i] − (eb · mul + (1−eb) · squared) = 0
    //    Gated by `act_shift · (1 − is_first_shift)` — fires on
    //    real-to-real transitions WITHIN an invocation only.
    let mut res_acc = zero.clone();
    let mut bp = one.clone();
    for i in 0..NUM_LIMBS {
        let mul_i = &col_evals_at_z[mul_limb(i)];
        let sq_i = &col_evals_at_z[squared_limb(i)];
        let chosen = eb.mul(mul_i).add(&one_minus_eb.mul(sq_i));
        let body = res_shift[i].sub(&chosen);
        let gated = act_shift.mul(&inv_boundary_gate).mul(&body);
        res_acc = res_acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }

    // 2. base_invariant: base(ω·X) − base(X) = 0, gated by
    //    act_shift · (1 − is_first_shift).
    let mut base_acc = zero.clone();
    let mut bp = one.clone();
    for i in 0..NUM_LIMBS {
        let cur = &col_evals_at_z[base_limb(i)];
        let body = base_shift[i].sub(cur);
        let gated = act_shift.mul(&inv_boundary_gate).mul(&body);
        base_acc = base_acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }

    // 3. bit_index_decrement: `(1 − is_first_shift) · (bi_shift −
    //    (bi − act_shift)) = 0`. On invocation-boundary transitions
    //    (is_first_shift = 1) the body vanishes via the gate so the
    //    next invocation's bi = 255 doesn't violate the chain.
    //
    //    Within an invocation:
    //      real → real: act_shift = 1, bi_shift = bi − 1.
    //      real → padding (last invocation): act_shift = 0, bi = 0,
    //        bi_shift = 0 ⇒ 0 = 0 − 0 holds.
    let bi_cur = &col_evals_at_z[COL_BIT_INDEX];
    let bi_body = bi_shift.sub(&bi_cur.sub(act_shift));
    let body = inv_boundary_gate.mul(&bi_body);

    // 4. exponent_invariant: EXPONENT(ω·X) − EXPONENT(X) = 0, gated
    //    by act_shift · (1 − is_first_shift). β-RLC over 4 limbs.
    let mut exp_acc = zero.clone();
    let mut bp = one.clone();
    for i in 0..NUM_LIMBS {
        let cur = &col_evals_at_z[COL_EXPONENT_OFFSET + i];
        let nxt = &shifted_evals[10 + i];
        let inv_body = nxt.sub(cur);
        let gated = act_shift.mul(&inv_boundary_gate).mul(&inv_body);
        exp_acc = exp_acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }

    // 5. final_output_invariant: FINAL_OUTPUT(ω·X) − FINAL_OUTPUT(X) = 0,
    //    gated by act_shift · (1 − is_first_shift). β-RLC over 4 limbs.
    let mut fo_acc = zero.clone();
    let mut bp = one.clone();
    for i in 0..NUM_LIMBS {
        let cur = &col_evals_at_z[COL_FINAL_OUTPUT_OFFSET + i];
        let nxt = &shifted_evals[14 + i];
        let inv_body = nxt.sub(cur);
        let gated = act_shift.mul(&inv_boundary_gate).mul(&inv_body);
        fo_acc = fo_acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }

    // 6. final_output_at_last_row: pins FINAL_OUTPUT to the
    //    algorithm's actual output on the row where the next row
    //    EITHER starts a new invocation (is_first_shift = 1) OR is
    //    padding (act_shift = 0). The combined indicator is
    //    `active · (is_first_shift + (1 − act_shift) − is_first_shift ·
    //    (1 − act_shift))` = `active · (is_first_shift OR (1 −
    //    act_shift))`. Since IS_FIRST_ROW = 1 implies act_shift = 1
    //    (the new invocation row IS active), the two conditions are
    //    disjoint: the simpler indicator
    //    `active · (is_first_shift + (1 − act_shift))` is correct.
    let active = &col_evals_at_z[COL_ACTIVE];
    let one_minus_act_shift = one.sub(act_shift);
    let last_indicator =
        active.mul(&is_first_shift.add(&one_minus_act_shift));
    let eb = &col_evals_at_z[COL_EXP_BIT];
    let one_minus_eb = one.sub(eb);
    let mut last_acc = zero.clone();
    let mut bp = one.clone();
    for i in 0..NUM_LIMBS {
        let mul_i = &col_evals_at_z[mul_limb(i)];
        let sq_i = &col_evals_at_z[squared_limb(i)];
        let chosen = eb.mul(mul_i).add(&one_minus_eb.mul(sq_i));
        let fo_i = &col_evals_at_z[COL_FINAL_OUTPUT_OFFSET + i];
        let body_i = fo_i.sub(&chosen);
        let gated = last_indicator.mul(&body_i);
        last_acc = last_acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }

    [res_acc, base_acc, body, exp_acc, fo_acc, last_acc]
}

/// Column indices that need shifted (ω·X) evaluations for cross-row constraints.
///
/// In order (existing indices preserved; new EXPONENT/FINAL_OUTPUT/
/// IS_FIRST_ROW shifts appended):
///   - 4 `result_in` limbs (for `result_chain`)
///   - 4 `base` limbs (for `base_invariant`)
///   - 1 `bit_index` (for `bit_index_decrement`)
///   - 1 `active` (gates `result_chain`, `base_invariant`, and the
///     two new invariance constraints)
///   - 4 `EXPONENT` limbs (for `exponent_invariant`)
///   - 4 `FINAL_OUTPUT` limbs (for `final_output_invariant`)
///   - 1 `IS_FIRST_ROW` (for #94 multi-invocation gating: each
///     cross-row body is gated by `(1 − IS_FIRST_ROW(ω·X))` so
///     transitions INTO an invocation-start row vanish naturally)
pub fn shifted_column_indices() -> Vec<usize> {
    let mut v = Vec::with_capacity(19);
    for i in 0..NUM_LIMBS {
        v.push(result_in_limb(i));
    }
    for i in 0..NUM_LIMBS {
        v.push(base_limb(i));
    }
    v.push(COL_BIT_INDEX);
    v.push(COL_ACTIVE);
    for i in 0..NUM_LIMBS {
        v.push(COL_EXPONENT_OFFSET + i);
    }
    for i in 0..NUM_LIMBS {
        v.push(COL_FINAL_OUTPUT_OFFSET + i);
    }
    v.push(COL_IS_FIRST_ROW);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    /// `2^7 = 128`. Almost all rows are no-ops (squaring 1 yields 1) up
    /// until row `255 - 7 = 248` which sees the only non-zero exponent bit.
    fn small_exp_witness() -> (U256, U256, Vec<ExpRow>) {
        let base = U256::from(2u64);
        let exponent = U256::from(7u64);
        let rows = exp_witness(base, exponent);
        (base, exponent, rows)
    }

    #[test]
    fn schoolbook_mul_matches_native_u128_for_small() {
        let (a, b) = ([3u64, 0, 0, 0], [5u64, 0, 0, 0]);
        let (out, c) = schoolbook_mul_mod_2_256(a, b);
        assert_eq!(out, [15, 0, 0, 0]);
        assert_eq!(c, [0, 0, 0, 0]);
    }

    #[test]
    fn schoolbook_mul_carries_propagate() {
        // a = 2^64 - 1 in limb 0
        let a = [u64::MAX, 0, 0, 0];
        let b = [u64::MAX, 0, 0, 0];
        let (out, c) = schoolbook_mul_mod_2_256(a, b);
        // (2^64 − 1)^2 = 2^128 − 2·2^64 + 1
        // limb0 = 1, limb1 = 2^64 − 2 = u64::MAX - 1, carry[0] = u64::MAX - 1
        assert_eq!(out[0], 1);
        assert_eq!(out[1], u64::MAX - 1);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 0);
        assert_eq!(c[0], u64::MAX - 1);
    }

    #[test]
    fn exp_witness_matches_reference_pow_mod() {
        let (base, exponent, rows) = small_exp_witness();
        assert_eq!(rows.len(), NUM_STEPS);
        // Last row's chosen output equals 2^7 = 128.
        let last = rows.last().unwrap();
        let chosen = if last.exp_bit == 1 { last.mul } else { last.squared };
        assert_eq!(chosen, [128, 0, 0, 0]);
        assert_eq!(chosen, pow_mod_2_256(base, exponent));
    }

    #[test]
    fn exp_witness_first_row_starts_from_one() {
        let (_, _, rows) = small_exp_witness();
        assert_eq!(rows[0].result_in, [1, 0, 0, 0]);
        assert_eq!(rows[0].bit_index, (NUM_STEPS - 1) as u8);
    }

    #[test]
    fn exp_witness_bit_index_decrements() {
        let (_, _, rows) = small_exp_witness();
        for r in 0..NUM_STEPS - 1 {
            assert_eq!(
                rows[r + 1].bit_index as i32,
                rows[r].bit_index as i32 - 1,
                "bit_index must decrement by 1"
            );
        }
    }

    #[test]
    fn populate_trace_round_trips() {
        let curve = CurveType::Bls48581;
        let (_, _, rows) = small_exp_witness();
        let cols = populate_trace(&rows, curve);
        assert_eq!(cols.len(), NUM_EXP_AIR_COLUMNS);
        for c in &cols {
            assert_eq!(c.len(), NUM_STEPS);
        }
        // Spot-check: row 0's result_in limb0 == 1.
        assert!(cols[result_in_limb(0)][0]
            .sub(&Scalar::one(curve))
            .is_zero());
        // Active is 1 everywhere.
        for r in 0..NUM_STEPS {
            assert!(cols[COL_ACTIVE][r]
                .sub(&Scalar::one(curve))
                .is_zero());
        }
    }

    #[test]
    fn evaluate_row_local_vanishes_on_valid_witness() {
        let curve = CurveType::Bls48581;
        let (_, _, rows) = small_exp_witness();
        let cols = populate_trace(&rows, curve);
        let beta = Scalar::from_u64(31, curve);
        let evals = evaluate_row_local(&col_refs(&cols), &beta);
        assert_eq!(evals.len(), 7);
        for ev in &evals {
            for (r, v) in ev.values.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local body '{}' nonzero at row {}",
                    ev.label,
                    r
                );
            }
        }
    }

    #[test]
    fn evaluate_row_local_at_point_zero_on_real_rows() {
        let curve = CurveType::Bls48581;
        let (_, _, rows) = small_exp_witness();
        let cols = populate_trace(&rows, curve);
        let beta = Scalar::from_u64(13, curve);
        for r in 0..NUM_STEPS {
            let row_vals: Vec<Scalar> = cols.iter().map(|c| c[r].clone()).collect();
            let bodies = evaluate_row_local_at_point(&row_vals, &beta);
            for (k, b) in bodies.iter().enumerate() {
                assert!(
                    b.is_zero(),
                    "row {}, body[{}] nonzero on valid witness",
                    r,
                    k
                );
            }
        }
    }

    #[test]
    fn evaluate_row_local_at_point_zero_on_padding_row() {
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let row_vals = vec![zero; NUM_EXP_AIR_COLUMNS];
        let beta = Scalar::from_u64(99, curve);
        let bodies = evaluate_row_local_at_point(&row_vals, &beta);
        for b in &bodies {
            assert!(b.is_zero(), "body must vanish on all-zero row");
        }
    }

    #[test]
    fn evaluate_cross_row_zero_on_valid_transitions() {
        let curve = CurveType::Bls48581;
        let (_, _, rows) = small_exp_witness();
        let cols = populate_trace(&rows, curve);
        let beta = Scalar::from_u64(41, curve);
        let shift_idxs = shifted_column_indices();
        // Walk every consecutive pair of real rows. Cross-row constraints
        // pertain to row r → r+1 transitions; the wrap-around (r=255 → 0)
        // is excluded by the verifier's (X − ω^{n−1}) factor and not
        // checked here.
        for r in 0..NUM_STEPS - 1 {
            let row_z: Vec<Scalar> = cols.iter().map(|c| c[r].clone()).collect();
            let shifted: Vec<Scalar> =
                shift_idxs.iter().map(|&i| cols[i][r + 1].clone()).collect();
            let bodies = evaluate_cross_row_at_point(&row_z, &shifted, &beta);
            for (k, b) in bodies.iter().enumerate() {
                assert!(
                    b.is_zero(),
                    "cross-row body[{}] fired at transition {} → {}",
                    k,
                    r,
                    r + 1
                );
            }
        }
    }

    #[test]
    fn cross_row_detects_tampered_result_chain() {
        let curve = CurveType::Bls48581;
        let (_, _, mut rows) = small_exp_witness();
        // Tamper: flip the next row's result_in[0]
        rows[10].result_in[0] = rows[10].result_in[0].wrapping_add(1);
        let cols = populate_trace(&rows, curve);
        let beta = Scalar::from_u64(7, curve);
        let shift_idxs = shifted_column_indices();
        // Transition 9 → 10 should now fire.
        let row_z: Vec<Scalar> = cols.iter().map(|c| c[9].clone()).collect();
        let shifted: Vec<Scalar> =
            shift_idxs.iter().map(|&i| cols[i][10].clone()).collect();
        let bodies = evaluate_cross_row_at_point(&row_z, &shifted, &beta);
        assert!(
            !bodies[0].is_zero(),
            "result_chain must fire when next row's result_in is tampered"
        );
    }

    #[test]
    fn cross_row_detects_tampered_base() {
        let curve = CurveType::Bls48581;
        let (_, _, mut rows) = small_exp_witness();
        // Mutate base on row 5 (still constant on every other row).
        rows[5].base[1] = 0xDEAD_BEEF;
        let cols = populate_trace(&rows, curve);
        let beta = Scalar::from_u64(7, curve);
        let shift_idxs = shifted_column_indices();
        // Transition 4 → 5 should fire (base on row 5 differs from row 4).
        let row_z: Vec<Scalar> = cols.iter().map(|c| c[4].clone()).collect();
        let shifted: Vec<Scalar> =
            shift_idxs.iter().map(|&i| cols[i][5].clone()).collect();
        let bodies = evaluate_cross_row_at_point(&row_z, &shifted, &beta);
        assert!(
            !bodies[1].is_zero(),
            "base_invariant must fire when base changes between rows"
        );
    }

    #[test]
    fn cross_row_detects_bit_index_skip() {
        let curve = CurveType::Bls48581;
        let (_, _, mut rows) = small_exp_witness();
        // Skip a bit_index value on row 7.
        rows[7].bit_index = rows[7].bit_index.wrapping_sub(1);
        let cols = populate_trace(&rows, curve);
        let beta = Scalar::from_u64(7, curve);
        let shift_idxs = shifted_column_indices();
        // Transition 6 → 7 should fire.
        let row_z: Vec<Scalar> = cols.iter().map(|c| c[6].clone()).collect();
        let shifted: Vec<Scalar> =
            shift_idxs.iter().map(|&i| cols[i][7].clone()).collect();
        let bodies = evaluate_cross_row_at_point(&row_z, &shifted, &beta);
        assert!(
            !bodies[2].is_zero(),
            "bit_index_decrement must fire on a bad transition"
        );
    }

    // ── #94 Phase 2 multi-invocation tests ────────────────────────

    #[test]
    fn populate_multi_trace_shape() {
        let curve = CurveType::Bls48581;
        let invs = [
            (U256::from(2u64), U256::from(7u64)),
            (U256::from(3u64), U256::from(11u64)),
        ];
        let cols = populate_multi_trace(&invs, curve);
        assert_eq!(cols.len(), NUM_EXP_AIR_COLUMNS);
        let total = invs.len() * NUM_STEPS;
        for c in &cols {
            assert_eq!(c.len(), total);
        }
        // IS_FIRST_ROW set on row 0 and row NUM_STEPS only.
        for r in 0..total {
            let expected = if r == 0 || r == NUM_STEPS { 1 } else { 0 };
            let actual = if cols[COL_IS_FIRST_ROW][r]
                .sub(&Scalar::one(curve))
                .is_zero()
            {
                1
            } else if cols[COL_IS_FIRST_ROW][r].is_zero() {
                0
            } else {
                panic!("non-binary IS_FIRST_ROW at row {}", r);
            };
            assert_eq!(actual, expected, "IS_FIRST_ROW row {}", r);
        }
        // bit_index = 255 at row 0 and row NUM_STEPS.
        for &r in &[0_usize, NUM_STEPS] {
            assert_eq!(
                cols[COL_BIT_INDEX][r].to_u64(),
                (NUM_STEPS - 1) as u64
            );
        }
        // Active = 1 everywhere (no padding within an invocation block).
        for r in 0..total {
            assert!(cols[COL_ACTIVE][r]
                .sub(&Scalar::one(curve))
                .is_zero());
        }
    }

    #[test]
    fn populate_multi_trace_per_invocation_exponent_and_final_output() {
        let curve = CurveType::Bls48581;
        let invs = [
            (U256::from(2u64), U256::from(7u64)),
            (U256::from(3u64), U256::from(11u64)),
        ];
        let cols = populate_multi_trace(&invs, curve);
        // Invocation 0 (rows 0..256): exponent=7, output=128.
        for r in 0..NUM_STEPS {
            assert_eq!(cols[COL_EXPONENT_OFFSET][r].to_u64(), 7);
            assert_eq!(cols[COL_FINAL_OUTPUT_OFFSET][r].to_u64(), 128);
        }
        // Invocation 1 (rows 256..512): exponent=11, output=3^11=177147.
        for r in NUM_STEPS..2 * NUM_STEPS {
            assert_eq!(cols[COL_EXPONENT_OFFSET][r].to_u64(), 11);
            assert_eq!(cols[COL_FINAL_OUTPUT_OFFSET][r].to_u64(), 177147);
        }
    }

    /// The defining property of Phase 1: cross-row constraints must
    /// vanish at the boundary between two invocations even though both
    /// rows are active. Without the `(1 − is_first_shift)` gating,
    /// `result_chain` would fire (invocation 1's result_in = 1 ≠
    /// invocation 0's last result_out), `base_invariant` would fire
    /// (different bases), `bit_index_decrement` would fire (255 ≠
    /// −1), `exponent_invariant` and `final_output_invariant` would
    /// fire. With Phase 1 gating, all six bodies vanish at the
    /// boundary.
    #[test]
    fn cross_row_vanishes_at_invocation_boundary() {
        let curve = CurveType::Bls48581;
        let invs = [
            (U256::from(2u64), U256::from(7u64)),
            (U256::from(3u64), U256::from(11u64)),
        ];
        let cols = populate_multi_trace(&invs, curve);
        let beta = Scalar::from_u64(7, curve);
        let shift_idxs = shifted_column_indices();

        // Boundary transition: row NUM_STEPS - 1 (last of invocation 0)
        // → row NUM_STEPS (first of invocation 1).
        let r = NUM_STEPS - 1;
        let row_z: Vec<Scalar> =
            cols.iter().map(|c| c[r].clone()).collect();
        let shifted: Vec<Scalar> =
            shift_idxs.iter().map(|&i| cols[i][r + 1].clone()).collect();
        let bodies = evaluate_cross_row_at_point(&row_z, &shifted, &beta);
        for (k, b) in bodies.iter().enumerate() {
            assert!(
                b.is_zero(),
                "cross-row body[{}] fired at invocation boundary {} → {}",
                k,
                r,
                r + 1
            );
        }
    }

    /// Sanity: every WITHIN-invocation transition still satisfies the
    /// cross-row constraints (the new boundary gating only kicks off
    /// at the boundary, not within invocations).
    #[test]
    fn cross_row_vanishes_within_each_invocation() {
        let curve = CurveType::Bls48581;
        let invs = [
            (U256::from(2u64), U256::from(7u64)),
            (U256::from(3u64), U256::from(11u64)),
        ];
        let cols = populate_multi_trace(&invs, curve);
        let beta = Scalar::from_u64(7, curve);
        let shift_idxs = shifted_column_indices();
        let total = invs.len() * NUM_STEPS;

        // All within-invocation transitions (skip the boundary at
        // row NUM_STEPS - 1 → NUM_STEPS, which the previous test
        // covers) and skip the wrap.
        for r in 0..total - 1 {
            if r == NUM_STEPS - 1 {
                continue; // boundary — separately tested
            }
            let row_z: Vec<Scalar> =
                cols.iter().map(|c| c[r].clone()).collect();
            let shifted: Vec<Scalar> =
                shift_idxs.iter().map(|&i| cols[i][r + 1].clone()).collect();
            let bodies =
                evaluate_cross_row_at_point(&row_z, &shifted, &beta);
            for (k, b) in bodies.iter().enumerate() {
                assert!(
                    b.is_zero(),
                    "cross-row body[{}] fired at within-invocation transition {} → {}",
                    k,
                    r,
                    r + 1
                );
            }
        }
    }

    /// Phase 1 gating: cross-row `result_chain` is SUPPRESSED at the
    /// invocation boundary, so a tampered `result_in` at an invocation
    /// start is NOT caught there. The proper soundness gate is the
    /// Phase 3 row-local pin `IS_FIRST_ROW · (result_in − init) = 0`,
    /// covered by `phase3_row_local_pins_invocation_start_result_in`.
    #[test]
    fn cross_row_does_not_catch_tampered_invocation_start_result_in() {
        let curve = CurveType::Bls48581;
        let invs = [
            (U256::from(2u64), U256::from(7u64)),
            (U256::from(3u64), U256::from(11u64)),
        ];
        let mut cols = populate_multi_trace(&invs, curve);
        cols[result_in_limb(0)][NUM_STEPS] = Scalar::from_u64(5, curve);
        let beta = Scalar::from_u64(7, curve);
        let shift_idxs = shifted_column_indices();
        let row_z: Vec<Scalar> = cols
            .iter()
            .map(|c| c[NUM_STEPS - 1].clone())
            .collect();
        let shifted: Vec<Scalar> = shift_idxs
            .iter()
            .map(|&i| cols[i][NUM_STEPS].clone())
            .collect();
        let bodies = evaluate_cross_row_at_point(&row_z, &shifted, &beta);
        // result_chain (body 0) is SUPPRESSED at boundary by Phase 1.
        assert!(bodies[0].is_zero());
    }

    /// Phase 3: the new row-local pin
    /// `IS_FIRST_ROW · β-RLC_i (result_in[i] − initializer[i]) = 0`
    /// catches what Phase 1's boundary gating leaves uncovered. On
    /// an honest invocation 1 start the body vanishes; tampering
    /// `result_in` makes it fire.
    #[test]
    fn phase3_row_local_pins_invocation_start_result_in() {
        let curve = CurveType::Bls48581;
        let invs = [
            (U256::from(2u64), U256::from(7u64)),
            (U256::from(3u64), U256::from(11u64)),
        ];
        // Honest case: row NUM_STEPS has IS_FIRST_ROW = 1 and
        // result_in = 1. Body 6 (`is_first_row_pins_result_in`)
        // vanishes.
        let cols = populate_multi_trace(&invs, curve);
        let beta = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> =
            cols.iter().map(|c| c[NUM_STEPS].clone()).collect();
        let bodies = evaluate_row_local_at_point(&row_vals, &beta);
        assert!(
            bodies[6].is_zero(),
            "honest invocation start must satisfy the result_in pin"
        );

        // Tamper: set invocation 1's result_in[0] to 5. Body 6 fires.
        let mut cols = populate_multi_trace(&invs, curve);
        cols[result_in_limb(0)][NUM_STEPS] = Scalar::from_u64(5, curve);
        let row_vals: Vec<Scalar> =
            cols.iter().map(|c| c[NUM_STEPS].clone()).collect();
        let bodies = evaluate_row_local_at_point(&row_vals, &beta);
        assert!(
            !bodies[6].is_zero(),
            "is_first_row_pins_result_in must fire on tampered \
             result_in[0] at an invocation start"
        );

        // Tamper a higher limb too: invocation 0's result_in[2] at row
        // 0 (also IS_FIRST_ROW = 1, should be 0).
        let mut cols = populate_multi_trace(&invs, curve);
        cols[result_in_limb(2)][0] = Scalar::from_u64(0xdeadbeef, curve);
        let row_vals: Vec<Scalar> =
            cols.iter().map(|c| c[0].clone()).collect();
        let bodies = evaluate_row_local_at_point(&row_vals, &beta);
        assert!(
            !bodies[6].is_zero(),
            "is_first_row_pins_result_in must fire on a non-zero \
             higher limb at invocation start"
        );

        // Non-IS_FIRST_ROW row: even with bogus result_in the pin
        // doesn't fire (it's gated by IS_FIRST_ROW).
        let mut cols = populate_multi_trace(&invs, curve);
        cols[result_in_limb(0)][5] = Scalar::from_u64(99, curve);
        let row_vals: Vec<Scalar> =
            cols.iter().map(|c| c[5].clone()).collect();
        let bodies = evaluate_row_local_at_point(&row_vals, &beta);
        assert!(
            bodies[6].is_zero(),
            "is_first_row_pins_result_in must NOT fire on rows where \
             IS_FIRST_ROW = 0"
        );
    }

    #[test]
    fn row_local_detects_tampered_squared_limb() {
        let curve = CurveType::Bls48581;
        let (_, _, rows) = small_exp_witness();
        let mut cols = populate_trace(&rows, curve);
        // Find a row where squared isn't zero — row 254 squares "result of
        // 2^6 = 64" so squared limb 0 is 4096 ≠ 0.
        let r = 254;
        cols[squared_limb(0)][r] = cols[squared_limb(0)][r]
            .add(&Scalar::one(curve));
        let beta = Scalar::from_u64(2, curve);
        let row_vals: Vec<Scalar> = cols.iter().map(|c| c[r].clone()).collect();
        let bodies = evaluate_row_local_at_point(&row_vals, &beta);
        // squared_mul_chain is body index 2.
        assert!(
            !bodies[2].is_zero(),
            "squared_mul_chain must fire when squared limb is tampered"
        );
    }
}
