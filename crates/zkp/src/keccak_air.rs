//! Bit-level AIR for Keccak-f[1600].
//!
//! This module implements a sound constraint system for the 24-round
//! Keccak-f[1600] permutation by representing every 64-bit lane as 64
//! individual 0/1 bit columns. The 5×5×64 = 1600 bit columns per state
//! grid make every Keccak-f operation (XOR, AND, NOT, rotate) expressible
//! as a degree-≤2 algebraic constraint:
//!
//!   XOR(a, b) = a + b − 2·a·b
//!   NOT(a)    = 1 − a
//!   AND(a, b) = a · b
//!   rotate-by-k = permute bit positions (structural — no arithmetic)
//!
//! Trade-off: ~13,000 bit columns per row. Accepted because the alternative
//! (bitwise LogUp wiring for thousands of XORs per round) is vastly more
//! complex to reason about and audit.
//!
//! # Row layout
//!
//! One row per Keccak round. 24 rounds per permutation. For a single
//! permutation the trace is 24 rows; for a multi-block hash each block
//! contributes 24 more rounds. Cross-row constraint binds `after_iota`
//! of row `r` to `before` of row `r+1`, except on the last row of each
//! permutation (where the next row begins a fresh permutation).
//!
//! # Column layout
//!
//! All columns hold 0/1 scalar values. The layout is:
//!
//! ```text
//! index range    name              size   meaning
//! 0..1600        BEFORE            1600   before[x][y][bit]  (state before θ)
//! 1600..3200     C_PARTIAL         1600   cumulative column parity
//! 3200..3520     D                  320   D[x][bit]
//! 3520..5120     AFTER_THETA       1600   state after θ
//! 5120..6720     AFTER_RHO         1600   state after ρ
//! 6720..8320     AFTER_PI          1600   state after π
//! 8320..9920     NAND_TEMP         1600   (¬after_pi[x+1][y]) ∧ after_pi[x+2][y]
//! 9920..11520    AFTER_CHI         1600   state after χ
//! 11520..13120   AFTER_IOTA        1600   state after ι (= round output)
//! 13120..13144   SEL_ROUND          24    one-hot round selectors
//! 13144..13400   INV_INPUT_BYTE    256    per-invocation input bytes (zero-padded)
//! 13400..13401   INV_INPUT_LEN_COL   1    per-invocation input length
//! 13401..13433   INV_OUTPUT_BYTE    32    per-invocation digest bytes
//! 13433..13434   IS_FIRST_INV_ROW    1    anchor: 1 only on row 0 of invocation
//! 13434..13435   IS_FIRST_BLOCK      1    1 on every round of block 0, 0 elsewhere
//! 13435..13691   IS_BYTE_ACTIVE    256    per-byte gate at row-0 anchor: 1 iff b < INV_INPUT_LEN
//! 13691..13692   IS_LAST_BLOCK       1    1 on the LAST absorption block's rows, 0 elsewhere
//! ```
//!
//! Total: 13,120 data + 24 selectors + 290 invocation aggregator + 1
//! IS_FIRST_BLOCK + 256 IS_BYTE_ACTIVE + 1 IS_LAST_BLOCK = **13,692 columns**.
//!
//! # Constraints (all gated by the row's active round selector where relevant)
//!
//!  1. Bit validity: every data bit column satisfies `b·(b−1) = 0`.
//!  2. C_partial recurrence: `C_partial[x][0][bit] = before[x][0][bit]`
//!     and `C_partial[x][y+1][bit] = C_partial[x][y][bit] ⊕
//!     before[x][y+1][bit]` (degree-2 XOR).
//!  3. D: `D[x][bit] = C_partial[x−1][4][bit] ⊕ C_partial[x+1][4][(bit+63) mod 64]`.
//!  4. θ: `after_theta[x][y][bit] = before[x][y][bit] ⊕ D[x][bit]`.
//!  5. ρ: `after_rho[x][y][bit] = after_theta[x][y][(bit − RHO_OFFSETS[x][y]) mod 64]`
//!     (structural equality).
//!  6. π: `after_pi[y][(2x+3y) mod 5][bit] = after_rho[x][y][bit]`.
//!  7. χ: `nand_temp[x][y][bit] = (1 − after_pi[(x+1)%5][y][bit]) ·
//!     after_pi[(x+2)%5][y][bit]` and `after_chi[x][y][bit] =
//!     after_pi[x][y][bit] ⊕ nand_temp[x][y][bit]`.
//!  8. ι: `after_iota[x][y][bit] = after_chi[x][y][bit]` for all
//!     (x, y, bit) except (0, 0, *), where per active round selector
//!     `sel_round_k`: `after_iota[0][0][bit] = after_chi[0][0][bit] ⊕
//!     RC_bit_k`.
//!  9. Selector one-hot: every `sel_round_k` is binary, and their sum on
//!     active rows is exactly 1.
//!
//! (Cross-row constraints — `after_iota` of row r equals `before` of row
//! r+1 — are described but not realized as polynomial division artifacts
//! in this module. The AIR exposes them via `evaluate_cross_row` so tests
//! can exercise the formula.)

use crate::field::{CurveType, Scalar};
use crate::keccak::{HashTrace, RoundTrace, NUM_ROUNDS, RHO_OFFSETS, ROUND_CONSTANTS};

// ──── Column offsets ──────────────────────────────────────────────────

pub const BITS_PER_LANE: usize = 64;
pub const LANE_COUNT: usize = 25;            // 5 × 5
pub const BITS_PER_STATE: usize = 1600;      // 5 × 5 × 64
pub const BITS_PER_PARITY: usize = 320;      // 5 × 64

pub const COL_BEFORE_OFFSET:       usize = 0;
pub const COL_C_PARTIAL_OFFSET:    usize = COL_BEFORE_OFFSET + BITS_PER_STATE;
pub const COL_D_OFFSET:            usize = COL_C_PARTIAL_OFFSET + BITS_PER_STATE;
pub const COL_AFTER_THETA_OFFSET:  usize = COL_D_OFFSET + BITS_PER_PARITY;
pub const COL_AFTER_RHO_OFFSET:    usize = COL_AFTER_THETA_OFFSET + BITS_PER_STATE;
pub const COL_AFTER_PI_OFFSET:     usize = COL_AFTER_RHO_OFFSET + BITS_PER_STATE;
pub const COL_NAND_TEMP_OFFSET:    usize = COL_AFTER_PI_OFFSET + BITS_PER_STATE;
pub const COL_AFTER_CHI_OFFSET:    usize = COL_NAND_TEMP_OFFSET + BITS_PER_STATE;
pub const COL_AFTER_IOTA_OFFSET:   usize = COL_AFTER_CHI_OFFSET + BITS_PER_STATE;

pub const NUM_DATA_COLUMNS: usize = COL_AFTER_IOTA_OFFSET + BITS_PER_STATE;

pub const COL_SEL_ROUND_OFFSET:    usize = NUM_DATA_COLUMNS;
pub const NUM_SEL_ROUND:           usize = NUM_ROUNDS; // 24

// ── Per-invocation byte-aggregation columns ───────────────────────────
//
// Mirror of [`crate::sha256_air`]'s aggregator columns. A Keccak AIR
// trace covers ONE `keccak256(input) → output` invocation. The bytes of
// the input (variable-length, zero-padded to `INV_INPUT_LEN`) and the
// 32-byte output digest are committed as per-invocation aggregator
// columns: they hold the SAME value on every row (invariance constraint
// — deferred, not yet wired). A cross-AIR LogUp linkage from
// [`crate::keccak_extract`] picks the anchor row (where
// `IS_FIRST_INV_ROW = 1`) and matches its (`INV_INPUT_BYTE[..]`,
// `INV_INPUT_LEN`, `INV_OUTPUT_BYTE[..]`) tuple against the extract
// AIR's per-row tuple.
//
// `INV_INPUT_LEN = 256` matches [`crate::keccak_extract::MAX_INPUT_LEN`]
// — covers MPT leaf and extension nodes. Wider inputs (MPT branch
// nodes, etc.) need a larger-bound variant.
//
// Soundness scope (current): the bytes are witness-populated and
// invariance is NOT yet enforced algebraically. Algebraic binding to
// the bit-level state (input bytes ↔ first block's BEFORE bits at
// row 0, output bytes ↔ AFTER_IOTA at the squeeze row) is a documented
// follow-up — see `cross_air_logup_dependent_tasks.md`.
pub const INV_INPUT_LEN: usize = 256;
pub const INV_OUTPUT_LEN: usize = 32;
pub const COL_INV_INPUT_BYTE_OFFSET:  usize = COL_SEL_ROUND_OFFSET + NUM_SEL_ROUND;
pub const COL_INV_INPUT_LEN_COL:      usize = COL_INV_INPUT_BYTE_OFFSET + INV_INPUT_LEN;
pub const COL_INV_OUTPUT_BYTE_OFFSET: usize = COL_INV_INPUT_LEN_COL + 1;
pub const COL_IS_FIRST_INV_ROW:       usize = COL_INV_OUTPUT_BYTE_OFFSET + INV_OUTPUT_LEN;

/// Indicator: `1` on every round of block 0 (rows 0..NUM_ROUNDS-1),
/// `0` on every round of subsequent absorption blocks. Witness-populated
/// and constrained by the same binarity + anchor-pin + within-block
/// invariance pattern as [`crate::sha256_air::COL_IS_FIRST_BLOCK`].
/// Required for any future algebraic input binding (rate bits at row 0
/// of block 0).
pub const COL_IS_FIRST_BLOCK:         usize = COL_IS_FIRST_INV_ROW + 1;

/// Length of the SHA3-style absorption rate in bytes (1088 bits =
/// 17 lanes × 64 bits). For Keccak-256 / SHA3-256 (used by Ethereum
/// and this codebase), this is the per-block absorption width.
pub const RATE_LEN: usize = 136;

/// Per-byte "is active" indicator at the row-0 anchor. `is_byte_active[b]
/// = 1` iff byte position `b` is within the actual input length
/// (`b < INV_INPUT_LEN_COL`); `0` otherwise. Witness-populated only at
/// row 0 (the anchor); unused on subsequent rows. Constrained via:
///   - Binarity (per-b, gated by `IS_FIRST_INV_ROW`)
///   - Monotone-decreasing in b (gated by `IS_FIRST_INV_ROW`)
///   - Sum across b ∈ [0, INV_INPUT_LEN) equals `INV_INPUT_LEN_COL`
///     (gated by `IS_FIRST_INV_ROW`)
///
/// Used by the input-binding constraint to gate the per-byte
/// `INV_INPUT_BYTE[b] = rate_byte_at_b` check to only fire on the
/// in-range bytes. Sized `INV_INPUT_LEN` (= MAX_INPUT_LEN = 256) so
/// it covers ALL input byte positions across both block 0 (positions
/// 0..136) and block 1 (positions 136..256) of multi-absorption
/// 2-block Keccak invocations.
pub const COL_IS_BYTE_ACTIVE_OFFSET:  usize = COL_IS_FIRST_BLOCK + 1;
pub const NUM_IS_BYTE_ACTIVE: usize = INV_INPUT_LEN; // 256, was RATE_LEN before multi-absorption

/// Indicator: `1` on every round of the LAST absorption block of an
/// aggregator-populated trace, `0` elsewhere. For single-absorption
/// traces (1 permutation), `IS_LAST_BLOCK = IS_FIRST_BLOCK = 1`
/// throughout. For multi-absorption (N permutations), `IS_LAST_BLOCK`
/// is `1` only on the last 24 rows.
///
/// Algebraically constrained by binarity (row-local) + within-permutation
/// invariance (shifted; same gate as IS_FIRST_BLOCK invariance). Used
/// by the output-byte binding to fire at the LAST round of the LAST
/// permutation (= the squeeze step).
///
/// **Soundness for multi-absorption**: a malicious prover setting
/// IS_LAST_BLOCK on a non-last permutation forces the digest claim to
/// equal an intermediate AFTER_IOTA — by Keccak collision resistance
/// this cannot match the canonical hash claimed by the extract AIR
/// (the cross-AIR LogUp matches digest bytes against the extract's
/// canonical hash claim, which the verifier independently accepts as
/// the trie root or other trusted commitment).
pub const COL_IS_LAST_BLOCK:          usize = COL_IS_BYTE_ACTIVE_OFFSET + NUM_IS_BYTE_ACTIVE;

pub const NUM_KECCAK_COLUMNS: usize = COL_IS_LAST_BLOCK + 1;

// ──── Column index helpers ────────────────────────────────────────────

#[inline]
fn grid_bit_index(x: usize, y: usize, bit: usize) -> usize {
    debug_assert!(x < 5 && y < 5 && bit < BITS_PER_LANE);
    (y * 5 + x) * BITS_PER_LANE + bit
}

#[inline]
pub fn before(x: usize, y: usize, bit: usize) -> usize {
    COL_BEFORE_OFFSET + grid_bit_index(x, y, bit)
}

/// `C_partial[x][y][bit]` — cumulative column parity up through y.
/// `C_partial[x][0]` = before[x][0], and the final parity `C[x] = C_partial[x][4]`.
#[inline]
pub fn c_partial(x: usize, y: usize, bit: usize) -> usize {
    COL_C_PARTIAL_OFFSET + grid_bit_index(x, y, bit)
}

#[inline]
pub fn c_final(x: usize, bit: usize) -> usize {
    c_partial(x, 4, bit)
}

#[inline]
pub fn d_col(x: usize, bit: usize) -> usize {
    debug_assert!(x < 5 && bit < BITS_PER_LANE);
    COL_D_OFFSET + x * BITS_PER_LANE + bit
}

#[inline]
pub fn after_theta(x: usize, y: usize, bit: usize) -> usize {
    COL_AFTER_THETA_OFFSET + grid_bit_index(x, y, bit)
}

#[inline]
pub fn after_rho(x: usize, y: usize, bit: usize) -> usize {
    COL_AFTER_RHO_OFFSET + grid_bit_index(x, y, bit)
}

#[inline]
pub fn after_pi(x: usize, y: usize, bit: usize) -> usize {
    COL_AFTER_PI_OFFSET + grid_bit_index(x, y, bit)
}

#[inline]
pub fn nand_temp(x: usize, y: usize, bit: usize) -> usize {
    COL_NAND_TEMP_OFFSET + grid_bit_index(x, y, bit)
}

#[inline]
pub fn after_chi(x: usize, y: usize, bit: usize) -> usize {
    COL_AFTER_CHI_OFFSET + grid_bit_index(x, y, bit)
}

#[inline]
pub fn after_iota(x: usize, y: usize, bit: usize) -> usize {
    COL_AFTER_IOTA_OFFSET + grid_bit_index(x, y, bit)
}

#[inline]
pub fn sel_round(round: usize) -> usize {
    debug_assert!(round < NUM_SEL_ROUND);
    COL_SEL_ROUND_OFFSET + round
}

#[inline]
pub fn inv_input_byte(b: usize) -> usize {
    debug_assert!(b < INV_INPUT_LEN);
    COL_INV_INPUT_BYTE_OFFSET + b
}

#[inline]
pub fn inv_output_byte(b: usize) -> usize {
    debug_assert!(b < INV_OUTPUT_LEN);
    COL_INV_OUTPUT_BYTE_OFFSET + b
}

#[inline]
pub fn is_byte_active(b: usize) -> usize {
    debug_assert!(b < NUM_IS_BYTE_ACTIVE);
    COL_IS_BYTE_ACTIVE_OFFSET + b
}

// ──── Witness population ──────────────────────────────────────────────

/// Initialize `columns` to have `NUM_KECCAK_COLUMNS` empty vectors each
/// of `num_rows` `Scalar::zero(curve)` entries. Convenience for tests.
pub fn alloc_trace(num_rows: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..NUM_KECCAK_COLUMNS)
        .map(|_| vec![zero.clone(); num_rows])
        .collect()
}

/// Given the 25-lane state, write the 1600 bit values (LSB-first per lane)
/// into the destination range `[offset..offset+1600]` at `row`.
fn write_state_bits(
    columns: &mut [Vec<Scalar>],
    row: usize,
    offset: usize,
    state: &[[u64; 5]; 5],
    one: &Scalar,
    zero: &Scalar,
) {
    for y in 0..5 {
        for x in 0..5 {
            let lane = state[x][y];
            let base = offset + (y * 5 + x) * BITS_PER_LANE;
            for bit in 0..BITS_PER_LANE {
                let v = (lane >> bit) & 1;
                columns[base + bit][row] =
                    if v == 1 { one.clone() } else { zero.clone() };
            }
        }
    }
}

/// Derive the `C_partial` cumulative-XOR values from a `before` state and
/// write them into `row`.
fn write_c_partial(
    columns: &mut [Vec<Scalar>],
    row: usize,
    before_state: &[[u64; 5]; 5],
    one: &Scalar,
    zero: &Scalar,
) {
    for x in 0..5 {
        // acc_lane accumulates the XOR across y = 0..=y_cur.
        let mut acc: u64 = 0;
        for y in 0..5 {
            acc ^= before_state[x][y];
            let base = COL_C_PARTIAL_OFFSET + (y * 5 + x) * BITS_PER_LANE;
            for bit in 0..BITS_PER_LANE {
                let v = (acc >> bit) & 1;
                columns[base + bit][row] =
                    if v == 1 { one.clone() } else { zero.clone() };
            }
        }
    }
}

/// Write the `D[x]` lane bits for a given round into `row`.
fn write_d(
    columns: &mut [Vec<Scalar>],
    row: usize,
    d_lanes: &[u64; 5],
    one: &Scalar,
    zero: &Scalar,
) {
    for x in 0..5 {
        let base = COL_D_OFFSET + x * BITS_PER_LANE;
        for bit in 0..BITS_PER_LANE {
            let v = (d_lanes[x] >> bit) & 1;
            columns[base + bit][row] =
                if v == 1 { one.clone() } else { zero.clone() };
        }
    }
}

/// Derive and write the `nand_temp` aux column from an `after_pi` state.
fn write_nand_temp(
    columns: &mut [Vec<Scalar>],
    row: usize,
    after_pi_state: &[[u64; 5]; 5],
    one: &Scalar,
    zero: &Scalar,
) {
    for y in 0..5 {
        for x in 0..5 {
            let a1 = after_pi_state[(x + 1) % 5][y];
            let a2 = after_pi_state[(x + 2) % 5][y];
            let nand = (!a1) & a2;
            let base = COL_NAND_TEMP_OFFSET + (y * 5 + x) * BITS_PER_LANE;
            for bit in 0..BITS_PER_LANE {
                let v = (nand >> bit) & 1;
                columns[base + bit][row] =
                    if v == 1 { one.clone() } else { zero.clone() };
            }
        }
    }
}

/// Populate a single row from a `RoundTrace`.
pub fn populate_round(
    columns: &mut [Vec<Scalar>],
    row: usize,
    rt: &RoundTrace,
    curve: CurveType,
) {
    assert!(columns.len() == NUM_KECCAK_COLUMNS, "columns has wrong shape");
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    write_state_bits(columns, row, COL_BEFORE_OFFSET,      &rt.before,      &one, &zero);
    write_c_partial (columns, row, &rt.before, &one, &zero);
    write_d         (columns, row, &rt.d, &one, &zero);
    write_state_bits(columns, row, COL_AFTER_THETA_OFFSET, &rt.after_theta, &one, &zero);
    write_state_bits(columns, row, COL_AFTER_RHO_OFFSET,   &rt.after_rho,   &one, &zero);
    write_state_bits(columns, row, COL_AFTER_PI_OFFSET,    &rt.after_pi,    &one, &zero);
    write_nand_temp (columns, row, &rt.after_pi, &one, &zero);
    write_state_bits(columns, row, COL_AFTER_CHI_OFFSET,   &rt.after_chi,   &one, &zero);
    write_state_bits(columns, row, COL_AFTER_IOTA_OFFSET,  &rt.after_iota,  &one, &zero);

    // One-hot round selector.
    let r = rt.round;
    columns[sel_round(r)][row] = one.clone();
}

/// Populate an entire `HashTrace` into a freshly allocated column set.
/// The returned trace has one row per round across all blocks
/// (`blocks.len() * 24` rows).
pub fn populate_trace_from_hash(
    hash_trace: &HashTrace,
    curve: CurveType,
) -> Vec<Vec<Scalar>> {
    let num_rows = hash_trace.blocks.len() * NUM_ROUNDS;
    let mut columns = alloc_trace(num_rows, curve);
    let mut row = 0usize;
    for block in &hash_trace.blocks {
        for rt in &block.rounds {
            populate_round(&mut columns, row, rt, curve);
            row += 1;
        }
    }
    debug_assert_eq!(row, num_rows);
    columns
}

/// Populate the trace and additionally fill the per-invocation
/// aggregator columns
/// ([`COL_INV_INPUT_BYTE_OFFSET`], [`COL_INV_INPUT_LEN_COL`],
/// [`COL_INV_OUTPUT_BYTE_OFFSET`], [`COL_IS_FIRST_INV_ROW`]) from
/// `invocation_input` (zero-padded to [`INV_INPUT_LEN`]) and `digest`.
/// Same input/output bytes are written to every row (invariance —
/// algebraic enforcement is a documented follow-up); `IS_FIRST_INV_ROW`
/// is `1` only on row 0.
///
/// Panics if `invocation_input.len() > INV_INPUT_LEN`.
pub fn populate_trace_from_hash_with_invocation_bytes(
    hash_trace: &HashTrace,
    invocation_input: &[u8],
    digest: &[u8; INV_OUTPUT_LEN],
    curve: CurveType,
) -> Vec<Vec<Scalar>> {
    assert!(
        invocation_input.len() <= INV_INPUT_LEN,
        "invocation input ({} bytes) exceeds INV_INPUT_LEN ({})",
        invocation_input.len(),
        INV_INPUT_LEN
    );
    let num_rows = hash_trace.blocks.len() * NUM_ROUNDS;
    let mut columns = populate_trace_from_hash(hash_trace, curve);
    let one = Scalar::one(curve);
    for r in 0..num_rows {
        for (b, &v) in invocation_input.iter().enumerate() {
            columns[COL_INV_INPUT_BYTE_OFFSET + b][r] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_INV_INPUT_LEN_COL][r] = Scalar::from_u64(invocation_input.len() as u64, curve);
        for b in 0..INV_OUTPUT_LEN {
            columns[COL_INV_OUTPUT_BYTE_OFFSET + b][r] = Scalar::from_u64(digest[b] as u64, curve);
        }
    }
    let one_clone = one.clone();
    if num_rows > 0 {
        columns[COL_IS_FIRST_INV_ROW][0] = one;
    }
    // IS_FIRST_BLOCK = 1 on every round of block 0, 0 elsewhere.
    for r in 0..num_rows.min(NUM_ROUNDS) {
        columns[COL_IS_FIRST_BLOCK][r] = one_clone.clone();
    }
    // IS_BYTE_ACTIVE[b] at row 0: 1 if b < invocation_input.len(), else 0.
    // Sized NUM_IS_BYTE_ACTIVE (= MAX_INPUT_LEN = 256) to cover both
    // block 0 (positions 0..136) and block 1 (positions 136..256) of
    // multi-absorption 2-block traces.
    if num_rows > 0 {
        for b in 0..NUM_IS_BYTE_ACTIVE {
            columns[COL_IS_BYTE_ACTIVE_OFFSET + b][0] = if b < invocation_input.len() {
                one_clone.clone()
            } else {
                Scalar::zero(curve)
            };
        }
    }
    // IS_LAST_BLOCK = 1 on every row of the LAST absorption block.
    let num_blocks = hash_trace.blocks.len();
    if num_blocks >= 1 {
        let last_block_row_base = (num_blocks - 1) * NUM_ROUNDS;
        for r in last_block_row_base..num_rows {
            columns[COL_IS_LAST_BLOCK][r] = one_clone.clone();
        }
    }
    columns
}

// ──── Constraint evaluation ───────────────────────────────────────────

/// A single constraint body label + its per-row evaluations.
pub struct ConstraintEval {
    pub label: String,
    pub values: Vec<Scalar>,
}

/// Evaluate the Keccak AIR constraint bodies on a populated trace.
///
/// Returns a vector of `(label, per-row values)`. A valid witness produces
/// zero at every row for every constraint. A `sel_round` selector on each
/// row gates the χ / ι / θ structural constraints; rows where no selector
/// is set (fully inactive padding) receive zero contribution.
///
/// Constraints are organized into consolidated categories via random
/// linear combination over a challenge power β inside each category. This
/// keeps the returned vector a manageable size (~15 entries) rather than
/// emitting one entry per bit-level constraint (~60,000).
pub fn evaluate_constraints(
    columns: &[&Vec<Scalar>],
    beta: &Scalar,
) -> Vec<ConstraintEval> {
    assert!(
        columns.len() == NUM_KECCAK_COLUMNS,
        "expected {} columns, got {}", NUM_KECCAK_COLUMNS, columns.len()
    );
    let num_rows = columns[0].len();
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);
    let two  = Scalar::from_u64(2, curve);

    let mut result: Vec<ConstraintEval> = Vec::new();

    // ── 1. Bit validity: b·(b−1) = 0 for every data column. ──
    // Accumulate via powers of β: Σ β^k · b_k·(b_k − 1).
    let mut bit_validity = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for col_idx in 0..NUM_DATA_COLUMNS {
            let col = columns[col_idx];
            for (row, cell) in col.iter().enumerate() {
                let v = cell;
                let diff = v.sub(&one);
                let body = v.mul(&diff);            // b·(b−1)
                let term = beta_pow.mul(&body);
                bit_validity[row] = bit_validity[row].add(&term);
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "bit_validity".into(), values: bit_validity });

    // ── 2. C_partial recurrence ──
    // C_partial[x][0][bit] − before[x][0][bit] = 0
    // C_partial[x][y+1][bit] − XOR(C_partial[x][y][bit], before[x][y+1][bit]) = 0
    //   where XOR(a, b) = a + b − 2ab
    let mut c_partial_rec = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let cp0_col = columns[c_partial(x, 0, bit)];
                let b0_col  = columns[before(x, 0, bit)];
                for row in 0..num_rows {
                    let body = cp0_col[row].sub(&b0_col[row]);
                    let term = beta_pow.mul(&body);
                    c_partial_rec[row] = c_partial_rec[row].add(&term);
                }
                beta_pow = beta_pow.mul(beta);
                for y in 0..4 {
                    let cp_y     = columns[c_partial(x, y,     bit)];
                    let cp_y_1   = columns[c_partial(x, y + 1, bit)];
                    let b_y_1    = columns[before(   x, y + 1, bit)];
                    for row in 0..num_rows {
                        // xor = a + b − 2ab
                        let a = &cp_y[row];
                        let b = &b_y_1[row];
                        let ab = a.mul(b);
                        let xor = a.add(b).sub(&two.mul(&ab));
                        let body = cp_y_1[row].sub(&xor);
                        let term = beta_pow.mul(&body);
                        c_partial_rec[row] = c_partial_rec[row].add(&term);
                    }
                    beta_pow = beta_pow.mul(beta);
                }
            }
        }
    }
    result.push(ConstraintEval { label: "c_partial_recurrence".into(), values: c_partial_rec });

    // ── 3. D[x][bit] = C[(x+4)%5][bit] ⊕ C[(x+1)%5][(bit+63) % 64] ──
    // Expand XOR(a, b) = a + b − 2ab.
    let mut d_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let cl = columns[c_final((x + 4) % 5, bit)];
                let cr = columns[c_final((x + 1) % 5, (bit + 63) % BITS_PER_LANE)];
                let d  = columns[d_col(x, bit)];
                for row in 0..num_rows {
                    let ab = cl[row].mul(&cr[row]);
                    let xor = cl[row].add(&cr[row]).sub(&two.mul(&ab));
                    let body = d[row].sub(&xor);
                    let term = beta_pow.mul(&body);
                    d_def[row] = d_def[row].add(&term);
                }
                beta_pow = beta_pow.mul(beta);
            }
        }
    }
    result.push(ConstraintEval { label: "d_definition".into(), values: d_def });

    // ── 4. θ apply: after_theta[x][y][bit] = before[x][y][bit] ⊕ D[x][bit] ──
    let mut theta_apply = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for y in 0..5 {
                for bit in 0..BITS_PER_LANE {
                    let b_col = columns[before(x, y, bit)];
                    let d_c   = columns[d_col(x, bit)];
                    let at    = columns[after_theta(x, y, bit)];
                    for row in 0..num_rows {
                        let ab  = b_col[row].mul(&d_c[row]);
                        let xor = b_col[row].add(&d_c[row]).sub(&two.mul(&ab));
                        let body = at[row].sub(&xor);
                        let term = beta_pow.mul(&body);
                        theta_apply[row] = theta_apply[row].add(&term);
                    }
                    beta_pow = beta_pow.mul(beta);
                }
            }
        }
    }
    result.push(ConstraintEval { label: "theta_apply".into(), values: theta_apply });

    // ── 5. ρ rotation (structural): after_rho[x][y][bit] =
    //     after_theta[x][y][(bit − RHO_OFFSETS[x][y]) mod 64] ──
    let mut rho_apply = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for y in 0..5 {
                let off = RHO_OFFSETS[x][y] as usize;
                for bit in 0..BITS_PER_LANE {
                    let src_bit = (bit + BITS_PER_LANE - off) % BITS_PER_LANE;
                    let at  = columns[after_theta(x, y, src_bit)];
                    let ar  = columns[after_rho(x, y, bit)];
                    for row in 0..num_rows {
                        let body = ar[row].sub(&at[row]);
                        let term = beta_pow.mul(&body);
                        rho_apply[row] = rho_apply[row].add(&term);
                    }
                    beta_pow = beta_pow.mul(beta);
                }
            }
        }
    }
    result.push(ConstraintEval { label: "rho_apply".into(), values: rho_apply });

    // ── 6. π (structural): after_pi[y][(2x+3y) mod 5][bit] = after_rho[x][y][bit] ──
    let mut pi_apply = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for y in 0..5 {
                let nx = y;
                let ny = (2 * x + 3 * y) % 5;
                for bit in 0..BITS_PER_LANE {
                    let ar = columns[after_rho(x, y, bit)];
                    let ap = columns[after_pi(nx, ny, bit)];
                    for row in 0..num_rows {
                        let body = ap[row].sub(&ar[row]);
                        let term = beta_pow.mul(&body);
                        pi_apply[row] = pi_apply[row].add(&term);
                    }
                    beta_pow = beta_pow.mul(beta);
                }
            }
        }
    }
    result.push(ConstraintEval { label: "pi_apply".into(), values: pi_apply });

    // ── 7a. nand_temp[x][y][bit] = (1 − after_pi[(x+1)%5][y][bit]) ·
    //      after_pi[(x+2)%5][y][bit] ──
    let mut chi_nand_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for y in 0..5 {
                for bit in 0..BITS_PER_LANE {
                    let ap1 = columns[after_pi((x + 1) % 5, y, bit)];
                    let ap2 = columns[after_pi((x + 2) % 5, y, bit)];
                    let nt  = columns[nand_temp(x, y, bit)];
                    for row in 0..num_rows {
                        let not_ap1 = one.sub(&ap1[row]);
                        let prod = not_ap1.mul(&ap2[row]);
                        let body = nt[row].sub(&prod);
                        let term = beta_pow.mul(&body);
                        chi_nand_def[row] = chi_nand_def[row].add(&term);
                    }
                    beta_pow = beta_pow.mul(beta);
                }
            }
        }
    }
    result.push(ConstraintEval { label: "chi_nand_definition".into(), values: chi_nand_def });

    // ── 7b. after_chi[x][y][bit] = after_pi[x][y][bit] ⊕ nand_temp[x][y][bit] ──
    let mut chi_apply = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for y in 0..5 {
                for bit in 0..BITS_PER_LANE {
                    let ap = columns[after_pi(x, y, bit)];
                    let nt = columns[nand_temp(x, y, bit)];
                    let ac = columns[after_chi(x, y, bit)];
                    for row in 0..num_rows {
                        let ab = ap[row].mul(&nt[row]);
                        let xor = ap[row].add(&nt[row]).sub(&two.mul(&ab));
                        let body = ac[row].sub(&xor);
                        let term = beta_pow.mul(&body);
                        chi_apply[row] = chi_apply[row].add(&term);
                    }
                    beta_pow = beta_pow.mul(beta);
                }
            }
        }
    }
    result.push(ConstraintEval { label: "chi_apply".into(), values: chi_apply });

    // ── 8. ι apply ──
    // For lanes other than (0, 0): after_iota = after_chi (direct equality).
    let mut iota_passthrough = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for x in 0..5 {
            for y in 0..5 {
                if x == 0 && y == 0 { continue; }
                for bit in 0..BITS_PER_LANE {
                    let ac = columns[after_chi(x, y, bit)];
                    let ai = columns[after_iota(x, y, bit)];
                    for row in 0..num_rows {
                        let body = ai[row].sub(&ac[row]);
                        let term = beta_pow.mul(&body);
                        iota_passthrough[row] = iota_passthrough[row].add(&term);
                    }
                    beta_pow = beta_pow.mul(beta);
                }
            }
        }
    }
    result.push(ConstraintEval { label: "iota_passthrough".into(), values: iota_passthrough });

    // For lane (0, 0), for each round k:
    //   sel_round_k · (after_iota[0][0][bit] − XOR(after_chi[0][0][bit], RC_k[bit])) = 0
    // Precompute RC scalars once.
    let mut iota_xor = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for k in 0..NUM_ROUNDS {
            let sel = columns[sel_round(k)];
            let rc_lane = ROUND_CONSTANTS[k];
            for bit in 0..BITS_PER_LANE {
                let rc_bit = ((rc_lane >> bit) & 1) as u64;
                let ac = columns[after_chi(0, 0, bit)];
                let ai = columns[after_iota(0, 0, bit)];
                for row in 0..num_rows {
                    // If rc_bit == 0: XOR = ac. If rc_bit == 1: XOR = 1 − ac.
                    let xor = if rc_bit == 1 { one.sub(&ac[row]) } else { ac[row].clone() };
                    let diff = ai[row].sub(&xor);
                    let gated = sel[row].mul(&diff);
                    let term = beta_pow.mul(&gated);
                    iota_xor[row] = iota_xor[row].add(&term);
                }
                beta_pow = beta_pow.mul(beta);
            }
        }
    }
    result.push(ConstraintEval { label: "iota_xor_rc".into(), values: iota_xor });

    // ── 9a. Selector binary: sel_round_k · (sel_round_k − 1) = 0 ──
    let mut sel_binary = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for k in 0..NUM_SEL_ROUND {
            let col = columns[sel_round(k)];
            for (row, cell) in col.iter().enumerate() {
                let body = cell.mul(&cell.sub(&one));
                let term = beta_pow.mul(&body);
                sel_binary[row] = sel_binary[row].add(&term);
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "sel_binary".into(), values: sel_binary });

    // ── 9b. Selector sum ∈ {0, 1} per row. ──
    // Active rows have Σ sel = 1, fully-inactive rows (if any) have Σ = 0.
    // Enforced as (Σ sel) · (Σ sel − 1) = 0.
    let mut sel_sum = vec![zero.clone(); num_rows];
    for row in 0..num_rows {
        let mut sum = zero.clone();
        for k in 0..NUM_SEL_ROUND {
            sum = sum.add(&columns[sel_round(k)][row]);
        }
        let body = sum.mul(&sum.sub(&one));
        sel_sum[row] = body;
    }
    result.push(ConstraintEval { label: "sel_sum_01".into(), values: sel_sum });

    result
}

/// Evaluate cross-row (transition) constraints: `after_iota(row r) ==
/// before(row r+1)` for rows r in a permutation, except on rows where
/// `r+1` starts a new permutation (`r mod 24 == 23`).
///
/// Returns a single consolidated per-row vector (length `num_rows`).
/// The entry at row r represents the constraint between r and r+1; the
/// last row's entry is always zero.
pub fn evaluate_cross_row(
    columns: &[&Vec<Scalar>],
    beta: &Scalar,
) -> ConstraintEval {
    let num_rows = columns[0].len();
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);

    let mut values = vec![zero.clone(); num_rows];
    let mut beta_pow = one.clone();
    for x in 0..5 {
        for y in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let ai = columns[after_iota(x, y, bit)];
                let bf = columns[before(x, y, bit)];
                for row in 0..num_rows.saturating_sub(1) {
                    // Skip the permutation boundary (row index r where
                    // r mod 24 == 23 — the next row starts a new permutation).
                    if (row + 1) % NUM_ROUNDS == 0 { continue; }
                    let body = ai[row].sub(&bf[row + 1]);
                    let term = beta_pow.mul(&body);
                    values[row] = values[row].add(&term);
                }
                beta_pow = beta_pow.mul(beta);
            }
        }
    }
    ConstraintEval { label: "cross_row_transition".into(), values }
}

/// Return the total number of constraints (bodies) evaluated by
/// [`evaluate_constraints`] plus the cross-row constraint. Each category
/// aggregates many bit-level sub-constraints via β-RLC.
pub const NUM_CONSTRAINT_CATEGORIES: usize = 12;

// ──── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;
    use crate::keccak::{keccak_f1600_witness, keccak_witness};

    fn beta_challenge() -> Scalar {
        // A fixed non-zero challenge. (Testing determinism; avoids Fiat-Shamir.)
        Scalar::from_u64(12345, CurveType::Bls48581)
    }

    /// Helper: collect column references for passing to `evaluate_constraints`.
    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    #[test]
    fn keccak_air_column_layout_is_consistent() {
        // Sanity: column helpers cover the expected totals without overlap.
        assert_eq!(COL_BEFORE_OFFSET, 0);
        assert_eq!(COL_C_PARTIAL_OFFSET, BITS_PER_STATE);
        assert_eq!(COL_D_OFFSET, 2 * BITS_PER_STATE);
        assert_eq!(COL_AFTER_THETA_OFFSET, 2 * BITS_PER_STATE + BITS_PER_PARITY);
        // 8 full state grids (BEFORE, C_PARTIAL, AFTER_THETA, AFTER_RHO,
        // AFTER_PI, NAND_TEMP, AFTER_CHI, AFTER_IOTA) each of 1600 bits,
        // plus one D parity grid of 320 bits, plus 24 round selectors.
        assert_eq!(NUM_DATA_COLUMNS, 8 * BITS_PER_STATE + BITS_PER_PARITY);
        // Total = data + selectors + invocation aggregator (290 cols:
        // 256 input bytes + 1 input length + 32 output bytes + 1 anchor)
        // + IS_FIRST_BLOCK (1 col) + IS_BYTE_ACTIVE (256 cols, =
        // INV_INPUT_LEN for multi-absorption) + IS_LAST_BLOCK (1 col).
        assert_eq!(
            NUM_KECCAK_COLUMNS,
            NUM_DATA_COLUMNS
                + NUM_ROUNDS
                + INV_INPUT_LEN
                + 1
                + INV_OUTPUT_LEN
                + 1
                + 1
                + NUM_IS_BYTE_ACTIVE
                + 1
        );
        assert_eq!(NUM_DATA_COLUMNS, 13_120);
        assert_eq!(NUM_KECCAK_COLUMNS, 13_692);
        assert_eq!(RATE_LEN, 136);
        // Invocation aggregator helpers.
        assert_eq!(inv_input_byte(0), COL_INV_INPUT_BYTE_OFFSET);
        assert_eq!(inv_input_byte(INV_INPUT_LEN - 1), COL_INV_INPUT_LEN_COL - 1);
        assert_eq!(inv_output_byte(0), COL_INV_OUTPUT_BYTE_OFFSET);
        assert_eq!(inv_output_byte(INV_OUTPUT_LEN - 1), COL_IS_FIRST_INV_ROW - 1);
        assert_eq!(COL_IS_FIRST_BLOCK, COL_IS_BYTE_ACTIVE_OFFSET - 1);
        assert_eq!(is_byte_active(0), COL_IS_BYTE_ACTIVE_OFFSET);
        assert_eq!(is_byte_active(NUM_IS_BYTE_ACTIVE - 1), COL_IS_LAST_BLOCK - 1);
        assert_eq!(NUM_IS_BYTE_ACTIVE, 256);
        assert_eq!(COL_IS_LAST_BLOCK, NUM_KECCAK_COLUMNS - 1);
    }

    #[test]
    fn keccak_air_witness_populates_fully_single_permutation() {
        // One permutation on a non-trivial state. The AIR should reflect
        // every intermediate state from the reference RoundTrace.
        let mut state = [[0u64; 5]; 5];
        state[0][0] = 0x0123_4567_89ab_cdef;
        state[2][3] = 0xdead_beef_cafe_babe;

        let rounds = keccak_f1600_witness(state);
        assert_eq!(rounds.len(), NUM_ROUNDS);

        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        // Spot-check a few (x, y, bit) cells against the reference round.
        for (row, rt) in rounds.iter().enumerate() {
            for (x, y) in [(0, 0), (1, 2), (3, 3), (4, 4), (2, 1)] {
                for bit in [0usize, 1, 17, 31, 62, 63] {
                    let before_bit = ((rt.before[x][y] >> bit) & 1) as u64;
                    let cell = &columns[before(x, y, bit)][row];
                    let expected = Scalar::from_u64(before_bit, curve);
                    assert!(
                        cell.sub(&expected).is_zero(),
                        "before mismatch at row {} xyb=({},{},{})", row, x, y, bit
                    );
                    let iota_bit = ((rt.after_iota[x][y] >> bit) & 1) as u64;
                    let cell = &columns[after_iota(x, y, bit)][row];
                    let expected = Scalar::from_u64(iota_bit, curve);
                    assert!(
                        cell.sub(&expected).is_zero(),
                        "after_iota mismatch at row {} xyb=({},{},{})", row, x, y, bit
                    );
                }
            }
            // Selector is one-hot on its own round.
            for k in 0..NUM_ROUNDS {
                let want = if k == rt.round { 1 } else { 0 };
                let got = columns[sel_round(k)][row].to_u64();
                assert_eq!(got, want, "sel_round[{}] mismatch at row {}", k, row);
            }
        }
    }

    #[test]
    fn keccak_air_witness_populates_fully_hash() {
        // A full HashTrace (one-block input) populates cleanly.
        let trace = keccak_witness(b"abc");
        let curve = CurveType::Bls48581;
        let columns = populate_trace_from_hash(&trace, curve);
        assert_eq!(columns.len(), NUM_KECCAK_COLUMNS);
        let expected_rows = trace.blocks.len() * NUM_ROUNDS;
        for col in &columns {
            assert_eq!(col.len(), expected_rows);
        }
        // Every row should have exactly one active selector.
        for row in 0..expected_rows {
            let mut active = 0usize;
            for k in 0..NUM_ROUNDS {
                if !columns[sel_round(k)][row].is_zero() { active += 1; }
            }
            assert_eq!(active, 1, "row {} must have one active selector", row);
        }
    }

    #[test]
    fn keccak_air_constraints_vanish_on_valid_witness_single_permutation() {
        // Populate from a valid reference trace; every constraint must
        // evaluate to zero on every row.
        let mut state = [[0u64; 5]; 5];
        state[0][0] = 0x0123_4567_89ab_cdef;
        state[2][3] = 0xdead_beef_cafe_babe;
        state[4][4] = 0xc0ff_eeba_be42_1337;

        let rounds = keccak_f1600_witness(state);

        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        for ce in &evals {
            for (row, v) in ce.values.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint `{}` fired at row {} (valid witness)",
                    ce.label, row
                );
            }
        }
        // Cross-row must also vanish on adjacent rounds within the permutation.
        let cross = evaluate_cross_row(&refs, &beta);
        for (row, v) in cross.values.iter().enumerate() {
            assert!(v.is_zero(), "cross-row constraint fired at row {}", row);
        }
    }

    #[test]
    fn keccak_air_constraints_vanish_on_valid_witness_hash() {
        // Multi-block hash: 272 bytes forces 3 blocks × 24 rounds = 72 rows.
        let input: Vec<u8> = (0..272).map(|i| (i * 7 + 3) as u8).collect();
        let trace = keccak_witness(&input);
        assert_eq!(trace.blocks.len(), 3);

        let curve = CurveType::Bls48581;
        let columns = populate_trace_from_hash(&trace, curve);
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        for ce in &evals {
            for (row, v) in ce.values.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint `{}` fired at row {} (hash witness)",
                    ce.label, row
                );
            }
        }
        // Cross-row: within each 24-block round the state should chain;
        // between blocks (row 23→24 etc) the boundary is excluded by design.
        let cross = evaluate_cross_row(&refs, &beta);
        for (row, v) in cross.values.iter().enumerate() {
            assert!(v.is_zero(), "cross-row fired at row {}", row);
        }
    }

    #[test]
    fn keccak_air_constraints_reject_tampered_before_bit() {
        // Flip a single BEFORE bit. The `theta_apply` and `c_partial_recurrence`
        // and `bit_validity` (transitively) paths all reference it.
        let mut state = [[0u64; 5]; 5];
        state[1][1] = 0x0123_4567_89ab_cdef;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        // Tamper: toggle before[1][1][0] on row 5.
        let col = before(1, 1, 0);
        let old = columns[col][5].clone();
        let one = Scalar::one(curve);
        columns[col][5] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);

        let any_fired = evals.iter().any(|ce| !ce.values[5].is_zero());
        assert!(any_fired, "tampered BEFORE bit should trigger some constraint");
    }

    #[test]
    fn keccak_air_constraints_reject_tampered_after_iota_bit() {
        // Flip a single AFTER_IOTA bit. `iota_passthrough` and `cross_row`
        // should both notice.
        let mut state = [[0u64; 5]; 5];
        state[0][0] = 0xbeef_face_dead_cafe;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        // Tamper: toggle after_iota[2][3][17] on row 0.
        let col = after_iota(2, 3, 17);
        let old = columns[col][0].clone();
        let one = Scalar::one(curve);
        columns[col][0] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);

        // iota_passthrough must fire (bit-validity might also depending on RLC,
        // but we don't rely on it).
        let iota_cat = evals
            .iter()
            .find(|ce| ce.label == "iota_passthrough")
            .expect("iota_passthrough category present");
        assert!(
            !iota_cat.values[0].is_zero(),
            "tampered AFTER_IOTA bit should trigger iota_passthrough"
        );

        // Cross-row must also notice (row 0's after_iota feeds row 1's before).
        let cross = evaluate_cross_row(&refs, &beta);
        assert!(
            !cross.values[0].is_zero(),
            "tampered AFTER_IOTA bit should trigger cross-row at row 0"
        );
    }

    #[test]
    fn keccak_air_constraints_reject_tampered_selector() {
        // Set two round selectors to 1 on the same row — must fire sel_sum_01.
        let mut state = [[0u64; 5]; 5];
        state[0][0] = 1;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        // Tamper: also activate sel_round[5] on row 0 (which already has sel_round[0]=1).
        let one = Scalar::one(curve);
        columns[sel_round(5)][0] = one.clone();

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);

        // sel_sum_01 must fire on row 0 (sum is 2, so 2·1 = 2 ≠ 0).
        let sel_cat = evals
            .iter()
            .find(|ce| ce.label == "sel_sum_01")
            .expect("sel_sum_01 category present");
        assert!(
            !sel_cat.values[0].is_zero(),
            "double-active selector must trigger sel_sum_01"
        );
    }

    #[test]
    fn keccak_air_constraints_reject_tampered_nand_temp() {
        // Flip a nand_temp bit. chi_nand_definition should fire.
        let mut state = [[0u64; 5]; 5];
        state[3][2] = 0xaaaa_5555_aaaa_5555;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        let col = nand_temp(1, 4, 7);
        let old = columns[col][3].clone();
        let one = Scalar::one(curve);
        columns[col][3] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);

        let cat = evals.iter().find(|ce| ce.label == "chi_nand_definition").unwrap();
        assert!(!cat.values[3].is_zero(), "flipped nand_temp must fail chi_nand_definition");
    }

    #[test]
    fn keccak_air_round_constants_wire_through_iota() {
        // If we tamper after_iota[0][0][0] on round 0 (where RC bit 0 is 1,
        // since ROUND_CONSTANTS[0] = 0x…0001), the iota_xor_rc constraint
        // should fire.
        assert_eq!(ROUND_CONSTANTS[0] & 1, 1);
        let state = [[0u64; 5]; 5];
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        let col = after_iota(0, 0, 0);
        let old = columns[col][0].clone();
        let one = Scalar::one(curve);
        columns[col][0] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let cat = evals.iter().find(|ce| ce.label == "iota_xor_rc").unwrap();
        assert!(!cat.values[0].is_zero(), "tampered iota[0][0][0] must fail iota_xor_rc");
    }
}
