//! ModExp internals AIR (skeleton).
//!
//! # Purpose
//!
//! Bridges the [`crate::modexp_precompile_air`] (which commits the
//! single-row witness `B^E mod M = output` taken on faith) to the actual
//! algebraic sliding-window modular exponentiation.
//!
//! # Algorithm shape (sliding window exponentiation)
//!
//! ```text
//!   result = 1
//!   for each chunk of WINDOW_BITS bits of E (high-to-low):
//!     # square WINDOW_BITS times
//!     for _ in 0..WINDOW_BITS:
//!       result = (result * result) mod M
//!     # conditional multiply by precomputed table[bits]
//!     if window_bits > 0:
//!       result = (result * table[window_bits]) mod M
//! ```
//!
//! In a constraint trace this is a tight loop of square-and-multiply
//! iterations. Each iteration commits the partial result before/after
//! the square+optional-multiply.
//!
//! # AIR shape (this skeleton)
//!
//! One TRACE ROW per iteration of the sliding window. Each row commits:
//!
//!   * b_limbs[LIMBS × LIMB_BYTES]      — base B (BE byte layout).
//!   * e_limbs[LIMBS × LIMB_BYTES]      — exponent E.
//!   * m_limbs[LIMBS × LIMB_BYTES]      — modulus M.
//!   * partial_result[LIMBS × LIMB_BYTES] — running result before this
//!     iteration's square+multiply.
//!   * next_partial_result[LIMBS × LIMB_BYTES] — running result after.
//!   * current_window_bits[WINDOW_BITS] — bits of E consumed this
//!     iteration (one bit per column).
//!   * iteration_index ∈ [0, ITERATIONS]
//!   * is_real, is_result_row ∈ {0, 1}.
//!   * result[LIMBS × LIMB_BYTES]        — finalized `B^E mod M`, exposed
//!     only on the FINAL row.
//!
//! # Soundness
//!
//! This AIR is a **skeleton**. Algebraic big-int multiplication / modular
//! reduction is **deferred**. What this AIR DOES bind:
//!
//!   - `is_real`, `is_result_row` binary.
//!   - Each `current_window_bits[k]` ∈ {0, 1}.
//!   - LE byte decomposition of `iteration_index_word`.
//!   - `result[i]` zero on non-final rows (gated).
//!   - Cross-row state continuity: `next_partial_result(X)` =
//!     `partial_result(ω·X)` when both rows are real and X is not the
//!     result row.
//!
//! What this AIR does NOT bind (deferred):
//!
//!   - Algebraic `next_partial_result = partial_result^2 * B^window mod M`.
//!   - `B`, `E`, `M` constancy across iterations of one invocation.
//!   - Initial partial_result = 1.
//!   - Bit-decomposition of `E` ↔ window bits (Schwartz-Zippel chain).
//!
//! # Cross-AIR LogUp descriptors
//!
//! [`make_modexp_internals_to_precompile_descriptor`] binds the
//! finalized `result[0..LIMBS*LIMB_BYTES]` on the FINAL row of this AIR
//! to the `output` column of [`crate::modexp_precompile_air`].

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum big-int byte length supported (matches precompile AIR).
pub const BIGINT_LENGTH: usize = 64;
/// Sliding window width (bits per window).
pub const WINDOW_BITS: usize = 4;
/// LE byte decomposition width for `iteration_index_word` (u32).
pub const ITERATION_INDEX_BYTES: usize = 4;
/// Result byte length (equals BIGINT_LENGTH — `B^E mod M` fits in
/// the same buffer).
pub const RESULT_LENGTH: usize = BIGINT_LENGTH;

/// EIP precompile id for ModExp.
pub const PC_MODEXP: u64 = 0x05;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_B_BYTES_OFFSET: usize = 0; // 0..64
pub const COL_E_BYTES_OFFSET: usize = COL_B_BYTES_OFFSET + BIGINT_LENGTH; // 64..128
pub const COL_M_BYTES_OFFSET: usize = COL_E_BYTES_OFFSET + BIGINT_LENGTH; // 128..192
pub const COL_PARTIAL_RESULT_OFFSET: usize =
    COL_M_BYTES_OFFSET + BIGINT_LENGTH; // 192..256
pub const COL_NEXT_PARTIAL_RESULT_OFFSET: usize =
    COL_PARTIAL_RESULT_OFFSET + BIGINT_LENGTH; // 256..320
pub const COL_RESULT_OFFSET: usize =
    COL_NEXT_PARTIAL_RESULT_OFFSET + BIGINT_LENGTH; // 320..384
pub const COL_CURRENT_WINDOW_BITS_OFFSET: usize =
    COL_RESULT_OFFSET + RESULT_LENGTH; // 384..388
pub const COL_ITERATION_INDEX: usize =
    COL_CURRENT_WINDOW_BITS_OFFSET + WINDOW_BITS; // 388
pub const COL_ITERATION_INDEX_BYTE_OFFSET: usize = COL_ITERATION_INDEX + 1; // 389..393
pub const COL_IS_RESULT_ROW: usize =
    COL_ITERATION_INDEX_BYTE_OFFSET + ITERATION_INDEX_BYTES; // 393
pub const COL_IS_REAL: usize = COL_IS_RESULT_ROW + 1; // 394

// ─── Mul-mod gadget columns (256-bit, 4×4 limb schoolbook) ────────────
//
// The mul-mod gadget on a row binds:
//   (A * B) mod M_lo = R   where M_lo is the low 256 bits of `m_bytes`.
//
// Encoding: each multiprecision value is committed as 4 × 64-bit
// little-endian limbs (A_lo = limb 0, A_hi = limb 3). Limb↔byte binding
// to the existing `b_bytes`/`m_bytes`/`partial_result` byte columns is
// DEFERRED — at this layer we treat the limb columns as an independent
// commitment used purely to validate the multiplicative identity.
//
// Identity (Barrett-style):
//   A * B = Q * M_lo + R    (over integers, with R < M_lo)
//
// Encoded limb-wise on radix 2^64. The AB product has 8 limbs (col
// sums of partial products with carry propagation); Q * M_lo likewise
// has 8 limbs. The low 4 limbs of (AB - QM) must equal R (with 4 carry
// cols across the low half); the high 4 limbs must vanish (4 carry
// cols across the high half).
//
// Range checks: Q < 2^256 and R < M_lo are DEFERRED (require 64-bit
// range arguments). The byte-level range checks already in place on
// `b_bytes`/etc. cover the legacy skeleton.
pub const MULMOD_LIMBS: usize = 4; // 4 × 64-bit limbs = 256 bits
pub const MULMOD_PP: usize = MULMOD_LIMBS * MULMOD_LIMBS; // 16

pub const COL_MM_A_LIMBS: usize = COL_IS_REAL + 1; // 395..399
pub const COL_MM_B_LIMBS: usize = COL_MM_A_LIMBS + MULMOD_LIMBS; // 399..403
pub const COL_MM_M_LO_LIMBS: usize = COL_MM_B_LIMBS + MULMOD_LIMBS; // 403..407
pub const COL_MM_Q_LIMBS: usize = COL_MM_M_LO_LIMBS + MULMOD_LIMBS; // 407..411
pub const COL_MM_R_LIMBS: usize = COL_MM_Q_LIMBS + MULMOD_LIMBS; // 411..415
pub const COL_MM_PP_AB: usize = COL_MM_R_LIMBS + MULMOD_LIMBS; // 415..431
pub const COL_MM_PP_QM: usize = COL_MM_PP_AB + MULMOD_PP; // 431..447
pub const COL_MM_AB_OUT: usize = COL_MM_PP_QM + MULMOD_PP; // 447..455 (8 limbs)
pub const COL_MM_AB_CARRY: usize = COL_MM_AB_OUT + 2 * MULMOD_LIMBS; // 455..462 (7 carries)
pub const COL_MM_QM_OUT: usize = COL_MM_AB_CARRY + (2 * MULMOD_LIMBS - 1); // 462..470 (8 limbs)
pub const COL_MM_QM_CARRY: usize = COL_MM_QM_OUT + 2 * MULMOD_LIMBS; // 470..477 (7 carries)
pub const COL_MM_ID_CARRY_LO: usize = COL_MM_QM_CARRY + (2 * MULMOD_LIMBS - 1); // 477..481 (4 carries)
pub const COL_MM_ID_CARRY_HI: usize = COL_MM_ID_CARRY_LO + MULMOD_LIMBS; // 481..485 (4 carries)
pub const COL_IS_MULMOD: usize = COL_MM_ID_CARRY_HI + MULMOD_LIMBS; // 485

// ─── Task #182: range-check + R<M uniqueness columns ──────────────────
//
// Bytes-per-limb LE decomposition columns. Each u64 limb is bound to
// 8 byte cells via Σ byte[k]·2^(8k) = limb. Byte cells are then range-
// checked against the 8-bit table (declared in lookup_declarations).
//
// Scope (time-boxed):
//   - r_limbs    → 32 byte cols (most important: R must be < 2^256)
//   - q_limbs    → 32 byte cols (Q range check)
//   - slack_low  → 1 u64 + 8 byte cols (R<M low-limb uniqueness check)
//
// Deferred (documented in module-level comments):
//   - a/b/m_lo limb byte cols (these mirror byte cols on the outer
//     legacy b/m byte side, but the algebraic link is not yet drawn).
//   - ab_carry / qm_carry 9-byte decomp (~70-bit range).
//   - id_carry_lo / id_carry_hi signed range check.
//   - 16-byte decomp of partial products.
//   - Full multi-limb R<M check (currently only the low u64 limb is
//     uniqueness-bound; high limbs equal-or-less is deferred until
//     comparison gadget lands).
pub const BYTES_PER_LIMB: usize = 8;

pub const COL_MM_R_LIMB_BYTES: usize = COL_IS_MULMOD + 1; // 486..518 (32 bytes)
pub const COL_MM_Q_LIMB_BYTES: usize =
    COL_MM_R_LIMB_BYTES + MULMOD_LIMBS * BYTES_PER_LIMB; // 518..550

/// Slack low limb: witness u64 such that
///   M_lo_limbs[0] = R_limbs[0] + 1 + slack_low_limb
/// (gated by is_mulmod). slack_low_limb is itself byte-decomposed and
/// 8-bit-range-checked → forces it into [0, 2^64), which in turn forces
/// R_limbs[0] < M_lo_limbs[0]. This is the low-limb piece of the full
/// R<M check; higher-limb equality/less-than is deferred.
pub const COL_MM_SLACK_LOW_LIMB: usize =
    COL_MM_Q_LIMB_BYTES + MULMOD_LIMBS * BYTES_PER_LIMB; // 550
pub const COL_MM_SLACK_LOW_BYTES: usize = COL_MM_SLACK_LOW_LIMB + 1; // 551..559

/// Soft selector: when 1, the low-limb R<M check fires for this row.
/// Honest witnesses set this to 1 ONLY when the modulus fits in a
/// single u64 (M_lo[1..3] all zero), in which case R<M reduces to the
/// low-limb compare and is fully algebraically sound. For multi-limb
/// M, the trace builder leaves this 0 and the low-limb check is
/// skipped. This is a SOFT selector (no must-fire constraint binds
/// it to M's high limbs) — kept for back-compat with existing tests.
pub const COL_MM_CHECK_R_LT_M_LOW: usize =
    COL_MM_SLACK_LOW_BYTES + BYTES_PER_LIMB; // 559

// ─── Task #196: Full multi-limb R<M via 256-bit slack ─────────────────
//
// Encode `slack_full := M_lo - R - 1` over 4 LE 64-bit limbs with a
// 3-borrow chain. The bottom limb reuses `COL_MM_SLACK_LOW_LIMB` /
// `COL_MM_SLACK_LOW_BYTES`; the high 3 limbs and their byte decomps
// are committed as NEW columns. Each limb is byte-range-checked into
// [0, 2^64), so `slack_full < 2^256`. The final borrow must be 0,
// which algebraically forces `M_lo > R` (i.e. R < M_lo) over the
// integers.
//
// Gated by `is_check_r_lt_m_full`, which is a HARD-firing selector
// for the mulmod identity: when set, the chain runs unconditionally.
// The honest witness sets this whenever `is_mulmod` is asserted; a
// malicious prover that sets it to 0 cannot benefit because the
// outer R-limb byte decomp still pins R, and Q*M + R = A*B already
// holds — the only thing R<M adds is uniqueness of R.
pub const COL_MM_SLACK_HI_LIMBS: usize =
    COL_MM_CHECK_R_LT_M_LOW + 1; // 560..563 (3 high u64 limbs)
pub const COL_MM_SLACK_HI_BYTES: usize =
    COL_MM_SLACK_HI_LIMBS + (MULMOD_LIMBS - 1); // 563..587 (24 bytes)
/// 3 binary borrows between the 4-limb subtraction columns.
pub const COL_MM_SLACK_BORROWS: usize =
    COL_MM_SLACK_HI_BYTES + (MULMOD_LIMBS - 1) * BYTES_PER_LIMB; // 587..590
/// Hard-firing selector for the full multi-limb R<M chain.
pub const COL_MM_CHECK_R_LT_M_FULL: usize =
    COL_MM_SLACK_BORROWS + (MULMOD_LIMBS - 1); // 590

// ─── Task #196: 9-byte LE range checks for AB / QM carries ────────────
//
// Each column-sum carry `ab_carry[k]` / `qm_carry[k]` (k=0..6) is
// bounded above by approximately MULMOD_LIMBS · 2^128 / 2^64 ≈ 2^66,
// safely fitting in 72 bits. Decompose each into 9 LE bytes and
// 8-bit-range-check the bytes via the lookup table.
pub const CARRY_BYTES: usize = 9;
pub const NUM_CARRIES: usize = 2 * MULMOD_LIMBS - 1; // 7
pub const COL_MM_AB_CARRY_BYTES: usize =
    COL_MM_CHECK_R_LT_M_FULL + 1; // 591..654 (7 × 9 = 63 bytes)
pub const COL_MM_QM_CARRY_BYTES: usize =
    COL_MM_AB_CARRY_BYTES + NUM_CARRIES * CARRY_BYTES; // 654..717

pub const NUM_COLUMNS: usize =
    COL_MM_QM_CARRY_BYTES + NUM_CARRIES * CARRY_BYTES; // 717

/// Row-local constraints:
///   0:                       is_real ∈ {0, 1}
///   1:                       is_result_row ∈ {0, 1}
///   2:                       is_result_row * (1 - is_real) = 0
///   3..3+WINDOW_BITS:        current_window_bits[k] ∈ {0, 1}
///   3+WB:                    LE byte-decomp of iteration_index_word
///   4+WB:                    (1 - is_result_row) * Σ result[i] = 0
///   5+WB:                    is_mulmod ∈ {0, 1}
///   6+WB..6+WB+16:           AB partial-product definition:
///                            pp_ab[i*4+j] = a_limbs[i] * b_limbs[j]
///   22+WB..22+WB+16:         QM partial-product definition:
///                            pp_qm[i*4+j] = q_limbs[i] * m_lo_limbs[j]
///   38+WB..38+WB+8:          AB column-sum + carry chain (radix 2^64):
///                            ab[k] + ab_carry[k]·2^64
///                              = Σ_{i+j=k} pp_ab[i][j]
///                                + (ab_carry[k-1] if k>0 else 0)
///   46+WB..46+WB+8:          QM column-sum + carry chain (radix 2^64).
///   54+WB..54+WB+4:          Identity low half (k=0..3):
///                            ab[k] + id_carry_lo[k-1]
///                              = qm_out[k] + r_limbs[k]
///                                + id_carry_lo[k]·2^64
///   58+WB..58+WB+4:          Identity high half (k=4..7):
///                            ab[k] + id_carry_hi[k-4-1]
///                              = qm_out[k]
///                                + id_carry_hi[k-4]·2^64
///                            (closes the 512-bit identity AB = QM + R.)
///
/// All gadget constraints are gated by `is_mulmod`.
pub const NUM_ROW_CONSTRAINTS_LEGACY: usize = 3 + WINDOW_BITS + 2; // 9
pub const NUM_ROW_CONSTRAINTS_MULMOD: usize = 1 // is_mulmod binary
    + MULMOD_PP // 16  PP-AB
    + MULMOD_PP // 16  PP-QM
    + 2 * MULMOD_LIMBS // 8  AB carry chain
    + 2 * MULMOD_LIMBS // 8  QM carry chain
    + MULMOD_LIMBS // 4  identity low half
    + MULMOD_LIMBS; // 4  identity high half
/// Task #182 binding constraints:
///   * R-limb byte decomp (4 × `Σ byte·2^8k = r_limbs[i]`).
///   * Q-limb byte decomp (4 ×).
///   * Slack low-limb byte decomp (1 ×).
///   * R<M low-limb uniqueness:
///       is_mulmod · (M_lo[0] - R[0] - 1 - slack_low_limb) = 0.
pub const NUM_ROW_CONSTRAINTS_RANGE: usize =
    MULMOD_LIMBS  // R limb byte decomp
    + MULMOD_LIMBS // Q limb byte decomp
    + 1            // slack low-limb byte decomp
    + 1            // is_check_r_lt_m_low binary
    + 1; // R<M low-limb uniqueness (gated)
/// Task #196 binding constraints:
///   * 3 high-limb slack byte decomps (`Σ byte·2^8k = slack_hi[i]`).
///   * 3 borrow-binary constraints.
///   * 4 limb-equations of the 256-bit subtraction
///     `M_lo[k] + borrow_in = R[k] + [k==0] + slack[k] + borrow_out·2^64`.
///     The final borrow_out (after k=3) is omitted ⇒ M_lo ≥ R+1.
///   * 1 is_check_r_lt_m_full binary.
///   * 7 ab_carry 9-byte decomps.
///   * 7 qm_carry 9-byte decomps.
pub const NUM_ROW_CONSTRAINTS_RANGE_V196: usize =
    (MULMOD_LIMBS - 1)  // 3 slack-hi byte decomps
    + (MULMOD_LIMBS - 1) // 3 borrow binaries
    + MULMOD_LIMBS       // 4 subtraction limb-equations
    + 1                  // is_check_r_lt_m_full binary
    + NUM_CARRIES        // 7 ab_carry decomps
    + NUM_CARRIES;       // 7 qm_carry decomps
pub const NUM_ROW_CONSTRAINTS: usize = NUM_ROW_CONSTRAINTS_LEGACY
    + NUM_ROW_CONSTRAINTS_MULMOD
    + NUM_ROW_CONSTRAINTS_RANGE
    + NUM_ROW_CONSTRAINTS_RANGE_V196; // 9 + 57 + 11 + 25 = 102

/// Shifted constraints:
///   0: partial_result-continuity
///      `next_partial(X) == partial(ω·X)` (gated by is_real chain, NOT
///      is_mulmod).
///   1: Task #182 mulmod cross-row chain.
///      `partial_result(ω·X) == R_limbs(X)` (gated by `is_mulmod(X) ·
///      is_real(ω·X)`). Algebraically binds each iteration's R limbs
///      to the next row's partial_result (BE byte view of partial_result
///      ↔ LE limb view of R: we bind only via the LOW 8 bytes — limb 0
///      — for now; full 256-bit binding deferred until limb↔byte cross-
///      decomposition is wired on partial_result).
pub const NUM_SHIFTED: usize = 2;

// ─── Witness ──────────────────────────────────────────────────────────

/// One row of the ModExp internals trace (one sliding-window iteration).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModExpInternalsRow {
    pub b_bytes: [u8; BIGINT_LENGTH],
    pub e_bytes: [u8; BIGINT_LENGTH],
    pub m_bytes: [u8; BIGINT_LENGTH],
    pub partial_result: [u8; BIGINT_LENGTH],
    pub next_partial_result: [u8; BIGINT_LENGTH],
    pub result: [u8; RESULT_LENGTH],
    pub current_window_bits: [u8; WINDOW_BITS],
    pub iteration_index: u32,
    pub is_result_row: bool,
    /// Mul-mod gadget witness. When `is_mulmod` is false all gadget
    /// columns must be zero (constraint system enforces this via the
    /// gating multiplications).
    pub mulmod: MulModWitness,
    pub is_mulmod: bool,
}

/// 4-limb (256-bit) mul-mod gadget witness for one row.
///
/// Encodes the identity `A * B = Q * M_lo + R` over little-endian
/// 64-bit limbs. All intermediate partial products and carries are
/// committed; the constraint system reconstructs the column sums
/// algebraically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MulModWitness {
    /// 4 LE 64-bit limbs of A.
    pub a_limbs: [u64; MULMOD_LIMBS],
    /// 4 LE 64-bit limbs of B.
    pub b_limbs: [u64; MULMOD_LIMBS],
    /// 4 LE 64-bit limbs of M_lo (low 256 bits of the modulus).
    pub m_lo_limbs: [u64; MULMOD_LIMBS],
    /// 4 LE 64-bit limbs of Q (Barrett-style quotient).
    pub q_limbs: [u64; MULMOD_LIMBS],
    /// 4 LE 64-bit limbs of R = A*B - Q*M_lo (must satisfy R < M_lo).
    pub r_limbs: [u64; MULMOD_LIMBS],
    /// 16 partial products `a_limbs[i] * b_limbs[j]` (row-major i*4+j).
    /// Each fits in 128 bits → stored as u128.
    pub pp_ab: [u128; MULMOD_PP],
    /// 16 partial products `q_limbs[i] * m_lo_limbs[j]`.
    pub pp_qm: [u128; MULMOD_PP],
    /// 8 output limbs of AB (low 4 + high 4); each is the radix-2^64
    /// residue of the column sum.
    pub ab_out: [u64; 2 * MULMOD_LIMBS],
    /// 7 inter-column carries for AB (between cols 0..1, 1..2, ..., 6..7).
    /// Carry from col k saturates at ~70 bits → store as u128.
    pub ab_carry: [u128; 2 * MULMOD_LIMBS - 1],
    /// 8 output limbs of Q*M_lo.
    pub qm_out: [u64; 2 * MULMOD_LIMBS],
    /// 7 inter-column carries for QM.
    pub qm_carry: [u128; 2 * MULMOD_LIMBS - 1],
    /// 4 identity carries (low half, k=0..3): borrow/carry between
    /// the column equations ab[k] = qm_out[k] + r_limbs[k] + carry.
    pub id_carry_lo: [i128; MULMOD_LIMBS],
    /// 4 identity carries (high half, k=4..7).
    pub id_carry_hi: [i128; MULMOD_LIMBS],
    /// Task #182: slack u64 for the R<M low-limb uniqueness check.
    /// Honest: slack_low_limb = M_lo[0] - R[0] - 1 when R[0] < M_lo[0],
    /// otherwise 0 (and `check_r_lt_m_low` is unset for this row).
    pub slack_low_limb: u64,
    /// Task #182: soft selector that enables the R<M low-limb check.
    /// Set to 1 by the honest witness ONLY when the modulus is
    /// single-limb (M_lo[1..3] all zero).
    pub check_r_lt_m_low: bool,
    /// Task #196: high-limb slack for the full 256-bit subtraction
    /// `M_lo - R - 1`. Combined with `slack_low_limb` for limb 0.
    pub slack_hi_limbs: [u64; MULMOD_LIMBS - 1],
    /// Task #196: 3 borrows of the 4-limb subtraction chain. Each is
    /// 0 or 1; the chain is sealed by omitting the final borrow.
    pub slack_borrows: [u64; MULMOD_LIMBS - 1],
    /// Task #196: hard-firing selector for the full R<M chain.
    pub check_r_lt_m_full: bool,
}

impl Default for ModExpInternalsRow {
    fn default() -> Self {
        Self {
            b_bytes: [0u8; BIGINT_LENGTH],
            e_bytes: [0u8; BIGINT_LENGTH],
            m_bytes: [0u8; BIGINT_LENGTH],
            partial_result: [0u8; BIGINT_LENGTH],
            next_partial_result: [0u8; BIGINT_LENGTH],
            result: [0u8; RESULT_LENGTH],
            current_window_bits: [0u8; WINDOW_BITS],
            iteration_index: 0,
            is_result_row: false,
            mulmod: MulModWitness::default(),
            is_mulmod: false,
        }
    }
}

impl MulModWitness {
    /// Compute the full mul-mod witness from raw 4-limb inputs.
    /// Computes `Q, R` such that `A * B = Q * M_lo + R` and fills all
    /// partial products + carries.
    ///
    /// Panics if `M_lo == 0` (caller must guard).
    pub fn from_inputs(
        a_limbs: [u64; MULMOD_LIMBS],
        b_limbs: [u64; MULMOD_LIMBS],
        m_lo_limbs: [u64; MULMOD_LIMBS],
    ) -> Self {
        use num_bigint::BigUint;
        // Reconstruct A, B, M as BigUint via LE limbs.
        let to_big = |ls: &[u64; MULMOD_LIMBS]| -> BigUint {
            let mut bytes = Vec::with_capacity(32);
            for l in ls.iter() {
                bytes.extend_from_slice(&l.to_le_bytes());
            }
            BigUint::from_bytes_le(&bytes)
        };
        let from_big = |x: &BigUint| -> [u64; MULMOD_LIMBS] {
            let mut bytes = x.to_bytes_le();
            bytes.resize(32, 0);
            let mut out = [0u64; MULMOD_LIMBS];
            for i in 0..MULMOD_LIMBS {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[8 * i..8 * (i + 1)]);
                out[i] = u64::from_le_bytes(b);
            }
            out
        };
        let a = to_big(&a_limbs);
        let b = to_big(&b_limbs);
        let m = to_big(&m_lo_limbs);
        assert!(!m.is_zero_helper(), "mulmod: M_lo must be nonzero");
        let ab = &a * &b;
        let q_big = &ab / &m;
        let r_big = &ab % &m;
        let qm_big = &q_big * &m;
        let q_limbs = from_big(&q_big);
        let r_limbs = from_big(&r_big);

        // 16 partial products A_i * B_j.
        let mut pp_ab = [0u128; MULMOD_PP];
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                pp_ab[i * MULMOD_LIMBS + j] =
                    (a_limbs[i] as u128) * (b_limbs[j] as u128);
            }
        }
        // 16 partial products Q_i * M_j.
        let mut pp_qm = [0u128; MULMOD_PP];
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                pp_qm[i * MULMOD_LIMBS + j] =
                    (q_limbs[i] as u128) * (m_lo_limbs[j] as u128);
            }
        }
        // Column-sum + carry chain (radix 2^64) for AB.
        let (ab_out, ab_carry) = column_sum_with_carries(&pp_ab);
        // Column-sum + carry chain for QM.
        let (qm_out, qm_carry) = column_sum_with_carries(&pp_qm);

        // Identity low half: ab[k] = qm_out[k] + r_limbs[k] + carry.
        // id_carry_lo[k] = (ab[k] + id_carry_lo[k-1] - qm_out[k] -
        // r_limbs[k]) / 2^64. The result must be a non-negative integer
        // bounded by O(MULMOD_LIMBS).
        let mut id_carry_lo = [0i128; MULMOD_LIMBS];
        let mut prev_carry: i128 = 0;
        for k in 0..MULMOD_LIMBS {
            let lhs = ab_out[k] as i128 + prev_carry;
            let rhs = qm_out[k] as i128 + r_limbs[k] as i128;
            let diff = lhs - rhs;
            // diff must be divisible by 2^64
            let carry = diff >> 64;
            id_carry_lo[k] = carry;
            prev_carry = carry;
        }
        // Identity high half: ab[k] = qm_out[k] + carry (no R).
        let mut id_carry_hi = [0i128; MULMOD_LIMBS];
        for k in 0..MULMOD_LIMBS {
            let kk = MULMOD_LIMBS + k;
            let lhs = ab_out[kk] as i128 + prev_carry;
            let rhs = qm_out[kk] as i128;
            let diff = lhs - rhs;
            let carry = diff >> 64;
            id_carry_hi[k] = carry;
            prev_carry = carry;
        }
        // Sanity: the final carry should be zero for an honest witness.
        debug_assert_eq!(prev_carry, 0, "mulmod identity did not close");
        debug_assert_eq!(qm_big + r_big.clone(), ab.clone(), "Q*M+R != A*B");

        // Task #182: low-limb R<M slack. Honest:
        //   slack_low_limb = M_lo[0] - R[0] - 1  when M is single-limb
        //   (high limbs zero) and R[0] < M_lo[0]; otherwise 0 and the
        //   row-level `check_r_lt_m_low` selector is unset.
        let is_single_limb_m =
            m_lo_limbs[1] == 0 && m_lo_limbs[2] == 0 && m_lo_limbs[3] == 0;
        let (slack_low_limb, check_r_lt_m_low) = if is_single_limb_m
            && r_limbs[0] < m_lo_limbs[0]
        {
            (m_lo_limbs[0] - r_limbs[0] - 1, true)
        } else {
            (0u64, false)
        };

        // Task #196: compute the full 256-bit slack = M_lo - R - 1 with
        // borrow propagation. M_lo ≥ R+1 is guaranteed by the honest
        // Barrett witness (R = AB mod M_lo, so R < M_lo).
        let mut slack_full = [0u64; MULMOD_LIMBS];
        let mut slack_borrows = [0u64; MULMOD_LIMBS - 1];
        let mut borrow_in: u128 = 0;
        let one_vec = [1u64, 0u64, 0u64, 0u64];
        for k in 0..MULMOD_LIMBS {
            // Compute slack[k] = M[k] - R[k] - one[k] - borrow_in, with
            // borrow_out propagated up.
            let m_k = m_lo_limbs[k] as u128;
            let r_k = r_limbs[k] as u128;
            let one_k = one_vec[k] as u128;
            let sub = r_k + one_k + borrow_in;
            let (slack_k, borrow_out) = if m_k >= sub {
                ((m_k - sub) as u64, 0u128)
            } else {
                // Wrap with 2^64 borrow.
                ((m_k + (1u128 << 64) - sub) as u64, 1u128)
            };
            slack_full[k] = slack_k;
            if k + 1 < MULMOD_LIMBS {
                slack_borrows[k] = borrow_out as u64;
            } else {
                // Final borrow must be zero for an honest R<M witness.
                debug_assert_eq!(
                    borrow_out, 0,
                    "Task #196 R<M chain did not close (R >= M)"
                );
            }
            borrow_in = borrow_out;
        }
        let slack_hi_limbs = [slack_full[1], slack_full[2], slack_full[3]];
        // Reuse `slack_low_limb` field for limb 0 of the full chain.
        let slack_low_limb_full = slack_full[0];
        // Distinguish v182 (single-limb, soft) vs v196 (full chain):
        // when the v182 selector fires, its slack equals slack_full[0]
        // anyway because high limbs of M are zero ⇒ slack_full[k]=0 for
        // k>=1 ⇒ borrow chain trivial. Otherwise we use slack_full[0].
        let slack_low_limb = if check_r_lt_m_low {
            slack_low_limb
        } else {
            slack_low_limb_full
        };

        Self {
            a_limbs,
            b_limbs,
            m_lo_limbs,
            q_limbs,
            r_limbs,
            pp_ab,
            pp_qm,
            ab_out,
            ab_carry,
            qm_out,
            qm_carry,
            id_carry_lo,
            id_carry_hi,
            slack_low_limb,
            check_r_lt_m_low,
            slack_hi_limbs,
            slack_borrows,
            check_r_lt_m_full: true,
        }
    }

    /// Build the squaring step witness (A == B == partial_result_lo).
    pub fn from_square(
        a_limbs: [u64; MULMOD_LIMBS],
        m_lo_limbs: [u64; MULMOD_LIMBS],
    ) -> Self {
        Self::from_inputs(a_limbs, a_limbs, m_lo_limbs)
    }
}

/// Helper: 8-limb column sum with 7 carries between adjacent columns,
/// radix 2^64.
fn column_sum_with_carries(
    pp: &[u128; MULMOD_PP],
) -> ([u64; 2 * MULMOD_LIMBS], [u128; 2 * MULMOD_LIMBS - 1]) {
    let mut out = [0u64; 2 * MULMOD_LIMBS];
    let mut carry = [0u128; 2 * MULMOD_LIMBS - 1];
    let mut prev: u128 = 0;
    for k in 0..(2 * MULMOD_LIMBS) {
        let mut col: u128 = prev;
        for i in 0..MULMOD_LIMBS {
            let j = k as i32 - i as i32;
            if j >= 0 && (j as usize) < MULMOD_LIMBS {
                col = col + pp[i * MULMOD_LIMBS + j as usize];
            }
        }
        out[k] = col as u64;
        let c = col >> 64;
        if k + 1 < 2 * MULMOD_LIMBS {
            carry[k] = c;
        }
        prev = c;
    }
    (out, carry)
}

// Tiny shim trait — BigUint::is_zero is gated behind num-traits which we
// don't import; use byte length check.
trait BigUintZeroHelper {
    fn is_zero_helper(&self) -> bool;
}
impl BigUintZeroHelper for num_bigint::BigUint {
    fn is_zero_helper(&self) -> bool {
        self.to_bytes_le() == vec![0u8]
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModExpInternalsTraceWitness {
    pub rows: Vec<ModExpInternalsRow>,
}

impl ModExpInternalsTraceWitness {
    pub fn from_rows(rows: Vec<ModExpInternalsRow>) -> Self {
        Self { rows }
    }
    pub fn push(&mut self, row: ModExpInternalsRow) {
        self.rows.push(row);
    }

    /// Build a SKELETON witness from a finalized `B^E mod M` output.
    /// Intermediate partial-result columns are zero (deferred); only the
    /// FINAL row carries the meaningful `result` field.
    pub fn from_result_skeleton(result: [u8; RESULT_LENGTH]) -> Self {
        let mut rows = Vec::with_capacity(2);
        rows.push(ModExpInternalsRow::default());
        rows.push(ModExpInternalsRow {
            result,
            iteration_index: 1,
            is_result_row: true,
            ..ModExpInternalsRow::default()
        });
        Self { rows }
    }

    /// Build a single-row witness containing a mul-mod gadget evaluation
    /// `R = A * B mod M_lo`. Other (legacy) columns left zero.
    pub fn from_mulmod_step(
        a_limbs: [u64; MULMOD_LIMBS],
        b_limbs: [u64; MULMOD_LIMBS],
        m_lo_limbs: [u64; MULMOD_LIMBS],
    ) -> Self {
        let mulmod = MulModWitness::from_inputs(a_limbs, b_limbs, m_lo_limbs);
        let row = ModExpInternalsRow {
            mulmod,
            is_mulmod: true,
            ..ModExpInternalsRow::default()
        };
        Self { rows: vec![row] }
    }
}

// ─── Field helpers ────────────────────────────────────────────────────

/// 2^64 as a `Scalar` (used as the radix for limb carry propagation).
fn two_64(curve: CurveType) -> Scalar {
    // 2^32 then square.
    let h = Scalar::from_u64(1u64 << 32, curve);
    h.mul(&h)
}

/// Embed a u128 into the prime field by splitting into hi/lo 64-bit
/// halves and combining via 2^64.
fn scalar_from_u128(v: u128, curve: CurveType) -> Scalar {
    let lo = (v as u64) as u128;
    let hi = (v >> 64) as u64;
    let lo_s = Scalar::from_u64(lo as u64, curve);
    let hi_s = Scalar::from_u64(hi, curve);
    lo_s.add(&hi_s.mul(&two_64(curve)))
}

/// Embed a signed i128 into the prime field. Negative values map to
/// `p - |v|`. This is sound because the constraint system reasons
/// modulo the scalar field prime, so the negative carry contributes
/// the correct algebraic value to the identity polynomial.
fn scalar_from_i128(v: i128, curve: CurveType) -> Scalar {
    if v >= 0 {
        scalar_from_u128(v as u128, curve)
    } else {
        let abs = scalar_from_u128((-v) as u128, curve);
        Scalar::zero(curve).sub(&abs)
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ModExpInternalsTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..BIGINT_LENGTH {
            columns[COL_B_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.b_bytes[k] as u64, curve);
            columns[COL_E_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.e_bytes[k] as u64, curve);
            columns[COL_M_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.m_bytes[k] as u64, curve);
            columns[COL_PARTIAL_RESULT_OFFSET + k][r] =
                Scalar::from_u64(row.partial_result[k] as u64, curve);
            columns[COL_NEXT_PARTIAL_RESULT_OFFSET + k][r] =
                Scalar::from_u64(row.next_partial_result[k] as u64, curve);
        }
        for k in 0..RESULT_LENGTH {
            columns[COL_RESULT_OFFSET + k][r] =
                Scalar::from_u64(row.result[k] as u64, curve);
        }
        for k in 0..WINDOW_BITS {
            columns[COL_CURRENT_WINDOW_BITS_OFFSET + k][r] =
                Scalar::from_u64(row.current_window_bits[k] as u64, curve);
        }
        columns[COL_ITERATION_INDEX][r] =
            Scalar::from_u64(row.iteration_index as u64, curve);
        let ii_le = row.iteration_index.to_le_bytes();
        for b in 0..ITERATION_INDEX_BYTES {
            columns[COL_ITERATION_INDEX_BYTE_OFFSET + b][r] =
                Scalar::from_u64(ii_le[b] as u64, curve);
        }
        columns[COL_IS_RESULT_ROW][r] = if row.is_result_row {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_REAL][r] = one.clone();

        // ── Mul-mod gadget columns ───────────────────────────────────
        let mm = &row.mulmod;
        for i in 0..MULMOD_LIMBS {
            columns[COL_MM_A_LIMBS + i][r] =
                Scalar::from_u64(mm.a_limbs[i], curve);
            columns[COL_MM_B_LIMBS + i][r] =
                Scalar::from_u64(mm.b_limbs[i], curve);
            columns[COL_MM_M_LO_LIMBS + i][r] =
                Scalar::from_u64(mm.m_lo_limbs[i], curve);
            columns[COL_MM_Q_LIMBS + i][r] =
                Scalar::from_u64(mm.q_limbs[i], curve);
            columns[COL_MM_R_LIMBS + i][r] =
                Scalar::from_u64(mm.r_limbs[i], curve);
        }
        for k in 0..MULMOD_PP {
            columns[COL_MM_PP_AB + k][r] =
                scalar_from_u128(mm.pp_ab[k], curve);
            columns[COL_MM_PP_QM + k][r] =
                scalar_from_u128(mm.pp_qm[k], curve);
        }
        for k in 0..(2 * MULMOD_LIMBS) {
            columns[COL_MM_AB_OUT + k][r] =
                Scalar::from_u64(mm.ab_out[k], curve);
            columns[COL_MM_QM_OUT + k][r] =
                Scalar::from_u64(mm.qm_out[k], curve);
        }
        for k in 0..(2 * MULMOD_LIMBS - 1) {
            columns[COL_MM_AB_CARRY + k][r] =
                scalar_from_u128(mm.ab_carry[k], curve);
            columns[COL_MM_QM_CARRY + k][r] =
                scalar_from_u128(mm.qm_carry[k], curve);
        }
        for k in 0..MULMOD_LIMBS {
            columns[COL_MM_ID_CARRY_LO + k][r] =
                scalar_from_i128(mm.id_carry_lo[k], curve);
            columns[COL_MM_ID_CARRY_HI + k][r] =
                scalar_from_i128(mm.id_carry_hi[k], curve);
        }
        columns[COL_IS_MULMOD][r] = if row.is_mulmod {
            one.clone()
        } else {
            zero.clone()
        };

        // ── Task #182: byte decompositions ───────────────────────────
        for i in 0..MULMOD_LIMBS {
            let r_bytes = mm.r_limbs[i].to_le_bytes();
            let q_bytes = mm.q_limbs[i].to_le_bytes();
            for k in 0..BYTES_PER_LIMB {
                columns[COL_MM_R_LIMB_BYTES + i * BYTES_PER_LIMB + k][r] =
                    Scalar::from_u64(r_bytes[k] as u64, curve);
                columns[COL_MM_Q_LIMB_BYTES + i * BYTES_PER_LIMB + k][r] =
                    Scalar::from_u64(q_bytes[k] as u64, curve);
            }
        }
        columns[COL_MM_SLACK_LOW_LIMB][r] =
            Scalar::from_u64(mm.slack_low_limb, curve);
        let slack_bytes = mm.slack_low_limb.to_le_bytes();
        for k in 0..BYTES_PER_LIMB {
            columns[COL_MM_SLACK_LOW_BYTES + k][r] =
                Scalar::from_u64(slack_bytes[k] as u64, curve);
        }
        columns[COL_MM_CHECK_R_LT_M_LOW][r] = if mm.check_r_lt_m_low {
            one.clone()
        } else {
            zero.clone()
        };

        // ── Task #196: high-limb slack + borrows + carry decomps ─────
        for i in 0..(MULMOD_LIMBS - 1) {
            columns[COL_MM_SLACK_HI_LIMBS + i][r] =
                Scalar::from_u64(mm.slack_hi_limbs[i], curve);
            let bytes = mm.slack_hi_limbs[i].to_le_bytes();
            for k in 0..BYTES_PER_LIMB {
                columns[COL_MM_SLACK_HI_BYTES + i * BYTES_PER_LIMB + k][r] =
                    Scalar::from_u64(bytes[k] as u64, curve);
            }
            columns[COL_MM_SLACK_BORROWS + i][r] =
                Scalar::from_u64(mm.slack_borrows[i], curve);
        }
        columns[COL_MM_CHECK_R_LT_M_FULL][r] = if mm.check_r_lt_m_full {
            one.clone()
        } else {
            zero.clone()
        };
        // 9-byte LE decomp of ab_carry[k] and qm_carry[k].
        for k in 0..NUM_CARRIES {
            let ab_bytes = mm.ab_carry[k].to_le_bytes(); // 16 bytes; take low 9
            let qm_bytes = mm.qm_carry[k].to_le_bytes();
            for b in 0..CARRY_BYTES {
                columns[COL_MM_AB_CARRY_BYTES + k * CARRY_BYTES + b][r] =
                    Scalar::from_u64(ab_bytes[b] as u64, curve);
                columns[COL_MM_QM_CARRY_BYTES + k * CARRY_BYTES + b][r] =
                    Scalar::from_u64(qm_bytes[b] as u64, curve);
            }
        }
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct ModExpInternalsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ModExpInternalsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for ModExpInternalsConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("is_result_row_binary".into());
        labels.push("is_result_row_le_is_real".into());
        for k in 0..WINDOW_BITS {
            labels.push(format!("current_window_bits_{}_binary", k));
        }
        labels.push("iteration_index_le_byte_decomp".into());
        labels.push("result_zero_off_final_row".into());
        // Mul-mod gadget labels.
        labels.push("is_mulmod_binary".into());
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                labels.push(format!("mulmod_pp_ab_{}_{}", i, j));
            }
        }
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                labels.push(format!("mulmod_pp_qm_{}_{}", i, j));
            }
        }
        for k in 0..(2 * MULMOD_LIMBS) {
            labels.push(format!("mulmod_ab_col_{}", k));
        }
        for k in 0..(2 * MULMOD_LIMBS) {
            labels.push(format!("mulmod_qm_col_{}", k));
        }
        for k in 0..MULMOD_LIMBS {
            labels.push(format!("mulmod_id_lo_{}", k));
        }
        for k in 0..MULMOD_LIMBS {
            labels.push(format!("mulmod_id_hi_{}", k));
        }
        // Task #182 range / R<M labels.
        for i in 0..MULMOD_LIMBS {
            labels.push(format!("mulmod_r_limb_{}_byte_decomp", i));
        }
        for i in 0..MULMOD_LIMBS {
            labels.push(format!("mulmod_q_limb_{}_byte_decomp", i));
        }
        labels.push("mulmod_slack_low_byte_decomp".into());
        labels.push("mulmod_check_r_lt_m_low_binary".into());
        labels.push("mulmod_r_lt_m_low_limb".into());
        // Task #196 labels.
        for i in 0..(MULMOD_LIMBS - 1) {
            labels.push(format!("mulmod_slack_hi_{}_byte_decomp", i));
        }
        for i in 0..(MULMOD_LIMBS - 1) {
            labels.push(format!("mulmod_slack_borrow_{}_binary", i));
        }
        for k in 0..MULMOD_LIMBS {
            labels.push(format!("mulmod_r_lt_m_full_limb_{}", k));
        }
        labels.push("mulmod_check_r_lt_m_full_binary".into());
        for k in 0..NUM_CARRIES {
            labels.push(format!("mulmod_ab_carry_{}_byte_decomp", k));
        }
        for k in 0..NUM_CARRIES {
            labels.push(format!("mulmod_qm_carry_{}_byte_decomp", k));
        }
        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real binary.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            c[r] = v.mul(&v.sub(&one));
        }
        out.push(c);

        // 1: is_result_row binary.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_RESULT_ROW][r];
            c[r] = v.mul(&v.sub(&one));
        }
        out.push(c);

        // 2: is_result_row * (1 - is_real).
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let rr = &columns[COL_IS_RESULT_ROW][r];
            let real = &columns[COL_IS_REAL][r];
            c[r] = rr.mul(&one.sub(real));
        }
        out.push(c);

        // 3..3+WINDOW_BITS: each window bit ∈ {0, 1}.
        for k in 0..WINDOW_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_CURRENT_WINDOW_BITS_OFFSET + k][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 3+WB: LE byte-decomp of iteration_index_word.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let mut sum = Scalar::zero(curve);
            for b in 0..ITERATION_INDEX_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                let term = columns[COL_ITERATION_INDEX_BYTE_OFFSET + b][r].mul(&w);
                sum = sum.add(&term);
            }
            c[r] = sum.sub(&columns[COL_ITERATION_INDEX][r]);
        }
        out.push(c);

        // 4+WB: (1 - is_result_row) * Σ result[i] = 0.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let mut sum = Scalar::zero(curve);
            for i in 0..RESULT_LENGTH {
                sum = sum.add(&columns[COL_RESULT_OFFSET + i][r]);
            }
            let gate = one.sub(&columns[COL_IS_RESULT_ROW][r]);
            c[r] = gate.mul(&sum);
        }
        out.push(c);

        // ─── Mul-mod gadget constraints ──────────────────────────────
        let r64 = two_64(curve);

        // is_mulmod binary.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_MULMOD][r];
            c[r] = v.mul(&v.sub(&one));
        }
        out.push(c);

        // PP-AB: pp_ab[i*4+j] = a_limbs[i] * b_limbs[j]  (gated).
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                let mut c = vec![Scalar::zero(curve); n];
                for r in 0..n {
                    let a = &columns[COL_MM_A_LIMBS + i][r];
                    let b = &columns[COL_MM_B_LIMBS + j][r];
                    let pp = &columns[COL_MM_PP_AB + i * MULMOD_LIMBS + j][r];
                    let body = a.mul(b).sub(pp);
                    c[r] = columns[COL_IS_MULMOD][r].mul(&body);
                }
                out.push(c);
            }
        }

        // PP-QM: pp_qm[i*4+j] = q_limbs[i] * m_lo_limbs[j]  (gated).
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                let mut c = vec![Scalar::zero(curve); n];
                for r in 0..n {
                    let a = &columns[COL_MM_Q_LIMBS + i][r];
                    let b = &columns[COL_MM_M_LO_LIMBS + j][r];
                    let pp = &columns[COL_MM_PP_QM + i * MULMOD_LIMBS + j][r];
                    let body = a.mul(b).sub(pp);
                    c[r] = columns[COL_IS_MULMOD][r].mul(&body);
                }
                out.push(c);
            }
        }

        // AB column sums + carry chain. For col k:
        //   ab_out[k] + ab_carry[k]·2^64
        //     = Σ_{i+j=k} pp_ab[i][j] + (ab_carry[k-1] if k>0 else 0)
        // The last column (k = 7) has no outgoing carry.
        for k in 0..(2 * MULMOD_LIMBS) {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                // RHS: column sum of partial products.
                let mut rhs = Scalar::zero(curve);
                for i in 0..MULMOD_LIMBS {
                    let j = k as i32 - i as i32;
                    if j >= 0 && (j as usize) < MULMOD_LIMBS {
                        rhs = rhs.add(
                            &columns[COL_MM_PP_AB + i * MULMOD_LIMBS + j as usize][r],
                        );
                    }
                }
                if k > 0 {
                    rhs = rhs.add(&columns[COL_MM_AB_CARRY + (k - 1)][r]);
                }
                // LHS: ab_out[k] + carry_out·2^64 (no carry out of last).
                let mut lhs = columns[COL_MM_AB_OUT + k][r].clone();
                if k + 1 < 2 * MULMOD_LIMBS {
                    lhs = lhs.add(&columns[COL_MM_AB_CARRY + k][r].mul(&r64));
                }
                let body = lhs.sub(&rhs);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }

        // QM column sums + carry chain (mirrors AB).
        for k in 0..(2 * MULMOD_LIMBS) {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut rhs = Scalar::zero(curve);
                for i in 0..MULMOD_LIMBS {
                    let j = k as i32 - i as i32;
                    if j >= 0 && (j as usize) < MULMOD_LIMBS {
                        rhs = rhs.add(
                            &columns[COL_MM_PP_QM + i * MULMOD_LIMBS + j as usize][r],
                        );
                    }
                }
                if k > 0 {
                    rhs = rhs.add(&columns[COL_MM_QM_CARRY + (k - 1)][r]);
                }
                let mut lhs = columns[COL_MM_QM_OUT + k][r].clone();
                if k + 1 < 2 * MULMOD_LIMBS {
                    lhs = lhs.add(&columns[COL_MM_QM_CARRY + k][r].mul(&r64));
                }
                let body = lhs.sub(&rhs);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }

        // Identity low half (k = 0..3):
        //   ab[k] + id_carry_lo[k-1] = qm_out[k] + r_limbs[k] + id_carry_lo[k]·2^64
        for k in 0..MULMOD_LIMBS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut lhs = columns[COL_MM_AB_OUT + k][r].clone();
                if k > 0 {
                    lhs = lhs.add(&columns[COL_MM_ID_CARRY_LO + (k - 1)][r]);
                }
                let rhs = columns[COL_MM_QM_OUT + k][r]
                    .add(&columns[COL_MM_R_LIMBS + k][r])
                    .add(&columns[COL_MM_ID_CARRY_LO + k][r].mul(&r64));
                let body = lhs.sub(&rhs);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }

        // Identity high half (k = 4..7):
        //   ab[kk] + carry_in = qm_out[kk] + id_carry_hi[k]·2^64
        // where carry_in for k=0 is id_carry_lo[MULMOD_LIMBS-1] (= 3).
        //
        // The LAST high limb (k = MULMOD_LIMBS-1) MUST have outgoing
        // carry = 0 for the integer identity to close. We enforce this
        // by setting the outgoing carry term to zero in that row:
        //   ab[7] + carry_in = qm_out[7]      (no ·2^64 term)
        // This is the algebraic seal of the full Barrett identity.
        for k in 0..MULMOD_LIMBS {
            let kk = MULMOD_LIMBS + k;
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let carry_in = if k == 0 {
                    columns[COL_MM_ID_CARRY_LO + (MULMOD_LIMBS - 1)][r].clone()
                } else {
                    columns[COL_MM_ID_CARRY_HI + (k - 1)][r].clone()
                };
                let lhs = columns[COL_MM_AB_OUT + kk][r].add(&carry_in);
                let mut rhs = columns[COL_MM_QM_OUT + kk][r].clone();
                if k + 1 < MULMOD_LIMBS {
                    // Outgoing carry for non-final high columns.
                    rhs = rhs
                        .add(&columns[COL_MM_ID_CARRY_HI + k][r].mul(&r64));
                }
                // For k+1 == MULMOD_LIMBS, the outgoing carry term is
                // omitted: ab[7] + carry_in = qm_out[7] exactly. This
                // algebraically forces the integer identity to close
                // (no residual high-word carry). Note id_carry_hi[3]
                // is then effectively unconstrained; honest witness
                // sets it to zero but the prover may set it freely
                // without affecting soundness (it appears in no
                // other equation).
                let body = lhs.sub(&rhs);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }

        // ─── Task #182 range / R<M constraints ───────────────────────
        // Byte weights for 8-byte LE decomposition.
        let byte_weights: Vec<Scalar> = (0..BYTES_PER_LIMB)
            .map(|k| Scalar::from_u64(1u64 << (8 * k), curve))
            .collect();

        // R-limb byte decomp: gated by is_mulmod.
        for i in 0..MULMOD_LIMBS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for k in 0..BYTES_PER_LIMB {
                    let term = columns[COL_MM_R_LIMB_BYTES + i * BYTES_PER_LIMB + k][r]
                        .mul(&byte_weights[k]);
                    sum = sum.add(&term);
                }
                let body = sum.sub(&columns[COL_MM_R_LIMBS + i][r]);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }

        // Q-limb byte decomp: gated by is_mulmod.
        for i in 0..MULMOD_LIMBS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for k in 0..BYTES_PER_LIMB {
                    let term = columns[COL_MM_Q_LIMB_BYTES + i * BYTES_PER_LIMB + k][r]
                        .mul(&byte_weights[k]);
                    sum = sum.add(&term);
                }
                let body = sum.sub(&columns[COL_MM_Q_LIMBS + i][r]);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }

        // Slack low-limb byte decomp: gated by is_mulmod.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let mut sum = Scalar::zero(curve);
            for k in 0..BYTES_PER_LIMB {
                let term = columns[COL_MM_SLACK_LOW_BYTES + k][r]
                    .mul(&byte_weights[k]);
                sum = sum.add(&term);
            }
            let body = sum.sub(&columns[COL_MM_SLACK_LOW_LIMB][r]);
            c[r] = columns[COL_IS_MULMOD][r].mul(&body);
        }
        out.push(c);

        // is_check_r_lt_m_low binary, gated by is_mulmod.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_MM_CHECK_R_LT_M_LOW][r];
            c[r] = columns[COL_IS_MULMOD][r].mul(&v.mul(&v.sub(&one)));
        }
        out.push(c);

        // R<M low-limb uniqueness:
        //   is_mulmod · check_r_lt_m_low · (M_lo[0] - R[0] - 1 - slack_low_limb) = 0
        // Combined with the slack low-limb being byte-decomposed (range
        // [0, 2^64)), this forces R[0] < M_lo[0] over the integers when
        // the soft selector is asserted. NOTE: only pins the LOW limb,
        // and the selector is soft (no must-fire) — full multi-limb
        // R<M is deferred.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let m0 = &columns[COL_MM_M_LO_LIMBS][r];
            let r0 = &columns[COL_MM_R_LIMBS][r];
            let slack = &columns[COL_MM_SLACK_LOW_LIMB][r];
            let chk = &columns[COL_MM_CHECK_R_LT_M_LOW][r];
            let body = m0.sub(r0).sub(&one).sub(slack);
            c[r] = columns[COL_IS_MULMOD][r].mul(chk).mul(&body);
        }
        out.push(c);

        // ─── Task #196: full multi-limb R<M + carry range checks ─────
        // Slack high-limb byte decomps (3 ×, gated by is_mulmod).
        for i in 0..(MULMOD_LIMBS - 1) {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for k in 0..BYTES_PER_LIMB {
                    let term = columns
                        [COL_MM_SLACK_HI_BYTES + i * BYTES_PER_LIMB + k][r]
                        .mul(&byte_weights[k]);
                    sum = sum.add(&term);
                }
                let body = sum.sub(&columns[COL_MM_SLACK_HI_LIMBS + i][r]);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }
        // Borrow-binary constraints (3 ×, gated by is_mulmod).
        for i in 0..(MULMOD_LIMBS - 1) {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_MM_SLACK_BORROWS + i][r];
                let body = v.mul(&v.sub(&one));
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }
        // 256-bit subtraction chain: 4 limb equations. Per limb k:
        //   M_lo[k] + borrow_in[k]·2^64 = R[k] + one[k] + slack[k] + borrow_out[k]
        // where borrow_in[0] = 0; borrow_in[k>0] = borrow_out[k-1];
        // borrow_out[k<3] = slack_borrows[k]; borrow_out[3] omitted ⇒ 0.
        //
        // Wait — for textbook subtraction with borrow, the equation
        // when we borrow is:
        //   M[k] + 2^64·borrow_out = R[k] + one[k] + slack[k] + borrow_in
        // i.e. we add 2^64 to M[k] when we borrow (borrow_out = 1).
        //
        // Gated by `is_mulmod · check_r_lt_m_full`.
        for k in 0..MULMOD_LIMBS {
            let mut c = vec![Scalar::zero(curve); n];
            let one_k = if k == 0 { one.clone() } else { Scalar::zero(curve) };
            for r in 0..n {
                let m_k = &columns[COL_MM_M_LO_LIMBS + k][r];
                let r_k = &columns[COL_MM_R_LIMBS + k][r];
                let slack_k = if k == 0 {
                    columns[COL_MM_SLACK_LOW_LIMB][r].clone()
                } else {
                    columns[COL_MM_SLACK_HI_LIMBS + (k - 1)][r].clone()
                };
                let borrow_in = if k == 0 {
                    Scalar::zero(curve)
                } else {
                    columns[COL_MM_SLACK_BORROWS + (k - 1)][r].clone()
                };
                let borrow_out = if k + 1 < MULMOD_LIMBS {
                    columns[COL_MM_SLACK_BORROWS + k][r].clone()
                } else {
                    // Final borrow forced to zero ⇒ seal.
                    Scalar::zero(curve)
                };
                let lhs = m_k.add(&borrow_out.mul(&r64));
                let rhs = r_k.add(&one_k).add(&slack_k).add(&borrow_in);
                let body = lhs.sub(&rhs);
                let gate = columns[COL_IS_MULMOD][r]
                    .mul(&columns[COL_MM_CHECK_R_LT_M_FULL][r]);
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }
        // is_check_r_lt_m_full binary (gated by is_mulmod).
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_MM_CHECK_R_LT_M_FULL][r];
            c[r] = columns[COL_IS_MULMOD][r].mul(&v.mul(&v.sub(&one)));
        }
        out.push(c);
        // AB carry 9-byte LE decomps (7 ×, gated by is_mulmod).
        // Carry-byte weights: 2^(8k) for k=0..8. The last (k=8) overflows
        // u64, so build via repeated multiplication by 2^8.
        let carry_weights: Vec<Scalar> = {
            let mut acc = Scalar::one(curve);
            let two_8 = Scalar::from_u64(1u64 << 8, curve);
            let mut v = Vec::with_capacity(CARRY_BYTES);
            for _ in 0..CARRY_BYTES {
                v.push(acc.clone());
                acc = acc.mul(&two_8);
            }
            v
        };
        for k in 0..NUM_CARRIES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for b in 0..CARRY_BYTES {
                    let term = columns
                        [COL_MM_AB_CARRY_BYTES + k * CARRY_BYTES + b][r]
                        .mul(&carry_weights[b]);
                    sum = sum.add(&term);
                }
                let body = sum.sub(&columns[COL_MM_AB_CARRY + k][r]);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }
        // QM carry 9-byte LE decomps (7 ×, gated by is_mulmod).
        for k in 0..NUM_CARRIES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for b in 0..CARRY_BYTES {
                    let term = columns
                        [COL_MM_QM_CARRY_BYTES + k * CARRY_BYTES + b][r]
                        .mul(&carry_weights[b]);
                    sum = sum.add(&term);
                }
                let body = sum.sub(&columns[COL_MM_QM_CARRY + k][r]);
                c[r] = columns[COL_IS_MULMOD][r].mul(&body);
            }
            out.push(c);
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        let v = &col_evals[COL_IS_REAL];
        acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
        ap = ap.mul(alpha);

        let v = &col_evals[COL_IS_RESULT_ROW];
        acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
        ap = ap.mul(alpha);

        let rr = &col_evals[COL_IS_RESULT_ROW];
        let real = &col_evals[COL_IS_REAL];
        acc = acc.add(&ap.mul(&rr.mul(&one.sub(real))));
        ap = ap.mul(alpha);

        for k in 0..WINDOW_BITS {
            let v = &col_evals[COL_CURRENT_WINDOW_BITS_OFFSET + k];
            acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
            ap = ap.mul(alpha);
        }

        let mut sum = Scalar::zero(curve);
        for b in 0..ITERATION_INDEX_BYTES {
            let w = Scalar::from_u64(1u64 << (8 * b), curve);
            sum = sum.add(&col_evals[COL_ITERATION_INDEX_BYTE_OFFSET + b].mul(&w));
        }
        acc = acc.add(&ap.mul(&sum.sub(&col_evals[COL_ITERATION_INDEX])));
        ap = ap.mul(alpha);

        let mut sum = Scalar::zero(curve);
        for i in 0..RESULT_LENGTH {
            sum = sum.add(&col_evals[COL_RESULT_OFFSET + i]);
        }
        let gate = one.sub(&col_evals[COL_IS_RESULT_ROW]);
        acc = acc.add(&ap.mul(&gate.mul(&sum)));
        ap = ap.mul(alpha);

        // ─── Mul-mod gadget ──────────────────────────────────────────
        let r64 = two_64(curve);
        let sel = &col_evals[COL_IS_MULMOD];

        // is_mulmod binary.
        acc = acc.add(&ap.mul(&sel.mul(&sel.sub(&one))));
        ap = ap.mul(alpha);

        // PP-AB.
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                let a = &col_evals[COL_MM_A_LIMBS + i];
                let b = &col_evals[COL_MM_B_LIMBS + j];
                let pp = &col_evals[COL_MM_PP_AB + i * MULMOD_LIMBS + j];
                acc = acc.add(&ap.mul(&sel.mul(&a.mul(b).sub(pp))));
                ap = ap.mul(alpha);
            }
        }
        // PP-QM.
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                let a = &col_evals[COL_MM_Q_LIMBS + i];
                let b = &col_evals[COL_MM_M_LO_LIMBS + j];
                let pp = &col_evals[COL_MM_PP_QM + i * MULMOD_LIMBS + j];
                acc = acc.add(&ap.mul(&sel.mul(&a.mul(b).sub(pp))));
                ap = ap.mul(alpha);
            }
        }
        // AB column sums.
        for k in 0..(2 * MULMOD_LIMBS) {
            let mut rhs = Scalar::zero(curve);
            for i in 0..MULMOD_LIMBS {
                let j = k as i32 - i as i32;
                if j >= 0 && (j as usize) < MULMOD_LIMBS {
                    rhs = rhs.add(
                        &col_evals[COL_MM_PP_AB + i * MULMOD_LIMBS + j as usize],
                    );
                }
            }
            if k > 0 {
                rhs = rhs.add(&col_evals[COL_MM_AB_CARRY + (k - 1)]);
            }
            let mut lhs = col_evals[COL_MM_AB_OUT + k].clone();
            if k + 1 < 2 * MULMOD_LIMBS {
                lhs = lhs.add(&col_evals[COL_MM_AB_CARRY + k].mul(&r64));
            }
            acc = acc.add(&ap.mul(&sel.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
        }
        // QM column sums.
        for k in 0..(2 * MULMOD_LIMBS) {
            let mut rhs = Scalar::zero(curve);
            for i in 0..MULMOD_LIMBS {
                let j = k as i32 - i as i32;
                if j >= 0 && (j as usize) < MULMOD_LIMBS {
                    rhs = rhs.add(
                        &col_evals[COL_MM_PP_QM + i * MULMOD_LIMBS + j as usize],
                    );
                }
            }
            if k > 0 {
                rhs = rhs.add(&col_evals[COL_MM_QM_CARRY + (k - 1)]);
            }
            let mut lhs = col_evals[COL_MM_QM_OUT + k].clone();
            if k + 1 < 2 * MULMOD_LIMBS {
                lhs = lhs.add(&col_evals[COL_MM_QM_CARRY + k].mul(&r64));
            }
            acc = acc.add(&ap.mul(&sel.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
        }
        // Identity low half.
        for k in 0..MULMOD_LIMBS {
            let mut lhs = col_evals[COL_MM_AB_OUT + k].clone();
            if k > 0 {
                lhs = lhs.add(&col_evals[COL_MM_ID_CARRY_LO + (k - 1)]);
            }
            let rhs = col_evals[COL_MM_QM_OUT + k]
                .add(&col_evals[COL_MM_R_LIMBS + k])
                .add(&col_evals[COL_MM_ID_CARRY_LO + k].mul(&r64));
            acc = acc.add(&ap.mul(&sel.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
        }
        // Identity high half.
        for k in 0..MULMOD_LIMBS {
            let kk = MULMOD_LIMBS + k;
            let carry_in = if k == 0 {
                col_evals[COL_MM_ID_CARRY_LO + (MULMOD_LIMBS - 1)].clone()
            } else {
                col_evals[COL_MM_ID_CARRY_HI + (k - 1)].clone()
            };
            let lhs = col_evals[COL_MM_AB_OUT + kk].add(&carry_in);
            let mut rhs = col_evals[COL_MM_QM_OUT + kk].clone();
            if k + 1 < MULMOD_LIMBS {
                rhs = rhs.add(&col_evals[COL_MM_ID_CARRY_HI + k].mul(&r64));
            }
            acc = acc.add(&ap.mul(&sel.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
        }

        // ─── Task #182 range / R<M ────────────────────────────────────
        let byte_weights: Vec<Scalar> = (0..BYTES_PER_LIMB)
            .map(|k| Scalar::from_u64(1u64 << (8 * k), curve))
            .collect();
        for i in 0..MULMOD_LIMBS {
            let mut sum = Scalar::zero(curve);
            for k in 0..BYTES_PER_LIMB {
                sum = sum.add(
                    &col_evals[COL_MM_R_LIMB_BYTES + i * BYTES_PER_LIMB + k]
                        .mul(&byte_weights[k]),
                );
            }
            acc = acc.add(
                &ap.mul(&sel.mul(&sum.sub(&col_evals[COL_MM_R_LIMBS + i]))),
            );
            ap = ap.mul(alpha);
        }
        for i in 0..MULMOD_LIMBS {
            let mut sum = Scalar::zero(curve);
            for k in 0..BYTES_PER_LIMB {
                sum = sum.add(
                    &col_evals[COL_MM_Q_LIMB_BYTES + i * BYTES_PER_LIMB + k]
                        .mul(&byte_weights[k]),
                );
            }
            acc = acc.add(
                &ap.mul(&sel.mul(&sum.sub(&col_evals[COL_MM_Q_LIMBS + i]))),
            );
            ap = ap.mul(alpha);
        }
        // Slack byte decomp.
        let mut sum = Scalar::zero(curve);
        for k in 0..BYTES_PER_LIMB {
            sum = sum.add(
                &col_evals[COL_MM_SLACK_LOW_BYTES + k].mul(&byte_weights[k]),
            );
        }
        acc = acc.add(
            &ap.mul(&sel.mul(&sum.sub(&col_evals[COL_MM_SLACK_LOW_LIMB]))),
        );
        ap = ap.mul(alpha);
        // is_check_r_lt_m_low binary (gated by is_mulmod).
        let chk = &col_evals[COL_MM_CHECK_R_LT_M_LOW];
        acc = acc.add(&ap.mul(&sel.mul(&chk.mul(&chk.sub(&one)))));
        ap = ap.mul(alpha);
        // R<M low-limb uniqueness (gated).
        let m0 = &col_evals[COL_MM_M_LO_LIMBS];
        let r0 = &col_evals[COL_MM_R_LIMBS];
        let slack = &col_evals[COL_MM_SLACK_LOW_LIMB];
        let body = m0.sub(r0).sub(&one).sub(slack);
        acc = acc.add(&ap.mul(&sel.mul(chk).mul(&body)));
        ap = ap.mul(alpha);

        // ─── Task #196: full multi-limb R<M + carry range ────────────
        // Slack high-limb byte decomps (3 ×).
        for i in 0..(MULMOD_LIMBS - 1) {
            let mut sum = Scalar::zero(curve);
            for k in 0..BYTES_PER_LIMB {
                sum = sum.add(
                    &col_evals[COL_MM_SLACK_HI_BYTES + i * BYTES_PER_LIMB + k]
                        .mul(&byte_weights[k]),
                );
            }
            acc = acc.add(
                &ap.mul(
                    &sel.mul(&sum.sub(&col_evals[COL_MM_SLACK_HI_LIMBS + i])),
                ),
            );
            ap = ap.mul(alpha);
        }
        // Borrow-binary (3 ×).
        for i in 0..(MULMOD_LIMBS - 1) {
            let v = &col_evals[COL_MM_SLACK_BORROWS + i];
            let body = v.mul(&v.sub(&one));
            acc = acc.add(&ap.mul(&sel.mul(&body)));
            ap = ap.mul(alpha);
        }
        // 4-limb subtraction chain.
        let chk_full = &col_evals[COL_MM_CHECK_R_LT_M_FULL];
        for k in 0..MULMOD_LIMBS {
            let one_k = if k == 0 { one.clone() } else { Scalar::zero(curve) };
            let m_k = &col_evals[COL_MM_M_LO_LIMBS + k];
            let r_k = &col_evals[COL_MM_R_LIMBS + k];
            let slack_k = if k == 0 {
                col_evals[COL_MM_SLACK_LOW_LIMB].clone()
            } else {
                col_evals[COL_MM_SLACK_HI_LIMBS + (k - 1)].clone()
            };
            let borrow_in = if k == 0 {
                Scalar::zero(curve)
            } else {
                col_evals[COL_MM_SLACK_BORROWS + (k - 1)].clone()
            };
            let borrow_out = if k + 1 < MULMOD_LIMBS {
                col_evals[COL_MM_SLACK_BORROWS + k].clone()
            } else {
                Scalar::zero(curve)
            };
            let lhs = m_k.add(&borrow_out.mul(&r64));
            let rhs = r_k.add(&one_k).add(&slack_k).add(&borrow_in);
            let body = lhs.sub(&rhs);
            acc = acc.add(&ap.mul(&sel.mul(chk_full).mul(&body)));
            ap = ap.mul(alpha);
        }
        // is_check_r_lt_m_full binary.
        let body = chk_full.mul(&chk_full.sub(&one));
        acc = acc.add(&ap.mul(&sel.mul(&body)));
        ap = ap.mul(alpha);
        // AB / QM carry decomps.
        // Carry-byte weights: 2^(8k) for k=0..8. The last (k=8) overflows
        // u64, so build via repeated multiplication by 2^8.
        let carry_weights: Vec<Scalar> = {
            let mut acc = Scalar::one(curve);
            let two_8 = Scalar::from_u64(1u64 << 8, curve);
            let mut v = Vec::with_capacity(CARRY_BYTES);
            for _ in 0..CARRY_BYTES {
                v.push(acc.clone());
                acc = acc.mul(&two_8);
            }
            v
        };
        for k in 0..NUM_CARRIES {
            let mut sum = Scalar::zero(curve);
            for b in 0..CARRY_BYTES {
                sum = sum.add(
                    &col_evals[COL_MM_AB_CARRY_BYTES + k * CARRY_BYTES + b]
                        .mul(&carry_weights[b]),
                );
            }
            acc = acc.add(
                &ap.mul(&sel.mul(&sum.sub(&col_evals[COL_MM_AB_CARRY + k]))),
            );
            ap = ap.mul(alpha);
        }
        for k in 0..NUM_CARRIES {
            let mut sum = Scalar::zero(curve);
            for b in 0..CARRY_BYTES {
                sum = sum.add(
                    &col_evals[COL_MM_QM_CARRY_BYTES + k * CARRY_BYTES + b]
                        .mul(&carry_weights[b]),
                );
            }
            acc = acc.add(
                &ap.mul(&sel.mul(&sum.sub(&col_evals[COL_MM_QM_CARRY + k]))),
            );
            ap = ap.mul(alpha);
        }
        let _ = ap;

        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);

        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        let v = &col_coeffs[COL_IS_RESULT_ROW];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        let rr = &col_coeffs[COL_IS_RESULT_ROW];
        let real = &col_coeffs[COL_IS_REAL];
        let one_minus = poly_sub(&one_poly, real, curve);
        let body = poly_mul(rr, &one_minus, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        for k in 0..WINDOW_BITS {
            let v = &col_coeffs[COL_CURRENT_WINDOW_BITS_OFFSET + k];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        let mut sum = vec![Scalar::zero(curve)];
        for b in 0..ITERATION_INDEX_BYTES {
            let w = Scalar::from_u64(1u64 << (8 * b), curve);
            let term =
                poly_scalar_mul(&col_coeffs[COL_ITERATION_INDEX_BYTE_OFFSET + b], &w);
            sum = poly_add(&sum, &term, curve);
        }
        let body = poly_sub(&sum, &col_coeffs[COL_ITERATION_INDEX], curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        let mut sum = vec![Scalar::zero(curve)];
        for i in 0..RESULT_LENGTH {
            sum = poly_add(&sum, &col_coeffs[COL_RESULT_OFFSET + i], curve);
        }
        let gate = poly_sub(&one_poly, &col_coeffs[COL_IS_RESULT_ROW], curve);
        let body = poly_mul(&gate, &sum, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // ─── Mul-mod gadget ──────────────────────────────────────────
        let r64 = two_64(curve);
        let sel = &col_coeffs[COL_IS_MULMOD];

        // is_mulmod binary.
        let sel_m1 = poly_sub(sel, &one_poly, curve);
        let body = poly_mul(sel, &sel_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // PP-AB.
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                let a = &col_coeffs[COL_MM_A_LIMBS + i];
                let b = &col_coeffs[COL_MM_B_LIMBS + j];
                let pp = &col_coeffs[COL_MM_PP_AB + i * MULMOD_LIMBS + j];
                let ab_prod = poly_mul(a, b, curve);
                let body = poly_mul(sel, &poly_sub(&ab_prod, pp, curve), curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
                ap = ap.mul(alpha);
            }
        }
        // PP-QM.
        for i in 0..MULMOD_LIMBS {
            for j in 0..MULMOD_LIMBS {
                let a = &col_coeffs[COL_MM_Q_LIMBS + i];
                let b = &col_coeffs[COL_MM_M_LO_LIMBS + j];
                let pp = &col_coeffs[COL_MM_PP_QM + i * MULMOD_LIMBS + j];
                let ab_prod = poly_mul(a, b, curve);
                let body = poly_mul(sel, &poly_sub(&ab_prod, pp, curve), curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
                ap = ap.mul(alpha);
            }
        }
        // AB column sums.
        for k in 0..(2 * MULMOD_LIMBS) {
            let mut rhs = vec![Scalar::zero(curve)];
            for i in 0..MULMOD_LIMBS {
                let j = k as i32 - i as i32;
                if j >= 0 && (j as usize) < MULMOD_LIMBS {
                    rhs = poly_add(
                        &rhs,
                        &col_coeffs[COL_MM_PP_AB + i * MULMOD_LIMBS + j as usize],
                        curve,
                    );
                }
            }
            if k > 0 {
                rhs = poly_add(
                    &rhs,
                    &col_coeffs[COL_MM_AB_CARRY + (k - 1)],
                    curve,
                );
            }
            let mut lhs = col_coeffs[COL_MM_AB_OUT + k].clone();
            if k + 1 < 2 * MULMOD_LIMBS {
                lhs = poly_add(
                    &lhs,
                    &poly_scalar_mul(&col_coeffs[COL_MM_AB_CARRY + k], &r64),
                    curve,
                );
            }
            let body = poly_mul(sel, &poly_sub(&lhs, &rhs, curve), curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // QM column sums.
        for k in 0..(2 * MULMOD_LIMBS) {
            let mut rhs = vec![Scalar::zero(curve)];
            for i in 0..MULMOD_LIMBS {
                let j = k as i32 - i as i32;
                if j >= 0 && (j as usize) < MULMOD_LIMBS {
                    rhs = poly_add(
                        &rhs,
                        &col_coeffs[COL_MM_PP_QM + i * MULMOD_LIMBS + j as usize],
                        curve,
                    );
                }
            }
            if k > 0 {
                rhs = poly_add(
                    &rhs,
                    &col_coeffs[COL_MM_QM_CARRY + (k - 1)],
                    curve,
                );
            }
            let mut lhs = col_coeffs[COL_MM_QM_OUT + k].clone();
            if k + 1 < 2 * MULMOD_LIMBS {
                lhs = poly_add(
                    &lhs,
                    &poly_scalar_mul(&col_coeffs[COL_MM_QM_CARRY + k], &r64),
                    curve,
                );
            }
            let body = poly_mul(sel, &poly_sub(&lhs, &rhs, curve), curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // Identity low half.
        for k in 0..MULMOD_LIMBS {
            let mut lhs = col_coeffs[COL_MM_AB_OUT + k].clone();
            if k > 0 {
                lhs = poly_add(
                    &lhs,
                    &col_coeffs[COL_MM_ID_CARRY_LO + (k - 1)],
                    curve,
                );
            }
            let mut rhs = poly_add(
                &col_coeffs[COL_MM_QM_OUT + k],
                &col_coeffs[COL_MM_R_LIMBS + k],
                curve,
            );
            rhs = poly_add(
                &rhs,
                &poly_scalar_mul(&col_coeffs[COL_MM_ID_CARRY_LO + k], &r64),
                curve,
            );
            let body = poly_mul(sel, &poly_sub(&lhs, &rhs, curve), curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // Identity high half.
        for k in 0..MULMOD_LIMBS {
            let kk = MULMOD_LIMBS + k;
            let carry_in = if k == 0 {
                col_coeffs[COL_MM_ID_CARRY_LO + (MULMOD_LIMBS - 1)].clone()
            } else {
                col_coeffs[COL_MM_ID_CARRY_HI + (k - 1)].clone()
            };
            let lhs = poly_add(&col_coeffs[COL_MM_AB_OUT + kk], &carry_in, curve);
            let mut rhs = col_coeffs[COL_MM_QM_OUT + kk].clone();
            if k + 1 < MULMOD_LIMBS {
                rhs = poly_add(
                    &rhs,
                    &poly_scalar_mul(&col_coeffs[COL_MM_ID_CARRY_HI + k], &r64),
                    curve,
                );
            }
            let body = poly_mul(sel, &poly_sub(&lhs, &rhs, curve), curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // ─── Task #182 range / R<M ───────────────────────────────────
        let byte_weights: Vec<Scalar> = (0..BYTES_PER_LIMB)
            .map(|k| Scalar::from_u64(1u64 << (8 * k), curve))
            .collect();
        for i in 0..MULMOD_LIMBS {
            let mut sum = vec![Scalar::zero(curve)];
            for k in 0..BYTES_PER_LIMB {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_MM_R_LIMB_BYTES + i * BYTES_PER_LIMB + k],
                    &byte_weights[k],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_mul(
                sel,
                &poly_sub(&sum, &col_coeffs[COL_MM_R_LIMBS + i], curve),
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        for i in 0..MULMOD_LIMBS {
            let mut sum = vec![Scalar::zero(curve)];
            for k in 0..BYTES_PER_LIMB {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_MM_Q_LIMB_BYTES + i * BYTES_PER_LIMB + k],
                    &byte_weights[k],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_mul(
                sel,
                &poly_sub(&sum, &col_coeffs[COL_MM_Q_LIMBS + i], curve),
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // Slack byte decomp.
        let mut sum = vec![Scalar::zero(curve)];
        for k in 0..BYTES_PER_LIMB {
            let term = poly_scalar_mul(
                &col_coeffs[COL_MM_SLACK_LOW_BYTES + k],
                &byte_weights[k],
            );
            sum = poly_add(&sum, &term, curve);
        }
        let body = poly_mul(
            sel,
            &poly_sub(&sum, &col_coeffs[COL_MM_SLACK_LOW_LIMB], curve),
            curve,
        );
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
        // is_check_r_lt_m_low binary (gated by is_mulmod).
        let chk = &col_coeffs[COL_MM_CHECK_R_LT_M_LOW];
        let chk_m1 = poly_sub(chk, &one_poly, curve);
        let chk_bin = poly_mul(chk, &chk_m1, curve);
        let body = poly_mul(sel, &chk_bin, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
        // R<M low-limb uniqueness (gated by chk).
        let m0 = &col_coeffs[COL_MM_M_LO_LIMBS];
        let r0 = &col_coeffs[COL_MM_R_LIMBS];
        let slack = &col_coeffs[COL_MM_SLACK_LOW_LIMB];
        let body_lt = poly_sub(
            &poly_sub(&poly_sub(m0, r0, curve), &one_poly, curve),
            slack,
            curve,
        );
        let gated = poly_mul(&poly_mul(sel, chk, curve), &body_lt, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &ap), curve);
        ap = ap.mul(alpha);

        // ─── Task #196: full multi-limb R<M + carry range ────────────
        // Slack high-limb byte decomps.
        for i in 0..(MULMOD_LIMBS - 1) {
            let mut sum = vec![Scalar::zero(curve)];
            for k in 0..BYTES_PER_LIMB {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_MM_SLACK_HI_BYTES + i * BYTES_PER_LIMB + k],
                    &byte_weights[k],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_mul(
                sel,
                &poly_sub(&sum, &col_coeffs[COL_MM_SLACK_HI_LIMBS + i], curve),
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // Borrow-binary.
        for i in 0..(MULMOD_LIMBS - 1) {
            let v = &col_coeffs[COL_MM_SLACK_BORROWS + i];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(sel, &poly_mul(v, &v_m1, curve), curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // 4-limb subtraction chain.
        let chk_full = &col_coeffs[COL_MM_CHECK_R_LT_M_FULL];
        let gate_full = poly_mul(sel, chk_full, curve);
        for k in 0..MULMOD_LIMBS {
            let one_k = if k == 0 { one_poly.clone() } else { vec![Scalar::zero(curve)] };
            let m_k = &col_coeffs[COL_MM_M_LO_LIMBS + k];
            let r_k = &col_coeffs[COL_MM_R_LIMBS + k];
            let slack_k = if k == 0 {
                col_coeffs[COL_MM_SLACK_LOW_LIMB].clone()
            } else {
                col_coeffs[COL_MM_SLACK_HI_LIMBS + (k - 1)].clone()
            };
            let borrow_in = if k == 0 {
                vec![Scalar::zero(curve)]
            } else {
                col_coeffs[COL_MM_SLACK_BORROWS + (k - 1)].clone()
            };
            let borrow_out = if k + 1 < MULMOD_LIMBS {
                col_coeffs[COL_MM_SLACK_BORROWS + k].clone()
            } else {
                vec![Scalar::zero(curve)]
            };
            let lhs = poly_add(m_k, &poly_scalar_mul(&borrow_out, &r64), curve);
            let mut rhs = poly_add(r_k, &one_k, curve);
            rhs = poly_add(&rhs, &slack_k, curve);
            rhs = poly_add(&rhs, &borrow_in, curve);
            let body = poly_mul(&gate_full, &poly_sub(&lhs, &rhs, curve), curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // is_check_r_lt_m_full binary.
        let chk_full_m1 = poly_sub(chk_full, &one_poly, curve);
        let body = poly_mul(sel, &poly_mul(chk_full, &chk_full_m1, curve), curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
        // AB / QM carry decomps.
        // Carry-byte weights: 2^(8k) for k=0..8. The last (k=8) overflows
        // u64, so build via repeated multiplication by 2^8.
        let carry_weights: Vec<Scalar> = {
            let mut acc = Scalar::one(curve);
            let two_8 = Scalar::from_u64(1u64 << 8, curve);
            let mut v = Vec::with_capacity(CARRY_BYTES);
            for _ in 0..CARRY_BYTES {
                v.push(acc.clone());
                acc = acc.mul(&two_8);
            }
            v
        };
        for k in 0..NUM_CARRIES {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..CARRY_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_MM_AB_CARRY_BYTES + k * CARRY_BYTES + b],
                    &carry_weights[b],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_mul(
                sel,
                &poly_sub(&sum, &col_coeffs[COL_MM_AB_CARRY + k], curve),
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        for k in 0..NUM_CARRIES {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..CARRY_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_MM_QM_CARRY_BYTES + k * CARRY_BYTES + b],
                    &carry_weights[b],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_mul(
                sel,
                &poly_sub(&sum, &col_coeffs[COL_MM_QM_CARRY + k], curve),
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Silence unused (last ap *= alpha cycle).
        let _ = ap;

        acc
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Shifted (in order, used by evaluate_shifted_at_point):
        //   0..BIGINT_LENGTH:           partial_result[k]
        //   BIGINT_LENGTH:              is_real
        //   BIGINT_LENGTH+1:            is_result_row
        //   (Task #182 mulmod chain — re-uses partial_result above and
        //   additionally needs the *next-row* partial_result low 8
        //   bytes, which is the same slice; no new columns needed.)
        let mut cols = Vec::with_capacity(BIGINT_LENGTH + 2);
        for k in 0..BIGINT_LENGTH {
            cols.push(COL_PARTIAL_RESULT_OFFSET + k);
        }
        cols.push(COL_IS_REAL);
        cols.push(COL_IS_RESULT_ROW);
        cols
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        let expected = BIGINT_LENGTH + 2;
        if shifted_evals.len() != expected || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let real_z = &col_evals_at_z[COL_IS_REAL];
        let real_next = &shifted_evals[BIGINT_LENGTH];
        let is_result = &col_evals_at_z[COL_IS_RESULT_ROW];
        let gating = real_z.mul(real_next).mul(&one.sub(is_result));

        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..BIGINT_LENGTH {
            let nxt = &shifted_evals[k];
            let cur_next = &col_evals_at_z[COL_NEXT_PARTIAL_RESULT_OFFSET + k];
            acc = acc.add(&bp.mul(&nxt.sub(cur_next)));
            bp = bp.mul(alpha);
        }
        let body = gating.mul(&acc);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let zero_at_last = z.sub(omega_n_minus_1);
        let mut total = ap.mul(&body).mul(&zero_at_last);

        // ─── Task #182 mulmod cross-row chain ─────────────────────────
        // Bind low 8 bytes of partial_result(ω·X) to R-limb[0]'s byte
        // decomposition columns at X. Gated by `is_mulmod(X) ·
        // is_real(ω·X)`. RLC the 8 byte equalities with β = α (re-using
        // α from outer transcript; mirrors existing convention).
        let sel = &col_evals_at_z[COL_IS_MULMOD];
        let g2 = sel.mul(real_next);
        let mut chain = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..BYTES_PER_LIMB {
            let nxt_byte = &shifted_evals[k];
            let r_byte = &col_evals_at_z[COL_MM_R_LIMB_BYTES + k];
            chain = chain.add(&bp.mul(&nxt_byte.sub(r_byte)));
            bp = bp.mul(alpha);
        }
        // alpha_offset+1 is the next shifted slot.
        let ap2 = ap.mul(alpha);
        total = total.add(&ap2.mul(&g2.mul(&chain)).mul(&zero_at_last));

        total
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let real = &column_coeffs[COL_IS_REAL];
        let real_next = poly_shift(real, omega);
        let is_result = &column_coeffs[COL_IS_RESULT_ROW];
        let one_minus = poly_sub(&one_poly, is_result, curve);
        let gating = poly_mul(&poly_mul(real, &real_next, curve), &one_minus, curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..BIGINT_LENGTH {
            let cur = &column_coeffs[COL_PARTIAL_RESULT_OFFSET + k];
            let nxt = poly_shift(cur, omega);
            let cur_next = &column_coeffs[COL_NEXT_PARTIAL_RESULT_OFFSET + k];
            let diff = poly_sub(&nxt, cur_next, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body0 = poly_mul(&gating, &acc, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut total = poly_scalar_mul(&body0, &ap);

        // ─── Task #182 mulmod cross-row chain ─────────────────────────
        // partial_result(ω·X)[k] = r_limb_bytes[k](X) for k in 0..8.
        // Gated by is_mulmod(X) · is_real(ω·X).
        let sel = &column_coeffs[COL_IS_MULMOD];
        let g2 = poly_mul(sel, &real_next, curve);
        let mut chain = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..BYTES_PER_LIMB {
            let cur = &column_coeffs[COL_PARTIAL_RESULT_OFFSET + k];
            let nxt = poly_shift(cur, omega);
            let r_byte = &column_coeffs[COL_MM_R_LIMB_BYTES + k];
            let diff = poly_sub(&nxt, r_byte, curve);
            chain = poly_add(&chain, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body1 = poly_mul(&g2, &chain, curve);
        let ap2 = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body1, &ap2), curve);

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_IS_RESULT_ROW, COL_IS_MULMOD]
    }

    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..BIGINT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_internals_b_byte_{}_8bit", k),
                    column_index: COL_B_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_internals_e_byte_{}_8bit", k),
                    column_index: COL_E_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_internals_m_byte_{}_8bit", k),
                    column_index: COL_M_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_internals_partial_byte_{}_8bit", k),
                    column_index: COL_PARTIAL_RESULT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_internals_next_partial_byte_{}_8bit", k),
                    column_index: COL_NEXT_PARTIAL_RESULT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..RESULT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_internals_result_byte_{}_8bit", k),
                    column_index: COL_RESULT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..ITERATION_INDEX_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_internals_iter_idx_byte_{}_8bit", b),
                    column_index: COL_ITERATION_INDEX_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // ─── Task #182: byte range checks for new decomp columns ──────
        for i in 0..MULMOD_LIMBS {
            for k in 0..BYTES_PER_LIMB {
                declarations.push((
                    LookupDeclaration {
                        label: format!(
                            "modexp_internals_mm_r_limb_{}_byte_{}_8bit",
                            i, k
                        ),
                        column_index: COL_MM_R_LIMB_BYTES
                            + i * BYTES_PER_LIMB
                            + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
                declarations.push((
                    LookupDeclaration {
                        label: format!(
                            "modexp_internals_mm_q_limb_{}_byte_{}_8bit",
                            i, k
                        ),
                        column_index: COL_MM_Q_LIMB_BYTES
                            + i * BYTES_PER_LIMB
                            + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        for k in 0..BYTES_PER_LIMB {
            declarations.push((
                LookupDeclaration {
                    label: format!(
                        "modexp_internals_mm_slack_low_byte_{}_8bit",
                        k
                    ),
                    column_index: COL_MM_SLACK_LOW_BYTES + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // Task #196: byte range checks for new decomp columns.
        for i in 0..(MULMOD_LIMBS - 1) {
            for k in 0..BYTES_PER_LIMB {
                declarations.push((
                    LookupDeclaration {
                        label: format!(
                            "modexp_internals_mm_slack_hi_{}_byte_{}_8bit",
                            i, k
                        ),
                        column_index: COL_MM_SLACK_HI_BYTES
                            + i * BYTES_PER_LIMB
                            + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        for k in 0..NUM_CARRIES {
            for b in 0..CARRY_BYTES {
                declarations.push((
                    LookupDeclaration {
                        label: format!(
                            "modexp_internals_mm_ab_carry_{}_byte_{}_8bit",
                            k, b
                        ),
                        column_index: COL_MM_AB_CARRY_BYTES + k * CARRY_BYTES + b,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
                declarations.push((
                    LookupDeclaration {
                        label: format!(
                            "modexp_internals_mm_qm_carry_{}_byte_{}_8bit",
                            k, b
                        ),
                        column_index: COL_MM_QM_CARRY_BYTES + k * CARRY_BYTES + b,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Placeholder sentinel for the precompile-side output column.
pub const PRECOMPILE_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (skeleton): binds the FINAL-row
/// `result[0..BIGINT_LENGTH]` of the internals AIR to the `output`
/// column of [`crate::modexp_precompile_air`].
pub fn make_modexp_internals_to_precompile_descriptor(
    internals_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(RESULT_LENGTH);
    for k in 0..RESULT_LENGTH {
        a_columns.push(COL_RESULT_OFFSET + k);
    }
    let b_columns: Vec<usize> = vec![PRECOMPILE_PLACEHOLDER; RESULT_LENGTH];
    CrossAirLogUpDescriptor {
        label: "modexp_internals_to_precompile_result_v1_stub".into(),
        a_layer_index: internals_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_RESULT_ROW),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} row {} nonzero: {:?}",
                    i, r, v,
                );
            }
        }
    }

    #[test]
    fn modexp_internals_air_zero_result_passes() {
        let w = ModExpInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn modexp_internals_air_real_result_passes() {
        // Build a precompile witness for B=3, E=5, M=7 (3^5 = 243 mod 7 = 5)
        // then take its output as the internals AIR's result.
        let precompile_w =
            crate::modexp_precompile_air::ModExpPrecompileWitness::from_inputs(
                &[3u8],
                &[5u8],
                &[7u8],
            );
        let mut result = [0u8; RESULT_LENGTH];
        let raw = &precompile_w.output[..RESULT_LENGTH.min(BIGINT_LENGTH)];
        result.copy_from_slice(raw);
        let w = ModExpInternalsTraceWitness::from_result_skeleton(result);
        assert_eq!(w.rows[1].result, result);

        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn modexp_internals_air_tampered_result_off_final_row_detected() {
        let w = ModExpInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_RESULT_OFFSET + 10][0] =
            Scalar::from_u64(0xab, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Last legacy constraint is `result_zero_off_final_row` at
        // index NUM_ROW_CONSTRAINTS_LEGACY - 1.
        let last = NUM_ROW_CONSTRAINTS_LEGACY - 1;
        assert!(
            !bodies[last][0].is_zero(),
            "expected gated-result constraint to fire on non-final row",
        );
    }

    #[test]
    fn modexp_internals_air_tampered_window_bit_detected() {
        let w = ModExpInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // current_window_bits[0] = 3 (not binary).
        cols[COL_CURRENT_WINDOW_BITS_OFFSET + 0][1] =
            Scalar::from_u64(3, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index: 3 (is_real_bin, is_result_row_bin, gating) + 0.
        assert!(
            !bodies[3][1].is_zero(),
            "expected current_window_bits binary constraint to fire",
        );
    }

    #[test]
    fn modexp_internals_air_descriptor_well_formed() {
        let d = make_modexp_internals_to_precompile_descriptor(0, 1);
        assert_eq!(d.label, "modexp_internals_to_precompile_result_v1_stub");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), RESULT_LENGTH);
        assert_eq!(d.b_columns.len(), RESULT_LENGTH);
        assert_eq!(d.a_selector_column, Some(COL_IS_RESULT_ROW));
        for k in 0..RESULT_LENGTH {
            assert_eq!(d.a_columns[k], COL_RESULT_OFFSET + k);
            assert_eq!(d.b_columns[k], PRECOMPILE_PLACEHOLDER);
        }
    }

    // ─── Mul-mod gadget tests ────────────────────────────────────────

    fn limbs_from_u64(x: u64) -> [u64; MULMOD_LIMBS] {
        let mut out = [0u64; MULMOD_LIMBS];
        out[0] = x;
        out
    }

    #[test]
    fn mulmod_witness_closes_small_inputs() {
        // 7 * 11 mod 13 = 77 mod 13 = 12.
        let mm = MulModWitness::from_inputs(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        assert_eq!(mm.r_limbs[0], 12);
        assert_eq!(mm.r_limbs[1], 0);
        assert_eq!(mm.q_limbs[0], 5); // 77 / 13 = 5
        // PP[0][0] = 77.
        assert_eq!(mm.pp_ab[0], 77);
        // AB output limb 0 = 77, others = 0.
        assert_eq!(mm.ab_out[0], 77);
        for k in 1..(2 * MULMOD_LIMBS) {
            assert_eq!(mm.ab_out[k], 0);
        }
        // Identity carries all zero.
        for k in 0..MULMOD_LIMBS {
            assert_eq!(mm.id_carry_lo[k], 0);
            assert_eq!(mm.id_carry_hi[k], 0);
        }
    }

    #[test]
    fn mulmod_air_passes_small_inputs() {
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn mulmod_air_passes_squaring_with_carry() {
        // Square a value whose product overflows the bottom limb to
        // exercise the AB column carry chain.
        let a = u64::MAX; // 2^64 - 1
        let m = 0xdeadbeef_cafebabeu64; // arbitrary nonzero modulus
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(a),
            limbs_from_u64(a),
            limbs_from_u64(m),
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} nonzero ({})",
                    i,
                    r,
                    cs.constraint_labels()[i]
                );
            }
        }
    }

    #[test]
    fn mulmod_air_passes_full_4_limb_inputs() {
        // Use four nonzero limbs each for A, B, M to exercise the full
        // 4×4 schoolbook with cross-column carries.
        let a = [0x1234_5678u64, 0x9abc_def0, 0xfeed_face, 0x1357_9bdf];
        let b = [0xdead_beefu64, 0xcafe_babe, 0xf00d_d00d, 0x0bad_face];
        let m = [0x0001_0001u64, 0x0002_0002, 0x0003_0003, 0x0004_0005];
        let w = ModExpInternalsTraceWitness::from_mulmod_step(a, b, m);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} nonzero ({})",
                    i,
                    r,
                    cs.constraint_labels()[i]
                );
            }
        }
    }

    #[test]
    fn mulmod_air_tampered_r_detected() {
        let mut w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        // Tamper R: change low limb.
        w.rows[0].mulmod.r_limbs[0] = 11; // honest is 12
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // The identity-low-k=0 constraint must fire.
        // Layout: legacy(9) + is_mulmod(1) + pp_ab(16) + pp_qm(16)
        //         + ab_col(8) + qm_col(8) = 58; then id_lo[0] = 58.
        let id_lo_0 = 58;
        assert!(
            !bodies[id_lo_0][0].is_zero(),
            "expected id_lo[0] to fire on tampered R",
        );
    }

    #[test]
    fn mulmod_air_tampered_pp_detected() {
        let mut w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        // Tamper PP[0][0]: honest = 77, set to 78.
        w.rows[0].mulmod.pp_ab[0] = 78;
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // PP-AB constraints start at index 9 (legacy) + 1 (is_mulmod) = 10.
        let pp_ab_0 = 10;
        assert!(
            !bodies[pp_ab_0][0].is_zero(),
            "expected PP-AB[0][0] to fire on tampered partial product",
        );
    }

    #[test]
    fn mulmod_air_tampered_q_detected() {
        let mut w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        // Tamper Q: honest = 5, set to 6 (which would make Q*M = 78 ≠ 77).
        w.rows[0].mulmod.q_limbs[0] = 6;
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Tampering Q breaks PP-QM[0][0] (constraint index 9+1+16 = 26).
        let pp_qm_0 = 26;
        assert!(
            !bodies[pp_qm_0][0].is_zero(),
            "expected PP-QM[0][0] to fire on tampered Q",
        );
    }

    #[test]
    fn mulmod_air_at_point_matches_on_domain() {
        // The verifier-side `evaluate_at_point` must match the prover
        // -side `evaluate_on_domain` at every row.
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);

        let alpha = Scalar::from_u64(123_456_789, CurveType::Bls48581);
        let n = cr[0].len();
        for row in 0..n {
            let col_evals: Vec<Scalar> =
                (0..NUM_COLUMNS).map(|c| cr[c][row].clone()).collect();
            let at_point = cs.evaluate_at_point(&col_evals, &alpha);
            // RLC combination of constraint bodies with α^i.
            let mut expected = Scalar::zero(CurveType::Bls48581);
            let mut ap = Scalar::one(CurveType::Bls48581);
            for body in bodies.iter() {
                expected = expected.add(&ap.mul(&body[row]));
                ap = ap.mul(&alpha);
            }
            assert!(
                at_point.sub(&expected).is_zero(),
                "row {} evaluate_at_point != on-domain RLC",
                row,
            );
        }
    }

    // ─── Task #182 range / R<M / cross-row tests ───────────────────

    #[test]
    fn mulmod_air_r_byte_decomp_passes() {
        // Honest witness must satisfy: Σ r_byte[i][k]·2^(8k) = r_limbs[i].
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // R-limb byte decomp constraints start right after the mulmod
        // block: legacy(9) + mulmod(57) = 66. R-decomp 66..70.
        for i in 0..MULMOD_LIMBS {
            assert!(bodies[66 + i][0].is_zero());
        }
        // Slack row exists and is byte-decomposed (single-limb modulus
        // case: check_r_lt_m_low must be 1, slack = 13 - 12 - 1 = 0).
        let row0 = 0;
        let slack = cr[COL_MM_SLACK_LOW_LIMB][row0].clone();
        assert!(slack.is_zero());
        let chk = cr[COL_MM_CHECK_R_LT_M_LOW][row0].clone();
        assert!(chk.sub(&Scalar::one(CurveType::Bls48581)).is_zero());
    }

    #[test]
    fn mulmod_air_tampered_r_byte_detected() {
        // Tamper a byte of r_limbs[0] decomp; the binding constraint
        // must fire.
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // r_limbs[0] = 12 → bytes [12, 0, ...]. Bump byte 0 to 13.
        cols[COL_MM_R_LIMB_BYTES + 0][0] =
            Scalar::from_u64(13, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // R-decomp constraint for limb 0 sits at index 66.
        assert!(
            !bodies[66][0].is_zero(),
            "expected R-limb-0 byte-decomp constraint to fire",
        );
    }

    #[test]
    fn mulmod_air_tampered_q_byte_detected() {
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // q_limbs[0] = 5; tamper byte 0 → 6.
        cols[COL_MM_Q_LIMB_BYTES + 0][0] =
            Scalar::from_u64(6, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Q-decomp constraints start at 66 + MULMOD_LIMBS = 70.
        assert!(
            !bodies[70][0].is_zero(),
            "expected Q-limb-0 byte-decomp constraint to fire",
        );
    }

    #[test]
    fn mulmod_air_tampered_slack_detected_when_check_fires() {
        // single-limb M=13, R=12, slack honest = 0; tamper to 1 so
        // M - R - 1 - slack = 13 - 12 - 1 - 1 = -1 ≠ 0.
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            limbs_from_u64(13),
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_MM_SLACK_LOW_LIMB][0] =
            Scalar::from_u64(1, CurveType::Bls48581);
        cols[COL_MM_SLACK_LOW_BYTES + 0][0] =
            Scalar::from_u64(1, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // R<M-low constraint is the LAST constraint of the v182 range
        // block (legacy + mulmod + range — but BEFORE the v196 block).
        let last_v182 = NUM_ROW_CONSTRAINTS_LEGACY
            + NUM_ROW_CONSTRAINTS_MULMOD
            + NUM_ROW_CONSTRAINTS_RANGE
            - 1;
        assert!(
            !bodies[last_v182][0].is_zero(),
            "expected R<M low-limb constraint to fire on tampered slack",
        );
    }

    #[test]
    fn mulmod_air_check_selector_only_set_for_single_limb_modulus() {
        // Multi-limb modulus → check_r_lt_m_low must be 0, R<M
        // constraint must NOT fire (deferred-soundness path).
        let m = [13u64, 1, 0, 0]; // high limb 1 is nonzero
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            m,
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let chk = cr[COL_MM_CHECK_R_LT_M_LOW][0].clone();
        assert!(chk.is_zero(), "check_r_lt_m_low must be 0 for multi-limb M");
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        let last_v182 = NUM_ROW_CONSTRAINTS_LEGACY
            + NUM_ROW_CONSTRAINTS_MULMOD
            + NUM_ROW_CONSTRAINTS_RANGE
            - 1;
        assert!(
            bodies[last_v182][0].is_zero(),
            "R<M low-limb constraint must not fire when soft check is 0",
        );
    }

    #[test]
    fn mulmod_air_byte_lookup_declarations_count() {
        let cs = ModExpInternalsConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        // Legacy declarations: 5 × 64 (b/e/m/partial/next_partial)
        // + RESULT_LENGTH (64) + ITERATION_INDEX_BYTES (4) = 388.
        let legacy = 5 * BIGINT_LENGTH + RESULT_LENGTH + ITERATION_INDEX_BYTES;
        // Task #182 byte decls: 32 R-bytes + 32 Q-bytes + 8 slack-low.
        let v182 = MULMOD_LIMBS * BYTES_PER_LIMB
            + MULMOD_LIMBS * BYTES_PER_LIMB
            + BYTES_PER_LIMB;
        // Task #196 byte decls: 24 slack-hi + 2 × 7 × 9 carry bytes.
        let v196 = (MULMOD_LIMBS - 1) * BYTES_PER_LIMB
            + 2 * NUM_CARRIES * CARRY_BYTES;
        assert_eq!(reqs.declarations.len(), legacy + v182 + v196);
    }

    #[test]
    fn mulmod_air_shifted_constraint_count() {
        let cs = ModExpInternalsConstraintSystem::new(1);
        assert_eq!(cs.num_shifted_constraints(), 2);
        // shifted_column_indices is reused — no new shifted cols needed
        // because the chain re-uses partial_result columns.
        assert_eq!(cs.shifted_column_indices().len(), BIGINT_LENGTH + 2);
    }

    // ─── Task #196 tests ────────────────────────────────────────────

    #[test]
    fn mulmod_air_full_r_lt_m_chain_multi_limb_passes() {
        // Multi-limb modulus, ensure the full chain closes.
        let a = limbs_from_u64(7);
        let b = limbs_from_u64(11);
        let m = [13u64, 1, 0, 0]; // high limb 1 is nonzero
        let w = ModExpInternalsTraceWitness::from_mulmod_step(a, b, m);
        // The witness must set check_r_lt_m_full = true.
        assert!(w.rows[0].mulmod.check_r_lt_m_full);
        // Final borrow must be 0 (sealed by witness builder).
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} nonzero ({})",
                    i,
                    r,
                    cs.constraint_labels()[i]
                );
            }
        }
    }

    #[test]
    fn mulmod_air_tampered_full_slack_hi_detected() {
        // Multi-limb M; tamper a high slack limb. The byte-decomp +
        // limb subtraction must catch this.
        let m = [u64::MAX, 0, 1, 0]; // a real multi-limb modulus
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            m,
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper slack_hi[1] (= the third u64 limb of the chain). The
        // byte decomp will mismatch.
        cols[COL_MM_SLACK_HI_LIMBS + 1][0] =
            cols[COL_MM_SLACK_HI_LIMBS + 1][0]
                .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Some constraint in the v196 block must fire.
        let v196_start = NUM_ROW_CONSTRAINTS_LEGACY
            + NUM_ROW_CONSTRAINTS_MULMOD
            + NUM_ROW_CONSTRAINTS_RANGE;
        let v196_end = v196_start + NUM_ROW_CONSTRAINTS_RANGE_V196;
        let mut any = false;
        for i in v196_start..v196_end {
            if !bodies[i][0].is_zero() {
                any = true;
                break;
            }
        }
        assert!(any, "expected some v196 constraint to fire on tampered slack_hi");
    }

    #[test]
    fn mulmod_air_tampered_carry_byte_detected() {
        // Use inputs that produce a nonzero ab_carry[0] but don't
        // overflow u128 in the column-sum helper. 4-limb A=B with
        // top-limb bits set generates carries through the middle
        // columns; we explicitly choose the smallest test exercising
        // ab_carry[0].
        let a = [u64::MAX, 0, 0, 0];
        let b = [u64::MAX, 0, 0, 0];
        let m = [0xdeadu64, 0xbeef, 0xcafe, 0xbabe];
        let w = ModExpInternalsTraceWitness::from_mulmod_step(a, b, m);
        assert_ne!(w.rows[0].mulmod.ab_carry[0], 0,
            "expected nonzero ab_carry[0] for this input");
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper one byte of ab_carry[0] decomp.
        cols[COL_MM_AB_CARRY_BYTES + 0][0] =
            cols[COL_MM_AB_CARRY_BYTES + 0][0]
                .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // AB carry decomp 0 sits at index:
        //   legacy(9) + mulmod(57) + range_v182(11)
        //   + slack_hi(3) + borrows(3) + sub_limbs(4) + chk_full(1)
        //   = 88
        let idx = NUM_ROW_CONSTRAINTS_LEGACY
            + NUM_ROW_CONSTRAINTS_MULMOD
            + NUM_ROW_CONSTRAINTS_RANGE
            + (MULMOD_LIMBS - 1)   // slack_hi byte decomps
            + (MULMOD_LIMBS - 1)   // borrow binaries
            + MULMOD_LIMBS         // sub limb-eqs
            + 1;                   // chk_full binary
        assert!(
            !bodies[idx][0].is_zero(),
            "expected AB carry[0] byte-decomp constraint to fire",
        );
    }

    #[test]
    fn mulmod_air_tampered_borrow_non_binary_detected() {
        let m = [13u64, 1, 0, 0];
        let w = ModExpInternalsTraceWitness::from_mulmod_step(
            limbs_from_u64(7),
            limbs_from_u64(11),
            m,
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper borrow[0] to 2 (non-binary).
        cols[COL_MM_SLACK_BORROWS + 0][0] =
            Scalar::from_u64(2, CurveType::Bls48581);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Borrow binary[0] sits at index:
        //   legacy(9) + mulmod(57) + range_v182(11) + slack_hi(3)
        //   = 80
        let idx = NUM_ROW_CONSTRAINTS_LEGACY
            + NUM_ROW_CONSTRAINTS_MULMOD
            + NUM_ROW_CONSTRAINTS_RANGE
            + (MULMOD_LIMBS - 1);
        assert!(
            !bodies[idx][0].is_zero(),
            "expected borrow[0] binary constraint to fire on tampered borrow",
        );
    }

    #[test]
    fn modexp_internals_air_column_layout_pinned() {
        assert_eq!(COL_B_BYTES_OFFSET, 0);
        assert_eq!(COL_E_BYTES_OFFSET, 64);
        assert_eq!(COL_M_BYTES_OFFSET, 128);
        assert_eq!(COL_PARTIAL_RESULT_OFFSET, 192);
        assert_eq!(COL_NEXT_PARTIAL_RESULT_OFFSET, 256);
        assert_eq!(COL_RESULT_OFFSET, 320);
        assert_eq!(COL_CURRENT_WINDOW_BITS_OFFSET, 384);
        assert_eq!(COL_ITERATION_INDEX, 388);
        assert_eq!(COL_ITERATION_INDEX_BYTE_OFFSET, 389);
        assert_eq!(COL_IS_RESULT_ROW, 393);
        assert_eq!(COL_IS_REAL, 394);
        // Mul-mod gadget layout pins.
        assert_eq!(COL_MM_A_LIMBS, 395);
        assert_eq!(COL_MM_B_LIMBS, 399);
        assert_eq!(COL_MM_M_LO_LIMBS, 403);
        assert_eq!(COL_MM_Q_LIMBS, 407);
        assert_eq!(COL_MM_R_LIMBS, 411);
        assert_eq!(COL_MM_PP_AB, 415);
        assert_eq!(COL_MM_PP_QM, 431);
        assert_eq!(COL_MM_AB_OUT, 447);
        assert_eq!(COL_MM_AB_CARRY, 455);
        assert_eq!(COL_MM_QM_OUT, 462);
        assert_eq!(COL_MM_QM_CARRY, 470);
        assert_eq!(COL_MM_ID_CARRY_LO, 477);
        assert_eq!(COL_MM_ID_CARRY_HI, 481);
        assert_eq!(COL_IS_MULMOD, 485);
        // Task #182 range / R<M column pins.
        assert_eq!(BYTES_PER_LIMB, 8);
        assert_eq!(COL_MM_R_LIMB_BYTES, 486);
        assert_eq!(COL_MM_Q_LIMB_BYTES, 518);
        assert_eq!(COL_MM_SLACK_LOW_LIMB, 550);
        assert_eq!(COL_MM_SLACK_LOW_BYTES, 551);
        assert_eq!(COL_MM_CHECK_R_LT_M_LOW, 559);
        // Task #196 column pins.
        assert_eq!(COL_MM_SLACK_HI_LIMBS, 560);
        assert_eq!(COL_MM_SLACK_HI_BYTES, 563);
        assert_eq!(COL_MM_SLACK_BORROWS, 587);
        assert_eq!(COL_MM_CHECK_R_LT_M_FULL, 590);
        assert_eq!(COL_MM_AB_CARRY_BYTES, 591);
        assert_eq!(COL_MM_QM_CARRY_BYTES, 591 + NUM_CARRIES * CARRY_BYTES);
        assert_eq!(NUM_CARRIES, 7);
        assert_eq!(CARRY_BYTES, 9);
        assert_eq!(NUM_COLUMNS, 591 + 2 * NUM_CARRIES * CARRY_BYTES);
        assert_eq!(NUM_COLUMNS, 717);
        assert_eq!(NUM_ROW_CONSTRAINTS_LEGACY, 9);
        assert_eq!(NUM_ROW_CONSTRAINTS_MULMOD, 57);
        assert_eq!(NUM_ROW_CONSTRAINTS_RANGE, 11);
        assert_eq!(NUM_ROW_CONSTRAINTS_RANGE_V196, 25);
        assert_eq!(NUM_ROW_CONSTRAINTS, 102);
        assert_eq!(NUM_SHIFTED, 2);
        assert_eq!(WINDOW_BITS, 4);
        assert_eq!(BIGINT_LENGTH, 64);
        assert_eq!(MULMOD_LIMBS, 4);
        assert_eq!(MULMOD_PP, 16);
        assert_eq!(PC_MODEXP, 0x05);

        let cs = ModExpInternalsConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert_eq!(cs.shifted_column_indices().len(), BIGINT_LENGTH + 2);
    }

    #[test]
    #[ignore = "slow: standalone prove+verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let w = ModExpInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = ModExpInternalsConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone modexp_internals_air proof must verify",
        );
    }
}
