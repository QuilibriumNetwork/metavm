//! RIPEMD-160 internals AIR (skeleton).
//!
//! # Purpose
//!
//! Bridges the [`crate::ripemd160_precompile_air`] (which commits a
//! single-row witness `ripemd160(input) = output` taken on faith) to the
//! actual algebraic computation. This module scaffolds the round/iteration
//! structure of RIPEMD-160 so a downstream gadget can later replace the
//! per-round witness columns with algebraic mixing / round-function
//! decomposition.
//!
//! # Algorithm shape (RIPEMD-160)
//!
//! RIPEMD-160 hashes 512-bit message blocks through 80 rounds per pass,
//! split into 5 phases of 16 rounds each, run in two PARALLEL passes
//! (left + right). After all 80 rounds in each pass, the two parallel
//! states are combined into the next chaining value.
//!
//! ```text
//!   for j in 0..80:                       # 80 rounds per pass
//!     T_left  = round_left (h_left,  m[r_l[j]], K_l[j], s_l[j])
//!     T_right = round_right(h_right, m[r_r[j]], K_r[j], s_r[j])
//!     // rotate state words
//!   h_next = combine(h_in, h_left, h_right)
//! ```
//!
//! # AIR shape (this skeleton)
//!
//! One TRACE ROW per round. Each row commits:
//!
//!   * state[5 × 4 bytes = 20 bytes]   — RIPEMD-160 internal state
//!     `(A, B, C, D, E)` as 4 little-endian bytes per 32-bit word.
//!   * round_index ∈ [0, 80]            — 0..80 maps to rounds within
//!     a pass; the boundary row at 80 holds the post-pass state.
//!   * message_word[4]                  — the m[r_l[j]] (or m[r_r[j]])
//!     32-bit word loaded for this round, as 4 LE bytes.
//!   * constant_k[4]                    — the round constant K_l[j]
//!     (or K_r[j]) as 4 LE bytes.
//!   * is_left_pass ∈ {0, 1}            — pass selector (0 = right).
//!   * result[20]                       — finalized digest exposed only on
//!     the FINAL row (`is_result_row = 1`).
//!
//! # Soundness
//!
//! This AIR is a **skeleton**. Algebraic mixing/round-function
//! decomposition is **deferred**. What this AIR DOES bind:
//!
//!   - `is_real`, `is_left_pass`, `is_result_row` binary.
//!   - `result[i]` exposed only when `is_result_row = 1` (gating).
//!   - LE byte decomposition of the 32-bit `round_index_word` column.
//!   - Cross-row state continuity: when `is_real(X) * is_real(ω·X) = 1`
//!     and `is_result_row(X) = 0`, the witness `next_state` column at
//!     row X equals the `state` column at row ω·X (shifted constraint).
//!
//! What this AIR does NOT bind (deferred to a follow-up):
//!
//!   - Algebraic computation `next_state = round(state, m, K, s)`.
//!   - Message schedule selection `m[r_l[j]] = …` (cross-row table).
//!   - Constants table `K_l[j] = …`.
//!   - Two-pass combine `h_next = combine(h_in, h_left, h_right)`.
//!   - Initial state `h_in = (IV_A, IV_B, IV_C, IV_D, IV_E)` pin.
//!
//! # Cross-AIR LogUp descriptors
//!
//! [`make_ripemd160_internals_to_precompile_descriptor`] binds the
//! finalized 20-byte `result` on the final row of this AIR to the
//! `ripemd_output` column of [`crate::ripemd160_precompile_air`]. This is
//! the shape-only link the downstream gadget will tighten once the
//! algebraic round computation lands.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// RIPEMD-160 digest length (bytes).
pub const RESULT_LENGTH: usize = 20;
/// RIPEMD-160 internal state: 5 × 32-bit words = 20 bytes.
pub const STATE_LENGTH: usize = 20;
/// Number of 32-bit message words per loaded slot in a round (one).
pub const MESSAGE_WORD_LENGTH: usize = 4;
/// Number of bytes per 32-bit constant K.
pub const CONSTANT_K_LENGTH: usize = 4;
/// Rounds per pass.
pub const ROUNDS_PER_PASS: usize = 80;
/// `round_index_word` LE byte decomposition width.
pub const ROUND_INDEX_BYTES: usize = 4;

/// EIP precompile id for RIPEMD-160 (mirrors precompile AIR).
pub const PC_RIPEMD160: u64 = 0x03;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_STATE_OFFSET: usize = 0; // 0..20
pub const COL_NEXT_STATE_OFFSET: usize = COL_STATE_OFFSET + STATE_LENGTH; // 20..40
pub const COL_MESSAGE_WORD_OFFSET: usize = COL_NEXT_STATE_OFFSET + STATE_LENGTH; // 40..44
pub const COL_CONSTANT_K_OFFSET: usize = COL_MESSAGE_WORD_OFFSET + MESSAGE_WORD_LENGTH; // 44..48
pub const COL_RESULT_OFFSET: usize = COL_CONSTANT_K_OFFSET + CONSTANT_K_LENGTH; // 48..68
pub const COL_ROUND_INDEX: usize = COL_RESULT_OFFSET + RESULT_LENGTH; // 68
pub const COL_ROUND_INDEX_BYTE_OFFSET: usize = COL_ROUND_INDEX + 1; // 69..73
pub const COL_IS_LEFT_PASS: usize = COL_ROUND_INDEX_BYTE_OFFSET + ROUND_INDEX_BYTES; // 73
pub const COL_IS_RESULT_ROW: usize = COL_IS_LEFT_PASS + 1; // 74
pub const COL_IS_REAL: usize = COL_IS_RESULT_ROW + 1; // 75

// ─── Algebraic round-function decomposition columns ────────────────────
//
// 32-bit RIPEMD-160 words are decomposed into 32 little-endian bits so
// the per-round nonlinear function f_j(B, C, D) can be enforced
// algebraically at the bit level. Below we lay out 4 bytes of f_result
// + 32 bits each for B, C, D, f_result, plus 5 round-group selectors.
//
// The 5 round-group selectors `is_round_group[0..5]` are one-hot under
// `is_real`. Round group 0 (rounds 0..15) uses f_0(x,y,z) = x ⊕ y ⊕ z;
// only that group is currently constrained algebraically. The remaining
// 4 groups (f_1..f_4) are deferred — see follow-up notes below.
pub const STATE_WORD_BYTES: usize = 4;
pub const STATE_WORD_BITS: usize = 32;
pub const NUM_ROUND_GROUPS: usize = 5;

/// Offset into the 20-byte state of words A, B, C, D, E (little-endian).
pub const COL_STATE_A_BYTE_OFFSET: usize = COL_STATE_OFFSET; // 0..4
pub const COL_STATE_B_BYTE_OFFSET: usize = COL_STATE_OFFSET + 4; // 4..8
pub const COL_STATE_C_BYTE_OFFSET: usize = COL_STATE_OFFSET + 8; // 8..12
pub const COL_STATE_D_BYTE_OFFSET: usize = COL_STATE_OFFSET + 12; // 12..16
pub const COL_STATE_E_BYTE_OFFSET: usize = COL_STATE_OFFSET + 16; // 16..20

/// 4 bytes of `f_result = f_j(B, C, D)`, LE byte order.
pub const COL_F_RESULT_OFFSET: usize = COL_IS_REAL + 1; // 76..80
/// 32 bits of B (LE).
pub const COL_B_BITS_OFFSET: usize = COL_F_RESULT_OFFSET + STATE_WORD_BYTES; // 80..112
/// 32 bits of C (LE).
pub const COL_C_BITS_OFFSET: usize = COL_B_BITS_OFFSET + STATE_WORD_BITS; // 112..144
/// 32 bits of D (LE).
pub const COL_D_BITS_OFFSET: usize = COL_C_BITS_OFFSET + STATE_WORD_BITS; // 144..176
/// 32 bits of `f_result` (LE).
pub const COL_F_BITS_OFFSET: usize = COL_D_BITS_OFFSET + STATE_WORD_BITS; // 176..208
/// One-hot selector per round group (j ∈ 0..5).
pub const COL_IS_ROUND_GROUP_OFFSET: usize = COL_F_BITS_OFFSET + STATE_WORD_BITS; // 208..213

// ─── ROL_s amount + T output column (Task #180) ──────────────────────
//
// `rol_amount` commits the rotation amount s_j for this round. We pin
// it per-round-group via a selector-driven lookup (one "common"
// rotation per group; the full per-round s_j table is deferred).
//
// `t_word` is the round-function output `T = ROL_s(A + f + W + K) + E`.
// We commit it as 4 LE bytes + 32 LE bits with the byte↔bit decomp
// constraint. The algebraic binding `T = ROL_s(...)+E` itself is
// deferred (involves modular u32 add + variable rotation).
pub const COL_ROL_AMOUNT: usize = COL_IS_ROUND_GROUP_OFFSET + NUM_ROUND_GROUPS; // 213
pub const COL_T_WORD_BYTE_OFFSET: usize = COL_ROL_AMOUNT + 1; // 214..218
pub const COL_T_WORD_BITS_OFFSET: usize = COL_T_WORD_BYTE_OFFSET + STATE_WORD_BYTES; // 218..250

// ─── T = ROL_s(A + f + W + K) + E computation columns (Task #195) ─────
//
// We commit a bit decomposition of the low-32 of S1 := A + f + W + K
// and a 2-bit carry, then a per-bit rotated word ROL_s(low-32(S1)),
// then a 1-bit carry of the rotated + E addition that produces t_word.
pub const COL_INTERMEDIATE_SUM_BITS_OFFSET: usize = COL_T_WORD_BITS_OFFSET + STATE_WORD_BITS; // 250..282
pub const COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET: usize =
    COL_INTERMEDIATE_SUM_BITS_OFFSET + STATE_WORD_BITS; // 282..284 (2 bits, value ∈ {0,1,2,3})
pub const INTERMEDIATE_SUM_CARRY_BITS: usize = 2;
pub const COL_ROTATED_WORD_BITS_OFFSET: usize =
    COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + INTERMEDIATE_SUM_CARRY_BITS; // 284..316
pub const COL_T_ADD_CARRY_BIT: usize =
    COL_ROTATED_WORD_BITS_OFFSET + STATE_WORD_BITS; // 316 (1 bit)

/// Task #318 / #309 / #190 mirror column: per-row byte anchor for
/// cross-AIR LogUp descriptors that want to bind an arbitrary host-derived
/// byte to this AIR's row without requiring B-side overrides on
/// `MESSAGE_WORD[0]`. Populated by the trace builder; defaults to
/// `row.message_word[0]` (back-compat) but the witness exposes
/// [`Ripemd160InternalsTraceWitness::mirror_byte0`] for per-row overrides.
/// No row-local constraint binds this column — soundness flows from the
/// cross-AIR LogUp closure.
pub const COL_MIRROR_BYTE0: usize = COL_T_ADD_CARRY_BIT + 1; // 317

pub const NUM_COLUMNS: usize = COL_MIRROR_BYTE0 + 1; // 318

/// K_L per round group (left branch, 4 bytes LE).
pub const K_L: [u32; NUM_ROUND_GROUPS] = [
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E,
];
/// K_R per round group (right branch, 4 bytes LE).
pub const K_R: [u32; NUM_ROUND_GROUPS] = [
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000,
];

/// Representative ROL_s rotation amount per round group (left branch).
///
/// **Time-box note (Task #180)**: the full per-round s_j table (16
/// values × 5 groups × 2 passes = 160 selectors) is deferred. We pin
/// ONE representative rotation per round group via the
/// `is_round_group` selector. Honest witnesses for arbitrary rounds
/// would need a wider per-round-index selector; for the skeleton
/// validating one round per group at the representative shift this is
/// sufficient. The deferred binding is the `(round_index_in_group ×
/// branch)` → `rol_amount` table.
///
/// Picked from RIPEMD-160 spec left-branch rotations: most common /
/// modal value per group.
pub const ROL_REPRESENTATIVE_L: [u8; NUM_ROUND_GROUPS] = [11, 7, 11, 11, 11];
/// Representative ROL_s rotation amount per round group (right branch).
pub const ROL_REPRESENTATIVE_R: [u8; NUM_ROUND_GROUPS] = [8, 9, 9, 15, 8];

/// Row-local constraints:
///   0:                 is_real ∈ {0, 1}
///   1:                 is_left_pass ∈ {0, 1}
///   2:                 is_result_row ∈ {0, 1}
///   3:                 is_result_row ≤ is_real
///   4:                 LE byte-decomp of round_index_word
///   5:                 (1 - is_result_row) * Σ result[i] = 0
///                      (result must be zero on non-final rows; gated)
///   6:                 Σ is_round_group[j] = is_real
///   7..12:             is_round_group[j] ∈ {0,1} (NUM_ROUND_GROUPS = 5)
///   12..16:            word→bit decomp for B, C, D, f_result (4)
///   16..48:            per-bit XOR3 (`f_0(B,C,D) = B⊕C⊕D`) gated by
///                      is_round_group[0] (32)
///   48..52:            constant_k[b] = is_left_pass*Σ is_round_group[j]*K_L_b[j]
///                              + (1-is_left_pass)*Σ is_round_group[j]*K_R_b[j]
///                      (4 bytes)
///   52..84:            per-bit f_1((B AND C) OR (NOT B AND D)) gated by
///                      is_round_group[1] (32)
///   84..116:           per-bit f_2((B OR NOT C) XOR D) gated by
///                      is_round_group[2] (32)
///   116..148:          per-bit f_3((B AND D) OR (C AND NOT D)) gated by
///                      is_round_group[3] (32)
///   148..180:          per-bit f_4(B XOR (C OR NOT D)) gated by
///                      is_round_group[4] (32)
///   180:               word→bit decomp for t_word
///   181:               rol_amount = is_left_pass * Σ is_round_group[j] *
///                      ROL_REPRESENTATIVE_L[j] +
///                      (1-is_left_pass) * Σ is_round_group[j] *
///                      ROL_REPRESENTATIVE_R[j]
///   182..186:          state-cycling A_next = state.E (4 bytes), gated
///                      by is_real * (1 - is_result_row)
///   186..190:          state-cycling B_next = t_word (4 bytes), gated
///   190..194:          state-cycling C_next = state.B (4 bytes), gated
///   194..198:          state-cycling D_next = state.C (4 bytes), gated.
///                      **NOTE**: spec says D_next = ROL_10(C); the
///                      ROL_10 bit-rotation is deferred — this binds
///                      the byte-level shape only (D_next byte k =
///                      state.C byte k) which is correct only when
///                      the held witness pre-rotates C honestly.
///                      Deferred: algebraic ROL_10 binding.
///   198..202:          state-cycling E_next = state.D (4 bytes), gated
///   202:               intermediate_sum bit-decomp
///                      Σ 2^i * intermediate_sum_bits[i] + 2^32 * carry_val
///                          = A_val + f_val + W_val + K_val
///                      (carry_val = carry_bits[0] + 2*carry_bits[1]).
///   203..205:          intermediate_sum carry_bits[0..2] ∈ {0,1}
///   205..237:          per-bit variable-ROL_s binding (32):
///                      rotated_word_bits[i] = Σ_{j,p}
///                          is_round_group[j] * pass_sel[p] *
///                          intermediate_sum_bits[(i - ROL[p][j]) mod 32]
///                      (pass_sel[L]=is_left_pass, pass_sel[R]=1-is_left_pass).
///                      ROL_REPRESENTATIVE_{L,R}[j] are the per-group
///                      representative rotation amounts; the full
///                      per-round s_j table (160 entries) remains
///                      deferred — see ROL_REPRESENTATIVE doc.
///   237:               T add carry: Σ 2^i * t_word_bits[i]
///                          + t_add_carry * 2^32 =
///                          Σ 2^i * rotated_word_bits[i] + E_val
///   238:               t_add_carry ∈ {0,1}
pub const NUM_ROW_CONSTRAINTS: usize = 239;

/// Shifted constraints:
///   0:                 state-continuity:
///                      is_real(X) * is_real(ω·X) * (1 - is_result_row(X)) *
///                      Σ_i β^i * (state(ω·X)[i] - next_state(X)[i]) = 0
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

/// One row of the RIPEMD-160 internals trace (one round).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ripemd160InternalsRow {
    /// Internal state at the start of this round (LE byte layout of
    /// (A,B,C,D,E) as 5 × 4 bytes).
    pub state: [u8; STATE_LENGTH],
    /// Internal state at the end of this round.
    pub next_state: [u8; STATE_LENGTH],
    /// Message word loaded this round (LE 4 bytes).
    pub message_word: [u8; MESSAGE_WORD_LENGTH],
    /// Round constant K (LE 4 bytes).
    pub constant_k: [u8; CONSTANT_K_LENGTH],
    /// 20-byte result digest (zero except on the final result row).
    pub result: [u8; RESULT_LENGTH],
    /// Round index (0..=80).
    pub round_index: u32,
    /// True if this is a left-pass round.
    pub is_left_pass: bool,
    /// True if this row is the FINAL row exposing the digest.
    pub is_result_row: bool,
    /// 4 LE bytes of f_j(B, C, D) for this round.
    pub f_result: [u8; STATE_WORD_BYTES],
    /// Round group index j ∈ 0..NUM_ROUND_GROUPS for this row's
    /// one-hot `is_round_group` selector.
    pub round_group: u8,
    /// 4 LE bytes of T = ROL_s(A + f + W + K) + E for this round
    /// (round-function output).
    pub t_word: [u8; STATE_WORD_BYTES],
    /// Rotation amount s for this round (pinned per-round-group to
    /// `ROL_REPRESENTATIVE_{L,R}[round_group]`).
    pub rol_amount: u8,
    /// 32-bit low part of A + f + W + K (pre-rotation summand).
    pub intermediate_sum_lo: u32,
    /// 2-bit carry from A + f + W + K (∈ {0,1,2,3}).
    pub intermediate_sum_carry: u8,
    /// 32-bit ROL_s(intermediate_sum_lo) (pre-E addition).
    pub rotated_word: u32,
    /// 1-bit carry from rotated_word + E (∈ {0,1}).
    pub t_add_carry: u8,
}

impl Default for Ripemd160InternalsRow {
    fn default() -> Self {
        Self {
            state: [0u8; STATE_LENGTH],
            next_state: [0u8; STATE_LENGTH],
            message_word: [0u8; MESSAGE_WORD_LENGTH],
            constant_k: [0u8; CONSTANT_K_LENGTH],
            result: [0u8; RESULT_LENGTH],
            round_index: 0,
            is_left_pass: true,
            is_result_row: false,
            f_result: [0u8; STATE_WORD_BYTES],
            round_group: 0,
            t_word: [0u8; STATE_WORD_BYTES],
            rol_amount: 0,
            intermediate_sum_lo: 0,
            intermediate_sum_carry: 0,
            rotated_word: 0,
            t_add_carry: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Ripemd160InternalsTraceWitness {
    pub rows: Vec<Ripemd160InternalsRow>,
    /// Task #318 / #309 mirror data: per-row byte value populated into
    /// [`COL_MIRROR_BYTE0`]. Empty `Vec` means "default to
    /// `rows[r].message_word[0]`" (back-compat). Otherwise must match
    /// `rows.len()` and is written verbatim per-row.
    #[doc(hidden)]
    pub mirror_byte0: Vec<u8>,
}

impl Ripemd160InternalsTraceWitness {
    pub fn from_rows(rows: Vec<Ripemd160InternalsRow>) -> Self {
        Self {
            rows,
            mirror_byte0: Vec::new(),
        }
    }

    pub fn push(&mut self, row: Ripemd160InternalsRow) {
        self.rows.push(row);
    }

    /// Builder: override `COL_MIRROR_BYTE0` values per-row. Used by
    /// integration tests that wire a cross-AIR LogUp descriptor against
    /// this AIR's mirror column without touching `MESSAGE_WORD`.
    pub fn with_mirror_byte0(mut self, mirror: Vec<u8>) -> Self {
        assert_eq!(
            mirror.len(),
            self.rows.len(),
            "mirror_byte0 length must equal rows.len()",
        );
        self.mirror_byte0 = mirror;
        self
    }

    /// Build a SKELETON witness from a finalized RIPEMD-160 digest.
    ///
    /// The intermediate state columns are populated as zeros (deferred);
    /// only the FINAL row carries the algebraically meaningful
    /// `result` field. This matches the soundness scope of the
    /// skeleton (downstream gadgets will replace the zero rows with
    /// algebraically computed round states).
    pub fn from_digest_skeleton(input: &[u8]) -> Self {
        use ripemd::{Digest, Ripemd160};
        let mut hasher = Ripemd160::new();
        hasher.update(input);
        let digest = hasher.finalize();
        let mut result = [0u8; RESULT_LENGTH];
        result.copy_from_slice(&digest);

        // 1 dummy round row (round 0) + 1 final result row, to exercise
        // the round/result-row column shape AND the cross-row state
        // continuity constraint. The skeleton's algebraic check across
        // these rows is `state(ω·X) == next_state(X)` (gated by
        // is_real * is_real_next * (1 - is_result_row(X))) — we set
        // both equal to zero, which is a trivially satisfied honest
        // witness for the deferred mixing.
        let mut rows = Vec::with_capacity(2);
        // Both rows are honest under f_0 (XOR3) with all-zero B,C,D —
        // f_result = 0 ⊕ 0 ⊕ 0 = 0. Left-pass with K_L[0] = 0 also
        // satisfies the constant_k binding. round_group = 0.
        rows.push(Ripemd160InternalsRow {
            state: [0u8; STATE_LENGTH],
            next_state: [0u8; STATE_LENGTH],
            message_word: [0u8; MESSAGE_WORD_LENGTH],
            constant_k: [0u8; CONSTANT_K_LENGTH],
            result: [0u8; RESULT_LENGTH],
            round_index: 0,
            is_left_pass: true,
            is_result_row: false,
            f_result: [0u8; STATE_WORD_BYTES],
            round_group: 0,
            t_word: [0u8; STATE_WORD_BYTES],
            rol_amount: ROL_REPRESENTATIVE_L[0],
            intermediate_sum_lo: 0,
            intermediate_sum_carry: 0,
            rotated_word: 0,
            t_add_carry: 0,
        });
        rows.push(Ripemd160InternalsRow {
            state: [0u8; STATE_LENGTH],
            next_state: [0u8; STATE_LENGTH],
            message_word: [0u8; MESSAGE_WORD_LENGTH],
            constant_k: [0u8; CONSTANT_K_LENGTH],
            result,
            round_index: ROUNDS_PER_PASS as u32,
            is_left_pass: true,
            is_result_row: true,
            f_result: [0u8; STATE_WORD_BYTES],
            round_group: 0,
            t_word: [0u8; STATE_WORD_BYTES],
            rol_amount: ROL_REPRESENTATIVE_L[0],
            intermediate_sum_lo: 0,
            intermediate_sum_carry: 0,
            rotated_word: 0,
            t_add_carry: 0,
        });
        Self {
            rows,
            mirror_byte0: Vec::new(),
        }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Ripemd160InternalsTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..STATE_LENGTH {
            columns[COL_STATE_OFFSET + k][r] =
                Scalar::from_u64(row.state[k] as u64, curve);
            columns[COL_NEXT_STATE_OFFSET + k][r] =
                Scalar::from_u64(row.next_state[k] as u64, curve);
        }
        for k in 0..MESSAGE_WORD_LENGTH {
            columns[COL_MESSAGE_WORD_OFFSET + k][r] =
                Scalar::from_u64(row.message_word[k] as u64, curve);
        }
        for k in 0..CONSTANT_K_LENGTH {
            columns[COL_CONSTANT_K_OFFSET + k][r] =
                Scalar::from_u64(row.constant_k[k] as u64, curve);
        }
        for k in 0..RESULT_LENGTH {
            columns[COL_RESULT_OFFSET + k][r] =
                Scalar::from_u64(row.result[k] as u64, curve);
        }
        columns[COL_ROUND_INDEX][r] =
            Scalar::from_u64(row.round_index as u64, curve);
        let ri_le = row.round_index.to_le_bytes();
        for b in 0..ROUND_INDEX_BYTES {
            columns[COL_ROUND_INDEX_BYTE_OFFSET + b][r] =
                Scalar::from_u64(ri_le[b] as u64, curve);
        }
        columns[COL_IS_LEFT_PASS][r] = if row.is_left_pass {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_RESULT_ROW][r] = if row.is_result_row {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_REAL][r] = one.clone();

        // f_result (4 LE bytes).
        for k in 0..STATE_WORD_BYTES {
            columns[COL_F_RESULT_OFFSET + k][r] =
                Scalar::from_u64(row.f_result[k] as u64, curve);
        }

        // 32-bit LE bit-decomposition of B, C, D, f_result. The
        // 4-byte LE word at byte offset O is bits [O,O+1,O+2,O+3] in
        // order from LSB to MSB.
        let b_word = u32::from_le_bytes([
            row.state[COL_STATE_B_BYTE_OFFSET + 0 - COL_STATE_OFFSET],
            row.state[COL_STATE_B_BYTE_OFFSET + 1 - COL_STATE_OFFSET],
            row.state[COL_STATE_B_BYTE_OFFSET + 2 - COL_STATE_OFFSET],
            row.state[COL_STATE_B_BYTE_OFFSET + 3 - COL_STATE_OFFSET],
        ]);
        let c_word = u32::from_le_bytes([
            row.state[COL_STATE_C_BYTE_OFFSET + 0 - COL_STATE_OFFSET],
            row.state[COL_STATE_C_BYTE_OFFSET + 1 - COL_STATE_OFFSET],
            row.state[COL_STATE_C_BYTE_OFFSET + 2 - COL_STATE_OFFSET],
            row.state[COL_STATE_C_BYTE_OFFSET + 3 - COL_STATE_OFFSET],
        ]);
        let d_word = u32::from_le_bytes([
            row.state[COL_STATE_D_BYTE_OFFSET + 0 - COL_STATE_OFFSET],
            row.state[COL_STATE_D_BYTE_OFFSET + 1 - COL_STATE_OFFSET],
            row.state[COL_STATE_D_BYTE_OFFSET + 2 - COL_STATE_OFFSET],
            row.state[COL_STATE_D_BYTE_OFFSET + 3 - COL_STATE_OFFSET],
        ]);
        let f_word = u32::from_le_bytes(row.f_result);
        for i in 0..STATE_WORD_BITS {
            columns[COL_B_BITS_OFFSET + i][r] =
                Scalar::from_u64(((b_word >> i) & 1) as u64, curve);
            columns[COL_C_BITS_OFFSET + i][r] =
                Scalar::from_u64(((c_word >> i) & 1) as u64, curve);
            columns[COL_D_BITS_OFFSET + i][r] =
                Scalar::from_u64(((d_word >> i) & 1) as u64, curve);
            columns[COL_F_BITS_OFFSET + i][r] =
                Scalar::from_u64(((f_word >> i) & 1) as u64, curve);
        }

        // One-hot is_round_group selector.
        let g = (row.round_group as usize).min(NUM_ROUND_GROUPS - 1);
        for j in 0..NUM_ROUND_GROUPS {
            columns[COL_IS_ROUND_GROUP_OFFSET + j][r] = if j == g {
                one.clone()
            } else {
                zero.clone()
            };
        }

        // rol_amount + t_word + t_word bits (Task #180).
        columns[COL_ROL_AMOUNT][r] =
            Scalar::from_u64(row.rol_amount as u64, curve);
        for k in 0..STATE_WORD_BYTES {
            columns[COL_T_WORD_BYTE_OFFSET + k][r] =
                Scalar::from_u64(row.t_word[k] as u64, curve);
        }
        let t_word_u32 = u32::from_le_bytes(row.t_word);
        for i in 0..STATE_WORD_BITS {
            columns[COL_T_WORD_BITS_OFFSET + i][r] =
                Scalar::from_u64(((t_word_u32 >> i) & 1) as u64, curve);
        }

        // Task #195: T = ROL_s(A + f + W + K) + E intermediate witnesses.
        for i in 0..STATE_WORD_BITS {
            columns[COL_INTERMEDIATE_SUM_BITS_OFFSET + i][r] =
                Scalar::from_u64(((row.intermediate_sum_lo >> i) & 1) as u64, curve);
        }
        for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
            columns[COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b][r] =
                Scalar::from_u64(((row.intermediate_sum_carry >> b) & 1) as u64, curve);
        }
        for i in 0..STATE_WORD_BITS {
            columns[COL_ROTATED_WORD_BITS_OFFSET + i][r] =
                Scalar::from_u64(((row.rotated_word >> i) & 1) as u64, curve);
        }
        columns[COL_T_ADD_CARRY_BIT][r] =
            Scalar::from_u64(row.t_add_carry as u64, curve);

        // Task #318 mirror: prefer explicit per-row override; fall back
        // to `message_word[0]` when the caller didn't populate
        // `mirror_byte0`. No row-local constraint binds this column;
        // soundness is via cross-AIR LogUp closure.
        let mirror_byte = if witness.mirror_byte0.is_empty() {
            row.message_word[0]
        } else {
            witness.mirror_byte0[r]
        };
        columns[COL_MIRROR_BYTE0][r] = Scalar::from_u64(mirror_byte as u64, curve);
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

pub struct Ripemd160InternalsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Ripemd160InternalsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Ripemd160InternalsConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "is_real_binary".to_string(),
            "is_left_pass_binary".to_string(),
            "is_result_row_binary".to_string(),
            "is_result_row_le_is_real".to_string(),
            "round_index_le_byte_decomp".to_string(),
            "result_zero_off_final_row".to_string(),
            "round_group_sum_eq_is_real".to_string(),
        ];
        for j in 0..NUM_ROUND_GROUPS {
            labels.push(format!("is_round_group_{}_binary", j));
        }
        labels.push("b_word_bit_decomp".to_string());
        labels.push("c_word_bit_decomp".to_string());
        labels.push("d_word_bit_decomp".to_string());
        labels.push("f_word_bit_decomp".to_string());
        for i in 0..STATE_WORD_BITS {
            labels.push(format!("f0_xor3_bit_{}", i));
        }
        for b in 0..CONSTANT_K_LENGTH {
            labels.push(format!("constant_k_byte_{}_round_group_pin", b));
        }
        for i in 0..STATE_WORD_BITS {
            labels.push(format!("f1_bit_{}", i));
        }
        for i in 0..STATE_WORD_BITS {
            labels.push(format!("f2_bit_{}", i));
        }
        for i in 0..STATE_WORD_BITS {
            labels.push(format!("f3_bit_{}", i));
        }
        for i in 0..STATE_WORD_BITS {
            labels.push(format!("f4_bit_{}", i));
        }
        labels.push("t_word_bit_decomp".to_string());
        labels.push("rol_amount_round_group_pin".to_string());
        for b in 0..STATE_WORD_BYTES {
            labels.push(format!("state_cycling_a_next_byte_{}", b));
        }
        for b in 0..STATE_WORD_BYTES {
            labels.push(format!("state_cycling_b_next_byte_{}", b));
        }
        for b in 0..STATE_WORD_BYTES {
            labels.push(format!("state_cycling_c_next_byte_{}", b));
        }
        for b in 0..STATE_WORD_BYTES {
            labels.push(format!("state_cycling_d_next_byte_{}", b));
        }
        for b in 0..STATE_WORD_BYTES {
            labels.push(format!("state_cycling_e_next_byte_{}", b));
        }
        labels.push("intermediate_sum_bit_decomp".to_string());
        for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
            labels.push(format!("intermediate_sum_carry_bit_{}_binary", b));
        }
        for i in 0..STATE_WORD_BITS {
            labels.push(format!("rotated_word_bit_{}_variable_rol_s", i));
        }
        labels.push("t_add_carry_word_relation".to_string());
        labels.push("t_add_carry_binary".to_string());
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

        // 1: is_left_pass binary.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_LEFT_PASS][r];
            c[r] = v.mul(&v.sub(&one));
        }
        out.push(c);

        // 2: is_result_row binary.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_RESULT_ROW][r];
            c[r] = v.mul(&v.sub(&one));
        }
        out.push(c);

        // 3: is_result_row ≤ is_real  ⇔  is_result_row * (1 - is_real) = 0.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let rr = &columns[COL_IS_RESULT_ROW][r];
            let real = &columns[COL_IS_REAL][r];
            c[r] = rr.mul(&one.sub(real));
        }
        out.push(c);

        // 4: LE byte-decomp of round_index_word.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let mut sum = Scalar::zero(curve);
            for b in 0..ROUND_INDEX_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                let term = columns[COL_ROUND_INDEX_BYTE_OFFSET + b][r].mul(&w);
                sum = sum.add(&term);
            }
            c[r] = sum.sub(&columns[COL_ROUND_INDEX][r]);
        }
        out.push(c);

        // 5: (1 - is_result_row) * Σ result[i] = 0.
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

        // 6: Σ is_round_group[j] = is_real.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let mut sum = Scalar::zero(curve);
            for j in 0..NUM_ROUND_GROUPS {
                sum = sum.add(&columns[COL_IS_ROUND_GROUP_OFFSET + j][r]);
            }
            c[r] = sum.sub(&columns[COL_IS_REAL][r]);
        }
        out.push(c);

        // 7..12: is_round_group[j] ∈ {0,1}.
        for j in 0..NUM_ROUND_GROUPS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_ROUND_GROUP_OFFSET + j][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 12..16: word→bit decomp for B, C, D, f_result.
        //   For each, Σ 2^i * bit_i = byte0 + byte1*2^8 + byte2*2^16 + byte3*2^24.
        let make_word_decomp = |bit_offset: usize,
                                 byte_offset: usize,
                                 columns: &[&Vec<Scalar>]|
         -> Vec<Scalar> {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut bit_sum = Scalar::zero(curve);
                for i in 0..STATE_WORD_BITS {
                    let w = Scalar::from_u64(1u64 << i, curve);
                    bit_sum = bit_sum.add(&columns[bit_offset + i][r].mul(&w));
                }
                let mut byte_sum = Scalar::zero(curve);
                for b in 0..STATE_WORD_BYTES {
                    let w = Scalar::from_u64(1u64 << (8 * b), curve);
                    byte_sum = byte_sum.add(&columns[byte_offset + b][r].mul(&w));
                }
                c[r] = bit_sum.sub(&byte_sum);
            }
            c
        };
        out.push(make_word_decomp(
            COL_B_BITS_OFFSET, COL_STATE_B_BYTE_OFFSET, columns,
        ));
        out.push(make_word_decomp(
            COL_C_BITS_OFFSET, COL_STATE_C_BYTE_OFFSET, columns,
        ));
        out.push(make_word_decomp(
            COL_D_BITS_OFFSET, COL_STATE_D_BYTE_OFFSET, columns,
        ));
        out.push(make_word_decomp(
            COL_F_BITS_OFFSET, COL_F_RESULT_OFFSET, columns,
        ));

        // 16..48: per-bit XOR3 (f_0(B,C,D) = B⊕C⊕D), gated by
        //   is_round_group[0]. Algebraic identity over {0,1}:
        //     x⊕y⊕z = x + y + z - 2(xy + xz + yz) + 4xyz.
        let two = Scalar::from_u64(2, curve);
        let four = Scalar::from_u64(4, curve);
        for i in 0..STATE_WORD_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let x = &columns[COL_B_BITS_OFFSET + i][r];
                let y = &columns[COL_C_BITS_OFFSET + i][r];
                let z = &columns[COL_D_BITS_OFFSET + i][r];
                let f = &columns[COL_F_BITS_OFFSET + i][r];
                let xy = x.mul(y);
                let xz = x.mul(z);
                let yz = y.mul(z);
                let xyz = xy.mul(z);
                let mut expected = x.add(y).add(z);
                let pairs = xy.add(&xz).add(&yz);
                expected = expected.sub(&two.mul(&pairs));
                expected = expected.add(&four.mul(&xyz));
                let gate = &columns[COL_IS_ROUND_GROUP_OFFSET + 0][r];
                c[r] = gate.mul(&f.sub(&expected));
            }
            out.push(c);
        }

        // 48..52: constant_k[b] pin.
        //   K_b = is_left_pass * Σ is_round_group[j] * K_L[j][b]
        //       + (1 - is_left_pass) * Σ is_round_group[j] * K_R[j][b]
        for b in 0..CONSTANT_K_LENGTH {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum_l = Scalar::zero(curve);
                let mut sum_r = Scalar::zero(curve);
                for j in 0..NUM_ROUND_GROUPS {
                    let kl_byte = ((K_L[j] >> (8 * b)) & 0xff) as u64;
                    let kr_byte = ((K_R[j] >> (8 * b)) & 0xff) as u64;
                    let sel = &columns[COL_IS_ROUND_GROUP_OFFSET + j][r];
                    sum_l = sum_l.add(&sel.mul(&Scalar::from_u64(kl_byte, curve)));
                    sum_r = sum_r.add(&sel.mul(&Scalar::from_u64(kr_byte, curve)));
                }
                let lp = &columns[COL_IS_LEFT_PASS][r];
                let expected =
                    lp.mul(&sum_l).add(&one.sub(lp).mul(&sum_r));
                c[r] = columns[COL_CONSTANT_K_OFFSET + b][r].sub(&expected);
            }
            out.push(c);
        }

        // 52..84: per-bit f_1((B AND C) OR (NOT B AND D)) gated by
        //   is_round_group[1]. Over {0,1} bits:
        //     f_1 = xy + (1-x)z = xy + z - xz
        //   (xy and (1-x)z are disjoint, so OR = sum).
        for i in 0..STATE_WORD_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let x = &columns[COL_B_BITS_OFFSET + i][r];
                let y = &columns[COL_C_BITS_OFFSET + i][r];
                let z = &columns[COL_D_BITS_OFFSET + i][r];
                let f = &columns[COL_F_BITS_OFFSET + i][r];
                let xy = x.mul(y);
                let xz = x.mul(z);
                let expected = xy.add(z).sub(&xz);
                let gate = &columns[COL_IS_ROUND_GROUP_OFFSET + 1][r];
                c[r] = gate.mul(&f.sub(&expected));
            }
            out.push(c);
        }

        // 84..116: per-bit f_2((B OR NOT C) XOR D) gated by
        //   is_round_group[2]. Over {0,1} bits:
        //     g := x OR (1-y) = x + (1-y) - x(1-y) = 1 - y + xy
        //     f_2 = g XOR z = g + z - 2gz
        for i in 0..STATE_WORD_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let x = &columns[COL_B_BITS_OFFSET + i][r];
                let y = &columns[COL_C_BITS_OFFSET + i][r];
                let z = &columns[COL_D_BITS_OFFSET + i][r];
                let f = &columns[COL_F_BITS_OFFSET + i][r];
                // g = 1 - y + x*y
                let xy = x.mul(y);
                let g = one.sub(y).add(&xy);
                let gz = g.mul(z);
                let two_gz = Scalar::from_u64(2, curve).mul(&gz);
                let expected = g.add(z).sub(&two_gz);
                let gate = &columns[COL_IS_ROUND_GROUP_OFFSET + 2][r];
                c[r] = gate.mul(&f.sub(&expected));
            }
            out.push(c);
        }

        // 116..148: per-bit f_3((B AND D) OR (C AND NOT D)) gated by
        //   is_round_group[3]. xz and y(1-z) are disjoint
        //   (when z=0 first term is 0; when z=1 second is 0).
        //     f_3 = xz + y(1-z) = xz + y - yz
        for i in 0..STATE_WORD_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let x = &columns[COL_B_BITS_OFFSET + i][r];
                let y = &columns[COL_C_BITS_OFFSET + i][r];
                let z = &columns[COL_D_BITS_OFFSET + i][r];
                let f = &columns[COL_F_BITS_OFFSET + i][r];
                let xz = x.mul(z);
                let yz = y.mul(z);
                let expected = xz.add(y).sub(&yz);
                let gate = &columns[COL_IS_ROUND_GROUP_OFFSET + 3][r];
                c[r] = gate.mul(&f.sub(&expected));
            }
            out.push(c);
        }

        // 148..180: per-bit f_4(B XOR (C OR NOT D)) gated by
        //   is_round_group[4]. Over {0,1}:
        //     h := y OR (1-z) = y + (1-z) - y(1-z) = 1 - z + yz
        //     f_4 = x XOR h = x + h - 2xh
        for i in 0..STATE_WORD_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let x = &columns[COL_B_BITS_OFFSET + i][r];
                let y = &columns[COL_C_BITS_OFFSET + i][r];
                let z = &columns[COL_D_BITS_OFFSET + i][r];
                let f = &columns[COL_F_BITS_OFFSET + i][r];
                let yz = y.mul(z);
                let h = one.sub(z).add(&yz);
                let xh = x.mul(&h);
                let two_xh = Scalar::from_u64(2, curve).mul(&xh);
                let expected = x.add(&h).sub(&two_xh);
                let gate = &columns[COL_IS_ROUND_GROUP_OFFSET + 4][r];
                c[r] = gate.mul(&f.sub(&expected));
            }
            out.push(c);
        }

        // 180: word→bit decomp for t_word.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut bit_sum = Scalar::zero(curve);
                for i in 0..STATE_WORD_BITS {
                    let w = Scalar::from_u64(1u64 << i, curve);
                    bit_sum = bit_sum.add(
                        &columns[COL_T_WORD_BITS_OFFSET + i][r].mul(&w),
                    );
                }
                let mut byte_sum = Scalar::zero(curve);
                for b in 0..STATE_WORD_BYTES {
                    let w = Scalar::from_u64(1u64 << (8 * b), curve);
                    byte_sum = byte_sum.add(
                        &columns[COL_T_WORD_BYTE_OFFSET + b][r].mul(&w),
                    );
                }
                c[r] = bit_sum.sub(&byte_sum);
            }
            out.push(c);
        }

        // 181: rol_amount per-round-group pin.
        //   rol_amount = is_left_pass * Σ is_round_group[j] *
        //       ROL_REPRESENTATIVE_L[j] +
        //     (1 - is_left_pass) * Σ is_round_group[j] *
        //       ROL_REPRESENTATIVE_R[j]
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum_l = Scalar::zero(curve);
                let mut sum_r = Scalar::zero(curve);
                for j in 0..NUM_ROUND_GROUPS {
                    let sel = &columns[COL_IS_ROUND_GROUP_OFFSET + j][r];
                    sum_l = sum_l.add(
                        &sel.mul(&Scalar::from_u64(
                            ROL_REPRESENTATIVE_L[j] as u64, curve,
                        )),
                    );
                    sum_r = sum_r.add(
                        &sel.mul(&Scalar::from_u64(
                            ROL_REPRESENTATIVE_R[j] as u64, curve,
                        )),
                    );
                }
                let lp = &columns[COL_IS_LEFT_PASS][r];
                let expected = lp.mul(&sum_l).add(&one.sub(lp).mul(&sum_r));
                c[r] = columns[COL_ROL_AMOUNT][r].sub(&expected);
            }
            out.push(c);
        }

        // 182..202: state-cycling, all gated by is_real * (1 - is_result_row).
        //   A_next = state.E      → next_state[A_BYTE_OFFSET + b] = state[E + b]
        //   B_next = t_word       → next_state[B_BYTE_OFFSET + b] = t_word[b]
        //   C_next = state.B
        //   D_next = state.C      (ROL_10 deferred)
        //   E_next = state.D
        // a_off (in state), and offset into the LE byte layout.
        let push_cycling =
            |out: &mut Vec<Vec<Scalar>>,
             cur_byte_col: &dyn Fn(usize) -> usize,
             next_state_off: usize,
             columns: &[&Vec<Scalar>]| {
                for b in 0..STATE_WORD_BYTES {
                    let mut c = vec![Scalar::zero(curve); n];
                    for r in 0..n {
                        let real = &columns[COL_IS_REAL][r];
                        let rr = &columns[COL_IS_RESULT_ROW][r];
                        let gate = real.mul(&one.sub(rr));
                        let cur = &columns[cur_byte_col(b)][r];
                        let nxt = &columns[next_state_off + b][r];
                        c[r] = gate.mul(&nxt.sub(cur));
                    }
                    out.push(c);
                }
            };
        // A_next = state.E
        push_cycling(
            &mut out,
            &|b| COL_STATE_E_BYTE_OFFSET + b,
            COL_NEXT_STATE_OFFSET + (COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET),
            columns,
        );
        // B_next = t_word
        push_cycling(
            &mut out,
            &|b| COL_T_WORD_BYTE_OFFSET + b,
            COL_NEXT_STATE_OFFSET + (COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET),
            columns,
        );
        // C_next = state.B
        push_cycling(
            &mut out,
            &|b| COL_STATE_B_BYTE_OFFSET + b,
            COL_NEXT_STATE_OFFSET + (COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET),
            columns,
        );
        // D_next = state.C (ROL_10 deferred — shape-only)
        push_cycling(
            &mut out,
            &|b| COL_STATE_C_BYTE_OFFSET + b,
            COL_NEXT_STATE_OFFSET + (COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET),
            columns,
        );
        // E_next = state.D
        push_cycling(
            &mut out,
            &|b| COL_STATE_D_BYTE_OFFSET + b,
            COL_NEXT_STATE_OFFSET + (COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET),
            columns,
        );

        // ─── 202: intermediate_sum bit-decomp ────────────────────────
        // Σ 2^i * intermediate_sum_bits[i] + 2^32 * carry_val
        //   = A_val + f_val + W_val + K_val
        // where carry_val = carry_bits[0] + 2*carry_bits[1] and the
        // *_val terms are reconstructed from the existing LE byte
        // columns (state.A, message_word, constant_k) and from the
        // f_bits sum.
        let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                // LHS: bits + carry * 2^32.
                let mut bit_sum = Scalar::zero(curve);
                for i in 0..STATE_WORD_BITS {
                    let w = Scalar::from_u64(1u64 << i, curve);
                    bit_sum = bit_sum.add(
                        &columns[COL_INTERMEDIATE_SUM_BITS_OFFSET + i][r].mul(&w),
                    );
                }
                let mut carry_val = Scalar::zero(curve);
                for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
                    let w = Scalar::from_u64(1u64 << b, curve);
                    carry_val = carry_val.add(
                        &columns[COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b][r]
                            .mul(&w),
                    );
                }
                let lhs = bit_sum.add(&carry_val.mul(&two_pow_32));
                // RHS: A_val + f_val + W_val + K_val.
                let mut a_val = Scalar::zero(curve);
                for b in 0..STATE_WORD_BYTES {
                    let w = Scalar::from_u64(1u64 << (8 * b), curve);
                    a_val = a_val.add(
                        &columns[COL_STATE_A_BYTE_OFFSET + b][r].mul(&w),
                    );
                }
                let mut w_val = Scalar::zero(curve);
                for b in 0..MESSAGE_WORD_LENGTH {
                    let w = Scalar::from_u64(1u64 << (8 * b), curve);
                    w_val = w_val.add(
                        &columns[COL_MESSAGE_WORD_OFFSET + b][r].mul(&w),
                    );
                }
                let mut k_val = Scalar::zero(curve);
                for b in 0..CONSTANT_K_LENGTH {
                    let w = Scalar::from_u64(1u64 << (8 * b), curve);
                    k_val = k_val.add(
                        &columns[COL_CONSTANT_K_OFFSET + b][r].mul(&w),
                    );
                }
                let mut f_val = Scalar::zero(curve);
                for i in 0..STATE_WORD_BITS {
                    let w = Scalar::from_u64(1u64 << i, curve);
                    f_val = f_val.add(
                        &columns[COL_F_BITS_OFFSET + i][r].mul(&w),
                    );
                }
                let rhs = a_val.add(&f_val).add(&w_val).add(&k_val);
                // On non-real rows, A_val = ... = 0, bit_sum = 0, carry = 0;
                // so LHS = 0 = RHS trivially.
                c[r] = lhs.sub(&rhs);
            }
            out.push(c);
        }

        // 203, 204: intermediate_sum carry_bits binary.
        for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 205..237: per-bit variable ROL_s binding.
        //   rotated_word_bits[i] = Σ_{j,p} is_round_group[j] * pass_sel[p]
        //                          * intermediate_sum_bits[(i - ROL[p][j]) mod 32]
        // pass_sel[L] = is_left_pass, pass_sel[R] = 1 - is_left_pass.
        // On padding (is_round_group all zero), the RHS collapses to 0 and
        // we require rotated_word_bits[i] = 0 on padding rows. The trace
        // builder zeros those columns on padding, so this holds.
        for i in 0..STATE_WORD_BITS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lp = &columns[COL_IS_LEFT_PASS][r];
                let one_minus_lp = one.sub(lp);
                let mut rhs = Scalar::zero(curve);
                for j in 0..NUM_ROUND_GROUPS {
                    let sel = &columns[COL_IS_ROUND_GROUP_OFFSET + j][r];
                    // Left pass.
                    let s_l = ROL_REPRESENTATIVE_L[j] as usize;
                    let src_l_idx = (i + STATE_WORD_BITS - (s_l % STATE_WORD_BITS))
                        % STATE_WORD_BITS;
                    let src_l =
                        &columns[COL_INTERMEDIATE_SUM_BITS_OFFSET + src_l_idx][r];
                    rhs = rhs.add(&sel.mul(lp).mul(src_l));
                    // Right pass.
                    let s_r = ROL_REPRESENTATIVE_R[j] as usize;
                    let src_r_idx = (i + STATE_WORD_BITS - (s_r % STATE_WORD_BITS))
                        % STATE_WORD_BITS;
                    let src_r =
                        &columns[COL_INTERMEDIATE_SUM_BITS_OFFSET + src_r_idx][r];
                    rhs = rhs.add(&sel.mul(&one_minus_lp).mul(src_r));
                }
                let dst = &columns[COL_ROTATED_WORD_BITS_OFFSET + i][r];
                c[r] = dst.sub(&rhs);
            }
            out.push(c);
        }

        // 237: T add carry — t_word + carry * 2^32 = rotated_word + E.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut t_val = Scalar::zero(curve);
                for i in 0..STATE_WORD_BITS {
                    let w = Scalar::from_u64(1u64 << i, curve);
                    t_val = t_val.add(
                        &columns[COL_T_WORD_BITS_OFFSET + i][r].mul(&w),
                    );
                }
                let carry = &columns[COL_T_ADD_CARRY_BIT][r];
                let lhs = t_val.add(&carry.mul(&two_pow_32));
                let mut rot_val = Scalar::zero(curve);
                for i in 0..STATE_WORD_BITS {
                    let w = Scalar::from_u64(1u64 << i, curve);
                    rot_val = rot_val.add(
                        &columns[COL_ROTATED_WORD_BITS_OFFSET + i][r].mul(&w),
                    );
                }
                let mut e_val = Scalar::zero(curve);
                for b in 0..STATE_WORD_BYTES {
                    let w = Scalar::from_u64(1u64 << (8 * b), curve);
                    e_val = e_val.add(
                        &columns[COL_STATE_E_BYTE_OFFSET + b][r].mul(&w),
                    );
                }
                let rhs = rot_val.add(&e_val);
                c[r] = lhs.sub(&rhs);
            }
            out.push(c);
        }

        // 238: t_add_carry binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_T_ADD_CARRY_BIT][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        out
    }

    #[allow(unused_assignments)]
    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        // 0: is_real binary.
        let v = &col_evals[COL_IS_REAL];
        acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
        ap = ap.mul(alpha);

        // 1: is_left_pass binary.
        let v = &col_evals[COL_IS_LEFT_PASS];
        acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
        ap = ap.mul(alpha);

        // 2: is_result_row binary.
        let v = &col_evals[COL_IS_RESULT_ROW];
        acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
        ap = ap.mul(alpha);

        // 3: is_result_row * (1 - is_real) = 0.
        let rr = &col_evals[COL_IS_RESULT_ROW];
        let real = &col_evals[COL_IS_REAL];
        acc = acc.add(&ap.mul(&rr.mul(&one.sub(real))));
        ap = ap.mul(alpha);

        // 4: LE byte-decomp of round_index_word.
        let mut sum = Scalar::zero(curve);
        for b in 0..ROUND_INDEX_BYTES {
            let w = Scalar::from_u64(1u64 << (8 * b), curve);
            sum = sum.add(&col_evals[COL_ROUND_INDEX_BYTE_OFFSET + b].mul(&w));
        }
        acc = acc.add(&ap.mul(&sum.sub(&col_evals[COL_ROUND_INDEX])));
        ap = ap.mul(alpha);

        // 5: (1 - is_result_row) * Σ result[i] = 0.
        let mut sum = Scalar::zero(curve);
        for i in 0..RESULT_LENGTH {
            sum = sum.add(&col_evals[COL_RESULT_OFFSET + i]);
        }
        let gate = one.sub(&col_evals[COL_IS_RESULT_ROW]);
        acc = acc.add(&ap.mul(&gate.mul(&sum)));
        ap = ap.mul(alpha);

        // 6: Σ is_round_group[j] - is_real.
        let mut sum = Scalar::zero(curve);
        for j in 0..NUM_ROUND_GROUPS {
            sum = sum.add(&col_evals[COL_IS_ROUND_GROUP_OFFSET + j]);
        }
        acc = acc.add(&ap.mul(&sum.sub(&col_evals[COL_IS_REAL])));
        ap = ap.mul(alpha);

        // 7..12: is_round_group binary.
        for j in 0..NUM_ROUND_GROUPS {
            let v = &col_evals[COL_IS_ROUND_GROUP_OFFSET + j];
            acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
            ap = ap.mul(alpha);
        }

        // 12..16: word→bit decomp for B, C, D, f.
        let word_decomp = |bit_offset: usize, byte_offset: usize| -> Scalar {
            let mut bit_sum = Scalar::zero(curve);
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                bit_sum = bit_sum.add(&col_evals[bit_offset + i].mul(&w));
            }
            let mut byte_sum = Scalar::zero(curve);
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                byte_sum = byte_sum.add(&col_evals[byte_offset + b].mul(&w));
            }
            bit_sum.sub(&byte_sum)
        };
        for (bit_off, byte_off) in [
            (COL_B_BITS_OFFSET, COL_STATE_B_BYTE_OFFSET),
            (COL_C_BITS_OFFSET, COL_STATE_C_BYTE_OFFSET),
            (COL_D_BITS_OFFSET, COL_STATE_D_BYTE_OFFSET),
            (COL_F_BITS_OFFSET, COL_F_RESULT_OFFSET),
        ] {
            acc = acc.add(&ap.mul(&word_decomp(bit_off, byte_off)));
            ap = ap.mul(alpha);
        }

        // 16..48: per-bit XOR3 gated by is_round_group[0].
        let two = Scalar::from_u64(2, curve);
        let four = Scalar::from_u64(4, curve);
        for i in 0..STATE_WORD_BITS {
            let x = &col_evals[COL_B_BITS_OFFSET + i];
            let y = &col_evals[COL_C_BITS_OFFSET + i];
            let z = &col_evals[COL_D_BITS_OFFSET + i];
            let f = &col_evals[COL_F_BITS_OFFSET + i];
            let xy = x.mul(y);
            let xz = x.mul(z);
            let yz = y.mul(z);
            let xyz = xy.mul(z);
            let mut expected = x.add(y).add(z);
            let pairs = xy.add(&xz).add(&yz);
            expected = expected.sub(&two.mul(&pairs));
            expected = expected.add(&four.mul(&xyz));
            let gate = &col_evals[COL_IS_ROUND_GROUP_OFFSET + 0];
            acc = acc.add(&ap.mul(&gate.mul(&f.sub(&expected))));
            ap = ap.mul(alpha);
        }

        // 48..52: constant_k[b] pin.
        for b in 0..CONSTANT_K_LENGTH {
            let mut sum_l = Scalar::zero(curve);
            let mut sum_r = Scalar::zero(curve);
            for j in 0..NUM_ROUND_GROUPS {
                let kl_byte = ((K_L[j] >> (8 * b)) & 0xff) as u64;
                let kr_byte = ((K_R[j] >> (8 * b)) & 0xff) as u64;
                let sel = &col_evals[COL_IS_ROUND_GROUP_OFFSET + j];
                sum_l = sum_l.add(&sel.mul(&Scalar::from_u64(kl_byte, curve)));
                sum_r = sum_r.add(&sel.mul(&Scalar::from_u64(kr_byte, curve)));
            }
            let lp = &col_evals[COL_IS_LEFT_PASS];
            let expected = lp.mul(&sum_l).add(&one.sub(lp).mul(&sum_r));
            acc = acc.add(&ap.mul(&col_evals[COL_CONSTANT_K_OFFSET + b].sub(&expected)));
            ap = ap.mul(alpha);
        }

        // 52..84: f_1 = xy + z - xz, gated by is_round_group[1].
        for i in 0..STATE_WORD_BITS {
            let x = &col_evals[COL_B_BITS_OFFSET + i];
            let y = &col_evals[COL_C_BITS_OFFSET + i];
            let z = &col_evals[COL_D_BITS_OFFSET + i];
            let f = &col_evals[COL_F_BITS_OFFSET + i];
            let xy = x.mul(y);
            let xz = x.mul(z);
            let expected = xy.add(z).sub(&xz);
            let gate = &col_evals[COL_IS_ROUND_GROUP_OFFSET + 1];
            acc = acc.add(&ap.mul(&gate.mul(&f.sub(&expected))));
            ap = ap.mul(alpha);
        }

        // 84..116: f_2 = g + z - 2gz, g = 1 - y + xy, gated by is_round_group[2].
        for i in 0..STATE_WORD_BITS {
            let x = &col_evals[COL_B_BITS_OFFSET + i];
            let y = &col_evals[COL_C_BITS_OFFSET + i];
            let z = &col_evals[COL_D_BITS_OFFSET + i];
            let f = &col_evals[COL_F_BITS_OFFSET + i];
            let xy = x.mul(y);
            let g = one.sub(y).add(&xy);
            let gz = g.mul(z);
            let two_gz = two.mul(&gz);
            let expected = g.add(z).sub(&two_gz);
            let gate = &col_evals[COL_IS_ROUND_GROUP_OFFSET + 2];
            acc = acc.add(&ap.mul(&gate.mul(&f.sub(&expected))));
            ap = ap.mul(alpha);
        }

        // 116..148: f_3 = xz + y - yz, gated by is_round_group[3].
        for i in 0..STATE_WORD_BITS {
            let x = &col_evals[COL_B_BITS_OFFSET + i];
            let y = &col_evals[COL_C_BITS_OFFSET + i];
            let z = &col_evals[COL_D_BITS_OFFSET + i];
            let f = &col_evals[COL_F_BITS_OFFSET + i];
            let xz = x.mul(z);
            let yz = y.mul(z);
            let expected = xz.add(y).sub(&yz);
            let gate = &col_evals[COL_IS_ROUND_GROUP_OFFSET + 3];
            acc = acc.add(&ap.mul(&gate.mul(&f.sub(&expected))));
            ap = ap.mul(alpha);
        }

        // 148..180: f_4 = x + h - 2xh, h = 1 - z + yz, gated by is_round_group[4].
        for i in 0..STATE_WORD_BITS {
            let x = &col_evals[COL_B_BITS_OFFSET + i];
            let y = &col_evals[COL_C_BITS_OFFSET + i];
            let z = &col_evals[COL_D_BITS_OFFSET + i];
            let f = &col_evals[COL_F_BITS_OFFSET + i];
            let yz = y.mul(z);
            let h = one.sub(z).add(&yz);
            let xh = x.mul(&h);
            let two_xh = two.mul(&xh);
            let expected = x.add(&h).sub(&two_xh);
            let gate = &col_evals[COL_IS_ROUND_GROUP_OFFSET + 4];
            acc = acc.add(&ap.mul(&gate.mul(&f.sub(&expected))));
            ap = ap.mul(alpha);
        }

        // 180: t_word bit-decomp.
        {
            let mut bit_sum = Scalar::zero(curve);
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                bit_sum = bit_sum.add(&col_evals[COL_T_WORD_BITS_OFFSET + i].mul(&w));
            }
            let mut byte_sum = Scalar::zero(curve);
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                byte_sum = byte_sum.add(&col_evals[COL_T_WORD_BYTE_OFFSET + b].mul(&w));
            }
            acc = acc.add(&ap.mul(&bit_sum.sub(&byte_sum)));
            ap = ap.mul(alpha);
        }

        // 181: rol_amount pin.
        {
            let mut sum_l = Scalar::zero(curve);
            let mut sum_r = Scalar::zero(curve);
            for j in 0..NUM_ROUND_GROUPS {
                let sel = &col_evals[COL_IS_ROUND_GROUP_OFFSET + j];
                sum_l = sum_l.add(&sel.mul(&Scalar::from_u64(
                    ROL_REPRESENTATIVE_L[j] as u64, curve,
                )));
                sum_r = sum_r.add(&sel.mul(&Scalar::from_u64(
                    ROL_REPRESENTATIVE_R[j] as u64, curve,
                )));
            }
            let lp = &col_evals[COL_IS_LEFT_PASS];
            let expected = lp.mul(&sum_l).add(&one.sub(lp).mul(&sum_r));
            acc = acc.add(&ap.mul(&col_evals[COL_ROL_AMOUNT].sub(&expected)));
            ap = ap.mul(alpha);
        }

        // 182..202: state cycling gated by is_real * (1 - is_result_row).
        let real = &col_evals[COL_IS_REAL];
        let rr = &col_evals[COL_IS_RESULT_ROW];
        let gate = real.mul(&one.sub(rr));
        let cycling_pairs: [(usize, usize); 5] = [
            (COL_STATE_E_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_T_WORD_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_STATE_B_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_STATE_C_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_STATE_D_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET)),
        ];
        for (cur_off, next_off) in cycling_pairs.iter() {
            for b in 0..STATE_WORD_BYTES {
                let cur = &col_evals[cur_off + b];
                let nxt = &col_evals[next_off + b];
                acc = acc.add(&ap.mul(&gate.mul(&nxt.sub(cur))));
                ap = ap.mul(alpha);
            }
        }

        // 202: intermediate_sum bit-decomp.
        let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
        {
            let mut bit_sum = Scalar::zero(curve);
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                bit_sum = bit_sum.add(
                    &col_evals[COL_INTERMEDIATE_SUM_BITS_OFFSET + i].mul(&w),
                );
            }
            let mut carry_val = Scalar::zero(curve);
            for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
                let w = Scalar::from_u64(1u64 << b, curve);
                carry_val = carry_val.add(
                    &col_evals[COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b].mul(&w),
                );
            }
            let lhs = bit_sum.add(&carry_val.mul(&two_pow_32));
            let mut a_val = Scalar::zero(curve);
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                a_val = a_val.add(&col_evals[COL_STATE_A_BYTE_OFFSET + b].mul(&w));
            }
            let mut w_val = Scalar::zero(curve);
            for b in 0..MESSAGE_WORD_LENGTH {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                w_val = w_val.add(
                    &col_evals[COL_MESSAGE_WORD_OFFSET + b].mul(&w),
                );
            }
            let mut k_val = Scalar::zero(curve);
            for b in 0..CONSTANT_K_LENGTH {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                k_val = k_val.add(
                    &col_evals[COL_CONSTANT_K_OFFSET + b].mul(&w),
                );
            }
            let mut f_val = Scalar::zero(curve);
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                f_val = f_val.add(&col_evals[COL_F_BITS_OFFSET + i].mul(&w));
            }
            let rhs = a_val.add(&f_val).add(&w_val).add(&k_val);
            acc = acc.add(&ap.mul(&lhs.sub(&rhs)));
            ap = ap.mul(alpha);
        }

        // 203, 204: intermediate_sum carry_bits binary.
        for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
            let v = &col_evals[COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b];
            acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
            ap = ap.mul(alpha);
        }

        // 205..237: per-bit variable ROL_s binding.
        let lp = &col_evals[COL_IS_LEFT_PASS];
        let one_minus_lp_e = one.sub(lp);
        for i in 0..STATE_WORD_BITS {
            let mut rhs = Scalar::zero(curve);
            for j in 0..NUM_ROUND_GROUPS {
                let sel = &col_evals[COL_IS_ROUND_GROUP_OFFSET + j];
                let s_l = ROL_REPRESENTATIVE_L[j] as usize;
                let src_l_idx = (i + STATE_WORD_BITS - (s_l % STATE_WORD_BITS))
                    % STATE_WORD_BITS;
                let src_l =
                    &col_evals[COL_INTERMEDIATE_SUM_BITS_OFFSET + src_l_idx];
                rhs = rhs.add(&sel.mul(lp).mul(src_l));
                let s_r = ROL_REPRESENTATIVE_R[j] as usize;
                let src_r_idx = (i + STATE_WORD_BITS - (s_r % STATE_WORD_BITS))
                    % STATE_WORD_BITS;
                let src_r =
                    &col_evals[COL_INTERMEDIATE_SUM_BITS_OFFSET + src_r_idx];
                rhs = rhs.add(&sel.mul(&one_minus_lp_e).mul(src_r));
            }
            let dst = &col_evals[COL_ROTATED_WORD_BITS_OFFSET + i];
            acc = acc.add(&ap.mul(&dst.sub(&rhs)));
            ap = ap.mul(alpha);
        }

        // 237: T add — t_word + carry*2^32 = rotated_word + E.
        {
            let mut t_val = Scalar::zero(curve);
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                t_val = t_val.add(
                    &col_evals[COL_T_WORD_BITS_OFFSET + i].mul(&w),
                );
            }
            let carry = &col_evals[COL_T_ADD_CARRY_BIT];
            let lhs = t_val.add(&carry.mul(&two_pow_32));
            let mut rot_val = Scalar::zero(curve);
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                rot_val = rot_val.add(
                    &col_evals[COL_ROTATED_WORD_BITS_OFFSET + i].mul(&w),
                );
            }
            let mut e_val = Scalar::zero(curve);
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                e_val = e_val.add(
                    &col_evals[COL_STATE_E_BYTE_OFFSET + b].mul(&w),
                );
            }
            let rhs = rot_val.add(&e_val);
            acc = acc.add(&ap.mul(&lhs.sub(&rhs)));
            ap = ap.mul(alpha);
        }

        // 238: t_add_carry binary.
        {
            let v = &col_evals[COL_T_ADD_CARRY_BIT];
            acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
            ap = ap.mul(alpha);
        }

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

        // 0: is_real binary.
        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // 1: is_left_pass binary.
        let v = &col_coeffs[COL_IS_LEFT_PASS];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // 2: is_result_row binary.
        let v = &col_coeffs[COL_IS_RESULT_ROW];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // 3: is_result_row * (1 - is_real).
        let rr = &col_coeffs[COL_IS_RESULT_ROW];
        let real = &col_coeffs[COL_IS_REAL];
        let one_minus = poly_sub(&one_poly, real, curve);
        let body = poly_mul(rr, &one_minus, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // 4: LE byte-decomp.
        let mut sum = vec![Scalar::zero(curve)];
        for b in 0..ROUND_INDEX_BYTES {
            let w = Scalar::from_u64(1u64 << (8 * b), curve);
            let term =
                poly_scalar_mul(&col_coeffs[COL_ROUND_INDEX_BYTE_OFFSET + b], &w);
            sum = poly_add(&sum, &term, curve);
        }
        let body = poly_sub(&sum, &col_coeffs[COL_ROUND_INDEX], curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // 5: (1 - is_result_row) * Σ result[i].
        let mut sum = vec![Scalar::zero(curve)];
        for i in 0..RESULT_LENGTH {
            sum = poly_add(&sum, &col_coeffs[COL_RESULT_OFFSET + i], curve);
        }
        let gate = poly_sub(&one_poly, &col_coeffs[COL_IS_RESULT_ROW], curve);
        let body = poly_mul(&gate, &sum, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // 6: Σ is_round_group[j] - is_real.
        let mut sum = vec![Scalar::zero(curve)];
        for j in 0..NUM_ROUND_GROUPS {
            sum = poly_add(&sum, &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + j], curve);
        }
        let body = poly_sub(&sum, &col_coeffs[COL_IS_REAL], curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // 7..12: is_round_group binary.
        for j in 0..NUM_ROUND_GROUPS {
            let v = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + j];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 12..16: word→bit decomp B, C, D, f.
        let word_decomp_poly = |bit_offset: usize,
                                 byte_offset: usize|
         -> Vec<Scalar> {
            let mut bit_sum = vec![Scalar::zero(curve)];
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                bit_sum = poly_add(
                    &bit_sum,
                    &poly_scalar_mul(&col_coeffs[bit_offset + i], &w),
                    curve,
                );
            }
            let mut byte_sum = vec![Scalar::zero(curve)];
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                byte_sum = poly_add(
                    &byte_sum,
                    &poly_scalar_mul(&col_coeffs[byte_offset + b], &w),
                    curve,
                );
            }
            poly_sub(&bit_sum, &byte_sum, curve)
        };
        for (bit_off, byte_off) in [
            (COL_B_BITS_OFFSET, COL_STATE_B_BYTE_OFFSET),
            (COL_C_BITS_OFFSET, COL_STATE_C_BYTE_OFFSET),
            (COL_D_BITS_OFFSET, COL_STATE_D_BYTE_OFFSET),
            (COL_F_BITS_OFFSET, COL_F_RESULT_OFFSET),
        ] {
            let body = word_decomp_poly(bit_off, byte_off);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 16..48: per-bit XOR3 gated by is_round_group[0].
        let two = Scalar::from_u64(2, curve);
        let four = Scalar::from_u64(4, curve);
        let neg_two = Scalar::zero(curve).sub(&two);
        for i in 0..STATE_WORD_BITS {
            let x = &col_coeffs[COL_B_BITS_OFFSET + i];
            let y = &col_coeffs[COL_C_BITS_OFFSET + i];
            let z = &col_coeffs[COL_D_BITS_OFFSET + i];
            let f = &col_coeffs[COL_F_BITS_OFFSET + i];
            let xy = poly_mul(x, y, curve);
            let xz = poly_mul(x, z, curve);
            let yz = poly_mul(y, z, curve);
            let xyz = poly_mul(&xy, z, curve);
            let mut expected = poly_add(&poly_add(x, y, curve), z, curve);
            let pairs = poly_add(&poly_add(&xy, &xz, curve), &yz, curve);
            expected = poly_add(&expected, &poly_scalar_mul(&pairs, &neg_two), curve);
            expected = poly_add(&expected, &poly_scalar_mul(&xyz, &four), curve);
            let diff = poly_sub(f, &expected, curve);
            let gate = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + 0];
            let body = poly_mul(gate, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 48..52: constant_k[b] pin.
        for b in 0..CONSTANT_K_LENGTH {
            let mut sum_l = vec![Scalar::zero(curve)];
            let mut sum_r = vec![Scalar::zero(curve)];
            for j in 0..NUM_ROUND_GROUPS {
                let kl_byte = ((K_L[j] >> (8 * b)) & 0xff) as u64;
                let kr_byte = ((K_R[j] >> (8 * b)) & 0xff) as u64;
                let sel = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + j];
                sum_l = poly_add(
                    &sum_l,
                    &poly_scalar_mul(sel, &Scalar::from_u64(kl_byte, curve)),
                    curve,
                );
                sum_r = poly_add(
                    &sum_r,
                    &poly_scalar_mul(sel, &Scalar::from_u64(kr_byte, curve)),
                    curve,
                );
            }
            let lp = &col_coeffs[COL_IS_LEFT_PASS];
            let one_minus_lp = poly_sub(&one_poly, lp, curve);
            let expected = poly_add(
                &poly_mul(lp, &sum_l, curve),
                &poly_mul(&one_minus_lp, &sum_r, curve),
                curve,
            );
            let body = poly_sub(&col_coeffs[COL_CONSTANT_K_OFFSET + b], &expected, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 52..84: f_1 = xy + z - xz gated by is_round_group[1].
        for i in 0..STATE_WORD_BITS {
            let x = &col_coeffs[COL_B_BITS_OFFSET + i];
            let y = &col_coeffs[COL_C_BITS_OFFSET + i];
            let z = &col_coeffs[COL_D_BITS_OFFSET + i];
            let f = &col_coeffs[COL_F_BITS_OFFSET + i];
            let xy = poly_mul(x, y, curve);
            let xz = poly_mul(x, z, curve);
            let expected = poly_sub(&poly_add(&xy, z, curve), &xz, curve);
            let diff = poly_sub(f, &expected, curve);
            let gate = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + 1];
            let body = poly_mul(gate, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 84..116: f_2 = g + z - 2gz, g = 1 - y + xy, gated by is_round_group[2].
        for i in 0..STATE_WORD_BITS {
            let x = &col_coeffs[COL_B_BITS_OFFSET + i];
            let y = &col_coeffs[COL_C_BITS_OFFSET + i];
            let z = &col_coeffs[COL_D_BITS_OFFSET + i];
            let f = &col_coeffs[COL_F_BITS_OFFSET + i];
            let xy = poly_mul(x, y, curve);
            // g = 1 - y + xy = (1_poly - y) + xy
            let one_minus_y = poly_sub(&one_poly, y, curve);
            let g = poly_add(&one_minus_y, &xy, curve);
            let gz = poly_mul(&g, z, curve);
            let two_gz = poly_scalar_mul(&gz, &two);
            let g_plus_z = poly_add(&g, z, curve);
            let expected = poly_sub(&g_plus_z, &two_gz, curve);
            let diff = poly_sub(f, &expected, curve);
            let gate = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + 2];
            let body = poly_mul(gate, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 116..148: f_3 = xz + y - yz gated by is_round_group[3].
        for i in 0..STATE_WORD_BITS {
            let x = &col_coeffs[COL_B_BITS_OFFSET + i];
            let y = &col_coeffs[COL_C_BITS_OFFSET + i];
            let z = &col_coeffs[COL_D_BITS_OFFSET + i];
            let f = &col_coeffs[COL_F_BITS_OFFSET + i];
            let xz = poly_mul(x, z, curve);
            let yz = poly_mul(y, z, curve);
            let expected = poly_sub(&poly_add(&xz, y, curve), &yz, curve);
            let diff = poly_sub(f, &expected, curve);
            let gate = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + 3];
            let body = poly_mul(gate, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 148..180: f_4 = x + h - 2xh, h = 1 - z + yz, gated by is_round_group[4].
        for i in 0..STATE_WORD_BITS {
            let x = &col_coeffs[COL_B_BITS_OFFSET + i];
            let y = &col_coeffs[COL_C_BITS_OFFSET + i];
            let z = &col_coeffs[COL_D_BITS_OFFSET + i];
            let f = &col_coeffs[COL_F_BITS_OFFSET + i];
            let yz = poly_mul(y, z, curve);
            let one_minus_z = poly_sub(&one_poly, z, curve);
            let h = poly_add(&one_minus_z, &yz, curve);
            let xh = poly_mul(x, &h, curve);
            let two_xh = poly_scalar_mul(&xh, &two);
            let x_plus_h = poly_add(x, &h, curve);
            let expected = poly_sub(&x_plus_h, &two_xh, curve);
            let diff = poly_sub(f, &expected, curve);
            let gate = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + 4];
            let body = poly_mul(gate, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 180: t_word bit-decomp.
        {
            let mut bit_sum = vec![Scalar::zero(curve)];
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                bit_sum = poly_add(
                    &bit_sum,
                    &poly_scalar_mul(&col_coeffs[COL_T_WORD_BITS_OFFSET + i], &w),
                    curve,
                );
            }
            let mut byte_sum = vec![Scalar::zero(curve)];
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                byte_sum = poly_add(
                    &byte_sum,
                    &poly_scalar_mul(&col_coeffs[COL_T_WORD_BYTE_OFFSET + b], &w),
                    curve,
                );
            }
            let body = poly_sub(&bit_sum, &byte_sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 181: rol_amount pin.
        {
            let mut sum_l = vec![Scalar::zero(curve)];
            let mut sum_r = vec![Scalar::zero(curve)];
            for j in 0..NUM_ROUND_GROUPS {
                let sel = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + j];
                sum_l = poly_add(
                    &sum_l,
                    &poly_scalar_mul(sel, &Scalar::from_u64(
                        ROL_REPRESENTATIVE_L[j] as u64, curve,
                    )),
                    curve,
                );
                sum_r = poly_add(
                    &sum_r,
                    &poly_scalar_mul(sel, &Scalar::from_u64(
                        ROL_REPRESENTATIVE_R[j] as u64, curve,
                    )),
                    curve,
                );
            }
            let lp = &col_coeffs[COL_IS_LEFT_PASS];
            let one_minus_lp = poly_sub(&one_poly, lp, curve);
            let expected = poly_add(
                &poly_mul(lp, &sum_l, curve),
                &poly_mul(&one_minus_lp, &sum_r, curve),
                curve,
            );
            let body = poly_sub(&col_coeffs[COL_ROL_AMOUNT], &expected, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 182..202: state cycling gated by is_real * (1 - is_result_row).
        let real = &col_coeffs[COL_IS_REAL];
        let rr = &col_coeffs[COL_IS_RESULT_ROW];
        let one_minus_rr = poly_sub(&one_poly, rr, curve);
        let cyc_gate = poly_mul(real, &one_minus_rr, curve);
        let cycling_pairs: [(usize, usize); 5] = [
            (COL_STATE_E_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_T_WORD_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_STATE_B_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_STATE_C_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET)),
            (COL_STATE_D_BYTE_OFFSET,
             COL_NEXT_STATE_OFFSET + (COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET)),
        ];
        for (cur_off, next_off) in cycling_pairs.iter() {
            for b in 0..STATE_WORD_BYTES {
                let cur = &col_coeffs[cur_off + b];
                let nxt = &col_coeffs[next_off + b];
                let diff = poly_sub(nxt, cur, curve);
                let body = poly_mul(&cyc_gate, &diff, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
                ap = ap.mul(alpha);
            }
        }

        // 202: intermediate_sum bit-decomp.
        let two_pow_32_scalar = Scalar::from_u64(1u64 << 32, curve);
        {
            let mut bit_sum = vec![Scalar::zero(curve)];
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                bit_sum = poly_add(
                    &bit_sum,
                    &poly_scalar_mul(
                        &col_coeffs[COL_INTERMEDIATE_SUM_BITS_OFFSET + i], &w,
                    ),
                    curve,
                );
            }
            let mut carry_val = vec![Scalar::zero(curve)];
            for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
                let w = Scalar::from_u64(1u64 << b, curve);
                carry_val = poly_add(
                    &carry_val,
                    &poly_scalar_mul(
                        &col_coeffs[COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b], &w,
                    ),
                    curve,
                );
            }
            let lhs = poly_add(
                &bit_sum,
                &poly_scalar_mul(&carry_val, &two_pow_32_scalar),
                curve,
            );
            let mut a_val = vec![Scalar::zero(curve)];
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                a_val = poly_add(
                    &a_val,
                    &poly_scalar_mul(&col_coeffs[COL_STATE_A_BYTE_OFFSET + b], &w),
                    curve,
                );
            }
            let mut w_val = vec![Scalar::zero(curve)];
            for b in 0..MESSAGE_WORD_LENGTH {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                w_val = poly_add(
                    &w_val,
                    &poly_scalar_mul(
                        &col_coeffs[COL_MESSAGE_WORD_OFFSET + b], &w,
                    ),
                    curve,
                );
            }
            let mut k_val = vec![Scalar::zero(curve)];
            for b in 0..CONSTANT_K_LENGTH {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                k_val = poly_add(
                    &k_val,
                    &poly_scalar_mul(
                        &col_coeffs[COL_CONSTANT_K_OFFSET + b], &w,
                    ),
                    curve,
                );
            }
            let mut f_val = vec![Scalar::zero(curve)];
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                f_val = poly_add(
                    &f_val,
                    &poly_scalar_mul(&col_coeffs[COL_F_BITS_OFFSET + i], &w),
                    curve,
                );
            }
            let rhs = poly_add(
                &poly_add(&poly_add(&a_val, &f_val, curve), &w_val, curve),
                &k_val,
                curve,
            );
            let body = poly_sub(&lhs, &rhs, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 203, 204: intermediate_sum carry_bits binary.
        for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
            let v = &col_coeffs[COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 205..237: per-bit variable ROL_s binding.
        let lp_poly = &col_coeffs[COL_IS_LEFT_PASS];
        let one_minus_lp_poly = poly_sub(&one_poly, lp_poly, curve);
        for i in 0..STATE_WORD_BITS {
            let mut rhs = vec![Scalar::zero(curve)];
            for j in 0..NUM_ROUND_GROUPS {
                let sel = &col_coeffs[COL_IS_ROUND_GROUP_OFFSET + j];
                let s_l = ROL_REPRESENTATIVE_L[j] as usize;
                let src_l_idx = (i + STATE_WORD_BITS - (s_l % STATE_WORD_BITS))
                    % STATE_WORD_BITS;
                let src_l =
                    &col_coeffs[COL_INTERMEDIATE_SUM_BITS_OFFSET + src_l_idx];
                let term_l = poly_mul(
                    &poly_mul(sel, lp_poly, curve),
                    src_l,
                    curve,
                );
                rhs = poly_add(&rhs, &term_l, curve);
                let s_r = ROL_REPRESENTATIVE_R[j] as usize;
                let src_r_idx = (i + STATE_WORD_BITS - (s_r % STATE_WORD_BITS))
                    % STATE_WORD_BITS;
                let src_r =
                    &col_coeffs[COL_INTERMEDIATE_SUM_BITS_OFFSET + src_r_idx];
                let term_r = poly_mul(
                    &poly_mul(sel, &one_minus_lp_poly, curve),
                    src_r,
                    curve,
                );
                rhs = poly_add(&rhs, &term_r, curve);
            }
            let dst = &col_coeffs[COL_ROTATED_WORD_BITS_OFFSET + i];
            let body = poly_sub(dst, &rhs, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 237: T add.
        {
            let mut t_val = vec![Scalar::zero(curve)];
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                t_val = poly_add(
                    &t_val,
                    &poly_scalar_mul(&col_coeffs[COL_T_WORD_BITS_OFFSET + i], &w),
                    curve,
                );
            }
            let carry = &col_coeffs[COL_T_ADD_CARRY_BIT];
            let lhs = poly_add(
                &t_val,
                &poly_scalar_mul(carry, &two_pow_32_scalar),
                curve,
            );
            let mut rot_val = vec![Scalar::zero(curve)];
            for i in 0..STATE_WORD_BITS {
                let w = Scalar::from_u64(1u64 << i, curve);
                rot_val = poly_add(
                    &rot_val,
                    &poly_scalar_mul(
                        &col_coeffs[COL_ROTATED_WORD_BITS_OFFSET + i], &w,
                    ),
                    curve,
                );
            }
            let mut e_val = vec![Scalar::zero(curve)];
            for b in 0..STATE_WORD_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                e_val = poly_add(
                    &e_val,
                    &poly_scalar_mul(
                        &col_coeffs[COL_STATE_E_BYTE_OFFSET + b], &w,
                    ),
                    curve,
                );
            }
            let rhs = poly_add(&rot_val, &e_val, curve);
            let body = poly_sub(&lhs, &rhs, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // 238: t_add_carry binary.
        {
            let v = &col_coeffs[COL_T_ADD_CARRY_BIT];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // Silence "unused assignment to ap" lint if any.
        let _ = ap;

        acc
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Shifted: state[0..STATE_LENGTH], is_real, is_result_row.
        let mut cols = Vec::with_capacity(STATE_LENGTH + 2);
        for k in 0..STATE_LENGTH {
            cols.push(COL_STATE_OFFSET + k);
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
        let expected = STATE_LENGTH + 2;
        if shifted_evals.len() != expected || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let real_z = &col_evals_at_z[COL_IS_REAL];
        let real_next = &shifted_evals[STATE_LENGTH];
        let is_result = &col_evals_at_z[COL_IS_RESULT_ROW];
        let gating = real_z
            .mul(real_next)
            .mul(&one.sub(is_result));

        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..STATE_LENGTH {
            let nxt = &shifted_evals[k];
            let cur_next_state = &col_evals_at_z[COL_NEXT_STATE_OFFSET + k];
            acc = acc.add(&bp.mul(&nxt.sub(cur_next_state)));
            bp = bp.mul(alpha);
        }
        let body = gating.mul(&acc);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&body).mul(&z.sub(omega_n_minus_1))
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
        let one_minus_rr = poly_sub(&one_poly, is_result, curve);
        let gating = poly_mul(&poly_mul(real, &real_next, curve), &one_minus_rr, curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..STATE_LENGTH {
            let cur = &column_coeffs[COL_STATE_OFFSET + k];
            let nxt = poly_shift(cur, omega);
            let cur_next_state = &column_coeffs[COL_NEXT_STATE_OFFSET + k];
            let diff = poly_sub(&nxt, cur_next_state, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body = poly_mul(&gating, &acc, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let total = poly_scalar_mul(&body, &ap);

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_IS_RESULT_ROW, COL_IS_LEFT_PASS]
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
        let tables = vec![LookupTable::range(256), LookupTable::range(2)];
        let mut declarations = Vec::new();
        for k in 0..STATE_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_state_byte_{}_8bit", k),
                    column_index: COL_STATE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_next_state_byte_{}_8bit", k),
                    column_index: COL_NEXT_STATE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_result_byte_{}_8bit", k),
                    column_index: COL_RESULT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MESSAGE_WORD_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_message_byte_{}_8bit", k),
                    column_index: COL_MESSAGE_WORD_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..CONSTANT_K_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_constant_k_byte_{}_8bit", k),
                    column_index: COL_CONSTANT_K_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..ROUND_INDEX_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_round_index_byte_{}_8bit", b),
                    column_index: COL_ROUND_INDEX_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..STATE_WORD_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_f_result_byte_{}_8bit", b),
                    column_index: COL_F_RESULT_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // Bit columns: bit-binary enforced via max_bits=1 range lookup.
        for (label, off) in [
            ("ripemd160_internals_b_bit", COL_B_BITS_OFFSET),
            ("ripemd160_internals_c_bit", COL_C_BITS_OFFSET),
            ("ripemd160_internals_d_bit", COL_D_BITS_OFFSET),
            ("ripemd160_internals_f_bit", COL_F_BITS_OFFSET),
        ] {
            for i in 0..STATE_WORD_BITS {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_1bit", label, i),
                        column_index: off + i,
                        max_bits: 1,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        for j in 0..NUM_ROUND_GROUPS {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_is_round_group_{}_1bit", j),
                    column_index: COL_IS_ROUND_GROUP_OFFSET + j,
                    max_bits: 1,
                    selector_column: None,
                },
                0,
            ));
        }
        // t_word bytes 8-bit, t_word bits 1-bit, rol_amount 8-bit.
        for b in 0..STATE_WORD_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_t_word_byte_{}_8bit", b),
                    column_index: COL_T_WORD_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for i in 0..STATE_WORD_BITS {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_internals_t_word_bit_{}_1bit", i),
                    column_index: COL_T_WORD_BITS_OFFSET + i,
                    max_bits: 1,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "ripemd160_internals_rol_amount_8bit".to_string(),
                column_index: COL_ROL_AMOUNT,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        // Task #195 columns.
        for i in 0..STATE_WORD_BITS {
            declarations.push((
                LookupDeclaration {
                    label: format!(
                        "ripemd160_internals_intermediate_sum_bit_{}_1bit", i,
                    ),
                    column_index: COL_INTERMEDIATE_SUM_BITS_OFFSET + i,
                    max_bits: 1,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..INTERMEDIATE_SUM_CARRY_BITS {
            declarations.push((
                LookupDeclaration {
                    label: format!(
                        "ripemd160_internals_intermediate_sum_carry_bit_{}_1bit", b,
                    ),
                    column_index: COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET + b,
                    max_bits: 1,
                    selector_column: None,
                },
                0,
            ));
        }
        for i in 0..STATE_WORD_BITS {
            declarations.push((
                LookupDeclaration {
                    label: format!(
                        "ripemd160_internals_rotated_word_bit_{}_1bit", i,
                    ),
                    column_index: COL_ROTATED_WORD_BITS_OFFSET + i,
                    max_bits: 1,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "ripemd160_internals_t_add_carry_bit_1bit".to_string(),
                column_index: COL_T_ADD_CARRY_BIT,
                max_bits: 1,
                selector_column: None,
            },
            0,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Placeholder sentinel for the precompile-side ripemd_output column.
pub const PRECOMPILE_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (skeleton): binds the FINAL-row
/// `result[0..20]` of the internals AIR to the
/// `ripemd_output[0..20]` column of [`crate::ripemd160_precompile_air`].
///
/// A side selector: `COL_IS_RESULT_ROW` (only the final row participates).
/// B side: the precompile AIR's `is_real` column (with the matching
/// `ripemd_output` byte columns). The orchestrator substitutes the real
/// indices.
pub fn make_ripemd160_internals_to_precompile_descriptor(
    internals_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(RESULT_LENGTH);
    for k in 0..RESULT_LENGTH {
        a_columns.push(COL_RESULT_OFFSET + k);
    }
    // B side: PRECOMPILE_PLACEHOLDER for ripemd_output[0..20] —
    // orchestrator substitutes ripemd160_precompile_air::COL_RIPEMD_OUTPUT_OFFSET + k.
    let b_columns: Vec<usize> = vec![PRECOMPILE_PLACEHOLDER; RESULT_LENGTH];
    CrossAirLogUpDescriptor {
        label: "ripemd160_internals_to_precompile_result_v1_stub".into(),
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

    /// RIPEMD-160 of the empty string.
    const RIPEMD_EMPTY: [u8; RESULT_LENGTH] = [
        0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9, 0xfc, 0x54,
        0x61, 0x28, 0x08, 0x97, 0x7e, 0xe8, 0xf5, 0x48,
        0xb2, 0x25, 0x8d, 0x31,
    ];

    /// RIPEMD-160 of "abc".
    const RIPEMD_ABC: [u8; RESULT_LENGTH] = [
        0x8e, 0xb2, 0x08, 0xf7, 0xe0, 0x5d, 0x98, 0x7a,
        0x9b, 0x04, 0x4a, 0x8e, 0x98, 0xc6, 0xb0, 0x87,
        0xf1, 0x5a, 0x0b, 0xfc,
    ];

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
    fn ripemd160_internals_air_empty_input_known_hash() {
        let w = Ripemd160InternalsTraceWitness::from_digest_skeleton(b"");
        assert_eq!(w.rows.len(), 2);
        assert_eq!(w.rows[1].result, RIPEMD_EMPTY);
        assert!(w.rows[1].is_result_row);
        assert!(!w.rows[0].is_result_row);

        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_internals_air_abc_known_hash() {
        let w = Ripemd160InternalsTraceWitness::from_digest_skeleton(b"abc");
        assert_eq!(w.rows[1].result, RIPEMD_ABC);

        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_internals_air_tampered_result_on_nonfinal_row_detected() {
        // Tamper a result byte on the non-final row — constraint 5
        // (gated `Σ result = 0` on non-result rows) MUST fire.
        let w = Ripemd160InternalsTraceWitness::from_digest_skeleton(b"abc");
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Row 0 is NOT the result row; smear a result byte to nonzero.
        cols[COL_RESULT_OFFSET + 0][0] = Scalar::from_u64(0x42, CurveType::Bls48581);

        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 5 = "result_zero_off_final_row".
        assert!(
            !bodies[5][0].is_zero(),
            "expected gated-result constraint to fire on non-final row",
        );
    }

    #[test]
    fn ripemd160_internals_air_tampered_round_index_decomp_detected() {
        let w = Ripemd160InternalsTraceWitness::from_digest_skeleton(b"abc");
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper round_index without touching the LE bytes — constraint 4 fires.
        cols[COL_ROUND_INDEX][1] = Scalar::from_u64(999, CurveType::Bls48581);

        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[4][1].is_zero(),
            "expected round_index byte-decomp constraint to fire",
        );
    }

    #[test]
    fn ripemd160_internals_air_descriptor_well_formed() {
        let d = make_ripemd160_internals_to_precompile_descriptor(0, 1);
        assert_eq!(d.label, "ripemd160_internals_to_precompile_result_v1_stub");
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

    /// Build a single round-group-0 (f_0 = XOR3) honest row with the
    /// given B, C, D 32-bit words. Used by algebraic-round tests.
    fn honest_f0_row(b: u32, c: u32, d: u32, is_left_pass: bool) -> Ripemd160InternalsRow {
        let f = b ^ c ^ d;
        let mut state = [0u8; STATE_LENGTH];
        state[COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&b.to_le_bytes());
        state[COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&c.to_le_bytes());
        state[COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&d.to_le_bytes());
        // Sample A and E words from the state too (default zero).
        let a_word = u32::from_le_bytes([
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 0],
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 1],
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 2],
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 3],
        ]);
        let e_word = u32::from_le_bytes([
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 0],
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 1],
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 2],
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 3],
        ]);
        let k = if is_left_pass { K_L[0] } else { K_R[0] };
        let rol = if is_left_pass {
            ROL_REPRESENTATIVE_L[0]
        } else {
            ROL_REPRESENTATIVE_R[0]
        };
        // Compute honest intermediate sum + rotation + final T.
        let sum_full = (a_word as u64) + (f as u64) + 0u64 /* W=0 */ + (k as u64);
        let intermediate_sum_lo = (sum_full & 0xFFFF_FFFF) as u32;
        let intermediate_sum_carry = (sum_full >> 32) as u8;
        let rotated_word = intermediate_sum_lo.rotate_left(rol as u32);
        let t_full = (rotated_word as u64) + (e_word as u64);
        let t_lo = (t_full & 0xFFFF_FFFF) as u32;
        let t_add_carry = (t_full >> 32) as u8;
        let t_word = t_lo.to_le_bytes();
        let mut next_state = [0u8; STATE_LENGTH];
        // A_next ← state.E
        next_state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(
                &state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET
                    ..COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 4],
            );
        // B_next ← t_word
        next_state[COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&t_word);
        // C_next ← state.B
        next_state[COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(
                &state[COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET
                    ..COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET + 4],
            );
        // D_next ← state.C
        next_state[COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(
                &state[COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET
                    ..COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET + 4],
            );
        // E_next ← state.D
        next_state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(
                &state[COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET
                    ..COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET + 4],
            );
        Ripemd160InternalsRow {
            state,
            next_state,
            message_word: [0u8; MESSAGE_WORD_LENGTH],
            constant_k: k.to_le_bytes(),
            result: [0u8; RESULT_LENGTH],
            round_index: 0,
            is_left_pass,
            is_result_row: false,
            f_result: f.to_le_bytes(),
            round_group: 0,
            t_word,
            rol_amount: rol,
            intermediate_sum_lo,
            intermediate_sum_carry,
            rotated_word,
            t_add_carry,
        }
    }

    #[test]
    fn ripemd160_internals_air_f0_xor3_honest_left_pass() {
        // f_0(B,C,D) = B ⊕ C ⊕ D over a varied non-zero bit pattern.
        let row = honest_f0_row(0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE, true);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_internals_air_f0_xor3_honest_right_pass() {
        // Right-pass: constant_k must equal K_R[0] = 0x50A28BE6.
        let row = honest_f0_row(0x0000_0000, 0xFFFF_FFFF, 0xAAAA_5555, false);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_internals_air_tampered_f_result_xor3_detected() {
        // Flip one f_result bit — the gated XOR3 row-local constraint must fire.
        let mut row = honest_f0_row(0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE, true);
        let mut f_word = u32::from_le_bytes(row.f_result);
        f_word ^= 1u32 << 7; // flip bit 7
        row.f_result = f_word.to_le_bytes();
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Constraint index of XOR3 for bit 7 = 16 + 7 = 23.
        assert!(
            !bodies[16 + 7][0].is_zero(),
            "expected f_0 XOR3 constraint for bit 7 to fire",
        );
    }

    #[test]
    fn ripemd160_internals_air_tampered_constant_k_pin_detected() {
        // Tamper constant_k byte 0 on a left-pass round-group-0 row.
        // The 48..52 constant_k pin must fire.
        let mut row = honest_f0_row(0, 0, 0, true);
        row.constant_k[0] = 0x42;
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert!(
            !bodies[48][0].is_zero(),
            "expected constant_k byte 0 pin to fire on tampered K",
        );
    }

    #[test]
    fn ripemd160_internals_air_round_group_sum_one_hot_pinned() {
        // Tamper the one-hot: set is_round_group[0]=1 AND
        // is_round_group[1]=1 simultaneously. Constraint 6
        // (sum == is_real) must fire.
        let row = honest_f0_row(0, 0, 0, true);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_ROUND_GROUP_OFFSET + 1][0] =
            Scalar::one(CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 6 = sum_eq_is_real.
        assert!(
            !bodies[6][0].is_zero(),
            "expected round_group sum to fire when two selectors are hot",
        );
    }

    #[test]
    fn ripemd160_internals_air_word_bit_decomp_tampering_detected() {
        // Flip a B bit but leave the B byte unchanged — bit-decomp must fire.
        let row = honest_f0_row(0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE, true);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Flip B bit 3 from its honest value.
        let cur = cols[COL_B_BITS_OFFSET + 3][0].clone();
        let one = Scalar::one(CurveType::Bls48581);
        cols[COL_B_BITS_OFFSET + 3][0] = one.sub(&cur);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 12 = b_word_bit_decomp.
        assert!(
            !bodies[12][0].is_zero(),
            "expected b-word bit-decomp constraint to fire",
        );
    }

    #[test]
    fn ripemd160_internals_air_constants_pinned_values() {
        // Sanity: K_L[0] = 0, K_R[0] = 0x50A28BE6 per RIPEMD-160 spec.
        assert_eq!(K_L[0], 0x00000000);
        assert_eq!(K_L[1], 0x5A827999);
        assert_eq!(K_L[2], 0x6ED9EBA1);
        assert_eq!(K_L[3], 0x8F1BBCDC);
        assert_eq!(K_L[4], 0xA953FD4E);
        assert_eq!(K_R[0], 0x50A28BE6);
        assert_eq!(K_R[1], 0x5C4DD124);
        assert_eq!(K_R[2], 0x6D703EF3);
        assert_eq!(K_R[3], 0x7A6D76E9);
        assert_eq!(K_R[4], 0x00000000);
        assert_eq!(NUM_ROUND_GROUPS, 5);
    }

    #[test]
    fn ripemd160_internals_air_evaluate_at_point_matches_on_domain() {
        // Cross-check evaluate_at_point against the on-domain bodies
        // at every row, so the verifier-side closed-form stays in
        // lockstep with the row-by-row evaluator.
        let row = honest_f0_row(0x1234_5678, 0x9ABC_DEF0, 0x0F0F_0F0F, true);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Random alpha; aggregate row 0 with α-powers and compare to
        // evaluate_at_point.
        let alpha = Scalar::from_u64(7, CurveType::Bls48581);
        let mut expected = Scalar::zero(CurveType::Bls48581);
        let mut ap = Scalar::one(CurveType::Bls48581);
        for body in bodies.iter() {
            expected = expected.add(&ap.mul(&body[0]));
            ap = ap.mul(&alpha);
        }
        let col_evals_at_0: Vec<Scalar> =
            trace.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let got = cs.evaluate_at_point(&col_evals_at_0, &alpha);
        assert!(
            got.sub(&expected).is_zero(),
            "evaluate_at_point disagrees with evaluate_on_domain aggregate",
        );
    }

    #[test]
    fn ripemd160_internals_air_column_layout_pinned() {
        assert_eq!(COL_STATE_OFFSET, 0);
        assert_eq!(COL_NEXT_STATE_OFFSET, 20);
        assert_eq!(COL_MESSAGE_WORD_OFFSET, 40);
        assert_eq!(COL_CONSTANT_K_OFFSET, 44);
        assert_eq!(COL_RESULT_OFFSET, 48);
        assert_eq!(COL_ROUND_INDEX, 68);
        assert_eq!(COL_ROUND_INDEX_BYTE_OFFSET, 69);
        assert_eq!(COL_IS_LEFT_PASS, 73);
        assert_eq!(COL_IS_RESULT_ROW, 74);
        assert_eq!(COL_IS_REAL, 75);
        assert_eq!(COL_F_RESULT_OFFSET, 76);
        assert_eq!(COL_B_BITS_OFFSET, 80);
        assert_eq!(COL_C_BITS_OFFSET, 112);
        assert_eq!(COL_D_BITS_OFFSET, 144);
        assert_eq!(COL_F_BITS_OFFSET, 176);
        assert_eq!(COL_IS_ROUND_GROUP_OFFSET, 208);
        assert_eq!(COL_ROL_AMOUNT, 213);
        assert_eq!(COL_T_WORD_BYTE_OFFSET, 214);
        assert_eq!(COL_T_WORD_BITS_OFFSET, 218);
        assert_eq!(COL_INTERMEDIATE_SUM_BITS_OFFSET, 250);
        assert_eq!(COL_INTERMEDIATE_SUM_CARRY_BITS_OFFSET, 282);
        assert_eq!(COL_ROTATED_WORD_BITS_OFFSET, 284);
        assert_eq!(COL_T_ADD_CARRY_BIT, 316);
        assert_eq!(COL_MIRROR_BYTE0, 317);
        assert_eq!(NUM_COLUMNS, 318);
        assert_eq!(NUM_ROW_CONSTRAINTS, 239);
        assert_eq!(NUM_SHIFTED, 1);
        assert_eq!(ROUNDS_PER_PASS, 80);
        assert_eq!(PC_RIPEMD160, 0x03);

        let cs = Ripemd160InternalsConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert_eq!(cs.shifted_column_indices().len(), STATE_LENGTH + 2);
    }

    /// Build an honest row in a non-zero round group `j` with the
    /// specified B, C, D and let the helper compute the honest f_j(B,C,D).
    fn honest_row_in_group(
        j: usize,
        b: u32,
        c: u32,
        d: u32,
        is_left_pass: bool,
    ) -> Ripemd160InternalsRow {
        assert!(j < NUM_ROUND_GROUPS);
        // Compute f_j honestly at the 32-bit level.
        let f: u32 = match j {
            0 => b ^ c ^ d,
            1 => (b & c) | (!b & d),
            2 => (b | !c) ^ d,
            3 => (b & d) | (c & !d),
            4 => b ^ (c | !d),
            _ => unreachable!(),
        };
        let mut state = [0u8; STATE_LENGTH];
        state[COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&b.to_le_bytes());
        state[COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&c.to_le_bytes());
        state[COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&d.to_le_bytes());
        let k = if is_left_pass { K_L[j] } else { K_R[j] };
        let rol = if is_left_pass {
            ROL_REPRESENTATIVE_L[j]
        } else {
            ROL_REPRESENTATIVE_R[j]
        };
        // Sample A, E words.
        let a_word = u32::from_le_bytes([
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 0],
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 1],
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 2],
            state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET + 3],
        ]);
        let e_word = u32::from_le_bytes([
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 0],
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 1],
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 2],
            state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET + 3],
        ]);
        let sum_full = (a_word as u64) + (f as u64) + 0u64 + (k as u64);
        let intermediate_sum_lo = (sum_full & 0xFFFF_FFFF) as u32;
        let intermediate_sum_carry = (sum_full >> 32) as u8;
        let rotated_word = intermediate_sum_lo.rotate_left(rol as u32);
        let t_full = (rotated_word as u64) + (e_word as u64);
        let t_lo = (t_full & 0xFFFF_FFFF) as u32;
        let t_add_carry = (t_full >> 32) as u8;
        let t_word = t_lo.to_le_bytes();
        let mut next_state = [0u8; STATE_LENGTH];
        // Build honest cycling next_state.
        for b_i in 0..4 {
            next_state[(COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET) + b_i] =
                state[(COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET) + b_i];
            next_state[(COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET) + b_i] =
                t_word[b_i];
            next_state[(COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET) + b_i] =
                state[(COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET) + b_i];
            next_state[(COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET) + b_i] =
                state[(COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET) + b_i];
            next_state[(COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET) + b_i] =
                state[(COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET) + b_i];
        }
        Ripemd160InternalsRow {
            state,
            next_state,
            message_word: [0u8; MESSAGE_WORD_LENGTH],
            constant_k: k.to_le_bytes(),
            result: [0u8; RESULT_LENGTH],
            round_index: 0,
            is_left_pass,
            is_result_row: false,
            f_result: f.to_le_bytes(),
            round_group: j as u8,
            t_word,
            rol_amount: rol,
            intermediate_sum_lo,
            intermediate_sum_carry,
            rotated_word,
            t_add_carry,
        }
    }

    #[test]
    fn ripemd160_internals_air_f1_honest_left_and_right() {
        for &is_left in &[true, false] {
            let row = honest_row_in_group(
                1, 0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE, is_left,
            );
            let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
            let trace = build_trace_polynomials(&w, CurveType::Bls48581);
            let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
            let cr: Vec<&Vec<Scalar>> =
                trace.columns.iter().map(|p| &p.evaluations).collect();
            assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
        }
    }

    #[test]
    fn ripemd160_internals_air_f2_honest() {
        let row = honest_row_in_group(2, 0x0F0F_0F0F, 0xAAAA_5555, 0x1234_5678, true);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_internals_air_f3_honest() {
        let row = honest_row_in_group(3, 0x1234_5678, 0x9ABC_DEF0, 0xCAFE_BABE, false);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_internals_air_f4_honest() {
        let row = honest_row_in_group(4, 0xDEAD_BEEF, 0x0FF0_0FF0, 0xA5A5_A5A5, true);
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_internals_air_tampered_f1_detected() {
        let mut row = honest_row_in_group(1, 0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE, true);
        // Flip a bit in f_result; the f1 row-local constraint for that bit
        // (gated by is_round_group[1]) must fire.
        let mut f_word = u32::from_le_bytes(row.f_result);
        f_word ^= 1u32 << 5; // flip bit 5
        row.f_result = f_word.to_le_bytes();
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // f1 starts at constraint index 52; bit 5 = 52 + 5 = 57.
        assert!(
            !bodies[52 + 5][0].is_zero(),
            "expected f_1 constraint for bit 5 to fire on tampered f_result",
        );
    }

    #[test]
    fn ripemd160_internals_air_tampered_f2_detected() {
        let mut row = honest_row_in_group(2, 0x0F0F_0F0F, 0xAAAA_5555, 0x1234_5678, true);
        let mut f_word = u32::from_le_bytes(row.f_result);
        f_word ^= 1u32 << 13;
        row.f_result = f_word.to_le_bytes();
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // f2 starts at 84; bit 13 → 84 + 13 = 97.
        assert!(!bodies[84 + 13][0].is_zero(),
            "expected f_2 constraint for bit 13 to fire");
    }

    #[test]
    fn ripemd160_internals_air_tampered_f3_detected() {
        let mut row = honest_row_in_group(3, 0x1234_5678, 0x9ABC_DEF0, 0xCAFE_BABE, false);
        let mut f_word = u32::from_le_bytes(row.f_result);
        f_word ^= 1u32 << 21;
        row.f_result = f_word.to_le_bytes();
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert!(!bodies[116 + 21][0].is_zero(),
            "expected f_3 constraint for bit 21 to fire");
    }

    #[test]
    fn ripemd160_internals_air_tampered_f4_detected() {
        let mut row = honest_row_in_group(4, 0xDEAD_BEEF, 0x0FF0_0FF0, 0xA5A5_A5A5, true);
        let mut f_word = u32::from_le_bytes(row.f_result);
        f_word ^= 1u32 << 31;
        row.f_result = f_word.to_le_bytes();
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert!(!bodies[148 + 31][0].is_zero(),
            "expected f_4 constraint for bit 31 to fire");
    }

    #[test]
    fn ripemd160_internals_air_rol_amount_tamper_detected() {
        // Honest row in group 1 → ROL_REPRESENTATIVE_L[1] = 7. Tamper to 11.
        let mut row = honest_row_in_group(1, 0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE, true);
        row.rol_amount = 11;
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Constraint 181 = rol_amount pin.
        assert!(!bodies[181][0].is_zero(),
            "expected rol_amount pin to fire when tampered");
    }

    #[test]
    fn ripemd160_internals_air_state_cycling_a_next_tamper_detected() {
        // Honest row: A_next = state.E. Tamper next_state byte A0.
        let mut row = honest_row_in_group(0, 0x1111_1111, 0x2222_2222, 0x3333_3333, true);
        // Force state.E to a known non-zero value to make tampering meaningful.
        row.state[COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET] = 0xAB;
        row.next_state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET] = 0xAB;
        // Now tamper: change next_state.A0 to a wrong value.
        row.next_state[COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET] = 0x42;
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Constraint 182 = state-cycling A_next byte 0.
        assert!(!bodies[182][0].is_zero(),
            "expected state-cycling A_next byte 0 to fire");
    }

    #[test]
    fn ripemd160_internals_air_state_cycling_b_next_tamper_detected() {
        // B_next = t_word. Tamper next_state.B0 to break the binding.
        let mut row = honest_row_in_group(0, 0, 0, 0, true);
        row.t_word[0] = 0x77;
        // With honest binding, next_state.B[0] should equal 0x77.
        row.next_state[COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET] = 0x77;
        // Tamper.
        row.next_state[COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET] = 0x00;
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Constraint 186 = state-cycling B_next byte 0.
        assert!(!bodies[186][0].is_zero(),
            "expected state-cycling B_next byte 0 to fire");
    }

    /// Helper: build a fully-populated honest row with chosen A, W, E,
    /// (B, C, D) for a given round group and pass.
    fn honest_row_full(
        j: usize,
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        e: u32,
        w: u32,
        is_left_pass: bool,
    ) -> Ripemd160InternalsRow {
        assert!(j < NUM_ROUND_GROUPS);
        let f: u32 = match j {
            0 => b ^ c ^ d,
            1 => (b & c) | (!b & d),
            2 => (b | !c) ^ d,
            3 => (b & d) | (c & !d),
            4 => b ^ (c | !d),
            _ => unreachable!(),
        };
        let mut state = [0u8; STATE_LENGTH];
        let copy_word = |state: &mut [u8; STATE_LENGTH], off: usize, word: u32| {
            state[(off - COL_STATE_OFFSET)..(off - COL_STATE_OFFSET) + 4]
                .copy_from_slice(&word.to_le_bytes());
        };
        copy_word(&mut state, COL_STATE_A_BYTE_OFFSET, a);
        copy_word(&mut state, COL_STATE_B_BYTE_OFFSET, b);
        copy_word(&mut state, COL_STATE_C_BYTE_OFFSET, c);
        copy_word(&mut state, COL_STATE_D_BYTE_OFFSET, d);
        copy_word(&mut state, COL_STATE_E_BYTE_OFFSET, e);
        let k = if is_left_pass { K_L[j] } else { K_R[j] };
        let rol = if is_left_pass {
            ROL_REPRESENTATIVE_L[j]
        } else {
            ROL_REPRESENTATIVE_R[j]
        };
        let sum_full =
            (a as u64) + (f as u64) + (w as u64) + (k as u64);
        let intermediate_sum_lo = (sum_full & 0xFFFF_FFFF) as u32;
        let intermediate_sum_carry = (sum_full >> 32) as u8;
        let rotated_word = intermediate_sum_lo.rotate_left(rol as u32);
        let t_full = (rotated_word as u64) + (e as u64);
        let t_lo = (t_full & 0xFFFF_FFFF) as u32;
        let t_add_carry = (t_full >> 32) as u8;
        let t_word = t_lo.to_le_bytes();
        let mut next_state = [0u8; STATE_LENGTH];
        // A_next = state.E
        next_state[(COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET)
            ..(COL_STATE_A_BYTE_OFFSET - COL_STATE_OFFSET) + 4]
            .copy_from_slice(&e.to_le_bytes());
        // B_next = t_word
        next_state[(COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET)
            ..(COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET) + 4]
            .copy_from_slice(&t_word);
        // C_next = state.B (no ROL_10 — deferred shape only)
        next_state[(COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET)
            ..(COL_STATE_C_BYTE_OFFSET - COL_STATE_OFFSET) + 4]
            .copy_from_slice(&b.to_le_bytes());
        // D_next = state.C
        next_state[(COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET)
            ..(COL_STATE_D_BYTE_OFFSET - COL_STATE_OFFSET) + 4]
            .copy_from_slice(&c.to_le_bytes());
        // E_next = state.D
        next_state[(COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET)
            ..(COL_STATE_E_BYTE_OFFSET - COL_STATE_OFFSET) + 4]
            .copy_from_slice(&d.to_le_bytes());
        Ripemd160InternalsRow {
            state,
            next_state,
            message_word: w.to_le_bytes(),
            constant_k: k.to_le_bytes(),
            result: [0u8; RESULT_LENGTH],
            round_index: 0,
            is_left_pass,
            is_result_row: false,
            f_result: f.to_le_bytes(),
            round_group: j as u8,
            t_word,
            rol_amount: rol,
            intermediate_sum_lo,
            intermediate_sum_carry,
            rotated_word,
            t_add_carry,
        }
    }

    #[test]
    fn ripemd160_internals_air_t_full_honest_all_groups() {
        // Validate the full T = ROL_s(A + f + W + K) + E chain honestly
        // for both passes across all 5 round groups.
        let a = 0x0123_4567u32;
        let b = 0xDEAD_BEEFu32;
        let c = 0x1234_5678u32;
        let d = 0xCAFE_BABEu32;
        let e = 0xFEDC_BA98u32;
        let w = 0xAA55_AA55u32;
        for j in 0..NUM_ROUND_GROUPS {
            for &is_left in &[true, false] {
                let row = honest_row_full(j, a, b, c, d, e, w, is_left);
                let witness =
                    Ripemd160InternalsTraceWitness::from_rows(vec![row]);
                let trace =
                    build_trace_polynomials(&witness, CurveType::Bls48581);
                let cs =
                    Ripemd160InternalsConstraintSystem::new(trace.num_rows);
                let cr: Vec<&Vec<Scalar>> =
                    trace.columns.iter().map(|p| &p.evaluations).collect();
                assert_all_zero(
                    &cs.evaluate_on_domain(&cr, trace.num_rows),
                );
            }
        }
    }

    #[test]
    fn ripemd160_internals_air_tampered_intermediate_sum_bit_detected() {
        // Flip an intermediate_sum_bits column entry; the
        // intermediate_sum bit-decomp (constraint 202) AND the rotated_word
        // ROL_s binding (which reads from intermediate_sum_bits) must fire.
        let row = honest_row_full(
            0, 0x0123_4567, 0xDEAD_BEEF, 0x1234_5678,
            0xCAFE_BABE, 0xFEDC_BA98, 0xAA55_AA55, true,
        );
        let witness = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&witness, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let one = Scalar::one(CurveType::Bls48581);
        let cur = cols[COL_INTERMEDIATE_SUM_BITS_OFFSET + 11][0].clone();
        cols[COL_INTERMEDIATE_SUM_BITS_OFFSET + 11][0] = one.sub(&cur);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 202 = intermediate_sum_bit_decomp.
        assert!(!bodies[202][0].is_zero(),
            "expected intermediate_sum bit-decomp to fire on tampered bit");
    }

    #[test]
    fn ripemd160_internals_air_tampered_rotated_word_bit_detected() {
        // Flip a rotated_word_bits column entry — the per-bit ROL_s
        // binding (constraints 205..237) MUST fire.
        let row = honest_row_full(
            1, 0x0123_4567, 0xDEAD_BEEF, 0x1234_5678,
            0xCAFE_BABE, 0xFEDC_BA98, 0xAA55_AA55, true,
        );
        let witness = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&witness, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let one = Scalar::one(CurveType::Bls48581);
        let cur = cols[COL_ROTATED_WORD_BITS_OFFSET + 3][0].clone();
        cols[COL_ROTATED_WORD_BITS_OFFSET + 3][0] = one.sub(&cur);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // ROL_s binding for bit 3 = constraint 205 + 3 = 208.
        assert!(!bodies[205 + 3][0].is_zero(),
            "expected ROL_s binding for bit 3 to fire on tampered rotated bit");
    }

    #[test]
    fn ripemd160_internals_air_tampered_t_word_detected_in_add() {
        // Tamper t_word — the T add carry equation (constraint 237) fires.
        let mut row = honest_row_full(
            2, 0x0123_4567, 0xDEAD_BEEF, 0x1234_5678,
            0xCAFE_BABE, 0xFEDC_BA98, 0xAA55_AA55, true,
        );
        // Bump t_word by 1.
        let mut tw = u32::from_le_bytes(row.t_word);
        tw = tw.wrapping_add(1);
        row.t_word = tw.to_le_bytes();
        // Fix next_state.B = t_word to keep state-cycling honest.
        row.next_state[COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET
            ..COL_STATE_B_BYTE_OFFSET - COL_STATE_OFFSET + 4]
            .copy_from_slice(&row.t_word);
        let witness = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&witness, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Constraint 237 = t_add_carry_word_relation.
        assert!(!bodies[237][0].is_zero(),
            "expected t-add equation to fire on tampered t_word");
    }

    #[test]
    fn ripemd160_internals_air_tampered_t_add_carry_binary_detected() {
        // Tamper t_add_carry to 2 (not binary) — constraint 238 fires.
        let row = honest_row_full(
            0, 0x0123_4567, 0xDEAD_BEEF, 0x1234_5678,
            0xCAFE_BABE, 0xFEDC_BA98, 0xAA55_AA55, true,
        );
        let witness = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&witness, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_T_ADD_CARRY_BIT][0] =
            Scalar::from_u64(2, CurveType::Bls48581);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!bodies[238][0].is_zero(),
            "expected t_add_carry binary constraint to fire");
    }

    #[test]
    fn ripemd160_internals_air_t_word_bit_decomp_tamper_detected() {
        let mut row = honest_row_in_group(1, 0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE, true);
        // Set t_word to a varied value (honest binding is built later in
        // the trace builder when it re-decomps; we'll tamper by changing
        // one bit column directly).
        row.t_word = 0x12345678u32.to_le_bytes();
        let w = Ripemd160InternalsTraceWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Flip bit 7 of t_word.
        let cur = cols[COL_T_WORD_BITS_OFFSET + 7][0].clone();
        let one = Scalar::one(CurveType::Bls48581);
        cols[COL_T_WORD_BITS_OFFSET + 7][0] = one.sub(&cur);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 180 = t_word_bit_decomp.
        assert!(!bodies[180][0].is_zero(),
            "expected t_word bit-decomp to fire when bit is tampered");
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

        let w = Ripemd160InternalsTraceWitness::from_digest_skeleton(b"abc");
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = Ripemd160InternalsConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone ripemd160_internals_air proof must verify",
        );
    }
}
