//! BLAKE2 F compression internals AIR (skeleton).
//!
//! # Purpose
//!
//! Bridges the [`crate::blake2f_precompile_air`] (which commits the
//! single-row witness `h_out = F(h_in, m, t, f, rounds)` taken on faith)
//! to the actual algebraic round-by-round computation of BLAKE2b's F
//! function per EIP-152.
//!
//! # Algorithm shape (BLAKE2b F)
//!
//! ```text
//!   v[0..8]  = h_in[0..8]
//!   v[8..16] = IV[0..8]
//!   v[12]   ^= t[0]
//!   v[13]   ^= t[1]
//!   if f: v[14] ^= 0xFFFFFFFFFFFFFFFF
//!   for r in 0..rounds:                  # 12 in mainline BLAKE2b
//!     G(v, ...) applied 8 times using σ[r % 10] message schedule
//!   for i in 0..8:
//!     h_out[i] = h_in[i] ^ v[i] ^ v[i+8]
//! ```
//!
//! The original BLAKE2b specifies exactly **12** rounds, but EIP-152
//! takes a `rounds` parameter (`u32 BE`) — implementations support a
//! variable-rounds variant. This AIR scaffolds the **12-round** mainline
//! shape; deviations are exposed by the `rounds` column for the
//! downstream gadget to bind algebraically.
//!
//! # AIR shape (this skeleton)
//!
//! One TRACE ROW per round. Each row commits:
//!
//!   * v[16 × 8 bytes = 128 bytes]   — internal state at round start.
//!   * next_v[16 × 8 bytes = 128 bytes] — internal state at round end.
//!   * m[16 × 8 bytes = 128 bytes]   — full message block (constant
//!     across all rows of one invocation).
//!   * t[2 × 8 bytes = 16 bytes]     — offset counter.
//!   * f_byte ∈ {0, 1}               — finalization flag.
//!   * round_index ∈ [0, MAINLINE_ROUNDS]  — round counter.
//!   * result h[8 × 8 bytes = 64 bytes] — final 8-limb output, exposed
//!     only on the FINAL row (`is_result_row = 1`).
//!   * is_real, is_result_row ∈ {0, 1}.
//!
//! # Soundness
//!
//! This AIR is a **skeleton**. Algebraic G-mixing / message schedule /
//! XOR finalization are **deferred**. What this AIR DOES bind:
//!
//!   - `is_real`, `is_result_row`, `f_byte` binary.
//!   - `result[i]` zero on non-final rows (gated).
//!   - LE byte decomposition of `round_index_word`.
//!   - Cross-row state continuity: when `is_real(X) * is_real(ω·X) = 1`
//!     and `is_result_row(X) = 0`, witness `next_v(X)` = `v(ω·X)`.
//!
//! What this AIR does NOT bind (deferred):
//!
//!   - Algebraic G-function `next_v = G(v, m, σ[r])`.
//!   - IV pin / counter XOR / final flag XOR for initial `v`.
//!   - XOR finalization `h_out[i] = h_in[i] ^ v[i] ^ v[i+8]`.
//!   - Message-schedule constancy across the rounds of one invocation.
//!
//! # Cross-AIR LogUp descriptors
//!
//! [`make_blake2_f_internals_to_precompile_descriptor`] binds the
//! finalized 64-byte `result_h` on the final row to the
//! `h_out` column of [`crate::blake2f_precompile_air`].

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of `v` 64-bit limbs.
pub const V_LIMBS: usize = 16;
/// Number of `m` 64-bit limbs.
pub const M_LIMBS: usize = 16;
/// Number of `t` 64-bit limbs.
pub const T_LIMBS: usize = 2;
/// Bytes per 64-bit limb.
pub const LIMB_BYTES: usize = 8;
/// Total V byte length.
pub const V_LENGTH: usize = V_LIMBS * LIMB_BYTES; // 128
/// Total M byte length.
pub const M_LENGTH: usize = M_LIMBS * LIMB_BYTES; // 128
/// Total T byte length.
pub const T_LENGTH: usize = T_LIMBS * LIMB_BYTES; // 16
/// Number of h output 64-bit limbs.
pub const H_LIMBS: usize = 8;
/// Result h byte length.
pub const RESULT_LENGTH: usize = H_LIMBS * LIMB_BYTES; // 64
/// Mainline BLAKE2b round count.
pub const MAINLINE_ROUNDS: usize = 12;
/// LE byte decomposition width for `round_index_word` (u32).
pub const ROUND_INDEX_BYTES: usize = 4;

/// EIP precompile id for BLAKE2F (mirrors precompile AIR).
pub const PC_BLAKE2F: u64 = 0x09;

/// BLAKE2b IV (RFC 7693 §2.6 / SHA-512 initial hash values).
pub const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// f-byte mask (all-ones u64 when finalization flag is set; zero otherwise).
pub const F_BYTE_MASK_ONES: u64 = 0xFFFFFFFFFFFFFFFFu64;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_V_OFFSET: usize = 0; // 0..128
pub const COL_NEXT_V_OFFSET: usize = COL_V_OFFSET + V_LENGTH; // 128..256
pub const COL_M_OFFSET: usize = COL_NEXT_V_OFFSET + V_LENGTH; // 256..384
pub const COL_T_OFFSET: usize = COL_M_OFFSET + M_LENGTH; // 384..400
pub const COL_F_BYTE: usize = COL_T_OFFSET + T_LENGTH; // 400
pub const COL_ROUND_INDEX: usize = COL_F_BYTE + 1; // 401
pub const COL_ROUND_INDEX_BYTE_OFFSET: usize = COL_ROUND_INDEX + 1; // 402..406
pub const COL_RESULT_OFFSET: usize =
    COL_ROUND_INDEX_BYTE_OFFSET + ROUND_INDEX_BYTES; // 406..470
pub const COL_IS_RESULT_ROW: usize = COL_RESULT_OFFSET + RESULT_LENGTH; // 470
pub const COL_IS_REAL: usize = COL_IS_RESULT_ROW + 1; // 471

// ─── G-mixing intermediate witness columns ────────────────────────────
//
// Per row (one BLAKE2b round), 8 G calls fire. Each G call mixes four
// 64-bit words `v[a], v[b], v[c], v[d]` with two message limbs
// `m[i], m[j]`. We witness ALL FOUR intermediates after each of the
// 8 internal steps of G as a packed 64-bit scalar, plus four carry
// witnesses (one per add) so the additive layer can be algebraically
// bound via `prev + other [+ msg] = next + carry · 2^64`.
//
// Step-by-step intermediates per G (each is a packed u64 scalar):
//
//   * `v_a1`  — v[a] after step 1 (3-way add: v_a + v_b + m[i])
//   * `v_d1`  — v[d] after step 2 (ROTR_32(v_d XOR v_a1)) — XOR/ROTR
//                  algebraically DEFERRED; witnessed as free u64.
//   * `v_c1`  — v[c] after step 3 (2-way add: v_c + v_d1)
//   * `v_b1`  — v[b] after step 4 (ROTR_24(v_b XOR v_c1)) — DEFERRED.
//   * `v_a2`  — v[a] after step 5 (3-way add: v_a1 + v_b1 + m[j])
//   * `v_d2`  — v[d] after step 6 (ROTR_16(v_d1 XOR v_a2)) — DEFERRED.
//   * `v_c2`  — v[c] after step 7 (2-way add: v_c1 + v_d2)
//   * `v_b2`  — v[b] after step 8 (ROTR_63(v_b1 XOR v_c2)) — DEFERRED.
//
// Per-G carry columns: `c1, c2, c3, c4` (one per add). 3-way adds
// (c1, c3) admit values in {0, 1, 2}; 2-way adds (c2, c4) admit {0, 1}.
//
// Per row we also commit 16 message-schedule columns `m_sched[0..16]`
// (packed u64), the σ-permuted view of the message used by G. Algebraic
// binding of `m_sched` to the on-row `m` bytes via the SIGMA table is
// **DEFERRED**: a future gadget AIR will close that loop. Until then
// `m_sched` is a free-witness column constrained only by the add layer.
pub const G_PER_ROUND: usize = 8;
pub const G_INTERMEDIATES_PER_G: usize = 8; // v_a1,v_d1,v_c1,v_b1,v_a2,v_d2,v_c2,v_b2
pub const G_CARRIES_PER_G: usize = 4;
pub const M_SCHED_LIMBS: usize = M_LIMBS;

pub const COL_G_INTERMEDIATE_OFFSET: usize = COL_IS_REAL + 1; // 472
pub const G_INTERMEDIATE_COL_COUNT: usize =
    G_PER_ROUND * G_INTERMEDIATES_PER_G; // 64
pub const COL_G_CARRY_OFFSET: usize =
    COL_G_INTERMEDIATE_OFFSET + G_INTERMEDIATE_COL_COUNT; // 536
pub const G_CARRY_COL_COUNT: usize = G_PER_ROUND * G_CARRIES_PER_G; // 32
pub const COL_M_SCHED_OFFSET: usize =
    COL_G_CARRY_OFFSET + G_CARRY_COL_COUNT; // 568

// ─── Per-bit decomposition columns for ALL 32 G×step XOR/ROTR instances
// ─────────────────────────────────────────────────────────────────────
//
// Task #181 algebraically pinned ONE representative XOR + ROTR step end-to-end
// (G0 step 2). Task #194 mechanically replicates the construction for all
// 8 G calls × 4 XOR/ROTR steps = 32 instances, covering every deferred
// XOR/ROTR step in one BLAKE2b round.
//
// Each instance adds 3 × 64 = 192 bit columns that decompose the THREE
// 64-bit limbs participating in `dst = ROTR_n(B XOR A)`:
//
//   * `A`   — the second-operand (just-computed) 64-bit value; always read
//             from an intermediate column.
//   * `B`   — the first-operand (input/forwarded) 64-bit value; sourced
//             either from the on-row V bytes (column-phase G0..G3 steps 2/4),
//             from `G_INPUT_FROM_INTERMEDIATE` (diagonal-phase G4..G7
//             steps 2/4), or from a same-G intermediate (steps 6/8).
//   * `dst` — the ROTR(B XOR A) output, always an intermediate column.
//
// Algebraic checks added per instance (all gated by `is_real * (1 - is_result_row)`):
//
//   * Each bit cell `b` satisfies `b · (b - 1) = 0`        (192 constraints).
//   * `Σ 2^i · bit_A[i]   == A`   (packed equality)        (1 constraint).
//   * `Σ 2^i · bit_B[i]   == B`   (packed equality)        (1 constraint).
//   * `Σ 2^i · bit_dst[i] == dst` (packed equality)        (1 constraint).
//   * For each i in 0..64, ROTR_n fused with XOR:
//        bit_dst[(i + n) mod 64]
//          - (bit_A[i] + bit_B[i] - 2 · bit_A[i] · bit_B[i]) = 0
//                                                          (64 constraints).
//
// Net total per instance: 192 + 3 + 64 = 259 new row-local constraints.
// Total across 32 instances: 32 × 259 = 8288 row-local constraints,
// 32 × 192 = 6144 new bit-decomposition columns.

/// Number of bit-decomposition limbs per XOR/ROTR instance (`A`, `B`, `dst`).
pub const XOR_ROTR_BIT_LIMBS_PER_INSTANCE: usize = 3;
/// Bits per 64-bit limb.
pub const XOR_ROTR_BITS_PER_LIMB: usize = 64;
/// Columns per XOR/ROTR instance.
pub const XOR_ROTR_BIT_COUNT_PER_INSTANCE: usize =
    XOR_ROTR_BIT_LIMBS_PER_INSTANCE * XOR_ROTR_BITS_PER_LIMB; // 192

/// 4 XOR/ROTR steps per G call (steps 2, 4, 6, 8).
pub const XOR_ROTR_STEPS_PER_G: usize = 4;
/// Total XOR/ROTR instances pinned in a single BLAKE2b round.
pub const XOR_ROTR_INSTANCE_COUNT: usize =
    G_PER_ROUND * XOR_ROTR_STEPS_PER_G; // 32

/// Where an operand for an XOR/ROTR instance is sourced from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum XorRotrOperand {
    /// Packed view of `V_OFFSET + v_idx*8 .. + 8` bytes (column-phase input).
    InputVBytes(usize),
    /// `col_g_intermediate(g_idx, slot)` of a (possibly earlier-G) intermediate.
    Intermediate { g_idx: usize, slot: usize },
}

/// One XOR/ROTR instance: `dst = ROTR_rot(B XOR A)`. Per-row witnesses are
/// the 3 × 64-bit-decompositions of A, B, dst, with the (B XOR A) bits and
/// the ROTR shift fused into a single row-local constraint per bit.
#[derive(Copy, Clone, Debug)]
pub struct XorRotrInstance {
    /// Owning G index (0..8).
    pub g_idx: usize,
    /// Step within the G call (one of 2, 4, 6, 8).
    pub step: usize,
    /// Right-rotation amount (one of 32, 24, 16, 63).
    pub rot: usize,
    /// Operand A — always a same-G intermediate slot.
    pub a: XorRotrOperand,
    /// Operand B — V-input bytes (G0..G3 steps 2/4), diagonal-phase
    /// intermediate (G4..G7 steps 2/4), or same-G intermediate (steps 6/8).
    pub b: XorRotrOperand,
    /// Output dst — always a same-G intermediate slot.
    pub dst: XorRotrOperand,
}

/// Build the 32 XOR/ROTR instance descriptors for one BLAKE2b round.
///
/// Layout (column-major over g_idx, row-major over step):
///   index = 4*g_idx + step_idx, where step_idx ∈ {0,1,2,3} for steps {2,4,6,8}.
pub const fn make_xor_rotr_instances() -> [XorRotrInstance; XOR_ROTR_INSTANCE_COUNT] {
    let mut out = [XorRotrInstance {
        g_idx: 0,
        step: 0,
        rot: 0,
        a: XorRotrOperand::Intermediate { g_idx: 0, slot: 0 },
        b: XorRotrOperand::Intermediate { g_idx: 0, slot: 0 },
        dst: XorRotrOperand::Intermediate { g_idx: 0, slot: 0 },
    }; XOR_ROTR_INSTANCE_COUNT];
    let mut g_idx = 0;
    while g_idx < G_PER_ROUND {
        // Source for v_b (slot 1) and v_d (slot 3). Only steps 2 and 4
        // touch the original v_b/v_d; later steps fold same-G intermediates.
        // For G0..G3 this is the on-row V bytes at V_MIX_INDICES[g_idx][slot].
        // For G4..G7 this is the referenced earlier-G intermediate via
        // G_INPUT_FROM_INTERMEDIATE.
        let v_b_in: XorRotrOperand;
        let v_d_in: XorRotrOperand;
        if g_idx < 4 {
            v_b_in = XorRotrOperand::InputVBytes(V_MIX_INDICES[g_idx][1]);
            v_d_in = XorRotrOperand::InputVBytes(V_MIX_INDICES[g_idx][3]);
        } else {
            // Diagonal-phase: G_INPUT_FROM_INTERMEDIATE[g][slot] = (g_src, s).
            let spec = match G_INPUT_FROM_INTERMEDIATE[g_idx] {
                Some(s) => s,
                None => [(0, 0); 4],
            };
            v_b_in = XorRotrOperand::Intermediate {
                g_idx: spec[1].0, slot: spec[1].1,
            };
            v_d_in = XorRotrOperand::Intermediate {
                g_idx: spec[3].0, slot: spec[3].1,
            };
        }
        // Step 2: v_d1 = ROTR_32(v_d_in XOR v_a1)
        out[g_idx * XOR_ROTR_STEPS_PER_G + 0] = XorRotrInstance {
            g_idx,
            step: 2,
            rot: 32,
            a: XorRotrOperand::Intermediate { g_idx, slot: 0 }, // v_a1
            b: v_d_in,
            dst: XorRotrOperand::Intermediate { g_idx, slot: 1 }, // v_d1
        };
        // Step 4: v_b1 = ROTR_24(v_b_in XOR v_c1)
        out[g_idx * XOR_ROTR_STEPS_PER_G + 1] = XorRotrInstance {
            g_idx,
            step: 4,
            rot: 24,
            a: XorRotrOperand::Intermediate { g_idx, slot: 2 }, // v_c1
            b: v_b_in,
            dst: XorRotrOperand::Intermediate { g_idx, slot: 3 }, // v_b1
        };
        // Step 6: v_d2 = ROTR_16(v_d1 XOR v_a2)
        out[g_idx * XOR_ROTR_STEPS_PER_G + 2] = XorRotrInstance {
            g_idx,
            step: 6,
            rot: 16,
            a: XorRotrOperand::Intermediate { g_idx, slot: 4 }, // v_a2
            b: XorRotrOperand::Intermediate { g_idx, slot: 1 }, // v_d1
            dst: XorRotrOperand::Intermediate { g_idx, slot: 5 }, // v_d2
        };
        // Step 8: v_b2 = ROTR_63(v_b1 XOR v_c2)
        out[g_idx * XOR_ROTR_STEPS_PER_G + 3] = XorRotrInstance {
            g_idx,
            step: 8,
            rot: 63,
            a: XorRotrOperand::Intermediate { g_idx, slot: 6 }, // v_c2
            b: XorRotrOperand::Intermediate { g_idx, slot: 3 }, // v_b1
            dst: XorRotrOperand::Intermediate { g_idx, slot: 7 }, // v_b2
        };
        g_idx += 1;
    }
    out
}

pub const XOR_ROTR_INSTANCES: [XorRotrInstance; XOR_ROTR_INSTANCE_COUNT] =
    make_xor_rotr_instances();

/// First column of the XOR/ROTR bit block.
pub const COL_XOR_ROTR_BIT_OFFSET: usize =
    COL_M_SCHED_OFFSET + M_SCHED_LIMBS; // 584
/// Total columns added by the 32-instance bit-decomposition layer.
pub const XOR_ROTR_BIT_COL_COUNT: usize =
    XOR_ROTR_INSTANCE_COUNT * XOR_ROTR_BIT_COUNT_PER_INSTANCE; // 6144

/// Returns the base column of the `block_idx` ∈ {0=A, 1=B, 2=dst} bit
/// limb for the `inst_idx`-th instance.
#[inline]
pub fn xor_rotr_bit_base(inst_idx: usize, block_idx: usize) -> usize {
    debug_assert!(inst_idx < XOR_ROTR_INSTANCE_COUNT);
    debug_assert!(block_idx < XOR_ROTR_BIT_LIMBS_PER_INSTANCE);
    COL_XOR_ROTR_BIT_OFFSET
        + inst_idx * XOR_ROTR_BIT_COUNT_PER_INSTANCE
        + block_idx * XOR_ROTR_BITS_PER_LIMB
}

/// Returns column index of the `bit_idx`-th LSB-first bit of `block_idx`
/// ∈ {0=A, 1=B, 2=dst} for the `inst_idx`-th instance.
#[inline]
pub fn col_xor_rotr_bit(inst_idx: usize, block_idx: usize, bit_idx: usize) -> usize {
    debug_assert!(bit_idx < XOR_ROTR_BITS_PER_LIMB);
    xor_rotr_bit_base(inst_idx, block_idx) + bit_idx
}

/// Number of SIGMA rows (BLAKE2b mainline = 10 distinct rounds; rounds
/// 10 and 11 reuse rows 0 and 1 via `round_index mod 10`).
pub const SIGMA_ROWS: usize = 10;

/// `round_index = 10 * round_quotient_10 + round_mod_10`. Mainline
/// rounds 0..11 give a quotient ∈ {0, 1}; the FINAL row pins
/// `round_index = MAINLINE_ROUNDS = 12` so quotient may be 1 + mod 2.
/// We only constrain the quotient binary when the row is a real
/// G-mixing row (gated by `is_real * (1 - is_result_row)`).
pub const COL_ROUND_MOD_10: usize =
    COL_XOR_ROTR_BIT_OFFSET + XOR_ROTR_BIT_COL_COUNT; // 6728
pub const COL_ROUND_QUOTIENT_10: usize = COL_ROUND_MOD_10 + 1; // 6729

/// One-hot selectors: `is_sigma_row[r] = 1` iff `round_mod_10 = r`.
/// On result / padding rows the sum-to-one gate forces ALL ten to 0.
pub const COL_IS_SIGMA_ROW_OFFSET: usize =
    COL_ROUND_QUOTIENT_10 + 1; // 6730

/// Column index for `is_sigma_row[r]`.
#[inline]
pub fn col_is_sigma_row(r: usize) -> usize {
    debug_assert!(r < SIGMA_ROWS);
    COL_IS_SIGMA_ROW_OFFSET + r
}

// ─── IV-init / final h_out finalize columns (Task #206) ───────────────
//
// `h_in[0..64]`: per-row byte view of the F invocation's input chain.
//   Constant across all rows of one BLAKE2b F invocation. Used on
//   `is_first_round` to algebraically bind v[0..8] = h_in[0..8], and on
//   `is_final_round` to drive the h_out = h_in XOR v_final[i] XOR v_final[i+8]
//   chain.
//
// `v_final[0..128]`: per-row byte view of the v state AFTER all 12 G-mixing
//   rounds complete. Populated only on the FINAL (`is_result_row`) row.
//
// `is_first_round`: selector for the FIRST G-mixing row of an invocation.
//   Drives the IV-pin + counter/flag XOR + h_in binding constraints.
//
// `is_final_round` is the existing `is_result_row` column. The result row
// holds `result_h[0..64]`, which the final XOR chain pins to
// `h_in XOR v_final[i] XOR v_final[i+8]`.
pub const H_IN_LENGTH: usize = H_LIMBS * LIMB_BYTES; // 64
pub const V_FINAL_LENGTH: usize = V_LIMBS * LIMB_BYTES; // 128

pub const COL_H_IN_OFFSET: usize =
    COL_IS_SIGMA_ROW_OFFSET + SIGMA_ROWS; // 6740
pub const COL_V_FINAL_OFFSET: usize =
    COL_H_IN_OFFSET + H_IN_LENGTH; // 6804
pub const COL_IS_FIRST_ROUND: usize =
    COL_V_FINAL_OFFSET + V_FINAL_LENGTH; // 6932

// ─── IV XOR bit-decomposition columns (Task #215) ─────────────────────
//
// Three IV-init XOR instances close the algebraic binding for
//   v[12] = IV[4] XOR t.lo
//   v[13] = IV[5] XOR t.hi
//   v[14] = IV[6] XOR f_byte_mask    (f_byte_mask = f_byte ? all-ones : 0)
// each via a per-bit decomposition of (A, B, dst) where A = IV constant,
// B = (t.lo | t.hi | f_byte * all-ones), dst = v[12 | 13 | 14] limb. All
// three use rot=0 (pure XOR, no rotation), reusing the existing
// XorRotrInstance / 192-col / 259-constraint shape but gated by
// `is_first_round` instead of `is_real * (1 - is_result_row)`.
pub const IV_XOR_INSTANCE_COUNT: usize = 3;
pub const IV_XOR_BIT_LIMBS_PER_INSTANCE: usize = XOR_ROTR_BIT_LIMBS_PER_INSTANCE; // 3
pub const IV_XOR_BITS_PER_LIMB: usize = XOR_ROTR_BITS_PER_LIMB; // 64
pub const IV_XOR_BIT_COUNT_PER_INSTANCE: usize =
    IV_XOR_BIT_LIMBS_PER_INSTANCE * IV_XOR_BITS_PER_LIMB; // 192
pub const IV_XOR_BIT_COL_COUNT: usize =
    IV_XOR_INSTANCE_COUNT * IV_XOR_BIT_COUNT_PER_INSTANCE; // 576

pub const COL_IV_XOR_BIT_OFFSET: usize = COL_IS_FIRST_ROUND + 1; // 6933

/// Per-instance IV constant index (IV[4], IV[5], IV[6]).
pub const IV_XOR_IV_INDEX: [usize; IV_XOR_INSTANCE_COUNT] = [4, 5, 6];
/// Per-instance v-slot index (v[12], v[13], v[14]).
pub const IV_XOR_V_INDEX: [usize; IV_XOR_INSTANCE_COUNT] = [12, 13, 14];

/// Source kind for the B operand of an IV XOR instance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IvXorBSource {
    /// Packed view of `T_OFFSET + byte_off .. + 8` bytes.
    TBytes(usize),
    /// `f_byte` column times the all-ones u64 mask. (B = f_byte * mask.)
    FByteMask,
}

/// Per-instance B operand description.
pub const IV_XOR_B_SOURCE: [IvXorBSource; IV_XOR_INSTANCE_COUNT] = [
    IvXorBSource::TBytes(0),         // t.lo (bytes 0..8)
    IvXorBSource::TBytes(LIMB_BYTES), // t.hi (bytes 8..16)
    IvXorBSource::FByteMask,         // f_byte * 0xFFFF..FF
];

/// Returns the base column of the `block_idx` ∈ {0=A, 1=B, 2=dst} bit
/// limb for the `inst_idx`-th IV XOR instance.
#[inline]
pub fn iv_xor_bit_base(inst_idx: usize, block_idx: usize) -> usize {
    debug_assert!(inst_idx < IV_XOR_INSTANCE_COUNT);
    debug_assert!(block_idx < IV_XOR_BIT_LIMBS_PER_INSTANCE);
    COL_IV_XOR_BIT_OFFSET
        + inst_idx * IV_XOR_BIT_COUNT_PER_INSTANCE
        + block_idx * IV_XOR_BITS_PER_LIMB
}

/// Returns column index of the `bit_idx`-th LSB-first bit of `block_idx`
/// ∈ {0=A, 1=B, 2=dst} for the `inst_idx`-th IV XOR instance.
#[inline]
pub fn col_iv_xor_bit(inst_idx: usize, block_idx: usize, bit_idx: usize) -> usize {
    debug_assert!(bit_idx < IV_XOR_BITS_PER_LIMB);
    iv_xor_bit_base(inst_idx, block_idx) + bit_idx
}

// ─── h_out finalize triple-XOR bit columns (Task #216) ────────────────
//
// 8 instances close the algebraic binding for the BLAKE2b finalization:
//   h_out[i] = h_in[i] XOR v_final[i] XOR v_final[i+8],   i ∈ 0..8.
//
// Each instance commits 4 × 64 LSB-first bit columns decomposing the
// FOUR participating 64-bit limbs:
//   * h_in_bit[i, b]       — bit b of h_in[i]
//   * v_lo_bit[i, b]       — bit b of v_final[i]
//   * v_hi_bit[i, b]       — bit b of v_final[i+8]
//   * h_out_bit[i, b]      — bit b of result_h[i]
//
// All four operate at the packed-u64 level (8 LE bytes → u64). The
// XOR-of-three identity is fused into a degree-3 row-local constraint:
//
//   h_out_bit[i, b] = h_in_bit[i, b] + v_lo_bit[i, b] + v_hi_bit[i, b]
//                  - 2 ·(h_in·v_lo + h_in·v_hi + v_lo·v_hi)
//                  + 4 ·(h_in·v_lo·v_hi)
//
// All gated by `is_final_round` (= COL_IS_RESULT_ROW). Per instance:
//   * 256 binary cell constraints (one per bit),
//   * 4   pack-equality constraints (h_in, v_lo, v_hi, h_out),
//   * 64  fused triple-XOR constraints.
// Net per instance: 256 + 4 + 64 = 324. Total: 8 × 324 = 2592.
pub const H_OUT_XOR_INSTANCE_COUNT: usize = H_LIMBS; // 8
pub const H_OUT_XOR_BIT_LIMBS_PER_INSTANCE: usize = 4; // h_in, v_lo, v_hi, h_out
pub const H_OUT_XOR_BITS_PER_LIMB: usize = 64;
pub const H_OUT_XOR_BIT_COUNT_PER_INSTANCE: usize =
    H_OUT_XOR_BIT_LIMBS_PER_INSTANCE * H_OUT_XOR_BITS_PER_LIMB; // 256
pub const H_OUT_XOR_BIT_COL_COUNT: usize =
    H_OUT_XOR_INSTANCE_COUNT * H_OUT_XOR_BIT_COUNT_PER_INSTANCE; // 2048

pub const COL_H_OUT_XOR_BIT_OFFSET: usize =
    COL_IV_XOR_BIT_OFFSET + IV_XOR_BIT_COL_COUNT; // 7509

/// Returns the base column of the `block_idx` ∈ {0=h_in, 1=v_lo, 2=v_hi,
/// 3=h_out} bit limb for the `inst_idx`-th h_out XOR instance.
#[inline]
pub fn h_out_xor_bit_base(inst_idx: usize, block_idx: usize) -> usize {
    debug_assert!(inst_idx < H_OUT_XOR_INSTANCE_COUNT);
    debug_assert!(block_idx < H_OUT_XOR_BIT_LIMBS_PER_INSTANCE);
    COL_H_OUT_XOR_BIT_OFFSET
        + inst_idx * H_OUT_XOR_BIT_COUNT_PER_INSTANCE
        + block_idx * H_OUT_XOR_BITS_PER_LIMB
}

/// Returns column index of the `bit_idx`-th LSB-first bit of `block_idx`
/// ∈ {0=h_in, 1=v_lo, 2=v_hi, 3=h_out} for the `inst_idx`-th instance.
#[inline]
pub fn col_h_out_xor_bit(
    inst_idx: usize,
    block_idx: usize,
    bit_idx: usize,
) -> usize {
    debug_assert!(bit_idx < H_OUT_XOR_BITS_PER_LIMB);
    h_out_xor_bit_base(inst_idx, block_idx) + bit_idx
}

pub const NUM_COLUMNS: usize =
    COL_H_OUT_XOR_BIT_OFFSET + H_OUT_XOR_BIT_COL_COUNT; // 9557

// ─── Legacy G0-step-2 aliases (back-compat) ──────────────────────────
//
// The original Task #181 code referenced these names directly; they now
// point at the G0-step-2 instance (instance 0) of the generic 32-slot
// XOR/ROTR layer.
pub const G0_STEP2_BIT_LIMBS: usize = XOR_ROTR_BIT_LIMBS_PER_INSTANCE;
pub const G0_STEP2_BITS_PER_LIMB: usize = XOR_ROTR_BITS_PER_LIMB;
pub const G0_STEP2_BIT_COUNT: usize = XOR_ROTR_BIT_COUNT_PER_INSTANCE;
pub const COL_G0_STEP2_BIT_OFFSET: usize = COL_XOR_ROTR_BIT_OFFSET;
pub const G0_STEP2_VA1_BIT_OFFSET: usize = COL_XOR_ROTR_BIT_OFFSET;
pub const G0_STEP2_VD_BIT_OFFSET: usize =
    COL_XOR_ROTR_BIT_OFFSET + XOR_ROTR_BITS_PER_LIMB;
pub const G0_STEP2_VD1_BIT_OFFSET: usize =
    COL_XOR_ROTR_BIT_OFFSET + 2 * XOR_ROTR_BITS_PER_LIMB;

/// Returns column index of the i-th LSB-first bit of G0's v_a1.
#[inline]
pub fn col_g0_step2_va1_bit(i: usize) -> usize {
    col_xor_rotr_bit(0, 0, i)
}
/// Returns column index of the i-th LSB-first bit of G0's v_d input.
#[inline]
pub fn col_g0_step2_vd_bit(i: usize) -> usize {
    col_xor_rotr_bit(0, 1, i)
}
/// Returns column index of the i-th LSB-first bit of G0's v_d1.
#[inline]
pub fn col_g0_step2_vd1_bit(i: usize) -> usize {
    col_xor_rotr_bit(0, 2, i)
}

/// Returns column index for G g_idx's k-th intermediate (k in 0..8).
#[inline]
pub fn col_g_intermediate(g_idx: usize, k: usize) -> usize {
    debug_assert!(g_idx < G_PER_ROUND);
    debug_assert!(k < G_INTERMEDIATES_PER_G);
    COL_G_INTERMEDIATE_OFFSET + g_idx * G_INTERMEDIATES_PER_G + k
}

/// Returns column index for G g_idx's k-th carry (k in 0..4).
#[inline]
pub fn col_g_carry(g_idx: usize, k: usize) -> usize {
    debug_assert!(g_idx < G_PER_ROUND);
    debug_assert!(k < G_CARRIES_PER_G);
    COL_G_CARRY_OFFSET + g_idx * G_CARRIES_PER_G + k
}

/// Returns column index for the k-th packed message-schedule limb.
#[inline]
pub fn col_m_sched(k: usize) -> usize {
    debug_assert!(k < M_SCHED_LIMBS);
    COL_M_SCHED_OFFSET + k
}

// G call to v-index mapping. G_g maps the 4 internal indices
// (a, b, c, d) to v[V_MIX_INDICES[g][0..4]].
pub const V_MIX_INDICES: [[usize; 4]; G_PER_ROUND] = [
    [0, 4, 8, 12],   // G0 (column)
    [1, 5, 9, 13],   // G1 (column)
    [2, 6, 10, 14],  // G2 (column)
    [3, 7, 11, 15],  // G3 (column)
    [0, 5, 10, 15],  // G4 (diagonal)
    [1, 6, 11, 12],  // G5 (diagonal)
    [2, 7, 8, 13],   // G6 (diagonal)
    [3, 4, 9, 14],   // G7 (diagonal)
];

/// For G4..G7 (diagonal phase), each input v_a/v_b/v_c/v_d threads from
/// the corresponding column-phase G's output rather than from the
/// original v bytes. Entry `(g_src, slot_src)` = read intermediate slot
/// `slot_src` from G `g_src` (slot 4 = v_a2, 5 = v_d2, 6 = v_c2,
/// 7 = v_b2). G0..G3 read from v bytes directly (entry `None`).
pub const G_INPUT_FROM_INTERMEDIATE:
    [Option<[(usize, usize); 4]>; G_PER_ROUND] = [
    None, None, None, None,
    // G4 = (v[0], v[5], v[10], v[15])
    // v[0]  was written by G0.v_a2 (slot 4)
    // v[5]  was written by G1.v_b2 (slot 7)
    // v[10] was written by G2.v_c2 (slot 6)
    // v[15] was written by G3.v_d2 (slot 5)
    Some([(0, 4), (1, 7), (2, 6), (3, 5)]),
    // G5 = (v[1], v[6], v[11], v[12])
    Some([(1, 4), (2, 7), (3, 6), (0, 5)]),
    // G6 = (v[2], v[7], v[8],  v[13])
    Some([(2, 4), (3, 7), (0, 6), (1, 5)]),
    // G7 = (v[3], v[4], v[9],  v[14])
    Some([(3, 4), (0, 7), (1, 6), (2, 5)]),
];

// Per-G message-schedule index offsets within m_sched[0..16]:
// G_g uses m_sched[2g] and m_sched[2g+1].
#[inline]
pub fn g_msched_i(g_idx: usize) -> usize { 2 * g_idx }
#[inline]
pub fn g_msched_j(g_idx: usize) -> usize { 2 * g_idx + 1 }

/// BLAKE2 SIGMA permutation table (10 rounds; rounds 10/11 wrap via mod).
pub const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// Row-local constraints:
///   0:                 is_real ∈ {0, 1}
///   1:                 is_result_row ∈ {0, 1}
///   2:                 f_byte ∈ {0, 1}
///   3:                 is_result_row * (1 - is_real) = 0
///   4:                 LE byte-decomp of round_index_word
///   5:                 (1 - is_result_row) * Σ result[i] = 0
///   6..6+12*G_PER_ROUND-1: per-G additive layer (12 constraints per G):
///       step-1 add  : v_a + v_b + m_i = v_a1 + c1 · 2^64
///       step-3 add  : v_c + v_d1      = v_c1 + c2 · 2^64
///       step-5 add  : v_a1 + v_b1 + m_j = v_a2 + c3 · 2^64
///       step-7 add  : v_c1 + v_d2     = v_c2 + c4 · 2^64
///       carry c1    : c1 · (c1 - 1) · (c1 - 2) = 0   (range {0,1,2})
///       carry c2    : c2 · (c2 - 1)            = 0   (range {0,1})
///       carry c3    : c3 · (c3 - 1) · (c3 - 2) = 0   (range {0,1,2})
///       carry c4    : c4 · (c4 - 1)            = 0   (range {0,1})
///       pack v_a2   : v_a2 == packed(next_v[a])
///       pack v_b2   : v_b2 == packed(next_v[b])
///       pack v_c2   : v_c2 == packed(next_v[c])
///       pack v_d2   : v_d2 == packed(next_v[d])
///
///   All G constraints are gated by `is_real * (1 - is_result_row)`,
///   so the FINAL row (is_result_row = 1) is exempt from G binding.
pub const NUM_G_CONSTRAINTS_PER_G: usize = 12;
pub const NUM_G_CONSTRAINTS_TOTAL: usize =
    G_PER_ROUND * NUM_G_CONSTRAINTS_PER_G; // 96

/// Row-local constraints added per XOR/ROTR instance:
///   * 192 binary cell constraints  (one per bit column)
///   * 3   pack-equality constraints (A, B, dst)
///   * 64  fused XOR + ROTR_n constraints
pub const NUM_XOR_ROTR_BINARY_PER_INSTANCE: usize =
    XOR_ROTR_BIT_COUNT_PER_INSTANCE; // 192
pub const NUM_XOR_ROTR_PACK_PER_INSTANCE: usize = 3;
pub const NUM_XOR_ROTR_BITWISE_PER_INSTANCE: usize =
    XOR_ROTR_BITS_PER_LIMB; // 64
pub const NUM_XOR_ROTR_CONSTRAINTS_PER_INSTANCE: usize =
    NUM_XOR_ROTR_BINARY_PER_INSTANCE
        + NUM_XOR_ROTR_PACK_PER_INSTANCE
        + NUM_XOR_ROTR_BITWISE_PER_INSTANCE; // 259
pub const NUM_XOR_ROTR_CONSTRAINTS_TOTAL: usize =
    XOR_ROTR_INSTANCE_COUNT * NUM_XOR_ROTR_CONSTRAINTS_PER_INSTANCE; // 8288

// Legacy aliases (G0 step 2 was the original single-instance layer).
pub const NUM_G0_STEP2_BINARY: usize = NUM_XOR_ROTR_BINARY_PER_INSTANCE;
pub const NUM_G0_STEP2_PACK: usize = NUM_XOR_ROTR_PACK_PER_INSTANCE;
pub const NUM_G0_STEP2_XOR_ROTR: usize = NUM_XOR_ROTR_BITWISE_PER_INSTANCE;
pub const NUM_G0_STEP2_CONSTRAINTS: usize =
    NUM_XOR_ROTR_CONSTRAINTS_PER_INSTANCE;

// ─── σ-permutation message-schedule binding ──────────────────────────
//
// We add 10 + 16 = 160 constraints `is_sigma_row[r] * (m_sched[k] -
// pack_m(SIGMA[r][k])) = 0` plus a small auxiliary layer that pins
// `round_mod_10` and the one-hot selectors:
//
//   * (1)  round_index - 10·q - round_mod_10 = 0        (ungated)
//   * (1)  q · (q - 1) = 0                              (gated: is_real * (1 - is_result_row))
//   * (10) is_sigma_row[r] · (is_sigma_row[r] - 1) = 0  (ungated)
//   * (1)  Σ is_sigma_row[r] = is_real * (1 - is_result_row)        (ungated)
//   * (1)  gate · (Σ r · is_sigma_row[r] - round_mod_10) = 0        (gated)
//   * (160) is_sigma_row[r] · (m_sched[k] - pack_m(SIGMA[r][k])) = 0 (ungated; selector handles gating)
//
// On padding / result rows the sum-to-one and quotient layer force every
// is_sigma_row[r] = 0, so the 160 σ-binding constraints all reduce to 0.
pub const NUM_SIGMA_AUX_CONSTRAINTS: usize = 1 + 1 + SIGMA_ROWS + 1 + 1; // 14
pub const NUM_SIGMA_BIND_CONSTRAINTS: usize = SIGMA_ROWS * M_SCHED_LIMBS; // 160
pub const NUM_SIGMA_CONSTRAINTS_TOTAL: usize =
    NUM_SIGMA_AUX_CONSTRAINTS + NUM_SIGMA_BIND_CONSTRAINTS; // 174

// ─── IV-init / final h_out finalize constraints (Task #206) ──────────
//
// Constraints (all packed-level, evaluated as u64 limb identities):
//   * 1: `is_first_round ∈ {0, 1}` binary.
//   * 8: On `is_first_round`, `pack_v(v[i]) == pack_h_in(i)` for i ∈ 0..8.
//   * 4: On `is_first_round`, `pack_v(v[i+8]) == IV[i]` for i ∈ 0..4.
//   * 1: On `is_first_round`, `pack_v(v[15]) == IV[7]`.
//
// (v[12..15] = IV[4..7] XOR (t_lo, t_hi, f_byte_mask) require per-bit XOR
//  decomposition and are pinned alongside the final h_out XOR chain in a
//  follow-up step that introduces ~19 XOR instances (3 IV-init + 16
//  h_out triple-chain) of the same 192-col / 259-constraint shape as the
//  existing XOR/ROTR layer. See task #206 follow-up for details.)
//
// The final h_out chain `result_h[i] = h_in[i] XOR v_final[i] XOR
// v_final[i+8]` is currently pinned only at the **packed-byte equality**
// level by adding NO new constraints (since `v_final` is a free witness
// column that's zero in current skeleton tests); per-bit XOR closure is
// deferred to the same follow-up step.
pub const NUM_IV_INIT_BINARY: usize = 1;
pub const NUM_IV_INIT_H_PACK: usize = 8;
pub const NUM_IV_INIT_IV_PACK: usize = 5; // 4 (v[8..11]) + 1 (v[15])
pub const NUM_IV_INIT_CONSTRAINTS: usize =
    NUM_IV_INIT_BINARY + NUM_IV_INIT_H_PACK + NUM_IV_INIT_IV_PACK; // 14

// ─── IV XOR constraints per instance (Task #215) ──────────────────────
//
// Each of the 3 IV XOR instances (v[12], v[13], v[14]) adds:
//   * 192 binary cell constraints (one per bit column),
//   * 3   pack-equality constraints (A == IV[k], B == target u64, dst == pack_v(v_idx)),
//   * 64  fused XOR (rot=0) constraints  dst_bit[i] = A_bit[i] XOR B_bit[i].
// All gated by `is_first_round`. Net per instance: 192 + 3 + 64 = 259.
pub const NUM_IV_XOR_BINARY_PER_INSTANCE: usize =
    IV_XOR_BIT_COUNT_PER_INSTANCE; // 192
pub const NUM_IV_XOR_PACK_PER_INSTANCE: usize = 3;
pub const NUM_IV_XOR_BITWISE_PER_INSTANCE: usize =
    IV_XOR_BITS_PER_LIMB; // 64
pub const NUM_IV_XOR_CONSTRAINTS_PER_INSTANCE: usize =
    NUM_IV_XOR_BINARY_PER_INSTANCE
        + NUM_IV_XOR_PACK_PER_INSTANCE
        + NUM_IV_XOR_BITWISE_PER_INSTANCE; // 259
pub const NUM_IV_XOR_CONSTRAINTS_TOTAL: usize =
    IV_XOR_INSTANCE_COUNT * NUM_IV_XOR_CONSTRAINTS_PER_INSTANCE; // 777

// ─── h_out finalize triple-XOR constraints (Task #216) ────────────────
//
// 8 instances closing `h_out[i] = h_in[i] XOR v_final[i] XOR v_final[i+8]`.
// Each instance contributes:
//   * 256 binary cell constraints (4 limbs × 64 bits),
//   * 4   pack-equality constraints (h_in / v_lo / v_hi / h_out),
//   * 64  fused triple-XOR constraints.
// All gated by `is_final_round` (= `is_result_row`).
pub const NUM_H_OUT_XOR_BINARY_PER_INSTANCE: usize =
    H_OUT_XOR_BIT_COUNT_PER_INSTANCE; // 256
pub const NUM_H_OUT_XOR_PACK_PER_INSTANCE: usize =
    H_OUT_XOR_BIT_LIMBS_PER_INSTANCE; // 4
pub const NUM_H_OUT_XOR_BITWISE_PER_INSTANCE: usize =
    H_OUT_XOR_BITS_PER_LIMB; // 64
pub const NUM_H_OUT_XOR_CONSTRAINTS_PER_INSTANCE: usize =
    NUM_H_OUT_XOR_BINARY_PER_INSTANCE
        + NUM_H_OUT_XOR_PACK_PER_INSTANCE
        + NUM_H_OUT_XOR_BITWISE_PER_INSTANCE; // 324
pub const NUM_H_OUT_XOR_CONSTRAINTS_TOTAL: usize =
    H_OUT_XOR_INSTANCE_COUNT * NUM_H_OUT_XOR_CONSTRAINTS_PER_INSTANCE; // 2592

pub const NUM_ROW_CONSTRAINTS: usize =
    6 + NUM_G_CONSTRAINTS_TOTAL
        + NUM_XOR_ROTR_CONSTRAINTS_TOTAL
        + NUM_SIGMA_CONSTRAINTS_TOTAL
        + NUM_IV_INIT_CONSTRAINTS
        + NUM_IV_XOR_CONSTRAINTS_TOTAL
        + NUM_H_OUT_XOR_CONSTRAINTS_TOTAL; // 11947

/// Shifted constraints:
///   0:                 v-state continuity (next_v(X) == v(ω·X) gated)
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

/// Per-G witnessed intermediates (each a 64-bit limb packed in a single
/// column) and carry witnesses (each a small non-negative integer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GIntermediate {
    /// `v_a1, v_d1, v_c1, v_b1, v_a2, v_d2, v_c2, v_b2` packed u64.
    pub limbs: [u64; G_INTERMEDIATES_PER_G],
    /// Carries `c1, c2, c3, c4` from the four adds.
    pub carries: [u64; G_CARRIES_PER_G],
}

/// One row of the BLAKE2 F internals trace (one round).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blake2fInternalsRow {
    pub v: [u8; V_LENGTH],
    pub next_v: [u8; V_LENGTH],
    pub m: [u8; M_LENGTH],
    pub t: [u8; T_LENGTH],
    pub f_byte: u8,
    pub round_index: u32,
    pub result_h: [u8; RESULT_LENGTH],
    pub is_result_row: bool,
    /// σ-permuted message schedule for this row (packed u64 limbs).
    /// On padding/result rows this is left as zeros.
    pub m_sched: [u64; M_SCHED_LIMBS],
    /// Per-G intermediates + carry witnesses (8 G calls per round).
    /// On padding/result rows these are zeros.
    pub g_intermediates: [GIntermediate; G_PER_ROUND],
    /// Input chain (h_in[0..64]) of the BLAKE2b F invocation.
    /// Constant across all rows of one invocation. Used by the
    /// IV-init binding on `is_first_round` and the h_out XOR chain
    /// on `is_result_row` (= `is_final_round`).
    pub h_in: [u8; H_IN_LENGTH],
    /// Final v state after all 12 G-mixing rounds, byte view.
    /// Populated only on the FINAL row (`is_result_row = true`).
    pub v_final: [u8; V_FINAL_LENGTH],
    /// Selector: true on the FIRST G-mixing row of an invocation.
    /// Drives the IV-init / counter / flag XOR binding constraints.
    pub is_first_round: bool,
}

impl Default for Blake2fInternalsRow {
    fn default() -> Self {
        Self {
            v: [0u8; V_LENGTH],
            next_v: [0u8; V_LENGTH],
            m: [0u8; M_LENGTH],
            t: [0u8; T_LENGTH],
            f_byte: 0,
            round_index: 0,
            result_h: [0u8; RESULT_LENGTH],
            is_result_row: false,
            m_sched: [0u64; M_SCHED_LIMBS],
            g_intermediates: [GIntermediate::default(); G_PER_ROUND],
            h_in: [0u8; H_IN_LENGTH],
            v_final: [0u8; V_FINAL_LENGTH],
            is_first_round: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Blake2fInternalsTraceWitness {
    pub rows: Vec<Blake2fInternalsRow>,
}

/// Return 2^64 as a Scalar (constructed as (2^32)^2 to fit `from_u64`).
#[inline]
pub fn two_pow_64_scalar(curve: CurveType) -> Scalar {
    let half = Scalar::from_u64(1u64 << 32, curve);
    half.mul(&half)
}

/// G-constraint gate: `is_real * (1 - is_result_row)` evaluated on the
/// domain at row `r`.
#[inline]
fn gate_g(columns: &[&Vec<Scalar>], r: usize, one: &Scalar) -> Scalar {
    let real = &columns[COL_IS_REAL][r];
    let is_result = &columns[COL_IS_RESULT_ROW][r];
    real.mul(&one.sub(is_result))
}

/// Read 8 LE bytes from a slice as a packed u64 limb.
#[inline]
pub fn pack_u64_le(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() >= 8);
    let mut out = [0u8; 8];
    out.copy_from_slice(&bytes[0..8]);
    u64::from_le_bytes(out)
}

/// Write a packed u64 to 8 LE bytes.
#[inline]
pub fn write_u64_le(out: &mut [u8], v: u64) {
    debug_assert!(out.len() >= 8);
    out[0..8].copy_from_slice(&v.to_le_bytes());
}

/// Resolve a host-side u64 value for an XOR/ROTR operand on a given row.
/// Used by the trace builder to populate the bit-decomposition columns.
#[inline]
pub fn read_xor_rotr_operand(
    operand: XorRotrOperand,
    row: &Blake2fInternalsRow,
) -> u64 {
    match operand {
        XorRotrOperand::InputVBytes(v_idx) => {
            pack_u64_le(&row.v[v_idx * LIMB_BYTES..])
        }
        XorRotrOperand::Intermediate { g_idx, slot } => {
            row.g_intermediates[g_idx].limbs[slot]
        }
    }
}

/// Compute one BLAKE2b G call given the four input v-words and message
/// pair (m_i, m_j). Returns the eight intermediate u64 values + four
/// carry witnesses (carries from the four adds, range {0,1,2}/{0,1}).
pub fn blake2_g_intermediates(
    v_a: u64,
    v_b: u64,
    v_c: u64,
    v_d: u64,
    m_i: u64,
    m_j: u64,
) -> GIntermediate {
    // step 1: v_a += v_b + m_i
    let s1 = (v_a as u128) + (v_b as u128) + (m_i as u128);
    let v_a1 = s1 as u64;
    let c1 = (s1 >> 64) as u64;
    // step 2: v_d = ROTR_32(v_d XOR v_a1)
    let v_d1 = (v_d ^ v_a1).rotate_right(32);
    // step 3: v_c += v_d1
    let s3 = (v_c as u128) + (v_d1 as u128);
    let v_c1 = s3 as u64;
    let c2 = (s3 >> 64) as u64;
    // step 4: v_b = ROTR_24(v_b XOR v_c1)
    let v_b1 = (v_b ^ v_c1).rotate_right(24);
    // step 5: v_a += v_b1 + m_j
    let s5 = (v_a1 as u128) + (v_b1 as u128) + (m_j as u128);
    let v_a2 = s5 as u64;
    let c3 = (s5 >> 64) as u64;
    // step 6: v_d = ROTR_16(v_d1 XOR v_a2)
    let v_d2 = (v_d1 ^ v_a2).rotate_right(16);
    // step 7: v_c += v_d2
    let s7 = (v_c1 as u128) + (v_d2 as u128);
    let v_c2 = s7 as u64;
    let c4 = (s7 >> 64) as u64;
    // step 8: v_b = ROTR_63(v_b1 XOR v_c2)
    let v_b2 = (v_b1 ^ v_c2).rotate_right(63);
    GIntermediate {
        limbs: [v_a1, v_d1, v_c1, v_b1, v_a2, v_d2, v_c2, v_b2],
        carries: [c1, c2, c3, c4],
    }
}

/// Run a single BLAKE2b round on the given 16-word `v` and 16-word
/// permuted message schedule, returning the updated `v`. Populates the
/// supplied row with all G intermediates + carries + next_v + m_sched.
pub fn build_round_row(
    v_in: [u64; V_LIMBS],
    m_words: [u64; M_LIMBS],
    t: [u8; T_LENGTH],
    f_byte: u8,
    round_index: u32,
) -> Blake2fInternalsRow {
    let sigma = SIGMA[(round_index as usize) % 10];
    let mut m_sched = [0u64; M_SCHED_LIMBS];
    for k in 0..16 {
        m_sched[k] = m_words[sigma[k]];
    }
    let mut v = v_in;
    let mut g_intermediates = [GIntermediate::default(); G_PER_ROUND];
    for g_idx in 0..G_PER_ROUND {
        let ai = V_MIX_INDICES[g_idx][0];
        let bi = V_MIX_INDICES[g_idx][1];
        let ci = V_MIX_INDICES[g_idx][2];
        let di = V_MIX_INDICES[g_idx][3];
        let g = blake2_g_intermediates(
            v[ai],
            v[bi],
            v[ci],
            v[di],
            m_sched[2 * g_idx],
            m_sched[2 * g_idx + 1],
        );
        // Final words land in v.
        v[ai] = g.limbs[4]; // v_a2
        v[bi] = g.limbs[7]; // v_b2
        v[ci] = g.limbs[6]; // v_c2
        v[di] = g.limbs[5]; // v_d2
        g_intermediates[g_idx] = g;
    }
    let mut v_bytes = [0u8; V_LENGTH];
    let mut next_v_bytes = [0u8; V_LENGTH];
    for k in 0..V_LIMBS {
        write_u64_le(&mut v_bytes[k * LIMB_BYTES..], v_in[k]);
        write_u64_le(&mut next_v_bytes[k * LIMB_BYTES..], v[k]);
    }
    let mut m_bytes = [0u8; M_LENGTH];
    for k in 0..M_LIMBS {
        write_u64_le(&mut m_bytes[k * LIMB_BYTES..], m_words[k]);
    }
    Blake2fInternalsRow {
        v: v_bytes,
        next_v: next_v_bytes,
        m: m_bytes,
        t,
        f_byte,
        round_index,
        result_h: [0u8; RESULT_LENGTH],
        is_result_row: false,
        m_sched,
        g_intermediates,
        h_in: [0u8; H_IN_LENGTH],
        v_final: [0u8; V_FINAL_LENGTH],
        is_first_round: false,
    }
}

impl Blake2fInternalsTraceWitness {
    pub fn from_rows(rows: Vec<Blake2fInternalsRow>) -> Self {
        Self { rows }
    }
    pub fn push(&mut self, row: Blake2fInternalsRow) {
        self.rows.push(row);
    }

    /// Build an honest single-round witness: one non-result G-mixing
    /// round row + a final result row carrying `result_h`. The G-mixing
    /// row's intermediate witness, carries, and m_sched are populated
    /// algebraically so all 96 G constraints (plus is_real/etc.) hold.
    pub fn single_round_then_result(
        v_in: [u64; V_LIMBS],
        m_words: [u64; M_LIMBS],
        t: [u8; T_LENGTH],
        f_byte: u8,
        round_index: u32,
        result_h: [u8; RESULT_LENGTH],
    ) -> Self {
        let mut rows = Vec::with_capacity(2);
        let row0 = build_round_row(v_in, m_words, t, f_byte, round_index);
        rows.push(row0);
        rows.push(Blake2fInternalsRow {
            v: [0u8; V_LENGTH],
            next_v: [0u8; V_LENGTH],
            m: [0u8; M_LENGTH],
            t: [0u8; T_LENGTH],
            f_byte,
            round_index,
            result_h,
            is_result_row: true,
            m_sched: [0u64; M_SCHED_LIMBS],
            g_intermediates: [GIntermediate::default(); G_PER_ROUND],
            h_in: [0u8; H_IN_LENGTH],
            v_final: [0u8; V_FINAL_LENGTH],
            is_first_round: false,
        });
        Self { rows }
    }

    /// Build a SKELETON witness from a finalized BLAKE2b F invocation
    /// output. Intermediate v/m/t columns are populated as zeros — the
    /// downstream gadget will replace them with algebraically-computed
    /// round states. Only the FINAL row carries the meaningful
    /// `result_h`.
    pub fn from_result_skeleton(h_out: [u8; RESULT_LENGTH]) -> Self {
        // Populate v_final[i+8] = h_out[i] so the Task #216 h_out triple-XOR
        // finalize identity (h_out[i] = h_in[i] XOR v_final[i] XOR v_final[i+8])
        // holds trivially with h_in = v_final[0..8] = 0.
        let mut v_final = [0u8; V_FINAL_LENGTH];
        for i in 0..H_LIMBS {
            let dst = (i + 8) * LIMB_BYTES;
            let src = i * LIMB_BYTES;
            v_final[dst..dst + LIMB_BYTES]
                .copy_from_slice(&h_out[src..src + LIMB_BYTES]);
        }
        let mut rows = Vec::with_capacity(2);
        rows.push(Blake2fInternalsRow::default());
        rows.push(Blake2fInternalsRow {
            v: [0u8; V_LENGTH],
            next_v: [0u8; V_LENGTH],
            m: [0u8; M_LENGTH],
            t: [0u8; T_LENGTH],
            f_byte: 1,
            round_index: MAINLINE_ROUNDS as u32,
            result_h: h_out,
            is_result_row: true,
            m_sched: [0u64; M_SCHED_LIMBS],
            g_intermediates: [GIntermediate::default(); G_PER_ROUND],
            h_in: [0u8; H_IN_LENGTH],
            v_final,
            is_first_round: false,
        });
        Self { rows }
    }

    /// Build a 1-row witness containing only the FINAL/result row, with
    /// `h_in`, `v_final`, and `result_h = h_in XOR v_final[0..8] XOR
    /// v_final[8..16]` populated so the Task #216 h_out triple-XOR
    /// finalize layer accepts the witness. The G-mixing / IV-init
    /// layers stay quiescent (is_real = 1, is_result_row = 1).
    pub fn final_row_only(
        h_in: [u8; H_IN_LENGTH],
        v_final: [u8; V_FINAL_LENGTH],
    ) -> Self {
        let mut result_h = [0u8; RESULT_LENGTH];
        for i in 0..H_LIMBS {
            let h_in_word = pack_u64_le(&h_in[i * LIMB_BYTES..]);
            let v_lo = pack_u64_le(&v_final[i * LIMB_BYTES..]);
            let v_hi = pack_u64_le(&v_final[(i + 8) * LIMB_BYTES..]);
            let h_out_word = h_in_word ^ v_lo ^ v_hi;
            write_u64_le(&mut result_h[i * LIMB_BYTES..], h_out_word);
        }
        let mut row = Blake2fInternalsRow::default();
        row.is_result_row = true;
        row.round_index = MAINLINE_ROUNDS as u32;
        row.h_in = h_in;
        row.v_final = v_final;
        row.result_h = result_h;
        Self { rows: vec![row] }
    }

    /// Build an IV-init witness pair: ONE first-round row with the
    /// initial v-state algebraically computed from `(h_in, t, f_byte)`
    /// per the BLAKE2b spec (`v[0..8] = h_in`, `v[8..11] = IV[0..3]`,
    /// `v[12] = IV[4] XOR t.lo`, `v[13] = IV[5] XOR t.hi`,
    /// `v[14] = IV[6] XOR (f_byte ? 0xFFF..F : 0)`, `v[15] = IV[7]`)
    /// followed by a zero-result row. The first row is gated by
    /// `is_first_round = true` so the IV-init binding fires.
    pub fn iv_init_then_result(
        h_in: [u8; H_IN_LENGTH],
        t: [u8; T_LENGTH],
        f_byte: u8,
        result_h: [u8; RESULT_LENGTH],
    ) -> Self {
        // Compute initial v[0..16] per the IV-init spec.
        let mut h_limbs = [0u64; H_LIMBS];
        for i in 0..H_LIMBS {
            h_limbs[i] = pack_u64_le(&h_in[i * LIMB_BYTES..]);
        }
        let t_lo = pack_u64_le(&t[0..LIMB_BYTES]);
        let t_hi = pack_u64_le(&t[LIMB_BYTES..]);
        let f_mask: u64 =
            if f_byte != 0 { F_BYTE_MASK_ONES } else { 0 };
        let mut v_init = [0u64; V_LIMBS];
        for i in 0..8 {
            v_init[i] = h_limbs[i];
        }
        for i in 0..4 {
            v_init[8 + i] = BLAKE2B_IV[i];
        }
        v_init[12] = BLAKE2B_IV[4] ^ t_lo;
        v_init[13] = BLAKE2B_IV[5] ^ t_hi;
        v_init[14] = BLAKE2B_IV[6] ^ f_mask;
        v_init[15] = BLAKE2B_IV[7];

        let mut v_bytes = [0u8; V_LENGTH];
        for k in 0..V_LIMBS {
            write_u64_le(&mut v_bytes[k * LIMB_BYTES..], v_init[k]);
        }
        let mut row0 = Blake2fInternalsRow::default();
        row0.v = v_bytes;
        row0.t = t;
        row0.f_byte = f_byte;
        row0.round_index = 0;
        row0.h_in = h_in;
        row0.is_first_round = true;

        let mut row_result = Blake2fInternalsRow::default();
        row_result.f_byte = f_byte;
        row_result.round_index = MAINLINE_ROUNDS as u32;
        row_result.result_h = result_h;
        row_result.is_result_row = true;
        row_result.h_in = h_in;
        // v_final is left as zeros for the skeleton; honest builders
        // can populate it once round-by-round threading is in place.

        Self { rows: vec![row0, row_result] }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Blake2fInternalsTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..V_LENGTH {
            columns[COL_V_OFFSET + k][r] =
                Scalar::from_u64(row.v[k] as u64, curve);
            columns[COL_NEXT_V_OFFSET + k][r] =
                Scalar::from_u64(row.next_v[k] as u64, curve);
        }
        for k in 0..M_LENGTH {
            columns[COL_M_OFFSET + k][r] =
                Scalar::from_u64(row.m[k] as u64, curve);
        }
        for k in 0..T_LENGTH {
            columns[COL_T_OFFSET + k][r] =
                Scalar::from_u64(row.t[k] as u64, curve);
        }
        columns[COL_F_BYTE][r] = Scalar::from_u64(row.f_byte as u64, curve);
        columns[COL_ROUND_INDEX][r] =
            Scalar::from_u64(row.round_index as u64, curve);
        let ri_le = row.round_index.to_le_bytes();
        for b in 0..ROUND_INDEX_BYTES {
            columns[COL_ROUND_INDEX_BYTE_OFFSET + b][r] =
                Scalar::from_u64(ri_le[b] as u64, curve);
        }
        for k in 0..RESULT_LENGTH {
            columns[COL_RESULT_OFFSET + k][r] =
                Scalar::from_u64(row.result_h[k] as u64, curve);
        }
        columns[COL_IS_RESULT_ROW][r] = if row.is_result_row {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_REAL][r] = one.clone();

        // G intermediates + carries.
        for g_idx in 0..G_PER_ROUND {
            let g = &row.g_intermediates[g_idx];
            for k in 0..G_INTERMEDIATES_PER_G {
                columns[col_g_intermediate(g_idx, k)][r] =
                    Scalar::from_u64(g.limbs[k], curve);
            }
            for k in 0..G_CARRIES_PER_G {
                columns[col_g_carry(g_idx, k)][r] =
                    Scalar::from_u64(g.carries[k], curve);
            }
        }
        // Message schedule.
        for k in 0..M_SCHED_LIMBS {
            columns[col_m_sched(k)][r] = Scalar::from_u64(row.m_sched[k], curve);
        }

        // ─── σ-permutation aux columns ─────────────────────────────
        // Pin `round_index = 10*q + round_mod_10`. On non-result G-mixing
        // rows the one-hot `is_sigma_row[r]` lights the `round_index mod 10`
        // selector; on result/padding rows ALL ten selectors stay zero
        // (sum-to-one constraint forces this).
        let ri = row.round_index as u64;
        let sigma_idx = (ri % SIGMA_ROWS as u64) as usize;
        let q10 = ri / SIGMA_ROWS as u64;
        columns[COL_ROUND_MOD_10][r] = Scalar::from_u64(sigma_idx as u64, curve);
        columns[COL_ROUND_QUOTIENT_10][r] = Scalar::from_u64(q10, curve);
        if !row.is_result_row {
            // Real G-mixing row: light exactly one of the 10 selectors.
            columns[col_is_sigma_row(sigma_idx)][r] = one.clone();
        }
        // result rows leave is_sigma_row[*] = 0 (initial zero fill).

        // ─── IV-init / final h_out finalize columns (Task #206) ──────
        for k in 0..H_IN_LENGTH {
            columns[COL_H_IN_OFFSET + k][r] =
                Scalar::from_u64(row.h_in[k] as u64, curve);
        }
        for k in 0..V_FINAL_LENGTH {
            columns[COL_V_FINAL_OFFSET + k][r] =
                Scalar::from_u64(row.v_final[k] as u64, curve);
        }
        columns[COL_IS_FIRST_ROUND][r] = if row.is_first_round {
            one.clone()
        } else {
            zero.clone()
        };

        // ─── IV XOR bit-decomposition columns (Task #215) ─────────────
        // Populate the 3 × 192 bit columns only on the FIRST-round row
        // (is_first_round = 1). On all other rows, leave as zero — the
        // gating by `is_first_round` zeroes every constraint regardless.
        if row.is_first_round {
            let t_lo = pack_u64_le(&row.t[0..LIMB_BYTES]);
            let t_hi = pack_u64_le(&row.t[LIMB_BYTES..]);
            let f_mask: u64 = if row.f_byte != 0 { F_BYTE_MASK_ONES } else { 0 };
            for inst_idx in 0..IV_XOR_INSTANCE_COUNT {
                let iv_const = BLAKE2B_IV[IV_XOR_IV_INDEX[inst_idx]];
                let b_val = match IV_XOR_B_SOURCE[inst_idx] {
                    IvXorBSource::TBytes(off) => {
                        if off == 0 { t_lo } else { t_hi }
                    }
                    IvXorBSource::FByteMask => f_mask,
                };
                let dst_val = pack_u64_le(&row.v[IV_XOR_V_INDEX[inst_idx] * LIMB_BYTES..]);
                for i in 0..IV_XOR_BITS_PER_LIMB {
                    let bit_a = ((iv_const >> i) & 1) as u64;
                    let bit_b = ((b_val >> i) & 1) as u64;
                    let bit_d = ((dst_val >> i) & 1) as u64;
                    columns[col_iv_xor_bit(inst_idx, 0, i)][r] =
                        Scalar::from_u64(bit_a, curve);
                    columns[col_iv_xor_bit(inst_idx, 1, i)][r] =
                        Scalar::from_u64(bit_b, curve);
                    columns[col_iv_xor_bit(inst_idx, 2, i)][r] =
                        Scalar::from_u64(bit_d, curve);
                }
            }
        }

        // ─── h_out finalize triple-XOR bit columns (Task #216) ────────
        // Populated only on the FINAL (`is_result_row = 1`) row. All
        // other rows leave the bit columns at zero — the gating by
        // `is_result_row` zeroes every constraint regardless.
        if row.is_result_row {
            for inst_idx in 0..H_OUT_XOR_INSTANCE_COUNT {
                let h_in_val = pack_u64_le(
                    &row.h_in[inst_idx * LIMB_BYTES..],
                );
                let v_lo_val = pack_u64_le(
                    &row.v_final[inst_idx * LIMB_BYTES..],
                );
                let v_hi_val = pack_u64_le(
                    &row.v_final[(inst_idx + 8) * LIMB_BYTES..],
                );
                let h_out_val = pack_u64_le(
                    &row.result_h[inst_idx * LIMB_BYTES..],
                );
                for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                    let bit_h_in = ((h_in_val >> i) & 1) as u64;
                    let bit_v_lo = ((v_lo_val >> i) & 1) as u64;
                    let bit_v_hi = ((v_hi_val >> i) & 1) as u64;
                    let bit_h_out = ((h_out_val >> i) & 1) as u64;
                    columns[col_h_out_xor_bit(inst_idx, 0, i)][r] =
                        Scalar::from_u64(bit_h_in, curve);
                    columns[col_h_out_xor_bit(inst_idx, 1, i)][r] =
                        Scalar::from_u64(bit_v_lo, curve);
                    columns[col_h_out_xor_bit(inst_idx, 2, i)][r] =
                        Scalar::from_u64(bit_v_hi, curve);
                    columns[col_h_out_xor_bit(inst_idx, 3, i)][r] =
                        Scalar::from_u64(bit_h_out, curve);
                }
            }
        }

        // XOR/ROTR bit decompositions for all 32 G×step instances. Each
        // instance writes its A, B, dst 64-bit values as 64 LSB-first bit
        // columns. Populated only on G-mixing (non-result) rows; result
        // rows leave these as zeros which trivially satisfy the gated
        // constraints.
        if !row.is_result_row {
            for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
                let inst = &XOR_ROTR_INSTANCES[inst_idx];
                let a_u = read_xor_rotr_operand(inst.a, row);
                let b_u = read_xor_rotr_operand(inst.b, row);
                let dst_u = read_xor_rotr_operand(inst.dst, row);
                for i in 0..XOR_ROTR_BITS_PER_LIMB {
                    let bit_a = ((a_u >> i) & 1) as u64;
                    let bit_b = ((b_u >> i) & 1) as u64;
                    let bit_d = ((dst_u >> i) & 1) as u64;
                    columns[col_xor_rotr_bit(inst_idx, 0, i)][r] =
                        Scalar::from_u64(bit_a, curve);
                    columns[col_xor_rotr_bit(inst_idx, 1, i)][r] =
                        Scalar::from_u64(bit_b, curve);
                    columns[col_xor_rotr_bit(inst_idx, 2, i)][r] =
                        Scalar::from_u64(bit_d, curve);
                }
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

pub struct Blake2fInternalsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Blake2fInternalsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Blake2fInternalsConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "is_real_binary".into(),
            "is_result_row_binary".into(),
            "f_byte_binary".into(),
            "is_result_row_le_is_real".into(),
            "round_index_le_byte_decomp".into(),
            "result_zero_off_final_row".into(),
        ];
        for g_idx in 0..G_PER_ROUND {
            labels.push(format!("g{}_add1", g_idx));
            labels.push(format!("g{}_add3", g_idx));
            labels.push(format!("g{}_add5", g_idx));
            labels.push(format!("g{}_add7", g_idx));
            labels.push(format!("g{}_c1_range_3way", g_idx));
            labels.push(format!("g{}_c2_binary", g_idx));
            labels.push(format!("g{}_c3_range_3way", g_idx));
            labels.push(format!("g{}_c4_binary", g_idx));
            labels.push(format!("g{}_pack_v_a2", g_idx));
            labels.push(format!("g{}_pack_v_b2", g_idx));
            labels.push(format!("g{}_pack_v_c2", g_idx));
            labels.push(format!("g{}_pack_v_d2", g_idx));
        }
        // XOR/ROTR bit-decomposition layer (32 instances).
        for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
            let inst = &XOR_ROTR_INSTANCES[inst_idx];
            let tag = format!("g{}_step{}_rotr{}", inst.g_idx, inst.step, inst.rot);
            for i in 0..XOR_ROTR_BITS_PER_LIMB {
                labels.push(format!("{}_a_bit_{}_binary", tag, i));
            }
            for i in 0..XOR_ROTR_BITS_PER_LIMB {
                labels.push(format!("{}_b_bit_{}_binary", tag, i));
            }
            for i in 0..XOR_ROTR_BITS_PER_LIMB {
                labels.push(format!("{}_dst_bit_{}_binary", tag, i));
            }
            labels.push(format!("{}_pack_a", tag));
            labels.push(format!("{}_pack_b", tag));
            labels.push(format!("{}_pack_dst", tag));
            for i in 0..XOR_ROTR_BITS_PER_LIMB {
                labels.push(format!("{}_xor_rotr_bit_{}", tag, i));
            }
        }
        // σ-permutation aux layer.
        labels.push("sigma_round_index_decomp".into());
        labels.push("sigma_q_binary".into());
        for r in 0..SIGMA_ROWS {
            labels.push(format!("sigma_is_sigma_row_{}_binary", r));
        }
        labels.push("sigma_one_hot_sum_to_one".into());
        labels.push("sigma_one_hot_binds_round_mod_10".into());
        // σ-binding constraints (10 × 16 = 160).
        for sr in 0..SIGMA_ROWS {
            for k in 0..M_SCHED_LIMBS {
                labels.push(format!("sigma_bind_r{}_k{}", sr, k));
            }
        }
        // IV-init / final h_out finalize layer (Task #206).
        labels.push("iv_init_is_first_round_binary".into());
        for i in 0..8 {
            labels.push(format!("iv_init_v{}_eq_h_in_{}", i, i));
        }
        for i in 0..4 {
            labels.push(format!("iv_init_v{}_eq_iv_{}", i + 8, i));
        }
        labels.push("iv_init_v15_eq_iv_7".into());
        // IV XOR layer (Task #215): 3 instances × 259 constraints each.
        for inst_idx in 0..IV_XOR_INSTANCE_COUNT {
            let v_idx = IV_XOR_V_INDEX[inst_idx];
            let iv_idx = IV_XOR_IV_INDEX[inst_idx];
            let tag = format!("iv_xor_v{}_iv{}", v_idx, iv_idx);
            for i in 0..IV_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_a_bit_{}_binary", tag, i));
            }
            for i in 0..IV_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_b_bit_{}_binary", tag, i));
            }
            for i in 0..IV_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_dst_bit_{}_binary", tag, i));
            }
            labels.push(format!("{}_pack_a", tag));
            labels.push(format!("{}_pack_b", tag));
            labels.push(format!("{}_pack_dst", tag));
            for i in 0..IV_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_xor_bit_{}", tag, i));
            }
        }
        // h_out finalize triple-XOR layer (Task #216): 8 instances ×
        // 324 constraints each.
        for inst_idx in 0..H_OUT_XOR_INSTANCE_COUNT {
            let tag = format!("h_out_xor_i{}", inst_idx);
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_h_in_bit_{}_binary", tag, i));
            }
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_v_lo_bit_{}_binary", tag, i));
            }
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_v_hi_bit_{}_binary", tag, i));
            }
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_h_out_bit_{}_binary", tag, i));
            }
            labels.push(format!("{}_pack_h_in", tag));
            labels.push(format!("{}_pack_v_lo", tag));
            labels.push(format!("{}_pack_v_hi", tag));
            labels.push(format!("{}_pack_h_out", tag));
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                labels.push(format!("{}_xor3_bit_{}", tag, i));
            }
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

        // 2: f_byte binary.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_F_BYTE][r];
            c[r] = v.mul(&v.sub(&one));
        }
        out.push(c);

        // 3: is_result_row * (1 - is_real).
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

        // ─── G-mixing layer (12 constraints per G × 8 G = 96) ─────
        let two_pow_64 = two_pow_64_scalar(curve);
        let two = Scalar::from_u64(2, curve);
        let limb_byte_weights: Vec<Scalar> = (0..LIMB_BYTES)
            .map(|b| Scalar::from_u64(1u64 << (8 * b), curve))
            .collect();

        let pack_v_at_row = |v_idx: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_V_OFFSET + v_idx * LIMB_BYTES + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let pack_next_v_at_row = |v_idx: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_NEXT_V_OFFSET + v_idx * LIMB_BYTES + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };

        // Helper: read G g_idx's v_a/v_b/v_c/v_d input at row r.
        // For G0..G3: pack from v-byte columns. For G4..G7: look up the
        // referenced earlier G's intermediate column.
        let g_input_at_row = |g_idx: usize, slot: usize, r: usize| -> Scalar {
            if let Some(spec) = G_INPUT_FROM_INTERMEDIATE[g_idx] {
                let (g_src, s_src) = spec[slot];
                columns[col_g_intermediate(g_src, s_src)][r].clone()
            } else {
                pack_v_at_row(V_MIX_INDICES[g_idx][slot], r)
            }
        };

        for g_idx in 0..G_PER_ROUND {
            let ai = V_MIX_INDICES[g_idx][0];
            let bi = V_MIX_INDICES[g_idx][1];
            let ci = V_MIX_INDICES[g_idx][2];
            let di = V_MIX_INDICES[g_idx][3];
            let m_i_col = col_m_sched(g_msched_i(g_idx));
            let m_j_col = col_m_sched(g_msched_j(g_idx));
            let v_a1 = col_g_intermediate(g_idx, 0);
            let v_d1 = col_g_intermediate(g_idx, 1);
            let v_c1 = col_g_intermediate(g_idx, 2);
            let v_b1 = col_g_intermediate(g_idx, 3);
            let v_a2 = col_g_intermediate(g_idx, 4);
            let v_d2 = col_g_intermediate(g_idx, 5);
            let v_c2 = col_g_intermediate(g_idx, 6);
            let v_b2 = col_g_intermediate(g_idx, 7);
            let cc1 = col_g_carry(g_idx, 0);
            let cc2 = col_g_carry(g_idx, 1);
            let cc3 = col_g_carry(g_idx, 2);
            let cc4 = col_g_carry(g_idx, 3);
            let pack_to_next_v = g_idx >= 4; // diagonal phase only

            // add1: v_a_in + v_b_in + m_i - v_a1 - c1·2^64 = 0
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v_a_in = g_input_at_row(g_idx, 0, r);
                let v_b_in = g_input_at_row(g_idx, 1, r);
                let lhs = v_a_in.add(&v_b_in).add(&columns[m_i_col][r]);
                let rhs = columns[v_a1][r]
                    .add(&columns[cc1][r].mul(&two_pow_64));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&lhs.sub(&rhs));
            }
            out.push(c);

            // add3: v_c_in + v_d1 - v_c1 - c2·2^64 = 0
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v_c_in = g_input_at_row(g_idx, 2, r);
                let lhs = v_c_in.add(&columns[v_d1][r]);
                let rhs = columns[v_c1][r]
                    .add(&columns[cc2][r].mul(&two_pow_64));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&lhs.sub(&rhs));
            }
            out.push(c);

            // add5: v_a1 + v_b1 + m_j - v_a2 - c3·2^64 = 0
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = columns[v_a1][r]
                    .add(&columns[v_b1][r])
                    .add(&columns[m_j_col][r]);
                let rhs = columns[v_a2][r]
                    .add(&columns[cc3][r].mul(&two_pow_64));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&lhs.sub(&rhs));
            }
            out.push(c);

            // add7: v_c1 + v_d2 - v_c2 - c4·2^64 = 0
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = columns[v_c1][r].add(&columns[v_d2][r]);
                let rhs = columns[v_c2][r]
                    .add(&columns[cc4][r].mul(&two_pow_64));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&lhs.sub(&rhs));
            }
            out.push(c);

            // c1 ∈ {0,1,2}: c1·(c1-1)·(c1-2) = 0
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[cc1][r];
                let body = v.mul(&v.sub(&one)).mul(&v.sub(&two));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&body);
            }
            out.push(c);

            // c2 binary.
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[cc2][r];
                let body = v.mul(&v.sub(&one));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&body);
            }
            out.push(c);

            // c3 ∈ {0,1,2}.
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[cc3][r];
                let body = v.mul(&v.sub(&one)).mul(&v.sub(&two));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&body);
            }
            out.push(c);

            // c4 binary.
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[cc4][r];
                let body = v.mul(&v.sub(&one));
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&body);
            }
            out.push(c);

            // pack v_a2 == next_v[a] (only diagonal phase G4..G7)
            let mut c = vec![Scalar::zero(curve); n];
            if pack_to_next_v {
                for r in 0..n {
                    let pack = pack_next_v_at_row(ai, r);
                    let gate = gate_g(columns, r, &one);
                    c[r] = gate.mul(&columns[v_a2][r].sub(&pack));
                }
            }
            out.push(c);

            // pack v_b2 == next_v[b]
            let mut c = vec![Scalar::zero(curve); n];
            if pack_to_next_v {
                for r in 0..n {
                    let pack = pack_next_v_at_row(bi, r);
                    let gate = gate_g(columns, r, &one);
                    c[r] = gate.mul(&columns[v_b2][r].sub(&pack));
                }
            }
            out.push(c);

            // pack v_c2 == next_v[c]
            let mut c = vec![Scalar::zero(curve); n];
            if pack_to_next_v {
                for r in 0..n {
                    let pack = pack_next_v_at_row(ci, r);
                    let gate = gate_g(columns, r, &one);
                    c[r] = gate.mul(&columns[v_c2][r].sub(&pack));
                }
            }
            out.push(c);

            // pack v_d2 == next_v[d]
            let mut c = vec![Scalar::zero(curve); n];
            if pack_to_next_v {
                for r in 0..n {
                    let pack = pack_next_v_at_row(di, r);
                    let gate = gate_g(columns, r, &one);
                    c[r] = gate.mul(&columns[v_d2][r].sub(&pack));
                }
            }
            out.push(c);
        }

        // ─── XOR/ROTR bit-decomposition layer (32 instances) ──────────
        // Powers of two as Scalars (indexable up to 2^63).
        let mut bit_weights: Vec<Scalar> =
            Vec::with_capacity(XOR_ROTR_BITS_PER_LIMB);
        bit_weights.push(Scalar::one(curve));
        for i in 1..XOR_ROTR_BITS_PER_LIMB {
            bit_weights.push(bit_weights[i - 1].add(&bit_weights[i - 1]));
        }
        let two_scalar = Scalar::from_u64(2, curve);

        // Helper to evaluate an operand on-domain at a given row.
        let operand_at_row = |op: XorRotrOperand, r: usize| -> Scalar {
            match op {
                XorRotrOperand::InputVBytes(v_idx) => pack_v_at_row(v_idx, r),
                XorRotrOperand::Intermediate { g_idx, slot } => {
                    columns[col_g_intermediate(g_idx, slot)][r].clone()
                }
            }
        };

        for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
            let inst = &XOR_ROTR_INSTANCES[inst_idx];
            let a_base = xor_rotr_bit_base(inst_idx, 0);
            let b_base = xor_rotr_bit_base(inst_idx, 1);
            let d_base = xor_rotr_bit_base(inst_idx, 2);

            // 192 binary cell constraints: A bits, B bits, dst bits.
            for blk in 0..XOR_ROTR_BIT_LIMBS_PER_INSTANCE {
                let base = xor_rotr_bit_base(inst_idx, blk);
                for i in 0..XOR_ROTR_BITS_PER_LIMB {
                    let mut c = vec![Scalar::zero(curve); n];
                    for r in 0..n {
                        let v = &columns[base + i][r];
                        let body = v.mul(&v.sub(&one));
                        let gate = gate_g(columns, r, &one);
                        c[r] = gate.mul(&body);
                    }
                    out.push(c);
                }
            }

            // Helper to pack 64 LSB-first bit columns at a row into a Scalar.
            let pack_bits_at_row = |base: usize, r: usize| -> Scalar {
                let mut acc = Scalar::zero(curve);
                for i in 0..XOR_ROTR_BITS_PER_LIMB {
                    acc = acc.add(&columns[base + i][r].mul(&bit_weights[i]));
                }
                acc
            };

            // pack A == reference value
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(a_base, r);
                let target = operand_at_row(inst.a, r);
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // pack B == reference value
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(b_base, r);
                let target = operand_at_row(inst.b, r);
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // pack dst == reference value
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(d_base, r);
                let target = operand_at_row(inst.dst, r);
                let gate = gate_g(columns, r, &one);
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // 64 fused XOR + ROTR_n constraints. ROTR_n(x) maps bit i of
            // x to bit (i - n) mod 64 of the output: bit_dst[j] = bit_x[(j + n) mod 64].
            // Equivalently, bit_dst[(i - n + 64) mod 64] = bit_x[i] for source i.
            //   bit_dst[(i + 64 - rot) mod 64] - (a + b - 2·a·b) = 0
            let rot = inst.rot;
            for i in 0..XOR_ROTR_BITS_PER_LIMB {
                let dst_bit_idx =
                    (i + XOR_ROTR_BITS_PER_LIMB - rot) % XOR_ROTR_BITS_PER_LIMB;
                let mut c = vec![Scalar::zero(curve); n];
                for r in 0..n {
                    let a = &columns[a_base + i][r];
                    let b = &columns[b_base + i][r];
                    let xor = a.add(b).sub(&two_scalar.mul(&a.mul(b)));
                    let dst_bit = &columns[d_base + dst_bit_idx][r];
                    let gate = gate_g(columns, r, &one);
                    c[r] = gate.mul(&dst_bit.sub(&xor));
                }
                out.push(c);
            }
        }

        // ─── σ-permutation layer ──────────────────────────────────────
        let ten = Scalar::from_u64(SIGMA_ROWS as u64, curve);

        // Helper: pack m[v_idx*8 .. v_idx*8+8] little-endian into a Scalar.
        let pack_m_at_row = |m_idx: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_M_OFFSET + m_idx * LIMB_BYTES + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };

        // (1) round_index - 10 * q - round_mod_10 = 0  (ungated).
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let q = &columns[COL_ROUND_QUOTIENT_10][r];
            let m10 = &columns[COL_ROUND_MOD_10][r];
            let ri = &columns[COL_ROUND_INDEX][r];
            c[r] = ri.sub(&ten.mul(q)).sub(m10);
        }
        out.push(c);

        // (2) q · (q - 1) = 0, gated by `is_real * (1 - is_result_row)`.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let q = &columns[COL_ROUND_QUOTIENT_10][r];
            let body = q.mul(&q.sub(&one));
            let gate = gate_g(columns, r, &one);
            c[r] = gate.mul(&body);
        }
        out.push(c);

        // (3) is_sigma_row[r] binary (ungated, 10 constraints).
        for sr in 0..SIGMA_ROWS {
            let col = col_is_sigma_row(sr);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[col][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // (4) Σ is_sigma_row[r] - is_real*(1 - is_result_row) = 0 (ungated).
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let mut sum = Scalar::zero(curve);
            for sr in 0..SIGMA_ROWS {
                sum = sum.add(&columns[col_is_sigma_row(sr)][r]);
            }
            let gate = gate_g(columns, r, &one);
            c[r] = sum.sub(&gate);
        }
        out.push(c);

        // (5) gate * (Σ r·is_sigma_row[r] - round_mod_10) = 0.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let mut sum = Scalar::zero(curve);
            for sr in 0..SIGMA_ROWS {
                let w = Scalar::from_u64(sr as u64, curve);
                sum = sum.add(&columns[col_is_sigma_row(sr)][r].mul(&w));
            }
            let body = sum.sub(&columns[COL_ROUND_MOD_10][r]);
            let gate = gate_g(columns, r, &one);
            c[r] = gate.mul(&body);
        }
        out.push(c);

        // (6) For each (σ-row sr, m-slot k):
        //     is_sigma_row[sr] · (m_sched[k] - pack_m(SIGMA[sr][k])) = 0.
        // The selector is binary and zero on result/padding rows, so this
        // self-gates without any explicit is_real / (1-is_result_row) factor.
        for sr in 0..SIGMA_ROWS {
            let sel_col = col_is_sigma_row(sr);
            for k in 0..M_SCHED_LIMBS {
                let m_sched_col = col_m_sched(k);
                let m_src_idx = SIGMA[sr][k];
                let mut c = vec![Scalar::zero(curve); n];
                for r in 0..n {
                    let m_target = pack_m_at_row(m_src_idx, r);
                    let body = columns[m_sched_col][r].sub(&m_target);
                    let sel = &columns[sel_col][r];
                    c[r] = sel.mul(&body);
                }
                out.push(c);
            }
        }

        // ─── IV-init / final h_out finalize layer (Task #206) ─────────
        //
        // All constraints are gated by `is_first_round` and act at the
        // packed u64 level. The XOR-based bindings for v[12..15] and the
        // final h_out chain are deferred (see NUM_IV_INIT_CONSTRAINTS).
        let pack_h_in_at_row = |k: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_H_IN_OFFSET + k * LIMB_BYTES + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let iv_scalar = |i: usize| -> Scalar {
            let lo = (BLAKE2B_IV[i] & 0xffff_ffffu64) as u64;
            let hi = (BLAKE2B_IV[i] >> 32) as u64;
            let lo_s = Scalar::from_u64(lo, curve);
            let hi_s = Scalar::from_u64(hi, curve);
            let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
            lo_s.add(&hi_s.mul(&two_pow_32))
        };

        // (0) is_first_round binary.
        let mut c = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_FIRST_ROUND][r];
            c[r] = v.mul(&v.sub(&one));
        }
        out.push(c);

        // (1..8) is_first_round * (pack_v(v[i]) - pack_h_in[i]) = 0 for i ∈ 0..8.
        for i in 0..8 {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = pack_v_at_row(i, r);
                let rhs = pack_h_in_at_row(i, r);
                let gate = &columns[COL_IS_FIRST_ROUND][r];
                c[r] = gate.mul(&lhs.sub(&rhs));
            }
            out.push(c);
        }

        // (9..12) is_first_round * (pack_v(v[i+8]) - IV[i]) = 0 for i ∈ 0..4.
        for i in 0..4 {
            let iv_s = iv_scalar(i);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = pack_v_at_row(i + 8, r);
                let gate = &columns[COL_IS_FIRST_ROUND][r];
                c[r] = gate.mul(&lhs.sub(&iv_s));
            }
            out.push(c);
        }

        // (13) is_first_round * (pack_v(v[15]) - IV[7]) = 0.
        let mut c = vec![Scalar::zero(curve); n];
        let iv7 = iv_scalar(7);
        for r in 0..n {
            let lhs = pack_v_at_row(15, r);
            let gate = &columns[COL_IS_FIRST_ROUND][r];
            c[r] = gate.mul(&lhs.sub(&iv7));
        }
        out.push(c);

        // ─── IV XOR bit-decomposition layer (Task #215) ───────────────
        // 3 instances closing v[12]=IV[4]^t.lo, v[13]=IV[5]^t.hi,
        // v[14]=IV[6]^f_byte_mask. All gated by `is_first_round`.
        let f_byte_mask_scalar = {
            // 0xFFFFFFFFFFFFFFFF = (2^64) - 1. Build it as
            // (2^32 - 1) * 2^32 + (2^32 - 1) to avoid u64 wrap.
            let lo_pow = Scalar::from_u64(0xFFFFFFFFu64, curve);
            let hi_pow = Scalar::from_u64(0xFFFFFFFFu64, curve);
            let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
            lo_pow.add(&hi_pow.mul(&two_pow_32))
        };
        let pack_t_at_row = |off: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_T_OFFSET + off + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let pack_v_target = |v_idx: usize, r: usize| -> Scalar {
            pack_v_at_row(v_idx, r)
        };
        for inst_idx in 0..IV_XOR_INSTANCE_COUNT {
            let iv_idx = IV_XOR_IV_INDEX[inst_idx];
            let v_idx = IV_XOR_V_INDEX[inst_idx];
            let a_base = iv_xor_bit_base(inst_idx, 0);
            let b_base = iv_xor_bit_base(inst_idx, 1);
            let d_base = iv_xor_bit_base(inst_idx, 2);
            let iv_const = iv_scalar(iv_idx);
            let b_target_at_row = |r: usize| -> Scalar {
                match IV_XOR_B_SOURCE[inst_idx] {
                    IvXorBSource::TBytes(off) => pack_t_at_row(off, r),
                    IvXorBSource::FByteMask => {
                        columns[COL_F_BYTE][r].mul(&f_byte_mask_scalar)
                    }
                }
            };

            // 192 binary cell constraints (A, B, dst bits), gated by is_first_round.
            for blk in 0..IV_XOR_BIT_LIMBS_PER_INSTANCE {
                let base = iv_xor_bit_base(inst_idx, blk);
                for i in 0..IV_XOR_BITS_PER_LIMB {
                    let mut c = vec![Scalar::zero(curve); n];
                    for r in 0..n {
                        let v = &columns[base + i][r];
                        let body = v.mul(&v.sub(&one));
                        let gate = &columns[COL_IS_FIRST_ROUND][r];
                        c[r] = gate.mul(&body);
                    }
                    out.push(c);
                }
            }

            // pack helpers.
            let pack_bits_at_row = |base: usize, r: usize| -> Scalar {
                let mut acc = Scalar::zero(curve);
                for i in 0..IV_XOR_BITS_PER_LIMB {
                    acc = acc.add(&columns[base + i][r].mul(&bit_weights[i]));
                }
                acc
            };

            // pack A == IV[iv_idx]
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(a_base, r);
                let gate = &columns[COL_IS_FIRST_ROUND][r];
                c[r] = gate.mul(&packed.sub(&iv_const));
            }
            out.push(c);

            // pack B == target u64 value
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(b_base, r);
                let target = b_target_at_row(r);
                let gate = &columns[COL_IS_FIRST_ROUND][r];
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // pack dst == pack_v(v_idx)
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(d_base, r);
                let target = pack_v_target(v_idx, r);
                let gate = &columns[COL_IS_FIRST_ROUND][r];
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // 64 XOR (rot=0) constraints: dst_bit[i] = A_bit[i] XOR B_bit[i].
            for i in 0..IV_XOR_BITS_PER_LIMB {
                let mut c = vec![Scalar::zero(curve); n];
                for r in 0..n {
                    let a = &columns[a_base + i][r];
                    let b = &columns[b_base + i][r];
                    let xor = a.add(b).sub(&two_scalar.mul(&a.mul(b)));
                    let dst_bit = &columns[d_base + i][r];
                    let gate = &columns[COL_IS_FIRST_ROUND][r];
                    c[r] = gate.mul(&dst_bit.sub(&xor));
                }
                out.push(c);
            }
        }

        // ─── h_out finalize triple-XOR layer (Task #216) ──────────────
        // 8 instances closing `result_h[i] = h_in[i] XOR v_final[i]
        // XOR v_final[i+8]` at the per-bit level, gated by
        // `is_result_row` (= is_final_round). Each instance commits
        // 4 × 64 LSB-first bit columns + 256 binary + 4 pack + 64 XOR3.
        let four_scalar = Scalar::from_u64(4, curve);
        let pack_h_in_limb_at_row = |i: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_H_IN_OFFSET + i * LIMB_BYTES + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let pack_v_final_limb_at_row = |i: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_V_FINAL_OFFSET + i * LIMB_BYTES + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let pack_result_limb_at_row = |i: usize, r: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &columns[COL_RESULT_OFFSET + i * LIMB_BYTES + b][r];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        for inst_idx in 0..H_OUT_XOR_INSTANCE_COUNT {
            let h_in_base = h_out_xor_bit_base(inst_idx, 0);
            let v_lo_base = h_out_xor_bit_base(inst_idx, 1);
            let v_hi_base = h_out_xor_bit_base(inst_idx, 2);
            let h_out_base = h_out_xor_bit_base(inst_idx, 3);

            // 256 binary cell constraints (4 × 64 bits), gated by is_result_row.
            for blk in 0..H_OUT_XOR_BIT_LIMBS_PER_INSTANCE {
                let base = h_out_xor_bit_base(inst_idx, blk);
                for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                    let mut c = vec![Scalar::zero(curve); n];
                    for r in 0..n {
                        let v = &columns[base + i][r];
                        let body = v.mul(&v.sub(&one));
                        let gate = &columns[COL_IS_RESULT_ROW][r];
                        c[r] = gate.mul(&body);
                    }
                    out.push(c);
                }
            }

            let pack_bits_at_row = |base: usize, r: usize| -> Scalar {
                let mut acc = Scalar::zero(curve);
                for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                    acc = acc.add(&columns[base + i][r].mul(&bit_weights[i]));
                }
                acc
            };

            // pack h_in: Σ 2^b · bit_h_in[b] == packed(h_in[i]).
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(h_in_base, r);
                let target = pack_h_in_limb_at_row(inst_idx, r);
                let gate = &columns[COL_IS_RESULT_ROW][r];
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // pack v_lo: Σ 2^b · bit_v_lo[b] == packed(v_final[i]).
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(v_lo_base, r);
                let target = pack_v_final_limb_at_row(inst_idx, r);
                let gate = &columns[COL_IS_RESULT_ROW][r];
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // pack v_hi: Σ 2^b · bit_v_hi[b] == packed(v_final[i+8]).
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(v_hi_base, r);
                let target = pack_v_final_limb_at_row(inst_idx + 8, r);
                let gate = &columns[COL_IS_RESULT_ROW][r];
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // pack h_out: Σ 2^b · bit_h_out[b] == packed(result_h[i]).
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let packed = pack_bits_at_row(h_out_base, r);
                let target = pack_result_limb_at_row(inst_idx, r);
                let gate = &columns[COL_IS_RESULT_ROW][r];
                c[r] = gate.mul(&packed.sub(&target));
            }
            out.push(c);

            // 64 fused XOR-of-3 constraints:
            //   h_out_bit[i] = h_in_bit + v_lo_bit + v_hi_bit
            //                - 2·(h·l + h·h2 + l·h2)
            //                + 4·(h · l · h2)
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                let mut c = vec![Scalar::zero(curve); n];
                for r in 0..n {
                    let h = &columns[h_in_base + i][r];
                    let l = &columns[v_lo_base + i][r];
                    let h2 = &columns[v_hi_base + i][r];
                    let h_l = h.mul(l);
                    let h_h2 = h.mul(h2);
                    let l_h2 = l.mul(h2);
                    let h_l_h2 = h_l.mul(h2);
                    let sum_lin = h.add(l).add(h2);
                    let pair_sum = h_l.add(&h_h2).add(&l_h2);
                    let two_pair = two_scalar.mul(&pair_sum);
                    let four_trip = four_scalar.mul(&h_l_h2);
                    let xor3 = sum_lin.sub(&two_pair).add(&four_trip);
                    let h_out_bit = &columns[h_out_base + i][r];
                    let gate = &columns[COL_IS_RESULT_ROW][r];
                    c[r] = gate.mul(&h_out_bit.sub(&xor3));
                }
                out.push(c);
            }
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

        let v = &col_evals[COL_F_BYTE];
        acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
        ap = ap.mul(alpha);

        let rr = &col_evals[COL_IS_RESULT_ROW];
        let real = &col_evals[COL_IS_REAL];
        acc = acc.add(&ap.mul(&rr.mul(&one.sub(real))));
        ap = ap.mul(alpha);

        let mut sum = Scalar::zero(curve);
        for b in 0..ROUND_INDEX_BYTES {
            let w = Scalar::from_u64(1u64 << (8 * b), curve);
            sum = sum.add(&col_evals[COL_ROUND_INDEX_BYTE_OFFSET + b].mul(&w));
        }
        acc = acc.add(&ap.mul(&sum.sub(&col_evals[COL_ROUND_INDEX])));
        ap = ap.mul(alpha);

        let mut sum = Scalar::zero(curve);
        for i in 0..RESULT_LENGTH {
            sum = sum.add(&col_evals[COL_RESULT_OFFSET + i]);
        }
        let gate = one.sub(&col_evals[COL_IS_RESULT_ROW]);
        acc = acc.add(&ap.mul(&gate.mul(&sum)));
        ap = ap.mul(alpha);

        // G layer.
        let two_pow_64 = two_pow_64_scalar(curve);
        let two = Scalar::from_u64(2, curve);
        let limb_byte_weights: Vec<Scalar> = (0..LIMB_BYTES)
            .map(|b| Scalar::from_u64(1u64 << (8 * b), curve))
            .collect();
        let pack_v_at_point = |v_idx: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_V_OFFSET + v_idx * LIMB_BYTES + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let pack_next_v_at_point = |v_idx: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_NEXT_V_OFFSET + v_idx * LIMB_BYTES + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let gate = col_evals[COL_IS_REAL]
            .mul(&one.sub(&col_evals[COL_IS_RESULT_ROW]));

        let g_input_at_point = |g_idx: usize, slot: usize| -> Scalar {
            if let Some(spec) = G_INPUT_FROM_INTERMEDIATE[g_idx] {
                let (g_src, s_src) = spec[slot];
                col_evals[col_g_intermediate(g_src, s_src)].clone()
            } else {
                pack_v_at_point(V_MIX_INDICES[g_idx][slot])
            }
        };

        for g_idx in 0..G_PER_ROUND {
            let ai = V_MIX_INDICES[g_idx][0];
            let bi = V_MIX_INDICES[g_idx][1];
            let ci = V_MIX_INDICES[g_idx][2];
            let di = V_MIX_INDICES[g_idx][3];
            let m_i = &col_evals[col_m_sched(g_msched_i(g_idx))];
            let m_j = &col_evals[col_m_sched(g_msched_j(g_idx))];
            let v_a1 = &col_evals[col_g_intermediate(g_idx, 0)];
            let v_d1 = &col_evals[col_g_intermediate(g_idx, 1)];
            let v_c1 = &col_evals[col_g_intermediate(g_idx, 2)];
            let v_b1 = &col_evals[col_g_intermediate(g_idx, 3)];
            let v_a2 = &col_evals[col_g_intermediate(g_idx, 4)];
            let v_d2 = &col_evals[col_g_intermediate(g_idx, 5)];
            let v_c2 = &col_evals[col_g_intermediate(g_idx, 6)];
            let v_b2 = &col_evals[col_g_intermediate(g_idx, 7)];
            let cc1 = &col_evals[col_g_carry(g_idx, 0)];
            let cc2 = &col_evals[col_g_carry(g_idx, 1)];
            let cc3 = &col_evals[col_g_carry(g_idx, 2)];
            let cc4 = &col_evals[col_g_carry(g_idx, 3)];
            let pack_to_next_v = g_idx >= 4;
            let zero_s = Scalar::zero(curve);

            // add1
            let lhs = g_input_at_point(g_idx, 0)
                .add(&g_input_at_point(g_idx, 1))
                .add(m_i);
            let rhs = v_a1.add(&cc1.mul(&two_pow_64));
            acc = acc.add(&ap.mul(&gate.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
            // add3
            let lhs = g_input_at_point(g_idx, 2).add(v_d1);
            let rhs = v_c1.add(&cc2.mul(&two_pow_64));
            acc = acc.add(&ap.mul(&gate.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
            // add5
            let lhs = v_a1.add(v_b1).add(m_j);
            let rhs = v_a2.add(&cc3.mul(&two_pow_64));
            acc = acc.add(&ap.mul(&gate.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
            // add7
            let lhs = v_c1.add(v_d2);
            let rhs = v_c2.add(&cc4.mul(&two_pow_64));
            acc = acc.add(&ap.mul(&gate.mul(&lhs.sub(&rhs))));
            ap = ap.mul(alpha);
            // c1 range
            let body = cc1.mul(&cc1.sub(&one)).mul(&cc1.sub(&two));
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // c2 bin
            let body = cc2.mul(&cc2.sub(&one));
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // c3 range
            let body = cc3.mul(&cc3.sub(&one)).mul(&cc3.sub(&two));
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // c4 bin
            let body = cc4.mul(&cc4.sub(&one));
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // pack v_a2 (G4..G7 only)
            let body = if pack_to_next_v {
                v_a2.sub(&pack_next_v_at_point(ai))
            } else {
                zero_s.clone()
            };
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // pack v_b2
            let body = if pack_to_next_v {
                v_b2.sub(&pack_next_v_at_point(bi))
            } else {
                zero_s.clone()
            };
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // pack v_c2
            let body = if pack_to_next_v {
                v_c2.sub(&pack_next_v_at_point(ci))
            } else {
                zero_s.clone()
            };
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // pack v_d2
            let body = if pack_to_next_v {
                v_d2.sub(&pack_next_v_at_point(di))
            } else {
                zero_s.clone()
            };
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
        }

        // ─── XOR/ROTR bit-decomposition layer (32 instances) ──────────
        let mut bit_weights: Vec<Scalar> =
            Vec::with_capacity(XOR_ROTR_BITS_PER_LIMB);
        bit_weights.push(Scalar::one(curve));
        for i in 1..XOR_ROTR_BITS_PER_LIMB {
            bit_weights.push(bit_weights[i - 1].add(&bit_weights[i - 1]));
        }
        let two_scalar = Scalar::from_u64(2, curve);

        let operand_at_point = |op: XorRotrOperand| -> Scalar {
            match op {
                XorRotrOperand::InputVBytes(v_idx) => pack_v_at_point(v_idx),
                XorRotrOperand::Intermediate { g_idx, slot } => {
                    col_evals[col_g_intermediate(g_idx, slot)].clone()
                }
            }
        };

        for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
            let inst = &XOR_ROTR_INSTANCES[inst_idx];
            let a_base = xor_rotr_bit_base(inst_idx, 0);
            let b_base = xor_rotr_bit_base(inst_idx, 1);
            let d_base = xor_rotr_bit_base(inst_idx, 2);

            // 192 binary cell constraints.
            for blk in 0..XOR_ROTR_BIT_LIMBS_PER_INSTANCE {
                let base = xor_rotr_bit_base(inst_idx, blk);
                for i in 0..XOR_ROTR_BITS_PER_LIMB {
                    let v = &col_evals[base + i];
                    let body = v.mul(&v.sub(&one));
                    acc = acc.add(&ap.mul(&gate.mul(&body)));
                    ap = ap.mul(alpha);
                }
            }

            let pack_bits_at_point = |base: usize| -> Scalar {
                let mut acc = Scalar::zero(curve);
                for i in 0..XOR_ROTR_BITS_PER_LIMB {
                    acc = acc.add(&col_evals[base + i].mul(&bit_weights[i]));
                }
                acc
            };
            // pack A
            let body = pack_bits_at_point(a_base).sub(&operand_at_point(inst.a));
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // pack B
            let body = pack_bits_at_point(b_base).sub(&operand_at_point(inst.b));
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
            // pack dst
            let body = pack_bits_at_point(d_base).sub(&operand_at_point(inst.dst));
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);

            // XOR + ROTR_n fused. See evaluate_on_domain for derivation:
            // bit_dst[(i + 64 - rot) mod 64] = bit_A[i] XOR bit_B[i].
            let rot = inst.rot;
            for i in 0..XOR_ROTR_BITS_PER_LIMB {
                let dst_bit_idx =
                    (i + XOR_ROTR_BITS_PER_LIMB - rot) % XOR_ROTR_BITS_PER_LIMB;
                let a = &col_evals[a_base + i];
                let b = &col_evals[b_base + i];
                let xor = a.add(b).sub(&two_scalar.mul(&a.mul(b)));
                let dst_bit = &col_evals[d_base + dst_bit_idx];
                let body = dst_bit.sub(&xor);
                acc = acc.add(&ap.mul(&gate.mul(&body)));
                ap = ap.mul(alpha);
            }
        }

        // ─── σ-permutation layer (point eval) ─────────────────────────
        let ten = Scalar::from_u64(SIGMA_ROWS as u64, curve);

        let pack_m_at_point = |m_idx: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_M_OFFSET + m_idx * LIMB_BYTES + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };

        // (1) round_index - 10*q - round_mod_10 = 0 (ungated).
        let q = &col_evals[COL_ROUND_QUOTIENT_10];
        let m10 = &col_evals[COL_ROUND_MOD_10];
        let ri = &col_evals[COL_ROUND_INDEX];
        let body = ri.sub(&ten.mul(q)).sub(m10);
        acc = acc.add(&ap.mul(&body));
        ap = ap.mul(alpha);

        // (2) q binary, gated.
        let body = q.mul(&q.sub(&one));
        acc = acc.add(&ap.mul(&gate.mul(&body)));
        ap = ap.mul(alpha);

        // (3) is_sigma_row[r] binary, ungated.
        for sr in 0..SIGMA_ROWS {
            let v = &col_evals[col_is_sigma_row(sr)];
            let body = v.mul(&v.sub(&one));
            acc = acc.add(&ap.mul(&body));
            ap = ap.mul(alpha);
        }

        // (4) Σ is_sigma_row[r] - gate = 0 (ungated).
        let mut sum = Scalar::zero(curve);
        for sr in 0..SIGMA_ROWS {
            sum = sum.add(&col_evals[col_is_sigma_row(sr)]);
        }
        let body = sum.sub(&gate);
        acc = acc.add(&ap.mul(&body));
        ap = ap.mul(alpha);

        // (5) gate · (Σ r · is_sigma_row[r] - round_mod_10) = 0.
        let mut sum = Scalar::zero(curve);
        for sr in 0..SIGMA_ROWS {
            let w = Scalar::from_u64(sr as u64, curve);
            sum = sum.add(&col_evals[col_is_sigma_row(sr)].mul(&w));
        }
        let body = sum.sub(m10);
        acc = acc.add(&ap.mul(&gate.mul(&body)));
        ap = ap.mul(alpha);

        // (6) σ-binding 160 constraints.
        for sr in 0..SIGMA_ROWS {
            let sel = &col_evals[col_is_sigma_row(sr)];
            for k in 0..M_SCHED_LIMBS {
                let m_sched = &col_evals[col_m_sched(k)];
                let m_src_idx = SIGMA[sr][k];
                let body = m_sched.sub(&pack_m_at_point(m_src_idx));
                acc = acc.add(&ap.mul(&sel.mul(&body)));
                ap = ap.mul(alpha);
            }
        }

        // ─── IV-init / final h_out finalize layer (Task #206) ─────────
        let pack_h_in_at_point = |k: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_H_IN_OFFSET + k * LIMB_BYTES + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let iv_scalar_pt = |i: usize| -> Scalar {
            let lo = (BLAKE2B_IV[i] & 0xffff_ffffu64) as u64;
            let hi = (BLAKE2B_IV[i] >> 32) as u64;
            let lo_s = Scalar::from_u64(lo, curve);
            let hi_s = Scalar::from_u64(hi, curve);
            let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
            lo_s.add(&hi_s.mul(&two_pow_32))
        };
        let first_round = &col_evals[COL_IS_FIRST_ROUND];

        // (0) is_first_round binary.
        let body = first_round.mul(&first_round.sub(&one));
        acc = acc.add(&ap.mul(&body));
        ap = ap.mul(alpha);

        // (1..8) is_first_round * (pack_v(v[i]) - pack_h_in[i]) = 0.
        for i in 0..8 {
            let lhs = pack_v_at_point(i);
            let rhs = pack_h_in_at_point(i);
            let body = first_round.mul(&lhs.sub(&rhs));
            acc = acc.add(&ap.mul(&body));
            ap = ap.mul(alpha);
        }

        // (9..12) is_first_round * (pack_v(v[i+8]) - IV[i]) = 0 for i ∈ 0..4.
        for i in 0..4 {
            let lhs = pack_v_at_point(i + 8);
            let iv_s = iv_scalar_pt(i);
            let body = first_round.mul(&lhs.sub(&iv_s));
            acc = acc.add(&ap.mul(&body));
            ap = ap.mul(alpha);
        }

        // (13) is_first_round * (pack_v(v[15]) - IV[7]) = 0.
        let lhs = pack_v_at_point(15);
        let iv7 = iv_scalar_pt(7);
        let body = first_round.mul(&lhs.sub(&iv7));
        acc = acc.add(&ap.mul(&body));
        ap = ap.mul(alpha);

        // ─── IV XOR bit-decomposition layer (Task #215) ───────────────
        let f_byte_mask_scalar_pt = {
            let lo_pow = Scalar::from_u64(0xFFFFFFFFu64, curve);
            let hi_pow = Scalar::from_u64(0xFFFFFFFFu64, curve);
            let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
            lo_pow.add(&hi_pow.mul(&two_pow_32))
        };
        let pack_t_at_point = |off: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_T_OFFSET + off + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        for inst_idx in 0..IV_XOR_INSTANCE_COUNT {
            let iv_idx = IV_XOR_IV_INDEX[inst_idx];
            let v_idx = IV_XOR_V_INDEX[inst_idx];
            let a_base = iv_xor_bit_base(inst_idx, 0);
            let b_base = iv_xor_bit_base(inst_idx, 1);
            let d_base = iv_xor_bit_base(inst_idx, 2);
            let iv_const = iv_scalar_pt(iv_idx);
            let b_target = match IV_XOR_B_SOURCE[inst_idx] {
                IvXorBSource::TBytes(off) => pack_t_at_point(off),
                IvXorBSource::FByteMask => {
                    col_evals[COL_F_BYTE].mul(&f_byte_mask_scalar_pt)
                }
            };
            let dst_target = pack_v_at_point(v_idx);

            // 192 binary cell constraints.
            for blk in 0..IV_XOR_BIT_LIMBS_PER_INSTANCE {
                let base = iv_xor_bit_base(inst_idx, blk);
                for i in 0..IV_XOR_BITS_PER_LIMB {
                    let v = &col_evals[base + i];
                    let body = v.mul(&v.sub(&one));
                    acc = acc.add(&ap.mul(&first_round.mul(&body)));
                    ap = ap.mul(alpha);
                }
            }

            let pack_bits_at_point = |base: usize| -> Scalar {
                let mut acc = Scalar::zero(curve);
                for i in 0..IV_XOR_BITS_PER_LIMB {
                    acc = acc.add(&col_evals[base + i].mul(&bit_weights[i]));
                }
                acc
            };

            // pack A
            let body = pack_bits_at_point(a_base).sub(&iv_const);
            acc = acc.add(&ap.mul(&first_round.mul(&body)));
            ap = ap.mul(alpha);
            // pack B
            let body = pack_bits_at_point(b_base).sub(&b_target);
            acc = acc.add(&ap.mul(&first_round.mul(&body)));
            ap = ap.mul(alpha);
            // pack dst
            let body = pack_bits_at_point(d_base).sub(&dst_target);
            acc = acc.add(&ap.mul(&first_round.mul(&body)));
            ap = ap.mul(alpha);

            // 64 XOR (rot=0).
            for i in 0..IV_XOR_BITS_PER_LIMB {
                let a = &col_evals[a_base + i];
                let b = &col_evals[b_base + i];
                let xor = a.add(b).sub(&two_scalar.mul(&a.mul(b)));
                let dst_bit = &col_evals[d_base + i];
                let body = dst_bit.sub(&xor);
                acc = acc.add(&ap.mul(&first_round.mul(&body)));
                ap = ap.mul(alpha);
            }
        }

        // ─── h_out finalize triple-XOR layer (Task #216, point eval) ──
        let four_scalar = Scalar::from_u64(4, curve);
        let is_result = &col_evals[COL_IS_RESULT_ROW];
        let pack_h_in_limb_at_point = |i: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_H_IN_OFFSET + i * LIMB_BYTES + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let pack_v_final_limb_at_point = |i: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_V_FINAL_OFFSET + i * LIMB_BYTES + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        let pack_result_limb_at_point = |i: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            for b in 0..LIMB_BYTES {
                let byte = &col_evals[COL_RESULT_OFFSET + i * LIMB_BYTES + b];
                acc = acc.add(&byte.mul(&limb_byte_weights[b]));
            }
            acc
        };
        for inst_idx in 0..H_OUT_XOR_INSTANCE_COUNT {
            let h_in_base = h_out_xor_bit_base(inst_idx, 0);
            let v_lo_base = h_out_xor_bit_base(inst_idx, 1);
            let v_hi_base = h_out_xor_bit_base(inst_idx, 2);
            let h_out_base = h_out_xor_bit_base(inst_idx, 3);

            // 256 binary cell constraints.
            for blk in 0..H_OUT_XOR_BIT_LIMBS_PER_INSTANCE {
                let base = h_out_xor_bit_base(inst_idx, blk);
                for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                    let v = &col_evals[base + i];
                    let body = v.mul(&v.sub(&one));
                    acc = acc.add(&ap.mul(&is_result.mul(&body)));
                    ap = ap.mul(alpha);
                }
            }

            let pack_bits_at_point = |base: usize| -> Scalar {
                let mut acc = Scalar::zero(curve);
                for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                    acc = acc.add(&col_evals[base + i].mul(&bit_weights[i]));
                }
                acc
            };

            // pack h_in
            let body = pack_bits_at_point(h_in_base)
                .sub(&pack_h_in_limb_at_point(inst_idx));
            acc = acc.add(&ap.mul(&is_result.mul(&body)));
            ap = ap.mul(alpha);
            // pack v_lo
            let body = pack_bits_at_point(v_lo_base)
                .sub(&pack_v_final_limb_at_point(inst_idx));
            acc = acc.add(&ap.mul(&is_result.mul(&body)));
            ap = ap.mul(alpha);
            // pack v_hi
            let body = pack_bits_at_point(v_hi_base)
                .sub(&pack_v_final_limb_at_point(inst_idx + 8));
            acc = acc.add(&ap.mul(&is_result.mul(&body)));
            ap = ap.mul(alpha);
            // pack h_out
            let body = pack_bits_at_point(h_out_base)
                .sub(&pack_result_limb_at_point(inst_idx));
            acc = acc.add(&ap.mul(&is_result.mul(&body)));
            ap = ap.mul(alpha);

            // 64 fused XOR-of-3 constraints.
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                let h = &col_evals[h_in_base + i];
                let l = &col_evals[v_lo_base + i];
                let h2 = &col_evals[v_hi_base + i];
                let h_l = h.mul(l);
                let h_h2 = h.mul(h2);
                let l_h2 = l.mul(h2);
                let h_l_h2 = h_l.mul(h2);
                let sum_lin = h.add(l).add(h2);
                let pair_sum = h_l.add(&h_h2).add(&l_h2);
                let two_pair = two_scalar.mul(&pair_sum);
                let four_trip = four_scalar.mul(&h_l_h2);
                let xor3 = sum_lin.sub(&two_pair).add(&four_trip);
                let h_out_bit = &col_evals[h_out_base + i];
                let body = h_out_bit.sub(&xor3);
                acc = acc.add(&ap.mul(&is_result.mul(&body)));
                ap = ap.mul(alpha);
            }
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

        let v = &col_coeffs[COL_F_BYTE];
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

        let mut sum = vec![Scalar::zero(curve)];
        for i in 0..RESULT_LENGTH {
            sum = poly_add(&sum, &col_coeffs[COL_RESULT_OFFSET + i], curve);
        }
        let gate_result = poly_sub(&one_poly, &col_coeffs[COL_IS_RESULT_ROW], curve);
        let body = poly_mul(&gate_result, &sum, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // G layer.
        let two_pow_64 = two_pow_64_scalar(curve);
        let two = Scalar::from_u64(2, curve);
        let limb_byte_weights: Vec<Scalar> = (0..LIMB_BYTES)
            .map(|b| Scalar::from_u64(1u64 << (8 * b), curve))
            .collect();
        let pack_v_poly = |v_idx: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp = &col_coeffs[COL_V_OFFSET + v_idx * LIMB_BYTES + b];
                acc = poly_add(&acc, &poly_scalar_mul(bp, &limb_byte_weights[b]), curve);
            }
            acc
        };
        let pack_next_v_poly = |v_idx: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp = &col_coeffs[COL_NEXT_V_OFFSET + v_idx * LIMB_BYTES + b];
                acc = poly_add(&acc, &poly_scalar_mul(bp, &limb_byte_weights[b]), curve);
            }
            acc
        };
        // gate = is_real * (1 - is_result_row)
        let one_minus_result =
            poly_sub(&one_poly, &col_coeffs[COL_IS_RESULT_ROW], curve);
        let gate_poly =
            poly_mul(&col_coeffs[COL_IS_REAL], &one_minus_result, curve);

        let g_input_poly = |g_idx: usize, slot: usize| -> Vec<Scalar> {
            if let Some(spec) = G_INPUT_FROM_INTERMEDIATE[g_idx] {
                let (g_src, s_src) = spec[slot];
                col_coeffs[col_g_intermediate(g_src, s_src)].clone()
            } else {
                pack_v_poly(V_MIX_INDICES[g_idx][slot])
            }
        };

        for g_idx in 0..G_PER_ROUND {
            let ai = V_MIX_INDICES[g_idx][0];
            let bi = V_MIX_INDICES[g_idx][1];
            let ci = V_MIX_INDICES[g_idx][2];
            let di = V_MIX_INDICES[g_idx][3];
            let m_i = &col_coeffs[col_m_sched(g_msched_i(g_idx))];
            let m_j = &col_coeffs[col_m_sched(g_msched_j(g_idx))];
            let v_a1 = &col_coeffs[col_g_intermediate(g_idx, 0)];
            let v_d1 = &col_coeffs[col_g_intermediate(g_idx, 1)];
            let v_c1 = &col_coeffs[col_g_intermediate(g_idx, 2)];
            let v_b1 = &col_coeffs[col_g_intermediate(g_idx, 3)];
            let v_a2 = &col_coeffs[col_g_intermediate(g_idx, 4)];
            let v_d2 = &col_coeffs[col_g_intermediate(g_idx, 5)];
            let v_c2 = &col_coeffs[col_g_intermediate(g_idx, 6)];
            let v_b2 = &col_coeffs[col_g_intermediate(g_idx, 7)];
            let cc1 = &col_coeffs[col_g_carry(g_idx, 0)];
            let cc2 = &col_coeffs[col_g_carry(g_idx, 1)];
            let cc3 = &col_coeffs[col_g_carry(g_idx, 2)];
            let cc4 = &col_coeffs[col_g_carry(g_idx, 3)];
            let pack_to_next_v = g_idx >= 4;
            let zero_body = vec![Scalar::zero(curve)];

            // Helper to fold a (gate * body) under current ap.
            let push = |body: Vec<Scalar>, ap_local: &Scalar| -> Vec<Scalar> {
                let term = poly_mul(&gate_poly, &body, curve);
                poly_scalar_mul(&term, ap_local)
            };

            // add1
            let v_a_p = g_input_poly(g_idx, 0);
            let v_b_p = g_input_poly(g_idx, 1);
            let lhs = poly_add(&poly_add(&v_a_p, &v_b_p, curve), m_i, curve);
            let rhs = poly_add(v_a1, &poly_scalar_mul(cc1, &two_pow_64), curve);
            let body = poly_sub(&lhs, &rhs, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // add3
            let v_c_p = g_input_poly(g_idx, 2);
            let lhs = poly_add(&v_c_p, v_d1, curve);
            let rhs = poly_add(v_c1, &poly_scalar_mul(cc2, &two_pow_64), curve);
            let body = poly_sub(&lhs, &rhs, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // add5
            let lhs = poly_add(&poly_add(v_a1, v_b1, curve), m_j, curve);
            let rhs = poly_add(v_a2, &poly_scalar_mul(cc3, &two_pow_64), curve);
            let body = poly_sub(&lhs, &rhs, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // add7
            let lhs = poly_add(v_c1, v_d2, curve);
            let rhs = poly_add(v_c2, &poly_scalar_mul(cc4, &two_pow_64), curve);
            let body = poly_sub(&lhs, &rhs, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // c1 range cubic
            let two_poly = vec![two.clone()];
            let cc1_m1 = poly_sub(cc1, &one_poly, curve);
            let cc1_m2 = poly_sub(cc1, &two_poly, curve);
            let body = poly_mul(&poly_mul(cc1, &cc1_m1, curve), &cc1_m2, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // c2 binary
            let cc2_m1 = poly_sub(cc2, &one_poly, curve);
            let body = poly_mul(cc2, &cc2_m1, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // c3 range cubic
            let cc3_m1 = poly_sub(cc3, &one_poly, curve);
            let cc3_m2 = poly_sub(cc3, &two_poly, curve);
            let body = poly_mul(&poly_mul(cc3, &cc3_m1, curve), &cc3_m2, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // c4 binary
            let cc4_m1 = poly_sub(cc4, &one_poly, curve);
            let body = poly_mul(cc4, &cc4_m1, curve);
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack v_a2
            let body = if pack_to_next_v {
                poly_sub(v_a2, &pack_next_v_poly(ai), curve)
            } else {
                zero_body.clone()
            };
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack v_b2
            let body = if pack_to_next_v {
                poly_sub(v_b2, &pack_next_v_poly(bi), curve)
            } else {
                zero_body.clone()
            };
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack v_c2
            let body = if pack_to_next_v {
                poly_sub(v_c2, &pack_next_v_poly(ci), curve)
            } else {
                zero_body.clone()
            };
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack v_d2
            let body = if pack_to_next_v {
                poly_sub(v_d2, &pack_next_v_poly(di), curve)
            } else {
                zero_body.clone()
            };
            acc = poly_add(&acc, &push(body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // ─── XOR/ROTR bit-decomposition layer (32 instances) ──────────
        let mut bit_weights: Vec<Scalar> =
            Vec::with_capacity(XOR_ROTR_BITS_PER_LIMB);
        bit_weights.push(Scalar::one(curve));
        for i in 1..XOR_ROTR_BITS_PER_LIMB {
            bit_weights.push(bit_weights[i - 1].add(&bit_weights[i - 1]));
        }
        let two_scalar = Scalar::from_u64(2, curve);
        let push_gated = |body: Vec<Scalar>, ap_local: &Scalar| -> Vec<Scalar> {
            let term = poly_mul(&gate_poly, &body, curve);
            poly_scalar_mul(&term, ap_local)
        };

        let operand_poly = |op: XorRotrOperand| -> Vec<Scalar> {
            match op {
                XorRotrOperand::InputVBytes(v_idx) => pack_v_poly(v_idx),
                XorRotrOperand::Intermediate { g_idx, slot } => {
                    col_coeffs[col_g_intermediate(g_idx, slot)].clone()
                }
            }
        };

        for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
            let inst = &XOR_ROTR_INSTANCES[inst_idx];
            let a_base = xor_rotr_bit_base(inst_idx, 0);
            let b_base = xor_rotr_bit_base(inst_idx, 1);
            let d_base = xor_rotr_bit_base(inst_idx, 2);

            // 192 binary cell constraints.
            for blk in 0..XOR_ROTR_BIT_LIMBS_PER_INSTANCE {
                let base = xor_rotr_bit_base(inst_idx, blk);
                for i in 0..XOR_ROTR_BITS_PER_LIMB {
                    let v = &col_coeffs[base + i];
                    let v_m1 = poly_sub(v, &one_poly, curve);
                    let body = poly_mul(v, &v_m1, curve);
                    acc = poly_add(&acc, &push_gated(body, &ap), curve);
                    ap = ap.mul(alpha);
                }
            }

            let pack_bits_poly = |base: usize| -> Vec<Scalar> {
                let mut acc = vec![Scalar::zero(curve)];
                for i in 0..XOR_ROTR_BITS_PER_LIMB {
                    let bp = &col_coeffs[base + i];
                    acc = poly_add(
                        &acc,
                        &poly_scalar_mul(bp, &bit_weights[i]),
                        curve,
                    );
                }
                acc
            };

            // pack A
            let body = poly_sub(&pack_bits_poly(a_base), &operand_poly(inst.a), curve);
            acc = poly_add(&acc, &push_gated(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack B
            let body = poly_sub(&pack_bits_poly(b_base), &operand_poly(inst.b), curve);
            acc = poly_add(&acc, &push_gated(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack dst
            let body = poly_sub(&pack_bits_poly(d_base), &operand_poly(inst.dst), curve);
            acc = poly_add(&acc, &push_gated(body, &ap), curve);
            ap = ap.mul(alpha);

            // XOR + ROTR_n fused. See evaluate_on_domain for derivation.
            let rot = inst.rot;
            for i in 0..XOR_ROTR_BITS_PER_LIMB {
                let dst_bit_idx =
                    (i + XOR_ROTR_BITS_PER_LIMB - rot) % XOR_ROTR_BITS_PER_LIMB;
                let a = &col_coeffs[a_base + i];
                let b = &col_coeffs[b_base + i];
                let ab = poly_mul(a, b, curve);
                let two_ab = poly_scalar_mul(&ab, &two_scalar);
                let a_plus_b = poly_add(a, b, curve);
                let xor = poly_sub(&a_plus_b, &two_ab, curve);
                let dst_bit = &col_coeffs[d_base + dst_bit_idx];
                let body = poly_sub(dst_bit, &xor, curve);
                acc = poly_add(&acc, &push_gated(body, &ap), curve);
                ap = ap.mul(alpha);
            }
        }

        // ─── σ-permutation layer (poly form) ──────────────────────────
        let ten = Scalar::from_u64(SIGMA_ROWS as u64, curve);

        let pack_m_poly = |m_idx: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp = &col_coeffs[COL_M_OFFSET + m_idx * LIMB_BYTES + b];
                acc = poly_add(&acc, &poly_scalar_mul(bp, &limb_byte_weights[b]), curve);
            }
            acc
        };

        // (1) round_index - 10*q - round_mod_10 = 0 (ungated).
        let q_p = &col_coeffs[COL_ROUND_QUOTIENT_10];
        let m10_p = &col_coeffs[COL_ROUND_MOD_10];
        let ri_p = &col_coeffs[COL_ROUND_INDEX];
        let ten_q = poly_scalar_mul(q_p, &ten);
        let body = poly_sub(&poly_sub(ri_p, &ten_q, curve), m10_p, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // (2) q binary, gated.
        let q_m1 = poly_sub(q_p, &one_poly, curve);
        let body = poly_mul(q_p, &q_m1, curve);
        acc = poly_add(&acc, &push_gated(body, &ap), curve);
        ap = ap.mul(alpha);

        // (3) is_sigma_row[r] binary (ungated, 10 constraints).
        for sr in 0..SIGMA_ROWS {
            let v = &col_coeffs[col_is_sigma_row(sr)];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // (4) Σ is_sigma_row - gate = 0 (ungated).
        let mut sum = vec![Scalar::zero(curve)];
        for sr in 0..SIGMA_ROWS {
            sum = poly_add(&sum, &col_coeffs[col_is_sigma_row(sr)], curve);
        }
        let body = poly_sub(&sum, &gate_poly, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // (5) gate * (Σ r·is_sigma_row[r] - round_mod_10) = 0.
        let mut sum = vec![Scalar::zero(curve)];
        for sr in 0..SIGMA_ROWS {
            let w = Scalar::from_u64(sr as u64, curve);
            sum = poly_add(
                &sum,
                &poly_scalar_mul(&col_coeffs[col_is_sigma_row(sr)], &w),
                curve,
            );
        }
        let body = poly_sub(&sum, m10_p, curve);
        acc = poly_add(&acc, &push_gated(body, &ap), curve);
        ap = ap.mul(alpha);

        // (6) σ-binding 160 constraints, self-gated by `is_sigma_row[sr]`.
        for sr in 0..SIGMA_ROWS {
            let sel = &col_coeffs[col_is_sigma_row(sr)];
            for k in 0..M_SCHED_LIMBS {
                let m_sched = &col_coeffs[col_m_sched(k)];
                let m_src_idx = SIGMA[sr][k];
                let body = poly_sub(m_sched, &pack_m_poly(m_src_idx), curve);
                let prod = poly_mul(sel, &body, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&prod, &ap), curve);
                ap = ap.mul(alpha);
            }
        }

        // ─── IV-init / final h_out finalize layer (Task #206) ─────────
        let pack_h_in_poly = |k: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp =
                    &col_coeffs[COL_H_IN_OFFSET + k * LIMB_BYTES + b];
                acc = poly_add(
                    &acc,
                    &poly_scalar_mul(bp, &limb_byte_weights[b]),
                    curve,
                );
            }
            acc
        };
        let iv_scalar_pl = |i: usize| -> Scalar {
            let lo = (BLAKE2B_IV[i] & 0xffff_ffffu64) as u64;
            let hi = (BLAKE2B_IV[i] >> 32) as u64;
            let lo_s = Scalar::from_u64(lo, curve);
            let hi_s = Scalar::from_u64(hi, curve);
            let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
            lo_s.add(&hi_s.mul(&two_pow_32))
        };
        let first_round_p = &col_coeffs[COL_IS_FIRST_ROUND];

        // (0) is_first_round binary.
        let fr_m1 = poly_sub(first_round_p, &one_poly, curve);
        let body = poly_mul(first_round_p, &fr_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // (1..8) is_first_round * (pack_v(v[i]) - pack_h_in[i]) = 0.
        for i in 0..8 {
            let lhs = pack_v_poly(i);
            let rhs = pack_h_in_poly(i);
            let diff = poly_sub(&lhs, &rhs, curve);
            let body = poly_mul(first_round_p, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // (9..12) is_first_round * (pack_v(v[i+8]) - IV[i]) = 0 for i ∈ 0..4.
        for i in 0..4 {
            let lhs = pack_v_poly(i + 8);
            let iv_s = iv_scalar_pl(i);
            let iv_poly = vec![iv_s];
            let diff = poly_sub(&lhs, &iv_poly, curve);
            let body = poly_mul(first_round_p, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }

        // (13) is_first_round * (pack_v(v[15]) - IV[7]) = 0.
        let lhs = pack_v_poly(15);
        let iv7 = iv_scalar_pl(7);
        let iv_poly = vec![iv7];
        let diff = poly_sub(&lhs, &iv_poly, curve);
        let body = poly_mul(first_round_p, &diff, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);

        // ─── IV XOR bit-decomposition layer (Task #215, poly form) ────
        let f_byte_mask_pl = {
            let lo_pow = Scalar::from_u64(0xFFFFFFFFu64, curve);
            let hi_pow = Scalar::from_u64(0xFFFFFFFFu64, curve);
            let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
            lo_pow.add(&hi_pow.mul(&two_pow_32))
        };
        let pack_t_poly = |off: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp = &col_coeffs[COL_T_OFFSET + off + b];
                acc = poly_add(
                    &acc,
                    &poly_scalar_mul(bp, &limb_byte_weights[b]),
                    curve,
                );
            }
            acc
        };
        let push_first_round =
            |body: Vec<Scalar>, ap_local: &Scalar| -> Vec<Scalar> {
                let term = poly_mul(first_round_p, &body, curve);
                poly_scalar_mul(&term, ap_local)
            };
        for inst_idx in 0..IV_XOR_INSTANCE_COUNT {
            let iv_idx = IV_XOR_IV_INDEX[inst_idx];
            let v_idx = IV_XOR_V_INDEX[inst_idx];
            let a_base = iv_xor_bit_base(inst_idx, 0);
            let b_base = iv_xor_bit_base(inst_idx, 1);
            let d_base = iv_xor_bit_base(inst_idx, 2);
            let iv_const = iv_scalar_pl(iv_idx);
            let iv_const_poly = vec![iv_const];
            let b_target_poly: Vec<Scalar> = match IV_XOR_B_SOURCE[inst_idx] {
                IvXorBSource::TBytes(off) => pack_t_poly(off),
                IvXorBSource::FByteMask => {
                    poly_scalar_mul(&col_coeffs[COL_F_BYTE], &f_byte_mask_pl)
                }
            };
            let dst_target_poly = pack_v_poly(v_idx);

            // 192 binary cell constraints.
            for blk in 0..IV_XOR_BIT_LIMBS_PER_INSTANCE {
                let base = iv_xor_bit_base(inst_idx, blk);
                for i in 0..IV_XOR_BITS_PER_LIMB {
                    let v = &col_coeffs[base + i];
                    let v_m1 = poly_sub(v, &one_poly, curve);
                    let body = poly_mul(v, &v_m1, curve);
                    acc = poly_add(&acc, &push_first_round(body, &ap), curve);
                    ap = ap.mul(alpha);
                }
            }

            let pack_bits_poly = |base: usize| -> Vec<Scalar> {
                let mut acc = vec![Scalar::zero(curve)];
                for i in 0..IV_XOR_BITS_PER_LIMB {
                    let bp = &col_coeffs[base + i];
                    acc = poly_add(
                        &acc,
                        &poly_scalar_mul(bp, &bit_weights[i]),
                        curve,
                    );
                }
                acc
            };
            // pack A
            let body = poly_sub(&pack_bits_poly(a_base), &iv_const_poly, curve);
            acc = poly_add(&acc, &push_first_round(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack B
            let body = poly_sub(&pack_bits_poly(b_base), &b_target_poly, curve);
            acc = poly_add(&acc, &push_first_round(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack dst
            let body = poly_sub(&pack_bits_poly(d_base), &dst_target_poly, curve);
            acc = poly_add(&acc, &push_first_round(body, &ap), curve);
            ap = ap.mul(alpha);

            // 64 XOR (rot=0).
            for i in 0..IV_XOR_BITS_PER_LIMB {
                let a = &col_coeffs[a_base + i];
                let b = &col_coeffs[b_base + i];
                let ab = poly_mul(a, b, curve);
                let two_ab = poly_scalar_mul(&ab, &two_scalar);
                let a_plus_b = poly_add(a, b, curve);
                let xor = poly_sub(&a_plus_b, &two_ab, curve);
                let dst_bit = &col_coeffs[d_base + i];
                let body = poly_sub(dst_bit, &xor, curve);
                acc = poly_add(&acc, &push_first_round(body, &ap), curve);
                ap = ap.mul(alpha);
            }
        }

        // ─── h_out finalize triple-XOR layer (Task #216, poly form) ───
        let four_scalar = Scalar::from_u64(4, curve);
        let is_result_p = &col_coeffs[COL_IS_RESULT_ROW];
        let push_is_result =
            |body: Vec<Scalar>, ap_local: &Scalar| -> Vec<Scalar> {
                let term = poly_mul(is_result_p, &body, curve);
                poly_scalar_mul(&term, ap_local)
            };
        let pack_h_in_limb_poly = |i: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp = &col_coeffs[COL_H_IN_OFFSET + i * LIMB_BYTES + b];
                acc = poly_add(
                    &acc,
                    &poly_scalar_mul(bp, &limb_byte_weights[b]),
                    curve,
                );
            }
            acc
        };
        let pack_v_final_limb_poly = |i: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp = &col_coeffs[COL_V_FINAL_OFFSET + i * LIMB_BYTES + b];
                acc = poly_add(
                    &acc,
                    &poly_scalar_mul(bp, &limb_byte_weights[b]),
                    curve,
                );
            }
            acc
        };
        let pack_result_limb_poly = |i: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            for b in 0..LIMB_BYTES {
                let bp = &col_coeffs[COL_RESULT_OFFSET + i * LIMB_BYTES + b];
                acc = poly_add(
                    &acc,
                    &poly_scalar_mul(bp, &limb_byte_weights[b]),
                    curve,
                );
            }
            acc
        };
        for inst_idx in 0..H_OUT_XOR_INSTANCE_COUNT {
            let h_in_base = h_out_xor_bit_base(inst_idx, 0);
            let v_lo_base = h_out_xor_bit_base(inst_idx, 1);
            let v_hi_base = h_out_xor_bit_base(inst_idx, 2);
            let h_out_base = h_out_xor_bit_base(inst_idx, 3);

            // 256 binary cell constraints.
            for blk in 0..H_OUT_XOR_BIT_LIMBS_PER_INSTANCE {
                let base = h_out_xor_bit_base(inst_idx, blk);
                for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                    let v = &col_coeffs[base + i];
                    let v_m1 = poly_sub(v, &one_poly, curve);
                    let body = poly_mul(v, &v_m1, curve);
                    acc = poly_add(&acc, &push_is_result(body, &ap), curve);
                    ap = ap.mul(alpha);
                }
            }

            let pack_bits_poly = |base: usize| -> Vec<Scalar> {
                let mut acc = vec![Scalar::zero(curve)];
                for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                    let bp = &col_coeffs[base + i];
                    acc = poly_add(
                        &acc,
                        &poly_scalar_mul(bp, &bit_weights[i]),
                        curve,
                    );
                }
                acc
            };

            // pack h_in
            let body =
                poly_sub(&pack_bits_poly(h_in_base), &pack_h_in_limb_poly(inst_idx), curve);
            acc = poly_add(&acc, &push_is_result(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack v_lo
            let body = poly_sub(
                &pack_bits_poly(v_lo_base),
                &pack_v_final_limb_poly(inst_idx),
                curve,
            );
            acc = poly_add(&acc, &push_is_result(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack v_hi
            let body = poly_sub(
                &pack_bits_poly(v_hi_base),
                &pack_v_final_limb_poly(inst_idx + 8),
                curve,
            );
            acc = poly_add(&acc, &push_is_result(body, &ap), curve);
            ap = ap.mul(alpha);
            // pack h_out
            let body = poly_sub(
                &pack_bits_poly(h_out_base),
                &pack_result_limb_poly(inst_idx),
                curve,
            );
            acc = poly_add(&acc, &push_is_result(body, &ap), curve);
            ap = ap.mul(alpha);

            // 64 fused XOR-of-3 constraints.
            for i in 0..H_OUT_XOR_BITS_PER_LIMB {
                let h = &col_coeffs[h_in_base + i];
                let l = &col_coeffs[v_lo_base + i];
                let h2 = &col_coeffs[v_hi_base + i];
                let h_l = poly_mul(h, l, curve);
                let h_h2 = poly_mul(h, h2, curve);
                let l_h2 = poly_mul(l, h2, curve);
                let h_l_h2 = poly_mul(&h_l, h2, curve);
                let sum_lin = poly_add(&poly_add(h, l, curve), h2, curve);
                let pair_sum = poly_add(
                    &poly_add(&h_l, &h_h2, curve),
                    &l_h2,
                    curve,
                );
                let two_pair = poly_scalar_mul(&pair_sum, &two_scalar);
                let four_trip = poly_scalar_mul(&h_l_h2, &four_scalar);
                let xor3 = poly_add(
                    &poly_sub(&sum_lin, &two_pair, curve),
                    &four_trip,
                    curve,
                );
                let h_out_bit = &col_coeffs[h_out_base + i];
                let body = poly_sub(h_out_bit, &xor3, curve);
                acc = poly_add(&acc, &push_is_result(body, &ap), curve);
                ap = ap.mul(alpha);
            }
        }

        acc
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Shifted: v[0..V_LENGTH], is_real, is_result_row.
        let mut cols = Vec::with_capacity(V_LENGTH + 2);
        for k in 0..V_LENGTH {
            cols.push(COL_V_OFFSET + k);
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
        let expected = V_LENGTH + 2;
        if shifted_evals.len() != expected || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let real_z = &col_evals_at_z[COL_IS_REAL];
        let real_next = &shifted_evals[V_LENGTH];
        let is_result = &col_evals_at_z[COL_IS_RESULT_ROW];
        let gating = real_z.mul(real_next).mul(&one.sub(is_result));

        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..V_LENGTH {
            let nxt = &shifted_evals[k];
            let cur_next = &col_evals_at_z[COL_NEXT_V_OFFSET + k];
            acc = acc.add(&bp.mul(&nxt.sub(cur_next)));
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
        let one_minus = poly_sub(&one_poly, is_result, curve);
        let gating = poly_mul(&poly_mul(real, &real_next, curve), &one_minus, curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..V_LENGTH {
            let cur = &column_coeffs[COL_V_OFFSET + k];
            let nxt = poly_shift(cur, omega);
            let cur_next = &column_coeffs[COL_NEXT_V_OFFSET + k];
            let diff = poly_sub(&nxt, cur_next, curve);
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
        vec![COL_IS_REAL, COL_IS_RESULT_ROW]
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
        for k in 0..V_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_v_byte_{}_8bit", k),
                    column_index: COL_V_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_next_v_byte_{}_8bit", k),
                    column_index: COL_NEXT_V_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..M_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_m_byte_{}_8bit", k),
                    column_index: COL_M_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..T_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_t_byte_{}_8bit", k),
                    column_index: COL_T_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..RESULT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_result_byte_{}_8bit", k),
                    column_index: COL_RESULT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..ROUND_INDEX_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_round_index_byte_{}_8bit", b),
                    column_index: COL_ROUND_INDEX_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..H_IN_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_h_in_byte_{}_8bit", k),
                    column_index: COL_H_IN_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..V_FINAL_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2_f_internals_v_final_byte_{}_8bit", k),
                    column_index: COL_V_FINAL_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // h_out finalize XOR bit columns (Task #216): per-cell binary
        // constraints are gated by is_result_row, so we range-check the
        // bit cells as 1-bit values. We use the existing 8-bit range
        // table since no 1-bit table exists; 1-bit values fit in 8 bits.
        // No declarations added here — the binary constraints in the
        // row-local layer enforce 0/1 directly.
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Placeholder sentinel for the precompile-side h_out column.
pub const PRECOMPILE_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (skeleton): binds the FINAL-row
/// `result_h[0..64]` of the internals AIR to the
/// `h_out[0..64]` column of [`crate::blake2f_precompile_air`].
pub fn make_blake2_f_internals_to_precompile_descriptor(
    internals_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(RESULT_LENGTH);
    for k in 0..RESULT_LENGTH {
        a_columns.push(COL_RESULT_OFFSET + k);
    }
    let b_columns: Vec<usize> = vec![PRECOMPILE_PLACEHOLDER; RESULT_LENGTH];
    CrossAirLogUpDescriptor {
        label: "blake2_f_internals_to_precompile_result_v1_stub".into(),
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
    use crate::blake2f_precompile_air::INPUT_LENGTH as BLAKE2F_INPUT_LENGTH;

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

    /// Honest skeleton witness with zero result passes all constraints.
    #[test]
    fn blake2_f_internals_air_zero_result_passes() {
        let w = Blake2fInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    /// Honest skeleton witness from the precompile's own reference F
    /// invocation passes.
    #[test]
    fn blake2_f_internals_air_known_result_passes() {
        // Build a real precompile witness from an EIP-152-shaped input,
        // then take its h_out as our internals AIR's `result_h`.
        let mut input = [0u8; BLAKE2F_INPUT_LENGTH];
        // 12 rounds (BE u32) at bytes 0..4.
        input[3] = 12;
        // f flag at last byte = 1 (final block).
        input[BLAKE2F_INPUT_LENGTH - 1] = 1;
        let precompile_w =
            crate::blake2f_precompile_air::Blake2fPrecompileWitness::from_input(input);
        let w = Blake2fInternalsTraceWitness::from_result_skeleton(
            precompile_w.h_out,
        );
        assert_eq!(w.rows[1].result_h, precompile_w.h_out);

        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn blake2_f_internals_air_tampered_result_off_final_row_detected() {
        let w = Blake2fInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Row 0 is NOT the result row; smear a byte to nonzero.
        cols[COL_RESULT_OFFSET + 7][0] = Scalar::from_u64(0x5a, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[5][0].is_zero(),
            "expected gated-result constraint to fire on non-final row",
        );
    }

    #[test]
    fn blake2_f_internals_air_tampered_f_byte_detected() {
        let w = Blake2fInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // f_byte must be 0 or 1 — set to 2 on row 1.
        cols[COL_F_BYTE][1] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][1].is_zero(),
            "expected f_byte binary constraint to fire",
        );
    }

    #[test]
    fn blake2_f_internals_air_descriptor_well_formed() {
        let d = make_blake2_f_internals_to_precompile_descriptor(0, 1);
        assert_eq!(d.label, "blake2_f_internals_to_precompile_result_v1_stub");
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

    #[test]
    fn blake2_f_internals_air_column_layout_pinned() {
        assert_eq!(COL_V_OFFSET, 0);
        assert_eq!(COL_NEXT_V_OFFSET, 128);
        assert_eq!(COL_M_OFFSET, 256);
        assert_eq!(COL_T_OFFSET, 384);
        assert_eq!(COL_F_BYTE, 400);
        assert_eq!(COL_ROUND_INDEX, 401);
        assert_eq!(COL_ROUND_INDEX_BYTE_OFFSET, 402);
        assert_eq!(COL_RESULT_OFFSET, 406);
        assert_eq!(COL_IS_RESULT_ROW, 470);
        assert_eq!(COL_IS_REAL, 471);
        assert_eq!(COL_G_INTERMEDIATE_OFFSET, 472);
        assert_eq!(G_INTERMEDIATE_COL_COUNT, 64);
        assert_eq!(COL_G_CARRY_OFFSET, 536);
        assert_eq!(G_CARRY_COL_COUNT, 32);
        assert_eq!(COL_M_SCHED_OFFSET, 568);
        assert_eq!(M_SCHED_LIMBS, 16);
        assert_eq!(COL_XOR_ROTR_BIT_OFFSET, 584);
        assert_eq!(COL_G0_STEP2_BIT_OFFSET, 584);
        assert_eq!(G0_STEP2_VA1_BIT_OFFSET, 584);
        assert_eq!(G0_STEP2_VD_BIT_OFFSET, 648);
        assert_eq!(G0_STEP2_VD1_BIT_OFFSET, 712);
        assert_eq!(G0_STEP2_BIT_COUNT, 192);
        assert_eq!(XOR_ROTR_INSTANCE_COUNT, 32);
        assert_eq!(XOR_ROTR_BIT_COUNT_PER_INSTANCE, 192);
        assert_eq!(XOR_ROTR_BIT_COL_COUNT, 32 * 192);
        assert_eq!(COL_ROUND_MOD_10, 6728);
        assert_eq!(COL_ROUND_QUOTIENT_10, 6729);
        assert_eq!(COL_IS_SIGMA_ROW_OFFSET, 6730);
        assert_eq!(SIGMA_ROWS, 10);
        // IV-init / final h_out columns (Task #206).
        assert_eq!(COL_H_IN_OFFSET, 6740);
        assert_eq!(COL_V_FINAL_OFFSET, 6740 + 64);
        assert_eq!(COL_IS_FIRST_ROUND, 6740 + 64 + 128);
        assert_eq!(H_IN_LENGTH, 64);
        assert_eq!(V_FINAL_LENGTH, 128);
        // Base layout (pre Task #215 / #216).
        let base_cols = 584 + 32 * 192 + 2 + 10 + 64 + 128 + 1; // 6933
        // Task #215 added 3 IV XOR instances × 192 bit cols = 576.
        // Task #216 added 8 h_out XOR instances × 256 bit cols = 2048.
        assert_eq!(COL_IV_XOR_BIT_OFFSET, base_cols);
        assert_eq!(IV_XOR_BIT_COL_COUNT, 3 * 192);
        assert_eq!(COL_H_OUT_XOR_BIT_OFFSET, base_cols + 3 * 192);
        assert_eq!(H_OUT_XOR_BIT_COL_COUNT, 8 * 256);
        assert_eq!(NUM_COLUMNS, base_cols + 3 * 192 + 8 * 256);
        assert_eq!(NUM_G0_STEP2_BINARY, 192);
        assert_eq!(NUM_G0_STEP2_PACK, 3);
        assert_eq!(NUM_G0_STEP2_XOR_ROTR, 64);
        assert_eq!(NUM_G0_STEP2_CONSTRAINTS, 259);
        assert_eq!(NUM_XOR_ROTR_CONSTRAINTS_PER_INSTANCE, 259);
        assert_eq!(NUM_XOR_ROTR_CONSTRAINTS_TOTAL, 32 * 259);
        assert_eq!(NUM_SIGMA_AUX_CONSTRAINTS, 14);
        assert_eq!(NUM_SIGMA_BIND_CONSTRAINTS, 160);
        assert_eq!(NUM_SIGMA_CONSTRAINTS_TOTAL, 174);
        assert_eq!(NUM_IV_INIT_CONSTRAINTS, 14);
        // IV XOR layer (Task #215): 3 instances × 259 constraints = 777.
        assert_eq!(NUM_IV_XOR_CONSTRAINTS_PER_INSTANCE, 259);
        assert_eq!(NUM_IV_XOR_CONSTRAINTS_TOTAL, 3 * 259);
        // h_out finalize triple-XOR layer (Task #216):
        // 256 binary + 4 pack + 64 XOR3 = 324 per instance × 8 = 2592.
        assert_eq!(H_OUT_XOR_INSTANCE_COUNT, 8);
        assert_eq!(H_OUT_XOR_BIT_COUNT_PER_INSTANCE, 4 * 64);
        assert_eq!(NUM_H_OUT_XOR_BINARY_PER_INSTANCE, 256);
        assert_eq!(NUM_H_OUT_XOR_PACK_PER_INSTANCE, 4);
        assert_eq!(NUM_H_OUT_XOR_BITWISE_PER_INSTANCE, 64);
        assert_eq!(NUM_H_OUT_XOR_CONSTRAINTS_PER_INSTANCE, 324);
        assert_eq!(NUM_H_OUT_XOR_CONSTRAINTS_TOTAL, 8 * 324);
        assert_eq!(
            NUM_ROW_CONSTRAINTS,
            6 + 96 + 32 * 259 + 174 + 14 + 3 * 259 + 8 * 324,
        );
        // BLAKE2b IV constants pinned (RFC 7693 §2.6 / SHA-512 IH).
        assert_eq!(BLAKE2B_IV[0], 0x6a09e667f3bcc908);
        assert_eq!(BLAKE2B_IV[7], 0x5be0cd19137e2179);
        assert_eq!(F_BYTE_MASK_ONES, 0xFFFFFFFFFFFFFFFFu64);
        assert_eq!(NUM_G_CONSTRAINTS_PER_G, 12);
        assert_eq!(NUM_G_CONSTRAINTS_TOTAL, 96);
        assert_eq!(NUM_SHIFTED, 1);
        assert_eq!(MAINLINE_ROUNDS, 12);
        assert_eq!(PC_BLAKE2F, 0x09);

        let cs = Blake2fInternalsConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert_eq!(cs.shifted_column_indices().len(), V_LENGTH + 2);
    }

    // ─── G-mixing layer tests ─────────────────────────────────────

    /// Reference BLAKE2 G that mutates a `[u64; 16]` state — used as
    /// the spec oracle for `blake2_g_intermediates` & `build_round_row`.
    fn ref_blake2_g(
        v: &mut [u64; 16],
        a: usize, b: usize, c: usize, d: usize,
        x: u64, y: u64,
    ) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }

    #[test]
    fn blake2_g_intermediates_matches_reference_zero() {
        let g = blake2_g_intermediates(0, 0, 0, 0, 0, 0);
        assert_eq!(g.limbs, [0u64; 8]);
        assert_eq!(g.carries, [0u64; 4]);
    }

    #[test]
    fn blake2_g_intermediates_matches_reference_nonzero() {
        // Use BLAKE2b IV + a sample message to exercise carry paths.
        let v_a = 0x6a09e667f3bcc908u64;
        let v_b = 0xbb67ae8584caa73bu64;
        let v_c = 0x3c6ef372fe94f82bu64;
        let v_d = 0xa54ff53a5f1d36f1u64;
        let m_i = 0xdeadbeefcafebabeu64;
        let m_j = 0x0123456789abcdefu64;
        let g = blake2_g_intermediates(v_a, v_b, v_c, v_d, m_i, m_j);

        let mut state = [0u64; 16];
        state[0] = v_a;
        state[1] = v_b;
        state[2] = v_c;
        state[3] = v_d;
        ref_blake2_g(&mut state, 0, 1, 2, 3, m_i, m_j);
        assert_eq!(g.limbs[4], state[0], "v_a2");
        assert_eq!(g.limbs[7], state[1], "v_b2");
        assert_eq!(g.limbs[6], state[2], "v_c2");
        assert_eq!(g.limbs[5], state[3], "v_d2");

        // Re-execute the add chain in u128 to verify carry witnesses.
        let s1 = (v_a as u128) + (v_b as u128) + (m_i as u128);
        assert_eq!(g.carries[0], (s1 >> 64) as u64);
        let s3 = (v_c as u128) + (g.limbs[1] as u128);
        assert_eq!(g.carries[1], (s3 >> 64) as u64);
        let s5 = (g.limbs[0] as u128)
            + (g.limbs[3] as u128)
            + (m_j as u128);
        assert_eq!(g.carries[2], (s5 >> 64) as u64);
        let s7 = (g.limbs[2] as u128) + (g.limbs[5] as u128);
        assert_eq!(g.carries[3], (s7 >> 64) as u64);
        // Carries are in range {0,1,2} / {0,1}.
        assert!(g.carries[0] < 3);
        assert!(g.carries[1] < 2);
        assert!(g.carries[2] < 3);
        assert!(g.carries[3] < 2);
    }

    #[test]
    fn blake2_f_internals_g_constraints_zero_witness_passes() {
        // Skeleton (all zero) witness — G constraints all hold trivially.
        let w = Blake2fInternalsTraceWitness::from_result_skeleton([0u8; RESULT_LENGTH]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} nonzero on zero-witness",
                    i, r,
                );
            }
        }
    }

    #[test]
    fn blake2_f_internals_g_constraints_honest_round_passes() {
        // Honest single non-trivial round + zero-result row.
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let mut m_words = [0u64; M_LIMBS];
        for k in 0..M_LIMBS {
            m_words[k] = 0xdeadbeefcafebabeu64.wrapping_add(k as u64);
        }
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} nonzero on honest round",
                    i, r,
                );
            }
        }
    }

    #[test]
    fn blake2_f_internals_g_tampered_v_a1_detected() {
        // Tamper an intermediate so add1 fails on G0.
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let m_words = [0xdeadbeefu64; M_LIMBS];
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Bump v_a1 on G0 (col_g_intermediate(0, 0)) by 1 on row 0.
        let bad_col = col_g_intermediate(0, 0);
        cols[bad_col][0] = cols[bad_col][0]
            .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // First G constraint is at index 6 (G0 add1).
        // Tampering v_a1 perturbs add1, add5, pack_v_a2... at least one fires.
        let g0_add1 = 6;
        assert!(
            !bodies[g0_add1][0].is_zero(),
            "expected G0 add1 to fire on tampered v_a1",
        );
    }

    #[test]
    fn blake2_f_internals_g_tampered_carry_out_of_range_detected() {
        // Set c1 on G0 to 3 (out of range {0,1,2}); cubic carry constraint
        // c·(c-1)·(c-2) must fire.
        let v_in = [0u64; V_LIMBS];
        let m_words = [0u64; M_LIMBS];
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let bad_col = col_g_carry(0, 0);
        cols[bad_col][0] = Scalar::from_u64(3, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // G0's c1_range_3way is at offset 6 + 4 = 10.
        let g0_c1_range = 6 + 4;
        assert!(
            !bodies[g0_c1_range][0].is_zero(),
            "expected G0 c1 range constraint to fire on c1=3",
        );
    }

    #[test]
    fn blake2_f_internals_g_evaluate_at_point_matches_on_domain() {
        // Honest round; verify evaluate_at_point yields the α-folded
        // RLC of the bodies (uses subtraction since Scalar has no PartialEq).
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x1234_5678_9abc_def0u64.wrapping_mul(k as u64 + 7);
        }
        let m_words = [0x55aa55aa55aa55aau64; M_LIMBS];
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // Use alpha = 7.
        let alpha = Scalar::from_u64(7, CurveType::Bls48581);
        for row in 0..trace.num_rows {
            let col_evals: Vec<Scalar> =
                trace.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let at_point = cs.evaluate_at_point(&col_evals, &alpha);
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

    // ─── G0 step-2 bit-decomposition tests ────────────────────────

    /// Constraint indices for the G0-step-2 layer within the row-local
    /// constraint vector. Order: 192 binary (va1[0..64], vd[0..64],
    /// vd1[0..64]) then 3 pack (va1, vd, vd1) then 64 xor+rotr.
    const G0_STEP2_LAYER_BASE: usize = 6 + 96; // 102
    const G0_STEP2_VA1_BIN_BASE: usize = G0_STEP2_LAYER_BASE; // 102
    const G0_STEP2_VD_BIN_BASE: usize =
        G0_STEP2_VA1_BIN_BASE + G0_STEP2_BITS_PER_LIMB; // 166
    const G0_STEP2_VD1_BIN_BASE: usize =
        G0_STEP2_VD_BIN_BASE + G0_STEP2_BITS_PER_LIMB; // 230
    const G0_STEP2_PACK_VA1: usize =
        G0_STEP2_VD1_BIN_BASE + G0_STEP2_BITS_PER_LIMB; // 294
    #[allow(dead_code)]
    const G0_STEP2_PACK_VD: usize = G0_STEP2_PACK_VA1 + 1; // 295
    #[allow(dead_code)]
    const G0_STEP2_PACK_VD1: usize = G0_STEP2_PACK_VA1 + 2; // 296
    const G0_STEP2_XOR_ROTR_BASE: usize = G0_STEP2_PACK_VA1 + 3; // 297

    #[test]
    fn blake2_f_internals_g0_step2_layer_honest_passes() {
        // Same honest round used by the broader G-layer test.
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let mut m_words = [0u64; M_LIMBS];
        for k in 0..M_LIMBS {
            m_words[k] = 0xdeadbeefcafebabeu64.wrapping_add(k as u64);
        }
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for i in G0_STEP2_LAYER_BASE..NUM_ROW_CONSTRAINTS {
            for (r, v) in bodies[i].iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "G0-step2 constraint {} row {} nonzero",
                    i, r,
                );
            }
        }
    }

    #[test]
    fn blake2_f_internals_g0_step2_tampered_bit_binary_detected() {
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let m_words = [0xdeadbeefu64; M_LIMBS];
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Bump v_a1 bit 0 to 2 (non-binary) on row 0.
        cols[col_g0_step2_va1_bit(0)][0] =
            Scalar::from_u64(2, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // va1 bit 0 binary constraint at index G0_STEP2_VA1_BIN_BASE + 0.
        assert!(
            !bodies[G0_STEP2_VA1_BIN_BASE][0].is_zero(),
            "binary constraint should fire on bit=2",
        );
    }

    #[test]
    fn blake2_f_internals_g0_step2_tampered_xor_rotr_detected() {
        // Honest round, then flip one v_d1 bit. The XOR-ROTR fused
        // constraint at the bit's destination index must fire.
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let m_words = [0xdeadbeefu64; M_LIMBS];
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Flip v_d1 bit at destination index 5 (1 - x).
        let dst = 5usize;
        let one_s = Scalar::one(CurveType::Bls48581);
        let cur = cols[col_g0_step2_vd1_bit(dst)][0].clone();
        cols[col_g0_step2_vd1_bit(dst)][0] = one_s.sub(&cur);
        // ALSO bump the corresponding pack-bits column to keep the bit
        // binary; the test specifically checks the XOR-ROTR constraint
        // fires. After flipping, the bit is in {0,1} so binary holds.
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Source index i such that (i + 32) % 64 == 5 → i = (5 - 32) mod 64
        // = 5 + 32 = 37. (since -32 mod 64 = 32.)
        let src = (dst + 64 - 32) % 64; // = 37
        assert!(
            !bodies[G0_STEP2_XOR_ROTR_BASE + src][0].is_zero(),
            "XOR+ROTR_32 constraint at src bit {} should fire", src,
        );
    }

    #[test]
    fn blake2_f_internals_g0_step2_tampered_pack_detected() {
        // Flip a bit of v_a1 to 1 (originally was 0) without updating the
        // packed v_a1 column. The pack constraint must fire AND we will
        // see the XOR-ROTR constraint also fire (since v_d1 not updated).
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let m_words = [0xdeadbeefu64; M_LIMBS];
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Find a v_a1 bit that's 0 and flip to 1.
        let mut chosen: Option<usize> = None;
        for i in 0..64 {
            if cols[col_g0_step2_va1_bit(i)][0].is_zero() {
                chosen = Some(i);
                break;
            }
        }
        let i = chosen.expect("some v_a1 bit should be zero");
        cols[col_g0_step2_va1_bit(i)][0] =
            Scalar::one(CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[G0_STEP2_PACK_VA1][0].is_zero(),
            "pack(v_a1 bits) == v_a1_packed should fire on flipped bit",
        );
    }

    #[test]
    fn blake2_f_internals_g0_step2_evaluate_at_point_matches_on_domain() {
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x1234_5678_9abc_def0u64.wrapping_mul(k as u64 + 7);
        }
        let m_words = [0x55aa55aa55aa55aau64; M_LIMBS];
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        let alpha = Scalar::from_u64(11, CurveType::Bls48581);
        for row in 0..trace.num_rows {
            let col_evals: Vec<Scalar> =
                trace.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let at_point = cs.evaluate_at_point(&col_evals, &alpha);
            let mut expected = Scalar::zero(CurveType::Bls48581);
            let mut ap = Scalar::one(CurveType::Bls48581);
            for body in bodies.iter() {
                expected = expected.add(&ap.mul(&body[row]));
                ap = ap.mul(&alpha);
            }
            assert!(
                at_point.sub(&expected).is_zero(),
                "row {} evaluate_at_point != on-domain RLC w/ bit layer",
                row,
            );
        }
    }

    #[test]
    fn blake2_f_internals_two_pow_64_scalar_value() {
        let t = two_pow_64_scalar(CurveType::Bls48581);
        // Verify (2^32) * (2^32) = 2^64. We can re-derive and subtract.
        let half = Scalar::from_u64(1u64 << 32, CurveType::Bls48581);
        let again = half.mul(&half);
        assert!(t.sub(&again).is_zero());
    }

    // ─── Full 32-instance XOR/ROTR layer tests ─────────────────────────

    /// Helper: derive the offset (within the row-local constraint vector)
    /// of the start of instance `inst_idx`'s 259 constraints.
    fn xor_rotr_instance_layer_base(inst_idx: usize) -> usize {
        6 + 96 + inst_idx * NUM_XOR_ROTR_CONSTRAINTS_PER_INSTANCE
    }

    #[test]
    fn blake2_f_internals_xor_rotr_instance_table_pinned() {
        assert_eq!(XOR_ROTR_INSTANCE_COUNT, 32);
        assert_eq!(XOR_ROTR_INSTANCES.len(), 32);
        // First instance is G0 step 2 (matches Task #181's original pin).
        let g0s2 = &XOR_ROTR_INSTANCES[0];
        assert_eq!(g0s2.g_idx, 0);
        assert_eq!(g0s2.step, 2);
        assert_eq!(g0s2.rot, 32);
        assert_eq!(g0s2.a, XorRotrOperand::Intermediate { g_idx: 0, slot: 0 });
        assert_eq!(g0s2.b, XorRotrOperand::InputVBytes(12));
        assert_eq!(g0s2.dst, XorRotrOperand::Intermediate { g_idx: 0, slot: 1 });
        // G0 step 4: ROTR_24
        let g0s4 = &XOR_ROTR_INSTANCES[1];
        assert_eq!(g0s4.step, 4);
        assert_eq!(g0s4.rot, 24);
        assert_eq!(g0s4.b, XorRotrOperand::InputVBytes(V_MIX_INDICES[0][1]));
        // G0 step 6: same-G intermediates (v_d1 XOR v_a2)
        let g0s6 = &XOR_ROTR_INSTANCES[2];
        assert_eq!(g0s6.step, 6);
        assert_eq!(g0s6.rot, 16);
        assert_eq!(g0s6.a, XorRotrOperand::Intermediate { g_idx: 0, slot: 4 });
        assert_eq!(g0s6.b, XorRotrOperand::Intermediate { g_idx: 0, slot: 1 });
        // G0 step 8
        let g0s8 = &XOR_ROTR_INSTANCES[3];
        assert_eq!(g0s8.step, 8);
        assert_eq!(g0s8.rot, 63);
        // G4 step 2 must source `v_d_in` from G3.v_d2 (diagonal phase).
        let g4s2 = &XOR_ROTR_INSTANCES[4 * 4 + 0];
        assert_eq!(g4s2.g_idx, 4);
        assert_eq!(g4s2.step, 2);
        assert_eq!(g4s2.b, XorRotrOperand::Intermediate { g_idx: 3, slot: 5 });
        // G7 step 4 must source `v_b_in` from G0.v_b2 (= slot 7) on G7.
        // G7's spec is `[(3, 4), (0, 7), (1, 6), (2, 5)]`; slot index 1 = b_in.
        let g7s4 = &XOR_ROTR_INSTANCES[7 * 4 + 1];
        assert_eq!(g7s4.g_idx, 7);
        assert_eq!(g7s4.step, 4);
        assert_eq!(g7s4.b, XorRotrOperand::Intermediate { g_idx: 0, slot: 7 });
    }

    #[test]
    fn blake2_f_internals_xor_rotr_full_layer_honest_passes() {
        // Honest round → all 32 × 259 = 8288 instance constraints must
        // evaluate to zero, plus the 102 prior constraints.
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let mut m_words = [0u64; M_LIMBS];
        for k in 0..M_LIMBS {
            m_words[k] = 0xdeadbeefcafebabeu64.wrapping_add(k as u64);
        }
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} row {} nonzero",
                    i, r,
                );
            }
        }
    }

    /// Spot-check tampering on every (g_idx, step) instance: flip the LSB
    /// of A's bit-column on the G-mixing row; the instance's A binary
    /// (if flipped to 2) or pack (if flipped to in-range bit) constraint
    /// must fire for that exact instance.
    #[test]
    fn blake2_f_internals_xor_rotr_per_instance_binary_tamper_fires() {
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let m_words = [0xdeadbeefu64; M_LIMBS];
        for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
            let w = Blake2fInternalsTraceWitness::single_round_then_result(
                v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
            );
            let trace = build_trace_polynomials(&w, CurveType::Bls48581);
            let mut cols: Vec<Vec<Scalar>> = trace
                .columns
                .iter()
                .map(|p| p.evaluations.clone())
                .collect();
            // Flip A bit 0 to 2 (non-binary).
            cols[col_xor_rotr_bit(inst_idx, 0, 0)][0] =
                Scalar::from_u64(2, CurveType::Bls48581);
            let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            // This instance's A binary constraint is at
            // (layer_base + 0): the first of its 192 binary slots.
            let bin_idx = xor_rotr_instance_layer_base(inst_idx);
            assert!(
                !bodies[bin_idx][0].is_zero(),
                "instance {} (g{} step{}) binary tamper did not fire",
                inst_idx,
                XOR_ROTR_INSTANCES[inst_idx].g_idx,
                XOR_ROTR_INSTANCES[inst_idx].step,
            );
        }
    }

    /// Spot-check tampering on every instance's XOR+ROTR_n fused
    /// constraint: flip a dst bit to its complement. Source `src` is
    /// `(dst_idx + 64 - rot) % 64`; that bit's XOR+ROTR_n constraint
    /// must fire.
    #[test]
    fn blake2_f_internals_xor_rotr_per_instance_xor_rotr_tamper_fires() {
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let m_words = [0xdeadbeefu64; M_LIMBS];
        let one_s = Scalar::one(CurveType::Bls48581);
        for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
            let inst = XOR_ROTR_INSTANCES[inst_idx];
            let w = Blake2fInternalsTraceWitness::single_round_then_result(
                v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
            );
            let trace = build_trace_polynomials(&w, CurveType::Bls48581);
            let mut cols: Vec<Vec<Scalar>> = trace
                .columns
                .iter()
                .map(|p| p.evaluations.clone())
                .collect();
            // Flip dst bit 5 → 1 - bit5; remains in {0,1}.
            let dst_idx = 5usize;
            let cur = cols[col_xor_rotr_bit(inst_idx, 2, dst_idx)][0].clone();
            cols[col_xor_rotr_bit(inst_idx, 2, dst_idx)][0] = one_s.sub(&cur);
            let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            // src such that (src + 64 - rot) % 64 == dst_idx, i.e.
            // src = (dst_idx + rot) % 64.
            let src = (dst_idx + inst.rot) % 64;
            // Per-instance XOR+ROTR base = layer_base + 192 (binary) + 3 (pack).
            let xor_base = xor_rotr_instance_layer_base(inst_idx)
                + NUM_XOR_ROTR_BINARY_PER_INSTANCE
                + NUM_XOR_ROTR_PACK_PER_INSTANCE;
            assert!(
                !bodies[xor_base + src][0].is_zero(),
                "instance {} (g{} step{} rot{}) XOR+ROTR tamper at \
                 dst={} src={} did not fire",
                inst_idx, inst.g_idx, inst.step, inst.rot, dst_idx, src,
            );
        }
    }

    /// Spot-check: for each instance, flip an A bit to its complement
    /// (keeping it in {0,1}); the pack(A) constraint must fire.
    #[test]
    fn blake2_f_internals_xor_rotr_per_instance_pack_tamper_fires() {
        let mut v_in = [0u64; V_LIMBS];
        for k in 0..V_LIMBS {
            v_in[k] = 0x0123456789abcdefu64.wrapping_mul(k as u64 + 1);
        }
        let m_words = [0xdeadbeefu64; M_LIMBS];
        let one_s = Scalar::one(CurveType::Bls48581);
        for inst_idx in 0..XOR_ROTR_INSTANCE_COUNT {
            let w = Blake2fInternalsTraceWitness::single_round_then_result(
                v_in, m_words, [0u8; T_LENGTH], 0, 0, [0u8; RESULT_LENGTH],
            );
            let trace = build_trace_polynomials(&w, CurveType::Bls48581);
            let mut cols: Vec<Vec<Scalar>> = trace
                .columns
                .iter()
                .map(|p| p.evaluations.clone())
                .collect();
            // Flip A bit 0 to complement (still in {0,1}).
            let cur = cols[col_xor_rotr_bit(inst_idx, 0, 0)][0].clone();
            cols[col_xor_rotr_bit(inst_idx, 0, 0)][0] = one_s.sub(&cur);
            let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            // pack(A) is at layer_base + 192.
            let pack_a_idx = xor_rotr_instance_layer_base(inst_idx)
                + NUM_XOR_ROTR_BINARY_PER_INSTANCE;
            assert!(
                !bodies[pack_a_idx][0].is_zero(),
                "instance {} (g{} step{}) pack(A) tamper did not fire",
                inst_idx,
                XOR_ROTR_INSTANCES[inst_idx].g_idx,
                XOR_ROTR_INSTANCES[inst_idx].step,
            );
        }
    }

    // ─── σ-permutation tests ──────────────────────────────────────

    /// Index of the first σ-binding constraint `sigma_bind_r{sr}_k{k}` in
    /// the row-local constraint list. Matches the push order in
    /// `evaluate_on_domain`.
    fn sigma_bind_base() -> usize {
        // 6 leading + 96 G + 32 * 259 XOR/ROTR + (1+1+10+1+1) σ-aux.
        6 + NUM_G_CONSTRAINTS_TOTAL
            + NUM_XOR_ROTR_CONSTRAINTS_TOTAL
            + NUM_SIGMA_AUX_CONSTRAINTS
    }

    fn sigma_bind_idx(sr: usize, k: usize) -> usize {
        sigma_bind_base() + sr * M_SCHED_LIMBS + k
    }

    /// σ-aux constraint indices (offsets into the constraint list).
    fn sigma_aux_decomp_idx() -> usize {
        6 + NUM_G_CONSTRAINTS_TOTAL + NUM_XOR_ROTR_CONSTRAINTS_TOTAL
    }
    fn sigma_aux_one_hot_sum_idx() -> usize {
        // ordering: decomp, q-bin, 10 sel-bin, sum-to-one, one-hot-binds.
        sigma_aux_decomp_idx() + 1 + 1 + SIGMA_ROWS
    }
    fn sigma_aux_one_hot_binds_idx() -> usize {
        sigma_aux_one_hot_sum_idx() + 1
    }

    /// Build a real single-round honest witness: G-mixing row at
    /// `round_index` + final result row. Returns the trace.
    fn honest_round_trace(round_index: u32) -> TracePolynomials {
        // BLAKE2b IV-shaped v + simple message words exercising every slot.
        let v_in: [u64; V_LIMBS] = [
            0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
            0x510e527fade682d1, 0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
            0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
            0x510e527fade682d1, 0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
        ];
        let mut m: [u64; M_LIMBS] = [0u64; M_LIMBS];
        for k in 0..M_LIMBS {
            m[k] = 0x1111111111111111u64.wrapping_mul(k as u64 + 1);
        }
        let w = Blake2fInternalsTraceWitness::single_round_then_result(
            v_in, m, [0u8; T_LENGTH], 0, round_index, [0u8; RESULT_LENGTH],
        );
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    #[test]
    fn blake2_f_internals_sigma_round0_identity_passes() {
        // Round 0: SIGMA[0] = identity. `m_sched[k] = m_words[k]`.
        let trace = honest_round_trace(0);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn blake2_f_internals_sigma_round3_permutation_passes() {
        // Round 3: nontrivial SIGMA. Verifies the witness builder permutes
        // m correctly and every σ-binding constraint accepts.
        let trace = honest_round_trace(3);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn blake2_f_internals_sigma_round11_wraps_to_row1() {
        // Round 11 → 11 mod 10 = 1 → SIGMA[1]. Witness builder must light
        // is_sigma_row[1] not is_sigma_row[11] (which doesn't exist).
        let trace = honest_round_trace(11);
        let curve = CurveType::Bls48581;
        let one_s = Scalar::one(curve);
        let is_sr1 = &trace.columns[col_is_sigma_row(1)].evaluations[0];
        assert!(
            is_sr1.sub(&one_s).is_zero(),
            "is_sigma_row[1] should be 1 on round-11 row",
        );
        for r in [0, 2, 3, 4, 5, 6, 7, 8, 9].iter() {
            let v = &trace.columns[col_is_sigma_row(*r)].evaluations[0];
            assert!(v.is_zero(), "is_sigma_row[{}] should be 0", r);
        }
        // round_mod_10 = 1, q = 1.
        assert!(
            trace.columns[COL_ROUND_MOD_10].evaluations[0]
                .sub(&one_s)
                .is_zero(),
        );
        assert!(
            trace.columns[COL_ROUND_QUOTIENT_10].evaluations[0]
                .sub(&one_s)
                .is_zero(),
        );
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn blake2_f_internals_sigma_tampered_m_sched_detected_round0() {
        // Round 0: SIGMA[0] = identity. Tampering m_sched[5] breaks the
        // `is_sigma_row[0] * (m_sched[5] - m[5]) = 0` constraint.
        let trace = honest_round_trace(0);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[col_m_sched(5)][0] = Scalar::from_u64(0xdeadbeef, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[sigma_bind_idx(0, 5)][0].is_zero(),
            "expected σ-binding (sr=0, k=5) to fire after tampering m_sched[5]",
        );
    }

    #[test]
    fn blake2_f_internals_sigma_tampered_m_sched_detected_round3() {
        // Round 3: SIGMA[3] = [7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8].
        // m_sched[0] should equal m[7]. Tampering m_sched[0] breaks the
        // (sr=3, k=0) constraint.
        let trace = honest_round_trace(3);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[col_m_sched(0)][0] =
            Scalar::from_u64(0xc0ffee_u64, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[sigma_bind_idx(3, 0)][0].is_zero(),
            "expected σ-binding (sr=3, k=0) to fire after tampering m_sched[0]",
        );
    }

    #[test]
    fn blake2_f_internals_sigma_tampered_one_hot_detected() {
        // Replace the correct one-hot selector with a wrong sigma row.
        // Round 0 lights is_sigma_row[0]; flip to is_sigma_row[3].
        let trace = honest_round_trace(0);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let curve = CurveType::Bls48581;
        cols[col_is_sigma_row(0)][0] = Scalar::zero(curve);
        cols[col_is_sigma_row(3)][0] = Scalar::one(curve);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // The one-hot-binds-round_mod_10 constraint should fire: Σ r*sel = 3
        // but round_mod_10 = 0.
        assert!(
            !bodies[sigma_aux_one_hot_binds_idx()][0].is_zero(),
            "expected one-hot-binds-round_mod_10 to fire after wrong sigma row",
        );
    }

    #[test]
    fn blake2_f_internals_sigma_tampered_sum_to_one_detected() {
        // Zero all selectors: sum-to-one constraint must fire.
        let trace = honest_round_trace(0);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let curve = CurveType::Bls48581;
        for sr in 0..SIGMA_ROWS {
            cols[col_is_sigma_row(sr)][0] = Scalar::zero(curve);
        }
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[sigma_aux_one_hot_sum_idx()][0].is_zero(),
            "expected sum-to-one to fire when all selectors zeroed",
        );
    }

    #[test]
    fn blake2_f_internals_sigma_tampered_round_decomp_detected() {
        // round_index = 10*q + round_mod_10. Tamper round_mod_10 alone.
        let trace = honest_round_trace(0);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_ROUND_MOD_10][0] = Scalar::from_u64(5, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[sigma_aux_decomp_idx()][0].is_zero(),
            "expected round_index decomposition to fire",
        );
    }

    #[test]
    fn blake2_f_internals_sigma_evaluate_at_point_matches() {
        // Sanity: evaluate_at_point on an honest trace at row 0 returns 0
        // for every alpha (rank-1: the gated row-local sum is identically
        // zero per-row, so any alpha works). Spot-check with alpha = 7.
        let trace = honest_round_trace(0);
        let curve = CurveType::Bls48581;
        let alpha = Scalar::from_u64(7, curve);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        let v = cs.evaluate_at_point(&col_evals, &alpha);
        assert!(v.is_zero(), "evaluate_at_point should be 0 on honest row");
    }

    // ─── IV-init / final h_out finalize tests (Task #206) ─────────

    /// Base offset of the IV-init layer within the row-local constraint
    /// vector: 6 (head) + 96 (G) + 32 * 259 (XOR/ROTR) + 174 (σ) = 8564.
    const IV_INIT_LAYER_BASE: usize =
        6 + NUM_G_CONSTRAINTS_TOTAL
            + NUM_XOR_ROTR_CONSTRAINTS_TOTAL
            + NUM_SIGMA_CONSTRAINTS_TOTAL;

    /// Constructor: build a 2-row IV-init witness with chosen (h_in, t,
    /// f_byte) and a zero result row, plus populate the σ aux columns to
    /// avoid triggering σ-layer constraints on the first row.
    fn iv_init_witness(
        h_in: [u8; H_IN_LENGTH],
        t: [u8; T_LENGTH],
        f_byte: u8,
    ) -> Blake2fInternalsTraceWitness {
        Blake2fInternalsTraceWitness::iv_init_then_result(
            h_in,
            t,
            f_byte,
            [0u8; RESULT_LENGTH],
        )
    }

    #[test]
    fn iv_init_constants_match_blake2b_iv() {
        // RFC 7693 §2.6: BLAKE2b IV[0..8] = SHA-512 initial hash values.
        assert_eq!(BLAKE2B_IV[0], 0x6a09e667f3bcc908);
        assert_eq!(BLAKE2B_IV[1], 0xbb67ae8584caa73b);
        assert_eq!(BLAKE2B_IV[2], 0x3c6ef372fe94f82b);
        assert_eq!(BLAKE2B_IV[3], 0xa54ff53a5f1d36f1);
        assert_eq!(BLAKE2B_IV[4], 0x510e527fade682d1);
        assert_eq!(BLAKE2B_IV[5], 0x9b05688c2b3e6c1f);
        assert_eq!(BLAKE2B_IV[6], 0x1f83d9abfb41bd6b);
        assert_eq!(BLAKE2B_IV[7], 0x5be0cd19137e2179);
        assert_eq!(F_BYTE_MASK_ONES, u64::MAX);
    }

    #[test]
    fn iv_init_honest_witness_passes_all_iv_constraints() {
        // Mix a nontrivial h_in + nonzero t + f=1 so v[12..15] take
        // distinct values from IV. The 14 IV-init constraints should
        // ALL fire as zero on row 0 (first round) and row 1 (result; no
        // gating).
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i * 17 + 3) as u8;
        }
        let mut t = [0u8; T_LENGTH];
        t[0] = 0x12; t[1] = 0x34; t[8] = 0x56; t[9] = 0x78;
        let w = iv_init_witness(h_in, t, 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for k in 0..NUM_IV_INIT_CONSTRAINTS {
            for (r, val) in bodies[IV_INIT_LAYER_BASE + k].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "IV-init constraint {} row {} nonzero on honest witness",
                    k, r,
                );
            }
        }
    }

    #[test]
    fn iv_init_tampered_h_in_binding_detected() {
        // Tamper v[3] (one of the h_in-bound limbs) on row 0. The
        // per-i pack(v[i]) == pack_h_in[i] constraint at i=3 must fire.
        let h_in = [0xa5u8; H_IN_LENGTH];
        let w = iv_init_witness(h_in, [0u8; T_LENGTH], 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Bump v_byte at index 3*8 by 1 on row 0.
        let col = COL_V_OFFSET + 3 * LIMB_BYTES;
        cols[col][0] = cols[col][0]
            .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Layout: index 0 is is_first_round binary; 1..=8 are h_in bindings
        // for i = 0..7. i=3 is at IV_INIT_LAYER_BASE + 1 + 3 = base + 4.
        let idx = IV_INIT_LAYER_BASE + 1 + 3;
        assert!(
            !bodies[idx][0].is_zero(),
            "expected pack(v[3]) == pack_h_in[3] to fire on tampered v",
        );
    }

    #[test]
    fn iv_init_tampered_iv_low_limb_detected() {
        // Tamper v[8] (which must equal IV[0]) — the IV pack constraint
        // at i=0 (one of the four "v[8..11] == IV[0..3]") must fire.
        let h_in = [0u8; H_IN_LENGTH];
        let w = iv_init_witness(h_in, [0u8; T_LENGTH], 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let col = COL_V_OFFSET + 8 * LIMB_BYTES;
        cols[col][0] = cols[col][0]
            .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Layout: 1 binary + 8 h_in + 4 IV[0..3] starting at base + 9.
        let idx = IV_INIT_LAYER_BASE + 1 + 8 + 0;
        assert!(
            !bodies[idx][0].is_zero(),
            "expected pack(v[8]) == IV[0] to fire on tampered v[8]",
        );
    }

    #[test]
    fn iv_init_tampered_v15_iv7_detected() {
        // v[15] must equal IV[7]. Bump v[15] byte 0 and check the
        // v15==IV[7] constraint fires.
        let h_in = [0u8; H_IN_LENGTH];
        let w = iv_init_witness(h_in, [0u8; T_LENGTH], 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let col = COL_V_OFFSET + 15 * LIMB_BYTES;
        cols[col][0] = cols[col][0]
            .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Last IV-init constraint: base + 1 + 8 + 4 = base + 13.
        let idx = IV_INIT_LAYER_BASE + 1 + 8 + 4;
        assert!(
            !bodies[idx][0].is_zero(),
            "expected pack(v[15]) == IV[7] to fire on tampered v[15]",
        );
    }

    #[test]
    fn iv_init_is_first_round_binary_constraint_fires() {
        // Set is_first_round = 2 (non-binary) on row 0; the binary
        // constraint at the top of the layer must fire.
        let w = iv_init_witness([0u8; H_IN_LENGTH], [0u8; T_LENGTH], 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_FIRST_ROUND][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[IV_INIT_LAYER_BASE][0].is_zero(),
            "expected is_first_round binary to fire on non-binary value",
        );
    }

    #[test]
    fn iv_init_evaluate_at_point_matches_on_domain() {
        // Honest IV-init witness; verify evaluate_at_point matches the
        // α-folded RLC of the on-domain bodies for a sample alpha.
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i * 11 + 1) as u8;
        }
        let mut t = [0u8; T_LENGTH];
        t[0] = 0xaa;
        let w = iv_init_witness(h_in, t, 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        let alpha = Scalar::from_u64(13, CurveType::Bls48581);
        for row in 0..trace.num_rows {
            let col_evals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            let at_point = cs.evaluate_at_point(&col_evals, &alpha);
            let mut expected = Scalar::zero(CurveType::Bls48581);
            let mut ap = Scalar::one(CurveType::Bls48581);
            for body in bodies.iter() {
                expected = expected.add(&ap.mul(&body[row]));
                ap = ap.mul(&alpha);
            }
            assert!(
                at_point.sub(&expected).is_zero(),
                "row {} IV-init evaluate_at_point != on-domain RLC",
                row,
            );
        }
    }

    // ─── IV XOR bit-decomposition tests (Task #215) ───────────────

    /// Base offset of the IV XOR layer within the row-local constraint
    /// vector. Layout order matches `evaluate_on_domain`:
    ///   head(6) + G(96) + XOR/ROTR(32*259) + σ(174) + IV-init(14).
    const IV_XOR_LAYER_BASE: usize = 6
        + NUM_G_CONSTRAINTS_TOTAL
        + NUM_XOR_ROTR_CONSTRAINTS_TOTAL
        + NUM_SIGMA_CONSTRAINTS_TOTAL
        + NUM_IV_INIT_CONSTRAINTS;

    fn iv_xor_instance_layer_base(inst_idx: usize) -> usize {
        IV_XOR_LAYER_BASE + inst_idx * NUM_IV_XOR_CONSTRAINTS_PER_INSTANCE
    }

    #[test]
    fn iv_xor_layout_pinned() {
        // The three instances close v[12], v[13], v[14] against IV[4..7].
        assert_eq!(IV_XOR_IV_INDEX, [4, 5, 6]);
        assert_eq!(IV_XOR_V_INDEX, [12, 13, 14]);
        // Per-instance constraint counts.
        assert_eq!(NUM_IV_XOR_BINARY_PER_INSTANCE, 192);
        assert_eq!(NUM_IV_XOR_PACK_PER_INSTANCE, 3);
        assert_eq!(NUM_IV_XOR_BITWISE_PER_INSTANCE, 64);
        // B operand sources.
        assert_eq!(IV_XOR_B_SOURCE[0], IvXorBSource::TBytes(0));
        assert_eq!(IV_XOR_B_SOURCE[1], IvXorBSource::TBytes(LIMB_BYTES));
        assert_eq!(IV_XOR_B_SOURCE[2], IvXorBSource::FByteMask);
    }

    #[test]
    fn iv_xor_honest_witness_passes_all_constraints() {
        // Nontrivial h_in + t + f=1 → v[12] = IV[4]^t.lo, v[13] = IV[5]^t.hi,
        // v[14] = IV[6]^0xFFFF..FF. iv_init_then_result computes v
        // algebraically; the IV XOR layer must accept all 3 * 259 = 777
        // constraints on row 0 (gated by is_first_round=1).
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i * 13 + 7) as u8;
        }
        let mut t = [0u8; T_LENGTH];
        // Distinct nonzero t.lo and t.hi exercise both TBytes instances.
        t[0] = 0xab; t[1] = 0xcd; t[2] = 0xef;
        t[8] = 0x12; t[9] = 0x34; t[10] = 0x56;
        let w = iv_init_witness(h_in, t, 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for k in 0..NUM_IV_XOR_CONSTRAINTS_TOTAL {
            for (r, val) in bodies[IV_XOR_LAYER_BASE + k].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "IV XOR constraint {} row {} nonzero on honest witness",
                    k, r,
                );
            }
        }
    }

    #[test]
    fn iv_xor_honest_witness_f_byte_zero_passes() {
        // f_byte = 0 → mask = 0 → v[14] = IV[6]. Still must satisfy all
        // 777 IV XOR constraints (B operand for instance 2 is just zero).
        let h_in = [0xa5u8; H_IN_LENGTH];
        let mut t = [0u8; T_LENGTH];
        t[0] = 0xff;
        let w = iv_init_witness(h_in, t, 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for k in 0..NUM_IV_XOR_CONSTRAINTS_TOTAL {
            for (r, val) in bodies[IV_XOR_LAYER_BASE + k].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "IV XOR (f=0) constraint {} row {} nonzero",
                    k, r,
                );
            }
        }
    }

    /// Tampering t.lo on row 0 without rewriting the v[12] bit decomp
    /// must fire one of the instance-0 constraints (pack(B) ≠ t.lo).
    #[test]
    fn iv_xor_tampered_t_lo_detected() {
        let h_in = [0u8; H_IN_LENGTH];
        let mut t = [0u8; T_LENGTH];
        t[0] = 0x55;
        let w = iv_init_witness(h_in, t, 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Bump t-byte 0 (which feeds t.lo target for instance 0) on row 0
        // without updating the B bit decomp of instance 0. The pack(B)
        // constraint of instance 0 must fire.
        cols[COL_T_OFFSET][0] = cols[COL_T_OFFSET][0]
            .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Layout per instance: 192 binary + pack_A + pack_B + pack_dst + 64 XOR.
        // pack_B is at +193.
        let pack_b_idx = iv_xor_instance_layer_base(0)
            + NUM_IV_XOR_BINARY_PER_INSTANCE + 1;
        assert!(
            !bodies[pack_b_idx][0].is_zero(),
            "expected instance 0 pack(B) to fire when t.lo tampered",
        );
    }

    /// Tampering t.hi must fire instance 1's pack(B) constraint.
    #[test]
    fn iv_xor_tampered_t_hi_detected() {
        let h_in = [0u8; H_IN_LENGTH];
        let mut t = [0u8; T_LENGTH];
        t[8] = 0xaa;
        let w = iv_init_witness(h_in, t, 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Bump t-byte 8 (start of t.hi) on row 0.
        cols[COL_T_OFFSET + LIMB_BYTES][0] = cols[COL_T_OFFSET + LIMB_BYTES][0]
            .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Instance 1's pack(B).
        let pack_b_idx = iv_xor_instance_layer_base(1)
            + NUM_IV_XOR_BINARY_PER_INSTANCE + 1;
        assert!(
            !bodies[pack_b_idx][0].is_zero(),
            "expected instance 1 pack(B) to fire when t.hi tampered",
        );
    }

    /// Tampering f_byte must fire instance 2's pack(B) constraint.
    /// (f_byte * 0xFFFF..FF is the B target for instance 2.)
    #[test]
    fn iv_xor_tampered_f_byte_detected() {
        // f_byte starts at 1; flip to 0 on row 0 without rewriting the
        // dst bits — instance 2's pack(B) must fire (target becomes 0
        // but B bits still pack to all-ones).
        let h_in = [0u8; H_IN_LENGTH];
        let w = iv_init_witness(h_in, [0u8; T_LENGTH], 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_F_BYTE][0] = Scalar::zero(CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let pack_b_idx = iv_xor_instance_layer_base(2)
            + NUM_IV_XOR_BINARY_PER_INSTANCE + 1;
        assert!(
            !bodies[pack_b_idx][0].is_zero(),
            "expected instance 2 pack(B) to fire when f_byte tampered",
        );
    }

    /// Tampering v[12] dst-byte without updating the dst bit decomp must
    /// fire instance 0's pack(dst) constraint.
    #[test]
    fn iv_xor_tampered_v12_detected() {
        let h_in = [0u8; H_IN_LENGTH];
        let mut t = [0u8; T_LENGTH];
        t[0] = 0x11;
        let w = iv_init_witness(h_in, t, 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Bump v[12] byte 0.
        cols[COL_V_OFFSET + 12 * LIMB_BYTES][0] =
            cols[COL_V_OFFSET + 12 * LIMB_BYTES][0]
                .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // pack(dst) is at +194.
        let pack_dst_idx = iv_xor_instance_layer_base(0)
            + NUM_IV_XOR_BINARY_PER_INSTANCE + 2;
        assert!(
            !bodies[pack_dst_idx][0].is_zero(),
            "expected instance 0 pack(dst) to fire when v[12] tampered",
        );
    }

    /// Tampering a bit cell (non-binary) on any IV XOR instance must fire
    /// the matching binary constraint.
    #[test]
    fn iv_xor_per_instance_binary_tamper_fires() {
        let h_in = [0u8; H_IN_LENGTH];
        let mut t = [0u8; T_LENGTH];
        t[0] = 0x21;
        t[8] = 0x43;
        for inst_idx in 0..IV_XOR_INSTANCE_COUNT {
            let w = iv_init_witness(h_in, t, 1);
            let trace = build_trace_polynomials(&w, CurveType::Bls48581);
            let mut cols: Vec<Vec<Scalar>> = trace
                .columns
                .iter()
                .map(|p| p.evaluations.clone())
                .collect();
            // Flip A bit 0 to 2 (non-binary) on row 0.
            cols[col_iv_xor_bit(inst_idx, 0, 0)][0] =
                Scalar::from_u64(2, CurveType::Bls48581);
            let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            let bin_idx = iv_xor_instance_layer_base(inst_idx);
            assert!(
                !bodies[bin_idx][0].is_zero(),
                "IV XOR instance {} binary tamper did not fire",
                inst_idx,
            );
        }
    }

    /// Flipping a dst bit to its complement (still in {0,1}) without
    /// updating A/B must fire the XOR constraint at that bit position.
    #[test]
    fn iv_xor_per_instance_xor_tamper_fires() {
        let h_in = [0u8; H_IN_LENGTH];
        let mut t = [0u8; T_LENGTH];
        t[0] = 0xab;
        t[8] = 0xcd;
        let one_s = Scalar::one(CurveType::Bls48581);
        for inst_idx in 0..IV_XOR_INSTANCE_COUNT {
            let w = iv_init_witness(h_in, t, 1);
            let trace = build_trace_polynomials(&w, CurveType::Bls48581);
            let mut cols: Vec<Vec<Scalar>> = trace
                .columns
                .iter()
                .map(|p| p.evaluations.clone())
                .collect();
            // Flip dst bit 3.
            let dst_idx = 3usize;
            let cur = cols[col_iv_xor_bit(inst_idx, 2, dst_idx)][0].clone();
            cols[col_iv_xor_bit(inst_idx, 2, dst_idx)][0] = one_s.sub(&cur);
            let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            // XOR constraints start at layer_base + 192 (binary) + 3 (pack).
            let xor_base = iv_xor_instance_layer_base(inst_idx)
                + NUM_IV_XOR_BINARY_PER_INSTANCE
                + NUM_IV_XOR_PACK_PER_INSTANCE;
            // rot=0 → dst bit i corresponds to source bit i.
            assert!(
                !bodies[xor_base + dst_idx][0].is_zero(),
                "IV XOR instance {} XOR tamper at dst={} did not fire",
                inst_idx, dst_idx,
            );
        }
    }

    /// All IV XOR constraints are gated by `is_first_round` — on a witness
    /// where `is_first_round = 0` everywhere (e.g. `from_result_skeleton`),
    /// even an entirely zero bit-decomp witness must pass all 777
    /// constraints.
    #[test]
    fn iv_xor_gated_off_on_non_first_round_rows() {
        let w = Blake2fInternalsTraceWitness::from_result_skeleton(
            [0u8; RESULT_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for k in 0..NUM_IV_XOR_CONSTRAINTS_TOTAL {
            for (r, val) in bodies[IV_XOR_LAYER_BASE + k].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "IV XOR constraint {} row {} nonzero on non-first-round witness",
                    k, r,
                );
            }
        }
    }

    #[test]
    fn iv_xor_evaluate_at_point_matches_on_domain() {
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i * 7 + 9) as u8;
        }
        let mut t = [0u8; T_LENGTH];
        t[0] = 0x33; t[8] = 0x55;
        let w = iv_init_witness(h_in, t, 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        let alpha = Scalar::from_u64(19, CurveType::Bls48581);
        for row in 0..trace.num_rows {
            let col_evals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            let at_point = cs.evaluate_at_point(&col_evals, &alpha);
            let mut expected = Scalar::zero(CurveType::Bls48581);
            let mut ap = Scalar::one(CurveType::Bls48581);
            for body in bodies.iter() {
                expected = expected.add(&ap.mul(&body[row]));
                ap = ap.mul(&alpha);
            }
            assert!(
                at_point.sub(&expected).is_zero(),
                "row {} IV XOR evaluate_at_point != on-domain RLC",
                row,
            );
        }
    }

    // ─── h_out finalize triple-XOR tests (Task #216) ──────────────

    /// Base offset of the h_out finalize XOR layer within the row-local
    /// constraint vector. Layout order matches `evaluate_on_domain`:
    ///   head(6) + G(96) + XOR/ROTR(32*259) + σ(174) + IV-init(14)
    ///   + IV XOR(3*259).
    const H_OUT_XOR_LAYER_BASE: usize = 6
        + NUM_G_CONSTRAINTS_TOTAL
        + NUM_XOR_ROTR_CONSTRAINTS_TOTAL
        + NUM_SIGMA_CONSTRAINTS_TOTAL
        + NUM_IV_INIT_CONSTRAINTS
        + NUM_IV_XOR_CONSTRAINTS_TOTAL;

    fn h_out_xor_instance_layer_base(inst_idx: usize) -> usize {
        H_OUT_XOR_LAYER_BASE
            + inst_idx * NUM_H_OUT_XOR_CONSTRAINTS_PER_INSTANCE
    }

    #[test]
    fn h_out_xor_honest_witness_passes_all_constraints() {
        // Nontrivial h_in + v_final; honest result_h is the per-limb
        // triple-XOR. All 8 × 324 = 2592 finalize constraints must hold.
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i.wrapping_mul(31) + 7) as u8;
        }
        let mut v_final = [0u8; V_FINAL_LENGTH];
        for (i, b) in v_final.iter_mut().enumerate() {
            *b = (i.wrapping_mul(53) + 113) as u8;
        }
        let w = Blake2fInternalsTraceWitness::final_row_only(h_in, v_final);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        for k in 0..NUM_H_OUT_XOR_CONSTRAINTS_TOTAL {
            for (r, val) in bodies[H_OUT_XOR_LAYER_BASE + k].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "h_out XOR constraint {} row {} nonzero on honest witness",
                    k, r,
                );
            }
        }
    }

    #[test]
    fn h_out_xor_zero_witness_passes() {
        // All-zero h_in + v_final + result_h trivially satisfies all
        // finalize constraints (gated by is_result_row which is 1 on
        // the result row; bits are zero so XOR-of-3 = 0 = h_out_bit).
        let w = Blake2fInternalsTraceWitness::final_row_only(
            [0u8; H_IN_LENGTH],
            [0u8; V_FINAL_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn h_out_xor_tampered_result_h_byte_detected() {
        // Honest h_in / v_final / result_h, then bump result_h byte 0
        // (the LSB of limb 0). The pack(h_out) constraint for inst 0
        // (or one of the XOR3 bit constraints) must fire.
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i.wrapping_mul(11) + 1) as u8;
        }
        let mut v_final = [0u8; V_FINAL_LENGTH];
        for (i, b) in v_final.iter_mut().enumerate() {
            *b = (i.wrapping_mul(7) + 2) as u8;
        }
        let w = Blake2fInternalsTraceWitness::final_row_only(h_in, v_final);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper byte 0 of result_h (instance 0, byte 0): adds 1.
        let bad_col = COL_RESULT_OFFSET; // instance 0 LSB
        let r0 = 0;
        cols[bad_col][r0] = cols[bad_col][r0]
            .add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // pack h_out for instance 0 is at:
        //   layer_base + 256 (binary) + 3 (pack h_in, v_lo, v_hi) = +259.
        let pack_h_out_idx = h_out_xor_instance_layer_base(0)
            + NUM_H_OUT_XOR_BINARY_PER_INSTANCE
            + 3;
        assert!(
            !bodies[pack_h_out_idx][r0].is_zero(),
            "expected pack(h_out) for instance 0 to fire on tampered result_h byte",
        );
    }

    #[test]
    fn h_out_xor_tampered_h_out_bit_detected() {
        // Honest witness, then flip an h_out bit cell on the result row
        // (in {0,1}). The XOR-of-3 fused constraint at that bit fires.
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i.wrapping_mul(19) + 5) as u8;
        }
        let mut v_final = [0u8; V_FINAL_LENGTH];
        for (i, b) in v_final.iter_mut().enumerate() {
            *b = (i.wrapping_mul(23) + 11) as u8;
        }
        let w = Blake2fInternalsTraceWitness::final_row_only(h_in, v_final);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let one_s = Scalar::one(CurveType::Bls48581);
        // Flip h_out bit 7 of instance 3.
        let inst_idx = 3usize;
        let bit_idx = 7usize;
        let col = col_h_out_xor_bit(inst_idx, 3, bit_idx);
        let r0 = 0;
        let cur = cols[col][r0].clone();
        cols[col][r0] = one_s.sub(&cur);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // XOR3 constraints are at:
        //   layer_base + 256 (binary) + 4 (pack) = +260, then 64 entries.
        let xor3_base = h_out_xor_instance_layer_base(inst_idx)
            + NUM_H_OUT_XOR_BINARY_PER_INSTANCE
            + NUM_H_OUT_XOR_PACK_PER_INSTANCE;
        assert!(
            !bodies[xor3_base + bit_idx][r0].is_zero(),
            "expected XOR3 constraint at instance {} bit {} to fire",
            inst_idx, bit_idx,
        );
    }

    #[test]
    fn h_out_xor_tampered_h_in_bit_binary_detected() {
        // Set an h_in bit cell to 2 (non-binary) on row 0; the binary
        // cell constraint at the top of the instance layer must fire.
        let w = Blake2fInternalsTraceWitness::final_row_only(
            [0u8; H_IN_LENGTH],
            [0u8; V_FINAL_LENGTH],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let inst_idx = 5usize;
        let col = col_h_out_xor_bit(inst_idx, 0, 0); // h_in bit 0
        let r0 = 0;
        cols[col][r0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // First binary constraint for this instance is at layer_base + 0.
        let bin_idx = h_out_xor_instance_layer_base(inst_idx);
        assert!(
            !bodies[bin_idx][r0].is_zero(),
            "expected h_in bit 0 binary constraint (instance {}) to fire",
            inst_idx,
        );
    }

    #[test]
    fn h_out_xor_evaluate_at_point_matches_on_domain() {
        // Honest finalize witness; verify evaluate_at_point matches the
        // α-folded RLC of the on-domain bodies.
        let mut h_in = [0u8; H_IN_LENGTH];
        for (i, b) in h_in.iter_mut().enumerate() {
            *b = (i.wrapping_mul(13) + 17) as u8;
        }
        let mut v_final = [0u8; V_FINAL_LENGTH];
        for (i, b) in v_final.iter_mut().enumerate() {
            *b = (i.wrapping_mul(41) + 29) as u8;
        }
        let w = Blake2fInternalsTraceWitness::final_row_only(h_in, v_final);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);
        for row in 0..trace.num_rows {
            let col_evals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            let at_point = cs.evaluate_at_point(&col_evals, &alpha);
            let mut expected = Scalar::zero(CurveType::Bls48581);
            let mut ap = Scalar::one(CurveType::Bls48581);
            for body in bodies.iter() {
                expected = expected.add(&ap.mul(&body[row]));
                ap = ap.mul(&alpha);
            }
            assert!(
                at_point.sub(&expected).is_zero(),
                "row {} h_out XOR evaluate_at_point != on-domain RLC",
                row,
            );
        }
    }

    // ─── Coefficient-form vs evaluator consistency diagnostics (#225) ───
    //
    // Localizes any disagreement between `build_constraint_polynomial`
    // (poly form) and `evaluate_at_point` (per-row body form). On the
    // honest single-round trace these MUST agree at every domain root
    // ω^j: if any root j fails, the row index pinpoints which
    // constraint contribution mis-builds its polynomial.
    //
    // n = 16 (the minimum padded size from `nearest_power_of_two(2)`).

    /// Row-local: at every domain root ω^j the polynomial-form combined
    /// constraint (built from column coefficients via
    /// `build_constraint_polynomial`) must equal the per-row evaluator
    /// `evaluate_at_point(col(ω^j), α)`.
    #[test]
    fn blake2_f_internals_coefficient_vs_point_form_consistent() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let trace = honest_round_trace(0);
        let n = trace.padded_size;
        assert!(n >= 16, "expected padded_size >= 16, got {}", n);

        // Column coefficient form via IFFT.
        let coeff_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| CommitmentScheme::ifft(&scheme, &p.evaluations, n))
            .collect();

        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(29, curve);

        // Build the combined row-local constraint polynomial.
        let c_coeffs = cs.build_constraint_polynomial(&coeff_form, &alpha, n);

        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let mut omega_pow = Scalar::one(curve);
        for j in 0..(n as usize) {
            // Path A: evaluate c_coeffs at ω^j.
            let c_at_root = CommitmentScheme::eval_poly_at(
                &scheme, &c_coeffs, &omega_pow,
            );
            // Path B: evaluate each column poly at ω^j, then call
            // evaluate_at_point. This should equal the per-row body RLC
            // computed on-domain (which we know matches
            // `evaluate_on_domain` from existing `*_evaluate_at_point_*`
            // tests).
            let col_at_root: Vec<Scalar> = coeff_form
                .iter()
                .map(|c| {
                    CommitmentScheme::eval_poly_at(&scheme, c, &omega_pow)
                })
                .collect();
            let c_point = cs.evaluate_at_point(&col_at_root, &alpha);

            let diff = c_at_root.sub(&c_point);
            assert!(
                diff.is_zero(),
                "row-local poly form disagrees with evaluator at ω^{} \
                 (row index {}); build_constraint_polynomial vs \
                 evaluate_at_point mismatch — localizes the bug to a \
                 per-category builder for this row.",
                j, j,
            );

            omega_pow = omega_pow.mul(&omega);
        }
    }

    /// Combined row-local + shifted form: at every domain root ω^j the
    /// sum (row_poly + shifted_poly) evaluated at ω^j must equal
    /// row evaluator + shifted-row evaluator, accounting for the
    /// `(z - ω^{n-1})` factor baked into `evaluate_shifted_at_point`.
    #[test]
    fn blake2_f_internals_shifted_coefficient_vs_point_form_consistent() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let trace = honest_round_trace(0);
        let n = trace.padded_size;

        let coeff_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| CommitmentScheme::ifft(&scheme, &p.evaluations, n))
            .collect();

        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(31, curve);

        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(n - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(&omega);
        }

        // Row + shifted polynomials.
        let row_poly = cs.build_constraint_polynomial(
            &coeff_form, &alpha, n,
        );
        let shifted_poly = cs.build_shifted_constraint_polynomial(
            &coeff_form, &alpha, n, &omega, cs.num_constraints(),
        );

        // Sum them.
        let mut full = row_poly.clone();
        if full.len() < shifted_poly.len() {
            full.resize(shifted_poly.len(), Scalar::zero(curve));
        }
        let mut sp = shifted_poly.clone();
        if sp.len() < full.len() {
            sp.resize(full.len(), Scalar::zero(curve));
        }
        for i in 0..full.len() {
            full[i] = full[i].add(&sp[i]);
        }

        // Shifted column indices, in order.
        let shifted_cols = cs.shifted_column_indices();
        let alpha_offset = cs.num_constraints();

        let mut omega_pow = Scalar::one(curve);
        for j in 0..(n as usize) {
            // Evaluate every column poly at z = ω^j AND at ω^{j+1}.
            let z = omega_pow.clone();
            let z_next = z.mul(&omega);

            let col_at_z: Vec<Scalar> = coeff_form
                .iter()
                .map(|c| CommitmentScheme::eval_poly_at(&scheme, c, &z))
                .collect();

            // Shifted evals: column(ω*z) for each shifted column index.
            let shifted_evals: Vec<Scalar> = shifted_cols
                .iter()
                .map(|&ci| {
                    CommitmentScheme::eval_poly_at(
                        &scheme, &coeff_form[ci], &z_next,
                    )
                })
                .collect();

            // Path A: poly form (already includes (z - ω^{n-1}) factor
            // baked into build_shifted_constraint_polynomial).
            let c_full = CommitmentScheme::eval_poly_at(&scheme, &full, &z);

            // Path B: row evaluator + shifted evaluator.
            let row_point = cs.evaluate_at_point(&col_at_z, &alpha);
            let shifted_point = cs.evaluate_shifted_at_point(
                &col_at_z,
                &shifted_evals,
                &z,
                &omega_n_minus_1,
                &alpha,
                alpha_offset,
            );
            let c_point = row_point.add(&shifted_point);

            let diff = c_full.sub(&c_point);
            assert!(
                diff.is_zero(),
                "row+shifted poly form disagrees with evaluator at ω^{} \
                 (row index {}); a per-category build_*_poly disagrees \
                 with its evaluate_*_at_point — localizes the prove/verify \
                 mismatch.",
                j, j,
            );

            omega_pow = z_next;
        }
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

        let trace = honest_round_trace(0);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = Blake2fInternalsConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone blake2_f_internals_air proof must verify",
        );
    }
}
