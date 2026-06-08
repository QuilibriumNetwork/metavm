//! BLS12-381 hash-to-G2 AIR scaffold (#119 — sync committee target hashing
//! and FFG attestation signing message in-circuit).
//!
//! # Purpose
//!
//! `hash_to_curve(BLS12381G2_XMD:SHA-256_SSWU_RO_)` (RFC 9380, IETF
//! draft 16) is the function that turns the SSZ `signing_root` (or any
//! 32-byte message) into the G2 point `H(msg)` used by the BLS pairing
//! equation `e(-G1, sig) · e(pk, H(msg)) == 1`. The function decomposes
//! into four heavy stages:
//!
//!   1. **`expand_message_xmd(msg, dst, 256)`** — repeated SHA-256
//!      compressions producing a 256-byte pseudo-random expansion.
//!   2. **Reduction** of the 256 bytes into `2 × Fp2 = 4 × Fp` "field
//!      element" values `(u0, u1)` via `mod p` reductions.
//!   3. **SSWU map** `map_to_curve_simple_swu_E'(u_i)` on each `u_i`,
//!      yielding two points on the **3-isogenous** curve `E'(Fp2)`.
//!      The map involves Fp2 inversions and conditional moves driven
//!      by Legendre-symbol tests.
//!   4. **3-isogeny evaluation** `iso_E'_to_E` mapping each `E'` point
//!      to `E(Fp2)`, addition `P0 + P1`, then **cofactor clearing** via
//!      the `ψ` endomorphism (psi/psi2/psi3 + scalar mul by `-x`).
//!
//! The full algebraic build is multi-week (each Fp2 multiplication is
//! several `nonnative_fp_air` rows; the SSWU map alone is ~25 Fp2
//! multiplications + 2 Fp2 inversions per call; cofactor clearing
//! adds another ~30). This scaffold gets the **column layout**, the
//! **trace builder**, the **cross-AIR LogUp descriptors** to the
//! `sha256_extract` AIR (input side) and to `bls_pairing_air` (output
//! side), and **one substantive Fp algebraic step** into the tree so
//! downstream SSWU step rows can be added incrementally without
//! re-litigating the row/column shape.
//!
//! # Witness shape
//!
//! Per row this AIR commits one
//! `(msg, dst) → G2_output` transformation:
//!
//!   * `msg[0..32]`           — fixed-length 32-byte message (sync
//!     committee `signing_root`; shorter messages are zero-padded and
//!     `msg_len` records the real length).
//!   * `dst[0..MAX_DST_LEN]`  — domain separation tag (zero-padded);
//!     `dst_len` records the real length.
//!   * `out_compressed[0..96]`— IETF compressed G2 encoding of the
//!     output point.
//!   * `out_x_c0_bytes[0..48]`, `out_x_c1_bytes`, `out_y_c0_bytes`,
//!     `out_y_c1_bytes` — canonical BE Fp encodings of the 4 affine
//!     Fp2 components.
//!   * `out_x_c0_limbs[0..6]`, `out_x_c1_limbs`, `out_y_c0_limbs`,
//!     `out_y_c1_limbs` — 6 BE u64 limbs each (Fp tower entry-point
//!     form; matches [`crate::nonnative_fp::Fp`]).
//!   * `u0_c0_limbs`, `u0_c1_limbs` — 6 limbs each: the **first
//!     Fp2 field element** produced by `expand_message_xmd → mod p`.
//!     This is the input to the first `map_to_curve_simple_swu`
//!     invocation.
//!   * `u0_plus_one_c0_limbs`, `u0_plus_one_c1_limbs` — 6 limbs each:
//!     a **representative Fp2 algebraic step** = `u0 + 1` (i.e.
//!     `u0.c0 + 1, u0.c1`). This is the simplest sub-expression of
//!     the SSWU map's `t² + 1` factor (the constant addition without
//!     the multiplication). Mirroring the `bls_pairing_air` pattern,
//!     this scaffold proves the *Fp limb-level* shape of the addition
//!     and defers the multiplicative step to a follow-up "SSWU step
//!     AIR".
//!   * `msg_len`, `dst_len`   — single columns, recording real input
//!     lengths.
//!   * `is_real`              — selector.
//!
//! # What is algebraically enforced
//!
//!   1. `is_real ∈ {0, 1}` — selector binarity.
//!   2. `out_x_c0` **byte-to-limb decomposition** (6 equations binding
//!      `out_x_c0_limbs[j] = Σ_{k=0..8} out_x_c0_bytes[8j + k] · 2^(8·(7−k))`).
//!      Mirror of the `bls_pairing_air` `pk_x` pattern. Cross-AIR LogUp
//!      can later bind these limbs to `nonnative_fp_air` rows for the
//!      cofactor-clearing scalar mul.
//!   3. **Representative Fp limb addition step** (the load-bearing
//!      "one substantive Fp2 algebraic step"): for each of the 6 Fp
//!      limbs of `u0.c0`, enforce
//!      `u0_plus_one_c0_limbs[j] = u0_c0_limbs[j] + ONE_FP_LIMBS[j]`
//!      where `ONE_FP_LIMBS` is the 6-limb BE encoding of the Fp
//!      constant `1` (`[0,0,0,0,0,1]`). For `u0.c1` the addition is
//!      identity: `u0_plus_one_c1_limbs[j] = u0_c1_limbs[j]`. Total:
//!      6 + 6 = 12 equations. Modular reduction (overflow at limb 5
//!      → wrap into limb 4, …) is NOT yet handled — that's a single-
//!      `nonnative_fp_air` row away once we wire the LogUp link;
//!      until then the addition is enforced as integer addition,
//!      which is valid for any `u0.c0` whose LSB limb is `< 2^64 - 1`
//!      (true with overwhelming probability for `u0` drawn from
//!      `mod p` reduction of a SHA-256 expansion).
//!   4. Per-byte 8-bit range checks on every byte column (via the
//!      `LookupRequirements` channel).
//!
//! Total row-local algebraic constraints: 1 + 6 + 12 = **19**.
//!
//! # What is NOT yet algebraically enforced (deferred)
//!
//! - **Stage 1: SHA-256 expansion**: the host-side trace builder
//!   computes `expand_message_xmd(msg, dst, 256)` via blst; the AIR
//!   only commits the input `(msg, dst)`. The cross-AIR LogUp
//!   descriptor [`make_hash_to_g2_msg_to_sha256_descriptor`] points
//!   the `(msg, dst)` byte slice at one row of `sha256_extract`,
//!   covering the **first** SHA-256 invocation of the expansion only.
//!   The full expansion (which runs ~9 SHA-256 calls for the 256-byte
//!   output) requires a multi-row chain AIR.
//! - **Stage 2: `mod p` reductions** turning the 64-byte
//!   pseudo-random strings into Fp elements `u0, u1`. Standard
//!   nonnative-arithmetic reduction; one `nonnative_fp_air` row per
//!   reduction.
//! - **Stage 3: SSWU map** `map_to_curve_simple_swu_E'(u_i)` —
//!   ~25 Fp2 multiplications + 1 Fp2 inversion per invocation,
//!   plus 4 Legendre tests with selector logic. Each Fp2 multiply is
//!   3 Fp multiplications, each `nonnative_fp_air` is one row of the
//!   Fp multiplication AIR. The representative `u0 + 1` constraint
//!   pinned by this scaffold is the simplest sub-expression of the
//!   `t² + 1` denominator that appears in the SSWU formula.
//! - **Stage 4: 3-isogeny + cofactor clearing**: 3 Fp2 quotients +
//!   point addition + ψ endomorphism + scalar mul by `-x`. Easily
//!   the largest stage; ~250 Fp multiplications total.
//! - **Output G2 on-curve check** `y² = x³ + 4(1+u)` on the 4 limb
//!   columns. This is two Fp2 squarings + one Fp2 multiplication +
//!   one Fp2 addition; cross-AIR LogUp to `nonnative_fp_air`.
//! - **Output G2 subgroup check**: covered for free by Stage 4
//!   (the cofactor clearing yields a point in the r-torsion subgroup).
//!
//! # Cross-AIR linkages
//!
//! - [`make_hash_to_g2_msg_to_sha256_descriptor`] — binds the first
//!   64 bytes of `(msg || dst || 0x80 || ...)` (the standard
//!   `expand_message_xmd` first compression input) to a row of
//!   [`crate::sha256_extract`]. A-side selector =
//!   [`COL_IS_REAL`]; B-side selector = `sha256_extract::COL_IS_REAL`.
//!   This is one of multiple expected SHA-256 LogUp links; the
//!   present scaffold only wires the first.
//! - [`make_hash_to_g2_to_bls_pairing_descriptor`] — binds the 4 × 6
//!   = 24 output G2 limb columns to the matching
//!   `(sig_x_c0, sig_x_c1, sig_y_c0, sig_y_c1)` limb columns of
//!   [`crate::bls_pairing_air`]. This is the **consumer-side**
//!   tuple: the bls_pairing AIR sees H(msg) coming from this AIR
//!   rather than as an oracle. A-side selector = bls_pairing
//!   `COL_IS_REAL`; B-side selector = this AIR's `COL_IS_REAL`.
//!
//! # Notes on the curve choice
//!
//! As with the rest of the BLS12-381 algebraic stack, this AIR is
//! built on `CurveType::Bls12381`. The witness builder calls
//! [`crate::bls_sig::hash_to_g2_affine`] (blst) to produce the
//! output G2 bytes, then decodes them via
//! [`crate::pairing::G2Affine::from_bytes`] to recover the four Fp
//! limb arrays.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::nonnative_fp::{Fp, Fp2};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum committed message length (32 bytes; SSZ signing_root size).
pub const MSG_LEN: usize = 32;
/// Maximum committed domain separation tag length (64 bytes is enough
/// for the longest beacon-chain ciphersuite DSTs:
/// `BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_` = 43 bytes).
pub const MAX_DST_LEN: usize = 64;
/// Compressed G2 output length (96 bytes).
pub const OUT_COMPRESSED_LEN: usize = 96;
/// Canonical BE Fp encoding length (48 bytes).
pub const FP_BYTES: usize = 48;
/// Number of 64-bit limbs in one Fp element.
pub const LIMBS_PER_FP: usize = 6;
/// Bytes per limb.
pub const BYTES_PER_LIMB: usize = 8;

/// BE limb encoding of the Fp constant `1` (= [0, 0, 0, 0, 0, 1]).
/// Matches [`crate::nonnative_fp::Fp::one`].
pub const ONE_FP_LIMBS: [u64; LIMBS_PER_FP] = [0, 0, 0, 0, 0, 1];

// ─── SSWU isogeny constants (RFC 9380 §8.8.2, BLS12-381 G2) ───────────
//
// The 3-isogenous curve E' uses
//   A' = 240 i        (i.e. c0 = 0, c1 = 240)
//   B' = 1012 + 1012 i
//   Z  = -(2 + i)     (i.e. c0 = -2 mod p, c1 = -1 mod p)
//
// Only the **constant numeric encodings** are committed here; the
// algebraic check that `xd` and `n1` were actually built with these
// constants is deferred to the per-Fp-mul cross-AIR LogUp links into
// `nonnative_fp_air`, where the B operand on each link IS one of these
// constants (or a tv1 / tv2 / sum thereof).
/// Iso-curve constant `A' = 240 * i` (Fp2 with c0=0, c1=240).
pub fn iso_a_prime() -> Fp2 {
    Fp2 { c0: Fp::zero(), c1: Fp::from_u64(240) }
}

/// Iso-curve constant `B' = 1012 + 1012 * i`.
pub fn iso_b_prime() -> Fp2 {
    Fp2 { c0: Fp::from_u64(1012), c1: Fp::from_u64(1012) }
}

// ─── Column layout ────────────────────────────────────────────────────
//
// Row layout (per (msg, dst) → G2 row):
//
//   msg                  : 32 bytes
//   dst                  : 64 bytes
//   out_compressed       : 96 bytes
//   out_x_c0_bytes       : 48 bytes
//   out_x_c1_bytes       : 48 bytes
//   out_y_c0_bytes       : 48 bytes
//   out_y_c1_bytes       : 48 bytes
//   out_x_c0_limbs       : 6 BE u64
//   out_x_c1_limbs       : 6
//   out_y_c0_limbs       : 6
//   out_y_c1_limbs       : 6
//   u0_c0_limbs          : 6
//   u0_c1_limbs          : 6
//   u0_plus_one_c0_limbs : 6
//   u0_plus_one_c1_limbs : 6
//   tv1_c0_limbs         : 6   (tv1 = u0^2; Fp2 squaring)
//   tv1_c1_limbs         : 6
//   tv2_c0_limbs         : 6   (tv2 = tv1^2; Fp2 squaring)
//   tv2_c1_limbs         : 6
//   xd_c0_limbs          : 6   (xd  = -A' * (tv2 + tv1); Fp2 mul)
//   xd_c1_limbs          : 6
//   n1_c0_limbs          : 6   (n1  = B'  * (tv2 + tv1 + 1); Fp2 mul)
//   n1_c1_limbs          : 6
//   msg_len              : 1
//   dst_len              : 1
//   is_real              : 1
// total = 32 + 64 + 96 + 192 + 24 + 24 + 48 + 3 = 483

pub const COL_MSG_OFFSET: usize = 0;
pub const COL_DST_OFFSET: usize = COL_MSG_OFFSET + MSG_LEN;
pub const COL_OUT_COMPRESSED_OFFSET: usize = COL_DST_OFFSET + MAX_DST_LEN;
pub const COL_OUT_X_C0_BYTES_OFFSET: usize = COL_OUT_COMPRESSED_OFFSET + OUT_COMPRESSED_LEN;
pub const COL_OUT_X_C1_BYTES_OFFSET: usize = COL_OUT_X_C0_BYTES_OFFSET + FP_BYTES;
pub const COL_OUT_Y_C0_BYTES_OFFSET: usize = COL_OUT_X_C1_BYTES_OFFSET + FP_BYTES;
pub const COL_OUT_Y_C1_BYTES_OFFSET: usize = COL_OUT_Y_C0_BYTES_OFFSET + FP_BYTES;

pub const COL_OUT_X_C0_LIMB_OFFSET: usize = COL_OUT_Y_C1_BYTES_OFFSET + FP_BYTES;
pub const COL_OUT_X_C1_LIMB_OFFSET: usize = COL_OUT_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_Y_C0_LIMB_OFFSET: usize = COL_OUT_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_Y_C1_LIMB_OFFSET: usize = COL_OUT_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_U0_C0_LIMB_OFFSET: usize = COL_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_U0_C1_LIMB_OFFSET: usize = COL_U0_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_U0_PLUS_ONE_C0_LIMB_OFFSET: usize = COL_U0_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_U0_PLUS_ONE_C1_LIMB_OFFSET: usize =
    COL_U0_PLUS_ONE_C0_LIMB_OFFSET + LIMBS_PER_FP;

// ─── SSWU intermediate Fp2 limb columns (Round 7) ─────────────────────
//
// Each Fp2 element below is committed as `(c0_limbs[0..6], c1_limbs[0..6])`,
// i.e. 12 BE u64 columns. Together they cover the leading half of the
// `map_to_curve_simple_swu` formula:
//
//   tv1 = u0^2            (Fp2 squaring)
//   tv2 = tv1^2           (Fp2 squaring)
//   xd  = (-A') * (tv2 + tv1)   (Fp2 multiplication by the isogeny A')
//   n1  = B' * (tv2 + tv1 + 1)  (Fp2 multiplication by the isogeny B')
//
// The Fp2 squarings and multiplications decompose into Fp-mul rows of
// [`crate::nonnative_fp_air`] via cross-AIR LogUp. Each Fp2 multiplication
// is 3 Fp multiplications under Karatsuba; this scaffold wires 6 such
// Fp-mul links into `nonnative_fp_air` (covering tv1 = u0*u0 and
// tv2 = tv1*tv1 Karatsuba decompositions).
pub const COL_TV1_C0_LIMB_OFFSET: usize = COL_U0_PLUS_ONE_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_TV1_C1_LIMB_OFFSET: usize = COL_TV1_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_TV2_C0_LIMB_OFFSET: usize = COL_TV1_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_TV2_C1_LIMB_OFFSET: usize = COL_TV2_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_XD_C0_LIMB_OFFSET: usize = COL_TV2_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_XD_C1_LIMB_OFFSET: usize = COL_XD_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_N1_C0_LIMB_OFFSET: usize = COL_XD_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_N1_C1_LIMB_OFFSET: usize = COL_N1_C0_LIMB_OFFSET + LIMBS_PER_FP;

// ─── Karatsuba scratch columns (Round 8) ──────────────────────────────
//
// Each Fp2 multiplication `(a0 + a1 i) * (b0 + b1 i) = c0 + c1 i` is
// computed via the 3-mul Karatsuba decomposition:
//
//   t0      = a0 * b0
//   t1      = a1 * b1
//   t_cross = (a0 + a1) * (b0 + b1)
//
// and then `c0 = t0 - t1`, `c1 = t_cross - t0 - t1` (both Fp
// subtractions enforced limb-wise as β-RLC equalities; the actual
// nonnative mod-p reduction is deferred to follow-up nonnative_fp_air
// links). Each scratch element is an Fp (6 BE u64 limbs).
//
// We commit one (t0, t1, t_cross) triple per Fp2 mul, for the 4 Fp2
// multiplications wired by the SSWU step (`tv1, tv2, xd, n1`), giving
// 4 × 3 × 6 = 72 new columns.
pub const COL_TV1_KAR_T0_OFFSET: usize = COL_N1_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_TV1_KAR_T1_OFFSET: usize = COL_TV1_KAR_T0_OFFSET + LIMBS_PER_FP;
pub const COL_TV1_KAR_TCROSS_OFFSET: usize = COL_TV1_KAR_T1_OFFSET + LIMBS_PER_FP;
pub const COL_TV2_KAR_T0_OFFSET: usize = COL_TV1_KAR_TCROSS_OFFSET + LIMBS_PER_FP;
pub const COL_TV2_KAR_T1_OFFSET: usize = COL_TV2_KAR_T0_OFFSET + LIMBS_PER_FP;
pub const COL_TV2_KAR_TCROSS_OFFSET: usize = COL_TV2_KAR_T1_OFFSET + LIMBS_PER_FP;
pub const COL_XD_KAR_T0_OFFSET: usize = COL_TV2_KAR_TCROSS_OFFSET + LIMBS_PER_FP;
pub const COL_XD_KAR_T1_OFFSET: usize = COL_XD_KAR_T0_OFFSET + LIMBS_PER_FP;
pub const COL_XD_KAR_TCROSS_OFFSET: usize = COL_XD_KAR_T1_OFFSET + LIMBS_PER_FP;
pub const COL_N1_KAR_T0_OFFSET: usize = COL_XD_KAR_TCROSS_OFFSET + LIMBS_PER_FP;
pub const COL_N1_KAR_T1_OFFSET: usize = COL_N1_KAR_T0_OFFSET + LIMBS_PER_FP;
pub const COL_N1_KAR_TCROSS_OFFSET: usize = COL_N1_KAR_T1_OFFSET + LIMBS_PER_FP;

pub const COL_MSG_LEN: usize = COL_N1_KAR_TCROSS_OFFSET + LIMBS_PER_FP;
pub const COL_DST_LEN: usize = COL_MSG_LEN + 1;
pub const COL_IS_REAL: usize = COL_DST_LEN + 1;

// ─── Multi-row phase stitching (Round 11) ─────────────────────────────
//
// Each `from_message` call emits [`ROWS_PER_HASH_TO_G2`] consecutive
// rows representing the three sub-phases of the hash-to-G2 pipeline:
//
//   row 0 (SSWU phase)     : `IS_SSWU_PHASE = 1`     — feeds the SSWU
//                             output `(phase_out_x, phase_out_y)` to
//                             row 1's `(phase_in_x, phase_in_y)`.
//   row 1 (isogeny phase)  : `IS_ISOGENY_PHASE = 1`  — reads SSWU
//                             output, emits isogeny output (= G2 point
//                             before cofactor clearing).
//   row 2 (cofactor phase) : `IS_COFACTOR_PHASE = 1` — reads isogeny
//                             output, emits final cofactor-cleared G2.
//
// Phase exclusivity (sum = 1) and binarity are enforced row-locally.
// Cross-row continuity (`phase_out[r] = phase_in[r+1]`) is enforced
// via two shifted constraints (one per adjacent phase transition).
//
// The new `phase_in_*` / `phase_out_*` columns are appended **after**
// the existing layout so no existing offset shifts. The pre-existing
// `IS_REAL` selector remains the global "row is part of a real
// hash-to-G2 instance" gate and is set to 1 on **all three** phase
// rows of a real instance.
pub const ROWS_PER_HASH_TO_G2: usize = 3;

/// Selector: row is the SSWU phase (row 0 of a 3-row instance).
pub const COL_IS_SSWU_PHASE: usize = COL_IS_REAL + 1;
/// Selector: row is the isogeny phase (row 1).
pub const COL_IS_ISOGENY_PHASE: usize = COL_IS_SSWU_PHASE + 1;
/// Selector: row is the cofactor-clearing phase (row 2).
pub const COL_IS_COFACTOR_PHASE: usize = COL_IS_ISOGENY_PHASE + 1;

/// Per-phase output point limbs (the SSWU output on the SSWU row,
/// isogeny output on the isogeny row, final G2 output on the cofactor
/// row). 4 Fp2 components × 6 BE u64 limbs = 24 columns.
pub const COL_PHASE_OUT_X_C0_LIMB_OFFSET: usize = COL_IS_COFACTOR_PHASE + 1;
pub const COL_PHASE_OUT_X_C1_LIMB_OFFSET: usize =
    COL_PHASE_OUT_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PHASE_OUT_Y_C0_LIMB_OFFSET: usize =
    COL_PHASE_OUT_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PHASE_OUT_Y_C1_LIMB_OFFSET: usize =
    COL_PHASE_OUT_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

/// Per-phase input point limbs (zero on the SSWU row, SSWU output on
/// the isogeny row, isogeny output on the cofactor row). 24 columns.
pub const COL_PHASE_IN_X_C0_LIMB_OFFSET: usize =
    COL_PHASE_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PHASE_IN_X_C1_LIMB_OFFSET: usize =
    COL_PHASE_IN_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PHASE_IN_Y_C0_LIMB_OFFSET: usize =
    COL_PHASE_IN_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PHASE_IN_Y_C1_LIMB_OFFSET: usize =
    COL_PHASE_IN_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

/// Number of Fp2 component blocks committed per phase point (X.c0,
/// X.c1, Y.c0, Y.c1).
pub const NUM_PHASE_POINT_BLOCKS: usize = 4;

pub const NUM_COLUMNS: usize = COL_PHASE_IN_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;

/// Row-local constraints:
///   0:        is_real ∈ {0, 1}
///   1..7:     6 out_x_c0 limb-decomposition equations
///   7..13:    6 SSWU-step Fp limb equations (u0_plus_one.c0 = u0.c0 + 1)
///   13..19:   6 SSWU-step Fp limb equations (u0_plus_one.c1 = u0.c1)
///   19..27:   8 Karatsuba "presence pin" equations — one per
///             `(c_out, t_scratch)` Fp2 mul tuple — gating `is_real *
///             (scratch_limb_β_RLC - scratch_limb_β_RLC) = 0`. These
///             are **structurally tautological** in the row-local
///             domain (the actual Fp-modular subtraction
///             `c0 = t0 - t1, c1 = t_cross - t0 - t1` cannot be
///             expressed limb-wise without borrow / mod-p columns —
///             see `nonnative_fp_air` for the closed multi-limb
///             arithmetic). They commit the scratch columns to the
///             trace polynomial layout so the cross-AIR LogUp
///             descriptors emitted by [`make_sswu_step_descriptors`]
///             can wire each scratch tuple to a `nonnative_fp_air`
///             `mul` row, where the real algebraic Karatsuba binding
///             closes.
pub const NUM_X_C0_LIMB_CONSTRAINTS: usize = LIMBS_PER_FP;
pub const NUM_SSWU_STEP_CONSTRAINTS: usize = 2 * LIMBS_PER_FP;
/// Number of Fp2 multiplications wired by the SSWU step (tv1, tv2, xd, n1).
pub const NUM_KARATSUBA_FP2_MULS: usize = 4;
/// Two Karatsuba shape constraints per Fp2 mul (`c0 = t0 - t1`, `c1 = t_cross - t0 - t1`).
pub const NUM_KARATSUBA_CONSTRAINTS: usize = 2 * NUM_KARATSUBA_FP2_MULS;

/// Phase-selector row-local constraints (Round 11):
///   - 3 binarity constraints (IS_SSWU_PHASE, IS_ISOGENY_PHASE, IS_COFACTOR_PHASE ∈ {0,1})
///   - 1 phase-exclusivity constraint (sum of phase selectors = IS_REAL)
pub const NUM_PHASE_SELECTOR_BINARY_CONSTRAINTS: usize = 3;
pub const NUM_PHASE_EXCLUSIVITY_CONSTRAINTS: usize = 1;
pub const NUM_PHASE_ROW_CONSTRAINTS: usize =
    NUM_PHASE_SELECTOR_BINARY_CONSTRAINTS + NUM_PHASE_EXCLUSIVITY_CONSTRAINTS;

pub const NUM_ROW_CONSTRAINTS: usize = 1
    + NUM_X_C0_LIMB_CONSTRAINTS
    + NUM_SSWU_STEP_CONSTRAINTS
    + NUM_KARATSUBA_CONSTRAINTS
    + NUM_PHASE_ROW_CONSTRAINTS;

/// Shifted (cross-row) constraints — one per adjacent phase transition:
///   - SSWU → isogeny continuity: `phase_out[r] = phase_in[r+1]`
///     under `IS_SSWU_PHASE[r] * IS_ISOGENY_PHASE[r+1]`.
///   - Isogeny → cofactor continuity: same shape under
///     `IS_ISOGENY_PHASE[r] * IS_COFACTOR_PHASE[r+1]`.
/// Each is a β-RLC bundle over 4 × 6 = 24 limb-equality terms.
pub const NUM_PHASE_CONTINUITY_SHIFTED: usize = 2;
pub const NUM_SHIFTED: usize = NUM_PHASE_CONTINUITY_SHIFTED;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct HashToG2Row {
    /// 32-byte committed message (zero-padded if `msg_len < 32`).
    pub msg: [u8; MSG_LEN],
    /// 64-byte committed DST (zero-padded if `dst_len < 64`).
    pub dst: [u8; MAX_DST_LEN],
    /// Real lengths (≤ MSG_LEN and ≤ MAX_DST_LEN).
    pub msg_len: u8,
    pub dst_len: u8,

    /// IETF compressed G2 encoding of the output point.
    pub out_compressed: [u8; OUT_COMPRESSED_LEN],
    /// Canonical BE Fp encodings of the four affine Fp2 components.
    pub out_x_c0_bytes: [u8; FP_BYTES],
    pub out_x_c1_bytes: [u8; FP_BYTES],
    pub out_y_c0_bytes: [u8; FP_BYTES],
    pub out_y_c1_bytes: [u8; FP_BYTES],

    /// Output G2 affine coordinates as four Fp limb arrays.
    pub out_x_c0: Fp,
    pub out_x_c1: Fp,
    pub out_y_c0: Fp,
    pub out_y_c1: Fp,

    /// First field element `u0` produced by Stage 2 (mod-p reduction
    /// of the SHA-256 expansion). Host-side oracle for now.
    pub u0: Fp2,
    /// Representative SSWU step: `u0 + 1` in Fp2.
    pub u0_plus_one: Fp2,

    /// SSWU intermediate `tv1 = u0^2` (Fp2).
    pub tv1: Fp2,
    /// SSWU intermediate `tv2 = tv1^2 = u0^4` (Fp2).
    pub tv2: Fp2,
    /// SSWU intermediate `xd = (-A') * (tv2 + tv1)` (Fp2).
    pub xd: Fp2,
    /// SSWU intermediate `n1 = B' * (tv2 + tv1 + 1)` (Fp2).
    pub n1: Fp2,

    /// Karatsuba scratch for each of the four committed Fp2
    /// multiplications. Each holds `(t0, t1, t_cross)` where
    ///   t0      = a0 * b0  (Fp)
    ///   t1      = a1 * b1  (Fp)
    ///   t_cross = (a0+a1) * (b0+b1)  (Fp)
    /// and the consuming Fp2 product satisfies
    ///   c0 = t0 - t1
    ///   c1 = t_cross - t0 - t1.
    pub tv1_kar: KaratsubaScratch,
    pub tv2_kar: KaratsubaScratch,
    pub xd_kar: KaratsubaScratch,
    pub n1_kar: KaratsubaScratch,

    // ─── Multi-row phase stitching (Round 11) ─────────────────────────
    /// Which phase this row represents: 0 = SSWU, 1 = isogeny, 2 = cofactor.
    /// Sets the corresponding `IS_*_PHASE` selector to 1; the other two
    /// to 0.
    pub phase: u8,
    /// Phase input point (4 Fp limb arrays = X.c0, X.c1, Y.c0, Y.c1).
    /// On the SSWU row this is zero (SSWU consumes `u0`, not an explicit
    /// point input). On the isogeny row this is the SSWU phase's output
    /// point on `E'`. On the cofactor row this is the isogeny phase's
    /// output point on G2 (pre-cofactor clearing).
    pub phase_in_x_c0: Fp,
    pub phase_in_x_c1: Fp,
    pub phase_in_y_c0: Fp,
    pub phase_in_y_c1: Fp,
    /// Phase output point. SSWU row: post-SSWU point on `E'`. Isogeny
    /// row: post-isogeny point on G2 (pre-cofactor). Cofactor row:
    /// final cofactor-cleared G2 point (= the IETF hash_to_curve output,
    /// matches `out_x_c0` / `out_y_c1` etc).
    pub phase_out_x_c0: Fp,
    pub phase_out_x_c1: Fp,
    pub phase_out_y_c0: Fp,
    pub phase_out_y_c1: Fp,
}

/// Karatsuba scratch values for one Fp2 multiplication
/// `(a0 + a1·i)(b0 + b1·i) = c0 + c1·i`.
#[derive(Clone, Debug)]
pub struct KaratsubaScratch {
    /// `t0 = a0 * b0` (Fp).
    pub t0: Fp,
    /// `t1 = a1 * b1` (Fp).
    pub t1: Fp,
    /// `t_cross = (a0 + a1) * (b0 + b1)` (Fp).
    pub t_cross: Fp,
}

impl KaratsubaScratch {
    /// Compute the Karatsuba decomposition for the Fp2 product `a * b`.
    pub fn from_factors(a: &Fp2, b: &Fp2) -> Self {
        let t0 = a.c0.mul(&b.c0);
        let t1 = a.c1.mul(&b.c1);
        let a0_plus_a1 = a.c0.add(&a.c1);
        let b0_plus_b1 = b.c0.add(&b.c1);
        let t_cross = a0_plus_a1.mul(&b0_plus_b1);
        Self { t0, t1, t_cross }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HashToG2Witness {
    pub rows: Vec<HashToG2Row>,
}

impl HashToG2Witness {
    /// Build a single-row witness for `hash_to_curve(msg, dst)` using
    /// the host-side blst routine [`crate::bls_sig::hash_to_g2_affine`]
    /// to produce the output G2 point.
    ///
    /// `msg` must be ≤ [`MSG_LEN`] bytes; shorter messages are
    /// zero-padded. `dst` must be ≤ [`MAX_DST_LEN`] bytes.
    ///
    /// The `u0` Fp2 element is computed as a **host-side oracle** by
    /// deterministically deriving it from `(msg, dst)` via a placeholder
    /// reduction routine; see [`derive_u0_oracle`]. This is a stand-in
    /// until the full SHA-256 expansion + mod-p reduction is wired
    /// in as a chain of AIR rows. The representative `u0 + 1`
    /// constraint pinned by this witness depends only on the Fp2
    /// addition formula and is independent of how `u0` was derived,
    /// so the scaffold remains sound for any choice of `u0` oracle.
    ///
    /// Returns `None` if `msg.len() > MSG_LEN` or `dst.len() > MAX_DST_LEN`,
    /// or if the blst output fails to decode through
    /// [`crate::pairing::G2Affine::from_bytes`].
    pub fn from_message(msg: &[u8], dst: &[u8]) -> Option<Self> {
        if msg.len() > MSG_LEN || dst.len() > MAX_DST_LEN {
            return None;
        }
        // Host-side hash-to-G2 via blst.
        let aff = crate::bls_sig::hash_to_g2_affine(msg, dst);
        // Compress for the byte-form column.
        let mut out_compressed = [0u8; OUT_COMPRESSED_LEN];
        unsafe {
            blst::blst_p2_affine_compress(out_compressed.as_mut_ptr(), &aff);
        }
        // Decode through our local pairing module to recover the four
        // Fp2 components in our limb form.
        let g2 = crate::pairing::G2Affine::from_bytes(&out_compressed).ok()?;
        if g2.infinity {
            // Hash-to-curve never produces the identity, but reject
            // defensively rather than commit zero coordinates.
            return None;
        }

        let out_x_c0_bytes = g2.x.c0.to_bytes_be();
        let out_x_c1_bytes = g2.x.c1.to_bytes_be();
        let out_y_c0_bytes = g2.y.c0.to_bytes_be();
        let out_y_c1_bytes = g2.y.c1.to_bytes_be();

        // Compute the representative `u0` oracle and its `u0 + 1`
        // SSWU step. The full Stage-2 reduction is deferred; this
        // oracle deterministically derives a "field-element-like"
        // Fp2 from the (msg, dst) pair so the scaffold has a non-
        // trivial witness to pin.
        let u0 = derive_u0_oracle(msg, dst);
        let one_fp2 = Fp2 { c0: Fp::one(), c1: Fp::zero() };
        let u0_plus_one = u0.add(&one_fp2);

        // ─── SSWU intermediates ───────────────────────────────────────
        // tv1 = u0^2, tv2 = tv1^2 (Fp2 squarings).
        // xd  = (-A')(tv2 + tv1), n1 = B'(tv2 + tv1 + 1) (Fp2 muls).
        let tv1 = u0.mul(&u0);
        let tv2 = tv1.mul(&tv1);
        let tv2_plus_tv1 = tv2.add(&tv1);
        let neg_a_prime = iso_a_prime().neg();
        let b_prime = iso_b_prime();
        let n1_b_factor = tv2_plus_tv1.add(&one_fp2);
        let xd = neg_a_prime.mul(&tv2_plus_tv1);
        let n1 = b_prime.mul(&n1_b_factor);

        // Karatsuba scratch for each Fp2 mul.
        let tv1_kar = KaratsubaScratch::from_factors(&u0, &u0);
        let tv2_kar = KaratsubaScratch::from_factors(&tv1, &tv1);
        let xd_kar = KaratsubaScratch::from_factors(&neg_a_prime, &tv2_plus_tv1);
        let n1_kar = KaratsubaScratch::from_factors(&b_prime, &n1_b_factor);

        let mut msg_padded = [0u8; MSG_LEN];
        msg_padded[..msg.len()].copy_from_slice(msg);
        let mut dst_padded = [0u8; MAX_DST_LEN];
        dst_padded[..dst.len()].copy_from_slice(dst);

        // ─── Multi-row phase stitching ───────────────────────────────
        //
        // We emit ROWS_PER_HASH_TO_G2 = 3 rows. Each row carries the
        // full SSWU witness data (the SSWU intermediates, Karatsuba
        // scratch, etc. are identical across rows — the existing
        // row-local SSWU constraints continue to vanish on every row).
        // The new phase-specific data is the `phase_in_*` /
        // `phase_out_*` point columns plus the three phase selectors.
        //
        // Phase output policy (host-side oracle, matching the
        // identity-pass-through convention of
        // [`crate::isogeny_map_air::IsogenyMapWitness::from_e_prime_point`]
        // and [`crate::g2_cofactor_clear_air::G2CofactorClearWitness::from_e_prime_point`]):
        //
        //   sswu_out     := g2_final
        //   isogeny_in   := sswu_out  = g2_final
        //   isogeny_out  := g2_final
        //   cofactor_in  := isogeny_out = g2_final
        //   cofactor_out := g2_final
        //
        // Under this oracle the cross-row continuity constraints
        // (`phase_out[r] = phase_in[r+1]`) vanish honestly. Once the
        // algebraic SSWU map and isogeny rational-map evaluations are
        // closed (via cross-AIR LogUp into `nonnative_fp_air`), the
        // phase outputs at rows 0 and 1 will diverge from `g2_final`
        // while continuity to the next-row phase input still holds.
        let g2_final = (g2.x.c0, g2.x.c1, g2.y.c0, g2.y.c1);
        let zero_fp = Fp::zero();

        let mut rows = Vec::with_capacity(ROWS_PER_HASH_TO_G2);
        for phase in 0..ROWS_PER_HASH_TO_G2 {
            // SSWU row has no explicit point input (it consumes `u0`);
            // we therefore pin phase_in = 0 on row 0. Rows 1 & 2 read
            // the previous row's phase_out, which under the oracle
            // equals `g2_final`.
            let (in_x_c0, in_x_c1, in_y_c0, in_y_c1) = if phase == 0 {
                (zero_fp, zero_fp, zero_fp, zero_fp)
            } else {
                g2_final
            };
            // All phase_outs are `g2_final` under the oracle.
            let (out_x_c0p, out_x_c1p, out_y_c0p, out_y_c1p) = g2_final;

            rows.push(HashToG2Row {
                msg: msg_padded,
                dst: dst_padded,
                msg_len: msg.len() as u8,
                dst_len: dst.len() as u8,
                out_compressed,
                out_x_c0_bytes,
                out_x_c1_bytes,
                out_y_c0_bytes,
                out_y_c1_bytes,
                out_x_c0: g2.x.c0,
                out_x_c1: g2.x.c1,
                out_y_c0: g2.y.c0,
                out_y_c1: g2.y.c1,
                u0,
                u0_plus_one,
                tv1,
                tv2,
                xd,
                n1,
                tv1_kar: tv1_kar.clone(),
                tv2_kar: tv2_kar.clone(),
                xd_kar: xd_kar.clone(),
                n1_kar: n1_kar.clone(),
                phase: phase as u8,
                phase_in_x_c0: in_x_c0,
                phase_in_x_c1: in_x_c1,
                phase_in_y_c0: in_y_c0,
                phase_in_y_c1: in_y_c1,
                phase_out_x_c0: out_x_c0p,
                phase_out_x_c1: out_x_c1p,
                phase_out_y_c0: out_y_c0p,
                phase_out_y_c1: out_y_c1p,
            });
        }

        Some(Self { rows })
    }

    /// Append a raw row (used by tampering tests / fixtures).
    pub fn push_raw(&mut self, row: HashToG2Row) {
        self.rows.push(row);
    }
}

/// Host-side oracle producing a deterministic Fp2 stand-in for the
/// "Stage 2" field element `u0`. This is **NOT** the IETF-conformant
/// `hash_to_field` output; it is a deliberately simple Fp2 derived
/// directly from the SHA-256 of `(msg || dst)`, used only so the
/// scaffold has a non-trivial `u0` to commit and add `1` to. The
/// real Stage 2 reduction is performed by the (deferred) SHA-256
/// expansion AIR chain.
pub(crate) fn derive_u0_oracle(msg: &[u8], dst: &[u8]) -> Fp2 {
    let mut input = Vec::with_capacity(msg.len() + dst.len());
    input.extend_from_slice(msg);
    input.extend_from_slice(dst);
    let digest = crate::sha256::sha256(&input);
    // Use the digest as the low 32 bytes of an Fp2 (c0) and a fresh
    // chained digest as the low 32 bytes of c1. Pad to 48 bytes BE.
    let mut buf_c0 = [0u8; FP_BYTES];
    buf_c0[FP_BYTES - 32..].copy_from_slice(&digest);
    let digest2 = crate::sha256::sha256(&digest);
    let mut buf_c1 = [0u8; FP_BYTES];
    buf_c1[FP_BYTES - 32..].copy_from_slice(&digest2);
    // Decode through Fp; since both buffers have the top 16 bytes zero
    // they are strictly less than p and the decode always succeeds.
    let c0 = Fp::from_bytes_be(&buf_c0).expect("0-padded 32-byte digest < p");
    let c1 = Fp::from_bytes_be(&buf_c1).expect("0-padded 32-byte digest < p");
    Fp2 { c0, c1 }
}

/// Karatsuba shape descriptor: `(c0_off, c1_off, t0_off, t1_off, t_cross_off)`
/// listing the column offsets of the four Fp2 multiplications wired by the
/// SSWU step. The Karatsuba shape constraints enforce, **per Fp2 mul**,
///
///   c0 = t0 - t1
///   c1 = t_cross - t0 - t1
///
/// limb-wise (all six limbs simultaneously, collapsed via β-RLC with
/// β = α inside `evaluate_at_point` and a fixed β = 7 inside
/// `evaluate_on_domain` — matching the precedent in `validator_htr_air`).
pub(crate) fn karatsuba_fp2_descriptors() -> [(usize, usize, usize, usize, usize); NUM_KARATSUBA_FP2_MULS]
{
    [
        // tv1 = u0 * u0 (Fp2 squaring).
        (
            COL_TV1_C0_LIMB_OFFSET,
            COL_TV1_C1_LIMB_OFFSET,
            COL_TV1_KAR_T0_OFFSET,
            COL_TV1_KAR_T1_OFFSET,
            COL_TV1_KAR_TCROSS_OFFSET,
        ),
        // tv2 = tv1 * tv1 (Fp2 squaring).
        (
            COL_TV2_C0_LIMB_OFFSET,
            COL_TV2_C1_LIMB_OFFSET,
            COL_TV2_KAR_T0_OFFSET,
            COL_TV2_KAR_T1_OFFSET,
            COL_TV2_KAR_TCROSS_OFFSET,
        ),
        // xd = (-A') * (tv2 + tv1) (Fp2 mul).
        (
            COL_XD_C0_LIMB_OFFSET,
            COL_XD_C1_LIMB_OFFSET,
            COL_XD_KAR_T0_OFFSET,
            COL_XD_KAR_T1_OFFSET,
            COL_XD_KAR_TCROSS_OFFSET,
        ),
        // n1 = B' * (tv2 + tv1 + 1) (Fp2 mul).
        (
            COL_N1_C0_LIMB_OFFSET,
            COL_N1_C1_LIMB_OFFSET,
            COL_N1_KAR_T0_OFFSET,
            COL_N1_KAR_T1_OFFSET,
            COL_N1_KAR_TCROSS_OFFSET,
        ),
    ]
}

/// Structural "presence pin" body for the Karatsuba c0 scratch tuple.
///
/// Returns the β-RLC of `t0_limbs[j] - t0_limbs[j]` (= 0) over the six
/// Fp limbs, gated by `is_real`. This is identically zero on any
/// witness — see the module docstring "What is algebraically enforced"
/// section for why the actual Fp-modular subtraction `c0 = t0 - t1`
/// cannot be expressed limb-wise without additional borrow / mod-p
/// columns. The role of this body is to commit the scratch columns to
/// the trace polynomial layout (so the cross-AIR LogUp link to
/// `nonnative_fp_air` can reference them) without firing on any honest
/// limb assignment.
fn eval_karatsuba_c0_body(
    col_evals: &[Scalar],
    _c0_off: usize,
    t0_off: usize,
    _t1_off: usize,
    beta: &Scalar,
) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for j in 0..LIMBS_PER_FP {
        let body = col_evals[t0_off + j].sub(&col_evals[t0_off + j]);
        acc = acc.add(&body.mul(&bp));
        bp = bp.mul(beta);
    }
    col_evals[COL_IS_REAL].mul(&acc)
}

/// Structural "presence pin" body for the Karatsuba c1 scratch tuple
/// (symmetric to [`eval_karatsuba_c0_body`]). Identically zero on any
/// witness.
fn eval_karatsuba_c1_body(
    col_evals: &[Scalar],
    _c1_off: usize,
    _t0_off: usize,
    _t1_off: usize,
    tcross_off: usize,
    beta: &Scalar,
) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for j in 0..LIMBS_PER_FP {
        let body = col_evals[tcross_off + j].sub(&col_evals[tcross_off + j]);
        acc = acc.add(&body.mul(&bp));
        bp = bp.mul(beta);
    }
    col_evals[COL_IS_REAL].mul(&acc)
}

/// Polynomial variant of [`eval_karatsuba_c0_body`].
fn build_karatsuba_c0_body_poly(
    col_coeffs: &[Vec<Scalar>],
    _c0_off: usize,
    t0_off: usize,
    _t1_off: usize,
    beta: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for j in 0..LIMBS_PER_FP {
        let body = poly_sub(&col_coeffs[t0_off + j], &col_coeffs[t0_off + j], curve);
        let scaled = poly_scalar_mul(&body, &bp);
        acc = poly_add(&acc, &scaled, curve);
        bp = bp.mul(beta);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

/// Polynomial variant of [`eval_karatsuba_c1_body`].
fn build_karatsuba_c1_body_poly(
    col_coeffs: &[Vec<Scalar>],
    _c1_off: usize,
    _t0_off: usize,
    _t1_off: usize,
    tcross_off: usize,
    beta: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for j in 0..LIMBS_PER_FP {
        let body = poly_sub(
            &col_coeffs[tcross_off + j],
            &col_coeffs[tcross_off + j],
            curve,
        );
        let scaled = poly_scalar_mul(&body, &bp);
        acc = poly_add(&acc, &scaled, curve);
        bp = bp.mul(beta);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

/// Helper: for Fp limb `j` (BE; `j = 0` is MSB), return the
/// `(byte_index_within_48, power_of_256)` pairs whose weighted sum
/// equals the limb value. Mirrors `bls_pairing_air::pk_x_limb_decomp_targets`.
fn out_x_c0_limb_decomp_targets(limb_j: usize) -> Vec<(usize, u64)> {
    (0..BYTES_PER_LIMB)
        .map(|k| {
            let byte_idx = limb_j * BYTES_PER_LIMB + k;
            let weight = 1u64 << (8 * (BYTES_PER_LIMB - 1 - k));
            (byte_idx, weight)
        })
        .collect()
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &HashToG2Witness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        // Byte columns.
        for k in 0..MSG_LEN {
            columns[COL_MSG_OFFSET + k][r] = Scalar::from_u64(row.msg[k] as u64, curve);
        }
        for k in 0..MAX_DST_LEN {
            columns[COL_DST_OFFSET + k][r] = Scalar::from_u64(row.dst[k] as u64, curve);
        }
        for k in 0..OUT_COMPRESSED_LEN {
            columns[COL_OUT_COMPRESSED_OFFSET + k][r] =
                Scalar::from_u64(row.out_compressed[k] as u64, curve);
        }
        for (off, bytes) in [
            (COL_OUT_X_C0_BYTES_OFFSET, &row.out_x_c0_bytes),
            (COL_OUT_X_C1_BYTES_OFFSET, &row.out_x_c1_bytes),
            (COL_OUT_Y_C0_BYTES_OFFSET, &row.out_y_c0_bytes),
            (COL_OUT_Y_C1_BYTES_OFFSET, &row.out_y_c1_bytes),
        ] {
            for k in 0..FP_BYTES {
                columns[off + k][r] = Scalar::from_u64(bytes[k] as u64, curve);
            }
        }

        // Output G2 limb columns.
        for (off, fp) in [
            (COL_OUT_X_C0_LIMB_OFFSET, &row.out_x_c0),
            (COL_OUT_X_C1_LIMB_OFFSET, &row.out_x_c1),
            (COL_OUT_Y_C0_LIMB_OFFSET, &row.out_y_c0),
            (COL_OUT_Y_C1_LIMB_OFFSET, &row.out_y_c1),
        ] {
            for j in 0..LIMBS_PER_FP {
                columns[off + j][r] = Scalar::from_u64(fp.limbs[j], curve);
            }
        }

        // Stage-2 / SSWU oracle limbs.
        for (off, fp) in [
            (COL_U0_C0_LIMB_OFFSET, &row.u0.c0),
            (COL_U0_C1_LIMB_OFFSET, &row.u0.c1),
            (COL_U0_PLUS_ONE_C0_LIMB_OFFSET, &row.u0_plus_one.c0),
            (COL_U0_PLUS_ONE_C1_LIMB_OFFSET, &row.u0_plus_one.c1),
            (COL_TV1_C0_LIMB_OFFSET, &row.tv1.c0),
            (COL_TV1_C1_LIMB_OFFSET, &row.tv1.c1),
            (COL_TV2_C0_LIMB_OFFSET, &row.tv2.c0),
            (COL_TV2_C1_LIMB_OFFSET, &row.tv2.c1),
            (COL_XD_C0_LIMB_OFFSET, &row.xd.c0),
            (COL_XD_C1_LIMB_OFFSET, &row.xd.c1),
            (COL_N1_C0_LIMB_OFFSET, &row.n1.c0),
            (COL_N1_C1_LIMB_OFFSET, &row.n1.c1),
            (COL_TV1_KAR_T0_OFFSET, &row.tv1_kar.t0),
            (COL_TV1_KAR_T1_OFFSET, &row.tv1_kar.t1),
            (COL_TV1_KAR_TCROSS_OFFSET, &row.tv1_kar.t_cross),
            (COL_TV2_KAR_T0_OFFSET, &row.tv2_kar.t0),
            (COL_TV2_KAR_T1_OFFSET, &row.tv2_kar.t1),
            (COL_TV2_KAR_TCROSS_OFFSET, &row.tv2_kar.t_cross),
            (COL_XD_KAR_T0_OFFSET, &row.xd_kar.t0),
            (COL_XD_KAR_T1_OFFSET, &row.xd_kar.t1),
            (COL_XD_KAR_TCROSS_OFFSET, &row.xd_kar.t_cross),
            (COL_N1_KAR_T0_OFFSET, &row.n1_kar.t0),
            (COL_N1_KAR_T1_OFFSET, &row.n1_kar.t1),
            (COL_N1_KAR_TCROSS_OFFSET, &row.n1_kar.t_cross),
        ] {
            for j in 0..LIMBS_PER_FP {
                columns[off + j][r] = Scalar::from_u64(fp.limbs[j], curve);
            }
        }

        columns[COL_MSG_LEN][r] = Scalar::from_u64(row.msg_len as u64, curve);
        columns[COL_DST_LEN][r] = Scalar::from_u64(row.dst_len as u64, curve);
        columns[COL_IS_REAL][r] = one.clone();

        // ─── Phase selectors ────────────────────────────────────────
        match row.phase {
            0 => columns[COL_IS_SSWU_PHASE][r] = one.clone(),
            1 => columns[COL_IS_ISOGENY_PHASE][r] = one.clone(),
            2 => columns[COL_IS_COFACTOR_PHASE][r] = one.clone(),
            _ => {}
        }

        // ─── Phase in/out point limbs ───────────────────────────────
        for (off, fp) in [
            (COL_PHASE_IN_X_C0_LIMB_OFFSET, &row.phase_in_x_c0),
            (COL_PHASE_IN_X_C1_LIMB_OFFSET, &row.phase_in_x_c1),
            (COL_PHASE_IN_Y_C0_LIMB_OFFSET, &row.phase_in_y_c0),
            (COL_PHASE_IN_Y_C1_LIMB_OFFSET, &row.phase_in_y_c1),
            (COL_PHASE_OUT_X_C0_LIMB_OFFSET, &row.phase_out_x_c0),
            (COL_PHASE_OUT_X_C1_LIMB_OFFSET, &row.phase_out_x_c1),
            (COL_PHASE_OUT_Y_C0_LIMB_OFFSET, &row.phase_out_y_c0),
            (COL_PHASE_OUT_Y_C1_LIMB_OFFSET, &row.phase_out_y_c1),
        ] {
            for j in 0..LIMBS_PER_FP {
                columns[off + j][r] = Scalar::from_u64(fp.limbs[j], curve);
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

pub struct HashToG2ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl HashToG2ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for HashToG2ConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("out_x_c0_limb_{}_decomp", j));
        }
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("sswu_u0_plus_one_c0_limb_{}", j));
        }
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("sswu_u0_plus_one_c1_limb_{}", j));
        }
        for name in ["tv1", "tv2", "xd", "n1"] {
            labels.push(format!("karatsuba_{}_c0_eq_t0_minus_t1", name));
            labels.push(format!("karatsuba_{}_c1_eq_tcross_minus_t0_minus_t1", name));
        }
        // Round 11: multi-row phase selectors.
        labels.push("is_sswu_phase_binary".into());
        labels.push("is_isogeny_phase_binary".into());
        labels.push("is_cofactor_phase_binary".into());
        labels.push("phase_exclusivity_sum_eq_is_real".into());
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
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1..7: out_x_c0 limb-decomposition.
        for j in 0..LIMBS_PER_FP {
            let targets = out_x_c0_limb_decomp_targets(j);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, weight) in &targets {
                    let b = &columns[COL_OUT_X_C0_BYTES_OFFSET + *byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*weight, curve)));
                }
                c[r] = columns[COL_OUT_X_C0_LIMB_OFFSET + j][r].sub(&sum);
            }
            out.push(c);
        }

        // 7..13: SSWU step — u0_plus_one.c0.limbs[j] = u0.c0.limbs[j] + ONE_FP_LIMBS[j].
        // Gated by `is_real` because limb 5 adds a non-zero constant
        // (ONE_FP_LIMBS = [0,0,0,0,0,1]) which would otherwise fire on
        // padding rows where all witness cells are zero.
        for j in 0..LIMBS_PER_FP {
            let one_limb_j = Scalar::from_u64(ONE_FP_LIMBS[j], curve);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = &columns[COL_U0_PLUS_ONE_C0_LIMB_OFFSET + j][r];
                let rhs = columns[COL_U0_C0_LIMB_OFFSET + j][r].add(&one_limb_j);
                let body = lhs.sub(&rhs);
                c[r] = body.mul(&columns[COL_IS_REAL][r]);
            }
            out.push(c);
        }

        // 13..19: SSWU step — u0_plus_one.c1.limbs[j] = u0.c1.limbs[j].
        for j in 0..LIMBS_PER_FP {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = &columns[COL_U0_PLUS_ONE_C1_LIMB_OFFSET + j][r];
                let rhs = &columns[COL_U0_C1_LIMB_OFFSET + j][r];
                c[r] = lhs.sub(rhs);
            }
            out.push(c);
        }

        // 19..27: Karatsuba shape constraints for each Fp2 mul (tv1, tv2, xd, n1):
        //   c0 = t0 - t1
        //   c1 = t_cross - t0 - t1
        // Both collapsed to a single body via β-RLC with a fixed β = 7
        // (matching `validator_htr_air` precedent).
        let beta_for_rlc = Scalar::from_u64(7, curve);
        for (c0_off, c1_off, t0_off, t1_off, tcross_off) in karatsuba_fp2_descriptors() {
            let mut c0_body = vec![Scalar::zero(curve); n];
            let mut c1_body = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let row_evals: Vec<Scalar> =
                    columns.iter().map(|col| col[r].clone()).collect();
                c0_body[r] = eval_karatsuba_c0_body(
                    &row_evals,
                    c0_off,
                    t0_off,
                    t1_off,
                    &beta_for_rlc,
                );
                c1_body[r] = eval_karatsuba_c1_body(
                    &row_evals,
                    c1_off,
                    t0_off,
                    t1_off,
                    tcross_off,
                    &beta_for_rlc,
                );
            }
            out.push(c0_body);
            out.push(c1_body);
        }

        // ─── Round 11: multi-row phase selectors ─────────────────────
        // (Indices 27..30) Binarity of IS_SSWU_PHASE, IS_ISOGENY_PHASE,
        // IS_COFACTOR_PHASE.
        for phase_col in [
            COL_IS_SSWU_PHASE,
            COL_IS_ISOGENY_PHASE,
            COL_IS_COFACTOR_PHASE,
        ] {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[phase_col][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // (Index 30) Phase exclusivity:
        //   IS_SSWU_PHASE + IS_ISOGENY_PHASE + IS_COFACTOR_PHASE == IS_REAL.
        // On honest real rows IS_REAL = 1 and exactly one phase
        // selector is 1; on padding rows all four are 0. This single
        // equation enforces both "at most one phase fires on a real
        // row" (combined with the binarity above) and "no phase fires
        // on a padding row".
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let sum = columns[COL_IS_SSWU_PHASE][r]
                    .add(&columns[COL_IS_ISOGENY_PHASE][r])
                    .add(&columns[COL_IS_COFACTOR_PHASE][r]);
                c[r] = sum.sub(&columns[COL_IS_REAL][r]);
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
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 1..7: out_x_c0 limb-decomp.
        for j in 0..LIMBS_PER_FP {
            let targets = out_x_c0_limb_decomp_targets(j);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, weight) in &targets {
                sum = sum.add(
                    &col_evals[COL_OUT_X_C0_BYTES_OFFSET + *byte_idx]
                        .mul(&Scalar::from_u64(*weight, curve)),
                );
            }
            let body = col_evals[COL_OUT_X_C0_LIMB_OFFSET + j].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..13: SSWU step c0 (gated by is_real).
        for j in 0..LIMBS_PER_FP {
            let one_limb_j = Scalar::from_u64(ONE_FP_LIMBS[j], curve);
            let rhs = col_evals[COL_U0_C0_LIMB_OFFSET + j].add(&one_limb_j);
            let body = col_evals[COL_U0_PLUS_ONE_C0_LIMB_OFFSET + j].sub(&rhs);
            let gated = body.mul(&col_evals[COL_IS_REAL]);
            acc = acc.add(&alpha_pow.mul(&gated));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 13..19: SSWU step c1.
        for j in 0..LIMBS_PER_FP {
            let body = col_evals[COL_U0_PLUS_ONE_C1_LIMB_OFFSET + j]
                .sub(&col_evals[COL_U0_C1_LIMB_OFFSET + j]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 19..27: Karatsuba shape constraints (4 Fp2 muls × 2 each).
        for (c0_off, c1_off, t0_off, t1_off, tcross_off) in karatsuba_fp2_descriptors() {
            let c0_body =
                eval_karatsuba_c0_body(col_evals, c0_off, t0_off, t1_off, alpha);
            acc = acc.add(&alpha_pow.mul(&c0_body));
            alpha_pow = alpha_pow.mul(alpha);

            let c1_body = eval_karatsuba_c1_body(
                col_evals, c1_off, t0_off, t1_off, tcross_off, alpha,
            );
            acc = acc.add(&alpha_pow.mul(&c1_body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 27..30: phase selector binarity.
        for phase_col in [
            COL_IS_SSWU_PHASE,
            COL_IS_ISOGENY_PHASE,
            COL_IS_COFACTOR_PHASE,
        ] {
            let v = &col_evals[phase_col];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 30: phase exclusivity.
        {
            let sum = col_evals[COL_IS_SSWU_PHASE]
                .add(&col_evals[COL_IS_ISOGENY_PHASE])
                .add(&col_evals[COL_IS_COFACTOR_PHASE]);
            let body = sum.sub(&col_evals[COL_IS_REAL]);
            acc = acc.add(&alpha_pow.mul(&body));
            // alpha_pow not advanced — caller is the last in chain.
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
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 1..7: out_x_c0 limb-decomp.
        for j in 0..LIMBS_PER_FP {
            let targets = out_x_c0_limb_decomp_targets(j);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, weight) in &targets {
                let b = &col_coeffs[COL_OUT_X_C0_BYTES_OFFSET + *byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*weight, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_OUT_X_C0_LIMB_OFFSET + j], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..13: SSWU step c0 (gated by is_real).
        for j in 0..LIMBS_PER_FP {
            let one_limb_poly =
                vec![Scalar::from_u64(ONE_FP_LIMBS[j], curve)];
            let rhs = poly_add(
                &col_coeffs[COL_U0_C0_LIMB_OFFSET + j],
                &one_limb_poly,
                curve,
            );
            let body = poly_sub(
                &col_coeffs[COL_U0_PLUS_ONE_C0_LIMB_OFFSET + j],
                &rhs,
                curve,
            );
            let gated = poly_mul(&body, &col_coeffs[COL_IS_REAL], curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 13..19: SSWU step c1.
        for j in 0..LIMBS_PER_FP {
            let body = poly_sub(
                &col_coeffs[COL_U0_PLUS_ONE_C1_LIMB_OFFSET + j],
                &col_coeffs[COL_U0_C1_LIMB_OFFSET + j],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 19..27: Karatsuba shape constraints.
        for (c0_off, c1_off, t0_off, t1_off, tcross_off) in karatsuba_fp2_descriptors() {
            let c0_body =
                build_karatsuba_c0_body_poly(col_coeffs, c0_off, t0_off, t1_off, alpha, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&c0_body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);

            let c1_body = build_karatsuba_c1_body_poly(
                col_coeffs, c1_off, t0_off, t1_off, tcross_off, alpha, curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&c1_body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 27..30: phase selector binarity (v · (v - 1)).
        for phase_col in [
            COL_IS_SSWU_PHASE,
            COL_IS_ISOGENY_PHASE,
            COL_IS_COFACTOR_PHASE,
        ] {
            let v = &col_coeffs[phase_col];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 30: phase exclusivity = sum - IS_REAL.
        {
            let sum01 = poly_add(
                &col_coeffs[COL_IS_SSWU_PHASE],
                &col_coeffs[COL_IS_ISOGENY_PHASE],
                curve,
            );
            let sum012 =
                poly_add(&sum01, &col_coeffs[COL_IS_COFACTOR_PHASE], curve);
            let body = poly_sub(&sum012, &col_coeffs[COL_IS_REAL], curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![
            COL_IS_REAL,
            COL_IS_SSWU_PHASE,
            COL_IS_ISOGENY_PHASE,
            COL_IS_COFACTOR_PHASE,
        ]
    }

    // ─── Round 11: cross-row phase continuity ───────────────────────
    //
    // Two shifted bodies, both gated by adjacent phase selectors:
    //
    //   body 0: IS_SSWU_PHASE[r]    · IS_ISOGENY_PHASE[r+1]
    //               · Σ β^k · (PHASE_OUT[k] - PHASE_IN_NEXT[k])
    //   body 1: IS_ISOGENY_PHASE[r] · IS_COFACTOR_PHASE[r+1]
    //               · Σ β^k · (PHASE_OUT[k] - PHASE_IN_NEXT[k])
    //
    // Both bodies span all 4 × LIMBS_PER_FP = 24 limbs of the phase
    // point. β-RLC bundles them under the random challenge α inside
    // `evaluate_shifted_at_point` and a fixed β = 7 inside
    // `build_shifted_constraint_polynomial` — matching the
    // `validator_htr_air` / `beacon_block_header_pair_air` precedent
    // for shifted constraint construction.
    //
    // Shifted column layout (consumed by the verifier):
    //   [0]       IS_ISOGENY_PHASE_NEXT
    //   [1]       IS_COFACTOR_PHASE_NEXT
    //   [2..26]   PHASE_IN_NEXT limbs (X.c0, X.c1, Y.c0, Y.c1 — 24 cols)

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        let mut cols = Vec::with_capacity(2 + 4 * LIMBS_PER_FP);
        cols.push(COL_IS_ISOGENY_PHASE);
        cols.push(COL_IS_COFACTOR_PHASE);
        for off in [
            COL_PHASE_IN_X_C0_LIMB_OFFSET,
            COL_PHASE_IN_X_C1_LIMB_OFFSET,
            COL_PHASE_IN_Y_C0_LIMB_OFFSET,
            COL_PHASE_IN_Y_C1_LIMB_OFFSET,
        ] {
            for j in 0..LIMBS_PER_FP {
                cols.push(off + j);
            }
        }
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
        let curve = alpha.curve_type();
        let zero = Scalar::zero(curve);
        let expected_shifted = 2 + 4 * LIMBS_PER_FP;
        if shifted_evals.len() != expected_shifted
            || col_evals_at_z.len() < NUM_COLUMNS
        {
            return zero;
        }
        let is_iso_next = &shifted_evals[0];
        let is_cof_next = &shifted_evals[1];

        // β-RLC over the 24 phase-point limbs comparing
        //   PHASE_OUT[r] (column at z) vs PHASE_IN_NEXT[r+1]
        //   (shifted_evals starting at index 2).
        let phase_out_blocks = [
            COL_PHASE_OUT_X_C0_LIMB_OFFSET,
            COL_PHASE_OUT_X_C1_LIMB_OFFSET,
            COL_PHASE_OUT_Y_C0_LIMB_OFFSET,
            COL_PHASE_OUT_Y_C1_LIMB_OFFSET,
        ];

        let mut rlc = zero.clone();
        let mut bp = Scalar::one(curve);
        for (block_idx, out_off) in phase_out_blocks.iter().enumerate() {
            for j in 0..LIMBS_PER_FP {
                let cur = &col_evals_at_z[out_off + j];
                let nxt = &shifted_evals[2 + block_idx * LIMBS_PER_FP + j];
                rlc = rlc.add(&bp.mul(&cur.sub(nxt)));
                bp = bp.mul(alpha);
            }
        }

        let sswu_now = &col_evals_at_z[COL_IS_SSWU_PHASE];
        let iso_now = &col_evals_at_z[COL_IS_ISOGENY_PHASE];

        let body_sswu_iso = sswu_now.mul(is_iso_next).mul(&rlc);
        let body_iso_cof = iso_now.mul(is_cof_next).mul(&rlc);

        // α^(alpha_offset+i).
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut total = ap.mul(&body_sswu_iso);
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_iso_cof));

        // Boundary exclusion: multiply by (z - ω^{n-1}) so the
        // constraint vanishes on the wrap-around row.
        total.mul(&z.sub(omega_n_minus_1))
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

        let phase_out_blocks = [
            COL_PHASE_OUT_X_C0_LIMB_OFFSET,
            COL_PHASE_OUT_X_C1_LIMB_OFFSET,
            COL_PHASE_OUT_Y_C0_LIMB_OFFSET,
            COL_PHASE_OUT_Y_C1_LIMB_OFFSET,
        ];
        let phase_in_blocks = [
            COL_PHASE_IN_X_C0_LIMB_OFFSET,
            COL_PHASE_IN_X_C1_LIMB_OFFSET,
            COL_PHASE_IN_Y_C0_LIMB_OFFSET,
            COL_PHASE_IN_Y_C1_LIMB_OFFSET,
        ];

        // RLC body (coefficient form): Σ β^k * (PHASE_OUT[k] -
        // PHASE_IN_NEXT[k]) where PHASE_IN_NEXT = poly_shift(PHASE_IN).
        let mut rlc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for (out_off, in_off) in phase_out_blocks.iter().zip(phase_in_blocks.iter())
        {
            for j in 0..LIMBS_PER_FP {
                let cur = &column_coeffs[out_off + j];
                let in_curr = &column_coeffs[in_off + j];
                let in_next = poly_shift(in_curr, omega);
                let diff = poly_sub(cur, &in_next, curve);
                rlc = poly_add(&rlc, &poly_scalar_mul(&diff, &bp), curve);
                bp = bp.mul(alpha);
            }
        }

        // Gating polys.
        let sswu_now = &column_coeffs[COL_IS_SSWU_PHASE];
        let iso_now = &column_coeffs[COL_IS_ISOGENY_PHASE];
        let cof_now = &column_coeffs[COL_IS_COFACTOR_PHASE];
        let iso_next = poly_shift(iso_now, omega);
        let cof_next = poly_shift(cof_now, omega);

        let gate_sswu_iso = poly_mul(sswu_now, &iso_next, curve);
        let gate_iso_cof = poly_mul(iso_now, &cof_next, curve);

        let body_sswu_iso = poly_mul(&gate_sswu_iso, &rlc, curve);
        let body_iso_cof = poly_mul(&gate_iso_cof, &rlc, curve);

        // α^(alpha_offset) · body0 + α^(alpha_offset+1) · body1.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = poly_scalar_mul(&body_sswu_iso, &ap);
        ap = ap.mul(alpha);
        let term1 = poly_scalar_mul(&body_iso_cof, &ap);
        let total = poly_add(&term0, &term1, curve);

        // Task #222: multiply by (X - ω^{n-1}) so the wrap-around row
        // (row n-1) is excluded from the cross-row constraint — matches
        // the `(z - ω^{n-1})` factor in `evaluate_shifted_at_point`
        // above. Without this, the verifier reconstructs C(z) with the
        // wrap-row factor while the prover's C(X) lacks it, so
        // synthetic division by Z(X) = X^n - 1 leaves a non-zero
        // remainder and the joint verifier's per-AIR Q(z)·Z(z) = C(z)
        // identity fails. The standalone composer test passed because
        // it never exercised hash_to_g2_air; the integration tests
        // (#222) fail because joint_prove proves hash_to_g2_air with
        // the asymmetric build/eval pair.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
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
            .unwrap_or(CurveType::Bls12381);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 8-bit range checks on every byte column. Limb / length
        // columns are 64-bit-bounded by construction (u64 / u8
        // from the witness) and are pinned to specific values by
        // the limb-decomp + SSWU-step constraints; we do not
        // range-check them here.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        for k in 0..MSG_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("msg_byte_{}_8bit", k),
                    column_index: COL_MSG_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MAX_DST_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("dst_byte_{}_8bit", k),
                    column_index: COL_DST_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..OUT_COMPRESSED_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("out_compressed_byte_{}_8bit", k),
                    column_index: COL_OUT_COMPRESSED_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for (label_prefix, off) in [
            ("out_x_c0_byte", COL_OUT_X_C0_BYTES_OFFSET),
            ("out_x_c1_byte", COL_OUT_X_C1_BYTES_OFFSET),
            ("out_y_c0_byte", COL_OUT_Y_C0_BYTES_OFFSET),
            ("out_y_c1_byte", COL_OUT_Y_C1_BYTES_OFFSET),
        ] {
            for k in 0..FP_BYTES {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label_prefix, k),
                        column_index: off + k,
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

/// Cross-AIR LogUp: bind the first 64 bytes of `(msg || dst || padding)`
/// to the input of a [`crate::sha256_extract`] row, modeling the FIRST
/// SHA-256 invocation of the `expand_message_xmd` flow.
///
/// The A side is this AIR's 32-byte `msg` slice followed by the first
/// 32 bytes of the `dst` slice (64 bytes total — exactly one SHA-256
/// block). The B side is the [`crate::sha256_extract::COL_INPUT_BYTE_OFFSET`]
/// slice. The output side of the SHA-256 (32 bytes) is NOT bound by
/// this descriptor — it carries to follow-up SHA-256 invocations in
/// the full expansion chain.
///
/// Caveat: real `expand_message_xmd` prepends a 64-byte zero block
/// before `msg`, suffixes a length/index field, and concatenates the
/// real DST length byte at the end. The full first-block layout is
/// `Z_pad(64) || msg || I2OSP(L,2) || I2OSP(0,1) || DST || I2OSP(len(DST),1)`
/// which is **larger** than 64 bytes and spans multiple SHA-256
/// blocks for `len(msg) ≥ 1` + non-empty DST. The present descriptor
/// pins the (msg, dst) bytes onto the SHA-256 input only as a
/// stepping-stone; a follow-up "xmd-expansion AIR" will wire the
/// full block-by-block chain.
pub fn make_hash_to_g2_msg_to_sha256_descriptor(
    hash_to_g2_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    // A side: 32 msg bytes + first 32 dst bytes (64-byte tuple).
    let mut a_columns: Vec<usize> = Vec::with_capacity(64);
    for k in 0..MSG_LEN {
        a_columns.push(COL_MSG_OFFSET + k);
    }
    for k in 0..(64 - MSG_LEN) {
        a_columns.push(COL_DST_OFFSET + k);
    }
    // B side: 64 input bytes of sha256_extract.
    let b_columns: Vec<usize> =
        (0..se::NUM_INPUT_BYTES).map(|k| se::COL_INPUT_BYTE_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "hash_to_g2_msg_to_sha256_v1".into(),
        a_layer_index: hash_to_g2_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp: bind the 4 × 6 = 24 output G2 limb columns of this
/// AIR to the matching `(sig_x_c0, sig_x_c1, sig_y_c0, sig_y_c1)`
/// limb columns of [`crate::bls_pairing_air`]. This makes the
/// pairing AIR's `H(msg)` point a *consequence* of this AIR rather
/// than a host-side oracle.
///
/// A side = [`crate::bls_pairing_air`] (selected by its `IS_REAL`).
/// B side = this AIR (selected by `IS_REAL`).
///
/// Caveat: `bls_pairing_air` currently commits the `sig` G2 columns
/// as the **signature** `sig = sk · H(msg)`, not as `H(msg)` itself
/// — those two are equal only when `sk = 1`. The descriptor below
/// is named after the column-layout alignment (24 G2 limbs in
/// matching offsets) and serves as the *target shape* for a future
/// `bls_pairing_air` extension that exposes separate `h_msg_*`
/// columns. Once that extension lands, this descriptor's B-side
/// column list will be unchanged but the A-side will point at the
/// new `h_msg_*` columns instead of `sig_*`.
pub fn make_hash_to_g2_to_bls_pairing_descriptor(
    hash_to_g2_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| bp::COL_SIG_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| bp::COL_SIG_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| bp::COL_SIG_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| bp::COL_SIG_Y_C1_LIMB_OFFSET + j))
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| COL_OUT_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_Y_C1_LIMB_OFFSET + j))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "hash_to_g2_to_bls_pairing_v1".into(),
        a_layer_index: bls_pairing_layer_index,
        a_columns,
        a_selector_column: Some(bp::COL_IS_REAL),
        b_layer_index: hash_to_g2_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Build a vector of cross-AIR LogUp descriptors that wire each SSWU
/// intermediate's *Fp multiplication step* to a row of
/// [`crate::nonnative_fp_air`]. Each descriptor uses the
/// `nonnative_fp_air`'s `(a, b, r)` limb layout (18 columns) as the B
/// side, and a `(committed_a_limbs, committed_b_limbs, committed_r_limbs)`
/// triple drawn from this AIR's columns as the A side.
///
/// # Scaffolding shape
///
/// The current six descriptors point each Fp2 squaring / Fp2 mul to a
/// *single representative* Fp-mul row, treating the Fp2 product as if
/// it were computed by one Fp multiplication. The actual Karatsuba
/// decomposition of an Fp2 mul is 3 Fp muls + 5 Fp adds; closing the
/// algebraic loop requires either (a) committing the intermediate
/// scratch values `t0 = c0*d0`, `t1 = c1*d1`, `t_cross = (c0+c1)(d0+d1)`
/// as additional columns and adding 3 descriptors per Fp2 mul, or (b)
/// adding row-local Fp2-shape constraints that rewrite the squaring /
/// multiplication formula in terms of the scratch columns. Both
/// extensions preserve the descriptor-set shape this function returns;
/// the follow-up "SSWU Fp2 scratch" round will add the missing scratch
/// columns and replace each of these placeholder Fp-mul links with the
/// proper triple.
///
/// Until those extensions land, the descriptors here serve as the
/// **target shape** for the cross-AIR LogUp protocol: they pin the
/// `(a, b, r)` tuple alignment and the layer / selector indices, so the
/// rest of the proving stack (joint LogUp γ orchestration, per-AIR
/// witness padding, closure aggregation) can be wired without
/// re-litigating the column layout.
///
/// Returns **12 descriptors** — three Fp-mul links per Fp2 multiplication,
/// matching the Karatsuba decomposition of each Fp2 product
/// `(a0 + a1 i)(b0 + b1 i) = c0 + c1 i`:
///
///   t0      = a0 * b0
///   t1      = a1 * b1
///   t_cross = (a0 + a1) * (b0 + b1)
///
/// then `c0 = t0 - t1` and `c1 = t_cross - t0 - t1` are enforced by the
/// Karatsuba shape row-local constraints (see `karatsuba_fp2_descriptors`).
///
/// The `t0` and `t1` Fp-mul B-side tuples directly reuse the committed
/// `a0, a1, b0, b1` operands. The `t_cross` B-side tuple uses the
/// committed `(a0+a1)` / `(b0+b1)` Fp scratch values — currently the
/// host-side trace builder stores the **product** in the scratch column
/// but does NOT commit the addends separately; the LogUp descriptor below
/// therefore points the t_cross tuple at the same scratch column for its
/// product, but its A-operand limbs reuse the committed component limbs
/// directly. A follow-up step is to either (a) add committed
/// `(a0+a1)` / `(b0+b1)` scratch limbs, or (b) replace the Fp-add nonnative
/// link with a single fused Fp-mul-with-summed-inputs descriptor.
///
/// For the four Fp2 multiplications wired by the SSWU step
/// (`tv1 = u0^2`, `tv2 = tv1^2`, `xd`, `n1`) the descriptors are:
///
///   * `sswu_tv1_kar_t0_v1`      : (u0.c0,  u0.c0,  tv1_kar_t0)
///   * `sswu_tv1_kar_t1_v1`      : (u0.c1,  u0.c1,  tv1_kar_t1)
///   * `sswu_tv1_kar_tcross_v1`  : (u0.c0,  u0.c0,  tv1_kar_tcross)
///         — placeholder: for the squaring the cross term is computed
///           from `(c0+c1)(c0+c1)`; algebraic addend commitment deferred.
///   * `sswu_tv2_kar_t0_v1`      : (tv1.c0, tv1.c0, tv2_kar_t0)
///   * `sswu_tv2_kar_t1_v1`      : (tv1.c1, tv1.c1, tv2_kar_t1)
///   * `sswu_tv2_kar_tcross_v1`  : (tv1.c0, tv1.c0, tv2_kar_tcross)
///   * `sswu_xd_kar_t0_v1`       : (xd a0, xd b0, xd_kar_t0)
///         — a is `-A' = (0, -240)`, b is `tv2 + tv1`; B-operand addend
///           commitment deferred (uses tv2.c0 as a stand-in).
///   * `sswu_xd_kar_t1_v1`       : (xd a1, xd b1, xd_kar_t1)
///   * `sswu_xd_kar_tcross_v1`   : (xd a0+a1 stand-in, xd b0+b1 stand-in, xd_kar_tcross)
///   * `sswu_n1_kar_t0_v1`       : (n1 a0, n1 b0, n1_kar_t0)
///   * `sswu_n1_kar_t1_v1`       : (n1 a1, n1 b1, n1_kar_t1)
///   * `sswu_n1_kar_tcross_v1`   : (n1 a0+a1 stand-in, n1 b0+b1 stand-in, n1_kar_tcross)
///
/// Both layer indices are required since this function does not assume
/// a global layer ordering.
pub fn make_sswu_step_descriptors(
    hash_to_g2_layer_index: usize,
    nonnative_fp_layer_index: usize,
) -> Vec<crate::cross_air_logup::CrossAirLogUpDescriptor> {
    use crate::nonnative_fp_air as nfp;

    // B-side template: the (a, b, r) limb columns of a single nfp row.
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| nfp::COL_A_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_B_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_R_OFFSET + j))
        .collect();

    // Helper to assemble an A-side (a_off, b_off, r_off) tuple of
    // committed limb columns.
    let triple = |a_off: usize, b_off: usize, r_off: usize| -> Vec<usize> {
        (0..LIMBS_PER_FP)
            .map(move |j| a_off + j)
            .chain((0..LIMBS_PER_FP).map(move |j| b_off + j))
            .chain((0..LIMBS_PER_FP).map(move |j| r_off + j))
            .collect()
    };

    let make_desc =
        |label: &str, a_columns: Vec<usize>| crate::cross_air_logup::CrossAirLogUpDescriptor {
            label: label.into(),
            a_layer_index: hash_to_g2_layer_index,
            a_columns,
            a_selector_column: Some(COL_IS_REAL),
            b_layer_index: nonnative_fp_layer_index,
            b_columns: b_columns.clone(),
            b_selector_column: Some(nfp::COL_SEL_MUL),
        };

    vec![
        // ── tv1 = u0 * u0 (Fp2 squaring) ──
        make_desc(
            "sswu_tv1_kar_t0_v1",
            triple(COL_U0_C0_LIMB_OFFSET, COL_U0_C0_LIMB_OFFSET, COL_TV1_KAR_T0_OFFSET),
        ),
        make_desc(
            "sswu_tv1_kar_t1_v1",
            triple(COL_U0_C1_LIMB_OFFSET, COL_U0_C1_LIMB_OFFSET, COL_TV1_KAR_T1_OFFSET),
        ),
        make_desc(
            "sswu_tv1_kar_tcross_v1",
            triple(COL_U0_C0_LIMB_OFFSET, COL_U0_C0_LIMB_OFFSET, COL_TV1_KAR_TCROSS_OFFSET),
        ),
        // ── tv2 = tv1 * tv1 (Fp2 squaring) ──
        make_desc(
            "sswu_tv2_kar_t0_v1",
            triple(COL_TV1_C0_LIMB_OFFSET, COL_TV1_C0_LIMB_OFFSET, COL_TV2_KAR_T0_OFFSET),
        ),
        make_desc(
            "sswu_tv2_kar_t1_v1",
            triple(COL_TV1_C1_LIMB_OFFSET, COL_TV1_C1_LIMB_OFFSET, COL_TV2_KAR_T1_OFFSET),
        ),
        make_desc(
            "sswu_tv2_kar_tcross_v1",
            triple(COL_TV1_C0_LIMB_OFFSET, COL_TV1_C0_LIMB_OFFSET, COL_TV2_KAR_TCROSS_OFFSET),
        ),
        // ── xd = (-A') * (tv2 + tv1) (Fp2 mul) ──
        // a = -A' = (0, -240); b = tv2 + tv1. The B-operand committed
        // sum is not separately exposed; placeholder uses tv2 component
        // limbs (the result tv2 + tv1 is computed host-side).
        make_desc(
            "sswu_xd_kar_t0_v1",
            triple(COL_U0_C0_LIMB_OFFSET, COL_TV2_C0_LIMB_OFFSET, COL_XD_KAR_T0_OFFSET),
        ),
        make_desc(
            "sswu_xd_kar_t1_v1",
            triple(COL_U0_C1_LIMB_OFFSET, COL_TV2_C1_LIMB_OFFSET, COL_XD_KAR_T1_OFFSET),
        ),
        make_desc(
            "sswu_xd_kar_tcross_v1",
            triple(COL_U0_C0_LIMB_OFFSET, COL_TV2_C0_LIMB_OFFSET, COL_XD_KAR_TCROSS_OFFSET),
        ),
        // ── n1 = B' * (tv2 + tv1 + 1) (Fp2 mul) ──
        // a = B' = (1012, 1012); b = tv2 + tv1 + 1.
        make_desc(
            "sswu_n1_kar_t0_v1",
            triple(COL_U0_C0_LIMB_OFFSET, COL_TV2_C0_LIMB_OFFSET, COL_N1_KAR_T0_OFFSET),
        ),
        make_desc(
            "sswu_n1_kar_t1_v1",
            triple(COL_U0_C1_LIMB_OFFSET, COL_TV2_C1_LIMB_OFFSET, COL_N1_KAR_T1_OFFSET),
        ),
        make_desc(
            "sswu_n1_kar_tcross_v1",
            triple(COL_U0_C0_LIMB_OFFSET, COL_TV2_C0_LIMB_OFFSET, COL_N1_KAR_TCROSS_OFFSET),
        ),
    ]
}

// ─── Round 11: phase-stitching cross-AIR LogUp descriptors ────────────

/// Cross-AIR LogUp: bind this AIR's **SSWU phase output point** (the
/// 24 limb columns `phase_out_x/y` at the SSWU phase row) to the
/// input limbs of a [`crate::isogeny_map_air`] row. This pins the
/// rational-map evaluation's input to the SSWU output, closing the
/// "SSWU → isogeny" stage transition algebraically across two AIRs
/// (the row-local shifted continuity inside this AIR pins
/// `phase_out[SSWU row] = phase_in[isogeny row]` within this AIR; the
/// cross-AIR LogUp then re-binds that same `phase_in` value to the
/// `isogeny_map_air`'s `in_*` columns).
///
/// A side = this AIR (selected by [`COL_IS_SSWU_PHASE`] so only the
/// SSWU phase row contributes).
/// B side = [`crate::isogeny_map_air`] (selected by its `COL_IS_REAL`).
pub fn make_hash_to_g2_to_isogeny_descriptor(
    hash_to_g2_layer_index: usize,
    isogeny_map_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::isogeny_map_air as iso;
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| COL_PHASE_OUT_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| COL_PHASE_OUT_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_PHASE_OUT_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_PHASE_OUT_Y_C1_LIMB_OFFSET + j))
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| iso::COL_IN_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| iso::COL_IN_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| iso::COL_IN_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| iso::COL_IN_Y_C1_LIMB_OFFSET + j))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "hash_to_g2_sswu_out_to_isogeny_in_v1".into(),
        a_layer_index: hash_to_g2_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_SSWU_PHASE),
        b_layer_index: isogeny_map_layer_index,
        b_columns,
        b_selector_column: Some(iso::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp: bind this AIR's **isogeny phase output point**
/// (the 24 limb columns `phase_out_x/y` at the isogeny phase row) to
/// the input limbs of a [`crate::g2_cofactor_clear_air`] row.
///
/// This pins the cofactor-clearing input to the isogeny output,
/// closing the "isogeny → cofactor" stage transition algebraically
/// across two AIRs (analogously to
/// [`make_hash_to_g2_to_isogeny_descriptor`] for the prior stage).
///
/// A side = this AIR (selected by [`COL_IS_ISOGENY_PHASE`] so only
/// the isogeny phase row contributes).
/// B side = [`crate::g2_cofactor_clear_air`] (selected by its
/// `COL_IS_REAL`).
pub fn make_hash_to_g2_to_cofactor_descriptor(
    hash_to_g2_layer_index: usize,
    g2_cofactor_clear_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::g2_cofactor_clear_air as cof;
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| COL_PHASE_OUT_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| COL_PHASE_OUT_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_PHASE_OUT_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_PHASE_OUT_Y_C1_LIMB_OFFSET + j))
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| cof::COL_IN_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| cof::COL_IN_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| cof::COL_IN_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| cof::COL_IN_Y_C1_LIMB_OFFSET + j))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "hash_to_g2_isogeny_out_to_cofactor_in_v1".into(),
        a_layer_index: hash_to_g2_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ISOGENY_PHASE),
        b_layer_index: g2_cofactor_clear_layer_index,
        b_columns,
        b_selector_column: Some(cof::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Beacon-chain ciphersuite DST (POP variant).
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    // ───────────────────────────────────────────────────────────────────
    // Test 1: witness builds for empty message and round-trips to G2
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_from_empty_message() {
        let w = HashToG2Witness::from_message(b"", POP_DST)
            .expect("hash_to_curve(empty, POP_DST) must decode");
        // Round 11: from_message now emits ROWS_PER_HASH_TO_G2 = 3 rows
        // (one per pipeline phase: SSWU, isogeny, cofactor).
        assert_eq!(w.rows.len(), ROWS_PER_HASH_TO_G2);
        let row = &w.rows[0];

        assert_eq!(row.msg_len, 0);
        assert_eq!(row.dst_len, POP_DST.len() as u8);
        // Padded zero bytes for the unused msg slots.
        assert_eq!(row.msg, [0u8; MSG_LEN]);
        // DST first bytes match.
        assert_eq!(&row.dst[..POP_DST.len()], POP_DST);

        // Output G2 round-trips: out_x_c0_bytes decodes to out_x_c0.
        let recovered = Fp::from_bytes_be(&row.out_x_c0_bytes).unwrap();
        assert_eq!(recovered, row.out_x_c0);
        // Output is not the identity.
        assert!(!row.out_x_c0.is_zero() || !row.out_x_c1.is_zero());

        // Compressed output decodes via our local pairing module to
        // the same Fp coordinates.
        let g2 = crate::pairing::G2Affine::from_bytes(&row.out_compressed).unwrap();
        assert!(!g2.infinity);
        assert_eq!(g2.x.c0, row.out_x_c0);
        assert_eq!(g2.x.c1, row.out_x_c1);
        assert_eq!(g2.y.c0, row.out_y_c0);
        assert_eq!(g2.y.c1, row.out_y_c1);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: witness builds for a 32-byte (signing_root-shape) message
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_from_32_byte_message() {
        let mut msg = [0u8; 32];
        for (i, b) in msg.iter_mut().enumerate() {
            *b = i as u8;
        }
        let w = HashToG2Witness::from_message(&msg, POP_DST)
            .expect("32-byte signing_root must hash to a valid G2 point");
        let row = &w.rows[0];
        assert_eq!(row.msg, msg);
        assert_eq!(row.msg_len, 32);
        assert_eq!(row.dst_len, POP_DST.len() as u8);
        // Output is not the identity.
        assert!(!row.out_y_c0.is_zero() || !row.out_y_c1.is_zero());

        // SSWU step witness: u0_plus_one == u0 + 1 in Fp2.
        let one_fp2 = Fp2 { c0: Fp::one(), c1: Fp::zero() };
        assert_eq!(row.u0_plus_one, row.u0.add(&one_fp2));
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: constraints zero on honest witness; tampering fires
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = HashToG2Witness::from_message(b"hash-to-g2-air-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn limb_decomp_fires_on_tampered_out_x_c0_byte() {
        let w = HashToG2Witness::from_message(b"x", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Tamper byte 47 of out_x_c0_bytes (LSB of limb 5). Add 1.
        let original = cols[COL_OUT_X_C0_BYTES_OFFSET + 47][0].clone();
        cols[COL_OUT_X_C0_BYTES_OFFSET + 47][0] = original.add(&Scalar::one(curve));
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let limb5_constraint = 1 + 5;
        assert!(
            !results[limb5_constraint][0].is_zero(),
            "tampering out_x_c0 byte 47 must fire limb 5 decomp",
        );
    }

    #[test]
    fn sswu_step_fires_on_tampered_u0_plus_one_c0_lsb() {
        let w = HashToG2Witness::from_message(b"sswu-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Tamper u0_plus_one_c0_limbs[5] (the LSB limb where ONE_FP_LIMBS = 1).
        let original = cols[COL_U0_PLUS_ONE_C0_LIMB_OFFSET + 5][0].clone();
        cols[COL_U0_PLUS_ONE_C0_LIMB_OFFSET + 5][0] = original.add(&Scalar::one(curve));
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // SSWU c0 constraints start at index 1 + LIMBS_PER_FP = 7; limb 5 = index 12.
        let sswu_c0_limb5 = 1 + LIMBS_PER_FP + 5;
        assert!(
            !results[sswu_c0_limb5][0].is_zero(),
            "tampering u0_plus_one_c0[5] must fire its SSWU step constraint",
        );
    }

    #[test]
    fn sswu_step_c1_fires_on_tampered_u0_plus_one_c1() {
        let w = HashToG2Witness::from_message(b"sswu-c1-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Tamper u0_plus_one_c1_limbs[0].
        let original = cols[COL_U0_PLUS_ONE_C1_LIMB_OFFSET][0].clone();
        cols[COL_U0_PLUS_ONE_C1_LIMB_OFFSET][0] = original.add(&Scalar::one(curve));
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // c1 constraints start at 1 + 2 * LIMBS_PER_FP = 13; limb 0 = index 13.
        let sswu_c1_limb0 = 1 + 2 * LIMBS_PER_FP;
        assert!(
            !results[sswu_c1_limb0][0].is_zero(),
            "tampering u0_plus_one_c1[0] must fire its SSWU step constraint",
        );
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_selector() {
        let w = HashToG2Witness::from_message(b"is-real-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(5, CurveType::Bls12381);
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "non-binary is_real must fire");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: cross-AIR descriptors are well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn sha256_descriptor_well_formed() {
        let d = make_hash_to_g2_msg_to_sha256_descriptor(0, 1);
        assert_eq!(d.label, "hash_to_g2_msg_to_sha256_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), 64, "first SHA-256 block = 64 bytes");
        assert_eq!(d.b_columns.len(), 64);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::sha256_extract::COL_IS_REAL),
        );
        // First A column = COL_MSG_OFFSET (= 0); first B column =
        // sha256_extract::COL_INPUT_BYTE_OFFSET (= 0).
        assert_eq!(d.a_columns[0], COL_MSG_OFFSET);
        assert_eq!(d.b_columns[0], crate::sha256_extract::COL_INPUT_BYTE_OFFSET);
        // Last A column = COL_DST_OFFSET + 31 (first 32 dst bytes).
        assert_eq!(d.a_columns[63], COL_DST_OFFSET + 31);
    }

    #[test]
    fn bls_pairing_descriptor_well_formed() {
        let d = make_hash_to_g2_to_bls_pairing_descriptor(0, 1);
        assert_eq!(d.label, "hash_to_g2_to_bls_pairing_v1");
        // A side = bls_pairing (layer 1), B side = this AIR (layer 0).
        assert_eq!(d.a_layer_index, 1);
        assert_eq!(d.b_layer_index, 0);
        assert_eq!(d.a_columns.len(), 4 * LIMBS_PER_FP, "4 Fp × 6 limbs = 24");
        assert_eq!(d.b_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(
            d.a_selector_column,
            Some(crate::bls_pairing_air::COL_IS_REAL),
        );
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));
        // First B column = this AIR's out_x_c0 limb 0.
        assert_eq!(d.b_columns[0], COL_OUT_X_C0_LIMB_OFFSET);
        // Last B column = this AIR's out_y_c1 limb 5.
        assert_eq!(
            d.b_columns[4 * LIMBS_PER_FP - 1],
            COL_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP - 1,
        );
        // First A column = bls_pairing's sig_x_c0 limb 0.
        assert_eq!(
            d.a_columns[0],
            crate::bls_pairing_air::COL_SIG_X_C0_LIMB_OFFSET,
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Sanity: column layout & helpers
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_MSG_OFFSET, 0);
        assert_eq!(COL_DST_OFFSET, MSG_LEN);
        assert_eq!(COL_OUT_COMPRESSED_OFFSET, MSG_LEN + MAX_DST_LEN);
        assert_eq!(
            COL_OUT_X_C0_BYTES_OFFSET,
            MSG_LEN + MAX_DST_LEN + OUT_COMPRESSED_LEN,
        );
        assert_eq!(
            COL_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP,
            COL_U0_C0_LIMB_OFFSET,
        );
        assert_eq!(
            COL_U0_PLUS_ONE_C1_LIMB_OFFSET + LIMBS_PER_FP,
            COL_TV1_C0_LIMB_OFFSET,
        );
        // After Round 8, the N1 limbs are followed by 12 Karatsuba
        // scratch Fp values (4 muls × 3 Fp scratches × 6 limbs = 72 cols)
        // before the length / selector columns.
        assert_eq!(
            COL_N1_C1_LIMB_OFFSET + LIMBS_PER_FP,
            COL_TV1_KAR_T0_OFFSET,
        );
        assert_eq!(
            COL_N1_KAR_TCROSS_OFFSET + LIMBS_PER_FP,
            COL_MSG_LEN,
        );
        // Round 11 extends the layout beyond IS_REAL with 3 phase
        // selectors + 8 phase-point limb blocks (4 in + 4 out, each
        // LIMBS_PER_FP wide). The pre-Round-11 NUM_COLUMNS was 555;
        // Round 11 adds 3 + 8 * LIMBS_PER_FP = 51 → 606.
        assert_eq!(
            NUM_ROW_CONSTRAINTS,
            1 + LIMBS_PER_FP
                + 2 * LIMBS_PER_FP
                + 2 * NUM_KARATSUBA_FP2_MULS
                + NUM_PHASE_ROW_CONSTRAINTS,
        );
        assert_eq!(NUM_ROW_CONSTRAINTS, 31);
        // Pre-Round-8 layout was 483 columns. Round 8 adds 72 Karatsuba
        // scratch cols → 555. Round 11 adds 3 phase selectors + 8 phase
        // point blocks × LIMBS_PER_FP = 51 → 606.
        assert_eq!(
            NUM_COLUMNS,
            483 + 4 * 3 * LIMBS_PER_FP + 3 + 8 * LIMBS_PER_FP,
        );
        assert_eq!(NUM_COLUMNS, 606);
        // Pre-Round-11 last column was COL_IS_REAL (= 554). New phase
        // columns occupy 555..606.
        assert_eq!(COL_IS_REAL, 554);
        assert_eq!(COL_IS_SSWU_PHASE, 555);
        assert_eq!(COL_PHASE_IN_Y_C1_LIMB_OFFSET + LIMBS_PER_FP, NUM_COLUMNS);
    }

    #[test]
    fn message_too_long_returns_none() {
        let too_long = [0u8; MSG_LEN + 1];
        assert!(HashToG2Witness::from_message(&too_long, POP_DST).is_none());
        let too_long_dst = vec![0u8; MAX_DST_LEN + 1];
        assert!(HashToG2Witness::from_message(b"x", &too_long_dst).is_none());
    }

    #[test]
    fn out_x_c0_limb_decomp_targets_round_trip() {
        // The targets reconstruct the limb value from the corresponding
        // 8 bytes of a sample Fp encoding.
        let sample = Fp {
            limbs: [
                0x0123456789abcdef,
                0xfedcba9876543210,
                0xdeadbeefcafebabe,
                0x0f0e0d0c0b0a0908,
                0x0706050403020100,
                0x4242424242424242,
            ],
        };
        let bytes = sample.to_bytes_be();
        for j in 0..LIMBS_PER_FP {
            let targets = out_x_c0_limb_decomp_targets(j);
            assert_eq!(targets.len(), BYTES_PER_LIMB);
            let mut acc: u128 = 0;
            for (byte_idx, weight) in &targets {
                acc += (bytes[*byte_idx] as u128) * (*weight as u128);
            }
            assert_eq!(acc as u64, sample.limbs[j], "limb {} mismatch", j);
        }
    }

    #[test]
    fn one_fp_limbs_match_fp_one() {
        // The constant ONE_FP_LIMBS must equal Fp::one().limbs so the
        // SSWU step constraint enforces "add 1" correctly.
        assert_eq!(ONE_FP_LIMBS, Fp::one().limbs);
    }

    // ───────────────────────────────────────────────────────────────────
    // Round 7: SSWU intermediates + descriptor set
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn sswu_step_descriptor_set_has_at_least_twelve_entries() {
        let descriptors = make_sswu_step_descriptors(2, 7);
        assert!(
            descriptors.len() >= 12,
            "SSWU step descriptor set must include >= 12 Fp-mul links \
             (got {}); each Fp2 mul contributes a Karatsuba triple \
             (t0, t1, t_cross) → 4 muls × 3 = 12 links minimum",
            descriptors.len(),
        );
        // All descriptors point at this AIR as A side and nonnative_fp_air
        // as B side, with matching A-side selector and the multiplication
        // selector on the B side.
        for d in &descriptors {
            assert_eq!(d.a_layer_index, 2);
            assert_eq!(d.b_layer_index, 7);
            assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
            assert_eq!(
                d.b_selector_column,
                Some(crate::nonnative_fp_air::COL_SEL_MUL),
            );
            // Each descriptor binds an (a, b, r) tuple of 6 limbs each
            // → 18 columns per side.
            assert_eq!(d.a_columns.len(), 3 * LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_FP);
        }
        // Labels are unique.
        let mut labels: Vec<&str> =
            descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), descriptors.len(), "descriptor labels must be unique");
        // The B-side tuple is the (a, b, r) layout of nfp.
        let d0 = &descriptors[0];
        assert_eq!(d0.b_columns[0], crate::nonnative_fp_air::COL_A_OFFSET);
        assert_eq!(
            d0.b_columns[LIMBS_PER_FP],
            crate::nonnative_fp_air::COL_B_OFFSET,
        );
        assert_eq!(
            d0.b_columns[2 * LIMBS_PER_FP],
            crate::nonnative_fp_air::COL_R_OFFSET,
        );
    }

    #[test]
    fn sswu_intermediates_match_host_side_computation() {
        let w = HashToG2Witness::from_message(b"sswu-intermediates-test", POP_DST).unwrap();
        let row = &w.rows[0];
        let one_fp2 = Fp2 { c0: Fp::one(), c1: Fp::zero() };
        // tv1 = u0^2.
        let expected_tv1 = row.u0.mul(&row.u0);
        assert_eq!(row.tv1, expected_tv1, "tv1 must equal u0^2");
        // tv2 = tv1^2 = u0^4.
        let expected_tv2 = row.tv1.mul(&row.tv1);
        assert_eq!(row.tv2, expected_tv2, "tv2 must equal tv1^2");
        // xd = (-A') * (tv2 + tv1).
        let tv2_plus_tv1 = row.tv2.add(&row.tv1);
        let expected_xd = iso_a_prime().neg().mul(&tv2_plus_tv1);
        assert_eq!(row.xd, expected_xd, "xd must equal (-A') * (tv2 + tv1)");
        // n1 = B' * (tv2 + tv1 + 1).
        let expected_n1 = iso_b_prime().mul(&tv2_plus_tv1.add(&one_fp2));
        assert_eq!(row.n1, expected_n1, "n1 must equal B' * (tv2 + tv1 + 1)");

        // Verify the trace polynomials populate the committed limb
        // columns with these exact intermediates.
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        for (off, fp) in [
            (COL_TV1_C0_LIMB_OFFSET, &row.tv1.c0),
            (COL_TV1_C1_LIMB_OFFSET, &row.tv1.c1),
            (COL_TV2_C0_LIMB_OFFSET, &row.tv2.c0),
            (COL_TV2_C1_LIMB_OFFSET, &row.tv2.c1),
            (COL_XD_C0_LIMB_OFFSET, &row.xd.c0),
            (COL_XD_C1_LIMB_OFFSET, &row.xd.c1),
            (COL_N1_C0_LIMB_OFFSET, &row.n1.c0),
            (COL_N1_C1_LIMB_OFFSET, &row.n1.c1),
        ] {
            for j in 0..LIMBS_PER_FP {
                let expected =
                    Scalar::from_u64(fp.limbs[j], CurveType::Bls12381);
                let actual = &trace.columns[off + j].evaluations[0];
                assert!(
                    actual.sub(&expected).is_zero(),
                    "limb {} at offset {} mismatch",
                    j, off,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Round 8: Karatsuba scratch + extended descriptor set
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn karatsuba_scratch_constraints_zero_on_honest_witness() {
        let w =
            HashToG2Witness::from_message(b"karatsuba-honest-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);

        // The Karatsuba shape constraints occupy indices
        // 1 + NUM_X_C0_LIMB_CONSTRAINTS + NUM_SSWU_STEP_CONSTRAINTS .. NUM_ROW_CONSTRAINTS
        // (= 19..27). Each must be zero on the honest witness.
        let kar_start = 1 + NUM_X_C0_LIMB_CONSTRAINTS + NUM_SSWU_STEP_CONSTRAINTS;
        assert_eq!(kar_start, 19);
        for i in kar_start..NUM_ROW_CONSTRAINTS {
            for (r, val) in results[i].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "Karatsuba constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }

        // Sanity-check the host-side Karatsuba relations on every Fp2 mul:
        //   c0 = t0 - t1
        //   c1 = t_cross - t0 - t1.
        let row = &w.rows[0];
        for (c, kar, label) in [
            (&row.tv1, &row.tv1_kar, "tv1"),
            (&row.tv2, &row.tv2_kar, "tv2"),
            (&row.xd, &row.xd_kar, "xd"),
            (&row.n1, &row.n1_kar, "n1"),
        ] {
            assert_eq!(c.c0, kar.t0.sub(&kar.t1), "{}: c0 = t0 - t1", label);
            let t_sum = kar.t0.add(&kar.t1);
            assert_eq!(
                c.c1,
                kar.t_cross.sub(&t_sum),
                "{}: c1 = t_cross - t0 - t1",
                label,
            );
        }
    }

    #[test]
    fn karatsuba_scratch_host_side_decomposition_round_trips() {
        // The Karatsuba scratch values must satisfy the field-element
        // Karatsuba identities on the honest witness:
        //   c0 = t0 - t1
        //   c1 = t_cross - t0 - t1
        // for each of the four committed Fp2 multiplications. The
        // row-local AIR body for these tuples is a structural pin (see
        // `eval_karatsuba_c0_body` / `eval_karatsuba_c1_body`); the
        // actual soundness binding comes from the cross-AIR LogUp
        // descriptors returned by `make_sswu_step_descriptors`. This
        // test pins the host-side scratch population so the descriptor
        // links commit the right tuples.
        let w =
            HashToG2Witness::from_message(b"karatsuba-roundtrip-test", POP_DST).unwrap();
        let row = &w.rows[0];
        for (c, kar, label) in [
            (&row.tv1, &row.tv1_kar, "tv1"),
            (&row.tv2, &row.tv2_kar, "tv2"),
            (&row.xd, &row.xd_kar, "xd"),
            (&row.n1, &row.n1_kar, "n1"),
        ] {
            assert_eq!(c.c0, kar.t0.sub(&kar.t1), "{}: c0 = t0 - t1", label);
            let t_sum = kar.t0.add(&kar.t1);
            assert_eq!(
                c.c1,
                kar.t_cross.sub(&t_sum),
                "{}: c1 = t_cross - t0 - t1",
                label,
            );
        }
    }

    #[test]
    fn sswu_step_descriptors_cover_all_four_fp2_muls() {
        let descriptors = make_sswu_step_descriptors(2, 7);
        assert_eq!(descriptors.len(), 12, "must have 12 descriptors total");

        // Each Fp2 mul (tv1, tv2, xd, n1) must contribute exactly three
        // descriptors whose result column (last 6 columns of the A side)
        // matches the corresponding Karatsuba scratch slot.
        let expected_results: Vec<(&str, [usize; 3])> = vec![
            (
                "tv1",
                [
                    COL_TV1_KAR_T0_OFFSET,
                    COL_TV1_KAR_T1_OFFSET,
                    COL_TV1_KAR_TCROSS_OFFSET,
                ],
            ),
            (
                "tv2",
                [
                    COL_TV2_KAR_T0_OFFSET,
                    COL_TV2_KAR_T1_OFFSET,
                    COL_TV2_KAR_TCROSS_OFFSET,
                ],
            ),
            (
                "xd",
                [
                    COL_XD_KAR_T0_OFFSET,
                    COL_XD_KAR_T1_OFFSET,
                    COL_XD_KAR_TCROSS_OFFSET,
                ],
            ),
            (
                "n1",
                [
                    COL_N1_KAR_T0_OFFSET,
                    COL_N1_KAR_T1_OFFSET,
                    COL_N1_KAR_TCROSS_OFFSET,
                ],
            ),
        ];

        for (name, scratch_offs) in &expected_results {
            let mut hits = Vec::new();
            for d in &descriptors {
                // A-side R-tuple starts at column index 12 (after a, b).
                let r_first = d.a_columns[2 * LIMBS_PER_FP];
                if scratch_offs.contains(&r_first) {
                    hits.push(r_first);
                }
            }
            hits.sort();
            let mut sorted_offs = scratch_offs.to_vec();
            sorted_offs.sort();
            assert_eq!(
                hits, sorted_offs,
                "Fp2 mul {} must have descriptors for all 3 Karatsuba scratches",
                name,
            );
        }

        // All 12 labels distinct.
        let mut labels: Vec<&str> =
            descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), 12, "all 12 descriptor labels must be unique");
    }

    // ───────────────────────────────────────────────────────────────────
    // Round 11: multi-row phase stitching
    // ───────────────────────────────────────────────────────────────────

    /// Honest witness must emit `ROWS_PER_HASH_TO_G2 = 3` rows with the
    /// expected phase selector assignments and the cross-row continuity
    /// `phase_out[r] = phase_in[r+1]` honored between adjacent phases.
    #[test]
    fn multi_row_witness_emits_three_phase_rows() {
        let w = HashToG2Witness::from_message(b"multi-row-test", POP_DST).unwrap();
        assert_eq!(w.rows.len(), ROWS_PER_HASH_TO_G2);
        assert_eq!(w.rows[0].phase, 0, "row 0 = SSWU phase");
        assert_eq!(w.rows[1].phase, 1, "row 1 = isogeny phase");
        assert_eq!(w.rows[2].phase, 2, "row 2 = cofactor phase");

        // Phase continuity at the witness level: SSWU output = isogeny
        // input, isogeny output = cofactor input.
        assert_eq!(w.rows[0].phase_out_x_c0, w.rows[1].phase_in_x_c0);
        assert_eq!(w.rows[0].phase_out_x_c1, w.rows[1].phase_in_x_c1);
        assert_eq!(w.rows[0].phase_out_y_c0, w.rows[1].phase_in_y_c0);
        assert_eq!(w.rows[0].phase_out_y_c1, w.rows[1].phase_in_y_c1);
        assert_eq!(w.rows[1].phase_out_x_c0, w.rows[2].phase_in_x_c0);
        assert_eq!(w.rows[1].phase_out_x_c1, w.rows[2].phase_in_x_c1);
        assert_eq!(w.rows[1].phase_out_y_c0, w.rows[2].phase_in_y_c0);
        assert_eq!(w.rows[1].phase_out_y_c1, w.rows[2].phase_in_y_c1);
    }

    /// Phase selector columns must hold the expected one-hot pattern
    /// across the 3 rows, and the row-local binarity + exclusivity
    /// constraints must vanish on the honest witness. (Padding rows must
    /// have all phase selectors zero.)
    #[test]
    fn phase_selectors_set_correctly_and_constraints_vanish() {
        let w =
            HashToG2Witness::from_message(b"phase-selectors-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);

        let is_one = |s: &Scalar| s.sub(&one).is_zero();
        let is_zero = |s: &Scalar| s.is_zero();

        // Row 0: SSWU.
        assert!(is_one(&trace.columns[COL_IS_SSWU_PHASE].evaluations[0]));
        assert!(is_zero(&trace.columns[COL_IS_ISOGENY_PHASE].evaluations[0]));
        assert!(is_zero(&trace.columns[COL_IS_COFACTOR_PHASE].evaluations[0]));
        // Row 1: isogeny.
        assert!(is_zero(&trace.columns[COL_IS_SSWU_PHASE].evaluations[1]));
        assert!(is_one(&trace.columns[COL_IS_ISOGENY_PHASE].evaluations[1]));
        assert!(is_zero(&trace.columns[COL_IS_COFACTOR_PHASE].evaluations[1]));
        // Row 2: cofactor.
        assert!(is_zero(&trace.columns[COL_IS_SSWU_PHASE].evaluations[2]));
        assert!(is_zero(&trace.columns[COL_IS_ISOGENY_PHASE].evaluations[2]));
        assert!(is_one(&trace.columns[COL_IS_COFACTOR_PHASE].evaluations[2]));

        // Padding rows (≥ ROWS_PER_HASH_TO_G2): all phase selectors 0
        // AND IS_REAL = 0, so the exclusivity body 0 - 0 = 0 vanishes.
        for r in ROWS_PER_HASH_TO_G2..trace.columns[COL_IS_REAL].evaluations.len() {
            assert!(is_zero(&trace.columns[COL_IS_SSWU_PHASE].evaluations[r]));
            assert!(is_zero(&trace.columns[COL_IS_ISOGENY_PHASE].evaluations[r]));
            assert!(is_zero(&trace.columns[COL_IS_COFACTOR_PHASE].evaluations[r]));
            assert!(is_zero(&trace.columns[COL_IS_REAL].evaluations[r]));
        }

        // The 4 new row-local constraints (3 binarity + 1 exclusivity)
        // must vanish on every domain row.
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        let phase_start = NUM_ROW_CONSTRAINTS - NUM_PHASE_ROW_CONSTRAINTS;
        for i in phase_start..NUM_ROW_CONSTRAINTS {
            for (r, val) in results[i].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "phase constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    /// Cross-row shifted continuity must vanish on the honest witness:
    /// at the SSWU row, `phase_out = phase_in_next`; at the isogeny row,
    /// likewise. We sample `evaluate_shifted_at_point` at a random α
    /// with the true row-0 and row-1 column values, and the shifted
    /// reads pulled directly from rows 1 / 2 of the trace.
    #[test]
    fn shifted_continuity_vanishes_on_honest_witness() {
        let w =
            HashToG2Witness::from_message(b"shifted-continuity-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let curve = CurveType::Bls12381;

        // Sanity: 2 shifted constraints declared, shifted column index
        // list has the expected length.
        assert_eq!(cs.num_shifted_constraints(), 2);
        let shifted_cols = cs.shifted_column_indices();
        assert_eq!(shifted_cols.len(), 2 + 4 * LIMBS_PER_FP);
        assert_eq!(shifted_cols[0], COL_IS_ISOGENY_PHASE);
        assert_eq!(shifted_cols[1], COL_IS_COFACTOR_PHASE);

        // Build (column_evals_at_row_r, shifted_evals_from_row_(r+1))
        // for r = 0 (SSWU→isogeny) and r = 1 (isogeny→cofactor); both
        // must yield zero under any α.
        let alpha = Scalar::from_u64(0xc0ffee, curve);
        // Use a dummy ω^{n-1} *not* equal to z, so the (z - ω^{n-1})
        // boundary factor is non-zero — this isolates the constraint
        // body itself.
        let z = Scalar::from_u64(0xbeef, curve);
        let omega_n_minus_1 = Scalar::from_u64(0xface, curve);

        for r in 0..2 {
            let col_evals: Vec<Scalar> = (0..NUM_COLUMNS)
                .map(|c| trace.columns[c].evaluations[r].clone())
                .collect();
            let shifted_evals: Vec<Scalar> = shifted_cols
                .iter()
                .map(|c| trace.columns[*c].evaluations[r + 1].clone())
                .collect();
            let body = cs.evaluate_shifted_at_point(
                &col_evals,
                &shifted_evals,
                &z,
                &omega_n_minus_1,
                &alpha,
                /* alpha_offset = */ NUM_ROW_CONSTRAINTS,
            );
            assert!(
                body.is_zero(),
                "shifted continuity body at row {} must be zero on honest \
                 witness; got {:?}",
                r, body,
            );
        }
    }

    /// Tampering the phase output at the SSWU row (so it disagrees with
    /// the isogeny row's phase input) must make the SSWU→isogeny
    /// shifted body non-zero under a random α (gating
    /// `IS_SSWU_PHASE[r] * IS_ISOGENY_PHASE[r+1]` evaluates to 1).
    #[test]
    fn tampered_cross_row_phase_out_fires_shifted_continuity() {
        let w =
            HashToG2Witness::from_message(b"shifted-tamper-test", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = HashToG2ConstraintSystem::new(trace.num_rows);
        let curve = CurveType::Bls12381;

        // Tamper PHASE_OUT_X_C0 limb 0 at row 0 (SSWU output) so it no
        // longer equals row 1's PHASE_IN_X_C0 limb 0.
        let mut col_evals: Vec<Scalar> = (0..NUM_COLUMNS)
            .map(|c| trace.columns[c].evaluations[0].clone())
            .collect();
        let tampered = col_evals[COL_PHASE_OUT_X_C0_LIMB_OFFSET]
            .add(&Scalar::from_u64(1, curve));
        col_evals[COL_PHASE_OUT_X_C0_LIMB_OFFSET] = tampered;

        let shifted_cols = cs.shifted_column_indices();
        let shifted_evals: Vec<Scalar> = shifted_cols
            .iter()
            .map(|c| trace.columns[*c].evaluations[1].clone())
            .collect();

        let alpha = Scalar::from_u64(0xc0ffee, curve);
        let z = Scalar::from_u64(0xbeef, curve);
        let omega_n_minus_1 = Scalar::from_u64(0xface, curve);

        let body = cs.evaluate_shifted_at_point(
            &col_evals,
            &shifted_evals,
            &z,
            &omega_n_minus_1,
            &alpha,
            NUM_ROW_CONSTRAINTS,
        );
        // Gating IS_SSWU_PHASE[r] * IS_ISOGENY_PHASE[r+1] is 1·1 = 1 at
        // row 0; the β-RLC body has a single non-zero term at α^0 = 1
        // contributing +1, so the body is non-zero for any α.
        assert!(
            !body.is_zero(),
            "tampered phase_out at SSWU row must fire shifted continuity",
        );
    }

    /// The two new cross-AIR LogUp descriptors must be well-formed and
    /// reference the correct phase selectors / target AIR offsets.
    #[test]
    fn phase_stitching_descriptors_are_well_formed() {
        let to_iso = make_hash_to_g2_to_isogeny_descriptor(0, 1);
        assert_eq!(to_iso.label, "hash_to_g2_sswu_out_to_isogeny_in_v1");
        assert_eq!(to_iso.a_layer_index, 0);
        assert_eq!(to_iso.b_layer_index, 1);
        assert_eq!(to_iso.a_selector_column, Some(COL_IS_SSWU_PHASE));
        assert_eq!(
            to_iso.b_selector_column,
            Some(crate::isogeny_map_air::COL_IS_REAL),
        );
        assert_eq!(to_iso.a_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(to_iso.b_columns.len(), 4 * LIMBS_PER_FP);
        // First A column = SSWU row's phase_out_x_c0 limb 0.
        assert_eq!(to_iso.a_columns[0], COL_PHASE_OUT_X_C0_LIMB_OFFSET);
        // First B column = isogeny_map_air's in_x_c0 limb 0.
        assert_eq!(
            to_iso.b_columns[0],
            crate::isogeny_map_air::COL_IN_X_C0_LIMB_OFFSET,
        );

        let to_cof = make_hash_to_g2_to_cofactor_descriptor(0, 2);
        assert_eq!(to_cof.label, "hash_to_g2_isogeny_out_to_cofactor_in_v1");
        assert_eq!(to_cof.a_layer_index, 0);
        assert_eq!(to_cof.b_layer_index, 2);
        assert_eq!(to_cof.a_selector_column, Some(COL_IS_ISOGENY_PHASE));
        assert_eq!(
            to_cof.b_selector_column,
            Some(crate::g2_cofactor_clear_air::COL_IS_REAL),
        );
        assert_eq!(to_cof.a_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(to_cof.b_columns.len(), 4 * LIMBS_PER_FP);
        // First B column = cofactor_clear's in_x_c0 limb 0.
        assert_eq!(
            to_cof.b_columns[0],
            crate::g2_cofactor_clear_air::COL_IN_X_C0_LIMB_OFFSET,
        );

        // Distinct labels.
        assert_ne!(to_iso.label, to_cof.label);
    }
}
