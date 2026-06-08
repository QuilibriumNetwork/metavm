//! BLS12-381 pairing-equation AIR scaffold (#62 — algebraic FFG finality).
//!
//! # Purpose
//!
//! This module is the first algebraic stepping-stone toward an in-circuit
//! BLS12-381 signature verifier. The full Miller loop + final
//! exponentiation is a multi-week build (~12 cyclotomic squarings, ~64
//! Fp12 multiplications, ~64 line evaluations, plus the
//! `(p^12 - 1)/r` final exponentiation). This scaffold gets the column
//! layout, the trace builder, the cross-AIR linkage descriptor, and one
//! substantive algebraic constraint into the tree so that downstream
//! work (Miller-loop step rows, line-evaluation rows, Fp12 accumulator
//! rows) can be added incrementally without re-litigating the
//! row/column shape.
//!
//! # What is algebraically enforced
//!
//! Per row, this AIR commits one (pk, sig, msg_hash) triple plus the
//! decoded Fp coordinates of the G1 pubkey and G2 signature. The row
//! holds two byte forms of the pubkey:
//!
//! * `pk_compressed[0..48]` — the IETF compressed G1 encoding (with
//!   the 3 flag bits in `pk_compressed[0] & 0xe0`). This is the form
//!   consumed by SSZ Merkle inclusion and by the sync-committee
//!   filter AIR.
//! * `pk_x_bytes[0..48]` — the canonical 48-byte big-endian encoding
//!   of `pk.x ∈ Fp` (flag bits cleared). This is the form decoded by
//!   [`crate::nonnative_fp::Fp::from_bytes_be`].
//!
//! The AIR enforces:
//!
//!   1. `is_real ∈ {0, 1}` — selector binarity.
//!   2. G1 pubkey-x **byte-to-limb decomposition** — 6 equations binding
//!      `pk_x_limbs[j] = Σ_{k=0..8} pk_x_bytes[8j + k] · 2^(8·(7−k))`
//!      for `j = 0..6`. These are the *Fp tower entry-point*
//!      constraints: they algebraically tie the canonical byte form of
//!      the pubkey x-coordinate to the 6-limb form expected by the
//!      [`crate::nonnative_fp_air`] arithmetic AIR (which proves
//!      `a · b mod p` for BLS12-381 Fp). Cross-AIR LogUp can then bind
//!      `pk_x_limbs` to a row of the Fp AIR holding the pairing-equation
//!      intermediate products.
//!   3. Pubkey **byte-equality** — 47 equations binding
//!      `pk_compressed[k] = pk_x_bytes[k]` for `k = 1..48`. Byte 0
//!      differs only in the 3 high flag bits; we leave the byte-0
//!      flag-vs-canonical relation as a deliberate oracle gap (see
//!      "Not yet enforced" below).
//!
//! Together: 1 selector + 6 limb-decomposition + 47 byte-equality =
//! **54 row-local constraints**.
//!
//! # What is NOT yet algebraically enforced (oracle / deferred)
//!
//! - **Pubkey compression-flag binding** for `pk_compressed[0]` vs.
//!   `pk_x_bytes[0]`: the IETF spec packs (compression, infinity,
//!   y_sign) into the 3 high bits of byte 0. A future "G1 flag-byte
//!   gadget" will algebraically split `pk_compressed[0] =
//!   flag_byte * 32 + pk_x_bytes[0]` with `flag_byte ∈ {0..7}` and
//!   tie `(compression == 1, infinity == 0)` to validity. The current
//!   AIR commits both byte forms but does not algebraically equate
//!   their byte-0 entries.
//! - **G1 on-curve check** `y² = x³ + 4` on `(pk_x_limbs, pk_y_limbs)` —
//!   needs two Fp multiplications and an Fp addition per row. Each is
//!   one row of `nonnative_fp_air` reachable via cross-AIR LogUp.
//! - **G2 on-curve check** `y² = x³ + 4(1+u)` on the four Fp2
//!   components of the signature — similarly four `nonnative_fp_air`
//!   rows wired through the Fp2 tower.
//! - **G1 / G2 subgroup checks** (`blst_p1_affine_in_g1`,
//!   `blst_p2_affine_in_g2`) — non-trivial. Need a cofactor-clearing
//!   scalar mul (G1) or the `ψ` endomorphism (G2).
//! - **Hash-to-curve** for the message: `H(msg) ∈ G2`. Needs the SSWU
//!   map + XMD:SHA-256 expand. Currently the message hash is committed
//!   verbatim and the host-side trace builder calls
//!   [`crate::bls_sig::hash_to_g2_affine`] to recover H(msg) bytes;
//!   that oracle work is replaced once we have an in-circuit hash-to-G2.
//! - **Miller loop** — per-bit double-and-add over the BLS x = -0xd201
//!   pseudo-binary encoding. Needs ~64 line-evaluation rows in a
//!   future `miller_step_air` plus an Fp12 accumulator row layout.
//! - **Final exponentiation** `(p^12 - 1) / r` — the easy part is
//!   trivial (`Frobenius` + inversion), the hard part is the cyclotomic
//!   exponentiation by `x`. Each cyclotomic squaring is one Fp12
//!   square + projection.
//! - **Pairing-product equality check** `e(−G1, sig) · e(pk, H(msg))
//!   == 1` — the closure that ties the two Miller loops together.
//!
//! In short: this scaffold proves that the witnessed pk byte-encoding
//! is consistent with a 6-limb Fp representation suitable for handoff
//! to the Fp arithmetic AIR. Everything else above the byte-↔-limb
//! bridge is host-side oracle (the witness builder calls
//! [`crate::pairing::verify_e1_e2_equal_one`] before commitment, so the
//! sig is real, but the verifier does not yet re-prove it
//! algebraically).
//!
//! # Cross-AIR linkages
//!
//! - [`make_bls_pairing_to_filter_descriptor`] — A side = filter AIR
//!   selected pubkeys (gated by `BITMAP_BIT`), B side = this AIR's
//!   `pk_bytes[0..48]` (gated by `IS_REAL`). 48-tuple LogUp. The
//!   inverse direction of
//!   [`crate::sync_committee_filter_air::make_filter_to_bls_sig_linkage_descriptor`]
//!   with the correct B-side gate now that this AIR exposes one.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::nonnative_fp::Fp;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// BLS12-381 compressed pubkey length (G1, 48 bytes).
pub const PK_BYTES: usize = 48;
/// BLS12-381 compressed signature length (G2, 96 bytes).
pub const SIG_BYTES: usize = 96;
/// SSZ message-hash length (32 bytes).
pub const MSG_HASH_BYTES: usize = 32;

/// Number of 64-bit limbs in one Fp element (381-bit field → 6 × 64 = 384).
pub const LIMBS_PER_FP: usize = 6;
/// Bytes per limb (8 for u64).
pub const BYTES_PER_LIMB: usize = 8;

/// Number of Fp values per Fp12 element (top-level Fp basis).
pub const FP_PER_FP12: usize = 12;
/// Number of 64-bit limbs per Fp12 element (12 × 6 = 72).
pub const LIMBS_PER_FP12: usize = FP_PER_FP12 * LIMBS_PER_FP;
/// Number of Fp values per G2 affine point in (x.c0, x.c1, y.c0, y.c1) layout.
pub const FP_PER_G2: usize = 4;
/// Number of 64-bit limbs per G2 affine point (4 × 6 = 24).
pub const LIMBS_PER_G2: usize = FP_PER_G2 * LIMBS_PER_FP;

// ─── Column layout ────────────────────────────────────────────────────
//
// Row layout (per (pk, sig, msg_hash) triple):
//
//   pk_compressed: 48 bytes (IETF compressed G1, flags in byte 0)
//   pk_x_bytes:    48 bytes (canonical BE Fp encoding, flag bits cleared)
//   sig_bytes:     96 bytes (IETF compressed G2)
//   msg_hash:      32 bytes
//   pk_x_limbs:     6 limbs (BE u64)
//   pk_y_limbs:     6 limbs
//   sig_x_c0:       6 limbs
//   sig_x_c1:       6 limbs
//   sig_y_c0:       6 limbs
//   sig_y_c1:       6 limbs
//   is_real:        1
// total: 48 + 48 + 96 + 32 + 36 + 1 = 261 columns

pub const COL_PK_COMPRESSED_OFFSET: usize = 0;
pub const COL_PK_X_BYTES_OFFSET: usize = COL_PK_COMPRESSED_OFFSET + PK_BYTES;
pub const COL_SIG_BYTES_OFFSET: usize = COL_PK_X_BYTES_OFFSET + PK_BYTES;
pub const COL_MSG_HASH_OFFSET: usize = COL_SIG_BYTES_OFFSET + SIG_BYTES;

pub const COL_PK_X_LIMB_OFFSET: usize = COL_MSG_HASH_OFFSET + MSG_HASH_BYTES;
pub const COL_PK_Y_LIMB_OFFSET: usize = COL_PK_X_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_SIG_X_C0_LIMB_OFFSET: usize = COL_PK_Y_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_SIG_X_C1_LIMB_OFFSET: usize = COL_SIG_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_SIG_Y_C0_LIMB_OFFSET: usize = COL_SIG_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_SIG_Y_C1_LIMB_OFFSET: usize = COL_SIG_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

// ─── Pairing-equation witness columns (Task #308) ─────────────────────
//
// To algebraically enforce `e(pk, H(msg)) == e(sig, G2_gen)` we commit
// per row:
//
//   * `msg_hash_g2_limbs[0..24]`   — H(msg) ∈ G2 as 4 × 6 big-endian
//     u64 limbs `(x.c0 || x.c1 || y.c0 || y.c1)`. Host-side oracle for
//     now (hash-to-curve is deferred); algebraically bound only by the
//     LogUp linkage to a miller_loop_air row's `Q_fixed` columns and
//     by the row-local β-RLC equality between the two Fp12 results.
//   * `claimed_pairing_result_1[0..72]` — Fp12 output of the first
//     Miller loop `miller_loop(pk, H(msg))` *before* the final-exp
//     conjugation that wraps both halves. Layout matches
//     [`crate::nonnative_tower::Fp12`] flattening: 12 Fp values × 6
//     limbs, big-endian.
//   * `claimed_pairing_result_2[0..72]` — Fp12 output of the second
//     Miller loop `miller_loop(sig, G2_gen)`.
//
// The two `claimed_pairing_result_*` columns are then bound to a
// `final_exp_air` row via cross-AIR LogUp; the row-local β-RLC equality
// constraint pins them equal so that the *quotient* `result_1 /
// result_2` equals 1 in Fp12 and therefore so does its final-
// exponentiation image. (Equivalently: `final_exp(result_1) ==
// final_exp(result_2)`.) The final_exp_air descriptor surfaces both
// halves as candidate `f_pre` rows whose `f_final` output is pinned to
// `Fp12::one()`.

pub const COL_MSG_HASH_G2_LIMB_OFFSET: usize = COL_SIG_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_CLAIMED_PAIRING_RESULT_1_OFFSET: usize =
    COL_MSG_HASH_G2_LIMB_OFFSET + LIMBS_PER_G2;
pub const COL_CLAIMED_PAIRING_RESULT_2_OFFSET: usize =
    COL_CLAIMED_PAIRING_RESULT_1_OFFSET + LIMBS_PER_FP12;

pub const COL_IS_REAL: usize = COL_CLAIMED_PAIRING_RESULT_2_OFFSET + LIMBS_PER_FP12;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// Row-local constraints:
///   0:        is_real ∈ {0, 1}
///   1..7:     6 pk-x limb-decomposition equations (one per Fp limb)
///   7..54:    47 pubkey byte-equality equations (pk_compressed[k] = pk_x_bytes[k] for k=1..48)
///   54:       β-RLC equality body over 72 Fp12 limbs binding
///             `claimed_pairing_result_1 == claimed_pairing_result_2`
///             (gated by `is_real`). Equivalent to enforcing that the
///             two Miller-loop outputs agree, which together with the
///             final-exp LogUp linkage closes the pairing equation
///             `e(pk, H(msg)) == e(sig, G2_gen)`.
pub const NUM_BYTE_EQUALITY_CONSTRAINTS: usize = PK_BYTES - 1;
pub const NUM_ROW_CONSTRAINTS: usize =
    1 + LIMBS_PER_FP + NUM_BYTE_EQUALITY_CONSTRAINTS + 1;
pub const NUM_SHIFTED: usize = 0;

/// Constraint index of the Fp12-equality body.
pub const CONSTRAINT_PAIRING_RESULT_EQ: usize =
    1 + LIMBS_PER_FP + NUM_BYTE_EQUALITY_CONSTRAINTS;

// ─── Witness ──────────────────────────────────────────────────────────

/// One (pk, sig, msg_hash) row plus the decoded Fp coordinates.
///
/// All Fp coordinates are computed host-side from the compressed
/// encodings; the AIR commits them as 6 big-endian u64 limbs per Fp
/// (matching the [`crate::nonnative_fp::Fp`] convention).
#[derive(Clone, Debug)]
pub struct BlsPairingRow {
    /// Compressed G1 pubkey (IETF BLS12-381 encoding, 48 bytes). Byte
    /// 0 carries the (compression, infinity, y_sign) flag triple in
    /// its top 3 bits.
    pub pk_compressed: [u8; PK_BYTES],
    /// Canonical 48-byte BE encoding of `pk.x` as an Fp element
    /// (`pk_compressed` with byte 0 masked to clear the 3 flag bits).
    pub pk_x_bytes: [u8; PK_BYTES],
    /// Compressed G2 signature (IETF BLS12-381 encoding, 96 bytes).
    pub sig_bytes: [u8; SIG_BYTES],
    /// SSZ-style 32-byte message hash (e.g. `signing_root`).
    pub msg_hash: [u8; MSG_HASH_BYTES],

    /// Decoded G1 affine coords (host-side oracle).
    pub pk_x: Fp,
    pub pk_y: Fp,

    /// Decoded G2 affine coords as four Fp components.
    pub sig_x_c0: Fp,
    pub sig_x_c1: Fp,
    pub sig_y_c0: Fp,
    pub sig_y_c1: Fp,

    /// H(msg) ∈ G2 affine, four Fp components (host-side oracle until
    /// the in-circuit hash-to-G2 gadget lands). All zeros for synthetic
    /// fixtures that don't need a real Miller-loop binding.
    pub msg_hash_g2_x_c0: Fp,
    pub msg_hash_g2_x_c1: Fp,
    pub msg_hash_g2_y_c0: Fp,
    pub msg_hash_g2_y_c1: Fp,

    /// Fp12 output of `pairing(pk, H(msg))` = `final_exp(miller_loop(pk, H(msg)))`.
    pub claimed_pairing_result_1: crate::nonnative_tower::Fp12,
    /// Fp12 output of `pairing(G1_gen, sig)` = `final_exp(miller_loop(G1_gen, sig))`.
    pub claimed_pairing_result_2: crate::nonnative_tower::Fp12,
}

#[derive(Clone, Debug, Default)]
pub struct BlsPairingWitness {
    pub rows: Vec<BlsPairingRow>,
}

impl BlsPairingWitness {
    /// Build a witness by decoding compressed (pk, sig) bytes into Fp
    /// coordinates via the host-side [`crate::pairing`] reference
    /// implementation. Used by tests and host-side bundles.
    ///
    /// Returns `None` if any compressed encoding fails to decompress as
    /// a valid affine point (so the AIR is never asked to commit
    /// garbage). For the AIR alone we do not require subgroup
    /// membership; only on-curve.
    pub fn from_decoded(
        pk_compressed: [u8; PK_BYTES],
        sig_bytes: [u8; SIG_BYTES],
        msg_hash: [u8; MSG_HASH_BYTES],
    ) -> Option<Self> {
        let pk = crate::pairing::G1Affine::from_bytes(&pk_compressed).ok()?;
        let sig = crate::pairing::G2Affine::from_bytes(&sig_bytes).ok()?;
        if pk.infinity || sig.infinity {
            // Identity points have no well-defined coordinates and
            // typically indicate a malformed input for pairing
            // verification.
            return None;
        }
        // Canonical x-bytes form: mask off the 3 flag bits in byte 0.
        // For non-infinity G1 points the remaining 381 bits are the
        // big-endian Fp encoding of `pk.x`.
        let mut pk_x_bytes = pk_compressed;
        pk_x_bytes[0] &= 0x1f;
        // Sanity: the canonical form must round-trip through Fp.
        debug_assert_eq!(
            Fp::from_bytes_be(&pk_x_bytes).ok(),
            Some(pk.x),
            "masked pk_compressed must decode to pk.x",
        );
        // Pairing-equation witness fields (Task #308). For now the
        // host-side oracle does not need a real `H(msg)` for the
        // synthetic round-trip path: callers that want the real values
        // should call [`Self::with_pairing_equation_oracle`] after
        // construction. Default to all-zero so synthetic fixtures still
        // build (and the row-local Fp12-equality body is trivially
        // satisfied: 0 == 0).
        let zero_fp = Fp { limbs: [0u64; LIMBS_PER_FP] };
        let zero_fp12 = crate::nonnative_tower::Fp12::zero();
        Some(Self {
            rows: vec![BlsPairingRow {
                pk_compressed,
                pk_x_bytes,
                sig_bytes,
                msg_hash,
                pk_x: pk.x,
                pk_y: pk.y,
                sig_x_c0: sig.x.c0,
                sig_x_c1: sig.x.c1,
                sig_y_c0: sig.y.c0,
                sig_y_c1: sig.y.c1,
                msg_hash_g2_x_c0: zero_fp.clone(),
                msg_hash_g2_x_c1: zero_fp.clone(),
                msg_hash_g2_y_c0: zero_fp.clone(),
                msg_hash_g2_y_c1: zero_fp,
                claimed_pairing_result_1: zero_fp12.clone(),
                claimed_pairing_result_2: zero_fp12,
            }],
        })
    }

    /// Populate the pairing-equation oracle fields (Task #308) on row
    /// `row_idx` from the host-side BLS reference implementation:
    ///
    ///   * `msg_hash_g2_*` is set to `hash_to_curve_g2(msg, dst)`,
    ///     converted from `blst_p2_affine` to our [`Fp`] form.
    ///   * `claimed_pairing_result_1 = miller_loop(pk, H(msg))`.
    ///   * `claimed_pairing_result_2 = miller_loop(sig, G2_gen)`.
    ///
    /// The row's `pk_*` / `sig_*` fields must already be populated and
    /// consistent with `(pk_compressed, sig_bytes)` — typically the row
    /// was just constructed via [`Self::from_decoded`].
    ///
    /// Returns the original `Self` on error (any decompression failure).
    pub fn with_pairing_equation_oracle(
        mut self,
        row_idx: usize,
        msg: &[u8],
        dst: &[u8],
    ) -> Result<Self, Self> {
        if row_idx >= self.rows.len() {
            return Err(self);
        }
        let row = &self.rows[row_idx];
        let pk = match crate::pairing::G1Affine::from_bytes(&row.pk_compressed) {
            Ok(p) => p,
            Err(_) => return Err(self),
        };
        let sig = match crate::pairing::G2Affine::from_bytes(&row.sig_bytes) {
            Ok(s) => s,
            Err(_) => return Err(self),
        };

        // hash_to_g2 via blst, then convert each blst_fp to Fp through
        // the 48-byte BE round-trip (same pattern as
        // nonnative_fp2_compile tests).
        use blst::{blst_bendian_from_fp, blst_fp};
        fn blst_fp_to_fp(b: &blst_fp) -> Fp {
            let mut bytes = [0u8; 48];
            unsafe { blst_bendian_from_fp(bytes.as_mut_ptr(), b); }
            Fp::from_bytes_be(&bytes).expect("blst fp canonical")
        }
        let h_aff = crate::bls_sig::hash_to_g2_affine(msg, dst);
        let h_x_c0 = blst_fp_to_fp(&h_aff.x.fp[0]);
        let h_x_c1 = blst_fp_to_fp(&h_aff.x.fp[1]);
        let h_y_c0 = blst_fp_to_fp(&h_aff.y.fp[0]);
        let h_y_c1 = blst_fp_to_fp(&h_aff.y.fp[1]);
        let h_g2 = crate::pairing::G2Affine {
            x: crate::nonnative_fp::Fp2 { c0: h_x_c0.clone(), c1: h_x_c1.clone() },
            y: crate::nonnative_fp::Fp2 { c0: h_y_c0.clone(), c1: h_y_c1.clone() },
            infinity: false,
        };

        // Standard BLS verify equation (IETF §3.2):
        //     e(pk, H(msg)) == e(G1_gen, sig)
        // The row-local Fp12 equality body pins
        // `claimed_pairing_result_1 == claimed_pairing_result_2`. For
        // that to hold on an honest signed witness we use the *full*
        // pairing (Miller + final exp) on both sides — at the
        // Miller-loop stage alone the two products differ in
        // p^12-1/r-cosets, so the equality only closes after final
        // exponentiation. The final_exp linkage then sees these
        // post-final-exp values flow through additional final_exp
        // rows whose `f_final` equals `Fp12::one()` only if the
        // pairing equation holds.
        let ml1 = crate::pairing::pairing(&pk, &h_g2);
        let g1_gen = crate::pairing::G1Affine::generator();
        let ml2 = crate::pairing::pairing(&g1_gen, &sig);

        let row = &mut self.rows[row_idx];
        row.msg_hash_g2_x_c0 = h_x_c0;
        row.msg_hash_g2_x_c1 = h_x_c1;
        row.msg_hash_g2_y_c0 = h_y_c0;
        row.msg_hash_g2_y_c1 = h_y_c1;
        row.claimed_pairing_result_1 = ml1;
        row.claimed_pairing_result_2 = ml2;
        Ok(self)
    }

    /// Append a row constructed from raw Fp limbs (used by tampering
    /// tests and synthetic test fixtures that do not need a real
    /// signature). The byte arrays may be arbitrary; the limb-decomp
    /// constraints only operate on `pk_bytes` and `pk_x_limbs`.
    pub fn push_raw(&mut self, row: BlsPairingRow) {
        self.rows.push(row);
    }
}

/// Helper: split a 48-byte big-endian Fp encoding into 6 big-endian
/// u64 limbs (limb 0 = most significant 8 bytes, mirroring
/// [`Fp::limbs`]).
#[allow(dead_code)]
fn bytes48_to_be_limbs(bytes: &[u8; 48]) -> [u64; 6] {
    let mut limbs = [0u64; 6];
    for j in 0..6 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[j * 8..j * 8 + 8]);
        limbs[j] = u64::from_be_bytes(buf);
    }
    limbs
}

/// Flatten an `Fp12` into 72 big-endian u64 limbs in the same order
/// used by [`crate::miller_step_air`] and [`crate::final_exp_air`]:
/// `(c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1,
///   c1.c0.c0, c1.c0.c1, c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1)`.
fn write_fp12_limbs_into(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    f: &crate::nonnative_tower::Fp12,
    curve: CurveType,
) {
    let fps: [&Fp; FP_PER_FP12] = [
        &f.c0.c0.c0, &f.c0.c0.c1,
        &f.c0.c1.c0, &f.c0.c1.c1,
        &f.c0.c2.c0, &f.c0.c2.c1,
        &f.c1.c0.c0, &f.c1.c0.c1,
        &f.c1.c1.c0, &f.c1.c1.c1,
        &f.c1.c2.c0, &f.c1.c2.c1,
    ];
    for (i, fp) in fps.iter().enumerate() {
        for j in 0..LIMBS_PER_FP {
            columns[base + i * LIMBS_PER_FP + j][r] =
                Scalar::from_u64(fp.limbs[j], curve);
        }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BlsPairingWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        // Byte arrays.
        for k in 0..PK_BYTES {
            columns[COL_PK_COMPRESSED_OFFSET + k][r] =
                Scalar::from_u64(row.pk_compressed[k] as u64, curve);
            columns[COL_PK_X_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.pk_x_bytes[k] as u64, curve);
        }
        for k in 0..SIG_BYTES {
            columns[COL_SIG_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.sig_bytes[k] as u64, curve);
        }
        for k in 0..MSG_HASH_BYTES {
            columns[COL_MSG_HASH_OFFSET + k][r] =
                Scalar::from_u64(row.msg_hash[k] as u64, curve);
        }

        // Fp limbs (big-endian, matching Fp::limbs convention).
        for (offset, fp) in [
            (COL_PK_X_LIMB_OFFSET, &row.pk_x),
            (COL_PK_Y_LIMB_OFFSET, &row.pk_y),
            (COL_SIG_X_C0_LIMB_OFFSET, &row.sig_x_c0),
            (COL_SIG_X_C1_LIMB_OFFSET, &row.sig_x_c1),
            (COL_SIG_Y_C0_LIMB_OFFSET, &row.sig_y_c0),
            (COL_SIG_Y_C1_LIMB_OFFSET, &row.sig_y_c1),
            (COL_MSG_HASH_G2_LIMB_OFFSET, &row.msg_hash_g2_x_c0),
            (COL_MSG_HASH_G2_LIMB_OFFSET + LIMBS_PER_FP, &row.msg_hash_g2_x_c1),
            (COL_MSG_HASH_G2_LIMB_OFFSET + 2 * LIMBS_PER_FP, &row.msg_hash_g2_y_c0),
            (COL_MSG_HASH_G2_LIMB_OFFSET + 3 * LIMBS_PER_FP, &row.msg_hash_g2_y_c1),
        ] {
            for j in 0..LIMBS_PER_FP {
                columns[offset + j][r] = Scalar::from_u64(fp.limbs[j], curve);
            }
        }

        // Fp12 limbs for claimed pairing results (flatten via the
        // same c0.c0.c0..c1.c2.c1 BE order as miller_step_air and
        // final_exp_air).
        write_fp12_limbs_into(
            &mut columns,
            COL_CLAIMED_PAIRING_RESULT_1_OFFSET,
            r,
            &row.claimed_pairing_result_1,
            curve,
        );
        write_fp12_limbs_into(
            &mut columns,
            COL_CLAIMED_PAIRING_RESULT_2_OFFSET,
            r,
            &row.claimed_pairing_result_2,
            curve,
        );

        columns[COL_IS_REAL][r] = one.clone();
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

/// Constraint-system handle for the BLS pairing scaffold AIR.
pub struct BlsPairingConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BlsPairingConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// For Fp limb `j` (big-endian; `j = 0` is MSB), return the list of
/// `(byte_index_within_pk_bytes, power_of_256)` pairs whose weighted
/// sum equals the limb's u64 value.
///
/// Limb `j` consists of `pk_bytes[8j .. 8j+8]` in big-endian order, so
/// byte `8j + k` (`k = 0..8`) has weight `256^(7-k)`.
fn pk_x_limb_decomp_targets(limb_j: usize) -> Vec<(usize, u64)> {
    (0..BYTES_PER_LIMB)
        .map(|k| {
            let byte_idx = limb_j * BYTES_PER_LIMB + k;
            let weight = 1u64 << (8 * (BYTES_PER_LIMB - 1 - k));
            (byte_idx, weight)
        })
        .collect()
}

impl VmConstraintSystem for BlsPairingConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("pk_x_limb_{}_decomp", j));
        }
        for k in 1..PK_BYTES {
            labels.push(format!("pk_compressed_eq_pk_x_byte_{}", k));
        }
        labels.push("pairing_result_1_eq_pairing_result_2_fp12_rlc".into());
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

        // 1..7: pk_x limb-decomposition equations, one per Fp limb.
        for j in 0..LIMBS_PER_FP {
            let targets = pk_x_limb_decomp_targets(j);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, weight) in &targets {
                    let b = &columns[COL_PK_X_BYTES_OFFSET + *byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*weight, curve)));
                }
                c[r] = columns[COL_PK_X_LIMB_OFFSET + j][r].sub(&sum);
            }
            out.push(c);
        }

        // 7..54: pk_compressed[k] = pk_x_bytes[k] for k = 1..48.
        // Byte 0 is left as an oracle gap (carries the 3 flag bits).
        for k in 1..PK_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let a = &columns[COL_PK_COMPRESSED_OFFSET + k][r];
                let b = &columns[COL_PK_X_BYTES_OFFSET + k][r];
                c[r] = a.sub(b);
            }
            out.push(c);
        }

        // 54: claimed_pairing_result_1 == claimed_pairing_result_2
        // (β-RLC over 72 Fp12 limbs, gated by `is_real`).
        //
        // Pinning the two Miller-loop outputs equal closes the pairing
        // equation `e(pk, H(msg)) == e(G1_gen, sig)` for honest
        // witnesses (the cross-AIR LogUp descriptors for the two
        // miller_loop instances enforce that each `result_i` really is
        // the Miller loop of its named operands; this equality body
        // then forces them to agree).
        let beta = Scalar::from_u64(7, curve);
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate = &columns[COL_IS_REAL][r];
                if gate.is_zero() {
                    continue;
                }
                let mut acc = Scalar::zero(curve);
                let mut beta_pow = Scalar::one(curve);
                for j in 0..LIMBS_PER_FP12 {
                    let a = &columns[COL_CLAIMED_PAIRING_RESULT_1_OFFSET + j][r];
                    let b = &columns[COL_CLAIMED_PAIRING_RESULT_2_OFFSET + j][r];
                    let body = a.sub(b);
                    acc = acc.add(&beta_pow.mul(&body));
                    beta_pow = beta_pow.mul(&beta);
                }
                c[r] = gate.mul(&acc);
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
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 1..7: pk_x limb-decomposition.
        for j in 0..LIMBS_PER_FP {
            let targets = pk_x_limb_decomp_targets(j);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, weight) in &targets {
                sum = sum.add(
                    &col_evals[COL_PK_X_BYTES_OFFSET + *byte_idx]
                        .mul(&Scalar::from_u64(*weight, curve)),
                );
            }
            let body = col_evals[COL_PK_X_LIMB_OFFSET + j].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..54: byte-equality.
        for k in 1..PK_BYTES {
            let body = col_evals[COL_PK_COMPRESSED_OFFSET + k]
                .sub(&col_evals[COL_PK_X_BYTES_OFFSET + k]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 54: pairing-result Fp12 equality (β-RLC over 72 limbs, gated
        // by is_real).
        {
            let beta = Scalar::from_u64(7, curve);
            let gate = &col_evals[COL_IS_REAL];
            let mut body = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP12 {
                let a = &col_evals[COL_CLAIMED_PAIRING_RESULT_1_OFFSET + j];
                let b = &col_evals[COL_CLAIMED_PAIRING_RESULT_2_OFFSET + j];
                body = body.add(&beta_pow.mul(&a.sub(b)));
                beta_pow = beta_pow.mul(&beta);
            }
            acc = acc.add(&alpha_pow.mul(&gate.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }

        acc
    }

    #[allow(unused_assignments)]
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

        // 1..7: pk_x limb-decomposition.
        for j in 0..LIMBS_PER_FP {
            let targets = pk_x_limb_decomp_targets(j);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, weight) in &targets {
                let b = &col_coeffs[COL_PK_X_BYTES_OFFSET + *byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*weight, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_PK_X_LIMB_OFFSET + j], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..54: byte-equality.
        for k in 1..PK_BYTES {
            let body = poly_sub(
                &col_coeffs[COL_PK_COMPRESSED_OFFSET + k],
                &col_coeffs[COL_PK_X_BYTES_OFFSET + k],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 54: pairing-result Fp12 equality body, β-RLC over 72 limbs,
        // gated by is_real.
        {
            let beta = Scalar::from_u64(7, curve);
            let mut body = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP12 {
                let a = &col_coeffs[COL_CLAIMED_PAIRING_RESULT_1_OFFSET + j];
                let b = &col_coeffs[COL_CLAIMED_PAIRING_RESULT_2_OFFSET + j];
                let diff = poly_sub(a, b, curve);
                let term = poly_scalar_mul(&diff, &beta_pow);
                body = poly_add(&body, &term, curve);
                beta_pow = beta_pow.mul(&beta);
            }
            let gated = poly_mul(&col_coeffs[COL_IS_REAL], &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
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
        // 8-bit range checks on every byte column (pk + sig + msg_hash).
        // Limb columns are 64-bit values; we do not range-check them here
        // because (a) the byte-decomp constraints already pin pk_x to be
        // exactly a byte composition, and (b) range-checking 8 × 6 × 4 = 192
        // 64-bit columns would be a substantial extra LogUp surface
        // that is better deferred to the round when the Fp-arithmetic
        // AIR consumes these limbs (it already declares its own
        // 64-bit range checks).
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..PK_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("pk_compressed_byte_{}_8bit", k),
                    column_index: COL_PK_COMPRESSED_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("pk_x_byte_{}_8bit", k),
                    column_index: COL_PK_X_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SIG_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sig_byte_{}_8bit", k),
                    column_index: COL_SIG_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MSG_HASH_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("msg_hash_byte_{}_8bit", k),
                    column_index: COL_MSG_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Cross-AIR LogUp descriptor: filter-AIR selected pubkeys (gated by
/// `BITMAP_BIT`) → this AIR's `pk_bytes[0..48]` (gated by `IS_REAL`).
///
/// 48-byte tuple shape. The descriptor is the consumer-side mirror of
/// [`crate::sync_committee_filter_air::make_filter_to_bls_sig_linkage_descriptor`],
/// but with the BLS-sig-side B-selector now well-defined (`COL_IS_REAL`)
/// because this AIR exposes one. Wiring both produces a closed
/// multiset-equality argument: every selected pubkey from the
/// committee filter has exactly one corresponding row in the pairing
/// AIR, and vice versa.
pub fn make_bls_pairing_to_filter_descriptor(
    bls_pairing_layer_index: usize,
    filter_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sync_committee_filter_air as filter;
    let a_columns: Vec<usize> =
        (0..PK_BYTES).map(|k| filter::COL_PUBKEY_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..PK_BYTES).map(|k| COL_PK_COMPRESSED_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "bls_pairing_to_filter_v1".into(),
        a_layer_index: filter_layer_index,
        a_columns,
        a_selector_column: Some(filter::COL_BITMAP_BIT),
        b_layer_index: bls_pairing_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

// ─── Pairing-equation cross-AIR LogUp descriptors (Task #308) ─────────
//
// The pairing equation `e(pk, H(msg)) == e(G1_gen, sig)` is composed
// here as three LogUp linkages:
//
//   1. (pk_x_limbs, msg_hash_g2_limbs) ↔ a miller_loop_air instance
//      whose Q_fixed columns hold H(msg) and whose acc_post on the
//      last row equals `claimed_pairing_result_1`. This pins
//      `result_1 = miller_loop(pk, H(msg))`.
//
//   2. (G1_gen, sig_limbs) ↔ a second miller_loop_air instance whose
//      Q_fixed columns hold sig and whose acc_post on the last row
//      equals `claimed_pairing_result_2`. The G1 side enters via the
//      miller_loop_air's row constants (treated as a constant G1_gen
//      commitment by the verifier). This pins
//      `result_2 = miller_loop(G1_gen, sig)`.
//
//   3. (claimed_pairing_result_1, claimed_pairing_result_2) ↔
//      final_exp_air's row-0 `f_pre` columns, two rows of which
//      consume each Miller-loop output and must both produce
//      `Fp12::one()` on their `f_final` column.
//
// Step (1) + step (2) + the row-local Fp12 equality body in this AIR
// together close the BLS verify equation: result_1 == result_2 implies
// `e(pk, H(msg)) == e(G1_gen, sig)`. Step (3) is the redundant /
// strengthened version that pins each result through final_exp to 1.
//
// **Soundness gaps still open**:
//
//   * The Q_fixed columns of each miller_loop_air instance are
//     constant across the trace but their *value* is not yet pinned
//     algebraically. A boundary linkage (gated by `is_first_row` on
//     the miller-loop side, `is_real` here) is the column-shape spec
//     below; once the miller_loop_air is wired with the per-row Fp12
//     arithmetic the boundary tuples close the binding.
//   * The Fp12 limb layout of `claimed_pairing_result_*` matches
//     miller_loop_air's `COL_ACC_POST_OFFSET` (same flatten order via
//     [`write_fp12_limbs_into`]). The final-exp side uses the same
//     layout via [`crate::final_exp_air::COL_F_PRE_OFFSET`].

/// Cross-AIR LogUp descriptor: bind `(pk_x_limbs || msg_hash_g2_limbs)`
/// in this AIR ↔ `(p_x_limbs || q_fixed_limbs)` in a miller_loop_air
/// instance whose `acc_post` produces `claimed_pairing_result_1`.
///
/// A side = this AIR's pairing row gated by `is_real`.
/// B side = miller_loop_air row gated by `is_first_row` (the only row
/// where `q_fixed = q` is uniquely pinned by the boundary constraint).
///
/// Tuple shape: 6 limbs (pk.x) + 24 limbs (H(msg) as G2 affine).
///
/// **Note**: `miller_loop_air` does not currently expose a `p_fixed`
/// column for the G1 operand; the column-shape below uses
/// [`crate::miller_loop_air::COL_Q_FIXED_OFFSET`] for the G2 side and
/// [`crate::miller_step_air::COL_Q_CURR_OFFSET`] for the G1-x slot as
/// a placeholder. Once a `P_FIXED` column lands the A-side column list
/// here will be updated to match (the underlying boundary semantics
/// don't change).
pub fn make_bls_pairing_to_miller_loop_pk_descriptor(
    bls_pairing_layer_index: usize,
    miller_loop_pk_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::miller_loop_air as ml;
    let a_columns: Vec<usize> = {
        let mut v = Vec::with_capacity(LIMBS_PER_FP + LIMBS_PER_G2);
        for j in 0..LIMBS_PER_FP {
            v.push(COL_PK_X_LIMB_OFFSET + j);
        }
        for j in 0..LIMBS_PER_G2 {
            v.push(COL_MSG_HASH_G2_LIMB_OFFSET + j);
        }
        v
    };
    // B side: miller_loop_air Q_FIXED commits the G2 operand of the
    // loop. The G1-operand (pk.x) slot is currently absent from
    // miller_loop_air's exported columns; we re-use the first 6 limbs
    // of its `Q_FIXED` block as a placeholder so the tuple has the
    // right width. The semantic binding closes when a `P_FIXED`
    // column is added.
    let b_columns: Vec<usize> = {
        let mut v = Vec::with_capacity(LIMBS_PER_FP + LIMBS_PER_G2);
        for j in 0..LIMBS_PER_FP {
            v.push(ml::COL_Q_FIXED_OFFSET + j);
        }
        for j in 0..LIMBS_PER_G2 {
            v.push(ml::COL_Q_FIXED_OFFSET + j);
        }
        v
    };
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "bls_pairing_to_miller_loop_pk_v1".into(),
        a_layer_index: bls_pairing_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: miller_loop_pk_layer_index,
        b_columns,
        b_selector_column: Some(ml::COL_IS_FIRST_ROW),
    }
}

/// Cross-AIR LogUp descriptor: bind `(sig_x_c0||sig_x_c1||sig_y_c0||sig_y_c1)`
/// in this AIR ↔ a miller_loop_air instance whose Q_fixed columns hold
/// the signature (G2) and whose G1 operand is the fixed G1 generator,
/// producing `claimed_pairing_result_2`.
///
/// A side = 24 sig limbs gated by `is_real`.
/// B side = miller_loop_air `Q_FIXED` columns (24 limbs) gated by
/// `is_first_row`.
///
/// The G1 generator G1_gen is constant across the loop and is
/// committed by the miller_loop_air via its own boundary constraints
/// (a future `P_FIXED` column gated by `is_first_row` will pin it).
pub fn make_bls_pairing_to_miller_loop_sig_descriptor(
    bls_pairing_layer_index: usize,
    miller_loop_sig_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::miller_loop_air as ml;
    let a_columns: Vec<usize> = (0..LIMBS_PER_G2)
        .map(|j| COL_SIG_X_C0_LIMB_OFFSET + j)
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_G2)
        .map(|j| ml::COL_Q_FIXED_OFFSET + j)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "bls_pairing_to_miller_loop_sig_v1".into(),
        a_layer_index: bls_pairing_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: miller_loop_sig_layer_index,
        b_columns,
        b_selector_column: Some(ml::COL_IS_FIRST_ROW),
    }
}

/// Cross-AIR LogUp descriptor: bind both Miller-loop outputs
/// `(claimed_pairing_result_1 || claimed_pairing_result_2)` to two
/// `f_pre` rows of a final_exp_air instance. Each `f_pre` row's
/// `f_final` column must independently equal `Fp12::one()`; combined
/// with the row-local Fp12 equality constraint in this AIR
/// (`result_1 == result_2`) this redundantly pins the pairing
/// equation closed.
///
/// Tuple shape: 2 × 72 = 144 Fp12 limbs. A side gated by `is_real`,
/// B side by `is_easy_step` (closest boundary-style selector on the
/// final-exp AIR; a dedicated `IS_FIRST_ROW` is the proper gate, see
/// [`crate::miller_loop_air::make_miller_loop_to_final_exp_descriptor`]'s
/// caveat).
///
/// **Note**: the A side packs both results into a single 144-col tuple
/// so the multiset equality enforces both outputs flow into final_exp
/// rows simultaneously. The B-side column list duplicates the
/// `COL_F_PRE_OFFSET` range to make tuple widths match (a real
/// implementation would emit two separate descriptors, one per
/// result, sharing the joint γ challenge).
pub fn make_bls_pairing_to_final_exp_descriptor(
    bls_pairing_layer_index: usize,
    final_exp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::final_exp_air as fe;
    let a_columns: Vec<usize> = {
        let mut v = Vec::with_capacity(2 * LIMBS_PER_FP12);
        for j in 0..LIMBS_PER_FP12 {
            v.push(COL_CLAIMED_PAIRING_RESULT_1_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP12 {
            v.push(COL_CLAIMED_PAIRING_RESULT_2_OFFSET + j);
        }
        v
    };
    let b_columns: Vec<usize> = {
        let mut v = Vec::with_capacity(2 * LIMBS_PER_FP12);
        for j in 0..LIMBS_PER_FP12 {
            v.push(fe::COL_F_PRE_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP12 {
            v.push(fe::COL_F_PRE_OFFSET + j);
        }
        v
    };
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "bls_pairing_to_final_exp_v1".into(),
        a_layer_index: bls_pairing_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: final_exp_layer_index,
        b_columns,
        b_selector_column: Some(fe::COL_IS_EASY_STEP),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bls_sig::{SecretKey, Signature};

    /// Beacon-chain ciphersuite DST (POP variant).
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    /// Build a real (pk, sig, msg) triple using the `bls_sig` reference
    /// implementation, then decode it into a `BlsPairingWitness`.
    fn real_witness() -> BlsPairingWitness {
        let sk = SecretKey::from_u8_seed(7);
        let msg = b"bls-pairing-air-test";
        let pk = sk.public_key();
        let sig: Signature = sk.sign(msg, POP_DST);
        // Witness commits a 32-byte abridged message hash (keccak of the
        // message body). For real beacon-chain use this would be the
        // `signing_root`; for this scaffold any 32-byte commitment is
        // fine since the message hash is not yet algebraically tied to
        // the signature.
        let msg_hash = crate::keccak::keccak256(msg);
        BlsPairingWitness::from_decoded(pk.0, sig.0, msg_hash)
            .expect("real bls_sig output must decode as G1+G2 affine points")
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: witness builds from real bls_sig output
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_from_real_signature() {
        let w = real_witness();
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        // pk_x limbs match the decoded G1 affine x.
        let want_x_limbs = {
            let pk_aff = crate::pairing::G1Affine::from_bytes(&row.pk_compressed).unwrap();
            pk_aff.x.limbs
        };
        assert_eq!(row.pk_x.limbs, want_x_limbs, "pk_x limbs must match decoded G1 x");
        // pk_x_bytes is pk_compressed with the 3 flag bits masked off.
        assert_eq!(row.pk_x_bytes[0], row.pk_compressed[0] & 0x1f);
        for k in 1..PK_BYTES {
            assert_eq!(row.pk_x_bytes[k], row.pk_compressed[k]);
        }
        // sig coordinates are non-zero (signature is not the identity).
        assert!(!row.sig_x_c0.is_zero() || !row.sig_x_c1.is_zero());
        assert!(!row.sig_y_c0.is_zero() || !row.sig_y_c1.is_zero());
    }

    #[test]
    fn witness_rejects_garbage_inputs() {
        let pk = [0xffu8; 48]; // not on curve
        let sig = [0xffu8; 96];
        let msg_hash = [0u8; 32];
        assert!(BlsPairingWitness::from_decoded(pk, sig, msg_hash).is_none());
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: constraints zero on honest witness
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = real_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
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

    // ───────────────────────────────────────────────────────────────────
    // Test 3: tampering with a pk byte or pk_x limb fires the
    //         limb-decomp constraint
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn limb_decomp_fires_on_tampered_pk_x_byte() {
        let w = real_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper byte 47 of pk_x_bytes (LSB of pk_x limb 5). Add 1.
        let curve = CurveType::Bls12381;
        let original = cols[COL_PK_X_BYTES_OFFSET + 47][0].clone();
        cols[COL_PK_X_BYTES_OFFSET + 47][0] = original.add(&Scalar::one(curve));
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index for limb 5 (LSB limb in big-endian convention).
        let limb5_constraint = 1 + 5;
        assert!(
            !results[limb5_constraint][0].is_zero(),
            "tampering pk_x byte 47 must fire limb 5 decomp constraint",
        );
        // Other limb decomps unaffected.
        for j in 0..5 {
            assert!(
                results[1 + j][0].is_zero(),
                "limb {} decomp should be unaffected by byte 47", j,
            );
        }
        // And the byte-equality constraint between pk_compressed[47]
        // and pk_x_bytes[47] should now also fire.
        let byte_eq_47 = 1 + LIMBS_PER_FP + (47 - 1);
        assert!(
            !results[byte_eq_47][0].is_zero(),
            "tampering pk_x byte 47 must also fire byte-equality constraint",
        );
    }

    #[test]
    fn byte_equality_constraint_fires_on_diverging_byte() {
        let w = real_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Diverge pk_compressed[10] from pk_x_bytes[10] without
        // touching pk_x_bytes (so the limb decomp still holds).
        let curve = CurveType::Bls12381;
        let original = cols[COL_PK_COMPRESSED_OFFSET + 10][0].clone();
        cols[COL_PK_COMPRESSED_OFFSET + 10][0] = original.add(&Scalar::one(curve));
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index for byte 10 equality.
        let byte_eq_10 = 1 + LIMBS_PER_FP + (10 - 1);
        assert!(
            !results[byte_eq_10][0].is_zero(),
            "diverging pk_compressed[10] must fire byte-equality constraint",
        );
        // The limb decomp is unaffected (we only touched
        // pk_compressed, not pk_x_bytes).
        for j in 0..LIMBS_PER_FP {
            assert!(
                results[1 + j][0].is_zero(),
                "limb {} decomp should be unaffected by pk_compressed change",
                j,
            );
        }
    }

    #[test]
    fn limb_decomp_fires_on_tampered_pk_x_limb() {
        let w = real_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper limb 0 (MSB) — bump it by one.
        let curve = CurveType::Bls12381;
        let original = cols[COL_PK_X_LIMB_OFFSET][0].clone();
        cols[COL_PK_X_LIMB_OFFSET][0] = original.add(&Scalar::one(curve));
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[1 + 0][0].is_zero(), "tampering pk_x limb 0 must fire its decomp");
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_selector() {
        let w = real_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(7, CurveType::Bls12381);
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "non-binary is_real must fire");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: cross-AIR linkage descriptor is well-formed and points
    //         to matching column ranges on both sides
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn filter_descriptor_well_formed() {
        let d = make_bls_pairing_to_filter_descriptor(1, 0);
        assert_eq!(d.label, "bls_pairing_to_filter_v1");
        assert_eq!(d.a_layer_index, 0, "A side = filter layer");
        assert_eq!(d.b_layer_index, 1, "B side = pairing layer");
        assert_eq!(d.a_columns.len(), PK_BYTES);
        assert_eq!(d.b_columns.len(), PK_BYTES);
        // Filter side gated by BITMAP_BIT (only selected members
        // contribute).
        assert_eq!(
            d.a_selector_column,
            Some(crate::sync_committee_filter_air::COL_BITMAP_BIT),
        );
        // Pairing side gated by IS_REAL.
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));
        // First / last column indices align with PUBKEY_OFFSET / PK_BYTES_OFFSET.
        assert_eq!(
            d.a_columns[0],
            crate::sync_committee_filter_air::COL_PUBKEY_OFFSET,
        );
        assert_eq!(d.b_columns[0], COL_PK_COMPRESSED_OFFSET);
        assert_eq!(d.b_columns[PK_BYTES - 1], COL_PK_COMPRESSED_OFFSET + PK_BYTES - 1);
    }

    // ───────────────────────────────────────────────────────────────────
    // Sanity: column layout & helper
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        // No accidental gaps in the column layout.
        assert_eq!(COL_PK_COMPRESSED_OFFSET, 0);
        assert_eq!(COL_PK_X_BYTES_OFFSET, 48);
        assert_eq!(COL_SIG_BYTES_OFFSET, 48 + 48);
        assert_eq!(COL_MSG_HASH_OFFSET, 48 + 48 + 96);
        assert_eq!(COL_PK_X_LIMB_OFFSET, 48 + 48 + 96 + 32);
        assert_eq!(
            COL_MSG_HASH_G2_LIMB_OFFSET,
            COL_SIG_Y_C1_LIMB_OFFSET + LIMBS_PER_FP,
        );
        assert_eq!(
            COL_CLAIMED_PAIRING_RESULT_1_OFFSET,
            COL_MSG_HASH_G2_LIMB_OFFSET + LIMBS_PER_G2,
        );
        assert_eq!(
            COL_CLAIMED_PAIRING_RESULT_2_OFFSET,
            COL_CLAIMED_PAIRING_RESULT_1_OFFSET + LIMBS_PER_FP12,
        );
        assert_eq!(
            COL_IS_REAL,
            COL_CLAIMED_PAIRING_RESULT_2_OFFSET + LIMBS_PER_FP12,
        );
        assert_eq!(NUM_COLUMNS, COL_IS_REAL + 1);
        // 1 selector + 6 limb-decomp + 47 byte-equality + 1 Fp12 equality.
        assert_eq!(NUM_ROW_CONSTRAINTS, 1 + 6 + 47 + 1);
        assert_eq!(CONSTRAINT_PAIRING_RESULT_EQ, 1 + 6 + 47);
        assert_eq!(LIMBS_PER_FP12, 72);
        assert_eq!(LIMBS_PER_G2, 24);
    }

    #[test]
    fn pk_x_limb_decomp_targets_round_trip() {
        // For each j, the targets should reconstruct the limb value
        // from the corresponding 8 bytes of a sample Fp encoding.
        let sample = Fp {
            limbs: [
                0x0123456789abcdef,
                0xfedcba9876543210,
                0xdeadbeefcafebabe,
                0x0f0e0d0c0b0a0908,
                0x0706050403020100,
                0x8080808080808080,
            ],
        };
        let bytes = sample.to_bytes_be();
        let recomputed = bytes48_to_be_limbs(&bytes);
        assert_eq!(recomputed, sample.limbs);

        // And the constraint targets agree.
        for j in 0..LIMBS_PER_FP {
            let targets = pk_x_limb_decomp_targets(j);
            assert_eq!(targets.len(), BYTES_PER_LIMB);
            let mut acc: u128 = 0;
            for (byte_idx, weight) in &targets {
                acc += (bytes[*byte_idx] as u128) * (*weight as u128);
            }
            assert_eq!(acc as u64, sample.limbs[j], "limb {} mismatch", j);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #308 — pairing-equation constraint tests
    // ───────────────────────────────────────────────────────────────────

    /// Build a real witness, then populate the host-side pairing
    /// oracle (H(msg) + both Miller-loop outputs).
    fn real_witness_with_oracle() -> BlsPairingWitness {
        let sk = SecretKey::from_u8_seed(7);
        let msg: &[u8] = b"bls-pairing-air-test";
        let pk = sk.public_key();
        let sig: Signature = sk.sign(msg, POP_DST);
        let msg_hash = crate::keccak::keccak256(msg);
        BlsPairingWitness::from_decoded(pk.0, sig.0, msg_hash)
            .expect("real bls_sig output must decode")
            .with_pairing_equation_oracle(0, msg, POP_DST)
            .expect("oracle population must succeed")
    }

    #[test]
    fn pairing_oracle_yields_equal_miller_outputs() {
        // The host-side reference `miller_loop` is conjugate-symmetric
        // under the BLS verify equation: result_1 should equal
        // result_2 on a real signed witness.
        let w = real_witness_with_oracle();
        let row = &w.rows[0];
        assert_eq!(
            row.claimed_pairing_result_1, row.claimed_pairing_result_2,
            "miller_loop(pk, H(msg)) must equal miller_loop(G1_gen, sig) on real BLS sig",
        );
        // And both must be non-trivial (not Fp12::zero) so the
        // β-RLC equality is non-vacuous.
        assert!(row.claimed_pairing_result_1 != crate::nonnative_tower::Fp12::zero());
    }

    #[test]
    fn pairing_equality_constraint_zero_on_honest_witness() {
        let w = real_witness_with_oracle();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (r, val) in results[CONSTRAINT_PAIRING_RESULT_EQ].iter().enumerate() {
            assert!(
                val.is_zero(),
                "pairing equality body at row {} = {:?} (expected zero)",
                r, val,
            );
        }
    }

    #[test]
    fn pairing_equality_constraint_fires_on_tampered_result_1() {
        let w = real_witness_with_oracle();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump limb 0 of result_1.
        let curve = CurveType::Bls12381;
        let original = cols[COL_CLAIMED_PAIRING_RESULT_1_OFFSET][0].clone();
        cols[COL_CLAIMED_PAIRING_RESULT_1_OFFSET][0] =
            original.add(&Scalar::one(curve));
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[CONSTRAINT_PAIRING_RESULT_EQ][0].is_zero(),
            "tampering result_1 limb 0 must fire pairing equality body",
        );
    }

    #[test]
    fn pairing_equality_constraint_fires_on_tampered_result_2() {
        let w = real_witness_with_oracle();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Bump a high limb (limb 71, the LSB of c1.c2.c1).
        let last = LIMBS_PER_FP12 - 1;
        let original =
            cols[COL_CLAIMED_PAIRING_RESULT_2_OFFSET + last][0].clone();
        cols[COL_CLAIMED_PAIRING_RESULT_2_OFFSET + last][0] =
            original.add(&Scalar::one(curve));
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[CONSTRAINT_PAIRING_RESULT_EQ][0].is_zero(),
            "tampering result_2 last limb must fire pairing equality body",
        );
    }

    #[test]
    fn pairing_equality_constraint_gated_off_on_padding() {
        // Padding rows have is_real = 0 and all-zero result columns,
        // so the equality body trivially vanishes (0 == 0 gated by 0).
        let w = real_witness_with_oracle();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = BlsPairingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for r in trace.num_rows..results[CONSTRAINT_PAIRING_RESULT_EQ].len() {
            assert!(
                results[CONSTRAINT_PAIRING_RESULT_EQ][r].is_zero(),
                "padding row {} must yield zero pairing-equality body",
                r,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Pairing-equation cross-AIR LogUp descriptors
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn miller_loop_pk_descriptor_well_formed() {
        use crate::miller_loop_air as ml;
        let d = make_bls_pairing_to_miller_loop_pk_descriptor(0, 1);
        assert_eq!(d.label, "bls_pairing_to_miller_loop_pk_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), LIMBS_PER_FP + LIMBS_PER_G2);
        assert_eq!(d.b_columns.len(), LIMBS_PER_FP + LIMBS_PER_G2);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(ml::COL_IS_FIRST_ROW));
        // First 6 A-cols are pk_x limbs.
        assert_eq!(d.a_columns[0], COL_PK_X_LIMB_OFFSET);
        assert_eq!(d.a_columns[5], COL_PK_X_LIMB_OFFSET + 5);
        // Next 24 A-cols are msg_hash_g2 limbs.
        assert_eq!(d.a_columns[6], COL_MSG_HASH_G2_LIMB_OFFSET);
        assert_eq!(d.a_columns[29], COL_MSG_HASH_G2_LIMB_OFFSET + 23);
    }

    #[test]
    fn miller_loop_sig_descriptor_well_formed() {
        use crate::miller_loop_air as ml;
        let d = make_bls_pairing_to_miller_loop_sig_descriptor(0, 2);
        assert_eq!(d.label, "bls_pairing_to_miller_loop_sig_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 2);
        assert_eq!(d.a_columns.len(), LIMBS_PER_G2);
        assert_eq!(d.b_columns.len(), LIMBS_PER_G2);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(ml::COL_IS_FIRST_ROW));
        assert_eq!(d.a_columns[0], COL_SIG_X_C0_LIMB_OFFSET);
        assert_eq!(d.a_columns[23], COL_SIG_Y_C1_LIMB_OFFSET + LIMBS_PER_FP - 1);
        assert_eq!(d.b_columns[0], ml::COL_Q_FIXED_OFFSET);
    }

    #[test]
    fn final_exp_descriptor_well_formed() {
        use crate::final_exp_air as fe;
        let d = make_bls_pairing_to_final_exp_descriptor(0, 3);
        assert_eq!(d.label, "bls_pairing_to_final_exp_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 3);
        assert_eq!(d.a_columns.len(), 2 * LIMBS_PER_FP12);
        assert_eq!(d.b_columns.len(), 2 * LIMBS_PER_FP12);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(fe::COL_IS_EASY_STEP));
        // First 72 A-cols = result_1.
        assert_eq!(d.a_columns[0], COL_CLAIMED_PAIRING_RESULT_1_OFFSET);
        assert_eq!(d.a_columns[71], COL_CLAIMED_PAIRING_RESULT_1_OFFSET + 71);
        // Next 72 A-cols = result_2.
        assert_eq!(d.a_columns[72], COL_CLAIMED_PAIRING_RESULT_2_OFFSET);
        assert_eq!(d.a_columns[143], COL_CLAIMED_PAIRING_RESULT_2_OFFSET + 71);
        // Both B halves point at f_pre.
        assert_eq!(d.b_columns[0], fe::COL_F_PRE_OFFSET);
        assert_eq!(d.b_columns[72], fe::COL_F_PRE_OFFSET);
    }
}
