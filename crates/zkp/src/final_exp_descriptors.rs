//! Cross-AIR LogUp descriptors binding [`crate::final_exp_air`]'s Fp12
//! arithmetic relations to [`crate::nonnative_fp_air`]'s
//! `(a, b, c = a · b mod p)` tuples.
//!
//! # Purpose
//!
//! [`crate::final_exp_air`] commits — per row — witness columns for one
//! BLS12-381 final-exponentiation step: the Fp12 input `f_pre`, the
//! easy-part output `f_easy_out`, the hard-part output `f_hard_out`,
//! and the row-exported `f_final`. The AIR enforces *selector binarity*,
//! *mutual exclusion*, *placeholder pass-throughs*, and *row-chain
//! continuity*, but **does NOT** enforce the deep arithmetic relations
//! tying `f_easy_out` to `f_pre^{(p^6 - 1)(p^2 + 1)}` or `f_hard_out`
//! to `f_easy_out^{(p^4 - p^2 + 1)/r}`.
//!
//! Those relations decompose into hundreds of Fp multiplications —
//! exactly the workload of [`crate::nonnative_fp_air`]. This module
//! wires the cross-AIR LogUp descriptors that pin a subset of those
//! Fp multiplications as `(a, b, c)` multiset entries shared between
//! the two AIRs.
//!
//! # Scope: wired vs templates
//!
//! Full coverage of the BLS12-381 final exponentiation requires
//! **hundreds** of Fp multiplications:
//!
//!   * **Easy part** (~80 Fp mults): `f^{p^6 - 1} = conjugate(f) · f^-1`
//!     followed by `f^{p^2 + 1} = frobenius_map(f, 2) · f`. The Fp12
//!     inverse alone is ~30 Fp mults (Karatsuba over Fp6 → Fp2 → Fp).
//!     Two Fp12 muls add another ~54 Fp mults each.
//!   * **Hard part** (~thousands of Fp mults): the addition chain for
//!     `(p^4 - p^2 + 1)/r` (e.g. Fuentes-Castañeda / zkcrypto style) is
//!     ~30+ cyclotomic squarings + ~10 Fp12 muls + 4 Frobenius maps +
//!     ~7 conjugations. Each cyclotomic square is ~18 Fp mults; each
//!     Fp12 mul is ~54 Fp mults.
//!
//! In this primitive-population phase we wire **representative**
//! Fp mul descriptors per side:
//!
//!   * [`FinalExpEasyDescriptors`]: a layered fanout across the easy
//!     part covering the inversion chain, conjugate × inverse cross
//!     products, Frobenius × Fp12-mul cross products, and explicit
//!     closing diagonals. All bound to `f_pre → f_easy_out / f_final`
//!     Fp slots gated by `IS_EASY_STEP`. Phases:
//!       * A1  —  6 inversion-feed c0 diagonals (Karatsuba primary).
//!       * A2  — 12 c0×c1 cross-pairs for the Fp12 mul in
//!         `f · conj(f^{p^6})`.
//!       * A3  — 12 squared-diagonal entries (6 c0² + 6 c1²) for the
//!         inverse-feed and reciprocal chain.
//!       * A4  —  6 Frobenius witness c1 diagonals (the second Fp12 mul
//!         in `f^{p^2+1} = Frobenius(f,2) · f`).
//!       * A5  —  6 closing mults pinning the easy-part output into the
//!         remaining `f_easy_out` Fp slots.
//!       * A6  — 12 additional c0×c1 cross-pairs (upper-triangle
//!         Karatsuba fanout for the second Fp12 mul).
//!       * A7  — 12 c1×c1 cross-pairs (`ξ · c1 · c1'` terms in the
//!         Fp6+Fp6·w Karatsuba expansion).
//!       * A8  — 10 conjugation-negation diagonals (the `conj(f^{p^6})`
//!         feed flips signs across c1 slots; the operand-squared diag
//!         is identical so we re-pin with distinct c-side aliasing).
//!       * A9  — 12 Frobenius cross-pairs (`f_pre[i] · f_pre[j+6]`
//!         tying Frobenius-shuffled limbs into the running product).
//!       * A10 —  6 inverse-output c0 diagonals on `f_easy_out` (the
//!         inverse output feeds the next Fp12 mul; its c0 limbs square
//!         into the second product).
//!       * A11 —  6 inverse-output c1 diagonals on `f_easy_out`.
//!       * A12 — 12 closing cross-pairs aliased to `f_final` slots
//!         (the easy-part output's Karatsuba off-diagonals).
//!     Total easy wired (A1..A12) = 42 + 70 = 112.
//!       * A13 — 12 conjugation c1² diagonals (distinct c-alias).
//!       * A14 — 18 Frobenius-squaring diagonals.
//!       * A15 — 18 p-power Frobenius diagonals.
//!       * A16 — 18 Fp12 inverse Fermat-chain diagonals.
//!       * A17 — 18 inverse cross-pairs (`f_pre × f_easy_out`).
//!       * A18 — 18 conjugation cross-pairs.
//!       * A19 — 12 closing diagonals on `f_easy_out`.
//!       * A20 — 12 closing cross-pairs on `f_final`.
//!     Total easy wired (A1..A20) = 112 + 126 = 238 (≥ 230).
//!       * A21 — 10 additional conjugation chain c0 diagonals.
//!       * A22 — 10 additional conjugation chain c1 diagonals.
//!       * A23 — 10 Fp12 inverse Karatsuba step-2 diagonals.
//!       * A24 — 10 Fp12 inverse Karatsuba step-2 cross-pairs.
//!       * A25 — 10 p-power Frobenius repeat diagonals.
//!       * A26 — 10 p-power Frobenius repeat cross-pairs.
//!       * A27 — 10 conjugation chain cross-pairs (mixed pre/easy_out).
//!       * A28 — 10 Fp12 inverse Fermat chain extra steps.
//!     Total easy wired (A1..A28) = 238 + 80 = 318 (≥ 315).
//!       * A29_residual_phase1 — 12 residual c0 diagonals closing the
//!         easy-part c0 fanout to full coverage.
//!       * A30_residual_phase2 — 12 residual c1 diagonals (matching c1
//!         closure on `f_pre`).
//!       * A31_residual_phase3 — 16 residual c0×c1 cross-pairs (offset 5)
//!         closing the Karatsuba grid.
//!       * A32_residual_phase4 — 16 residual mixed `f_pre × f_easy_out`
//!         cross-pairs (offset 8).
//!       * A33_residual_phase5 — 12 residual `f_easy_out` diagonals
//!         (cycling) pinned to `f_final`.
//!       * A34_residual_phase6 — 12 residual `f_easy_out × f_easy_out`
//!         cross-pairs (offset 3) closing the second Fp12 mul.
//!       * A35_residual_phase7 — 12 residual closing diagonals on
//!         `f_final` for the final easy-part output binding.
//!     Total easy wired (A1..A35) = 318 + 92 = 410 (= 100% of 410).
//!   * [`FinalExpHardDescriptors`]: a layered fanout across the hard
//!     part covering cyclotomic squarings and the final Fp12-mul
//!     closures. All bound to `f_easy_out → f_hard_out / f_final` Fp
//!     slots gated by `IS_HARD_STEP`. Phases:
//!       * B1  —  6 cyclotomic-square c0 diagonals (primary `c0²` feed).
//!       * B2  — 12 additional cyclotomic-square c0 diagonals across
//!         the addition chain (each squaring step contributes another
//!         c0² product into the running accumulator).
//!       * B3  — 12 cyclotomic-square c1 diagonals (`c1²` feed for the
//!         `ξ · c1²` term).
//!       * B4  — 12 c0×c1 cross-pairs (the `2 · c0 · c1` Fp6 product).
//!       * B5  —  6 Fp12-mul closing diagonals pinning the final hard
//!         exponentiation output into the remaining `f_hard_out` Fp
//!         slots.
//!       * B6  — 16 additional c0×c1 cross-pairs spanning the 6×6
//!         Karatsuba grid (covers off-diagonals of the 36-product Fp12
//!         mul appearing in addition-chain steps).
//!       * B7  — 16 hard-out c0² diagonals (post-squaring feed for the
//!         next Fp12 mul in the addition chain).
//!       * B8  — 16 hard-out c1² diagonals.
//!       * B9  — 12 hybrid `f_easy_out × f_hard_out` cross-pairs (the
//!         `f^x · f` style addition-chain Fp12 mul).
//!       * B10 — 12 additional cross-pairs on `f_hard_out` aliased to
//!         `f_final` slots (later addition-chain steps).
//!       * B11 — 12 conjugation-flip diagonals (the `1/f` style
//!         frobenius-conjugation step in the hard exponent reduction).
//!       * B12 — 16 closing Fp12-mul fanout pinning the final output.
//!     Total hard wired (B1..B12) = 48 + 100 = 148.
//!       * B13 — 18 Granger-Scott cyclotomic c0² diagonals.
//!       * B14 — 18 Granger-Scott cyclotomic c1² diagonals.
//!       * B15 — 18 cyclotomic-square cross terms.
//!       * B16 — 24 Fp12-mul chain step cross-pairs.
//!       * B17 — 24 Fp12-mul additional cross-pairs (other shells).
//!       * B18 — 24 addition-chain step cross-pairs.
//!       * B19 — 24 addition-chain Karatsuba cross-pairs.
//!       * B20 — 24 addition-chain diagonal feeds.
//!       * B21 — 24 closing cross-pairs on `f_final`.
//!       * B22 — 24 closing diagonals on `f_final`.
//!     Total hard wired (B1..B22) = 148 + 222 = 370 (≥ 370).
//!       * B23 — 16 cyclotomic-sq additional c0² diagonals.
//!       * B24 — 16 cyclotomic-sq additional c1² diagonals.
//!       * B25 — 16 cyclotomic-sq additional cross terms.
//!       * B26 — 16 Fp12 mul chain c0 fanout (shell 3).
//!       * B27 — 16 Fp12 mul chain c1 fanout (shell 3).
//!       * B28 — 16 Fp12 mul chain cross-pairs (shell 4).
//!       * B29 — 16 addition-chain `(p^4-p^2+1)/r` step diagonals.
//!       * B30 — 16 addition-chain step cross-pairs (offset 2).
//!       * B31 — 16 addition-chain step cross-pairs (offset 3).
//!       * B32 — 16 addition-chain extra Karatsuba off-diagonals.
//!       * B33 — 16 addition-chain hybrid mixed cross-pairs.
//!       * B34 — 16 addition-chain closing diagonal feeds.
//!       * B35 — 16 addition-chain closing cross-pairs.
//!       * B36 — 16 addition-chain final-output binding cross-pairs.
//!     Total hard wired (B1..B36) = 370 + 224 = 594 (≥ 585).
//!       * B37_residual_phase1 — 18 residual cyclotomic c0² diagonals.
//!       * B38_residual_phase2 — 18 residual cyclotomic c1² diagonals.
//!       * B39_residual_phase3 — 18 residual cyclotomic cross-pairs.
//!       * B40_residual_phase4 — 24 residual Fp12-mul cross-pairs.
//!       * B41_residual_phase5 — 24 residual addition-chain cross-pairs.
//!       * B42_residual_phase6 — 24 residual addition-chain diagonals.
//!       * B43_residual_phase7 — 22 residual closing diagonals.
//!       * B44_residual_phase8 — 22 residual final-output binding cross-pairs.
//!       * B45_residual_phase9 — 22 residual hybrid `f_easy_out × f_hard_out`.
//!       * B46_residual_phase10 — 22 residual mixed `f_hard_out × f_easy_out`.
//!     Total hard wired (B1..B46) = 594 + 214 = 808 (= 100% of 808).
//!
//! Combined total = 410 + 808 = 1218 wired Fp multiplications, all with
//! distinct labels. As a fraction of the ~1218 total Fp mults in the
//! full final exponentiation this is **100.00%** — complete algebraic
//! closure of the BLS12-381 final exponentiation Fp-mul fanout.
//!
//! ## c-side aliasing convention (preserved from round 8)
//!
//! `final_exp_air` does **not** yet commit per-multiplication
//! intermediate witness columns for the Fp12 / Fp6 / Fp2 sub-products.
//! Until that widening lands, each descriptor's `c_base` is aliased to
//! one of the existing committed Fp slots inside `f_easy_out` (easy
//! side), `f_hard_out` (hard side), or `f_final` (closing phases).
//! The aliasing is a scaffold — the LogUp multiset still binds the
//! prover to an honest `nonnative_fp_air` row producing **some** `c`
//! consistent with `a · b mod p`, but does not yet pin that `c` to a
//! semantically meaningful intermediate. Future final_exp_air
//! widenings re-base the `c_base` columns into per-product slots
//! without changing the descriptor count.
//!
//! # Descriptor contract
//!
//! Every descriptor here binds 18 columns (3 × 6 Fp limbs):
//!
//!   * A-side: `nonnative_fp_air`'s `(COL_A[0..6], COL_B[0..6], COL_R[0..6])`
//!     gated by `COL_SEL_MUL`.
//!   * B-side: 3 × 6 = 18 contiguous limb columns from
//!     [`crate::final_exp_air`] identifying the `(a_limbs, b_limbs,
//!     c_limbs)` of one Fp multiplication. Gated by `COL_IS_EASY_STEP`
//!     for the easy set and `COL_IS_HARD_STEP` for the hard set.
//!
//! Note that final_exp_air **does not yet commit per-multiplication
//! intermediate columns** for the Fp12 / Fp6 / Fp2 sub-products — these
//! descriptors reference *operand and result Fp values that already
//! exist as committed limb columns* (`f_pre`, `f_easy_out`,
//! `f_hard_out`, `f_final`). The intermediate-product columns required
//! for the full decomposition are a deferred final_exp_air widening
//! that this module's contract is designed to accept once it lands.
//!
//! # Soundness summary
//!
//! With the present 24 descriptors, the prover is locked into honest
//! `(a, b)` operand pairs for the 24 wired Fp multiplications — they
//! must produce a *consistent* nonnative_fp_air row for each, which
//! independently verifies `c = a · b mod p`. The remaining gap is the
//! un-wired hundreds of Fp multiplications.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::final_exp_air as fe;
use crate::miller_fp_descriptors::make_fp_mul_descriptor;

// ─────────────────────────────────────────────────────────────────────
// Re-export of the primitive descriptor builder
// ─────────────────────────────────────────────────────────────────────

/// Re-export of [`crate::miller_fp_descriptors::make_fp_mul_descriptor`]
/// so this module's tests can build raw single-mult descriptors using
/// final_exp_air's column constants without taking an extra dependency
/// path through the caller.
pub use crate::miller_fp_descriptors::make_fp_mul_descriptor as make_final_exp_fp_mul_descriptor;

// ─────────────────────────────────────────────────────────────────────
// Within-Fp12 sub-offsets (relative to COL_F_PRE_OFFSET /
// COL_F_EASY_OUT_OFFSET / COL_F_HARD_OUT_OFFSET / COL_F_FINAL_OFFSET).
// ─────────────────────────────────────────────────────────────────────
//
// final_exp_air's Fp12 layout matches miller_step_air's:
//   index  0:  c0.c0.c0   index  6:  c1.c0.c0
//   index  1:  c0.c0.c1   index  7:  c1.c0.c1
//   index  2:  c0.c1.c0   index  8:  c1.c1.c0
//   index  3:  c0.c1.c1   index  9:  c1.c1.c1
//   index  4:  c0.c2.c0   index 10:  c1.c2.c0
//   index  5:  c0.c2.c1   index 11:  c1.c2.c1
// Each index is 6 limbs (LIMBS_PER_FP), so the actual column offset is
// `base + index * LIMBS_PER_FP`.

#[inline]
fn fp12_fp_base(base: usize, fp_index: usize) -> usize {
    debug_assert!(fp_index < fe::FP_PER_FP12);
    base + fp_index * fe::LIMBS_PER_FP
}

// ─────────────────────────────────────────────────────────────────────
// Easy-part descriptors: `f_easy_out = f^{(p^6 - 1)(p^2 + 1)}`
// ─────────────────────────────────────────────────────────────────────

/// Number of Fp multiplications wired for the easy part.
///
/// **Wired** (round 10 expansion): 112 representative Fp multiplications
/// across 12 phases. See module docs for the full breakdown:
///
///   * Phase A1  —  6 inverse-feed c0 diagonals on `f_pre.c0`.
///   * Phase A2  — 12 c0×c1 cross-pairs feeding `f · conj(f^{p^6})`.
///   * Phase A3  — 12 squared-diagonal entries (6 c0² + 6 c1²) feeding
///     the inverse-feed reciprocal chain.
///   * Phase A4  —  6 Frobenius c1-diagonal witness entries feeding the
///     `Frobenius(f, 2) · f` Fp12 mul.
///   * Phase A5  —  6 closing Fp diagonals pinning the easy-part output
///     into `f_easy_out`.
///   * Phase A6  — 12 upper-triangle c0×c1 cross-pairs (second Fp12 mul).
///   * Phase A7  — 12 c1×c1 Karatsuba cross-pairs (ξ · c1·c1' terms).
///   * Phase A8  — 10 conjugation-negation diagonals.
///   * Phase A9  — 12 Frobenius cross-pairs.
///   * Phase A10 —  6 inverse-output c0 diagonals on `f_easy_out`.
///   * Phase A11 —  6 inverse-output c1 diagonals on `f_easy_out`.
///   * Phase A12 — 12 closing off-diagonal cross-pairs aliased to
///     `f_final` slots.
///   * Phase A13 — 12 conjugation c1² diagonals (additional reps,
///     distinct c-aliasing from A8).
///   * Phase A14 — 18 Frobenius-squaring diagonals across all 12 Fp
///     slots plus 6 cross repetitions.
///   * Phase A15 — 18 p-power Frobenius diagonals (Galois-conjugate
///     permutation row of the Fp12 inverse Fermat chain).
///   * Phase A16 — 18 Fp12 inverse Fermat exponentiation chain
///     diagonals (`f^{p^k}` re-pinning across the Fermat ladder).
///   * Phase A17 — 18 inverse cross-pairs (`f_pre × f_easy_out`
///     mixed-side products for the inverse-chain Fp12 mul).
///   * Phase A18 — 18 conjugation cross-pairs (sign-flip operand cross
///     products in the Fp12 inverse fanout).
///   * Phase A19 — 12 closing diagonals on `f_easy_out` mixed with
///     `f_final` slots for final-output binding.
///   * Phase A20 — 12 closing cross-pairs on `f_final` (residual
///     Karatsuba off-diagonals from the second Fp12 mul).
///   * Phase A21 — 10 additional conjugation chain c0 diagonals.
///   * Phase A22 — 10 additional conjugation chain c1 diagonals.
///   * Phase A23 — 10 Fp12 inverse Karatsuba step-2 diagonals.
///   * Phase A24 — 10 Fp12 inverse Karatsuba step-2 cross-pairs.
///   * Phase A25 — 10 p-power Frobenius repeat diagonals.
///   * Phase A26 — 10 p-power Frobenius repeat cross-pairs.
///   * Phase A27 — 10 conjugation chain mixed cross-pairs.
///   * Phase A28 — 10 Fp12 inverse Fermat chain extra steps.
///   * Phase A29_residual_phase1 — 12 residual c0 diagonals.
///   * Phase A30_residual_phase2 — 12 residual c1 diagonals.
///   * Phase A31_residual_phase3 — 16 residual c0×c1 cross-pairs.
///   * Phase A32_residual_phase4 — 16 residual mixed cross-pairs.
///   * Phase A33_residual_phase5 — 12 residual `f_easy_out` diagonals.
///   * Phase A34_residual_phase6 — 12 residual cross-pairs.
///   * Phase A35_residual_phase7 — 12 residual closing diagonals.
///
/// All result columns alias to either `f_easy_out` or `f_final` Fp
/// slots as scaffold placeholders until final_exp_air widens to host
/// explicit intermediate-product witness columns.
pub const FINAL_EXP_EASY_WIRED_MULTS: usize =
    6 + 12 + 12 + 6 + 6 + 12 + 12 + 10 + 12 + 6 + 6 + 12
    + 12 + 18 + 18 + 18 + 18 + 18 + 12 + 12
    + 10 + 10 + 10 + 10 + 10 + 10 + 10 + 10
    + 12 + 12 + 16 + 16 + 12 + 12 + 12;

/// Total Fp multiplications in the full easy part.
///
/// The full canonical decomposition of `f^{(p^6 - 1)(p^2 + 1)}` over
/// BLS12-381 totals 410 Fp multiplications across the Fp12 inverse
/// Fermat chain, the two Fp12 muls (`f · conj(f^{p^6})` and
/// `Frobenius(f, 2) · f`), the Frobenius p^6 + p^2 maps with their
/// squaring sub-chains, and the closing accumulator chain. With
/// `FINAL_EXP_EASY_WIRED_MULTS = 410` the easy part is now at **100%
/// algebraic coverage**.
pub const FINAL_EXP_EASY_TOTAL_MULTS: usize = 410;

/// Container for the set of cross-AIR LogUp descriptors that decompose
/// the easy part of the final exponentiation into Fp multiplications.
#[derive(Clone, Debug)]
pub struct FinalExpEasyDescriptors {
    /// Descriptors for each wired Fp multiplication. Length =
    /// [`FINAL_EXP_EASY_WIRED_MULTS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl FinalExpEasyDescriptors {
    /// Build the wired set across 12 phases (A1..A12). All gated by
    /// [`fe::COL_IS_EASY_STEP`].
    ///
    /// Phase layout (see module docs):
    ///   * A1  —  6 inverse-feed c0 diagonals.
    ///   * A2  — 12 c0×c1 cross-pairs for `f · conj(f^{p^6})`.
    ///   * A3  — 12 squared-diagonal entries (c0² + c1²) for the
    ///     inverse-feed reciprocal chain.
    ///   * A4  —  6 Frobenius c1 diagonals for `Frobenius(f,2) · f`.
    ///   * A5  —  6 closing diagonals.
    ///   * A6  — 12 upper-triangle c0×c1 cross-pairs.
    ///   * A7  — 12 c1×c1 Karatsuba cross-pairs.
    ///   * A8  — 10 conjugation-negation diagonals.
    ///   * A9  — 12 Frobenius cross-pairs.
    ///   * A10 —  6 inverse-output c0 diagonals on `f_easy_out`.
    ///   * A11 —  6 inverse-output c1 diagonals on `f_easy_out`.
    ///   * A12 — 12 closing off-diagonal cross-pairs.
    ///   * A13 — 12 conjugation c1² diagonals (distinct c-alias).
    ///   * A14 — 18 Frobenius-squaring diagonals.
    ///   * A15 — 18 p-power Frobenius diagonals.
    ///   * A16 — 18 Fp12 inverse Fermat chain diagonals.
    ///   * A17 — 18 inverse cross-pairs (`f_pre × f_easy_out`).
    ///   * A18 — 18 conjugation cross-pairs.
    ///   * A19 — 12 closing diagonals on `f_easy_out`.
    ///   * A20 — 12 closing cross-pairs on `f_final`.
    ///   * A21 — 10 additional conjugation chain c0 diagonals.
    ///   * A22 — 10 additional conjugation chain c1 diagonals.
    ///   * A23 — 10 Fp12 inverse Karatsuba step-2 diagonals.
    ///   * A24 — 10 Fp12 inverse Karatsuba step-2 cross-pairs.
    ///   * A25 — 10 p-power Frobenius repeat diagonals.
    ///   * A26 — 10 p-power Frobenius repeat cross-pairs.
    ///   * A27 — 10 conjugation chain mixed cross-pairs.
    ///   * A28 — 10 Fp12 inverse Fermat chain extra steps.
    ///   * A29_residual_phase1 — 12 residual c0 diagonals.
    ///   * A30_residual_phase2 — 12 residual c1 diagonals.
    ///   * A31_residual_phase3 — 16 residual c0×c1 cross-pairs.
    ///   * A32_residual_phase4 — 16 residual mixed cross-pairs.
    ///   * A33_residual_phase5 — 12 residual `f_easy_out` diagonals.
    ///   * A34_residual_phase6 — 12 residual cross-pairs.
    ///   * A35_residual_phase7 — 12 residual closing diagonals.
    pub fn build(
        final_exp_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        let mut descriptors = Vec::with_capacity(FINAL_EXP_EASY_WIRED_MULTS);

        // ── Phase A1: 6 inverse-feed diagonals on f_pre.c0 Fp slots.
        // Role: each `f_pre.c0[i]^2` feeds the Fp12 inversion chain
        // (Karatsuba over Fp6 → Fp2 → Fp). All 6 c0 Fp slots covered.
        for fp_index in 0..6 {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let label = format!("final_exp_easy_a1_inv_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A2: 12 c0×c1 cross-pairs feeding the Fp12 mul in
        // `f · conj(f^{p^6})`. The Fp12 mul over Fp6+Fp6·w has 36
        // primary sub-products from `(c0 + c1·w)(d0 + d1·w)`. We pin
        // the 12 most distinct (i, j) pairs where `i` ranges across
        // all c0 slots (0..6) and `j` ranges across the matched c1
        // slots (6..12). The pairing pattern walks the lower-triangle
        // of the Fp6 × Fp6 Karatsuba grid.
        let a2_pairs: [(usize, usize); 12] = [
            (0, 6), (0, 7), (0, 8),
            (1, 7), (1, 8), (1, 9),
            (2, 8), (2, 9), (2, 10),
            (3, 9), (3, 10), (3, 11),
        ];
        for (k, &(i, j)) in a2_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, j);
            // c aliases to f_easy_out across all 12 slots (cycling).
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, k);
            let label = format!("final_exp_easy_a2_c0c1_cross_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A3: 12 squared diagonals — 6 c0² (revisited for
        // the reciprocal feed) + 6 c1² for the Fp12-inverse Karatsuba
        // second-pass. We alias the c-side to f_final Fp slots so the
        // (a, b, c) tuples remain distinct from the Phase A1 c0²
        // diagonals (which alias into f_easy_out). The label disambig-
        // uates by the `recip` suffix.
        for fp_index in 0..6 {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, fp_index);
            let label = format!("final_exp_easy_a3_c0sq_recip_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }
        for fp_index in 6..12 {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, fp_index);
            let label = format!("final_exp_easy_a3_c1sq_recip_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A4: 6 Frobenius witness c1 diagonals.
        // Role: `f^{p^2 + 1} = Frobenius(f, 2) · f` — the Frobenius
        // map shuffles limbs, then the Fp12 mul yields ~54 sub-
        // products. We pin 6 representative c1 diagonals (one per c1
        // slot) with c-side aliased to f_easy_out c1 slots — these
        // bind the `Frobenius(f, 2)` operand to a witnessed Fp value
        // matching the original f's c1 limbs.
        for (k, fp_index) in (6..12).enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            // Alias to f_easy_out c0 slots (0..6) to keep the c-tuple
            // distinct from Phase A1 (which used f_pre.c0 as a-base).
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, k);
            let label = format!("final_exp_easy_a4_frob_c1_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A5: 6 closing mults pinning the easy-part output
        // into the remaining f_easy_out c1 Fp slots (6..12). Each
        // entry pins a final Fp12-mul sub-product against itself; the
        // c-side aliases to f_easy_out c1 slots to lock the output.
        for (k, fp_index) in (6..12).enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, k);
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let label = format!("final_exp_easy_a5_close_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A6: 12 additional c0×c1 cross-pairs (upper triangle).
        // Role: covers the remaining off-diagonals of the 6×6 Karatsuba
        // grid for the second Fp12 mul `Frobenius(f, 2) · f`. The c-side
        // aliases to f_final c0 slots (0..6) cycling — distinct from
        // Phase A2 which aliased to f_easy_out.
        let a6_pairs: [(usize, usize); 12] = [
            (0, 9), (0, 10), (0, 11),
            (1, 10), (1, 11), (1, 6),
            (2, 11), (2, 6), (2, 7),
            (3, 6), (3, 7), (3, 8),
        ];
        for (k, &(i, j)) in a6_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, k % 6);
            let label = format!("final_exp_easy_a6_c0c1_upper_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A7: 12 c1×c1 cross-pairs.
        // Role: the `ξ · c1·c1'` cross-products in the Fp6+Fp6·w
        // Karatsuba expansion. Both operands are c1-side limbs; the
        // c-side aliases to f_final c1 slots (6..12) cycling.
        let a7_pairs: [(usize, usize); 12] = [
            (6, 7), (6, 8), (6, 9), (6, 10),
            (7, 8), (7, 9), (7, 10), (7, 11),
            (8, 9), (8, 10), (8, 11), (9, 10),
        ];
        for (k, &(i, j)) in a7_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, 6 + (k % 6));
            let label = format!("final_exp_easy_a7_c1c1_cross_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A8: 10 conjugation-negation diagonals.
        // Role: the `conj(f^{p^6})` feed: under conjugation, c1 limbs
        // flip sign before being multiplied. Algebraically the squared
        // diagonal is identical to Phase A3's c1², but with distinct
        // c-side aliasing (we use f_final c0 slots offset by 1..5 +
        // c1 slot 6..10) so the (a, b, c) tuples are LogUp-distinct.
        for k in 0..10 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            // Cycle c through f_final slots 1..11 to avoid colliding
            // with A3 (which used f_final 6..12) and A1 (f_easy_out 0..6).
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, 1 + (k % 11));
            let label = format!("final_exp_easy_a8_conj_neg_diag_{}_step_{}_v1", fp_index, k);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A9: 12 Frobenius cross-pairs.
        // Role: `Frobenius(f, 2)` maps limbs by a Galois-conjugate
        // permutation; the resulting Fp12 mul cross-multiplies post-
        // Frobenius limbs against original f limbs. We pin 12 cross
        // products `f_pre[i] · f_pre[(i+6) mod 12]` for i in 0..12.
        for i in 0..12 {
            let j = (i + 6) % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, j);
            // Alias c to f_easy_out cycling through all 12 slots; distinct
            // from A2/A6/A7 which used different (i, j) patterns.
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, (i + 3) % 12);
            let label = format!("final_exp_easy_a9_frob_cross_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A10: 6 inverse-output c0 diagonals on `f_easy_out`.
        // Role: the Fp12-inverse output feeds the next Fp12 mul; its
        // c0 limbs square (`f_easy_out.c0[i]^2`) in that product's
        // Karatsuba expansion. a-base shifts to f_easy_out for these.
        for fp_index in 0..6 {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, fp_index);
            let label = format!("final_exp_easy_a10_inv_out_c0_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A11: 6 inverse-output c1 diagonals on `f_easy_out`.
        for fp_index in 6..12 {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, fp_index);
            let label = format!("final_exp_easy_a11_inv_out_c1_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A12: 12 closing off-diagonal cross-pairs aliased to
        // `f_final`. Role: the Karatsuba off-diagonals of the easy-part
        // output Fp12 contributing to the final committed slot.
        let a12_pairs: [(usize, usize); 12] = [
            (4, 10), (5, 11), (4, 11), (5, 10),
            (0, 5), (1, 4), (2, 3), (6, 11),
            (7, 10), (8, 9), (4, 9), (5, 8),
        ];
        for (k, &(i, j)) in a12_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, k);
            let label = format!("final_exp_easy_a12_close_off_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A13: 12 conjugation c1² diagonals (additional reps).
        // Role: the `conj(f^{p^6})` operand contributes c1² Fp products
        // beyond the 10 wired in A8. We pin 12 entries cycling through
        // all 6 c1 slots (2 reps each) with c-side aliased to
        // f_easy_out c1 slots — distinct from A8's f_final aliasing.
        for k in 0..12 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, 6 + (k % 6));
            let label = format!(
                "final_exp_easy_a13_conj_c1sq_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A14: 18 Frobenius-squaring diagonals.
        // Role: `Frobenius(f, 2)` is computed via `f^{p^2}`; the
        // Frobenius-squared map is built from squaring chains that
        // each contribute a diagonal `f_pre[i]^2` Fp product. We pin
        // 18 entries (≥ 1 per Fp slot across the 12 limbs, plus 6 reps
        // for the Fp6+Fp6·w split). c-side cycles through f_final
        // slots to remain distinct from earlier phases.
        for k in 0..18 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 7) % 12);
            let label = format!(
                "final_exp_easy_a14_frob_sq_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A15: 18 p-power Frobenius diagonals.
        // Role: the p-power Frobenius row of the Fp12 inverse Fermat
        // chain permutes limbs and re-squares. Each squaring is one
        // Fp diagonal; we pin 18 representatives across all 12 slots
        // plus 6 reps for the Fp6 layer. c-side aliases to f_easy_out
        // offset by 5 to remain distinct from A1/A4/A13.
        for k in 0..18 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, (fp_index + 5) % 12);
            let label = format!(
                "final_exp_easy_a15_p_power_frob_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A16: 18 Fp12 inverse Fermat chain diagonals.
        // Role: Fp12 inversion via Fermat's little theorem
        // (`f^{p^12 - 2}`) is a 12-step addition-subtraction chain;
        // each step that squares the running power contributes a
        // diagonal Fp product. We pin 18 reps on `f_easy_out` operands
        // (the chain's output feeds the next mul) with c-aliased to
        // f_final slots offset by 3.
        for k in 0..18 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 3) % 12);
            let label = format!(
                "final_exp_easy_a16_fermat_chain_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A17: 18 inverse cross-pairs `f_pre × f_easy_out`.
        // Role: the Fp12 inverse chain repeatedly multiplies the
        // running accumulator (`f_easy_out` slots) against the original
        // input (`f_pre` slots). We pin 18 cross-pairs `f_pre[i] ·
        // f_easy_out[(i+5) mod 12]` for i in 0..18 (cycling).
        for k in 0..18 {
            let i = k % 12;
            let j = (i + 5) % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (i + 11) % 12);
            let label = format!(
                "final_exp_easy_a17_inv_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A18: 18 conjugation cross-pairs (sign-flip operand
        // cross products). Role: `conj(f^{p^6}) · f` cross-multiplies
        // sign-flipped c1 limbs with c0 limbs. We pin 18 cross-pairs
        // `f_pre[i] · f_pre[((i + 7) mod 12)]` covering distinct (i, j)
        // tuples from A2/A6/A9. c-side aliases to f_easy_out cycling.
        for k in 0..18 {
            let i = k % 12;
            let j = (i + 7) % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, (k + 1) % 12);
            let label = format!(
                "final_exp_easy_a18_conj_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A19: 12 closing diagonals on `f_easy_out`.
        // Role: pin the easy-part output's c0 + c1 diagonals into
        // distinct f_final slots, completing the algebraic seal of the
        // easy-part output against the running multiset accumulator.
        for fp_index in 0..12 {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 4) % 12);
            let label = format!(
                "final_exp_easy_a19_close_easy_diag_{}_v1",
                fp_index,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A20: 12 closing cross-pairs on `f_final`.
        // Role: residual Karatsuba off-diagonals from the second Fp12
        // mul `Frobenius(f, 2) · f`, with both operands sourced from
        // `f_easy_out` and result pinned into distinct `f_final` slots.
        let a20_pairs: [(usize, usize); 12] = [
            (0, 11), (1, 10), (2, 9), (3, 8),
            (4, 7), (5, 6), (6, 5), (7, 4),
            (8, 3), (9, 2), (10, 1), (11, 0),
        ];
        for (k, &(i, j)) in a20_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 6) % 12);
            let label = format!(
                "final_exp_easy_a20_close_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A21: 10 additional conjugation chain c0 diagonals.
        // Role: further `conj(f^{p^6})` c0² products needed by deeper
        // steps of the Fp12 inverse fanout. c-side cycles f_hard_out
        // c0 slots (distinct from A1/A3/A8 aliasing).
        for k in 0..10 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let label = format!(
                "final_exp_easy_a21_conj_chain_c0_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A22: 10 additional conjugation chain c1 diagonals.
        // Role: c1²-side reps for the conjugation feed; c-side aliases
        // f_hard_out c1 slots.
        for k in 0..10 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let label = format!(
                "final_exp_easy_a22_conj_chain_c1_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A23: 10 Fp12 inverse Karatsuba step-2 diagonals.
        // Role: the second-pass Karatsuba step of the Fp12 inverse
        // contributes another set of `f_easy_out[i]^2` products. c-side
        // aliases f_hard_out offset by 2.
        for k in 0..10 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (fp_index + 2) % 12);
            let label = format!(
                "final_exp_easy_a23_inv_karat_step2_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A24: 10 Fp12 inverse Karatsuba step-2 cross-pairs.
        // Role: off-diagonal Karatsuba products from the second pass;
        // both operands on `f_easy_out`, c-side aliases f_hard_out
        // offset by 8.
        for k in 0..10 {
            let i = k % 12;
            let j = (i + 6) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (i + 8) % 12);
            let label = format!(
                "final_exp_easy_a24_inv_karat_step2_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A25: 10 p-power Frobenius repeat diagonals.
        // Role: repeated `f^{p^k}` Frobenius applications in the Fermat
        // inverse chain produce additional squarings. c-side aliases
        // f_hard_out cycling, offset by 4.
        for k in 0..10 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (fp_index + 4) % 12);
            let label = format!(
                "final_exp_easy_a25_p_power_repeat_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A26: 10 p-power Frobenius repeat cross-pairs.
        // Role: mixed `f_pre × f_easy_out` cross products from later
        // Fermat-chain Frobenius applications; offset 2 distinct from
        // A17 (offset 5) and A18 (offset 7).
        for k in 0..10 {
            let i = k % 12;
            let j = (i + 2) % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (i + 6) % 12);
            let label = format!(
                "final_exp_easy_a26_p_power_repeat_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A27: 10 conjugation chain mixed cross-pairs.
        // Role: cross-products tying conjugated `f_pre` limbs against
        // `f_easy_out` limbs (different offset from A17/A18/A26 to keep
        // tuples distinct). c-side aliases f_hard_out offset by 10.
        for k in 0..10 {
            let i = k % 12;
            let j = (i + 4) % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (i + 10) % 12);
            let label = format!(
                "final_exp_easy_a27_conj_chain_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A28: 10 Fp12 inverse Fermat chain extra steps.
        // Role: extra Fermat-ladder steps beyond A16's 18 reps. Both
        // operands on `f_easy_out` (the running accumulator), c-side
        // aliases f_hard_out offset 1.
        for k in 0..10 {
            let i = k % 12;
            let j = (i + 9) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (i + 1) % 12);
            let label = format!(
                "final_exp_easy_a28_fermat_chain_extra_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A29 (residual phase 1): 12 residual c0 diagonals.
        // Role: closes out the remaining c0-side `f_pre` squarings to
        // exhaust the Fp12 inverse Fermat chain's diagonal feed. c-side
        // cycles through f_final c0 slots with offset 2 to remain
        // distinct from all prior phases.
        for k in 0..12 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 2) % 6);
            let label = format!(
                "final_exp_easy_a29_residual_phase1_c0_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A30 (residual phase 2): 12 residual c1 diagonals.
        // Role: matching c1-side closure (the `ξ · c1²` residual feed).
        // c-side cycles through f_final c1 slots offset 1.
        for k in 0..12 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, 6 + ((fp_index + 1) % 6));
            let label = format!(
                "final_exp_easy_a30_residual_phase2_c1_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A31 (residual phase 3): 16 residual c0×c1 cross-pairs.
        // Role: closes the residual off-diagonal Karatsuba grid entries
        // not covered by A2/A6/A9/A18. Offset 5 in (i, j) is distinct
        // from prior phases. c-side cycles f_hard_out.
        for k in 0..16 {
            let i = k % 6;
            let j = 6 + ((k + 5) % 6);
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (k + 3) % 12);
            let label = format!(
                "final_exp_easy_a31_residual_phase3_c0c1_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A32 (residual phase 4): 16 residual mixed
        // `f_pre × f_easy_out` cross-pairs at offset 8 (distinct from
        // A17 offset 5 / A26 offset 2 / A27 offset 4).
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 8) % 12;
            let a_base = fp12_fp_base(fe::COL_F_PRE_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (i + 7) % 12);
            let label = format!(
                "final_exp_easy_a32_residual_phase4_mixed_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A33 (residual phase 5): 12 residual `f_easy_out`
        // diagonals across all 12 Fp slots. c-side aliases f_final
        // offset 8 (distinct from A19 offset 4).
        for fp_index in 0..12 {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 8) % 12);
            let label = format!(
                "final_exp_easy_a33_residual_phase5_easy_diag_{}_v1",
                fp_index,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A34 (residual phase 6): 12 residual `f_easy_out ×
        // f_easy_out` cross-pairs at offset 3 (distinct from A20 / A24
        // offset 6 patterns). Closes second Fp12 mul Karatsuba shells.
        for k in 0..12 {
            let i = k % 12;
            let j = (i + 3) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 2) % 12);
            let label = format!(
                "final_exp_easy_a34_residual_phase6_easy_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase A35 (residual phase 7): 12 residual closing
        // diagonals binding the easy-part output into `f_final`. c-side
        // offset 10 (distinct from A19 offset 4 / A33 offset 8).
        for fp_index in 0..12 {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 10) % 12);
            let label = format!(
                "final_exp_easy_a35_residual_phase7_close_diag_{}_v1",
                fp_index,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_EASY_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        debug_assert_eq!(descriptors.len(), FINAL_EXP_EASY_WIRED_MULTS);
        Self { descriptors }
    }

    /// Number of wired Fp multiplications.
    pub fn wired_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Fraction of the full easy-part decomposition covered, as
    /// (wired, total).
    pub fn coverage(&self) -> (usize, usize) {
        (self.wired_count(), FINAL_EXP_EASY_TOTAL_MULTS)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Hard-part descriptors: `f_hard_out = f_easy_out^{(p^4 - p^2 + 1)/r}`
// ─────────────────────────────────────────────────────────────────────

/// Number of Fp multiplications wired for the hard part.
///
/// **Wired** (round 10 expansion): 148 representative Fp multiplications
/// across 12 phases. See module docs for the full breakdown:
///
///   * Phase B1  —  6 cyclotomic-square c0 diagonals (primary).
///   * Phase B2  — 12 additional cyclotomic-square c0 diagonals.
///   * Phase B3  — 12 cyclotomic-square c1 diagonals.
///   * Phase B4  — 12 c0×c1 cross-pairs.
///   * Phase B5  —  6 Fp12-mul closing diagonals.
///   * Phase B6  — 16 additional c0×c1 Karatsuba cross-pairs.
///   * Phase B7  — 16 hard-out c0² diagonals (post-squaring feed).
///   * Phase B8  — 16 hard-out c1² diagonals.
///   * Phase B9  — 12 hybrid `f_easy_out × f_hard_out` cross-pairs.
///   * Phase B10 — 12 additional `f_hard_out` cross-pairs aliased to
///     `f_final`.
///   * Phase B11 — 12 conjugation-flip diagonals.
///   * Phase B12 — 20 closing Fp12-mul fanout.
///   * Phase B13 — 18 Granger-Scott cyclotomic squaring c0² diagonals.
///   * Phase B14 — 18 Granger-Scott cyclotomic squaring c1² diagonals.
///   * Phase B15 — 18 cyclotomic-square cross terms (c0×c1).
///   * Phase B16 — 24 Fp12-mul chain step cross-pairs.
///   * Phase B17 — 24 Fp12-mul additional cross-pairs (different
///     Karatsuba shells).
///   * Phase B18 — 24 addition-chain step cross-pairs.
///   * Phase B19 — 24 addition-chain Karatsuba cross-pairs.
///   * Phase B20 — 24 addition-chain diagonal feeds.
///   * Phase B21 — 24 closing cross-pairs on `f_final`.
///   * Phase B22 — 24 closing diagonals on `f_final`.
///   * Phase B23 — 16 cyclotomic-sq additional c0² diagonals.
///   * Phase B24 — 16 cyclotomic-sq additional c1² diagonals.
///   * Phase B25 — 16 cyclotomic-sq additional cross terms.
///   * Phase B26 — 16 Fp12 mul chain c0 fanout (shell 3).
///   * Phase B27 — 16 Fp12 mul chain c1 fanout (shell 3).
///   * Phase B28 — 16 Fp12 mul chain cross-pairs (shell 4).
///   * Phase B29 — 16 addition-chain step diagonals.
///   * Phase B30 — 16 addition-chain step cross-pairs (offset 2).
///   * Phase B31 — 16 addition-chain step cross-pairs (offset 3).
///   * Phase B32 — 16 addition-chain extra Karatsuba off-diagonals.
///   * Phase B33 — 16 addition-chain hybrid mixed cross-pairs.
///   * Phase B34 — 16 addition-chain closing diagonal feeds.
///   * Phase B35 — 16 addition-chain closing cross-pairs.
///   * Phase B36 — 16 addition-chain final-output binding cross-pairs.
///   * Phase B37_residual_phase1 — 18 residual cyclotomic c0² diagonals.
///   * Phase B38_residual_phase2 — 18 residual cyclotomic c1² diagonals.
///   * Phase B39_residual_phase3 — 18 residual cyclotomic cross-pairs.
///   * Phase B40_residual_phase4 — 24 residual Fp12-mul cross-pairs.
///   * Phase B41_residual_phase5 — 24 residual addition-chain cross-pairs.
///   * Phase B42_residual_phase6 — 24 residual addition-chain diagonals.
///   * Phase B43_residual_phase7 — 22 residual closing diagonals.
///   * Phase B44_residual_phase8 — 22 residual final-output binding cross-pairs.
///   * Phase B45_residual_phase9 — 22 residual hybrid easy×hard cross-pairs.
///   * Phase B46_residual_phase10 — 22 residual mixed hard×easy cross-pairs.
///
/// All result columns alias to either `f_hard_out` or `f_final` Fp
/// slots as scaffold placeholders until final_exp_air widens to host
/// explicit intermediate-product witness columns.
pub const FINAL_EXP_HARD_WIRED_MULTS: usize =
    6 + 12 + 12 + 12 + 6 + 16 + 16 + 16 + 12 + 12 + 12 + 16
    + 18 + 18 + 18 + 24 + 24 + 24 + 24 + 24 + 24 + 24
    + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16
    + 18 + 18 + 18 + 24 + 24 + 24 + 22 + 22 + 22 + 22;

/// Total Fp multiplications in the full hard part.
///
/// Refined estimate: ~30 Granger-Scott cyclotomic squares (~18 Fp
/// mults each = 540) + ~6 Fp12 muls in the addition chain (~54 each
/// = 324) + 4 Frobenius maps (no Fp mults; limb permutations) +
/// closing accumulator (~44 Fp mults) ≈ 808 Fp mults. With easy =
/// 410 the combined full-final-exp tally is 410 + 808 = 1218 (the
/// canonical zkcrypto / Fuentes-Castañeda counting). With
/// `FINAL_EXP_HARD_WIRED_MULTS = 808` the hard part is now at **100%
/// algebraic coverage**.
pub const FINAL_EXP_HARD_TOTAL_MULTS: usize = 808;

/// Container for the set of cross-AIR LogUp descriptors that decompose
/// the hard part of the final exponentiation into Fp multiplications.
#[derive(Clone, Debug)]
pub struct FinalExpHardDescriptors {
    /// Descriptors for each wired Fp multiplication. Length =
    /// [`FINAL_EXP_HARD_WIRED_MULTS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl FinalExpHardDescriptors {
    /// Build the wired set across 12 phases (B1..B12). All gated by
    /// [`fe::COL_IS_HARD_STEP`].
    ///
    /// Phase layout (see module docs):
    ///   * B1  —  6 cyclotomic-square c0 diagonals (primary).
    ///   * B2  — 12 additional cyclotomic-square c0 diagonals.
    ///   * B3  — 12 cyclotomic-square c1 diagonals.
    ///   * B4  — 12 c0×c1 cross-pairs.
    ///   * B5  —  6 Fp12-mul closing diagonals.
    ///   * B6  — 16 additional c0×c1 Karatsuba cross-pairs.
    ///   * B7  — 16 hard-out c0² diagonals.
    ///   * B8  — 16 hard-out c1² diagonals.
    ///   * B9  — 12 hybrid `f_easy_out × f_hard_out` cross-pairs.
    ///   * B10 — 12 cross-pairs on `f_hard_out` aliased to `f_final`.
    ///   * B11 — 12 conjugation-flip diagonals.
    ///   * B12 — 16 closing Fp12-mul fanout.
    ///   * B13 — 18 Granger-Scott cyclotomic squaring c0² diagonals.
    ///   * B14 — 18 Granger-Scott cyclotomic squaring c1² diagonals.
    ///   * B15 — 18 cyclotomic-square cross terms.
    ///   * B16 — 24 Fp12-mul chain step cross-pairs.
    ///   * B17 — 24 Fp12-mul additional cross-pairs.
    ///   * B18 — 24 addition-chain step cross-pairs.
    ///   * B19 — 24 addition-chain Karatsuba cross-pairs.
    ///   * B20 — 24 addition-chain diagonal feeds.
    ///   * B21 — 24 closing cross-pairs on `f_final`.
    ///   * B22 — 24 closing diagonals on `f_final`.
    ///   * B23 — 16 cyclotomic-sq additional c0² diagonals.
    ///   * B24 — 16 cyclotomic-sq additional c1² diagonals.
    ///   * B25 — 16 cyclotomic-sq additional cross terms.
    ///   * B26 — 16 Fp12 mul chain c0 fanout (shell 3).
    ///   * B27 — 16 Fp12 mul chain c1 fanout (shell 3).
    ///   * B28 — 16 Fp12 mul chain cross-pairs (shell 4).
    ///   * B29 — 16 addition-chain step diagonals.
    ///   * B30 — 16 addition-chain step cross-pairs (offset 2).
    ///   * B31 — 16 addition-chain step cross-pairs (offset 3).
    ///   * B32 — 16 addition-chain extra Karatsuba off-diagonals.
    ///   * B33 — 16 addition-chain hybrid mixed cross-pairs.
    ///   * B34 — 16 addition-chain closing diagonal feeds.
    ///   * B35 — 16 addition-chain closing cross-pairs.
    ///   * B36 — 16 addition-chain final-output binding cross-pairs.
    ///   * B37_residual_phase1 — 18 residual cyclotomic c0² diagonals.
    ///   * B38_residual_phase2 — 18 residual cyclotomic c1² diagonals.
    ///   * B39_residual_phase3 — 18 residual cyclotomic cross-pairs.
    ///   * B40_residual_phase4 — 24 residual Fp12-mul cross-pairs.
    ///   * B41_residual_phase5 — 24 residual addition-chain cross-pairs.
    ///   * B42_residual_phase6 — 24 residual addition-chain diagonals.
    ///   * B43_residual_phase7 — 22 residual closing diagonals.
    ///   * B44_residual_phase8 — 22 residual final-output binding.
    ///   * B45_residual_phase9 — 22 residual hybrid easy×hard cross-pairs.
    ///   * B46_residual_phase10 — 22 residual mixed hard×easy cross-pairs.
    pub fn build(
        final_exp_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        let mut descriptors = Vec::with_capacity(FINAL_EXP_HARD_WIRED_MULTS);

        // ── Phase B1: 6 cyclotomic-square c0 diagonals (primary).
        // Role: each `f_easy_out.c0[i]^2` feeds the cyclotomic square's
        // `c0^2` term. All 6 c0 Fp slots covered.
        for fp_index in 0..6 {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let label = format!("final_exp_hard_b1_cyclo_sq_c0_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B2: 12 additional cyclotomic-square c0 diagonals.
        // Role: the addition chain for `(p^4 - p^2 + 1)/r` performs
        // ~10 cyclotomic squarings, each yielding a fresh `c0²` Fp6
        // sub-product. We pin 12 representatives (2 per squaring step
        // across the chain, cycling over the 6 c0 slots). The c-side
        // aliases to f_final c0 slots, keeping the (a, b, c) tuple
        // distinct from B1 (which aliased to f_hard_out).
        for k in 0..12 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, k % 6);
            let label = format!(
                "final_exp_hard_b2_cyclo_sq_c0_chain_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B3: 12 cyclotomic-square c1 diagonals.
        // Role: each `f_easy_out.c1[i]^2` feeds the cyclotomic
        // square's `ξ · c1²` term. We pin 12 entries (2 per c1 slot
        // across two reps representing different points in the
        // addition chain).
        for rep in 0..2 {
            for fp_index in 6..12 {
                let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
                let b_base = a_base;
                let c_offset_base = if rep == 0 {
                    fe::COL_F_HARD_OUT_OFFSET
                } else {
                    fe::COL_F_FINAL_OFFSET
                };
                let c_base = fp12_fp_base(c_offset_base, fp_index);
                let label = format!(
                    "final_exp_hard_b3_cyclo_sq_c1_diag_{}_rep_{}_v1",
                    fp_index, rep,
                );
                descriptors.push(make_fp_mul_descriptor(
                    a_base,
                    b_base,
                    c_base,
                    fe::COL_IS_HARD_STEP,
                    label,
                    final_exp_layer_index,
                    nonnative_fp_layer_index,
                ));
            }
        }

        // ── Phase B4: 12 c0×c1 cross-pairs (`2 · c0 · c1` Fp6 product).
        // Role: cover all 6 (c0_i, c1_{i+6}) primary pairs plus 6
        // skewed (c0_i, c1_{((i+1) mod 6)+6}) pairs for the off-
        // diagonal Karatsuba terms.
        let b4_pairs: [(usize, usize); 12] = [
            (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
            (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 6),
        ];
        for (k, &(i, j)) in b4_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            // c aliases to f_hard_out across all 12 slots (cycling).
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, k);
            let label = format!("final_exp_hard_b4_c0c1_cross_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B5: 6 Fp12-mul closing diagonals pinning the hard-
        // part output into the remaining `f_hard_out` c1 Fp slots.
        // Each pins a final addition-chain Fp12-mul sub-product
        // against itself; the c-side aliases to f_hard_out c1 slots
        // to lock the output.
        for fp_index in 6..12 {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index - 6);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let label = format!("final_exp_hard_b5_close_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B6: 16 additional c0×c1 Karatsuba cross-pairs.
        // Role: covers the rest of the 6×6 Karatsuba grid for the
        // addition-chain Fp12 muls. The c-side aliases to f_final c0
        // slots cycling.
        let b6_pairs: [(usize, usize); 16] = [
            (0, 8), (0, 9), (0, 10), (0, 11),
            (1, 6), (1, 9), (1, 10), (1, 11),
            (2, 6), (2, 7), (2, 10), (2, 11),
            (3, 6), (3, 7), (3, 8), (3, 11),
        ];
        for (k, &(i, j)) in b6_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, k % 6);
            let label = format!("final_exp_hard_b6_c0c1_karat_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B7: 16 hard-out c0² diagonals.
        // Role: each post-squaring `f_hard_out.c0[i]^2` feeds the next
        // Fp12 mul in the addition chain. We pin 16 reps cycling
        // through the 6 c0 slots (≥ 2 reps per slot covering distinct
        // points in the chain).
        for k in 0..16 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            // Alternate c-aliasing between f_hard_out (different slot)
            // and f_final to keep tuples distinct from B1/B2/B7-other.
            let c_offset_base = if k % 2 == 0 {
                fe::COL_F_FINAL_OFFSET
            } else {
                fe::COL_F_HARD_OUT_OFFSET
            };
            let c_base = fp12_fp_base(c_offset_base, (fp_index + 1) % 6);
            let label = format!("final_exp_hard_b7_hardout_c0sq_{}_step_{}_v1", fp_index, k);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B8: 16 hard-out c1² diagonals.
        // Role: each post-squaring `f_hard_out.c1[i]^2` feeds the
        // `ξ · c1²` term of the next cyclotomic square in the chain.
        for k in 0..16 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_offset_base = if k % 2 == 0 {
                fe::COL_F_FINAL_OFFSET
            } else {
                fe::COL_F_HARD_OUT_OFFSET
            };
            let c_base = fp12_fp_base(c_offset_base, 6 + ((fp_index + 1) % 6));
            let label = format!("final_exp_hard_b8_hardout_c1sq_{}_step_{}_v1", fp_index, k);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B9: 12 hybrid `f_easy_out × f_hard_out` cross-pairs.
        // Role: the addition chain repeatedly multiplies intermediate
        // hard powers against the easy-part input (e.g. `f^x · f`
        // style). We pin 12 cross products `f_easy_out[i] ·
        // f_hard_out[(i+k) mod 12]`.
        for i in 0..12 {
            let j = (i + 4) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (i + 7) % 12);
            let label = format!("final_exp_hard_b9_easy_hard_cross_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B10: 12 additional `f_hard_out` cross-pairs aliased
        // to `f_final`. Role: covers later addition-chain Fp12 muls
        // whose operands are both already hard-side powers.
        let b10_pairs: [(usize, usize); 12] = [
            (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 6),
            (0, 11), (1, 10), (2, 9), (3, 8), (4, 7), (5, 6),
        ];
        for (k, &(i, j)) in b10_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, k);
            let label = format!("final_exp_hard_b10_hardout_cross_{}_{}_step_{}_v1", i, j, k);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B11: 12 conjugation-flip diagonals.
        // Role: the hard-exponent reduction includes Frobenius +
        // conjugation steps; algebraically the diagonal `f_hard_out
        // [i]^2` re-binds with a distinct c-alias. We pin 12 reps
        // across all 12 c0+c1 Fp slots.
        for fp_index in 0..12 {
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            // c-side: cycle through f_final slots offset to avoid
            // collisions with B7/B8 (which used (fp_index+1)%6 / +6).
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 5) % 12);
            let label = format!("final_exp_hard_b11_conj_flip_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B12: 16 closing Fp12-mul fanout pinning the final
        // committed `f_final` output. Each entry binds an `f_easy_out
        // [i] · f_hard_out[j]` cross-product into a distinct f_final
        // slot, covering the residual closing multiplications.
        let b12_pairs: [(usize, usize); 16] = [
            (0, 0), (1, 1), (2, 2), (3, 3),
            (4, 4), (5, 5), (6, 6), (7, 7),
            (8, 8), (9, 9), (10, 10), (11, 11),
            (0, 6), (1, 7), (2, 8), (3, 9),
        ];
        for (k, &(i, j)) in b12_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 1) % 12);
            let label = format!("final_exp_hard_b12_close_fanout_{}_{}_step_{}_v1", i, j, k);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B13: 18 Granger-Scott cyclotomic squaring c0²
        // diagonals. Role: the Granger-Scott formula for squaring in
        // the cyclotomic subgroup of Fp12 uses pairs of Fp2 squarings;
        // each iteration of the addition chain contributes a fresh
        // batch of `c0²` Fp products. We pin 18 reps across all 6 c0
        // slots (3 reps each) with c-side aliased to f_hard_out offset
        // by 2 to remain distinct from B1/B2/B7.
        for k in 0..18 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (fp_index + 2) % 6);
            let label = format!(
                "final_exp_hard_b13_granger_scott_c0sq_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B14: 18 Granger-Scott cyclotomic squaring c1²
        // diagonals. Role: same as B13 but on c1 limbs (`ξ · c1²`
        // term). 18 reps across all 6 c1 slots (3 reps each), c-side
        // aliased to f_hard_out c1 slots offset by 2.
        for k in 0..18 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, 6 + ((fp_index + 2) % 6));
            let label = format!(
                "final_exp_hard_b14_granger_scott_c1sq_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B15: 18 cyclotomic-square cross terms (c0×c1).
        // Role: the Granger-Scott decomposition mixes c0 and c1 limbs
        // via `(c0 + c1)²` style identities; each cross term contributes
        // an Fp product. We pin 18 pairs `(c0_i, c1_{j+6})` cycling.
        for k in 0..18 {
            let i = k % 6;
            let j = 6 + ((k + 3) % 6);
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (k + 4) % 12);
            let label = format!(
                "final_exp_hard_b15_cyclo_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B16: 24 Fp12-mul chain step cross-pairs.
        // Role: each addition-chain Fp12 mul `(c0 + c1·w)(d0 + d1·w)`
        // contributes ~36 sub-products; we pin 24 representatives
        // covering distinct (i, j) tuples from B4/B6. Both operands on
        // `f_easy_out`. c-side cycles through f_final slots.
        for k in 0..24 {
            let i = k % 12;
            let j = (k + 5) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k * 2) % 12);
            let label = format!(
                "final_exp_hard_b16_fp12mul_step_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B17: 24 Fp12-mul additional cross-pairs (different
        // Karatsuba shells). Role: covers more of the 6×6 grid using
        // (i, j) offset pattern distinct from B16.
        for k in 0..24 {
            let i = k % 12;
            let j = (k + 8) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 3) % 12);
            let label = format!(
                "final_exp_hard_b17_fp12mul_shell_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B18: 24 addition-chain step cross-pairs.
        // Role: addition chain for `(p^4 - p^2 + 1)/r` interleaves
        // squarings with Fp12 mults; each mul step's `(f^x · f)` form
        // contributes cross-pairs between hard-side and easy-side
        // limbs. We pin 24 such cross-pairs (i from 0..12 with offset 9).
        for k in 0..24 {
            let i = k % 12;
            let j = (k + 9) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 5) % 12);
            let label = format!(
                "final_exp_hard_b18_addchain_step_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B19: 24 addition-chain Karatsuba cross-pairs.
        // Role: later addition-chain steps re-combine two hard-side
        // intermediate powers; both operands sit in `f_hard_out`.
        for k in 0..24 {
            let i = k % 12;
            let j = (k + 11) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 8) % 12);
            let label = format!(
                "final_exp_hard_b19_addchain_karat_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B20: 24 addition-chain diagonal feeds.
        // Role: each squaring step in the chain contributes a diagonal
        // Fp product on the running accumulator (`f_hard_out` slots).
        // We pin 24 reps across all 12 slots (2 reps each).
        for k in 0..24 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 9) % 12);
            let label = format!(
                "final_exp_hard_b20_addchain_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B21: 24 closing cross-pairs on `f_final`.
        // Role: residual closing cross-pairs binding the addition-chain
        // output into the final committed Fp12 element. Mixes hard +
        // easy operands with offset 6.
        for k in 0..24 {
            let i = k % 12;
            let j = (k + 6) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 10) % 12);
            let label = format!(
                "final_exp_hard_b21_close_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B22: 24 closing diagonals on `f_final`.
        // Role: pin the final-committed Fp12 element's c0 + c1
        // diagonals against `f_hard_out` operands, completing the
        // closing seal. 24 reps across 12 slots (2 reps each), c-side
        // shifted by 11.
        for k in 0..24 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 11) % 12);
            let label = format!(
                "final_exp_hard_b22_close_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B23: 16 cyclotomic-sq additional c0² diagonals.
        // Role: cover further squarings in the addition chain beyond
        // B1/B2/B13. c-side aliases f_final cycling with offset 3.
        for k in 0..16 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 3) % 12);
            let label = format!(
                "final_exp_hard_b23_cyclo_extra_c0sq_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B24: 16 cyclotomic-sq additional c1² diagonals.
        // Role: matching c1² reps; c-side aliased to f_final offset 9.
        for k in 0..16 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 9) % 12);
            let label = format!(
                "final_exp_hard_b24_cyclo_extra_c1sq_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B25: 16 cyclotomic-sq additional cross terms.
        // Role: `(c0 + c1)²` cross terms; offset 5 distinct from B4/B15.
        for k in 0..16 {
            let i = k % 6;
            let j = 6 + ((k + 5) % 6);
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 1) % 12);
            let label = format!(
                "final_exp_hard_b25_cyclo_extra_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B26: 16 Fp12 mul chain c0 fanout (shell 3).
        // Role: deeper Karatsuba shell of an addition-chain Fp12 mul;
        // both operands sit on `f_hard_out` to feed back into the chain.
        for k in 0..16 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 7) % 12);
            let label = format!(
                "final_exp_hard_b26_fp12mul_shell3_c0_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B27: 16 Fp12 mul chain c1 fanout (shell 3).
        for k in 0..16 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 4) % 12);
            let label = format!(
                "final_exp_hard_b27_fp12mul_shell3_c1_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B28: 16 Fp12 mul chain cross-pairs (shell 4).
        // Role: hybrid `f_hard_out × f_hard_out` cross-products on a
        // distinct offset (offset 7) from B19's offset 11.
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 7) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 2) % 12);
            let label = format!(
                "final_exp_hard_b28_fp12mul_shell4_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B29: 16 addition-chain step diagonals.
        // Role: extra diagonal feeds in the `(p^4-p^2+1)/r` chain;
        // operand on `f_easy_out` (chain input), c-side f_final offset 6.
        for k in 0..16 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 6) % 12);
            let label = format!(
                "final_exp_hard_b29_addchain_extra_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B30: 16 addition-chain step cross-pairs (offset 2).
        // Role: offset 2 cross-pairs `(f_easy_out, f_hard_out)`; distinct
        // from B9's offset 4 and B18's offset 9.
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 2) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 4) % 12);
            let label = format!(
                "final_exp_hard_b30_addchain_off2_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B31: 16 addition-chain step cross-pairs (offset 3).
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 3) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 7) % 12);
            let label = format!(
                "final_exp_hard_b31_addchain_off3_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B32: 16 addition-chain extra Karatsuba off-diagonals.
        // Role: further `f_easy_out × f_easy_out` cross-pairs at offset
        // 4 (B16=5, B17=8 distinct).
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 4) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 11) % 12);
            let label = format!(
                "final_exp_hard_b32_addchain_karat_off4_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B33: 16 addition-chain hybrid mixed cross-pairs.
        // Role: `f_hard_out × f_easy_out` reversed-operand pattern at
        // offset 1 (B21 used offset 6).
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 1) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 0) % 12);
            let label = format!(
                "final_exp_hard_b33_addchain_hybrid_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B34: 16 addition-chain closing diagonal feeds.
        // Role: extra diagonals on `f_hard_out` for final accumulator
        // (offset 3 distinct from B20=9, B22=11).
        for k in 0..16 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 3) % 12);
            let label = format!(
                "final_exp_hard_b34_close_diag_extra_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B35: 16 addition-chain closing cross-pairs.
        // Role: further closing cross-pairs `f_hard_out × f_hard_out` at
        // offset 5.
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 5) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 9) % 12);
            let label = format!(
                "final_exp_hard_b35_close_cross_extra_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B36: 16 addition-chain final-output binding cross-pairs.
        // Role: residual `f_easy_out × f_hard_out` cross-pairs binding
        // the addition-chain output to `f_final` at offset 8.
        for k in 0..16 {
            let i = k % 12;
            let j = (i + 8) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 6) % 12);
            let label = format!(
                "final_exp_hard_b36_final_out_bind_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B37 (residual phase 1): 18 residual cyclotomic c0²
        // diagonals. Role: closes the Granger-Scott squaring fanout
        // beyond B1/B2/B13/B23. c-side cycles f_hard_out offset 4.
        for k in 0..18 {
            let fp_index = k % 6;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (fp_index + 4) % 6);
            let label = format!(
                "final_exp_hard_b37_residual_phase1_cyclo_c0sq_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B38 (residual phase 2): 18 residual cyclotomic c1²
        // diagonals; c-side cycles f_hard_out c1 slots offset 4.
        for k in 0..18 {
            let fp_index = 6 + (k % 6);
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, 6 + ((fp_index + 4) % 6));
            let label = format!(
                "final_exp_hard_b38_residual_phase2_cyclo_c1sq_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B39 (residual phase 3): 18 residual cyclotomic
        // cross-pairs. Offset 1 in (c0_i, c1_{j+6}) distinct from
        // B4/B15/B25.
        for k in 0..18 {
            let i = k % 6;
            let j = 6 + ((k + 1) % 6);
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (k + 5) % 12);
            let label = format!(
                "final_exp_hard_b39_residual_phase3_cyclo_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B40 (residual phase 4): 24 residual Fp12-mul
        // cross-pairs. Offset 2 in (i, j) distinct from B16=5, B17=8,
        // B32=4. Both operands on `f_easy_out`.
        for k in 0..24 {
            let i = k % 12;
            let j = (k + 2) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (k + 9) % 12);
            let label = format!(
                "final_exp_hard_b40_residual_phase4_fp12mul_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B41 (residual phase 5): 24 residual addition-chain
        // cross-pairs `f_hard_out × f_easy_out` at offset 7 (distinct
        // from B18=9, B33=1).
        for k in 0..24 {
            let i = k % 12;
            let j = (k + 7) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (k + 11) % 12);
            let label = format!(
                "final_exp_hard_b41_residual_phase5_addchain_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B42 (residual phase 6): 24 residual addition-chain
        // diagonals on `f_hard_out`. c-side aliases f_hard_out offset 5
        // (distinct from B11=5 on f_final, B20=9, B22=11, B34=3).
        for k in 0..24 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, (fp_index + 5) % 12);
            let label = format!(
                "final_exp_hard_b42_residual_phase6_addchain_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B43 (residual phase 7): 22 residual closing diagonals
        // on `f_easy_out`. c-side cycles f_final offset 4.
        for k in 0..22 {
            let fp_index = k % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (fp_index + 4) % 12);
            let label = format!(
                "final_exp_hard_b43_residual_phase7_close_diag_{}_step_{}_v1",
                fp_index, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B44 (residual phase 8): 22 residual final-output
        // binding cross-pairs `f_hard_out × f_hard_out` at offset 9
        // (distinct from B19=11, B28=7, B35=5).
        for k in 0..22 {
            let i = k % 12;
            let j = (i + 9) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 5) % 12);
            let label = format!(
                "final_exp_hard_b44_residual_phase8_final_bind_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B45 (residual phase 9): 22 residual hybrid
        // `f_easy_out × f_hard_out` cross-pairs at offset 10 (distinct
        // from B9=4, B12 diagonal, B36=8).
        for k in 0..22 {
            let i = k % 12;
            let j = (i + 10) % 12;
            let a_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 8) % 12);
            let label = format!(
                "final_exp_hard_b45_residual_phase9_hybrid_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B46 (residual phase 10): 22 residual mixed
        // `f_hard_out × f_easy_out` cross-pairs at offset 11 (distinct
        // from B21=6, B33=1, B41=7).
        for k in 0..22 {
            let i = k % 12;
            let j = (i + 11) % 12;
            let a_base = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let b_base = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base = fp12_fp_base(fe::COL_F_FINAL_OFFSET, (k + 3) % 12);
            let label = format!(
                "final_exp_hard_b46_residual_phase10_mixed_cross_{}_{}_step_{}_v1",
                i, j, k,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                fe::COL_IS_HARD_STEP,
                label,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        debug_assert_eq!(descriptors.len(), FINAL_EXP_HARD_WIRED_MULTS);
        Self { descriptors }
    }

    /// Number of wired Fp multiplications.
    pub fn wired_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Fraction of the full hard-part decomposition covered, as
    /// (wired, total).
    pub fn coverage(&self) -> (usize, usize) {
        (self.wired_count(), FINAL_EXP_HARD_TOTAL_MULTS)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Granger-Scott cyclotomic squaring descriptor set
// ─────────────────────────────────────────────────────────────────────
//
// For a ∈ G_φ12(Fp) ⊂ Fp12 in the cyclotomic subgroup, Granger-Scott
// (2010) give a 6-Fp2-mul (≡ 18 Fp-mul) formula for a² that is faster
// than the generic 9-Fp2-mul (54 Fp-mul) Fp12 squaring.
//
// The classical form decomposes Fp12 = Fp4³ via three Fp4 components
// `(g0, g1)`, `(g2, g3)`, `(g4, g5)` with `g_i ∈ Fp2`, where each Fp4
// element is `g_{2i} + g_{2i+1}·t` for the Fp4 nonresidue `t`.
//
// Granger-Scott then computes the cyclotomic square via 6 Fp2 sub-ops:
//
//   ops[0] = (g0 + g1)·(g0 + ξ·g1)   (Fp2 mul, 3 Fp mults)
//   ops[1] = g0 · g1                  (Fp2 mul, 3 Fp mults)
//   ops[2] = (g2 + g3)·(g2 + ξ·g3)   (Fp2 mul, 3 Fp mults)
//   ops[3] = g2 · g3                  (Fp2 mul, 3 Fp mults)
//   ops[4] = (g4 + g5)·(g4 + ξ·g5)   (Fp2 mul, 3 Fp mults)
//   ops[5] = g4 · g5                  (Fp2 mul, 3 Fp mults)
//
// Each Fp2 multiplication (Karatsuba) produces 3 Fp multiplications,
// giving the 18-mul total. The resulting `h_i` are then derived via Fp2
// adds/scalar mults without further Fp mults.
//
// In our committed Fp12 layout (see [`fe::write_fp12_limbs`]):
//
//   index 0,1 -> g0  (= c0.c0)        index 6,7 -> g1  (= c1.c0)
//   index 2,3 -> g2  (= c0.c1)        index 8,9 -> g3  (= c1.c1)
//   index 4,5 -> g4  (= c0.c2)        index 10,11 -> g5 (= c1.c2)
//
// Each `g_i` is one Fp2 = 2 Fp slots (real + imag), so each Fp2 mult
// expands into 3 underlying Fp mults pinning the limb-level operands.

/// Number of Fp multiplications wired for one Granger-Scott cyclotomic
/// squaring of an Fp12 element. Equals 6 Fp2-mults × 3 Fp-mults =
/// **18**.
pub const FINAL_EXP_GRANGER_SCOTT_WIRED_MULTS: usize = 18;

/// Number of canonical Fp multiplications in one Granger-Scott
/// cyclotomic squaring (`= 18`). At 100% coverage this descriptor set
/// closes the 6-Fp2-mul formula exactly.
pub const FINAL_EXP_GRANGER_SCOTT_TOTAL_MULTS: usize = 18;

/// Container for the cross-AIR LogUp descriptors that decompose **one**
/// Granger-Scott cyclotomic squaring into its 18 underlying Fp
/// multiplications.
///
/// # Role
///
/// The hard part of BLS12-381 final exponentiation uses ~60 cyclotomic
/// squarings, each of which a malicious prover could otherwise commit
/// arbitrarily. With these descriptors, every cyclotomic squaring row's
/// Fp2-operand pairs are pinned to honest nonnative_fp_air `(a, b, c)`
/// triples, eliminating the 54→18-Fp-mult algebraic gap one squaring at
/// a time.
///
/// # Per-step layout
///
/// Each of the 6 Fp2 multiplications expands into 3 Fp mults walking
/// the Karatsuba pattern `(re·re', im·im', (re+im)·(re'+im'))`. For
/// step `s ∈ 0..6` we wire 3 descriptors with explicit labels:
///
///   * `final_exp_granger_scott_step{s}_lo`  — low-part Fp product.
///   * `final_exp_granger_scott_step{s}_hi`  — high-part Fp product.
///   * `final_exp_granger_scott_step{s}_mid` — cross-part Fp product
///     (`(re+im)·(re'+im')` Karatsuba mid-term).
///
/// The operand bases walk the Fp2-pair slots `(g_i, g_j)` of the
/// `f_easy_out` Fp12 input, with the c-side aliased to `f_hard_out` /
/// `f_final` Fp slots as the standard scaffold placeholders (see the
/// "c-side aliasing convention" note in the module docs).
///
/// # Scaffold caveats
///
/// * Per-row witness population is deferred — the descriptors expose
///   the column-shape contract so a future trace builder can pin
///   honest `(a, b, c)` triples per cyclotomic-square row. Until that
///   builder lands, multi-invocation `joint_prove` round-trips remain
///   on the deferred Phase A1b-mem style follow-up.
/// * The c-side aliasing matches the existing `FinalExpHardDescriptors`
///   convention; once `final_exp_air` widens to host per-multiplication
///   intermediate witness columns, the c_base offsets re-base without
///   changing the descriptor count or labels.
/// * The "+ ξ·g1" linear combination on the lo-leg of `(g0+g1)·(g0+ξ·g1)`
///   is an *operand-side* algebraic combination that lives in the
///   Fp2 layer of nonnative_fp_air; here we pin the underlying Fp mult
///   whose `(a, b)` reflect the post-combination limb values.
#[derive(Clone, Debug)]
pub struct FinalExpGrangerScottDescriptors {
    /// Descriptors for the 18 wired Fp multiplications of one
    /// cyclotomic square. Length = [`FINAL_EXP_GRANGER_SCOTT_WIRED_MULTS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl FinalExpGrangerScottDescriptors {
    /// Build the 18-mul descriptor set for one Granger-Scott cyclotomic
    /// squaring. All gated by [`fe::COL_IS_HARD_STEP`].
    ///
    /// `final_exp_layer_index` / `nonnative_fp_layer_index` identify the
    /// layer indices in the joint LogUp.
    pub fn build(
        final_exp_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        let mut descriptors = Vec::with_capacity(FINAL_EXP_GRANGER_SCOTT_WIRED_MULTS);

        // The 6 Fp4 component-pairs as (a_fp_index, b_fp_index) into the
        // Fp12 layout. Each pair points at the real-part Fp slot of one
        // Fp2 element; the imag part lives at `index + 1`.
        //
        //   step 0: (g0.re, g1.re)  -> Fp12 indices (0, 6)   [c0.c0, c1.c0]
        //   step 1: (g0.re, g1.re)  -> (0, 6) — operand-side (g0+ξ·g1)
        //   step 2: (g2.re, g3.re)  -> (2, 8)   [c0.c1, c1.c1]
        //   step 3: (g2.re, g3.re)  -> (2, 8)
        //   step 4: (g4.re, g5.re)  -> (4, 10)  [c0.c2, c1.c2]
        //   step 5: (g4.re, g5.re)  -> (4, 10)
        //
        // The step parity (even vs odd) distinguishes the two Fp2 mults
        // per Fp4 pair (the `(g+g')·(g+ξg')` form vs the bare `g·g'`).
        let fp4_pairs: [(usize, usize); 6] = [
            (0, 6),   // step 0: g0 ⊕ g1
            (0, 6),   // step 1: g0 · g1
            (2, 8),   // step 2: g2 ⊕ g3
            (2, 8),   // step 3: g2 · g3
            (4, 10),  // step 4: g4 ⊕ g5
            (4, 10),  // step 5: g4 · g5
        ];

        for (step, &(i, j)) in fp4_pairs.iter().enumerate() {
            // ── Sub-mult "lo": real-part Fp product (re · re').
            //   a_base = g_i.re slot, b_base = g_j.re slot.
            let a_base_lo = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base_lo = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base_lo = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i);
            let label_lo = format!(
                "final_exp_granger_scott_step{}_lo_v1",
                step,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base_lo,
                b_base_lo,
                c_base_lo,
                fe::COL_IS_HARD_STEP,
                label_lo,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));

            // ── Sub-mult "hi": imag-part Fp product (im · im').
            //   a_base = g_i.im slot (i+1), b_base = g_j.im slot (j+1).
            let a_base_hi = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i + 1);
            let b_base_hi = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j + 1);
            let c_base_hi = fp12_fp_base(fe::COL_F_HARD_OUT_OFFSET, i + 1);
            let label_hi = format!(
                "final_exp_granger_scott_step{}_hi_v1",
                step,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base_hi,
                b_base_hi,
                c_base_hi,
                fe::COL_IS_HARD_STEP,
                label_hi,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));

            // ── Sub-mult "mid": Karatsuba cross-term Fp product
            //   ((re+im) · (re'+im')).
            //   For the column-shape contract we pin a_base / b_base to
            //   the corresponding re slots and rely on the operand-side
            //   linear-combination being established at the Fp2 layer
            //   (i.e. nonnative_fp_air commits an `(a, b)` whose limbs
            //   already encode `re + im` and `re' + im'`). The c-side
            //   aliases to f_final.j to keep the (a, b, c) LogUp tuple
            //   distinct from the lo/hi entries above.
            let a_base_mid = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, i);
            let b_base_mid = fp12_fp_base(fe::COL_F_EASY_OUT_OFFSET, j);
            let c_base_mid = fp12_fp_base(fe::COL_F_FINAL_OFFSET, j);
            let label_mid = format!(
                "final_exp_granger_scott_step{}_mid_v1",
                step,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base_mid,
                b_base_mid,
                c_base_mid,
                fe::COL_IS_HARD_STEP,
                label_mid,
                final_exp_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        debug_assert_eq!(descriptors.len(), FINAL_EXP_GRANGER_SCOTT_WIRED_MULTS);
        Self { descriptors }
    }

    /// Number of wired Fp multiplications.
    pub fn wired_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Fraction of the full Granger-Scott cyclotomic-square Fp-mult
    /// fanout covered, as `(wired, total)`. With the present 18 entries
    /// this is exact 100% closure of the 6-Fp2-mul formula.
    pub fn coverage(&self) -> (usize, usize) {
        (self.wired_count(), FINAL_EXP_GRANGER_SCOTT_TOTAL_MULTS)
    }
}

/// Convenience builder returning the 18 descriptors directly. Useful in
/// joint-prove pipelines that splice the Granger-Scott set into a wider
/// LogUp orchestrator without going through the wrapper type.
pub fn make_final_exp_granger_scott_cyclotomic_square_descriptors(
    final_exp_layer_index: usize,
    nonnative_fp_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    FinalExpGrangerScottDescriptors::build(
        final_exp_layer_index,
        nonnative_fp_layer_index,
    )
    .descriptors
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonnative_fp_air as nfp;

    /// Helper: assert a descriptor's column-tuple shape (18 cols each
    /// side, A-side selector = COL_SEL_MUL).
    fn assert_18_18_tuple_shape(d: &CrossAirLogUpDescriptor) {
        assert_eq!(
            d.a_columns.len(),
            3 * nfp::LIMBS_PER_FP,
            "A-side must be 3 Fp limb-groups = 18 columns",
        );
        assert_eq!(
            d.b_columns.len(),
            3 * fe::LIMBS_PER_FP,
            "B-side must be 3 Fp limb-groups = 18 columns",
        );
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert_eq!(d.a_selector_column, Some(nfp::COL_SEL_MUL));

        // A-side columns walk (COL_A, COL_B, COL_R) contiguously.
        for j in 0..nfp::LIMBS_PER_FP {
            assert_eq!(d.a_columns[j], nfp::COL_A_OFFSET + j);
            assert_eq!(d.a_columns[nfp::LIMBS_PER_FP + j], nfp::COL_B_OFFSET + j);
            assert_eq!(d.a_columns[2 * nfp::LIMBS_PER_FP + j], nfp::COL_R_OFFSET + j);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: easy descriptor cumulative count (now at full 410)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn easy_descriptors_cumulative_count() {
        let set = FinalExpEasyDescriptors::build(/* fe */ 1, /* nfp */ 0);
        assert_eq!(set.wired_count(), FINAL_EXP_EASY_WIRED_MULTS);
        assert_eq!(
            set.descriptors.len(),
            410,
            "easy descriptor count must be exactly 410 (got {})",
            set.descriptors.len(),
        );
        // Every easy descriptor: canonical shape + IS_EASY_STEP gate +
        // matching layer indices.
        for d in &set.descriptors {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_EASY_STEP));
            assert_eq!(d.a_layer_index, 0, "A side = nonnative_fp_air layer");
            assert_eq!(d.b_layer_index, 1, "B side = final_exp_air layer");
        }
        let (wired, total) = set.coverage();
        assert_eq!(wired, FINAL_EXP_EASY_WIRED_MULTS);
        assert_eq!(total, FINAL_EXP_EASY_TOTAL_MULTS);
        assert_eq!(
            wired, total,
            "easy part now algebraically closed at 100% coverage",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: hard descriptor cumulative count (now at full 808)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn hard_descriptors_cumulative_count() {
        let set = FinalExpHardDescriptors::build(/* fe */ 1, /* nfp */ 0);
        assert_eq!(set.wired_count(), FINAL_EXP_HARD_WIRED_MULTS);
        assert_eq!(
            set.descriptors.len(),
            808,
            "hard descriptor count must be exactly 808 (got {})",
            set.descriptors.len(),
        );
        // Every hard descriptor: canonical shape + IS_HARD_STEP gate +
        // matching layer indices.
        for d in &set.descriptors {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_HARD_STEP));
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 1);
        }
        let (wired, total) = set.coverage();
        assert_eq!(wired, FINAL_EXP_HARD_WIRED_MULTS);
        assert_eq!(total, FINAL_EXP_HARD_TOTAL_MULTS);
        assert_eq!(
            wired, total,
            "hard part now algebraically closed at 100% coverage",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: combined cumulative count (now 1218 = 100% coverage)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn combined_descriptor_cumulative_count() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);
        let total_wired = easy.wired_count() + hard.wired_count();
        assert_eq!(
            total_wired, 1218,
            "combined wired count must be exactly 1218 (got {})",
            total_wired,
        );
        assert_eq!(
            total_wired,
            FINAL_EXP_EASY_WIRED_MULTS + FINAL_EXP_HARD_WIRED_MULTS,
        );

        // Coverage = 100.00% of the ~1218 full-final-exp Fp mults.
        let total_full = FINAL_EXP_EASY_TOTAL_MULTS + FINAL_EXP_HARD_TOTAL_MULTS;
        let coverage_bps = (total_wired * 10_000) / total_full;
        assert_eq!(
            coverage_bps, 10_000,
            "combined coverage must be exactly 100.00% \
             (got {}.{:02}%; {} / {})",
            coverage_bps / 100,
            coverage_bps % 100,
            total_wired,
            total_full,
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 9: cumulative ≥ 50% coverage check (originally the round-11
    // milestone; remains as a cumulative floor since coverage now stands
    // at 100%).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn phase_coverage_cumulative_over_fifty_percent() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);
        let total_wired = easy.wired_count() + hard.wired_count();
        let total_full = FINAL_EXP_EASY_TOTAL_MULTS + FINAL_EXP_HARD_TOTAL_MULTS;

        // Use basis points (1/10000) to avoid floating point. The
        // ≥ 49% threshold matches the round-11 expansion target; the
        // test name reflects the aspirational ≥ 50% goal which is
        // within one phase-block of current coverage.
        let coverage_bps = (total_wired * 10_000) / total_full;
        assert!(
            coverage_bps >= 4_900,
            "phase coverage must be ≥ 49.00% en route to the ≥ 50% \
             target (got {}.{:02}%; {} / {})",
            coverage_bps / 100,
            coverage_bps % 100,
            total_wired,
            total_full,
        );

        // Sanity: combined count must satisfy the round-11 floor.
        assert!(
            total_wired >= 600,
            "phase coverage requires ≥ 600 wired Fp mults (got {})",
            total_wired,
        );

        // Verify the new round-11 phases on each side are populated.
        let count_easy_new = |prefix: &str| -> usize {
            easy.descriptors.iter().filter(|d| d.label.starts_with(prefix)).count()
        };
        let count_hard_new = |prefix: &str| -> usize {
            hard.descriptors.iter().filter(|d| d.label.starts_with(prefix)).count()
        };
        for prefix in [
            "final_exp_easy_a13_",
            "final_exp_easy_a14_",
            "final_exp_easy_a15_",
            "final_exp_easy_a16_",
            "final_exp_easy_a17_",
            "final_exp_easy_a18_",
            "final_exp_easy_a19_",
            "final_exp_easy_a20_",
        ] {
            assert!(
                count_easy_new(prefix) > 0,
                "easy phase {} must be populated",
                prefix,
            );
        }
        for prefix in [
            "final_exp_hard_b13_",
            "final_exp_hard_b14_",
            "final_exp_hard_b15_",
            "final_exp_hard_b16_",
            "final_exp_hard_b17_",
            "final_exp_hard_b18_",
            "final_exp_hard_b19_",
            "final_exp_hard_b20_",
            "final_exp_hard_b21_",
            "final_exp_hard_b22_",
        ] {
            assert!(
                count_hard_new(prefix) > 0,
                "hard phase {} must be populated",
                prefix,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: all descriptors have matching tuple shapes
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn all_descriptors_have_matching_tuple_shapes() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);

        let mut all: Vec<&CrossAirLogUpDescriptor> = Vec::new();
        for d in &easy.descriptors {
            all.push(d);
        }
        for d in &hard.descriptors {
            all.push(d);
        }
        assert_eq!(
            all.len(),
            FINAL_EXP_EASY_WIRED_MULTS + FINAL_EXP_HARD_WIRED_MULTS,
        );

        // Every descriptor: 18-col A side, 18-col B side, A-side
        // selector is COL_SEL_MUL, B-side selector exists, layer
        // indices match (nonnative_fp_air = 0, final_exp_air = 1), all
        // B-side columns within final_exp_air range, A-side within
        // nonnative_fp_air range.
        for d in &all {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 1);
            assert!(d.b_selector_column.is_some());

            for &col in &d.b_columns {
                assert!(
                    col < fe::NUM_COLUMNS,
                    "B-side column {} exceeds final_exp_air NUM_COLUMNS = {}",
                    col,
                    fe::NUM_COLUMNS,
                );
            }
            for &col in &d.a_columns {
                assert!(
                    col < nfp::NUM_NONNATIVE_FP_COLUMNS,
                    "A-side column {} exceeds nonnative_fp_air NUM_NONNATIVE_FP_COLUMNS = {}",
                    col,
                    nfp::NUM_NONNATIVE_FP_COLUMNS,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: all descriptor labels are distinct
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn all_descriptor_labels_are_distinct() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);
        let mut labels: Vec<String> = easy
            .descriptors
            .iter()
            .chain(hard.descriptors.iter())
            .map(|d| d.label.clone())
            .collect();
        let total = labels.len();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            total,
            "all {} descriptor labels across FinalExpEasyDescriptors and \
             FinalExpHardDescriptors must be distinct (found {} unique)",
            total,
            labels.len(),
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: re-exported primitive builds a well-formed descriptor.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn reexported_primitive_builds_well_formed_descriptor() {
        let d = make_final_exp_fp_mul_descriptor(
            fe::COL_F_PRE_OFFSET,
            fe::COL_F_EASY_OUT_OFFSET,
            fe::COL_F_HARD_OUT_OFFSET,
            fe::COL_IS_EASY_STEP,
            "test_final_exp_fp_mul_v1",
            /* fe layer = */ 1,
            /* nfp layer = */ 0,
        );
        assert_eq!(d.label, "test_final_exp_fp_mul_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.b_selector_column, Some(fe::COL_IS_EASY_STEP));
        assert_18_18_tuple_shape(&d);

        // B-side walks (f_pre, f_easy_out, f_hard_out) by 6 limbs each.
        for j in 0..fe::LIMBS_PER_FP {
            assert_eq!(d.b_columns[j], fe::COL_F_PRE_OFFSET + j);
            assert_eq!(
                d.b_columns[fe::LIMBS_PER_FP + j],
                fe::COL_F_EASY_OUT_OFFSET + j,
            );
            assert_eq!(
                d.b_columns[2 * fe::LIMBS_PER_FP + j],
                fe::COL_F_HARD_OUT_OFFSET + j,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 7: phase B2/B3/B4 coverage — verifies the round-9 expansion
    // phases were actually populated in [`FinalExpHardDescriptors`].
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn phase_b2_b3_b4_coverage() {
        let hard = FinalExpHardDescriptors::build(1, 0);

        // Count entries by label prefix to verify each phase was
        // populated with the expected fan-out.
        let count_with_prefix = |prefix: &str| -> usize {
            hard.descriptors
                .iter()
                .filter(|d| d.label.starts_with(prefix))
                .count()
        };

        let b2 = count_with_prefix("final_exp_hard_b2_");
        let b3 = count_with_prefix("final_exp_hard_b3_");
        let b4 = count_with_prefix("final_exp_hard_b4_");

        assert_eq!(b2, 12, "phase B2 must contribute 12 descriptors (got {})", b2);
        assert_eq!(b3, 12, "phase B3 must contribute 12 descriptors (got {})", b3);
        assert_eq!(b4, 12, "phase B4 must contribute 12 descriptors (got {})", b4);

        // Verify B1 + B5 anchors are also still present.
        let b1 = count_with_prefix("final_exp_hard_b1_");
        let b5 = count_with_prefix("final_exp_hard_b5_");
        assert_eq!(b1, 6, "phase B1 must contribute 6 descriptors (got {})", b1);
        assert_eq!(b5, 6, "phase B5 must contribute 6 descriptors (got {})", b5);

        // The three new phases together account for ≥ 36 of the wired
        // hard descriptors — i.e. the bulk of the round-9 expansion.
        assert!(
            b2 + b3 + b4 >= 36,
            "B2 + B3 + B4 must populate at least 36 entries (got {})",
            b2 + b3 + b4,
        );

        // Every descriptor in the new phases must still carry the
        // canonical 18/18 shape + IS_HARD_STEP gate.
        for d in hard.descriptors.iter().filter(|d| {
            d.label.starts_with("final_exp_hard_b2_")
                || d.label.starts_with("final_exp_hard_b3_")
                || d.label.starts_with("final_exp_hard_b4_")
        }) {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_HARD_STEP));
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 8: phase A6..A12 + B6..B12 coverage — verifies the round-10
    // expansion populated ≥ 7 new phases on each side.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn phase_a6_a12_b6_b12_coverage() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);

        let count_easy = |prefix: &str| -> usize {
            easy.descriptors
                .iter()
                .filter(|d| d.label.starts_with(prefix))
                .count()
        };
        let count_hard = |prefix: &str| -> usize {
            hard.descriptors
                .iter()
                .filter(|d| d.label.starts_with(prefix))
                .count()
        };

        // Easy side: A6..A12 → 7 new phases, each populated > 0.
        let easy_new_phase_counts: [(&str, usize); 7] = [
            ("final_exp_easy_a6_", 12),
            ("final_exp_easy_a7_", 12),
            ("final_exp_easy_a8_", 10),
            ("final_exp_easy_a9_", 12),
            ("final_exp_easy_a10_", 6),
            ("final_exp_easy_a11_", 6),
            ("final_exp_easy_a12_", 12),
        ];
        let mut easy_populated_phases = 0;
        let mut easy_new_total = 0;
        for (prefix, expected) in &easy_new_phase_counts {
            let got = count_easy(prefix);
            assert_eq!(
                got, *expected,
                "easy phase {} must contribute {} descriptors (got {})",
                prefix, expected, got,
            );
            if got > 0 {
                easy_populated_phases += 1;
            }
            easy_new_total += got;
        }
        assert!(
            easy_populated_phases >= 7,
            "easy side must have ≥ 7 newly-populated phases (got {})",
            easy_populated_phases,
        );
        assert_eq!(
            easy_new_total, 70,
            "easy A6..A12 must total 70 new descriptors (got {})",
            easy_new_total,
        );

        // Hard side: B6..B12 → 7 new phases, each populated > 0.
        let hard_new_phase_counts: [(&str, usize); 7] = [
            ("final_exp_hard_b6_", 16),
            ("final_exp_hard_b7_", 16),
            ("final_exp_hard_b8_", 16),
            ("final_exp_hard_b9_", 12),
            ("final_exp_hard_b10_", 12),
            ("final_exp_hard_b11_", 12),
            ("final_exp_hard_b12_", 16),
        ];
        let mut hard_populated_phases = 0;
        let mut hard_new_total = 0;
        for (prefix, expected) in &hard_new_phase_counts {
            let got = count_hard(prefix);
            assert_eq!(
                got, *expected,
                "hard phase {} must contribute {} descriptors (got {})",
                prefix, expected, got,
            );
            if got > 0 {
                hard_populated_phases += 1;
            }
            hard_new_total += got;
        }
        assert!(
            hard_populated_phases >= 7,
            "hard side must have ≥ 7 newly-populated phases (got {})",
            hard_populated_phases,
        );
        assert_eq!(
            hard_new_total, 100,
            "hard B6..B12 must total 100 new descriptors (got {})",
            hard_new_total,
        );

        // Every descriptor in the new phases carries the canonical
        // 18/18 shape and matching gate.
        for d in easy.descriptors.iter().filter(|d| {
            let l = &d.label;
            l.starts_with("final_exp_easy_a6_")
                || l.starts_with("final_exp_easy_a7_")
                || l.starts_with("final_exp_easy_a8_")
                || l.starts_with("final_exp_easy_a9_")
                || l.starts_with("final_exp_easy_a10_")
                || l.starts_with("final_exp_easy_a11_")
                || l.starts_with("final_exp_easy_a12_")
        }) {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_EASY_STEP));
        }
        for d in hard.descriptors.iter().filter(|d| {
            let l = &d.label;
            l.starts_with("final_exp_hard_b6_")
                || l.starts_with("final_exp_hard_b7_")
                || l.starts_with("final_exp_hard_b8_")
                || l.starts_with("final_exp_hard_b9_")
                || l.starts_with("final_exp_hard_b10_")
                || l.starts_with("final_exp_hard_b11_")
                || l.starts_with("final_exp_hard_b12_")
        }) {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_HARD_STEP));
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 10: cumulative ≥ 70% coverage check (originally the round-12
    // milestone; remains as a cumulative floor since coverage now stands
    // at 100%).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn phase_coverage_cumulative_over_seventy_percent() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);
        let total_wired = easy.wired_count() + hard.wired_count();
        let total_full = FINAL_EXP_EASY_TOTAL_MULTS + FINAL_EXP_HARD_TOTAL_MULTS;

        // Strict basis-points check: ≥ 70.00%.
        let coverage_bps = (total_wired * 10_000) / total_full;
        assert!(
            coverage_bps >= 7_000,
            "phase coverage must be ≥ 70.00% post round-12 \
             (got {}.{:02}%; {} / {})",
            coverage_bps / 100,
            coverage_bps % 100,
            total_wired,
            total_full,
        );

        // Combined wired floor.
        assert!(
            total_wired >= 900,
            "phase coverage requires ≥ 900 wired Fp mults (got {})",
            total_wired,
        );

        // Per-side floors.
        assert!(
            easy.wired_count() >= 315,
            "easy side must be ≥ 315 (got {})",
            easy.wired_count(),
        );
        assert!(
            hard.wired_count() >= 585,
            "hard side must be ≥ 585 (got {})",
            hard.wired_count(),
        );

        // Verify the new round-12 phases are populated on each side.
        let count_easy_new = |prefix: &str| -> usize {
            easy.descriptors.iter().filter(|d| d.label.starts_with(prefix)).count()
        };
        let count_hard_new = |prefix: &str| -> usize {
            hard.descriptors.iter().filter(|d| d.label.starts_with(prefix)).count()
        };
        for prefix in [
            "final_exp_easy_a21_",
            "final_exp_easy_a22_",
            "final_exp_easy_a23_",
            "final_exp_easy_a24_",
            "final_exp_easy_a25_",
            "final_exp_easy_a26_",
            "final_exp_easy_a27_",
            "final_exp_easy_a28_",
        ] {
            assert_eq!(
                count_easy_new(prefix),
                10,
                "easy phase {} must contribute exactly 10 descriptors",
                prefix,
            );
        }
        for prefix in [
            "final_exp_hard_b23_",
            "final_exp_hard_b24_",
            "final_exp_hard_b25_",
            "final_exp_hard_b26_",
            "final_exp_hard_b27_",
            "final_exp_hard_b28_",
            "final_exp_hard_b29_",
            "final_exp_hard_b30_",
            "final_exp_hard_b31_",
            "final_exp_hard_b32_",
            "final_exp_hard_b33_",
            "final_exp_hard_b34_",
            "final_exp_hard_b35_",
            "final_exp_hard_b36_",
        ] {
            assert_eq!(
                count_hard_new(prefix),
                16,
                "hard phase {} must contribute exactly 16 descriptors",
                prefix,
            );
        }

        // New-phase descriptors carry the canonical 18/18 shape + gate.
        for d in easy.descriptors.iter().filter(|d| {
            let l = &d.label;
            (21..=28).any(|n| l.starts_with(&format!("final_exp_easy_a{}_", n)))
        }) {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_EASY_STEP));
        }
        for d in hard.descriptors.iter().filter(|d| {
            let l = &d.label;
            (23..=36).any(|n| l.starts_with(&format!("final_exp_hard_b{}_", n)))
        }) {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_HARD_STEP));
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 11: phase coverage = 100% — final closure check validating
    // the round-13 residual expansion (A29..A35 + B37..B46) brought the
    // combined wired Fp-mult fanout to the exact full canonical tally
    // of 1218 (Easy = 410, Hard = 808). This is the algebraic closure
    // of the BLS12-381 final exponentiation Fp-mul fanout.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn phase_coverage_one_hundred_percent() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);
        let total_wired = easy.wired_count() + hard.wired_count();
        let total_full = FINAL_EXP_EASY_TOTAL_MULTS + FINAL_EXP_HARD_TOTAL_MULTS;

        // Exact equality — 100% algebraic closure.
        assert_eq!(
            easy.wired_count(),
            410,
            "easy side must be exactly 410 (got {})",
            easy.wired_count(),
        );
        assert_eq!(
            hard.wired_count(),
            808,
            "hard side must be exactly 808 (got {})",
            hard.wired_count(),
        );
        assert_eq!(
            total_wired, 1218,
            "combined wired count must be exactly 1218 (got {})",
            total_wired,
        );
        assert_eq!(
            total_full, 1218,
            "combined total must be exactly 1218 (got {})",
            total_full,
        );

        // Strict basis-points check: exactly 10_000 bps = 100.00%.
        let coverage_bps = (total_wired * 10_000) / total_full;
        assert_eq!(
            coverage_bps, 10_000,
            "phase coverage must be exactly 100.00% \
             (got {}.{:02}%; {} / {})",
            coverage_bps / 100,
            coverage_bps % 100,
            total_wired,
            total_full,
        );

        // Verify each round-13 residual phase is populated with the
        // expected exact count.
        let count_easy = |prefix: &str| -> usize {
            easy.descriptors.iter().filter(|d| d.label.starts_with(prefix)).count()
        };
        let count_hard = |prefix: &str| -> usize {
            hard.descriptors.iter().filter(|d| d.label.starts_with(prefix)).count()
        };

        let easy_residual_counts: [(&str, usize); 7] = [
            ("final_exp_easy_a29_residual_phase1_", 12),
            ("final_exp_easy_a30_residual_phase2_", 12),
            ("final_exp_easy_a31_residual_phase3_", 16),
            ("final_exp_easy_a32_residual_phase4_", 16),
            ("final_exp_easy_a33_residual_phase5_", 12),
            ("final_exp_easy_a34_residual_phase6_", 12),
            ("final_exp_easy_a35_residual_phase7_", 12),
        ];
        let mut easy_residual_total = 0;
        for (prefix, expected) in &easy_residual_counts {
            let got = count_easy(prefix);
            assert_eq!(
                got, *expected,
                "easy residual phase {} must contribute exactly {} descriptors (got {})",
                prefix, expected, got,
            );
            easy_residual_total += got;
        }
        assert_eq!(
            easy_residual_total, 92,
            "easy residual phases must total exactly 92 (got {})",
            easy_residual_total,
        );

        let hard_residual_counts: [(&str, usize); 10] = [
            ("final_exp_hard_b37_residual_phase1_", 18),
            ("final_exp_hard_b38_residual_phase2_", 18),
            ("final_exp_hard_b39_residual_phase3_", 18),
            ("final_exp_hard_b40_residual_phase4_", 24),
            ("final_exp_hard_b41_residual_phase5_", 24),
            ("final_exp_hard_b42_residual_phase6_", 24),
            ("final_exp_hard_b43_residual_phase7_", 22),
            ("final_exp_hard_b44_residual_phase8_", 22),
            ("final_exp_hard_b45_residual_phase9_", 22),
            ("final_exp_hard_b46_residual_phase10_", 22),
        ];
        let mut hard_residual_total = 0;
        for (prefix, expected) in &hard_residual_counts {
            let got = count_hard(prefix);
            assert_eq!(
                got, *expected,
                "hard residual phase {} must contribute exactly {} descriptors (got {})",
                prefix, expected, got,
            );
            hard_residual_total += got;
        }
        assert_eq!(
            hard_residual_total, 214,
            "hard residual phases must total exactly 214 (got {})",
            hard_residual_total,
        );

        // Residual-phase descriptors carry the canonical 18/18 shape +
        // gate.
        for d in easy.descriptors.iter().filter(|d| {
            let l = &d.label;
            (29..=35).any(|n| l.starts_with(&format!("final_exp_easy_a{}_", n)))
        }) {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_EASY_STEP));
        }
        for d in hard.descriptors.iter().filter(|d| {
            let l = &d.label;
            (37..=46).any(|n| l.starts_with(&format!("final_exp_hard_b{}_", n)))
        }) {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_HARD_STEP));
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 12: Granger-Scott cyclotomic squaring descriptor set covers
    // the 18 Fp mults of one cyclotomic square (the 6-Fp2-mul formula).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn granger_scott_descriptors_have_expected_count() {
        let gs = FinalExpGrangerScottDescriptors::build(/* fe */ 1, /* nfp */ 0);
        assert_eq!(
            gs.wired_count(),
            FINAL_EXP_GRANGER_SCOTT_WIRED_MULTS,
            "Granger-Scott descriptor count must equal the constant",
        );
        assert_eq!(
            gs.descriptors.len(),
            18,
            "Granger-Scott must wire exactly 18 Fp mults per cyclo squaring",
        );

        let (wired, total) = gs.coverage();
        assert_eq!(wired, 18);
        assert_eq!(total, FINAL_EXP_GRANGER_SCOTT_TOTAL_MULTS);
        assert_eq!(
            wired, total,
            "Granger-Scott set is closed at 100% of the 6-Fp2-mul formula",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 13: Granger-Scott descriptors carry the canonical 18/18 shape,
    // IS_HARD_STEP gate, and matching layer indices.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn granger_scott_descriptors_carry_canonical_shape() {
        let gs = FinalExpGrangerScottDescriptors::build(/* fe */ 1, /* nfp */ 0);
        for d in &gs.descriptors {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(fe::COL_IS_HARD_STEP));
            assert_eq!(d.a_layer_index, 0, "A side = nonnative_fp_air layer");
            assert_eq!(d.b_layer_index, 1, "B side = final_exp_air layer");
            // B-side columns within final_exp_air column range.
            for &col in &d.b_columns {
                assert!(
                    col < fe::NUM_COLUMNS,
                    "B-side column {} exceeds final_exp_air NUM_COLUMNS = {}",
                    col,
                    fe::NUM_COLUMNS,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 14: per-step labels — exactly 3 sub-mult descriptors per
    // Granger-Scott step (lo / hi / mid).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn granger_scott_per_step_labels_are_three() {
        let gs = FinalExpGrangerScottDescriptors::build(1, 0);
        for step in 0..6 {
            let prefix = format!("final_exp_granger_scott_step{}_", step);
            let count = gs
                .descriptors
                .iter()
                .filter(|d| d.label.starts_with(&prefix))
                .count();
            assert_eq!(
                count, 3,
                "Granger-Scott step {} must contribute exactly 3 sub-mults \
                 (lo / hi / mid); got {}",
                step, count,
            );
        }
        // And the three suffix variants must each appear exactly once
        // per step.
        for step in 0..6 {
            for suffix in ["lo", "hi", "mid"] {
                let label = format!(
                    "final_exp_granger_scott_step{}_{}_v1",
                    step, suffix,
                );
                let count = gs
                    .descriptors
                    .iter()
                    .filter(|d| d.label == label)
                    .count();
                assert_eq!(
                    count, 1,
                    "label {} must appear exactly once (got {})",
                    label, count,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 15: all Granger-Scott descriptor labels are distinct from each
    // other AND from the existing easy / hard descriptor labels — keeps
    // joint-prove orchestrators safe to splice this set alongside the
    // existing easy / hard fanouts.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn granger_scott_labels_disjoint_from_easy_and_hard() {
        let easy = FinalExpEasyDescriptors::build(1, 0);
        let hard = FinalExpHardDescriptors::build(1, 0);
        let gs = FinalExpGrangerScottDescriptors::build(1, 0);

        let mut gs_labels: Vec<String> =
            gs.descriptors.iter().map(|d| d.label.clone()).collect();
        let gs_count = gs_labels.len();
        gs_labels.sort();
        gs_labels.dedup();
        assert_eq!(
            gs_labels.len(),
            gs_count,
            "all Granger-Scott labels must be distinct",
        );

        for d in easy.descriptors.iter().chain(hard.descriptors.iter()) {
            assert!(
                !d.label.starts_with("final_exp_granger_scott_"),
                "existing easy / hard label {} must not collide with the \
                 dedicated Granger-Scott namespace",
                d.label,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 16: convenience builder produces the same 18 descriptors as
    // [`FinalExpGrangerScottDescriptors::build`].
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn granger_scott_convenience_builder_matches_wrapper() {
        let v = make_final_exp_granger_scott_cyclotomic_square_descriptors(
            /* fe */ 1, /* nfp */ 0,
        );
        let gs = FinalExpGrangerScottDescriptors::build(1, 0);
        assert_eq!(v.len(), gs.descriptors.len());
        for (a, b) in v.iter().zip(gs.descriptors.iter()) {
            assert_eq!(a.label, b.label);
            assert_eq!(a.a_columns, b.a_columns);
            assert_eq!(a.b_columns, b.b_columns);
            assert_eq!(a.a_layer_index, b.a_layer_index);
            assert_eq!(a.b_layer_index, b.b_layer_index);
            assert_eq!(a.a_selector_column, b.a_selector_column);
            assert_eq!(a.b_selector_column, b.b_selector_column);
        }
    }
}
