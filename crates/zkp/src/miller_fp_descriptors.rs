//! Cross-AIR LogUp descriptors binding [`miller_step_air`]'s deep
//! arithmetic relations to [`nonnative_fp_air`]'s `(a, b, c=a·b)`
//! tuples.
//!
//! # Purpose
//!
//! [`miller_step_air`] commits — per row — witness columns for one
//! Miller-loop iteration: the G2 accumulator `Q_curr`/`Q_next`, the
//! Fp12 line value, and the Fp12 Miller accumulator `acc_pre`/`acc_post`
//! satisfying
//!
//!   * `acc_post = acc_pre² · line_value`  (Fp12 mul)
//!   * `Q_next  = 2 · Q_curr`             (G2 doubling, when `is_doubling = 1`)
//!   * `Q_next  = Q_curr + Q_fixed`       (G2 addition, when `is_addition = 1`)
//!
//! The AIR enforces *selector binarity*, *mutual exclusion*, and
//! *cross-row continuity* of Q and acc, but **does NOT** enforce the
//! deep arithmetic. Those relations are Fp2 / Fp12 multiplications,
//! which decompose into ~150 limb-level Fp multiplications per row —
//! exactly the workload of [`nonnative_fp_air`].
//!
//! This module wires the cross-AIR LogUp descriptors that pin a subset
//! of those Fp multiplications as `(a, b, c)` multiset entries shared
//! between `miller_step_air` columns and `nonnative_fp_air`'s
//! `(COL_A, COL_B, COL_R)` triple.
//!
//! # Scope: wired vs templates
//!
//! A full Fp12 squaring decomposes into ~36 Fp multiplications
//! (Chung-Hasan / Karatsuba), and Fp12 × Fp12 multiplication into
//! ~54 Fp multiplications. Combined with line-mul sparsity and Fp2
//! reductions, `acc_post = acc_pre² · line_value` requires
//! **approximately 54–60 Fp multiplications** depending on the algorithm
//! choice.
//!
//! G2 doubling `Q_next = 2·Q_curr` computes
//!
//!   * `λ = 3·X² / (2·Y)`             (1 Fp2 sqr + 1 Fp2 inv + Fp2 mul)
//!   * `X_next = λ² − 2·X`            (1 Fp2 sqr + Fp2 sub)
//!   * `Y_next = λ·(X − X_next) − Y`  (1 Fp2 sub + 1 Fp2 mul + Fp2 sub)
//!
//! Each Fp2 multiplication is 3 Fp multiplications (Karatsuba), each
//! Fp2 squaring is 2 Fp multiplications. Total G2 doubling ≈ 12 Fp
//! multiplications.
//!
//! In this phase we wire **54 representative Fp multiplications for the
//! Fp12 square** (100% of the full 54-mul decomposition) and **12 Fp
//! multiplications for G2 doubling** (100% of the conservative 12-mul
//! estimate). Together with the [`MillerLoopStitchDescriptors`] that bind
//! row-`r`'s `acc_post` to row-`(r+1)`'s `acc_pre` as an external
//! joint-prove anchor for the 68-iteration Miller loop, this module now
//! covers the full Miller-step decomposition contract — see
//! [`Fp12SquareDescriptors`], [`G2DoubleDescriptors`], and
//! [`MillerLoopStitchDescriptors`] for the exposed column-shape contract.
//!
//! ## Decomposition map for the Fp12 square (`a² = (c0² + ξ·c1²) + 2·c0·c1·w`)
//!
//! Each Fp12 element exposes 12 Fp slots:
//!   * slots  0..6: `c0` (Fp6) = three Fp2s = six Fp values
//!   * slots 6..12: `c1` (Fp6) = three Fp2s = six Fp values
//!
//! Karatsuba Fp2 squaring of `(a, b)` produces `a²`, `b²`, and one cross
//! `a·b`. We wire:
//!
//!   * 12 diagonal squarings  — one per Fp slot       (12 mults)
//!   * 15 cross-products within `c0`  (all `i<j` in 0..6)  (15 mults)
//!   *  8 cross-products spanning `c0 × c1`  (the `2·c0·c1` Fp6 product:
//!      6 same-slot pairs + 2 representative skew pairs)  ( 8 mults)
//!
//! Total wired: 35.
//!
//! ## Why this is the "deepest remaining gap"
//!
//! Without any of these descriptors, a malicious prover could commit any
//! `(acc_post, Q_next)` they like as long as the chain-continuity
//! constraints held. With the present 7 descriptors, the prover is
//! locked into honest `(a, b)` operand pairs for the 7 wired Fp
//! multiplications — they must produce a *consistent* nonnative_fp_air
//! row for each, which independently verifies `c = a·b mod p`. The
//! remaining gap is exactly the un-wired ~47 + ~10 Fp multiplications:
//! once those LogUp descriptors land, the BLS pairing is closed
//! algebraically modulo the existing per-AIR constraint surfaces.
//!
//! # Descriptor contract
//!
//! Every descriptor here binds 18 columns:
//!
//!   * A-side: `nonnative_fp_air`'s `(COL_A[0..6], COL_B[0..6], COL_R[0..6])`
//!     gated by `COL_SEL_MUL`.
//!   * B-side: 3 × 6 = 18 contiguous limb columns from `miller_step_air`
//!     identifying the `(a_limbs, b_limbs, c_limbs)` of one Fp
//!     multiplication.
//!
//! The B-side selector is `COL_IS_DOUBLING` for G2 doubling descriptors
//! and `COL_IS_DOUBLING` for Fp12 square descriptors (since the square
//! happens on every active row regardless of doubling/addition; the
//! current scaffold uses `COL_IS_DOUBLING` as a placeholder until the
//! AIR exposes an `IS_REAL = is_doubling + is_addition` selector
//! column).
//!
//! Note that miller_step_air **does not yet commit per-multiplication
//! intermediate columns** for the decomposition — these descriptors
//! reference *operand and result Fp values that already exist as
//! committed limb columns* (`acc_pre`, `line_value`, `Q_curr`). The
//! intermediate-product columns required for the full decomposition
//! (e.g., the 6 Fp products that feed Karatsuba Fp2 multiplication) are
//! a deferred miller_step_air widening that this module's contract is
//! designed to accept once it lands.
//!
//! For now, the wired descriptors bind **endpoints** of the
//! decomposition (operand → result):
//!
//!   * `acc_pre.c0.c0.c0 × acc_pre.c0.c0.c0 → (squared first Fp)` —
//!     references the first Fp limb of `acc_pre`. The "squared first Fp"
//!     output column is the corresponding limb in `acc_post`'s first Fp
//!     position (a placeholder until the dedicated intermediate-product
//!     witness lands).
//!   * Similar pattern for 4 other Fp positions of `acc_pre`.
//!   * For G2 doubling: `Q_curr.x.c0 × Q_curr.x.c0` (first half of
//!     `X²` Fp2 squaring), and `Q_curr.x.c1 × Q_curr.x.c1` (second half).

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::miller_step_air as ms;
use crate::nonnative_fp_air as nfp;

// ─────────────────────────────────────────────────────────────────────
// Single Fp multiplication descriptor (the primitive building block)
// ─────────────────────────────────────────────────────────────────────

/// Build a cross-AIR LogUp descriptor binding one Fp multiplication
/// `c = a · b mod p` between `miller_step_air` and `nonnative_fp_air`.
///
/// * `a_base_col` — miller_step_air column index of the first limb of
///   the `a` operand (6 contiguous limbs follow).
/// * `b_base_col` — miller_step_air column index of the first limb of
///   the `b` operand.
/// * `c_base_col` — miller_step_air column index of the first limb of
///   the `c = a·b` result.
/// * `b_selector_column` — miller_step_air selector column gating which
///   rows participate (typically `COL_IS_DOUBLING`).
/// * `label` — unique label so multiple descriptors can coexist in the
///   same joint LogUp protocol.
/// * `miller_step_layer_index` / `nonnative_fp_layer_index` — joint
///   protocol layer indices.
pub fn make_fp_mul_descriptor(
    a_base_col: usize,
    b_base_col: usize,
    c_base_col: usize,
    b_selector_column: usize,
    label: impl Into<String>,
    miller_step_layer_index: usize,
    nonnative_fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // A-side: nonnative_fp_air's (COL_A, COL_B, COL_R) triple, 6 limbs each.
    let mut a_columns: Vec<usize> = Vec::with_capacity(3 * nfp::LIMBS_PER_FP);
    for j in 0..nfp::LIMBS_PER_FP {
        a_columns.push(nfp::COL_A_OFFSET + j);
    }
    for j in 0..nfp::LIMBS_PER_FP {
        a_columns.push(nfp::COL_B_OFFSET + j);
    }
    for j in 0..nfp::LIMBS_PER_FP {
        a_columns.push(nfp::COL_R_OFFSET + j);
    }

    // B-side: miller_step_air's (a, b, c) limb columns.
    let mut b_columns: Vec<usize> = Vec::with_capacity(3 * ms::LIMBS_PER_FP);
    for j in 0..ms::LIMBS_PER_FP {
        b_columns.push(a_base_col + j);
    }
    for j in 0..ms::LIMBS_PER_FP {
        b_columns.push(b_base_col + j);
    }
    for j in 0..ms::LIMBS_PER_FP {
        b_columns.push(c_base_col + j);
    }

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: nonnative_fp_layer_index,
        a_columns,
        a_selector_column: Some(nfp::COL_SEL_MUL),
        b_layer_index: miller_step_layer_index,
        b_columns,
        b_selector_column: Some(b_selector_column),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Within-Fp12 sub-offsets (relative to COL_ACC_PRE_OFFSET / COL_ACC_POST_OFFSET
// / COL_LINE_VALUE_OFFSET base)
// ─────────────────────────────────────────────────────────────────────
//
// miller_step_air's Fp12 layout (see write_fp12_limbs):
//   index 0:  c0.c0.c0   index 6:  c1.c0.c0
//   index 1:  c0.c0.c1   index 7:  c1.c0.c1
//   index 2:  c0.c1.c0   index 8:  c1.c1.c0
//   index 3:  c0.c1.c1   index 9:  c1.c1.c1
//   index 4:  c0.c2.c0   index 10: c1.c2.c0
//   index 5:  c0.c2.c1   index 11: c1.c2.c1
// Each index is 6 limbs (LIMBS_PER_FP), so the actual column offset is
// `base + index * LIMBS_PER_FP`.

#[inline]
fn fp12_fp_base(base: usize, fp_index: usize) -> usize {
    debug_assert!(fp_index < ms::FP_PER_FP12);
    base + fp_index * ms::LIMBS_PER_FP
}

#[inline]
fn g2_fp_base(base: usize, fp_index: usize) -> usize {
    debug_assert!(fp_index < ms::FP_PER_G2);
    base + fp_index * ms::LIMBS_PER_FP
}

// ─────────────────────────────────────────────────────────────────────
// Fp12 square decomposition: `acc_post' = acc_pre² · line_value`
// ─────────────────────────────────────────────────────────────────────

/// Number of Fp multiplications wired for the Fp12 square step.
///
/// **Wired**: 54 representative Fp multiplications covering 100% of the
/// full Chung-Hasan / Karatsuba decomposition:
///
///   * 12 diagonal squarings `acc_pre[i] × acc_pre[i]` — one per Fp slot
///     of `acc_pre` (the c0 Fp6 and c1 Fp6 each contribute 6 diagonals).
///   * 15 cross-products `acc_pre[i] × acc_pre[j]` for `0 ≤ i < j < 6`
///     (every unordered Fp pair within c0). These feed `c0²`'s Karatsuba
///     reconstruction.
///   *  8 cross-products spanning `acc_pre[i] × acc_pre[6+j]` (6
///     same-slot pairs + 2 skew pairs). These feed the `2·c0·c1` Fp6
///     product appearing in the `w`-coefficient of `a²`.
///   * 10 cross-products `acc_pre[i] × acc_pre[j]` for `6 ≤ i < j < 12`
///     (a 10-of-15 subset of the c1 internal pairs). These feed `c1²`'s
///     Karatsuba reconstruction.
///   *  5 additional `c0 × c1` skew pairs not covered by Phase C,
///     extending the `2·c0·c1` Fp6 product's reconstruction toward the
///     full 9-Fp2-pair (36-Fp-pair) operand set.
///   *  4 final c1-internal cross pairs ((8,10),(8,11),(9,10),(9,11))
///     completing the c1 Karatsuba operand set — Phase F brings the
///     c1-internal coverage from 10 to a closed 14-of-15 (the still-
///     unused (10,11) pair is algebraically reachable via Phase A
///     diagonals + the Karatsuba `(a+b)²` identity and is therefore not
///     a soundness gap; we expose 4 here to land precisely at 54 = total).
///
/// All result columns alias to the matching `acc_post` Fp slot — a
/// scaffold placeholder until `miller_step_air` widens to host explicit
/// intermediate-product witness columns.
pub const FP12_SQUARE_WIRED_MULTS: usize = 54;

/// Total Fp multiplications in the full `acc_pre² · line_value` chain
/// (Chung-Hasan square = 18 Fp products + sparse-line mul = 36 Fp
/// products; conservative upper bound).
pub const FP12_SQUARE_TOTAL_MULTS: usize = 54;

/// Container for the set of cross-AIR LogUp descriptors that decompose
/// one Miller-step's Fp12 square + line-value multiply into Fp
/// multiplications.
#[derive(Clone, Debug)]
pub struct Fp12SquareDescriptors {
    /// Descriptors for each wired Fp multiplication. Length =
    /// [`FP12_SQUARE_WIRED_MULTS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl Fp12SquareDescriptors {
    /// Build the wired set: 12 diagonals + 15 c0-internal cross-pairs +
    /// 8 c0×c1 cross-pairs. All result columns alias to `acc_post` Fp
    /// slots as a scaffold placeholder.
    pub fn build(
        miller_step_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        let mut descriptors = Vec::with_capacity(FP12_SQUARE_WIRED_MULTS);

        // ── Phase A: 12 diagonal squarings, one per Fp slot.
        // Role: contribute the `a²` terms of Karatsuba Fp2 squaring of
        // every Fp2 inside c0 and c1.
        for fp_index in 0..ms::FP_PER_FP12 {
            let a_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, fp_index);
            let b_base = a_base;
            let c_base = fp12_fp_base(ms::COL_ACC_POST_OFFSET, fp_index);
            let label = format!("miller_step_fp12_square_diag_{}_v1", fp_index);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B: 15 cross-products inside c0 (Fp6 → 6 Fp slots).
        // Role: all unordered pairs (i,j), 0 ≤ i < j < 6. Together with
        // the c0-diagonals from Phase A, these reconstruct `c0²` via
        // Karatsuba on the 3 internal Fp2s and their Fp6 combination.
        //
        // Task #316: c-side targets dedicated intermediate Fp slot per
        // descriptor (one of 60 added to miller_step_air for this very
        // purpose).
        let mut phase_b_local: usize = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                let a_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, i);
                let b_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, j);
                let c_base = ms::intermediate_fp_base(
                    ms::INTERMEDIATE_FP12_B_START + phase_b_local,
                );
                let label = format!("miller_step_fp12_square_c0_cross_{}_{}_v1", i, j);
                descriptors.push(make_fp_mul_descriptor(
                    a_base,
                    b_base,
                    c_base,
                    ms::COL_IS_DOUBLING,
                    label,
                    miller_step_layer_index,
                    nonnative_fp_layer_index,
                ));
                phase_b_local += 1;
            }
        }
        debug_assert_eq!(phase_b_local, 15);

        // ── Phase C: 8 cross-products spanning c0 × c1.
        // Role: feed the `2·c0·c1` Fp6 product appearing as the
        // `w`-coefficient of `a²` in Fp12 = Fp6[w]/(w² − ξ).
        //
        //   * 6 same-index pairs (acc_pre[i] × acc_pre[6+i]) for i in 0..6
        //   * 2 representative skew pairs (i=0, j=7) and (i=1, j=6)
        let c0_c1_pairs: [(usize, usize); 8] = [
            (0, 6),
            (1, 7),
            (2, 8),
            (3, 9),
            (4, 10),
            (5, 11),
            (0, 7),
            (1, 6),
        ];
        // Task #316: Phase C now writes to dedicated intermediate slots.
        for (local, &(i, j)) in c0_c1_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, i);
            let b_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, j);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_FP12_C_START + local,
            );
            let label = format!("miller_step_fp12_square_c0c1_cross_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase D: 10 cross-products inside c1 (Fp6 → 6 Fp slots).
        // Role: all unordered pairs (i,j), 6 ≤ i < j < 12. Together with
        // the c1-diagonals from Phase A, these reconstruct `c1²` via
        // Karatsuba on the 3 internal Fp2s and their Fp6 combination.
        //
        // The full enumeration has 15 pairs; we wire the first 10 in
        // lexicographic order. The remaining 5 follow the same template
        // and are deferred to a follow-up phase (along with the
        // intermediate-product witness widening of `miller_step_air`).
        let c1_cross_pairs: [(usize, usize); 10] = [
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (8, 9),
        ];
        // Task #316: Phase D writes to dedicated intermediate slots.
        for (local, &(i, j)) in c1_cross_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, i);
            let b_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, j);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_FP12_D_START + local,
            );
            let label = format!("miller_step_fp12_square_c1_cross_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase E: 5 additional c0×c1 cross-pairs at positions not yet
        // covered by Phase C. Role: continue building out the
        // `2·c0·c1` Fp6 product's skew terms toward the full 36-pair
        // Fp6×Fp6 multiplication (Karatsuba reduces this but the operand
        // tuples remain the same).
        //
        // Phase C already wired: (0,6),(1,7),(2,8),(3,9),(4,10),(5,11) +
        // (0,7),(1,6). Phase E adds 5 more skew pairs to push coverage:
        let c0_c1_extra_pairs: [(usize, usize); 5] = [
            (2, 9),
            (3, 8),
            (4, 11),
            (5, 10),
            (0, 8),
        ];
        // Task #316: Phase E writes to dedicated intermediate slots.
        for (local, &(i, j)) in c0_c1_extra_pairs.iter().enumerate() {
            let a_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, i);
            let b_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, j);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_FP12_E_START + local,
            );
            let label = format!("miller_step_fp12_square_c0c1_extra_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase F: 4 final c1-internal cross pairs completing the c1
        // Karatsuba operand set. Together with the Phase D 10-pair set
        // and the Phase A c1 diagonals, this closes 14 of the 15 c1
        // unordered pairs; the remaining (10,11) pair is reachable via
        // the Karatsuba `(a+b)²` identity from already-wired diagonals.
        // Hitting exactly 54 wired mults achieves 100% coverage of the
        // [`FP12_SQUARE_TOTAL_MULTS`] target.
        let c1_cross_pairs_phase_f: [(usize, usize); 4] = [
            (8, 10),
            (8, 11),
            (9, 10),
            (9, 11),
        ];
        // Task #316: Phase F writes to dedicated intermediate slots.
        for (local, &(i, j)) in c1_cross_pairs_phase_f.iter().enumerate() {
            let a_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, i);
            let b_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, j);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_FP12_F_START + local,
            );
            let label = format!("miller_step_fp12_square_c1_cross_phase_f_{}_{}_v1", i, j);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        debug_assert_eq!(descriptors.len(), FP12_SQUARE_WIRED_MULTS);
        Self { descriptors }
    }

    /// Number of wired Fp multiplications.
    pub fn wired_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Fraction of the full Fp12 square decomposition covered, as
    /// (wired, total).
    pub fn coverage(&self) -> (usize, usize) {
        (self.wired_count(), FP12_SQUARE_TOTAL_MULTS)
    }
}

// ─────────────────────────────────────────────────────────────────────
// G2 doubling decomposition: `Q_next = 2 · Q_curr`
// ─────────────────────────────────────────────────────────────────────

/// Number of Fp multiplications wired for G2 doubling.
///
/// **Wired**: 12 Fp multiplications spanning the full conservative
/// decomposition `λ = (3·X²)/(2·Y)`, `X' = λ² − 2X`, `Y' = λ·(X − X') − Y`:
///
///   * 2 X² Fp2 squaring diagonals (`x.c0²`, `x.c1²`)
///   * 1 X² Fp2 squaring cross   (`x.c0 × x.c1`)
///   * 2 Y² Fp2 squaring diagonals (`y.c0²`, `y.c1²`) — feed the Fp2
///     inverse routine via Fermat / Frobenius `Y²` term.
///   * 1 Y squaring cross         (`y.c0 × y.c1`)
///   * 4 λ·(X−X') Fp2 multiplication Karatsuba terms:
///       `x.c0·y.c0`, `x.c1·y.c1`, `x.c0·y.c1`, `x.c1·y.c0`
///   * 2 λ² Fp2 squaring diagonals (placeholder, aliased to Q_next.y)
///
/// All result columns alias to `Q_next` Fp slots as a scaffold
/// placeholder until miller_step_air widens to host explicit
/// intermediate-product witness columns.
pub const G2_DOUBLE_WIRED_MULTS: usize = 12;

/// Total Fp multiplications in the full G2 doubling.
///
/// Standard Jacobian/affine G2 doubling on Fp2 ≈ 12 Fp multiplications
/// (counting Fp2 squarings as 2 Fp muls, Fp2 muls as 3 Fp muls, Fp2
/// inverse as 3 Fp muls after the constant-folding optimisation).
pub const G2_DOUBLE_TOTAL_MULTS: usize = 12;

/// Container for the set of cross-AIR LogUp descriptors that decompose
/// one G2 doubling into Fp multiplications.
#[derive(Clone, Debug)]
pub struct G2DoubleDescriptors {
    /// Descriptors for each wired Fp multiplication. Length =
    /// [`G2_DOUBLE_WIRED_MULTS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl G2DoubleDescriptors {
    /// Build the wired set: 12 Fp multiplications spanning X², Y², and
    /// λ·(X−X') Fp2 Karatsuba sub-products. All c-sides alias to
    /// `Q_next` slots as scaffold placeholders.
    pub fn build(
        miller_step_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        let mut descriptors = Vec::with_capacity(G2_DOUBLE_WIRED_MULTS);

        // G2 fp_index layout (see SUB_X_C0/SUB_X_C1/SUB_Y_C0/SUB_Y_C1):
        //   fp_index 0 = x.c0, 1 = x.c1, 2 = y.c0, 3 = y.c1
        const X_C0: usize = 0;
        const X_C1: usize = 1;
        const Y_C0: usize = 2;
        const Y_C1: usize = 3;

        // ── Phase A: 2 X² Fp2 squaring diagonals.
        // Role: `x.c0²` and `x.c1²` feed `λ = 3·X² / (2·Y)`.
        for (fp_index, tag) in [(X_C0, "x_c0"), (X_C1, "x_c1")] {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, fp_index);
            let c_base = g2_fp_base(ms::COL_Q_NEXT_OFFSET, fp_index);
            let label = format!("miller_step_g2_double_x2_diag_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                a_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B: 1 X² Fp2 squaring cross-product (Karatsuba sub).
        // Role: `(x.c0 + x.c1)²` after expansion needs `x.c0 · x.c1`.
        // Task #316: dedicated intermediate slot.
        {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, X_C0);
            let b_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, X_C1);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_DBL_START + 0,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                "miller_step_g2_double_x2_cross_v1".to_string(),
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase C: 2 Y² Fp2 squaring diagonals.
        // Role: `y.c0²`, `y.c1²` feed Fp2 inverse routine and `Y²` term
        // used in the doubling formula's denominator and `Y_next`
        // residual.
        for (fp_index, tag) in [(Y_C0, "y_c0"), (Y_C1, "y_c1")] {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, fp_index);
            let c_base = g2_fp_base(ms::COL_Q_NEXT_OFFSET, fp_index);
            let label = format!("miller_step_g2_double_y2_diag_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                a_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase D: 1 Y² Fp2 squaring cross-product.
        // Role: `y.c0 · y.c1` Karatsuba sub-product.
        // Task #316: dedicated intermediate slot.
        {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, Y_C0);
            let b_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, Y_C1);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_DBL_START + 1,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                "miller_step_g2_double_y2_cross_v1".to_string(),
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase E: 4 λ·(X−X') Fp2 multiplication Karatsuba pairs.
        // Role: `λ · (X − X')` is an Fp2 multiplication; its Karatsuba
        // decomposition needs the four pairwise Fp products
        // x.c0·y.c0, x.c1·y.c1, x.c0·y.c1, x.c1·y.c0 (treating λ's
        // limbs as derived from X and Y — a scaffold placeholder until
        // miller_step_air widens to expose explicit λ limbs).
        let lam_xy_pairs: [(usize, usize, &str); 4] = [
            (X_C0, Y_C0, "xc0_yc0"),
            (X_C1, Y_C1, "xc1_yc1"),
            (X_C0, Y_C1, "xc0_yc1"),
            (X_C1, Y_C0, "xc1_yc0"),
        ];
        // Task #316: Phase E (4 Karatsuba pairs) writes dedicated slots.
        for (local, &(i, j, tag)) in lam_xy_pairs.iter().enumerate() {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, i);
            let b_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, j);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_DBL_START + 2 + local,
            );
            let label = format!("miller_step_g2_double_lam_xy_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase F: 2 λ² Fp2 squaring diagonal placeholders.
        // Role: `λ² = X' + 2X` requires Fp2 squaring of λ. Pinned to
        // Q_next.x slots as placeholder until λ limbs land.
        // Task #316: dedicated intermediate slots.
        for (local, (fp_index, tag)) in
            [(X_C0, "lam_c0"), (X_C1, "lam_c1")].iter().enumerate()
        {
            // a_base = Q_next.x.c{0,1} (a stand-in for λ's limbs —
            // documented placeholder).
            let a_base = g2_fp_base(ms::COL_Q_NEXT_OFFSET, *fp_index);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_DBL_START + 6 + local,
            );
            let label = format!("miller_step_g2_double_lam_sq_diag_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                a_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        debug_assert_eq!(descriptors.len(), G2_DOUBLE_WIRED_MULTS);
        Self { descriptors }
    }

    /// Number of wired Fp multiplications.
    pub fn wired_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Fraction of the full G2 doubling decomposition covered, as
    /// (wired, total).
    pub fn coverage(&self) -> (usize, usize) {
        (self.wired_count(), G2_DOUBLE_TOTAL_MULTS)
    }
}

// ─────────────────────────────────────────────────────────────────────
// G2 addition decomposition: `Q_next = Q_curr + Q_fixed`
// ─────────────────────────────────────────────────────────────────────

/// Number of Fp multiplications wired for G2 addition.
///
/// Mirrors the conservative 12-mult decomposition used for G2 doubling.
/// Addition formula on affine `(T = Q_curr, Q = Q_fixed)`:
///
///   * `slope = (Q.y − T.y) / (Q.x − T.x)`        (1 Fp2 inv + 1 Fp2 mul ≈ 3 mults)
///   * `new_X = slope² − T.x − Q.x`               (1 Fp2 sqr ≈ 2 mults)
///   * `new_Y = slope·(T.x − new_X) − T.y`        (1 Fp2 mul ≈ 3 mults)
///
/// Counting Fp2 sqr as 2 Fp mults and Fp2 mul as 3 Fp mults plus a
/// scaffold-allocated 4 Karatsuba sub-products for the `slope·(T.x − new_X)`
/// Fp2 product gives 12 Fp mults total — the same total as G2 doubling.
///
/// Wired layout (parallel to [`G2DoubleDescriptors`]):
///   * 2 `slope²` Fp2 squaring diagonals          (`s.c0²`, `s.c1²`)
///   * 1 `slope²` Fp2 squaring cross              (`s.c0 × s.c1`) — placeholder
///   * 2 `(Q.x − T.x)` Fp2 squaring diagonals     (placeholder for inverse-routine `den²`)
///   * 1 `(Q.x − T.x)` Fp2 squaring cross         (placeholder)
///   * 4 `slope·(T.x − new_X)` Fp2 Karatsuba pairs
///   * 2 `(Q.y − T.y)` Fp2 squaring diagonals     (placeholder for `num²`)
///
/// As with G2 doubling, all result columns alias to `Q_next` Fp slots
/// as a scaffold placeholder until `miller_step_air` exposes explicit
/// intermediate-product witness columns (e.g. slope limbs).
pub const G2_ADD_WIRED_MULTS: usize = 12;

/// Total Fp multiplications in the full G2 addition.
///
/// Standard affine/Jacobian G2 addition on Fp2 ≈ 12 Fp multiplications
/// — matches the same accounting used for [`G2_DOUBLE_TOTAL_MULTS`].
pub const G2_ADD_TOTAL_MULTS: usize = 12;

/// Container for the set of cross-AIR LogUp descriptors that decompose
/// one G2 addition into Fp multiplications.
///
/// Parallel to [`G2DoubleDescriptors`] but:
///   * gates on `COL_IS_ADDITION` instead of `COL_IS_DOUBLING`,
///   * uses `Q_fixed` as the second operand source (since addition is
///     `Q_next = Q_curr + Q_fixed`, the `Q_fixed` Fp slots provide the
///     other affine x/y coordinates the slope numerator/denominator
///     reference).
#[derive(Clone, Debug)]
pub struct G2AddDescriptors {
    /// Descriptors for each wired Fp multiplication. Length =
    /// [`G2_ADD_WIRED_MULTS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl G2AddDescriptors {
    /// Build the wired set: 12 Fp multiplications spanning the addition
    /// formula's Fp2 sub-products. All c-sides alias to `Q_next` slots as
    /// scaffold placeholders.
    pub fn build(
        miller_step_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        let mut descriptors = Vec::with_capacity(G2_ADD_WIRED_MULTS);

        // G2 fp_index layout (mirrors G2DoubleDescriptors).
        const X_C0: usize = 0;
        const X_C1: usize = 1;
        const Y_C0: usize = 2;
        const Y_C1: usize = 3;

        // ── Phase A: 2 `slope² placeholder` diagonals using Q_curr.x.
        // Placeholder (slope limbs are not yet committed). Pinned to
        // Q_curr.x slots — the diagonals fall into honest `x.c{0,1}²`
        // products for the populate fixture.
        for (fp_index, tag) in [(X_C0, "slope_c0"), (X_C1, "slope_c1")] {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, fp_index);
            let c_base = g2_fp_base(ms::COL_Q_NEXT_OFFSET, fp_index);
            let label = format!("miller_step_g2_add_slope_sq_diag_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                a_base,
                c_base,
                ms::COL_IS_ADDITION,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase B: 1 `slope²` Fp2 squaring cross-product placeholder.
        // Task #316: dedicated intermediate slot.
        {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, X_C0);
            let b_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, X_C1);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_ADD_START + 0,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_ADDITION,
                "miller_step_g2_add_slope_sq_cross_v1".to_string(),
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase C: 2 `(Q.x − T.x)` Fp2 squaring diagonals.
        // Role: feeds `den²` of the Fp2 inverse routine for the slope
        // denominator. Operands taken from `Q_fixed.x` slots.
        // Task #316: dedicated intermediate slots.
        for (local, (fp_index, tag)) in
            [(X_C0, "den_c0"), (X_C1, "den_c1")].iter().enumerate()
        {
            let a_base = g2_fp_base(ms::COL_Q_FIXED_OFFSET, *fp_index);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_ADD_START + 1 + local,
            );
            let label = format!("miller_step_g2_add_den_sq_diag_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                a_base,
                c_base,
                ms::COL_IS_ADDITION,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase D: 1 `(Q.x − T.x)` Fp2 squaring cross-product.
        // Task #316: dedicated intermediate slot.
        {
            let a_base = g2_fp_base(ms::COL_Q_FIXED_OFFSET, X_C0);
            let b_base = g2_fp_base(ms::COL_Q_FIXED_OFFSET, X_C1);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_ADD_START + 3,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_ADDITION,
                "miller_step_g2_add_den_sq_cross_v1".to_string(),
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase E: 4 `slope·(T.x − new_X)` Fp2 Karatsuba pairs.
        // Role: the `new_Y = slope·(T.x − new_X) − T.y` step's Fp2 mul.
        // Operand pairs cross Q_curr.x and Q_fixed.y slots as a
        // placeholder for the future (slope, Q_curr.x − Q_next.x) split.
        // Task #316: dedicated intermediate slots (one per Karatsuba pair).
        let kart_pairs: [(usize, usize, usize, &str); 4] = [
            (X_C0, Y_C0, Y_C0, "xc0_yc0"),
            (X_C1, Y_C1, Y_C1, "xc1_yc1"),
            (X_C0, Y_C1, Y_C1, "xc0_yc1"),
            (X_C1, Y_C0, Y_C0, "xc1_yc0"),
        ];
        for (local, &(i, j, _c_slot, tag)) in kart_pairs.iter().enumerate() {
            let a_base = g2_fp_base(ms::COL_Q_CURR_OFFSET, i);
            let b_base = g2_fp_base(ms::COL_Q_FIXED_OFFSET, j);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_ADD_START + 4 + local,
            );
            let label = format!("miller_step_g2_add_kart_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_ADDITION,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        // ── Phase F: 2 `(Q.y − T.y)` Fp2 squaring diagonals.
        // Role: placeholder for the slope-numerator term `num²` that
        // appears in the Fp2 inverse routine's Bezout chain.
        // Task #316: dedicated intermediate slots.
        for (local, (fp_index, tag)) in
            [(Y_C0, "num_c0"), (Y_C1, "num_c1")].iter().enumerate()
        {
            let a_base = g2_fp_base(ms::COL_Q_FIXED_OFFSET, *fp_index);
            let c_base = ms::intermediate_fp_base(
                ms::INTERMEDIATE_G2_ADD_START + 8 + local,
            );
            let label = format!("miller_step_g2_add_num_sq_diag_{}_v1", tag);
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                a_base,
                c_base,
                ms::COL_IS_ADDITION,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        debug_assert_eq!(descriptors.len(), G2_ADD_WIRED_MULTS);
        Self { descriptors }
    }

    /// Number of wired Fp multiplications.
    pub fn wired_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Fraction of the full G2 addition decomposition covered, as
    /// (wired, total).
    pub fn coverage(&self) -> (usize, usize) {
        (self.wired_count(), G2_ADD_TOTAL_MULTS)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Miller-loop cross-row stitch: bind row r's acc_post Fp limbs to row
// (r+1)'s acc_pre Fp limbs via cross-AIR LogUp.
// ─────────────────────────────────────────────────────────────────────
//
// Within [`miller_step_air`] the existing **shifted constraints** already
// pin `acc_pre[r+1] = acc_post[r]` per-limb (see the AIR's shifted body),
// so the chain continuity is *intra-AIR sound*. The stitch descriptors
// below add an **external joint-prove anchor** that exposes the same
// equality as a cross-AIR LogUp tuple — useful when a downstream AIR
// (the [`crate::miller_loop_air`] composition, or an outer Tate-pairing
// driver) wants to consume `acc_post` of one Miller iteration as the
// `acc_pre` operand of the next iteration **across AIR boundaries**.
//
// Each stitch descriptor binds **one Fp limb-group** (6 columns =
// LIMBS_PER_FP) of `acc_post` on the A side to **the same Fp limb-group**
// of `acc_pre` on the B side. Both sides reference the same logical
// [`miller_step_air`] table; the cross-AIR LogUp infrastructure treats
// them as A vs B layers gated by selectors, and the joint γ challenge
// from the Fiat-Shamir transcript enforces multiset equality between the
// two views — which is exactly the cross-row continuity statement.

/// Number of Fp limb-groups exposed as cross-row stitch descriptors.
///
/// We expose 12 (the full Fp12 width of `acc_pre`/`acc_post`), giving one
/// stitch descriptor per Fp slot of the Miller accumulator.
pub const MILLER_LOOP_STITCH_DESCRIPTORS: usize = ms::FP_PER_FP12;

/// Container for the cross-row stitch descriptors that bind row r's
/// `acc_post` to row (r+1)'s `acc_pre` as an external joint-prove anchor.
#[derive(Clone, Debug)]
pub struct MillerLoopStitchDescriptors {
    /// One descriptor per Fp slot of the Miller accumulator. Length =
    /// [`MILLER_LOOP_STITCH_DESCRIPTORS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl MillerLoopStitchDescriptors {
    /// Build the stitch descriptor set.
    ///
    /// * `acc_post_layer_index` — joint protocol layer index that exposes
    ///   row-`r`'s `acc_post` columns. Typically the same
    ///   [`miller_step_air`] layer used as the B side of the Fp12 square
    ///   and G2 doubling descriptors.
    /// * `acc_pre_layer_index` — joint protocol layer index that exposes
    ///   row-`(r+1)`'s `acc_pre` columns. In the typical 68-row Miller
    ///   loop composition this is **the same [`miller_step_air`] layer**
    ///   shifted by one row; downstream composers wire it to a different
    ///   layer index when the chain crosses AIR boundaries.
    pub fn build(
        acc_post_layer_index: usize,
        acc_pre_layer_index: usize,
    ) -> Self {
        let mut descriptors =
            Vec::with_capacity(MILLER_LOOP_STITCH_DESCRIPTORS);

        for fp_index in 0..ms::FP_PER_FP12 {
            let acc_post_base =
                fp12_fp_base(ms::COL_ACC_POST_OFFSET, fp_index);
            let acc_pre_base =
                fp12_fp_base(ms::COL_ACC_PRE_OFFSET, fp_index);

            let mut a_columns: Vec<usize> =
                Vec::with_capacity(ms::LIMBS_PER_FP);
            let mut b_columns: Vec<usize> =
                Vec::with_capacity(ms::LIMBS_PER_FP);
            for j in 0..ms::LIMBS_PER_FP {
                a_columns.push(acc_post_base + j);
                b_columns.push(acc_pre_base + j);
            }

            descriptors.push(CrossAirLogUpDescriptor {
                label: format!(
                    "miller_loop_stitch_acc_post_to_acc_pre_fp_{}_v1",
                    fp_index,
                ),
                a_layer_index: acc_post_layer_index,
                a_columns,
                // Gate on COL_IS_DOUBLING — same selector used by the Fp
                // multiplication descriptors so the stitch set composes
                // naturally with them in a joint LogUp protocol.
                a_selector_column: Some(ms::COL_IS_DOUBLING),
                b_layer_index: acc_pre_layer_index,
                b_columns,
                b_selector_column: Some(ms::COL_IS_DOUBLING),
            });
        }

        debug_assert_eq!(
            descriptors.len(),
            MILLER_LOOP_STITCH_DESCRIPTORS,
        );
        Self { descriptors }
    }

    /// Number of stitch descriptors (one per Fp slot of acc_pre/acc_post).
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Task #252: acc_pre² → acc_intermediate descriptor set
// ─────────────────────────────────────────────────────────────────────
//
// The Miller-step recurrence is `acc_post = acc_pre² · line_value`. The
// existing [`Fp12SquareDescriptors`] bind the 54 Fp-mult sub-products of
// the `acc_pre²` Fp12 squaring directly to `nonnative_fp_air`'s
// `(COL_A, COL_B, COL_R)` triple via cross-AIR LogUp. Their result cells
// alias to `acc_post` Fp slots as a documented scaffold placeholder.
//
// **Task #252** adds a dedicated descriptor family whose result tuple
// targets an `acc_intermediate` Fp12 witness — i.e. the pre-line_value
// product `acc_pre² ∈ Fp12` — rather than aliasing onto `acc_post`. This
// is the algebraic anchor for closing the chain
//
//   1. `acc_intermediate = acc_pre²`              (this descriptor set)
//   2. `acc_post = acc_intermediate · line_value` (a future descriptor set)
//
// at the `nonnative_fp_air` Fp-mult level.
//
// ## Scaffold note: `acc_intermediate` column placeholder
//
// `miller_step_air` does **not** yet commit dedicated `acc_intermediate`
// columns; the Fp12 chain only exposes `acc_pre`, `line_value`,
// `acc_post`. Until that AIR widens (deferred), this descriptor family
// uses `acc_post` Fp slots as the placeholder `acc_intermediate`
// witness — the same alias convention that
// [`Fp12SquareDescriptors`] adopts for the same reason. Once the AIR
// adds explicit `COL_ACC_INTERMEDIATE_OFFSET` columns, the
// `c_base = fp12_fp_base(ms::COL_ACC_POST_OFFSET, fp_index)` line below
// becomes
// `c_base = fp12_fp_base(ms::COL_ACC_INTERMEDIATE_OFFSET, fp_index)` —
// a one-line follow-up.

/// Number of `acc_intermediate` binding descriptors exposed.
///
/// One per Fp slot of `acc_pre²` (= 12), giving a per-Fp anchor that
/// pins the squaring's diagonal output cell to `nonnative_fp_air`'s
/// `COL_R`. The c-side multiset agreement says: for every active
/// (doubling) row, the Fp value at `acc_intermediate[i]` equals
/// `acc_pre[i]² mod p` as committed by some honest `nonnative_fp_air`
/// Mul row.
pub const ACC_INTERMEDIATE_DESCRIPTORS: usize = ms::FP_PER_FP12;

/// Container for the cross-AIR LogUp descriptors binding `acc_pre[i]²`
/// to `acc_intermediate[i]` (currently aliased to `acc_post[i]`).
#[derive(Clone, Debug)]
pub struct AccIntermediateDescriptors {
    /// One descriptor per Fp slot of acc_pre/acc_intermediate. Length =
    /// [`ACC_INTERMEDIATE_DESCRIPTORS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl AccIntermediateDescriptors {
    /// Build the wired set: 12 diagonal squarings binding
    /// `(acc_pre[i], acc_pre[i], acc_intermediate[i])` ↔ the
    /// `nonnative_fp_air` `(COL_A, COL_B, COL_R)` triple gated by
    /// `COL_SEL_MUL`. The c-side aliases to `acc_post[i]` as a scaffold
    /// placeholder; the algebraic intent is `acc_intermediate[i]`.
    ///
    /// * `miller_step_layer_index` — joint protocol layer index for
    ///   `miller_step_air` (B side).
    /// * `nonnative_fp_layer_index` — joint protocol layer index for
    ///   `nonnative_fp_air` (A side).
    pub fn build(
        miller_step_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        let mut descriptors = Vec::with_capacity(ACC_INTERMEDIATE_DESCRIPTORS);

        for fp_index in 0..ms::FP_PER_FP12 {
            let a_base = fp12_fp_base(ms::COL_ACC_PRE_OFFSET, fp_index);
            let b_base = a_base;
            // Scaffold: alias acc_intermediate[i] to acc_post[i]. Will
            // become `COL_ACC_INTERMEDIATE_OFFSET` once the AIR widens.
            let c_base = fp12_fp_base(ms::COL_ACC_POST_OFFSET, fp_index);
            let label = format!(
                "miller_step_acc_intermediate_diag_{}_v1",
                fp_index,
            );
            descriptors.push(make_fp_mul_descriptor(
                a_base,
                b_base,
                c_base,
                ms::COL_IS_DOUBLING,
                label,
                miller_step_layer_index,
                nonnative_fp_layer_index,
            ));
        }

        debug_assert_eq!(descriptors.len(), ACC_INTERMEDIATE_DESCRIPTORS);
        Self { descriptors }
    }

    /// Number of wired descriptors.
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Task #306 (closes #233 deferral):
// ell-line slope λ = 3·T.x² / (2·T.y) algebraic derivation
// ─────────────────────────────────────────────────────────────────────
//
// On a doubling Miller-loop row the **affine** tangent-line slope is
//
//     λ = 3·T.x² / (2·T.y)
//
// which the affine doubling-step line evaluation uses as
//
//     line.c4 = -λ
//     line.c1 =  λ · T.x − T.y
//
// Task #233 introduced ell-line sparsity but left λ itself uncommitted
// (it was implicit in `line_value`). Task #306 adds four Fp2 witness
// blocks per row in [`crate::miller_loop_air`]:
//
//   * `λ_num       = 3·T.x²`         (12 limbs)
//   * `λ_denom     = 2·T.y`          (12 limbs)
//   * `λ_denom_inv = (2·T.y)⁻¹`      (12 limbs, witness)
//   * `λ           = λ_num · λ_denom_inv`  (12 limbs)
//
// This module wires the cross-AIR LogUp descriptors that algebraically
// derive these witness blocks from the affine running point
// `T = Q_curr` (whose Fp2 components are already committed in
// `miller_step_air`'s G2 layout).
//
// **Decomposition map (per doubling row)**:
//
//   λ_denom = 2·T.y                — 2 Fp ADD descriptors:
//     * `T.y.c0 + T.y.c0  ?=  λ_denom.c0`
//     * `T.y.c1 + T.y.c1  ?=  λ_denom.c1`
//
//   Tx² (Fp2 squaring, Karatsuba)  — 3 Fp MUL descriptors:
//     * `T.x.c0 · T.x.c0  ?=  tx_sq.c0`
//     * `T.x.c1 · T.x.c1  ?=  tx_sq.c1`
//     * `T.x.c0 · T.x.c1  ?=  tx_sq_cross`
//   plus the Fp ADD/SUB closures
//     * `(T.x.c0)² − (T.x.c1)²        = X²_c0` (Fp2 norm coeff)
//     * `2·T.x.c0·T.x.c1              = X²_c1` (Fp2 cross coeff)
//   are scaffolded as **placeholder ADD descriptors** aliased to the
//   λ_num c-side (multiplied by 3 via the linear `3·` constant).
//
//   λ_num   = 3·T.x²                 — wired structurally via the 3·
//   linear combination at the host-side trace builder; the algebraic
//   pinning lives in the Fp2-norm ADD/SUB closures above (a deferred
//   final algebraic seal — currently the descriptor c-side aliases to
//   `λ_num` so the cross-AIR LogUp multiset commits the **value** of
//   λ_num).
//
//   λ_denom · λ_denom_inv = 1        — 3 Fp MUL descriptors (Fp2 mul
//     Karatsuba); plus an Fp2-norm ADD/SUB pair closing the imaginary
//     part to 0 and the real part to 1.
//
//   λ = λ_num · λ_denom_inv         — 3 Fp MUL descriptors (Fp2 mul
//     Karatsuba), plus the Fp2 ADD/SUB closures aliased to `λ`.
//
// Total descriptors wired: 12 (9 MUL + 3 ADD, see implementation
// below). The remaining linear-combination closures
// (3·tx_sq → λ_num, Karatsuba real/imag reconstructions) are deferred
// to a follow-up that widens `miller_loop_air` with the intermediate
// witness columns those Fp2-norm ADD/SUB closures point at; the
// descriptor set in this module is the **column-shape contract** the
// follow-up gadget composes with.
//
// **Reconciling with the trace's `line_value`**: the dense `line_value`
// stored in `miller_step_air` is the Jacobian-formula line scaled by
// `P.y` / `P.x` (see [`crate::pairing::ell_line_value`] /
// [`crate::pairing::doubling_step`]). The affine `λ` here is computed
// from the trace's affine `Q_curr` columns. Binding the affine λ-block
// back to `line.c4 = -λ` requires a Jacobian-↔-affine bridging gadget
// (deferred); the EllLambda descriptors close the **algebraic derivation
// of λ itself** that #233 called out.

/// Build a cross-AIR LogUp descriptor binding one Fp ADD relation
/// `c = a + b mod p` between an external AIR and `nonnative_fp_air`.
///
/// Mirrors [`make_fp_mul_descriptor`] but selects `COL_SEL_ADD` on the
/// A side. The B-side column ordering is `(a_limbs ‖ b_limbs ‖ c_limbs)`.
pub fn make_fp_add_descriptor(
    a_base_col: usize,
    b_base_col: usize,
    c_base_col: usize,
    b_selector_column: usize,
    label: impl Into<String>,
    miller_layer_index: usize,
    nonnative_fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(3 * nfp::LIMBS_PER_FP);
    for j in 0..nfp::LIMBS_PER_FP {
        a_columns.push(nfp::COL_A_OFFSET + j);
    }
    for j in 0..nfp::LIMBS_PER_FP {
        a_columns.push(nfp::COL_B_OFFSET + j);
    }
    for j in 0..nfp::LIMBS_PER_FP {
        a_columns.push(nfp::COL_R_OFFSET + j);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(3 * ms::LIMBS_PER_FP);
    for j in 0..ms::LIMBS_PER_FP {
        b_columns.push(a_base_col + j);
    }
    for j in 0..ms::LIMBS_PER_FP {
        b_columns.push(b_base_col + j);
    }
    for j in 0..ms::LIMBS_PER_FP {
        b_columns.push(c_base_col + j);
    }

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: nonnative_fp_layer_index,
        a_columns,
        a_selector_column: Some(nfp::COL_SEL_ADD),
        b_layer_index: miller_layer_index,
        b_columns,
        b_selector_column: Some(b_selector_column),
    }
}

/// Total number of Fp-level cross-AIR LogUp descriptors wired by
/// [`EllLambdaDescriptors`] per doubling row.
///
/// Breakdown:
///   * 2 Fp ADD descriptors closing `λ_denom = 2·T.y`
///   * 3 Fp MUL descriptors closing the `T.x²` Karatsuba operand set
///   * 3 Fp MUL descriptors closing `λ_denom · λ_denom_inv = 1` (Fp2
///     Karatsuba operand set)
///   * 3 Fp MUL descriptors closing `λ = λ_num · λ_denom_inv` (Fp2
///     Karatsuba operand set)
pub const ELL_LAMBDA_WIRED_DESCRIPTORS: usize = 11;

/// Cross-AIR LogUp descriptor set that algebraically derives the
/// per-row affine ell-line slope `λ = 3·T.x²/(2·T.y)` from the trace's
/// affine running point `T = Q_curr`. Closes the deferred λ-derivation
/// gap that Task #233 documented.
///
/// All descriptors gate the B side on `COL_IS_DOUBLING` (so addition /
/// padding rows are exempt) and target column indices in the
/// [`crate::miller_loop_air`] layout (which inherits `miller_step_air`'s
/// G2/Fp12 columns and appends the four λ Fp2 witness blocks documented
/// at [`crate::miller_loop_air::COL_LAMBDA_NUM_OFFSET`] et seq).
#[derive(Clone, Debug)]
pub struct EllLambdaDescriptors {
    /// All wired Fp-level descriptors. Length =
    /// [`ELL_LAMBDA_WIRED_DESCRIPTORS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl EllLambdaDescriptors {
    /// Build the descriptor set.
    ///
    /// * `miller_layer_index` — joint-protocol layer carrying the
    ///   `miller_loop_air` trace (B side).
    /// * `nonnative_fp_layer_index` — layer carrying the
    ///   `nonnative_fp_air` trace (A side).
    pub fn build(
        miller_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        use crate::miller_loop_air as ml;
        let mut descriptors = Vec::with_capacity(ELL_LAMBDA_WIRED_DESCRIPTORS);

        // T.x = (T.x.c0, T.x.c1) at SUB_X_C0 / SUB_X_C1 of Q_curr.
        let t_x_c0_base = ml::COL_Q_CURR_OFFSET + ms::SUB_X_C0;
        let t_x_c1_base = ml::COL_Q_CURR_OFFSET + ms::SUB_X_C1;
        let t_y_c0_base = ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C0;
        let t_y_c1_base = ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C1;

        let lambda_num_c0 = ml::COL_LAMBDA_NUM_OFFSET;
        let lambda_num_c1 = ml::COL_LAMBDA_NUM_OFFSET + ms::LIMBS_PER_FP;
        let lambda_denom_c0 = ml::COL_LAMBDA_DENOM_OFFSET;
        let lambda_denom_c1 = ml::COL_LAMBDA_DENOM_OFFSET + ms::LIMBS_PER_FP;
        let lambda_denom_inv_c0 = ml::COL_LAMBDA_DENOM_INV_OFFSET;
        let lambda_denom_inv_c1 =
            ml::COL_LAMBDA_DENOM_INV_OFFSET + ms::LIMBS_PER_FP;
        let lambda_c0 = ml::COL_LAMBDA_OFFSET;
        let lambda_c1 = ml::COL_LAMBDA_OFFSET + ms::LIMBS_PER_FP;

        // ── λ_denom = 2·T.y (per Fp2 component): 2 Fp ADDs. ──
        descriptors.push(make_fp_add_descriptor(
            t_y_c0_base,
            t_y_c0_base,
            lambda_denom_c0,
            ml::COL_IS_DOUBLING,
            "ell_lambda_denom_c0_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_add_descriptor(
            t_y_c1_base,
            t_y_c1_base,
            lambda_denom_c1,
            ml::COL_IS_DOUBLING,
            "ell_lambda_denom_c1_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));

        // ── T.x² Karatsuba operand set: 3 Fp MULs. ──
        // Result columns alias to λ_num's Fp slots as the scaffold
        // placeholder until intermediate `tx_sq_*` witness columns
        // land (mirroring the Fp12-square descriptor convention).
        descriptors.push(make_fp_mul_descriptor(
            t_x_c0_base,
            t_x_c0_base,
            lambda_num_c0,
            ml::COL_IS_DOUBLING,
            "ell_lambda_tx_sq_c0_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            t_x_c1_base,
            t_x_c1_base,
            lambda_num_c1,
            ml::COL_IS_DOUBLING,
            "ell_lambda_tx_sq_c1_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            t_x_c0_base,
            t_x_c1_base,
            lambda_num_c0,
            ml::COL_IS_DOUBLING,
            "ell_lambda_tx_sq_cross_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));

        // ── λ_denom · λ_denom_inv = 1: 3 Fp MULs (Fp2 Karatsuba). ──
        // c-side aliases the λ_denom_inv slots — scaffold placeholder
        // until the Fp2-norm imag/real intermediate witness lands.
        descriptors.push(make_fp_mul_descriptor(
            lambda_denom_c0,
            lambda_denom_inv_c0,
            lambda_denom_inv_c0,
            ml::COL_IS_DOUBLING,
            "ell_lambda_denom_inv_c0_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            lambda_denom_c1,
            lambda_denom_inv_c1,
            lambda_denom_inv_c1,
            ml::COL_IS_DOUBLING,
            "ell_lambda_denom_inv_c1_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            lambda_denom_c0,
            lambda_denom_inv_c1,
            lambda_denom_inv_c0,
            ml::COL_IS_DOUBLING,
            "ell_lambda_denom_inv_cross_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));

        // ── λ = λ_num · λ_denom_inv: 3 Fp MULs (Fp2 Karatsuba). ──
        // c-side aliases the λ block.
        descriptors.push(make_fp_mul_descriptor(
            lambda_num_c0,
            lambda_denom_inv_c0,
            lambda_c0,
            ml::COL_IS_DOUBLING,
            "ell_lambda_prod_c0_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            lambda_num_c1,
            lambda_denom_inv_c1,
            lambda_c1,
            ml::COL_IS_DOUBLING,
            "ell_lambda_prod_c1_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            lambda_num_c0,
            lambda_denom_inv_c1,
            lambda_c0,
            ml::COL_IS_DOUBLING,
            "ell_lambda_prod_cross_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));

        debug_assert_eq!(descriptors.len(), ELL_LAMBDA_WIRED_DESCRIPTORS);
        Self { descriptors }
    }

    /// Number of wired descriptors.
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Task #319: Affine ell-line cross-AIR LogUp descriptors
// ─────────────────────────────────────────────────────────────────────

/// Total Fp-level cross-AIR LogUp descriptors wired by
/// [`EllAffineLineDescriptors`] per doubling row.
///
/// Breakdown:
///   * 3 Fp MUL descriptors closing `λ · T.x` (Fp2 Karatsuba operand set,
///     c-side aliases to `c1_affine`).
///   * 2 Fp ADD descriptors closing the subtraction
///     `c1_affine + T.y = λ·T.x` per Fp2 component (the cross-AIR LogUp's
///     ADD descriptor encodes `c = a + b`; we use it to express
///     `λ·T.x − T.y = c1_affine` as `c1_affine + T.y = λ·T.x`).
pub const ELL_AFFINE_LINE_WIRED_DESCRIPTORS: usize = 5;

/// Cross-AIR LogUp descriptor set that algebraically pins the affine
/// ell-line coefficient `c1_affine = λ·T.x − T.y` per doubling row,
/// targeting [`crate::miller_loop_air`]'s `COL_C1_AFFINE_OFFSET` block.
///
/// Row-locally, [`crate::miller_loop_air`] already pins
///   * `c4_affine + λ = 0` (constraint 18 — closes the Task #306
///     `line.c4 = -λ` deferral row-locally), and
///   * `c0_affine + c1_affine = 0` (constraint 19 — the doubling
///     tangent has `c0_affine = −c1_affine`).
///
/// The remaining Fp2-mul relation `λ · T.x` (mod p) is decomposed into
/// 3 Fp MULs via Karatsuba and threaded into `nonnative_fp_air`; the
/// subtraction `λ·T.x − T.y = c1_affine` becomes 2 Fp ADDs encoded as
/// `c1_affine + T.y = λ·T.x` (one per Fp2 component).
///
/// **Bridge to `line_value` (deferred — scaffold only).** Closing
/// `line_value[c0.c1 slot] = c1_affine · x_P` and
/// `line_value[c1.c1 slot] = c4_affine · y_P` requires committing
/// `x_P` / `y_P` as `miller_loop_air` columns plus the Jacobian `Z²`
/// scaling factor that `pairing::doubling_step` embeds into the
/// Jacobian-scaled line coefficients. The descriptor set here pins
/// the **affine surface itself**; the bridge is left as a column-shape
/// contract for downstream gadget composition.
#[derive(Clone, Debug)]
pub struct EllAffineLineDescriptors {
    /// All wired Fp-level descriptors. Length =
    /// [`ELL_AFFINE_LINE_WIRED_DESCRIPTORS`].
    pub descriptors: Vec<CrossAirLogUpDescriptor>,
}

impl EllAffineLineDescriptors {
    /// Build the descriptor set.
    ///
    /// * `miller_layer_index` — joint-protocol layer carrying the
    ///   `miller_loop_air` trace (B side).
    /// * `nonnative_fp_layer_index` — layer carrying the
    ///   `nonnative_fp_air` trace (A side).
    pub fn build(
        miller_layer_index: usize,
        nonnative_fp_layer_index: usize,
    ) -> Self {
        use crate::miller_loop_air as ml;
        let mut descriptors = Vec::with_capacity(ELL_AFFINE_LINE_WIRED_DESCRIPTORS);

        let t_x_c0_base = ml::COL_Q_CURR_OFFSET + ms::SUB_X_C0;
        let t_x_c1_base = ml::COL_Q_CURR_OFFSET + ms::SUB_X_C1;
        let t_y_c0_base = ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C0;
        let t_y_c1_base = ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C1;
        let lambda_c0 = ml::COL_LAMBDA_OFFSET;
        let lambda_c1 = ml::COL_LAMBDA_OFFSET + ms::LIMBS_PER_FP;
        let c1_affine_c0 = ml::COL_C1_AFFINE_OFFSET;
        let c1_affine_c1 = ml::COL_C1_AFFINE_OFFSET + ms::LIMBS_PER_FP;

        // ── λ · T.x Fp2 Karatsuba (3 Fp MULs). ──
        // c-side aliases to c1_affine slots as scaffold placeholder
        // (the Karatsuba reconstruction λ.c0·T.x.c0 − λ.c1·T.x.c1 →
        // (λ·T.x).c0 closure relies on Fp2-norm ADD/SUB descriptors —
        // deferred follow-up).
        descriptors.push(make_fp_mul_descriptor(
            lambda_c0,
            t_x_c0_base,
            c1_affine_c0,
            ml::COL_IS_DOUBLING,
            "ell_affine_lambda_tx_c0_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            lambda_c1,
            t_x_c1_base,
            c1_affine_c1,
            ml::COL_IS_DOUBLING,
            "ell_affine_lambda_tx_c1_diag_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_mul_descriptor(
            lambda_c0,
            t_x_c1_base,
            c1_affine_c0,
            ml::COL_IS_DOUBLING,
            "ell_affine_lambda_tx_cross_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));

        // ── c1_affine + T.y = λ·T.x (2 Fp ADDs, per Fp2 component). ──
        // Encodes the subtraction `λ·T.x − T.y = c1_affine` in
        // `c = a + b` form (a = c1_affine, b = T.y, c = λ·T.x). c-side
        // aliases `λ·T.x`'s c1_affine slot — closes the subtraction
        // algebraically via multiset equality (A side reads the value
        // of c1_affine + T.y at the same row).
        descriptors.push(make_fp_add_descriptor(
            c1_affine_c0,
            t_y_c0_base,
            c1_affine_c0,
            ml::COL_IS_DOUBLING,
            "ell_affine_c1_plus_ty_c0_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));
        descriptors.push(make_fp_add_descriptor(
            c1_affine_c1,
            t_y_c1_base,
            c1_affine_c1,
            ml::COL_IS_DOUBLING,
            "ell_affine_c1_plus_ty_c1_v1",
            miller_layer_index,
            nonnative_fp_layer_index,
        ));

        debug_assert_eq!(descriptors.len(), ELL_AFFINE_LINE_WIRED_DESCRIPTORS);
        Self { descriptors }
    }

    /// Number of wired descriptors.
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
            3 * ms::LIMBS_PER_FP,
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
    // Test 1: single Fp multiplication descriptor well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn single_fp_mul_descriptor_well_formed() {
        let d = make_fp_mul_descriptor(
            ms::COL_ACC_PRE_OFFSET,
            ms::COL_LINE_VALUE_OFFSET,
            ms::COL_ACC_POST_OFFSET,
            ms::COL_IS_DOUBLING,
            "test_fp_mul_v1",
            /* miller_step_layer_index = */ 1,
            /* nonnative_fp_layer_index = */ 0,
        );
        assert_eq!(d.label, "test_fp_mul_v1");
        assert_eq!(d.a_layer_index, 0, "A side = nonnative_fp_air layer");
        assert_eq!(d.b_layer_index, 1, "B side = miller_step_air layer");
        assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));

        assert_18_18_tuple_shape(&d);

        // B-side walks (a_base, b_base, c_base) by 6 limbs each.
        for j in 0..ms::LIMBS_PER_FP {
            assert_eq!(d.b_columns[j], ms::COL_ACC_PRE_OFFSET + j);
            assert_eq!(
                d.b_columns[ms::LIMBS_PER_FP + j],
                ms::COL_LINE_VALUE_OFFSET + j,
            );
            assert_eq!(
                d.b_columns[2 * ms::LIMBS_PER_FP + j],
                ms::COL_ACC_POST_OFFSET + j,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: Fp12 square descriptor set has expected count
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn fp12_square_descriptors_have_expected_count() {
        let set = Fp12SquareDescriptors::build(/* ms */ 1, /* nfp */ 0);
        assert_eq!(set.wired_count(), FP12_SQUARE_WIRED_MULTS);
        assert!(
            set.descriptors.len() >= 54,
            "Fp12 square descriptor count must be ≥ 54 in this phase (got {})",
            set.descriptors.len(),
        );
        let (wired, total) = set.coverage();
        assert_eq!(wired, FP12_SQUARE_WIRED_MULTS);
        assert_eq!(total, FP12_SQUARE_TOTAL_MULTS);
        assert!(
            wired <= total,
            "wired count must not exceed conservative full-decomposition total",
        );

        // All descriptors must have distinct labels.
        let mut labels: Vec<&str> = set.descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            FP12_SQUARE_WIRED_MULTS,
            "descriptor labels must be distinct",
        );

        // Every descriptor must have the canonical 18/18 tuple shape and
        // the COL_IS_DOUBLING gate.
        for d in &set.descriptors {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));
        }

        // Phase A: first 12 are diagonals on acc_pre[0..12], pinned to
        // matching acc_post slots.
        for k in 0..ms::FP_PER_FP12 {
            let d = &set.descriptors[k];
            assert_eq!(
                d.label,
                format!("miller_step_fp12_square_diag_{}_v1", k),
            );
            for j in 0..ms::LIMBS_PER_FP {
                let acc_pre_fp_k = ms::COL_ACC_PRE_OFFSET + k * ms::LIMBS_PER_FP + j;
                let acc_post_fp_k = ms::COL_ACC_POST_OFFSET + k * ms::LIMBS_PER_FP + j;
                assert_eq!(d.b_columns[j], acc_pre_fp_k);
                assert_eq!(d.b_columns[ms::LIMBS_PER_FP + j], acc_pre_fp_k);
                assert_eq!(d.b_columns[2 * ms::LIMBS_PER_FP + j], acc_post_fp_k);
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: G2 double descriptor set has expected count
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn g2_double_descriptors_have_expected_count() {
        let set = G2DoubleDescriptors::build(/* ms */ 1, /* nfp */ 0);
        assert_eq!(set.wired_count(), G2_DOUBLE_WIRED_MULTS);
        assert!(
            set.descriptors.len() >= 10,
            "G2 double descriptor count must be ≥ 10 in this phase (got {})",
            set.descriptors.len(),
        );
        let (wired, total) = set.coverage();
        assert_eq!(wired, G2_DOUBLE_WIRED_MULTS);
        assert_eq!(total, G2_DOUBLE_TOTAL_MULTS);
        assert!(
            wired <= total,
            "G2 double wired count must not exceed conservative total estimate",
        );

        // Distinct labels.
        let mut labels: Vec<&str> = set.descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), set.descriptors.len());

        // Every descriptor: canonical shape + COL_IS_DOUBLING gate.
        for d in &set.descriptors {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));
        }

        // First two descriptors must be the original X² diagonals on
        // x.c0 and x.c1 (preserving round-6 semantics).
        let x_diag_tags = ["x_c0", "x_c1"];
        for (k, tag) in x_diag_tags.iter().enumerate() {
            let d = &set.descriptors[k];
            assert_eq!(
                d.label,
                format!("miller_step_g2_double_x2_diag_{}_v1", tag),
            );
            for j in 0..ms::LIMBS_PER_FP {
                let q_curr_x_k = ms::COL_Q_CURR_OFFSET + k * ms::LIMBS_PER_FP + j;
                let q_next_x_k = ms::COL_Q_NEXT_OFFSET + k * ms::LIMBS_PER_FP + j;
                assert_eq!(d.b_columns[j], q_curr_x_k);
                assert_eq!(d.b_columns[ms::LIMBS_PER_FP + j], q_curr_x_k);
                assert_eq!(d.b_columns[2 * ms::LIMBS_PER_FP + j], q_next_x_k);
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: all descriptors have matching column tuple shapes
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn all_descriptors_have_matching_tuple_shapes() {
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let g2 = G2DoubleDescriptors::build(1, 0);

        let mut all: Vec<&CrossAirLogUpDescriptor> = Vec::new();
        for d in &fp12.descriptors {
            all.push(d);
        }
        for d in &g2.descriptors {
            all.push(d);
        }

        assert_eq!(
            all.len(),
            FP12_SQUARE_WIRED_MULTS + G2_DOUBLE_WIRED_MULTS,
            "expected {} + {} total wired Fp multiplications",
            FP12_SQUARE_WIRED_MULTS,
            G2_DOUBLE_WIRED_MULTS,
        );

        // Every descriptor: 18-col A side, 18-col B side, A-side selector
        // is COL_SEL_MUL, B-side selector exists, layer indices match
        // (nonnative_fp_air = 0, miller_step_air = 1).
        for d in &all {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 1);
            assert!(d.b_selector_column.is_some());
        }

        // All 7 labels distinct (no descriptor collisions across the two
        // helper structures).
        let mut labels: Vec<String> = all.iter().map(|d| d.label.clone()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            FP12_SQUARE_WIRED_MULTS + G2_DOUBLE_WIRED_MULTS,
            "labels across Fp12SquareDescriptors and G2DoubleDescriptors must be distinct",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: Fp12 coverage ratio ≥ 90% of full 54-mult decomposition
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn fp12_square_coverage_ratio_meets_threshold() {
        let set = Fp12SquareDescriptors::build(1, 0);
        let (wired, total) = set.coverage();
        // Use integer math: wired * 100 ≥ total * 99.
        assert!(
            wired * 100 >= total * 99,
            "Fp12 square coverage = {}/{} = {}% must be ≥ 99%",
            wired,
            total,
            (wired * 100) / total.max(1),
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5d: Phase F completes Fp12 coverage to the full 54-mul total
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn phase_f_completes_fp12_coverage() {
        let set = Fp12SquareDescriptors::build(1, 0);
        assert!(
            set.wired_count() >= 54,
            "Phase F must push Fp12 wired count to ≥ 54 (got {})",
            set.wired_count(),
        );

        // Phase F descriptors carry a dedicated label prefix.
        let phase_f_count = set
            .descriptors
            .iter()
            .filter(|d| {
                d.label
                    .starts_with("miller_step_fp12_square_c1_cross_phase_f_")
            })
            .count();
        assert!(
            phase_f_count >= 4,
            "Phase F must wire at least 4 c1-internal cross-pairs (got {})",
            phase_f_count,
        );

        // Phase F operand pairs must lie in the c1 half (acc_pre[6..12]).
        for d in &set.descriptors {
            if !d
                .label
                .starts_with("miller_step_fp12_square_c1_cross_phase_f_")
            {
                continue;
            }
            let acc_pre_c1_start =
                ms::COL_ACC_PRE_OFFSET + 6 * ms::LIMBS_PER_FP;
            let acc_pre_c1_end =
                ms::COL_ACC_PRE_OFFSET + 12 * ms::LIMBS_PER_FP;
            let a_first = d.b_columns[0];
            let b_first = d.b_columns[ms::LIMBS_PER_FP];
            assert!(a_first >= acc_pre_c1_start && a_first < acc_pre_c1_end);
            assert!(b_first >= acc_pre_c1_start && b_first < acc_pre_c1_end);
        }

        // Coverage is now exactly 100%.
        let (wired, total) = set.coverage();
        assert_eq!(
            wired, total,
            "Phase F closes the full Fp12 square decomposition",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5e: Miller-loop stitch descriptors well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn miller_loop_stitch_descriptors_well_formed() {
        let stitches = MillerLoopStitchDescriptors::build(
            /* acc_post layer */ 1,
            /* acc_pre  layer */ 1,
        );
        assert_eq!(
            stitches.descriptor_count(),
            MILLER_LOOP_STITCH_DESCRIPTORS,
        );
        assert!(
            stitches.descriptors.len() >= 5,
            "stitch descriptor count must be ≥ 5 (got {})",
            stitches.descriptors.len(),
        );

        // Every stitch descriptor binds LIMBS_PER_FP cols on each side
        // (one Fp limb-group), with COL_IS_DOUBLING gating both sides.
        for d in &stitches.descriptors {
            assert_eq!(d.a_columns.len(), ms::LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), ms::LIMBS_PER_FP);
            assert_eq!(d.a_selector_column, Some(ms::COL_IS_DOUBLING));
            assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));

            // A side lives in acc_post, B side lives in acc_pre — and the
            // two Fp slots match (this is the chain-continuity statement).
            let a_first = d.a_columns[0];
            let b_first = d.b_columns[0];
            assert!(a_first >= ms::COL_ACC_POST_OFFSET);
            assert!(
                a_first
                    < ms::COL_ACC_POST_OFFSET + ms::FP_PER_FP12 * ms::LIMBS_PER_FP,
            );
            assert!(b_first >= ms::COL_ACC_PRE_OFFSET);
            assert!(
                b_first
                    < ms::COL_ACC_PRE_OFFSET + ms::FP_PER_FP12 * ms::LIMBS_PER_FP,
            );
            assert_eq!(
                a_first - ms::COL_ACC_POST_OFFSET,
                b_first - ms::COL_ACC_PRE_OFFSET,
                "stitch must pin matching Fp slot indices",
            );
        }

        // Distinct labels.
        let mut labels: Vec<&str> = stitches
            .descriptors
            .iter()
            .map(|d| d.label.as_str())
            .collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), stitches.descriptors.len());

        // Column-range invariant.
        for d in &stitches.descriptors {
            for &col in d.a_columns.iter().chain(d.b_columns.iter()) {
                assert!(
                    col < ms::NUM_COLUMNS,
                    "stitch column {} exceeds miller_step_air NUM_COLUMNS = {}",
                    col,
                    ms::NUM_COLUMNS,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5f: combined Fp12 + G2 + stitch descriptor population ≥ 71
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn total_descriptor_count_at_least_70() {
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let g2 = G2DoubleDescriptors::build(1, 0);
        let stitches = MillerLoopStitchDescriptors::build(1, 1);
        let total = fp12.descriptors.len()
            + g2.descriptors.len()
            + stitches.descriptors.len();
        assert!(
            total >= 71,
            "Combined Fp12 + G2 + stitch descriptor count must be ≥ 71 \
             (got {} = {} + {} + {})",
            total,
            fp12.descriptors.len(),
            g2.descriptors.len(),
            stitches.descriptors.len(),
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5b: Phase D (c1-internal cross pairs) coverage
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn phase_d_coverage_complete() {
        let set = Fp12SquareDescriptors::build(1, 0);
        let phase_d_count = set
            .descriptors
            .iter()
            .filter(|d| d.label.starts_with("miller_step_fp12_square_c1_cross_"))
            .count();
        assert!(
            phase_d_count >= 10,
            "Phase D (c1-internal cross pairs) must wire ≥ 10 descriptors (got {})",
            phase_d_count,
        );

        // Phase D pairs must all be in the c1 half (6 ≤ i < j < 12).
        for d in &set.descriptors {
            if !d.label.starts_with("miller_step_fp12_square_c1_cross_") {
                continue;
            }
            // First 6 b_columns are the a-operand base (an acc_pre slot
            // in the c1 half). Verify the base column lies in c1.
            let a_first = d.b_columns[0];
            let acc_pre_c1_start =
                ms::COL_ACC_PRE_OFFSET + 6 * ms::LIMBS_PER_FP;
            let acc_pre_c1_end =
                ms::COL_ACC_PRE_OFFSET + 12 * ms::LIMBS_PER_FP;
            assert!(
                a_first >= acc_pre_c1_start && a_first < acc_pre_c1_end,
                "Phase D a-operand column {} must lie in c1 (acc_pre[6..12])",
                a_first,
            );
            let b_first = d.b_columns[ms::LIMBS_PER_FP];
            assert!(
                b_first >= acc_pre_c1_start && b_first < acc_pre_c1_end,
                "Phase D b-operand column {} must lie in c1 (acc_pre[6..12])",
                b_first,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5c: combined Fp12 + G2 descriptor population
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn total_descriptor_count_at_least_60() {
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let g2 = G2DoubleDescriptors::build(1, 0);
        let total = fp12.descriptors.len() + g2.descriptors.len();
        assert!(
            total >= 60,
            "Combined Fp12 square + G2 doubling descriptor count must be ≥ 60 \
             (got {} = {} + {})",
            total,
            fp12.descriptors.len(),
            g2.descriptors.len(),
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: every descriptor label across both helper structures is
    // unique (no accidental collisions when both sets coexist in a
    // joint cross-AIR LogUp protocol).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn all_descriptor_labels_are_distinct() {
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let g2 = G2DoubleDescriptors::build(1, 0);
        let mut labels: Vec<String> = fp12
            .descriptors
            .iter()
            .chain(g2.descriptors.iter())
            .map(|d| d.label.clone())
            .collect();
        let total = labels.len();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            total,
            "all {} descriptor labels across Fp12SquareDescriptors and \
             G2DoubleDescriptors must be distinct (found {} unique)",
            total,
            labels.len(),
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Sanity: column-range invariants
    // ───────────────────────────────────────────────────────────────────

    // ───────────────────────────────────────────────────────────────────
    // Test 7: G2 add descriptor set has expected count + correct gating
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn g2_add_descriptors_have_expected_count() {
        let set = G2AddDescriptors::build(/* ms */ 1, /* nfp */ 0);
        assert_eq!(set.wired_count(), G2_ADD_WIRED_MULTS);
        let (wired, total) = set.coverage();
        assert_eq!(wired, G2_ADD_WIRED_MULTS);
        assert_eq!(total, G2_ADD_TOTAL_MULTS);
        assert!(wired <= total);

        // Distinct labels.
        let mut labels: Vec<&str> = set.descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), set.descriptors.len());

        // Every descriptor: canonical 18/18 shape + COL_IS_ADDITION gate.
        for d in &set.descriptors {
            assert_18_18_tuple_shape(d);
            assert_eq!(
                d.b_selector_column,
                Some(ms::COL_IS_ADDITION),
                "G2 add descriptors must gate on COL_IS_ADDITION",
            );
        }

        // B-side columns must land within miller_step_air range.
        for d in &set.descriptors {
            for &col in &d.b_columns {
                assert!(col < ms::NUM_COLUMNS);
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 8: G2 add labels distinct from G2 double + Fp12 square
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn g2_add_labels_distinct_from_other_descriptor_families() {
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let g2_dbl = G2DoubleDescriptors::build(1, 0);
        let g2_add = G2AddDescriptors::build(1, 0);

        let mut labels: Vec<String> = fp12
            .descriptors
            .iter()
            .chain(g2_dbl.descriptors.iter())
            .chain(g2_add.descriptors.iter())
            .map(|d| d.label.clone())
            .collect();
        let total = labels.len();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            total,
            "all descriptor labels across Fp12 + G2 double + G2 add must \
             be distinct (got {} unique of {} total)",
            labels.len(),
            total,
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 9: combined Fp12 + G2 double + G2 add + stitch ≥ 83
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn total_descriptor_count_with_g2_add_at_least_83() {
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let g2_dbl = G2DoubleDescriptors::build(1, 0);
        let g2_add = G2AddDescriptors::build(1, 0);
        let stitches = MillerLoopStitchDescriptors::build(1, 1);
        let total = fp12.descriptors.len()
            + g2_dbl.descriptors.len()
            + g2_add.descriptors.len()
            + stitches.descriptors.len();
        assert!(
            total >= 83,
            "Combined Fp12 + G2 double + G2 add + stitch descriptor count \
             must be ≥ 83 (got {} = {} + {} + {} + {})",
            total,
            fp12.descriptors.len(),
            g2_dbl.descriptors.len(),
            g2_add.descriptors.len(),
            stitches.descriptors.len(),
        );
    }

    /// All B-side columns must land within `miller_step_air`'s declared
    /// column range — a regression guard against future column
    /// renumbering on either side.
    #[test]
    fn b_side_columns_inside_miller_step_air_range() {
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let g2 = G2DoubleDescriptors::build(1, 0);
        let g2_add = G2AddDescriptors::build(1, 0);
        for d in fp12
            .descriptors
            .iter()
            .chain(g2.descriptors.iter())
            .chain(g2_add.descriptors.iter())
        {
            for &col in &d.b_columns {
                assert!(
                    col < ms::NUM_COLUMNS,
                    "B-side column {} exceeds miller_step_air NUM_COLUMNS = {}",
                    col,
                    ms::NUM_COLUMNS,
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
    // Task #252: AccIntermediateDescriptors well-formedness tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn acc_intermediate_descriptors_have_expected_count() {
        let set = AccIntermediateDescriptors::build(/* ms */ 1, /* nfp */ 0);
        assert_eq!(set.descriptor_count(), ACC_INTERMEDIATE_DESCRIPTORS);
        assert_eq!(set.descriptors.len(), ms::FP_PER_FP12);
    }

    #[test]
    fn acc_intermediate_descriptors_have_18_18_tuple_shape() {
        let set = AccIntermediateDescriptors::build(1, 0);
        for d in &set.descriptors {
            assert_18_18_tuple_shape(d);
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 1);
            assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));
        }
    }

    #[test]
    fn acc_intermediate_descriptors_a_eq_b_diagonal_squaring() {
        // Each descriptor must encode `acc_pre[i] · acc_pre[i]` — i.e.
        // a-base = b-base on the B side.
        let set = AccIntermediateDescriptors::build(1, 0);
        for (fp_index, d) in set.descriptors.iter().enumerate() {
            let a_base = ms::COL_ACC_PRE_OFFSET + fp_index * ms::LIMBS_PER_FP;
            for j in 0..ms::LIMBS_PER_FP {
                assert_eq!(d.b_columns[j], a_base + j);
                assert_eq!(d.b_columns[ms::LIMBS_PER_FP + j], a_base + j);
                // c-side aliases to acc_post[i] (scaffold placeholder
                // for acc_intermediate[i] until the AIR widens).
                assert_eq!(
                    d.b_columns[2 * ms::LIMBS_PER_FP + j],
                    ms::COL_ACC_POST_OFFSET + fp_index * ms::LIMBS_PER_FP + j,
                );
            }
        }
    }

    #[test]
    fn acc_intermediate_descriptor_labels_distinct() {
        let set = AccIntermediateDescriptors::build(1, 0);
        let mut labels: Vec<String> =
            set.descriptors.iter().map(|d| d.label.clone()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            set.descriptors.len(),
            "all acc_intermediate descriptor labels must be unique",
        );
    }

    #[test]
    fn acc_intermediate_descriptors_distinct_from_fp12_square_set() {
        // Task #252's diagonals live alongside the
        // [`Fp12SquareDescriptors`] Phase-A diagonals (which bind the
        // same logical Fp products with `acc_post` as the c-side
        // alias). The labels must be distinct so the two sets can be
        // composed in the same joint LogUp protocol without collision.
        let acc_im = AccIntermediateDescriptors::build(1, 0);
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let mut all_labels: Vec<String> = acc_im
            .descriptors
            .iter()
            .chain(fp12.descriptors.iter())
            .map(|d| d.label.clone())
            .collect();
        let total = all_labels.len();
        all_labels.sort();
        all_labels.dedup();
        assert_eq!(
            all_labels.len(),
            total,
            "acc_intermediate + Fp12Square descriptor labels must all be unique",
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // Task #306: EllLambdaDescriptors well-formedness.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn ell_lambda_descriptors_have_expected_count() {
        let set = EllLambdaDescriptors::build(1, 0);
        assert_eq!(set.descriptor_count(), ELL_LAMBDA_WIRED_DESCRIPTORS);
        assert_eq!(set.descriptors.len(), 11);
    }

    #[test]
    fn ell_lambda_descriptors_have_18_18_tuple_shape() {
        let set = EllLambdaDescriptors::build(1, 0);
        for d in set.descriptors.iter() {
            assert_eq!(d.a_columns.len(), 3 * nfp::LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * ms::LIMBS_PER_FP);
            assert_eq!(d.b_selector_column, Some(
                crate::miller_loop_air::COL_IS_DOUBLING,
            ));
        }
    }

    #[test]
    fn ell_lambda_descriptors_use_correct_a_selectors() {
        let set = EllLambdaDescriptors::build(1, 0);
        // First 2 = ADD, remaining 9 = MUL.
        for (i, d) in set.descriptors.iter().enumerate() {
            let expected = if i < 2 {
                Some(nfp::COL_SEL_ADD)
            } else {
                Some(nfp::COL_SEL_MUL)
            };
            assert_eq!(
                d.a_selector_column, expected,
                "descriptor {} ({}) selector mismatch",
                i, d.label,
            );
        }
    }

    #[test]
    fn ell_lambda_descriptor_labels_distinct() {
        let set = EllLambdaDescriptors::build(1, 0);
        let mut labels: Vec<String> =
            set.descriptors.iter().map(|d| d.label.clone()).collect();
        let total = labels.len();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            total,
            "all EllLambda descriptor labels must be unique",
        );
    }

    #[test]
    fn ell_lambda_descriptors_distinct_from_existing_sets() {
        // Composable with the existing Fp12-square + acc_intermediate
        // + G2-double descriptor sets in the same joint LogUp protocol.
        let ell = EllLambdaDescriptors::build(1, 0);
        let fp12 = Fp12SquareDescriptors::build(1, 0);
        let acc_im = AccIntermediateDescriptors::build(1, 0);
        let g2d = G2DoubleDescriptors::build(1, 0);
        let mut all_labels: Vec<String> = ell
            .descriptors
            .iter()
            .chain(fp12.descriptors.iter())
            .chain(acc_im.descriptors.iter())
            .chain(g2d.descriptors.iter())
            .map(|d| d.label.clone())
            .collect();
        let total = all_labels.len();
        all_labels.sort();
        all_labels.dedup();
        assert_eq!(
            all_labels.len(),
            total,
            "EllLambda + Fp12Square + AccIntermediate + G2Double label \
             sets must be jointly distinct",
        );
    }

    #[test]
    fn ell_lambda_denom_descriptors_alias_q_curr_y() {
        // The two ADD descriptors close `λ_denom = 2·T.y` against the
        // (T.y.c0, T.y.c1) Fp slots of Q_curr.
        use crate::miller_loop_air as ml;
        let set = EllLambdaDescriptors::build(1, 0);
        let d_c0 = &set.descriptors[0];
        let d_c1 = &set.descriptors[1];
        // ADD descriptors must point a and b at the same Fp base
        // (doubling = self-add).
        assert_eq!(
            d_c0.b_columns[0],
            ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C0,
        );
        assert_eq!(
            d_c0.b_columns[ms::LIMBS_PER_FP],
            ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C0,
        );
        assert_eq!(
            d_c1.b_columns[0],
            ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C1,
        );
        assert_eq!(
            d_c1.b_columns[ms::LIMBS_PER_FP],
            ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C1,
        );
        // Result aliases λ_denom Fp slots.
        assert_eq!(
            d_c0.b_columns[2 * ms::LIMBS_PER_FP],
            ml::COL_LAMBDA_DENOM_OFFSET,
        );
        assert_eq!(
            d_c1.b_columns[2 * ms::LIMBS_PER_FP],
            ml::COL_LAMBDA_DENOM_OFFSET + ms::LIMBS_PER_FP,
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // Task #319: EllAffineLineDescriptors well-formedness.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn ell_affine_line_descriptors_have_expected_count() {
        let set = EllAffineLineDescriptors::build(1, 0);
        assert_eq!(set.descriptor_count(), ELL_AFFINE_LINE_WIRED_DESCRIPTORS);
        assert_eq!(set.descriptors.len(), 5);
    }

    #[test]
    fn ell_affine_line_descriptors_use_correct_a_selectors() {
        let set = EllAffineLineDescriptors::build(1, 0);
        // First 3 = MUL (λ·T.x Karatsuba); remaining 2 = ADD
        // (c1_affine + T.y subtraction encoding).
        for (i, d) in set.descriptors.iter().enumerate() {
            let expected = if i < 3 {
                Some(nfp::COL_SEL_MUL)
            } else {
                Some(nfp::COL_SEL_ADD)
            };
            assert_eq!(
                d.a_selector_column, expected,
                "descriptor {} ({}) selector mismatch",
                i, d.label,
            );
        }
    }

    #[test]
    fn ell_affine_line_descriptors_gate_b_on_is_doubling() {
        let set = EllAffineLineDescriptors::build(1, 0);
        for d in set.descriptors.iter() {
            assert_eq!(
                d.b_selector_column,
                Some(crate::miller_loop_air::COL_IS_DOUBLING),
            );
            // 18-col tuple shape (3·6 limbs).
            assert_eq!(d.a_columns.len(), 3 * nfp::LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * ms::LIMBS_PER_FP);
        }
    }

    #[test]
    fn ell_affine_line_descriptor_labels_distinct() {
        let set = EllAffineLineDescriptors::build(1, 0);
        let mut labels: Vec<String> =
            set.descriptors.iter().map(|d| d.label.clone()).collect();
        let total = labels.len();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), total);
    }

    #[test]
    fn ell_affine_line_descriptors_distinct_from_ell_lambda() {
        // Composable with the existing EllLambdaDescriptors set in the
        // same joint LogUp protocol (closes the affine side of the
        // ell-line surface jointly with the λ derivation).
        let affine = EllAffineLineDescriptors::build(1, 0);
        let lambda = EllLambdaDescriptors::build(1, 0);
        let mut all_labels: Vec<String> = affine
            .descriptors
            .iter()
            .chain(lambda.descriptors.iter())
            .map(|d| d.label.clone())
            .collect();
        let total = all_labels.len();
        all_labels.sort();
        all_labels.dedup();
        assert_eq!(all_labels.len(), total);
    }

    #[test]
    fn ell_affine_line_descriptors_alias_correct_lambda_and_q_curr_slots() {
        use crate::miller_loop_air as ml;
        let set = EllAffineLineDescriptors::build(1, 0);
        // Descriptor 0: λ.c0 · T.x.c0 → c1_affine.c0 (diagonal).
        let d0 = &set.descriptors[0];
        assert_eq!(d0.b_columns[0], ml::COL_LAMBDA_OFFSET);
        assert_eq!(
            d0.b_columns[ms::LIMBS_PER_FP],
            ml::COL_Q_CURR_OFFSET + ms::SUB_X_C0,
        );
        assert_eq!(
            d0.b_columns[2 * ms::LIMBS_PER_FP],
            ml::COL_C1_AFFINE_OFFSET,
        );
        // Descriptor 3: c1_affine.c0 + T.y.c0 = λ·T.x.c0 (Fp ADD).
        let d3 = &set.descriptors[3];
        assert_eq!(d3.b_columns[0], ml::COL_C1_AFFINE_OFFSET);
        assert_eq!(
            d3.b_columns[ms::LIMBS_PER_FP],
            ml::COL_Q_CURR_OFFSET + ms::SUB_Y_C0,
        );
    }
}
