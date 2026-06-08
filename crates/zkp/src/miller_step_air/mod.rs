//! BLS12-381 Miller-loop **single-step** AIR.
//!
//! # Purpose
//!
//! This module is the next algebraic stepping-stone after
//! [`crate::bls_pairing_air`]: it lays down the column layout, witness
//! shape, trace builder, and cross-AIR linkage descriptor for ONE
//! iteration of the optimal-ate Miller loop on BLS12-381. The full
//! pairing has roughly 64 Miller iterations + a final exponentiation;
//! the in-circuit version is built up incrementally by stitching
//! rows of this AIR together via shifted constraints, then composing
//! with the Fp arithmetic AIR ([`crate::nonnative_fp_air`]) through
//! cross-AIR LogUp.
//!
//! # Per-row witness (one Miller iteration)
//!
//! Each row commits:
//!
//!   * `Q_curr` — current G2 accumulator point (affine), as 4 Fp values
//!     `(x.c0, x.c1, y.c0, y.c1)` × 6 BE u64 limbs = 24 limb cells.
//!   * `Q_next` — `2·Q_curr` (when `is_doubling = 1`) or
//!     `Q_curr + Q_fixed` (when `is_addition = 1`), same layout.
//!   * `Q_fixed` — the original signature G2 point (constant across the
//!     loop), included so addition steps have access to it.
//!   * `line_value` — the Fp12 line-function value at this step
//!     evaluated at the G1 point P (12 Fp = 72 limb cells).
//!   * `acc_pre` — Fp12 Miller accumulator before this step (72 limbs).
//!   * `acc_post` — Fp12 Miller accumulator after this step =
//!     `acc_pre² · line_value`, same layout.
//!   * `is_doubling`, `is_addition` — exclusive {0,1} selectors.
//!
//! Column total: 4·6 (Q_curr) + 4·6 (Q_next) + 4·6 (Q_fixed) +
//! 12·6 (line) + 12·6 (acc_pre) + 12·6 (acc_post) + 2 (selectors)
//! + 60·6 (per-descriptor intermediate slots, Task #316) =
//! **650 columns**.
//!
//! # What is algebraically enforced (this AIR)
//!
//! 1. `is_doubling ∈ {0, 1}` — selector binarity.
//! 2. `is_addition ∈ {0, 1}` — selector binarity.
//! 3. `is_doubling · is_addition = 0` — mutual exclusion (at most one
//!    of the two operations fires per row; both zero ⇒ padding row).
//! 4. **Shifted row-chain continuity** (Q):
//!    `Q_curr[r+1] = Q_next[r]` for each of the 24 G2 limb columns.
//!    This is the load-bearing compositional property: stitching N rows
//!    end-to-end yields a 1-row → N-row Miller-loop trace with no gap
//!    between iterations.
//! 5. **Shifted row-chain continuity** (acc):
//!    `acc_pre[r+1] = acc_post[r]` for each of the 72 Fp12 limb columns.
//!
//! Constraints 4 and 5 are gated so that they hold trivially at the
//! padding boundary (when row `r+1` is all-zero padding, `acc_post[r]`
//! must be carried into the padding as zero — this is consistent with
//! how the trace builder lays out padded rows for a partial loop).
//!
//! # What is NOT yet enforced (deferred)
//!
//! The **deep arithmetic relations** — that `Q_next` actually equals
//! `2·Q_curr` (resp. `Q_curr + Q_fixed`), and that `acc_post` actually
//! equals `acc_pre² · line_value` — are NOT enforced row-locally
//! here. They are Fp2 / Fp12 multiplications, which decompose into
//! ~150 limb-level Fp multiplications and reductions per row — exactly
//! the workload of [`crate::nonnative_fp_air`]. The intended composition
//! is:
//!
//!   * Each Miller-step row exposes its Fp / Fp2 / Fp12 operands and
//!     results as committed limb columns.
//!   * For each such (operand₁, operand₂, result) triple, a row in
//!     `nonnative_fp_air` proves the corresponding `a · b ≡ r (mod p)`.
//!   * A cross-AIR LogUp descriptor between this AIR and
//!     `nonnative_fp_air` pins each triple as a multiset entry, closing
//!     the algebraic argument.
//!
//! Wiring those LogUp descriptors is several follow-up phases of work
//! (one per Fp / Fp2 op in `doubling_step` + `addition_step` + `ell` +
//! `Fp12::mul` + `Fp12::square` — call it ~250 LogUp tuples per row,
//! 250·64 = 16,000 tuples for the full loop). The present AIR
//! commits the witnesses so those LogUps have a stable target.
//!
//! Additional deferred items:
//!
//! * **Loop unrolling** — this AIR is ONE iteration. The full BLS12-381
//!   Miller loop scans bits of `|x| = 0xd201_0000_0001_0000` from MSB-1
//!   down to bit 0, yielding ~63 doubling rows + ~6 addition rows
//!   (depending on the bit pattern). Stitching 63 rows via the shifted
//!   continuity constraints (4) and (5) above is the next phase.
//! * **Final conjugation** — `x < 0` for BLS12-381 so the final
//!   accumulator must be conjugated. One extra row with an
//!   `is_conjugate` selector, deferred.
//! * **Final exponentiation** `(p¹² − 1)/r` — separate AIR/phase.
//! * **G1 point P** — the line `ell` evaluation multiplies the line
//!   coefficients by `(x_P, y_P)`. Those Fp values must enter this AIR
//!   from a cross-AIR LogUp into the BLS pairing AIR's `pk_x_limbs` /
//!   `pk_y_limbs` (G1 case) or sig coords (G2 case). For now we commit
//!   `line_value` as a flat Fp12 oracle; the `ell` decomposition lives
//!   in the witness builder.
//!
//! # Cross-AIR linkages
//!
//! * [`make_miller_step_q_curr_to_bls_pairing_descriptor`] — A side =
//!   this AIR's `Q_curr` G2 limbs gated by `IS_FIRST_ROW` (encoded as
//!   `is_doubling + is_addition` on the first row), B side = the
//!   `bls_pairing_air`'s `sig_x_c0 / sig_x_c1 / sig_y_c0 / sig_y_c1`
//!   limbs gated by `IS_REAL`. 24-tuple LogUp. This binds the
//!   *initial* Q to the signature; subsequent Q_curr rows are bound by
//!   the shifted continuity constraint (4) above.
//!
//! # Soundness summary
//!
//! After this phase: a malicious prover cannot break the row-chain
//! continuity or selector binarity. They CAN still cheat on the
//! per-row arithmetic (commit any Q_next/acc_post they like) — that
//! gap closes when the nonnative_fp_air cross-AIR LogUps are added.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::nonnative_fp::Fp;
use crate::nonnative_tower::Fp12;
use crate::pairing::G2Affine;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of 64-bit limbs per Fp element (381-bit field → 6 × 64).
pub const LIMBS_PER_FP: usize = 6;
/// Number of Fp values per Fp2 element.
pub const FP_PER_FP2: usize = 2;
/// Number of Fp2 values per G2 affine point (x, y).
pub const FP2_PER_G2: usize = 2;
/// Number of Fp values per G2 affine point (x.c0, x.c1, y.c0, y.c1).
pub const FP_PER_G2: usize = FP2_PER_G2 * FP_PER_FP2;
/// Number of limbs per G2 affine point = 4 × 6 = 24.
pub const LIMBS_PER_G2: usize = FP_PER_G2 * LIMBS_PER_FP;
/// Number of Fp values per Fp12 element (top-level Fp basis).
pub const FP_PER_FP12: usize = 12;
/// Number of limbs per Fp12 element = 12 × 6 = 72.
pub const LIMBS_PER_FP12: usize = FP_PER_FP12 * LIMBS_PER_FP;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_Q_CURR_OFFSET: usize = 0;
pub const COL_Q_NEXT_OFFSET: usize = COL_Q_CURR_OFFSET + LIMBS_PER_G2;
pub const COL_Q_FIXED_OFFSET: usize = COL_Q_NEXT_OFFSET + LIMBS_PER_G2;
pub const COL_LINE_VALUE_OFFSET: usize = COL_Q_FIXED_OFFSET + LIMBS_PER_G2;
pub const COL_ACC_PRE_OFFSET: usize = COL_LINE_VALUE_OFFSET + LIMBS_PER_FP12;
pub const COL_ACC_POST_OFFSET: usize = COL_ACC_PRE_OFFSET + LIMBS_PER_FP12;
pub const COL_IS_DOUBLING: usize = COL_ACC_POST_OFFSET + LIMBS_PER_FP12;
pub const COL_IS_ADDITION: usize = COL_IS_DOUBLING + 1;

// ─── Task #316: Per-descriptor intermediate result columns ────────────
//
// Task #307 populated 90 cross-AIR LogUp descriptors (`Fp12SquareDescriptors` +
// `AccIntermediateDescriptors` + `G2DoubleDescriptors` + `G2AddDescriptors`).
// 60 of those collided on `c_base` because multiple descriptors aliased
// the same `acc_post[k]` or `Q_next[k]` slot as their result column —
// the documented "last-write-wins" scaffold pattern. This widening adds
// **one dedicated 6-limb Fp witness slot per collision descriptor**, so
// every descriptor has its own (a, b, c) result column triple and all
// 90 closures can hold simultaneously.
//
// Layout (60 Fp slots = 360 limb columns):
//
//   * `[0..42)`  Fp12 phases B/C/D/E/F (15 + 8 + 10 + 5 + 4 = 42 slots).
//     Sub-ranges in build order:
//       - B[0..15]  → intermediate[0..15]
//       - C[0..8]   → intermediate[15..23]
//       - D[0..10]  → intermediate[23..33]
//       - E[0..5]   → intermediate[33..38]
//       - F[0..4]   → intermediate[38..42]
//   * `[42..50)` G2-doubling B/D/E/F descriptors (8 slots), corresponding
//     to G2Double descriptor indices [2, 5, 6, 7, 8, 9, 10, 11].
//   * `[50..60)` G2-add B/C/D/E/F descriptors (10 slots), corresponding
//     to G2Add descriptor indices [2, 3, 4, 5, 6, 7, 8, 9, 10, 11].

/// Number of dedicated intermediate Fp slots (one per collision descriptor).
pub const NUM_INTERMEDIATE_FP_SLOTS: usize = 60;
/// Total intermediate limb columns (60 Fp × 6 limbs).
pub const NUM_INTERMEDIATE_LIMBS: usize =
    NUM_INTERMEDIATE_FP_SLOTS * LIMBS_PER_FP;
/// Offset of the first intermediate Fp slot's first limb column.
pub const COL_INTERMEDIATE_OFFSET: usize = COL_IS_ADDITION + 1;

// Sub-range starts within the intermediate block (Fp-slot units).
pub const INTERMEDIATE_FP12_B_START: usize = 0;
pub const INTERMEDIATE_FP12_C_START: usize = INTERMEDIATE_FP12_B_START + 15;
pub const INTERMEDIATE_FP12_D_START: usize = INTERMEDIATE_FP12_C_START + 8;
pub const INTERMEDIATE_FP12_E_START: usize = INTERMEDIATE_FP12_D_START + 10;
pub const INTERMEDIATE_FP12_F_START: usize = INTERMEDIATE_FP12_E_START + 5;
pub const INTERMEDIATE_G2_DBL_START: usize = INTERMEDIATE_FP12_F_START + 4;
pub const INTERMEDIATE_G2_ADD_START: usize = INTERMEDIATE_G2_DBL_START + 8;

/// Compute the limb base column for intermediate Fp slot `k`.
#[inline]
pub const fn intermediate_fp_base(slot: usize) -> usize {
    COL_INTERMEDIATE_OFFSET + slot * LIMBS_PER_FP
}

pub const NUM_COLUMNS: usize = COL_INTERMEDIATE_OFFSET + NUM_INTERMEDIATE_LIMBS;

// Within-G2 sub-offsets (relative to a base of either COL_Q_CURR_OFFSET,
// COL_Q_NEXT_OFFSET, or COL_Q_FIXED_OFFSET).
pub const SUB_X_C0: usize = 0 * LIMBS_PER_FP;
pub const SUB_X_C1: usize = 1 * LIMBS_PER_FP;
pub const SUB_Y_C0: usize = 2 * LIMBS_PER_FP;
pub const SUB_Y_C1: usize = 3 * LIMBS_PER_FP;

/// Row-local constraints:
///   0: `is_doubling ∈ {0, 1}`
///   1: `is_addition ∈ {0, 1}`
///   2: `is_doubling · is_addition = 0`
pub const NUM_ROW_CONSTRAINTS: usize = 3;

/// Shifted-row constraint categories (one body each, combined per-row
/// via a β-RLC over their limb sub-bodies):
///   0: `Q_curr[r+1] = Q_next[r]` (24 sub-bodies)
///   1: `acc_pre[r+1] = acc_post[r]` (72 sub-bodies)
pub const NUM_SHIFTED: usize = 2;

// ─── Witness ──────────────────────────────────────────────────────────

/// One Miller iteration step witness.
///
/// All Fp / Fp2 / Fp12 elements are stored in the host-side reference
/// types ([`Fp`], [`Fp12`], [`G2Affine`]); the trace builder flattens
/// them into 6-limb big-endian columns matching the
/// [`crate::nonnative_fp_air`] convention.
#[derive(Clone, Debug)]
pub struct MillerStepRow {
    /// Current G2 accumulator point.
    pub q_curr: G2Affine,
    /// Next G2 accumulator point: `2·q_curr` if doubling, or
    /// `q_curr + q_fixed` if addition.
    pub q_next: G2Affine,
    /// Original signature G2 point (constant across all rows of one
    /// Miller loop).
    pub q_fixed: G2Affine,
    /// Fp12 line-function value at this step, already evaluated at G1
    /// point P (i.e. the `ell()` output).
    pub line_value: Fp12,
    /// Miller accumulator before this step.
    pub acc_pre: Fp12,
    /// Miller accumulator after this step: `acc_pre² · line_value` on
    /// doubling rows, `acc_pre · line_value` on addition rows (matching
    /// the real Miller-loop recurrence, where the `f.square()` is fused
    /// into the doubling step only).
    pub acc_post: Fp12,
    /// Selector: this row is a doubling step.
    pub is_doubling: bool,
    /// Selector: this row is an addition step (BLS x has a set bit at
    /// this iteration).
    pub is_addition: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MillerStepWitness {
    pub rows: Vec<MillerStepRow>,
}

impl MillerStepWitness {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Append a doubling-step row. Computes `q_next = 2 · q_curr` and
    /// `acc_post = acc_pre² · line_value` via the host-side reference
    /// implementation, so the row is internally consistent by
    /// construction.
    pub fn push_doubling(
        &mut self,
        q_curr: G2Affine,
        q_fixed: G2Affine,
        line_value: Fp12,
        acc_pre: Fp12,
    ) {
        let q_next = q_curr.double();
        let acc_post = acc_pre.square().mul(&line_value);
        self.rows.push(MillerStepRow {
            q_curr,
            q_next,
            q_fixed,
            line_value,
            acc_pre,
            acc_post,
            is_doubling: true,
            is_addition: false,
        });
    }

    /// Append an addition-step row. Computes `q_next = q_curr + q_fixed`
    /// and `acc_post = acc_pre · line_value` host-side.
    ///
    /// Note: in the real Miller-loop arithmetic, an addition step does
    /// **not** square the accumulator (only doubling steps do — the
    /// `f = f.square()` step happens once per outer-loop iteration,
    /// followed by `f = ell(f, doubling_coeffs)` and optionally
    /// `f = ell(f, addition_coeffs)`). So addition rows reduce to
    /// `acc_post = acc_pre · line_value` (one Fp12 multiply per row,
    /// no square). The constraint surface treats the per-row arithmetic
    /// relation as oracle / deferred regardless of row type, so this
    /// row-type-dependent semantics is encoded entirely in the witness
    /// builder.
    pub fn push_addition(
        &mut self,
        q_curr: G2Affine,
        q_fixed: G2Affine,
        line_value: Fp12,
        acc_pre: Fp12,
    ) {
        let q_next = q_curr.add(&q_fixed);
        let acc_post = acc_pre.mul(&line_value);
        self.rows.push(MillerStepRow {
            q_curr,
            q_next,
            q_fixed,
            line_value,
            acc_pre,
            acc_post,
            is_doubling: false,
            is_addition: true,
        });
    }

    /// Append a raw row without recomputing `q_next` / `acc_post` —
    /// used by tampering tests to commit values that violate the
    /// (deferred) arithmetic relations.
    pub fn push_raw(&mut self, row: MillerStepRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

/// Flatten an [`Fp`] into 6 big-endian u64 scalars and write into
/// `columns[base..base+6]` at row `r`.
fn write_fp_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    fp: &Fp,
    curve: CurveType,
) {
    for j in 0..LIMBS_PER_FP {
        columns[base + j][r] = Scalar::from_u64(fp.limbs[j], curve);
    }
}

/// Flatten a G2 affine point into 24 limb cells starting at `base`.
fn write_g2_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    q: &G2Affine,
    curve: CurveType,
) {
    write_fp_limbs(columns, base + SUB_X_C0, r, &q.x.c0, curve);
    write_fp_limbs(columns, base + SUB_X_C1, r, &q.x.c1, curve);
    write_fp_limbs(columns, base + SUB_Y_C0, r, &q.y.c0, curve);
    write_fp_limbs(columns, base + SUB_Y_C1, r, &q.y.c1, curve);
}

/// Flatten an Fp12 into 72 limb cells starting at `base`. Layout (in
/// order of [`Fp12`]'s `c0/c1` of `Fp6`'s `c0/c1/c2` of `Fp2`'s
/// `c0/c1`):
///
///   `[c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1,
///     c1.c0.c0, c1.c0.c1, c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1]`
///
/// each one 6 limbs big-endian.
fn write_fp12_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    f: &Fp12,
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
        write_fp_limbs(columns, base + i * LIMBS_PER_FP, r, fp, curve);
    }
}

pub fn build_trace_polynomials(
    witness: &MillerStepWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        write_g2_limbs(&mut columns, COL_Q_CURR_OFFSET, r, &row.q_curr, curve);
        write_g2_limbs(&mut columns, COL_Q_NEXT_OFFSET, r, &row.q_next, curve);
        write_g2_limbs(&mut columns, COL_Q_FIXED_OFFSET, r, &row.q_fixed, curve);
        write_fp12_limbs(&mut columns, COL_LINE_VALUE_OFFSET, r, &row.line_value, curve);
        write_fp12_limbs(&mut columns, COL_ACC_PRE_OFFSET, r, &row.acc_pre, curve);
        write_fp12_limbs(&mut columns, COL_ACC_POST_OFFSET, r, &row.acc_post, curve);
        columns[COL_IS_DOUBLING][r] = if row.is_doubling { one.clone() } else { zero.clone() };
        columns[COL_IS_ADDITION][r] = if row.is_addition { one.clone() } else { zero.clone() };
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

/// Constraint-system handle for the single-step Miller AIR.
pub struct MillerStepConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl MillerStepConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for MillerStepConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_doubling_binary".into(),
            "is_addition_binary".into(),
            "selectors_mutually_exclusive".into(),
        ]
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

        // 0: is_doubling binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_DOUBLING][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1: is_addition binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_ADDITION][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 2: mutual exclusion: is_doubling * is_addition = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let d = &columns[COL_IS_DOUBLING][r];
                let a = &columns[COL_IS_ADDITION][r];
                c[r] = d.mul(a);
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

        // 0
        {
            let v = &col_evals[COL_IS_DOUBLING];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1
        {
            let v = &col_evals[COL_IS_ADDITION];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2
        {
            let d = &col_evals[COL_IS_DOUBLING];
            let a = &col_evals[COL_IS_ADDITION];
            acc = acc.add(&alpha_pow.mul(&d.mul(a)));
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

        // 0: is_doubling binary.
        {
            let v = &col_coeffs[COL_IS_DOUBLING];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_addition binary.
        {
            let v = &col_coeffs[COL_IS_ADDITION];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: mutual exclusion.
        {
            let d = &col_coeffs[COL_IS_DOUBLING];
            let a = &col_coeffs[COL_IS_ADDITION];
            let body = poly_mul(d, a, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_DOUBLING, COL_IS_ADDITION]
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
        // Limb columns are 64-bit values; range-checking them at this
        // layer would add 290 × 64-bit checks per row, dwarfing the
        // (3) row-local constraint surface. The intended composition
        // is that nonnative_fp_air consumes these limbs via cross-AIR
        // LogUp and contributes the 64-bit range checks itself.
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Every column whose value at row r appears in any shifted
        // (cross-row) constraint as `col[r+1]` must be listed here.
        //
        // Shifted constraint 0 needs `Q_curr[r+1]` (24 columns).
        // Shifted constraint 1 needs `acc_pre[r+1]` (72 columns).
        let mut idx: Vec<usize> = Vec::with_capacity(LIMBS_PER_G2 + LIMBS_PER_FP12);
        for j in 0..LIMBS_PER_G2 {
            idx.push(COL_Q_CURR_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP12 {
            idx.push(COL_ACC_PRE_OFFSET + j);
        }
        idx
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }
}

// ─── Shifted (cross-row) constraint helpers ───────────────────────────
//
// Each shifted-constraint category yields ONE body per row, formed as a
// β-RLC over the per-limb sub-bodies. β = α to keep a single Schwartz-
// Zippel challenge.

/// Per-row evaluation of shifted constraint `i` at row `r`, given full
/// columns and the `(curr, next)` row slice already provided by the
/// caller. Returns the body value (zero on honest rows).
pub fn evaluate_shifted_row(
    columns: &[&Vec<Scalar>],
    r: usize,
    next: usize,
    constraint_index: usize,
    alpha: &Scalar,
) -> Scalar {
    let curve = alpha.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut beta_pow = Scalar::one(curve);

    match constraint_index {
        0 => {
            // Q_curr[r+1] = Q_next[r] — 24 sub-bodies.
            for j in 0..LIMBS_PER_G2 {
                let a = &columns[COL_Q_CURR_OFFSET + j][next];
                let b = &columns[COL_Q_NEXT_OFFSET + j][r];
                let body = a.sub(b);
                acc = acc.add(&beta_pow.mul(&body));
                beta_pow = beta_pow.mul(alpha);
            }
        }
        1 => {
            // acc_pre[r+1] = acc_post[r] — 72 sub-bodies.
            for j in 0..LIMBS_PER_FP12 {
                let a = &columns[COL_ACC_PRE_OFFSET + j][next];
                let b = &columns[COL_ACC_POST_OFFSET + j][r];
                let body = a.sub(b);
                acc = acc.add(&beta_pow.mul(&body));
                beta_pow = beta_pow.mul(alpha);
            }
        }
        _ => {}
    }
    acc
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Cross-AIR LogUp descriptor: this AIR's `Q_curr` G2 limbs (24 cols)
/// gated by `IS_DOUBLING + IS_ADDITION` ↔ `bls_pairing_air`'s
/// `sig_x_c0 / sig_x_c1 / sig_y_c0 / sig_y_c1` limbs (24 cols) gated by
/// `IS_REAL`.
///
/// 24-tuple LogUp. Pins the initial Q_curr to the signature
/// committed by the BLS pairing AIR. (Subsequent Q_curr rows are pinned
/// by the shifted continuity constraint in this AIR.)
///
/// **Caveat**: with only a `(is_doubling + is_addition)` gate, this
/// descriptor matches Q_curr from *every* active Miller-step row, not
/// just the first one. The full algebraic binding to the *first row's*
/// Q_curr requires a dedicated `IS_FIRST_ROW` selector column (a
/// "boundary" selector that is 1 only at row 0 of the loop). That
/// selector is straightforward to add but does not yet exist in this
/// scaffold; the descriptor below is the column-shape spec the future
/// boundary-gated linkage will use.
pub fn make_miller_step_q_curr_to_bls_pairing_descriptor(
    miller_step_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;
    // A-side columns: bls_pairing_air sig_x_c0 / sig_x_c1 / sig_y_c0 /
    // sig_y_c1 limbs (4 × 6 = 24).
    let a_columns: Vec<usize> = {
        let mut v = Vec::with_capacity(LIMBS_PER_G2);
        for j in 0..LIMBS_PER_FP {
            v.push(bp::COL_SIG_X_C0_LIMB_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP {
            v.push(bp::COL_SIG_X_C1_LIMB_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP {
            v.push(bp::COL_SIG_Y_C0_LIMB_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP {
            v.push(bp::COL_SIG_Y_C1_LIMB_OFFSET + j);
        }
        v
    };
    // B-side columns: Q_curr in this AIR, in (x.c0, x.c1, y.c0, y.c1)
    // order to align with the A-side ordering above.
    let b_columns: Vec<usize> = (0..LIMBS_PER_G2)
        .map(|j| COL_Q_CURR_OFFSET + j)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "miller_step_q_curr_to_bls_pairing_v1".into(),
        a_layer_index: bls_pairing_layer_index,
        a_columns,
        a_selector_column: Some(bp::COL_IS_REAL),
        b_layer_index: miller_step_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_DOUBLING),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairing::G2Affine;

    fn sample_q() -> G2Affine {
        G2Affine::generator()
    }

    fn sample_q2() -> G2Affine {
        G2Affine::generator().double()
    }

    fn sample_line() -> Fp12 {
        // Any non-trivial Fp12 value works as a stand-in for an `ell`
        // output; the (deferred) algebraic check that this equals the
        // actual line evaluation is wired through cross-AIR LogUp.
        Fp12::one().add(&Fp12::one())
    }

    fn sample_acc_pre() -> Fp12 {
        Fp12::one()
    }

    fn honest_witness(num_rows: usize) -> MillerStepWitness {
        let mut w = MillerStepWitness::new();
        let q_fixed = sample_q();
        let mut q = sample_q();
        let mut acc = sample_acc_pre();
        let line = sample_line();
        for i in 0..num_rows {
            if i % 2 == 0 {
                w.push_doubling(q, q_fixed, line, acc.clone());
            } else {
                w.push_addition(q, q_fixed, line, acc.clone());
            }
            // Continuity: next row's Q_curr / acc_pre = this row's
            // Q_next / acc_post.
            let row = w.rows.last().unwrap();
            q = row.q_next;
            acc = row.acc_post.clone();
        }
        w
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: witness builds + columns populate
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_and_trace_has_expected_shape() {
        let w = honest_witness(2);
        assert_eq!(w.rows.len(), 2);
        assert!(w.rows[0].is_doubling && !w.rows[0].is_addition);
        assert!(!w.rows[1].is_doubling && w.rows[1].is_addition);

        // Row 0 doubling: q_next == 2 q_curr (host-side reference).
        assert_eq!(w.rows[0].q_next, sample_q2());
        // Row 0 acc_post == acc_pre² · line_value.
        let expected_acc_post = sample_acc_pre().square().mul(&sample_line());
        assert_eq!(w.rows[0].acc_post, expected_acc_post);

        // Continuity holds in the witness: row 1's q_curr/acc_pre
        // equals row 0's q_next/acc_post.
        assert_eq!(w.rows[1].q_curr, w.rows[0].q_next);
        assert_eq!(w.rows[1].acc_pre, w.rows[0].acc_post);

        // Trace has expected column count and the selectors are set.
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);
        assert!(
            trace.columns[COL_IS_DOUBLING].evaluations[0].sub(&one).is_zero(),
            "row 0 is_doubling should be 1",
        );
        assert!(
            trace.columns[COL_IS_ADDITION].evaluations[0].is_zero(),
            "row 0 is_addition should be 0",
        );
        assert!(
            trace.columns[COL_IS_DOUBLING].evaluations[1].is_zero(),
            "row 1 is_doubling should be 0",
        );
        assert!(
            trace.columns[COL_IS_ADDITION].evaluations[1].sub(&one).is_zero(),
            "row 1 is_addition should be 1",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: constraints zero on honest witness
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = MillerStepConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "row-local constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
        // Shifted constraints zero at row 0 (the only row that has a
        // valid `next` neighbor in the witness — row 1 → padding).
        let curve = CurveType::Bls12381;
        let alpha = Scalar::from_u64(7, curve);
        for cidx in 0..NUM_SHIFTED {
            let body = evaluate_shifted_row(&col_refs, 0, 1, cidx, &alpha);
            assert!(
                body.is_zero(),
                "shifted constraint {} at row 0 = {:?} (expected zero)",
                cidx, body,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: tampered Q_next breaks chain continuity (shifted constraint 0)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_q_next_breaks_chain_continuity() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();

        // Tamper limb 0 of Q_next.x.c0 on row 0 (add 1). This breaks the
        // shifted continuity to row 1's Q_curr.
        let curve = CurveType::Bls12381;
        let original = cols[COL_Q_NEXT_OFFSET + SUB_X_C0][0].clone();
        cols[COL_Q_NEXT_OFFSET + SUB_X_C0][0] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();

        // Row-local constraints unaffected — they don't touch Q.
        let cs = MillerStepConstraintSystem::new(trace.num_rows);
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "row-local constraint {} at row {} unexpectedly fired = {:?}",
                    i, r, val,
                );
            }
        }

        // Shifted constraint 0 (Q chain continuity) fires at row 0.
        let alpha = Scalar::from_u64(7, curve);
        let body = evaluate_shifted_row(&col_refs, 0, 1, 0, &alpha);
        assert!(
            !body.is_zero(),
            "tampered Q_next must break Q chain continuity (shifted 0)",
        );

        // Shifted constraint 1 (acc chain) unaffected.
        let acc_body = evaluate_shifted_row(&col_refs, 0, 1, 1, &alpha);
        assert!(
            acc_body.is_zero(),
            "tampering Q must NOT affect acc chain continuity",
        );
    }

    #[test]
    fn tampered_acc_post_breaks_acc_continuity() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        // Tamper limb 5 of acc_post.c0.c0.c0 on row 0.
        let original = cols[COL_ACC_POST_OFFSET + 5][0].clone();
        cols[COL_ACC_POST_OFFSET + 5][0] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let alpha = Scalar::from_u64(11, curve);

        let body = evaluate_shifted_row(&col_refs, 0, 1, 1, &alpha);
        assert!(
            !body.is_zero(),
            "tampered acc_post must break acc chain continuity (shifted 1)",
        );
        // Q-side chain unaffected.
        let q_body = evaluate_shifted_row(&col_refs, 0, 1, 0, &alpha);
        assert!(
            q_body.is_zero(),
            "tampering acc_post must NOT affect Q chain continuity",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: row-local selector constraints fire on tampering
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn is_doubling_binary_fires_on_nonbinary_selector() {
        let w = honest_witness(1);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_DOUBLING][0] = Scalar::from_u64(3, CurveType::Bls12381);
        let cs = MillerStepConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[0][0].is_zero(),
            "non-binary is_doubling must fire constraint 0",
        );
    }

    #[test]
    fn mutual_exclusion_fires_when_both_selectors_set() {
        let w = honest_witness(1);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);
        cols[COL_IS_DOUBLING][0] = one.clone();
        cols[COL_IS_ADDITION][0] = one;
        let cs = MillerStepConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[2][0].is_zero(),
            "is_doubling = is_addition = 1 must fire mutual-exclusion (constraint 2)",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: cross-AIR linkage descriptor is well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn q_curr_to_bls_pairing_descriptor_well_formed() {
        let d = make_miller_step_q_curr_to_bls_pairing_descriptor(1, 0);
        assert_eq!(d.label, "miller_step_q_curr_to_bls_pairing_v1");
        assert_eq!(d.a_layer_index, 0, "A side = BLS pairing layer");
        assert_eq!(d.b_layer_index, 1, "B side = miller_step layer");
        assert_eq!(d.a_columns.len(), LIMBS_PER_G2);
        assert_eq!(d.b_columns.len(), LIMBS_PER_G2);

        // A-side first column = sig_x_c0 limb 0, last = sig_y_c1 limb 5.
        use crate::bls_pairing_air as bp;
        assert_eq!(d.a_columns[0], bp::COL_SIG_X_C0_LIMB_OFFSET);
        assert_eq!(
            d.a_columns[LIMBS_PER_G2 - 1],
            bp::COL_SIG_Y_C1_LIMB_OFFSET + LIMBS_PER_FP - 1,
        );
        // B-side first / last column.
        assert_eq!(d.b_columns[0], COL_Q_CURR_OFFSET);
        assert_eq!(d.b_columns[LIMBS_PER_G2 - 1], COL_Q_CURR_OFFSET + LIMBS_PER_G2 - 1);

        // Selectors.
        assert_eq!(d.a_selector_column, Some(bp::COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(COL_IS_DOUBLING));
    }

    // ───────────────────────────────────────────────────────────────────
    // Sanity: column layout & shifted-index list
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_Q_CURR_OFFSET, 0);
        assert_eq!(COL_Q_NEXT_OFFSET, LIMBS_PER_G2);
        assert_eq!(COL_Q_FIXED_OFFSET, 2 * LIMBS_PER_G2);
        assert_eq!(COL_LINE_VALUE_OFFSET, 3 * LIMBS_PER_G2);
        assert_eq!(COL_ACC_PRE_OFFSET, 3 * LIMBS_PER_G2 + LIMBS_PER_FP12);
        assert_eq!(COL_ACC_POST_OFFSET, 3 * LIMBS_PER_G2 + 2 * LIMBS_PER_FP12);
        assert_eq!(COL_IS_DOUBLING, 3 * LIMBS_PER_G2 + 3 * LIMBS_PER_FP12);
        assert_eq!(COL_IS_ADDITION, COL_IS_DOUBLING + 1);
        assert_eq!(COL_INTERMEDIATE_OFFSET, COL_IS_ADDITION + 1);
        assert_eq!(
            NUM_COLUMNS,
            COL_INTERMEDIATE_OFFSET + NUM_INTERMEDIATE_LIMBS,
        );
        // 3·24 + 3·72 + 2 + 60·6 = 72 + 216 + 2 + 360 = 650.
        assert_eq!(NUM_COLUMNS, 650);
        // Task #316: intermediate sub-range invariants.
        assert_eq!(NUM_INTERMEDIATE_FP_SLOTS, 60);
        assert_eq!(INTERMEDIATE_FP12_B_START, 0);
        assert_eq!(INTERMEDIATE_FP12_C_START, 15);
        assert_eq!(INTERMEDIATE_FP12_D_START, 23);
        assert_eq!(INTERMEDIATE_FP12_E_START, 33);
        assert_eq!(INTERMEDIATE_FP12_F_START, 38);
        assert_eq!(INTERMEDIATE_G2_DBL_START, 42);
        assert_eq!(INTERMEDIATE_G2_ADD_START, 50);
    }

    #[test]
    fn shifted_column_indices_cover_q_curr_and_acc_pre() {
        let cs = MillerStepConstraintSystem::new(2);
        let idx = cs.shifted_column_indices();
        assert_eq!(idx.len(), LIMBS_PER_G2 + LIMBS_PER_FP12);
        for j in 0..LIMBS_PER_G2 {
            assert_eq!(idx[j], COL_Q_CURR_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP12 {
            assert_eq!(idx[LIMBS_PER_G2 + j], COL_ACC_PRE_OFFSET + j);
        }
    }
}
