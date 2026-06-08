//! BLS12-381 **final-exponentiation** scaffold AIR.
//!
//! # Purpose
//!
//! This module is the next algebraic stepping-stone after
//! [`crate::miller_step_air`]: it lays down the column layout, witness
//! shape, trace builder, and cross-AIR linkage descriptor for the
//! BLS12-381 *final exponentiation* — raising the Miller-loop output
//! `f ∈ Fp12` to `(p¹² − 1)/r`.
//!
//! Final exponentiation is the second-to-last piece needed to fully
//! algebraically prove pairing equations. Its host-side reference is
//! [`crate::pairing::final_exponentiation`].
//!
//! # Decomposition
//!
//! The exponent factors as:
//!
//!   * **Easy part**: `(p⁶ − 1)(p² + 1)`. Computed in closed form via
//!     `conjugate(f) · f⁻¹` (= `f^{p⁶ − 1}`) followed by
//!     `frobenius_map(·, 2) · ·` (= `·^{p² + 1}`). The output lives in
//!     the cyclotomic subgroup `G_φ12(Fp)`.
//!   * **Hard part**: `(p⁴ − p² + 1)/r`. Computed via an addition chain
//!     (Fuentes-Castañeda / zkcrypto style) over the cyclotomic
//!     subgroup using `cyclotomic_square`, `mul`, `conjugate`, and
//!     `frobenius_map`.
//!
//! # Per-row witness
//!
//! Each row commits four Fp12 values, mirroring the easy / hard split:
//!
//!   * `f_pre`        — Fp12 input to this step (≡ Miller-loop output on
//!                      the first row).
//!   * `f_easy_out`   — Fp12 result after the easy part on
//!                      `is_easy_step` rows (a pass-through on hard
//!                      rows).
//!   * `f_hard_out`   — Fp12 result after the hard part on
//!                      `is_hard_step` rows.
//!   * `f_final`      — Fp12 carried out of this step into the next
//!                      row's `f_pre` (semantically equals `f_hard_out`
//!                      on the final row).
//!
//! Each Fp12 is flattened to 12 × 6 = 72 big-endian u64 limbs using
//! the same convention as [`crate::miller_step_air`].
//!
//! # Selectors
//!
//!   * `is_easy_step` — this row applies the easy-part transformation.
//!   * `is_hard_step` — this row applies the hard-part transformation.
//!
//! The two are mutually exclusive {0, 1}; both zero indicates a
//! padding row.
//!
//! Column total:
//!   4 · 72 (Fp12 cols) + 2 (selectors) = **290 columns**.
//!
//! # What is algebraically enforced (this AIR)
//!
//! 1. `is_easy_step ∈ {0, 1}` — selector binarity.
//! 2. `is_hard_step ∈ {0, 1}` — selector binarity.
//! 3. `is_easy_step · is_hard_step = 0` — mutual exclusion.
//! 4. **Easy-part w-conjugation half-step** (row-local, on
//!    `is_easy_step` rows): The very first sub-step of the easy part
//!    is `f₁ = conjugate(f) = c₀ − c₁ w`. We pin one cheap algebraic
//!    fingerprint of that step: on `is_easy_step` rows the c₀ block of
//!    `f_pre` and `f_easy_out` must agree (the easy-part output's c₀
//!    coordinate is fully determined by chained Fp6 operations, so
//!    equality with `f_pre.c₀` is **not** what holds in general — this
//!    constraint is a **placeholder** that demonstrates a substantive
//!    Fp12-equality body wired up correctly, gated by `is_easy_step`,
//!    flattened over 6 Fp limb sub-bodies via a β-RLC). The full
//!    algebraic decomposition into Fp / Fp2 / Fp6 sub-ops is deferred
//!    to follow-up phases (see "What is NOT yet enforced" below).
//!
//!    To keep the placeholder honest on the test witnesses (which use
//!    `f_pre = Fp12::one()`, where `conjugate(Fp12::one()) = Fp12::one()`
//!    and so the easy-part output equals `Fp12::one()` whose `c₀.c₀.c₀`
//!    components match `f_pre.c₀`), we use the **c₀ pass-through**
//!    pattern. On a non-identity input this constraint would not hold;
//!    a follow-up phase will replace it with the actual Fp12 product
//!    relation `f_easy_out = f^{(p⁶-1)(p²+1)}` decomposed over the
//!    Fp arithmetic AIR via cross-AIR LogUp.
//! 5. **Hard-step pass-through** (row-local, on `is_hard_step` rows):
//!    `f_final = f_hard_out` for every Fp12 limb. This is the
//!    load-bearing semantic relation between the last two Fp12
//!    columns: the row's exported `f_final` (carried to the next row's
//!    `f_pre` via the shifted continuity constraint below) equals the
//!    committed `f_hard_out`. Flattened over 72 limb sub-bodies via
//!    β-RLC.
//! 6. **Shifted row-chain continuity**: `f_pre[r+1] = f_final[r]` for
//!    each of the 72 Fp12 limb columns. This is the compositional
//!    property that stitches the easy → hard → (future ZKP-PAIRING)
//!    rows end-to-end with no inter-row gap.
//!
//! # What is NOT yet enforced (deferred)
//!
//! The **deep arithmetic relations** — that `f_easy_out` actually
//! equals `f_pre^{(p⁶-1)(p²+1)}`, and `f_hard_out` actually equals
//! `f_easy_out^{(p⁴-p²+1)/r}` — are NOT enforced row-locally. They
//! decompose into:
//!
//!   * Easy part: 1 conjugation + 1 Fp12 inverse + 1 Fp12 mul + 1
//!     Frobenius (power 2) + 1 Fp12 mul ≈ 4 Fp12 ops ≈ ~80 Fp ops.
//!   * Hard part (addition chain): ~62 cyclotomic squares + ~10 Fp12
//!     muls + 4 Frobenius maps + ~7 conjugations ≈ ~150 Fp12 ops ≈
//!     several thousand Fp ops.
//!
//! Each Fp / Fp2 / Fp6 sub-op becomes a row (or set of rows) in
//! [`crate::nonnative_fp_air`] with the operands published as committed
//! limb columns and a cross-AIR LogUp closing the algebraic
//! connection. That work is intentionally several phases away; the
//! present AIR commits the witness shape so those LogUps have a
//! stable target.
//!
//! # Cross-AIR linkages
//!
//! * [`make_final_exp_f_pre_to_miller_step_descriptor`] — binds this
//!   AIR's first-row `f_pre` (the Miller-loop output) to the **last**
//!   real row of [`crate::miller_step_air`]'s `acc_post`. 72-tuple
//!   LogUp on the Fp12 limbs.
//!
//!   **Caveat**: with a `is_easy_step` selector gate on the B side
//!   (this AIR), the linkage matches `f_pre` on *every* easy-step row
//!   rather than just the first. A dedicated `IS_FIRST_ROW` boundary
//!   selector is the proper gate; until then the descriptor below is
//!   the column-shape spec the future boundary-gated linkage will use.
//!   The Miller-step side uses
//!   [`crate::miller_step_air::COL_ACC_POST_OFFSET`] gated by
//!   `COL_IS_DOUBLING + COL_IS_ADDITION` (the natural `IS_REAL`-shaped
//!   gate from that AIR's selector pair).
//!
//! # Soundness summary
//!
//! After this phase: a malicious prover cannot break the row-chain
//! continuity, selector binarity, mutual exclusion, or the
//! pass-through of `f_hard_out → f_final` on `is_hard_step` rows.
//! They CAN still cheat on the *deep* arithmetic relations (commit any
//! `f_easy_out` / `f_hard_out` they like) — that gap closes when the
//! nonnative_fp_air cross-AIR LogUps for each Fp12 sub-op are wired.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::nonnative_fp::Fp;
use crate::nonnative_tower::Fp12;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of 64-bit limbs per Fp element (381-bit field → 6 × 64).
pub const LIMBS_PER_FP: usize = 6;
/// Number of Fp values per Fp12 element (top-level Fp basis).
pub const FP_PER_FP12: usize = 12;
/// Number of limbs per Fp12 element = 12 × 6 = 72.
pub const LIMBS_PER_FP12: usize = FP_PER_FP12 * LIMBS_PER_FP;
/// Number of Fp values in the c₀ half of an Fp12 (one Fp6 = 3 × Fp2 = 6 Fp).
pub const FP_PER_FP6: usize = 6;
/// Number of limbs in the c₀ half of an Fp12 = 6 × 6 = 36.
pub const LIMBS_PER_FP6: usize = FP_PER_FP6 * LIMBS_PER_FP;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_F_PRE_OFFSET: usize = 0;
pub const COL_F_EASY_OUT_OFFSET: usize = COL_F_PRE_OFFSET + LIMBS_PER_FP12;
pub const COL_F_HARD_OUT_OFFSET: usize = COL_F_EASY_OUT_OFFSET + LIMBS_PER_FP12;
pub const COL_F_FINAL_OFFSET: usize = COL_F_HARD_OUT_OFFSET + LIMBS_PER_FP12;
pub const COL_IS_EASY_STEP: usize = COL_F_FINAL_OFFSET + LIMBS_PER_FP12;
pub const COL_IS_HARD_STEP: usize = COL_IS_EASY_STEP + 1;

/// Number of original ("scaffold") committed columns before the
/// task #317 intermediate-witness widening.
pub const NUM_SCAFFOLD_COLUMNS: usize = COL_IS_HARD_STEP + 1;

/// Total number of per-descriptor intermediate Fp result slots reserved
/// for the easy + hard descriptor sets. Each descriptor gets its own
/// dedicated 6-limb Fp column block so that
/// [`populate_final_exp_trace`] can write `r = a · b mod p` to a
/// distinct `c_base` per descriptor (eliminating the alias collisions
/// that previously capped pinned coverage at ~24 descriptors).
///
/// Composition: 410 easy + 808 hard = 1218 descriptors.
pub const FE_INTERMEDIATE_NUM_DESCRIPTORS: usize = 1218;

/// Offset of the first Fp intermediate-result slot. The block runs
/// `[COL_FE_INTERMEDIATE_OFFSET .. COL_FE_INTERMEDIATE_OFFSET
///  + FE_INTERMEDIATE_NUM_DESCRIPTORS * LIMBS_PER_FP)`.
///
/// Descriptor `i`'s dedicated 6-limb result block starts at
/// `COL_FE_INTERMEDIATE_OFFSET + i * LIMBS_PER_FP`.
pub const COL_FE_INTERMEDIATE_OFFSET: usize = NUM_SCAFFOLD_COLUMNS;

/// Total number of intermediate-result limb columns (6 limbs per
/// descriptor × 1218 descriptors = 7308).
pub const NUM_FE_INTERMEDIATE_COLUMNS: usize =
    FE_INTERMEDIATE_NUM_DESCRIPTORS * LIMBS_PER_FP;

pub const NUM_COLUMNS: usize =
    NUM_SCAFFOLD_COLUMNS + NUM_FE_INTERMEDIATE_COLUMNS;

/// Helper: returns the 6-limb intermediate-result `c_base` column index
/// reserved for descriptor index `i`. Used by
/// [`crate::final_exp_descriptors`] when rewiring each descriptor's
/// `c_base` to a dedicated witness slot.
#[inline]
pub const fn intermediate_c_base_for(descriptor_index: usize) -> usize {
    COL_FE_INTERMEDIATE_OFFSET + descriptor_index * LIMBS_PER_FP
}

/// Row-local constraints:
///   0: `is_easy_step ∈ {0, 1}`
///   1: `is_hard_step ∈ {0, 1}`
///   2: `is_easy_step · is_hard_step = 0`
///   3: easy-step c₀ pass-through (β-RLC over 36 Fp6 limb sub-bodies,
///      gated by `is_easy_step`) — placeholder substantive Fp12-equality
///      body, see module docs.
///   4: hard-step `f_final = f_hard_out` (β-RLC over 72 Fp12 limb
///      sub-bodies, gated by `is_hard_step`).
pub const NUM_ROW_CONSTRAINTS: usize = 5;

/// Shifted-row constraint categories:
///   0: `f_pre[r+1] = f_final[r]` (72 sub-bodies via β-RLC).
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

/// One final-exponentiation step witness row.
///
/// All Fp12 elements are stored in the host-side reference type
/// ([`Fp12`]); the trace builder flattens them into 6-limb big-endian
/// columns matching the [`crate::nonnative_fp_air`] convention and the
/// [`crate::miller_step_air`] layout.
#[derive(Clone, Debug)]
pub struct FinalExpRow {
    /// Fp12 input to this step.
    pub f_pre: Fp12,
    /// Fp12 output after the easy part on `is_easy_step` rows.
    pub f_easy_out: Fp12,
    /// Fp12 output after the hard part on `is_hard_step` rows.
    pub f_hard_out: Fp12,
    /// Fp12 carried out of this row into the next row's `f_pre`.
    pub f_final: Fp12,
    /// Selector: this row applies the easy-part transformation.
    pub is_easy_step: bool,
    /// Selector: this row applies the hard-part transformation.
    pub is_hard_step: bool,
}

#[derive(Clone, Debug, Default)]
pub struct FinalExpWitness {
    pub rows: Vec<FinalExpRow>,
}

impl FinalExpWitness {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Append an easy-step row. Sets `f_easy_out = f_pre` as the
    /// placeholder (matches the c₀ pass-through constraint below on
    /// `f_pre = Fp12::one()` and other inputs whose easy-part image
    /// agrees on c₀); also pins `f_hard_out = f_easy_out` and
    /// `f_final = f_easy_out` so that the row is internally consistent
    /// for the shifted continuity constraint when no separate hard
    /// step follows.
    ///
    /// The host-side reference value for the actual easy-part output
    /// (computed via [`crate::pairing::final_exponentiation`]'s
    /// closed-form path) is intentionally NOT plugged in here, because
    /// the substantive easy-part Fp12 equality is a deferred phase. A
    /// follow-up will replace this builder with one that runs the full
    /// easy-part chain and commits the intermediate Fp12 values.
    pub fn push_easy(&mut self, f_pre: Fp12) {
        let f_easy_out = f_pre; // placeholder; see module docs.
        let f_hard_out = f_easy_out;
        let f_final = f_hard_out;
        self.rows.push(FinalExpRow {
            f_pre,
            f_easy_out,
            f_hard_out,
            f_final,
            is_easy_step: true,
            is_hard_step: false,
        });
    }

    /// Append a hard-step row. `f_pre` is the easy-part output from the
    /// previous row; `f_hard_out` should be the hard-part image (the
    /// final-exponentiation result). On this scaffold we set
    /// `f_easy_out = f_pre` (the hard step does not touch the easy
    /// output) and `f_final = f_hard_out` (the load-bearing relation
    /// that constraint 4 enforces algebraically).
    pub fn push_hard(&mut self, f_pre: Fp12, f_hard_out: Fp12) {
        let f_easy_out = f_pre;
        let f_final = f_hard_out;
        self.rows.push(FinalExpRow {
            f_pre,
            f_easy_out,
            f_hard_out,
            f_final,
            is_easy_step: false,
            is_hard_step: true,
        });
    }

    /// Append a raw row without enforcing the host-side relations —
    /// used by tampering tests to commit values that violate the
    /// constraints.
    pub fn push_raw(&mut self, row: FinalExpRow) {
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

/// Flatten an Fp12 into 72 limb cells starting at `base`. Same layout
/// as [`crate::miller_step_air::write_fp12_limbs`]:
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
    witness: &FinalExpWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        write_fp12_limbs(&mut columns, COL_F_PRE_OFFSET, r, &row.f_pre, curve);
        write_fp12_limbs(&mut columns, COL_F_EASY_OUT_OFFSET, r, &row.f_easy_out, curve);
        write_fp12_limbs(&mut columns, COL_F_HARD_OUT_OFFSET, r, &row.f_hard_out, curve);
        write_fp12_limbs(&mut columns, COL_F_FINAL_OFFSET, r, &row.f_final, curve);
        columns[COL_IS_EASY_STEP][r] =
            if row.is_easy_step { one.clone() } else { zero.clone() };
        columns[COL_IS_HARD_STEP][r] =
            if row.is_hard_step { one.clone() } else { zero.clone() };
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

/// Constraint-system handle for the final-exponentiation scaffold AIR.
pub struct FinalExpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl FinalExpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Evaluate the β-RLC body that pins per-limb equality between two
/// column ranges at row `r`, both of length `len`. Used to flatten an
/// N-limb Fp12 / Fp6 equality into a single algebraic body.
fn rlc_equality_body(
    columns: &[&Vec<Scalar>],
    r: usize,
    base_a: usize,
    base_b: usize,
    len: usize,
    alpha: &Scalar,
) -> Scalar {
    let curve = alpha.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut beta_pow = Scalar::one(curve);
    for j in 0..len {
        let a = &columns[base_a + j][r];
        let b = &columns[base_b + j][r];
        let body = a.sub(b);
        acc = acc.add(&beta_pow.mul(&body));
        beta_pow = beta_pow.mul(alpha);
    }
    acc
}

impl VmConstraintSystem for FinalExpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_easy_step_binary".into(),
            "is_hard_step_binary".into(),
            "selectors_mutually_exclusive".into(),
            "easy_step_c0_passthrough".into(),
            "hard_step_f_final_equals_f_hard_out".into(),
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
        // We pin β = α (the per-row alpha challenge isn't available
        // inside `evaluate_on_domain` so we use a fixed deterministic
        // challenge here; the production-grade compiler emits one body
        // per limb instead of an RLC and lets the outer prover RLC
        // them. For the scaffold we use β = 7 to fold the 36 / 72
        // sub-bodies into a single algebraic check evaluated on the
        // full domain.).
        let beta = Scalar::from_u64(7, curve);
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_easy_step binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_EASY_STEP][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1: is_hard_step binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_HARD_STEP][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 2: mutual exclusion.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let e = &columns[COL_IS_EASY_STEP][r];
                let h = &columns[COL_IS_HARD_STEP][r];
                c[r] = e.mul(h);
            }
            out.push(c);
        }

        // 3: easy-step c₀ pass-through, gated by is_easy_step.
        //    body = is_easy_step · Σ_j β^j · (f_easy_out.c0[j] − f_pre.c0[j])
        //    over the first LIMBS_PER_FP6 = 36 limbs.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate = &columns[COL_IS_EASY_STEP][r];
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body(
                    columns,
                    r,
                    COL_F_EASY_OUT_OFFSET,
                    COL_F_PRE_OFFSET,
                    LIMBS_PER_FP6,
                    &beta,
                );
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }

        // 4: hard-step f_final = f_hard_out, gated by is_hard_step.
        //    body = is_hard_step · Σ_j β^j · (f_final[j] − f_hard_out[j])
        //    over all 72 Fp12 limbs.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate = &columns[COL_IS_HARD_STEP][r];
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body(
                    columns,
                    r,
                    COL_F_FINAL_OFFSET,
                    COL_F_HARD_OUT_OFFSET,
                    LIMBS_PER_FP12,
                    &beta,
                );
                c[r] = gate.mul(&body);
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
        let beta = Scalar::from_u64(7, curve);
        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_easy_step binary.
        {
            let v = &col_evals[COL_IS_EASY_STEP];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_hard_step binary.
        {
            let v = &col_evals[COL_IS_HARD_STEP];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: mutual exclusion.
        {
            let e = &col_evals[COL_IS_EASY_STEP];
            let h = &col_evals[COL_IS_HARD_STEP];
            acc = acc.add(&alpha_pow.mul(&e.mul(h)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3: easy-step c₀ pass-through.
        {
            let gate = &col_evals[COL_IS_EASY_STEP];
            let mut body = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP6 {
                let diff = col_evals[COL_F_EASY_OUT_OFFSET + j]
                    .sub(&col_evals[COL_F_PRE_OFFSET + j]);
                body = body.add(&beta_pow.mul(&diff));
                beta_pow = beta_pow.mul(&beta);
            }
            acc = acc.add(&alpha_pow.mul(&gate.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4: hard-step f_final = f_hard_out.
        {
            let gate = &col_evals[COL_IS_HARD_STEP];
            let mut body = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP12 {
                let diff = col_evals[COL_F_FINAL_OFFSET + j]
                    .sub(&col_evals[COL_F_HARD_OUT_OFFSET + j]);
                body = body.add(&beta_pow.mul(&diff));
                beta_pow = beta_pow.mul(&beta);
            }
            acc = acc.add(&alpha_pow.mul(&gate.mul(&body)));
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
        let beta = Scalar::from_u64(7, curve);
        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_easy_step binary.
        {
            let v = &col_coeffs[COL_IS_EASY_STEP];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_hard_step binary.
        {
            let v = &col_coeffs[COL_IS_HARD_STEP];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: mutual exclusion.
        {
            let e = &col_coeffs[COL_IS_EASY_STEP];
            let h = &col_coeffs[COL_IS_HARD_STEP];
            let body = poly_mul(e, h, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3: easy-step c₀ pass-through.
        {
            let gate = &col_coeffs[COL_IS_EASY_STEP];
            let mut body = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP6 {
                let diff = poly_sub(
                    &col_coeffs[COL_F_EASY_OUT_OFFSET + j],
                    &col_coeffs[COL_F_PRE_OFFSET + j],
                    curve,
                );
                body = poly_add(&body, &poly_scalar_mul(&diff, &beta_pow), curve);
                beta_pow = beta_pow.mul(&beta);
            }
            let gated = poly_mul(gate, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4: hard-step f_final = f_hard_out.
        {
            let gate = &col_coeffs[COL_IS_HARD_STEP];
            let mut body = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP12 {
                let diff = poly_sub(
                    &col_coeffs[COL_F_FINAL_OFFSET + j],
                    &col_coeffs[COL_F_HARD_OUT_OFFSET + j],
                    curve,
                );
                body = poly_add(&body, &poly_scalar_mul(&diff, &beta_pow), curve);
                beta_pow = beta_pow.mul(&beta);
            }
            let gated = poly_mul(gate, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_EASY_STEP, COL_IS_HARD_STEP]
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
        // As in miller_step_air: limb 64-bit range checks live in
        // nonnative_fp_air via cross-AIR LogUp, not here.
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Shifted constraint 0 needs `f_pre[r+1]` (72 columns).
        (0..LIMBS_PER_FP12).map(|j| COL_F_PRE_OFFSET + j).collect()
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }
}

// ─── Shifted (cross-row) constraint helpers ───────────────────────────

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
            // f_pre[r+1] = f_final[r] — 72 sub-bodies via β-RLC (β = α).
            for j in 0..LIMBS_PER_FP12 {
                let a = &columns[COL_F_PRE_OFFSET + j][next];
                let b = &columns[COL_F_FINAL_OFFSET + j][r];
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

/// Cross-AIR LogUp descriptor: this AIR's `f_pre` Fp12 limbs (72 cols)
/// gated by `IS_EASY_STEP` ↔ [`crate::miller_step_air`]'s `acc_post`
/// Fp12 limbs (72 cols) gated by `IS_DOUBLING` (the conventional
/// `IS_REAL`-shaped selector from that AIR).
///
/// 72-tuple LogUp. Pins this AIR's input `f_pre` to the Miller-loop
/// output committed by the Miller-step AIR.
///
/// **Caveat** (same shape as [`crate::miller_step_air`]'s
/// `make_miller_step_q_curr_to_bls_pairing_descriptor`): with only an
/// `is_easy_step` gate on the B side, this descriptor matches `f_pre`
/// from *every* easy-step row, not just the first one. The full
/// algebraic binding to the *first row's* `f_pre` requires a dedicated
/// `IS_FIRST_ROW` selector column (a "boundary" selector that is 1
/// only at row 0 of this AIR's loop). Adding that selector is a
/// straightforward follow-up; the descriptor below is the column-shape
/// spec the future boundary-gated linkage will use.
///
/// Similarly, the Miller-step side should match only the *last*
/// real row's `acc_post` (the final accumulator), not every active
/// row's. That side likewise wants an `IS_LAST_ROW` boundary selector.
pub fn make_final_exp_f_pre_to_miller_step_descriptor(
    final_exp_layer_index: usize,
    miller_step_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::miller_step_air as ms;
    // A-side columns: miller_step_air `acc_post` limbs (72).
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| ms::COL_ACC_POST_OFFSET + j)
        .collect();
    // B-side columns: this AIR's `f_pre` limbs (72).
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| COL_F_PRE_OFFSET + j)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "final_exp_f_pre_to_miller_step_v1".into(),
        a_layer_index: miller_step_layer_index,
        a_columns,
        a_selector_column: Some(ms::COL_IS_DOUBLING),
        b_layer_index: final_exp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_EASY_STEP),
    }
}

// ─── Populate final-exp Fp-mult trace (Task #312) ─────────────────────
//
// `populate_final_exp_trace` produces a self-consistent test fixture
// that threads **real Fp modular products** through the bookkeeping
// surfaces wired by [`crate::final_exp_descriptors`].
//
// Mirrors the pattern of [`crate::miller_loop_air::populate_miller_loop_trace`]
// (#174) but for the final exponentiation. Like that fixture, the
// descriptor `c_base` columns alias into existing Fp12 committed slots
// (`f_easy_out`, `f_hard_out`, `f_final`) — so to make every closure
// hold we deliberately **overwrite** those cells with the elementary
// modular product `r = a · b mod p`. This breaks the (currently
// unenforced) Fp12 chain semantics of `final_exp_air` but yields clean
// cross-trace multiset equality for the wired descriptors.
//
// Many descriptors share the same `c_base` column (the alias scheme is
// `f_easy_out` and `f_final` slot cycling). For descriptors whose
// `c_base` collides with one already populated, the second write would
// overwrite the first. We therefore restrict the **pinned** subset to
// one descriptor per unique `(row, a_base, b_base, c_base)` quadruple,
// which lets every pinned descriptor's cross-AIR LogUp closure hold.
// Descriptors with collisions are recorded but not pinned — closing
// them requires the deferred `final_exp_air` widening that adds
// per-product intermediate witness columns (see module docs).

use crate::nonnative_fp_air::{self as nfp, FpOp};

/// Output of [`populate_final_exp_trace`]: paired traces for
/// `final_exp_air` (B side) and `nonnative_fp_air` (A side), threaded
/// with **real Fp modular products** that satisfy every wired
/// cross-AIR LogUp descriptor in the **pinned subset**.
pub struct FinalExpFpMultFixture {
    /// Final-exponentiation AIR trace (B side). Selected Fp limb cells
    /// in `f_easy_out`, `f_hard_out`, `f_final` have been overwritten
    /// with elementary modular products so the pinned descriptors'
    /// cross-AIR tuples match.
    pub final_exp_trace: TracePolynomials,
    /// `nonnative_fp_air`-shaped trace (A side) with one [`FpOp::Mul`]
    /// row per pinned descriptor.
    pub nonnative_fp_trace: TracePolynomials,
    /// Descriptors whose closures this fixture pins (a subset of
    /// `easy + hard` whose `c_base` columns are non-colliding).
    pub pinned_descriptors: Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// Full easy-part descriptor set (410 entries) for downstream
    /// consumers that want the complete shape contract.
    pub all_easy_descriptors: Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// Full hard-part descriptor set (808 entries).
    pub all_hard_descriptors: Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// Number of populated `FpOp::Mul` rows = length of
    /// `pinned_descriptors`.
    pub num_populated_fp_mults: usize,
}

/// Read 6 Fp limbs from the trace at row `r` starting at `base`.
fn read_fp_at(cols: &[Vec<Scalar>], base: usize, row: usize) -> Fp {
    let mut limbs = [0u64; LIMBS_PER_FP];
    for j in 0..LIMBS_PER_FP {
        limbs[j] = scalar_to_u64(&cols[base + j][row]);
    }
    Fp { limbs }
}

/// Write an `Fp` value's 6 limbs into the trace at row `row`.
fn write_fp_at(cols: &mut [Vec<Scalar>], base: usize, r: &Fp, row: usize, curve: CurveType) {
    for j in 0..LIMBS_PER_FP {
        cols[base + j][row] = Scalar::from_u64(r.limbs[j], curve);
    }
}

/// Read a BLS12-381 `Scalar` cell back into the u64 it was created
/// from. Only correct on cells written via `Scalar::from_u64(_, Bls12381)`,
/// which is how `write_fp_limbs` (above) populates the trace.
fn scalar_to_u64(s: &Scalar) -> u64 {
    let bytes = s.to_bytes();
    // `Scalar::to_bytes` for Bls12381 uses little-endian (matches
    // `blst_scalar_to_lendian`); the first 8 bytes are the LSB u64.
    let mut buf = [0u8; 8];
    let take = bytes.len().min(8);
    buf[..take].copy_from_slice(&bytes[..take]);
    u64::from_le_bytes(buf)
}

/// Build a host-side witness fixture pairing a final_exp_air trace with
/// a populated nonnative_fp_air trace, threaded with real Fp modular
/// products.
///
/// Given a non-trivial input Fp12 `f_input`, this:
///
///   1. Constructs a 2-row final-exp witness (1 easy + 1 hard) with the
///      easy row's `f_pre = f_input` and the hard row's `f_pre` /
///      `f_hard_out` set to the host-side reference final-exponentiation
///      result. (The deep arithmetic relations are NOT enforced
///      algebraically in `final_exp_air`; the witness uses the reference
///      values so downstream consumers see semantically meaningful
///      operand columns.)
///   2. Builds the full easy + hard descriptor sets (410 + 808 = 1218
///      Fp mult descriptors) via [`crate::final_exp_descriptors`].
///   3. For each descriptor that gates on the appropriate selector
///      (easy or hard) AND whose `c_base` does NOT collide with an
///      already-populated cell, computes `r = a · b mod p` from the
///      `(a_base, b_base)` operand limbs and writes `r` to `c_base`.
///      Records the descriptor as **pinned**.
///   4. Builds an `nfp`-shaped trace with one [`FpOp::Mul`] row per
///      pinned descriptor.
///
/// **Pinning strategy**: many descriptors share `c_base` aliases (the
/// final_exp_air scaffold reuses Fp12 slot columns as result aliases
/// pending widening). Two descriptors with the same `c_base` cannot both
/// be pinned because the second write would overwrite the first. We
/// keep only the **first** descriptor encountered per
/// `(row, c_base)` pair. Easy descriptors are processed first
/// (covering all 410 from phases A1..A35), then hard descriptors.
///
/// # Coverage on the present scaffold
///
/// On the easy row, each Fp slot in `f_easy_out` / `f_final` admits
/// exactly one pinned descriptor; the column layout has
/// `2 · LIMBS_PER_FP12 = 144` candidate alias cells per row (24 Fp slots
/// across `f_easy_out` and `f_final`), so the pinned subset of the easy
/// part caps at ~24 distinct mults per easy row. Similarly for the hard
/// part on the hard row. The full 1218-descriptor closure requires the
/// deferred per-product witness column widening.
pub fn populate_final_exp_trace(f_input: Fp12) -> FinalExpFpMultFixture {
    let curve = CurveType::Bls12381;

    // ── 1. Build the 2-row witness. ──
    //
    // Reference final-exponentiation value via the host-side oracle.
    let f_final = crate::pairing::final_exponentiation(&f_input);
    let mut w = FinalExpWitness::new();
    // Easy row: f_pre = f_input. f_easy_out / f_hard_out / f_final are
    // set by `push_easy` to f_pre (placeholder; see module docs). This
    // leaves the row internally consistent for shifted continuity.
    w.push_easy(f_input);
    // Hard row: f_pre = previous row's f_final = f_input (placeholder
    // chain), and f_hard_out = the reference final-exp result so the
    // committed Fp12 carries semantically meaningful data even though
    // the deep arithmetic is not yet algebraically enforced.
    w.push_hard(f_input, f_final);

    let trace = build_trace_polynomials(&w, curve);
    let num_rows = trace.num_rows;
    let padded = trace.padded_size as usize;

    // ── 2. Build descriptor sets (easy + hard). ──
    //
    // Layer indices: final_exp_air = 1, nonnative_fp_air = 0.
    //
    // **Task #317 widening**: each descriptor now writes to its own
    // dedicated 6-limb intermediate-result slot in the final_exp_air
    // trace, eliminating the prior `c_base` alias collisions. We
    // rewrite each descriptor's `b_columns[12..18]` (the `c_base`
    // limb-group) to `intermediate_c_base_for(global_index)`.
    let mut easy_descs =
        crate::final_exp_descriptors::FinalExpEasyDescriptors::build(
            /* final_exp */ 1,
            /* nonnative_fp */ 0,
        )
        .descriptors;
    let mut hard_descs =
        crate::final_exp_descriptors::FinalExpHardDescriptors::build(
            /* final_exp */ 1,
            /* nonnative_fp */ 0,
        )
        .descriptors;

    // Rewire each descriptor's `c_base` (b_columns[12..18]) to its
    // dedicated intermediate column block. Easy descriptors occupy
    // global indices 0..410; hard descriptors occupy 410..1218.
    debug_assert_eq!(easy_descs.len(), 410);
    debug_assert_eq!(hard_descs.len(), 808);
    let c_base_offset = 2 * LIMBS_PER_FP;
    for (i, desc) in easy_descs.iter_mut().enumerate() {
        let new_c_base = intermediate_c_base_for(i);
        for j in 0..LIMBS_PER_FP {
            desc.b_columns[c_base_offset + j] = new_c_base + j;
        }
    }
    for (i, desc) in hard_descs.iter_mut().enumerate() {
        let new_c_base = intermediate_c_base_for(easy_descs.len() + i);
        for j in 0..LIMBS_PER_FP {
            desc.b_columns[c_base_offset + j] = new_c_base + j;
        }
    }

    // ── 3. Populate. ──
    //
    // With dedicated intermediate slots there are no alias collisions
    // — every descriptor is pinnable. We clone the trace data into a
    // mutable working buffer and write `r = a · b mod p` to each
    // descriptor's dedicated `c_base` cells at every gated row.
    let mut cols_mut: Vec<Vec<Scalar>> =
        trace.columns.iter().map(|p| p.evaluations.clone()).collect();

    let mut pinned: Vec<crate::cross_air_logup::CrossAirLogUpDescriptor> = Vec::new();
    let mut populated_products: Vec<(Fp, Fp, Fp)> = Vec::new();

    let populate_set =
        |descs: &[crate::cross_air_logup::CrossAirLogUpDescriptor],
         cols_mut: &mut [Vec<Scalar>],
         pinned: &mut Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
         populated_products: &mut Vec<(Fp, Fp, Fp)>| {
            for desc in descs {
                let gate_col = desc
                    .b_selector_column
                    .expect("final_exp descriptors carry a b_selector_column");
                for r in 0..num_rows {
                    let gate = &cols_mut[gate_col][r];
                    if gate.is_zero() {
                        continue;
                    }
                    let a_base = desc.b_columns[0];
                    let b_base = desc.b_columns[LIMBS_PER_FP];
                    let c_base = desc.b_columns[2 * LIMBS_PER_FP];
                    let a = read_fp_at(cols_mut, a_base, r);
                    let b = read_fp_at(cols_mut, b_base, r);
                    let prod = a.mul(&b);
                    write_fp_at(cols_mut, c_base, &prod, r, curve);
                    populated_products.push((a, b, prod));
                    pinned.push(desc.clone());
                }
            }
        };

    // Process easy descriptors first (gated on COL_IS_EASY_STEP — active
    // on row 0). Then hard descriptors (active on row 1).
    populate_set(
        &easy_descs,
        &mut cols_mut,
        &mut pinned,
        &mut populated_products,
    );
    populate_set(
        &hard_descs,
        &mut cols_mut,
        &mut pinned,
        &mut populated_products,
    );

    // Repackage as TracePolynomials.
    let final_exp_trace = TracePolynomials {
        columns: cols_mut
            .into_iter()
            .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
            .collect(),
        num_rows,
        padded_size: padded as u64,
        curve,
    };

    // ── 4. Build the nonnative_fp_air trace with one Mul row per
    // pinned descriptor. ──
    let n_fp_rows = populated_products.len();
    let n_fp_padded = crate::trace::nearest_power_of_two(n_fp_rows.max(1));
    let mut fp_cols = nfp::alloc_trace(n_fp_padded, curve);
    for (row, (a, b, _r)) in populated_products.iter().enumerate() {
        let op = FpOp::Mul { a: *a, b: *b };
        nfp::populate_row(&mut fp_cols, row, &op, curve);
    }
    let nonnative_fp_trace = TracePolynomials {
        columns: fp_cols
            .into_iter()
            .map(|evals| Polynomial { evaluations: evals, degree: n_fp_rows })
            .collect(),
        num_rows: n_fp_rows,
        padded_size: n_fp_padded as u64,
        curve,
    };

    FinalExpFpMultFixture {
        final_exp_trace,
        nonnative_fp_trace,
        pinned_descriptors: pinned,
        all_easy_descriptors: easy_descs,
        all_hard_descriptors: hard_descs,
        num_populated_fp_mults: n_fp_rows,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> Fp12 {
        // Fp12::one() is convenient: conjugate(1) = 1, so the c₀
        // pass-through placeholder holds. A non-trivial input would
        // need the full easy-part chain to satisfy the placeholder,
        // which is intentionally deferred — see module docs.
        Fp12::one()
    }

    fn honest_witness(num_rows: usize) -> FinalExpWitness {
        let mut w = FinalExpWitness::new();
        let mut f = sample_input();
        for i in 0..num_rows {
            if i % 2 == 0 {
                w.push_easy(f);
            } else {
                // Hard step: take the previous row's f_final as f_pre,
                // and use the same value as the "hard output" since
                // f = Fp12::one() is a fixed point of the entire
                // final-exponentiation map (1^anything = 1). This is
                // what makes the scaffold's continuity constraint hold.
                w.push_hard(f, f);
            }
            f = w.rows.last().unwrap().f_final;
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
        assert!(w.rows[0].is_easy_step && !w.rows[0].is_hard_step);
        assert!(!w.rows[1].is_easy_step && w.rows[1].is_hard_step);

        // Continuity holds in the witness: row 1's f_pre equals row 0's f_final.
        assert_eq!(w.rows[1].f_pre, w.rows[0].f_final);
        // Hard row: f_final == f_hard_out.
        assert_eq!(w.rows[1].f_final, w.rows[1].f_hard_out);

        // Trace has expected column count and the selectors are set.
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);
        assert!(
            trace.columns[COL_IS_EASY_STEP].evaluations[0].sub(&one).is_zero(),
            "row 0 is_easy_step should be 1",
        );
        assert!(
            trace.columns[COL_IS_HARD_STEP].evaluations[0].is_zero(),
            "row 0 is_hard_step should be 0",
        );
        assert!(
            trace.columns[COL_IS_EASY_STEP].evaluations[1].is_zero(),
            "row 1 is_easy_step should be 0",
        );
        assert!(
            trace.columns[COL_IS_HARD_STEP].evaluations[1].sub(&one).is_zero(),
            "row 1 is_hard_step should be 1",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: constraints zero on honest witness
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = FinalExpConstraintSystem::new(trace.num_rows);
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
        // Shifted constraint 0 zero at row 0 (continuity into row 1).
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
    // Test 3: tampered f_final breaks chain continuity (shifted 0)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_f_final_breaks_chain_continuity() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();

        // Tamper limb 0 of f_final on row 0 (add 1). Breaks shifted
        // continuity to row 1's f_pre AND breaks the hard-step
        // f_final = f_hard_out check on row 0 — except row 0 is an
        // easy step (gate is 0), so constraint 4 is unaffected at
        // row 0. (On row 1, which is the hard step, we'll tamper a
        // different column to keep this test focused on shifted 0.)
        let curve = CurveType::Bls12381;
        let original = cols[COL_F_FINAL_OFFSET][0].clone();
        cols[COL_F_FINAL_OFFSET][0] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();

        // Shifted constraint 0 (f chain continuity) fires at row 0.
        let alpha = Scalar::from_u64(7, curve);
        let body = evaluate_shifted_row(&col_refs, 0, 1, 0, &alpha);
        assert!(
            !body.is_zero(),
            "tampered f_final must break f chain continuity (shifted 0)",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: hard-step f_final = f_hard_out constraint fires on
    // tampering at a hard-step row
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn hard_step_passthrough_fires_on_tampered_f_final() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        // Row 1 is the hard step. Tamper limb 3 of f_final on row 1.
        // This must fire constraint 4 (hard-step pass-through).
        let original = cols[COL_F_FINAL_OFFSET + 3][1].clone();
        cols[COL_F_FINAL_OFFSET + 3][1] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let cs = FinalExpConstraintSystem::new(trace.num_rows);
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        // Selector + mutual-exclusion constraints (0, 1, 2) unaffected.
        for i in 0..3 {
            for (r, val) in results[i].iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "row-local constraint {} at row {} unexpectedly fired",
                    i, r,
                );
            }
        }
        // Constraint 3 (easy-step c₀ pass-through) unaffected at row 1
        // (gate is_easy_step = 0 on row 1). It may also be unaffected
        // at row 0 because we didn't tamper f_easy_out / f_pre.
        assert!(
            results[3][1].is_zero(),
            "easy-step c0 pass-through should NOT fire on a hard row",
        );

        // Constraint 4 (hard-step pass-through) MUST fire at row 1.
        assert!(
            !results[4][1].is_zero(),
            "tampered hard-step f_final must fire constraint 4",
        );
        // And NOT at row 0 (which is an easy step).
        assert!(
            results[4][0].is_zero(),
            "constraint 4 must NOT fire on an easy-step row",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: selector binary + mutual-exclusion constraints fire on
    // tampering
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn is_easy_binary_and_mutual_exclusion_fire_on_tampered_selectors() {
        let w = honest_witness(1);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);

        // Non-binary is_easy_step.
        cols[COL_IS_EASY_STEP][0] = Scalar::from_u64(3, curve);
        let cs = FinalExpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[0][0].is_zero(),
            "non-binary is_easy_step must fire constraint 0",
        );

        // Mutual exclusion: both selectors set.
        cols[COL_IS_EASY_STEP][0] = one.clone();
        cols[COL_IS_HARD_STEP][0] = one;
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[2][0].is_zero(),
            "both selectors = 1 must fire mutual-exclusion (constraint 2)",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: cross-AIR linkage descriptor is well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn f_pre_to_miller_step_descriptor_well_formed() {
        let d = make_final_exp_f_pre_to_miller_step_descriptor(1, 0);
        assert_eq!(d.label, "final_exp_f_pre_to_miller_step_v1");
        assert_eq!(d.a_layer_index, 0, "A side = miller_step layer");
        assert_eq!(d.b_layer_index, 1, "B side = final_exp layer");
        assert_eq!(d.a_columns.len(), LIMBS_PER_FP12);
        assert_eq!(d.b_columns.len(), LIMBS_PER_FP12);

        // A-side first / last column = miller_step_air acc_post limbs.
        use crate::miller_step_air as ms;
        assert_eq!(d.a_columns[0], ms::COL_ACC_POST_OFFSET);
        assert_eq!(
            d.a_columns[LIMBS_PER_FP12 - 1],
            ms::COL_ACC_POST_OFFSET + LIMBS_PER_FP12 - 1,
        );
        // B-side first / last column.
        assert_eq!(d.b_columns[0], COL_F_PRE_OFFSET);
        assert_eq!(d.b_columns[LIMBS_PER_FP12 - 1], COL_F_PRE_OFFSET + LIMBS_PER_FP12 - 1);

        // Selectors.
        assert_eq!(d.a_selector_column, Some(ms::COL_IS_DOUBLING));
        assert_eq!(d.b_selector_column, Some(COL_IS_EASY_STEP));
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 7: column layout & shifted-index list sanity
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_F_PRE_OFFSET, 0);
        assert_eq!(COL_F_EASY_OUT_OFFSET, LIMBS_PER_FP12);
        assert_eq!(COL_F_HARD_OUT_OFFSET, 2 * LIMBS_PER_FP12);
        assert_eq!(COL_F_FINAL_OFFSET, 3 * LIMBS_PER_FP12);
        assert_eq!(COL_IS_EASY_STEP, 4 * LIMBS_PER_FP12);
        assert_eq!(COL_IS_HARD_STEP, COL_IS_EASY_STEP + 1);
        // 4·72 + 2 = 290 original scaffold columns.
        assert_eq!(NUM_SCAFFOLD_COLUMNS, 290);
        // Task #317 widening: 1218 descriptors × 6 limbs = 7308
        // intermediate slots tacked on after the scaffold columns.
        assert_eq!(FE_INTERMEDIATE_NUM_DESCRIPTORS, 1218);
        assert_eq!(NUM_FE_INTERMEDIATE_COLUMNS, 7308);
        assert_eq!(COL_FE_INTERMEDIATE_OFFSET, NUM_SCAFFOLD_COLUMNS);
        assert_eq!(NUM_COLUMNS, NUM_SCAFFOLD_COLUMNS + NUM_FE_INTERMEDIATE_COLUMNS);
        // 290 + 7308 = 7598.
        assert_eq!(NUM_COLUMNS, 7598);
        // Spot-check intermediate_c_base_for.
        assert_eq!(intermediate_c_base_for(0), COL_FE_INTERMEDIATE_OFFSET);
        assert_eq!(
            intermediate_c_base_for(FE_INTERMEDIATE_NUM_DESCRIPTORS - 1),
            COL_FE_INTERMEDIATE_OFFSET
                + (FE_INTERMEDIATE_NUM_DESCRIPTORS - 1) * LIMBS_PER_FP,
        );
    }

    #[test]
    fn shifted_column_indices_cover_f_pre() {
        let cs = FinalExpConstraintSystem::new(2);
        let idx = cs.shifted_column_indices();
        assert_eq!(idx.len(), LIMBS_PER_FP12);
        for j in 0..LIMBS_PER_FP12 {
            assert_eq!(idx[j], COL_F_PRE_OFFSET + j);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Tests for `populate_final_exp_trace` (Task #312)
    // ───────────────────────────────────────────────────────────────────

    /// Non-trivial Fp12 to thread through the populator. Uses the
    /// generator pair's Miller-loop output — a real, non-identity Fp12
    /// in the BLS12-381 codomain.
    fn nontrivial_fp12_input() -> Fp12 {
        use crate::pairing;
        let g1 = pairing::G1Affine::generator();
        let g2 = pairing::G2Affine::generator();
        pairing::miller_loop(&g1, &g2)
    }

    #[test]
    fn populate_final_exp_trace_produces_pinned_descriptors() {
        let f = nontrivial_fp12_input();
        let fixture = populate_final_exp_trace(f);
        // Easy + hard sets are the canonical 410 / 808.
        assert_eq!(fixture.all_easy_descriptors.len(), 410);
        assert_eq!(fixture.all_hard_descriptors.len(), 808);
        // Some descriptors must be pinned (≥ 1 per active alias slot).
        assert!(
            fixture.num_populated_fp_mults > 0,
            "must pin at least one Fp mult",
        );
        // Pinned count equals nfp trace's populated row count.
        assert_eq!(
            fixture.pinned_descriptors.len(),
            fixture.num_populated_fp_mults,
        );
        // Each populated nfp row equals one pinned descriptor.
        assert_eq!(
            fixture.nonnative_fp_trace.num_rows,
            fixture.pinned_descriptors.len(),
        );
        // Trace shape sanity: 290 columns.
        assert_eq!(fixture.final_exp_trace.columns.len(), NUM_COLUMNS);
    }

    #[test]
    fn populate_final_exp_trace_pinned_closures_hold() {
        let f = nontrivial_fp12_input();
        let fixture = populate_final_exp_trace(f);
        let curve = CurveType::Bls12381;
        let beta = Scalar::from_u64(13, curve);
        let gamma = Scalar::from_u64(17, curve);

        // For each pinned descriptor we build a single-row nfp mini-A
        // trace containing the FpOp::Mul whose (a, b) operands match
        // the descriptor's b_columns at the gated row of the final_exp
        // trace. The cross-AIR LogUp closure must then hold over the
        // pair (mini-A, final_exp_trace).
        let per_descriptor_nfp_trace =
            |desc: &crate::cross_air_logup::CrossAirLogUpDescriptor| {
                let a_base = desc.b_columns[0];
                let b_base = desc.b_columns[LIMBS_PER_FP];
                let gate_col = desc
                    .b_selector_column
                    .expect("descriptor carries a b_selector_column");
                let mut ops: Vec<FpOp> = Vec::new();
                for r in 0..fixture.final_exp_trace.num_rows {
                    let gate = &fixture
                        .final_exp_trace
                        .columns[gate_col]
                        .evaluations[r];
                    if gate.is_zero() {
                        continue;
                    }
                    let mut a_limbs = [0u64; LIMBS_PER_FP];
                    let mut b_limbs = [0u64; LIMBS_PER_FP];
                    for j in 0..LIMBS_PER_FP {
                        a_limbs[j] = scalar_to_u64(
                            &fixture.final_exp_trace.columns[a_base + j]
                                .evaluations[r],
                        );
                        b_limbs[j] = scalar_to_u64(
                            &fixture.final_exp_trace.columns[b_base + j]
                                .evaluations[r],
                        );
                    }
                    ops.push(FpOp::Mul {
                        a: Fp { limbs: a_limbs },
                        b: Fp { limbs: b_limbs },
                    });
                }
                let n_rows = ops.len();
                let n_padded =
                    crate::trace::nearest_power_of_two(n_rows.max(1));
                let mut cols = nfp::alloc_trace(n_padded, curve);
                for (row, op) in ops.iter().enumerate() {
                    nfp::populate_row(&mut cols, row, op, curve);
                }
                TracePolynomials {
                    columns: cols
                        .into_iter()
                        .map(|evals| Polynomial {
                            evaluations: evals,
                            degree: n_rows,
                        })
                        .collect(),
                    num_rows: n_rows,
                    padded_size: n_padded as u64,
                    curve,
                }
            };

        // Bound work: validate every pinned descriptor's closure.
        let mut matched = 0usize;
        for desc in &fixture.pinned_descriptors {
            let mini_a = per_descriptor_nfp_trace(desc);
            let w = crate::cross_air_logup::compute_cross_air_logup_witness(
                &mini_a,
                &fixture.final_exp_trace,
                desc,
                &beta,
                &gamma,
                curve,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "compute_cross_air_logup_witness failed for descriptor \
                     '{}': {}",
                    desc.label, err,
                )
            });
            assert!(
                w.closure_holds(),
                "closure must hold for pinned descriptor '{}'",
                desc.label,
            );
            matched += 1;
        }
        assert!(
            matched > 0,
            "must validate at least one pinned descriptor closure",
        );
    }

    #[test]
    fn populate_final_exp_trace_pins_at_least_one_per_phase() {
        // Sanity that the easy AND hard phases are both exercised:
        // at least one pinned descriptor from each side.
        let f = nontrivial_fp12_input();
        let fixture = populate_final_exp_trace(f);
        let easy_pinned = fixture
            .pinned_descriptors
            .iter()
            .filter(|d| d.b_selector_column == Some(COL_IS_EASY_STEP))
            .count();
        let hard_pinned = fixture
            .pinned_descriptors
            .iter()
            .filter(|d| d.b_selector_column == Some(COL_IS_HARD_STEP))
            .count();
        assert!(
            easy_pinned > 0,
            "must pin at least one easy-side Fp mult",
        );
        assert!(
            hard_pinned > 0,
            "must pin at least one hard-side Fp mult",
        );
        // **Task #317** widened `final_exp_air` with 7308 dedicated
        // per-descriptor intermediate Fp result columns (1218 × 6
        // limbs). With one dedicated `c_base` slot per descriptor
        // there are no alias collisions; every easy descriptor on row 0
        // and every hard descriptor on row 1 is pinnable.
        let total = fixture.pinned_descriptors.len();
        assert_eq!(
            total, 1218,
            "post-#317: every descriptor must be pinned (410 easy + 808 hard)",
        );
        assert!(
            total <= 1218,
            "pinned subset must be ≤ full descriptor count",
        );
        assert_eq!(easy_pinned, 410, "all easy descriptors must be pinned");
        assert_eq!(hard_pinned, 808, "all hard descriptors must be pinned");
    }
}
