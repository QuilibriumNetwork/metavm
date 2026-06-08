//! BN254 **final-exponentiation** scaffold AIR.
//!
//! # Purpose
//!
//! Companion to [`crate::final_exp_air`] (the BLS12-381 final-exp
//! scaffold), specialised for BN254. Lays down column layout, witness
//! shape, trace builder, and cross-AIR linkage descriptors targeting
//! [`crate::bn254_pairing_internals_air`]'s Miller-loop output
//! (`acc_post`).
//!
//! Final exponentiation for BN254 raises the Miller-loop output
//! `f ∈ Fp12` to `(p¹² − 1)/r`.
//!
//! # Decomposition
//!
//!   * **Easy part**: `(p⁶ − 1)(p² + 1)`. Closed-form
//!     `conjugate(f) · f⁻¹` (= `f^{p⁶ − 1}`) followed by
//!     `frobenius_map(·, 2) · ·` (= `·^{p² + 1}`). Output lives in the
//!     cyclotomic subgroup `G_φ12(Fp)`.
//!   * **Hard part**: `(p⁴ − p² + 1)/r`. Computed via the
//!     Fuentes-Castañeda / Devegili–Scott addition chain using the
//!     BN254 curve parameter `z = 4965661367192848881`, exploiting
//!     `cyclotomic_square`, `mul`, `conjugate`, and `frobenius_map`.
//!
//! See module docs in [`crate::final_exp_air`] for the broader
//! soundness story; the BN254 variant follows the same shape but with
//! 4-limb Fp (BN254 is a 254-bit field) so each Fp12 occupies
//! 12 × 4 = 48 limbs (vs 72 limbs on BLS12-381).
//!
//! # Per-row witness
//!
//! Each row commits four Fp12 values, mirroring the easy / hard split:
//!
//!   * `f_pre`        — Fp12 input to this step (≡ Miller-loop output on
//!                      the first row).
//!   * `f_easy_out`   — Fp12 result after the easy part on
//!                      `is_easy_step` rows.
//!   * `f_hard_out`   — Fp12 result after the hard part on
//!                      `is_hard_step` rows.
//!   * `f_final`      — Fp12 carried into the next row's `f_pre`
//!                      (semantically equals `f_hard_out` on hard rows).
//!
//! Each Fp12 is flattened to 12 × 4 = 48 big-endian u64 limbs.
//!
//! # Selectors
//!
//!   * `is_easy_step` — this row applies the easy-part transformation.
//!   * `is_hard_step` — this row applies the hard-part transformation.
//!
//! The two are mutually exclusive {0, 1}; both zero indicates a
//! padding row.
//!
//! Column total: 4 · 48 (Fp12 cols) + 2 (selectors) = **194 columns**.
//!
//! # What is algebraically enforced (this AIR)
//!
//! 1. `is_easy_step ∈ {0, 1}` — selector binarity.
//! 2. `is_hard_step ∈ {0, 1}` — selector binarity.
//! 3. `is_easy_step · is_hard_step = 0` — mutual exclusion.
//! 4. **Easy-step c₀ pass-through** (row-local, on `is_easy_step` rows):
//!    placeholder substantive Fp12-equality body that pins
//!    `f_easy_out.c0 == f_pre.c0` (24 limb sub-bodies via β-RLC),
//!    holding on `Fp12::one()` witnesses since `conjugate(1) = 1`. The
//!    real algebraic body decomposes over `nonnative_fp_air` and is
//!    deferred — see [`crate::final_exp_air`] for the full story.
//! 5. **Hard-step pass-through** (row-local, on `is_hard_step` rows):
//!    `f_final = f_hard_out` for every Fp12 limb (β-RLC over 48 limbs).
//! 6. **Shifted row-chain continuity**: `f_pre[r+1] = f_final[r]` for
//!    each of the 48 Fp12 limb columns.
//!
//! # What is NOT yet enforced (deferred)
//!
//! The deep arithmetic relations (`f_easy_out = f_pre^{easy_exp}` and
//! `f_hard_out = f_easy_out^{hard_exp}`) decompose into Fp / Fp2 / Fp6
//! sub-ops over `nonnative_fp_air` and are deferred to follow-up
//! phases.
//!
//! # Cross-AIR linkages
//!
//! * [`make_bn254_final_exp_f_pre_to_miller_output_descriptor`] — binds
//!   this AIR's `f_pre` (48 cols) gated by `IS_EASY_STEP` ↔
//!   [`crate::bn254_pairing_internals_air`]'s `acc_post` (48 cols) gated
//!   by `COL_SEL_MILLER_DOUBLING`. Same boundary-selector caveat as the
//!   BLS12-381 sibling: without a dedicated `IS_FIRST_ROW` / `IS_LAST_ROW`
//!   boundary selector the linkage matches across all real rows. The
//!   descriptor is the column-shape spec the future boundary-gated
//!   linkage will use.
//!
//! * [`make_bn254_final_exp_f_final_to_miller_output_descriptor`] — same
//!   shape on the B side but bound to `f_final` (the result), useful
//!   when composing this AIR's output into downstream pairing-equation
//!   AIRs.

use crate::bn254_pairing_internals_air as bn254;
use crate::bn254_pairing_internals_air::{Fp, Fp12};
use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of 64-bit limbs per BN254 Fp element (254-bit field → 4 × 64).
pub const LIMBS_PER_FP: usize = bn254::LIMBS_PER_BN254_FP;
/// Number of distinct Fp values per Fp12 element.
pub const FP_PER_FP12: usize = bn254::FP_PER_FP12;
/// Number of limbs per Fp12 element = 12 × 4 = 48.
pub const LIMBS_PER_FP12: usize = bn254::LIMBS_PER_FP12;
/// Number of Fp values in the c₀ half of an Fp12 (one Fp6 = 6 Fp).
pub const FP_PER_FP6: usize = 6;
/// Number of limbs in the c₀ half of an Fp12 = 6 × 4 = 24.
pub const LIMBS_PER_FP6: usize = FP_PER_FP6 * LIMBS_PER_FP;

/// BN254 curve parameter z (used by the hard-part addition chain).
/// `z = 4965661367192848881`.
pub const BN254_Z: u64 = 4_965_661_367_192_848_881;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_F_PRE_OFFSET: usize = 0;
pub const COL_F_EASY_OUT_OFFSET: usize = COL_F_PRE_OFFSET + LIMBS_PER_FP12;
pub const COL_F_HARD_OUT_OFFSET: usize = COL_F_EASY_OUT_OFFSET + LIMBS_PER_FP12;
pub const COL_F_FINAL_OFFSET: usize = COL_F_HARD_OUT_OFFSET + LIMBS_PER_FP12;
pub const COL_IS_EASY_STEP: usize = COL_F_FINAL_OFFSET + LIMBS_PER_FP12;
pub const COL_IS_HARD_STEP: usize = COL_IS_EASY_STEP + 1;
pub const NUM_COLUMNS: usize = COL_IS_HARD_STEP + 1;

/// Row-local constraints:
///   0: `is_easy_step ∈ {0, 1}`
///   1: `is_hard_step ∈ {0, 1}`
///   2: `is_easy_step · is_hard_step = 0`
///   3: easy-step c₀ pass-through (β-RLC over 24 Fp6 limb sub-bodies,
///      gated by `is_easy_step`).
///   4: hard-step `f_final = f_hard_out` (β-RLC over 48 Fp12 limb
///      sub-bodies, gated by `is_hard_step`).
pub const NUM_ROW_CONSTRAINTS: usize = 5;

/// Shifted-row constraint categories:
///   0: `f_pre[r+1] = f_final[r]` (48 sub-bodies via β-RLC).
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Bn254FinalExpRow {
    pub f_pre: Fp12,
    pub f_easy_out: Fp12,
    pub f_hard_out: Fp12,
    pub f_final: Fp12,
    pub is_easy_step: bool,
    pub is_hard_step: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Bn254FinalExpWitness {
    pub rows: Vec<Bn254FinalExpRow>,
}

impl Bn254FinalExpWitness {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Append an easy-step row. Sets `f_easy_out = f_pre` as the
    /// placeholder (matches the c₀ pass-through constraint on
    /// `f_pre = Fp12::one()`); pins `f_hard_out = f_easy_out` and
    /// `f_final = f_easy_out` for internal continuity.
    pub fn push_easy(&mut self, f_pre: Fp12) {
        let f_easy_out = f_pre;
        let f_hard_out = f_easy_out;
        let f_final = f_hard_out;
        self.rows.push(Bn254FinalExpRow {
            f_pre,
            f_easy_out,
            f_hard_out,
            f_final,
            is_easy_step: true,
            is_hard_step: false,
        });
    }

    /// Append a hard-step row. Sets `f_easy_out = f_pre` and
    /// `f_final = f_hard_out` (the load-bearing relation that constraint
    /// 4 enforces).
    pub fn push_hard(&mut self, f_pre: Fp12, f_hard_out: Fp12) {
        let f_easy_out = f_pre;
        let f_final = f_hard_out;
        self.rows.push(Bn254FinalExpRow {
            f_pre,
            f_easy_out,
            f_hard_out,
            f_final,
            is_easy_step: false,
            is_hard_step: true,
        });
    }

    /// Append a raw row, used by tampering tests.
    pub fn push_raw(&mut self, row: Bn254FinalExpRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

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

fn write_fp12_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    f: &Fp12,
    curve: CurveType,
) {
    // Same flattening order as bn254_pairing_internals_air and
    // miller_step_air.
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
    witness: &Bn254FinalExpWitness,
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

pub struct Bn254FinalExpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bn254FinalExpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

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

impl VmConstraintSystem for Bn254FinalExpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "bn254_is_easy_step_binary".into(),
            "bn254_is_hard_step_binary".into(),
            "bn254_selectors_mutually_exclusive".into(),
            "bn254_easy_step_c0_passthrough".into(),
            "bn254_hard_step_f_final_equals_f_hard_out".into(),
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
        // β = 7 — same convention as the BLS12-381 sibling.
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
        // 3: easy-step c₀ pass-through (over LIMBS_PER_FP6 = 24 limbs).
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
        // 4: hard-step f_final = f_hard_out (over all 48 Fp12 limbs).
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

        // 0
        {
            let v = &col_evals[COL_IS_EASY_STEP];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1
        {
            let v = &col_evals[COL_IS_HARD_STEP];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2
        {
            let e = &col_evals[COL_IS_EASY_STEP];
            let h = &col_evals[COL_IS_HARD_STEP];
            acc = acc.add(&alpha_pow.mul(&e.mul(h)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3
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
        // 4
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

        // 0
        {
            let v = &col_coeffs[COL_IS_EASY_STEP];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1
        {
            let v = &col_coeffs[COL_IS_HARD_STEP];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2
        {
            let e = &col_coeffs[COL_IS_EASY_STEP];
            let h = &col_coeffs[COL_IS_HARD_STEP];
            let body = poly_mul(e, h, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3
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
        // 4
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
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        (0..LIMBS_PER_FP12).map(|j| COL_F_PRE_OFFSET + j).collect()
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }
}

// ─── Shifted (cross-row) constraint helpers ───────────────────────────

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
    if constraint_index == 0 {
        for j in 0..LIMBS_PER_FP12 {
            let a = &columns[COL_F_PRE_OFFSET + j][next];
            let b = &columns[COL_F_FINAL_OFFSET + j][r];
            let body = a.sub(b);
            acc = acc.add(&beta_pow.mul(&body));
            beta_pow = beta_pow.mul(alpha);
        }
    }
    acc
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Cross-AIR LogUp descriptor: this AIR's `f_pre` Fp12 limbs (48 cols)
/// gated by `IS_EASY_STEP` ↔
/// [`crate::bn254_pairing_internals_air`]'s `acc_post` Fp12 limbs
/// (48 cols) gated by `COL_SEL_MILLER_DOUBLING`.
///
/// Pins this AIR's input `f_pre` to the Miller-loop output committed by
/// the BN254 pairing-internals AIR. Caveat (same shape as the BLS12-381
/// sibling): without a dedicated `IS_FIRST_ROW` boundary selector this
/// matches `f_pre` on every easy-step row, not just the first. A
/// follow-up phase will introduce that boundary selector.
pub fn make_bn254_final_exp_f_pre_to_miller_output_descriptor(
    final_exp_layer_index: usize,
    pairing_internals_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| bn254::COL_ACC_POST_OFFSET + j)
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| COL_F_PRE_OFFSET + j)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "bn254_final_exp_f_pre_to_miller_output_v1".into(),
        a_layer_index: pairing_internals_layer_index,
        a_columns,
        a_selector_column: Some(bn254::COL_SEL_MILLER_DOUBLING),
        b_layer_index: final_exp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_EASY_STEP),
    }
}

/// Cross-AIR LogUp descriptor: this AIR's `f_final` (48 cols) gated by
/// `IS_HARD_STEP` ↔ [`crate::bn254_pairing_internals_air`]'s `acc_post`
/// (48 cols) gated by `COL_SEL_FINAL_EXP_HARD`.
///
/// Bind point for composing this scaffold's output with downstream
/// pairing-equation AIRs that consume the final-exp result via the
/// pairing-internals AIR's final-exp-hard rows.
pub fn make_bn254_final_exp_f_final_to_miller_output_descriptor(
    final_exp_layer_index: usize,
    pairing_internals_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| bn254::COL_ACC_POST_OFFSET + j)
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| COL_F_FINAL_OFFSET + j)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "bn254_final_exp_f_final_to_miller_output_v1".into(),
        a_layer_index: pairing_internals_layer_index,
        a_columns,
        a_selector_column: Some(bn254::COL_SEL_FINAL_EXP_HARD),
        b_layer_index: final_exp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_HARD_STEP),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> Fp12 {
        // Fp12::one() is a fixed point of the entire final-exp map, so
        // the scaffold's placeholder c₀ pass-through and continuity
        // constraints all hold on identity witnesses.
        Fp12::one()
    }

    fn honest_witness(num_rows: usize) -> Bn254FinalExpWitness {
        let mut w = Bn254FinalExpWitness::new();
        let mut f = sample_input();
        for i in 0..num_rows {
            if i % 2 == 0 {
                w.push_easy(f);
            } else {
                w.push_hard(f, f);
            }
            f = w.rows.last().unwrap().f_final;
        }
        w
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: column layout sanity
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(LIMBS_PER_FP, 4);
        assert_eq!(LIMBS_PER_FP12, 48);
        assert_eq!(LIMBS_PER_FP6, 24);
        assert_eq!(COL_F_PRE_OFFSET, 0);
        assert_eq!(COL_F_EASY_OUT_OFFSET, LIMBS_PER_FP12);
        assert_eq!(COL_F_HARD_OUT_OFFSET, 2 * LIMBS_PER_FP12);
        assert_eq!(COL_F_FINAL_OFFSET, 3 * LIMBS_PER_FP12);
        assert_eq!(COL_IS_EASY_STEP, 4 * LIMBS_PER_FP12);
        assert_eq!(COL_IS_HARD_STEP, COL_IS_EASY_STEP + 1);
        // 4·48 + 2 = 194.
        assert_eq!(NUM_COLUMNS, 194);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: witness builds + trace shape
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_and_trace_has_expected_shape() {
        let w = honest_witness(2);
        assert_eq!(w.rows.len(), 2);
        assert!(w.rows[0].is_easy_step && !w.rows[0].is_hard_step);
        assert!(!w.rows[1].is_easy_step && w.rows[1].is_hard_step);
        assert_eq!(w.rows[1].f_pre, w.rows[0].f_final);
        assert_eq!(w.rows[1].f_final, w.rows[1].f_hard_out);

        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);
        assert!(trace.columns[COL_IS_EASY_STEP].evaluations[0].sub(&one).is_zero());
        assert!(trace.columns[COL_IS_HARD_STEP].evaluations[0].is_zero());
        assert!(trace.columns[COL_IS_EASY_STEP].evaluations[1].is_zero());
        assert!(trace.columns[COL_IS_HARD_STEP].evaluations[1].sub(&one).is_zero());
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: constraints zero on honest witness
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = Bn254FinalExpConstraintSystem::new(trace.num_rows);
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
        let curve = CurveType::Bls12381;
        let alpha = Scalar::from_u64(7, curve);
        for cidx in 0..NUM_SHIFTED {
            let body = evaluate_shifted_row(&col_refs, 0, 1, cidx, &alpha);
            assert!(body.is_zero(), "shifted constraint {} fired", cidx);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: tampered f_final breaks hard-step pass-through
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn hard_step_passthrough_fires_on_tampered_f_final() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        // Row 1 is the hard step. Tamper limb 3 of f_final on row 1.
        let original = cols[COL_F_FINAL_OFFSET + 3][1].clone();
        cols[COL_F_FINAL_OFFSET + 3][1] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let cs = Bn254FinalExpConstraintSystem::new(trace.num_rows);
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        // Selector + mutex constraints unaffected.
        for i in 0..3 {
            for (r, val) in results[i].iter().enumerate() {
                assert!(val.is_zero(), "constraint {} at row {} fired", i, r);
            }
        }
        // Easy-step c₀ pass-through unaffected (row 1's gate = 0).
        assert!(results[3][1].is_zero());
        // Hard-step pass-through MUST fire at row 1.
        assert!(
            !results[4][1].is_zero(),
            "tampered hard-step f_final must fire constraint 4",
        );
        assert!(results[4][0].is_zero(), "constraint 4 must NOT fire on easy row");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: selector binary + mutual-exclusion fire on tampering
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn selector_constraints_fire_on_tampering() {
        let w = honest_witness(1);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);

        // Non-binary is_easy_step.
        cols[COL_IS_EASY_STEP][0] = Scalar::from_u64(3, curve);
        let cs = Bn254FinalExpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[0][0].is_zero(),
            "non-binary is_easy_step must fire constraint 0",
        );

        // Mutual exclusion violation.
        cols[COL_IS_EASY_STEP][0] = one.clone();
        cols[COL_IS_HARD_STEP][0] = one;
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[2][0].is_zero(),
            "both selectors = 1 must fire constraint 2 (mutex)",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: tampered f_final breaks chain continuity
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_f_final_breaks_chain_continuity() {
        let w = honest_witness(2);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        // Tamper limb 0 of f_final on row 0. Row 0 is the easy step, so
        // constraint 4 (gated by is_hard_step) does NOT fire on row 0,
        // but shifted continuity to row 1's f_pre breaks.
        let original = cols[COL_F_FINAL_OFFSET][0].clone();
        cols[COL_F_FINAL_OFFSET][0] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let alpha = Scalar::from_u64(7, curve);
        let body = evaluate_shifted_row(&col_refs, 0, 1, 0, &alpha);
        assert!(
            !body.is_zero(),
            "tampered f_final must break f-chain continuity (shifted 0)",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 7: cross-AIR linkage descriptors are well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn f_pre_to_miller_output_descriptor_well_formed() {
        let d = make_bn254_final_exp_f_pre_to_miller_output_descriptor(1, 0);
        assert_eq!(d.label, "bn254_final_exp_f_pre_to_miller_output_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), LIMBS_PER_FP12);
        assert_eq!(d.b_columns.len(), LIMBS_PER_FP12);
        assert_eq!(d.a_columns[0], bn254::COL_ACC_POST_OFFSET);
        assert_eq!(
            d.a_columns[LIMBS_PER_FP12 - 1],
            bn254::COL_ACC_POST_OFFSET + LIMBS_PER_FP12 - 1,
        );
        assert_eq!(d.b_columns[0], COL_F_PRE_OFFSET);
        assert_eq!(d.b_columns[LIMBS_PER_FP12 - 1], COL_F_PRE_OFFSET + LIMBS_PER_FP12 - 1);
        assert_eq!(d.a_selector_column, Some(bn254::COL_SEL_MILLER_DOUBLING));
        assert_eq!(d.b_selector_column, Some(COL_IS_EASY_STEP));
    }

    #[test]
    fn f_final_to_miller_output_descriptor_well_formed() {
        let d = make_bn254_final_exp_f_final_to_miller_output_descriptor(1, 0);
        assert_eq!(d.label, "bn254_final_exp_f_final_to_miller_output_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), LIMBS_PER_FP12);
        assert_eq!(d.b_columns.len(), LIMBS_PER_FP12);
        assert_eq!(d.b_columns[0], COL_F_FINAL_OFFSET);
        assert_eq!(d.a_selector_column, Some(bn254::COL_SEL_FINAL_EXP_HARD));
        assert_eq!(d.b_selector_column, Some(COL_IS_HARD_STEP));
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 8: shifted column indices cover f_pre
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn shifted_column_indices_cover_f_pre() {
        let cs = Bn254FinalExpConstraintSystem::new(2);
        let idx = cs.shifted_column_indices();
        assert_eq!(idx.len(), LIMBS_PER_FP12);
        for j in 0..LIMBS_PER_FP12 {
            assert_eq!(idx[j], COL_F_PRE_OFFSET + j);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 9: BN254_Z constant is the expected curve parameter
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn bn254_z_parameter_correct() {
        assert_eq!(BN254_Z, 4_965_661_367_192_848_881);
    }
}
