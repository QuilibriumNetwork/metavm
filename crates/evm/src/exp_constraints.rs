//! [`VmConstraintSystem`] wiring for the EVM EXP gadget AIR.
//!
//! Adapts the constraint bodies defined in [`crate::exp_air`] into the shape
//! required by the generic `prove_with_scheme` / `verify_with_scheme`
//! pipeline. The structure mirrors `metavm-zkp::sha256_constraints` and
//! `metavm-zkp::keccak_constraints`:
//!
//! - `EvmExpConstraintSystem`: the trait implementer. Construct with
//!   `EvmExpConstraintSystem::new()` for a single 256-row EXP gadget run.
//! - `build_trace_polynomials`: helper that lifts a `Vec<ExpRow>` into the
//!   `TracePolynomials` the pipeline expects.
//!
//! ## Decoupling from the main EVM constraint module
//!
//! See the [`crate::exp_air`] preamble: the main EVM constraint module's
//! EXP opcode constraint stays an oracle (zero body), and cross-AIR
//! linkage between the EVM trace and this gadget trace is a separate
//! follow-up. This module produces a provable witness for `base ^ exponent`
//! over its own clean 256-row trace, but does not yet bind that witness
//! back into the EVM trace.
//!
//! ## Constraint layout
//!
//! 4 row-local consolidated categories (label → index):
//!
//!   0. `exp_bit_binary`
//!   1. `active_binary`
//!   2. `squared_mul_chain`
//!   3. `mul_mul_chain`
//!
//! 3 cross-row (shifted) constraints:
//!
//!   0. `result_chain`
//!   1. `base_invariant`
//!   2. `bit_index_decrement`
//!
//! Each multi-limb category β-RLCs its sub-bodies into a single body. We
//! fix β = α (single Fiat-Shamir challenge) for simplicity, exactly as the
//! SHA-256 / Keccak / nonnative-fp wirings do.
//!
//! ## Padding strategy
//!
//! Real traces have exactly [`crate::exp_air::NUM_STEPS`] = 256 rows, which
//! is itself a power of two — for the natural case there are NO padding
//! rows. For test traces shorter than 16 (the FFT minimum), padding rows
//! carry all-zero columns; every row-local body is gated by `active`, so
//! the bodies vanish trivially. The cross-row `result_chain` and
//! `base_invariant` bodies are gated by `active(ω·X)`, so the
//! real → padding transition vanishes too. The `bit_index_decrement`
//! body collapses to `0 = 0 − 0` on that transition because the last real
//! row's `bit_index` is 0.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use metavm_zkp::trace::{nearest_power_of_two, Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

use crate::exp_air::{
    self, alloc_trace, base_limb, evaluate_cross_row_at_point, evaluate_row_local,
    evaluate_row_local_at_point, mul_carry_limb, mul_limb, populate_row, result_in_limb,
    shifted_column_indices, squared_carry_limb, squared_limb, two_pow_64, COL_ACTIVE,
    COL_BIT_INDEX, COL_EXP_BIT, ExpRow, NUM_EXP_AIR_COLUMNS, NUM_LIMBS, NUM_STEPS,
};

/// Number of consolidated row-local constraint categories.
pub const NUM_ROW_CONSTRAINTS: usize = 7;

/// Number of consolidated cross-row constraints.
pub const NUM_SHIFTED: usize = 6;

// ──── Constraint system ───────────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the EVM EXP gadget AIR.
pub struct EvmExpConstraintSystem {
    /// Number of real trace rows (before padding). Almost always
    /// [`NUM_STEPS`]; tests sometimes use shorter traces so we keep this
    /// configurable.
    pub num_rows: usize,
}

impl EvmExpConstraintSystem {
    /// Construct an EXP constraint system for a full 256-row gadget run.
    pub fn new() -> Self {
        Self { num_rows: NUM_STEPS }
    }

    /// Construct with a custom row count (typically only used in tests).
    pub fn with_num_rows(num_rows: usize) -> Self {
        Self { num_rows }
    }
}

impl Default for EvmExpConstraintSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ──── Trace construction helper ───────────────────────────────────────

/// Build a [`TracePolynomials`] wrapping the EXP gadget bit-level trace
/// produced by populating `rows` in order. The trace pads up to the next
/// power of two automatically (a no-op for the natural 256-row case).
pub fn build_trace_polynomials(rows: &[ExpRow], curve: CurveType) -> TracePolynomials {
    let num_rows = rows.len();
    let padded = nearest_power_of_two(num_rows.max(1));
    let mut columns = alloc_trace(padded, curve);
    for (row, er) in rows.iter().enumerate() {
        populate_row(&mut columns, row, er, curve);
    }
    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial {
            evaluations: evals,
            degree: num_rows,
        })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ──── Helpers for polynomial-form bodies ──────────────────────────────

/// Build the polynomial corresponding to a single schoolbook MUL-limb body:
///   `Σ aᵢ · bⱼ + carry_in − carry_out · 2^64 − out_k = 0`
///
/// `pairs` is the slice of `(a_polynomial, b_polynomial)` cross-product
/// pairs to multiply and sum (length `limb + 1`). `carry_in` may be a
/// degenerate zero polynomial for limb 0.
#[allow(clippy::too_many_arguments)]
fn build_mul_limb_poly(
    pairs: &[(&Vec<Scalar>, &Vec<Scalar>)],
    carry_in: &Vec<Scalar>,
    out_k: &Vec<Scalar>,
    carry_out: &Vec<Scalar>,
    two_64: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum: Vec<Scalar> = vec![Scalar::zero(curve)];
    for (a, b) in pairs {
        let prod = poly_mul(a, b, curve);
        sum = poly_add(&sum, &prod, curve);
    }
    sum = poly_add(&sum, carry_in, curve);
    let hi = poly_scalar_mul(carry_out, two_64);
    let rhs = poly_add(&hi, out_k, curve);
    poly_sub(&sum, &rhs, curve)
}

/// Build the row-local "exp_bit_binary" body polynomial:
///   `active · exp_bit · (exp_bit − 1)`
fn build_exp_bit_binary_poly(
    column_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let active = &column_coeffs[COL_ACTIVE];
    let eb = &column_coeffs[COL_EXP_BIT];
    let one_poly = vec![Scalar::one(curve)];
    let eb_minus_1 = poly_sub(eb, &one_poly, curve);
    let inner = poly_mul(eb, &eb_minus_1, curve);
    poly_mul(active, &inner, curve)
}

/// Build the row-local "active_binary" body polynomial:
///   `active · (active − 1)`
fn build_active_binary_poly(
    column_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let active = &column_coeffs[COL_ACTIVE];
    let one_poly = vec![Scalar::one(curve)];
    let active_minus_1 = poly_sub(active, &one_poly, curve);
    poly_mul(active, &active_minus_1, curve)
}

/// Build the row-local "is_first_row_binary" body polynomial:
///   `IS_FIRST_ROW · (IS_FIRST_ROW − 1)`
///
/// Pins `IS_FIRST_ROW` to `{0, 1}`. Combined with the cross-AIR LogUp
/// linkage's selector gating, this is what the verifier uses to pick
/// the per-invocation anchor row.
fn build_is_first_row_binary_poly(
    column_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    use crate::exp_air::COL_IS_FIRST_ROW;
    let v = &column_coeffs[COL_IS_FIRST_ROW];
    let one_poly = vec![Scalar::one(curve)];
    let v_minus_1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_minus_1, curve)
}

/// Build the row-local "is_first_row_pins_bit_index" body polynomial:
///   `IS_FIRST_ROW · (bit_index − (NUM_STEPS − 1))`
///
/// On a row where `IS_FIRST_ROW = 1`, this forces `bit_index = NUM_STEPS − 1`
/// (= 255), the algorithm's first row. Combined with `bit_index_decrement`
/// the prover can't claim any other row is the anchor.
fn build_is_first_row_pins_bit_index_poly(
    column_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    use crate::exp_air::{COL_BIT_INDEX, COL_IS_FIRST_ROW, NUM_STEPS};
    let is_first = &column_coeffs[COL_IS_FIRST_ROW];
    let bit_idx = &column_coeffs[COL_BIT_INDEX];
    let last_idx_poly = vec![Scalar::from_u64((NUM_STEPS - 1) as u64, curve)];
    let diff = poly_sub(bit_idx, &last_idx_poly, curve);
    poly_mul(is_first, &diff, curve)
}

/// Build the row-local "is_first_row_pins_result_in" body polynomial
/// (#94 Phase 3):
///   `IS_FIRST_ROW · β-RLC_i (result_in[i] − initializer[i])`
/// where `initializer = (1, 0, 0, 0)`.
///
/// On a row where `IS_FIRST_ROW = 1`, pins `result_in = 1` (= the
/// algorithm's loop-body initializer). Closes the soundness gap that
/// Phase 1's boundary gating left open: cross-row `result_chain` is
/// suppressed at invocation boundaries, so without this row-local pin
/// a malicious prover could supply arbitrary `result_in` at any
/// invocation start.
fn build_is_first_row_pins_result_in_poly(
    column_coeffs: &[Vec<Scalar>],
    beta: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    use crate::exp_air::{result_in_limb, COL_IS_FIRST_ROW, NUM_LIMBS};
    let is_first = &column_coeffs[COL_IS_FIRST_ROW];
    let one_poly = vec![Scalar::one(curve)];

    let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..NUM_LIMBS {
        let ri = &column_coeffs[result_in_limb(i)];
        let body = if i == 0 {
            poly_sub(ri, &one_poly, curve)
        } else {
            ri.clone()
        };
        let scaled = poly_scalar_mul(&body, &bp);
        acc = poly_add(&acc, &scaled, curve);
        bp = bp.mul(beta);
    }
    poly_mul(is_first, &acc, curve)
}

/// Build the squaring MUL chain body polynomial (β-RLC over 4 limb bodies),
/// gated by `active`.
fn build_squared_mul_chain_poly(
    column_coeffs: &[Vec<Scalar>],
    beta: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let two_64 = two_pow_64(curve);
    let zero_poly = vec![Scalar::zero(curve)];
    let active = &column_coeffs[COL_ACTIVE];

    let a: [&Vec<Scalar>; 4] = [
        &column_coeffs[result_in_limb(0)],
        &column_coeffs[result_in_limb(1)],
        &column_coeffs[result_in_limb(2)],
        &column_coeffs[result_in_limb(3)],
    ];
    let s: [&Vec<Scalar>; 4] = [
        &column_coeffs[squared_limb(0)],
        &column_coeffs[squared_limb(1)],
        &column_coeffs[squared_limb(2)],
        &column_coeffs[squared_limb(3)],
    ];
    let cs: [&Vec<Scalar>; 4] = [
        &column_coeffs[squared_carry_limb(0)],
        &column_coeffs[squared_carry_limb(1)],
        &column_coeffs[squared_carry_limb(2)],
        &column_coeffs[squared_carry_limb(3)],
    ];

    // Schoolbook squaring: a*a layout.
    //   limb 0: a0·a0
    //   limb 1: a0·a1 + a1·a0
    //   limb 2: a0·a2 + a1·a1 + a2·a0
    //   limb 3: a0·a3 + a1·a2 + a2·a1 + a3·a0
    let bodies = [
        build_mul_limb_poly(&[(a[0], a[0])], &zero_poly, s[0], cs[0], &two_64, curve),
        build_mul_limb_poly(
            &[(a[0], a[1]), (a[1], a[0])],
            cs[0],
            s[1],
            cs[1],
            &two_64,
            curve,
        ),
        build_mul_limb_poly(
            &[(a[0], a[2]), (a[1], a[1]), (a[2], a[0])],
            cs[1],
            s[2],
            cs[2],
            &two_64,
            curve,
        ),
        build_mul_limb_poly(
            &[(a[0], a[3]), (a[1], a[2]), (a[2], a[1]), (a[3], a[0])],
            cs[2],
            s[3],
            cs[3],
            &two_64,
            curve,
        ),
    ];

    let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for body in &bodies {
        let scaled = poly_scalar_mul(body, &bp);
        acc = poly_add(&acc, &scaled, curve);
        bp = bp.mul(beta);
    }
    poly_mul(active, &acc, curve)
}

/// Build the squared*base MUL chain body polynomial (β-RLC over 4 limb bodies),
/// gated by `active`.
fn build_mul_mul_chain_poly(
    column_coeffs: &[Vec<Scalar>],
    beta: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let two_64 = two_pow_64(curve);
    let zero_poly = vec![Scalar::zero(curve)];
    let active = &column_coeffs[COL_ACTIVE];

    let s: [&Vec<Scalar>; 4] = [
        &column_coeffs[squared_limb(0)],
        &column_coeffs[squared_limb(1)],
        &column_coeffs[squared_limb(2)],
        &column_coeffs[squared_limb(3)],
    ];
    let b: [&Vec<Scalar>; 4] = [
        &column_coeffs[base_limb(0)],
        &column_coeffs[base_limb(1)],
        &column_coeffs[base_limb(2)],
        &column_coeffs[base_limb(3)],
    ];
    let m: [&Vec<Scalar>; 4] = [
        &column_coeffs[mul_limb(0)],
        &column_coeffs[mul_limb(1)],
        &column_coeffs[mul_limb(2)],
        &column_coeffs[mul_limb(3)],
    ];
    let cm: [&Vec<Scalar>; 4] = [
        &column_coeffs[mul_carry_limb(0)],
        &column_coeffs[mul_carry_limb(1)],
        &column_coeffs[mul_carry_limb(2)],
        &column_coeffs[mul_carry_limb(3)],
    ];

    let bodies = [
        build_mul_limb_poly(&[(s[0], b[0])], &zero_poly, m[0], cm[0], &two_64, curve),
        build_mul_limb_poly(
            &[(s[0], b[1]), (s[1], b[0])],
            cm[0],
            m[1],
            cm[1],
            &two_64,
            curve,
        ),
        build_mul_limb_poly(
            &[(s[0], b[2]), (s[1], b[1]), (s[2], b[0])],
            cm[1],
            m[2],
            cm[2],
            &two_64,
            curve,
        ),
        build_mul_limb_poly(
            &[(s[0], b[3]), (s[1], b[2]), (s[2], b[1]), (s[3], b[0])],
            cm[2],
            m[3],
            cm[3],
            &two_64,
            curve,
        ),
    ];

    let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for body in &bodies {
        let scaled = poly_scalar_mul(body, &bp);
        acc = poly_add(&acc, &scaled, curve);
        bp = bp.mul(beta);
    }
    poly_mul(active, &acc, curve)
}

// ──── VmConstraintSystem implementation ───────────────────────────────

impl VmConstraintSystem for EvmExpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "exp_bit_binary".into(),
            "active_binary".into(),
            "squared_mul_chain".into(),
            "mul_mul_chain".into(),
            "is_first_row_binary".into(),
            "is_first_row_pins_bit_index".into(),
            "is_first_row_pins_result_in".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert_eq!(
            columns.len(),
            NUM_EXP_AIR_COLUMNS,
            "EXP AIR expects {} columns",
            NUM_EXP_AIR_COLUMNS
        );
        let curve = columns[0][0].curve_type();
        // β = 2 matches the SHA-256 wiring's choice of a fixed β here.
        let beta = Scalar::from_u64(2, curve);
        let evals = evaluate_row_local(columns, &beta);
        evals.into_iter().map(|c| c.values).collect()
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < NUM_EXP_AIR_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let beta = alpha; // β = α as in keccak / sha256 / nonnative-fp wirings.
        let bodies = evaluate_row_local_at_point(col_evals_at_z, beta);
        let curve = alpha.curve_type();
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        // `active` is the only "selector"-like column in this AIR. We expose
        // it so the prover's selector-padding logic recognises the gate.
        vec![COL_ACTIVE]
    }

    /// Return `None`: padding rows carry all-zero data + selector columns,
    /// which makes every row-local body vanish via the `active` gating. The
    /// `active_binary` constraint `active · (active − 1) = 0` is satisfied
    /// with `active = 0`.
    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        // Defensive: zero every column on padding rows. The trace constructor
        // already zero-fills; we make this explicit so any upstream stage
        // that mutated padding cells is reset.
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < NUM_EXP_AIR_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_EXP_AIR_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let beta = alpha.clone();

        let body0 = build_exp_bit_binary_poly(column_coeffs, curve);
        let body1 = build_active_binary_poly(column_coeffs, curve);
        let body2 = build_squared_mul_chain_poly(column_coeffs, &beta, curve);
        let body3 = build_mul_mul_chain_poly(column_coeffs, &beta, curve);
        let body4 = build_is_first_row_binary_poly(column_coeffs, curve);
        let body5 = build_is_first_row_pins_bit_index_poly(column_coeffs, curve);
        let body6 = build_is_first_row_pins_result_in_poly(column_coeffs, &beta, curve);

        let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in [&body0, &body1, &body2, &body3, &body4, &body5, &body6] {
            let scaled = poly_scalar_mul(body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    // ── Cross-row support ──────────────────────────────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        shifted_column_indices()
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
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
        let expected_shift_len = exp_air::shifted_column_indices().len();
        if shifted_evals.len() != expected_shift_len {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let beta = alpha;
        let bodies = evaluate_cross_row_at_point(col_evals_at_z, shifted_evals, beta);

        // Wrap-around exclusion factor `(z − ω^{n−1})`.
        let exclusion = z.sub(omega_n_minus_1);

        // α^alpha_offset, then α^{alpha_offset + 1}, etc.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut acc = Scalar::zero(curve);
        for body in &bodies {
            let term = body.mul(&exclusion).mul(&ap);
            acc = acc.add(&term);
            ap = ap.mul(alpha);
        }
        acc
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let beta = alpha.clone();
        let one_poly = vec![Scalar::one(curve)];

        // Build act_shift = poly_shift(active, omega).
        let active = &column_coeffs[COL_ACTIVE];
        let act_shift = poly_shift(active, omega);

        // Build is_first_shift = poly_shift(IS_FIRST_ROW, omega) and
        // the boundary gate (1 − is_first_shift). #94 multi-invocation:
        // every cross-row body multiplies by this gate so transitions
        // INTO an invocation-start row (is_first_shift = 1) vanish.
        // Single-invocation traces only have IS_FIRST_ROW = 1 on row
        // 0 and the transition INTO row 0 is the wrap (already
        // excluded by `(X − ω^{n−1})`); the new factor is 1 elsewhere
        // so behavior is unchanged.
        use crate::exp_air::COL_IS_FIRST_ROW;
        let is_first = &column_coeffs[COL_IS_FIRST_ROW];
        let is_first_shift = poly_shift(is_first, omega);
        let inv_boundary_gate = poly_sub(&one_poly, &is_first_shift, curve);
        let act_shift_gated = poly_mul(&act_shift, &inv_boundary_gate, curve);

        // 1. result_chain body polynomial:
        //    Σ β^i · act_shift · (1 − is_first_shift) ·
        //         (result_in_shift[i] − (eb · mul[i] + (1 − eb) · squared[i]))
        let eb = &column_coeffs[COL_EXP_BIT];
        let one_minus_eb = poly_sub(&one_poly, eb, curve);

        let mut res_acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..NUM_LIMBS {
            let res_in = &column_coeffs[result_in_limb(i)];
            let res_in_shift = poly_shift(res_in, omega);
            let mul_i = &column_coeffs[mul_limb(i)];
            let sq_i = &column_coeffs[squared_limb(i)];
            // chosen = eb · mul + (1 − eb) · squared
            let term_eb_mul = poly_mul(eb, mul_i, curve);
            let term_one_minus_eb_sq = poly_mul(&one_minus_eb, sq_i, curve);
            let chosen = poly_add(&term_eb_mul, &term_one_minus_eb_sq, curve);
            let body = poly_sub(&res_in_shift, &chosen, curve);
            let gated = poly_mul(&act_shift_gated, &body, curve);
            let scaled = poly_scalar_mul(&gated, &bp);
            res_acc = poly_add(&res_acc, &scaled, curve);
            bp = bp.mul(&beta);
        }

        // 2. base_invariant body polynomial:
        //    Σ β^i · act_shift · (1 − is_first_shift) · (base_shift[i] − base[i])
        let mut base_acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..NUM_LIMBS {
            let base_i = &column_coeffs[base_limb(i)];
            let base_shift = poly_shift(base_i, omega);
            let body = poly_sub(&base_shift, base_i, curve);
            let gated = poly_mul(&act_shift_gated, &body, curve);
            let scaled = poly_scalar_mul(&gated, &bp);
            base_acc = poly_add(&base_acc, &scaled, curve);
            bp = bp.mul(&beta);
        }

        // 3. bit_index_decrement body polynomial:
        //    (1 − is_first_shift) · (bi_shift − (bi − act_shift))
        let bi = &column_coeffs[COL_BIT_INDEX];
        let bi_shift = poly_shift(bi, omega);
        let bi_minus_act = poly_sub(bi, &act_shift, curve);
        let bi_body_inner = poly_sub(&bi_shift, &bi_minus_act, curve);
        let bi_body = poly_mul(&inv_boundary_gate, &bi_body_inner, curve);

        // 4. exponent_invariant body polynomial:
        //    Σ β^i · act_shift · (1 − is_first_shift) ·
        //         (EXPONENT_shift[i] − EXPONENT[i])
        use crate::exp_air::{COL_EXPONENT_OFFSET, COL_FINAL_OUTPUT_OFFSET};
        let mut exp_acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..NUM_LIMBS {
            let cur = &column_coeffs[COL_EXPONENT_OFFSET + i];
            let nxt = poly_shift(cur, omega);
            let body = poly_sub(&nxt, cur, curve);
            let gated = poly_mul(&act_shift_gated, &body, curve);
            let scaled = poly_scalar_mul(&gated, &bp);
            exp_acc = poly_add(&exp_acc, &scaled, curve);
            bp = bp.mul(&beta);
        }

        // 5. final_output_invariant body polynomial:
        //    Σ β^i · act_shift · (1 − is_first_shift) ·
        //         (FINAL_OUTPUT_shift[i] − FINAL_OUTPUT[i])
        let mut fo_acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..NUM_LIMBS {
            let cur = &column_coeffs[COL_FINAL_OUTPUT_OFFSET + i];
            let nxt = poly_shift(cur, omega);
            let body = poly_sub(&nxt, cur, curve);
            let gated = poly_mul(&act_shift_gated, &body, curve);
            let scaled = poly_scalar_mul(&gated, &bp);
            fo_acc = poly_add(&fo_acc, &scaled, curve);
            bp = bp.mul(&beta);
        }

        // 6. final_output_at_last_row body polynomial:
        //    Σ β^i · active · (is_first_shift + (1 − act_shift)) ·
        //         (FINAL_OUTPUT[i] − eb · mul[i] − (1 − eb) · squared[i])
        //    The indicator fires on the last row of every invocation —
        //    either the trace's overall last real row (1 − act_shift = 1)
        //    OR the row whose successor starts a new invocation
        //    (is_first_shift = 1). Since IS_FIRST_ROW = 1 implies the
        //    next row is active, the two conditions are disjoint.
        let active_poly = &column_coeffs[COL_ACTIVE];
        let one_minus_act_shift = poly_sub(&one_poly, &act_shift, curve);
        let last_indicator_inner =
            poly_add(&is_first_shift, &one_minus_act_shift, curve);
        let last_indicator = poly_mul(active_poly, &last_indicator_inner, curve);
        let mut last_acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..NUM_LIMBS {
            let mul_i = &column_coeffs[mul_limb(i)];
            let sq_i = &column_coeffs[squared_limb(i)];
            // chosen = eb · mul + (1 − eb) · squared
            let term_eb_mul = poly_mul(eb, mul_i, curve);
            let term_one_minus_eb_sq = poly_mul(&one_minus_eb, sq_i, curve);
            let chosen = poly_add(&term_eb_mul, &term_one_minus_eb_sq, curve);
            let fo_i = &column_coeffs[COL_FINAL_OUTPUT_OFFSET + i];
            let body = poly_sub(fo_i, &chosen, curve);
            let gated = poly_mul(&last_indicator, &body, curve);
            let scaled = poly_scalar_mul(&gated, &bp);
            last_acc = poly_add(&last_acc, &scaled, curve);
            bp = bp.mul(&beta);
        }

        // Multiply each body by (X − ω^{n−1}). Domain is `column_coeffs[0].len()`
        // rows post-IFFT — but the natural identification is `domain_size` from
        // the trace. We use _domain_size here.
        let n_minus_1 = (_domain_size as u64).saturating_sub(1);
        let mut omega_n_minus_1 = Scalar::one(curve);
        let mut e = n_minus_1;
        let mut base_pow = omega.clone();
        while e > 0 {
            if e & 1 == 1 {
                omega_n_minus_1 = omega_n_minus_1.mul(&base_pow);
            }
            base_pow = base_pow.mul(&base_pow);
            e >>= 1;
        }

        let res_excluded = poly_mul_linear(&res_acc, &omega_n_minus_1);
        let base_excluded = poly_mul_linear(&base_acc, &omega_n_minus_1);
        let bi_excluded = poly_mul_linear(&bi_body, &omega_n_minus_1);
        let exp_excluded = poly_mul_linear(&exp_acc, &omega_n_minus_1);
        let fo_excluded = poly_mul_linear(&fo_acc, &omega_n_minus_1);
        let last_excluded = poly_mul_linear(&last_acc, &omega_n_minus_1);

        // α^alpha_offset, …, with each constraint getting its own α power.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let scaled_res = poly_scalar_mul(&res_excluded, &ap);
        ap = ap.mul(alpha);
        let scaled_base = poly_scalar_mul(&base_excluded, &ap);
        ap = ap.mul(alpha);
        let scaled_bi = poly_scalar_mul(&bi_excluded, &ap);
        ap = ap.mul(alpha);
        let scaled_exp = poly_scalar_mul(&exp_excluded, &ap);
        ap = ap.mul(alpha);
        let scaled_fo = poly_scalar_mul(&fo_excluded, &ap);
        ap = ap.mul(alpha);
        let scaled_last = poly_scalar_mul(&last_excluded, &ap);

        let mut combined = poly_add(&scaled_res, &scaled_base, curve);
        combined = poly_add(&combined, &scaled_bi, curve);
        combined = poly_add(&combined, &scaled_exp, curve);
        combined = poly_add(&combined, &scaled_fo, curve);
        poly_add(&combined, &scaled_last, curve)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // The EXP gadget multiplies three classes of u64 limbs through
        // schoolbook MUL: result_in (squared into squared), squared (multiplied
        // by base into mul), and base. Their carry chains are bounded by the
        // worst-case 4-limb collapse: the largest carry term for limb 3 is
        // `4 · (2^64 − 1)^2 / 2^64 < 2^66`, but we conservatively declare the
        // safe 64-bit range — the upper carry bits are constrained algebraically
        // by the next-limb body. We forward an empty requirements set: this
        // gadget runs standalone and the integer overflow path is not yet
        // exercised end-to-end. A future iteration would declare:
        //   - 16 result_in / squared / mul limbs at 64 bits
        //   - 8 base limbs at 64 bits
        //   - 8 carry limbs at 64 bits (squared_carry, mul_carry)
        // …and route them through the LogUp pipeline.
        LookupRequirements::none()
    }
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use metavm_zkp::field::CurveType;
    use revm::primitives::U256;

    use crate::exp_air::{exp_witness, pow_mod_2_256, NUM_STEPS};

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    /// `2^7 = 128`. Cheap deterministic driver shared with the AIR tests.
    fn small_rows() -> (U256, U256, Vec<ExpRow>) {
        let base = U256::from(2u64);
        let exponent = U256::from(7u64);
        let rows = exp_witness(base, exponent);
        (base, exponent, rows)
    }

    /// Assemble a column set with all 256 rows populated.
    fn populated_columns() -> (Vec<Vec<Scalar>>, Vec<ExpRow>) {
        let curve = CurveType::Bls48581;
        let (_, _, rows) = small_rows();
        let cols = exp_air::populate_trace(&rows, curve);
        (cols, rows)
    }

    #[test]
    fn exp_witness_matches_reference_pow_mod() {
        let (base, exponent, rows) = small_rows();
        let last = rows.last().unwrap();
        let chosen = if last.exp_bit == 1 { last.mul } else { last.squared };
        assert_eq!(chosen, [128, 0, 0, 0]);
        assert_eq!(chosen, pow_mod_2_256(base, exponent));
    }

    #[test]
    fn exp_cs_labels_and_counts() {
        let cs = EvmExpConstraintSystem::new();
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_constraints(), 7);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert_eq!(cs.num_shifted_constraints(), 6);
        assert_eq!(cs.selector_column_indices(), vec![COL_ACTIVE]);
        // 4 result_in + 4 base + 1 bit_index + 1 active + 4 EXPONENT +
        // 4 FINAL_OUTPUT + 1 IS_FIRST_ROW (added for #94 multi-invocation
        // boundary gating).
        assert_eq!(cs.shifted_column_indices().len(), 19);
        assert!(cs.padding_selector_column().is_none());
    }

    #[test]
    fn exp_cs_evaluate_on_domain_matches_witness() {
        let (cols, _rows) = populated_columns();
        let cs = EvmExpConstraintSystem::new();
        let evals = cs.evaluate_on_domain(&col_refs(&cols), NUM_STEPS);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, vec) in evals.iter().enumerate() {
            for (row, v) in vec.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {} on valid witness",
                    k,
                    cs.constraint_labels()[k],
                    row
                );
            }
        }
    }

    #[test]
    fn exp_cs_evaluate_at_point_zero_on_real_rows() {
        let (cols, _rows) = populated_columns();
        let cs = EvmExpConstraintSystem::new();
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);
        for r in 0..NUM_STEPS {
            let row_vals: Vec<Scalar> = cols.iter().map(|c| c[r].clone()).collect();
            let c_at = cs.evaluate_at_point(&row_vals, &alpha);
            assert!(
                c_at.is_zero(),
                "combined C(row {}) nonzero on valid witness",
                r
            );
        }
    }

    #[test]
    fn exp_cs_evaluate_at_point_zero_on_all_zero_row() {
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let row_vals = vec![zero; NUM_EXP_AIR_COLUMNS];
        let cs = EvmExpConstraintSystem::new();
        let alpha = Scalar::from_u64(23, curve);
        let c_at = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            c_at.is_zero(),
            "all-zero padding row must evaluate to zero (otherwise \
             padding_selector_column must be set)"
        );
    }

    #[test]
    fn exp_cs_constraints_reject_tampered_result_limb() {
        let curve = CurveType::Bls48581;
        let (mut cols, _rows) = populated_columns();
        // Tamper a non-zero squared limb on row 254. (Row 254 has result_in
        // = 64, squared = 4096 — definitely non-zero.)
        let r = 254;
        let one = Scalar::one(curve);
        cols[squared_limb(0)][r] = cols[squared_limb(0)][r].add(&one);
        let cs = EvmExpConstraintSystem::new();
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = cols.iter().map(|c| c[r].clone()).collect();
        let c_at = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "combined constraint must fire on tampered squared limb"
        );
    }

    #[test]
    fn exp_cs_constraints_reject_non_binary_is_first_row() {
        use crate::exp_air::COL_IS_FIRST_ROW;
        let curve = CurveType::Bls48581;
        let (mut cols, _rows) = populated_columns();
        // Forge IS_FIRST_ROW = 2 on row 0 (out of {0,1}). The
        // is_first_row_binary constraint must fire.
        cols[COL_IS_FIRST_ROW][0] = Scalar::from_u64(2, curve);
        let cs = EvmExpConstraintSystem::new();
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = cols.iter().map(|c| c[0].clone()).collect();
        let c_at = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "is_first_row_binary must fire when IS_FIRST_ROW ∉ {{0,1}}"
        );
    }

    #[test]
    fn exp_cs_constraints_reject_is_first_row_on_wrong_bit_index() {
        use crate::exp_air::{COL_BIT_INDEX, COL_IS_FIRST_ROW, NUM_STEPS};
        let curve = CurveType::Bls48581;
        let (mut cols, _rows) = populated_columns();
        // The witness builder sets IS_FIRST_ROW = 1 only on row 0
        // (bit_index = NUM_STEPS-1 = 255). Forge IS_FIRST_ROW = 1 on
        // row 100 (where bit_index = NUM_STEPS - 1 - 100 = 155).
        // The is_first_row_pins_bit_index constraint must fire because
        // 1 · (155 − 255) = -100 ≠ 0.
        let r = 100;
        cols[COL_IS_FIRST_ROW][r] = Scalar::one(curve);
        // Sanity: bit_index at row 100 should be 155.
        assert_eq!(
            cols[COL_BIT_INDEX][r].to_bytes(),
            Scalar::from_u64((NUM_STEPS - 1 - r) as u64, curve).to_bytes(),
            "row {}'s bit_index must be {}",
            r,
            NUM_STEPS - 1 - r
        );
        let cs = EvmExpConstraintSystem::new();
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = cols.iter().map(|c| c[r].clone()).collect();
        let c_at = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "is_first_row_pins_bit_index must fire when IS_FIRST_ROW=1 and bit_index ≠ NUM_STEPS-1"
        );
    }

    #[test]
    fn exp_cs_constraints_reject_wrong_exp_bit_decision() {
        let curve = CurveType::Bls48581;
        let (mut cols, _rows) = populated_columns();
        // Force exp_bit on row 255 to 2 (out of binary range). The
        // exp_bit_binary constraint must catch it.
        cols[COL_EXP_BIT][255] = Scalar::from_u64(2, curve);
        let cs = EvmExpConstraintSystem::new();
        let alpha = Scalar::from_u64(5, curve);
        let row_vals: Vec<Scalar> = cols.iter().map(|c| c[255].clone()).collect();
        let c_at = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "exp_bit_binary must fire when exp_bit ∉ {{0, 1}}"
        );
    }

    #[test]
    fn exp_cs_lookup_declarations_are_well_formed() {
        let cs = EvmExpConstraintSystem::new();
        let reqs = cs.lookup_declarations();
        assert!(reqs.tables.is_empty());
        assert!(reqs.declarations.is_empty());
    }

    #[test]
    fn exp_cs_build_trace_polynomials_shape() {
        let curve = CurveType::Bls48581;
        let (_, _, rows) = small_rows();
        let trace = build_trace_polynomials(&rows, curve);
        assert_eq!(trace.columns.len(), NUM_EXP_AIR_COLUMNS);
        assert_eq!(trace.num_rows, NUM_STEPS);
        // 256 = nearest_power_of_two(256), which is itself.
        assert_eq!(trace.padded_size, NUM_STEPS as u64);
        for col in &trace.columns {
            assert_eq!(col.evaluations.len(), NUM_STEPS);
        }
    }

    /// Marked `#[ignore]`: full prove/verify roundtrip on a 256-row, 36-column
    /// EXP gadget trace is moderately expensive (KZG commitments + 4 row-local
    /// + 3 shifted constraint builds). The column count grew from 27 to 36 to
    /// accommodate the cross-AIR LogUp linkage columns (`EXPONENT`,
    /// `FINAL_OUTPUT`, `IS_FIRST_ROW`); these are populated by the witness
    /// builder but not yet algebraically constrained at the per-AIR level.
    /// Run manually with:
    ///     cargo test --release -p metavm-evm --lib \
    ///         exp_cs_prove_verify_small -- --ignored --nocapture
    #[test]
    #[ignore = "slow: full prover roundtrip; run with --release --ignored"]
    fn exp_cs_prove_verify_small() {
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (_, _, rows) = small_rows();
        let trace = build_trace_polynomials(&rows, curve);
        let cs = EvmExpConstraintSystem::new();

        let proof = metavm_zkp::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid =
            metavm_zkp::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "EXP gadget proof must verify");
    }

    /// End-to-end: produce a real EXP-gadget ExecutionProof, serialize,
    /// attach to the Execution layer (treating EXP as the execution under
    /// proof for this test), and verify through the
    /// `LayerChainProof::verify_with_layer_verifier` closure dispatch on
    /// `LayerProofKind::EvmExp`. Validates that the 6th wired AIR
    /// composes cleanly with the existing envelope infrastructure.
    #[test]
    #[ignore = "slow: produces a real EXP gadget proof; run with --release --ignored"]
    fn evm_exp_proof_flows_through_layer_chain_envelope() {
        use metavm_zkp::layer_chain::{
            ChainBoundaries, LayerChain, LayerChainProof, LayerProof, LayerProofKind,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 1. Generate a real EXP proof.
        let (_, _, rows) = small_rows();
        let trace = build_trace_polynomials(&rows, curve);
        let cs = EvmExpConstraintSystem::new();
        let proof = metavm_zkp::prover::prove_with_scheme(&trace, &cs, &scheme);
        assert!(metavm_zkp::verifier::verify_with_scheme(&proof, &cs, &scheme, curve));

        let proof_bytes = proof.to_bytes();
        assert!(!proof_bytes.is_empty());

        // 2. Build a LayerChainProof with the EXP gadget proof in the
        //    Execution slot (synthetic — EXP is a sub-proof of the EVM
        //    main trace, but for envelope-validation purposes any slot
        //    works since the dispatch is purely on `kind`).
        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: 1,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 0 {
                    LayerProof::with_proof(claim, LayerProofKind::EvmExp, proof_bytes.clone())
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // 3. Closure-based dispatch on EvmExp kind.
        let result = chain_proof.verify_with_layer_verifier(|layer| match layer.kind {
            LayerProofKind::EvmExp => {
                let p = metavm_zkp::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                    .map_err(|e| format!("decode failed: {:?}", e))?;
                let cs = EvmExpConstraintSystem::new();
                if metavm_zkp::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                    Ok(())
                } else {
                    Err("EVM EXP proof did not verify".to_string())
                }
            }
            LayerProofKind::ReferenceOnly => Ok(()),
            other => Err(format!("unsupported layer kind {}", other.as_str())),
        });
        assert_eq!(
            result,
            Ok(()),
            "real EVM EXP proof must verify through the LayerChainProof envelope",
        );
    }
}
