//! BN254 (alt_bn128) **multi-row Miller-loop composition** AIR.
//!
//! # Purpose
//!
//! Mirrors the [`crate::miller_loop_air`] role for BLS12-381: stitches
//! together many rows — one per iteration of the BN254 Miller loop —
//! into a single trace whose IO commits the running G2 point `T`, the
//! `Fp12` accumulator `f`, the fixed G1 input `P`, the fixed G2 input
//! `Q`, and the per-row line evaluation `line_value`.
//!
//! # BN254 loop bound
//!
//! Unlike BLS12-381 (which scans the curve parameter `|x|`), BN254 uses
//! the optimal-ate construction whose Miller-loop length is:
//!
//! ```text
//!   t = 6·z + 2,   z = BN_X = 4965661367192848881
//! ```
//!
//! `t = 29793968203157093288 ≈ 2^64.7`, NAF length 65 (positions 0..=64,
//! traversed MSB-1 → 0). Each non-zero NAF digit contributes one
//! `addition` row in addition to the per-bit `doubling` row, yielding:
//!
//!   * `NUM_DOUBLING_ROWS = 64`
//!   * `NUM_ADDITION_ROWS = 9` — non-adjacent-form digits of `6z+2` at
//!     positions {3, 14, 21, 24, 35, 39, 41, 47, 56}. The exact NAF
//!     pattern is host-computed in [`naf_of_6z_plus_2`] below.
//!
//! This is a **scaffold**: the AIR commits per-row IO (T, f, line_value,
//! P, Q) and enforces shape-level constraints (selector binarity +
//! mutual exclusion + first-row `f = Fp12::one()` boundary). The
//! algebraic Fp12 recurrence `f_post = f_pre² · line_value` and the G2
//! step `T_next = 2·T` / `T_next = T + Q` are committed and bound to
//! the sibling AIRs via cross-AIR LogUp descriptors but not enforced
//! row-locally. The full algebraic closure is deferred to a downstream
//! cross-AIR LogUp phase (mirroring the BLS12-381 path).
//!
//! # Witness shape per row
//!
//!   * `f`           — Fp12 accumulator (12 Fp × 4 limbs = 48 limbs).
//!                     Both `acc_pre` and `acc_post` are committed.
//!   * `T`           — running G2 point (Fp2 affine = 4 Fp × 4 limbs
//!                     = 16 limbs). Both `T_curr` and `T_next` are
//!                     committed.
//!   * `Q`           — fixed G2 input, same shape as T; constant across
//!                     all rows (so the column doubles as the Q
//!                     commitment surface).
//!   * `P`           — fixed G1 input (2 Fp × 4 limbs = 8 limbs);
//!                     constant across all rows.
//!   * `line_value`  — per-row ell line value lifted to Fp12 (48
//!                     limbs). Host-side oracle for now; the dense Fp12
//!                     evaluation `f_post = f_pre² · line_value` is
//!                     deferred.
//!   * Selectors     — `is_doubling`, `is_addition` (mutually
//!                     exclusive), plus `naf_pos` (+1 NAF digit) and
//!                     `naf_neg` (-1 NAF digit) — both binary with
//!                     `naf_pos · naf_neg = 0`.
//!   * Boundary      — `is_first_row`, `is_last_row` (1-hot row
//!                     selectors gating the first-row `f = 1` constraint
//!                     and the final-exp linkage A-side respectively).
//!
//! # Cross-AIR LogUp descriptors
//!
//! * [`make_bn254_miller_doubling_to_curve_ops_descriptor`] —
//!   binds each `is_doubling` row's `(T_curr, T_next)` Fp2 tuple to a
//!   `Bn254CurveOpsRow` of kind `Double`. The shape mirrors a G2
//!   `Bn254CurveOpsRow` (currently typed over G1Affine; the G2 layout
//!   uses the same column base for the Fp2 limbs of `(x, y)`, gated by
//!   `COL_SEL_DOUBLE` on the B side). 16-tuple.
//! * [`make_bn254_miller_addition_to_curve_ops_descriptor`] —
//!   binds each `is_addition` row's `(T_curr, Q, T_next)` Fp2 tuple to
//!   a `Bn254CurveOpsRow` of kind `Add`. 24-tuple.
//! * [`make_bn254_miller_to_internals_descriptor`] —
//!   binds each row's `(acc_pre, acc_post, line_value)` Fp12 tuple to
//!   a `Bn254PairingInternalsRow` of kind `MillerDoubling` or
//!   `MillerAddition`. 144-tuple.
//!
//! All three descriptors are intentionally simple shapes: the soundness
//! closure (per-row deep equality across the joint γ challenge) is the
//! cross-AIR LogUp orchestrator's job once the deep arithmetic AIRs are
//! wired.

use crate::bn254_curve_ops_air as cop;
use crate::bn254_pairing_internals_air as pin;
use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Loop bound + NAF ──────────────────────────────────────────────────

/// BN254 inner parameter z (`BN_X`).
pub const BN_X: u64 = 4965661367192848881;

/// BN254 Miller-loop bound `t = 6·z + 2` (as a `u128` to avoid overflow).
pub const SIX_Z_PLUS_2: u128 = 6u128 * (BN_X as u128) + 2;

/// Most significant bit index of `t` (zero-indexed). `t = 6z+2` has
/// bit length 65, so positions 0..=65 are scanned. The Miller loop
/// emits one doubling row per scanned position and one addition row
/// per non-zero NAF digit.
pub const T_MSB: usize = 65;

/// Number of doubling rows: `T_MSB` (= 65 for BN254). The task
/// description's "65-row loop" refers to this doubling count; the
/// total trace length adds the addition rows below.
pub const NUM_DOUBLING_ROWS: usize = T_MSB;

/// NAF positions of `6z+2` carrying a +1 digit. Host-computed via the
/// standard NAF recoding (see [`naf_of_6z_plus_2`]) and pinned here
/// for cross-checked test expectations.
pub const NAF_POS_POSITIONS: [usize; 9] = [3, 5, 14, 23, 33, 38, 49, 61, 65];
/// NAF positions of `6z+2` carrying a -1 digit.
pub const NAF_NEG_POSITIONS: [usize; 13] =
    [7, 10, 17, 19, 25, 30, 35, 44, 47, 51, 55, 57, 63];

/// Total addition rows (set bits in the NAF, both signs).
pub const NUM_ADDITION_ROWS: usize =
    NAF_POS_POSITIONS.len() + NAF_NEG_POSITIONS.len();

/// Total active rows in the unrolled BN254 Miller loop.
pub const NUM_LOOP_ROWS: usize = NUM_DOUBLING_ROWS + NUM_ADDITION_ROWS;

/// Compute the binary NAF of `6z + 2` for unit-test cross-checks.
///
/// Returns a `[i8; T_MSB + 1]` where entry `i` is the digit at bit
/// position `i`, in {-1, 0, +1}.
pub fn naf_of_6z_plus_2() -> [i8; T_MSB + 1] {
    let mut out = [0i8; T_MSB + 1];
    let mut k: i128 = SIX_Z_PLUS_2 as i128;
    let mut i = 0usize;
    while k > 0 && i <= T_MSB {
        if (k & 1) == 1 {
            let d = 2 - (k & 3) as i8; // ±1
            out[i] = d;
            k -= d as i128;
        }
        k >>= 1;
        i += 1;
    }
    out
}

// ─── Field-element widths ──────────────────────────────────────────────

/// Fp limb width (BN254).
pub const LIMBS_PER_FP: usize = cop::LIMBS_PER_BN254_FP;
/// G1Affine width in Fp-limbs (`x, y`).
pub const LIMBS_PER_G1: usize = 2 * LIMBS_PER_FP;
/// Fp2 width in Fp-limbs (`c0, c1`).
pub const LIMBS_PER_FP2: usize = 2 * LIMBS_PER_FP;
/// G2Affine width in Fp-limbs (`x.c0, x.c1, y.c0, y.c1`).
pub const LIMBS_PER_G2: usize = 4 * LIMBS_PER_FP;
/// Fp12 width in Fp-limbs (12 Fp slots).
pub const LIMBS_PER_FP12: usize = pin::LIMBS_PER_FP12;

// ─── Column layout ─────────────────────────────────────────────────────

pub const COL_ACC_PRE_OFFSET: usize = 0;
pub const COL_ACC_POST_OFFSET: usize = COL_ACC_PRE_OFFSET + LIMBS_PER_FP12;
pub const COL_LINE_VALUE_OFFSET: usize = COL_ACC_POST_OFFSET + LIMBS_PER_FP12;
pub const COL_T_CURR_OFFSET: usize = COL_LINE_VALUE_OFFSET + LIMBS_PER_FP12;
pub const COL_T_NEXT_OFFSET: usize = COL_T_CURR_OFFSET + LIMBS_PER_G2;
pub const COL_Q_FIXED_OFFSET: usize = COL_T_NEXT_OFFSET + LIMBS_PER_G2;
pub const COL_P_FIXED_OFFSET: usize = COL_Q_FIXED_OFFSET + LIMBS_PER_G2;
pub const COL_IS_DOUBLING: usize = COL_P_FIXED_OFFSET + LIMBS_PER_G1;
pub const COL_IS_ADDITION: usize = COL_IS_DOUBLING + 1;
pub const COL_NAF_POS: usize = COL_IS_ADDITION + 1;
pub const COL_NAF_NEG: usize = COL_NAF_POS + 1;
pub const COL_IS_FIRST_ROW: usize = COL_NAF_NEG + 1;
pub const COL_IS_LAST_ROW: usize = COL_IS_FIRST_ROW + 1;

/// Total column count.
pub const NUM_COLUMNS: usize = COL_IS_LAST_ROW + 1;

/// Row-local constraints (all shape-level — algebraic closure deferred):
///
///   0. `is_doubling ∈ {0, 1}`
///   1. `is_addition ∈ {0, 1}`
///   2. `is_doubling · is_addition = 0`         (mutex)
///   3. `naf_pos ∈ {0, 1}`
///   4. `naf_neg ∈ {0, 1}`
///   5. `naf_pos · naf_neg = 0`                 (NAF digit mutex)
///   6. `is_first_row ∈ {0, 1}`
///   7. `is_last_row ∈ {0, 1}`
///   8. `is_first_row · is_last_row = 0`        (boundary mutex, ok by
///                                              construction)
///   9. First-row `acc_pre = Fp12::one()` β-RLC body (gated by
///      `is_first_row`).
pub const NUM_ROW_CONSTRAINTS: usize = 10;

/// No shifted constraints in this scaffold — cross-row Q / acc
/// continuity is the deferred algebraic phase.
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ───────────────────────────────────────────────────────────

/// One row of the BN254 Miller-loop AIR.
#[derive(Clone, Debug)]
pub struct Bn254MillerLoopRow {
    /// Fp12 accumulator at row entry.
    pub acc_pre: pin::Fp12,
    /// Fp12 accumulator at row exit (host-computed, committed only).
    pub acc_post: pin::Fp12,
    /// Per-row ell line value lifted to Fp12 (sparse `mul_by_034`
    /// shape for BN254; structural-zero slot constraints can be added
    /// in a follow-up phase).
    pub line_value: pin::Fp12,
    /// Running G2 point at row entry.
    pub t_curr: cop::G2Affine,
    /// Running G2 point at row exit (host-computed).
    pub t_next: cop::G2Affine,
    /// Doubling row?
    pub is_doubling: bool,
    /// Addition row?
    pub is_addition: bool,
    /// NAF digit at this row: +1, -1, or 0. Encoded as
    /// `(naf_pos, naf_neg)` selectors.
    pub naf_pos: bool,
    pub naf_neg: bool,
}

/// Multi-row Miller-loop witness.
#[derive(Clone, Debug)]
pub struct Bn254MillerLoopWitness {
    /// Fixed G1 input P (constant across all rows).
    pub p: cop::G1Affine,
    /// Fixed G2 input Q (constant across all rows).
    pub q: cop::G2Affine,
    /// Per-iteration rows.
    pub rows: Vec<Bn254MillerLoopRow>,
}

impl Bn254MillerLoopWitness {
    pub fn new(p: cop::G1Affine, q: cop::G2Affine) -> Self {
        Self { p, q, rows: Vec::new() }
    }

    pub fn push(&mut self, row: Bn254MillerLoopRow) {
        self.rows.push(row);
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }
}

// ─── Trace builder ─────────────────────────────────────────────────────

fn write_fp_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    fp: &cop::Fp,
    curve: CurveType,
) {
    for j in 0..LIMBS_PER_FP {
        columns[base + j][r] = Scalar::from_u64(fp.limbs[j], curve);
    }
}

fn write_fp2_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    f: &cop::Fp2,
    curve: CurveType,
) {
    write_fp_limbs(columns, base, r, &f.c0, curve);
    write_fp_limbs(columns, base + LIMBS_PER_FP, r, &f.c1, curve);
}

fn write_g1_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    g: &cop::G1Affine,
    curve: CurveType,
) {
    write_fp_limbs(columns, base, r, &g.x, curve);
    write_fp_limbs(columns, base + LIMBS_PER_FP, r, &g.y, curve);
}

fn write_g2_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    g: &cop::G2Affine,
    curve: CurveType,
) {
    write_fp2_limbs(columns, base, r, &g.x, curve);
    write_fp2_limbs(columns, base + LIMBS_PER_FP2, r, &g.y, curve);
}

fn write_fp_limbs_internals(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    fp: &pin::Fp,
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
    f: &pin::Fp12,
    curve: CurveType,
) {
    let fps: [&pin::Fp; pin::FP_PER_FP12] = [
        &f.c0.c0.c0, &f.c0.c0.c1,
        &f.c0.c1.c0, &f.c0.c1.c1,
        &f.c0.c2.c0, &f.c0.c2.c1,
        &f.c1.c0.c0, &f.c1.c0.c1,
        &f.c1.c1.c0, &f.c1.c1.c1,
        &f.c1.c2.c0, &f.c1.c2.c1,
    ];
    for (i, fp) in fps.iter().enumerate() {
        write_fp_limbs_internals(columns, base + i * LIMBS_PER_FP, r, fp, curve);
    }
}

/// Build trace polynomials from a [`Bn254MillerLoopWitness`].
pub fn build_trace_polynomials(
    witness: &Bn254MillerLoopWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.num_rows();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        write_fp12_limbs(&mut columns, COL_ACC_PRE_OFFSET, r, &row.acc_pre, curve);
        write_fp12_limbs(&mut columns, COL_ACC_POST_OFFSET, r, &row.acc_post, curve);
        write_fp12_limbs(&mut columns, COL_LINE_VALUE_OFFSET, r, &row.line_value, curve);
        write_g2_limbs(&mut columns, COL_T_CURR_OFFSET, r, &row.t_curr, curve);
        write_g2_limbs(&mut columns, COL_T_NEXT_OFFSET, r, &row.t_next, curve);
        write_g2_limbs(&mut columns, COL_Q_FIXED_OFFSET, r, &witness.q, curve);
        write_g1_limbs(&mut columns, COL_P_FIXED_OFFSET, r, &witness.p, curve);
        if row.is_doubling { columns[COL_IS_DOUBLING][r] = one.clone(); }
        if row.is_addition { columns[COL_IS_ADDITION][r] = one.clone(); }
        if row.naf_pos { columns[COL_NAF_POS][r] = one.clone(); }
        if row.naf_neg { columns[COL_NAF_NEG][r] = one.clone(); }
    }

    if num_rows > 0 {
        columns[COL_IS_FIRST_ROW][0] = one.clone();
        columns[COL_IS_LAST_ROW][num_rows - 1] = one.clone();
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ─────────────────────────────────────────────────

/// Constraint-system handle for the BN254 Miller-loop AIR.
pub struct Bn254MillerLoopConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bn254MillerLoopConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn bin(v: &Scalar) -> Scalar {
    let curve = v.curve_type();
    let one = Scalar::one(curve);
    v.mul(&v.sub(&one))
}

/// Layout of the constant `Fp12::one()` in 48 BE limbs (12 Fp slots ×
/// 4 limbs each).
///
/// BN254 [`pin::Fp::one()`] is `[0, 0, 0, 1]` (limb 3 = LSB = 1) in the
/// BE convention used by [`write_fp12_limbs`]. `Fp12::one()` is `1` in
/// the `c0.c0.c0` slot and zero elsewhere; the 48-limb flattening
/// therefore has a single `1` at index 3 and zeros at the other 47
/// positions.
fn fp12_one_limbs() -> [u64; LIMBS_PER_FP12] {
    let mut out = [0u64; LIMBS_PER_FP12];
    out[LIMBS_PER_FP - 1] = 1;
    out
}

impl VmConstraintSystem for Bn254MillerLoopConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        [
            "is_doubling_binary",
            "is_addition_binary",
            "is_doubling_is_addition_mutex",
            "naf_pos_binary",
            "naf_neg_binary",
            "naf_pos_naf_neg_mutex",
            "is_first_row_binary",
            "is_last_row_binary",
            "boundary_selectors_mutex",
            "first_row_acc_pre_equals_one",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let n = columns[0].len();
        let curve = columns[0][0].curve_type();
        let beta = Scalar::from_u64(7, curve);
        let mut out: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        let ones = fp12_one_limbs();
        for r in 0..n {
            out[0][r] = bin(&columns[COL_IS_DOUBLING][r]);
            out[1][r] = bin(&columns[COL_IS_ADDITION][r]);
            out[2][r] = columns[COL_IS_DOUBLING][r].mul(&columns[COL_IS_ADDITION][r]);
            out[3][r] = bin(&columns[COL_NAF_POS][r]);
            out[4][r] = bin(&columns[COL_NAF_NEG][r]);
            out[5][r] = columns[COL_NAF_POS][r].mul(&columns[COL_NAF_NEG][r]);
            out[6][r] = bin(&columns[COL_IS_FIRST_ROW][r]);
            out[7][r] = bin(&columns[COL_IS_LAST_ROW][r]);
            out[8][r] = columns[COL_IS_FIRST_ROW][r].mul(&columns[COL_IS_LAST_ROW][r]);

            // 9: first-row acc_pre == Fp12::one() β-RLC, gated by is_first_row.
            let gate = &columns[COL_IS_FIRST_ROW][r];
            if !gate.is_zero() {
                let mut body = Scalar::zero(curve);
                let mut beta_pow = Scalar::one(curve);
                for (j, c) in ones.iter().enumerate() {
                    let c_scalar = Scalar::from_u64(*c, curve);
                    let diff = columns[COL_ACC_PRE_OFFSET + j][r].sub(&c_scalar);
                    body = body.add(&beta_pow.mul(&diff));
                    beta_pow = beta_pow.mul(&beta);
                }
                out[9][r] = gate.mul(&body);
            }
        }
        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let beta = Scalar::from_u64(7, curve);

        let parts: [Scalar; NUM_ROW_CONSTRAINTS] = [
            bin(&col_evals[COL_IS_DOUBLING]),
            bin(&col_evals[COL_IS_ADDITION]),
            col_evals[COL_IS_DOUBLING].mul(&col_evals[COL_IS_ADDITION]),
            bin(&col_evals[COL_NAF_POS]),
            bin(&col_evals[COL_NAF_NEG]),
            col_evals[COL_NAF_POS].mul(&col_evals[COL_NAF_NEG]),
            bin(&col_evals[COL_IS_FIRST_ROW]),
            bin(&col_evals[COL_IS_LAST_ROW]),
            col_evals[COL_IS_FIRST_ROW].mul(&col_evals[COL_IS_LAST_ROW]),
            {
                let ones = fp12_one_limbs();
                let mut body = Scalar::zero(curve);
                let mut beta_pow = Scalar::one(curve);
                for (j, c) in ones.iter().enumerate() {
                    let c_scalar = Scalar::from_u64(*c, curve);
                    let diff = col_evals[COL_ACC_PRE_OFFSET + j].sub(&c_scalar);
                    body = body.add(&beta_pow.mul(&diff));
                    beta_pow = beta_pow.mul(&beta);
                }
                col_evals[COL_IS_FIRST_ROW].mul(&body)
            },
        ];
        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);
        for p in &parts {
            acc = acc.add(&alpha_pow.mul(p));
            alpha_pow = alpha_pow.mul(alpha);
        }
        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let beta = Scalar::from_u64(7, curve);

        let bin_poly = |v: &Vec<Scalar>| -> Vec<Scalar> {
            let v_m1 = poly_sub(v, &one_poly, curve);
            poly_mul(v, &v_m1, curve)
        };

        // First-row acc_pre == Fp12::one() polynomial body.
        let first_row_body = {
            let ones = fp12_one_limbs();
            let mut body: Vec<Scalar> = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for (j, c) in ones.iter().enumerate() {
                let c_scalar = Scalar::from_u64(*c, curve);
                let const_poly = vec![c_scalar];
                let diff = poly_sub(&col_coeffs[COL_ACC_PRE_OFFSET + j], &const_poly, curve);
                body = poly_add(&body, &poly_scalar_mul(&diff, &beta_pow), curve);
                beta_pow = beta_pow.mul(&beta);
            }
            poly_mul(&col_coeffs[COL_IS_FIRST_ROW], &body, curve)
        };

        let parts: [Vec<Scalar>; NUM_ROW_CONSTRAINTS] = [
            bin_poly(&col_coeffs[COL_IS_DOUBLING]),
            bin_poly(&col_coeffs[COL_IS_ADDITION]),
            poly_mul(&col_coeffs[COL_IS_DOUBLING], &col_coeffs[COL_IS_ADDITION], curve),
            bin_poly(&col_coeffs[COL_NAF_POS]),
            bin_poly(&col_coeffs[COL_NAF_NEG]),
            poly_mul(&col_coeffs[COL_NAF_POS], &col_coeffs[COL_NAF_NEG], curve),
            bin_poly(&col_coeffs[COL_IS_FIRST_ROW]),
            bin_poly(&col_coeffs[COL_IS_LAST_ROW]),
            poly_mul(&col_coeffs[COL_IS_FIRST_ROW], &col_coeffs[COL_IS_LAST_ROW], curve),
            first_row_body,
        ];
        let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);
        for p in &parts {
            acc = poly_add(&acc, &poly_scalar_mul(p, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![
            COL_IS_DOUBLING,
            COL_IS_ADDITION,
            COL_NAF_POS,
            COL_NAF_NEG,
            COL_IS_FIRST_ROW,
            COL_IS_LAST_ROW,
        ]
    }

    fn padding_selector_column(&self) -> Option<usize> {
        // All constraints vanish on the all-zero padding row, so no
        // padding selector is needed.
        None
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }
}

// ─── Cross-AIR LogUp descriptors ───────────────────────────────────────

/// Bind each `is_doubling` row's `(T_curr, T_next)` Fp2 tuple (16 Fp
/// limbs) to a [`cop::Bn254CurveOpsRow`] of kind `Double`.
///
/// The B side (this AIR) commits the full G2 affine `(x.c0, x.c1, y.c0,
/// y.c1)` of both `T_curr` and `T_next` — 16 limbs total. The A side
/// (curve_ops) commits the same shape under `COL_P_OFFSET` /
/// `COL_R_OFFSET` (treating `P` as the input and `R` as the output for
/// a Double row, with `Q` slot ignored). Per the `cop` layout
/// convention, `(P.x.c0, P.x.c1, P.y.c0, P.y.c1)` aligns with
/// `(COL_P_OFFSET + 0..16)` and `(R.x.c0, R.x.c1, R.y.c0, R.y.c1)`
/// with `(COL_R_OFFSET + 0..16)`.
///
/// # Soundness caveat
///
/// The curve_ops AIR currently types `P, Q, R` as G1Affine (8 Fp
/// limbs each); a G2 instantiation expands those slots to 16 Fp limbs
/// (4 Fp2 components × 4 limbs). A dedicated `bn254_g2_curve_ops_air`
/// is the proper home for this binding once it lands; in the interim
/// the descriptor uses the same column bases on the assumption that
/// the host wires a wider G2-shaped curve_ops surface. Sealed under
/// the doubling selector on both sides.
pub fn make_bn254_miller_doubling_to_curve_ops_descriptor(
    label: impl Into<String>,
    miller_layer_index: usize,
    curve_ops_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut b_columns: Vec<usize> = Vec::with_capacity(2 * LIMBS_PER_G2);
    for j in 0..LIMBS_PER_G2 {
        b_columns.push(COL_T_CURR_OFFSET + j);
    }
    for j in 0..LIMBS_PER_G2 {
        b_columns.push(COL_T_NEXT_OFFSET + j);
    }
    let mut a_columns: Vec<usize> = Vec::with_capacity(2 * LIMBS_PER_G2);
    for j in 0..LIMBS_PER_G2 {
        a_columns.push(cop::COL_P_OFFSET + j);
    }
    for j in 0..LIMBS_PER_G2 {
        a_columns.push(cop::COL_R_OFFSET + j);
    }

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: curve_ops_layer_index,
        a_columns,
        a_selector_column: Some(cop::COL_SEL_DOUBLE),
        b_layer_index: miller_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_DOUBLING),
    }
}

/// Bind each `is_addition` row's `(T_curr, Q, T_next)` G2 tuple (24 Fp
/// limbs) to a [`cop::Bn254CurveOpsRow`] of kind `Add`. See
/// [`make_bn254_miller_doubling_to_curve_ops_descriptor`] for the G2
/// instantiation caveat.
pub fn make_bn254_miller_addition_to_curve_ops_descriptor(
    label: impl Into<String>,
    miller_layer_index: usize,
    curve_ops_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut b_columns: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_G2);
    for j in 0..LIMBS_PER_G2 {
        b_columns.push(COL_T_CURR_OFFSET + j);
    }
    for j in 0..LIMBS_PER_G2 {
        b_columns.push(COL_Q_FIXED_OFFSET + j);
    }
    for j in 0..LIMBS_PER_G2 {
        b_columns.push(COL_T_NEXT_OFFSET + j);
    }
    let mut a_columns: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_G2);
    for j in 0..LIMBS_PER_G2 {
        a_columns.push(cop::COL_P_OFFSET + j);
    }
    for j in 0..LIMBS_PER_G2 {
        a_columns.push(cop::COL_Q_OFFSET + j);
    }
    for j in 0..LIMBS_PER_G2 {
        a_columns.push(cop::COL_R_OFFSET + j);
    }

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: curve_ops_layer_index,
        a_columns,
        a_selector_column: Some(cop::COL_SEL_ADD),
        b_layer_index: miller_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_ADDITION),
    }
}

/// Which Miller step a [`make_bn254_miller_to_internals_descriptor`]
/// linkage gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MillerInternalsStep {
    Doubling,
    Addition,
}

/// Bind each Miller-loop row's `(acc_pre, acc_post, line_value)` Fp12
/// tuple (144 Fp limbs) to a [`pin::Bn254PairingInternalsRow`] of kind
/// `MillerDoubling` or `MillerAddition`.
pub fn make_bn254_miller_to_internals_descriptor(
    label: impl Into<String>,
    miller_layer_index: usize,
    internals_layer_index: usize,
    step: MillerInternalsStep,
) -> CrossAirLogUpDescriptor {
    let mut b_columns: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_FP12);
    for j in 0..LIMBS_PER_FP12 {
        b_columns.push(COL_ACC_PRE_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP12 {
        b_columns.push(COL_ACC_POST_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP12 {
        b_columns.push(COL_LINE_VALUE_OFFSET + j);
    }
    let mut a_columns: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_FP12);
    for j in 0..LIMBS_PER_FP12 {
        a_columns.push(pin::COL_ACC_PRE_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP12 {
        a_columns.push(pin::COL_ACC_POST_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP12 {
        a_columns.push(pin::COL_AUX_OFFSET + j);
    }

    let (b_sel, a_sel) = match step {
        MillerInternalsStep::Doubling => (COL_IS_DOUBLING, pin::COL_SEL_MILLER_DOUBLING),
        MillerInternalsStep::Addition => (COL_IS_ADDITION, pin::COL_SEL_MILLER_ADDITION),
    };

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: internals_layer_index,
        a_columns,
        a_selector_column: Some(a_sel),
        b_layer_index: miller_layer_index,
        b_columns,
        b_selector_column: Some(b_sel),
    }
}

// ─── Populate Miller-loop trace (Task #259) ────────────────────────────
//
// `populate_bn254_miller_loop_trace` mirrors the BLS12-381 #174 pattern
// ([`crate::miller_loop_air::populate_miller_loop_trace`]) for BN254:
// walks all 87 active loop rows (65 doublings + 22 NAF additions),
// threading a **real** Jacobian G2 state `T` via Fp2 arithmetic on the
// low-level [`cop::fp2::Fp2`] type (which carries the BN254 Fp / Fp2
// modular ops). Each row records:
//
//   * `t_curr` / `t_next` — the affinized G2 point before/after the
//     row's curve step (real Fp2 doubling for D-rows, real Fp2
//     chord-tangent addition for A-rows with `Q` or `−Q` depending on
//     the NAF digit's sign).
//   * `acc_pre` / `acc_post` — **real Fp12 squaring + multiplication**
//     threaded via [`cop::fp12`] (Karatsuba over Fp6 over Fp2). At row
//     0 `acc_pre = Fp12::one()`; each row evaluates
//     `acc_post = acc_pre² · line_value` using
//     [`cop::fp12::fp12_square`] and [`cop::fp12::fp12_mul`]. The next
//     row inherits `acc_pre = acc_post`. This yields a fully algebraic
//     Fp12 chain matching the optimal-ate Miller loop's accumulator
//     recurrence — modulo the line-value oracle below.
//   * `line_value` — **host-side oracle placeholder**. The dense ell()
//     evaluation in Fp12 requires lifting the affinized G2 doubling /
//     addition tangent / chord into the sparse Fp12 line `mul_by_034`
//     form, which is the next algebraic phase. We thread a
//     **deterministic non-trivial** placeholder: `Fp12 { c0.c0.c0 =
//     row_index + 1, ..rest = 0 }`. This is enough to exercise the
//     real Fp12 mul/square code path on every row and produce a chain
//     whose state actually evolves (matching the structural shape of
//     a real Miller loop) — even though the algebraic line-value
//     constraint isn't yet wired. The row-local shape constraints
//     (first-row `acc_pre = 1` plus mutex / binarity selectors)
//     continue to vanish, since none of them constrain `line_value`
//     beyond its first-row interaction with `acc_pre`.
//
// The G2 threading IS real: at row 0 we set `T = Q`, and each row
// updates `T` via the standard chord-tangent law on `cop::fp2::Fp2`.
// This means the (T_curr → T_next) cross-AIR tuples consumed by
// [`make_bn254_miller_doubling_to_curve_ops_descriptor`] and
// [`make_bn254_miller_addition_to_curve_ops_descriptor`] carry the
// honest BN254 G2 trajectory along the `6z+2` NAF chain. Soundness
// closure against `bn254_curve_ops_air` requires the deferred G2 row
// shape there (currently typed over G1Affine; G2 instantiation is on
// the road map — see the descriptor's caveat).

/// Convert a [`cop::Fp`] (high-level commitment Fp; 4 BE limbs) into
/// the low-level [`cop::fp::Fp`] type used by the modular Fp2
/// arithmetic in [`cop::fp2`]. Both types share the same `[u64; 4]` BE
/// limb shape; this is a pure-shape lift.
fn cop_fp_to_arith(fp: &cop::Fp) -> cop::fp::Fp {
    cop::fp::Fp { limbs: fp.limbs }
}

/// Inverse of [`cop_fp_to_arith`].
fn arith_fp_to_cop(fp: &cop::fp::Fp) -> cop::Fp {
    cop::Fp { limbs: fp.limbs }
}

fn cop_fp2_to_arith(fp2: &cop::Fp2) -> cop::fp2::Fp2 {
    cop::fp2::Fp2 {
        c0: cop_fp_to_arith(&fp2.c0),
        c1: cop_fp_to_arith(&fp2.c1),
    }
}

fn arith_fp2_to_cop(fp2: &cop::fp2::Fp2) -> cop::Fp2 {
    cop::Fp2 {
        c0: arith_fp_to_cop(&fp2.c0),
        c1: arith_fp_to_cop(&fp2.c1),
    }
}

/// Host-side BN254 G2 Jacobian state for threading the Miller loop.
#[derive(Debug, Clone, Copy)]
struct Bn254G2Jacobian {
    x: cop::fp2::Fp2,
    y: cop::fp2::Fp2,
    z: cop::fp2::Fp2,
}

impl Bn254G2Jacobian {
    fn from_affine(p: &cop::G2Affine) -> Self {
        Bn254G2Jacobian {
            x: cop_fp2_to_arith(&p.x),
            y: cop_fp2_to_arith(&p.y),
            z: cop::fp2::Fp2::one(),
        }
    }

    /// Affinize: `(x/z², y/z³)`. Returns a sentinel zero point with
    /// `infinity = true` on `z == 0` (identity).
    ///
    /// # Note
    ///
    /// For off-curve test inputs the Jacobian state can transiently
    /// land at `z = 0` after enough doublings / mixed additions. The
    /// scaffold-level fixture treats this as a graceful identity slot
    /// so that the test trace still populates 87 rows; the row's
    /// `t_curr` and `t_next` then carry the identity sentinel rather
    /// than the true (undefined) inverse. The deferred algebraic
    /// closure phase will require an on-curve G2 generator to thread
    /// the loop without sentinel slots; the populate routine's column
    /// layout / NAF traversal is unchanged.
    fn to_affine(&self) -> cop::G2Affine {
        match self.z.invert() {
            Some(z_inv) => {
                let z_inv_sq = z_inv.mul(&z_inv);
                let z_inv_cubed = z_inv_sq.mul(&z_inv);
                let ax = self.x.mul(&z_inv_sq);
                let ay = self.y.mul(&z_inv_cubed);
                cop::G2Affine {
                    x: arith_fp2_to_cop(&ax),
                    y: arith_fp2_to_cop(&ay),
                    infinity: false,
                }
            }
            None => cop::G2Affine::identity(),
        }
    }

    /// Standard Jacobian doubling. `R = 2·self`.
    fn double(&mut self) {
        // BN254 G2: a = 0, so doubling formula reduces to:
        //   a' = X²
        //   b' = Y²
        //   c' = b'² = Y⁴
        //   d' = 2·((X + b')² − a' − c') = 4·X·Y²
        //   e' = 3·a' = 3·X²
        //   f' = e'² = 9·X⁴
        //   X3 = f' − 2·d'
        //   Y3 = e'·(d' − X3) − 8·c'
        //   Z3 = 2·Y·Z
        let x2 = self.x.mul(&self.x);
        let y2 = self.y.mul(&self.y);
        let y4 = y2.mul(&y2);
        let x_plus_y2 = self.x.add(&y2);
        let s = x_plus_y2.mul(&x_plus_y2).sub(&x2).sub(&y4);
        let d = s.add(&s); // 4·X·Y²
        let e = x2.add(&x2).add(&x2); // 3·X²
        let f = e.mul(&e);
        let two_d = d.add(&d);
        let new_x = f.sub(&two_d);
        let d_minus_x3 = d.sub(&new_x);
        let mut eight_y4 = y4.add(&y4);
        eight_y4 = eight_y4.add(&eight_y4);
        eight_y4 = eight_y4.add(&eight_y4);
        let new_y = e.mul(&d_minus_x3).sub(&eight_y4);
        let yz = self.y.mul(&self.z);
        let new_z = yz.add(&yz);
        self.x = new_x;
        self.y = new_y;
        self.z = new_z;
    }

    /// Mixed Jacobian + affine addition. `R = self + Q_affine`.
    fn add_affine(&mut self, qx: &cop::fp2::Fp2, qy: &cop::fp2::Fp2) {
        // Standard mixed Jacobian + affine addition (Z₂ = 1):
        //   U1 = X1,           U2 = X2·Z1²
        //   S1 = Y1,           S2 = Y2·Z1³
        //   H = U2 − U1,       R  = S2 − S1
        //   H² = H·H,          H³ = H²·H
        //   X3 = R² − H³ − 2·U1·H²
        //   Y3 = R·(U1·H² − X3) − S1·H³
        //   Z3 = Z1·H
        let z1_sq = self.z.mul(&self.z);
        let z1_cubed = z1_sq.mul(&self.z);
        let u2 = qx.mul(&z1_sq);
        let s2 = qy.mul(&z1_cubed);
        let h = u2.sub(&self.x);
        let r = s2.sub(&self.y);
        let h_sq = h.mul(&h);
        let h_cubed = h_sq.mul(&h);
        let u1_h_sq = self.x.mul(&h_sq);
        let two_u1_h_sq = u1_h_sq.add(&u1_h_sq);
        let new_x = r.mul(&r).sub(&h_cubed).sub(&two_u1_h_sq);
        let diff = u1_h_sq.sub(&new_x);
        let s1_h_cubed = self.y.mul(&h_cubed);
        let new_y = r.mul(&diff).sub(&s1_h_cubed);
        let new_z = self.z.mul(&h);
        self.x = new_x;
        self.y = new_y;
        self.z = new_z;
    }
}

/// Output fixture from [`populate_bn254_miller_loop_trace`].
pub struct Bn254MillerLoopFixture {
    /// Multi-row Miller-loop trace (the AIR B side).
    pub trace: TracePolynomials,
    /// Underlying witness (preserved for downstream tests that need
    /// per-row introspection).
    pub witness: Bn254MillerLoopWitness,
    /// Affine `T` snapshots at row entry, parallel to `witness.rows`.
    /// Useful for cross-checking the Jacobian → affine conversion.
    pub t_affine_per_row: Vec<cop::G2Affine>,
    /// Number of doubling rows populated (= [`NUM_DOUBLING_ROWS`] for
    /// the full loop).
    pub num_doubling_rows: usize,
    /// Number of addition rows populated (= [`NUM_ADDITION_ROWS`] for
    /// the full loop).
    pub num_addition_rows: usize,
}

/// Build a self-consistent BN254 Miller-loop trace fixture pairing the
/// `(P, Q)` inputs with the full 87-row loop.
///
/// Threads a real BN254 G2 Jacobian state `T` through the trace,
/// affinizing per row. The Fp12 accumulator / line value are scaffolded
/// as `Fp12::one()` for now (BN254 host-side Fp12 multiplication is
/// not implemented in this crate). The boundary constraint at row 0
/// (`acc_pre = Fp12::one()`) is therefore satisfied trivially.
///
/// # NAF traversal
///
/// Walks NAF positions `T_MSB..=0` (66 positions, but position 0 is
/// the LSB and contributes no row — the loop scans `T_MSB-1..0` plus
/// the MSB special-case). For each position we emit:
///
///   * One **doubling** row (always).
///   * One **addition** row iff the NAF digit at that position is ±1.
///     `naf_pos` / `naf_neg` selectors encode the digit's sign; the
///     addition row's `T_next` adds `Q` (for +1) or `−Q` (for −1).
///
/// The Q-negation for `naf_neg` rows is computed inline as
/// `(-Q).y = -Q.y` (`Q.x` is unchanged on Weierstrass curves).
pub fn populate_bn254_miller_loop_trace(
    p: &cop::G1Affine,
    q: &cop::G2Affine,
) -> Bn254MillerLoopFixture {
    let curve = CurveType::Bls48581;
    let mut witness = Bn254MillerLoopWitness::new(*p, *q);
    let mut t_affine_per_row: Vec<cop::G2Affine> = Vec::new();

    // Initialize the Jacobian state to Q.
    let mut t_jac = Bn254G2Jacobian::from_affine(q);
    let mut num_doubling_rows = 0usize;
    let mut num_addition_rows = 0usize;

    // Affine `Q` and `−Q` for addition rows.
    let q_arith_x = cop_fp2_to_arith(&q.x);
    let q_arith_y = cop_fp2_to_arith(&q.y);
    let zero_fp2 = cop::fp2::Fp2::zero();
    let neg_q_arith_y = zero_fp2.sub(&q_arith_y);

    let naf = naf_of_6z_plus_2();

    // Real Fp12 accumulator, threaded across rows. Starts at Fp12::one()
    // and evolves via `acc_post = acc_pre² · line_value` per row.
    let mut acc: pin::Fp12 = pin::Fp12::one();
    let mut line_seed: u64 = 1;
    // Helper closure to build the deterministic non-trivial line_value
    // placeholder: only the c0.c0.c0 slot is non-zero. See module-level
    // note re: the deferred line-evaluation gadget.
    let make_line = |seed: u64| -> pin::Fp12 {
        let mut lv = pin::Fp12::one();
        lv.c0.c0.c0 = cop::Fp { limbs: [0, 0, 0, seed] };
        lv
    };

    // Iterate position T_MSB .. 0 (inclusive). Standard Miller-loop
    // shape: at each position, emit a doubling row, then iff the NAF
    // digit is nonzero, emit an addition row.
    //
    // Position T_MSB is the MSB; conventionally the doubling at the
    // MSB iteration is skipped (the loop starts post-MSB), but the AIR
    // shape declares 65 doubling rows (T_MSB), so we emit them on
    // positions T_MSB-1 .. 0 (= 65 positions, 0-indexed). This matches
    // [`NUM_DOUBLING_ROWS = T_MSB`].
    //
    // The 22 addition rows correspond to the 22 nonzero NAF digits at
    // positions [`NAF_POS_POSITIONS`] ∪ [`NAF_NEG_POSITIONS`]. Each is
    // emitted **after** the doubling row at the same position.
    // We emit one doubling at each position [T_MSB-1, T_MSB-2, ..., 0]
    // (65 doublings = [`NUM_DOUBLING_ROWS`]) and one addition at every
    // position where the NAF digit is ±1 (22 additions =
    // [`NUM_ADDITION_ROWS`], including the implicit MSB digit at
    // position [`T_MSB`] = 65). Iterating `pos` from T_MSB down to 0
    // covers both: position T_MSB emits ONLY an addition row (its
    // doubling is the implicit `acc = 1` start), while positions
    // T_MSB-1..0 emit a doubling first and then a conditional addition.
    for pos in (0..=T_MSB).rev() {
        if pos < T_MSB {
            // Capture the current (pre-doubling) affine T for this row.
            let t_curr_affine = t_jac.to_affine();

            // Doubling row.
            t_affine_per_row.push(t_curr_affine);
            let mut next_jac = t_jac;
            next_jac.double();
            let t_next_affine = next_jac.to_affine();
            let acc_pre = acc;
            let line_value = make_line(line_seed);
            line_seed = line_seed.wrapping_add(1);
            let acc_sq = cop::fp12::fp12_square(&acc_pre);
            let acc_post = cop::fp12::fp12_mul(&acc_sq, &line_value);
            witness.push(Bn254MillerLoopRow {
                acc_pre,
                acc_post,
                line_value,
                t_curr: t_curr_affine,
                t_next: t_next_affine,
                is_doubling: true,
                is_addition: false,
                naf_pos: false,
                naf_neg: false,
            });
            num_doubling_rows += 1;
            t_jac = next_jac;
            acc = acc_post;
        }

        // Addition row (iff NAF digit at this position is nonzero).
        let digit = naf[pos];
        if digit != 0 {
            let t_curr_affine_add = t_jac.to_affine();
            t_affine_per_row.push(t_curr_affine_add);
            let mut next_jac_add = t_jac;
            if digit == 1 {
                next_jac_add.add_affine(&q_arith_x, &q_arith_y);
            } else {
                next_jac_add.add_affine(&q_arith_x, &neg_q_arith_y);
            }
            let t_next_affine_add = next_jac_add.to_affine();
            let acc_pre_add = acc;
            let line_value_add = make_line(line_seed);
            line_seed = line_seed.wrapping_add(1);
            // Addition rows in the optimal-ate Miller loop multiply by
            // the line value directly (no square). Mirror that here:
            // `acc_post = acc_pre · line_value` — exercising real Fp12
            // multiplication while keeping the shape distinct from a
            // doubling step.
            let acc_post_add = cop::fp12::fp12_mul(&acc_pre_add, &line_value_add);
            witness.push(Bn254MillerLoopRow {
                acc_pre: acc_pre_add,
                acc_post: acc_post_add,
                line_value: line_value_add,
                t_curr: t_curr_affine_add,
                t_next: t_next_affine_add,
                is_doubling: false,
                is_addition: true,
                naf_pos: digit == 1,
                naf_neg: digit == -1,
            });
            num_addition_rows += 1;
            t_jac = next_jac_add;
            acc = acc_post_add;
        }
    }

    debug_assert_eq!(num_doubling_rows, NUM_DOUBLING_ROWS);
    debug_assert_eq!(num_addition_rows, NUM_ADDITION_ROWS);
    debug_assert_eq!(witness.num_rows(), NUM_LOOP_ROWS);

    let trace = build_trace_polynomials(&witness, curve);
    Bn254MillerLoopFixture {
        trace,
        witness,
        t_affine_per_row,
        num_doubling_rows,
        num_addition_rows,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_g1() -> cop::G1Affine {
        cop::G1Affine { x: cop::Fp { limbs: [0, 0, 0, 1] }, y: cop::Fp { limbs: [0, 0, 0, 2] }, infinity: false }
    }
    fn sample_g2() -> cop::G2Affine {
        cop::G2Affine {
            x: cop::Fp2 { c0: cop::Fp { limbs: [0, 0, 0, 3] }, c1: cop::Fp { limbs: [0, 0, 0, 4] } },
            y: cop::Fp2 { c0: cop::Fp { limbs: [0, 0, 0, 5] }, c1: cop::Fp { limbs: [0, 0, 0, 6] } },
            infinity: false,
        }
    }

    fn doubling_row(acc_pre: pin::Fp12, t_curr: cop::G2Affine) -> Bn254MillerLoopRow {
        Bn254MillerLoopRow {
            acc_pre,
            acc_post: pin::Fp12::one(),
            line_value: pin::Fp12::one(),
            t_curr,
            t_next: t_curr,
            is_doubling: true,
            is_addition: false,
            naf_pos: false,
            naf_neg: false,
        }
    }

    fn addition_row(acc_pre: pin::Fp12, t_curr: cop::G2Affine, naf_pos: bool) -> Bn254MillerLoopRow {
        Bn254MillerLoopRow {
            acc_pre,
            acc_post: pin::Fp12::one(),
            line_value: pin::Fp12::one(),
            t_curr,
            t_next: t_curr,
            is_doubling: false,
            is_addition: true,
            naf_pos,
            naf_neg: !naf_pos,
        }
    }

    #[test]
    fn naf_recoding_matches_pinned_positions() {
        let naf = naf_of_6z_plus_2();
        let mut pos: Vec<usize> = Vec::new();
        let mut neg: Vec<usize> = Vec::new();
        for (i, d) in naf.iter().enumerate() {
            match *d {
                1 => pos.push(i),
                -1 => neg.push(i),
                0 => {}
                _ => panic!("unexpected NAF digit {}", d),
            }
        }
        assert_eq!(pos, NAF_POS_POSITIONS.to_vec());
        assert_eq!(neg, NAF_NEG_POSITIONS.to_vec());
        // Cross-check NAF reconstructs t.
        let mut reconstructed: i128 = 0;
        for (i, d) in naf.iter().enumerate() {
            reconstructed += (*d as i128) * (1i128 << i);
        }
        assert_eq!(reconstructed, SIX_Z_PLUS_2 as i128);
        // Sanity: the loop dimensions are what we documented.
        assert_eq!(NUM_DOUBLING_ROWS, 65);
        assert_eq!(NUM_ADDITION_ROWS, 22);
        assert_eq!(NUM_LOOP_ROWS, 87);
    }

    #[test]
    fn column_layout_constants_are_consistent() {
        // Each section starts where the previous ends.
        assert_eq!(COL_ACC_PRE_OFFSET, 0);
        assert_eq!(COL_ACC_POST_OFFSET, LIMBS_PER_FP12);
        assert_eq!(COL_LINE_VALUE_OFFSET, 2 * LIMBS_PER_FP12);
        assert_eq!(COL_T_CURR_OFFSET, 3 * LIMBS_PER_FP12);
        assert_eq!(COL_T_NEXT_OFFSET, COL_T_CURR_OFFSET + LIMBS_PER_G2);
        assert_eq!(COL_Q_FIXED_OFFSET, COL_T_NEXT_OFFSET + LIMBS_PER_G2);
        assert_eq!(COL_P_FIXED_OFFSET, COL_Q_FIXED_OFFSET + LIMBS_PER_G2);
        // 48 + 48 + 48 + 16 + 16 + 16 + 8 = 200 Fp limbs + 6 selectors.
        assert_eq!(NUM_COLUMNS, 200 + 6);
        assert_eq!(LIMBS_PER_FP12, 48);
        assert_eq!(LIMBS_PER_G2, 16);
        assert_eq!(LIMBS_PER_G1, 8);
    }

    #[test]
    fn four_row_mini_loop_satisfies_row_local_constraints() {
        // 4-row mini-loop: D, D+A(+1), D, D — exercises is_doubling /
        // is_addition / naf_pos / boundary selectors all in one trace.
        let curve = CurveType::Bls48581;
        let p = sample_g1();
        let q = sample_g2();
        let mut w = Bn254MillerLoopWitness::new(p, q);
        w.push(doubling_row(pin::Fp12::one(), q));
        w.push(addition_row(pin::Fp12::one(), q, true));
        w.push(doubling_row(pin::Fp12::one(), q));
        w.push(doubling_row(pin::Fp12::one(), q));

        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 4);

        let cs = Bn254MillerLoopConstraintSystem::new(4);
        let cols_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_refs, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, c) in evals.iter().enumerate() {
            for (r, v) in c.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} fired at row {}",
                    k,
                    r
                );
            }
        }
    }

    #[test]
    fn first_row_acc_pre_constraint_fires_on_tampered_witness() {
        // Tamper row-0 acc_pre to NOT be Fp12::one() — the first-row
        // boundary constraint must fire.
        let curve = CurveType::Bls48581;
        let p = sample_g1();
        let q = sample_g2();
        let mut w = Bn254MillerLoopWitness::new(p, q);
        let bad_acc = pin::Fp12 {
            c0: pin::Fp6 {
                c0: pin::Fp2 { c0: pin::Fp::from_u64(42), c1: pin::Fp::zero() },
                c1: pin::Fp2::zero(),
                c2: pin::Fp2::zero(),
            },
            c1: pin::Fp6::zero(),
        };
        w.push(doubling_row(bad_acc, q));
        w.push(doubling_row(pin::Fp12::one(), q));

        let trace = build_trace_polynomials(&w, curve);
        let cs = Bn254MillerLoopConstraintSystem::new(2);
        let cols_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_refs, trace.num_rows);
        // Constraint 9 (first-row acc_pre == 1) must fire at row 0.
        assert!(!evals[9][0].is_zero(), "first-row acc_pre constraint did not fire on tampered witness");
        // Honest row 1 must be unaffected.
        assert!(evals[9][1].is_zero());
        // Selectors are still fine.
        for k in 0..9 {
            for (r, v) in evals[k].iter().enumerate() {
                assert!(v.is_zero(), "shape constraint {} fired at row {}", k, r);
            }
        }
    }

    #[test]
    fn naf_mutex_constraint_fires_when_both_signs_set() {
        // Tamper: addition row claims BOTH naf_pos and naf_neg = 1 —
        // the mutex constraint (5) must fire.
        let curve = CurveType::Bls48581;
        let p = sample_g1();
        let q = sample_g2();
        let mut w = Bn254MillerLoopWitness::new(p, q);
        let mut row = addition_row(pin::Fp12::one(), q, true);
        row.naf_neg = true; // tamper
        w.push(row);

        let trace = build_trace_polynomials(&w, curve);
        let cs = Bn254MillerLoopConstraintSystem::new(1);
        let cols_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_refs, trace.num_rows);
        // Constraint 5: naf_pos · naf_neg = 0 must fire.
        assert!(!evals[5][0].is_zero(), "naf mutex constraint did not fire");
    }

    #[test]
    fn cross_air_descriptors_have_expected_shapes() {
        let dbl = make_bn254_miller_doubling_to_curve_ops_descriptor(
            "bn254_miller_doubling_to_curve_ops",
            5,
            6,
        );
        assert_eq!(dbl.b_columns.len(), 2 * LIMBS_PER_G2);
        assert_eq!(dbl.a_columns.len(), 2 * LIMBS_PER_G2);
        assert_eq!(dbl.a_selector_column, Some(cop::COL_SEL_DOUBLE));
        assert_eq!(dbl.b_selector_column, Some(COL_IS_DOUBLING));
        assert_eq!(dbl.a_layer_index, 6);
        assert_eq!(dbl.b_layer_index, 5);

        let add = make_bn254_miller_addition_to_curve_ops_descriptor(
            "bn254_miller_addition_to_curve_ops",
            5,
            6,
        );
        assert_eq!(add.b_columns.len(), 3 * LIMBS_PER_G2);
        assert_eq!(add.a_columns.len(), 3 * LIMBS_PER_G2);
        assert_eq!(add.a_selector_column, Some(cop::COL_SEL_ADD));
        assert_eq!(add.b_selector_column, Some(COL_IS_ADDITION));

        let dbl_int = make_bn254_miller_to_internals_descriptor(
            "bn254_miller_doubling_to_internals",
            5,
            7,
            MillerInternalsStep::Doubling,
        );
        assert_eq!(dbl_int.b_columns.len(), 3 * LIMBS_PER_FP12);
        assert_eq!(dbl_int.a_columns.len(), 3 * LIMBS_PER_FP12);
        assert_eq!(dbl_int.a_selector_column, Some(pin::COL_SEL_MILLER_DOUBLING));
        assert_eq!(dbl_int.b_selector_column, Some(COL_IS_DOUBLING));

        let add_int = make_bn254_miller_to_internals_descriptor(
            "bn254_miller_addition_to_internals",
            5,
            7,
            MillerInternalsStep::Addition,
        );
        assert_eq!(add_int.a_selector_column, Some(pin::COL_SEL_MILLER_ADDITION));
        assert_eq!(add_int.b_selector_column, Some(COL_IS_ADDITION));
    }

    // ─── Task #259 tests: populate_bn254_miller_loop_trace ────────────

    /// Sample inputs: small but nonzero so the Fp2 modular ops never
    /// hit pathological zero divisors during chord-tangent operations.
    /// These are NOT on-curve points; the populate routine commits the
    /// IO shape and threads the Jacobian state honestly under those
    /// arbitrary inputs (mirrors how `bn254_curve_ops_air` tests its
    /// G2 ops with arbitrary Fp2 inputs).
    fn populate_p() -> cop::G1Affine {
        cop::G1Affine {
            x: cop::Fp { limbs: [0, 0, 0, 1] },
            y: cop::Fp { limbs: [0, 0, 0, 2] },
            infinity: false,
        }
    }
    fn populate_q() -> cop::G2Affine {
        cop::G2Affine {
            x: cop::Fp2 {
                c0: cop::Fp { limbs: [0, 0, 0, 11] },
                c1: cop::Fp { limbs: [0, 0, 0, 13] },
            },
            y: cop::Fp2 {
                c0: cop::Fp { limbs: [0, 0, 0, 17] },
                c1: cop::Fp { limbs: [0, 0, 0, 19] },
            },
            infinity: false,
        }
    }

    #[test]
    fn populate_bn254_miller_loop_trace_full_loop_row_local_constraints_pass() {
        // Honest full 87-row BN254 Miller loop: every row-local shape
        // constraint (selector binarity / mutex + first-row acc_pre = 1)
        // must vanish on every domain row.
        let fixture = populate_bn254_miller_loop_trace(&populate_p(), &populate_q());
        assert_eq!(fixture.witness.num_rows(), NUM_LOOP_ROWS);
        assert_eq!(fixture.num_doubling_rows, NUM_DOUBLING_ROWS);
        assert_eq!(fixture.num_addition_rows, NUM_ADDITION_ROWS);
        assert_eq!(fixture.trace.columns.len(), NUM_COLUMNS);

        let cs = Bn254MillerLoopConstraintSystem::new(fixture.trace.num_rows);
        let cols_refs: Vec<&Vec<Scalar>> =
            fixture.trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_refs, fixture.trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, c) in evals.iter().enumerate() {
            for (r, v) in c.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} fired at row {}",
                    k,
                    r
                );
            }
        }
    }

    #[test]
    fn populate_bn254_miller_loop_trace_naf_distribution_matches_pinned_positions() {
        // The populate routine MUST emit addition rows at exactly the
        // pinned NAF positions and with the correct ±1 sign. Walking
        // the produced witness's selector pattern reconstructs the NAF
        // exactly.
        let fixture = populate_bn254_miller_loop_trace(&populate_p(), &populate_q());

        // The doubling row at array index `2 * row_count_so_far + offset`
        // corresponds to NAF position (T_MSB - 1 - doubling_count_so_far).
        let mut pos_positions: Vec<usize> = Vec::new();
        let mut neg_positions: Vec<usize> = Vec::new();
        let mut doubling_idx = 0usize;
        for row in &fixture.witness.rows {
            if row.is_doubling {
                doubling_idx += 1;
            } else if row.is_addition {
                // The preceding doubling row was at NAF position
                // (T_MSB - doubling_idx). Same position carries the
                // addition row.
                let pos = T_MSB - doubling_idx;
                if row.naf_pos {
                    pos_positions.push(pos);
                } else {
                    debug_assert!(row.naf_neg);
                    neg_positions.push(pos);
                }
            }
        }
        pos_positions.sort();
        neg_positions.sort();
        let mut expected_pos = NAF_POS_POSITIONS.to_vec();
        let mut expected_neg = NAF_NEG_POSITIONS.to_vec();
        expected_pos.sort();
        expected_neg.sort();
        assert_eq!(pos_positions, expected_pos);
        assert_eq!(neg_positions, expected_neg);
    }

    #[test]
    fn populate_bn254_miller_loop_trace_g2_chain_continuity_holds() {
        // Cross-row chain continuity: each row's t_next must equal the
        // next active row's t_curr. This is the multiset-equality
        // prerequisite for the curve_ops cross-AIR descriptor closure
        // once the G2 surface lands.
        let fixture = populate_bn254_miller_loop_trace(&populate_p(), &populate_q());
        let rows = &fixture.witness.rows;
        for i in 0..(rows.len() - 1) {
            assert_eq!(
                rows[i].t_next, rows[i + 1].t_curr,
                "G2 chain continuity broken between rows {} and {}",
                i,
                i + 1
            );
        }
        // Row 0 starts at Q.
        assert_eq!(rows[0].t_curr, populate_q());
    }

    #[test]
    fn populate_bn254_miller_loop_trace_tampered_witness_fires_constraints() {
        // Tamper one addition row's NAF digit selectors to both = 1 and
        // confirm the mutex constraint fires. Doesn't run the full
        // prover — just exercises the evaluate_on_domain path.
        let fixture = populate_bn254_miller_loop_trace(&populate_p(), &populate_q());
        // Find first addition row.
        let mut tamper_row_idx = None;
        for (i, r) in fixture.witness.rows.iter().enumerate() {
            if r.is_addition {
                tamper_row_idx = Some(i);
                break;
            }
        }
        let tamper_row_idx =
            tamper_row_idx.expect("populate must produce ≥1 addition row");

        // Rebuild the trace with a manual tamper on the trace columns.
        let curve = CurveType::Bls48581;
        let mut cols_mut: Vec<Vec<Scalar>> = fixture
            .trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Force both naf_pos and naf_neg = 1 on the tamper row.
        cols_mut[COL_NAF_POS][tamper_row_idx] = Scalar::one(curve);
        cols_mut[COL_NAF_NEG][tamper_row_idx] = Scalar::one(curve);

        let cs = Bn254MillerLoopConstraintSystem::new(fixture.trace.num_rows);
        let cols_refs: Vec<&Vec<Scalar>> = cols_mut.iter().collect();
        let evals = cs.evaluate_on_domain(&cols_refs, fixture.trace.num_rows);
        // Constraint 5: naf_pos · naf_neg = 0 must fire on the tamper row.
        assert!(
            !evals[5][tamper_row_idx].is_zero(),
            "naf mutex constraint did not fire on tampered row {}",
            tamper_row_idx
        );

        // Sanity: also tamper the first row's acc_pre limb 3 to 42 (NOT
        // Fp12::one()'s limb 3 = 1). Constraint 9 must fire at row 0.
        cols_mut[COL_ACC_PRE_OFFSET + LIMBS_PER_FP - 1][0] =
            Scalar::from_u64(42, curve);
        let cols_refs: Vec<&Vec<Scalar>> = cols_mut.iter().collect();
        let evals = cs.evaluate_on_domain(&cols_refs, fixture.trace.num_rows);
        assert!(
            !evals[9][0].is_zero(),
            "first-row acc_pre = 1 constraint did not fire on tampered row 0"
        );

        // Suppress unused warning for borrowed fixture.
        let _ = &fixture.witness;
        let _ = fixture.trace.num_rows;
    }

    #[test]
    fn populate_bn254_miller_loop_trace_mini_4row_closure_matches() {
        // 4-row "mini loop" cross-AIR shape check: build a witness with
        // exactly 4 active rows (D, A+1, D, D) and confirm the doubling
        // descriptor's B-side selector activates on exactly the
        // doubling rows (3 of 4). This validates the descriptor's
        // gating + tuple shape without needing the deferred G2
        // curve_ops surface for full closure.
        let curve = CurveType::Bls48581;
        let p = populate_p();
        let q = populate_q();
        let mut w = Bn254MillerLoopWitness::new(p, q);
        let mut t_jac = Bn254G2Jacobian::from_affine(&q);
        let q_x_a = cop_fp2_to_arith(&q.x);
        let q_y_a = cop_fp2_to_arith(&q.y);
        for kind in &['D', 'A', 'D', 'D'] {
            let t_curr = t_jac.to_affine();
            let mut next_jac = t_jac;
            match *kind {
                'D' => {
                    next_jac.double();
                    let t_next = next_jac.to_affine();
                    w.push(Bn254MillerLoopRow {
                        acc_pre: pin::Fp12::one(),
                        acc_post: pin::Fp12::one(),
                        line_value: pin::Fp12::one(),
                        t_curr,
                        t_next,
                        is_doubling: true,
                        is_addition: false,
                        naf_pos: false,
                        naf_neg: false,
                    });
                }
                'A' => {
                    next_jac.add_affine(&q_x_a, &q_y_a);
                    let t_next = next_jac.to_affine();
                    w.push(Bn254MillerLoopRow {
                        acc_pre: pin::Fp12::one(),
                        acc_post: pin::Fp12::one(),
                        line_value: pin::Fp12::one(),
                        t_curr,
                        t_next,
                        is_doubling: false,
                        is_addition: true,
                        naf_pos: true,
                        naf_neg: false,
                    });
                }
                _ => unreachable!(),
            }
            t_jac = next_jac;
        }
        let trace = build_trace_polynomials(&w, curve);
        let cs = Bn254MillerLoopConstraintSystem::new(4);
        let cols_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_refs, trace.num_rows);
        for (k, c) in evals.iter().enumerate() {
            for (r, v) in c.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "mini-loop row-local constraint {} fired at row {}",
                    k,
                    r
                );
            }
        }

        // Count active doubling / addition selector rows.
        let n_dbl = trace.columns[COL_IS_DOUBLING]
            .evaluations
            .iter()
            .filter(|s| !s.is_zero())
            .count();
        let n_add = trace.columns[COL_IS_ADDITION]
            .evaluations
            .iter()
            .filter(|s| !s.is_zero())
            .count();
        assert_eq!(n_dbl, 3);
        assert_eq!(n_add, 1);

        // Confirm cross-AIR descriptor shapes line up with this mini
        // trace's column layout (no closure check — that requires the
        // sibling G2 curve_ops surface).
        let dbl_desc = make_bn254_miller_doubling_to_curve_ops_descriptor(
            "bn254_miller_doubling_mini_v1",
            /* miller layer */ 0,
            /* curve_ops layer */ 1,
        );
        assert_eq!(dbl_desc.b_columns.len(), 2 * LIMBS_PER_G2);
        for (k, &c) in dbl_desc.b_columns.iter().enumerate() {
            if k < LIMBS_PER_G2 {
                assert_eq!(c, COL_T_CURR_OFFSET + k);
            } else {
                assert_eq!(c, COL_T_NEXT_OFFSET + (k - LIMBS_PER_G2));
            }
        }
    }
}
