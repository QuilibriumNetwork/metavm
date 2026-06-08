//! BLS12-381 **G2 / twist** curve-ops scaffold AIR (#271).
//!
//! Mirrors [`crate::bn254_curve_ops_air::g2`] but operates over the
//! BLS12-381 `Fp2 = Fp[u]/(u² + 1)` with 6-limb Fp. G2 is the twist
//!
//! ```text
//!   E'/Fp2 : y² = x³ + b'    with  b' = 4(1 + u) = 4 + 4u ∈ Fp2.
//! ```
//!
//! Unlike BN254's `b' = 3/(9 + u)` (which requires a separate
//! `(9 + u) · b' = 3` definition triple), BLS12-381's `b' = 4 + 4u` is
//! a tiny direct constant, so the curve-equation pinning only needs to
//! pin the 6+6 limbs of `b'` itself.
//!
//! # Doubling / addition formulas
//!
//! Coefficient-independent (same as BN254 G1 / G2):
//!
//! ```text
//!   Doubling:  λ = 3·P.x² / (2·P.y)       (all over Fp2)
//!              R.x = λ² − 2·P.x
//!              R.y = λ·(P.x − R.x) − P.y
//!
//!   Addition:  λ = (Q.y − P.y) / (Q.x − P.x)
//!              R.x = λ² − P.x − Q.x
//!              R.y = λ·(P.x − R.x) − P.y
//! ```
//!
//! # Witness layout
//!
//! Each row commits one G2 op `R = 2·P` or `R = P + Q`, plus the
//! per-row **intermediate Fp2 values** that decompose the formulas
//! into individual Fp2 add / sub / mul / inv operations. Each
//! intermediate Fp2 triple `(a_fp2, b_fp2, c_fp2)` is shipped to the
//! BLS12-381 [`super::fp2`] AIR via a cross-AIR LogUp descriptor under
//! the appropriate selector. The Fp2 AIR in turn ships every Fp triple
//! down to the BLS12-381 [`super::fp`] AIR, so the full algebraic
//! decomposition for one G2 row chains G2 → Fp2 → Fp.
//!
//! The G2 scaffold itself enforces only the binary / mutex shape
//! constraints plus the pinning of `b' = 4 + 4u`; soundness of the
//! curve law comes from the LogUp closures + Fp2 AIR + Fp AIR.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

use super::fp;
use super::fp2::{
    self as bls_fp2,
    LIMBS_PER_FP2_COMPONENT,
    COL_A_C0_OFFSET as FP2_COL_A_C0_OFFSET, COL_A_C1_OFFSET as FP2_COL_A_C1_OFFSET,
    COL_B_C0_OFFSET as FP2_COL_B_C0_OFFSET, COL_B_C1_OFFSET as FP2_COL_B_C1_OFFSET,
    COL_C_C0_OFFSET as FP2_COL_C_C0_OFFSET, COL_C_C1_OFFSET as FP2_COL_C_C1_OFFSET,
    COL_SEL_ADD as FP2_COL_SEL_ADD, COL_SEL_SUB as FP2_COL_SEL_SUB,
    COL_SEL_MUL as FP2_COL_SEL_MUL, COL_SEL_INV as FP2_COL_SEL_INV,
};

// ─── Column layout (limb counts) ──────────────────────────────────────

pub const LIMBS_PER_FP: usize = LIMBS_PER_FP2_COMPONENT;
pub const LIMBS_PER_FP2: usize = 2 * LIMBS_PER_FP;
pub const LIMBS_PER_G2: usize = 2 * LIMBS_PER_FP2;

// Inputs / output: 3 affine G2 points = 6 Fp2 = 12 limb-blocks = 72 limbs.
pub const COL_P_OFFSET: usize = 0;
pub const COL_PX_C0: usize = COL_P_OFFSET;
pub const COL_PX_C1: usize = COL_PX_C0 + LIMBS_PER_FP;
pub const COL_PY_C0: usize = COL_PX_C1 + LIMBS_PER_FP;
pub const COL_PY_C1: usize = COL_PY_C0 + LIMBS_PER_FP;

pub const COL_Q_OFFSET: usize = COL_P_OFFSET + LIMBS_PER_G2;
pub const COL_QX_C0: usize = COL_Q_OFFSET;
pub const COL_QX_C1: usize = COL_QX_C0 + LIMBS_PER_FP;
pub const COL_QY_C0: usize = COL_QX_C1 + LIMBS_PER_FP;
pub const COL_QY_C1: usize = COL_QY_C0 + LIMBS_PER_FP;

pub const COL_R_OFFSET: usize = COL_Q_OFFSET + LIMBS_PER_G2;
pub const COL_RX_C0: usize = COL_R_OFFSET;
pub const COL_RX_C1: usize = COL_RX_C0 + LIMBS_PER_FP;
pub const COL_RY_C0: usize = COL_RX_C1 + LIMBS_PER_FP;
pub const COL_RY_C1: usize = COL_RY_C0 + LIMBS_PER_FP;

// Shape selectors (single cols).
pub const COL_IS_REAL: usize = COL_R_OFFSET + LIMBS_PER_G2;
pub const COL_SEL_ADD: usize = COL_IS_REAL + 1;
pub const COL_SEL_DOUBLE: usize = COL_SEL_ADD + 1;
pub const COL_P_INFINITY: usize = COL_SEL_DOUBLE + 1;
pub const COL_Q_INFINITY: usize = COL_P_INFINITY + 1;
pub const COL_R_INFINITY: usize = COL_Q_INFINITY + 1;

// ─── Doubling intermediates (10 Fp2 values) ───────────────────────────
//
//   x_sq            = P.x · P.x                       (Fp2 Mul)
//   two_x_sq        = x_sq + x_sq                     (Fp2 Add)
//   lambda_num      = two_x_sq + x_sq    = 3·P.x²     (Fp2 Add)
//   lambda_denom    = P.y + P.y          = 2·P.y      (Fp2 Add)
//   lambda_denom_inv  s.t. denom · inv = 1            (Fp2 Inv)
//   lambda          = lambda_num · lambda_denom_inv   (Fp2 Mul)
//   lambda_sq       = lambda · lambda                 (Fp2 Mul)
//   two_px          = P.x + P.x                       (Fp2 Add)
//   diff_px_rx      = P.x − R.x                       (Fp2 Sub)
//   lambda_diff     = lambda · diff_px_rx             (Fp2 Mul)
//   R.x bound via:  lambda_sq − two_px      (Fp2 Sub)
//   R.y bound via:  lambda_diff − P.y       (Fp2 Sub)
pub const COL_DBL_X_SQ:             usize = COL_R_INFINITY + 1;
pub const COL_DBL_TWO_X_SQ:         usize = COL_DBL_X_SQ + LIMBS_PER_FP2;
pub const COL_DBL_LAMBDA_NUM:       usize = COL_DBL_TWO_X_SQ + LIMBS_PER_FP2;
pub const COL_DBL_LAMBDA_DENOM:     usize = COL_DBL_LAMBDA_NUM + LIMBS_PER_FP2;
pub const COL_DBL_LAMBDA_DENOM_INV: usize = COL_DBL_LAMBDA_DENOM + LIMBS_PER_FP2;
pub const COL_DBL_LAMBDA:           usize = COL_DBL_LAMBDA_DENOM_INV + LIMBS_PER_FP2;
pub const COL_DBL_LAMBDA_SQ:        usize = COL_DBL_LAMBDA + LIMBS_PER_FP2;
pub const COL_DBL_TWO_PX:           usize = COL_DBL_LAMBDA_SQ + LIMBS_PER_FP2;
pub const COL_DBL_DIFF_PX_RX:       usize = COL_DBL_TWO_PX + LIMBS_PER_FP2;
pub const COL_DBL_LAMBDA_DIFF:      usize = COL_DBL_DIFF_PX_RX + LIMBS_PER_FP2;

// ─── Addition intermediates (8 Fp2 values) ────────────────────────────
//
//   diff_qy_py      = Q.y − P.y                       (Fp2 Sub)
//   diff_qx_px      = Q.x − P.x                       (Fp2 Sub)
//   diff_qx_px_inv    s.t. diff · inv = 1             (Fp2 Inv)
//   lambda          = diff_qy_py · diff_qx_px_inv     (Fp2 Mul)
//   lambda_sq       = lambda · lambda                 (Fp2 Mul)
//   sum_px_qx       = P.x + Q.x                       (Fp2 Add)
//   diff_px_rx      = P.x − R.x                       (Fp2 Sub)
//   lambda_diff     = lambda · diff_px_rx             (Fp2 Mul)
//   R.x bound via:  lambda_sq − sum_px_qx     (Fp2 Sub)
//   R.y bound via:  lambda_diff − P.y         (Fp2 Sub)
pub const COL_ADD_DIFF_QY_PY:     usize = COL_DBL_LAMBDA_DIFF + LIMBS_PER_FP2;
pub const COL_ADD_DIFF_QX_PX:     usize = COL_ADD_DIFF_QY_PY + LIMBS_PER_FP2;
pub const COL_ADD_DIFF_QX_PX_INV: usize = COL_ADD_DIFF_QX_PX + LIMBS_PER_FP2;
pub const COL_ADD_LAMBDA:         usize = COL_ADD_DIFF_QX_PX_INV + LIMBS_PER_FP2;
pub const COL_ADD_LAMBDA_SQ:      usize = COL_ADD_LAMBDA + LIMBS_PER_FP2;
pub const COL_ADD_SUM_PX_QX:      usize = COL_ADD_LAMBDA_SQ + LIMBS_PER_FP2;
pub const COL_ADD_DIFF_PX_RX:     usize = COL_ADD_SUM_PX_QX + LIMBS_PER_FP2;
pub const COL_ADD_LAMBDA_DIFF:    usize = COL_ADD_DIFF_PX_RX + LIMBS_PER_FP2;

// ─── Curve-equation intermediates ─────────────────────────────────────
//
// `R.y² = R.x³ + b'` over Fp2, with `b' = 4 + 4u`. Decomposition:
//
//   rx_sq    = R.x · R.x          (Fp2 Mul)
//   rx_cubed = rx_sq · R.x        (Fp2 Mul)
//   ry_sq    = R.y · R.y          (Fp2 Mul)
//   ry_sq − rx_cubed = b_prime    (Fp2 Sub)  — bound via LogUp
//
// `b_prime` is row-locally pinned to `4 + 4u · sel_curve` via 12 limb
// constraints (6 for c0, 6 for c1).
pub const COL_RX_SQ:    usize = COL_ADD_LAMBDA_DIFF + LIMBS_PER_FP2;
pub const COL_RX_CUBED: usize = COL_RX_SQ + LIMBS_PER_FP2;
pub const COL_RY_SQ:    usize = COL_RX_CUBED + LIMBS_PER_FP2;
pub const COL_B_PRIME:  usize = COL_RY_SQ + LIMBS_PER_FP2;
pub const COL_SEL_CURVE: usize = COL_B_PRIME + LIMBS_PER_FP2;

pub const NUM_COLUMNS: usize = COL_SEL_CURVE + 1;

// ─── Row-local constraints ────────────────────────────────────────────
//
//   0. is_real     ∈ {0, 1}
//   1. sel_add     ∈ {0, 1}
//   2. sel_double  ∈ {0, 1}
//   3. sel_add · sel_double = 0                 (mutex)
//   4. sel_add + sel_double − is_real = 0       (one op on real rows)
//   5. p_infinity  ∈ {0, 1}
//   6. q_infinity  ∈ {0, 1}
//   7. r_infinity  ∈ {0, 1}
//   8. sel_curve   ∈ {0, 1}
//   9. sel_curve = is_real · (1 − r_infinity)
//
// Plus pin every limb of `b_prime` to `expected · sel_curve`:
//
//  10..15  b_prime.c0[0..5]    (expected (0,0,0,0,0,4))
//  16..21  b_prime.c1[0..5]    (expected (0,0,0,0,0,4))
pub const NUM_ROW_CONSTRAINTS: usize = 10 + 2 * LIMBS_PER_FP;
pub const NUM_SHIFTED: usize = 0;

// ─── Host-side reference: G2 doubling / addition over Fp2 ─────────────

/// G2 doubling. Panics on identity or vertical tangent (`P.y = 0`).
pub fn double_host(px: &bls_fp2::Fp2, py: &bls_fp2::Fp2) -> (bls_fp2::Fp2, bls_fp2::Fp2) {
    let two = bls_fp2::Fp2 { c0: fp::Fp::from_u64(2), c1: fp::Fp::zero() };
    let three = bls_fp2::Fp2 { c0: fp::Fp::from_u64(3), c1: fp::Fp::zero() };
    let x_sq = px.mul(px);
    let lambda_num = three.mul(&x_sq);
    let lambda_denom = two.mul(py);
    let lambda_denom_inv = lambda_denom.invert().expect("2·P.y must be invertible");
    let lambda = lambda_num.mul(&lambda_denom_inv);
    let lambda_sq = lambda.mul(&lambda);
    let two_px = two.mul(px);
    let rx = lambda_sq.sub(&two_px);
    let diff = px.sub(&rx);
    let ry = lambda.mul(&diff).sub(py);
    (rx, ry)
}

/// G2 addition `R = P + Q`. Panics on `Q.x == P.x`.
pub fn add_host(
    px: &bls_fp2::Fp2, py: &bls_fp2::Fp2,
    qx: &bls_fp2::Fp2, qy: &bls_fp2::Fp2,
) -> (bls_fp2::Fp2, bls_fp2::Fp2) {
    let diff_y = qy.sub(py);
    let diff_x = qx.sub(px);
    let diff_x_inv = diff_x.invert().expect("Q.x − P.x must be invertible");
    let lambda = diff_y.mul(&diff_x_inv);
    let lambda_sq = lambda.mul(&lambda);
    let sum_x = px.add(qx);
    let rx = lambda_sq.sub(&sum_x);
    let diff_px_rx = px.sub(&rx);
    let ry = lambda.mul(&diff_px_rx).sub(py);
    (rx, ry)
}

/// BLS12-381 G2 twist parameter `b' = 4 + 4u ∈ Fp2`.
pub fn twist_b() -> bls_fp2::Fp2 {
    bls_fp2::Fp2 { c0: fp::Fp::from_u64(4), c1: fp::Fp::from_u64(4) }
}

// ─── Row + witness ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpSelector {
    Add,
    Double,
}

#[derive(Clone, Debug)]
pub struct Bls12G2OpsRow {
    pub px: bls_fp2::Fp2,
    pub py: bls_fp2::Fp2,
    pub qx: bls_fp2::Fp2,
    pub qy: bls_fp2::Fp2,
    pub rx: bls_fp2::Fp2,
    pub ry: bls_fp2::Fp2,
    pub op: OpSelector,
    pub is_real: bool,
    pub p_infinity: bool,
    pub q_infinity: bool,
    pub r_infinity: bool,
}

impl Bls12G2OpsRow {
    /// Build a `R = 2·P` row from host-side Fp2 inputs.
    pub fn double_row(px: bls_fp2::Fp2, py: bls_fp2::Fp2) -> Self {
        let (rx, ry) = double_host(&px, &py);
        Self {
            px, py,
            qx: bls_fp2::Fp2::zero(), qy: bls_fp2::Fp2::zero(),
            rx, ry,
            op: OpSelector::Double,
            is_real: true,
            p_infinity: false, q_infinity: true, r_infinity: false,
        }
    }

    /// Build a `R = P + Q` row from host-side Fp2 inputs.
    pub fn add_row(
        px: bls_fp2::Fp2, py: bls_fp2::Fp2,
        qx: bls_fp2::Fp2, qy: bls_fp2::Fp2,
    ) -> Self {
        let (rx, ry) = add_host(&px, &py, &qx, &qy);
        Self {
            px, py, qx, qy, rx, ry,
            op: OpSelector::Add,
            is_real: true,
            p_infinity: false, q_infinity: false, r_infinity: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Bls12G2OpsWitness {
    pub rows: Vec<Bls12G2OpsRow>,
}

impl Bls12G2OpsWitness {
    pub fn new() -> Self { Self { rows: Vec::new() } }
    pub fn push(&mut self, row: Bls12G2OpsRow) { self.rows.push(row); }
}

// ─── Trace builder helpers ────────────────────────────────────────────

fn write_fp(columns: &mut [Vec<Scalar>], base: usize, r: usize, fp: &fp::Fp, curve: CurveType) {
    for j in 0..LIMBS_PER_FP {
        columns[base + j][r] = Scalar::from_u64(fp.limbs[j], curve);
    }
}

fn write_fp2(columns: &mut [Vec<Scalar>], base: usize, r: usize, v: &bls_fp2::Fp2, curve: CurveType) {
    write_fp(columns, base, r, &v.c0, curve);
    write_fp(columns, base + LIMBS_PER_FP, r, &v.c1, curve);
}

fn populate_double_intermediates(
    columns: &mut [Vec<Scalar>],
    r: usize,
    px: &bls_fp2::Fp2,
    py: &bls_fp2::Fp2,
    curve: CurveType,
) {
    let x_sq = px.mul(px);
    let two_x_sq = x_sq.add(&x_sq);
    let lambda_num = two_x_sq.add(&x_sq);
    let lambda_denom = py.add(py);
    let lambda_denom_inv = lambda_denom.invert().expect("2·P.y nonzero");
    let lambda = lambda_num.mul(&lambda_denom_inv);
    let lambda_sq = lambda.mul(&lambda);
    let two_px = px.add(px);
    let rx = lambda_sq.sub(&two_px);
    let diff_px_rx = px.sub(&rx);
    let lambda_diff = lambda.mul(&diff_px_rx);

    write_fp2(columns, COL_DBL_X_SQ, r, &x_sq, curve);
    write_fp2(columns, COL_DBL_TWO_X_SQ, r, &two_x_sq, curve);
    write_fp2(columns, COL_DBL_LAMBDA_NUM, r, &lambda_num, curve);
    write_fp2(columns, COL_DBL_LAMBDA_DENOM, r, &lambda_denom, curve);
    write_fp2(columns, COL_DBL_LAMBDA_DENOM_INV, r, &lambda_denom_inv, curve);
    write_fp2(columns, COL_DBL_LAMBDA, r, &lambda, curve);
    write_fp2(columns, COL_DBL_LAMBDA_SQ, r, &lambda_sq, curve);
    write_fp2(columns, COL_DBL_TWO_PX, r, &two_px, curve);
    write_fp2(columns, COL_DBL_DIFF_PX_RX, r, &diff_px_rx, curve);
    write_fp2(columns, COL_DBL_LAMBDA_DIFF, r, &lambda_diff, curve);
}

fn populate_add_intermediates(
    columns: &mut [Vec<Scalar>],
    r: usize,
    px: &bls_fp2::Fp2,
    py: &bls_fp2::Fp2,
    qx: &bls_fp2::Fp2,
    qy: &bls_fp2::Fp2,
    curve: CurveType,
) {
    let diff_y = qy.sub(py);
    let diff_x = qx.sub(px);
    let diff_x_inv = diff_x.invert().expect("Q.x − P.x nonzero");
    let lambda = diff_y.mul(&diff_x_inv);
    let lambda_sq = lambda.mul(&lambda);
    let sum_x = px.add(qx);
    let rx = lambda_sq.sub(&sum_x);
    let diff_px_rx = px.sub(&rx);
    let lambda_diff = lambda.mul(&diff_px_rx);

    write_fp2(columns, COL_ADD_DIFF_QY_PY, r, &diff_y, curve);
    write_fp2(columns, COL_ADD_DIFF_QX_PX, r, &diff_x, curve);
    write_fp2(columns, COL_ADD_DIFF_QX_PX_INV, r, &diff_x_inv, curve);
    write_fp2(columns, COL_ADD_LAMBDA, r, &lambda, curve);
    write_fp2(columns, COL_ADD_LAMBDA_SQ, r, &lambda_sq, curve);
    write_fp2(columns, COL_ADD_SUM_PX_QX, r, &sum_x, curve);
    write_fp2(columns, COL_ADD_DIFF_PX_RX, r, &diff_px_rx, curve);
    write_fp2(columns, COL_ADD_LAMBDA_DIFF, r, &lambda_diff, curve);
}

pub fn build_trace_polynomials(
    witness: &Bls12G2OpsWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        write_fp2(&mut columns, COL_PX_C0, r, &row.px, curve);
        write_fp2(&mut columns, COL_PY_C0, r, &row.py, curve);
        write_fp2(&mut columns, COL_QX_C0, r, &row.qx, curve);
        write_fp2(&mut columns, COL_QY_C0, r, &row.qy, curve);
        write_fp2(&mut columns, COL_RX_C0, r, &row.rx, curve);
        write_fp2(&mut columns, COL_RY_C0, r, &row.ry, curve);

        columns[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };
        match row.op {
            OpSelector::Add => {
                columns[COL_SEL_ADD][r] = if row.is_real { one.clone() } else { zero.clone() };
            }
            OpSelector::Double => {
                columns[COL_SEL_DOUBLE][r] = if row.is_real { one.clone() } else { zero.clone() };
            }
        }
        columns[COL_P_INFINITY][r] = if row.p_infinity { one.clone() } else { zero.clone() };
        columns[COL_Q_INFINITY][r] = if row.q_infinity { one.clone() } else { zero.clone() };
        columns[COL_R_INFINITY][r] = if row.r_infinity { one.clone() } else { zero.clone() };

        if row.is_real && !row.p_infinity && !row.r_infinity {
            match row.op {
                OpSelector::Double => {
                    populate_double_intermediates(&mut columns, r, &row.px, &row.py, curve);
                }
                OpSelector::Add => {
                    if !row.q_infinity {
                        populate_add_intermediates(
                            &mut columns, r, &row.px, &row.py, &row.qx, &row.qy, curve,
                        );
                    }
                }
            }

            // Curve-equation witnesses.
            let rx_sq = row.rx.mul(&row.rx);
            let rx_cubed = rx_sq.mul(&row.rx);
            let ry_sq = row.ry.mul(&row.ry);
            let b_prime = twist_b();

            write_fp2(&mut columns, COL_RX_SQ, r, &rx_sq, curve);
            write_fp2(&mut columns, COL_RX_CUBED, r, &rx_cubed, curve);
            write_fp2(&mut columns, COL_RY_SQ, r, &ry_sq, curve);
            write_fp2(&mut columns, COL_B_PRIME, r, &b_prime, curve);
            columns[COL_SEL_CURVE][r] = one.clone();
        }
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct Bls12G2OpsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bls12G2OpsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
}

fn bin(v: &Scalar) -> Scalar {
    let curve = v.curve_type();
    let one = Scalar::one(curve);
    v.mul(&v.sub(&one))
}

impl VmConstraintSystem for Bls12G2OpsConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = vec![
            "is_real_binary".into(),
            "sel_add_binary".into(),
            "sel_double_binary".into(),
            "selectors_mutex".into(),
            "selectors_sum_to_is_real".into(),
            "p_infinity_binary".into(),
            "q_infinity_binary".into(),
            "r_infinity_binary".into(),
            "sel_curve_binary".into(),
            "sel_curve_eq_real_and_not_r_infinity".into(),
        ];
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("b_prime_c0_limb{}_pinned", j));
        }
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("b_prime_c1_limb{}_pinned", j));
        }
        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let n = columns[0].len();
        let curve = columns[0][0].curve_type();
        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        // Expected b' = 4 + 4u. BE limbs with limb[5] = least significant.
        let mut expected_bp_c0 = [0u64; LIMBS_PER_FP];
        expected_bp_c0[LIMBS_PER_FP - 1] = 4;
        let mut expected_bp_c1 = [0u64; LIMBS_PER_FP];
        expected_bp_c1[LIMBS_PER_FP - 1] = 4;

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let sa = &columns[COL_SEL_ADD][r];
            let sd = &columns[COL_SEL_DOUBLE][r];
            let pi = &columns[COL_P_INFINITY][r];
            let qi = &columns[COL_Q_INFINITY][r];
            let ri = &columns[COL_R_INFINITY][r];
            let sc = &columns[COL_SEL_CURVE][r];
            out[0][r] = bin(is_real);
            out[1][r] = bin(sa);
            out[2][r] = bin(sd);
            out[3][r] = sa.mul(sd);
            out[4][r] = sa.add(sd).sub(is_real);
            out[5][r] = bin(pi);
            out[6][r] = bin(qi);
            out[7][r] = bin(ri);
            out[8][r] = bin(sc);
            out[9][r] = sc.add(&is_real.mul(ri)).sub(is_real);

            let pin = |val: &Scalar, expected: u64, sel: &Scalar| -> Scalar {
                let e = Scalar::from_u64(expected, curve);
                val.sub(&e.mul(sel))
            };
            let mut idx = 10;
            for j in 0..LIMBS_PER_FP {
                out[idx][r] = pin(&columns[COL_B_PRIME + j][r], expected_bp_c0[j], sc);
                idx += 1;
            }
            for j in 0..LIMBS_PER_FP {
                out[idx][r] = pin(
                    &columns[COL_B_PRIME + LIMBS_PER_FP + j][r],
                    expected_bp_c1[j],
                    sc,
                );
                idx += 1;
            }
            debug_assert_eq!(idx, NUM_ROW_CONSTRAINTS);
        }
        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let is_real = &col_evals[COL_IS_REAL];
        let sa = &col_evals[COL_SEL_ADD];
        let sd = &col_evals[COL_SEL_DOUBLE];
        let pi = &col_evals[COL_P_INFINITY];
        let qi = &col_evals[COL_Q_INFINITY];
        let ri = &col_evals[COL_R_INFINITY];
        let sc = &col_evals[COL_SEL_CURVE];
        let mut parts: Vec<Scalar> = vec![
            bin(is_real),
            bin(sa),
            bin(sd),
            sa.mul(sd),
            sa.add(sd).sub(is_real),
            bin(pi),
            bin(qi),
            bin(ri),
            bin(sc),
            sc.add(&is_real.mul(ri)).sub(is_real),
        ];
        let pin = |val: &Scalar, expected: u64| -> Scalar {
            let e = Scalar::from_u64(expected, curve);
            val.sub(&e.mul(sc))
        };
        let mut expected_bp_c0 = [0u64; LIMBS_PER_FP];
        expected_bp_c0[LIMBS_PER_FP - 1] = 4;
        let mut expected_bp_c1 = [0u64; LIMBS_PER_FP];
        expected_bp_c1[LIMBS_PER_FP - 1] = 4;
        for j in 0..LIMBS_PER_FP {
            parts.push(pin(&col_evals[COL_B_PRIME + j], expected_bp_c0[j]));
        }
        for j in 0..LIMBS_PER_FP {
            parts.push(pin(&col_evals[COL_B_PRIME + LIMBS_PER_FP + j], expected_bp_c1[j]));
        }
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

        let bin_poly = |v: &Vec<Scalar>| -> Vec<Scalar> {
            let v_m1 = poly_sub(v, &one_poly, curve);
            poly_mul(v, &v_m1, curve)
        };

        let mut parts: Vec<Vec<Scalar>> = vec![
            bin_poly(&col_coeffs[COL_IS_REAL]),
            bin_poly(&col_coeffs[COL_SEL_ADD]),
            bin_poly(&col_coeffs[COL_SEL_DOUBLE]),
            poly_mul(&col_coeffs[COL_SEL_ADD], &col_coeffs[COL_SEL_DOUBLE], curve),
            poly_sub(
                &poly_add(&col_coeffs[COL_SEL_ADD], &col_coeffs[COL_SEL_DOUBLE], curve),
                &col_coeffs[COL_IS_REAL],
                curve,
            ),
            bin_poly(&col_coeffs[COL_P_INFINITY]),
            bin_poly(&col_coeffs[COL_Q_INFINITY]),
            bin_poly(&col_coeffs[COL_R_INFINITY]),
            bin_poly(&col_coeffs[COL_SEL_CURVE]),
            poly_sub(
                &poly_add(
                    &col_coeffs[COL_SEL_CURVE],
                    &poly_mul(
                        &col_coeffs[COL_IS_REAL],
                        &col_coeffs[COL_R_INFINITY],
                        curve,
                    ),
                    curve,
                ),
                &col_coeffs[COL_IS_REAL],
                curve,
            ),
        ];
        let mut expected_bp_c0 = [0u64; LIMBS_PER_FP];
        expected_bp_c0[LIMBS_PER_FP - 1] = 4;
        let mut expected_bp_c1 = [0u64; LIMBS_PER_FP];
        expected_bp_c1[LIMBS_PER_FP - 1] = 4;
        let push_pin = |parts: &mut Vec<Vec<Scalar>>, col: usize, expected: u64| {
            let e = Scalar::from_u64(expected, curve);
            let e_times_sc = poly_scalar_mul(&col_coeffs[COL_SEL_CURVE], &e);
            parts.push(poly_sub(&col_coeffs[col], &e_times_sc, curve));
        };
        for j in 0..LIMBS_PER_FP {
            push_pin(&mut parts, COL_B_PRIME + j, expected_bp_c0[j]);
        }
        for j in 0..LIMBS_PER_FP {
            push_pin(&mut parts, COL_B_PRIME + LIMBS_PER_FP + j, expected_bp_c1[j]);
        }
        debug_assert_eq!(parts.len(), NUM_ROW_CONSTRAINTS);
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
            COL_IS_REAL, COL_SEL_ADD, COL_SEL_DOUBLE,
            COL_P_INFINITY, COL_Q_INFINITY, COL_R_INFINITY,
            COL_SEL_CURVE,
        ]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> { Vec::new() }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }
}

// ─── Cross-AIR LogUp descriptors: g2 ↔ fp2 ────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp2OpKind {
    Add,
    Sub,
    Mul,
    Inv,
}

fn fp2_sel_col(kind: Fp2OpKind) -> usize {
    match kind {
        Fp2OpKind::Add => FP2_COL_SEL_ADD,
        Fp2OpKind::Sub => FP2_COL_SEL_SUB,
        Fp2OpKind::Mul => FP2_COL_SEL_MUL,
        Fp2OpKind::Inv => FP2_COL_SEL_INV,
    }
}

/// Width of one `(a, b, c)` Fp2 triple in limbs (`3 × 2 × LIMBS_PER_FP`).
pub const FP2_TRIPLE_LIMBS: usize = 3 * LIMBS_PER_FP2;

/// Build a single `(a, b, c)` Fp2-triple LogUp descriptor.
pub fn make_bls12_g2_fp2_triple_descriptor(
    label: impl Into<String>,
    g2_layer_index: usize,
    fp2_layer_index: usize,
    a_base: usize,
    b_base: usize,
    c_base: usize,
    kind: Fp2OpKind,
    g2_sel_col: usize,
) -> CrossAirLogUpDescriptor {
    let mut b_columns: Vec<usize> = Vec::with_capacity(FP2_TRIPLE_LIMBS);
    for j in 0..LIMBS_PER_FP { b_columns.push(a_base + j); }                      // a.c0
    for j in 0..LIMBS_PER_FP { b_columns.push(a_base + LIMBS_PER_FP + j); }       // a.c1
    for j in 0..LIMBS_PER_FP { b_columns.push(b_base + j); }                      // b.c0
    for j in 0..LIMBS_PER_FP { b_columns.push(b_base + LIMBS_PER_FP + j); }       // b.c1
    for j in 0..LIMBS_PER_FP { b_columns.push(c_base + j); }                      // c.c0
    for j in 0..LIMBS_PER_FP { b_columns.push(c_base + LIMBS_PER_FP + j); }       // c.c1

    let mut a_columns: Vec<usize> = Vec::with_capacity(FP2_TRIPLE_LIMBS);
    for j in 0..LIMBS_PER_FP { a_columns.push(FP2_COL_A_C0_OFFSET + j); }
    for j in 0..LIMBS_PER_FP { a_columns.push(FP2_COL_A_C1_OFFSET + j); }
    for j in 0..LIMBS_PER_FP { a_columns.push(FP2_COL_B_C0_OFFSET + j); }
    for j in 0..LIMBS_PER_FP { a_columns.push(FP2_COL_B_C1_OFFSET + j); }
    for j in 0..LIMBS_PER_FP { a_columns.push(FP2_COL_C_C0_OFFSET + j); }
    for j in 0..LIMBS_PER_FP { a_columns.push(FP2_COL_C_C1_OFFSET + j); }

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: fp2_layer_index,
        a_columns,
        a_selector_column: Some(fp2_sel_col(kind)),
        b_layer_index: g2_layer_index,
        b_columns,
        b_selector_column: Some(g2_sel_col),
    }
}

/// All 12 Fp2-triple LogUp descriptors witnessing the G2 doubling chain
/// for one G2-ops row (gated by `COL_SEL_DOUBLE`).
pub fn bls12_g2_double_fp2_descriptors(
    g2_layer_index: usize,
    fp2_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    let mk = |label: &str, kind: Fp2OpKind, a: usize, b: usize, c: usize| {
        make_bls12_g2_fp2_triple_descriptor(
            label, g2_layer_index, fp2_layer_index, a, b, c, kind, COL_SEL_DOUBLE,
        )
    };
    vec![
        mk("bls12_g2_dbl_x_sq",         Fp2OpKind::Mul, COL_PX_C0,                  COL_PX_C0,                  COL_DBL_X_SQ),
        mk("bls12_g2_dbl_two_x_sq",     Fp2OpKind::Add, COL_DBL_X_SQ,               COL_DBL_X_SQ,               COL_DBL_TWO_X_SQ),
        mk("bls12_g2_dbl_lambda_num",   Fp2OpKind::Add, COL_DBL_TWO_X_SQ,           COL_DBL_X_SQ,               COL_DBL_LAMBDA_NUM),
        mk("bls12_g2_dbl_lambda_denom", Fp2OpKind::Add, COL_PY_C0,                  COL_PY_C0,                  COL_DBL_LAMBDA_DENOM),
        mk("bls12_g2_dbl_inv_denom",    Fp2OpKind::Inv, COL_DBL_LAMBDA_DENOM,       COL_DBL_LAMBDA_DENOM_INV,   COL_DBL_LAMBDA_DENOM_INV),
        mk("bls12_g2_dbl_lambda",       Fp2OpKind::Mul, COL_DBL_LAMBDA_NUM,         COL_DBL_LAMBDA_DENOM_INV,   COL_DBL_LAMBDA),
        mk("bls12_g2_dbl_lambda_sq",    Fp2OpKind::Mul, COL_DBL_LAMBDA,             COL_DBL_LAMBDA,             COL_DBL_LAMBDA_SQ),
        mk("bls12_g2_dbl_two_px",       Fp2OpKind::Add, COL_PX_C0,                  COL_PX_C0,                  COL_DBL_TWO_PX),
        mk("bls12_g2_dbl_r_x",          Fp2OpKind::Sub, COL_DBL_LAMBDA_SQ,          COL_DBL_TWO_PX,             COL_RX_C0),
        mk("bls12_g2_dbl_diff_px_rx",   Fp2OpKind::Sub, COL_PX_C0,                  COL_RX_C0,                  COL_DBL_DIFF_PX_RX),
        mk("bls12_g2_dbl_lambda_diff",  Fp2OpKind::Mul, COL_DBL_LAMBDA,             COL_DBL_DIFF_PX_RX,         COL_DBL_LAMBDA_DIFF),
        mk("bls12_g2_dbl_r_y",          Fp2OpKind::Sub, COL_DBL_LAMBDA_DIFF,        COL_PY_C0,                  COL_RY_C0),
    ]
}

/// All 10 Fp2-triple LogUp descriptors witnessing the G2 addition chain.
pub fn bls12_g2_add_fp2_descriptors(
    g2_layer_index: usize,
    fp2_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    let mk = |label: &str, kind: Fp2OpKind, a: usize, b: usize, c: usize| {
        make_bls12_g2_fp2_triple_descriptor(
            label, g2_layer_index, fp2_layer_index, a, b, c, kind, COL_SEL_ADD,
        )
    };
    vec![
        mk("bls12_g2_add_diff_qy_py",   Fp2OpKind::Sub, COL_QY_C0,                  COL_PY_C0,                  COL_ADD_DIFF_QY_PY),
        mk("bls12_g2_add_diff_qx_px",   Fp2OpKind::Sub, COL_QX_C0,                  COL_PX_C0,                  COL_ADD_DIFF_QX_PX),
        mk("bls12_g2_add_inv_diff",     Fp2OpKind::Inv, COL_ADD_DIFF_QX_PX,         COL_ADD_DIFF_QX_PX_INV,     COL_ADD_DIFF_QX_PX_INV),
        mk("bls12_g2_add_lambda",       Fp2OpKind::Mul, COL_ADD_DIFF_QY_PY,         COL_ADD_DIFF_QX_PX_INV,     COL_ADD_LAMBDA),
        mk("bls12_g2_add_lambda_sq",    Fp2OpKind::Mul, COL_ADD_LAMBDA,             COL_ADD_LAMBDA,             COL_ADD_LAMBDA_SQ),
        mk("bls12_g2_add_sum_px_qx",    Fp2OpKind::Add, COL_PX_C0,                  COL_QX_C0,                  COL_ADD_SUM_PX_QX),
        mk("bls12_g2_add_r_x",          Fp2OpKind::Sub, COL_ADD_LAMBDA_SQ,          COL_ADD_SUM_PX_QX,          COL_RX_C0),
        mk("bls12_g2_add_diff_px_rx",   Fp2OpKind::Sub, COL_PX_C0,                  COL_RX_C0,                  COL_ADD_DIFF_PX_RX),
        mk("bls12_g2_add_lambda_diff",  Fp2OpKind::Mul, COL_ADD_LAMBDA,             COL_ADD_DIFF_PX_RX,         COL_ADD_LAMBDA_DIFF),
        mk("bls12_g2_add_r_y",          Fp2OpKind::Sub, COL_ADD_LAMBDA_DIFF,        COL_PY_C0,                  COL_RY_C0),
    ]
}

/// All 4 Fp2-triple LogUp descriptors witnessing the **G2 twist curve
/// equation** `R.y² = R.x³ + b'` for one G2 row (gated by
/// `COL_SEL_CURVE`).
///
/// Triples:
///
/// 1. `curve_rx_sq`    : Mul · `(R.x, R.x, rx_sq)`
/// 2. `curve_rx_cubed` : Mul · `(rx_sq, R.x, rx_cubed)`
/// 3. `curve_ry_sq`    : Mul · `(R.y, R.y, ry_sq)`
/// 4. `curve_eq`       : Sub · `(ry_sq, rx_cubed, b_prime)`  — `b'` is
///                       row-locally pinned to `4 + 4u`.
pub fn bls12_g2_curve_eq_fp2_descriptors(
    g2_layer_index: usize,
    fp2_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    let mk = |label: &str, kind: Fp2OpKind, a: usize, b: usize, c: usize| {
        make_bls12_g2_fp2_triple_descriptor(
            label, g2_layer_index, fp2_layer_index, a, b, c, kind, COL_SEL_CURVE,
        )
    };
    vec![
        mk("bls12_g2_curve_rx_sq",    Fp2OpKind::Mul, COL_RX_C0,    COL_RX_C0,    COL_RX_SQ),
        mk("bls12_g2_curve_rx_cubed", Fp2OpKind::Mul, COL_RX_SQ,    COL_RX_C0,    COL_RX_CUBED),
        mk("bls12_g2_curve_ry_sq",    Fp2OpKind::Mul, COL_RY_C0,    COL_RY_C0,    COL_RY_SQ),
        mk("bls12_g2_curve_eq",       Fp2OpKind::Sub, COL_RY_SQ,    COL_RX_CUBED, COL_B_PRIME),
    ]
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fp_u64(v: u64) -> fp::Fp { fp::Fp::from_u64(v) }
    fn fp2_u64(c0: u64, c1: u64) -> bls_fp2::Fp2 {
        bls_fp2::Fp2 { c0: fp_u64(c0), c1: fp_u64(c1) }
    }

    /// A "test" G2 point. Not on the real BLS12-381 twist (we'd need a
    /// host-side sqrt in Fp2 to lift), but the doubling/addition formulas
    /// are coefficient-independent so any P with P.y ≠ 0 works.
    fn sample_p() -> (bls_fp2::Fp2, bls_fp2::Fp2) {
        (fp2_u64(1, 1), fp2_u64(2, 1))
    }
    #[allow(dead_code)]
    fn sample_q() -> (bls_fp2::Fp2, bls_fp2::Fp2) {
        (fp2_u64(7, 3), fp2_u64(5, 2))
    }

    #[test]
    fn g2_column_layout_packed() {
        assert_eq!(LIMBS_PER_FP, 6);
        assert_eq!(LIMBS_PER_FP2, 12);
        assert_eq!(LIMBS_PER_G2, 24);
        assert_eq!(COL_P_OFFSET, 0);
        assert_eq!(COL_Q_OFFSET, 24);
        assert_eq!(COL_R_OFFSET, 48);
        assert_eq!(COL_IS_REAL, 72);
        assert_eq!(COL_R_INFINITY, 77);
        assert_eq!(COL_DBL_X_SQ, 78);
        // 10 doubling Fp2 + 8 addition Fp2 + 4 curve-eq Fp2 = 22 Fp2 blocks.
        // 22 × 12 = 264 intermediate-data cols.
        assert_eq!(COL_DBL_LAMBDA_DIFF, 78 + 9 * LIMBS_PER_FP2);
        assert_eq!(COL_ADD_DIFF_QY_PY, 78 + 10 * LIMBS_PER_FP2);
        assert_eq!(COL_RX_SQ, 78 + 18 * LIMBS_PER_FP2);
        assert_eq!(COL_B_PRIME, 78 + 21 * LIMBS_PER_FP2);
        assert_eq!(COL_SEL_CURVE, 78 + 22 * LIMBS_PER_FP2);
        assert_eq!(NUM_COLUMNS, 78 + 22 * LIMBS_PER_FP2 + 1);
        // 78 + 22*12 + 1 = 343.
        assert_eq!(NUM_COLUMNS, 343);
        assert_eq!(NUM_ROW_CONSTRAINTS, 10 + 2 * LIMBS_PER_FP);
        assert_eq!(NUM_ROW_CONSTRAINTS, 22);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn double_host_then_add_host_satisfy_chord_tangent_identities() {
        let (px, py) = sample_p();
        let (two_p_x, two_p_y) = double_host(&px, &py);

        // Manual chain for doubling.
        let x_sq = px.mul(&px);
        let three = bls_fp2::Fp2 { c0: fp_u64(3), c1: fp::Fp::zero() };
        let two = bls_fp2::Fp2 { c0: fp_u64(2), c1: fp::Fp::zero() };
        let lambda_num = three.mul(&x_sq);
        let lambda_denom = two.mul(&py);
        let lambda = lambda_num.mul(&lambda_denom.invert().unwrap());
        let lambda_sq = lambda.mul(&lambda);
        let two_px = two.mul(&px);
        let rx_check = lambda_sq.sub(&two_px);
        assert_eq!(rx_check, two_p_x, "double_host R.x identity");
        let ry_check = lambda.mul(&px.sub(&two_p_x)).sub(&py);
        assert_eq!(ry_check, two_p_y, "double_host R.y identity");

        // Now check 2P used as Q in addition.
        let (rx, ry) = add_host(&px, &py, &two_p_x, &two_p_y);
        let diff_y = two_p_y.sub(&py);
        let diff_x = two_p_x.sub(&px);
        let lambda_add = diff_y.mul(&diff_x.invert().unwrap());
        let lambda_add_sq = lambda_add.mul(&lambda_add);
        let sum_x = px.add(&two_p_x);
        let rx_check2 = lambda_add_sq.sub(&sum_x);
        assert_eq!(rx_check2, rx, "add_host R.x identity");
        let ry_check2 = lambda_add.mul(&px.sub(&rx)).sub(&py);
        assert_eq!(ry_check2, ry, "add_host R.y identity");
    }

    #[test]
    fn constraints_zero_on_honest_double_and_add_witness() {
        let (px, py) = sample_p();
        let (two_p_x, two_p_y) = double_host(&px, &py);
        let (three_p_x, three_p_y) = add_host(&px, &py, &two_p_x, &two_p_y);

        let mut w = Bls12G2OpsWitness::new();
        w.push(Bls12G2OpsRow::double_row(px, py));
        let add_row = Bls12G2OpsRow {
            px, py, qx: two_p_x, qy: two_p_y,
            rx: three_p_x, ry: three_p_y,
            op: OpSelector::Add, is_real: true,
            p_infinity: false, q_infinity: false, r_infinity: false,
        };
        w.push(add_row);

        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 2);

        let cs = Bls12G2OpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(),
                        "honest double+add: constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn padding_rows_clean_and_tampered_constant_fires() {
        // All-zero padding: every constraint vanishes.
        let w = Bls12G2OpsWitness::new();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = Bls12G2OpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(), "padding: constraint {} row {} nonzero", i, r);
            }
        }

        // Tamper: flip b_prime.c1[5] from 4 to 7 on a real row. The pinning
        // constraint for that limb must fire.
        let (px, py) = sample_p();
        let mut w2 = Bls12G2OpsWitness::new();
        w2.push(Bls12G2OpsRow::double_row(px, py));
        let trace2 = build_trace_polynomials(&w2, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace2.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        cols[COL_B_PRIME + 2 * LIMBS_PER_FP - 1][0] = Scalar::from_u64(7, curve);
        let col_refs2: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results2 = cs.evaluate_on_domain(&col_refs2, trace2.num_rows);
        // Constraint index for c1[5]: 10 + LIMBS_PER_FP + (LIMBS_PER_FP - 1).
        let idx = 10 + LIMBS_PER_FP + (LIMBS_PER_FP - 1);
        assert!(!results2[idx][0].is_zero(),
                "tampered b_prime.c1[5] must fire constraint {}", idx);
    }

    #[test]
    fn twist_b_pinning_witness_matches_constant() {
        let (px, py) = sample_p();
        let mut w = Bls12G2OpsWitness::new();
        w.push(Bls12G2OpsRow::double_row(px, py));
        let curve = CurveType::Bls12381;
        let trace = build_trace_polynomials(&w, curve);
        // sel_curve = 1 on row 0.
        let one = Scalar::one(curve);
        assert!(trace.columns[COL_SEL_CURVE].evaluations[0].sub(&one).is_zero());
        // b_prime.c0 = (0,...,4); b_prime.c1 = (0,...,4).
        let four = Scalar::from_u64(4, curve);
        for j in 0..LIMBS_PER_FP - 1 {
            assert!(trace.columns[COL_B_PRIME + j].evaluations[0].is_zero(),
                    "b_prime.c0[{}] should be 0", j);
        }
        assert!(trace.columns[COL_B_PRIME + LIMBS_PER_FP - 1]
            .evaluations[0].sub(&four).is_zero());
        for j in 0..LIMBS_PER_FP - 1 {
            assert!(trace.columns[COL_B_PRIME + LIMBS_PER_FP + j]
                .evaluations[0].is_zero(), "b_prime.c1[{}] should be 0", j);
        }
        assert!(trace.columns[COL_B_PRIME + 2 * LIMBS_PER_FP - 1]
            .evaluations[0].sub(&four).is_zero());

        // Witness-level: twist_b() == 4 + 4u (host-side direct check).
        let bp = twist_b();
        assert_eq!(bp.c0, fp_u64(4));
        assert_eq!(bp.c1, fp_u64(4));
    }

    #[test]
    fn g2_double_descriptors_well_formed() {
        let ds = bls12_g2_double_fp2_descriptors(0, 1);
        assert_eq!(ds.len(), 12, "12 Fp2 ops in the G2 doubling chain");
        let mut labels: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for d in &ds {
            assert_eq!(d.a_layer_index, 1);
            assert_eq!(d.b_layer_index, 0);
            assert_eq!(d.a_columns.len(), FP2_TRIPLE_LIMBS);
            assert_eq!(d.b_columns.len(), FP2_TRIPLE_LIMBS);
            assert_eq!(d.b_selector_column, Some(COL_SEL_DOUBLE));
            for &c in &d.b_columns {
                assert!(c < NUM_COLUMNS, "descriptor `{}` col {} OOB", d.label, c);
            }
            assert!(d.label.starts_with("bls12_g2_dbl_"));
            assert!(labels.insert(d.label.clone()), "duplicate `{}`", d.label);
            // A-side first chunks: A.c0, A.c1, B.c0, B.c1, C.c0, C.c1.
            assert_eq!(d.a_columns[0], FP2_COL_A_C0_OFFSET);
            assert_eq!(d.a_columns[LIMBS_PER_FP], FP2_COL_A_C1_OFFSET);
            assert_eq!(d.a_columns[2 * LIMBS_PER_FP], FP2_COL_B_C0_OFFSET);
            assert_eq!(d.a_columns[3 * LIMBS_PER_FP], FP2_COL_B_C1_OFFSET);
            assert_eq!(d.a_columns[4 * LIMBS_PER_FP], FP2_COL_C_C0_OFFSET);
            assert_eq!(d.a_columns[5 * LIMBS_PER_FP], FP2_COL_C_C1_OFFSET);
        }
        // First descriptor: x_sq = P.x · P.x (Mul).
        let d0 = &ds[0];
        assert_eq!(d0.label, "bls12_g2_dbl_x_sq");
        assert_eq!(d0.a_selector_column, Some(FP2_COL_SEL_MUL));
        assert_eq!(d0.b_columns[0], COL_PX_C0);
        assert_eq!(d0.b_columns[2 * LIMBS_PER_FP], COL_PX_C0);
        assert_eq!(d0.b_columns[4 * LIMBS_PER_FP], COL_DBL_X_SQ);
        // Last: R.y binding.
        let last = ds.last().unwrap();
        assert_eq!(last.label, "bls12_g2_dbl_r_y");
        assert_eq!(last.a_selector_column, Some(FP2_COL_SEL_SUB));
        assert_eq!(last.b_columns[4 * LIMBS_PER_FP], COL_RY_C0);
    }

    #[test]
    fn g2_add_descriptors_well_formed() {
        let ds = bls12_g2_add_fp2_descriptors(0, 1);
        assert_eq!(ds.len(), 10);
        let mut labels: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for d in &ds {
            assert_eq!(d.b_selector_column, Some(COL_SEL_ADD));
            assert!(d.label.starts_with("bls12_g2_add_"));
            assert!(labels.insert(d.label.clone()), "duplicate `{}`", d.label);
            for &c in &d.b_columns {
                assert!(c < NUM_COLUMNS);
            }
        }
        let d0 = &ds[0];
        assert_eq!(d0.label, "bls12_g2_add_diff_qy_py");
        assert_eq!(d0.a_selector_column, Some(FP2_COL_SEL_SUB));
        assert_eq!(d0.b_columns[0], COL_QY_C0);
        assert_eq!(d0.b_columns[2 * LIMBS_PER_FP], COL_PY_C0);
        assert_eq!(d0.b_columns[4 * LIMBS_PER_FP], COL_ADD_DIFF_QY_PY);
    }

    #[test]
    fn g2_curve_eq_descriptors_well_formed() {
        let ds = bls12_g2_curve_eq_fp2_descriptors(0, 1);
        assert_eq!(ds.len(), 4);
        let mut labels: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for d in &ds {
            assert_eq!(d.b_layer_index, 0);
            assert_eq!(d.a_layer_index, 1);
            assert_eq!(d.b_selector_column, Some(COL_SEL_CURVE));
            assert!(labels.insert(d.label.clone()), "duplicate `{}`", d.label);
            for &c in &d.b_columns {
                assert!(c < NUM_COLUMNS);
            }
        }
        assert!(labels.contains("bls12_g2_curve_rx_sq"));
        assert!(labels.contains("bls12_g2_curve_rx_cubed"));
        assert!(labels.contains("bls12_g2_curve_ry_sq"));
        assert!(labels.contains("bls12_g2_curve_eq"));
        // curve_eq Sub triple's c-slot must point to b_prime.
        let ce = ds.iter().find(|d| d.label == "bls12_g2_curve_eq").unwrap();
        assert_eq!(ce.a_selector_column, Some(FP2_COL_SEL_SUB));
        assert_eq!(ce.b_columns[4 * LIMBS_PER_FP], COL_B_PRIME);
    }
}
