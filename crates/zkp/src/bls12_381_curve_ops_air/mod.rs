//! BLS12-381 curve operations — host-side scaffold AIR (#270 + #271).
//!
//! Mirrors [`crate::bn254_curve_ops_air`] for the BLS12-381 curve. The
//! sibling Fp / Fp2 / G2 modules expose a layered scaffold so that a
//! downstream BLS12-381 pairing precompile can attach algebraic
//! decomposition via the same cross-AIR LogUp pattern used by BN254:
//!
//! ```text
//!   Fp   = GF(p)                              p ≈ 2^381 (6 u64 limbs)
//!   Fp2  = Fp[u]  / (u^2 + 1)                 nonresidue = -1
//!   Fp6  = Fp2[v] / (v^3 - ξ)                 ξ = 1 + u  ∈ Fp2  *
//!   Fp12 = Fp6[w] / (w^2 - v)
//! ```
//!
//! \* BLS12-381's Fp6 nonresidue is `ξ = 1 + u`, distinct from BN254's
//! `9 + u`. The Fp / Fp2 layer is structurally identical between the
//! two curves modulo limb width.
//!
//! # Scope
//!
//! Phase #270 (this commit) adds the Fp scaffold, the G1 shape AIR,
//! the G1 add / double **algebraic intermediates** (mirroring task
//! #231 for BN254) and the G1 **curve-equation pin** `y² = x³ + 4`
//! (mirroring task #247 for BN254). Phase #271 adds Fp2 + G2. Each
//! sibling module commits the IO of a single curve / field operation
//! per row and ships the algebraic decomposition to the next-lower
//! AIR through cross-AIR LogUp descriptors against the BLS12-381 Fp
//! AIR ([`crate::nonnative_fp_air`]).
//!
//! Note: `b = 4` for BLS12-381 G1, **not** `3` like BN254.
//!
//! The host-side reference for Fp / Fp2 is re-exported from
//! [`crate::nonnative_fp`] — that crate already provides canonical
//! reduction modulo BLS12-381 `p` for both layers.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub mod fp;
pub mod fp2;
pub mod g2;

// ─── BLS12-381 base field constants ───────────────────────────────────

/// BLS12-381 base field modulus `p` as 6 big-endian u64 limbs.
///
/// `p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab`
///   ≈ 2^381.
pub const BLS12_381_P_LIMBS: [u64; 6] = crate::nonnative_fp::P_LIMBS;

/// Number of u64 limbs per BLS12-381 Fp element.
pub const LIMBS_PER_BLS12_FP: usize = 6;

// ─── Host-side BLS12-381 G1 affine ────────────────────────────────────

pub use crate::nonnative_fp::Fp as HostFp;
pub use crate::nonnative_fp::Fp2 as HostFp2;

/// Affine point on BLS12-381 G1: `y² = x³ + 4` over Fp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct G1Affine {
    pub x: HostFp,
    pub y: HostFp,
    pub infinity: bool,
}

impl G1Affine {
    pub const fn identity() -> Self {
        G1Affine { x: HostFp { limbs: [0u64; 6] }, y: HostFp { limbs: [0u64; 6] }, infinity: true }
    }

    /// Canonical BLS12-381 G1 generator (IETF pairing-friendly-curves
    /// §4.2.1), via [`crate::pairing::G1Affine::generator`].
    pub fn generator() -> Self {
        let g = crate::pairing::G1Affine::generator();
        G1Affine { x: g.x, y: g.y, infinity: g.infinity }
    }

    /// Affine curve equation check: `y² = x³ + 4`.
    pub fn is_on_curve(&self) -> bool {
        if self.infinity {
            return true;
        }
        let y2 = self.y.mul(&self.y);
        let rhs = self.x.mul(&self.x).mul(&self.x).add(&HostFp::from_u64(4));
        y2 == rhs
    }

    /// Compute `R = 2·P` using the host-side BLS12-381 Fp arithmetic.
    /// Panics if `P` is the identity or if `2·P.y == 0`.
    pub fn double_host(&self) -> Self {
        let p_x = self.x;
        let p_y = self.y;

        let x_sq = p_x.mul(&p_x);
        let two_x_sq = x_sq.add(&x_sq);
        let lambda_num = two_x_sq.add(&x_sq); // 3·x²
        let lambda_denom = p_y.add(&p_y);     // 2·y
        let lambda_denom_inv = lambda_denom.invert().expect("2·P.y must be invertible");
        let lambda = lambda_num.mul(&lambda_denom_inv);
        let lambda_sq = lambda.mul(&lambda);
        let two_px = p_x.add(&p_x);
        let r_x = lambda_sq.sub(&two_px);
        let diff = p_x.sub(&r_x);
        let r_y = lambda.mul(&diff).sub(&p_y);

        G1Affine { x: r_x, y: r_y, infinity: false }
    }

    /// Compute `R = P + Q` using the host-side BLS12-381 Fp arithmetic.
    /// Panics if `Q.x == P.x`.
    pub fn add_host(&self, other: &G1Affine) -> Self {
        let p_x = self.x;
        let p_y = self.y;
        let q_x = other.x;
        let q_y = other.y;

        let diff_y = q_y.sub(&p_y);
        let diff_x = q_x.sub(&p_x);
        let diff_x_inv = diff_x.invert().expect("Q.x − P.x must be invertible");
        let lambda = diff_y.mul(&diff_x_inv);
        let lambda_sq = lambda.mul(&lambda);
        let sum_x = p_x.add(&q_x);
        let r_x = lambda_sq.sub(&sum_x);
        let diff_px_rx = p_x.sub(&r_x);
        let r_y = lambda.mul(&diff_px_rx).sub(&p_y);

        G1Affine { x: r_x, y: r_y, infinity: false }
    }
}

/// Affine point on BLS12-381 G2 twist: `y² = x³ + 4(1+u)` over Fp2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct G2Affine {
    pub x: HostFp2,
    pub y: HostFp2,
    pub infinity: bool,
}

impl G2Affine {
    pub fn identity() -> Self {
        G2Affine {
            x: HostFp2::zero(),
            y: HostFp2::zero(),
            infinity: true,
        }
    }
}

// ─── G1 scaffold row + witness ────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpSelector {
    Add,
    Double,
}

#[derive(Clone, Debug)]
pub struct Bls12CurveOpsRow {
    pub p: G1Affine,
    pub q: G1Affine,
    pub r: G1Affine,
    pub op: OpSelector,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Bls12CurveOpsWitness {
    pub rows: Vec<Bls12CurveOpsRow>,
}

impl Bls12CurveOpsWitness {
    pub fn new() -> Self { Self { rows: Vec::new() } }
    pub fn push(&mut self, row: Bls12CurveOpsRow) { self.rows.push(row); }
}

// ─── Column layout (G1 scaffold) ──────────────────────────────────────

pub const LIMBS_PER_G1: usize = 2 * LIMBS_PER_BLS12_FP;

pub const COL_P_OFFSET: usize = 0;
pub const COL_Q_OFFSET: usize = COL_P_OFFSET + LIMBS_PER_G1;
pub const COL_R_OFFSET: usize = COL_Q_OFFSET + LIMBS_PER_G1;
pub const COL_IS_REAL: usize = COL_R_OFFSET + LIMBS_PER_G1;
pub const COL_SEL_ADD: usize = COL_IS_REAL + 1;
pub const COL_SEL_DOUBLE: usize = COL_SEL_ADD + 1;
pub const COL_P_INFINITY: usize = COL_SEL_DOUBLE + 1;
pub const COL_Q_INFINITY: usize = COL_P_INFINITY + 1;
pub const COL_R_INFINITY: usize = COL_Q_INFINITY + 1;

// ─── Algebraic intermediates for G1 doubling (#270, mirrors #231) ─────
//
// 10 Fp elements × 6 BE u64 limbs each = 60 columns. Each will be bound
// to the BLS12-381 Fp AIR (`nonnative_fp_air`) under the matching op
// selector via the LogUp descriptors at the bottom of this module.
//
//   x_sq              = P.x · P.x                       (Mul)
//   two_x_sq          = x_sq + x_sq                     (Add)
//   lambda_num        = two_x_sq + x_sq    = 3·P.x²     (Add)
//   lambda_denom      = P.y + P.y          = 2·P.y      (Add)
//   lambda_denom_inv    s.t. lambda_denom · lambda_denom_inv = 1   (Inv)
//   lambda            = lambda_num · lambda_denom_inv             (Mul)
//   lambda_sq         = lambda · lambda                            (Mul)
//   two_px            = P.x + P.x                                  (Add)
//   r_x  ≡ R.x        = lambda_sq − two_px              (Sub → R.x)
//   diff_px_rx        = P.x − R.x                                  (Sub)
//   lambda_diff       = lambda · diff_px_rx                        (Mul)
//   r_y  ≡ R.y        = lambda_diff − P.y               (Sub → R.y)
pub const COL_DBL_X_SQ:             usize = COL_R_INFINITY + 1;
pub const COL_DBL_TWO_X_SQ:         usize = COL_DBL_X_SQ + LIMBS_PER_BLS12_FP;
pub const COL_DBL_LAMBDA_NUM:       usize = COL_DBL_TWO_X_SQ + LIMBS_PER_BLS12_FP;
pub const COL_DBL_LAMBDA_DENOM:     usize = COL_DBL_LAMBDA_NUM + LIMBS_PER_BLS12_FP;
pub const COL_DBL_LAMBDA_DENOM_INV: usize = COL_DBL_LAMBDA_DENOM + LIMBS_PER_BLS12_FP;
pub const COL_DBL_LAMBDA:           usize = COL_DBL_LAMBDA_DENOM_INV + LIMBS_PER_BLS12_FP;
pub const COL_DBL_LAMBDA_SQ:        usize = COL_DBL_LAMBDA + LIMBS_PER_BLS12_FP;
pub const COL_DBL_TWO_PX:           usize = COL_DBL_LAMBDA_SQ + LIMBS_PER_BLS12_FP;
pub const COL_DBL_DIFF_PX_RX:       usize = COL_DBL_TWO_PX + LIMBS_PER_BLS12_FP;
pub const COL_DBL_LAMBDA_DIFF:      usize = COL_DBL_DIFF_PX_RX + LIMBS_PER_BLS12_FP;

// ─── Algebraic intermediates for G1 addition (#270, mirrors #231) ─────
//
// 8 Fp elements × 6 limbs = 48 columns.
pub const COL_ADD_DIFF_QY_PY:        usize = COL_DBL_LAMBDA_DIFF + LIMBS_PER_BLS12_FP;
pub const COL_ADD_DIFF_QX_PX:        usize = COL_ADD_DIFF_QY_PY + LIMBS_PER_BLS12_FP;
pub const COL_ADD_DIFF_QX_PX_INV:    usize = COL_ADD_DIFF_QX_PX + LIMBS_PER_BLS12_FP;
pub const COL_ADD_LAMBDA:            usize = COL_ADD_DIFF_QX_PX_INV + LIMBS_PER_BLS12_FP;
pub const COL_ADD_LAMBDA_SQ:         usize = COL_ADD_LAMBDA + LIMBS_PER_BLS12_FP;
pub const COL_ADD_SUM_PX_QX:         usize = COL_ADD_LAMBDA_SQ + LIMBS_PER_BLS12_FP;
pub const COL_ADD_DIFF_PX_RX:        usize = COL_ADD_SUM_PX_QX + LIMBS_PER_BLS12_FP;
pub const COL_ADD_LAMBDA_DIFF:       usize = COL_ADD_DIFF_PX_RX + LIMBS_PER_BLS12_FP;

// ─── Curve-equation intermediates (#270, mirrors #247) ────────────────
//
// Algebraic binding for `R = (R.x, R.y)` ∈ G1: enforces
// `R.y² = R.x³ + 4` (over Fp) on every real, non-infinity row.
//
//   rx_sq          = R.x · R.x                          (Mul)
//   rx_cubed       = rx_sq · R.x                        (Mul)
//   ry_sq          = R.y · R.y                          (Mul)
//   ry_sq − rx_cubed = four_const = (0,0,0,0,0, 4)·sel  (Sub)
//
// `four_const` is pinned row-locally so that the Fp-AIR Sub row's c-slot
// really equals the field element `4` (since `b = 4` for BLS12-381 G1).
pub const COL_RX_SQ:        usize = COL_ADD_LAMBDA_DIFF + LIMBS_PER_BLS12_FP;
pub const COL_RX_CUBED:     usize = COL_RX_SQ + LIMBS_PER_BLS12_FP;
pub const COL_RY_SQ:        usize = COL_RX_CUBED + LIMBS_PER_BLS12_FP;
pub const COL_FOUR_CONST:   usize = COL_RY_SQ + LIMBS_PER_BLS12_FP;
pub const COL_SEL_CURVE:    usize = COL_FOUR_CONST + LIMBS_PER_BLS12_FP;

pub const NUM_COLUMNS: usize = COL_SEL_CURVE + 1;

/// Row-local constraints (shape + curve-equation pin).
///
///   0. is_real ∈ {0, 1}
///   1. sel_add ∈ {0, 1}
///   2. sel_double ∈ {0, 1}
///   3. sel_add · sel_double = 0
///   4. sel_add + sel_double − is_real = 0
///   5. p_infinity ∈ {0, 1}
///   6. q_infinity ∈ {0, 1}
///   7. r_infinity ∈ {0, 1}
///   8. sel_curve ∈ {0, 1}
///   9. sel_curve + is_real·r_infinity − is_real = 0
///       (sel_curve = is_real · (1 − r_infinity))
///  10. four_const[0] = 0
///  11. four_const[1] = 0
///  12. four_const[2] = 0
///  13. four_const[3] = 0
///  14. four_const[4] = 0
///  15. four_const[5] − 4·sel_curve = 0       (BE LSB = 5)
///
/// The G1 curve-law algebra (λ·denom = num, λ² = λ·λ, …, ry² = rx³ + 4)
/// is closed via cross-AIR LogUp into [`crate::nonnative_fp_air`]; see
/// [`bls12_381_g1_double_fp_descriptors`],
/// [`bls12_381_g1_add_fp_descriptors`], and
/// [`bls12_381_g1_curve_eq_fp_descriptors`].
pub const NUM_ROW_CONSTRAINTS: usize = 16;

pub const NUM_SHIFTED: usize = 0;

// ─── Trace builder ────────────────────────────────────────────────────

fn write_fp_limbs(columns: &mut [Vec<Scalar>], base: usize, r: usize, v: &HostFp, curve: CurveType) {
    for j in 0..LIMBS_PER_BLS12_FP {
        columns[base + j][r] = Scalar::from_u64(v.limbs[j], curve);
    }
}

fn write_g1_limbs(columns: &mut [Vec<Scalar>], base: usize, r: usize, g: &G1Affine, curve: CurveType) {
    write_fp_limbs(columns, base, r, &g.x, curve);
    write_fp_limbs(columns, base + LIMBS_PER_BLS12_FP, r, &g.y, curve);
}

/// Compute and write the G1-doubling algebraic intermediates for row `r`.
fn populate_double_intermediates(
    columns: &mut [Vec<Scalar>],
    r: usize,
    p: &G1Affine,
    curve: CurveType,
) {
    let p_x = p.x;
    let p_y = p.y;

    let x_sq = p_x.mul(&p_x);
    let two_x_sq = x_sq.add(&x_sq);
    let lambda_num = two_x_sq.add(&x_sq); // 3·x²
    let lambda_denom = p_y.add(&p_y);     // 2·y
    let lambda_denom_inv = lambda_denom.invert().expect("2·y nonzero");
    let lambda = lambda_num.mul(&lambda_denom_inv);
    let lambda_sq = lambda.mul(&lambda);
    let two_px = p_x.add(&p_x);
    let r_x = lambda_sq.sub(&two_px);
    let diff_px_rx = p_x.sub(&r_x);
    let lambda_diff = lambda.mul(&diff_px_rx);

    write_fp_limbs(columns, COL_DBL_X_SQ, r, &x_sq, curve);
    write_fp_limbs(columns, COL_DBL_TWO_X_SQ, r, &two_x_sq, curve);
    write_fp_limbs(columns, COL_DBL_LAMBDA_NUM, r, &lambda_num, curve);
    write_fp_limbs(columns, COL_DBL_LAMBDA_DENOM, r, &lambda_denom, curve);
    write_fp_limbs(columns, COL_DBL_LAMBDA_DENOM_INV, r, &lambda_denom_inv, curve);
    write_fp_limbs(columns, COL_DBL_LAMBDA, r, &lambda, curve);
    write_fp_limbs(columns, COL_DBL_LAMBDA_SQ, r, &lambda_sq, curve);
    write_fp_limbs(columns, COL_DBL_TWO_PX, r, &two_px, curve);
    write_fp_limbs(columns, COL_DBL_DIFF_PX_RX, r, &diff_px_rx, curve);
    write_fp_limbs(columns, COL_DBL_LAMBDA_DIFF, r, &lambda_diff, curve);
}

/// Compute and write the G1-addition algebraic intermediates for row `r`.
fn populate_add_intermediates(
    columns: &mut [Vec<Scalar>],
    r: usize,
    p: &G1Affine,
    q: &G1Affine,
    curve: CurveType,
) {
    let p_x = p.x;
    let p_y = p.y;
    let q_x = q.x;
    let q_y = q.y;

    let diff_y = q_y.sub(&p_y);
    let diff_x = q_x.sub(&p_x);
    let diff_x_inv = diff_x.invert().expect("Q.x − P.x nonzero");
    let lambda = diff_y.mul(&diff_x_inv);
    let lambda_sq = lambda.mul(&lambda);
    let sum_x = p_x.add(&q_x);
    let r_x = lambda_sq.sub(&sum_x);
    let diff_px_rx = p_x.sub(&r_x);
    let lambda_diff = lambda.mul(&diff_px_rx);

    write_fp_limbs(columns, COL_ADD_DIFF_QY_PY, r, &diff_y, curve);
    write_fp_limbs(columns, COL_ADD_DIFF_QX_PX, r, &diff_x, curve);
    write_fp_limbs(columns, COL_ADD_DIFF_QX_PX_INV, r, &diff_x_inv, curve);
    write_fp_limbs(columns, COL_ADD_LAMBDA, r, &lambda, curve);
    write_fp_limbs(columns, COL_ADD_LAMBDA_SQ, r, &lambda_sq, curve);
    write_fp_limbs(columns, COL_ADD_SUM_PX_QX, r, &sum_x, curve);
    write_fp_limbs(columns, COL_ADD_DIFF_PX_RX, r, &diff_px_rx, curve);
    write_fp_limbs(columns, COL_ADD_LAMBDA_DIFF, r, &lambda_diff, curve);
}

pub fn build_trace_polynomials(
    witness: &Bls12CurveOpsWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        write_g1_limbs(&mut columns, COL_P_OFFSET, r, &row.p, curve);
        write_g1_limbs(&mut columns, COL_Q_OFFSET, r, &row.q, curve);
        write_g1_limbs(&mut columns, COL_R_OFFSET, r, &row.r, curve);
        columns[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };
        match row.op {
            OpSelector::Add => {
                columns[COL_SEL_ADD][r] = if row.is_real { one.clone() } else { zero.clone() };
            }
            OpSelector::Double => {
                columns[COL_SEL_DOUBLE][r] = if row.is_real { one.clone() } else { zero.clone() };
            }
        }
        columns[COL_P_INFINITY][r] = if row.p.infinity { one.clone() } else { zero.clone() };
        columns[COL_Q_INFINITY][r] = if row.q.infinity { one.clone() } else { zero.clone() };
        columns[COL_R_INFINITY][r] = if row.r.infinity { one.clone() } else { zero.clone() };

        // Algebraic intermediates + curve-equation pin on real, non-infinity rows.
        if row.is_real && !row.p.infinity && !row.r.infinity {
            match row.op {
                OpSelector::Double => {
                    populate_double_intermediates(&mut columns, r, &row.p, curve);
                }
                OpSelector::Add => {
                    if !row.q.infinity {
                        populate_add_intermediates(&mut columns, r, &row.p, &row.q, curve);
                    }
                }
            }

            let r_x = row.r.x;
            let r_y = row.r.y;
            let rx_sq = r_x.mul(&r_x);
            let rx_cubed = rx_sq.mul(&r_x);
            let ry_sq = r_y.mul(&r_y);
            let four_const = HostFp::from_u64(4);
            write_fp_limbs(&mut columns, COL_RX_SQ, r, &rx_sq, curve);
            write_fp_limbs(&mut columns, COL_RX_CUBED, r, &rx_cubed, curve);
            write_fp_limbs(&mut columns, COL_RY_SQ, r, &ry_sq, curve);
            write_fp_limbs(&mut columns, COL_FOUR_CONST, r, &four_const, curve);
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

pub struct Bls12CurveOpsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bls12CurveOpsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
}

fn bin(v: &Scalar) -> Scalar {
    let curve = v.curve_type();
    let one = Scalar::one(curve);
    v.mul(&v.sub(&one))
}

impl VmConstraintSystem for Bls12CurveOpsConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        [
            "is_real_binary",
            "sel_add_binary",
            "sel_double_binary",
            "selectors_mutex",
            "selectors_sum_to_is_real",
            "p_infinity_binary",
            "q_infinity_binary",
            "r_infinity_binary",
            "sel_curve_binary",
            "sel_curve_eq_real_and_not_r_infinity",
            "four_const_limb0_zero",
            "four_const_limb1_zero",
            "four_const_limb2_zero",
            "four_const_limb3_zero",
            "four_const_limb4_zero",
            "four_const_limb5_eq_four_times_sel_curve",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let n = columns[0].len();
        let curve = columns[0][0].curve_type();
        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n]).collect();

        let four_scalar = Scalar::from_u64(4, curve);
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
            // sel_curve + is_real·r_infinity − is_real = 0.
            out[9][r] = sc.add(&is_real.mul(ri)).sub(is_real);
            // four_const high limbs (0..5 in BE order) must be zero.
            for j in 0..LIMBS_PER_BLS12_FP - 1 {
                out[10 + j][r] = columns[COL_FOUR_CONST + j][r].clone();
            }
            // BE LSB (limb 5) must equal 4·sel_curve.
            let last_limb = &columns[COL_FOUR_CONST + LIMBS_PER_BLS12_FP - 1][r];
            out[10 + LIMBS_PER_BLS12_FP - 1][r] =
                last_limb.sub(&four_scalar.mul(sc));
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
        let four_scalar = Scalar::from_u64(4, curve);
        let mut parts: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        parts.push(bin(is_real));
        parts.push(bin(sa));
        parts.push(bin(sd));
        parts.push(sa.mul(sd));
        parts.push(sa.add(sd).sub(is_real));
        parts.push(bin(pi));
        parts.push(bin(qi));
        parts.push(bin(ri));
        parts.push(bin(sc));
        parts.push(sc.add(&is_real.mul(ri)).sub(is_real));
        for j in 0..LIMBS_PER_BLS12_FP - 1 {
            parts.push(col_evals[COL_FOUR_CONST + j].clone());
        }
        let last_limb = &col_evals[COL_FOUR_CONST + LIMBS_PER_BLS12_FP - 1];
        parts.push(last_limb.sub(&four_scalar.mul(sc)));

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

        let four_scalar = Scalar::from_u64(4, curve);
        let four_times_sc =
            poly_scalar_mul(&col_coeffs[COL_SEL_CURVE], &four_scalar);

        let mut parts: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        parts.push(bin_poly(&col_coeffs[COL_IS_REAL]));
        parts.push(bin_poly(&col_coeffs[COL_SEL_ADD]));
        parts.push(bin_poly(&col_coeffs[COL_SEL_DOUBLE]));
        parts.push(poly_mul(&col_coeffs[COL_SEL_ADD], &col_coeffs[COL_SEL_DOUBLE], curve));
        parts.push(poly_sub(
            &poly_add(&col_coeffs[COL_SEL_ADD], &col_coeffs[COL_SEL_DOUBLE], curve),
            &col_coeffs[COL_IS_REAL],
            curve,
        ));
        parts.push(bin_poly(&col_coeffs[COL_P_INFINITY]));
        parts.push(bin_poly(&col_coeffs[COL_Q_INFINITY]));
        parts.push(bin_poly(&col_coeffs[COL_R_INFINITY]));
        parts.push(bin_poly(&col_coeffs[COL_SEL_CURVE]));
        parts.push(poly_sub(
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
        ));
        for j in 0..LIMBS_PER_BLS12_FP - 1 {
            parts.push(col_coeffs[COL_FOUR_CONST + j].clone());
        }
        parts.push(poly_sub(
            &col_coeffs[COL_FOUR_CONST + LIMBS_PER_BLS12_FP - 1],
            &four_times_sc,
            curve,
        ));

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

// ─── Cross-AIR LogUp descriptors: curve_ops ↔ fp (#270) ───────────────
//
// Each descriptor binds one `(a, b, c)` Fp triple committed in the
// curve_ops AIR's per-row intermediate columns to the BLS12-381 Fp
// AIR's `(COL_A, COL_B, COL_R)` triple under the appropriate selector
// (`COL_SEL_ADD`, `COL_SEL_SUB`, `COL_SEL_MUL`, or `COL_SEL_INV`). The
// 18-column tuple `(a[0..6], b[0..6], c[0..6])` carries the full Fp
// triple (6 BE u64 limbs each).
//
// Soundness scope: under joint γ, multiset equality on the 18-tuples
// forces every committed curve_ops triple to coincide with some Fp-AIR
// row of the matching op-kind. Together with the Fp AIR's row-local
// constraints, the curve-law identities hold algebraically.

use crate::cross_air_logup::CrossAirLogUpDescriptor;

/// Selector tag for which Fp AIR operation a given triple decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpOpKind {
    Add,
    Sub,
    Mul,
    Inv,
}

fn fp_sel_col(kind: FpOpKind) -> usize {
    match kind {
        FpOpKind::Add => fp::COL_SEL_ADD,
        FpOpKind::Sub => fp::COL_SEL_SUB,
        FpOpKind::Mul => fp::COL_SEL_MUL,
        FpOpKind::Inv => fp::COL_SEL_INV,
    }
}

/// Build one Fp-triple LogUp descriptor for a curve_ops row.
pub fn make_bls12_381_curve_ops_fp_triple_descriptor(
    label: impl Into<String>,
    curve_ops_layer_index: usize,
    fp_layer_index: usize,
    a_base: usize,
    b_base: usize,
    c_base: usize,
    kind: FpOpKind,
    curve_sel_col: usize,
) -> CrossAirLogUpDescriptor {
    let mut b_columns: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_BLS12_FP);
    for j in 0..LIMBS_PER_BLS12_FP { b_columns.push(a_base + j); }
    for j in 0..LIMBS_PER_BLS12_FP { b_columns.push(b_base + j); }
    for j in 0..LIMBS_PER_BLS12_FP { b_columns.push(c_base + j); }

    let mut a_columns: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_BLS12_FP);
    for j in 0..LIMBS_PER_BLS12_FP { a_columns.push(fp::COL_A_OFFSET + j); }
    for j in 0..LIMBS_PER_BLS12_FP { a_columns.push(fp::COL_B_OFFSET + j); }
    for j in 0..LIMBS_PER_BLS12_FP { a_columns.push(fp::COL_C_OFFSET + j); }

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: fp_layer_index,
        a_columns,
        a_selector_column: Some(fp_sel_col(kind)),
        b_layer_index: curve_ops_layer_index,
        b_columns,
        b_selector_column: Some(curve_sel_col),
    }
}

/// Column base of `P.x` BE limbs on a curve_ops row.
pub const COL_PX: usize = COL_P_OFFSET;
/// Column base of `P.y` BE limbs.
pub const COL_PY: usize = COL_P_OFFSET + LIMBS_PER_BLS12_FP;
/// Column base of `Q.x` BE limbs.
pub const COL_QX: usize = COL_Q_OFFSET;
/// Column base of `Q.y` BE limbs.
pub const COL_QY: usize = COL_Q_OFFSET + LIMBS_PER_BLS12_FP;
/// Column base of `R.x` BE limbs.
pub const COL_RX: usize = COL_R_OFFSET;
/// Column base of `R.y` BE limbs.
pub const COL_RY: usize = COL_R_OFFSET + LIMBS_PER_BLS12_FP;

/// All Fp-triple LogUp descriptors witnessing the G1 doubling chain
/// (gated by `COL_SEL_DOUBLE`). 12 descriptors total.
pub fn bls12_381_g1_double_fp_descriptors(
    curve_ops_layer_index: usize,
    fp_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    let mk = |label: &str, kind: FpOpKind, a: usize, b: usize, c: usize| {
        make_bls12_381_curve_ops_fp_triple_descriptor(
            label, curve_ops_layer_index, fp_layer_index, a, b, c, kind, COL_SEL_DOUBLE,
        )
    };
    vec![
        mk("bls12_381_g1_dbl_x_sq",         FpOpKind::Mul, COL_PX,                  COL_PX,                  COL_DBL_X_SQ),
        mk("bls12_381_g1_dbl_two_x_sq",     FpOpKind::Add, COL_DBL_X_SQ,            COL_DBL_X_SQ,            COL_DBL_TWO_X_SQ),
        mk("bls12_381_g1_dbl_lambda_num",   FpOpKind::Add, COL_DBL_TWO_X_SQ,        COL_DBL_X_SQ,            COL_DBL_LAMBDA_NUM),
        mk("bls12_381_g1_dbl_lambda_denom", FpOpKind::Add, COL_PY,                  COL_PY,                  COL_DBL_LAMBDA_DENOM),
        mk("bls12_381_g1_dbl_inv_denom",    FpOpKind::Inv, COL_DBL_LAMBDA_DENOM,    COL_DBL_LAMBDA_DENOM_INV, COL_DBL_LAMBDA_DENOM_INV),
        mk("bls12_381_g1_dbl_lambda",       FpOpKind::Mul, COL_DBL_LAMBDA_NUM,      COL_DBL_LAMBDA_DENOM_INV, COL_DBL_LAMBDA),
        mk("bls12_381_g1_dbl_lambda_sq",    FpOpKind::Mul, COL_DBL_LAMBDA,          COL_DBL_LAMBDA,          COL_DBL_LAMBDA_SQ),
        mk("bls12_381_g1_dbl_two_px",       FpOpKind::Add, COL_PX,                  COL_PX,                  COL_DBL_TWO_PX),
        mk("bls12_381_g1_dbl_r_x",          FpOpKind::Sub, COL_DBL_LAMBDA_SQ,       COL_DBL_TWO_PX,          COL_RX),
        mk("bls12_381_g1_dbl_diff_px_rx",   FpOpKind::Sub, COL_PX,                  COL_RX,                  COL_DBL_DIFF_PX_RX),
        mk("bls12_381_g1_dbl_lambda_diff",  FpOpKind::Mul, COL_DBL_LAMBDA,          COL_DBL_DIFF_PX_RX,      COL_DBL_LAMBDA_DIFF),
        mk("bls12_381_g1_dbl_r_y",          FpOpKind::Sub, COL_DBL_LAMBDA_DIFF,     COL_PY,                  COL_RY),
    ]
}

/// All Fp-triple LogUp descriptors witnessing the G1 addition chain
/// (gated by `COL_SEL_ADD`). 10 descriptors total.
pub fn bls12_381_g1_add_fp_descriptors(
    curve_ops_layer_index: usize,
    fp_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    let mk = |label: &str, kind: FpOpKind, a: usize, b: usize, c: usize| {
        make_bls12_381_curve_ops_fp_triple_descriptor(
            label, curve_ops_layer_index, fp_layer_index, a, b, c, kind, COL_SEL_ADD,
        )
    };
    vec![
        mk("bls12_381_g1_add_diff_qy_py",   FpOpKind::Sub, COL_QY,                  COL_PY,                  COL_ADD_DIFF_QY_PY),
        mk("bls12_381_g1_add_diff_qx_px",   FpOpKind::Sub, COL_QX,                  COL_PX,                  COL_ADD_DIFF_QX_PX),
        mk("bls12_381_g1_add_inv_diff",     FpOpKind::Inv, COL_ADD_DIFF_QX_PX,      COL_ADD_DIFF_QX_PX_INV,  COL_ADD_DIFF_QX_PX_INV),
        mk("bls12_381_g1_add_lambda",       FpOpKind::Mul, COL_ADD_DIFF_QY_PY,      COL_ADD_DIFF_QX_PX_INV,  COL_ADD_LAMBDA),
        mk("bls12_381_g1_add_lambda_sq",    FpOpKind::Mul, COL_ADD_LAMBDA,          COL_ADD_LAMBDA,          COL_ADD_LAMBDA_SQ),
        mk("bls12_381_g1_add_sum_px_qx",    FpOpKind::Add, COL_PX,                  COL_QX,                  COL_ADD_SUM_PX_QX),
        mk("bls12_381_g1_add_r_x",          FpOpKind::Sub, COL_ADD_LAMBDA_SQ,       COL_ADD_SUM_PX_QX,       COL_RX),
        mk("bls12_381_g1_add_diff_px_rx",   FpOpKind::Sub, COL_PX,                  COL_RX,                  COL_ADD_DIFF_PX_RX),
        mk("bls12_381_g1_add_lambda_diff",  FpOpKind::Mul, COL_ADD_LAMBDA,          COL_ADD_DIFF_PX_RX,      COL_ADD_LAMBDA_DIFF),
        mk("bls12_381_g1_add_r_y",          FpOpKind::Sub, COL_ADD_LAMBDA_DIFF,     COL_PY,                  COL_RY),
    ]
}

/// All Fp-triple LogUp descriptors witnessing the **G1 curve equation**
/// `R.y² = R.x³ + 4` (gated by `COL_SEL_CURVE`). 4 descriptors total.
///
/// Triples:
/// 1. `curve_rx_sq`    : Mul · `(R.x, R.x, rx_sq)`
/// 2. `curve_rx_cubed` : Mul · `(rx_sq, R.x, rx_cubed)`
/// 3. `curve_ry_sq`    : Mul · `(R.y, R.y, ry_sq)`
/// 4. `curve_eq`       : Sub · `(ry_sq, rx_cubed, four_const)`
///                       — i.e. `ry_sq − rx_cubed = 4`.
pub fn bls12_381_g1_curve_eq_fp_descriptors(
    curve_ops_layer_index: usize,
    fp_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    let mk = |label: &str, kind: FpOpKind, a: usize, b: usize, c: usize| {
        make_bls12_381_curve_ops_fp_triple_descriptor(
            label, curve_ops_layer_index, fp_layer_index, a, b, c, kind, COL_SEL_CURVE,
        )
    };
    vec![
        mk("bls12_381_g1_curve_rx_sq",    FpOpKind::Mul, COL_RX,        COL_RX,        COL_RX_SQ),
        mk("bls12_381_g1_curve_rx_cubed", FpOpKind::Mul, COL_RX_SQ,     COL_RX,        COL_RX_CUBED),
        mk("bls12_381_g1_curve_ry_sq",    FpOpKind::Mul, COL_RY,        COL_RY,        COL_RY_SQ),
        mk("bls12_381_g1_curve_eq",       FpOpKind::Sub, COL_RY_SQ,     COL_RX_CUBED,  COL_FOUR_CONST),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_double_row() -> Bls12CurveOpsRow {
        let g = G1Affine::generator();
        let r = g.double_host();
        Bls12CurveOpsRow {
            p: g, q: G1Affine::identity(), r,
            op: OpSelector::Double, is_real: true,
        }
    }

    #[test]
    fn g1_scaffold_column_layout_packed() {
        // P + Q + R = 3·12 = 36 limbs; +6 shape selector/infinity cols = 42.
        // 10 doubling intermediates × 6 = 60, 8 add intermediates × 6 = 48,
        // 4 curve-eq intermediates × 6 = 24, plus 1 sel_curve = 1.
        // Total = 42 + 60 + 48 + 24 + 1 = 175.
        assert_eq!(LIMBS_PER_BLS12_FP, 6);
        assert_eq!(LIMBS_PER_G1, 12);
        assert_eq!(COL_P_OFFSET, 0);
        assert_eq!(COL_Q_OFFSET, 12);
        assert_eq!(COL_R_OFFSET, 24);
        assert_eq!(COL_IS_REAL, 36);
        assert_eq!(COL_R_INFINITY, 41);
        assert_eq!(COL_DBL_X_SQ, 42);
        assert_eq!(COL_ADD_DIFF_QY_PY, 42 + 10 * LIMBS_PER_BLS12_FP);
        assert_eq!(COL_RX_SQ, 42 + 18 * LIMBS_PER_BLS12_FP);
        assert_eq!(COL_SEL_CURVE, 42 + 22 * LIMBS_PER_BLS12_FP);
        assert_eq!(NUM_COLUMNS, 42 + 22 * LIMBS_PER_BLS12_FP + 1);
        assert_eq!(NUM_COLUMNS, 175);
    }

    #[test]
    fn g1_scaffold_shape_constraints_pass_on_padding() {
        let w = Bls12CurveOpsWitness::new();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = Bls12CurveOpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn bls12_381_p_limbs_match_nonnative_fp() {
        assert_eq!(BLS12_381_P_LIMBS, crate::nonnative_fp::P_LIMBS);
    }

    #[test]
    fn host_g1_generator_is_on_curve() {
        // y² = x³ + 4 (b = 4 for BLS12-381 G1).
        let g = G1Affine::generator();
        assert!(g.is_on_curve(), "G1 generator must satisfy y² = x³ + 4");
    }

    #[test]
    fn g1_doubling_known_vector_two_g_on_curve() {
        // Compute 2·G via the host-side decomposition and cross-check
        // against the production `pairing::G1Affine::double`.
        let g = G1Affine::generator();
        let two_g = g.double_host();
        assert!(!two_g.infinity);
        assert!(two_g.is_on_curve(), "2·G must lie on y² = x³ + 4");

        let g_pair = crate::pairing::G1Affine::generator();
        let two_g_pair = g_pair.double();
        assert_eq!(two_g.x.limbs, two_g_pair.x.limbs, "2·G.x must match pairing impl");
        assert_eq!(two_g.y.limbs, two_g_pair.y.limbs, "2·G.y must match pairing impl");

        // Building a witness with the host-computed R passes shape constraints.
        let mut w = Bls12CurveOpsWitness::new();
        w.push(Bls12CurveOpsRow {
            p: g, q: G1Affine::identity(), r: two_g,
            op: OpSelector::Double, is_real: true,
        });
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = Bls12CurveOpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(),
                        "2·G doubling: constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn g1_addition_g_plus_2g_equals_3g_on_curve() {
        let g = G1Affine::generator();
        let two_g = g.double_host();
        let three_g = g.add_host(&two_g);
        assert!(three_g.is_on_curve(), "G + 2G must lie on y² = x³ + 4");

        let mut w = Bls12CurveOpsWitness::new();
        w.push(Bls12CurveOpsRow {
            p: g, q: G1Affine::identity(), r: two_g,
            op: OpSelector::Double, is_real: true,
        });
        w.push(Bls12CurveOpsRow {
            p: g, q: two_g, r: three_g,
            op: OpSelector::Add, is_real: true,
        });
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = Bls12CurveOpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(),
                        "G + 2G row: constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn tampered_four_const_fires_pin() {
        let mut w = Bls12CurveOpsWitness::new();
        w.push(sample_double_row());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Overwrite four_const's BE LSB (limb 5) from 4 → 5; the pin must fire.
        cols[COL_FOUR_CONST + LIMBS_PER_BLS12_FP - 1][0] =
            Scalar::from_u64(5, curve);
        let cs = Bls12CurveOpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 15 = four_const[5] − 4·sel_curve must fire (5 - 4 = 1).
        assert!(!results[15][0].is_zero(),
                "tampered four_const LSB must fire the pin constraint");
    }

    #[test]
    fn tampered_selector_fires_mutex_constraint() {
        let mut w = Bls12CurveOpsWitness::new();
        w.push(sample_double_row());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Set sel_add = 1 on a row that already has sel_double = 1.
        cols[COL_SEL_ADD][0] = Scalar::one(curve);
        let cs = Bls12CurveOpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[3][0].is_zero(),
                "double selector must fire mutex constraint");
        assert!(!results[4][0].is_zero(),
                "double selector must fire sum-to-is_real constraint");
    }

    #[test]
    fn r_infinity_disables_sel_curve_pin() {
        let mut w = Bls12CurveOpsWitness::new();
        w.push(Bls12CurveOpsRow {
            p: G1Affine::generator(),
            q: G1Affine::identity(),
            r: G1Affine::identity(),
            op: OpSelector::Double,
            is_real: true,
        });
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        // sel_curve was not set because r.infinity is true.
        assert!(trace.columns[COL_SEL_CURVE].evaluations[0].is_zero());
        // four_const should be all zeros on this row.
        for j in 0..LIMBS_PER_BLS12_FP {
            assert!(trace.columns[COL_FOUR_CONST + j].evaluations[0].is_zero());
        }
        let cs = Bls12CurveOpsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(),
                        "R-infinity row: constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn selector_indices_include_sel_curve() {
        let cs = Bls12CurveOpsConstraintSystem::new(0);
        let idx = cs.selector_column_indices();
        assert!(idx.contains(&COL_IS_REAL));
        assert!(idx.contains(&COL_SEL_ADD));
        assert!(idx.contains(&COL_SEL_DOUBLE));
        assert!(idx.contains(&COL_SEL_CURVE));
        assert_eq!(cs.padding_selector_column(), None);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert!(cs.shifted_column_indices().is_empty());
    }

    #[test]
    fn bls12_381_g1_double_fp_descriptors_well_formed() {
        let ds = bls12_381_g1_double_fp_descriptors(0, 1);
        assert_eq!(ds.len(), 12);
        let mut labels: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for d in &ds {
            assert_eq!(d.a_layer_index, 1);
            assert_eq!(d.b_layer_index, 0);
            assert_eq!(d.a_columns.len(), 3 * LIMBS_PER_BLS12_FP);
            assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_BLS12_FP);
            assert_eq!(d.b_selector_column, Some(COL_SEL_DOUBLE));
            let s = d.a_selector_column.unwrap();
            assert!(
                s == fp::COL_SEL_ADD || s == fp::COL_SEL_SUB
                    || s == fp::COL_SEL_MUL || s == fp::COL_SEL_INV,
                "descriptor `{}`: A-side selector must be an Fp op kind",
                d.label,
            );
            assert_eq!(d.a_columns[0], fp::COL_A_OFFSET);
            assert_eq!(d.a_columns[LIMBS_PER_BLS12_FP], fp::COL_B_OFFSET);
            assert_eq!(d.a_columns[2 * LIMBS_PER_BLS12_FP], fp::COL_C_OFFSET);
            for &c in &d.b_columns {
                assert!(c < NUM_COLUMNS,
                        "descriptor `{}` references col {}", d.label, c);
            }
            assert!(d.label.starts_with("bls12_381_g1_dbl_"));
            assert!(labels.insert(d.label.as_str()), "duplicate label `{}`", d.label);
        }
        // First descriptor: x_sq = P.x · P.x (Mul).
        let d0 = &ds[0];
        assert_eq!(d0.label, "bls12_381_g1_dbl_x_sq");
        assert_eq!(d0.a_selector_column, Some(fp::COL_SEL_MUL));
        assert_eq!(d0.b_columns[0], COL_PX);
        assert_eq!(d0.b_columns[LIMBS_PER_BLS12_FP], COL_PX);
        assert_eq!(d0.b_columns[2 * LIMBS_PER_BLS12_FP], COL_DBL_X_SQ);
    }

    #[test]
    fn bls12_381_g1_add_fp_descriptors_well_formed() {
        let ds = bls12_381_g1_add_fp_descriptors(0, 1);
        assert_eq!(ds.len(), 10);
        for d in &ds {
            assert_eq!(d.b_selector_column, Some(COL_SEL_ADD));
            assert!(d.label.starts_with("bls12_381_g1_add_"));
            for &c in &d.b_columns {
                assert!(c < NUM_COLUMNS,
                        "descriptor `{}` col {} OOB", d.label, c);
            }
        }
    }

    #[test]
    fn bls12_381_g1_curve_eq_descriptors_well_formed() {
        let ds = bls12_381_g1_curve_eq_fp_descriptors(0, 1);
        assert_eq!(ds.len(), 4);
        let labels: Vec<&str> = ds.iter().map(|d| d.label.as_str()).collect();
        assert!(labels.contains(&"bls12_381_g1_curve_rx_sq"));
        assert!(labels.contains(&"bls12_381_g1_curve_rx_cubed"));
        assert!(labels.contains(&"bls12_381_g1_curve_ry_sq"));
        assert!(labels.contains(&"bls12_381_g1_curve_eq"));
        for d in &ds {
            assert_eq!(d.b_selector_column, Some(COL_SEL_CURVE));
        }
        let ce = ds.iter().find(|d| d.label == "bls12_381_g1_curve_eq").unwrap();
        assert_eq!(ce.a_selector_column, Some(fp::COL_SEL_SUB));
        assert_eq!(ce.b_columns[2 * LIMBS_PER_BLS12_FP], COL_FOUR_CONST);
    }
}
