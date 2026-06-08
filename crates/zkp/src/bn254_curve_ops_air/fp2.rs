//! Algebraic limb-level AIR for **BN254** non-native `Fp2` arithmetic.
//!
//! `Fp2 = Fp[u] / (u² + 1)`, with the BN254 base prime `p` from
//! [`super::fp::P_LIMBS`]. Each Fp2 element is `c0 + c1·u`, and both
//! components are 4-BE-u64-limb [`super::fp::Fp`] elements.
//!
//! # Operation algebra
//!
//! ```text
//!   (a.c0 + a.c1·u) + (b.c0 + b.c1·u) = (a.c0 + b.c0) + (a.c1 + b.c1)·u
//!   (a.c0 + a.c1·u) - (b.c0 + b.c1·u) = (a.c0 - b.c0) + (a.c1 - b.c1)·u
//!   (a.c0 + a.c1·u) * (b.c0 + b.c1·u) =
//!       (a.c0·b.c0 - a.c1·b.c1) + (a.c0·b.c1 + a.c1·b.c0)·u
//! ```
//!
//! Inversion uses the standard Frobenius-conjugate trick:
//! `(c0 + c1·u)^{-1} = (c0 - c1·u) / (c0² + c1²) mod p`. The
//! algebraic decomposition of `Inv` into Fp ops is committed via the
//! same intermediate-Fp witness columns plus an additional
//! `(norm = c0² + c1²)` and `norm_inv` pair.
//!
//! # Witness layout & soundness model
//!
//! Each row commits one Fp2 operation `c = op(a, b)`, plus a small
//! set of **intermediate Fp values** that the algebra reduces to. The
//! row-local constraints assemble `c.c0` and `c.c1` from the
//! intermediates using **Fp addition / subtraction** (modular under
//! `p`). The intermediate Fp products / sums / differences are then
//! shipped to [`super::fp`]'s `Fp` AIR through cross-AIR LogUp
//! descriptors — each one binds a `(a_op, b_op, c_op)` 12-limb tuple
//! at this row to a matching `(a, b, c)` row of the Fp AIR under the
//! appropriate selector (`SEL_MUL` for products, `SEL_ADD` for sums,
//! `SEL_SUB` for differences, `SEL_INV` for inverses).
//!
//! Because the Fp AIR already enforces canonical reduction modulo
//! `p` on every emitted `c`, every intermediate Fp value committed
//! in this Fp2 AIR is also canonical mod `p` whenever the LogUp
//! closures match. This is the same delegation pattern used by
//! [`crate::miller_fp_descriptors`] for BLS12-381 Fp2.
//!
//! # Row shape (Mul case, the worst)
//!
//! ```text
//! offset  name         size  notes
//! 0       a.c0          4    Fp BE limbs
//! 4       a.c1          4
//! 8       b.c0          4
//! 12      b.c1          4
//! 16      c.c0          4
//! 20      c.c1          4
//!
//! 24      m00           4    Mul: a.c0 · b.c0
//! 28      m11           4    Mul: a.c1 · b.c1
//! 32      m01           4    Mul: a.c0 · b.c1
//! 36      m10           4    Mul: a.c1 · b.c0
//!
//! 40      norm          4    Inv only: c0² + c1²
//! 44      norm_inv      4    Inv only: norm^{-1}
//!
//! 48      sel_add       1
//! 49      sel_sub       1
//! 50      sel_mul       1
//! 51      sel_inv       1
//! 52      is_real       1    sel_add + sel_sub + sel_mul + sel_inv
//! ```
//!
//! # Row-local constraints (algebraic, gated by `is_real`)
//!
//! Beyond the binary / mutex shape checks, only **shape** is enforced
//! row-locally: the algebraic glue between the intermediate Fp values
//! and `c.c0` / `c.c1` is enforced **through the cross-AIR LogUp
//! linkages**, not as polynomial constraints over this AIR's columns.
//! Concretely:
//!
//!   - For `Add`: ship `(a.c0, b.c0, c.c0)` and `(a.c1, b.c1, c.c1)`
//!     into the Fp AIR under `SEL_ADD`. Two linkages.
//!   - For `Sub`: ship the same two triples under `SEL_SUB`. Two
//!     linkages.
//!   - For `Mul`: ship `(a.c0, b.c0, m00)`, `(a.c1, b.c1, m11)`,
//!     `(a.c0, b.c1, m01)`, `(a.c1, b.c0, m10)` under `SEL_MUL`
//!     (four linkages), then `(m00, m11, c.c0)` under `SEL_SUB` and
//!     `(m01, m10, c.c1)` under `SEL_ADD` (two linkages). Six
//!     linkages per Mul row.
//!   - For `Inv`: ship `(a.c0, a.c0, m00)`, `(a.c1, a.c1, m11)`
//!     under `SEL_MUL` (the two squarings), `(m00, m11, norm)` under
//!     `SEL_ADD`, `(norm, norm_inv, _one_)` is implicit through
//!     `SEL_INV` (which the Fp AIR already pins to `c = 1`), so we
//!     ship `(norm, dummy, norm_inv)` under `SEL_INV`. Then
//!     `(a.c0, norm_inv, c.c0)` and `(a.c1, norm_inv, c.c1_neg)`
//!     under `SEL_MUL`, with `c.c1_neg + c.c1 = 0 mod p` enforced
//!     by an extra `SEL_SUB` link `(zero, c.c1_neg, c.c1)`. This
//!     module exposes the Add/Sub/Mul linkage builders; the
//!     full Inv decomposition is documented for the future phase.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

use super::fp::{
    self, Fp, COL_A_OFFSET as FP_COL_A_OFFSET, COL_B_OFFSET as FP_COL_B_OFFSET,
    COL_C_OFFSET as FP_COL_C_OFFSET, COL_SEL_ADD as FP_COL_SEL_ADD, COL_SEL_MUL as FP_COL_SEL_MUL,
    COL_SEL_SUB as FP_COL_SEL_SUB, LIMBS_PER_FP,
};

// ─── Column layout ────────────────────────────────────────────────────

/// Number of Fp limbs (4 for BN254).
pub const LIMBS_PER_FP2_COMPONENT: usize = LIMBS_PER_FP;

pub const COL_A_C0_OFFSET: usize = 0;
pub const COL_A_C1_OFFSET: usize = COL_A_C0_OFFSET + LIMBS_PER_FP;
pub const COL_B_C0_OFFSET: usize = COL_A_C1_OFFSET + LIMBS_PER_FP;
pub const COL_B_C1_OFFSET: usize = COL_B_C0_OFFSET + LIMBS_PER_FP;
pub const COL_C_C0_OFFSET: usize = COL_B_C1_OFFSET + LIMBS_PER_FP;
pub const COL_C_C1_OFFSET: usize = COL_C_C0_OFFSET + LIMBS_PER_FP;

// Mul intermediates (also reused by Inv squarings: m00 = c0², m11 = c1²).
pub const COL_M00_OFFSET: usize = COL_C_C1_OFFSET + LIMBS_PER_FP;
pub const COL_M11_OFFSET: usize = COL_M00_OFFSET + LIMBS_PER_FP;
pub const COL_M01_OFFSET: usize = COL_M11_OFFSET + LIMBS_PER_FP;
pub const COL_M10_OFFSET: usize = COL_M01_OFFSET + LIMBS_PER_FP;

// Inv-only intermediates.
pub const COL_NORM_OFFSET: usize = COL_M10_OFFSET + LIMBS_PER_FP;
pub const COL_NORM_INV_OFFSET: usize = COL_NORM_OFFSET + LIMBS_PER_FP;

pub const NUM_DATA_COLUMNS: usize = COL_NORM_INV_OFFSET + LIMBS_PER_FP;

pub const COL_SEL_ADD: usize = NUM_DATA_COLUMNS;
pub const COL_SEL_SUB: usize = COL_SEL_ADD + 1;
pub const COL_SEL_MUL: usize = COL_SEL_SUB + 1;
pub const COL_SEL_INV: usize = COL_SEL_MUL + 1;
pub const COL_IS_REAL: usize = COL_SEL_INV + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

// ─── Constraint count ─────────────────────────────────────────────────
//
//   0. sel_add ∈ {0,1}
//   1. sel_sub ∈ {0,1}
//   2. sel_mul ∈ {0,1}
//   3. sel_inv ∈ {0,1}
//   4. is_real ∈ {0,1}
//   5. is_real = sel_add + sel_sub + sel_mul + sel_inv  (sum binding)
//   6. sel_add · sel_sub = 0                            (pairwise mutex)
//   7. sel_add · sel_mul = 0
//   8. sel_add · sel_inv = 0
//   9. sel_sub · sel_mul = 0
//  10. sel_sub · sel_inv = 0
//  11. sel_mul · sel_inv = 0

pub const NUM_ROW_CONSTRAINTS: usize = 12;

/// No cross-row constraints — each Fp2 op row is stand-alone.
pub const NUM_SHIFTED: usize = 0;

// ─── Host-side Fp2 ────────────────────────────────────────────────────

/// Element of BN254 `Fp2 = Fp[u]/(u² + 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp2 {
    pub c0: Fp,
    pub c1: Fp,
}

impl Fp2 {
    #[inline]
    pub const fn new(c0: Fp, c1: Fp) -> Self { Fp2 { c0, c1 } }
    #[inline]
    pub fn zero() -> Self { Fp2 { c0: Fp::zero(), c1: Fp::zero() } }
    #[inline]
    pub fn one() -> Self { Fp2 { c0: Fp::one(), c1: Fp::zero() } }
    #[inline]
    pub fn is_zero(&self) -> bool { self.c0.is_zero() && self.c1.is_zero() }

    pub fn add(&self, other: &Self) -> Self {
        Fp2 { c0: self.c0.add(&other.c0), c1: self.c1.add(&other.c1) }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Fp2 { c0: self.c0.sub(&other.c0), c1: self.c1.sub(&other.c1) }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let m00 = self.c0.mul(&other.c0);
        let m11 = self.c1.mul(&other.c1);
        let m01 = self.c0.mul(&other.c1);
        let m10 = self.c1.mul(&other.c0);
        Fp2 { c0: m00.sub(&m11), c1: m01.add(&m10) }
    }

    /// Multiplicative inverse. Returns `None` for the zero element.
    /// Uses `(c0 + c1·u)^{-1} = (c0 - c1·u) / (c0² + c1²)`.
    pub fn invert(&self) -> Option<Self> {
        if self.is_zero() { return None; }
        let c0_sq = self.c0.mul(&self.c0);
        let c1_sq = self.c1.mul(&self.c1);
        let norm = c0_sq.add(&c1_sq);
        let norm_inv = norm.invert()?;
        let c0 = self.c0.mul(&norm_inv);
        let neg_c1 = Fp::zero().sub(&self.c1);
        let c1 = neg_c1.mul(&norm_inv);
        Some(Fp2 { c0, c1 })
    }
}

// ─── Op enum ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp2Op {
    Add { a: Fp2, b: Fp2 },
    Sub { a: Fp2, b: Fp2 },
    Mul { a: Fp2, b: Fp2 },
    Inv { a: Fp2 },
}

// ─── Trace builder ────────────────────────────────────────────────────

#[inline]
fn sc(v: u64, curve: CurveType) -> Scalar { Scalar::from_u64(v, curve) }

fn write_fp_limbs(columns: &mut [Vec<Scalar>], offset: usize, row: usize, fp: &Fp, curve: CurveType) {
    for i in 0..LIMBS_PER_FP {
        columns[offset + i][row] = sc(fp.limbs[i], curve);
    }
}

pub fn alloc_trace(num_rows: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..NUM_COLUMNS).map(|_| vec![zero.clone(); num_rows]).collect()
}

pub fn populate_row(columns: &mut [Vec<Scalar>], row: usize, op: &Fp2Op, curve: CurveType) {
    assert_eq!(columns.len(), NUM_COLUMNS, "columns shape");
    let one = Scalar::one(curve);
    let zero_fp = Fp::zero();

    // Caller is responsible for zero-initializing padding; here we only
    // overwrite the per-op columns. Selectors default to 0.
    match op {
        Fp2Op::Add { a, b } => {
            let c = a.add(b);
            write_fp_limbs(columns, COL_A_C0_OFFSET, row, &a.c0, curve);
            write_fp_limbs(columns, COL_A_C1_OFFSET, row, &a.c1, curve);
            write_fp_limbs(columns, COL_B_C0_OFFSET, row, &b.c0, curve);
            write_fp_limbs(columns, COL_B_C1_OFFSET, row, &b.c1, curve);
            write_fp_limbs(columns, COL_C_C0_OFFSET, row, &c.c0, curve);
            write_fp_limbs(columns, COL_C_C1_OFFSET, row, &c.c1, curve);
            columns[COL_SEL_ADD][row] = one.clone();
            columns[COL_IS_REAL][row] = one;
        }
        Fp2Op::Sub { a, b } => {
            let c = a.sub(b);
            write_fp_limbs(columns, COL_A_C0_OFFSET, row, &a.c0, curve);
            write_fp_limbs(columns, COL_A_C1_OFFSET, row, &a.c1, curve);
            write_fp_limbs(columns, COL_B_C0_OFFSET, row, &b.c0, curve);
            write_fp_limbs(columns, COL_B_C1_OFFSET, row, &b.c1, curve);
            write_fp_limbs(columns, COL_C_C0_OFFSET, row, &c.c0, curve);
            write_fp_limbs(columns, COL_C_C1_OFFSET, row, &c.c1, curve);
            columns[COL_SEL_SUB][row] = one.clone();
            columns[COL_IS_REAL][row] = one;
        }
        Fp2Op::Mul { a, b } => {
            let m00 = a.c0.mul(&b.c0);
            let m11 = a.c1.mul(&b.c1);
            let m01 = a.c0.mul(&b.c1);
            let m10 = a.c1.mul(&b.c0);
            let c = Fp2 { c0: m00.sub(&m11), c1: m01.add(&m10) };
            write_fp_limbs(columns, COL_A_C0_OFFSET, row, &a.c0, curve);
            write_fp_limbs(columns, COL_A_C1_OFFSET, row, &a.c1, curve);
            write_fp_limbs(columns, COL_B_C0_OFFSET, row, &b.c0, curve);
            write_fp_limbs(columns, COL_B_C1_OFFSET, row, &b.c1, curve);
            write_fp_limbs(columns, COL_C_C0_OFFSET, row, &c.c0, curve);
            write_fp_limbs(columns, COL_C_C1_OFFSET, row, &c.c1, curve);
            write_fp_limbs(columns, COL_M00_OFFSET, row, &m00, curve);
            write_fp_limbs(columns, COL_M11_OFFSET, row, &m11, curve);
            write_fp_limbs(columns, COL_M01_OFFSET, row, &m01, curve);
            write_fp_limbs(columns, COL_M10_OFFSET, row, &m10, curve);
            columns[COL_SEL_MUL][row] = one.clone();
            columns[COL_IS_REAL][row] = one;
        }
        Fp2Op::Inv { a } => {
            let c = a.invert().expect("Fp2::Inv on zero");
            let c0_sq = a.c0.mul(&a.c0);
            let c1_sq = a.c1.mul(&a.c1);
            let norm = c0_sq.add(&c1_sq);
            let norm_inv = norm.invert().expect("nonzero norm");
            write_fp_limbs(columns, COL_A_C0_OFFSET, row, &a.c0, curve);
            write_fp_limbs(columns, COL_A_C1_OFFSET, row, &a.c1, curve);
            write_fp_limbs(columns, COL_B_C0_OFFSET, row, &zero_fp, curve);
            write_fp_limbs(columns, COL_B_C1_OFFSET, row, &zero_fp, curve);
            write_fp_limbs(columns, COL_C_C0_OFFSET, row, &c.c0, curve);
            write_fp_limbs(columns, COL_C_C1_OFFSET, row, &c.c1, curve);
            write_fp_limbs(columns, COL_M00_OFFSET, row, &c0_sq, curve);
            write_fp_limbs(columns, COL_M11_OFFSET, row, &c1_sq, curve);
            write_fp_limbs(columns, COL_NORM_OFFSET, row, &norm, curve);
            write_fp_limbs(columns, COL_NORM_INV_OFFSET, row, &norm_inv, curve);
            columns[COL_SEL_INV][row] = one.clone();
            columns[COL_IS_REAL][row] = one;
        }
    }
}

pub fn populate_trace(ops: &[Fp2Op], curve: CurveType, num_rows: Option<usize>) -> Vec<Vec<Scalar>> {
    let rows = num_rows.unwrap_or(ops.len()).max(ops.len()).max(1);
    let mut columns = alloc_trace(rows, curve);
    for (row, op) in ops.iter().enumerate() {
        populate_row(&mut columns, row, op, curve);
    }
    columns
}

pub fn build_trace_polynomials(ops: &[Fp2Op], curve: CurveType) -> TracePolynomials {
    let num_rows = ops.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let columns = populate_trace(ops, curve, Some(padded));
    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct Bn254Fp2ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bn254Fp2ConstraintSystem {
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

impl VmConstraintSystem for Bn254Fp2ConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        [
            "sel_add_binary",
            "sel_sub_binary",
            "sel_mul_binary",
            "sel_inv_binary",
            "is_real_binary",
            "is_real_sum_eq_selectors",
            "mutex_add_sub",
            "mutex_add_mul",
            "mutex_add_inv",
            "mutex_sub_mul",
            "mutex_sub_inv",
            "mutex_mul_inv",
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
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        for r in 0..n {
            let sa = &columns[COL_SEL_ADD][r];
            let ss = &columns[COL_SEL_SUB][r];
            let sm = &columns[COL_SEL_MUL][r];
            let si = &columns[COL_SEL_INV][r];
            let ir = &columns[COL_IS_REAL][r];

            out[0][r] = bin(sa);
            out[1][r] = bin(ss);
            out[2][r] = bin(sm);
            out[3][r] = bin(si);
            out[4][r] = bin(ir);
            out[5][r] = sa.add(ss).add(sm).add(si).sub(ir);
            out[6][r] = sa.mul(ss);
            out[7][r] = sa.mul(sm);
            out[8][r] = sa.mul(si);
            out[9][r] = ss.mul(sm);
            out[10][r] = ss.mul(si);
            out[11][r] = sm.mul(si);
        }
        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let sa = &col_evals[COL_SEL_ADD];
        let ss = &col_evals[COL_SEL_SUB];
        let sm = &col_evals[COL_SEL_MUL];
        let si = &col_evals[COL_SEL_INV];
        let ir = &col_evals[COL_IS_REAL];
        let parts = [
            bin(sa),
            bin(ss),
            bin(sm),
            bin(si),
            bin(ir),
            sa.add(ss).add(sm).add(si).sub(ir),
            sa.mul(ss),
            sa.mul(sm),
            sa.mul(si),
            ss.mul(sm),
            ss.mul(si),
            sm.mul(si),
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

        let bin_poly = |v: &Vec<Scalar>| -> Vec<Scalar> {
            let v_m1 = poly_sub(v, &one_poly, curve);
            poly_mul(v, &v_m1, curve)
        };

        let sa = &col_coeffs[COL_SEL_ADD];
        let ss = &col_coeffs[COL_SEL_SUB];
        let sm = &col_coeffs[COL_SEL_MUL];
        let si = &col_coeffs[COL_SEL_INV];
        let ir = &col_coeffs[COL_IS_REAL];
        let sum_sels =
            poly_add(&poly_add(&poly_add(sa, ss, curve), sm, curve), si, curve);
        let parts: [Vec<Scalar>; NUM_ROW_CONSTRAINTS] = [
            bin_poly(sa),
            bin_poly(ss),
            bin_poly(sm),
            bin_poly(si),
            bin_poly(ir),
            poly_sub(&sum_sels, ir, curve),
            poly_mul(sa, ss, curve),
            poly_mul(sa, sm, curve),
            poly_mul(sa, si, curve),
            poly_mul(ss, sm, curve),
            poly_mul(ss, si, curve),
            poly_mul(sm, si, curve),
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
        vec![COL_SEL_ADD, COL_SEL_SUB, COL_SEL_MUL, COL_SEL_INV, COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> { Vec::new() }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }
}

// ─── Lookup declarations (limb range + binary) ────────────────────────

pub fn lookup_declarations() -> Vec<LookupDeclaration> {
    let mut decls = Vec::new();
    let range_64 = |col: usize, label: &str| LookupDeclaration {
        label: label.into(), column_index: col, max_bits: 64, selector_column: None,
    };
    let binary = |col: usize, label: &str| LookupDeclaration {
        label: label.into(), column_index: col, max_bits: 1, selector_column: None,
    };

    let limb_blocks: &[(usize, &str)] = &[
        (COL_A_C0_OFFSET, "a_c0"),
        (COL_A_C1_OFFSET, "a_c1"),
        (COL_B_C0_OFFSET, "b_c0"),
        (COL_B_C1_OFFSET, "b_c1"),
        (COL_C_C0_OFFSET, "c_c0"),
        (COL_C_C1_OFFSET, "c_c1"),
        (COL_M00_OFFSET, "m00"),
        (COL_M11_OFFSET, "m11"),
        (COL_M01_OFFSET, "m01"),
        (COL_M10_OFFSET, "m10"),
        (COL_NORM_OFFSET, "norm"),
        (COL_NORM_INV_OFFSET, "norm_inv"),
    ];
    for (off, label) in limb_blocks {
        for i in 0..LIMBS_PER_FP {
            decls.push(range_64(off + i, &format!("{}_limb_{}_range", label, i)));
        }
    }
    decls.push(binary(COL_SEL_ADD, "sel_add_bin"));
    decls.push(binary(COL_SEL_SUB, "sel_sub_bin"));
    decls.push(binary(COL_SEL_MUL, "sel_mul_bin"));
    decls.push(binary(COL_SEL_INV, "sel_inv_bin"));
    decls.push(binary(COL_IS_REAL, "is_real_bin"));
    decls
}

// ─── Cross-AIR LogUp descriptors (Fp2 → Fp) ───────────────────────────

/// Build a descriptor for one `(a_fp, b_fp, c_fp)` triple shipped to
/// the Fp AIR. The 12 columns on side A are `a_fp ∥ b_fp ∥ c_fp` on
/// this Fp2 AIR; side B is the Fp AIR's `(A, B, C)` limb block. The
/// `b_selector` chooses which Fp op (Add/Sub/Mul/Inv) on the Fp side
/// matches.
fn make_triple_descriptor(
    label: &str,
    fp2_layer_index: usize,
    fp_layer_index: usize,
    a_offset: usize,
    b_offset: usize,
    c_offset: usize,
    a_selector_on_fp2: usize,
    b_selector_on_fp: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_cols: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_FP);
    for i in 0..LIMBS_PER_FP { a_cols.push(a_offset + i); }
    for i in 0..LIMBS_PER_FP { a_cols.push(b_offset + i); }
    for i in 0..LIMBS_PER_FP { a_cols.push(c_offset + i); }

    let mut b_cols: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_FP);
    for i in 0..LIMBS_PER_FP { b_cols.push(FP_COL_A_OFFSET + i); }
    for i in 0..LIMBS_PER_FP { b_cols.push(FP_COL_B_OFFSET + i); }
    for i in 0..LIMBS_PER_FP { b_cols.push(FP_COL_C_OFFSET + i); }

    CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: fp2_layer_index,
        a_columns: a_cols,
        a_selector_column: Some(a_selector_on_fp2),
        b_layer_index: fp_layer_index,
        b_columns: b_cols,
        b_selector_column: Some(b_selector_on_fp),
    }
}

/// Linkage for `Fp2::Add`: `(a.c0, b.c0, c.c0)` on Add rows.
pub fn make_fp2_add_c0_to_fp_add_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_add_c0_to_fp_add_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C0_OFFSET, COL_B_C0_OFFSET, COL_C_C0_OFFSET,
        COL_SEL_ADD, FP_COL_SEL_ADD,
    )
}

/// Linkage for `Fp2::Add`: `(a.c1, b.c1, c.c1)` on Add rows.
pub fn make_fp2_add_c1_to_fp_add_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_add_c1_to_fp_add_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C1_OFFSET, COL_B_C1_OFFSET, COL_C_C1_OFFSET,
        COL_SEL_ADD, FP_COL_SEL_ADD,
    )
}

/// Linkage for `Fp2::Sub`: `(a.c0, b.c0, c.c0)` on Sub rows.
pub fn make_fp2_sub_c0_to_fp_sub_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_sub_c0_to_fp_sub_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C0_OFFSET, COL_B_C0_OFFSET, COL_C_C0_OFFSET,
        COL_SEL_SUB, FP_COL_SEL_SUB,
    )
}

/// Linkage for `Fp2::Sub`: `(a.c1, b.c1, c.c1)` on Sub rows.
pub fn make_fp2_sub_c1_to_fp_sub_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_sub_c1_to_fp_sub_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C1_OFFSET, COL_B_C1_OFFSET, COL_C_C1_OFFSET,
        COL_SEL_SUB, FP_COL_SEL_SUB,
    )
}

/// Linkage for `Fp2::Mul`: `(a.c0, b.c0, m00)` to Fp Mul rows.
pub fn make_fp2_mul_m00_to_fp_mul_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_mul_m00_to_fp_mul_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C0_OFFSET, COL_B_C0_OFFSET, COL_M00_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}

/// Linkage for `Fp2::Mul`: `(a.c1, b.c1, m11)` to Fp Mul rows.
pub fn make_fp2_mul_m11_to_fp_mul_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_mul_m11_to_fp_mul_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C1_OFFSET, COL_B_C1_OFFSET, COL_M11_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}

/// Linkage for `Fp2::Mul`: `(a.c0, b.c1, m01)` to Fp Mul rows.
pub fn make_fp2_mul_m01_to_fp_mul_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_mul_m01_to_fp_mul_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C0_OFFSET, COL_B_C1_OFFSET, COL_M01_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}

/// Linkage for `Fp2::Mul`: `(a.c1, b.c0, m10)` to Fp Mul rows.
pub fn make_fp2_mul_m10_to_fp_mul_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_mul_m10_to_fp_mul_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C1_OFFSET, COL_B_C0_OFFSET, COL_M10_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}

/// Linkage for `Fp2::Mul`: `(m00, m11, c.c0)` to Fp Sub rows.
pub fn make_fp2_mul_c0_combine_to_fp_sub_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_mul_c0_combine_to_fp_sub_v1",
        fp2_layer_index, fp_layer_index,
        COL_M00_OFFSET, COL_M11_OFFSET, COL_C_C0_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_SUB,
    )
}

/// Linkage for `Fp2::Mul`: `(m01, m10, c.c1)` to Fp Add rows.
pub fn make_fp2_mul_c1_combine_to_fp_add_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_mul_c1_combine_to_fp_add_v1",
        fp2_layer_index, fp_layer_index,
        COL_M01_OFFSET, COL_M10_OFFSET, COL_C_C1_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_ADD,
    )
}

/// Linkage for `Fp2::Inv` step 1: `(a.c0, a.c0, m00 = c0²)` to Fp Mul rows.
pub fn make_fp2_inv_sq0_to_fp_mul_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_inv_sq0_to_fp_mul_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C0_OFFSET, COL_A_C0_OFFSET, COL_M00_OFFSET,
        COL_SEL_INV, FP_COL_SEL_MUL,
    )
}

/// Linkage for `Fp2::Inv` step 2: `(a.c1, a.c1, m11 = c1²)` to Fp Mul rows.
pub fn make_fp2_inv_sq1_to_fp_mul_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_inv_sq1_to_fp_mul_v1",
        fp2_layer_index, fp_layer_index,
        COL_A_C1_OFFSET, COL_A_C1_OFFSET, COL_M11_OFFSET,
        COL_SEL_INV, FP_COL_SEL_MUL,
    )
}

/// Linkage for `Fp2::Inv` step 3: `(m00, m11, norm)` to Fp Add rows.
pub fn make_fp2_inv_norm_to_fp_add_linkage_descriptor(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bn254_fp2_inv_norm_to_fp_add_v1",
        fp2_layer_index, fp_layer_index,
        COL_M00_OFFSET, COL_M11_OFFSET, COL_NORM_OFFSET,
        COL_SEL_INV, FP_COL_SEL_ADD,
    )
}

// Note: the rest of the Inv decomposition (norm_inv, the two final
// products c.c0 = a.c0·norm_inv and the c.c1 sign-flip + product)
// requires committing an additional `c1_neg` intermediate column
// in this AIR; deferred to the algebraic-curve-law phase.

/// Convenience aggregator: return all Add / Sub / Mul-side linkage
/// descriptors (10 total) for a given pair of layer indices. The Inv
/// linkages are returned by [`fp2_inv_linkage_descriptors`].
pub fn fp2_to_fp_linkage_descriptors(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    vec![
        make_fp2_add_c0_to_fp_add_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_add_c1_to_fp_add_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_sub_c0_to_fp_sub_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_sub_c1_to_fp_sub_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_mul_m00_to_fp_mul_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_mul_m11_to_fp_mul_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_mul_m01_to_fp_mul_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_mul_m10_to_fp_mul_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_mul_c0_combine_to_fp_sub_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_mul_c1_combine_to_fp_add_linkage_descriptor(fp2_layer_index, fp_layer_index),
    ]
}

pub fn fp2_inv_linkage_descriptors(
    fp2_layer_index: usize,
    fp_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    vec![
        make_fp2_inv_sq0_to_fp_mul_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_inv_sq1_to_fp_mul_linkage_descriptor(fp2_layer_index, fp_layer_index),
        make_fp2_inv_norm_to_fp_add_linkage_descriptor(fp2_layer_index, fp_layer_index),
    ]
}

// Silence unused-import warning for `fp` (re-export anchor for callers).
#[allow(dead_code)]
const _FP_MODULE_REFERENCE: fn() -> usize = || fp::NUM_BN254_FP_COLUMNS;

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fp() -> Fp { Fp { limbs: [0, 0, 0, 7] } }
    fn sample_fp2_a() -> Fp2 { Fp2 { c0: Fp::from_u64(3), c1: Fp::from_u64(5) } }
    fn sample_fp2_b() -> Fp2 { Fp2 { c0: Fp::from_u64(11), c1: Fp::from_u64(13) } }

    fn assert_constraints_zero(columns: &[Vec<Scalar>]) {
        let cs = Bn254Fp2ConstraintSystem::new(columns[0].len());
        let refs: Vec<&Vec<Scalar>> = columns.iter().collect();
        let evals = cs.evaluate_on_domain(&refs, columns[0].len());
        for (i, col) in evals.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} fired at row {}", i, r);
            }
        }
    }

    #[test]
    fn fp2_air_column_layout_is_packed() {
        // 6 Fp limb blocks (a.c0, a.c1, b.c0, b.c1, c.c0, c.c1)
        // + 4 Mul intermediates (m00, m11, m01, m10)
        // + 2 Inv intermediates (norm, norm_inv)
        // = 12 Fp limb blocks × 4 limbs = 48 limb cols
        // + 5 selector/flag cols (sel_add/sub/mul/inv, is_real)
        // = 53 cols.
        assert_eq!(LIMBS_PER_FP2_COMPONENT, 4);
        assert_eq!(NUM_DATA_COLUMNS, 12 * LIMBS_PER_FP);
        assert_eq!(NUM_DATA_COLUMNS, 48);
        assert_eq!(COL_SEL_ADD, 48);
        assert_eq!(COL_IS_REAL, 52);
        assert_eq!(NUM_COLUMNS, 53);
        assert_eq!(NUM_ROW_CONSTRAINTS, 12);
        // Sanity on offsets.
        assert_eq!(COL_A_C0_OFFSET, 0);
        assert_eq!(COL_C_C1_OFFSET, 20);
        assert_eq!(COL_M00_OFFSET, 24);
        assert_eq!(COL_M10_OFFSET, 36);
        assert_eq!(COL_NORM_OFFSET, 40);
        assert_eq!(COL_NORM_INV_OFFSET, 44);
    }

    #[test]
    fn fp2_host_arithmetic_known_vectors() {
        // (3 + 5u) + (11 + 13u) = 14 + 18u.
        let a = sample_fp2_a();
        let b = sample_fp2_b();
        let sum = a.add(&b);
        assert_eq!(sum.c0, Fp::from_u64(14));
        assert_eq!(sum.c1, Fp::from_u64(18));
        // (3 + 5u)(11 + 13u) = (33 - 65) + (39 + 55)u = -32 + 94u.
        let prod = a.mul(&b);
        assert_eq!(prod.c1, Fp::from_u64(39 + 55));
        // -32 mod p = p - 32; verify via re-addition.
        let thirty_two = Fp::from_u64(32);
        let back = prod.c0.add(&thirty_two);
        assert!(back.is_zero(), "(33 - 65) mod p should be -32");
        // (1 + 0·u) = identity for mul.
        let one2 = Fp2::one();
        let any = sample_fp2_a();
        let p = one2.mul(&any);
        assert_eq!(p, any);
        // Mul · Inv = 1.
        let any_b = sample_fp2_b();
        let inv = any_b.invert().expect("nonzero");
        let chk = any_b.mul(&inv);
        assert_eq!(chk, Fp2::one(), "Fp2 inv round-trip failed");
        // Zero check.
        assert!(Fp2::zero().is_zero());
        assert!(Fp2::zero().invert().is_none());
        // Sub.
        let d = b.sub(&a);
        assert_eq!(d.c0, Fp::from_u64(11 - 3));
        assert_eq!(d.c1, Fp::from_u64(13 - 5));
        // Single sample_fp used to anchor the Fp import.
        assert_eq!(sample_fp().limbs[3], 7);
    }

    #[test]
    fn fp2_air_constraints_zero_on_honest_witness() {
        let ops = vec![
            Fp2Op::Add { a: sample_fp2_a(), b: sample_fp2_b() },
            Fp2Op::Sub { a: sample_fp2_a(), b: sample_fp2_b() },
            Fp2Op::Mul { a: sample_fp2_a(), b: sample_fp2_b() },
            Fp2Op::Inv { a: sample_fp2_a() },
        ];
        let columns = populate_trace(&ops, CurveType::Bls48581, Some(8));
        assert_constraints_zero(&columns);
    }

    #[test]
    fn fp2_air_rejects_double_selector() {
        let ops = vec![Fp2Op::Add { a: sample_fp2_a(), b: sample_fp2_b() }];
        let curve = CurveType::Bls48581;
        let mut columns = populate_trace(&ops, curve, Some(2));
        let one = Scalar::one(curve);
        // Set sel_mul = 1 simultaneously with sel_add = 1.
        columns[COL_SEL_MUL][0] = one.clone();
        // is_real stays 1 — sum_eq_selectors fires (1+0+1+0 ≠ 1).
        let cs = Bn254Fp2ConstraintSystem::new(columns[0].len());
        let refs: Vec<&Vec<Scalar>> = columns.iter().collect();
        let evals = cs.evaluate_on_domain(&refs, columns[0].len());
        // Pairwise mutex `sel_add · sel_mul` (constraint 7) and sum-eq
        // constraint (5) must fire.
        assert!(!evals[7][0].is_zero(), "mutex_add_mul must fire");
        assert!(!evals[5][0].is_zero(), "is_real_sum_eq_selectors must fire");
    }

    #[test]
    fn fp2_air_padding_rows_are_clean() {
        // Empty op list + padded shape: all constraints vanish on the
        // all-zero trace.
        let columns = populate_trace(&[], CurveType::Bls48581, Some(4));
        assert_constraints_zero(&columns);
    }

    #[test]
    fn fp2_air_lookup_declarations_are_well_formed() {
        let decls = lookup_declarations();
        assert!(!decls.is_empty(), "expected at least one lookup declaration");
        for d in &decls {
            assert!(d.column_index < NUM_COLUMNS,
                    "decl `{}` references out-of-range column {}", d.label, d.column_index);
            assert!(d.max_bits == 1 || d.max_bits == 64,
                    "unexpected max_bits {} for decl `{}`", d.max_bits, d.label);
        }
    }

    #[test]
    fn fp2_to_fp_linkage_descriptors_are_well_formed() {
        let descs = fp2_to_fp_linkage_descriptors(0, 1);
        assert_eq!(descs.len(), 10, "expected 10 Add/Sub/Mul linkages");
        for d in &descs {
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 1);
            assert_eq!(d.a_columns.len(), 3 * LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_FP);
            // B side tuple is always (A, B, C) on the Fp AIR.
            assert_eq!(d.b_columns[0], FP_COL_A_OFFSET);
            assert_eq!(d.b_columns[LIMBS_PER_FP], FP_COL_B_OFFSET);
            assert_eq!(d.b_columns[2 * LIMBS_PER_FP], FP_COL_C_OFFSET);
            // A-side selector is one of the Fp2 selectors.
            let asel = d.a_selector_column.unwrap();
            assert!(asel == COL_SEL_ADD || asel == COL_SEL_SUB || asel == COL_SEL_MUL);
            // B-side selector is one of the Fp selectors.
            let bsel = d.b_selector_column.unwrap();
            assert!(
                bsel == FP_COL_SEL_ADD || bsel == FP_COL_SEL_SUB || bsel == FP_COL_SEL_MUL,
            );
            assert!(!d.label.is_empty());
        }

        // Spot-check one descriptor matches the expected layout:
        // make_fp2_mul_m00_to_fp_mul_linkage_descriptor binds
        // (a.c0, b.c0, m00) on Fp2 (selector SEL_MUL) to (A, B, C) on
        // Fp (selector SEL_MUL).
        let d = make_fp2_mul_m00_to_fp_mul_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns[0], COL_A_C0_OFFSET);
        assert_eq!(d.a_columns[LIMBS_PER_FP], COL_B_C0_OFFSET);
        assert_eq!(d.a_columns[2 * LIMBS_PER_FP], COL_M00_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_SEL_MUL));
        assert_eq!(d.b_selector_column, Some(FP_COL_SEL_MUL));
        assert!(d.label.contains("fp2_mul_m00"));

        // Inv linkages (3 total).
        let inv = fp2_inv_linkage_descriptors(0, 1);
        assert_eq!(inv.len(), 3);
        // First: (a.c0, a.c0, m00) under SEL_INV → Fp SEL_MUL.
        assert_eq!(inv[0].a_columns[0], COL_A_C0_OFFSET);
        assert_eq!(inv[0].a_columns[LIMBS_PER_FP], COL_A_C0_OFFSET);
        assert_eq!(inv[0].a_columns[2 * LIMBS_PER_FP], COL_M00_OFFSET);
        assert_eq!(inv[0].a_selector_column, Some(COL_SEL_INV));
        assert_eq!(inv[0].b_selector_column, Some(FP_COL_SEL_MUL));
    }
}
