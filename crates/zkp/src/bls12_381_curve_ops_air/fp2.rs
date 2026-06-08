//! Algebraic limb-level AIR for **BLS12-381** non-native `Fp2`
//! arithmetic (#271).
//!
//! `Fp2 = Fp[u] / (u² + 1)`, with the BLS12-381 base prime `p` from
//! [`super::fp::P_LIMBS`]. Each Fp2 element is `c0 + c1·u`, and both
//! components are 6-BE-u64-limb [`super::fp::Fp`] elements.
//!
//! Mirrors [`crate::bn254_curve_ops_air::fp2`] with the BLS12-381 limb
//! count (6 instead of 4); the algebraic identities are identical
//! (`u² = -1` is the same for both curves' Fp2 layer).
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
//! Inversion: `(c0 + c1·u)^{-1} = (c0 - c1·u) / (c0² + c1²) mod p`.
//!
//! # Witness layout & soundness model
//!
//! Each row commits one Fp2 operation `c = op(a, b)`, plus a small
//! set of **intermediate Fp values** (`m00 = a.c0·b.c0`, etc.). The
//! row-local constraints enforce only the binary / mutex shape; the
//! algebraic glue between intermediates and `c.c0` / `c.c1` is
//! delegated to the BLS12-381 [`super::fp`] AIR through cross-AIR
//! LogUp descriptors — each one binds a 18-limb `(a, b, c)` triple
//! at this row to an Fp AIR row of the matching op kind.
//!
//! # Row shape (Mul, the worst case)
//!
//! ```text
//! offset (in limb-blocks, 6 u64 each)
//!  0    a.c0
//!  6    a.c1
//! 12    b.c0
//! 18    b.c1
//! 24    c.c0
//! 30    c.c1
//! 36    m00 = a.c0·b.c0
//! 42    m11 = a.c1·b.c1
//! 48    m01 = a.c0·b.c1
//! 54    m10 = a.c1·b.c0
//! 60    norm     (Inv only: c0² + c1²)
//! 66    norm_inv (Inv only: norm^{-1})
//! 72    sel_add   (single col)
//! 73    sel_sub
//! 74    sel_mul
//! 75    sel_inv
//! 76    is_real
//! ```

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

use super::fp::{
    self, Fp, LIMBS_PER_FP, COL_A_OFFSET as FP_COL_A_OFFSET,
    COL_B_OFFSET as FP_COL_B_OFFSET, COL_C_OFFSET as FP_COL_C_OFFSET,
    COL_SEL_ADD as FP_COL_SEL_ADD, COL_SEL_SUB as FP_COL_SEL_SUB,
    COL_SEL_MUL as FP_COL_SEL_MUL,
};

// ─── Host-side Fp2 (re-export) ────────────────────────────────────────

/// Re-export the BLS12-381 host-side `Fp2` from [`crate::nonnative_fp`],
/// which already provides canonical reduction modulo `p` for both
/// components.
pub use crate::nonnative_fp::Fp2;

// ─── Column layout ────────────────────────────────────────────────────

pub const LIMBS_PER_FP2_COMPONENT: usize = LIMBS_PER_FP;

pub const COL_A_C0_OFFSET: usize = 0;
pub const COL_A_C1_OFFSET: usize = COL_A_C0_OFFSET + LIMBS_PER_FP;
pub const COL_B_C0_OFFSET: usize = COL_A_C1_OFFSET + LIMBS_PER_FP;
pub const COL_B_C1_OFFSET: usize = COL_B_C0_OFFSET + LIMBS_PER_FP;
pub const COL_C_C0_OFFSET: usize = COL_B_C1_OFFSET + LIMBS_PER_FP;
pub const COL_C_C1_OFFSET: usize = COL_C_C0_OFFSET + LIMBS_PER_FP;

// Mul intermediates (also re-used by Inv squarings as m00 = c0², m11 = c1²).
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

// Shape constraints (no algebraic glue — that's the LogUp side):
//   0. sel_add ∈ {0,1}
//   1. sel_sub ∈ {0,1}
//   2. sel_mul ∈ {0,1}
//   3. sel_inv ∈ {0,1}
//   4. is_real ∈ {0,1}
//   5. is_real = sel_add + sel_sub + sel_mul + sel_inv
//   6..11. pairwise mutex (6 pairs)
pub const NUM_ROW_CONSTRAINTS: usize = 12;
pub const NUM_SHIFTED: usize = 0;

// ─── Op enum ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp2Op {
    Add { a: Fp2, b: Fp2 },
    Sub { a: Fp2, b: Fp2 },
    Mul { a: Fp2, b: Fp2 },
    Inv { a: Fp2 },
}

// ─── Trace builder ────────────────────────────────────────────────────

fn write_fp_limbs(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    row: usize,
    fp: &Fp,
    curve: CurveType,
) {
    for i in 0..LIMBS_PER_FP {
        columns[offset + i][row] = Scalar::from_u64(fp.limbs[i], curve);
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
            // Schoolbook (NOT Karatsuba) intermediates: m00 = a.c0·b.c0,
            // m11 = a.c1·b.c1, m01 = a.c0·b.c1, m10 = a.c1·b.c0. The
            // host-side Fp2::mul uses Karatsuba internally; we recompute
            // these four products explicitly so the LogUp ships honest
            // schoolbook Fp triples to the Fp AIR (which expects four
            // independent Mul rows on the side B selector).
            let m00 = a.c0.mul(&b.c0);
            let m11 = a.c1.mul(&b.c1);
            let m01 = a.c0.mul(&b.c1);
            let m10 = a.c1.mul(&b.c0);
            // c.c0 = m00 - m11 ; c.c1 = m01 + m10 (these match Karatsuba).
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
            // norm = c0² + c1². Use plain mul to match Fp Mul rows.
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

pub struct Bls12Fp2ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bls12Fp2ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
}

fn bin(v: &Scalar) -> Scalar {
    let curve = v.curve_type();
    let one = Scalar::one(curve);
    v.mul(&v.sub(&one))
}

impl VmConstraintSystem for Bls12Fp2ConstraintSystem {
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
        label: label.into(),
        column_index: col,
        max_bits: 64,
        selector_column: None,
    };
    let binary = |col: usize, label: &str| LookupDeclaration {
        label: label.into(),
        column_index: col,
        max_bits: 1,
        selector_column: None,
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

// Add linkages
pub fn make_fp2_add_c0_to_fp_add_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_add_c0_to_fp_add_v1",
        fp2, fp,
        COL_A_C0_OFFSET, COL_B_C0_OFFSET, COL_C_C0_OFFSET,
        COL_SEL_ADD, FP_COL_SEL_ADD,
    )
}
pub fn make_fp2_add_c1_to_fp_add_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_add_c1_to_fp_add_v1",
        fp2, fp,
        COL_A_C1_OFFSET, COL_B_C1_OFFSET, COL_C_C1_OFFSET,
        COL_SEL_ADD, FP_COL_SEL_ADD,
    )
}

// Sub linkages
pub fn make_fp2_sub_c0_to_fp_sub_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_sub_c0_to_fp_sub_v1",
        fp2, fp,
        COL_A_C0_OFFSET, COL_B_C0_OFFSET, COL_C_C0_OFFSET,
        COL_SEL_SUB, FP_COL_SEL_SUB,
    )
}
pub fn make_fp2_sub_c1_to_fp_sub_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_sub_c1_to_fp_sub_v1",
        fp2, fp,
        COL_A_C1_OFFSET, COL_B_C1_OFFSET, COL_C_C1_OFFSET,
        COL_SEL_SUB, FP_COL_SEL_SUB,
    )
}

// Mul intermediate linkages (4 squarings + 2 combining ops).
pub fn make_fp2_mul_m00_to_fp_mul_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_mul_m00_to_fp_mul_v1",
        fp2, fp,
        COL_A_C0_OFFSET, COL_B_C0_OFFSET, COL_M00_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}
pub fn make_fp2_mul_m11_to_fp_mul_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_mul_m11_to_fp_mul_v1",
        fp2, fp,
        COL_A_C1_OFFSET, COL_B_C1_OFFSET, COL_M11_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}
pub fn make_fp2_mul_m01_to_fp_mul_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_mul_m01_to_fp_mul_v1",
        fp2, fp,
        COL_A_C0_OFFSET, COL_B_C1_OFFSET, COL_M01_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}
pub fn make_fp2_mul_m10_to_fp_mul_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_mul_m10_to_fp_mul_v1",
        fp2, fp,
        COL_A_C1_OFFSET, COL_B_C0_OFFSET, COL_M10_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_MUL,
    )
}
pub fn make_fp2_mul_c0_combine_to_fp_sub_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_mul_c0_combine_to_fp_sub_v1",
        fp2, fp,
        COL_M00_OFFSET, COL_M11_OFFSET, COL_C_C0_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_SUB,
    )
}
pub fn make_fp2_mul_c1_combine_to_fp_add_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_mul_c1_combine_to_fp_add_v1",
        fp2, fp,
        COL_M01_OFFSET, COL_M10_OFFSET, COL_C_C1_OFFSET,
        COL_SEL_MUL, FP_COL_SEL_ADD,
    )
}

// Inv-decomposition partial linkages: squarings + norm sum.
pub fn make_fp2_inv_sq0_to_fp_mul_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_inv_sq0_to_fp_mul_v1",
        fp2, fp,
        COL_A_C0_OFFSET, COL_A_C0_OFFSET, COL_M00_OFFSET,
        COL_SEL_INV, FP_COL_SEL_MUL,
    )
}
pub fn make_fp2_inv_sq1_to_fp_mul_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_inv_sq1_to_fp_mul_v1",
        fp2, fp,
        COL_A_C1_OFFSET, COL_A_C1_OFFSET, COL_M11_OFFSET,
        COL_SEL_INV, FP_COL_SEL_MUL,
    )
}
pub fn make_fp2_inv_norm_to_fp_add_linkage_descriptor(fp2: usize, fp: usize) -> CrossAirLogUpDescriptor {
    make_triple_descriptor(
        "bls12_381_fp2_inv_norm_to_fp_add_v1",
        fp2, fp,
        COL_M00_OFFSET, COL_M11_OFFSET, COL_NORM_OFFSET,
        COL_SEL_INV, FP_COL_SEL_ADD,
    )
}

/// Aggregator: all 10 Add/Sub/Mul-side linkages.
pub fn fp2_to_fp_linkage_descriptors(fp2: usize, fp: usize) -> Vec<CrossAirLogUpDescriptor> {
    vec![
        make_fp2_add_c0_to_fp_add_linkage_descriptor(fp2, fp),
        make_fp2_add_c1_to_fp_add_linkage_descriptor(fp2, fp),
        make_fp2_sub_c0_to_fp_sub_linkage_descriptor(fp2, fp),
        make_fp2_sub_c1_to_fp_sub_linkage_descriptor(fp2, fp),
        make_fp2_mul_m00_to_fp_mul_linkage_descriptor(fp2, fp),
        make_fp2_mul_m11_to_fp_mul_linkage_descriptor(fp2, fp),
        make_fp2_mul_m01_to_fp_mul_linkage_descriptor(fp2, fp),
        make_fp2_mul_m10_to_fp_mul_linkage_descriptor(fp2, fp),
        make_fp2_mul_c0_combine_to_fp_sub_linkage_descriptor(fp2, fp),
        make_fp2_mul_c1_combine_to_fp_add_linkage_descriptor(fp2, fp),
    ]
}

/// Aggregator: 3 Inv-decomposition partial linkages.
pub fn fp2_inv_linkage_descriptors(fp2: usize, fp: usize) -> Vec<CrossAirLogUpDescriptor> {
    vec![
        make_fp2_inv_sq0_to_fp_mul_linkage_descriptor(fp2, fp),
        make_fp2_inv_sq1_to_fp_mul_linkage_descriptor(fp2, fp),
        make_fp2_inv_norm_to_fp_add_linkage_descriptor(fp2, fp),
    ]
}

// Anchor the Fp re-export so `cargo check` doesn't warn on unused imports.
#[allow(dead_code)]
const _FP_ANCHOR: fn() -> usize = || fp::NUM_NONNATIVE_FP_COLUMNS;

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_a() -> Fp2 { Fp2 { c0: Fp::from_u64(3), c1: Fp::from_u64(5) } }
    fn sample_b() -> Fp2 { Fp2 { c0: Fp::from_u64(11), c1: Fp::from_u64(13) } }

    fn assert_constraints_zero(columns: &[Vec<Scalar>]) {
        let cs = Bls12Fp2ConstraintSystem::new(columns[0].len());
        let refs: Vec<&Vec<Scalar>> = columns.iter().collect();
        let evals = cs.evaluate_on_domain(&refs, columns[0].len());
        for (i, col) in evals.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} fired at row {}", i, r);
            }
        }
    }

    #[test]
    fn fp2_column_layout_packed() {
        assert_eq!(LIMBS_PER_FP2_COMPONENT, 6);
        assert_eq!(NUM_DATA_COLUMNS, 12 * LIMBS_PER_FP);
        assert_eq!(NUM_DATA_COLUMNS, 72);
        assert_eq!(COL_SEL_ADD, 72);
        assert_eq!(COL_IS_REAL, 76);
        assert_eq!(NUM_COLUMNS, 77);
        assert_eq!(NUM_ROW_CONSTRAINTS, 12);
        assert_eq!(COL_A_C0_OFFSET, 0);
        assert_eq!(COL_C_C1_OFFSET, 30);
        assert_eq!(COL_M00_OFFSET, 36);
        assert_eq!(COL_M10_OFFSET, 54);
        assert_eq!(COL_NORM_OFFSET, 60);
        assert_eq!(COL_NORM_INV_OFFSET, 66);
    }

    #[test]
    fn fp2_host_arithmetic_known_vectors() {
        // (3 + 5u) + (11 + 13u) = 14 + 18u.
        let a = sample_a();
        let b = sample_b();
        let sum = a.add(&b);
        assert_eq!(sum.c0, Fp::from_u64(14));
        assert_eq!(sum.c1, Fp::from_u64(18));

        // (3 + 5u)(11 + 13u) = (33 - 65) + (39 + 55)u = -32 + 94u.
        let prod = a.mul(&b);
        assert_eq!(prod.c1, Fp::from_u64(94));
        // -32 mod p; check by re-adding 32.
        let back = prod.c0.add(&Fp::from_u64(32));
        assert!(back.is_zero(), "(33 - 65) mod p should be -32");

        // Identity.
        let one2 = Fp2::one();
        assert_eq!(one2.mul(&a), a);

        // Inv round-trip.
        let inv = b.invert().expect("nonzero");
        assert_eq!(b.mul(&inv), Fp2::one(), "Fp2 inv round-trip failed");

        // Zero handling.
        assert!(Fp2::zero().is_zero());
        assert!(Fp2::zero().invert().is_none());
    }

    #[test]
    fn fp2_constraints_zero_on_honest_witness() {
        let ops = vec![
            Fp2Op::Add { a: sample_a(), b: sample_b() },
            Fp2Op::Sub { a: sample_a(), b: sample_b() },
            Fp2Op::Mul { a: sample_a(), b: sample_b() },
            Fp2Op::Inv { a: sample_a() },
        ];
        let columns = populate_trace(&ops, CurveType::Bls12381, Some(8));
        assert_constraints_zero(&columns);
    }

    #[test]
    fn fp2_rejects_double_selector() {
        let ops = vec![Fp2Op::Add { a: sample_a(), b: sample_b() }];
        let curve = CurveType::Bls12381;
        let mut columns = populate_trace(&ops, curve, Some(2));
        // Set sel_mul = 1 simultaneously with sel_add = 1.
        columns[COL_SEL_MUL][0] = Scalar::one(curve);
        let cs = Bls12Fp2ConstraintSystem::new(columns[0].len());
        let refs: Vec<&Vec<Scalar>> = columns.iter().collect();
        let evals = cs.evaluate_on_domain(&refs, columns[0].len());
        // Pairwise mutex `sel_add · sel_mul` (constraint 7) and sum-eq
        // (5) must fire.
        assert!(!evals[7][0].is_zero(), "mutex_add_mul must fire");
        assert!(!evals[5][0].is_zero(), "is_real_sum_eq_selectors must fire");
    }

    #[test]
    fn fp2_padding_rows_are_clean() {
        let columns = populate_trace(&[], CurveType::Bls12381, Some(4));
        assert_constraints_zero(&columns);
    }

    #[test]
    fn fp2_to_fp_linkage_descriptors_well_formed() {
        let descs = fp2_to_fp_linkage_descriptors(0, 1);
        assert_eq!(descs.len(), 10);
        for d in &descs {
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 1);
            assert_eq!(d.a_columns.len(), 3 * LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_FP);
            // B side tuple is (A, B, C) on the Fp AIR.
            assert_eq!(d.b_columns[0], FP_COL_A_OFFSET);
            assert_eq!(d.b_columns[LIMBS_PER_FP], FP_COL_B_OFFSET);
            assert_eq!(d.b_columns[2 * LIMBS_PER_FP], FP_COL_C_OFFSET);
            let asel = d.a_selector_column.unwrap();
            assert!(asel == COL_SEL_ADD || asel == COL_SEL_SUB || asel == COL_SEL_MUL);
            let bsel = d.b_selector_column.unwrap();
            assert!(
                bsel == FP_COL_SEL_ADD || bsel == FP_COL_SEL_SUB || bsel == FP_COL_SEL_MUL,
            );
            assert!(d.label.starts_with("bls12_381_fp2_"));
        }

        // Spot-check: make_fp2_mul_m00_to_fp_mul.
        let d = make_fp2_mul_m00_to_fp_mul_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns[0], COL_A_C0_OFFSET);
        assert_eq!(d.a_columns[LIMBS_PER_FP], COL_B_C0_OFFSET);
        assert_eq!(d.a_columns[2 * LIMBS_PER_FP], COL_M00_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_SEL_MUL));
        assert_eq!(d.b_selector_column, Some(FP_COL_SEL_MUL));

        // Inv linkages: 3 total.
        let inv = fp2_inv_linkage_descriptors(0, 1);
        assert_eq!(inv.len(), 3);
        assert_eq!(inv[0].a_columns[0], COL_A_C0_OFFSET);
        assert_eq!(inv[0].a_columns[LIMBS_PER_FP], COL_A_C0_OFFSET);
        assert_eq!(inv[0].a_columns[2 * LIMBS_PER_FP], COL_M00_OFFSET);
        assert_eq!(inv[0].a_selector_column, Some(COL_SEL_INV));
        assert_eq!(inv[0].b_selector_column, Some(FP_COL_SEL_MUL));
    }

    #[test]
    fn fp2_lookup_declarations_well_formed() {
        let decls = lookup_declarations();
        assert!(!decls.is_empty());
        for d in &decls {
            assert!(d.column_index < NUM_COLUMNS,
                    "decl `{}` references OOB col {}", d.label, d.column_index);
            assert!(d.max_bits == 1 || d.max_bits == 64);
        }
    }
}
