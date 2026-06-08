//! [`VmConstraintSystem`] wiring for the BLS12-381 non-native `Fp` AIR.
//!
//! This module lifts the standalone constraint evaluator in
//! [`crate::nonnative_fp_air`] into the shape required by the generic
//! [`prove_with_scheme`] / [`verify_with_scheme`] pipeline — making this
//! the first end-to-end provable BLS12-381 non-native arithmetic circuit.
//!
//! The underlying AIR has **150 columns** and **19 row-local constraint
//! categories** (no cross-row transitions — each row is self-contained).
//! Every constraint body is either
//!
//!   * gated by one of the four op selectors (`sel_add`, `sel_sub`,
//!     `sel_mul`, `sel_inv`), so it vanishes trivially on padding rows where
//!     every selector is zero, or
//!   * a binary / selector-sum check that is also satisfied when every
//!     column is zero (`0·(0−1) = 0`, `0·(0−1) = 0`).
//!
//! We therefore return [`None`] from [`VmConstraintSystem::padding_selector_column`]
//! and leave every padding row all-zero.
//!
//! # Constraint layout
//!
//! 19 row-local categories (label → index):
//!
//!   0.  `add_limb_chain`
//!   1.  `add_trial_diff`
//!   2.  `add_reduce_flag_def`
//!   3.  `add_result_select`
//!   4.  `sub_limb_chain`
//!   5.  `sub_trial_sum`
//!   6.  `sub_add_back_flag_def`
//!   7.  `sub_result_select`
//!   8.  `mul_schoolbook_ab`           — gated by `sel_mul + sel_inv`
//!   9.  `mul_ab_top_carry_zero`       — gated by `sel_mul + sel_inv`
//!   10. `mul_schoolbook_qp`           — gated by `sel_mul + sel_inv`
//!   11. `mul_qp_top_carry_zero`       — gated by `sel_mul + sel_inv`
//!   12. `mul_sum_equals_prod`         — gated by `sel_mul + sel_inv`
//!   13. `mul_sum_top_carry_zero`      — gated by `sel_mul + sel_inv`
//!   14. `slack_chain`                 — gated by `sel_add + sel_sub + sel_mul + sel_inv`
//!   15. `slack_top_borrow_zero`       — gated by `sel_add + sel_sub + sel_mul + sel_inv`
//!   16. `binary_flags`
//!   17. `selector_sum_01`             — `sel_add + sel_sub + sel_mul + sel_inv ∈ {0,1}`
//!   18. `inv_result_is_one`           — `sel_inv · (r − 1) = 0`
//!
//! Within each category, limb-level sub-constraints are aggregated via
//! powers of a β challenge. We fix β = α (single challenge); the combined
//! polynomial remains a polynomial in α and Schwartz-Zippel still applies.
//!
//! # Range checks
//!
//! The AIR declares 64-bit range checks on every limb-valued column
//! (`a`, `b`, `r`, `add_unred_sum`, `add_trial_diff`, `sub_unred_diff`,
//! `sub_trial_sum`, `quotient`, `slack`, `prod`, `qp`, `mul_carry_ab`,
//! `mul_carry_qp`) and 1-bit range checks on every carry / borrow / flag
//! column.
//!
//! Forwarded verbatim from [`nonnative_fp_air::lookup_declarations`] into
//! a [`LookupRequirements`] with two tables — one 64-bit, one 1-bit.
//! Each declaration is marked always-active (`selector_column: None`)
//! because padding rows hold value zero which lies in both `[0, 2^64)` and
//! `{0, 1}`.
//!
//! # Cost model
//!
//! The AIR is two orders of magnitude smaller than the Keccak-f[1600]
//! bit-level AIR (149 cols vs 13,144). Prove / verify is tractable even
//! in debug mode for a handful of rows. However the schoolbook multiplier
//! poly-form constraints build ~150 polynomial multiplications per row
//! type (12 limbs × up to 6 partial products of degree-n polynomials) —
//! combined with the LogUp domain bump to 256 this can still run for
//! several minutes in debug. The full prove/verify fixture is therefore
//! marked `#[ignore]` with a documented `--release --ignored` run
//! invocation.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::nonnative_fp::P_LIMBS;
use crate::nonnative_fp_air::{
    self, a_limb, add_trial_diff, add_unred_sum, alloc_trace, b_limb,
    lookup_declarations as air_lookup_declarations, populate_row, prod_limb, qp_limb,
    quotient_limb, r_limb, slack_limb, sub_trial_sum, sub_unred_diff, two_pow_64, FpOp,
    COL_ADD_CARRIES_OFFSET, COL_ADD_REDUCE_FLAG, COL_ADD_SUB_BORROWS_OFFSET,
    COL_MUL_CARRY_AB_OFFSET, COL_MUL_CARRY_QP_OFFSET, COL_MUL_SUM_CARRY_OFFSET, COL_R_OFFSET,
    COL_SEL_ADD, COL_SEL_INV, COL_SEL_MUL, COL_SEL_SUB, COL_SLACK_BORROWS_OFFSET,
    COL_SUB_ADD_BACK_FLAG, COL_SUB_ADD_CARRIES_OFFSET, COL_SUB_BORROWS_OFFSET, LIMBS_PER_FP,
    LIMBS_PER_PROD, NUM_NONNATIVE_FP_COLUMNS,
};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Number of consolidated row-local constraint categories.
pub const NUM_ROW_CONSTRAINTS: usize = 19;

// ──── Constraint system ────────────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the BLS12-381 non-native `Fp`
/// AIR.
///
/// Stateless — every call operates solely on the columns supplied by the
/// prover / verifier and the fixed constants drawn from
/// [`nonnative_fp_air`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NonnativeFpConstraintSystem;

impl NonnativeFpConstraintSystem {
    pub const fn new() -> Self {
        Self
    }
}

// ──── Trace-builder helper ─────────────────────────────────────────────

/// Build a [`TracePolynomials`] from a sequence of [`FpOp`]s. Pads to the
/// next power of two automatically (padding rows are all-zero — consistent
/// with our "no selector on padding" strategy).
pub fn build_fp_trace_polynomials(ops: &[FpOp], curve: CurveType) -> TracePolynomials {
    let num_rows = ops.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let mut columns = alloc_trace(padded, curve);
    for (row, op) in ops.iter().enumerate() {
        populate_row(&mut columns, row, op, curve);
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

// ──── Per-point body evaluators ────────────────────────────────────────
//
// Each body mirrors the corresponding category in
// `nonnative_fp_air::evaluate_constraints`. Because `cols` here is a
// row-local slice (one value per column), the index logic is identical —
// we just drop the outer `for row in 0..num_rows` loop.

#[inline]
fn le_to_be_idx(lsb_first_index: usize, n_limbs: usize) -> usize {
    n_limbs - 1 - lsb_first_index
}

/// LSB-first accessors adapted from `nonnative_fp_air`.
struct RowAccess<'a> {
    cols: &'a [Scalar],
}

impl<'a> RowAccess<'a> {
    fn a_le(&self, i: usize) -> Scalar {
        self.cols[a_limb(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn b_le(&self, i: usize) -> Scalar {
        self.cols[b_limb(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn r_le(&self, i: usize) -> Scalar {
        self.cols[r_limb(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn add_unred_le(&self, i: usize) -> Scalar {
        self.cols[add_unred_sum(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn add_trial_le(&self, i: usize) -> Scalar {
        self.cols[add_trial_diff(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn sub_diff_le(&self, i: usize) -> Scalar {
        self.cols[sub_unred_diff(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn sub_trial_le(&self, i: usize) -> Scalar {
        self.cols[sub_trial_sum(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn quot_le(&self, i: usize) -> Scalar {
        self.cols[quotient_limb(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
    fn prod_le(&self, k: usize) -> Scalar {
        self.cols[prod_limb(le_to_be_idx(k, LIMBS_PER_PROD))].clone()
    }
    fn qp_le(&self, k: usize) -> Scalar {
        self.cols[qp_limb(le_to_be_idx(k, LIMBS_PER_PROD))].clone()
    }
    fn slack_le(&self, i: usize) -> Scalar {
        self.cols[slack_limb(le_to_be_idx(i, LIMBS_PER_FP))].clone()
    }
}

/// The p limbs in LSB-first order as scalars.
fn p_limbs_le(curve: CurveType) -> [Scalar; 6] {
    [
        Scalar::from_u64(P_LIMBS[5], curve),
        Scalar::from_u64(P_LIMBS[4], curve),
        Scalar::from_u64(P_LIMBS[3], curve),
        Scalar::from_u64(P_LIMBS[2], curve),
        Scalar::from_u64(P_LIMBS[1], curve),
        Scalar::from_u64(P_LIMBS[0], curve),
    ]
}

// ── CAT 0: add_limb_chain ──
fn eval_add_limb_chain(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let sel_add = &cols[COL_SEL_ADD];

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let c_in = if i == 0 {
            zero.clone()
        } else {
            cols[COL_ADD_CARRIES_OFFSET + i - 1].clone()
        };
        let c_out = cols[COL_ADD_CARRIES_OFFSET + i].clone();
        let lhs = a.a_le(i).add(&a.b_le(i)).add(&c_in);
        let rhs = a.add_unred_le(i).add(&c_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_add.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 1: add_trial_diff ──
fn eval_add_trial_diff(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let p_le = p_limbs_le(curve);
    let sel_add = &cols[COL_SEL_ADD];

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let b_in = if i == 0 {
            zero.clone()
        } else {
            cols[COL_ADD_SUB_BORROWS_OFFSET + i - 1].clone()
        };
        let b_out = cols[COL_ADD_SUB_BORROWS_OFFSET + i].clone();
        let lhs = a.add_unred_le(i).sub(&p_le[i]).sub(&b_in);
        let rhs = a.add_trial_le(i).sub(&b_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_add.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 2: add_reduce_flag_def ──
fn eval_add_reduce_flag_def(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let x = cols[COL_ADD_CARRIES_OFFSET + 5].clone();
    let y = one.sub(&cols[COL_ADD_SUB_BORROWS_OFFSET + 5]);
    let xy = x.mul(&y);
    let or = x.add(&y).sub(&xy);
    let flag = cols[COL_ADD_REDUCE_FLAG].clone();
    let body = flag.sub(&or);
    cols[COL_SEL_ADD].mul(&body)
}

// ── CAT 3: add_result_select ──
fn eval_add_result_select(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let a = RowAccess { cols };
    let sel_add = &cols[COL_SEL_ADD];

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let flag = cols[COL_ADD_REDUCE_FLAG].clone();
        let one_m = one.sub(&flag);
        let rhs = one_m.mul(&a.add_unred_le(i)).add(&flag.mul(&a.add_trial_le(i)));
        let body = a.r_le(i).sub(&rhs);
        let gated = sel_add.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 4: sub_limb_chain ──
fn eval_sub_limb_chain(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let sel_sub = &cols[COL_SEL_SUB];

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let b_in = if i == 0 {
            zero.clone()
        } else {
            cols[COL_SUB_BORROWS_OFFSET + i - 1].clone()
        };
        let b_out = cols[COL_SUB_BORROWS_OFFSET + i].clone();
        let lhs = a.a_le(i).sub(&a.b_le(i)).sub(&b_in);
        let rhs = a.sub_diff_le(i).sub(&b_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_sub.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 5: sub_trial_sum ──
fn eval_sub_trial_sum(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let p_le = p_limbs_le(curve);
    let sel_sub = &cols[COL_SEL_SUB];

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let c_in = if i == 0 {
            zero.clone()
        } else {
            cols[COL_SUB_ADD_CARRIES_OFFSET + i - 1].clone()
        };
        let c_out = cols[COL_SUB_ADD_CARRIES_OFFSET + i].clone();
        let lhs = a.sub_diff_le(i).add(&p_le[i]).add(&c_in);
        let rhs = a.sub_trial_le(i).add(&c_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_sub.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 6: sub_add_back_flag_def ──
fn eval_sub_add_back_flag_def(cols: &[Scalar]) -> Scalar {
    let flag = cols[COL_SUB_ADD_BACK_FLAG].clone();
    let top = cols[COL_SUB_BORROWS_OFFSET + 5].clone();
    let body = flag.sub(&top);
    cols[COL_SEL_SUB].mul(&body)
}

// ── CAT 7: sub_result_select ──
fn eval_sub_result_select(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let a = RowAccess { cols };
    let sel_sub = &cols[COL_SEL_SUB];

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let flag = cols[COL_SUB_ADD_BACK_FLAG].clone();
        let one_m = one.sub(&flag);
        let rhs = one_m.mul(&a.sub_diff_le(i)).add(&flag.mul(&a.sub_trial_le(i)));
        let body = a.r_le(i).sub(&rhs);
        let gated = sel_sub.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 8: mul_schoolbook_ab ──
fn eval_mul_schoolbook_ab(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let sel_mul = cols[COL_SEL_MUL].add(&cols[COL_SEL_INV]);
    let sel_mul = &sel_mul;

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for k in 0..LIMBS_PER_PROD {
        let mut col_sum = zero.clone();
        for i in 0..LIMBS_PER_FP {
            let j = k as isize - i as isize;
            if j < 0 || j >= LIMBS_PER_FP as isize {
                continue;
            }
            col_sum = col_sum.add(&a.a_le(i).mul(&a.b_le(j as usize)));
        }
        let c_in = if k == 0 {
            zero.clone()
        } else {
            cols[COL_MUL_CARRY_AB_OFFSET + k - 1].clone()
        };
        let c_out = cols[COL_MUL_CARRY_AB_OFFSET + k].clone();
        let lhs = col_sum.add(&c_in);
        let rhs = a.prod_le(k).add(&c_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_mul.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 9: mul_ab_top_carry_zero ──
fn eval_mul_ab_top_carry_zero(cols: &[Scalar]) -> Scalar {
    let top = cols[COL_MUL_CARRY_AB_OFFSET + 11].clone();
    cols[COL_SEL_MUL].add(&cols[COL_SEL_INV]).mul(&top)
}

// ── CAT 10: mul_schoolbook_qp ──
fn eval_mul_schoolbook_qp(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let p_le = p_limbs_le(curve);
    let sel_mul = cols[COL_SEL_MUL].add(&cols[COL_SEL_INV]);
    let sel_mul = &sel_mul;

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for k in 0..LIMBS_PER_PROD {
        let mut col_sum = zero.clone();
        for i in 0..LIMBS_PER_FP {
            let j = k as isize - i as isize;
            if j < 0 || j >= LIMBS_PER_FP as isize {
                continue;
            }
            col_sum = col_sum.add(&a.quot_le(i).mul(&p_le[j as usize]));
        }
        let c_in = if k == 0 {
            zero.clone()
        } else {
            cols[COL_MUL_CARRY_QP_OFFSET + k - 1].clone()
        };
        let c_out = cols[COL_MUL_CARRY_QP_OFFSET + k].clone();
        let lhs = col_sum.add(&c_in);
        let rhs = a.qp_le(k).add(&c_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_mul.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 11: mul_qp_top_carry_zero ──
fn eval_mul_qp_top_carry_zero(cols: &[Scalar]) -> Scalar {
    let top = cols[COL_MUL_CARRY_QP_OFFSET + 11].clone();
    cols[COL_SEL_MUL].add(&cols[COL_SEL_INV]).mul(&top)
}

// ── CAT 12: mul_sum_equals_prod ──
fn eval_mul_sum_equals_prod(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let sel_mul = cols[COL_SEL_MUL].add(&cols[COL_SEL_INV]);
    let sel_mul = &sel_mul;

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for k in 0..LIMBS_PER_PROD {
        let r_term = if k < LIMBS_PER_FP {
            a.r_le(k)
        } else {
            zero.clone()
        };
        let c_in = if k == 0 {
            zero.clone()
        } else {
            cols[COL_MUL_SUM_CARRY_OFFSET + k - 1].clone()
        };
        let c_out = cols[COL_MUL_SUM_CARRY_OFFSET + k].clone();
        let lhs = a.qp_le(k).add(&r_term).add(&c_in);
        let rhs = a.prod_le(k).add(&c_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_mul.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 13: mul_sum_top_carry_zero ──
fn eval_mul_sum_top_carry_zero(cols: &[Scalar]) -> Scalar {
    let top = cols[COL_MUL_SUM_CARRY_OFFSET + 11].clone();
    cols[COL_SEL_MUL].add(&cols[COL_SEL_INV]).mul(&top)
}

// ── CAT 14: slack_chain ──
fn eval_slack_chain(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let two64 = two_pow_64(curve);
    let a = RowAccess { cols };
    let p_le = p_limbs_le(curve);
    let sel_any = cols[COL_SEL_ADD]
        .add(&cols[COL_SEL_SUB])
        .add(&cols[COL_SEL_MUL])
        .add(&cols[COL_SEL_INV]);

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let p_minus_1_at_i = if i == 0 { p_le[0].sub(&one) } else { p_le[i].clone() };
        let b_in = if i == 0 {
            zero.clone()
        } else {
            cols[COL_SLACK_BORROWS_OFFSET + i - 1].clone()
        };
        let b_out = cols[COL_SLACK_BORROWS_OFFSET + i].clone();
        let lhs = p_minus_1_at_i.sub(&a.r_le(i)).sub(&b_in);
        let rhs = a.slack_le(i).sub(&b_out.mul(&two64));
        let body = lhs.sub(&rhs);
        let gated = sel_any.mul(&body);
        acc = acc.add(&bp.mul(&gated));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 15: slack_top_borrow_zero ──
fn eval_slack_top_borrow_zero(cols: &[Scalar]) -> Scalar {
    let sel_any = cols[COL_SEL_ADD]
        .add(&cols[COL_SEL_SUB])
        .add(&cols[COL_SEL_MUL])
        .add(&cols[COL_SEL_INV]);
    let top = cols[COL_SLACK_BORROWS_OFFSET + 5].clone();
    sel_any.mul(&top)
}

// List of indices of all binary-valued columns, in the exact order used by
// `nonnative_fp_air::evaluate_constraints` CAT 17 to assemble the β-RLC.
fn binary_column_indices() -> Vec<usize> {
    let mut v = Vec::new();
    for i in 0..LIMBS_PER_FP {
        v.push(COL_ADD_CARRIES_OFFSET + i);
    }
    for i in 0..LIMBS_PER_FP {
        v.push(COL_ADD_SUB_BORROWS_OFFSET + i);
    }
    v.push(COL_ADD_REDUCE_FLAG);
    for i in 0..LIMBS_PER_FP {
        v.push(COL_SUB_BORROWS_OFFSET + i);
    }
    for i in 0..LIMBS_PER_FP {
        v.push(COL_SUB_ADD_CARRIES_OFFSET + i);
    }
    v.push(COL_SUB_ADD_BACK_FLAG);
    for k in 0..LIMBS_PER_PROD {
        v.push(COL_MUL_SUM_CARRY_OFFSET + k);
    }
    v.push(COL_SEL_ADD);
    v.push(COL_SEL_SUB);
    v.push(COL_SEL_MUL);
    v.push(COL_SEL_INV);
    v
}

// ── CAT 16: binary_flags ──
fn eval_binary_flags(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for col_idx in binary_column_indices() {
        let v = cols[col_idx].clone();
        let body = v.mul(&v.sub(&one));
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

// ── CAT 17: selector_sum_01 ──
fn eval_selector_sum_01(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let s = cols[COL_SEL_ADD]
        .add(&cols[COL_SEL_SUB])
        .add(&cols[COL_SEL_MUL])
        .add(&cols[COL_SEL_INV]);
    s.mul(&s.sub(&one))
}

// ── CAT 18: inv_result_is_one ──
// On Inv rows, r must equal Fp::one() — i.e. r limbs in BE = [0,0,0,0,0,1].
// β-RLC over the 6 limbs: high limbs use body = r[i], low limb uses r[5] - 1.
fn eval_inv_result_is_one(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let sel = &cols[COL_SEL_INV];
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for be in 0..LIMBS_PER_FP {
        let r_be = cols[COL_R_OFFSET + be].clone();
        let target = if be == LIMBS_PER_FP - 1 {
            r_be.sub(&one)
        } else {
            r_be
        };
        let body = sel.mul(&target);
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

// ──── Polynomial-form builders for each category body ──────────────────
//
// Each returns the coefficient-form polynomial representing the category's
// β-aggregated body. The outer caller combines them with α-powers.
//
// In polynomial form:
//   * `cols[i]` is the coefficient-form polynomial of column `i`.
//   * `poly_mul`, `poly_add`, `poly_sub`, `poly_scalar_mul` provide the
//     standard coefficient-form operations.
//   * Constants like `2^64` and `p_le[i]` are lifted to degree-0 polys via
//     a single-element `Vec<Scalar>`.

fn scalar_poly(s: Scalar) -> Vec<Scalar> {
    vec![s]
}

fn build_add_limb_chain_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two64 = scalar_poly(two_pow_64(curve));
    let sel = &cols[COL_SEL_ADD];

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let a = &cols[a_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let b = &cols[b_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let unred = &cols[add_unred_sum(le_to_be_idx(i, LIMBS_PER_FP))];
        let c_out = &cols[COL_ADD_CARRIES_OFFSET + i];
        let c_out_scaled = poly_mul(c_out, &two64, curve);

        // lhs = a + b (+ c_in)
        let mut lhs = poly_add(a, b, curve);
        if i > 0 {
            let c_in = &cols[COL_ADD_CARRIES_OFFSET + i - 1];
            lhs = poly_add(&lhs, c_in, curve);
        }
        // rhs = unred + c_out * 2^64
        let rhs = poly_add(unred, &c_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_add_trial_diff_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two64 = scalar_poly(two_pow_64(curve));
    let p_le = p_limbs_le(curve);
    let sel = &cols[COL_SEL_ADD];

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let unred = &cols[add_unred_sum(le_to_be_idx(i, LIMBS_PER_FP))];
        let trial = &cols[add_trial_diff(le_to_be_idx(i, LIMBS_PER_FP))];
        let p_const = scalar_poly(p_le[i].clone());
        let b_out = &cols[COL_ADD_SUB_BORROWS_OFFSET + i];
        let b_out_scaled = poly_mul(b_out, &two64, curve);

        // lhs = unred - p - b_in
        let mut lhs = poly_sub(unred, &p_const, curve);
        if i > 0 {
            let b_in = &cols[COL_ADD_SUB_BORROWS_OFFSET + i - 1];
            lhs = poly_sub(&lhs, b_in, curve);
        }
        // rhs = trial - b_out*2^64
        let rhs = poly_sub(trial, &b_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_add_reduce_flag_def_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let x = &cols[COL_ADD_CARRIES_OFFSET + 5];
    let y_poly = poly_sub(&one_poly, &cols[COL_ADD_SUB_BORROWS_OFFSET + 5], curve);
    let xy = poly_mul(x, &y_poly, curve);
    let or = poly_sub(&poly_add(x, &y_poly, curve), &xy, curve);
    let flag = &cols[COL_ADD_REDUCE_FLAG];
    let body = poly_sub(flag, &or, curve);
    poly_mul(&cols[COL_SEL_ADD], &body, curve)
}

fn build_add_result_select_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let sel = &cols[COL_SEL_ADD];
    let flag = &cols[COL_ADD_REDUCE_FLAG];
    let one_m_flag = poly_sub(&one_poly, flag, curve);

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let r = &cols[r_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let unred = &cols[add_unred_sum(le_to_be_idx(i, LIMBS_PER_FP))];
        let trial = &cols[add_trial_diff(le_to_be_idx(i, LIMBS_PER_FP))];
        let t1 = poly_mul(&one_m_flag, unred, curve);
        let t2 = poly_mul(flag, trial, curve);
        let rhs = poly_add(&t1, &t2, curve);
        let body = poly_sub(r, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_sub_limb_chain_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two64 = scalar_poly(two_pow_64(curve));
    let sel = &cols[COL_SEL_SUB];

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let a = &cols[a_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let b = &cols[b_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let diff = &cols[sub_unred_diff(le_to_be_idx(i, LIMBS_PER_FP))];
        let b_out = &cols[COL_SUB_BORROWS_OFFSET + i];
        let b_out_scaled = poly_mul(b_out, &two64, curve);

        // lhs = a - b - b_in
        let mut lhs = poly_sub(a, b, curve);
        if i > 0 {
            let b_in = &cols[COL_SUB_BORROWS_OFFSET + i - 1];
            lhs = poly_sub(&lhs, b_in, curve);
        }
        // rhs = diff - b_out*2^64
        let rhs = poly_sub(diff, &b_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_sub_trial_sum_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two64 = scalar_poly(two_pow_64(curve));
    let p_le = p_limbs_le(curve);
    let sel = &cols[COL_SEL_SUB];

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let diff = &cols[sub_unred_diff(le_to_be_idx(i, LIMBS_PER_FP))];
        let trial = &cols[sub_trial_sum(le_to_be_idx(i, LIMBS_PER_FP))];
        let p_const = scalar_poly(p_le[i].clone());
        let c_out = &cols[COL_SUB_ADD_CARRIES_OFFSET + i];
        let c_out_scaled = poly_mul(c_out, &two64, curve);

        // lhs = diff + p + c_in
        let mut lhs = poly_add(diff, &p_const, curve);
        if i > 0 {
            let c_in = &cols[COL_SUB_ADD_CARRIES_OFFSET + i - 1];
            lhs = poly_add(&lhs, c_in, curve);
        }
        // rhs = trial + c_out*2^64
        let rhs = poly_add(trial, &c_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_sub_add_back_flag_def_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let flag = &cols[COL_SUB_ADD_BACK_FLAG];
    let top = &cols[COL_SUB_BORROWS_OFFSET + 5];
    let body = poly_sub(flag, top, curve);
    poly_mul(&cols[COL_SEL_SUB], &body, curve)
}

fn build_sub_result_select_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let sel = &cols[COL_SEL_SUB];
    let flag = &cols[COL_SUB_ADD_BACK_FLAG];
    let one_m_flag = poly_sub(&one_poly, flag, curve);

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let r = &cols[r_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let diff = &cols[sub_unred_diff(le_to_be_idx(i, LIMBS_PER_FP))];
        let trial = &cols[sub_trial_sum(le_to_be_idx(i, LIMBS_PER_FP))];
        let t1 = poly_mul(&one_m_flag, diff, curve);
        let t2 = poly_mul(flag, trial, curve);
        let rhs = poly_add(&t1, &t2, curve);
        let body = poly_sub(r, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_mul_schoolbook_ab_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two64 = scalar_poly(two_pow_64(curve));
    let sel = poly_add(&cols[COL_SEL_MUL], &cols[COL_SEL_INV], curve);
    let sel = &sel;

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for k in 0..LIMBS_PER_PROD {
        // col_sum = Σ_{i+j=k; i,j<6} a[i]*b[j]
        let mut col_sum = vec![Scalar::zero(curve)];
        for i in 0..LIMBS_PER_FP {
            let j = k as isize - i as isize;
            if j < 0 || j >= LIMBS_PER_FP as isize {
                continue;
            }
            let a = &cols[a_limb(le_to_be_idx(i, LIMBS_PER_FP))];
            let b = &cols[b_limb(le_to_be_idx(j as usize, LIMBS_PER_FP))];
            let ab = poly_mul(a, b, curve);
            col_sum = poly_add(&col_sum, &ab, curve);
        }
        let prod_k = &cols[prod_limb(le_to_be_idx(k, LIMBS_PER_PROD))];
        let c_out = &cols[COL_MUL_CARRY_AB_OFFSET + k];
        let c_out_scaled = poly_mul(c_out, &two64, curve);

        let lhs = if k == 0 {
            col_sum
        } else {
            let c_in = &cols[COL_MUL_CARRY_AB_OFFSET + k - 1];
            poly_add(&col_sum, c_in, curve)
        };
        let rhs = poly_add(prod_k, &c_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_mul_ab_top_carry_zero_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let top = &cols[COL_MUL_CARRY_AB_OFFSET + 11];
    let sel = poly_add(&cols[COL_SEL_MUL], &cols[COL_SEL_INV], curve);
    poly_mul(&sel, top, curve)
}

fn build_mul_schoolbook_qp_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two64 = scalar_poly(two_pow_64(curve));
    let p_le = p_limbs_le(curve);
    let sel = poly_add(&cols[COL_SEL_MUL], &cols[COL_SEL_INV], curve);
    let sel = &sel;

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for k in 0..LIMBS_PER_PROD {
        // col_sum = Σ_{i+j=k; i,j<6} q[i]*p[j]  (p is a constant → poly_scalar_mul)
        let mut col_sum = vec![Scalar::zero(curve)];
        for i in 0..LIMBS_PER_FP {
            let j = k as isize - i as isize;
            if j < 0 || j >= LIMBS_PER_FP as isize {
                continue;
            }
            let q = &cols[quotient_limb(le_to_be_idx(i, LIMBS_PER_FP))];
            let scaled = poly_scalar_mul(q, &p_le[j as usize]);
            col_sum = poly_add(&col_sum, &scaled, curve);
        }
        let qp_k = &cols[qp_limb(le_to_be_idx(k, LIMBS_PER_PROD))];
        let c_out = &cols[COL_MUL_CARRY_QP_OFFSET + k];
        let c_out_scaled = poly_mul(c_out, &two64, curve);

        let lhs = if k == 0 {
            col_sum
        } else {
            let c_in = &cols[COL_MUL_CARRY_QP_OFFSET + k - 1];
            poly_add(&col_sum, c_in, curve)
        };
        let rhs = poly_add(qp_k, &c_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_mul_qp_top_carry_zero_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let top = &cols[COL_MUL_CARRY_QP_OFFSET + 11];
    let sel = poly_add(&cols[COL_SEL_MUL], &cols[COL_SEL_INV], curve);
    poly_mul(&sel, top, curve)
}

fn build_mul_sum_equals_prod_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two64 = scalar_poly(two_pow_64(curve));
    let sel = poly_add(&cols[COL_SEL_MUL], &cols[COL_SEL_INV], curve);
    let sel = &sel;

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for k in 0..LIMBS_PER_PROD {
        let qp_k = &cols[qp_limb(le_to_be_idx(k, LIMBS_PER_PROD))];
        let prod_k = &cols[prod_limb(le_to_be_idx(k, LIMBS_PER_PROD))];
        let c_out = &cols[COL_MUL_SUM_CARRY_OFFSET + k];
        let c_out_scaled = poly_mul(c_out, &two64, curve);

        // lhs = qp[k] + r_pad[k] + c_in
        let mut lhs = if k < LIMBS_PER_FP {
            let r = &cols[r_limb(le_to_be_idx(k, LIMBS_PER_FP))];
            poly_add(qp_k, r, curve)
        } else {
            qp_k.clone()
        };
        if k > 0 {
            let c_in = &cols[COL_MUL_SUM_CARRY_OFFSET + k - 1];
            lhs = poly_add(&lhs, c_in, curve);
        }
        // rhs = prod + c_out*2^64
        let rhs = poly_add(prod_k, &c_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(sel, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_mul_sum_top_carry_zero_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let top = &cols[COL_MUL_SUM_CARRY_OFFSET + 11];
    let sel = poly_add(&cols[COL_SEL_MUL], &cols[COL_SEL_INV], curve);
    poly_mul(&sel, top, curve)
}

fn build_slack_chain_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let two64 = scalar_poly(two_pow_64(curve));
    let p_le = p_limbs_le(curve);
    let sel_any = poly_add(
        &poly_add(
            &poly_add(&cols[COL_SEL_ADD], &cols[COL_SEL_SUB], curve),
            &cols[COL_SEL_MUL],
            curve,
        ),
        &cols[COL_SEL_INV],
        curve,
    );

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for i in 0..LIMBS_PER_FP {
        let p_minus_1_at_i = if i == 0 {
            p_le[0].sub(&one)
        } else {
            p_le[i].clone()
        };
        let p_const = scalar_poly(p_minus_1_at_i);
        let r = &cols[r_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let slack = &cols[slack_limb(le_to_be_idx(i, LIMBS_PER_FP))];
        let b_out = &cols[COL_SLACK_BORROWS_OFFSET + i];
        let b_out_scaled = poly_mul(b_out, &two64, curve);

        // lhs = p_minus_1_at_i - r - b_in
        let mut lhs = poly_sub(&p_const, r, curve);
        if i > 0 {
            let b_in = &cols[COL_SLACK_BORROWS_OFFSET + i - 1];
            lhs = poly_sub(&lhs, b_in, curve);
        }
        // rhs = slack - b_out*2^64
        let rhs = poly_sub(slack, &b_out_scaled, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        let gated = poly_mul(&sel_any, &body, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_slack_top_borrow_zero_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let sel_any = poly_add(
        &poly_add(
            &poly_add(&cols[COL_SEL_ADD], &cols[COL_SEL_SUB], curve),
            &cols[COL_SEL_MUL],
            curve,
        ),
        &cols[COL_SEL_INV],
        curve,
    );
    let top = &cols[COL_SLACK_BORROWS_OFFSET + 5];
    poly_mul(&sel_any, top, curve)
}

fn build_binary_flags_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for col_idx in binary_column_indices() {
        let v = &cols[col_idx];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_selector_sum_01_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let sum = poly_add(
        &poly_add(
            &poly_add(&cols[COL_SEL_ADD], &cols[COL_SEL_SUB], curve),
            &cols[COL_SEL_MUL],
            curve,
        ),
        &cols[COL_SEL_INV],
        curve,
    );
    let sum_m1 = poly_sub(&sum, &one_poly, curve);
    poly_mul(&sum, &sum_m1, curve)
}

fn build_inv_result_is_one_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let sel = &cols[COL_SEL_INV];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for be in 0..LIMBS_PER_FP {
        let r_be = &cols[COL_R_OFFSET + be];
        let target = if be == LIMBS_PER_FP - 1 {
            poly_sub(r_be, &one_poly, curve)
        } else {
            r_be.clone()
        };
        let body = poly_mul(sel, &target, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

// ──── VmConstraintSystem implementation ────────────────────────────────

impl VmConstraintSystem for NonnativeFpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "add_limb_chain".into(),
            "add_trial_diff".into(),
            "add_reduce_flag_def".into(),
            "add_result_select".into(),
            "sub_limb_chain".into(),
            "sub_trial_sum".into(),
            "sub_add_back_flag_def".into(),
            "sub_result_select".into(),
            "mul_schoolbook_ab".into(),
            "mul_ab_top_carry_zero".into(),
            "mul_schoolbook_qp".into(),
            "mul_qp_top_carry_zero".into(),
            "mul_sum_equals_prod".into(),
            "mul_sum_top_carry_zero".into(),
            "slack_chain".into(),
            "slack_top_borrow_zero".into(),
            "binary_flags".into(),
            "selector_sum_01".into(),
            "inv_result_is_one".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(
            columns.len() == NUM_NONNATIVE_FP_COLUMNS,
            "nonnative Fp AIR expects {} columns",
            NUM_NONNATIVE_FP_COLUMNS
        );
        // Fixed non-zero β so the returned evaluations are deterministic.
        // Each sub-constraint vanishes on every row of a valid witness, so
        // the choice of β does not affect soundness of this helper.
        let curve = columns[0][0].curve_type();
        let beta = Scalar::from_u64(2, curve);
        let evals = nonnative_fp_air::evaluate_constraints(columns, &beta);
        evals.into_iter().map(|c| c.values).collect()
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < NUM_NONNATIVE_FP_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let beta = alpha;
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            eval_add_limb_chain(col_evals_at_z, beta),
            eval_add_trial_diff(col_evals_at_z, beta),
            eval_add_reduce_flag_def(col_evals_at_z),
            eval_add_result_select(col_evals_at_z, beta),
            eval_sub_limb_chain(col_evals_at_z, beta),
            eval_sub_trial_sum(col_evals_at_z, beta),
            eval_sub_add_back_flag_def(col_evals_at_z),
            eval_sub_result_select(col_evals_at_z, beta),
            eval_mul_schoolbook_ab(col_evals_at_z, beta),
            eval_mul_ab_top_carry_zero(col_evals_at_z),
            eval_mul_schoolbook_qp(col_evals_at_z, beta),
            eval_mul_qp_top_carry_zero(col_evals_at_z),
            eval_mul_sum_equals_prod(col_evals_at_z, beta),
            eval_mul_sum_top_carry_zero(col_evals_at_z),
            eval_slack_chain(col_evals_at_z, beta),
            eval_slack_top_borrow_zero(col_evals_at_z),
            eval_binary_flags(col_evals_at_z, beta),
            eval_selector_sum_01(col_evals_at_z),
            eval_inv_result_is_one(col_evals_at_z, beta),
        ];
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
        vec![COL_SEL_ADD, COL_SEL_SUB, COL_SEL_MUL, COL_SEL_INV]
    }

    /// Padding rows carry all-zero columns (including all three selectors),
    /// which makes every row-local body vanish. The `selector_sum_01`
    /// constraint `s·(s−1) = 0` is satisfied with s = 0.
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
        if columns.len() < NUM_NONNATIVE_FP_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_NONNATIVE_FP_COLUMNS) {
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
        use rayon::prelude::*;
        let curve = alpha.curve_type();
        let beta = alpha.clone();
        type BuilderRet = Vec<Scalar>;
        let category_builders: Vec<Box<dyn Fn() -> BuilderRet + Sync + Send>> = vec![
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_add_limb_chain_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_add_trial_diff_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_add_reduce_flag_def_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_add_result_select_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_sub_limb_chain_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_sub_trial_sum_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_sub_add_back_flag_def_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_sub_result_select_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_mul_schoolbook_ab_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_mul_ab_top_carry_zero_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_mul_schoolbook_qp_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_mul_qp_top_carry_zero_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_mul_sum_equals_prod_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_mul_sum_top_carry_zero_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_slack_chain_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_slack_top_borrow_zero_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_binary_flags_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_selector_sum_01_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_inv_result_is_one_poly(&cols, &b)
            }),
        ];
        let bodies: Vec<Vec<Scalar>> = category_builders.par_iter().map(|f| f()).collect();

        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            let scaled = poly_scalar_mul(body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    // No cross-row constraints — this AIR is purely row-local.

    fn lookup_declarations(&self) -> LookupRequirements {
        // Group every declaration from the AIR module into three tables:
        // 64-bit range (most limbs), 72-bit range (mul schoolbook carries
        // for full Fp · Fp products that can reach ~2^67), and 1-bit
        // (binary). `selector_column: None` for all — padding rows are
        // all-zero which lies in every range.
        let table64 = LookupTable::range(64);
        let table72 = LookupTable::range(72);
        let table1 = LookupTable::range(1);

        let mut declarations: Vec<(LookupDeclaration, usize)> = Vec::new();
        for d in air_lookup_declarations() {
            let table_idx = match d.max_bits {
                64 => 0,
                72 => 1,
                1 => 2,
                other => panic!(
                    "unexpected nonnative_fp_air declaration `{}` width {}",
                    d.label, other,
                ),
            };
            declarations.push((d, table_idx));
        }

        LookupRequirements {
            tables: vec![table64, table72, table1],
            declarations,
        }
    }
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;
    use crate::nonnative_fp::Fp;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    #[test]
    fn nonnative_fp_cs_labels_and_counts() {
        let cs = NonnativeFpConstraintSystem::new();
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), 0);
        assert_eq!(
            cs.selector_column_indices(),
            vec![COL_SEL_ADD, COL_SEL_SUB, COL_SEL_MUL, COL_SEL_INV],
        );
        assert!(cs.shifted_column_indices().is_empty());
        assert!(cs.padding_selector_column().is_none());
    }

    #[test]
    fn nonnative_fp_cs_lookup_declarations_are_well_formed() {
        let cs = NonnativeFpConstraintSystem::new();
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 3);
        assert_eq!(reqs.tables[0].bits, 64);
        assert_eq!(reqs.tables[1].bits, 72);
        assert_eq!(reqs.tables[2].bits, 1);
        // Every declaration must reference a valid column and be routed to
        // the table whose width matches its declared max_bits.
        assert!(!reqs.declarations.is_empty());
        for (d, table_idx) in &reqs.declarations {
            assert!(
                d.column_index < NUM_NONNATIVE_FP_COLUMNS,
                "declaration `{}` references out-of-range column {}",
                d.label,
                d.column_index
            );
            assert_eq!(d.max_bits, reqs.tables[*table_idx].bits);
            // Always-active: padding rows are all-zero so no selector gating
            // is needed.
            assert!(d.selector_column.is_none());
        }
    }

    #[test]
    fn nonnative_fp_cs_evaluate_on_domain_matches_air_helper() {
        let ops = vec![
            FpOp::Add { a: Fp::from_u64(3), b: Fp::from_u64(7) },
            FpOp::Sub { a: Fp::from_u64(100), b: Fp::from_u64(42) },
            FpOp::Mul { a: Fp::from_u64(5), b: Fp::from_u64(11) },
            FpOp::Inv { a: Fp::from_u64(13) },
        ];
        let curve = CurveType::Bls48581;
        let columns = nonnative_fp_air::populate_trace(&ops, curve, None);

        let cs = NonnativeFpConstraintSystem::new();
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, ops.len());
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

    /// `evaluate_at_point` over real rows of a valid witness must be zero:
    /// each category body is zero on each real row, so the α-RLC is zero.
    #[test]
    fn nonnative_fp_cs_evaluate_at_point_zero_on_real_rows() {
        let ops = vec![
            FpOp::Add { a: Fp::from_u64(3), b: Fp::from_u64(7) },
            FpOp::Sub { a: Fp::from_u64(100), b: Fp::from_u64(42) },
            FpOp::Mul { a: Fp::from_u64(5), b: Fp::from_u64(11) },
            FpOp::Mul { a: Fp::from_u64(1 << 30), b: Fp::from_u64(1 << 30) },
            FpOp::Inv { a: Fp::from_u64(31337) },
        ];
        let curve = CurveType::Bls48581;
        let columns = nonnative_fp_air::populate_trace(&ops, curve, None);

        let cs = NonnativeFpConstraintSystem::new();
        let alpha = Scalar::from_u64(17, curve);
        for row in 0..ops.len() {
            let col_vals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let c_at_row = cs.evaluate_at_point(&col_vals, &alpha);
            assert!(
                c_at_row.is_zero(),
                "combined constraint C(row {}) nonzero on valid witness",
                row
            );
        }
    }

    /// Mutating an `r_limb` on a valid Mul row must make `evaluate_at_point`
    /// non-zero there — the verifier's per-point check rejects the proof.
    #[test]
    fn nonnative_fp_cs_evaluate_at_point_detects_mutation() {
        let a = Fp::from_u64(123_456_789);
        let b = Fp::from_u64(987_654_321);
        let ops = vec![FpOp::Mul { a, b }];
        let curve = CurveType::Bls48581;
        let mut columns = nonnative_fp_air::populate_trace(&ops, curve, None);

        let one = Scalar::one(curve);
        columns[r_limb(5)][0] = columns[r_limb(5)][0].add(&one);

        let cs = NonnativeFpConstraintSystem::new();
        let alpha = Scalar::from_u64(19, curve);
        let col_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at = cs.evaluate_at_point(&col_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "tampered r_limb should make C(row 0) non-zero"
        );
    }

    /// The padding-row constraint contract: on an all-zero row, every
    /// body in `evaluate_at_point` is zero. This is the reason we can
    /// return `None` from `padding_selector_column`.
    #[test]
    fn nonnative_fp_cs_evaluate_at_point_zero_on_all_zero_row() {
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let col_vals = vec![zero.clone(); NUM_NONNATIVE_FP_COLUMNS];
        let cs = NonnativeFpConstraintSystem::new();
        let alpha = Scalar::from_u64(23, curve);
        let c_at = cs.evaluate_at_point(&col_vals, &alpha);
        assert!(
            c_at.is_zero(),
            "all-zero padding row must evaluate to zero (otherwise \
             padding_selector_column must be set)"
        );
    }

    /// The coefficient-form builder vanishes at every real domain row on a
    /// valid witness (same as the row-local bodies). Padding rows produce
    /// zero too because all columns are zero there.
    #[test]
    fn nonnative_fp_cs_build_constraint_polynomial_vanishes_on_domain() {
        let ops = vec![
            FpOp::Add { a: Fp::from_u64(3), b: Fp::from_u64(7) },
            FpOp::Sub { a: Fp::from_u64(100), b: Fp::from_u64(42) },
        ];
        let curve = CurveType::Bls48581;
        let trace = build_fp_trace_polynomials(&ops, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let cs = NonnativeFpConstraintSystem::new();
        let alpha = Scalar::from_u64(13, curve);
        let c_coeffs = cs.build_constraint_polynomial(&coeff_form, &alpha, n);

        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let mut z = Scalar::one(curve);
        for r in 0..(n as usize) {
            let v = CommitmentScheme::eval_poly_at(&scheme, &c_coeffs, &z);
            assert!(
                v.is_zero(),
                "C(ω^{}) not zero — row-local bodies should vanish on every domain point",
                r
            );
            z = z.mul(&omega);
        }
    }

    /// End-to-end prove/verify for a 4-op Fp trace.
    ///
    /// Marked `#[ignore]` because the schoolbook multiplier build_poly
    /// stages multiply ~150 degree-`domain_size` polynomials in BLS48-581,
    /// which is slow in debug mode. Re-enable with:
    ///
    ///     cargo test --release -p metavm-zkp --lib \
    ///         nonnative_fp_cs_prove_verify_small -- --ignored --nocapture
    #[test]
    #[ignore = "slow in debug: ~150 degree-n poly_muls under BLS48-581; run with --release --ignored"]
    fn nonnative_fp_cs_prove_verify_small() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let ops = vec![
            FpOp::Add { a: Fp::from_u64(3), b: Fp::from_u64(7) },
            FpOp::Sub { a: Fp::from_u64(100), b: Fp::from_u64(42) },
            FpOp::Mul { a: Fp::from_u64(5), b: Fp::from_u64(11) },
            FpOp::Mul { a: Fp::from_u64(1 << 30), b: Fp::from_u64(1 << 30) },
            FpOp::Inv { a: Fp::from_u64(0xC0FFEE) },
        ];

        let trace = build_fp_trace_polynomials(&ops, curve);
        let cs = NonnativeFpConstraintSystem::new();
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "nonnative Fp prove/verify must succeed");
    }

    /// End-to-end: produce a real Fp ExecutionProof, serialize, attach to
    /// the Attestation layer (BLS pairing is non-native Fp arithmetic),
    /// and verify through `LayerChainProof::verify_with_layer_verifier`.
    /// Validates the LayerProofKind::NonnativeFp dispatch path.
    #[test]
    #[ignore = "slow: produces a real Fp proof; run with --release --ignored"]
    fn nonnative_fp_proof_flows_through_layer_chain_envelope() {
        use crate::layer_chain::{
            ChainBoundaries, LayerChainProof, LayerProof, LayerProofKind,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let ops = vec![
            FpOp::Add { a: Fp::from_u64(3), b: Fp::from_u64(7) },
            FpOp::Mul { a: Fp::from_u64(5), b: Fp::from_u64(11) },
            FpOp::Inv { a: Fp::from_u64(0xC0FFEE) },
        ];

        let trace = build_fp_trace_polynomials(&ops, curve);
        let cs = NonnativeFpConstraintSystem::new();
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        assert!(crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve));

        let proof_bytes = proof.to_bytes();
        assert!(!proof_bytes.is_empty());

        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: 1,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = crate::layer_chain::LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 2 {
                    // Attestation layer carries the real Fp proof.
                    LayerProof::with_proof(
                        claim,
                        LayerProofKind::NonnativeFp,
                        proof_bytes.clone(),
                    )
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        let result = chain_proof.verify_with_layer_verifier(|layer| match layer.kind {
            LayerProofKind::NonnativeFp => {
                let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                    .map_err(|e| format!("decode failed: {:?}", e))?;
                let cs = NonnativeFpConstraintSystem::new();
                if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                    Ok(())
                } else {
                    Err("NonnativeFp proof did not verify".to_string())
                }
            }
            LayerProofKind::ReferenceOnly => Ok(()),
            other => Err(format!("unsupported layer kind {}", other.as_str())),
        });
        assert_eq!(
            result,
            Ok(()),
            "real NonnativeFp proof must verify through the LayerChainProof envelope",
        );
    }
}
