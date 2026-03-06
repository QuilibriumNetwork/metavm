//! Comparison constraints: LT, GT, EQ, ISZERO with algebraic verification.
//!
//! LT/GT use a 256-bit subtraction borrow chain:
//!   For LT: compute input0 - input1 with borrow chain in aux0[0..3],
//!           diff values in aux1[0..3]. output_l0 = aux0[3] (final borrow).
//!   For GT: compute input1 - input0 with same layout.
//!
//! EQ: output = 1 iff input0 == input1 (all 4 limb differences are zero).
//!   Constraint: output * diff_k = 0 for each k, plus
//!   (1 - output) * (diff0 + diff1 + diff2 + diff3) != 0 (via inverse witness).
//!
//! ISZERO: output = 1 iff input0 == 0.
//!   Constraint: output * (sum of input limbs) = 0.

use metavm_zkp::field::Scalar;
use crate::trace::*;

/// Evaluate LT constraints on the full trace domain (unused, kept for API consistency).
pub fn evaluate_lt(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[COL_SEL_LT][i];
        if sel.is_zero() {
            result.push(Scalar::zero(curve));
        } else {
            let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_lt_raw(&cols_at_i));
        }
    }
    result
}

/// Evaluate GT constraints on the full trace domain.
pub fn evaluate_gt(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[COL_SEL_GT][i];
        if sel.is_zero() {
            result.push(Scalar::zero(curve));
        } else {
            let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_gt_raw(&cols_at_i));
        }
    }
    result
}

/// Evaluate EQ constraints on the full trace domain.
pub fn evaluate_eq(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[COL_SEL_EQ][i];
        if sel.is_zero() {
            result.push(Scalar::zero(curve));
        } else {
            let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_eq_raw(&cols_at_i));
        }
    }
    result
}

/// Evaluate ISZERO constraints on the full trace domain.
pub fn evaluate_iszero(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[COL_SEL_ISZERO][i];
        if sel.is_zero() {
            result.push(Scalar::zero(curve));
        } else {
            let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_iszero_raw(&cols_at_i));
        }
    }
    result
}

/// LT raw constraint body (ungated).
///
/// Performs 256-bit subtraction: input0 - input1 with borrow chain.
/// aux0[0..3] = borrow chain, aux1[0..3] = diff values.
/// output_l0 = aux0[3] (final borrow = 1 means input0 < input1).
///
/// Combined constraint:
///   Σ(input0_lk - input1_lk - borrow_{k-1} + borrow_k*2^64 - diff_k) + Σ borrow_k*(borrow_k-1) + (output_l0 - borrow3) + Σ upper_output_limbs = 0
pub fn evaluate_lt_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = super::arith::two_pow_64_pub(curve);
    let one = Scalar::one(curve);

    // Limb 0: input0_l0 - input1_l0 + borrow0*2^64 - diff0 = 0
    let c0 = col_evals[COL_INPUT0_L0].sub(&col_evals[COL_INPUT1_L0])
        .add(&col_evals[COL_AUX0_L0].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L0]);

    // Limb 1: input0_l1 - input1_l1 - borrow0 + borrow1*2^64 - diff1 = 0
    let c1 = col_evals[COL_INPUT0_L1].sub(&col_evals[COL_INPUT1_L1])
        .sub(&col_evals[COL_AUX0_L0])
        .add(&col_evals[COL_AUX0_L1].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L1]);

    // Limb 2: input0_l2 - input1_l2 - borrow1 + borrow2*2^64 - diff2 = 0
    let c2 = col_evals[COL_INPUT0_L2].sub(&col_evals[COL_INPUT1_L2])
        .sub(&col_evals[COL_AUX0_L1])
        .add(&col_evals[COL_AUX0_L2].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L2]);

    // Limb 3: input0_l3 - input1_l3 - borrow2 + borrow3*2^64 - diff3 = 0
    let c3 = col_evals[COL_INPUT0_L3].sub(&col_evals[COL_INPUT1_L3])
        .sub(&col_evals[COL_AUX0_L2])
        .add(&col_evals[COL_AUX0_L3].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L3]);

    // Borrow binary constraints
    let bb0 = col_evals[COL_AUX0_L0].mul(&col_evals[COL_AUX0_L0].sub(&one));
    let bb1 = col_evals[COL_AUX0_L1].mul(&col_evals[COL_AUX0_L1].sub(&one));
    let bb2 = col_evals[COL_AUX0_L2].mul(&col_evals[COL_AUX0_L2].sub(&one));
    let bb3 = col_evals[COL_AUX0_L3].mul(&col_evals[COL_AUX0_L3].sub(&one));

    // output_l0 = borrow3 (the LT result)
    let output_check = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_AUX0_L3]);

    // Upper output limbs must be zero
    let upper = col_evals[COL_OUTPUT0_L1].add(&col_evals[COL_OUTPUT0_L2])
        .add(&col_evals[COL_OUTPUT0_L3]);

    c0.add(&c1).add(&c2).add(&c3)
        .add(&bb0).add(&bb1).add(&bb2).add(&bb3)
        .add(&output_check).add(&upper)
}

/// GT raw constraint body (ungated).
///
/// Same as LT but subtracts input1 - input0 instead.
/// aux0[0..3] = borrow chain, aux1[0..3] = diff values.
/// output_l0 = aux0[3] (final borrow = 1 means input1 < input0, i.e., input0 > input1).
pub fn evaluate_gt_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = super::arith::two_pow_64_pub(curve);
    let one = Scalar::one(curve);

    // Limb 0: input1_l0 - input0_l0 + borrow0*2^64 - diff0 = 0
    let c0 = col_evals[COL_INPUT1_L0].sub(&col_evals[COL_INPUT0_L0])
        .add(&col_evals[COL_AUX0_L0].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L0]);

    // Limb 1: input1_l1 - input0_l1 - borrow0 + borrow1*2^64 - diff1 = 0
    let c1 = col_evals[COL_INPUT1_L1].sub(&col_evals[COL_INPUT0_L1])
        .sub(&col_evals[COL_AUX0_L0])
        .add(&col_evals[COL_AUX0_L1].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L1]);

    // Limb 2: input1_l2 - input0_l2 - borrow1 + borrow2*2^64 - diff2 = 0
    let c2 = col_evals[COL_INPUT1_L2].sub(&col_evals[COL_INPUT0_L2])
        .sub(&col_evals[COL_AUX0_L1])
        .add(&col_evals[COL_AUX0_L2].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L2]);

    // Limb 3: input1_l3 - input0_l3 - borrow2 + borrow3*2^64 - diff3 = 0
    let c3 = col_evals[COL_INPUT1_L3].sub(&col_evals[COL_INPUT0_L3])
        .sub(&col_evals[COL_AUX0_L2])
        .add(&col_evals[COL_AUX0_L3].mul(&two_64))
        .sub(&col_evals[COL_AUX1_L3]);

    // Borrow binary constraints
    let bb0 = col_evals[COL_AUX0_L0].mul(&col_evals[COL_AUX0_L0].sub(&one));
    let bb1 = col_evals[COL_AUX0_L1].mul(&col_evals[COL_AUX0_L1].sub(&one));
    let bb2 = col_evals[COL_AUX0_L2].mul(&col_evals[COL_AUX0_L2].sub(&one));
    let bb3 = col_evals[COL_AUX0_L3].mul(&col_evals[COL_AUX0_L3].sub(&one));

    // output_l0 = borrow3 (the GT result)
    let output_check = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_AUX0_L3]);

    // Upper output limbs must be zero
    let upper = col_evals[COL_OUTPUT0_L1].add(&col_evals[COL_OUTPUT0_L2])
        .add(&col_evals[COL_OUTPUT0_L3]);

    c0.add(&c1).add(&c2).add(&c3)
        .add(&bb0).add(&bb1).add(&bb2).add(&bb3)
        .add(&output_check).add(&upper)
}

/// EQ raw constraint body (ungated).
///
/// output = 1 iff input0 == input1 across all 4 limbs.
/// aux1[0..3] = input0_lk - input1_lk (diff limbs).
///
/// Constraints:
///   1. output * (output - 1) = 0  (binary output)
///   2. output * diff_k = 0 for each k (if output=1, all diffs must be zero)
///   3. Upper output limbs must be zero
pub fn evaluate_eq_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);

    let out = &col_evals[COL_OUTPUT0_L0];

    // Binary output
    let binary = out.mul(&out.sub(&one));

    // output * diff_k = 0: if output=1, each diff must be zero
    let diff0 = col_evals[COL_INPUT0_L0].sub(&col_evals[COL_INPUT1_L0]);
    let diff1 = col_evals[COL_INPUT0_L1].sub(&col_evals[COL_INPUT1_L1]);
    let diff2 = col_evals[COL_INPUT0_L2].sub(&col_evals[COL_INPUT1_L2]);
    let diff3 = col_evals[COL_INPUT0_L3].sub(&col_evals[COL_INPUT1_L3]);

    let eq0 = out.mul(&diff0);
    let eq1 = out.mul(&diff1);
    let eq2 = out.mul(&diff2);
    let eq3 = out.mul(&diff3);

    // Upper output limbs must be zero
    let upper = col_evals[COL_OUTPUT0_L1].add(&col_evals[COL_OUTPUT0_L2])
        .add(&col_evals[COL_OUTPUT0_L3]);

    binary.add(&eq0).add(&eq1).add(&eq2).add(&eq3).add(&upper)
}

/// ISZERO raw constraint body (ungated).
///
/// output = 1 iff input0 == 0 across all 4 limbs.
///
/// Constraints:
///   1. output * (output - 1) = 0  (binary output)
///   2. output * input0_lk = 0 for each k (if output=1, all input limbs must be zero)
///   3. Upper output limbs must be zero
pub fn evaluate_iszero_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);

    let out = &col_evals[COL_OUTPUT0_L0];

    // Binary output
    let binary = out.mul(&out.sub(&one));

    // output * input_lk = 0
    let iz0 = out.mul(&col_evals[COL_INPUT0_L0]);
    let iz1 = out.mul(&col_evals[COL_INPUT0_L1]);
    let iz2 = out.mul(&col_evals[COL_INPUT0_L2]);
    let iz3 = out.mul(&col_evals[COL_INPUT0_L3]);

    // Upper output limbs must be zero
    let upper = col_evals[COL_OUTPUT0_L1].add(&col_evals[COL_OUTPUT0_L2])
        .add(&col_evals[COL_OUTPUT0_L3]);

    binary.add(&iz0).add(&iz1).add(&iz2).add(&iz3).add(&upper)
}

/// Evaluate the old combined compare constraint (binary output + upper limbs zero).
/// Kept for backward compatibility — gated by sel_compare_other for SLT/SGT.
pub fn evaluate_compare_other_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);

    let out_l0 = &col_evals[COL_OUTPUT0_L0];
    let binary_check = out_l0.mul(&out_l0.sub(&one));

    let l1_zero = col_evals[COL_OUTPUT0_L1].clone();
    let l2_zero = col_evals[COL_OUTPUT0_L2].clone();
    let l3_zero = col_evals[COL_OUTPUT0_L3].clone();

    binary_check.add(&l1_zero).add(&l2_zero).add(&l3_zero)
}
