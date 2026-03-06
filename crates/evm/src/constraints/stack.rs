//! Stack constraints: PUSH (output = immediate), DUP (output = input0), POP/SWAP.

use metavm_zkp::field::Scalar;
use crate::trace::*;

/// Evaluate PUSH constraint on the full trace domain (gated by sel_push).
pub fn evaluate_push(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let sel = &columns[COL_SEL_PUSH][i];
        let constraint = if !sel.is_zero() {
            // PUSH: output = immediate (all 4 limbs)
            let c0 = columns[COL_OUTPUT0_L0][i].sub(&columns[COL_IMMEDIATE_L0][i]);
            let c1 = columns[COL_OUTPUT0_L1][i].sub(&columns[COL_IMMEDIATE_L1][i]);
            let c2 = columns[COL_OUTPUT0_L2][i].sub(&columns[COL_IMMEDIATE_L2][i]);
            let c3 = columns[COL_OUTPUT0_L3][i].sub(&columns[COL_IMMEDIATE_L3][i]);
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };
        result.push(constraint);
    }

    result
}

/// Evaluate DUP constraint on the full trace domain (gated by sel_dup).
pub fn evaluate_dup(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let sel = &columns[COL_SEL_DUP][i];
        let constraint = if !sel.is_zero() {
            // DUP: output = input0 (duplicated value, all 4 limbs)
            let c0 = columns[COL_OUTPUT0_L0][i].sub(&columns[COL_INPUT0_L0][i]);
            let c1 = columns[COL_OUTPUT0_L1][i].sub(&columns[COL_INPUT0_L1][i]);
            let c2 = columns[COL_OUTPUT0_L2][i].sub(&columns[COL_INPUT0_L2][i]);
            let c3 = columns[COL_OUTPUT0_L3][i].sub(&columns[COL_INPUT0_L3][i]);
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };
        result.push(constraint);
    }

    result
}

/// Evaluate PUSH constraint raw, without instruction type gating.
///
/// Computes output = immediate for all 4 limbs. The caller gates this by
/// multiplying with `sel_push`.
pub fn evaluate_push_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_IMMEDIATE_L0]);
    let c1 = col_evals[COL_OUTPUT0_L1].sub(&col_evals[COL_IMMEDIATE_L1]);
    let c2 = col_evals[COL_OUTPUT0_L2].sub(&col_evals[COL_IMMEDIATE_L2]);
    let c3 = col_evals[COL_OUTPUT0_L3].sub(&col_evals[COL_IMMEDIATE_L3]);
    c0.add(&c1).add(&c2).add(&c3)
}

/// Evaluate DUP constraint raw, without instruction type gating.
///
/// Computes output = input0 for all 4 limbs. The caller gates this by
/// multiplying with `sel_dup`.
pub fn evaluate_dup_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_INPUT0_L0]);
    let c1 = col_evals[COL_OUTPUT0_L1].sub(&col_evals[COL_INPUT0_L1]);
    let c2 = col_evals[COL_OUTPUT0_L2].sub(&col_evals[COL_INPUT0_L2]);
    let c3 = col_evals[COL_OUTPUT0_L3].sub(&col_evals[COL_INPUT0_L3]);
    c0.add(&c1).add(&c2).add(&c3)
}

/// Evaluate POP constraint on the full trace domain (gated by sel_pop).
pub fn evaluate_pop(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let sel = &columns[COL_SEL_POP][i];
        let constraint = if !sel.is_zero() {
            // POP: all output limbs must be zero (POP produces no stack output)
            let c0 = columns[COL_OUTPUT0_L0][i].clone();
            let c1 = columns[COL_OUTPUT0_L1][i].clone();
            let c2 = columns[COL_OUTPUT0_L2][i].clone();
            let c3 = columns[COL_OUTPUT0_L3][i].clone();
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };
        result.push(constraint);
    }

    result
}

/// Evaluate SWAP constraint on the full trace domain (gated by sel_swap).
pub fn evaluate_swap(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let sel = &columns[COL_SEL_SWAP][i];
        let constraint = if !sel.is_zero() {
            // SWAP: output = input1 (swapped value, all 4 limbs)
            let c0 = columns[COL_OUTPUT0_L0][i].sub(&columns[COL_INPUT1_L0][i]);
            let c1 = columns[COL_OUTPUT0_L1][i].sub(&columns[COL_INPUT1_L1][i]);
            let c2 = columns[COL_OUTPUT0_L2][i].sub(&columns[COL_INPUT1_L2][i]);
            let c3 = columns[COL_OUTPUT0_L3][i].sub(&columns[COL_INPUT1_L3][i]);
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };
        result.push(constraint);
    }

    result
}

/// Evaluate POP constraint raw, without instruction type gating.
///
/// Computes output limbs all zero. The caller gates this by
/// multiplying with `sel_pop`.
pub fn evaluate_pop_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].clone();
    let c1 = col_evals[COL_OUTPUT0_L1].clone();
    let c2 = col_evals[COL_OUTPUT0_L2].clone();
    let c3 = col_evals[COL_OUTPUT0_L3].clone();
    c0.add(&c1).add(&c2).add(&c3)
}

/// Evaluate SWAP constraint raw, without instruction type gating.
///
/// Computes output = input1 for all 4 limbs. The caller gates this by
/// multiplying with `sel_swap`.
pub fn evaluate_swap_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_INPUT1_L0]);
    let c1 = col_evals[COL_OUTPUT0_L1].sub(&col_evals[COL_INPUT1_L1]);
    let c2 = col_evals[COL_OUTPUT0_L2].sub(&col_evals[COL_INPUT1_L2]);
    let c3 = col_evals[COL_OUTPUT0_L3].sub(&col_evals[COL_INPUT1_L3]);
    c0.add(&c1).add(&c2).add(&c3)
}
