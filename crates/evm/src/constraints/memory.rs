//! Memory constraints: MLOAD (output = mem_value), MSTORE (mem_value = input1).

use metavm_zkp::field::Scalar;
use crate::trace::*;

/// Evaluate MLOAD constraint on the full trace domain (gated by sel_mload).
pub fn evaluate_mload(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let sel = &columns[COL_SEL_MLOAD][i];
        let constraint = if !sel.is_zero() {
            // MLOAD: output = mem_value
            let c0 = columns[COL_OUTPUT0_L0][i].sub(&columns[COL_MEM_VALUE_L0][i]);
            let c1 = columns[COL_OUTPUT0_L1][i].sub(&columns[COL_MEM_VALUE_L1][i]);
            let c2 = columns[COL_OUTPUT0_L2][i].sub(&columns[COL_MEM_VALUE_L2][i]);
            let c3 = columns[COL_OUTPUT0_L3][i].sub(&columns[COL_MEM_VALUE_L3][i]);
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };
        result.push(constraint);
    }

    result
}

/// Evaluate MSTORE constraint on the full trace domain (gated by sel_mstore).
pub fn evaluate_mstore(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let sel = &columns[COL_SEL_MSTORE][i];
        let constraint = if !sel.is_zero() {
            // MSTORE: mem_value = input1 (value being stored, input0 = offset)
            let c0 = columns[COL_MEM_VALUE_L0][i].sub(&columns[COL_INPUT1_L0][i]);
            let c1 = columns[COL_MEM_VALUE_L1][i].sub(&columns[COL_INPUT1_L1][i]);
            let c2 = columns[COL_MEM_VALUE_L2][i].sub(&columns[COL_INPUT1_L2][i]);
            let c3 = columns[COL_MEM_VALUE_L3][i].sub(&columns[COL_INPUT1_L3][i]);
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };
        result.push(constraint);
    }

    result
}

/// Evaluate MLOAD constraint raw, without instruction type gating.
///
/// Computes output = mem_value for all 4 limbs. The caller gates this by
/// multiplying with `sel_mload`.
pub fn evaluate_mload_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_MEM_VALUE_L0]);
    let c1 = col_evals[COL_OUTPUT0_L1].sub(&col_evals[COL_MEM_VALUE_L1]);
    let c2 = col_evals[COL_OUTPUT0_L2].sub(&col_evals[COL_MEM_VALUE_L2]);
    let c3 = col_evals[COL_OUTPUT0_L3].sub(&col_evals[COL_MEM_VALUE_L3]);
    c0.add(&c1).add(&c2).add(&c3)
}

/// Evaluate MSTORE constraint raw, without instruction type gating.
///
/// Computes mem_value = input1 for all 4 limbs. The caller gates this by
/// multiplying with `sel_mstore`.
pub fn evaluate_mstore_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_MEM_VALUE_L0].sub(&col_evals[COL_INPUT1_L0]);
    let c1 = col_evals[COL_MEM_VALUE_L1].sub(&col_evals[COL_INPUT1_L1]);
    let c2 = col_evals[COL_MEM_VALUE_L2].sub(&col_evals[COL_INPUT1_L2]);
    let c3 = col_evals[COL_MEM_VALUE_L3].sub(&col_evals[COL_INPUT1_L3]);
    c0.add(&c1).add(&c2).add(&c3)
}

/// Evaluate MSTORE8 constraint on the full trace domain (gated by sel_mstore8).
pub fn evaluate_mstore8(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let two_56 = Scalar::from_u64(256, curve);
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let sel = &columns[COL_SEL_MSTORE8][i];
        let constraint = if !sel.is_zero() {
            // MSTORE8: byte extraction
            // aux0_l0 * 256 + mem_val_l0 - input1_l0 = 0
            let c0 = columns[COL_AUX0_L0][i].mul(&two_56)
                .add(&columns[COL_MEM_VALUE_L0][i])
                .sub(&columns[COL_INPUT1_L0][i]);
            // Upper mem_value limbs must be zero
            let c1 = columns[COL_MEM_VALUE_L1][i].clone();
            let c2 = columns[COL_MEM_VALUE_L2][i].clone();
            let c3 = columns[COL_MEM_VALUE_L3][i].clone();
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };
        result.push(constraint);
    }

    result
}

/// Evaluate MSIZE constraint on the full trace domain (gated by sel_msize).
///
/// MSIZE is an oracle operation: the constraint body is zero.
/// Correctness is verified externally.
pub fn evaluate_msize(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    vec![Scalar::zero(curve); num_rows]
}

/// Evaluate MSTORE8 constraint raw, without instruction type gating.
///
/// Computes aux0_l0 * 256 + mem_val_l0 - input1_l0 = 0 and upper mem_value limbs zero.
/// The caller gates this by multiplying with `sel_mstore8`.
pub fn evaluate_mstore8_raw(col_evals: &[Scalar]) -> Scalar {
    let two_56 = Scalar::from_u64(256, col_evals[0].curve_type());
    let c0 = col_evals[COL_AUX0_L0].mul(&two_56)
        .add(&col_evals[COL_MEM_VALUE_L0])
        .sub(&col_evals[COL_INPUT1_L0]);
    let c1 = col_evals[COL_MEM_VALUE_L1].clone();
    let c2 = col_evals[COL_MEM_VALUE_L2].clone();
    let c3 = col_evals[COL_MEM_VALUE_L3].clone();
    c0.add(&c1).add(&c2).add(&c3)
}

/// Evaluate MSIZE constraint raw, without instruction type gating.
///
/// MSIZE is an oracle operation: returns zero. The caller gates this by
/// multiplying with `sel_msize`.
pub fn evaluate_msize_raw(_col_evals: &[Scalar]) -> Scalar {
    Scalar::zero(_col_evals[0].curve_type())
}
