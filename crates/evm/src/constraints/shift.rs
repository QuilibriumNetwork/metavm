//! Shift constraints for EVM SHL, SHR, SAR with 256-bit limb decomposition.
//!
//! EVM shift semantics: shift_amount = input0 (top of stack), value = input1.
//!
//! SHL (left shift): output = input1 << input0 = input1 * 2^k (mod 2^256)
//!   Uses schoolbook 4-limb multiplication with input1 and immediate (2^k).
//!   Carry chain stored in aux0[0..3]. immediate[0..3] = 2^k.
//!
//! SHR (logical right shift): output = input1 >> input0 = floor(input1 / 2^k)
//!   Constraint: output * 2^k + remainder = input1 (mod 2^256)
//!   remainder in aux0[0..3], carry chain in aux1[0..3].
//!
//! SAR (arithmetic right shift): same algebraic constraint as SHR.
//!   output * 2^k + remainder = input1 (mod 2^256)
//!   Full soundness requires range checks on the output to verify sign extension.

use metavm_zkp::field::Scalar;
use crate::trace::*;
use super::arith::two_pow_64_pub;

/// Evaluate SHL constraint on the full domain (gated by sel_shl).
pub fn evaluate_shl(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[COL_SEL_SHL][i];
        if sel.is_zero() {
            result.push(zero.clone());
        } else {
            let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_shl_raw(&cols_at_i));
        }
    }
    result
}

/// SHL raw constraint body (combined 4-limb multiplication check).
///
/// Checks: input1 * immediate = output (mod 2^256) with carry chain in aux0.
///
/// Limb k: Sigma_{i+j=k} input1_i * imm_j + carry_{k-1} = carry_k * 2^64 + output_k
pub fn evaluate_shl_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64_pub(curve);

    // a = input1 (value), b = immediate (2^k), out = output, c = aux0 (carries)
    let a = [&col_evals[COL_INPUT1_L0], &col_evals[COL_INPUT1_L1],
             &col_evals[COL_INPUT1_L2], &col_evals[COL_INPUT1_L3]];
    let b = [&col_evals[COL_IMMEDIATE_L0], &col_evals[COL_IMMEDIATE_L1],
             &col_evals[COL_IMMEDIATE_L2], &col_evals[COL_IMMEDIATE_L3]];
    let out = [&col_evals[COL_OUTPUT0_L0], &col_evals[COL_OUTPUT0_L1],
               &col_evals[COL_OUTPUT0_L2], &col_evals[COL_OUTPUT0_L3]];
    let c = [&col_evals[COL_AUX0_L0], &col_evals[COL_AUX0_L1],
             &col_evals[COL_AUX0_L2], &col_evals[COL_AUX0_L3]];

    // Limb 0: a0*b0 - c0*2^64 - out0 = 0
    let l0 = a[0].mul(b[0])
        .sub(&c[0].mul(&two_64))
        .sub(out[0]);

    // Limb 1: a0*b1 + a1*b0 + c0 - c1*2^64 - out1 = 0
    let l1 = a[0].mul(b[1])
        .add(&a[1].mul(b[0]))
        .add(c[0])
        .sub(&c[1].mul(&two_64))
        .sub(out[1]);

    // Limb 2: a0*b2 + a1*b1 + a2*b0 + c1 - c2*2^64 - out2 = 0
    let l2 = a[0].mul(b[2])
        .add(&a[1].mul(b[1]))
        .add(&a[2].mul(b[0]))
        .add(c[1])
        .sub(&c[2].mul(&two_64))
        .sub(out[2]);

    // Limb 3: a0*b3 + a1*b2 + a2*b1 + a3*b0 + c2 - c3*2^64 - out3 = 0
    let l3 = a[0].mul(b[3])
        .add(&a[1].mul(b[2]))
        .add(&a[2].mul(b[1]))
        .add(&a[3].mul(b[0]))
        .add(c[2])
        .sub(&c[3].mul(&two_64))
        .sub(out[3]);

    l0.add(&l1).add(&l2).add(&l3)
}

/// Evaluate SHR constraint on the full domain (gated by sel_shr).
pub fn evaluate_shr(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[COL_SEL_SHR][i];
        if sel.is_zero() {
            result.push(zero.clone());
        } else {
            let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_shr_raw(&cols_at_i));
        }
    }
    result
}

/// SHR raw constraint body.
///
/// Checks: output * immediate + aux0 = input1 (mod 2^256), carry in aux1.
pub fn evaluate_shr_raw(col_evals: &[Scalar]) -> Scalar {
    evaluate_shift_div_raw(col_evals)
}

/// Evaluate SAR constraint on the full domain (gated by sel_sar).
pub fn evaluate_sar(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[COL_SEL_SAR][i];
        if sel.is_zero() {
            result.push(zero.clone());
        } else {
            let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_sar_raw(&cols_at_i));
        }
    }
    result
}

/// SAR raw constraint body: same as SHR algebraically.
pub fn evaluate_sar_raw(col_evals: &[Scalar]) -> Scalar {
    evaluate_shift_div_raw(col_evals)
}

/// Shared division-style constraint for SHR and SAR.
///
/// output * immediate + aux0 = input1 (mod 2^256), carry in aux1.
///
/// Limb k: Sigma_{i+j=k} out_i * imm_j + rem_k + carry_{k-1} = carry_k * 2^64 + input1_k
fn evaluate_shift_div_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64_pub(curve);

    // q = output (quotient), d = immediate (2^k), r = aux0 (remainder), val = input1
    let q = [&col_evals[COL_OUTPUT0_L0], &col_evals[COL_OUTPUT0_L1],
             &col_evals[COL_OUTPUT0_L2], &col_evals[COL_OUTPUT0_L3]];
    let d = [&col_evals[COL_IMMEDIATE_L0], &col_evals[COL_IMMEDIATE_L1],
             &col_evals[COL_IMMEDIATE_L2], &col_evals[COL_IMMEDIATE_L3]];
    let r = [&col_evals[COL_AUX0_L0], &col_evals[COL_AUX0_L1],
             &col_evals[COL_AUX0_L2], &col_evals[COL_AUX0_L3]];
    let val = [&col_evals[COL_INPUT1_L0], &col_evals[COL_INPUT1_L1],
               &col_evals[COL_INPUT1_L2], &col_evals[COL_INPUT1_L3]];
    let c = [&col_evals[COL_AUX1_L0], &col_evals[COL_AUX1_L1],
             &col_evals[COL_AUX1_L2], &col_evals[COL_AUX1_L3]];

    // Limb 0: q0*d0 + r0 - c0*2^64 - val0 = 0
    let l0 = q[0].mul(d[0])
        .add(r[0])
        .sub(&c[0].mul(&two_64))
        .sub(val[0]);

    // Limb 1: q0*d1 + q1*d0 + r1 + c0 - c1*2^64 - val1 = 0
    let l1 = q[0].mul(d[1])
        .add(&q[1].mul(d[0]))
        .add(r[1])
        .add(c[0])
        .sub(&c[1].mul(&two_64))
        .sub(val[1]);

    // Limb 2: q0*d2 + q1*d1 + q2*d0 + r2 + c1 - c2*2^64 - val2 = 0
    let l2 = q[0].mul(d[2])
        .add(&q[1].mul(d[1]))
        .add(&q[2].mul(d[0]))
        .add(r[2])
        .add(c[1])
        .sub(&c[2].mul(&two_64))
        .sub(val[2]);

    // Limb 3: q0*d3 + q1*d2 + q2*d1 + q3*d0 + r3 + c2 - c3*2^64 - val3 = 0
    let l3 = q[0].mul(d[3])
        .add(&q[1].mul(d[2]))
        .add(&q[2].mul(d[1]))
        .add(&q[3].mul(d[0]))
        .add(r[3])
        .add(c[2])
        .sub(&c[3].mul(&two_64))
        .sub(val[3]);

    l0.add(&l1).add(&l2).add(&l3)
}
