//! Arithmetic constraints for EVM ADD, SUB, MUL, DIV with 256-bit limb decomposition.
//!
//! 256-bit ADD with 4x64-bit limbs (a + b = c with carries):
//!   a0 + b0 = carry0 * 2^64 + c0       (carry0 binary)
//!   a1 + b1 + carry0 = carry1 * 2^64 + c1
//!   a2 + b2 + carry1 = carry2 * 2^64 + c2
//!   a3 + b3 + carry2 = carry3 * 2^64 + c3   (carry3 stored in aux0[0])
//!
//! Aux columns store carries: aux0_limb0..3 = carry0..carry3.

use metavm_zkp::field::{Scalar, CurveType};
use crate::trace::*;

/// Compute the field element 2^64 for the given curve.
pub fn two_pow_64_pub(curve: CurveType) -> Scalar {
    two_pow_64(curve)
}

fn two_pow_64(curve: CurveType) -> Scalar {
    use bls48581::bls48581::big;
    match curve {
        CurveType::Bls48581 => {
            let mut buf = [0u8; big::MODBYTES];
            buf[big::MODBYTES - 9] = 1;
            Scalar::Bls48581(big::BIG::frombytes(&buf))
        }
        CurveType::Bls12381 => {
            let mut scalar = blst::blst_scalar::default();
            scalar.b[8] = 1;
            let mut fr = blst::blst_fr::default();
            unsafe { blst::blst_fr_from_scalar(&mut fr, &scalar); }
            Scalar::Bls12381(fr)
        }
    }
}

/// Evaluate arithmetic constraints on the full trace domain.
pub fn evaluate_arith(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type = columns[COL_INSN_TYPE][i].to_u64() as u8;
        let funct = columns[COL_FUNCT][i].to_u64() as u8;

        let constraint = if insn_type == INSN_ARITH {
            match funct {
                FUNCT_ADD => {
                    // Limb-by-limb ADD with carry chain.
                    // Carry values stored in aux0 limbs.
                    // For simplicity, check: input0 + input1 = carry * 2^256 + output
                    // where carry = aux0[0] (single bit for 256-bit overflow).
                    //
                    // Detailed limb constraint:
                    // input0_l0 + input1_l0 = aux0_l0 * 2^64 + output0_l0
                    let sum_l0 = columns[COL_INPUT0_L0][i].add(&columns[COL_INPUT1_L0][i]);
                    let carry0_term = columns[COL_AUX0_L0][i].mul(&two_64);
                    let rhs_l0 = carry0_term.add(&columns[COL_OUTPUT0_L0][i]);
                    let c_l0 = sum_l0.sub(&rhs_l0);

                    // input0_l1 + input1_l1 + carry0 = aux0_l1 * 2^64 + output0_l1
                    let sum_l1 = columns[COL_INPUT0_L1][i].add(&columns[COL_INPUT1_L1][i]).add(&columns[COL_AUX0_L0][i]);
                    let carry1_term = columns[COL_AUX0_L1][i].mul(&two_64);
                    let rhs_l1 = carry1_term.add(&columns[COL_OUTPUT0_L1][i]);
                    let c_l1 = sum_l1.sub(&rhs_l1);

                    // input0_l2 + input1_l2 + carry1 = aux0_l2 * 2^64 + output0_l2
                    let sum_l2 = columns[COL_INPUT0_L2][i].add(&columns[COL_INPUT1_L2][i]).add(&columns[COL_AUX0_L1][i]);
                    let carry2_term = columns[COL_AUX0_L2][i].mul(&two_64);
                    let rhs_l2 = carry2_term.add(&columns[COL_OUTPUT0_L2][i]);
                    let c_l2 = sum_l2.sub(&rhs_l2);

                    // input0_l3 + input1_l3 + carry2 = aux0_l3 * 2^64 + output0_l3
                    let sum_l3 = columns[COL_INPUT0_L3][i].add(&columns[COL_INPUT1_L3][i]).add(&columns[COL_AUX0_L2][i]);
                    let carry3_term = columns[COL_AUX0_L3][i].mul(&two_64);
                    let rhs_l3 = carry3_term.add(&columns[COL_OUTPUT0_L3][i]);
                    let c_l3 = sum_l3.sub(&rhs_l3);

                    // Each carry must be binary
                    let cb0 = columns[COL_AUX0_L0][i].mul(&columns[COL_AUX0_L0][i].sub(&one));
                    let cb1 = columns[COL_AUX0_L1][i].mul(&columns[COL_AUX0_L1][i].sub(&one));
                    let cb2 = columns[COL_AUX0_L2][i].mul(&columns[COL_AUX0_L2][i].sub(&one));
                    let cb3 = columns[COL_AUX0_L3][i].mul(&columns[COL_AUX0_L3][i].sub(&one));

                    c_l0.add(&c_l1).add(&c_l2).add(&c_l3)
                        .add(&cb0).add(&cb1).add(&cb2).add(&cb3)
                }
                FUNCT_SUB => {
                    // SUB: output + input1 = input0 (mod 2^256)
                    // Per-limb with borrow chain in aux0:
                    // output_l0 + input1_l0 = borrow0 * 2^64 + input0_l0
                    // output_l1 + input1_l1 + borrow0 = borrow1 * 2^64 + input0_l1
                    // output_l2 + input1_l2 + borrow1 = borrow2 * 2^64 + input0_l2
                    // output_l3 + input1_l3 + borrow2 = borrow3 * 2^64 + input0_l3
                    let sum_out_l0 = columns[COL_OUTPUT0_L0][i].add(&columns[COL_INPUT1_L0][i]);
                    let borrow_term = columns[COL_AUX0_L0][i].mul(&two_64);
                    let c0 = sum_out_l0.sub(&columns[COL_INPUT0_L0][i]).sub(&borrow_term);

                    let sum_out_l1 = columns[COL_OUTPUT0_L1][i].add(&columns[COL_INPUT1_L1][i]).add(&columns[COL_AUX0_L0][i]);
                    let borrow1_term = columns[COL_AUX0_L1][i].mul(&two_64);
                    let c1 = sum_out_l1.sub(&columns[COL_INPUT0_L1][i]).sub(&borrow1_term);

                    let sum_out_l2 = columns[COL_OUTPUT0_L2][i].add(&columns[COL_INPUT1_L2][i]).add(&columns[COL_AUX0_L1][i]);
                    let borrow2_term = columns[COL_AUX0_L2][i].mul(&two_64);
                    let c2 = sum_out_l2.sub(&columns[COL_INPUT0_L2][i]).sub(&borrow2_term);

                    let sum_out_l3 = columns[COL_OUTPUT0_L3][i].add(&columns[COL_INPUT1_L3][i]).add(&columns[COL_AUX0_L2][i]);
                    let borrow3_term = columns[COL_AUX0_L3][i].mul(&two_64);
                    let c3 = sum_out_l3.sub(&columns[COL_INPUT0_L3][i]).sub(&borrow3_term);

                    // borrow binary constraints
                    let bb0 = columns[COL_AUX0_L0][i].mul(&columns[COL_AUX0_L0][i].sub(&one));
                    let bb1 = columns[COL_AUX0_L1][i].mul(&columns[COL_AUX0_L1][i].sub(&one));
                    let bb2 = columns[COL_AUX0_L2][i].mul(&columns[COL_AUX0_L2][i].sub(&one));
                    let bb3 = columns[COL_AUX0_L3][i].mul(&columns[COL_AUX0_L3][i].sub(&one));

                    c0.add(&c1).add(&c2).add(&c3)
                        .add(&bb0).add(&bb1).add(&bb2).add(&bb3)
                }
                FUNCT_MUL => {
                    // Full 256-bit MUL: 4-limb schoolbook multiplication with carry chain
                    let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                        .map(|c| columns[c][i].clone())
                        .collect();
                    let m0 = evaluate_arith_mul_limb0_raw(&cols_at_i);
                    let m1 = evaluate_arith_mul_limb1_raw(&cols_at_i);
                    let m2 = evaluate_arith_mul_limb2_raw(&cols_at_i);
                    let m3 = evaluate_arith_mul_limb3_raw(&cols_at_i);
                    m0.add(&m1).add(&m2).add(&m3)
                }
                FUNCT_DIV => {
                    // Full 256-bit DIV: quotient*divisor+remainder=dividend, 4-limb carry chain
                    let cols_at_i: Vec<Scalar> = (0..NUM_EVM_COLUMNS)
                        .map(|c| columns[c][i].clone())
                        .collect();
                    let d0 = evaluate_arith_div_limb0_raw(&cols_at_i);
                    let d1 = evaluate_arith_div_limb1_raw(&cols_at_i);
                    let d2 = evaluate_arith_div_limb2_raw(&cols_at_i);
                    let d3 = evaluate_arith_div_limb3_raw(&cols_at_i);
                    d0.add(&d1).add(&d2).add(&d3)
                }
                _ => Scalar::zero(curve),
            }
        } else {
            Scalar::zero(curve)
        };

        result.push(constraint);
    }

    result
}

/// Evaluate arithmetic constraint at a single point.
pub fn evaluate_arith_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < NUM_EVM_COLUMNS {
        return Scalar::zero(curve);
    }

    let insn_type = col_evals[COL_INSN_TYPE].to_u64() as u8;
    let funct = col_evals[COL_FUNCT].to_u64() as u8;
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);

    if insn_type != INSN_ARITH {
        return Scalar::zero(curve);
    }

    match funct {
        FUNCT_ADD => {
            // Same limb-by-limb constraint as domain evaluation
            let sum_l0 = col_evals[COL_INPUT0_L0].add(&col_evals[COL_INPUT1_L0]);
            let carry0_term = col_evals[COL_AUX0_L0].mul(&two_64);
            let c_l0 = sum_l0.sub(&carry0_term).sub(&col_evals[COL_OUTPUT0_L0]);

            let sum_l1 = col_evals[COL_INPUT0_L1].add(&col_evals[COL_INPUT1_L1]).add(&col_evals[COL_AUX0_L0]);
            let carry1_term = col_evals[COL_AUX0_L1].mul(&two_64);
            let c_l1 = sum_l1.sub(&carry1_term).sub(&col_evals[COL_OUTPUT0_L1]);

            let sum_l2 = col_evals[COL_INPUT0_L2].add(&col_evals[COL_INPUT1_L2]).add(&col_evals[COL_AUX0_L1]);
            let carry2_term = col_evals[COL_AUX0_L2].mul(&two_64);
            let c_l2 = sum_l2.sub(&carry2_term).sub(&col_evals[COL_OUTPUT0_L2]);

            let sum_l3 = col_evals[COL_INPUT0_L3].add(&col_evals[COL_INPUT1_L3]).add(&col_evals[COL_AUX0_L2]);
            let carry3_term = col_evals[COL_AUX0_L3].mul(&two_64);
            let c_l3 = sum_l3.sub(&carry3_term).sub(&col_evals[COL_OUTPUT0_L3]);

            let cb0 = col_evals[COL_AUX0_L0].mul(&col_evals[COL_AUX0_L0].sub(&one));
            let cb1 = col_evals[COL_AUX0_L1].mul(&col_evals[COL_AUX0_L1].sub(&one));
            let cb2 = col_evals[COL_AUX0_L2].mul(&col_evals[COL_AUX0_L2].sub(&one));
            let cb3 = col_evals[COL_AUX0_L3].mul(&col_evals[COL_AUX0_L3].sub(&one));

            c_l0.add(&c_l1).add(&c_l2).add(&c_l3)
                .add(&cb0).add(&cb1).add(&cb2).add(&cb3)
        }
        FUNCT_SUB => {
            let sum_l0 = col_evals[COL_OUTPUT0_L0].add(&col_evals[COL_INPUT1_L0]);
            let borrow_term = col_evals[COL_AUX0_L0].mul(&two_64);
            let c0 = sum_l0.sub(&col_evals[COL_INPUT0_L0]).sub(&borrow_term);

            let sum_l1 = col_evals[COL_OUTPUT0_L1].add(&col_evals[COL_INPUT1_L1]).add(&col_evals[COL_AUX0_L0]);
            let b1_term = col_evals[COL_AUX0_L1].mul(&two_64);
            let c1 = sum_l1.sub(&col_evals[COL_INPUT0_L1]).sub(&b1_term);

            let sum_l2 = col_evals[COL_OUTPUT0_L2].add(&col_evals[COL_INPUT1_L2]).add(&col_evals[COL_AUX0_L1]);
            let b2_term = col_evals[COL_AUX0_L2].mul(&two_64);
            let c2 = sum_l2.sub(&col_evals[COL_INPUT0_L2]).sub(&b2_term);

            let sum_l3 = col_evals[COL_OUTPUT0_L3].add(&col_evals[COL_INPUT1_L3]).add(&col_evals[COL_AUX0_L2]);
            let b3_term = col_evals[COL_AUX0_L3].mul(&two_64);
            let c3 = sum_l3.sub(&col_evals[COL_INPUT0_L3]).sub(&b3_term);

            let bb0 = col_evals[COL_AUX0_L0].mul(&col_evals[COL_AUX0_L0].sub(&one));
            let bb1 = col_evals[COL_AUX0_L1].mul(&col_evals[COL_AUX0_L1].sub(&one));
            let bb2 = col_evals[COL_AUX0_L2].mul(&col_evals[COL_AUX0_L2].sub(&one));
            let bb3 = col_evals[COL_AUX0_L3].mul(&col_evals[COL_AUX0_L3].sub(&one));

            c0.add(&c1).add(&c2).add(&c3)
                .add(&bb0).add(&bb1).add(&bb2).add(&bb3)
        }
        FUNCT_MUL => {
            // Full 256-bit MUL: all 4 limbs
            let m0 = evaluate_arith_mul_limb0_raw(col_evals);
            let m1 = evaluate_arith_mul_limb1_raw(col_evals);
            let m2 = evaluate_arith_mul_limb2_raw(col_evals);
            let m3 = evaluate_arith_mul_limb3_raw(col_evals);
            m0.add(&m1).add(&m2).add(&m3)
        }
        FUNCT_DIV => {
            // Full 256-bit DIV: all 4 limbs
            let d0 = evaluate_arith_div_limb0_raw(col_evals);
            let d1 = evaluate_arith_div_limb1_raw(col_evals);
            let d2 = evaluate_arith_div_limb2_raw(col_evals);
            let d3 = evaluate_arith_div_limb3_raw(col_evals);
            d0.add(&d1).add(&d2).add(&d3)
        }
        _ => Scalar::zero(curve),
    }
}

/// Evaluate the ADD constraint raw, without any instruction type gating.
///
/// Computes the 4-limb ADD constraint body with carry chain (combined, kept for backward compat).
pub fn evaluate_arith_add_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);

    // Limb 0: input0_l0 + input1_l0 = aux0_l0 * 2^64 + output0_l0
    let sum_l0 = col_evals[COL_INPUT0_L0].add(&col_evals[COL_INPUT1_L0]);
    let carry0_term = col_evals[COL_AUX0_L0].mul(&two_64);
    let c_l0 = sum_l0.sub(&carry0_term).sub(&col_evals[COL_OUTPUT0_L0]);

    // Limb 1: input0_l1 + input1_l1 + carry0 = aux0_l1 * 2^64 + output0_l1
    let sum_l1 = col_evals[COL_INPUT0_L1].add(&col_evals[COL_INPUT1_L1]).add(&col_evals[COL_AUX0_L0]);
    let carry1_term = col_evals[COL_AUX0_L1].mul(&two_64);
    let c_l1 = sum_l1.sub(&carry1_term).sub(&col_evals[COL_OUTPUT0_L1]);

    // Limb 2: input0_l2 + input1_l2 + carry1 = aux0_l2 * 2^64 + output0_l2
    let sum_l2 = col_evals[COL_INPUT0_L2].add(&col_evals[COL_INPUT1_L2]).add(&col_evals[COL_AUX0_L1]);
    let carry2_term = col_evals[COL_AUX0_L2].mul(&two_64);
    let c_l2 = sum_l2.sub(&carry2_term).sub(&col_evals[COL_OUTPUT0_L2]);

    // Limb 3: input0_l3 + input1_l3 + carry2 = aux0_l3 * 2^64 + output0_l3
    let sum_l3 = col_evals[COL_INPUT0_L3].add(&col_evals[COL_INPUT1_L3]).add(&col_evals[COL_AUX0_L2]);
    let carry3_term = col_evals[COL_AUX0_L3].mul(&two_64);
    let c_l3 = sum_l3.sub(&carry3_term).sub(&col_evals[COL_OUTPUT0_L3]);

    // Each carry must be binary: aux0_lk * (aux0_lk - 1) = 0
    let cb0 = col_evals[COL_AUX0_L0].mul(&col_evals[COL_AUX0_L0].sub(&one));
    let cb1 = col_evals[COL_AUX0_L1].mul(&col_evals[COL_AUX0_L1].sub(&one));
    let cb2 = col_evals[COL_AUX0_L2].mul(&col_evals[COL_AUX0_L2].sub(&one));
    let cb3 = col_evals[COL_AUX0_L3].mul(&col_evals[COL_AUX0_L3].sub(&one));

    c_l0.add(&c_l1).add(&c_l2).add(&c_l3)
        .add(&cb0).add(&cb1).add(&cb2).add(&cb3)
}

// --- Per-limb ADD constraint raw functions ---

/// ADD limb 0: input0_l0 + input1_l0 - carry0*2^64 - output_l0
pub fn evaluate_arith_add_limb0_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_INPUT0_L0].add(&col_evals[COL_INPUT1_L0]);
    let carry_term = col_evals[COL_AUX0_L0].mul(&two_64);
    sum.sub(&carry_term).sub(&col_evals[COL_OUTPUT0_L0])
}

/// ADD limb 1: input0_l1 + input1_l1 + carry0 - carry1*2^64 - output_l1
pub fn evaluate_arith_add_limb1_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_INPUT0_L1].add(&col_evals[COL_INPUT1_L1]).add(&col_evals[COL_AUX0_L0]);
    let carry_term = col_evals[COL_AUX0_L1].mul(&two_64);
    sum.sub(&carry_term).sub(&col_evals[COL_OUTPUT0_L1])
}

/// ADD limb 2: input0_l2 + input1_l2 + carry1 - carry2*2^64 - output_l2
pub fn evaluate_arith_add_limb2_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_INPUT0_L2].add(&col_evals[COL_INPUT1_L2]).add(&col_evals[COL_AUX0_L1]);
    let carry_term = col_evals[COL_AUX0_L2].mul(&two_64);
    sum.sub(&carry_term).sub(&col_evals[COL_OUTPUT0_L2])
}

/// ADD limb 3: input0_l3 + input1_l3 + carry2 - carry3*2^64 - output_l3
pub fn evaluate_arith_add_limb3_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_INPUT0_L3].add(&col_evals[COL_INPUT1_L3]).add(&col_evals[COL_AUX0_L2]);
    let carry_term = col_evals[COL_AUX0_L3].mul(&two_64);
    sum.sub(&carry_term).sub(&col_evals[COL_OUTPUT0_L3])
}

// --- Per-carry binary constraint raw functions ---

/// carry0 binary: aux0_l0 * (aux0_l0 - 1)
pub fn evaluate_carry0_binary_raw(col_evals: &[Scalar]) -> Scalar {
    let one = Scalar::one(col_evals[0].curve_type());
    col_evals[COL_AUX0_L0].mul(&col_evals[COL_AUX0_L0].sub(&one))
}

/// carry1 binary: aux0_l1 * (aux0_l1 - 1)
pub fn evaluate_carry1_binary_raw(col_evals: &[Scalar]) -> Scalar {
    let one = Scalar::one(col_evals[0].curve_type());
    col_evals[COL_AUX0_L1].mul(&col_evals[COL_AUX0_L1].sub(&one))
}

/// carry2 binary: aux0_l2 * (aux0_l2 - 1)
pub fn evaluate_carry2_binary_raw(col_evals: &[Scalar]) -> Scalar {
    let one = Scalar::one(col_evals[0].curve_type());
    col_evals[COL_AUX0_L2].mul(&col_evals[COL_AUX0_L2].sub(&one))
}

/// carry3 binary: aux0_l3 * (aux0_l3 - 1)
pub fn evaluate_carry3_binary_raw(col_evals: &[Scalar]) -> Scalar {
    let one = Scalar::one(col_evals[0].curve_type());
    col_evals[COL_AUX0_L3].mul(&col_evals[COL_AUX0_L3].sub(&one))
}

// --- Per-limb SUB constraint raw functions ---

/// SUB limb 0: output_l0 + input1_l0 - borrow0*2^64 - input0_l0
pub fn evaluate_arith_sub_limb0_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_OUTPUT0_L0].add(&col_evals[COL_INPUT1_L0]);
    let borrow_term = col_evals[COL_AUX0_L0].mul(&two_64);
    sum.sub(&col_evals[COL_INPUT0_L0]).sub(&borrow_term)
}

/// SUB limb 1: output_l1 + input1_l1 + borrow0 - borrow1*2^64 - input0_l1
pub fn evaluate_arith_sub_limb1_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_OUTPUT0_L1].add(&col_evals[COL_INPUT1_L1]).add(&col_evals[COL_AUX0_L0]);
    let borrow_term = col_evals[COL_AUX0_L1].mul(&two_64);
    sum.sub(&col_evals[COL_INPUT0_L1]).sub(&borrow_term)
}

/// SUB limb 2: output_l2 + input1_l2 + borrow1 - borrow2*2^64 - input0_l2
pub fn evaluate_arith_sub_limb2_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_OUTPUT0_L2].add(&col_evals[COL_INPUT1_L2]).add(&col_evals[COL_AUX0_L1]);
    let borrow_term = col_evals[COL_AUX0_L2].mul(&two_64);
    sum.sub(&col_evals[COL_INPUT0_L2]).sub(&borrow_term)
}

/// SUB limb 3: output_l3 + input1_l3 + borrow2 - borrow3*2^64 - input0_l3
pub fn evaluate_arith_sub_limb3_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let sum = col_evals[COL_OUTPUT0_L3].add(&col_evals[COL_INPUT1_L3]).add(&col_evals[COL_AUX0_L2]);
    let borrow_term = col_evals[COL_AUX0_L3].mul(&two_64);
    sum.sub(&col_evals[COL_INPUT0_L3]).sub(&borrow_term)
}

// --- Per-borrow binary constraint raw functions (same structure as carry) ---

/// borrow0 binary: aux0_l0 * (aux0_l0 - 1)
pub fn evaluate_borrow0_binary_raw(col_evals: &[Scalar]) -> Scalar {
    evaluate_carry0_binary_raw(col_evals)
}

/// borrow1 binary: aux0_l1 * (aux0_l1 - 1)
pub fn evaluate_borrow1_binary_raw(col_evals: &[Scalar]) -> Scalar {
    evaluate_carry1_binary_raw(col_evals)
}

/// borrow2 binary: aux0_l2 * (aux0_l2 - 1)
pub fn evaluate_borrow2_binary_raw(col_evals: &[Scalar]) -> Scalar {
    evaluate_carry2_binary_raw(col_evals)
}

/// borrow3 binary: aux0_l3 * (aux0_l3 - 1)
pub fn evaluate_borrow3_binary_raw(col_evals: &[Scalar]) -> Scalar {
    evaluate_carry3_binary_raw(col_evals)
}

// --- MUL constraint raw functions ---
//
// Full 256-bit schoolbook MUL: a * b = output (mod 2^256). Using 4×64-bit
// limbs with carry chain stored in aux1 (c0..c3):
//
// limb 0: a0·b0                              = c0·2^64 + out0   (c0 in aux1_l0)
// limb 1: a0·b1 + a1·b0 + c0                 = c1·2^64 + out1   (c1 in aux1_l1)
// limb 2: a0·b2 + a1·b1 + a2·b0 + c1         = c2·2^64 + out2   (c2 in aux1_l2)
// limb 3: a0·b3 + a1·b2 + a2·b1 + a3·b0 + c2 = c3·2^64 + out3   (c3 in aux1_l3)
//
// EVM MUL outputs only the low 256 bits, so the upper-half schoolbook
// (limbs 4..7) is intentionally not constrained: aux0 on MUL rows is
// unused storage (declared as a 64-bit range check for defensive
// well-formedness; no other constraint reads it). Adding the upper-half
// schoolbook would let aux0 carry the high 256 bits of the 512-bit
// product — useful for MULMOD's intermediate, not for plain MUL.

/// MUL limb 0: a0*b0 - c0*2^64 - out0 = 0
pub fn evaluate_arith_mul_limb0_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let product_l0 = col_evals[COL_INPUT0_L0].mul(&col_evals[COL_INPUT1_L0]);
    let hi_term = col_evals[COL_AUX1_L0].mul(&two_64);
    let rhs = hi_term.add(&col_evals[COL_OUTPUT0_L0]);
    product_l0.sub(&rhs)
}

/// MUL limb 1: a0*b1 + a1*b0 + c0 - c1*2^64 - out1 = 0
pub fn evaluate_arith_mul_limb1_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_INPUT0_L0].mul(&col_evals[COL_INPUT1_L1])
        .add(&col_evals[COL_INPUT0_L1].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_AUX1_L0]); // + carry_in c0
    let hi_term = col_evals[COL_AUX1_L1].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_OUTPUT0_L1])
}

/// MUL limb 2: a0*b2 + a1*b1 + a2*b0 + c1 - c2*2^64 - out2 = 0
pub fn evaluate_arith_mul_limb2_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_INPUT0_L0].mul(&col_evals[COL_INPUT1_L2])
        .add(&col_evals[COL_INPUT0_L1].mul(&col_evals[COL_INPUT1_L1]))
        .add(&col_evals[COL_INPUT0_L2].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_AUX1_L1]); // + carry_in c1
    let hi_term = col_evals[COL_AUX1_L2].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_OUTPUT0_L2])
}

/// MUL limb 3: a0*b3 + a1*b2 + a2*b1 + a3*b0 + c2 - c3*2^64 - out3 = 0
pub fn evaluate_arith_mul_limb3_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_INPUT0_L0].mul(&col_evals[COL_INPUT1_L3])
        .add(&col_evals[COL_INPUT0_L1].mul(&col_evals[COL_INPUT1_L2]))
        .add(&col_evals[COL_INPUT0_L2].mul(&col_evals[COL_INPUT1_L1]))
        .add(&col_evals[COL_INPUT0_L3].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_AUX1_L2]); // + carry_in c2
    let hi_term = col_evals[COL_AUX1_L3].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_OUTPUT0_L3])
}

// --- DIV constraint raw functions ---
//
// DIV: quotient * divisor + remainder = dividend
//   output = quotient, input0 = dividend, input1 = divisor, aux0 = remainder
//
// This uses the same carry structure as MUL but with output*input1 + aux0 = input0:
// limb 0: out0*in1_0 + rem0 = c0*2^64 + div0                    (c0 in aux1_l0)
// limb 1: out0*in1_1 + out1*in1_0 + rem1 + c0 = c1*2^64 + div1  (c1 in aux1_l1)
// limb 2: out0*in1_2 + out1*in1_1 + out2*in1_0 + rem2 + c1 = c2*2^64 + div2
// limb 3: out0*in1_3 + out1*in1_2 + out2*in1_1 + out3*in1_0 + rem3 + c2 = c3*2^64 + div3

/// DIV limb 0: out0*in1_0 + rem0 - c0*2^64 - div0 = 0
pub fn evaluate_arith_div_limb0_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let prod_l0 = col_evals[COL_OUTPUT0_L0].mul(&col_evals[COL_INPUT1_L0]);
    let lhs = prod_l0.add(&col_evals[COL_AUX0_L0]);
    let hi_term = col_evals[COL_AUX1_L0].mul(&two_64);
    let rhs = hi_term.add(&col_evals[COL_INPUT0_L0]);
    lhs.sub(&rhs)
}

/// DIV limb 1: out0*in1_1 + out1*in1_0 + rem1 + c0 - c1*2^64 - div1 = 0
pub fn evaluate_arith_div_limb1_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_OUTPUT0_L0].mul(&col_evals[COL_INPUT1_L1])
        .add(&col_evals[COL_OUTPUT0_L1].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_AUX0_L1]) // rem1
        .add(&col_evals[COL_AUX1_L0]); // carry_in c0
    let hi_term = col_evals[COL_AUX1_L1].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_INPUT0_L1])
}

/// DIV limb 2: out0*in1_2 + out1*in1_1 + out2*in1_0 + rem2 + c1 - c2*2^64 - div2 = 0
pub fn evaluate_arith_div_limb2_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_OUTPUT0_L0].mul(&col_evals[COL_INPUT1_L2])
        .add(&col_evals[COL_OUTPUT0_L1].mul(&col_evals[COL_INPUT1_L1]))
        .add(&col_evals[COL_OUTPUT0_L2].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_AUX0_L2]) // rem2
        .add(&col_evals[COL_AUX1_L1]); // carry_in c1
    let hi_term = col_evals[COL_AUX1_L2].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_INPUT0_L2])
}

/// DIV limb 3: out0*in1_3 + out1*in1_2 + out2*in1_1 + out3*in1_0 + rem3 + c2 - c3*2^64 - div3 = 0
pub fn evaluate_arith_div_limb3_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_OUTPUT0_L0].mul(&col_evals[COL_INPUT1_L3])
        .add(&col_evals[COL_OUTPUT0_L1].mul(&col_evals[COL_INPUT1_L2]))
        .add(&col_evals[COL_OUTPUT0_L2].mul(&col_evals[COL_INPUT1_L1]))
        .add(&col_evals[COL_OUTPUT0_L3].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_AUX0_L3]) // rem3
        .add(&col_evals[COL_AUX1_L2]); // carry_in c2
    let hi_term = col_evals[COL_AUX1_L3].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_INPUT0_L3])
}

// --- MOD constraint raw functions ---
//
// MOD: quotient * divisor + remainder = dividend
//   output = remainder, input0 = dividend, input1 = divisor, aux0 = quotient
//
// Same carry structure as DIV but with aux0 and output swapped:
// limb 0: q0*in1_0 + out0 = c0*2^64 + div0                    (c0 in aux1_l0)
// limb 1: q0*in1_1 + q1*in1_0 + out1 + c0 = c1*2^64 + div1    (c1 in aux1_l1)
// limb 2: q0*in1_2 + q1*in1_1 + q2*in1_0 + out2 + c1 = c2*2^64 + div2
// limb 3: q0*in1_3 + q1*in1_2 + q2*in1_1 + q3*in1_0 + out3 + c2 = c3*2^64 + div3

/// MOD limb 0: q0*in1_0 + out0 - c0*2^64 - div0 = 0
pub fn evaluate_arith_mod_limb0_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let prod_l0 = col_evals[COL_AUX0_L0].mul(&col_evals[COL_INPUT1_L0]);
    let lhs = prod_l0.add(&col_evals[COL_OUTPUT0_L0]);
    let hi_term = col_evals[COL_AUX1_L0].mul(&two_64);
    let rhs = hi_term.add(&col_evals[COL_INPUT0_L0]);
    lhs.sub(&rhs)
}

/// MOD limb 1: q0*in1_1 + q1*in1_0 + out1 + c0 - c1*2^64 - div1 = 0
pub fn evaluate_arith_mod_limb1_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_AUX0_L0].mul(&col_evals[COL_INPUT1_L1])
        .add(&col_evals[COL_AUX0_L1].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_OUTPUT0_L1])
        .add(&col_evals[COL_AUX1_L0]); // carry_in c0
    let hi_term = col_evals[COL_AUX1_L1].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_INPUT0_L1])
}

/// MOD limb 2: q0*in1_2 + q1*in1_1 + q2*in1_0 + out2 + c1 - c2*2^64 - div2 = 0
pub fn evaluate_arith_mod_limb2_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_AUX0_L0].mul(&col_evals[COL_INPUT1_L2])
        .add(&col_evals[COL_AUX0_L1].mul(&col_evals[COL_INPUT1_L1]))
        .add(&col_evals[COL_AUX0_L2].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_OUTPUT0_L2])
        .add(&col_evals[COL_AUX1_L1]); // carry_in c1
    let hi_term = col_evals[COL_AUX1_L2].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_INPUT0_L2])
}

/// MOD limb 3: q0*in1_3 + q1*in1_2 + q2*in1_1 + q3*in1_0 + out3 + c2 - c3*2^64 - div3 = 0
pub fn evaluate_arith_mod_limb3_raw(col_evals: &[Scalar]) -> Scalar {
    let two_64 = two_pow_64(col_evals[0].curve_type());
    let cross = col_evals[COL_AUX0_L0].mul(&col_evals[COL_INPUT1_L3])
        .add(&col_evals[COL_AUX0_L1].mul(&col_evals[COL_INPUT1_L2]))
        .add(&col_evals[COL_AUX0_L2].mul(&col_evals[COL_INPUT1_L1]))
        .add(&col_evals[COL_AUX0_L3].mul(&col_evals[COL_INPUT1_L0]))
        .add(&col_evals[COL_OUTPUT0_L3])
        .add(&col_evals[COL_AUX1_L2]); // carry_in c2
    let hi_term = col_evals[COL_AUX1_L3].mul(&two_64);
    cross.sub(&hi_term).sub(&col_evals[COL_INPUT0_L3])
}

/// Full 4-limb MUL constraint body (sum of all limbs).
pub fn evaluate_arith_mul_full_raw(col_evals: &[Scalar]) -> Scalar {
    let m0 = evaluate_arith_mul_limb0_raw(col_evals);
    let m1 = evaluate_arith_mul_limb1_raw(col_evals);
    let m2 = evaluate_arith_mul_limb2_raw(col_evals);
    let m3 = evaluate_arith_mul_limb3_raw(col_evals);
    m0.add(&m1).add(&m2).add(&m3)
}

/// Full 4-limb DIV constraint body (sum of all limbs).
pub fn evaluate_arith_div_full_raw(col_evals: &[Scalar]) -> Scalar {
    let d0 = evaluate_arith_div_limb0_raw(col_evals);
    let d1 = evaluate_arith_div_limb1_raw(col_evals);
    let d2 = evaluate_arith_div_limb2_raw(col_evals);
    let d3 = evaluate_arith_div_limb3_raw(col_evals);
    d0.add(&d1).add(&d2).add(&d3)
}

/// Full 4-limb MOD constraint body (sum of all limbs).
pub fn evaluate_arith_mod_full_raw(col_evals: &[Scalar]) -> Scalar {
    let m0 = evaluate_arith_mod_limb0_raw(col_evals);
    let m1 = evaluate_arith_mod_limb1_raw(col_evals);
    let m2 = evaluate_arith_mod_limb2_raw(col_evals);
    let m3 = evaluate_arith_mod_limb3_raw(col_evals);
    m0.add(&m1).add(&m2).add(&m3)
}

/// Evaluate the SUB constraint raw, without any instruction type gating.
///
/// SUB: output + input1 = input0 + borrow*2^256 (4-limb borrow chain, combined, kept for backward compat).
pub fn evaluate_arith_sub_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);

    // Limb 0: output_l0 + input1_l0 - borrow0*2^64 - input0_l0
    let sum_l0 = col_evals[COL_OUTPUT0_L0].add(&col_evals[COL_INPUT1_L0]);
    let borrow0_term = col_evals[COL_AUX0_L0].mul(&two_64);
    let c_l0 = sum_l0.sub(&col_evals[COL_INPUT0_L0]).sub(&borrow0_term);

    // Limb 1: output_l1 + input1_l1 + borrow0 - borrow1*2^64 - input0_l1
    let sum_l1 = col_evals[COL_OUTPUT0_L1].add(&col_evals[COL_INPUT1_L1]).add(&col_evals[COL_AUX0_L0]);
    let borrow1_term = col_evals[COL_AUX0_L1].mul(&two_64);
    let c_l1 = sum_l1.sub(&col_evals[COL_INPUT0_L1]).sub(&borrow1_term);

    // Limb 2: output_l2 + input1_l2 + borrow1 - borrow2*2^64 - input0_l2
    let sum_l2 = col_evals[COL_OUTPUT0_L2].add(&col_evals[COL_INPUT1_L2]).add(&col_evals[COL_AUX0_L1]);
    let borrow2_term = col_evals[COL_AUX0_L2].mul(&two_64);
    let c_l2 = sum_l2.sub(&col_evals[COL_INPUT0_L2]).sub(&borrow2_term);

    // Limb 3: output_l3 + input1_l3 + borrow2 - borrow3*2^64 - input0_l3
    let sum_l3 = col_evals[COL_OUTPUT0_L3].add(&col_evals[COL_INPUT1_L3]).add(&col_evals[COL_AUX0_L2]);
    let borrow3_term = col_evals[COL_AUX0_L3].mul(&two_64);
    let c_l3 = sum_l3.sub(&col_evals[COL_INPUT0_L3]).sub(&borrow3_term);

    // Each borrow must be binary
    let bb0 = col_evals[COL_AUX0_L0].mul(&col_evals[COL_AUX0_L0].sub(&one));
    let bb1 = col_evals[COL_AUX0_L1].mul(&col_evals[COL_AUX0_L1].sub(&one));
    let bb2 = col_evals[COL_AUX0_L2].mul(&col_evals[COL_AUX0_L2].sub(&one));
    let bb3 = col_evals[COL_AUX0_L3].mul(&col_evals[COL_AUX0_L3].sub(&one));

    c_l0.add(&c_l1).add(&c_l2).add(&c_l3)
        .add(&bb0).add(&bb1).add(&bb2).add(&bb3)
}
