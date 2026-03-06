//! ALU constraints for SBF 64-bit arithmetic operations.
//!
//! SBF ALU is very similar to RISC-V: dst = dst OP src.
//! - ADD: dst_before + src = carry * 2^64 + dst_after
//! - SUB: dst_after + src = dst_before + borrow * 2^64
//! - MUL: dst_before * src = aux0 * 2^64 + dst_after
//! - DIV: dst_after * src + aux0 = dst_before (quotient*divisor+remainder=dividend)
//! - MOD: aux0 * src + dst_after = dst_before (quotient*divisor+remainder=dividend)
//! - MOV: dst_after = src

use metavm_zkp::field::{Scalar, CurveType};
use crate::trace::*;

pub fn two_pow_64(curve: CurveType) -> Scalar {
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

/// Evaluate ALU constraints on the full domain.
pub fn evaluate_alu(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let two_64 = two_pow_64(curve);
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type = columns[COL_INSN_TYPE][i].to_u64() as u8;
        let funct = columns[COL_FUNCT][i].to_u64() as u8;

        let constraint = if insn_type == INSN_ALU64 || insn_type == INSN_ALU32 {
            let alu_constraint = match funct {
                FUNCT_ADD => {
                    // dst_before + src_val = carry * 2^64 + dst_after
                    // (binary check for carry is a separate constraint)
                    let sum = columns[COL_DST_VAL_BEFORE][i].add(&columns[COL_SRC_VAL][i]);
                    let carry_term = columns[COL_AUX0][i].mul(&two_64);
                    let rhs = carry_term.add(&columns[COL_DST_VAL_AFTER][i]);
                    sum.sub(&rhs)
                }
                FUNCT_SUB => {
                    // dst_after + src = dst_before + borrow*2^64
                    // (binary check for borrow is a separate constraint)
                    let lhs = columns[COL_DST_VAL_AFTER][i].add(&columns[COL_SRC_VAL][i]);
                    let borrow_term = columns[COL_AUX0][i].mul(&two_64);
                    let rhs = columns[COL_DST_VAL_BEFORE][i].add(&borrow_term);
                    lhs.sub(&rhs)
                }
                FUNCT_MUL => {
                    // dst_before * src = aux0 * 2^64 + dst_after
                    let product = columns[COL_DST_VAL_BEFORE][i].mul(&columns[COL_SRC_VAL][i]);
                    let hi_term = columns[COL_AUX0][i].mul(&two_64);
                    let rhs = hi_term.add(&columns[COL_DST_VAL_AFTER][i]);
                    product.sub(&rhs)
                }
                FUNCT_DIV => {
                    // dst_after * src + aux0 = dst_before
                    let prod = columns[COL_DST_VAL_AFTER][i].mul(&columns[COL_SRC_VAL][i]);
                    let lhs = prod.add(&columns[COL_AUX0][i]);
                    lhs.sub(&columns[COL_DST_VAL_BEFORE][i])
                }
                FUNCT_MOD => {
                    // aux0 * src + dst_after = dst_before
                    let prod = columns[COL_AUX0][i].mul(&columns[COL_SRC_VAL][i]);
                    let lhs = prod.add(&columns[COL_DST_VAL_AFTER][i]);
                    lhs.sub(&columns[COL_DST_VAL_BEFORE][i])
                }
                FUNCT_MOV => {
                    // dst_after = src_val
                    columns[COL_DST_VAL_AFTER][i].sub(&columns[COL_SRC_VAL][i])
                }
                _ => Scalar::zero(curve), // Bitwise/shift deferred to lookup
            };

            // For immediate-source ALU ops (opcode bit 3 = 0),
            // verify src_val = immediate to prevent source substitution attacks.
            let opcode_val = columns[COL_OPCODE][i].to_u64();
            let is_imm_source = (opcode_val & 0x08) == 0;
            if is_imm_source {
                let src_check = columns[COL_SRC_VAL][i].sub(&columns[COL_IMMEDIATE][i]);
                alu_constraint.add(&src_check)
            } else {
                alu_constraint
            }
        } else {
            Scalar::zero(curve)
        };

        result.push(constraint);
    }

    result
}

/// Evaluate ALU constraint at a single point (legacy, uses integer branching).
pub fn evaluate_alu_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < NUM_SBF_COLUMNS {
        return Scalar::zero(curve);
    }

    let insn_type = col_evals[COL_INSN_TYPE].to_u64() as u8;
    let funct = col_evals[COL_FUNCT].to_u64() as u8;
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);

    if insn_type != INSN_ALU64 && insn_type != INSN_ALU32 {
        return Scalar::zero(curve);
    }

    match funct {
        FUNCT_ADD => {
            let sum = col_evals[COL_DST_VAL_BEFORE].add(&col_evals[COL_SRC_VAL]);
            let carry_term = col_evals[COL_AUX0].mul(&two_64);
            let rhs = carry_term.add(&col_evals[COL_DST_VAL_AFTER]);
            let c1 = sum.sub(&rhs);
            let carry_m1 = col_evals[COL_AUX0].sub(&one);
            let c2 = col_evals[COL_AUX0].mul(&carry_m1);
            c1.add(&c2)
        }
        FUNCT_SUB => {
            let lhs = col_evals[COL_DST_VAL_AFTER].add(&col_evals[COL_SRC_VAL]);
            let borrow_term = col_evals[COL_AUX0].mul(&two_64);
            let rhs = col_evals[COL_DST_VAL_BEFORE].add(&borrow_term);
            let c1 = lhs.sub(&rhs);
            let borrow_m1 = col_evals[COL_AUX0].sub(&one);
            let c2 = col_evals[COL_AUX0].mul(&borrow_m1);
            c1.add(&c2)
        }
        FUNCT_MUL => {
            let product = col_evals[COL_DST_VAL_BEFORE].mul(&col_evals[COL_SRC_VAL]);
            let hi_term = col_evals[COL_AUX0].mul(&two_64);
            let rhs = hi_term.add(&col_evals[COL_DST_VAL_AFTER]);
            product.sub(&rhs)
        }
        FUNCT_DIV => {
            let prod = col_evals[COL_DST_VAL_AFTER].mul(&col_evals[COL_SRC_VAL]);
            let lhs = prod.add(&col_evals[COL_AUX0]);
            lhs.sub(&col_evals[COL_DST_VAL_BEFORE])
        }
        FUNCT_MOD => {
            let prod = col_evals[COL_AUX0].mul(&col_evals[COL_SRC_VAL]);
            let lhs = prod.add(&col_evals[COL_DST_VAL_AFTER]);
            lhs.sub(&col_evals[COL_DST_VAL_BEFORE])
        }
        FUNCT_MOV => {
            col_evals[COL_DST_VAL_AFTER].sub(&col_evals[COL_SRC_VAL])
        }
        _ => Scalar::zero(curve),
    }
}

/// Evaluate ADD constraint body without insn_type gating.
///
/// ADD: dst_before + src = carry*2^64 + dst_after, plus carry binary check.
pub fn evaluate_alu_add_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let sum = col_evals[COL_DST_VAL_BEFORE].add(&col_evals[COL_SRC_VAL]);
    let carry_term = col_evals[COL_AUX0].mul(&two_64);
    let rhs = carry_term.add(&col_evals[COL_DST_VAL_AFTER]);
    let c1 = sum.sub(&rhs);
    let carry_m1 = col_evals[COL_AUX0].sub(&one);
    let c2 = col_evals[COL_AUX0].mul(&carry_m1);
    c1.add(&c2)
}

/// Evaluate SUB constraint body without insn_type gating.
///
/// SUB: dst_after + src = dst_before + borrow*2^64, plus borrow binary check.
pub fn evaluate_alu_sub_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let lhs = col_evals[COL_DST_VAL_AFTER].add(&col_evals[COL_SRC_VAL]);
    let borrow_term = col_evals[COL_AUX0].mul(&two_64);
    let rhs = col_evals[COL_DST_VAL_BEFORE].add(&borrow_term);
    let c1 = lhs.sub(&rhs);
    let borrow_m1 = col_evals[COL_AUX0].sub(&one);
    let c2 = col_evals[COL_AUX0].mul(&borrow_m1);
    c1.add(&c2)
}

/// Evaluate MOV constraint body without insn_type gating.
///
/// MOV: dst_after = src_val.
pub fn evaluate_alu_mov_raw(col_evals: &[Scalar]) -> Scalar {
    col_evals[COL_DST_VAL_AFTER].sub(&col_evals[COL_SRC_VAL])
}

/// Evaluate MUL constraint body without insn_type gating.
///
/// MUL: dst_before * src_val = aux0 * 2^64 + dst_after.
pub fn evaluate_alu_mul_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let product = col_evals[COL_DST_VAL_BEFORE].mul(&col_evals[COL_SRC_VAL]);
    let hi_term = col_evals[COL_AUX0].mul(&two_64);
    let rhs = hi_term.add(&col_evals[COL_DST_VAL_AFTER]);
    product.sub(&rhs)
}

/// Evaluate DIV constraint body without insn_type gating.
///
/// DIV: dst_after * src + aux0 = dst_before (quotient*divisor+remainder=dividend).
pub fn evaluate_alu_div_raw(col_evals: &[Scalar]) -> Scalar {
    let prod = col_evals[COL_DST_VAL_AFTER].mul(&col_evals[COL_SRC_VAL]);
    let lhs = prod.add(&col_evals[COL_AUX0]);
    lhs.sub(&col_evals[COL_DST_VAL_BEFORE])
}

/// Evaluate MOD constraint body without insn_type gating.
///
/// MOD: aux0 * src + dst_after = dst_before (quotient*divisor+remainder=dividend).
pub fn evaluate_alu_mod_raw(col_evals: &[Scalar]) -> Scalar {
    let prod = col_evals[COL_AUX0].mul(&col_evals[COL_SRC_VAL]);
    let lhs = prod.add(&col_evals[COL_DST_VAL_AFTER]);
    lhs.sub(&col_evals[COL_DST_VAL_BEFORE])
}

/// ADD main equation only (no carry binary check).
///
/// ADD: dst_before + src - carry*2^64 - dst_after = 0.
/// The carry binary check (aux0*(aux0-1)) is a separate constraint.
pub fn evaluate_alu_add_main_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let sum = col_evals[COL_DST_VAL_BEFORE].add(&col_evals[COL_SRC_VAL]);
    let carry_term = col_evals[COL_AUX0].mul(&two_64);
    let rhs = carry_term.add(&col_evals[COL_DST_VAL_AFTER]);
    sum.sub(&rhs)
}

/// SUB main equation only (no borrow binary check).
///
/// SUB: dst_after + src - dst_before - borrow*2^64 = 0.
/// The borrow binary check (aux0*(aux0-1)) is a separate constraint.
pub fn evaluate_alu_sub_main_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let lhs = col_evals[COL_DST_VAL_AFTER].add(&col_evals[COL_SRC_VAL]);
    let borrow_term = col_evals[COL_AUX0].mul(&two_64);
    let rhs = col_evals[COL_DST_VAL_BEFORE].add(&borrow_term);
    lhs.sub(&rhs)
}

/// Evaluate AND constraint body without insn_type gating.
///
/// AND: dst_after = aux0, where aux0 = AND(dst_before, src_val).
pub fn evaluate_alu_and_raw(col_evals: &[Scalar]) -> Scalar {
    col_evals[COL_DST_VAL_AFTER].sub(&col_evals[COL_AUX0])
}

/// Evaluate OR constraint body without insn_type gating.
///
/// OR: dst_after = dst_before + src_val - aux0
/// => dst_after - dst_before - src_val + aux0 = 0
pub fn evaluate_alu_or_raw(col_evals: &[Scalar]) -> Scalar {
    col_evals[COL_DST_VAL_AFTER]
        .sub(&col_evals[COL_DST_VAL_BEFORE])
        .sub(&col_evals[COL_SRC_VAL])
        .add(&col_evals[COL_AUX0])
}

/// Evaluate XOR constraint body without insn_type gating.
///
/// XOR: dst_after = dst_before + src_val - 2*aux0
/// => dst_after - dst_before - src_val + 2*aux0 = 0
pub fn evaluate_alu_xor_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two = Scalar::from_u64(2, curve);
    col_evals[COL_DST_VAL_AFTER]
        .sub(&col_evals[COL_DST_VAL_BEFORE])
        .sub(&col_evals[COL_SRC_VAL])
        .add(&two.mul(&col_evals[COL_AUX0]))
}

/// Evaluate LSH constraint body without insn_type gating.
///
/// LSH: dst_before * aux0 - aux1 * 2^64 - dst_after = 0
/// where aux0 = 2^k (shift power), aux1 = overflow.
pub fn evaluate_alu_lsh_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let product = col_evals[COL_DST_VAL_BEFORE].mul(&col_evals[COL_AUX0]);
    let overflow_term = col_evals[COL_AUX1].mul(&two_64);
    product.sub(&overflow_term).sub(&col_evals[COL_DST_VAL_AFTER])
}

/// Evaluate RSH constraint body without insn_type gating.
///
/// RSH: dst_after * aux0 + aux1 - dst_before = 0
/// where aux0 = 2^k (shift power), aux1 = remainder = dst_before mod 2^k.
pub fn evaluate_alu_rsh_raw(col_evals: &[Scalar]) -> Scalar {
    let product = col_evals[COL_DST_VAL_AFTER].mul(&col_evals[COL_AUX0]);
    product.add(&col_evals[COL_AUX1]).sub(&col_evals[COL_DST_VAL_BEFORE])
}

/// Evaluate ARSH constraint body without insn_type gating.
///
/// ARSH: dst_after * aux0 + aux1 - dst_before - aux2 * 2^64 = 0
/// where aux0 = 2^k, aux1 = remainder = dst_before mod 2^k,
/// aux2 = sign_bit * (2^k - 1) (sign extension correction).
pub fn evaluate_alu_arsh_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let product = col_evals[COL_DST_VAL_AFTER].mul(&col_evals[COL_AUX0]);
    let correction_term = col_evals[COL_AUX2].mul(&two_64);
    product.add(&col_evals[COL_AUX1])
        .sub(&col_evals[COL_DST_VAL_BEFORE])
        .sub(&correction_term)
}

/// Evaluate NEG constraint body without insn_type gating.
///
/// NEG: dst_after + dst_before = aux0 * 2^64 (two's complement negation).
/// When dst_before = 0, dst_after = 0 and aux0 = 0.
/// When dst_before != 0, dst_after = 2^64 - dst_before and aux0 = 1.
/// Plus aux0 binary check: aux0 * (aux0 - 1) = 0.
pub fn evaluate_alu_neg_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    // Main equation: dst_after + dst_before - aux0 * 2^64 = 0
    let sum = col_evals[COL_DST_VAL_AFTER].add(&col_evals[COL_DST_VAL_BEFORE]);
    let carry_term = col_evals[COL_AUX0].mul(&two_64);
    let main = sum.sub(&carry_term);
    // Binary check: aux0 * (aux0 - 1) = 0
    let binary = col_evals[COL_AUX0].mul(&col_evals[COL_AUX0].sub(&one));
    main.add(&binary)
}

/// aux0 binary check: aux0 * (aux0 - 1) = 0.
///
/// This is zero iff aux0 is in {0, 1}. Used independently by both
/// ADD carry and SUB borrow constraints to prevent cancellation attacks.
pub fn evaluate_aux0_binary_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    col_evals[COL_AUX0].mul(&col_evals[COL_AUX0].sub(&one))
}

/// Evaluate CALL constraint body.
///
/// BPF_CALL invokes a helper function. The return value is verified externally
/// by the oracle mechanism (public input binding). Constraint body: zero.
pub fn evaluate_call_raw(col_evals: &[Scalar]) -> Scalar {
    Scalar::zero(col_evals[0].curve_type())
}

/// Evaluate EXIT constraint body.
///
/// BPF_EXIT terminates execution. Correctness is verified externally
/// by the oracle mechanism (public input binding). Constraint body: zero.
pub fn evaluate_exit_raw(col_evals: &[Scalar]) -> Scalar {
    Scalar::zero(col_evals[0].curve_type())
}

/// Evaluate aux0 binary checks for ADD carry and SUB borrow on the full domain.
///
/// Returns 2 vectors (one per constraint), each gated by the appropriate selector
/// via the domain integer branching. These are separated from the main ALU equations
/// to prevent cancellation attacks.
pub fn evaluate_aux0_binary_checks(columns: &[&Vec<Scalar>], num_rows: usize) -> (Vec<Scalar>, Vec<Scalar>) {
    let curve = columns[0][0].curve_type();
    let one = Scalar::one(curve);
    let mut add_carry_binary = Vec::with_capacity(num_rows);
    let mut sub_borrow_binary = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type = columns[COL_INSN_TYPE][i].to_u64() as u8;
        let funct = columns[COL_FUNCT][i].to_u64() as u8;

        let is_add = (insn_type == INSN_ALU64 || insn_type == INSN_ALU32) && funct == FUNCT_ADD;
        let is_sub = (insn_type == INSN_ALU64 || insn_type == INSN_ALU32) && funct == FUNCT_SUB;

        let binary_val = if is_add || is_sub {
            let aux0_m1 = columns[COL_AUX0][i].sub(&one);
            columns[COL_AUX0][i].mul(&aux0_m1)
        } else {
            Scalar::zero(curve)
        };

        add_carry_binary.push(if is_add { binary_val.clone() } else { Scalar::zero(curve) });
        sub_borrow_binary.push(if is_sub { binary_val } else { Scalar::zero(curve) });
    }

    (add_carry_binary, sub_borrow_binary)
}
