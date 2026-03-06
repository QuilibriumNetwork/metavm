//! ALU constraints for R-type, I-type, W-type, and MUL/DIV instructions.
//!
//! These constraints verify that rd_val_after matches the correct ALU operation
//! applied to the source operands. They are gated by the instruction type selector
//! so they only apply to ALU instruction rows.
//!
//! Auxiliary columns (aux0, aux1) are used for overflow/carry/borrow handling:
//! - ADD: rs1 + rs2 = carry * 2^64 + rd, carry ∈ {0,1}
//! - SUB: rd + rs2 = rs1 + borrow * 2^64, borrow ∈ {0,1}
//! - MUL: rs1 * rs2 = aux0 * 2^64 + rd (128-bit product)
//! - DIV: rd * rs2 + aux0 = rs1 (quotient*divisor + remainder = dividend)
//! - REM: aux0 * rs2 + rd = rs1 (quotient*divisor + remainder = dividend)
//! - SLT/SLTU: rd * (rd - 1) = 0 (binary result)

use metavm_zkp::field::{Scalar, CurveType};
use super::selectors;
use super::{COL_RD_VAL_AFTER, COL_RS1_VAL, COL_RS2_VAL, COL_MEM_VAL, COL_INSN_TYPE, COL_FUNCT, COL_IMMEDIATE, COL_AUX0, COL_AUX1, COL_AUX2};

/// Constant 2^64 as a Scalar.
pub fn two_pow_64(curve: CurveType) -> Scalar {
    use bls48581::bls48581::big;
    match curve {
        CurveType::Bls48581 => {
            let mut buf = [0u8; big::MODBYTES];
            buf[big::MODBYTES - 9] = 1; // byte at position 8 from the right = 2^64
            Scalar::Bls48581(big::BIG::frombytes(&buf))
        }
        CurveType::Bls12381 => {
            // 2^64 as a blst scalar
            // In blst little-endian representation: byte[8] = 1
            let mut scalar = blst::blst_scalar::default();
            scalar.b[8] = 1;
            let mut fr = blst::blst_fr::default();
            unsafe {
                blst::blst_fr_from_scalar(&mut fr, &scalar);
            }
            Scalar::Bls12381(fr)
        }
    }
}

/// Evaluate ALU constraints. Returns a vector of constraint evaluations per row.
pub fn evaluate_alu(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type_val = columns[COL_INSN_TYPE][i].to_u64();
        let funct_val = columns[COL_FUNCT][i].to_u64();

        let constraint = match insn_type_val as u8 {
            selectors::INSN_R_ALU => {
                match funct_val as u8 {
                    selectors::FUNCT_ADD => {
                        // rs1 + rs2 = carry * 2^64 + rd (main equation only)
                        let sum = columns[COL_RS1_VAL][i].add(&columns[COL_RS2_VAL][i]);
                        let carry_term = columns[COL_AUX0][i].mul(&two_64);
                        let rhs = carry_term.add(&columns[COL_RD_VAL_AFTER][i]);
                        sum.sub(&rhs)
                    }
                    selectors::FUNCT_SUB => {
                        // rd + rs2 = rs1 + borrow*2^64 (main equation only)
                        let lhs = columns[COL_RD_VAL_AFTER][i].add(&columns[COL_RS2_VAL][i]);
                        let borrow_term = columns[COL_AUX0][i].mul(&two_64);
                        let rhs = columns[COL_RS1_VAL][i].add(&borrow_term);
                        lhs.sub(&rhs)
                    }
                    selectors::FUNCT_SLT | selectors::FUNCT_SLTU => {
                        // rd must be binary: rd*(rd-1) = 0
                        let rd_minus_1 = columns[COL_RD_VAL_AFTER][i].sub(&one);
                        columns[COL_RD_VAL_AFTER][i].mul(&rd_minus_1)
                    }
                    _ => Scalar::zero(curve), // Bitwise/shift ops deferred to lookup
                }
            }
            selectors::INSN_I_ALU => {
                match funct_val as u8 {
                    selectors::FUNCT_ADDI => {
                        // rs1 + imm = carry * 2^64 + rd (main equation only)
                        let sum = columns[COL_RS1_VAL][i].add(&columns[COL_IMMEDIATE][i]);
                        let carry_term = columns[COL_AUX0][i].mul(&two_64);
                        let rhs = carry_term.add(&columns[COL_RD_VAL_AFTER][i]);
                        sum.sub(&rhs)
                    }
                    selectors::FUNCT_SLTI | selectors::FUNCT_SLTIU => {
                        // rd must be binary
                        let rd_minus_1 = columns[COL_RD_VAL_AFTER][i].sub(&one);
                        columns[COL_RD_VAL_AFTER][i].mul(&rd_minus_1)
                    }
                    _ => Scalar::zero(curve), // Bitwise/shift deferred
                }
            }
            selectors::INSN_MULDIV => {
                match funct_val as u8 {
                    0 => {
                        // MUL: rs1 * rs2 = aux0 * 2^64 + rd
                        let product = columns[COL_RS1_VAL][i].mul(&columns[COL_RS2_VAL][i]);
                        let hi_term = columns[COL_AUX0][i].mul(&two_64);
                        let rhs = hi_term.add(&columns[COL_RD_VAL_AFTER][i]);
                        product.sub(&rhs)
                    }
                    1 | 2 | 3 => {
                        // MULH/MULHSU/MULHU: rd * 2^64 + aux0 = rs1 * rs2
                        let product = columns[COL_RS1_VAL][i].mul(&columns[COL_RS2_VAL][i]);
                        let hi_term = columns[COL_RD_VAL_AFTER][i].mul(&two_64);
                        let rhs = hi_term.add(&columns[COL_AUX0][i]);
                        product.sub(&rhs)
                    }
                    4 | 5 => {
                        // DIV/DIVU: rd * rs2 + aux0 = rs1
                        let prod = columns[COL_RD_VAL_AFTER][i].mul(&columns[COL_RS2_VAL][i]);
                        let lhs = prod.add(&columns[COL_AUX0][i]);
                        lhs.sub(&columns[COL_RS1_VAL][i])
                    }
                    6 | 7 => {
                        // REM/REMU: aux0 * rs2 + rd = rs1
                        let prod = columns[COL_AUX0][i].mul(&columns[COL_RS2_VAL][i]);
                        let lhs = prod.add(&columns[COL_RD_VAL_AFTER][i]);
                        lhs.sub(&columns[COL_RS1_VAL][i])
                    }
                    8 => {
                        // MULW: lower32(rs1) * lower32(rs2) = aux0 * 2^32 + rd_lower32
                        // Partial: verify rs1*rs2 = aux0*2^64 + rd (same structure as MUL, approximate)
                        let product = columns[COL_RS1_VAL][i].mul(&columns[COL_RS2_VAL][i]);
                        let hi_term = columns[COL_AUX0][i].mul(&two_64);
                        let rhs = hi_term.add(&columns[COL_RD_VAL_AFTER][i]);
                        product.sub(&rhs)
                    }
                    9 | 10 => {
                        // DIVW/DIVUW: rd * rs2 + aux0 = rs1 (quotient*divisor+remainder=dividend)
                        // Approximate: uses full 64-bit values
                        let prod = columns[COL_RD_VAL_AFTER][i].mul(&columns[COL_RS2_VAL][i]);
                        let lhs = prod.add(&columns[COL_AUX0][i]);
                        lhs.sub(&columns[COL_RS1_VAL][i])
                    }
                    11 | 12 => {
                        // REMW/REMUW: aux0 * rs2 + rd = rs1
                        let prod = columns[COL_AUX0][i].mul(&columns[COL_RS2_VAL][i]);
                        let lhs = prod.add(&columns[COL_RD_VAL_AFTER][i]);
                        lhs.sub(&columns[COL_RS1_VAL][i])
                    }
                    _ => Scalar::zero(curve),
                }
            }
            selectors::INSN_W_ALU => {
                let two_32 = Scalar::from_u64(1u64 << 32, curve);
                let sign_ext_offset = two_64.sub(&two_32); // 2^64 - 2^32
                match funct_val as u8 {
                    selectors::FUNCT_ADDW => {
                        // ADDW: rd = sign_extend_32((rs1 + rs2)[31:0])
                        // Constraint: rs1 + rs2 - aux0 * 2^32 - rd + aux1 * (2^64 - 2^32) = 0
                        // (ignoring 64-bit carry — partially sound)
                        // Plus: aux1 binary (sign bit)
                        let sum = columns[COL_RS1_VAL][i].add(&columns[COL_RS2_VAL][i]);
                        let upper = columns[COL_AUX0][i].mul(&two_32);
                        let sign_term = columns[COL_AUX1][i].mul(&sign_ext_offset);
                        let c1 = sum.sub(&upper).sub(&columns[COL_RD_VAL_AFTER][i]).add(&sign_term);
                        let c2 = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));
                        c1.add(&c2)
                    }
                    selectors::FUNCT_SUBW => {
                        // SUBW: rd = sign_extend_32((rs1 - rs2)[31:0])
                        let diff = columns[COL_RS1_VAL][i].sub(&columns[COL_RS2_VAL][i]);
                        let upper = columns[COL_AUX0][i].mul(&two_32);
                        let sign_term = columns[COL_AUX1][i].mul(&sign_ext_offset);
                        let c1 = diff.sub(&upper).sub(&columns[COL_RD_VAL_AFTER][i]).add(&sign_term);
                        let c2 = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));
                        c1.add(&c2)
                    }
                    selectors::FUNCT_ADDIW => {
                        // ADDIW: rd = sign_extend_32((rs1 + imm)[31:0])
                        let sum = columns[COL_RS1_VAL][i].add(&columns[COL_IMMEDIATE][i]);
                        let upper = columns[COL_AUX0][i].mul(&two_32);
                        let sign_term = columns[COL_AUX1][i].mul(&sign_ext_offset);
                        let c1 = sum.sub(&upper).sub(&columns[COL_RD_VAL_AFTER][i]).add(&sign_term);
                        let c2 = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));
                        c1.add(&c2)
                    }
                    _ => Scalar::zero(curve), // Shift W-variants deferred to lookup
                }
            }
            _ => Scalar::zero(curve),
        };

        result.push(constraint);
    }

    result
}

/// Evaluate ALU constraint at a single point given column evaluations.
/// Column layout: 0:pc, 1:rd, 2:rd_val_before, 3:rd_val_after,
///   4:rs1, 5:rs1_val, 6:rs2, 7:rs2_val,
///   8:mem_addr, 9:mem_val, 10:next_pc, 11:privilege_mode,
///   12:insn_type, 13:funct, 14:immediate, 15:insn_len, 16:aux0, 17:aux1
pub fn evaluate_alu_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < 16 {
        return Scalar::zero(curve);
    }

    let insn_type_val = col_evals[COL_INSN_TYPE].to_u64() as u8;
    let funct_val = col_evals[COL_FUNCT].to_u64() as u8;
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    // Helper: get aux0 if available
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };

    match insn_type_val {
        selectors::INSN_R_ALU => {
            match funct_val {
                selectors::FUNCT_ADD => {
                    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_RS2_VAL]);
                    let carry_term = aux0.mul(&two_64);
                    let rhs = carry_term.add(&col_evals[COL_RD_VAL_AFTER]);
                    let c1 = sum.sub(&rhs);
                    let carry_m1 = aux0.sub(&one);
                    let c2 = aux0.mul(&carry_m1);
                    c1.add(&c2)
                }
                selectors::FUNCT_SUB => {
                    let lhs = col_evals[COL_RD_VAL_AFTER].add(&col_evals[COL_RS2_VAL]);
                    let borrow_term = aux0.mul(&two_64);
                    let rhs = col_evals[COL_RS1_VAL].add(&borrow_term);
                    let c1 = lhs.sub(&rhs);
                    let borrow_m1 = aux0.sub(&one);
                    let c2 = aux0.mul(&borrow_m1);
                    c1.add(&c2)
                }
                selectors::FUNCT_SLT | selectors::FUNCT_SLTU => {
                    let rd_m1 = col_evals[COL_RD_VAL_AFTER].sub(&one);
                    col_evals[COL_RD_VAL_AFTER].mul(&rd_m1)
                }
                _ => Scalar::zero(curve),
            }
        }
        selectors::INSN_I_ALU => {
            match funct_val {
                selectors::FUNCT_ADDI => {
                    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
                    let carry_term = aux0.mul(&two_64);
                    let rhs = carry_term.add(&col_evals[COL_RD_VAL_AFTER]);
                    let c1 = sum.sub(&rhs);
                    let carry_m1 = aux0.sub(&one);
                    let c2 = aux0.mul(&carry_m1);
                    c1.add(&c2)
                }
                selectors::FUNCT_SLTI | selectors::FUNCT_SLTIU => {
                    let rd_m1 = col_evals[COL_RD_VAL_AFTER].sub(&one);
                    col_evals[COL_RD_VAL_AFTER].mul(&rd_m1)
                }
                _ => Scalar::zero(curve),
            }
        }
        selectors::INSN_MULDIV => {
            match funct_val {
                0 => {
                    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
                    let hi_term = aux0.mul(&two_64);
                    let rhs = hi_term.add(&col_evals[COL_RD_VAL_AFTER]);
                    product.sub(&rhs)
                }
                1 | 2 | 3 => {
                    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
                    let hi_term = col_evals[COL_RD_VAL_AFTER].mul(&two_64);
                    let rhs = hi_term.add(aux0);
                    product.sub(&rhs)
                }
                4 | 5 => {
                    let prod = col_evals[COL_RD_VAL_AFTER].mul(&col_evals[COL_RS2_VAL]);
                    let lhs = prod.add(aux0);
                    lhs.sub(&col_evals[COL_RS1_VAL])
                }
                6 | 7 => {
                    let prod = aux0.mul(&col_evals[COL_RS2_VAL]);
                    let lhs = prod.add(&col_evals[COL_RD_VAL_AFTER]);
                    lhs.sub(&col_evals[COL_RS1_VAL])
                }
                8 => {
                    // MULW: approximate using rs1*rs2 = aux0*2^64 + rd
                    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
                    let hi_term = aux0.mul(&two_64);
                    let rhs = hi_term.add(&col_evals[COL_RD_VAL_AFTER]);
                    product.sub(&rhs)
                }
                9 | 10 => {
                    // DIVW/DIVUW: rd * rs2 + aux0 = rs1
                    let prod = col_evals[COL_RD_VAL_AFTER].mul(&col_evals[COL_RS2_VAL]);
                    let lhs = prod.add(aux0);
                    lhs.sub(&col_evals[COL_RS1_VAL])
                }
                11 | 12 => {
                    // REMW/REMUW: aux0 * rs2 + rd = rs1
                    let prod = aux0.mul(&col_evals[COL_RS2_VAL]);
                    let lhs = prod.add(&col_evals[COL_RD_VAL_AFTER]);
                    lhs.sub(&col_evals[COL_RS1_VAL])
                }
                _ => Scalar::zero(curve),
            }
        }
        selectors::INSN_W_ALU => {
            let two_32 = Scalar::from_u64(1u64 << 32, curve);
            let sign_ext_offset = two_64.sub(&two_32);
            let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
            match funct_val {
                selectors::FUNCT_ADDW => {
                    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_RS2_VAL]);
                    let upper = aux0.mul(&two_32);
                    let sign_term = aux1.mul(&sign_ext_offset);
                    let c1 = sum.sub(&upper).sub(&col_evals[COL_RD_VAL_AFTER]).add(&sign_term);
                    let c2 = aux1.mul(&aux1.sub(&one));
                    c1.add(&c2)
                }
                selectors::FUNCT_SUBW => {
                    let diff = col_evals[COL_RS1_VAL].sub(&col_evals[COL_RS2_VAL]);
                    let upper = aux0.mul(&two_32);
                    let sign_term = aux1.mul(&sign_ext_offset);
                    let c1 = diff.sub(&upper).sub(&col_evals[COL_RD_VAL_AFTER]).add(&sign_term);
                    let c2 = aux1.mul(&aux1.sub(&one));
                    c1.add(&c2)
                }
                selectors::FUNCT_ADDIW => {
                    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
                    let upper = aux0.mul(&two_32);
                    let sign_term = aux1.mul(&sign_ext_offset);
                    let c1 = sum.sub(&upper).sub(&col_evals[COL_RD_VAL_AFTER]).add(&sign_term);
                    let c2 = aux1.mul(&aux1.sub(&one));
                    c1.add(&c2)
                }
                _ => Scalar::zero(curve),
            }
        }
        _ => Scalar::zero(curve),
    }
}

/// Evaluate W-type ADDW constraint without instruction type gating.
///
/// ADDW: rs1 + rs2 - aux0*2^32 - rd + aux1*(2^64-2^32) = 0, aux1 binary
pub fn evaluate_alu_w_addw_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let sign_ext_offset = two_64.sub(&two_32);
    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_RS2_VAL]);
    let upper = aux0.mul(&two_32);
    let sign_term = aux1.mul(&sign_ext_offset);
    let c1 = sum.sub(&upper).sub(&col_evals[COL_RD_VAL_AFTER]).add(&sign_term);
    let c2 = aux1.mul(&aux1.sub(&one));
    c1.add(&c2)
}

/// Evaluate W-type SUBW constraint without instruction type gating.
///
/// SUBW: rs1 - rs2 - aux0*2^32 - rd + aux1*(2^64-2^32) = 0, aux1 binary
pub fn evaluate_alu_w_subw_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let sign_ext_offset = two_64.sub(&two_32);
    let diff = col_evals[COL_RS1_VAL].sub(&col_evals[COL_RS2_VAL]);
    let upper = aux0.mul(&two_32);
    let sign_term = aux1.mul(&sign_ext_offset);
    let c1 = diff.sub(&upper).sub(&col_evals[COL_RD_VAL_AFTER]).add(&sign_term);
    let c2 = aux1.mul(&aux1.sub(&one));
    c1.add(&c2)
}

/// Evaluate W-type ADDIW constraint without instruction type gating.
///
/// ADDIW: rs1 + imm - aux0*2^32 - rd + aux1*(2^64-2^32) = 0, aux1 binary
pub fn evaluate_alu_w_addiw_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let sign_ext_offset = two_64.sub(&two_32);
    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
    let upper = aux0.mul(&two_32);
    let sign_term = aux1.mul(&sign_ext_offset);
    let c1 = sum.sub(&upper).sub(&col_evals[COL_RD_VAL_AFTER]).add(&sign_term);
    let c2 = aux1.mul(&aux1.sub(&one));
    c1.add(&c2)
}

/// Evaluate MUL constraint without instruction type gating.
///
/// MUL: rs1 * rs2 = aux0 * 2^64 + rd (128-bit product, lower 64 bits in rd)
pub fn evaluate_mul_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
    let hi_term = aux0.mul(&two_64);
    let rhs = hi_term.add(&col_evals[COL_RD_VAL_AFTER]);
    product.sub(&rhs)
}

/// Evaluate DIV/DIVU constraint without instruction type gating.
///
/// DIV/DIVU: rd * rs2 + aux0 = rs1 (quotient*divisor + remainder = dividend)
pub fn evaluate_div_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let prod = col_evals[COL_RD_VAL_AFTER].mul(&col_evals[COL_RS2_VAL]);
    let lhs = prod.add(aux0);
    lhs.sub(&col_evals[COL_RS1_VAL])
}

/// Evaluate REM/REMU constraint without instruction type gating.
///
/// REM/REMU: aux0 * rs2 + rd = rs1 (quotient*divisor + remainder = dividend)
pub fn evaluate_rem_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let prod = aux0.mul(&col_evals[COL_RS2_VAL]);
    let lhs = prod.add(&col_evals[COL_RD_VAL_AFTER]);
    lhs.sub(&col_evals[COL_RS1_VAL])
}

/// Evaluate R-type ADD constraint without instruction type gating.
///
/// ADD: rs1 + rs2 = carry*2^64 + rd, carry in {0,1}
/// Returns (rs1+rs2 - carry*2^64 - rd) + carry*(carry-1)
pub fn evaluate_alu_r_add_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_RS2_VAL]);
    let carry_term = aux0.mul(&two_64);
    let rhs = carry_term.add(&col_evals[COL_RD_VAL_AFTER]);
    let c1 = sum.sub(&rhs);
    let carry_m1 = aux0.sub(&one);
    let c2 = aux0.mul(&carry_m1);
    c1.add(&c2)
}

/// Evaluate R-type ADD main equation only (no carry binary check).
///
/// ADD: rs1 + rs2 = carry*2^64 + rd
/// Returns rs1+rs2 - carry*2^64 - rd
pub fn evaluate_alu_r_add_main_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_RS2_VAL]);
    let carry_term = aux0.mul(&two_64);
    let rhs = carry_term.add(&col_evals[COL_RD_VAL_AFTER]);
    sum.sub(&rhs)
}

/// Evaluate R-type SUB constraint without instruction type gating.
///
/// SUB: rd + rs2 = rs1 + borrow*2^64, borrow in {0,1}
/// Returns (rd + rs2 - rs1 - borrow*2^64) + borrow*(borrow-1)
pub fn evaluate_alu_r_sub_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let lhs = col_evals[COL_RD_VAL_AFTER].add(&col_evals[COL_RS2_VAL]);
    let borrow_term = aux0.mul(&two_64);
    let rhs = col_evals[COL_RS1_VAL].add(&borrow_term);
    let c1 = lhs.sub(&rhs);
    let borrow_m1 = aux0.sub(&one);
    let c2 = aux0.mul(&borrow_m1);
    c1.add(&c2)
}

/// Evaluate R-type SUB main equation only (no borrow binary check).
///
/// SUB: rd + rs2 = rs1 + borrow*2^64
/// Returns rd + rs2 - rs1 - borrow*2^64
pub fn evaluate_alu_r_sub_main_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let lhs = col_evals[COL_RD_VAL_AFTER].add(&col_evals[COL_RS2_VAL]);
    let borrow_term = aux0.mul(&two_64);
    let rhs = col_evals[COL_RS1_VAL].add(&borrow_term);
    lhs.sub(&rhs)
}

/// Evaluate I-type ADDI constraint without instruction type gating.
///
/// ADDI: rs1 + immediate = carry*2^64 + rd, carry in {0,1}
/// Returns (rs1+imm - carry*2^64 - rd) + carry*(carry-1)
pub fn evaluate_alu_i_add_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
    let carry_term = aux0.mul(&two_64);
    let rhs = carry_term.add(&col_evals[COL_RD_VAL_AFTER]);
    let c1 = sum.sub(&rhs);
    let carry_m1 = aux0.sub(&one);
    let c2 = aux0.mul(&carry_m1);
    c1.add(&c2)
}

/// Evaluate I-type ADDI main equation only (no carry binary check).
///
/// ADDI: rs1 + immediate = carry*2^64 + rd
/// Returns rs1+imm - carry*2^64 - rd
pub fn evaluate_alu_i_add_main_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let sum = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
    let carry_term = aux0.mul(&two_64);
    let rhs = carry_term.add(&col_evals[COL_RD_VAL_AFTER]);
    sum.sub(&rhs)
}

/// Evaluate aux0 binary check (carry/borrow in {0,1}).
///
/// Returns aux0 * (aux0 - 1), which is zero iff aux0 ∈ {0, 1}.
/// Used as an independent constraint for R-ADD carry, R-SUB borrow, and I-ADDI carry.
pub fn evaluate_aux0_binary_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    aux0.mul(&aux0.sub(&one))
}

/// Evaluate aux0 binary checks for R-ADD carry, R-SUB borrow, and I-ADDI carry
/// on the full domain. Returns 3 vectors (one per constraint), each gated by the
/// appropriate selector.
///
/// These are separated from the main ALU equations to prevent cancellation attacks.
pub fn evaluate_aux0_binary_checks(columns: &[&Vec<Scalar>], num_rows: usize) -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
    let curve = columns[0][0].curve_type();
    let one = Scalar::one(curve);
    let mut r_add_binary = Vec::with_capacity(num_rows);
    let mut r_sub_binary = Vec::with_capacity(num_rows);
    let mut i_add_binary = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;
        let funct_val = columns[COL_FUNCT][i].to_u64() as u8;

        // aux0 * (aux0 - 1)
        let binary_check = columns[COL_AUX0][i].mul(&columns[COL_AUX0][i].sub(&one));

        let is_r_add = insn_type_val == selectors::INSN_R_ALU && funct_val == selectors::FUNCT_ADD;
        let is_r_sub = insn_type_val == selectors::INSN_R_ALU && funct_val == selectors::FUNCT_SUB;
        let is_i_add = insn_type_val == selectors::INSN_I_ALU && funct_val == selectors::FUNCT_ADDI;

        r_add_binary.push(if is_r_add { binary_check.clone() } else { Scalar::zero(curve) });
        r_sub_binary.push(if is_r_sub { binary_check.clone() } else { Scalar::zero(curve) });
        i_add_binary.push(if is_i_add { binary_check } else { Scalar::zero(curve) });
    }

    (r_add_binary, r_sub_binary, i_add_binary)
}

/// Convert a Scalar field element back to u64 (for small values like type/funct selectors).
pub fn scalar_to_u64(val: &Scalar) -> u64 {
    val.to_u64()
}

/// Evaluate R-type AND constraint without instruction type gating.
///
/// AND: rd = aux0, where aux0 = AND(rs1, rs2)
/// Returns rd_val_after - aux0
pub fn evaluate_r_and_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    col_evals[COL_RD_VAL_AFTER].sub(aux0)
}

/// Evaluate R-type OR constraint without instruction type gating.
///
/// OR(a,b) = a + b - AND(a,b), so rd = rs1 + rs2 - aux0
/// Returns rd_val_after - (rs1 + rs2 - aux0)
pub fn evaluate_r_or_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let expected = col_evals[COL_RS1_VAL].add(&col_evals[COL_RS2_VAL]).sub(aux0);
    col_evals[COL_RD_VAL_AFTER].sub(&expected)
}

/// Evaluate R-type XOR constraint without instruction type gating.
///
/// XOR(a,b) = a + b - 2*AND(a,b), so rd = rs1 + rs2 - 2*aux0
/// Returns rd_val_after - (rs1 + rs2 - 2*aux0)
pub fn evaluate_r_xor_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two = Scalar::from_u64(2, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let expected = col_evals[COL_RS1_VAL].add(&col_evals[COL_RS2_VAL]).sub(&two.mul(aux0));
    col_evals[COL_RD_VAL_AFTER].sub(&expected)
}

/// Evaluate I-type AND (ANDI) constraint without instruction type gating.
///
/// ANDI: rd = aux0, where aux0 = AND(rs1, immediate)
/// Returns rd_val_after - aux0
pub fn evaluate_i_and_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    col_evals[COL_RD_VAL_AFTER].sub(aux0)
}

/// Evaluate I-type OR (ORI) constraint without instruction type gating.
///
/// ORI: rd = rs1 + imm - AND(rs1, imm), so rd = rs1 + imm - aux0
/// Returns rd_val_after - (rs1 + imm - aux0)
pub fn evaluate_i_or_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let expected = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]).sub(aux0);
    col_evals[COL_RD_VAL_AFTER].sub(&expected)
}

/// Evaluate I-type XOR (XORI) constraint without instruction type gating.
///
/// XORI: rd = rs1 + imm - 2*AND(rs1, imm), so rd = rs1 + imm - 2*aux0
/// Returns rd_val_after - (rs1 + imm - 2*aux0)
pub fn evaluate_i_xor_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two = Scalar::from_u64(2, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let expected = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]).sub(&two.mul(aux0));
    col_evals[COL_RD_VAL_AFTER].sub(&expected)
}

/// Evaluate SLL constraint without instruction type gating.
///
/// SLL: rs1 * aux0 = aux1 * 2^64 + rd
/// aux0 = 2^k (power of 2), aux1 = overflow
/// Returns rs1 * aux0 - aux1 * 2^64 - rd
pub fn evaluate_sll_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let product = col_evals[COL_RS1_VAL].mul(aux0);
    let hi_term = aux1.mul(&two_64);
    let rhs = hi_term.add(&col_evals[COL_RD_VAL_AFTER]);
    product.sub(&rhs)
}

/// Evaluate SRL constraint without instruction type gating.
///
/// SRL: rd * aux0 + aux1 = rs1
/// aux0 = 2^k, aux1 = remainder
/// Returns rd * aux0 + aux1 - rs1
pub fn evaluate_srl_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let prod = col_evals[COL_RD_VAL_AFTER].mul(aux0);
    let lhs = prod.add(aux1);
    lhs.sub(&col_evals[COL_RS1_VAL])
}

/// Evaluate SRA constraint without instruction type gating.
///
/// SRA: rd * aux0 + aux1 - rs1 - aux2 * 2^64 = 0
/// aux0 = 2^k, aux1 = remainder, aux2 = sign_bit * (2^k - 1)
/// Returns rd * aux0 + aux1 - rs1 - aux2 * 2^64
pub fn evaluate_sra_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };
    let prod = col_evals[COL_RD_VAL_AFTER].mul(aux0);
    let lhs = prod.add(aux1);
    let overflow_term = aux2.mul(&two_64);
    let rhs = col_evals[COL_RS1_VAL].add(&overflow_term);
    lhs.sub(&rhs)
}

/// Evaluate W-SLL constraint without instruction type gating.
///
/// W-SLL: rs1_low * aux0 = aux1 * 2^32 + rd_low
/// aux0 = 2^k, aux1 = overflow (32-bit context)
/// Returns rs1 * aux0 - aux1 * 2^32 - rd
pub fn evaluate_w_sll_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let product = col_evals[COL_RS1_VAL].mul(aux0);
    let hi_term = aux1.mul(&two_32);
    let rhs = hi_term.add(&col_evals[COL_RD_VAL_AFTER]);
    product.sub(&rhs)
}

/// Evaluate W-SRL constraint without instruction type gating.
///
/// W-SRL: rd * aux0 + aux1 = rs1_low
/// Returns rd * aux0 + aux1 - rs1
pub fn evaluate_w_srl_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let prod = col_evals[COL_RD_VAL_AFTER].mul(aux0);
    let lhs = prod.add(aux1);
    lhs.sub(&col_evals[COL_RS1_VAL])
}

/// Evaluate W-SRA constraint without instruction type gating.
///
/// W-SRA: rd * aux0 + aux1 - rs1_low - aux2 * 2^32 = 0
/// Returns rd * aux0 + aux1 - rs1 - aux2 * 2^32
pub fn evaluate_w_sra_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };
    let prod = col_evals[COL_RD_VAL_AFTER].mul(aux0);
    let lhs = prod.add(aux1);
    let overflow_term = aux2.mul(&two_32);
    let rhs = col_evals[COL_RS1_VAL].add(&overflow_term);
    lhs.sub(&rhs)
}

/// Evaluate compare constraint (SLT/SLTU/SLTI/SLTIU) without instruction type gating.
///
/// Uses the signed comparison formula that unifies SLT and SLTU:
///   signed_lt = aux1*(1-aux2) + (1-aux1-aux2+2*aux1*aux2)*aux0
/// For SLTU: aux1=0, aux2=0 → signed_lt = aux0 (unsigned comparison)
/// For SLT: aux1=sign(rs1), aux2=sign(rs2) → signed_lt = full signed comparison
///
/// Returns: (rd - signed_lt) + aux0*(aux0-1) + aux1*(aux1-1) + aux2*(aux2-1)
pub fn evaluate_compare_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };

    // signed_lt = aux1*(1-aux2) + (1-aux1-aux2+2*aux1*aux2)*aux0
    let one_m_aux2 = one.sub(aux2);
    let term1 = aux1.mul(&one_m_aux2);
    let aux1_aux2 = aux1.mul(aux2);
    let same_sign = one.sub(aux1).sub(aux2).add(&two.mul(&aux1_aux2));
    let term2 = same_sign.mul(aux0);
    let signed_lt = term1.add(&term2);

    // body = (rd - signed_lt) + aux0*(aux0-1) + aux1*(aux1-1) + aux2*(aux2-1)
    let rd_check = col_evals[COL_RD_VAL_AFTER].sub(&signed_lt);
    let aux0_binary = aux0.mul(&aux0.sub(&one));
    let aux1_binary = aux1.mul(&aux1.sub(&one));
    let aux2_binary = aux2.mul(&aux2.sub(&one));

    rd_check.add(&aux0_binary).add(&aux1_binary).add(&aux2_binary)
}

/// Evaluate CSRRS/CSRRSI constraint without instruction type gating.
///
/// CSRRS: new_csr = old | source, where old = rd_val_after, source = rs1_val
/// aux0 = old | source = old + source - AND(old, source)
/// aux1 = AND(old, source)
/// Constraint: aux0 - rd_val_after - rs1_val + aux1 = 0
/// (For CSRRSI, rs1_val holds the uimm value in the trace)
pub fn evaluate_csrrs_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    // aux0 = old + source - AND(old, source) = rd_val_after + rs1_val - aux1
    // => aux0 - rd_val_after - rs1_val + aux1 = 0
    aux0.sub(&col_evals[COL_RD_VAL_AFTER]).sub(&col_evals[COL_RS1_VAL]).add(aux1)
}

/// Evaluate CSRRC/CSRRCI constraint without instruction type gating.
///
/// CSRRC: new_csr = old & ~source
/// old = rd_val_after, aux1 = AND(old, source), aux0 = old & ~source = old - aux1
/// Constraint: aux0 - rd_val_after + aux1 = 0
pub fn evaluate_csrrc_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    // aux0 = old - AND(old, source) = rd_val_after - aux1
    // => aux0 - rd_val_after + aux1 = 0
    aux0.sub(&col_evals[COL_RD_VAL_AFTER]).add(aux1)
}

/// Evaluate CSRRW/CSRRWI constraint without instruction type gating.
///
/// CSRRW: rd = old CSR value, new CSR = rs1_val (stored in aux0)
/// Constraint: aux0 - rs1_val = 0
/// Note: For CSRRWI (funct=3), rs1_val in trace holds the register value,
/// but the uimm is in the immediate field. Since both share a selector,
/// we verify aux0 = rs1_val which works for CSRRW.
/// CSRRWI verification relies on oracle data binding for the immediate source.
pub fn evaluate_csrrw_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    // aux0 = new CSR value = rs1_val
    aux0.sub(&col_evals[COL_RS1_VAL])
}

/// Evaluate LR (load reserved) constraint without instruction type gating.
///
/// LR: rd_val_after = mem_val (loaded value matches captured memory value).
/// Constraint: rd_val_after - mem_val = 0
pub fn evaluate_lr_raw(col_evals: &[Scalar]) -> Scalar {
    col_evals[COL_RD_VAL_AFTER].sub(&col_evals[COL_MEM_VAL])
}

/// Evaluate SC (store conditional) constraint without instruction type gating.
///
/// SC: aux0 = 1 if reservation valid (rd=0), aux0 = 0 if invalid.
/// Constraints combined: aux0*(aux0-1) + aux0*rd_val_after = 0
/// - aux0*(aux0-1): ensures aux0 is binary
/// - aux0*rd_val_after: if reservation valid (aux0=1), rd must be 0
pub fn evaluate_sc_raw(col_evals: &[Scalar]) -> Scalar {
    let one = Scalar::one(col_evals[0].curve_type());
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let binary = aux0.mul(&aux0.sub(&one));
    let valid_check = aux0.mul(&col_evals[COL_RD_VAL_AFTER]);
    binary.add(&valid_check)
}

/// Evaluate MULHU constraint without instruction type gating.
///
/// MULHU: rs1 * rs2 = rd * 2^64 + aux0
/// The 128-bit unsigned product has high 64 bits in rd and low 64 bits in aux0.
pub fn evaluate_mulhu_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
    let hi_term = col_evals[COL_RD_VAL_AFTER].mul(&two_64);
    let rhs = hi_term.add(aux0);
    product.sub(&rhs)
}

/// Evaluate shift constraint on domain for a specific instruction type/funct.
/// Returns a vector of constraint evaluations, zero for non-matching rows.
pub fn evaluate_shift_on_domain(
    columns: &[&Vec<Scalar>],
    num_rows: usize,
    target_insn_type: u8,
    target_funct: u8,
) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let mut result = vec![zero.clone(); num_rows];

    let is_w_type = target_insn_type == selectors::INSN_W_ALU;
    let is_sll = target_funct == selectors::FUNCT_SLL || target_funct == selectors::FUNCT_SLLI
        || target_funct == selectors::FUNCT_SLLW || target_funct == selectors::FUNCT_SLLIW;
    let is_sra = target_funct == selectors::FUNCT_SRA || target_funct == selectors::FUNCT_SRAI
        || target_funct == selectors::FUNCT_SRAW || target_funct == selectors::FUNCT_SRAIW;

    let modulus = if is_w_type { &two_32 } else { &two_64 };

    for i in 0..num_rows {
        let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;
        let funct_val = columns[COL_FUNCT][i].to_u64() as u8;

        if insn_type_val != target_insn_type {
            continue;
        }

        // For W-type, SLLIW/SRLIW/SRAIW share selectors with SLLW/SRLW/SRAW
        let matches = match target_insn_type {
            selectors::INSN_W_ALU => match target_funct {
                selectors::FUNCT_SLLW => funct_val == selectors::FUNCT_SLLW || funct_val == selectors::FUNCT_SLLIW,
                selectors::FUNCT_SRLW => funct_val == selectors::FUNCT_SRLW || funct_val == selectors::FUNCT_SRLIW,
                selectors::FUNCT_SRAW => funct_val == selectors::FUNCT_SRAW || funct_val == selectors::FUNCT_SRAIW,
                _ => false,
            },
            _ => funct_val == target_funct,
        };

        if !matches {
            continue;
        }

        if is_sll {
            // rs1 * aux0 - aux1 * modulus - rd = 0
            let product = columns[COL_RS1_VAL][i].mul(&columns[COL_AUX0][i]);
            let hi_term = columns[COL_AUX1][i].mul(modulus);
            let rhs = hi_term.add(&columns[COL_RD_VAL_AFTER][i]);
            result[i] = product.sub(&rhs);
        } else if is_sra {
            // rd * aux0 + aux1 - rs1 - aux2 * modulus = 0
            let prod = columns[COL_RD_VAL_AFTER][i].mul(&columns[COL_AUX0][i]);
            let lhs = prod.add(&columns[COL_AUX1][i]);
            let overflow_term = columns[COL_AUX2][i].mul(modulus);
            let rhs = columns[COL_RS1_VAL][i].add(&overflow_term);
            result[i] = lhs.sub(&rhs);
        } else {
            // SRL: rd * aux0 + aux1 - rs1 = 0
            let prod = columns[COL_RD_VAL_AFTER][i].mul(&columns[COL_AUX0][i]);
            let lhs = prod.add(&columns[COL_AUX1][i]);
            result[i] = lhs.sub(&columns[COL_RS1_VAL][i]);
        }
    }

    result
}

/// Legacy compatibility: Convert a BIG to u64.
pub fn big_to_u64(val: &bls48581::bls48581::big::BIG) -> u64 {
    use bls48581::bls48581::big;
    let mut buf = [0u8; big::MODBYTES];
    val.tobytes(&mut buf);
    let start = big::MODBYTES - 8;
    u64::from_be_bytes([
        buf[start], buf[start + 1], buf[start + 2], buf[start + 3],
        buf[start + 4], buf[start + 5], buf[start + 6], buf[start + 7],
    ])
}

/// CSRRWI I-variant: new CSR value = immediate (zero-extended uimm).
/// Constraint: aux0 - immediate = 0
pub fn evaluate_csrrw_i_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    aux0.sub(&col_evals[COL_IMMEDIATE])
}

/// CSRRSI I-variant: aux0 = old | source = old + source - AND(old, source).
/// Constraint: aux0 - rd_val_after - immediate + aux1 = 0
pub fn evaluate_csrrs_i_raw(col_evals: &[Scalar]) -> Scalar {
    let zero = Scalar::zero(col_evals[0].curve_type());
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    aux0.sub(&col_evals[COL_RD_VAL_AFTER]).sub(&col_evals[COL_IMMEDIATE]).add(aux1)
}

/// MULH: signed×signed high 64 bits via sign correction.
/// unsigned_product = rs1*rs2, signed correction subtracts sign_a*2^64*b + sign_b*2^64*a - sign_a*sign_b*2^128
/// Constraint: rs1*rs2 - aux1*2^64*rs2 - aux2*2^64*rs1 + aux1*aux2*2^128 - rd*2^64 - aux0
///           + aux1*(aux1-1) + aux2*(aux2-1) = 0
pub fn evaluate_mulh_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_128 = two_64.mul(&two_64);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };

    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
    let sign_a_rs2 = aux1.mul(&two_64).mul(&col_evals[COL_RS2_VAL]);
    let sign_b_rs1 = aux2.mul(&two_64).mul(&col_evals[COL_RS1_VAL]);
    let sign_ab = aux1.mul(aux2).mul(&two_128);
    let hi_term = col_evals[COL_RD_VAL_AFTER].mul(&two_64);

    let main = product.sub(&sign_a_rs2).sub(&sign_b_rs1).add(&sign_ab).sub(&hi_term).sub(aux0);
    let a1_bin = aux1.mul(&aux1.sub(&one));
    let a2_bin = aux2.mul(&aux2.sub(&one));

    main.add(&a1_bin).add(&a2_bin)
}

/// MULHSU: signed×unsigned high 64 bits. Only rs1 signed.
/// Constraint: rs1*rs2 - aux1*2^64*rs2 - rd*2^64 - aux0 + aux1*(aux1-1) = 0
pub fn evaluate_mulhsu_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };

    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
    let sign_a_rs2 = aux1.mul(&two_64).mul(&col_evals[COL_RS2_VAL]);
    let hi_term = col_evals[COL_RD_VAL_AFTER].mul(&two_64);

    let main = product.sub(&sign_a_rs2).sub(&hi_term).sub(aux0);
    let a1_bin = aux1.mul(&aux1.sub(&one));

    main.add(&a1_bin)
}

/// MULW: W-type multiplication with sign extension.
/// Constraint: rs1*rs2 - aux0*2^32 - rd + aux1*(2^64-2^32) + aux1*(aux1-1) = 0
pub fn evaluate_mulw_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let sign_ext_offset = two_64.sub(&two_32);
    let product = col_evals[COL_RS1_VAL].mul(&col_evals[COL_RS2_VAL]);
    let upper = aux0.mul(&two_32);
    let sign_term = aux1.mul(&sign_ext_offset);
    let c1 = product.sub(&upper).sub(&col_evals[COL_RD_VAL_AFTER]).add(&sign_term);
    let c2 = aux1.mul(&aux1.sub(&one));
    c1.add(&c2)
}

/// DIVW/DIVUW: quotient*divisor + remainder = dividend (mod 2^32).
/// Constraint: rd*rs2 + aux0 - rs1 - aux1*2^32 = 0
pub fn evaluate_divw_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let prod = col_evals[COL_RD_VAL_AFTER].mul(&col_evals[COL_RS2_VAL]);
    let correction = aux1.mul(&two_32);
    prod.add(aux0).sub(&col_evals[COL_RS1_VAL]).sub(&correction)
}

/// REMW/REMUW: quotient*divisor + remainder = dividend (mod 2^32).
/// Constraint: aux0*rs2 + rd - rs1 - aux1*2^32 = 0
pub fn evaluate_remw_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let prod = aux0.mul(&col_evals[COL_RS2_VAL]);
    let correction = aux1.mul(&two_32);
    prod.add(&col_evals[COL_RD_VAL_AFTER]).sub(&col_evals[COL_RS1_VAL]).sub(&correction)
}

/// AMO_SWAP: mem_val = rs2 (D) or sign_extend_32(rs2) (W).
/// Constraint: rs2 - aux0*2^32 - mem_val + aux1*(2^64-2^32) + aux1*(aux1-1) = 0
pub fn evaluate_amo_swap_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let sign_ext_offset = two_64.sub(&two_32);
    let c1 = col_evals[COL_RS2_VAL].sub(&aux0.mul(&two_32)).sub(&col_evals[COL_MEM_VAL]).add(&aux1.mul(&sign_ext_offset));
    let c2 = aux1.mul(&aux1.sub(&one));
    c1.add(&c2)
}

/// AMO_ADD: mem_val = old + rs2. Unified W/D via aux2 flag.
/// D (aux2=0): rd+rs2-mem_val-aux0*2^64=0
/// W (aux2=1): rd+rs2-aux0*2^32-mem_val+aux1*(2^64-2^32)=0
pub fn evaluate_amo_add_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_64 = two_pow_64(curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };

    let sign_ext_offset = two_64.sub(&two_32);
    let one_m_aux2 = one.sub(aux2);
    // aux0 * (aux2*2^32 + (1-aux2)*2^64)
    let modulus = aux2.mul(&two_32).add(&one_m_aux2.mul(&two_64));
    let correction_term = aux0.mul(&modulus);

    let sum = col_evals[COL_RD_VAL_AFTER].add(&col_evals[COL_RS2_VAL]);
    let c1 = sum.sub(&col_evals[COL_MEM_VAL]).sub(&correction_term).add(&aux1.mul(aux2).mul(&sign_ext_offset));

    let a1_bin = aux1.mul(&aux1.sub(&one));
    let a2_bin = aux2.mul(&aux2.sub(&one));
    // For D: aux0 is carry (binary). For W: aux0 is quotient (not necessarily binary).
    // Only enforce aux0 binary when D: (1-aux2)*aux0*(aux0-1)
    let a0_d_bin = one_m_aux2.mul(&aux0.mul(&aux0.sub(&one)));

    c1.add(&a1_bin).add(&a2_bin).add(&a0_d_bin)
}

/// AMO_AND: mem_val = old & rs2. Uses AND+OR=a+b identity.
/// aux0 = OR(old, rs2). Constraint: mem_val + aux0 - rd - rs2 - aux1*2^32 = 0
pub fn evaluate_amo_and_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    col_evals[COL_MEM_VAL].add(aux0)
        .sub(&col_evals[COL_RD_VAL_AFTER]).sub(&col_evals[COL_RS2_VAL])
        .sub(&aux1.mul(&two_32))
}

/// AMO_OR: mem_val = old | rs2. Uses AND+OR=a+b identity.
/// aux0 = AND(old, rs2). Constraint: mem_val + aux0 - rd - rs2 - aux1*2^32 = 0
pub fn evaluate_amo_or_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    col_evals[COL_MEM_VAL].add(aux0)
        .sub(&col_evals[COL_RD_VAL_AFTER]).sub(&col_evals[COL_RS2_VAL])
        .sub(&aux1.mul(&two_32))
}

/// AMO_XOR: mem_val = old ^ rs2. Uses XOR=a+b-2*AND identity.
/// aux0 = AND(old, rs2). Constraint: mem_val + 2*aux0 - rd - rs2 - aux1*2^32 = 0
pub fn evaluate_amo_xor_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two = Scalar::from_u64(2, curve);
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    col_evals[COL_MEM_VAL].add(&two.mul(aux0))
        .sub(&col_evals[COL_RD_VAL_AFTER]).sub(&col_evals[COL_RS2_VAL])
        .sub(&aux1.mul(&two_32))
}

/// AMO_MINU: mem_val = min(old, rs2) unsigned.
/// aux0 = borrow (old < rs2). Constraint: mem_val - rs2 - aux0*(rd-rs2) - aux1*2^32 + aux0*(aux0-1) = 0
pub fn evaluate_amo_minu_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let rd_m_rs2 = col_evals[COL_RD_VAL_AFTER].sub(&col_evals[COL_RS2_VAL]);
    let selection = aux0.mul(&rd_m_rs2);
    let correction = aux1.mul(&two_32);
    let c1 = col_evals[COL_MEM_VAL].sub(&col_evals[COL_RS2_VAL]).sub(&selection).sub(&correction);
    let c2 = aux0.mul(&aux0.sub(&one));
    c1.add(&c2)
}

/// AMO_MAXU: mem_val = max(old, rs2) unsigned.
/// aux0 = borrow (old < rs2). Constraint: mem_val - rd + aux0*(rd-rs2) - aux1*2^32 + aux0*(aux0-1) = 0
pub fn evaluate_amo_maxu_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let two_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let rd_m_rs2 = col_evals[COL_RD_VAL_AFTER].sub(&col_evals[COL_RS2_VAL]);
    let selection = aux0.mul(&rd_m_rs2);
    let correction = aux1.mul(&two_32);
    let c1 = col_evals[COL_MEM_VAL].sub(&col_evals[COL_RD_VAL_AFTER]).add(&selection).sub(&correction);
    let c2 = aux0.mul(&aux0.sub(&one));
    c1.add(&c2)
}

/// AMO_MINS: mem_val = min(old, rs2) signed.
/// Uses signed comparison: signed_lt = aux1*(1-aux2) + (1-aux1-aux2+2*aux1*aux2)*aux0
/// Constraint: mem_val - rs2 - signed_lt*(rd-rs2) + binary checks = 0
pub fn evaluate_amo_mins_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };

    // signed_lt = aux1*(1-aux2) + (1-aux1-aux2+2*aux1*aux2)*aux0
    let one_m_aux2 = one.sub(aux2);
    let term1 = aux1.mul(&one_m_aux2);
    let aux1_aux2 = aux1.mul(aux2);
    let same_sign = one.sub(aux1).sub(aux2).add(&two.mul(&aux1_aux2));
    let term2 = same_sign.mul(aux0);
    let signed_lt = term1.add(&term2);

    let rd_m_rs2 = col_evals[COL_RD_VAL_AFTER].sub(&col_evals[COL_RS2_VAL]);
    let selection = signed_lt.mul(&rd_m_rs2);
    let c1 = col_evals[COL_MEM_VAL].sub(&col_evals[COL_RS2_VAL]).sub(&selection);

    let a0_bin = aux0.mul(&aux0.sub(&one));
    let a1_bin = aux1.mul(&aux1.sub(&one));
    let a2_bin = aux2.mul(&aux2.sub(&one));

    c1.add(&a0_bin).add(&a1_bin).add(&a2_bin)
}

/// AMO_MAXS: mem_val = max(old, rs2) signed.
/// Constraint: mem_val - rd + signed_lt*(rd-rs2) + binary checks = 0
pub fn evaluate_amo_maxs_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };

    let one_m_aux2 = one.sub(aux2);
    let term1 = aux1.mul(&one_m_aux2);
    let aux1_aux2 = aux1.mul(aux2);
    let same_sign = one.sub(aux1).sub(aux2).add(&two.mul(&aux1_aux2));
    let term2 = same_sign.mul(aux0);
    let signed_lt = term1.add(&term2);

    let rd_m_rs2 = col_evals[COL_RD_VAL_AFTER].sub(&col_evals[COL_RS2_VAL]);
    let selection = signed_lt.mul(&rd_m_rs2);
    let c1 = col_evals[COL_MEM_VAL].sub(&col_evals[COL_RD_VAL_AFTER]).add(&selection);

    let a0_bin = aux0.mul(&aux0.sub(&one));
    let a1_bin = aux1.mul(&aux1.sub(&one));
    let a2_bin = aux2.mul(&aux2.sub(&one));

    c1.add(&a0_bin).add(&a1_bin).add(&a2_bin)
}
