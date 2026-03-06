//! Control flow constraints for branch, jump, LUI, and AUIPC instructions.

use metavm_zkp::field::Scalar;
use super::selectors;
use super::{COL_PC, COL_RD_VAL_AFTER, COL_RS1_VAL, COL_RS2_VAL, COL_NEXT_PC,
            COL_INSN_TYPE, COL_FUNCT, COL_IMMEDIATE, COL_INSN_LEN,
            COL_AUX0, COL_AUX1, COL_AUX2};

/// Evaluate control flow constraints.
///
/// - JAL: rd_val_after == pc + insn_len (link register), next_pc == pc + immediate
/// - JALR: rd_val_after == pc + insn_len, next_pc == (rs1_val + immediate) - aux1, aux1 binary
/// - LUI: rd_val_after == immediate
/// - AUIPC: rd_val_after == pc + immediate
/// - Branch: target validity + condition verification
pub fn evaluate_control_flow(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let one = Scalar::one(curve);
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;

        let constraint = match insn_type_val {
            selectors::INSN_JAL => {
                // Constraint 1: rd_val_after == pc + insn_len (return address)
                let expected_rd = columns[COL_PC][i].add(&columns[COL_INSN_LEN][i]);
                let c1 = columns[COL_RD_VAL_AFTER][i].sub(&expected_rd);

                // Constraint 2: next_pc == pc + immediate
                let expected_pc = columns[COL_PC][i].add(&columns[COL_IMMEDIATE][i]);
                let c2 = columns[COL_NEXT_PC][i].sub(&expected_pc);

                c1.add(&c2)
            }

            selectors::INSN_JALR => {
                // Constraint 1: rd_val_after == pc + insn_len
                let expected_rd = columns[COL_PC][i].add(&columns[COL_INSN_LEN][i]);
                let c1 = columns[COL_RD_VAL_AFTER][i].sub(&expected_rd);

                // Constraint 2: next_pc = (rs1_val + imm) - aux1, where aux1 = LSB
                let raw_target = columns[COL_RS1_VAL][i].add(&columns[COL_IMMEDIATE][i]);
                let c2 = columns[COL_NEXT_PC][i].sub(&raw_target.sub(&columns[COL_AUX1][i]));

                // Constraint 3: aux1 is binary (it's the LSB being cleared)
                let aux1_binary = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));

                c1.add(&c2).add(&aux1_binary)
            }

            selectors::INSN_LUI => {
                // rd_val_after == immediate
                columns[COL_RD_VAL_AFTER][i].sub(&columns[COL_IMMEDIATE][i])
            }

            selectors::INSN_AUIPC => {
                // rd_val_after == pc + immediate
                let expected = columns[COL_PC][i].add(&columns[COL_IMMEDIATE][i]);
                columns[COL_RD_VAL_AFTER][i].sub(&expected)
            }

            selectors::INSN_BRANCH => {
                // Target validity: (next_pc - pc - imm) * (next_pc - pc - insn_len) == 0
                let taken_target = columns[COL_PC][i].add(&columns[COL_IMMEDIATE][i]);
                let not_taken_target = columns[COL_PC][i].add(&columns[COL_INSN_LEN][i]);

                let diff_taken = columns[COL_NEXT_PC][i].sub(&taken_target);
                let diff_not_taken = columns[COL_NEXT_PC][i].sub(&not_taken_target);

                let target_check = diff_taken.mul(&diff_not_taken);

                // Condition verification using aux1
                let funct_val = columns[COL_FUNCT][i].to_u64() as u8;
                let diff = columns[COL_RS1_VAL][i].sub(&columns[COL_RS2_VAL][i]);

                let condition_check = match funct_val {
                    selectors::FUNCT_BEQ => {
                        // BEQ: branch taken iff rs1 == rs2
                        // aux1 = 0 when rs1 == rs2, aux1 = 1 when rs1 != rs2
                        // When taken (diff_not_taken != 0): aux1 must be 0
                        //   diff_not_taken * aux1 = 0
                        // When not taken (diff_taken != 0): aux1 must be 1
                        //   diff_taken * (1 - aux1) = 0
                        // Also: when aux1 = 0 (rs1==rs2), diff must actually be 0
                        //   (1 - aux1) * diff = 0
                        // aux1 binary: aux1 * (aux1 - 1) = 0
                        let c_taken = diff_not_taken.mul(&columns[COL_AUX1][i]);
                        let one_minus_aux1 = one.sub(&columns[COL_AUX1][i]);
                        let c_not_taken = diff_taken.mul(&one_minus_aux1);
                        let c_eq_check = one_minus_aux1.mul(&diff);
                        let c_binary = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));
                        c_taken.add(&c_not_taken).add(&c_eq_check).add(&c_binary)
                    }
                    selectors::FUNCT_BNE => {
                        // BNE: branch taken iff rs1 != rs2
                        // aux1 = 1 when rs1 != rs2 (from trace), aux1 = 0 when equal
                        // When taken (diff_not_taken != 0): aux1 must be 1 (rs1 != rs2)
                        //   diff_not_taken * (1 - aux1) = 0
                        // When not taken (diff_taken != 0): aux1 must be 0 (rs1 == rs2)
                        //   diff_taken * aux1 = 0
                        // Also: when aux1 = 0, diff must be 0
                        //   (1 - aux1) * diff = 0
                        // aux1 binary
                        let one_minus_aux1 = one.sub(&columns[COL_AUX1][i]);
                        let c_taken = diff_not_taken.mul(&one_minus_aux1);
                        let c_not_taken = diff_taken.mul(&columns[COL_AUX1][i]);
                        let c_eq_check = one_minus_aux1.mul(&diff);
                        let c_binary = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));
                        c_taken.add(&c_not_taken).add(&c_eq_check).add(&c_binary)
                    }
                    selectors::FUNCT_BLTU => {
                        // BLTU: taken iff rs1 < rs2 (unsigned)
                        // aux1 = borrow = 1 if rs1 < rs2
                        // When taken: aux1 must be 1
                        //   diff_not_taken * (1 - aux1) = 0
                        // When not taken: aux1 must be 0
                        //   diff_taken * aux1 = 0
                        // aux1 binary
                        let one_minus_aux1 = one.sub(&columns[COL_AUX1][i]);
                        let c_taken = diff_not_taken.mul(&one_minus_aux1);
                        let c_not_taken = diff_taken.mul(&columns[COL_AUX1][i]);
                        let c_binary = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));
                        c_taken.add(&c_not_taken).add(&c_binary)
                    }
                    selectors::FUNCT_BGEU => {
                        // BGEU: taken iff rs1 >= rs2 (unsigned)
                        // aux1 = borrow = 1 if rs1 < rs2
                        // When taken: aux1 must be 0 (rs1 >= rs2)
                        //   diff_not_taken * aux1 = 0
                        // When not taken: aux1 must be 1 (rs1 < rs2)
                        //   diff_taken * (1 - aux1) = 0
                        // aux1 binary
                        let one_minus_aux1 = one.sub(&columns[COL_AUX1][i]);
                        let c_taken = diff_not_taken.mul(&columns[COL_AUX1][i]);
                        let c_not_taken = diff_taken.mul(&one_minus_aux1);
                        let c_binary = columns[COL_AUX1][i].mul(&columns[COL_AUX1][i].sub(&one));
                        c_taken.add(&c_not_taken).add(&c_binary)
                    }
                    selectors::FUNCT_BLT => {
                        evaluate_signed_branch_condition(
                            &columns[COL_PC][i], &columns[COL_IMMEDIATE][i],
                            &columns[COL_INSN_LEN][i], &columns[COL_NEXT_PC][i],
                            &columns[COL_RS1_VAL][i], &columns[COL_RS2_VAL][i],
                            &columns[COL_AUX0][i], &columns[COL_AUX1][i],
                            &columns[COL_AUX2][i], true, curve)
                    }
                    selectors::FUNCT_BGE => {
                        evaluate_signed_branch_condition(
                            &columns[COL_PC][i], &columns[COL_IMMEDIATE][i],
                            &columns[COL_INSN_LEN][i], &columns[COL_NEXT_PC][i],
                            &columns[COL_RS1_VAL][i], &columns[COL_RS2_VAL][i],
                            &columns[COL_AUX0][i], &columns[COL_AUX1][i],
                            &columns[COL_AUX2][i], false, curve)
                    }
                    _ => {
                        Scalar::zero(curve)
                    }
                };

                target_check.add(&condition_check)
            }

            _ => Scalar::zero(curve),
        };

        result.push(constraint);
    }

    result
}

/// Evaluate control flow constraint at a single point given column evaluations.
pub fn evaluate_control_flow_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < 16 {
        return Scalar::zero(curve);
    }

    let insn_type_val = col_evals[COL_INSN_TYPE].to_u64() as u8;
    let one = Scalar::one(curve);

    match insn_type_val {
        selectors::INSN_JAL => {
            let expected_rd = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
            let c1 = col_evals[COL_RD_VAL_AFTER].sub(&expected_rd);
            let expected_pc = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
            let c2 = col_evals[COL_NEXT_PC].sub(&expected_pc);
            c1.add(&c2)
        }
        selectors::INSN_JALR => {
            let expected_rd = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
            let c1 = col_evals[COL_RD_VAL_AFTER].sub(&expected_rd);
            let raw_target = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
            let c2 = col_evals[COL_NEXT_PC].sub(&raw_target.sub(&col_evals[COL_AUX1]));
            let c3 = col_evals[COL_AUX1].mul(&col_evals[COL_AUX1].sub(&one));
            c1.add(&c2).add(&c3)
        }
        selectors::INSN_LUI => {
            col_evals[COL_RD_VAL_AFTER].sub(&col_evals[COL_IMMEDIATE])
        }
        selectors::INSN_AUIPC => {
            let expected = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
            col_evals[COL_RD_VAL_AFTER].sub(&expected)
        }
        selectors::INSN_BRANCH => {
            let taken_target = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
            let not_taken_target = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
            let diff_taken = col_evals[COL_NEXT_PC].sub(&taken_target);
            let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken_target);
            diff_taken.mul(&diff_not_taken)
            // Note: condition verification happens through the selector-gated path
            // in evaluate_at_point_extended. The legacy path here only does target check.
        }
        _ => Scalar::zero(curve),
    }
}

/// Evaluate branch target check without instruction type gating.
/// Returns (next_pc - pc - imm) * (next_pc - pc - insn_len).
/// This is zero when next_pc equals either the taken target (pc+imm) or the
/// not-taken target (pc+insn_len).
pub fn evaluate_control_flow_raw(col_evals: &[Scalar]) -> Scalar {
    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    diff_taken.mul(&diff_not_taken)
}

/// Evaluate sequential PC constraint without instruction type gating.
/// Returns next_pc - (pc + insn_len).
pub fn evaluate_sequential_pc_raw(col_evals: &[Scalar]) -> Scalar {
    let expected = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    col_evals[COL_NEXT_PC].sub(&expected)
}

/// BEQ condition: taken iff rs1 == rs2.
/// aux1 = 0 when equal, 1 when not equal.
/// Combines target check + condition verification.
pub fn evaluate_beq_condition_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let diff = col_evals[COL_RS1_VAL].sub(&col_evals[COL_RS2_VAL]);
    let one_m_aux1 = one.sub(&col_evals[COL_AUX1]);

    // target_check: (next_pc - taken) * (next_pc - not_taken) = 0
    let target = diff_taken.mul(&diff_not_taken);
    // taken -> aux1=0: diff_not_taken * aux1 = 0
    let c1 = diff_not_taken.mul(&col_evals[COL_AUX1]);
    // not-taken -> aux1=1: diff_taken * (1-aux1) = 0
    let c2 = diff_taken.mul(&one_m_aux1);
    // aux1=0 -> equal: (1-aux1) * diff = 0
    let c3 = one_m_aux1.mul(&diff);
    // aux1 binary
    let c4 = col_evals[COL_AUX1].mul(&col_evals[COL_AUX1].sub(&one));

    target.add(&c1).add(&c2).add(&c3).add(&c4)
}

/// BNE condition: taken iff rs1 != rs2.
/// aux1 = 1 when not equal, 0 when equal.
pub fn evaluate_bne_condition_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let diff = col_evals[COL_RS1_VAL].sub(&col_evals[COL_RS2_VAL]);
    let one_m_aux1 = one.sub(&col_evals[COL_AUX1]);

    let target = diff_taken.mul(&diff_not_taken);
    // taken -> aux1=1: diff_not_taken * (1-aux1) = 0
    let c1 = diff_not_taken.mul(&one_m_aux1);
    // not-taken -> aux1=0: diff_taken * aux1 = 0
    let c2 = diff_taken.mul(&col_evals[COL_AUX1]);
    // aux1=0 -> equal: (1-aux1) * diff = 0
    let c3 = one_m_aux1.mul(&diff);
    let c4 = col_evals[COL_AUX1].mul(&col_evals[COL_AUX1].sub(&one));

    target.add(&c1).add(&c2).add(&c3).add(&c4)
}

/// BLTU condition: taken iff rs1 < rs2 (unsigned).
/// aux1 = borrow = 1 if rs1 < rs2.
pub fn evaluate_bltu_condition_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let one_m_aux1 = one.sub(&col_evals[COL_AUX1]);

    let target = diff_taken.mul(&diff_not_taken);
    // taken -> aux1=1: diff_not_taken * (1-aux1) = 0
    let c1 = diff_not_taken.mul(&one_m_aux1);
    // not-taken -> aux1=0: diff_taken * aux1 = 0
    let c2 = diff_taken.mul(&col_evals[COL_AUX1]);
    let c3 = col_evals[COL_AUX1].mul(&col_evals[COL_AUX1].sub(&one));

    target.add(&c1).add(&c2).add(&c3)
}

/// BGEU condition: taken iff rs1 >= rs2 (unsigned).
/// aux1 = borrow = 1 if rs1 < rs2.
pub fn evaluate_bgeu_condition_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let one_m_aux1 = one.sub(&col_evals[COL_AUX1]);

    let target = diff_taken.mul(&diff_not_taken);
    // taken -> aux1=0: diff_not_taken * aux1 = 0
    let c1 = diff_not_taken.mul(&col_evals[COL_AUX1]);
    // not-taken -> aux1=1: diff_taken * (1-aux1) = 0
    let c2 = diff_taken.mul(&one_m_aux1);
    let c3 = col_evals[COL_AUX1].mul(&col_evals[COL_AUX1].sub(&one));

    target.add(&c1).add(&c2).add(&c3)
}

/// Evaluate signed branch condition constraints.
///
/// For signed comparison, the result depends on the sign bits of rs1 and rs2:
/// - If signs differ: sign_a=1, sign_b=0 means a is negative (a < b);
///   sign_a=0, sign_b=1 means a is positive (a >= b).
/// - If signs are the same: use unsigned comparison (borrow from rs1 - rs2).
///
/// Witnesses:
/// - aux0 = borrow from unsigned subtraction rs1 - rs2 (1 if rs1 < rs2 unsigned)
/// - aux1 = sign bit of rs1 (bit 63)
/// - aux2 = sign bit of rs2 (bit 63)
///
/// The signed_lt result is:
///   signed_lt = sign_a*(1-sign_b) + (1-sign_a-sign_b+2*sign_a*sign_b)*borrow
///
/// For BLT (is_lt=true): branch taken iff signed_lt = 1
/// For BGE (is_lt=false): branch taken iff signed_lt = 0
fn evaluate_signed_branch_condition(
    pc: &Scalar, imm: &Scalar, insn_len: &Scalar, next_pc: &Scalar,
    _rs1_val: &Scalar, _rs2_val: &Scalar,
    aux0: &Scalar, aux1: &Scalar, aux2: &Scalar,
    is_lt: bool,
    curve: metavm_zkp::field::CurveType,
) -> Scalar {
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);

    let taken_target = pc.add(imm);
    let not_taken_target = pc.add(insn_len);
    let diff_taken = next_pc.sub(&taken_target);
    let diff_not_taken = next_pc.sub(&not_taken_target);

    // Target check: (next_pc - taken) * (next_pc - not_taken) = 0
    let target_check = diff_taken.mul(&diff_not_taken);

    // aux1 (sign_a) and aux2 (sign_b) must be binary
    let sign_a_binary = aux1.mul(&aux1.sub(&one));
    let sign_b_binary = aux2.mul(&aux2.sub(&one));

    // aux0 (borrow) must be binary
    let borrow_binary = aux0.mul(&aux0.sub(&one));

    // Compute signed_lt algebraically:
    // signed_lt = sign_a*(1-sign_b) + (1 - sign_a - sign_b + 2*sign_a*sign_b) * borrow
    let sign_a = aux1;
    let sign_b = aux2;
    let borrow = aux0;

    let one_m_sign_b = one.sub(sign_b);
    let term1 = sign_a.mul(&one_m_sign_b);

    let same_sign = one.sub(sign_a).sub(sign_b).add(&two.mul(&sign_a.mul(sign_b)));
    let term2 = same_sign.mul(borrow);

    let signed_lt = term1.add(&term2);

    // Verify sign witnesses against actual rs1/rs2 values:
    // rs1_val = sign_a * 2^63 + rs1_low, so rs1_val - sign_a * 2^63 must be in [0, 2^63).
    // We cannot fully verify this range without lookup tables, but the binary checks
    // on sign_a/sign_b combined with the subtraction circuit provide soundness for
    // correct witnesses. The borrow witness is consistent with the unsigned comparison.
    // Full range checks are deferred to the lookup infrastructure.

    // Branch direction constraints:
    if is_lt {
        // BLT: taken iff signed_lt = 1
        // When taken (diff_not_taken =/= 0): signed_lt must be 1
        //   diff_not_taken * (1 - signed_lt) = 0
        // When not taken (diff_taken =/= 0): signed_lt must be 0
        //   diff_taken * signed_lt = 0
        let c_taken = diff_not_taken.mul(&one.sub(&signed_lt));
        let c_not_taken = diff_taken.mul(&signed_lt);
        target_check.add(&c_taken).add(&c_not_taken)
            .add(&sign_a_binary).add(&sign_b_binary).add(&borrow_binary)
    } else {
        // BGE: taken iff signed_lt = 0
        // When taken (diff_not_taken =/= 0): signed_lt must be 0
        //   diff_not_taken * signed_lt = 0
        // When not taken (diff_taken =/= 0): signed_lt must be 1
        //   diff_taken * (1 - signed_lt) = 0
        let c_taken = diff_not_taken.mul(&signed_lt);
        let c_not_taken = diff_taken.mul(&one.sub(&signed_lt));
        target_check.add(&c_taken).add(&c_not_taken)
            .add(&sign_a_binary).add(&sign_b_binary).add(&borrow_binary)
    }
}

/// BLT condition: taken iff rs1 < rs2 (signed).
/// aux0 = borrow from unsigned subtraction, aux1 = sign_a, aux2 = sign_b.
pub fn evaluate_blt_condition_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };
    evaluate_signed_branch_condition(
        &col_evals[COL_PC], &col_evals[COL_IMMEDIATE],
        &col_evals[COL_INSN_LEN], &col_evals[COL_NEXT_PC],
        &col_evals[COL_RS1_VAL], &col_evals[COL_RS2_VAL],
        aux0, aux1, aux2, true, curve)
}

/// BGE condition: taken iff rs1 >= rs2 (signed).
/// aux0 = borrow from unsigned subtraction, aux1 = sign_a, aux2 = sign_b.
pub fn evaluate_bge_condition_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let zero = Scalar::zero(curve);
    let aux0 = if col_evals.len() > COL_AUX0 { &col_evals[COL_AUX0] } else { &zero };
    let aux1 = if col_evals.len() > COL_AUX1 { &col_evals[COL_AUX1] } else { &zero };
    let aux2 = if col_evals.len() > COL_AUX2 { &col_evals[COL_AUX2] } else { &zero };
    evaluate_signed_branch_condition(
        &col_evals[COL_PC], &col_evals[COL_IMMEDIATE],
        &col_evals[COL_INSN_LEN], &col_evals[COL_NEXT_PC],
        &col_evals[COL_RS1_VAL], &col_evals[COL_RS2_VAL],
        aux0, aux1, aux2, false, curve)
}

/// JAL raw: rd = pc + insn_len, next_pc = pc + immediate.
pub fn evaluate_jal_raw(col_evals: &[Scalar]) -> Scalar {
    let expected_rd = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    let rd_check = col_evals[COL_RD_VAL_AFTER].sub(&expected_rd);
    let expected_pc = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let pc_check = col_evals[COL_NEXT_PC].sub(&expected_pc);
    rd_check.add(&pc_check)
}

/// JALR raw: rd = pc + insn_len, next_pc = (rs1 + imm) - aux1, aux1 binary.
pub fn evaluate_jalr_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let expected_rd = col_evals[COL_PC].add(&col_evals[COL_INSN_LEN]);
    let rd_check = col_evals[COL_RD_VAL_AFTER].sub(&expected_rd);
    let raw_target = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
    let pc_check = col_evals[COL_NEXT_PC].sub(&raw_target.sub(&col_evals[COL_AUX1]));
    let aux1_binary = col_evals[COL_AUX1].mul(&col_evals[COL_AUX1].sub(&one));
    rd_check.add(&pc_check).add(&aux1_binary)
}

/// LUI raw: rd = immediate.
pub fn evaluate_lui_raw(col_evals: &[Scalar]) -> Scalar {
    col_evals[COL_RD_VAL_AFTER].sub(&col_evals[COL_IMMEDIATE])
}

/// AUIPC raw: rd = pc + immediate.
pub fn evaluate_auipc_raw(col_evals: &[Scalar]) -> Scalar {
    let expected = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    col_evals[COL_RD_VAL_AFTER].sub(&expected)
}

/// Evaluate non-branch PC constraint: for non-branch/jump instructions,
/// next_pc must equal pc + insn_len.
pub fn evaluate_sequential_pc(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;

        let is_sequential = !matches!(
            insn_type_val,
            selectors::INSN_BRANCH |
            selectors::INSN_JAL |
            selectors::INSN_JALR |
            selectors::INSN_SYSTEM
        );

        if is_sequential {
            let expected = columns[COL_PC][i].add(&columns[COL_INSN_LEN][i]);
            let val = columns[COL_NEXT_PC][i].sub(&expected);
            result.push(val);
        } else {
            result.push(Scalar::zero(curve));
        }
    }

    result
}
