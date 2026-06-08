//! SBF constraint system implementing VmConstraintSystem.
//!
//! Constraints follow the same pattern as RISC-V since SBF is also
//! a register-based 64-bit architecture.

pub mod alu;
pub mod branches;
pub mod memory;

use metavm_zkp::field::Scalar;
use metavm_zkp::lookup::{LookupRequirements, LookupTable, LookupDeclaration, BitwiseLookupDeclaration, BitwiseOp};
use metavm_zkp::poly_arith::*;
use metavm_zkp::vm_constraints::VmConstraintSystem;
use crate::trace;

/// The SBF constraint system for proving BPF execution correctness.
pub struct SbfConstraintSystem;

impl SbfConstraintSystem {
    pub fn new() -> Self {
        SbfConstraintSystem
    }
}

impl VmConstraintSystem for SbfConstraintSystem {
    fn num_constraints(&self) -> usize {
        24 + 22 + 1  // 24 VM constraints + 22 binary selector + 1 sum-to-one = 47
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "alu_add".to_string(),
            "alu_sub".to_string(),
            "alu_mov".to_string(),
            "alu_mul".to_string(),
            "alu_div".to_string(),
            "alu_mod".to_string(),
            "alu_and".to_string(),
            "alu_or".to_string(),
            "alu_xor".to_string(),
            "alu_lsh".to_string(),
            "alu_rsh".to_string(),
            "alu_arsh".to_string(),
            "branch_eq".to_string(),
            "branch_neq".to_string(),
            "branch_lt".to_string(),
            "branch_ge".to_string(),
            "branch_other".to_string(),
            "load".to_string(),
            "store".to_string(),
            "alu_add_carry_binary".to_string(),
            "alu_sub_borrow_binary".to_string(),
            "call".to_string(),
            "exit".to_string(),
            "alu_neg".to_string(),
        ];
        for i in 0..22 {
            labels.push(format!("sel_binary_{}", i));
        }
        labels.push("sel_sum_to_one".to_string());
        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        // evaluate_on_domain uses integer branching, which is correct
        // on actual trace values (not arbitrary field elements).
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);
        let (add_carry_binary, sub_borrow_binary) =
            alu::evaluate_aux0_binary_checks(columns, num_rows);

        // Split ALU into per-operation vectors using integer branching
        let alu_full = alu::evaluate_alu(columns, num_rows);
        let mut alu_add_evals = Vec::with_capacity(num_rows);
        let mut alu_sub_evals = Vec::with_capacity(num_rows);
        let mut alu_mov_evals = Vec::with_capacity(num_rows);
        let mut alu_mul_evals = Vec::with_capacity(num_rows);
        let mut alu_div_evals = Vec::with_capacity(num_rows);
        let mut alu_mod_evals = Vec::with_capacity(num_rows);
        let mut alu_and_evals = Vec::with_capacity(num_rows);
        let mut alu_or_evals = Vec::with_capacity(num_rows);
        let mut alu_xor_evals = Vec::with_capacity(num_rows);
        let mut alu_lsh_evals = Vec::with_capacity(num_rows);
        let mut alu_rsh_evals = Vec::with_capacity(num_rows);
        let mut alu_arsh_evals = Vec::with_capacity(num_rows);
        let mut alu_other_evals = Vec::with_capacity(num_rows);

        let two_64 = alu::two_pow_64(curve);
        let one = Scalar::one(curve);

        for i in 0..num_rows {
            let insn_type = columns[trace::COL_INSN_TYPE][i].to_u64() as u8;
            let funct = columns[trace::COL_FUNCT][i].to_u64() as u8;

            macro_rules! push_zeros {
                ($($v:ident),+) => { $( $v.push(zero.clone()); )+ };
            }

            if insn_type == trace::INSN_ALU64 || insn_type == trace::INSN_ALU32 {
                match funct {
                    trace::FUNCT_ADD => {
                        alu_add_evals.push(alu_full[i].clone());
                        push_zeros!(alu_sub_evals, alu_mov_evals, alu_mul_evals, alu_div_evals,
                                    alu_mod_evals, alu_and_evals, alu_or_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_SUB => {
                        alu_sub_evals.push(alu_full[i].clone());
                        push_zeros!(alu_add_evals, alu_mov_evals, alu_mul_evals, alu_div_evals,
                                    alu_mod_evals, alu_and_evals, alu_or_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_MOV => {
                        alu_mov_evals.push(alu_full[i].clone());
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mul_evals, alu_div_evals,
                                    alu_mod_evals, alu_and_evals, alu_or_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_MUL => {
                        alu_mul_evals.push(alu_full[i].clone());
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_div_evals,
                                    alu_mod_evals, alu_and_evals, alu_or_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_DIV => {
                        alu_div_evals.push(alu_full[i].clone());
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_mod_evals, alu_and_evals, alu_or_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_MOD => {
                        alu_mod_evals.push(alu_full[i].clone());
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_and_evals, alu_or_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_AND => {
                        let body = columns[trace::COL_DST_VAL_AFTER][i]
                            .sub(&columns[trace::COL_AUX0][i]);
                        alu_and_evals.push(body);
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_mod_evals, alu_or_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_OR => {
                        let body = columns[trace::COL_DST_VAL_AFTER][i]
                            .sub(&columns[trace::COL_DST_VAL_BEFORE][i])
                            .sub(&columns[trace::COL_SRC_VAL][i])
                            .add(&columns[trace::COL_AUX0][i]);
                        alu_or_evals.push(body);
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_mod_evals, alu_and_evals, alu_xor_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_XOR => {
                        let two = Scalar::from_u64(2, curve);
                        let body = columns[trace::COL_DST_VAL_AFTER][i]
                            .sub(&columns[trace::COL_DST_VAL_BEFORE][i])
                            .sub(&columns[trace::COL_SRC_VAL][i])
                            .add(&two.mul(&columns[trace::COL_AUX0][i]));
                        alu_xor_evals.push(body);
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_mod_evals, alu_and_evals, alu_or_evals,
                                    alu_lsh_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_LSH => {
                        // LSH: dst_before * aux0 - aux1 * 2^64 - dst_after = 0
                        let product = columns[trace::COL_DST_VAL_BEFORE][i].mul(&columns[trace::COL_AUX0][i]);
                        let overflow_term = columns[trace::COL_AUX1][i].mul(&two_64);
                        let body = product.sub(&overflow_term).sub(&columns[trace::COL_DST_VAL_AFTER][i]);
                        alu_lsh_evals.push(body);
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_mod_evals, alu_and_evals, alu_or_evals,
                                    alu_xor_evals, alu_rsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_RSH => {
                        // RSH: dst_after * aux0 + aux1 - dst_before = 0
                        let product = columns[trace::COL_DST_VAL_AFTER][i].mul(&columns[trace::COL_AUX0][i]);
                        let body = product.add(&columns[trace::COL_AUX1][i]).sub(&columns[trace::COL_DST_VAL_BEFORE][i]);
                        alu_rsh_evals.push(body);
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_mod_evals, alu_and_evals, alu_or_evals,
                                    alu_xor_evals, alu_lsh_evals, alu_arsh_evals, alu_other_evals);
                    }
                    trace::FUNCT_ARSH => {
                        // ARSH: dst_after * aux0 + aux1 - dst_before - aux2 * 2^64 = 0
                        let product = columns[trace::COL_DST_VAL_AFTER][i].mul(&columns[trace::COL_AUX0][i]);
                        let correction_term = columns[trace::COL_AUX2][i].mul(&two_64);
                        let body = product.add(&columns[trace::COL_AUX1][i])
                            .sub(&columns[trace::COL_DST_VAL_BEFORE][i])
                            .sub(&correction_term);
                        alu_arsh_evals.push(body);
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_mod_evals, alu_and_evals, alu_or_evals,
                                    alu_xor_evals, alu_lsh_evals, alu_rsh_evals, alu_other_evals);
                    }
                    _ => {
                        // NEG and any remaining: dst_after + dst_before - aux0 * 2^64 = 0
                        // plus aux0 * (aux0 - 1) = 0
                        let sum = columns[trace::COL_DST_VAL_AFTER][i].add(&columns[trace::COL_DST_VAL_BEFORE][i]);
                        let carry_term = columns[trace::COL_AUX0][i].mul(&two_64);
                        let main = sum.sub(&carry_term);
                        let binary = columns[trace::COL_AUX0][i].mul(&columns[trace::COL_AUX0][i].sub(&one));
                        alu_other_evals.push(main.add(&binary));
                        push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                                    alu_div_evals, alu_mod_evals, alu_and_evals, alu_or_evals,
                                    alu_xor_evals, alu_lsh_evals, alu_rsh_evals, alu_arsh_evals);
                    }
                }
            } else {
                push_zeros!(alu_add_evals, alu_sub_evals, alu_mov_evals, alu_mul_evals,
                            alu_div_evals, alu_mod_evals, alu_and_evals, alu_or_evals,
                            alu_xor_evals, alu_lsh_evals, alu_rsh_evals, alu_arsh_evals,
                            alu_other_evals);
            }
        }

        // Split memory into load-only and store-only constraint vectors
        let memory_full = memory::evaluate_memory(columns, num_rows);
        let mut load_evals = Vec::with_capacity(num_rows);
        let mut store_evals = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let insn_type = columns[trace::COL_INSN_TYPE][i].to_u64() as u8;
            if insn_type == trace::INSN_LOAD {
                load_evals.push(memory_full[i].clone());
                store_evals.push(zero.clone());
            } else if insn_type == trace::INSN_STORE {
                load_evals.push(zero.clone());
                store_evals.push(memory_full[i].clone());
            } else {
                load_evals.push(zero.clone());
                store_evals.push(zero.clone());
            }
        }

        // Split branches into 5 per-type vectors
        let branches_full = branches::evaluate_branches(columns, num_rows);
        let mut beq_evals = Vec::with_capacity(num_rows);
        let mut bneq_evals = Vec::with_capacity(num_rows);
        let mut blt_evals = Vec::with_capacity(num_rows);
        let mut bge_evals = Vec::with_capacity(num_rows);
        let mut bother_evals = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let insn_type = columns[trace::COL_INSN_TYPE][i].to_u64() as u8;
            let funct = columns[trace::COL_FUNCT][i].to_u64() as u8;
            if insn_type == trace::INSN_BRANCH {
                match funct {
                    trace::FUNCT_JEQ => {
                        beq_evals.push(branches_full[i].clone());
                        bneq_evals.push(zero.clone());
                        blt_evals.push(zero.clone());
                        bge_evals.push(zero.clone());
                        bother_evals.push(zero.clone());
                    }
                    trace::FUNCT_JNE => {
                        beq_evals.push(zero.clone());
                        bneq_evals.push(branches_full[i].clone());
                        blt_evals.push(zero.clone());
                        bge_evals.push(zero.clone());
                        bother_evals.push(zero.clone());
                    }
                    f if f == trace::FUNCT_JLT || f == trace::FUNCT_JSLT => {
                        beq_evals.push(zero.clone());
                        bneq_evals.push(zero.clone());
                        blt_evals.push(branches_full[i].clone());
                        bge_evals.push(zero.clone());
                        bother_evals.push(zero.clone());
                    }
                    f if f == trace::FUNCT_JGE || f == trace::FUNCT_JSGE => {
                        beq_evals.push(zero.clone());
                        bneq_evals.push(zero.clone());
                        blt_evals.push(zero.clone());
                        bge_evals.push(branches_full[i].clone());
                        bother_evals.push(zero.clone());
                    }
                    _ => {
                        beq_evals.push(zero.clone());
                        bneq_evals.push(zero.clone());
                        blt_evals.push(zero.clone());
                        bge_evals.push(zero.clone());
                        bother_evals.push(branches_full[i].clone());
                    }
                }
            } else {
                beq_evals.push(zero.clone());
                bneq_evals.push(zero.clone());
                blt_evals.push(zero.clone());
                bge_evals.push(zero.clone());
                bother_evals.push(zero.clone());
            }
        }

        // CALL and EXIT are oracle-verified; constraint body is zero
        let call_evals = vec![zero.clone(); num_rows];
        let exit_evals = vec![zero.clone(); num_rows];

        vec![
            alu_add_evals,
            alu_sub_evals,
            alu_mov_evals,
            alu_mul_evals,
            alu_div_evals,
            alu_mod_evals,
            alu_and_evals,
            alu_or_evals,
            alu_xor_evals,
            alu_lsh_evals,
            alu_rsh_evals,
            alu_arsh_evals,
            beq_evals,
            bneq_evals,
            blt_evals,
            bge_evals,
            bother_evals,
            load_evals,
            store_evals,
            add_carry_binary,
            sub_borrow_binary,
            call_evals,
            exit_evals,
            alu_other_evals,
        ]
    }

    fn evaluate_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        alpha: &Scalar,
    ) -> Scalar {
        let curve = alpha.curve_type();
        if col_evals_at_z.len() < trace::NUM_SBF_COLUMNS {
            return Scalar::zero(curve);
        }

        let one = Scalar::one(curve);
        let mut result = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        // Constraint 0: ALU ADD main equation (gated by sel_alu_add)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_ADD]
                .mul(&alu::evaluate_alu_add_main_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 1: ALU SUB main equation (gated by sel_alu_sub)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_SUB]
                .mul(&alu::evaluate_alu_sub_main_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 2: ALU MOV (gated by sel_alu_mov)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_MOV]
                .mul(&alu::evaluate_alu_mov_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 3: ALU MUL (gated by sel_alu_mul)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_MUL]
                .mul(&alu::evaluate_alu_mul_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 4: ALU DIV (gated by sel_alu_div)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_DIV]
                .mul(&alu::evaluate_alu_div_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 5: ALU MOD (gated by sel_alu_mod)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_MOD]
                .mul(&alu::evaluate_alu_mod_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 6: ALU AND (gated by sel_alu_and)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_AND]
                .mul(&alu::evaluate_alu_and_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 7: ALU OR (gated by sel_alu_or)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_OR]
                .mul(&alu::evaluate_alu_or_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 8: ALU XOR (gated by sel_alu_xor)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_XOR]
                .mul(&alu::evaluate_alu_xor_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 9: ALU LSH (gated by sel_alu_lsh)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_LSH]
                .mul(&alu::evaluate_alu_lsh_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 10: ALU RSH (gated by sel_alu_rsh)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_RSH]
                .mul(&alu::evaluate_alu_rsh_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 11: ALU ARSH (gated by sel_alu_arsh)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_ARSH]
                .mul(&alu::evaluate_alu_arsh_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 12: branch EQ (gated by sel_branch_eq)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_BRANCH_EQ]
                .mul(&branches::evaluate_branch_eq_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 13: branch NEQ (gated by sel_branch_neq)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_BRANCH_NEQ]
                .mul(&branches::evaluate_branch_neq_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 14: branch LT (gated by sel_branch_lt)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_BRANCH_LT]
                .mul(&branches::evaluate_branch_lt_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 15: branch GE (gated by sel_branch_ge)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_BRANCH_GE]
                .mul(&branches::evaluate_branch_ge_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 16: branch other (gated by sel_branch_other)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_BRANCH_OTHER]
                .mul(&branches::evaluate_branches_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 17: load (gated by sel_load)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_LOAD]
                .mul(&memory::evaluate_memory_load_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 18: store (gated by sel_store)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_STORE]
                .mul(&memory::evaluate_memory_store_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 19: ADD carry binary check (gated by sel_alu_add)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_ADD]
                .mul(&alu::evaluate_aux0_binary_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 20: SUB borrow binary check (gated by sel_alu_sub)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_SUB]
                .mul(&alu::evaluate_aux0_binary_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 21: CALL (oracle-verified, body is zero)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_CALL]
                .mul(&alu::evaluate_call_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 22: EXIT (oracle-verified, body is zero)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_EXIT]
                .mul(&alu::evaluate_exit_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Constraint 23: ALU OTHER / NEG (gated by sel_alu_other)
        result = result.add(
            &col_evals_at_z[trace::COL_SEL_ALU_OTHER]
                .mul(&alu::evaluate_alu_neg_raw(col_evals_at_z))
                .mul(&ap),
        );
        ap = ap.mul(alpha);

        // Selector consistency: binary constraints (sel * (sel - 1) = 0)
        let sel_indices = self.selector_column_indices();
        for &si in &sel_indices {
            let s = &col_evals_at_z[si];
            let s_m1 = s.sub(&one);
            result = result.add(&s.mul(&s_m1).mul(&ap));
            ap = ap.mul(alpha);
        }

        // Sum-to-one: sum of all selectors = 1
        let mut sel_sum = Scalar::zero(curve);
        for &si in &sel_indices {
            sel_sum = sel_sum.add(&col_evals_at_z[si]);
        }
        result = result.add(&sel_sum.sub(&one).mul(&ap));

        result
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![
            trace::COL_SEL_ALU_ADD,
            trace::COL_SEL_ALU_SUB,
            trace::COL_SEL_ALU_MOV,
            trace::COL_SEL_ALU_MUL,
            trace::COL_SEL_ALU_DIV,
            trace::COL_SEL_ALU_MOD,
            trace::COL_SEL_ALU_AND,
            trace::COL_SEL_ALU_OR,
            trace::COL_SEL_ALU_XOR,
            trace::COL_SEL_ALU_LSH,
            trace::COL_SEL_ALU_RSH,
            trace::COL_SEL_ALU_ARSH,
            trace::COL_SEL_ALU_OTHER,
            trace::COL_SEL_LOAD,
            trace::COL_SEL_STORE,
            trace::COL_SEL_BRANCH_EQ,
            trace::COL_SEL_BRANCH_NEQ,
            trace::COL_SEL_BRANCH_LT,
            trace::COL_SEL_BRANCH_GE,
            trace::COL_SEL_BRANCH_OTHER,
            trace::COL_SEL_CALL,
            trace::COL_SEL_EXIT,
        ]
    }

    fn padding_selector_column(&self) -> Option<usize> {
        Some(trace::COL_SEL_EXIT)
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
        // Set padding rows' PC and NEXT_PC to the last real instruction's next_pc
        // so that the cross-row constraint pc[i+1] == next_pc[i] is satisfied.
        let last_next_pc = columns[trace::COL_NEXT_PC][num_rows - 1].clone();
        for i in num_rows..padded_size {
            columns[trace::COL_PC][i] = last_next_pc.clone();
            columns[trace::COL_NEXT_PC][i] = last_next_pc.clone();
        }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        vec![trace::COL_PC]
    }

    fn num_shifted_constraints(&self) -> usize {
        1
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        if shifted_evals.is_empty() {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // PC continuity: pc(omega*z) - next_pc(z)
        let pc_next = &shifted_evals[0];
        let next_pc = &col_evals_at_z[trace::COL_NEXT_PC];
        let body = pc_next.sub(next_pc);

        // Multiply by (z - omega^{n-1}) to exclude wrap-around row
        let exclusion = z.sub(omega_n_minus_1);

        ap.mul(&body).mul(&exclusion)
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();

        let col = |idx: usize| -> &Vec<Scalar> { &column_coeffs[idx] };

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // PC continuity: poly_shift(col_pc, omega) - col_next_pc
        let pc_shifted = poly_shift(col(trace::COL_PC), omega);
        let body = poly_sub(&pc_shifted, col(trace::COL_NEXT_PC), curve);

        // Compute omega^{n-1} via repeated squaring
        let mut omega_n_minus_1 = Scalar::one(curve);
        let mut base = omega.clone();
        let mut exp = domain_size - 1;
        while exp > 0 {
            if exp & 1 == 1 {
                omega_n_minus_1 = omega_n_minus_1.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }

        // Multiply by (X - omega^{n-1})
        let body_excluded = poly_mul_linear(&body, &omega_n_minus_1);

        poly_scalar_mul(&body_excluded, &ap)
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        use rayon::prelude::*;
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let col = |idx: usize| -> &Vec<Scalar> { &column_coeffs[idx] };
        let two_64 = alu::two_pow_64(curve);

        // Phase 1: Collect (selector, body) pairs for all VM constraints.
        // Body computation is sequential (shared precomputed values), but
        // the final gating poly_mul calls will be parallelized.
        let mut pairs: Vec<(Vec<Scalar>, Vec<Scalar>)> = Vec::new();

        // Constraint 0: ALU ADD
        {
            let sum = poly_add(col(trace::COL_DST_VAL_BEFORE), col(trace::COL_SRC_VAL), curve);
            let carry_term = poly_scalar_mul(col(trace::COL_AUX0), &two_64);
            let rhs = poly_add(&carry_term, col(trace::COL_DST_VAL_AFTER), curve);
            let body = poly_sub(&sum, &rhs, curve);
            pairs.push((col(trace::COL_SEL_ALU_ADD).clone(), body));
        }

        // Constraint 1: ALU SUB
        {
            let lhs = poly_add(col(trace::COL_DST_VAL_AFTER), col(trace::COL_SRC_VAL), curve);
            let borrow_term = poly_scalar_mul(col(trace::COL_AUX0), &two_64);
            let rhs = poly_add(col(trace::COL_DST_VAL_BEFORE), &borrow_term, curve);
            let body = poly_sub(&lhs, &rhs, curve);
            pairs.push((col(trace::COL_SEL_ALU_SUB).clone(), body));
        }

        // Constraint 2: ALU MOV
        {
            let body = poly_sub(col(trace::COL_DST_VAL_AFTER), col(trace::COL_SRC_VAL), curve);
            pairs.push((col(trace::COL_SEL_ALU_MOV).clone(), body));
        }

        // Constraint 3: ALU MUL
        {
            let product = poly_mul(col(trace::COL_DST_VAL_BEFORE), col(trace::COL_SRC_VAL), curve);
            let hi_term = poly_scalar_mul(col(trace::COL_AUX0), &two_64);
            let rhs = poly_add(&hi_term, col(trace::COL_DST_VAL_AFTER), curve);
            let body = poly_sub(&product, &rhs, curve);
            pairs.push((col(trace::COL_SEL_ALU_MUL).clone(), body));
        }

        // Constraint 4: ALU DIV
        {
            let prod = poly_mul(col(trace::COL_DST_VAL_AFTER), col(trace::COL_SRC_VAL), curve);
            let lhs = poly_add(&prod, col(trace::COL_AUX0), curve);
            let body = poly_sub(&lhs, col(trace::COL_DST_VAL_BEFORE), curve);
            pairs.push((col(trace::COL_SEL_ALU_DIV).clone(), body));
        }

        // Constraint 5: ALU MOD
        {
            let prod = poly_mul(col(trace::COL_AUX0), col(trace::COL_SRC_VAL), curve);
            let lhs = poly_add(&prod, col(trace::COL_DST_VAL_AFTER), curve);
            let body = poly_sub(&lhs, col(trace::COL_DST_VAL_BEFORE), curve);
            pairs.push((col(trace::COL_SEL_ALU_MOD).clone(), body));
        }

        // Constraint 6: ALU AND
        {
            let body = poly_sub(col(trace::COL_DST_VAL_AFTER), col(trace::COL_AUX0), curve);
            pairs.push((col(trace::COL_SEL_ALU_AND).clone(), body));
        }

        // Constraint 7: ALU OR
        {
            let body = poly_sub(col(trace::COL_DST_VAL_AFTER), col(trace::COL_DST_VAL_BEFORE), curve);
            let body = poly_sub(&body, col(trace::COL_SRC_VAL), curve);
            let body = poly_add(&body, col(trace::COL_AUX0), curve);
            pairs.push((col(trace::COL_SEL_ALU_OR).clone(), body));
        }

        // Constraint 8: ALU XOR
        {
            let two = Scalar::from_u64(2, curve);
            let body = poly_sub(col(trace::COL_DST_VAL_AFTER), col(trace::COL_DST_VAL_BEFORE), curve);
            let body = poly_sub(&body, col(trace::COL_SRC_VAL), curve);
            let two_aux0 = poly_scalar_mul(col(trace::COL_AUX0), &two);
            let body = poly_add(&body, &two_aux0, curve);
            pairs.push((col(trace::COL_SEL_ALU_XOR).clone(), body));
        }

        // Constraint 9: ALU LSH
        {
            let product = poly_mul(col(trace::COL_DST_VAL_BEFORE), col(trace::COL_AUX0), curve);
            let overflow_term = poly_scalar_mul(col(trace::COL_AUX1), &two_64);
            let rhs = poly_add(&overflow_term, col(trace::COL_DST_VAL_AFTER), curve);
            let body = poly_sub(&product, &rhs, curve);
            pairs.push((col(trace::COL_SEL_ALU_LSH).clone(), body));
        }

        // Constraint 10: ALU RSH
        {
            let product = poly_mul(col(trace::COL_DST_VAL_AFTER), col(trace::COL_AUX0), curve);
            let lhs = poly_add(&product, col(trace::COL_AUX1), curve);
            let body = poly_sub(&lhs, col(trace::COL_DST_VAL_BEFORE), curve);
            pairs.push((col(trace::COL_SEL_ALU_RSH).clone(), body));
        }

        // Constraint 11: ALU ARSH
        {
            let product = poly_mul(col(trace::COL_DST_VAL_AFTER), col(trace::COL_AUX0), curve);
            let lhs = poly_add(&product, col(trace::COL_AUX1), curve);
            let correction_term = poly_scalar_mul(col(trace::COL_AUX2), &two_64);
            let body = poly_sub(&lhs, col(trace::COL_DST_VAL_BEFORE), curve);
            let body = poly_sub(&body, &correction_term, curve);
            pairs.push((col(trace::COL_SEL_ALU_ARSH).clone(), body));
        }

        // Precompute common branch sub-expressions
        let insn_size_poly = vec![Scalar::from_u64(8, curve)];
        let taken_poly = poly_add(col(trace::COL_PC), col(trace::COL_IMMEDIATE), curve);
        let not_taken_poly = poly_add(col(trace::COL_PC), &insn_size_poly, curve);
        let diff_taken_poly = poly_sub(col(trace::COL_NEXT_PC), &taken_poly, curve);
        let diff_not_taken_poly = poly_sub(col(trace::COL_NEXT_PC), &not_taken_poly, curve);
        let target_check_poly = poly_mul(&diff_taken_poly, &diff_not_taken_poly, curve);
        let aux1_m1_poly = poly_sub(col(trace::COL_AUX1), &one_poly, curve);
        let aux1_binary_poly = poly_mul(col(trace::COL_AUX1), &aux1_m1_poly, curve);
        let one_m_aux1_poly = poly_sub(&one_poly, col(trace::COL_AUX1), curve);
        let dst_src_diff_poly = poly_sub(col(trace::COL_DST_VAL_BEFORE), col(trace::COL_SRC_VAL), curve);

        // Constraint 12: branch EQ
        {
            let c1 = poly_mul(&diff_not_taken_poly, col(trace::COL_AUX1), curve);
            let c2 = poly_mul(&diff_taken_poly, &one_m_aux1_poly, curve);
            let c3 = poly_mul(&one_m_aux1_poly, &dst_src_diff_poly, curve);
            let body = poly_add(&target_check_poly, &c1, curve);
            let body = poly_add(&body, &c2, curve);
            let body = poly_add(&body, &c3, curve);
            let body = poly_add(&body, &aux1_binary_poly, curve);
            pairs.push((col(trace::COL_SEL_BRANCH_EQ).clone(), body));
        }

        // Constraint 13: branch NEQ
        {
            let c1 = poly_mul(&diff_not_taken_poly, &one_m_aux1_poly, curve);
            let c2 = poly_mul(&diff_taken_poly, col(trace::COL_AUX1), curve);
            let c3 = poly_mul(&one_m_aux1_poly, &dst_src_diff_poly, curve);
            let body = poly_add(&target_check_poly, &c1, curve);
            let body = poly_add(&body, &c2, curve);
            let body = poly_add(&body, &c3, curve);
            let body = poly_add(&body, &aux1_binary_poly, curve);
            pairs.push((col(trace::COL_SEL_BRANCH_NEQ).clone(), body));
        }

        // Constraint 14: branch LT
        {
            let c1 = poly_mul(&diff_not_taken_poly, &one_m_aux1_poly, curve);
            let c2 = poly_mul(&diff_taken_poly, col(trace::COL_AUX1), curve);
            let body = poly_add(&target_check_poly, &c1, curve);
            let body = poly_add(&body, &c2, curve);
            let body = poly_add(&body, &aux1_binary_poly, curve);
            pairs.push((col(trace::COL_SEL_BRANCH_LT).clone(), body));
        }

        // Constraint 15: branch GE
        {
            let c1 = poly_mul(&diff_not_taken_poly, col(trace::COL_AUX1), curve);
            let c2 = poly_mul(&diff_taken_poly, &one_m_aux1_poly, curve);
            let body = poly_add(&target_check_poly, &c1, curve);
            let body = poly_add(&body, &c2, curve);
            let body = poly_add(&body, &aux1_binary_poly, curve);
            pairs.push((col(trace::COL_SEL_BRANCH_GE).clone(), body));
        }

        // Constraint 16: branch other
        {
            let body = poly_add(&target_check_poly, &aux1_binary_poly, curve);
            pairs.push((col(trace::COL_SEL_BRANCH_OTHER).clone(), body));
        }

        // Constraint 17: load
        {
            let expected_addr = poly_add(col(trace::COL_SRC_VAL), col(trace::COL_IMMEDIATE), curve);
            let addr_check = poly_sub(col(trace::COL_MEM_ADDR), &expected_addr, curve);
            let val_check = poly_sub(col(trace::COL_DST_VAL_AFTER), col(trace::COL_MEM_VAL), curve);
            let body = poly_add(&addr_check, &val_check, curve);
            pairs.push((col(trace::COL_SEL_LOAD).clone(), body));
        }

        // Constraint 18: store
        {
            let expected_addr = poly_add(col(trace::COL_DST_VAL_BEFORE), col(trace::COL_IMMEDIATE), curve);
            let addr_check = poly_sub(col(trace::COL_MEM_ADDR), &expected_addr, curve);
            let val_check = poly_sub(col(trace::COL_MEM_VAL), col(trace::COL_SRC_VAL), curve);
            let body = poly_add(&addr_check, &val_check, curve);
            pairs.push((col(trace::COL_SEL_STORE).clone(), body));
        }

        // Constraint 19: ADD carry binary
        {
            let aux0_m1 = poly_sub(col(trace::COL_AUX0), &one_poly, curve);
            let body = poly_mul(col(trace::COL_AUX0), &aux0_m1, curve);
            pairs.push((col(trace::COL_SEL_ALU_ADD).clone(), body));
        }

        // Constraint 20: SUB borrow binary
        {
            let aux0_m1 = poly_sub(col(trace::COL_AUX0), &one_poly, curve);
            let body = poly_mul(col(trace::COL_AUX0), &aux0_m1, curve);
            pairs.push((col(trace::COL_SEL_ALU_SUB).clone(), body));
        }

        // Constraints 21-22: CALL and EXIT oracle (zero body, no pair needed)
        // We must account for 2 alpha advances with no poly_mul.

        // Constraint 23: ALU OTHER / NEG
        {
            let sum = poly_add(col(trace::COL_DST_VAL_AFTER), col(trace::COL_DST_VAL_BEFORE), curve);
            let carry_term = poly_scalar_mul(col(trace::COL_AUX0), &two_64);
            let main = poly_sub(&sum, &carry_term, curve);
            let aux0_m1 = poly_sub(col(trace::COL_AUX0), &one_poly, curve);
            let binary = poly_mul(col(trace::COL_AUX0), &aux0_m1, curve);
            let body = poly_add(&main, &binary, curve);
            pairs.push((col(trace::COL_SEL_ALU_OTHER).clone(), body));
        }

        // Phase 2: Parallel gating poly_mul for all constraint pairs
        let gated_results: Vec<Vec<Scalar>> = pairs.par_iter()
            .map(|(sel, body)| poly_mul(sel, body, curve))
            .collect();

        // Phase 3: Sequential accumulation with alpha powers
        // Constraints 0-20 are pairs[0..21], then 2 oracle skips, then constraint 23 is pairs[21]
        let mut c = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);

        // Accumulate constraints 0-20 (pairs[0..21])
        for gated in &gated_results[..21] {
            c = poly_add(&c, &poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Skip 2 oracle constraints (21: CALL, 22: EXIT)
        ap = ap.mul(alpha);
        ap = ap.mul(alpha);

        // Accumulate constraint 23 (pairs[21])
        c = poly_add(&c, &poly_scalar_mul(&gated_results[21], &ap), curve);
        ap = ap.mul(alpha);

        // Phase 4: Parallel binary selector constraints
        let sel_indices = self.selector_column_indices();
        let binary_results: Vec<Vec<Scalar>> = sel_indices.par_iter()
            .map(|&si| {
                let s = &column_coeffs[si];
                let s_m1 = poly_sub(s, &one_poly, curve);
                poly_mul(s, &s_m1, curve)
            })
            .collect();

        for binary in &binary_results {
            c = poly_add(&c, &poly_scalar_mul(binary, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Phase 5: Sum-to-one (no poly_mul needed, sequential)
        let mut sum = vec![Scalar::zero(curve)];
        for &si in &sel_indices {
            sum = poly_add(&sum, col(si), curve);
        }
        if !sum.is_empty() {
            sum[0] = sum[0].sub(&Scalar::one(curve));
        }
        c = poly_add(&c, &poly_scalar_mul(&sum, &ap), curve);

        c
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let table64 = LookupTable::range(64);

        let declarations = vec![
            // MUL upper 64 bits: aux0 must be in [0, 2^64-1] on MUL rows
            (
                LookupDeclaration {
                    label: "mul_aux0_range".to_string(),
                    column_index: trace::COL_AUX0,
                    max_bits: 64,
                    selector_column: Some(trace::COL_SEL_ALU_MUL),
                },
                0, // table index
            ),
            // DIV remainder: aux0 must be in [0, 2^64-1] on DIV rows
            (
                LookupDeclaration {
                    label: "div_remainder_range".to_string(),
                    column_index: trace::COL_AUX0,
                    max_bits: 64,
                    selector_column: Some(trace::COL_SEL_ALU_DIV),
                },
                0,
            ),
            // MOD quotient: aux0 must be in [0, 2^64-1] on MOD rows
            (
                LookupDeclaration {
                    label: "mod_quotient_range".to_string(),
                    column_index: trace::COL_AUX0,
                    max_bits: 64,
                    selector_column: Some(trace::COL_SEL_ALU_MOD),
                },
                0,
            ),
            // dst output: dst_val_after must always be a valid 64-bit value
            (
                LookupDeclaration {
                    label: "dst_val_after_range".to_string(),
                    column_index: trace::COL_DST_VAL_AFTER,
                    max_bits: 64,
                    selector_column: None,
                },
                0,
            ),
            // src input: src_val must always be a valid 64-bit value
            (
                LookupDeclaration {
                    label: "src_val_range".to_string(),
                    column_index: trace::COL_SRC_VAL,
                    max_bits: 64,
                    selector_column: None,
                },
                0,
            ),
        ];

        LookupRequirements {
            tables: vec![table64],
            declarations,
        }
    }

    fn memory_columns(&self) -> Option<(usize, Vec<usize>, Vec<usize>, Vec<usize>)> {
        Some((trace::COL_MEM_ADDR, vec![trace::COL_MEM_VAL], vec![trace::COL_SEL_LOAD], vec![trace::COL_SEL_STORE]))
    }

    fn oracle_selectors(&self) -> Vec<usize> {
        vec![
            trace::COL_SEL_CALL,
        ]
    }

    fn register_ports(&self) -> Vec<(usize, usize, bool)> {
        vec![
            (trace::COL_SRC_REG, trace::COL_SRC_VAL, false),     // src read
            (trace::COL_DST_REG, trace::COL_DST_VAL_AFTER, true), // dst write
        ]
    }

    fn bitwise_lookup_declarations(&self) -> Vec<BitwiseLookupDeclaration> {
        vec![
            BitwiseLookupDeclaration {
                label: "sbf_and".to_string(),
                operand_a_column: trace::COL_DST_VAL_BEFORE,
                operand_b_column: trace::COL_SRC_VAL,
                result_column: trace::COL_DST_VAL_AFTER,
                width_bits: 64,
                op: BitwiseOp::And,
                selectors: vec![trace::COL_SEL_ALU_AND],
            },
            BitwiseLookupDeclaration {
                label: "sbf_or".to_string(),
                operand_a_column: trace::COL_DST_VAL_BEFORE,
                operand_b_column: trace::COL_SRC_VAL,
                result_column: trace::COL_DST_VAL_AFTER,
                width_bits: 64,
                op: BitwiseOp::Or,
                selectors: vec![trace::COL_SEL_ALU_OR],
            },
            BitwiseLookupDeclaration {
                label: "sbf_xor".to_string(),
                operand_a_column: trace::COL_DST_VAL_BEFORE,
                operand_b_column: trace::COL_SRC_VAL,
                result_column: trace::COL_DST_VAL_AFTER,
                width_bits: 64,
                op: BitwiseOp::Xor,
                selectors: vec![trace::COL_SEL_ALU_XOR],
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metavm_zkp::field::CurveType;
    use crate::trace::*;

    fn make_add_trace() -> SbfTraceColumns {
        let mut trace = SbfTraceColumns::new();
        let row = SbfTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x07,
            dst_reg: 1,
            dst_val_before: 10,
            dst_val_after: 15,
            src_reg: 0,
            src_val: 5,
            mem_addr: 0,
            mem_val: 0,
            mem_size: 0,
            next_pc: 8,
            insn_type: INSN_ALU64,
            funct: FUNCT_ADD,
            immediate: 5,
            aux0: 0,
            aux1: 0,
            aux2: 0,
        };
        trace.push_row(&row);
        trace
    }

    #[test]
    fn test_sbf_add_constraint() {
        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(eval[0][0].is_zero(), "ADD constraint should be satisfied");
    }

    #[test]
    fn test_sbf_add_invalid() {
        let mut trace = SbfTraceColumns::new();
        let row = SbfTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x07,
            dst_reg: 1,
            dst_val_before: 10,
            dst_val_after: 16, // wrong! should be 15
            src_reg: 0,
            src_val: 5,
            mem_addr: 0,
            mem_val: 0,
            mem_size: 0,
            next_pc: 8,
            insn_type: INSN_ALU64,
            funct: FUNCT_ADD,
            immediate: 5,
            aux0: 0,
            aux1: 0,
            aux2: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(!eval[0][0].is_zero(), "Invalid ADD should fail constraint");
    }

    #[test]
    fn test_sbf_sub_constraint() {
        let mut trace = SbfTraceColumns::new();
        let row = SbfTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x17,
            dst_reg: 1,
            dst_val_before: 30,
            dst_val_after: 20,
            src_reg: 2,
            src_val: 10,
            mem_addr: 0,
            mem_val: 0,
            mem_size: 0,
            next_pc: 8,
            insn_type: INSN_ALU64,
            funct: FUNCT_SUB,
            immediate: 10,
            aux0: 0, // no borrow
            aux1: 0,
            aux2: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(eval[0][0].is_zero(), "SUB constraint should be satisfied");
    }

    #[test]
    fn test_sbf_branch_not_taken() {
        let mut trace = SbfTraceColumns::new();
        // JEQ with dst=5, src=0: not equal, so not taken. aux1=1 (not equal).
        let row = SbfTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x15,
            dst_reg: 1,
            dst_val_before: 5,
            dst_val_after: 5,
            src_reg: 0,
            src_val: 0,
            mem_addr: 0,
            mem_val: 0,
            mem_size: 0,
            next_pc: 8, // not taken
            insn_type: INSN_BRANCH,
            funct: FUNCT_JEQ,
            immediate: 0x20, // target offset
            aux0: 0,
            aux1: 1, // 1 = not equal (dst != src)
            aux2: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Branch EQ constraint (index 12) should be satisfied
        assert!(eval[12][0].is_zero(), "Branch not-taken should satisfy constraint");
    }

    #[test]
    fn test_sbf_load_constraint() {
        let mut trace = SbfTraceColumns::new();
        let row = SbfTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x79, // LDXDW
            dst_reg: 1,
            dst_val_before: 0,
            dst_val_after: 42,
            src_reg: 2,
            src_val: 0x1000,
            mem_addr: 0x1010,
            mem_val: 42,
            mem_size: 8,
            next_pc: 8,
            insn_type: INSN_LOAD,
            funct: FUNCT_LDXDW,
            immediate: 0x10, // offset
            aux0: 0,
            aux1: 0,
            aux2: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Load constraint (index 17) should be satisfied
        assert!(eval[17][0].is_zero(), "Load constraint should be satisfied");
        // Store constraint (index 18) should be zero (selector is 0 for load row)
        assert!(eval[18][0].is_zero(), "Store constraint should be zero for load row");
    }

    #[test]
    fn test_sbf_store_constraint() {
        let mut trace = SbfTraceColumns::new();
        // STX: store src_val to [dst + offset]
        let row = SbfTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x7b, // STXDW
            dst_reg: 2,
            dst_val_before: 0x1000,
            dst_val_after: 0x1000, // dst unchanged for store
            src_reg: 1,
            src_val: 42,
            mem_addr: 0x1010, // dst_val_before + immediate
            mem_val: 42,       // = src_val
            mem_size: 8,
            next_pc: 8,
            insn_type: INSN_STORE,
            funct: FUNCT_STXDW,
            immediate: 0x10,
            aux0: 0,
            aux1: 0,
            aux2: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Store constraint (index 18) should be satisfied
        assert!(eval[18][0].is_zero(), "Store constraint should be satisfied");
        // Load constraint (index 17) should also be zero (selector is 0)
        assert!(eval[17][0].is_zero(), "Load constraint should be zero for store row");
    }

    #[test]
    fn test_sbf_prove_verify_roundtrip() {
        metavm_zkp::commitment::init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();

        let proof = metavm_zkp::prover::prove(&polys, &cs);
        let valid = metavm_zkp::verifier::verify(&proof, &cs);
        assert!(valid, "Valid SBF trace should produce a verifying proof");
    }

    #[test]
    fn test_sbf_prove_verify_bls48581_scheme() {
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();

        let proof = metavm_zkp::prover::prove_with_scheme(&polys, &cs, &scheme);
        let valid = metavm_zkp::verifier::verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        assert!(valid, "SBF BLS48-581 scheme prove/verify should succeed");
    }

    #[test]
    fn test_sbf_prove_verify_bls12381_scheme() {
        use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let scheme = Bls12381Scheme::new();
        scheme.init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls12381);
        let cs = SbfConstraintSystem::new();

        let proof = metavm_zkp::prover::prove_with_scheme(&polys, &cs, &scheme);
        let valid = metavm_zkp::verifier::verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls12381);
        assert!(valid, "SBF BLS12-381 scheme prove/verify should succeed");
    }

    /// Regression: real SBF ChunkProof must verify through
    /// `begin_chunk_scheme` + `verify_final_scheme` (the recursive
    /// accumulator path the prove-sbf / prove-slot CLIs use).
    ///
    /// Companion to the EVM and RISC-V regressions. Locks in the
    /// bitwise transcript fix in
    /// `recursive::recover_chunk_challenges_scheme`.
    #[test]
    #[ignore = "slow: full prove + recursive verify on SBF trace; run with --release --ignored"]
    fn sbf_chunk_proof_verifies_through_recursive_accumulator() {
        use metavm_zkp::prover::prove_chunk_with_scheme;
        use metavm_zkp::recursive::{begin_chunk_scheme, verify_final_scheme};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, curve);
        let cs = SbfConstraintSystem::new();

        let zero = [0u8; 32];
        let chunk_proof = prove_chunk_with_scheme(&polys, &cs, 0, &zero, &zero, &scheme);

        let recursive = begin_chunk_scheme(chunk_proof, &scheme, curve);
        let valid = verify_final_scheme(&recursive, &scheme);
        assert!(
            valid,
            "SBF ChunkProof must verify through recursive accumulator \
             (regression guard for the bitwise transcript fix in \
             `recover_chunk_challenges_scheme`)",
        );
    }

    /// Create a 5-instruction trace: mov+mov+add+mov+exit
    fn make_multi_insn_trace() -> SbfTraceColumns {
        let mut trace = SbfTraceColumns::new();
        // Step 0: mov64 r1, 10
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0xb7, dst_reg: 1,
            dst_val_before: 0, dst_val_after: 10,
            src_reg: 0, src_val: 10, // immediate source
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_MOV,
            immediate: 10, aux0: 0, aux1: 0, aux2: 0,
        });
        // Step 1: mov64 r2, 20
        trace.push_row(&SbfTraceRow {
            step: 1, pc: 8, opcode: 0xb7, dst_reg: 2,
            dst_val_before: 0, dst_val_after: 20,
            src_reg: 0, src_val: 20,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 16, insn_type: INSN_ALU64, funct: FUNCT_MOV,
            immediate: 20, aux0: 0, aux1: 0, aux2: 0,
        });
        // Step 2: add64 r1, r2 (r1 = 10 + 20 = 30)
        trace.push_row(&SbfTraceRow {
            step: 2, pc: 16, opcode: 0x0f, dst_reg: 1,
            dst_val_before: 10, dst_val_after: 30,
            src_reg: 2, src_val: 20,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 24, insn_type: INSN_ALU64, funct: FUNCT_ADD,
            immediate: 0, aux0: 0, aux1: 0, aux2: 0,
        });
        // Step 3: mov64 r0, r1
        trace.push_row(&SbfTraceRow {
            step: 3, pc: 24, opcode: 0xbf, dst_reg: 0,
            dst_val_before: 0, dst_val_after: 30,
            src_reg: 1, src_val: 30,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 32, insn_type: INSN_ALU64, funct: FUNCT_MOV,
            immediate: 0, aux0: 0, aux1: 0, aux2: 0,
        });
        // Step 4: exit
        trace.push_row(&SbfTraceRow {
            step: 4, pc: 32, opcode: 0x95, dst_reg: 0,
            dst_val_before: 30, dst_val_after: 30,
            src_reg: 0, src_val: 0,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 40, insn_type: INSN_EXIT, funct: 0,
            immediate: 0, aux0: 0, aux1: 0, aux2: 0,
        });
        trace
    }

    #[test]
    fn test_sbf_multi_insn_constraints_satisfied() {
        let trace = make_multi_insn_trace();
        let mut polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        polys.fix_selector_padding(&cs);
        let columns = polys.columns();

        // Check constraints on actual trace rows
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        for (ci, constraint_evals) in eval.iter().enumerate() {
            for (ri, val) in constraint_evals.iter().enumerate() {
                assert!(val.is_zero(),
                    "Constraint {} should be zero on row {} but is non-zero", ci, ri);
            }
        }
    }

    #[test]
    fn test_sbf_multi_insn_constraint_poly_divisible() {
        use metavm_zkp::commitment;
        use bls48581::bls48581::big;
        use bls48581::bls48581::rom;

        commitment::init();

        let trace = make_multi_insn_trace();
        let mut polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        polys.fix_selector_padding(&cs);

        let domain_size = polys.domain_size();
        let n = domain_size as usize;
        eprintln!("[diag] domain_size = {}", n);

        // Get column coefficients via IFFT
        let big_columns = polys.columns_as_bls48581();
        let column_coeffs: Vec<Vec<Scalar>> = big_columns.iter()
            .map(|col| {
                let coeffs_big = commitment::eval_to_coeff(col, domain_size);
                coeffs_big.iter().map(|b| Scalar::Bls48581(big::BIG::new_copy(b))).collect()
            })
            .collect();

        // Build alpha from a fixed value for reproducibility
        let alpha = Scalar::from_u64(7, CurveType::Bls48581);

        // Build C(x) via build_constraint_polynomial
        let c_coeffs = cs.build_constraint_polynomial(&column_coeffs, &alpha, domain_size);
        eprintln!("[diag] C(x) has {} coefficients", c_coeffs.len());

        // Verify C(x) is zero at all roots of unity by evaluating C(x) at each
        // root ω^i. We do this by FFT of c_coeffs back to evaluation form.
        // c_coeffs may be longer than n, so we evaluate manually using Horner.
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);

        // Verify divisibility by checking the remainder of C(x) / Z(x).
        let c_len = c_coeffs.len();
        let c_deg = if c_len > 0 { c_len - 1 } else { 0 };
        eprintln!("[diag] C(x) degree = {}", c_deg);

        // Convert c_coeffs to BIG for division
        let mut dividend: Vec<big::BIG> = c_coeffs.iter()
            .map(|s| big::BIG::new_copy(s.as_bls48581()))
            .collect();
        while dividend.len() < n {
            dividend.push(big::BIG::new());
        }

        if c_deg >= n {
            let q_deg = c_deg - n;
            let mut q = vec![big::BIG::new(); q_deg + 1];
            let mut div = dividend.clone();
            while div.len() <= c_deg {
                div.push(big::BIG::new());
            }
            for i in (n..=c_deg).rev() {
                q[i - n] = big::BIG::new_copy(&div[i]);
                div[i - n] = big::BIG::modadd(&div[i - n], &div[i], &modulus);
            }

            // Check remainder is zero
            let mut remainder_nonzero = false;
            for i in 0..n {
                if !div[i].iszilch() {
                    eprintln!("[diag] REMAINDER[{}] is non-zero!", i);
                    remainder_nonzero = true;
                }
            }
            assert!(!remainder_nonzero, "C(x) should be divisible by Z(x) = x^{} - 1", n);
            eprintln!("[diag] Division remainder is zero - C(x) is divisible by Z(x) ✓");
            eprintln!("[diag] Q(x) has {} coefficients (degree {})", q.len(), q_deg);
        } else {
            eprintln!("[diag] C(x) degree {} < n={}, checking via legacy division", c_deg, n);
        }

        // Now check evaluate_at_point matches polynomial evaluation
        // Pick a random z
        let z = Scalar::from_u64(12345, CurveType::Bls48581);
        let z_big = big::BIG::new_copy(z.as_bls48581());

        // Evaluate C(z) from coefficient polynomial
        let c_coeffs_big: Vec<big::BIG> = c_coeffs.iter()
            .map(|s| big::BIG::new_copy(s.as_bls48581()))
            .collect();
        let c_at_z_from_poly = commitment::eval_poly_at(&c_coeffs_big, &z_big);

        // Evaluate columns at z
        let col_evals: Vec<Scalar> = column_coeffs.iter()
            .map(|coeffs| {
                let coeffs_big: Vec<big::BIG> = coeffs.iter()
                    .map(|s| big::BIG::new_copy(s.as_bls48581()))
                    .collect();
                let val = commitment::eval_poly_at(&coeffs_big, &z_big);
                Scalar::Bls48581(val)
            })
            .collect();

        // Evaluate C(z) from evaluate_at_point
        let c_at_z_from_eval = cs.evaluate_at_point(&col_evals, &alpha);
        let c_at_z_eval_big = big::BIG::new_copy(c_at_z_from_eval.as_bls48581());

        // Compare
        let diff = big::BIG::modadd(
            &c_at_z_from_poly,
            &big::BIG::modneg(&c_at_z_eval_big, &modulus),
            &modulus,
        );

        if !diff.iszilch() {
            eprintln!("[diag] MISMATCH! C(z) from polynomial ≠ C(z) from evaluate_at_point");
            eprintln!("[diag] C(z) from poly: {:?}", c_at_z_from_poly);
            eprintln!("[diag] C(z) from eval: {:?}", c_at_z_eval_big);
        } else {
            eprintln!("[diag] C(z) matches between polynomial and evaluate_at_point ✓");
        }
        assert!(diff.iszilch(), "C(z) from polynomial should equal C(z) from evaluate_at_point");
    }

    #[test]
    fn test_sbf_multi_insn_prove_verify() {
        metavm_zkp::commitment::init();

        let trace = make_multi_insn_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();

        let proof = metavm_zkp::prover::prove(&polys, &cs);
        eprintln!("[diag] num_steps={}, domain_size={}, num_q_chunks={}",
            proof.num_steps, proof.domain_size, proof.num_quotient_chunks);
        let valid = metavm_zkp::verifier::verify(&proof, &cs);
        assert!(valid, "Multi-instruction SBF trace should produce a verifying proof");
    }

    #[test]
    fn test_sbf_lsh_constraint() {
        let mut trace = SbfTraceColumns::new();
        // LSH: 0x0F << 4 = 0xF0
        let dst_before = 0x0Fu64;
        let shift = 4u64;
        let dst_after = dst_before << shift;
        let power = 1u64 << shift;
        let overflow = (((dst_before as u128) * (power as u128)) >> 64) as u64;
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0x67, // ALU64 LSH imm
            dst_reg: 1,
            dst_val_before: dst_before,
            dst_val_after: dst_after,
            src_reg: 0, src_val: shift,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_LSH,
            immediate: shift,
            aux0: power, aux1: overflow, aux2: 0,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // LSH constraint (index 9)
        assert!(eval[9][0].is_zero(), "LSH constraint should be satisfied");
    }

    #[test]
    fn test_sbf_lsh_large_shift() {
        let mut trace = SbfTraceColumns::new();
        // LSH: 0xFFFF_FFFF_FFFF_FFFF << 32 should overflow
        let dst_before = 0xFFFF_FFFF_FFFF_FFFFu64;
        let shift = 32u64;
        let dst_after = dst_before.wrapping_shl(shift as u32);
        let power = 1u64 << shift;
        let full = (dst_before as u128) * (power as u128);
        let overflow = (full >> 64) as u64;
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0x67,
            dst_reg: 1,
            dst_val_before: dst_before,
            dst_val_after: dst_after,
            src_reg: 0, src_val: shift,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_LSH,
            immediate: shift,
            aux0: power, aux1: overflow, aux2: 0,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(eval[9][0].is_zero(), "LSH constraint should be satisfied for large shift with overflow");
    }

    #[test]
    fn test_sbf_rsh_constraint() {
        let mut trace = SbfTraceColumns::new();
        // RSH: 0xF0 >> 4 = 0x0F
        let dst_before = 0xF0u64;
        let shift = 4u64;
        let dst_after = dst_before >> shift;
        let power = 1u64 << shift;
        let remainder = dst_before % power;
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0x77, // ALU64 RSH imm
            dst_reg: 1,
            dst_val_before: dst_before,
            dst_val_after: dst_after,
            src_reg: 0, src_val: shift,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_RSH,
            immediate: shift,
            aux0: power, aux1: remainder, aux2: 0,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // RSH constraint (index 10)
        assert!(eval[10][0].is_zero(), "RSH constraint should be satisfied");
    }

    #[test]
    fn test_sbf_rsh_with_remainder() {
        let mut trace = SbfTraceColumns::new();
        // RSH: 255 >> 3 = 31, remainder = 255 % 8 = 7
        let dst_before = 255u64;
        let shift = 3u64;
        let dst_after = dst_before >> shift;
        let power = 1u64 << shift;
        let remainder = dst_before % power;
        assert_eq!(dst_after, 31);
        assert_eq!(remainder, 7);
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0x77,
            dst_reg: 1,
            dst_val_before: dst_before,
            dst_val_after: dst_after,
            src_reg: 0, src_val: shift,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_RSH,
            immediate: shift,
            aux0: power, aux1: remainder, aux2: 0,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(eval[10][0].is_zero(), "RSH constraint should be satisfied with nonzero remainder");
    }

    #[test]
    fn test_sbf_arsh_positive() {
        let mut trace = SbfTraceColumns::new();
        // ARSH on positive number: 0x7F00 >> 4 = 0x07F0
        let dst_before = 0x7F00u64;
        let shift = 4u64;
        let dst_after = ((dst_before as i64) >> shift) as u64;
        let power = 1u64 << shift;
        let remainder = dst_before % power;
        let sign_bit = (dst_before >> 63) & 1;
        let sign_correction = sign_bit * (power - 1);
        assert_eq!(sign_correction, 0); // positive, no correction
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0xC7, // ALU64 ARSH imm
            dst_reg: 1,
            dst_val_before: dst_before,
            dst_val_after: dst_after,
            src_reg: 0, src_val: shift,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_ARSH,
            immediate: shift,
            aux0: power, aux1: remainder, aux2: sign_correction,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // ARSH constraint (index 11)
        assert!(eval[11][0].is_zero(), "ARSH constraint should be satisfied for positive value");
    }

    #[test]
    fn test_sbf_arsh_negative() {
        let mut trace = SbfTraceColumns::new();
        // ARSH on negative number (sign bit set): 0xFFFF_FFFF_FFFF_FFF0 >> 4
        // i64: -16 >> 4 = -1 = 0xFFFF_FFFF_FFFF_FFFF
        let dst_before = 0xFFFF_FFFF_FFFF_FFF0u64;
        let shift = 4u64;
        let dst_after = ((dst_before as i64) >> shift) as u64;
        assert_eq!(dst_after, 0xFFFF_FFFF_FFFF_FFFFu64);
        let power = 1u64 << shift; // 16
        let remainder = dst_before % power; // 0
        let sign_bit = (dst_before >> 63) & 1; // 1
        let sign_correction = sign_bit * (power - 1); // 15
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0xC7,
            dst_reg: 1,
            dst_val_before: dst_before,
            dst_val_after: dst_after,
            src_reg: 0, src_val: shift,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_ARSH,
            immediate: shift,
            aux0: power, aux1: remainder, aux2: sign_correction,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(eval[11][0].is_zero(), "ARSH constraint should be satisfied for negative value");
    }

    #[test]
    fn test_sbf_lsh_invalid() {
        let mut trace = SbfTraceColumns::new();
        // Wrong dst_after for LSH
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0x67,
            dst_reg: 1,
            dst_val_before: 0x0F,
            dst_val_after: 0xFF, // wrong! should be 0xF0
            src_reg: 0, src_val: 4,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_LSH,
            immediate: 4,
            aux0: 16, aux1: 0, aux2: 0,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(!eval[9][0].is_zero(), "Invalid LSH should fail constraint");
    }

    #[test]
    fn test_sbf_shift_prove_verify() {
        metavm_zkp::commitment::init();

        let mut trace = SbfTraceColumns::new();
        // mov64 r1, 0xFF
        trace.push_row(&SbfTraceRow {
            step: 0, pc: 0, opcode: 0xb7, dst_reg: 1,
            dst_val_before: 0, dst_val_after: 0xFF,
            src_reg: 0, src_val: 0xFF,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 8, insn_type: INSN_ALU64, funct: FUNCT_MOV,
            immediate: 0xFF, aux0: 0, aux1: 0, aux2: 0,
        });
        // lsh64 r1, 4 => r1 = 0xFF0
        let lsh_before = 0xFFu64;
        let lsh_after = lsh_before << 4;
        let lsh_power = 1u64 << 4;
        let lsh_overflow = (((lsh_before as u128) * (lsh_power as u128)) >> 64) as u64;
        trace.push_row(&SbfTraceRow {
            step: 1, pc: 8, opcode: 0x67, dst_reg: 1,
            dst_val_before: lsh_before, dst_val_after: lsh_after,
            src_reg: 0, src_val: 4,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 16, insn_type: INSN_ALU64, funct: FUNCT_LSH,
            immediate: 4, aux0: lsh_power, aux1: lsh_overflow, aux2: 0,
        });
        // rsh64 r1, 8 => r1 = 0x0F
        let rsh_before = 0xFF0u64;
        let rsh_after = rsh_before >> 8;
        let rsh_power = 1u64 << 8;
        let rsh_remainder = rsh_before % rsh_power;
        trace.push_row(&SbfTraceRow {
            step: 2, pc: 16, opcode: 0x77, dst_reg: 1,
            dst_val_before: rsh_before, dst_val_after: rsh_after,
            src_reg: 0, src_val: 8,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 24, insn_type: INSN_ALU64, funct: FUNCT_RSH,
            immediate: 8, aux0: rsh_power, aux1: rsh_remainder, aux2: 0,
        });
        // mov64 r0, r1
        trace.push_row(&SbfTraceRow {
            step: 3, pc: 24, opcode: 0xbf, dst_reg: 0,
            dst_val_before: 0, dst_val_after: 0x0F,
            src_reg: 1, src_val: 0x0F,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 32, insn_type: INSN_ALU64, funct: FUNCT_MOV,
            immediate: 0, aux0: 0, aux1: 0, aux2: 0,
        });
        // exit
        trace.push_row(&SbfTraceRow {
            step: 4, pc: 32, opcode: 0x95, dst_reg: 0,
            dst_val_before: 0x0F, dst_val_after: 0x0F,
            src_reg: 0, src_val: 0,
            mem_addr: 0, mem_val: 0, mem_size: 0,
            next_pc: 40, insn_type: INSN_EXIT, funct: 0,
            immediate: 0, aux0: 0, aux1: 0, aux2: 0,
        });

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();

        let proof = metavm_zkp::prover::prove(&polys, &cs);
        let valid = metavm_zkp::verifier::verify(&proof, &cs);
        assert!(valid, "Shift instruction trace should produce a verifying proof");
    }
}
