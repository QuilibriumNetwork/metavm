pub mod selectors;
pub mod alu;
pub mod control_flow;
pub mod memory_ops;

use metavm_zkp::field::{Scalar, CurveType};
use metavm_zkp::trace::{TracePolynomials, Polynomial};
use metavm_zkp::vm_constraints::VmConstraintSystem;
use crate::trace::TraceColumns;
use metavm_zkp::lookup::{LookupRequirements, LookupTable, LookupDeclaration};

// RISC-V trace column index constants.
// These match the order produced by TraceColumns (skipping column 0 = step).
pub const COL_PC: usize = 0;
pub const COL_RD: usize = 1;
pub const COL_RD_VAL_BEFORE: usize = 2;
pub const COL_RD_VAL_AFTER: usize = 3;
pub const COL_RS1: usize = 4;
pub const COL_RS1_VAL: usize = 5;
pub const COL_RS2: usize = 6;
pub const COL_RS2_VAL: usize = 7;
pub const COL_MEM_ADDR: usize = 8;
pub const COL_MEM_VAL: usize = 9;
pub const COL_NEXT_PC: usize = 10;
pub const COL_PRIVILEGE_MODE: usize = 11;
pub const COL_INSN_TYPE: usize = 12;
pub const COL_FUNCT: usize = 13;
pub const COL_IMMEDIATE: usize = 14;
pub const COL_INSN_LEN: usize = 15;
pub const COL_AUX0: usize = 16;
pub const COL_AUX1: usize = 17;
pub const COL_AUX2: usize = 18;
// Selector columns (one-hot instruction type + funct indicators)
pub const COL_SEL_R_ALU_ADD: usize = 19;
pub const COL_SEL_R_ALU_SUB: usize = 20;
pub const COL_SEL_R_AND: usize = 21;
pub const COL_SEL_R_OR: usize = 22;
pub const COL_SEL_R_XOR: usize = 23;
pub const COL_SEL_R_SLL: usize = 24;
pub const COL_SEL_R_SRL: usize = 25;
pub const COL_SEL_R_SRA: usize = 26;
pub const COL_SEL_R_COMPARE: usize = 27;
pub const COL_SEL_I_ALU_ADD: usize = 28;
pub const COL_SEL_I_AND: usize = 29;
pub const COL_SEL_I_OR: usize = 30;
pub const COL_SEL_I_XOR: usize = 31;
pub const COL_SEL_I_SLL: usize = 32;
pub const COL_SEL_I_SRL: usize = 33;
pub const COL_SEL_I_SRA: usize = 34;
pub const COL_SEL_I_COMPARE: usize = 35;
pub const COL_SEL_W_ALU_ADD: usize = 36;
pub const COL_SEL_W_ALU_SUB: usize = 37;
pub const COL_SEL_W_ALU_ADDI: usize = 38;
pub const COL_SEL_W_SLL: usize = 39;
pub const COL_SEL_W_SRL: usize = 40;
pub const COL_SEL_W_SRA: usize = 41;
pub const COL_SEL_W_ALU_OTHER: usize = 42;
pub const COL_SEL_MUL: usize = 43;
pub const COL_SEL_DIV: usize = 44;
pub const COL_SEL_REM: usize = 45;
pub const COL_SEL_MULDIV_OTHER: usize = 46;
pub const COL_SEL_LOAD: usize = 47;
pub const COL_SEL_STORE: usize = 48;
pub const COL_SEL_BEQ: usize = 49;
pub const COL_SEL_BNE: usize = 50;
pub const COL_SEL_BLTU: usize = 51;
pub const COL_SEL_BGEU: usize = 52;
pub const COL_SEL_BLT: usize = 53;
pub const COL_SEL_BGE: usize = 54;
pub const COL_SEL_JAL: usize = 55;
pub const COL_SEL_JALR: usize = 56;
pub const COL_SEL_LUI: usize = 57;
pub const COL_SEL_AUIPC: usize = 58;
pub const COL_SEL_CSRRW: usize = 59;
pub const COL_SEL_CSRRS: usize = 60;
pub const COL_SEL_CSRRC: usize = 61;
pub const COL_SEL_SYSTEM: usize = 62;
pub const COL_SEL_LR: usize = 63;
pub const COL_SEL_SC: usize = 64;
pub const COL_SEL_AMO_SWAP: usize = 65;
pub const COL_SEL_AMO_ADD: usize = 66;
pub const COL_SEL_AMO_BITWISE: usize = 67;
pub const COL_SEL_AMO_COMPARE: usize = 68;
pub const COL_SEL_ATOMIC_OTHER: usize = 69;
pub const COL_SEL_MULHU: usize = 70;
pub const COL_SEL_CSRRW_I: usize = 71;
pub const COL_SEL_CSRRS_I: usize = 72;
pub const COL_SEL_MULH: usize = 73;
pub const COL_SEL_MULHSU: usize = 74;
pub const COL_SEL_MULW: usize = 75;
pub const COL_SEL_DIVW: usize = 76;
pub const COL_SEL_REMW: usize = 77;
pub const COL_SEL_AMO_OR: usize = 78;
pub const COL_SEL_AMO_XOR: usize = 79;
pub const COL_SEL_AMO_MAXU: usize = 80;
pub const COL_SEL_AMO_MINS: usize = 81;
pub const COL_SEL_AMO_MAXS: usize = 82;

/// Convert all columns of a `TraceColumns` into polynomials.
/// Defaults to BLS48-581 for backward compatibility.
pub fn trace_polys_from_columns(columns: &TraceColumns) -> TracePolynomials {
    trace_polys_from_columns_with_curve(columns, CurveType::Bls48581)
}

/// Convert all columns of a `TraceColumns` into polynomials for a specific curve.
pub fn trace_polys_from_columns_with_curve(columns: &TraceColumns, curve: CurveType) -> TracePolynomials {
    let num_rows = columns.pc.len();

    let polys = vec![
        Polynomial::from_u64_vec_with_curve(&columns.pc, curve),
        Polynomial::from_u64_vec_with_curve(&columns.rd, curve),
        Polynomial::from_u64_vec_with_curve(&columns.rd_val_before, curve),
        Polynomial::from_u64_vec_with_curve(&columns.rd_val_after, curve),
        Polynomial::from_u64_vec_with_curve(&columns.rs1, curve),
        Polynomial::from_u64_vec_with_curve(&columns.rs1_val, curve),
        Polynomial::from_u64_vec_with_curve(&columns.rs2, curve),
        Polynomial::from_u64_vec_with_curve(&columns.rs2_val, curve),
        Polynomial::from_u64_vec_with_curve(&columns.mem_addr, curve),
        Polynomial::from_u64_vec_with_curve(&columns.mem_val, curve),
        Polynomial::from_u64_vec_with_curve(&columns.next_pc, curve),
        Polynomial::from_u64_vec_with_curve(&columns.privilege_mode, curve),
        Polynomial::from_u64_vec_with_curve(&columns.insn_type, curve),
        Polynomial::from_u64_vec_with_curve(&columns.funct, curve),
        Polynomial::from_u64_vec_with_curve(&columns.immediate, curve),
        Polynomial::from_u64_vec_with_curve(&columns.insn_len, curve),
        Polynomial::from_u64_vec_with_curve(&columns.aux0, curve),
        Polynomial::from_u64_vec_with_curve(&columns.aux1, curve),
        Polynomial::from_u64_vec_with_curve(&columns.aux2, curve),
        // Selector columns
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_alu_add, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_alu_sub, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_and, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_or, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_xor, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_sll, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_srl, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_sra, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_r_compare, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_alu_add, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_and, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_or, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_xor, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_sll, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_srl, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_sra, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_i_compare, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_w_alu_add, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_w_alu_sub, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_w_alu_addi, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_w_sll, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_w_srl, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_w_sra, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_w_alu_other, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_mul, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_div, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_rem, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_muldiv_other, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_load, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_store, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_beq, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_bne, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_bltu, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_bgeu, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_blt, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_bge, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_jal, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_jalr, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_lui, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_auipc, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_csrrw, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_csrrs, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_csrrc, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_system, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_lr, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_sc, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_swap, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_add, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_bitwise, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_compare, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_atomic_other, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_mulhu, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_csrrw_i, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_csrrs_i, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_mulh, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_mulhsu, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_mulw, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_divw, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_remw, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_or, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_xor, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_maxu, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_mins, curve),
        Polynomial::from_u64_vec_with_curve(&columns.sel_amo_maxs, curve),
    ];

    TracePolynomials::from_polynomials(polys, num_rows, curve)
}

/// The RISC-V constraint system for RV64IMAC execution traces.
///
/// Supports two modes:
/// - `new()`: Legacy 3-constraint system for backward compatibility
/// - `full()`: Extended constraint system with selector-based instruction verification
pub struct RiscvConstraintSystem {
    use_extended: bool,
}

impl RiscvConstraintSystem {
    /// Create the legacy 3-constraint system (backward compatible).
    pub fn new() -> Self {
        RiscvConstraintSystem { use_extended: false }
    }

    /// Create the full extended constraint system with selector-based AIR.
    pub fn full() -> Self {
        RiscvConstraintSystem { use_extended: true }
    }
}

impl VmConstraintSystem for RiscvConstraintSystem {
    fn num_constraints(&self) -> usize {
        if self.use_extended {
            // 63 VM constraints (0-62) + 64 binary selector constraints + 1 sum-to-one = 128
            128
        } else {
            3
        }
    }

    fn constraint_labels(&self) -> Vec<String> {
        if self.use_extended {
            let mut labels = vec![
                "sequential_pc".to_string(),
                "alu_r_add".to_string(),
                "alu_r_sub".to_string(),
                "alu_i_add".to_string(),
                "beq_condition".to_string(),
                "bne_condition".to_string(),
                "bltu_condition".to_string(),
                "bgeu_condition".to_string(),
                "blt_condition".to_string(),
                "bge_condition".to_string(),
                "jal_body".to_string(),
                "jalr_body".to_string(),
                "lui_body".to_string(),
                "auipc_body".to_string(),
                "memory_ops".to_string(),
                "alu_r_add_carry_binary".to_string(),
                "alu_r_sub_borrow_binary".to_string(),
                "alu_i_add_carry_binary".to_string(),
                "alu_w_addw".to_string(),
                "alu_w_subw".to_string(),
                "alu_w_addiw".to_string(),
                "alu_mul".to_string(),
                "alu_div".to_string(),
                "alu_rem".to_string(),
                "alu_r_and".to_string(),
                "alu_r_or".to_string(),
                "alu_r_xor".to_string(),
                "alu_i_and".to_string(),
                "alu_i_or".to_string(),
                "alu_i_xor".to_string(),
                "alu_r_sll".to_string(),
                "alu_r_srl".to_string(),
                "alu_r_sra".to_string(),
                "alu_i_sll".to_string(),
                "alu_i_srl".to_string(),
                "alu_i_sra".to_string(),
                "alu_w_sll".to_string(),
                "alu_w_srl".to_string(),
                "alu_w_sra".to_string(),
                "r_compare".to_string(),
                "i_compare".to_string(),
                "csrrc".to_string(),
                "lr".to_string(),
                "sc".to_string(),
                "alu_mulhu".to_string(),
                "csrrw_r".to_string(),
                "csrrw_i".to_string(),
                "csrrs_r".to_string(),
                "csrrs_i".to_string(),
                "mulh".to_string(),
                "mulhsu".to_string(),
                "mulw".to_string(),
                "divw".to_string(),
                "remw".to_string(),
                "amo_swap".to_string(),
                "amo_add".to_string(),
                "amo_and".to_string(),
                "amo_or".to_string(),
                "amo_xor".to_string(),
                "amo_minu".to_string(),
                "amo_maxu".to_string(),
                "amo_mins".to_string(),
                "amo_maxs".to_string(),
            ];
            let sel_names = [
                "r_alu_add", "r_alu_sub",
                "r_and", "r_or", "r_xor",
                "r_sll", "r_srl", "r_sra", "r_compare",
                "i_alu_add",
                "i_and", "i_or", "i_xor",
                "i_sll", "i_srl", "i_sra", "i_compare",
                "w_alu_add", "w_alu_sub", "w_alu_addi",
                "w_sll", "w_srl", "w_sra", "w_alu_other",
                "mul", "div", "rem", "muldiv_other",
                "load", "store",
                "beq", "bne", "bltu", "bgeu", "blt", "bge",
                "jal", "jalr", "lui", "auipc",
                "csrrw", "csrrs", "csrrc",
                "system",
                "lr", "sc", "amo_swap", "amo_add",
                "amo_bitwise", "amo_compare", "atomic_other",
                "mulhu",
                "csrrw_i", "csrrs_i",
                "mulh", "mulhsu", "mulw", "divw", "remw",
                "amo_or", "amo_xor",
                "amo_maxu", "amo_mins", "amo_maxs",
            ];
            for name in &sel_names {
                labels.push(format!("sel_{}_binary", name));
            }
            labels.push("selector_sum_to_one".to_string());
            labels
        } else {
            vec![
                "pc_transition".to_string(),
                "register_x0_zero".to_string(),
                "alu_correctness".to_string(),
            ]
        }
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        if self.use_extended {
            self.evaluate_extended_on_domain(columns, num_rows)
        } else {
            self.evaluate_legacy_on_domain(columns, num_rows)
        }
    }

    fn evaluate_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        alpha: &Scalar,
    ) -> Scalar {
        if self.use_extended {
            self.evaluate_at_point_extended(col_evals_at_z, alpha)
        } else {
            self.evaluate_at_point_legacy(col_evals_at_z, alpha)
        }
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        if self.use_extended {
            vec![
                COL_SEL_R_ALU_ADD, COL_SEL_R_ALU_SUB,
                COL_SEL_R_AND, COL_SEL_R_OR, COL_SEL_R_XOR,
                COL_SEL_R_SLL, COL_SEL_R_SRL, COL_SEL_R_SRA, COL_SEL_R_COMPARE,
                COL_SEL_I_ALU_ADD,
                COL_SEL_I_AND, COL_SEL_I_OR, COL_SEL_I_XOR,
                COL_SEL_I_SLL, COL_SEL_I_SRL, COL_SEL_I_SRA, COL_SEL_I_COMPARE,
                COL_SEL_W_ALU_ADD, COL_SEL_W_ALU_SUB, COL_SEL_W_ALU_ADDI,
                COL_SEL_W_SLL, COL_SEL_W_SRL, COL_SEL_W_SRA, COL_SEL_W_ALU_OTHER,
                COL_SEL_MUL, COL_SEL_DIV, COL_SEL_REM, COL_SEL_MULDIV_OTHER,
                COL_SEL_LOAD, COL_SEL_STORE,
                COL_SEL_BEQ, COL_SEL_BNE, COL_SEL_BLTU, COL_SEL_BGEU,
                COL_SEL_BLT, COL_SEL_BGE,
                COL_SEL_JAL, COL_SEL_JALR, COL_SEL_LUI, COL_SEL_AUIPC,
                COL_SEL_CSRRW, COL_SEL_CSRRS, COL_SEL_CSRRC,
                COL_SEL_SYSTEM,
                COL_SEL_LR, COL_SEL_SC, COL_SEL_AMO_SWAP, COL_SEL_AMO_ADD,
                COL_SEL_AMO_BITWISE, COL_SEL_AMO_COMPARE, COL_SEL_ATOMIC_OTHER,
                COL_SEL_MULHU,
                COL_SEL_CSRRW_I, COL_SEL_CSRRS_I,
                COL_SEL_MULH, COL_SEL_MULHSU, COL_SEL_MULW, COL_SEL_DIVW, COL_SEL_REMW,
                COL_SEL_AMO_OR, COL_SEL_AMO_XOR,
                COL_SEL_AMO_MAXU, COL_SEL_AMO_MINS, COL_SEL_AMO_MAXS,
            ]
        } else {
            Vec::new()
        }
    }

    fn padding_selector_column(&self) -> Option<usize> {
        if self.use_extended {
            Some(COL_SEL_SYSTEM) // 62
        } else {
            None
        }
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if !self.use_extended || num_rows == 0 || num_rows >= padded_size {
            return;
        }
        // Set padding rows' PC and NEXT_PC to the last real instruction's next_pc
        // value. This ensures the cross-row constraint pc[i+1] == next_pc[i] is
        // satisfied at the boundary between real trace and padding rows, and also
        // within the padding region itself (all padding rows are identical).
        let last_next_pc = columns[COL_NEXT_PC][num_rows - 1].clone();
        for i in num_rows..padded_size {
            columns[COL_PC][i] = last_next_pc.clone();
            columns[COL_NEXT_PC][i] = last_next_pc.clone();
        }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        if self.use_extended {
            vec![COL_PC]
        } else {
            Vec::new()
        }
    }

    fn num_shifted_constraints(&self) -> usize {
        if self.use_extended { 1 } else { 0 }
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
        if !self.use_extended || shifted_evals.is_empty() {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // PC continuity: pc(omega*z) - next_pc(z)
        let pc_next = &shifted_evals[0]; // col_pc evaluated at omega*z
        let next_pc = &col_evals_at_z[COL_NEXT_PC];
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
        if !self.use_extended {
            return vec![Scalar::zero(alpha.curve_type())];
        }
        let curve = alpha.curve_type();
        use metavm_zkp::poly_arith::*;

        let col = |idx: usize| -> &Vec<Scalar> { &column_coeffs[idx] };

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // PC continuity: poly_shift(col_pc, omega) - col_next_pc
        let pc_shifted = poly_shift(col(COL_PC), omega);
        let body = poly_sub(&pc_shifted, col(COL_NEXT_PC), curve);

        // Multiply by (X - omega^{n-1}) to exclude last row
        // omega^{n-1} = omega^(domain_size - 1)
        let n = domain_size;
        let mut omega_n_minus_1 = Scalar::one(curve);
        let mut base = omega.clone();
        let mut exp = n - 1;
        while exp > 0 {
            if exp & 1 == 1 {
                omega_n_minus_1 = omega_n_minus_1.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }

        let body_excluded = poly_mul_linear(&body, &omega_n_minus_1);

        poly_scalar_mul(&body_excluded, &ap)
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        if !self.use_extended {
            // Legacy mode: return zero polynomial (legacy uses evaluate_on_domain path)
            return vec![Scalar::zero(alpha.curve_type())];
        }

        use metavm_zkp::poly_arith::*;
        use rayon::prelude::*;
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let col = |idx: usize| -> &Vec<Scalar> { &column_coeffs[idx] };
        let two_64 = alu::two_pow_64(curve);

        let mut pairs: Vec<(Vec<Scalar>, Vec<Scalar>)> = Vec::new();

        // Constraint 0: sequential PC
        // (all non-branch/jump selectors) * (next_pc - pc - insn_len)
        {
            let mut seq = col(COL_SEL_R_ALU_ADD).clone();
            for &si in &[COL_SEL_R_ALU_SUB,
                          COL_SEL_R_AND, COL_SEL_R_OR, COL_SEL_R_XOR,
                          COL_SEL_R_SLL, COL_SEL_R_SRL, COL_SEL_R_SRA, COL_SEL_R_COMPARE,
                          COL_SEL_I_ALU_ADD,
                          COL_SEL_I_AND, COL_SEL_I_OR, COL_SEL_I_XOR,
                          COL_SEL_I_SLL, COL_SEL_I_SRL, COL_SEL_I_SRA, COL_SEL_I_COMPARE,
                          COL_SEL_W_ALU_ADD, COL_SEL_W_ALU_SUB, COL_SEL_W_ALU_ADDI,
                          COL_SEL_W_SLL, COL_SEL_W_SRL, COL_SEL_W_SRA, COL_SEL_W_ALU_OTHER,
                          COL_SEL_MUL, COL_SEL_DIV, COL_SEL_REM, COL_SEL_MULDIV_OTHER,
                          COL_SEL_LOAD, COL_SEL_STORE, COL_SEL_LUI,
                          COL_SEL_AUIPC,
                          COL_SEL_CSRRW, COL_SEL_CSRRS, COL_SEL_CSRRC,
                          COL_SEL_LR, COL_SEL_SC, COL_SEL_AMO_SWAP,
                          COL_SEL_AMO_ADD, COL_SEL_AMO_BITWISE,
                          COL_SEL_AMO_COMPARE, COL_SEL_ATOMIC_OTHER,
                          COL_SEL_MULHU,
                          COL_SEL_CSRRW_I, COL_SEL_CSRRS_I,
                          COL_SEL_MULH, COL_SEL_MULHSU, COL_SEL_MULW, COL_SEL_DIVW, COL_SEL_REMW,
                          COL_SEL_AMO_OR, COL_SEL_AMO_XOR,
                          COL_SEL_AMO_MAXU, COL_SEL_AMO_MINS, COL_SEL_AMO_MAXS] {
                seq = poly_add(&seq, col(si), curve);
            }
            let expected = poly_add(col(COL_PC), col(COL_INSN_LEN), curve);
            let body = poly_sub(col(COL_NEXT_PC), &expected, curve);
            pairs.push((seq, body));
        }

        // Constraint 1: R-type ADD main: sel_r_alu_add * (rs1 + rs2 - carry*2^64 - rd)
        {
            let sum = poly_add(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let carry_term = poly_scalar_mul(col(COL_AUX0), &two_64);
            let rhs = poly_add(&carry_term, col(COL_RD_VAL_AFTER), curve);
            let body = poly_sub(&sum, &rhs, curve);
            pairs.push((col(COL_SEL_R_ALU_ADD).clone(), body));
        }

        // Constraint 2: R-type SUB main: sel_r_alu_sub * (rd + rs2 - rs1 - borrow*2^64)
        {
            let lhs = poly_add(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let borrow_term = poly_scalar_mul(col(COL_AUX0), &two_64);
            let rhs = poly_add(col(COL_RS1_VAL), &borrow_term, curve);
            let body = poly_sub(&lhs, &rhs, curve);
            pairs.push((col(COL_SEL_R_ALU_SUB).clone(), body));
        }

        // Constraint 3: I-type ADDI main: sel_i_alu_add * (rs1 + imm - carry*2^64 - rd)
        {
            let sum = poly_add(col(COL_RS1_VAL), col(COL_IMMEDIATE), curve);
            let carry_term = poly_scalar_mul(col(COL_AUX0), &two_64);
            let rhs = poly_add(&carry_term, col(COL_RD_VAL_AFTER), curve);
            let body = poly_sub(&sum, &rhs, curve);
            pairs.push((col(COL_SEL_I_ALU_ADD).clone(), body));
        }

        // Precompute common branch sub-expressions
        let taken = poly_add(col(COL_PC), col(COL_IMMEDIATE), curve);
        let not_taken = poly_add(col(COL_PC), col(COL_INSN_LEN), curve);
        let diff_taken = poly_sub(col(COL_NEXT_PC), &taken, curve);
        let diff_not_taken = poly_sub(col(COL_NEXT_PC), &not_taken, curve);
        let target_check = poly_mul(&diff_taken, &diff_not_taken, curve);
        let one_m_aux1 = poly_sub(&one_poly, col(COL_AUX1), curve);
        let rs_diff = poly_sub(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
        let aux1_binary = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);

        // Constraint 4: BEQ condition (sel_beq * body)
        // target + taken->aux1=0 + not-taken->(1-aux1)=0 + (1-aux1)*diff=0 + aux1 binary
        {
            let c1 = poly_mul(&diff_not_taken, col(COL_AUX1), curve);
            let c2 = poly_mul(&diff_taken, &one_m_aux1, curve);
            let c3 = poly_mul(&one_m_aux1, &rs_diff, curve);
            let body = poly_add(&target_check,
                &poly_add(&c1, &poly_add(&c2, &poly_add(&c3, &aux1_binary, curve), curve), curve), curve);
            pairs.push((col(COL_SEL_BEQ).clone(), body));
        }

        // Constraint 5: BNE condition (sel_bne * body)
        {
            let c1 = poly_mul(&diff_not_taken, &one_m_aux1, curve);
            let c2 = poly_mul(&diff_taken, col(COL_AUX1), curve);
            let c3 = poly_mul(&one_m_aux1, &rs_diff, curve);
            let body = poly_add(&target_check,
                &poly_add(&c1, &poly_add(&c2, &poly_add(&c3, &aux1_binary, curve), curve), curve), curve);
            pairs.push((col(COL_SEL_BNE).clone(), body));
        }

        // Constraint 6: BLTU condition (sel_bltu * body)
        {
            let c1 = poly_mul(&diff_not_taken, &one_m_aux1, curve);
            let c2 = poly_mul(&diff_taken, col(COL_AUX1), curve);
            let body = poly_add(&target_check,
                &poly_add(&c1, &poly_add(&c2, &aux1_binary, curve), curve), curve);
            pairs.push((col(COL_SEL_BLTU).clone(), body));
        }

        // Constraint 7: BGEU condition (sel_bgeu * body)
        {
            let c1 = poly_mul(&diff_not_taken, col(COL_AUX1), curve);
            let c2 = poly_mul(&diff_taken, &one_m_aux1, curve);
            let body = poly_add(&target_check,
                &poly_add(&c1, &poly_add(&c2, &aux1_binary, curve), curve), curve);
            pairs.push((col(COL_SEL_BGEU).clone(), body));
        }

        // Precompute signed branch sub-expressions
        let two_s = Scalar::from_u64(2, curve);
        let one_m_aux2 = poly_sub(&one_poly, col(COL_AUX2), curve);
        let aux0_binary = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
        let aux1_sign_binary = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
        let aux2_sign_binary = poly_mul(col(COL_AUX2), &poly_sub(col(COL_AUX2), &one_poly, curve), curve);
        // signed_lt = aux1*(1-aux2) + (1-aux1-aux2+2*aux1*aux2)*aux0
        let term1_signed = poly_mul(col(COL_AUX1), &one_m_aux2, curve);
        let aux1_aux2 = poly_mul(col(COL_AUX1), col(COL_AUX2), curve);
        let same_sign = poly_add(
            &poly_sub(&poly_sub(&one_poly, col(COL_AUX1), curve), col(COL_AUX2), curve),
            &poly_scalar_mul(&aux1_aux2, &two_s),
            curve,
        );
        let term2_signed = poly_mul(&same_sign, col(COL_AUX0), curve);
        let signed_lt = poly_add(&term1_signed, &term2_signed, curve);
        let one_m_signed_lt = poly_sub(&one_poly, &signed_lt, curve);

        // Constraint 8: BLT condition (sel_blt * body)
        // taken iff signed_lt=1: diff_not_taken*(1-signed_lt)=0, diff_taken*signed_lt=0
        {
            let c1 = poly_mul(&diff_not_taken, &one_m_signed_lt, curve);
            let c2 = poly_mul(&diff_taken, &signed_lt, curve);
            let body = poly_add(&target_check,
                &poly_add(&c1,
                    &poly_add(&c2,
                        &poly_add(&aux0_binary,
                            &poly_add(&aux1_sign_binary, &aux2_sign_binary, curve),
                        curve),
                    curve),
                curve),
            curve);
            pairs.push((col(COL_SEL_BLT).clone(), body));
        }

        // Constraint 9: BGE condition (sel_bge * body)
        // taken iff signed_lt=0: diff_not_taken*signed_lt=0, diff_taken*(1-signed_lt)=0
        {
            let c1 = poly_mul(&diff_not_taken, &signed_lt, curve);
            let c2 = poly_mul(&diff_taken, &one_m_signed_lt, curve);
            let body = poly_add(&target_check,
                &poly_add(&c1,
                    &poly_add(&c2,
                        &poly_add(&aux0_binary,
                            &poly_add(&aux1_sign_binary, &aux2_sign_binary, curve),
                        curve),
                    curve),
                curve),
            curve);
            pairs.push((col(COL_SEL_BGE).clone(), body));
        }

        // Constraint 10: JAL body: rd = pc + insn_len, next_pc = pc + immediate
        {
            let rd_check = poly_sub(col(COL_RD_VAL_AFTER), &not_taken, curve);
            let pc_check = poly_sub(col(COL_NEXT_PC), &taken, curve);
            let body = poly_add(&rd_check, &pc_check, curve);
            pairs.push((col(COL_SEL_JAL).clone(), body));
        }

        // Constraint 11: JALR body: rd = pc+insn_len, next_pc = (rs1+imm)-aux1, aux1 binary
        {
            let rd_check = poly_sub(col(COL_RD_VAL_AFTER), &not_taken, curve);
            let raw_target = poly_add(col(COL_RS1_VAL), col(COL_IMMEDIATE), curve);
            let pc_check = poly_sub(col(COL_NEXT_PC), &poly_sub(&raw_target, col(COL_AUX1), curve), curve);
            let body = poly_add(&rd_check, &poly_add(&pc_check, &aux1_binary, curve), curve);
            pairs.push((col(COL_SEL_JALR).clone(), body));
        }

        // Constraint 12: LUI body: rd = immediate
        {
            let body = poly_sub(col(COL_RD_VAL_AFTER), col(COL_IMMEDIATE), curve);
            pairs.push((col(COL_SEL_LUI).clone(), body));
        }

        // Constraint 13: AUIPC body: rd = pc + immediate
        {
            let expected = poly_add(col(COL_PC), col(COL_IMMEDIATE), curve);
            let body = poly_sub(col(COL_RD_VAL_AFTER), &expected, curve);
            pairs.push((col(COL_SEL_AUIPC).clone(), body));
        }

        // Constraint 14: memory (sel_load + sel_store) * body
        {
            let sel = poly_add(col(COL_SEL_LOAD), col(COL_SEL_STORE), curve);
            let expected_addr = poly_add(col(COL_RS1_VAL), col(COL_IMMEDIATE), curve);
            let body = poly_sub(col(COL_MEM_ADDR), &expected_addr, curve);
            pairs.push((sel, body));
        }

        // Constraint 14: R-ADD carry binary: sel_r_alu_add * aux0*(aux0-1)
        {
            let binary = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            pairs.push((col(COL_SEL_R_ALU_ADD).clone(), binary));
        }

        // Constraint 15: R-SUB borrow binary: sel_r_alu_sub * aux0*(aux0-1)
        {
            let binary = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            pairs.push((col(COL_SEL_R_ALU_SUB).clone(), binary));
        }

        // Constraint 16: I-ADDI carry binary: sel_i_alu_add * aux0*(aux0-1)
        {
            let binary = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            pairs.push((col(COL_SEL_I_ALU_ADD).clone(), binary));
        }

        // Constraint 17: W-type ADDW: sel_w_alu_add * (rs1+rs2 - aux0*2^32 - rd + aux1*(2^64-2^32) + aux1*(aux1-1))
        {
            let two_32 = Scalar::from_u64(1u64 << 32, curve);
            let sign_ext_offset = two_64.sub(&two_32);
            let sum = poly_add(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let upper = poly_scalar_mul(col(COL_AUX0), &two_32);
            let sign_term = poly_scalar_mul(col(COL_AUX1), &sign_ext_offset);
            let c1 = poly_add(&poly_sub(&poly_sub(&sum, &upper, curve), col(COL_RD_VAL_AFTER), curve), &sign_term, curve);
            let aux1_binary = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let body = poly_add(&c1, &aux1_binary, curve);
            pairs.push((col(COL_SEL_W_ALU_ADD).clone(), body));
        }

        // Constraint 18: W-type SUBW: sel_w_alu_sub * (rs1-rs2 - aux0*2^32 - rd + aux1*(2^64-2^32) + aux1*(aux1-1))
        {
            let two_32 = Scalar::from_u64(1u64 << 32, curve);
            let sign_ext_offset = two_64.sub(&two_32);
            let diff = poly_sub(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let upper = poly_scalar_mul(col(COL_AUX0), &two_32);
            let sign_term = poly_scalar_mul(col(COL_AUX1), &sign_ext_offset);
            let c1 = poly_add(&poly_sub(&poly_sub(&diff, &upper, curve), col(COL_RD_VAL_AFTER), curve), &sign_term, curve);
            let aux1_binary = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let body = poly_add(&c1, &aux1_binary, curve);
            pairs.push((col(COL_SEL_W_ALU_SUB).clone(), body));
        }

        // Constraint 19: W-type ADDIW: sel_w_alu_addi * (rs1+imm - aux0*2^32 - rd + aux1*(2^64-2^32) + aux1*(aux1-1))
        {
            let two_32 = Scalar::from_u64(1u64 << 32, curve);
            let sign_ext_offset = two_64.sub(&two_32);
            let sum = poly_add(col(COL_RS1_VAL), col(COL_IMMEDIATE), curve);
            let upper = poly_scalar_mul(col(COL_AUX0), &two_32);
            let sign_term = poly_scalar_mul(col(COL_AUX1), &sign_ext_offset);
            let c1 = poly_add(&poly_sub(&poly_sub(&sum, &upper, curve), col(COL_RD_VAL_AFTER), curve), &sign_term, curve);
            let aux1_binary = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let body = poly_add(&c1, &aux1_binary, curve);
            pairs.push((col(COL_SEL_W_ALU_ADDI).clone(), body));
        }

        // Constraint 20: MUL: sel_mul * (rs1*rs2 - aux0*2^64 - rd)
        {
            let product = poly_mul(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let hi_term = poly_scalar_mul(col(COL_AUX0), &two_64);
            let rhs = poly_add(&hi_term, col(COL_RD_VAL_AFTER), curve);
            let body = poly_sub(&product, &rhs, curve);
            pairs.push((col(COL_SEL_MUL).clone(), body));
        }

        // Constraint 21: DIV/DIVU: sel_div * (rd*rs2 + aux0 - rs1)
        {
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let lhs = poly_add(&prod, col(COL_AUX0), curve);
            let body = poly_sub(&lhs, col(COL_RS1_VAL), curve);
            pairs.push((col(COL_SEL_DIV).clone(), body));
        }

        // Constraint 22: REM/REMU: sel_rem * (aux0*rs2 + rd - rs1)
        {
            let prod = poly_mul(col(COL_AUX0), col(COL_RS2_VAL), curve);
            let lhs = poly_add(&prod, col(COL_RD_VAL_AFTER), curve);
            let body = poly_sub(&lhs, col(COL_RS1_VAL), curve);
            pairs.push((col(COL_SEL_REM).clone(), body));
        }

        // Constraint 23: R-AND: sel_r_and * (rd_val_after - aux0)
        // aux0 = AND(rs1, rs2), so rd_val_after must equal aux0
        {
            let body = poly_sub(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            pairs.push((col(COL_SEL_R_AND).clone(), body));
        }

        // Constraint 24: R-OR: sel_r_or * (rd_val_after - rs1 - rs2 + aux0)
        // OR(a,b) = a + b - AND(a,b), so rd = rs1 + rs2 - aux0
        {
            let expected = poly_sub(&poly_add(col(COL_RS1_VAL), col(COL_RS2_VAL), curve), col(COL_AUX0), curve);
            let body = poly_sub(col(COL_RD_VAL_AFTER), &expected, curve);
            pairs.push((col(COL_SEL_R_OR).clone(), body));
        }

        // Constraint 25: R-XOR: sel_r_xor * (rd_val_after - rs1 - rs2 + 2*aux0)
        // XOR(a,b) = a + b - 2*AND(a,b), so rd = rs1 + rs2 - 2*aux0
        {
            let two = Scalar::from_u64(2, curve);
            let two_aux0 = poly_scalar_mul(col(COL_AUX0), &two);
            let expected = poly_sub(&poly_add(col(COL_RS1_VAL), col(COL_RS2_VAL), curve), &two_aux0, curve);
            let body = poly_sub(col(COL_RD_VAL_AFTER), &expected, curve);
            pairs.push((col(COL_SEL_R_XOR).clone(), body));
        }

        // Constraint 26: I-AND: sel_i_and * (rd_val_after - aux0)
        // aux0 = AND(rs1, immediate), so rd_val_after must equal aux0
        {
            let body = poly_sub(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            pairs.push((col(COL_SEL_I_AND).clone(), body));
        }

        // Constraint 27: I-OR: sel_i_or * (rd_val_after - rs1 - immediate + aux0)
        // OR(a,b) = a + b - AND(a,b), so rd = rs1 + imm - aux0
        {
            let expected = poly_sub(&poly_add(col(COL_RS1_VAL), col(COL_IMMEDIATE), curve), col(COL_AUX0), curve);
            let body = poly_sub(col(COL_RD_VAL_AFTER), &expected, curve);
            pairs.push((col(COL_SEL_I_OR).clone(), body));
        }

        // Constraint 28: I-XOR: sel_i_xor * (rd_val_after - rs1 - immediate + 2*aux0)
        // XOR(a,b) = a + b - 2*AND(a,b), so rd = rs1 + imm - 2*aux0
        {
            let two = Scalar::from_u64(2, curve);
            let two_aux0 = poly_scalar_mul(col(COL_AUX0), &two);
            let expected = poly_sub(&poly_add(col(COL_RS1_VAL), col(COL_IMMEDIATE), curve), &two_aux0, curve);
            let body = poly_sub(col(COL_RD_VAL_AFTER), &expected, curve);
            pairs.push((col(COL_SEL_I_XOR).clone(), body));
        }

        // Constraint 29: R-SLL: sel_r_sll * (rs1 * aux0 - aux1 * 2^64 - rd)
        {
            let product = poly_mul(col(COL_RS1_VAL), col(COL_AUX0), curve);
            let hi_term = poly_scalar_mul(col(COL_AUX1), &two_64);
            let rhs = poly_add(&hi_term, col(COL_RD_VAL_AFTER), curve);
            let body = poly_sub(&product, &rhs, curve);
            pairs.push((col(COL_SEL_R_SLL).clone(), body));
        }

        // Constraint 30: R-SRL: sel_r_srl * (rd * aux0 + aux1 - rs1)
        {
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            let lhs = poly_add(&prod, col(COL_AUX1), curve);
            let body = poly_sub(&lhs, col(COL_RS1_VAL), curve);
            pairs.push((col(COL_SEL_R_SRL).clone(), body));
        }

        // Constraint 31: R-SRA: sel_r_sra * (rd * aux0 + aux1 - rs1 - aux2 * 2^64)
        {
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            let lhs = poly_add(&prod, col(COL_AUX1), curve);
            let overflow_term = poly_scalar_mul(col(COL_AUX2), &two_64);
            let rhs = poly_add(col(COL_RS1_VAL), &overflow_term, curve);
            let body = poly_sub(&lhs, &rhs, curve);
            pairs.push((col(COL_SEL_R_SRA).clone(), body));
        }

        // Constraint 32: I-SLL: sel_i_sll * (rs1 * aux0 - aux1 * 2^64 - rd)
        {
            let product = poly_mul(col(COL_RS1_VAL), col(COL_AUX0), curve);
            let hi_term = poly_scalar_mul(col(COL_AUX1), &two_64);
            let rhs = poly_add(&hi_term, col(COL_RD_VAL_AFTER), curve);
            let body = poly_sub(&product, &rhs, curve);
            pairs.push((col(COL_SEL_I_SLL).clone(), body));
        }

        // Constraint 33: I-SRL: sel_i_srl * (rd * aux0 + aux1 - rs1)
        {
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            let lhs = poly_add(&prod, col(COL_AUX1), curve);
            let body = poly_sub(&lhs, col(COL_RS1_VAL), curve);
            pairs.push((col(COL_SEL_I_SRL).clone(), body));
        }

        // Constraint 34: I-SRA: sel_i_sra * (rd * aux0 + aux1 - rs1 - aux2 * 2^64)
        {
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            let lhs = poly_add(&prod, col(COL_AUX1), curve);
            let overflow_term = poly_scalar_mul(col(COL_AUX2), &two_64);
            let rhs = poly_add(col(COL_RS1_VAL), &overflow_term, curve);
            let body = poly_sub(&lhs, &rhs, curve);
            pairs.push((col(COL_SEL_I_SRA).clone(), body));
        }

        // W-type shifts use TWO_32 = 2^32 instead of TWO_64
        let two_32 = Scalar::from_u64(1u64 << 32, curve);

        // Constraint 35: W-SLL: sel_w_sll * (rs1_low * aux0 - aux1 * 2^32 - rd_low)
        // rs1_low = rs1 mod 2^32 (ensured by trace), aux0 = 2^k, aux1 = overflow
        // rd is sign-extended 32-bit result, constraint verifies 32-bit multiplication
        {
            let product = poly_mul(col(COL_RS1_VAL), col(COL_AUX0), curve);
            let hi_term = poly_scalar_mul(col(COL_AUX1), &two_32);
            let rhs = poly_add(&hi_term, col(COL_RD_VAL_AFTER), curve);
            let body = poly_sub(&product, &rhs, curve);
            pairs.push((col(COL_SEL_W_SLL).clone(), body));
        }

        // Constraint 36: W-SRL: sel_w_srl * (rd * aux0 + aux1 - rs1_low)
        {
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            let lhs = poly_add(&prod, col(COL_AUX1), curve);
            let body = poly_sub(&lhs, col(COL_RS1_VAL), curve);
            pairs.push((col(COL_SEL_W_SRL).clone(), body));
        }

        // Constraint 37: W-SRA: sel_w_sra * (rd * aux0 + aux1 - rs1_low - aux2 * 2^32)
        {
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_AUX0), curve);
            let lhs = poly_add(&prod, col(COL_AUX1), curve);
            let overflow_term = poly_scalar_mul(col(COL_AUX2), &two_32);
            let rhs = poly_add(col(COL_RS1_VAL), &overflow_term, curve);
            let body = poly_sub(&lhs, &rhs, curve);
            pairs.push((col(COL_SEL_W_SRA).clone(), body));
        }

        // Constraint 39: R-type compare (SLT/SLTU): sel_r_compare * compare_body
        // signed_lt = aux1*(1-aux2) + (1-aux1-aux2+2*aux1*aux2)*aux0
        // body = (rd - signed_lt) + aux0*(aux0-1) + aux1*(aux1-1) + aux2*(aux2-1)
        {
            let two = Scalar::from_u64(2, curve);
            let one_m_aux2 = poly_sub(&one_poly, col(COL_AUX2), curve);
            let term1 = poly_mul(col(COL_AUX1), &one_m_aux2, curve);
            let a1_a2 = poly_mul(col(COL_AUX1), col(COL_AUX2), curve);
            let same_sign = poly_add(
                &poly_sub(&poly_sub(&one_poly, col(COL_AUX1), curve), col(COL_AUX2), curve),
                &poly_scalar_mul(&a1_a2, &two),
                curve,
            );
            let term2 = poly_mul(&same_sign, col(COL_AUX0), curve);
            let signed_lt = poly_add(&term1, &term2, curve);
            let rd_check = poly_sub(col(COL_RD_VAL_AFTER), &signed_lt, curve);
            let a0_bin = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            let a1_bin = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let a2_bin = poly_mul(col(COL_AUX2), &poly_sub(col(COL_AUX2), &one_poly, curve), curve);
            let body = poly_add(&rd_check, &poly_add(&a0_bin, &poly_add(&a1_bin, &a2_bin, curve), curve), curve);
            pairs.push((col(COL_SEL_R_COMPARE).clone(), body));
        }

        // Constraint 40: I-type compare (SLTI/SLTIU): sel_i_compare * compare_body
        // Same body as R-type compare (aux values differ in trace generation)
        {
            let two = Scalar::from_u64(2, curve);
            let one_m_aux2 = poly_sub(&one_poly, col(COL_AUX2), curve);
            let term1 = poly_mul(col(COL_AUX1), &one_m_aux2, curve);
            let a1_a2 = poly_mul(col(COL_AUX1), col(COL_AUX2), curve);
            let same_sign = poly_add(
                &poly_sub(&poly_sub(&one_poly, col(COL_AUX1), curve), col(COL_AUX2), curve),
                &poly_scalar_mul(&a1_a2, &two),
                curve,
            );
            let term2 = poly_mul(&same_sign, col(COL_AUX0), curve);
            let signed_lt = poly_add(&term1, &term2, curve);
            let rd_check = poly_sub(col(COL_RD_VAL_AFTER), &signed_lt, curve);
            let a0_bin = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            let a1_bin = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let a2_bin = poly_mul(col(COL_AUX2), &poly_sub(col(COL_AUX2), &one_poly, curve), curve);
            let body = poly_add(&rd_check, &poly_add(&a0_bin, &poly_add(&a1_bin, &a2_bin, curve), curve), curve);
            pairs.push((col(COL_SEL_I_COMPARE).clone(), body));
        }

        // Constraint 41: CSRRC: sel_csrrc * (aux0 - rd_val_after + aux1)
        // Verifies: new_csr = old & ~source, where aux0 = old - AND(old, source), aux1 = AND(old, source)
        // Works for both CSRRC (R-variant) and CSRRCI (I-variant) since source cancels out.
        {
            let body = poly_sub(col(COL_AUX0), col(COL_RD_VAL_AFTER), curve);
            let body = poly_add(&body, col(COL_AUX1), curve);
            pairs.push((col(COL_SEL_CSRRC).clone(), body));
        }

        // Constraint 42: LR: sel_lr * (rd_val_after - mem_val)
        // Load reserved: the loaded value must match the captured memory value.
        {
            let body = poly_sub(col(COL_RD_VAL_AFTER), col(COL_MEM_VAL), curve);
            pairs.push((col(COL_SEL_LR).clone(), body));
        }

        // Constraint 43: SC: sel_sc * (aux0*(aux0-1) + aux0*rd_val_after)
        // Store conditional: aux0 is binary reservation flag, valid SC requires rd=0.
        {
            let a0_m1 = poly_sub(col(COL_AUX0), &one_poly, curve);
            let binary = poly_mul(col(COL_AUX0), &a0_m1, curve);
            let valid_check = poly_mul(col(COL_AUX0), col(COL_RD_VAL_AFTER), curve);
            let body = poly_add(&binary, &valid_check, curve);
            pairs.push((col(COL_SEL_SC).clone(), body));
        }

        // Constraint 44: MULHU: sel_mulhu * (rs1*rs2 - rd*2^64 - aux0)
        // 128-bit unsigned product = (high 64 bits in rd) * 2^64 + (low 64 bits in aux0)
        {
            let product = poly_mul(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let hi_term = poly_scalar_mul(col(COL_RD_VAL_AFTER), &two_64);
            let rhs = poly_add(&hi_term, col(COL_AUX0), curve);
            let body = poly_sub(&product, &rhs, curve);
            pairs.push((col(COL_SEL_MULHU).clone(), body));
        }

        // Constraint 45: CSRRW_R: sel_csrrw * (aux0 - rs1_val)
        {
            let body = poly_sub(col(COL_AUX0), col(COL_RS1_VAL), curve);
            pairs.push((col(COL_SEL_CSRRW).clone(), body));
        }

        // Constraint 46: CSRRW_I: sel_csrrw_i * (aux0 - immediate)
        {
            let body = poly_sub(col(COL_AUX0), col(COL_IMMEDIATE), curve);
            pairs.push((col(COL_SEL_CSRRW_I).clone(), body));
        }

        // Constraint 47: CSRRS_R: sel_csrrs * (aux0 - rd_val_after - rs1_val + aux1)
        {
            let body = poly_add(
                &poly_sub(&poly_sub(col(COL_AUX0), col(COL_RD_VAL_AFTER), curve), col(COL_RS1_VAL), curve),
                col(COL_AUX1), curve);
            pairs.push((col(COL_SEL_CSRRS).clone(), body));
        }

        // Constraint 48: CSRRS_I: sel_csrrs_i * (aux0 - rd_val_after - immediate + aux1)
        {
            let body = poly_add(
                &poly_sub(&poly_sub(col(COL_AUX0), col(COL_RD_VAL_AFTER), curve), col(COL_IMMEDIATE), curve),
                col(COL_AUX1), curve);
            pairs.push((col(COL_SEL_CSRRS_I).clone(), body));
        }

        // Constraint 49: MULH: sel_mulh * (rs1*rs2 - aux1*2^64*rs2 - aux2*2^64*rs1 + aux1*aux2*2^128 - rd*2^64 - aux0 + aux1*(aux1-1) + aux2*(aux2-1))
        {
            let two_128 = poly_scalar_mul(&one_poly, &two_64.mul(&two_64));
            let product = poly_mul(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let sa_rs2 = poly_mul(col(COL_AUX1), &poly_scalar_mul(col(COL_RS2_VAL), &two_64), curve);
            let sb_rs1 = poly_mul(col(COL_AUX2), &poly_scalar_mul(col(COL_RS1_VAL), &two_64), curve);
            let sa_sb = poly_mul(col(COL_AUX1), col(COL_AUX2), curve);
            let sa_sb_2128 = poly_mul(&sa_sb, &two_128, curve);
            let hi_term = poly_scalar_mul(col(COL_RD_VAL_AFTER), &two_64);
            let main = poly_sub(&poly_sub(&poly_add(&poly_sub(&product, &sa_rs2, curve), &sa_sb_2128, curve), &sb_rs1, curve), &poly_add(&hi_term, col(COL_AUX0), curve), curve);
            let a1_bin = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let a2_bin = poly_mul(col(COL_AUX2), &poly_sub(col(COL_AUX2), &one_poly, curve), curve);
            let body = poly_add(&main, &poly_add(&a1_bin, &a2_bin, curve), curve);
            pairs.push((col(COL_SEL_MULH).clone(), body));
        }

        // Constraint 50: MULHSU: sel_mulhsu * (rs1*rs2 - aux1*2^64*rs2 - rd*2^64 - aux0 + aux1*(aux1-1))
        {
            let product = poly_mul(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let sa_rs2 = poly_mul(col(COL_AUX1), &poly_scalar_mul(col(COL_RS2_VAL), &two_64), curve);
            let hi_term = poly_scalar_mul(col(COL_RD_VAL_AFTER), &two_64);
            let main = poly_sub(&poly_sub(&product, &sa_rs2, curve), &poly_add(&hi_term, col(COL_AUX0), curve), curve);
            let a1_bin = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let body = poly_add(&main, &a1_bin, curve);
            pairs.push((col(COL_SEL_MULHSU).clone(), body));
        }

        // Constraint 51: MULW: sel_mulw * (rs1*rs2 - aux0*2^32 - rd + aux1*(2^64-2^32) + aux1*(aux1-1))
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let sign_ext_offset = two_64.sub(&two_32_c);
            let product = poly_mul(col(COL_RS1_VAL), col(COL_RS2_VAL), curve);
            let upper = poly_scalar_mul(col(COL_AUX0), &two_32_c);
            let sign_term = poly_scalar_mul(col(COL_AUX1), &sign_ext_offset);
            let c1 = poly_add(&poly_sub(&poly_sub(&product, &upper, curve), col(COL_RD_VAL_AFTER), curve), &sign_term, curve);
            let a1_bin = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let body = poly_add(&c1, &a1_bin, curve);
            pairs.push((col(COL_SEL_MULW).clone(), body));
        }

        // Constraint 52: DIVW: sel_divw * (rd*rs2 + aux0 - rs1 - aux1*2^32)
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let prod = poly_mul(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let correction = poly_scalar_mul(col(COL_AUX1), &two_32_c);
            let body = poly_sub(&poly_sub(&poly_add(&prod, col(COL_AUX0), curve), col(COL_RS1_VAL), curve), &correction, curve);
            pairs.push((col(COL_SEL_DIVW).clone(), body));
        }

        // Constraint 53: REMW: sel_remw * (aux0*rs2 + rd - rs1 - aux1*2^32)
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let prod = poly_mul(col(COL_AUX0), col(COL_RS2_VAL), curve);
            let correction = poly_scalar_mul(col(COL_AUX1), &two_32_c);
            let body = poly_sub(&poly_sub(&poly_add(&prod, col(COL_RD_VAL_AFTER), curve), col(COL_RS1_VAL), curve), &correction, curve);
            pairs.push((col(COL_SEL_REMW).clone(), body));
        }

        // Constraint 54: AMO_SWAP: sel_amo_swap * (rs2 - aux0*2^32 - mem_val + aux1*(2^64-2^32) + aux1*(aux1-1))
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let sign_ext_offset = two_64.sub(&two_32_c);
            let c1 = poly_add(
                &poly_sub(&poly_sub(col(COL_RS2_VAL), &poly_scalar_mul(col(COL_AUX0), &two_32_c), curve), col(COL_MEM_VAL), curve),
                &poly_scalar_mul(col(COL_AUX1), &sign_ext_offset), curve);
            let a1_bin = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let body = poly_add(&c1, &a1_bin, curve);
            pairs.push((col(COL_SEL_AMO_SWAP).clone(), body));
        }

        // Constraint 55: AMO_ADD: sel_amo_add * (rd+rs2-mem_val - aux0*(aux2*2^32+(1-aux2)*2^64) + aux1*aux2*(2^64-2^32) + binaries)
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let sign_ext_offset = two_64.sub(&two_32_c);
            let one_m_aux2 = poly_sub(&one_poly, col(COL_AUX2), curve);
            let modulus = poly_add(&poly_scalar_mul(col(COL_AUX2), &two_32_c), &poly_scalar_mul(&one_m_aux2, &two_64), curve);
            let correction_term = poly_mul(col(COL_AUX0), &modulus, curve);
            let sum = poly_add(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let sign_corr = poly_mul(&poly_mul(col(COL_AUX1), col(COL_AUX2), curve), &poly_scalar_mul(&one_poly, &sign_ext_offset), curve);
            let c1 = poly_add(&poly_sub(&poly_sub(&sum, col(COL_MEM_VAL), curve), &correction_term, curve), &sign_corr, curve);
            let a1_bin = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let a2_bin = poly_mul(col(COL_AUX2), &poly_sub(col(COL_AUX2), &one_poly, curve), curve);
            let a0_d_bin = poly_mul(&one_m_aux2, &poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve), curve);
            let body = poly_add(&c1, &poly_add(&a1_bin, &poly_add(&a2_bin, &a0_d_bin, curve), curve), curve);
            pairs.push((col(COL_SEL_AMO_ADD).clone(), body));
        }

        // Constraint 56: AMO_AND: sel_amo_bitwise * (mem_val + aux0 - rd - rs2 - aux1*2^32)
        // NOTE: COL_SEL_AMO_BITWISE is now AMO_AND only
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let body = poly_sub(
                &poly_sub(&poly_add(col(COL_MEM_VAL), col(COL_AUX0), curve), col(COL_RD_VAL_AFTER), curve),
                &poly_add(col(COL_RS2_VAL), &poly_scalar_mul(col(COL_AUX1), &two_32_c), curve), curve);
            pairs.push((col(COL_SEL_AMO_BITWISE).clone(), body));
        }

        // Constraint 57: AMO_OR: sel_amo_or * (mem_val + aux0 - rd - rs2 - aux1*2^32)
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let body = poly_sub(
                &poly_sub(&poly_add(col(COL_MEM_VAL), col(COL_AUX0), curve), col(COL_RD_VAL_AFTER), curve),
                &poly_add(col(COL_RS2_VAL), &poly_scalar_mul(col(COL_AUX1), &two_32_c), curve), curve);
            pairs.push((col(COL_SEL_AMO_OR).clone(), body));
        }

        // Constraint 58: AMO_XOR: sel_amo_xor * (mem_val + 2*aux0 - rd - rs2 - aux1*2^32)
        {
            let two = Scalar::from_u64(2, curve);
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let two_aux0 = poly_scalar_mul(col(COL_AUX0), &two);
            let body = poly_sub(
                &poly_sub(&poly_add(col(COL_MEM_VAL), &two_aux0, curve), col(COL_RD_VAL_AFTER), curve),
                &poly_add(col(COL_RS2_VAL), &poly_scalar_mul(col(COL_AUX1), &two_32_c), curve), curve);
            pairs.push((col(COL_SEL_AMO_XOR).clone(), body));
        }

        // Constraint 59: AMO_MINU: sel_amo_compare * (mem_val - rs2 - aux0*(rd-rs2) - aux1*2^32 + aux0*(aux0-1))
        // NOTE: COL_SEL_AMO_COMPARE is now AMO_MINU only
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let rd_m_rs2 = poly_sub(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let selection = poly_mul(col(COL_AUX0), &rd_m_rs2, curve);
            let correction = poly_scalar_mul(col(COL_AUX1), &two_32_c);
            let c1 = poly_sub(&poly_sub(&poly_sub(col(COL_MEM_VAL), col(COL_RS2_VAL), curve), &selection, curve), &correction, curve);
            let a0_bin = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            let body = poly_add(&c1, &a0_bin, curve);
            pairs.push((col(COL_SEL_AMO_COMPARE).clone(), body));
        }

        // Constraint 60: AMO_MAXU: sel_amo_maxu * (mem_val - rd + aux0*(rd-rs2) - aux1*2^32 + aux0*(aux0-1))
        {
            let two_32_c = Scalar::from_u64(1u64 << 32, curve);
            let rd_m_rs2 = poly_sub(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let selection = poly_mul(col(COL_AUX0), &rd_m_rs2, curve);
            let correction = poly_scalar_mul(col(COL_AUX1), &two_32_c);
            let c1 = poly_sub(&poly_add(&poly_sub(col(COL_MEM_VAL), col(COL_RD_VAL_AFTER), curve), &selection, curve), &correction, curve);
            let a0_bin = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            let body = poly_add(&c1, &a0_bin, curve);
            pairs.push((col(COL_SEL_AMO_MAXU).clone(), body));
        }

        // Constraint 61: AMO_MINS: sel_amo_mins * (mem_val - rs2 - signed_lt*(rd-rs2) + binaries)
        {
            let two_s = Scalar::from_u64(2, curve);
            let one_m_a2 = poly_sub(&one_poly, col(COL_AUX2), curve);
            let t1 = poly_mul(col(COL_AUX1), &one_m_a2, curve);
            let a1a2 = poly_mul(col(COL_AUX1), col(COL_AUX2), curve);
            let ss = poly_add(
                &poly_sub(&poly_sub(&one_poly, col(COL_AUX1), curve), col(COL_AUX2), curve),
                &poly_scalar_mul(&a1a2, &two_s), curve);
            let t2 = poly_mul(&ss, col(COL_AUX0), curve);
            let slt = poly_add(&t1, &t2, curve);
            let rd_m_rs2 = poly_sub(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let sel_term = poly_mul(&slt, &rd_m_rs2, curve);
            let c1 = poly_sub(&poly_sub(col(COL_MEM_VAL), col(COL_RS2_VAL), curve), &sel_term, curve);
            let a0b = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            let a1b = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let a2b = poly_mul(col(COL_AUX2), &poly_sub(col(COL_AUX2), &one_poly, curve), curve);
            let body = poly_add(&c1, &poly_add(&a0b, &poly_add(&a1b, &a2b, curve), curve), curve);
            pairs.push((col(COL_SEL_AMO_MINS).clone(), body));
        }

        // Constraint 62: AMO_MAXS: sel_amo_maxs * (mem_val - rd + signed_lt*(rd-rs2) + binaries)
        {
            let two_s = Scalar::from_u64(2, curve);
            let one_m_a2 = poly_sub(&one_poly, col(COL_AUX2), curve);
            let t1 = poly_mul(col(COL_AUX1), &one_m_a2, curve);
            let a1a2 = poly_mul(col(COL_AUX1), col(COL_AUX2), curve);
            let ss = poly_add(
                &poly_sub(&poly_sub(&one_poly, col(COL_AUX1), curve), col(COL_AUX2), curve),
                &poly_scalar_mul(&a1a2, &two_s), curve);
            let t2 = poly_mul(&ss, col(COL_AUX0), curve);
            let slt = poly_add(&t1, &t2, curve);
            let rd_m_rs2 = poly_sub(col(COL_RD_VAL_AFTER), col(COL_RS2_VAL), curve);
            let sel_term = poly_mul(&slt, &rd_m_rs2, curve);
            let c1 = poly_add(&poly_sub(col(COL_MEM_VAL), col(COL_RD_VAL_AFTER), curve), &sel_term, curve);
            let a0b = poly_mul(col(COL_AUX0), &poly_sub(col(COL_AUX0), &one_poly, curve), curve);
            let a1b = poly_mul(col(COL_AUX1), &poly_sub(col(COL_AUX1), &one_poly, curve), curve);
            let a2b = poly_mul(col(COL_AUX2), &poly_sub(col(COL_AUX2), &one_poly, curve), curve);
            let body = poly_add(&c1, &poly_add(&a0b, &poly_add(&a1b, &a2b, curve), curve), curve);
            pairs.push((col(COL_SEL_AMO_MAXS).clone(), body));
        }

        // Phase 2: Parallel gating (poly_mul of selector * body for all constraint pairs)
        let gated_results: Vec<Vec<Scalar>> = pairs.par_iter()
            .map(|(sel, body)| poly_mul(sel, body, curve))
            .collect();

        // Phase 3: Sequential accumulation with alpha powers
        let mut c = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for gated in &gated_results {
            c = poly_add(&c, &poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

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

        // Phase 5: Sum-to-one (sum_of_selectors - 1)
        let mut sel_sum = vec![Scalar::zero(curve)];
        for &si in &sel_indices {
            sel_sum = poly_add(&sel_sum, col(si), curve);
        }
        if !sel_sum.is_empty() {
            sel_sum[0] = sel_sum[0].sub(&Scalar::one(curve));
        }
        c = poly_add(&c, &poly_scalar_mul(&sel_sum, &ap), curve);

        c
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        if !self.use_extended {
            return LookupRequirements::none();
        }

        let range_16 = LookupTable::range(16);
        let mut decls = Vec::new();

        // MUL: aux0 holds upper 64 bits of 128-bit product.
        // Without range check, a malicious prover can pick aux0 ∉ [0, 2^64)
        // and forge an incorrect multiplication result.
        decls.push((LookupDeclaration {
            label: "mul_aux0_64bit".to_string(),
            column_index: COL_AUX0,
            max_bits: 64,
            selector_column: Some(COL_SEL_MUL),
        }, 0));

        // MULHU: aux0 holds lower 64 bits of 128-bit product.
        // Same range-check requirement as MUL's aux0.
        decls.push((LookupDeclaration {
            label: "mulhu_aux0_64bit".to_string(),
            column_index: COL_AUX0,
            max_bits: 64,
            selector_column: Some(COL_SEL_MULHU),
        }, 0));

        // DIV: aux0 holds remainder. Without 0 ≤ remainder < divisor,
        // the quotient is not unique.
        decls.push((LookupDeclaration {
            label: "div_aux0_64bit".to_string(),
            column_index: COL_AUX0,
            max_bits: 64,
            selector_column: Some(COL_SEL_DIV),
        }, 0));

        // REM: aux0 holds quotient (same issue as DIV).
        decls.push((LookupDeclaration {
            label: "rem_aux0_64bit".to_string(),
            column_index: COL_AUX0,
            max_bits: 64,
            selector_column: Some(COL_SEL_REM),
        }, 0));

        // W-type ALU: aux0 holds upper 32 bits of result before truncation.
        decls.push((LookupDeclaration {
            label: "addw_aux0_32bit".to_string(),
            column_index: COL_AUX0,
            max_bits: 32,
            selector_column: Some(COL_SEL_W_ALU_ADD),
        }, 0));
        decls.push((LookupDeclaration {
            label: "subw_aux0_32bit".to_string(),
            column_index: COL_AUX0,
            max_bits: 32,
            selector_column: Some(COL_SEL_W_ALU_SUB),
        }, 0));
        decls.push((LookupDeclaration {
            label: "addiw_aux0_32bit".to_string(),
            column_index: COL_AUX0,
            max_bits: 32,
            selector_column: Some(COL_SEL_W_ALU_ADDI),
        }, 0));

        // Data columns: all register values should be in [0, 2^64).
        // This prevents a malicious prover from using arbitrary field elements.
        decls.push((LookupDeclaration {
            label: "rd_val_after_64bit".to_string(),
            column_index: COL_RD_VAL_AFTER,
            max_bits: 64,
            selector_column: None,
        }, 0));
        decls.push((LookupDeclaration {
            label: "rs1_val_64bit".to_string(),
            column_index: COL_RS1_VAL,
            max_bits: 64,
            selector_column: None,
        }, 0));
        decls.push((LookupDeclaration {
            label: "rs2_val_64bit".to_string(),
            column_index: COL_RS2_VAL,
            max_bits: 64,
            selector_column: None,
        }, 0));

        LookupRequirements {
            tables: vec![range_16],
            declarations: decls,
        }
    }

    fn memory_columns(&self) -> Option<(usize, Vec<usize>, Vec<usize>, Vec<usize>)> {
        if !self.use_extended {
            return None;
        }
        Some((COL_MEM_ADDR, vec![COL_MEM_VAL], vec![COL_SEL_LOAD, COL_SEL_LR],
              vec![COL_SEL_STORE, COL_SEL_SC, COL_SEL_AMO_SWAP,
                   COL_SEL_AMO_ADD, COL_SEL_AMO_BITWISE,
                   COL_SEL_AMO_COMPARE, COL_SEL_ATOMIC_OTHER,
                   COL_SEL_AMO_OR, COL_SEL_AMO_XOR,
                   COL_SEL_AMO_MAXU, COL_SEL_AMO_MINS, COL_SEL_AMO_MAXS]))
    }

    fn oracle_selectors(&self) -> Vec<usize> {
        Vec::new()
    }

    fn register_ports(&self) -> Vec<(usize, usize, bool)> {
        if !self.use_extended { return Vec::new(); }
        vec![
            (COL_RS1, COL_RS1_VAL, false),      // rs1 read
            (COL_RS2, COL_RS2_VAL, false),      // rs2 read
            (COL_RD, COL_RD_VAL_AFTER, true),   // rd write
        ]
    }
}

impl RiscvConstraintSystem {
    /// Legacy evaluation (backward compatible with original 3 constraints).
    fn evaluate_legacy_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let mut evaluations: Vec<Vec<Scalar>> = Vec::with_capacity(3);

        // Column indices: 0:pc, 1:rd, 2:rd_val_before, 3:rd_val_after,
        //   4:rs1, 5:rs1_val, 6:rs2, 7:rs2_val,
        //   8:mem_addr, 9:mem_val, 10:next_pc, 11:privilege_mode,
        //   12:insn_type, 13:funct, 14:immediate, 15:insn_len, 16:aux0, 17:aux1

        // Constraint 0: PC transition (next_pc - pc - 4)
        {
            let four = Scalar::from_u64(4, curve);
            let mut pc_constraint = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let pc_plus_4 = columns[0][i].add(&four);
                let val = columns[10][i].sub(&pc_plus_4);
                pc_constraint.push(val);
            }
            evaluations.push(pc_constraint);
        }

        // Constraint 1: x0 always zero
        {
            let mut x0_constraint = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                if columns[1][i].is_zero() {
                    x0_constraint.push(columns[3][i].clone());
                } else {
                    x0_constraint.push(Scalar::zero(curve));
                }
            }
            evaluations.push(x0_constraint);
        }

        // Constraint 2: ALU correctness (rd_val_after - rs1_val - rs2_val)
        {
            let mut alu_constraint = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let sum = columns[5][i].add(&columns[7][i]);
                let val = columns[3][i].sub(&sum);
                alu_constraint.push(val);
            }
            evaluations.push(alu_constraint);
        }

        evaluations
    }

    /// Extended evaluation with selector-based intra-row constraints.
    fn evaluate_extended_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);

        let (r_add_binary, r_sub_binary, i_add_binary) =
            alu::evaluate_aux0_binary_checks(columns, num_rows);

        // Split ALU combined evaluation into per-type vectors
        let alu_combined = alu::evaluate_alu(columns, num_rows);
        let mut alu_r_add = vec![zero.clone(); num_rows];
        let mut alu_r_sub = vec![zero.clone(); num_rows];
        let mut alu_i_add = vec![zero.clone(); num_rows];
        let mut alu_w_addw = vec![zero.clone(); num_rows];
        let mut alu_w_subw = vec![zero.clone(); num_rows];
        let mut alu_w_addiw = vec![zero.clone(); num_rows];
        let mut alu_mul = vec![zero.clone(); num_rows];
        let mut alu_div = vec![zero.clone(); num_rows];
        let mut alu_rem = vec![zero.clone(); num_rows];
        for i in 0..num_rows {
            let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;
            let funct_val = columns[COL_FUNCT][i].to_u64() as u8;
            match insn_type_val {
                selectors::INSN_R_ALU if funct_val == selectors::FUNCT_ADD => {
                    alu_r_add[i] = alu_combined[i].clone();
                }
                selectors::INSN_R_ALU if funct_val == selectors::FUNCT_SUB => {
                    alu_r_sub[i] = alu_combined[i].clone();
                }
                selectors::INSN_I_ALU if funct_val == selectors::FUNCT_ADDI => {
                    alu_i_add[i] = alu_combined[i].clone();
                }
                selectors::INSN_W_ALU => match funct_val {
                    selectors::FUNCT_ADDW => { alu_w_addw[i] = alu_combined[i].clone(); }
                    selectors::FUNCT_SUBW => { alu_w_subw[i] = alu_combined[i].clone(); }
                    selectors::FUNCT_ADDIW => { alu_w_addiw[i] = alu_combined[i].clone(); }
                    _ => {}
                },
                selectors::INSN_MULDIV => match funct_val {
                    0 => { alu_mul[i] = alu_combined[i].clone(); }
                    4 | 5 => { alu_div[i] = alu_combined[i].clone(); }
                    6 | 7 => { alu_rem[i] = alu_combined[i].clone(); }
                    _ => {}
                },
                _ => {}
            }
        }

        // Bitwise constraint evaluations (per-row, gated by selector)
        let mut alu_r_and = vec![zero.clone(); num_rows];
        let mut alu_r_or = vec![zero.clone(); num_rows];
        let mut alu_r_xor = vec![zero.clone(); num_rows];
        let mut alu_i_and = vec![zero.clone(); num_rows];
        let mut alu_i_or = vec![zero.clone(); num_rows];
        let mut alu_i_xor = vec![zero.clone(); num_rows];
        for i in 0..num_rows {
            let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;
            let funct_val = columns[COL_FUNCT][i].to_u64() as u8;
            match insn_type_val {
                selectors::INSN_R_ALU => match funct_val {
                    selectors::FUNCT_AND => {
                        // AND: rd = aux0 (aux0 = AND(rs1, rs2))
                        alu_r_and[i] = columns[COL_RD_VAL_AFTER][i].sub(&columns[COL_AUX0][i]);
                    }
                    selectors::FUNCT_OR => {
                        // OR: rd = rs1 + rs2 - aux0
                        let expected = columns[COL_RS1_VAL][i].add(&columns[COL_RS2_VAL][i])
                            .sub(&columns[COL_AUX0][i]);
                        alu_r_or[i] = columns[COL_RD_VAL_AFTER][i].sub(&expected);
                    }
                    selectors::FUNCT_XOR => {
                        // XOR: rd = rs1 + rs2 - 2*aux0
                        let two = Scalar::from_u64(2, curve);
                        let expected = columns[COL_RS1_VAL][i].add(&columns[COL_RS2_VAL][i])
                            .sub(&two.mul(&columns[COL_AUX0][i]));
                        alu_r_xor[i] = columns[COL_RD_VAL_AFTER][i].sub(&expected);
                    }
                    _ => {}
                },
                selectors::INSN_I_ALU => match funct_val {
                    selectors::FUNCT_ANDI => {
                        // ANDI: rd = aux0 (aux0 = AND(rs1, imm))
                        alu_i_and[i] = columns[COL_RD_VAL_AFTER][i].sub(&columns[COL_AUX0][i]);
                    }
                    selectors::FUNCT_ORI => {
                        // ORI: rd = rs1 + imm - aux0
                        let expected = columns[COL_RS1_VAL][i].add(&columns[COL_IMMEDIATE][i])
                            .sub(&columns[COL_AUX0][i]);
                        alu_i_or[i] = columns[COL_RD_VAL_AFTER][i].sub(&expected);
                    }
                    selectors::FUNCT_XORI => {
                        // XORI: rd = rs1 + imm - 2*aux0
                        let two = Scalar::from_u64(2, curve);
                        let expected = columns[COL_RS1_VAL][i].add(&columns[COL_IMMEDIATE][i])
                            .sub(&two.mul(&columns[COL_AUX0][i]));
                        alu_i_xor[i] = columns[COL_RD_VAL_AFTER][i].sub(&expected);
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // Split control flow evaluation into per-branch-type vectors
        let cf_combined = control_flow::evaluate_control_flow(columns, num_rows);
        let mut beq_evals = vec![zero.clone(); num_rows];
        let mut bne_evals = vec![zero.clone(); num_rows];
        let mut bltu_evals = vec![zero.clone(); num_rows];
        let mut bgeu_evals = vec![zero.clone(); num_rows];
        let mut blt_evals = vec![zero.clone(); num_rows];
        let mut bge_evals = vec![zero.clone(); num_rows];
        let mut jal_evals = vec![zero.clone(); num_rows];
        let mut jalr_evals = vec![zero.clone(); num_rows];
        let mut lui_evals = vec![zero.clone(); num_rows];
        let mut auipc_evals = vec![zero.clone(); num_rows];

        for i in 0..num_rows {
            let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;
            let funct_val = columns[COL_FUNCT][i].to_u64() as u8;
            match insn_type_val {
                selectors::INSN_BRANCH => match funct_val {
                    selectors::FUNCT_BEQ => beq_evals[i] = cf_combined[i].clone(),
                    selectors::FUNCT_BNE => bne_evals[i] = cf_combined[i].clone(),
                    selectors::FUNCT_BLTU => bltu_evals[i] = cf_combined[i].clone(),
                    selectors::FUNCT_BGEU => bgeu_evals[i] = cf_combined[i].clone(),
                    selectors::FUNCT_BLT => blt_evals[i] = cf_combined[i].clone(),
                    selectors::FUNCT_BGE => bge_evals[i] = cf_combined[i].clone(),
                    _ => {},
                },
                selectors::INSN_JAL => jal_evals[i] = cf_combined[i].clone(),
                selectors::INSN_JALR => jalr_evals[i] = cf_combined[i].clone(),
                selectors::INSN_LUI => lui_evals[i] = cf_combined[i].clone(),
                selectors::INSN_AUIPC => auipc_evals[i] = cf_combined[i].clone(),
                _ => {},
            }
        }

        // New constraint evaluations for constraints 45-62
        let mut csrrw_r_evals = vec![zero.clone(); num_rows];
        let mut csrrw_i_evals = vec![zero.clone(); num_rows];
        let mut csrrs_r_evals = vec![zero.clone(); num_rows];
        let mut csrrs_i_evals = vec![zero.clone(); num_rows];
        let mut mulh_evals = vec![zero.clone(); num_rows];
        let mut mulhsu_evals = vec![zero.clone(); num_rows];
        let mut mulw_evals = vec![zero.clone(); num_rows];
        let mut divw_evals = vec![zero.clone(); num_rows];
        let mut remw_evals = vec![zero.clone(); num_rows];
        let mut amo_swap_evals = vec![zero.clone(); num_rows];
        let mut amo_add_evals = vec![zero.clone(); num_rows];
        let mut amo_and_evals = vec![zero.clone(); num_rows];
        let mut amo_or_evals = vec![zero.clone(); num_rows];
        let mut amo_xor_evals = vec![zero.clone(); num_rows];
        let mut amo_minu_evals = vec![zero.clone(); num_rows];
        let mut amo_maxu_evals = vec![zero.clone(); num_rows];
        let mut amo_mins_evals = vec![zero.clone(); num_rows];
        let mut amo_maxs_evals = vec![zero.clone(); num_rows];

        for i in 0..num_rows {
            // Build a row_evals slice for this row so we can reuse the raw functions
            let row_evals: Vec<Scalar> = (0..columns.len()).map(|c| columns[c][i].clone()).collect();

            if !columns[COL_SEL_CSRRW][i].is_zero() {
                csrrw_r_evals[i] = alu::evaluate_csrrw_raw(&row_evals);
            }
            if columns.len() > COL_SEL_CSRRW_I && !columns[COL_SEL_CSRRW_I][i].is_zero() {
                csrrw_i_evals[i] = alu::evaluate_csrrw_i_raw(&row_evals);
            }
            if !columns[COL_SEL_CSRRS][i].is_zero() {
                csrrs_r_evals[i] = alu::evaluate_csrrs_raw(&row_evals);
            }
            if columns.len() > COL_SEL_CSRRS_I && !columns[COL_SEL_CSRRS_I][i].is_zero() {
                csrrs_i_evals[i] = alu::evaluate_csrrs_i_raw(&row_evals);
            }
            if columns.len() > COL_SEL_MULH && !columns[COL_SEL_MULH][i].is_zero() {
                mulh_evals[i] = alu::evaluate_mulh_raw(&row_evals);
            }
            if columns.len() > COL_SEL_MULHSU && !columns[COL_SEL_MULHSU][i].is_zero() {
                mulhsu_evals[i] = alu::evaluate_mulhsu_raw(&row_evals);
            }
            if columns.len() > COL_SEL_MULW && !columns[COL_SEL_MULW][i].is_zero() {
                mulw_evals[i] = alu::evaluate_mulw_raw(&row_evals);
            }
            if columns.len() > COL_SEL_DIVW && !columns[COL_SEL_DIVW][i].is_zero() {
                divw_evals[i] = alu::evaluate_divw_raw(&row_evals);
            }
            if columns.len() > COL_SEL_REMW && !columns[COL_SEL_REMW][i].is_zero() {
                remw_evals[i] = alu::evaluate_remw_raw(&row_evals);
            }
            if !columns[COL_SEL_AMO_SWAP][i].is_zero() {
                amo_swap_evals[i] = alu::evaluate_amo_swap_raw(&row_evals);
            }
            if !columns[COL_SEL_AMO_ADD][i].is_zero() {
                amo_add_evals[i] = alu::evaluate_amo_add_raw(&row_evals);
            }
            if !columns[COL_SEL_AMO_BITWISE][i].is_zero() {
                amo_and_evals[i] = alu::evaluate_amo_and_raw(&row_evals);
            }
            if columns.len() > COL_SEL_AMO_OR && !columns[COL_SEL_AMO_OR][i].is_zero() {
                amo_or_evals[i] = alu::evaluate_amo_or_raw(&row_evals);
            }
            if columns.len() > COL_SEL_AMO_XOR && !columns[COL_SEL_AMO_XOR][i].is_zero() {
                amo_xor_evals[i] = alu::evaluate_amo_xor_raw(&row_evals);
            }
            if !columns[COL_SEL_AMO_COMPARE][i].is_zero() {
                amo_minu_evals[i] = alu::evaluate_amo_minu_raw(&row_evals);
            }
            if columns.len() > COL_SEL_AMO_MAXU && !columns[COL_SEL_AMO_MAXU][i].is_zero() {
                amo_maxu_evals[i] = alu::evaluate_amo_maxu_raw(&row_evals);
            }
            if columns.len() > COL_SEL_AMO_MINS && !columns[COL_SEL_AMO_MINS][i].is_zero() {
                amo_mins_evals[i] = alu::evaluate_amo_mins_raw(&row_evals);
            }
            if columns.len() > COL_SEL_AMO_MAXS && !columns[COL_SEL_AMO_MAXS][i].is_zero() {
                amo_maxs_evals[i] = alu::evaluate_amo_maxs_raw(&row_evals);
            }
        }

        vec![
            control_flow::evaluate_sequential_pc(columns, num_rows),  // 0
            alu_r_add,                                                 // 1
            alu_r_sub,                                                 // 2
            alu_i_add,                                                 // 3
            beq_evals,                                                 // 4
            bne_evals,                                                 // 5
            bltu_evals,                                                // 6
            bgeu_evals,                                                // 7
            blt_evals,                                                 // 8
            bge_evals,                                                 // 9
            jal_evals,                                                 // 10
            jalr_evals,                                                // 11
            lui_evals,                                                 // 12
            auipc_evals,                                               // 13
            memory_ops::evaluate_memory_ops(columns, num_rows),        // 14
            r_add_binary,                                              // 15
            r_sub_binary,                                              // 16
            i_add_binary,                                              // 17
            alu_w_addw,                                                // 18
            alu_w_subw,                                                // 19
            alu_w_addiw,                                               // 20
            alu_mul,                                                   // 21
            alu_div,                                                   // 22
            alu_rem,                                                   // 23
            alu_r_and,                                                 // 24
            alu_r_or,                                                  // 25
            alu_r_xor,                                                 // 26
            alu_i_and,                                                 // 27
            alu_i_or,                                                  // 28
            alu_i_xor,                                                 // 29
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_R_ALU, selectors::FUNCT_SLL),  // 30
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_R_ALU, selectors::FUNCT_SRL),  // 31
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_R_ALU, selectors::FUNCT_SRA),  // 32
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_I_ALU, selectors::FUNCT_SLLI), // 33
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_I_ALU, selectors::FUNCT_SRLI), // 34
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_I_ALU, selectors::FUNCT_SRAI), // 35
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_W_ALU, selectors::FUNCT_SLLW), // 36
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_W_ALU, selectors::FUNCT_SRLW), // 37
            alu::evaluate_shift_on_domain(columns, num_rows, selectors::INSN_W_ALU, selectors::FUNCT_SRAW), // 38
            csrrw_r_evals,    // 39 (constraint 45)
            csrrw_i_evals,    // 40 (constraint 46)
            csrrs_r_evals,    // 41 (constraint 47)
            csrrs_i_evals,    // 42 (constraint 48)
            mulh_evals,       // 43 (constraint 49)
            mulhsu_evals,     // 44 (constraint 50)
            mulw_evals,       // 45 (constraint 51)
            divw_evals,       // 46 (constraint 52)
            remw_evals,       // 47 (constraint 53)
            amo_swap_evals,   // 48 (constraint 54)
            amo_add_evals,    // 49 (constraint 55)
            amo_and_evals,    // 50 (constraint 56)
            amo_or_evals,     // 51 (constraint 57)
            amo_xor_evals,    // 52 (constraint 58)
            amo_minu_evals,   // 53 (constraint 59)
            amo_maxu_evals,   // 54 (constraint 60)
            amo_mins_evals,   // 55 (constraint 61)
            amo_maxs_evals,   // 56 (constraint 62)
        ]
    }

    /// Evaluate constraints directly on TracePolynomials (primary entry point).
    ///
    /// Returns one Vec<Scalar> per constraint.
    pub fn evaluate_on_trace(
        &self,
        trace: &metavm_zkp::trace::TracePolynomials,
    ) -> Vec<Vec<Scalar>> {
        let columns = trace.columns();
        if self.use_extended {
            self.evaluate_extended_on_domain(&columns, trace.num_rows)
        } else {
            self.evaluate_legacy_on_domain(&columns, trace.num_rows)
        }
    }

    /// Legacy pointwise evaluation at z.
    fn evaluate_at_point_legacy(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        let curve = alpha.curve_type();

        if col_evals.len() < 16 {
            return Scalar::zero(curve);
        }

        let four = Scalar::from_u64(4, curve);

        // Constraint 0: next_pc - pc - 4
        let pc_plus_4 = col_evals[0].add(&four);
        let c0 = col_evals[10].sub(&pc_plus_4);

        // Constraint 1: x0 check — at arbitrary z, skip (it's zero)
        let c1 = Scalar::zero(curve);

        // Constraint 2: rd_val_after - rs1_val - rs2_val
        let rs_sum = col_evals[5].add(&col_evals[7]);
        let c2 = col_evals[3].sub(&rs_sum);

        // Combine: C(z) = c0 + alpha * c1 + alpha^2 * c2
        let alpha_c1 = alpha.mul(&c1);
        let alpha_sq = alpha.mul(alpha);
        let alpha_sq_c2 = alpha_sq.mul(&c2);
        c0.add(&alpha_c1).add(&alpha_sq_c2)
    }

    /// Extended pointwise evaluation at z using algebraic selector gating.
    ///
    /// All constraints are gated by selector column evaluations (field elements),
    /// not by integer branching on `.to_u64()`. This makes the evaluation correct
    /// at arbitrary field points, not just on trace rows.
    fn evaluate_at_point_extended(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        let curve = alpha.curve_type();

        if col_evals.len() < 19 {
            return Scalar::zero(curve);
        }

        let one = Scalar::one(curve);
        let mut result = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        // Constraint 0: sequential PC (gated by non-branch/jump selectors)
        let seq_sel = col_evals[COL_SEL_R_ALU_ADD].add(&col_evals[COL_SEL_R_ALU_SUB])
            .add(&col_evals[COL_SEL_R_AND]).add(&col_evals[COL_SEL_R_OR])
            .add(&col_evals[COL_SEL_R_XOR])
            .add(&col_evals[COL_SEL_R_SLL]).add(&col_evals[COL_SEL_R_SRL])
            .add(&col_evals[COL_SEL_R_SRA]).add(&col_evals[COL_SEL_R_COMPARE])
            .add(&col_evals[COL_SEL_I_ALU_ADD])
            .add(&col_evals[COL_SEL_I_AND]).add(&col_evals[COL_SEL_I_OR])
            .add(&col_evals[COL_SEL_I_XOR])
            .add(&col_evals[COL_SEL_I_SLL]).add(&col_evals[COL_SEL_I_SRL])
            .add(&col_evals[COL_SEL_I_SRA]).add(&col_evals[COL_SEL_I_COMPARE])
            .add(&col_evals[COL_SEL_W_ALU_ADD]).add(&col_evals[COL_SEL_W_ALU_SUB])
            .add(&col_evals[COL_SEL_W_ALU_ADDI])
            .add(&col_evals[COL_SEL_W_SLL]).add(&col_evals[COL_SEL_W_SRL])
            .add(&col_evals[COL_SEL_W_SRA]).add(&col_evals[COL_SEL_W_ALU_OTHER])
            .add(&col_evals[COL_SEL_MUL]).add(&col_evals[COL_SEL_DIV])
            .add(&col_evals[COL_SEL_REM]).add(&col_evals[COL_SEL_MULDIV_OTHER])
            .add(&col_evals[COL_SEL_LOAD]).add(&col_evals[COL_SEL_STORE])
            .add(&col_evals[COL_SEL_LUI]).add(&col_evals[COL_SEL_AUIPC])
            .add(&col_evals[COL_SEL_CSRRW]).add(&col_evals[COL_SEL_CSRRS])
            .add(&col_evals[COL_SEL_CSRRC])
            .add(&col_evals[COL_SEL_LR]).add(&col_evals[COL_SEL_SC])
            .add(&col_evals[COL_SEL_AMO_SWAP]).add(&col_evals[COL_SEL_AMO_ADD])
            .add(&col_evals[COL_SEL_AMO_BITWISE]).add(&col_evals[COL_SEL_AMO_COMPARE])
            .add(&col_evals[COL_SEL_ATOMIC_OTHER])
            .add(&col_evals[COL_SEL_MULHU])
            .add(&col_evals[COL_SEL_CSRRW_I]).add(&col_evals[COL_SEL_CSRRS_I])
            .add(&col_evals[COL_SEL_MULH]).add(&col_evals[COL_SEL_MULHSU])
            .add(&col_evals[COL_SEL_MULW]).add(&col_evals[COL_SEL_DIVW])
            .add(&col_evals[COL_SEL_REMW])
            .add(&col_evals[COL_SEL_AMO_OR]).add(&col_evals[COL_SEL_AMO_XOR])
            .add(&col_evals[COL_SEL_AMO_MAXU]).add(&col_evals[COL_SEL_AMO_MINS])
            .add(&col_evals[COL_SEL_AMO_MAXS]);
        result = result.add(&seq_sel.mul(&control_flow::evaluate_sequential_pc_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 1: R-type ADD main (gated by sel_r_alu_add)
        result = result.add(&col_evals[COL_SEL_R_ALU_ADD]
            .mul(&alu::evaluate_alu_r_add_main_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 2: R-type SUB main (gated by sel_r_alu_sub)
        result = result.add(&col_evals[COL_SEL_R_ALU_SUB]
            .mul(&alu::evaluate_alu_r_sub_main_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 3: I-type ADDI main (gated by sel_i_alu_add)
        result = result.add(&col_evals[COL_SEL_I_ALU_ADD]
            .mul(&alu::evaluate_alu_i_add_main_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 4: BEQ condition (gated by sel_beq)
        result = result.add(&col_evals[COL_SEL_BEQ]
            .mul(&control_flow::evaluate_beq_condition_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 5: BNE condition (gated by sel_bne)
        result = result.add(&col_evals[COL_SEL_BNE]
            .mul(&control_flow::evaluate_bne_condition_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 6: BLTU condition (gated by sel_bltu)
        result = result.add(&col_evals[COL_SEL_BLTU]
            .mul(&control_flow::evaluate_bltu_condition_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 7: BGEU condition (gated by sel_bgeu)
        result = result.add(&col_evals[COL_SEL_BGEU]
            .mul(&control_flow::evaluate_bgeu_condition_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 8: BLT condition (gated by sel_blt)
        result = result.add(&col_evals[COL_SEL_BLT]
            .mul(&control_flow::evaluate_blt_condition_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 9: BGE condition (gated by sel_bge)
        result = result.add(&col_evals[COL_SEL_BGE]
            .mul(&control_flow::evaluate_bge_condition_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 10: JAL body (gated by sel_jal)
        result = result.add(&col_evals[COL_SEL_JAL]
            .mul(&control_flow::evaluate_jal_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 11: JALR body (gated by sel_jalr)
        result = result.add(&col_evals[COL_SEL_JALR]
            .mul(&control_flow::evaluate_jalr_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 12: LUI body (gated by sel_lui)
        result = result.add(&col_evals[COL_SEL_LUI]
            .mul(&control_flow::evaluate_lui_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 13: AUIPC body (gated by sel_auipc)
        result = result.add(&col_evals[COL_SEL_AUIPC]
            .mul(&control_flow::evaluate_auipc_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 14: memory ops (gated by sel_load + sel_store)
        let mem_sel = col_evals[COL_SEL_LOAD].add(&col_evals[COL_SEL_STORE]);
        result = result.add(&mem_sel.mul(&memory_ops::evaluate_memory_ops_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 14: R-ADD carry binary (gated by sel_r_alu_add)
        let aux0_binary = alu::evaluate_aux0_binary_raw(col_evals);
        result = result.add(&col_evals[COL_SEL_R_ALU_ADD]
            .mul(&aux0_binary).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 15: R-SUB borrow binary (gated by sel_r_alu_sub)
        result = result.add(&col_evals[COL_SEL_R_ALU_SUB]
            .mul(&aux0_binary).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 16: I-ADDI carry binary (gated by sel_i_alu_add)
        result = result.add(&col_evals[COL_SEL_I_ALU_ADD]
            .mul(&aux0_binary).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 17: W-type ADDW (gated by sel_w_alu_add)
        result = result.add(&col_evals[COL_SEL_W_ALU_ADD]
            .mul(&alu::evaluate_alu_w_addw_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 18: W-type SUBW (gated by sel_w_alu_sub)
        result = result.add(&col_evals[COL_SEL_W_ALU_SUB]
            .mul(&alu::evaluate_alu_w_subw_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 19: W-type ADDIW (gated by sel_w_alu_addi)
        result = result.add(&col_evals[COL_SEL_W_ALU_ADDI]
            .mul(&alu::evaluate_alu_w_addiw_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 20: MUL (gated by sel_mul)
        result = result.add(&col_evals[COL_SEL_MUL]
            .mul(&alu::evaluate_mul_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 21: DIV/DIVU (gated by sel_div)
        result = result.add(&col_evals[COL_SEL_DIV]
            .mul(&alu::evaluate_div_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 22: REM/REMU (gated by sel_rem)
        result = result.add(&col_evals[COL_SEL_REM]
            .mul(&alu::evaluate_rem_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 23: R-AND (gated by sel_r_and)
        result = result.add(&col_evals[COL_SEL_R_AND]
            .mul(&alu::evaluate_r_and_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 24: R-OR (gated by sel_r_or)
        result = result.add(&col_evals[COL_SEL_R_OR]
            .mul(&alu::evaluate_r_or_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 25: R-XOR (gated by sel_r_xor)
        result = result.add(&col_evals[COL_SEL_R_XOR]
            .mul(&alu::evaluate_r_xor_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 26: I-AND (gated by sel_i_and)
        result = result.add(&col_evals[COL_SEL_I_AND]
            .mul(&alu::evaluate_i_and_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 27: I-OR (gated by sel_i_or)
        result = result.add(&col_evals[COL_SEL_I_OR]
            .mul(&alu::evaluate_i_or_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 28: I-XOR (gated by sel_i_xor)
        result = result.add(&col_evals[COL_SEL_I_XOR]
            .mul(&alu::evaluate_i_xor_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 29: R-SLL (gated by sel_r_sll)
        result = result.add(&col_evals[COL_SEL_R_SLL]
            .mul(&alu::evaluate_sll_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 30: R-SRL (gated by sel_r_srl)
        result = result.add(&col_evals[COL_SEL_R_SRL]
            .mul(&alu::evaluate_srl_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 31: R-SRA (gated by sel_r_sra)
        result = result.add(&col_evals[COL_SEL_R_SRA]
            .mul(&alu::evaluate_sra_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 32: I-SLL (gated by sel_i_sll)
        result = result.add(&col_evals[COL_SEL_I_SLL]
            .mul(&alu::evaluate_sll_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 33: I-SRL (gated by sel_i_srl)
        result = result.add(&col_evals[COL_SEL_I_SRL]
            .mul(&alu::evaluate_srl_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 34: I-SRA (gated by sel_i_sra)
        result = result.add(&col_evals[COL_SEL_I_SRA]
            .mul(&alu::evaluate_sra_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 35: W-SLL (gated by sel_w_sll)
        result = result.add(&col_evals[COL_SEL_W_SLL]
            .mul(&alu::evaluate_w_sll_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 36: W-SRL (gated by sel_w_srl)
        result = result.add(&col_evals[COL_SEL_W_SRL]
            .mul(&alu::evaluate_w_srl_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 37: W-SRA (gated by sel_w_sra)
        result = result.add(&col_evals[COL_SEL_W_SRA]
            .mul(&alu::evaluate_w_sra_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 39: R-type compare (SLT/SLTU) (gated by sel_r_compare)
        result = result.add(&col_evals[COL_SEL_R_COMPARE]
            .mul(&alu::evaluate_compare_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 40: I-type compare (SLTI/SLTIU) (gated by sel_i_compare)
        result = result.add(&col_evals[COL_SEL_I_COMPARE]
            .mul(&alu::evaluate_compare_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 41: CSRRC (gated by sel_csrrc)
        result = result.add(&col_evals[COL_SEL_CSRRC]
            .mul(&alu::evaluate_csrrc_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 42: LR (gated by sel_lr)
        result = result.add(&col_evals[COL_SEL_LR]
            .mul(&alu::evaluate_lr_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 43: SC (gated by sel_sc)
        result = result.add(&col_evals[COL_SEL_SC]
            .mul(&alu::evaluate_sc_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 44: MULHU (gated by sel_mulhu)
        result = result.add(&col_evals[COL_SEL_MULHU]
            .mul(&alu::evaluate_mulhu_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 45: CSRRW_R (gated by sel_csrrw -- now R-variant only)
        result = result.add(&col_evals[COL_SEL_CSRRW]
            .mul(&alu::evaluate_csrrw_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 46: CSRRW_I (gated by sel_csrrw_i)
        result = result.add(&col_evals[COL_SEL_CSRRW_I]
            .mul(&alu::evaluate_csrrw_i_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 47: CSRRS_R (gated by sel_csrrs -- now R-variant only)
        result = result.add(&col_evals[COL_SEL_CSRRS]
            .mul(&alu::evaluate_csrrs_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 48: CSRRS_I (gated by sel_csrrs_i)
        result = result.add(&col_evals[COL_SEL_CSRRS_I]
            .mul(&alu::evaluate_csrrs_i_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 49: MULH (gated by sel_mulh)
        result = result.add(&col_evals[COL_SEL_MULH]
            .mul(&alu::evaluate_mulh_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 50: MULHSU (gated by sel_mulhsu)
        result = result.add(&col_evals[COL_SEL_MULHSU]
            .mul(&alu::evaluate_mulhsu_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 51: MULW (gated by sel_mulw)
        result = result.add(&col_evals[COL_SEL_MULW]
            .mul(&alu::evaluate_mulw_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 52: DIVW (gated by sel_divw)
        result = result.add(&col_evals[COL_SEL_DIVW]
            .mul(&alu::evaluate_divw_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 53: REMW (gated by sel_remw)
        result = result.add(&col_evals[COL_SEL_REMW]
            .mul(&alu::evaluate_remw_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 54: AMO_SWAP (gated by sel_amo_swap)
        result = result.add(&col_evals[COL_SEL_AMO_SWAP]
            .mul(&alu::evaluate_amo_swap_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 55: AMO_ADD (gated by sel_amo_add)
        result = result.add(&col_evals[COL_SEL_AMO_ADD]
            .mul(&alu::evaluate_amo_add_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 56: AMO_AND (gated by sel_amo_bitwise -- now AND only)
        result = result.add(&col_evals[COL_SEL_AMO_BITWISE]
            .mul(&alu::evaluate_amo_and_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 57: AMO_OR (gated by sel_amo_or)
        result = result.add(&col_evals[COL_SEL_AMO_OR]
            .mul(&alu::evaluate_amo_or_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 58: AMO_XOR (gated by sel_amo_xor)
        result = result.add(&col_evals[COL_SEL_AMO_XOR]
            .mul(&alu::evaluate_amo_xor_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 59: AMO_MINU (gated by sel_amo_compare -- now MINU only)
        result = result.add(&col_evals[COL_SEL_AMO_COMPARE]
            .mul(&alu::evaluate_amo_minu_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 60: AMO_MAXU (gated by sel_amo_maxu)
        result = result.add(&col_evals[COL_SEL_AMO_MAXU]
            .mul(&alu::evaluate_amo_maxu_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 61: AMO_MINS (gated by sel_amo_mins)
        result = result.add(&col_evals[COL_SEL_AMO_MINS]
            .mul(&alu::evaluate_amo_mins_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Constraint 62: AMO_MAXS (gated by sel_amo_maxs)
        result = result.add(&col_evals[COL_SEL_AMO_MAXS]
            .mul(&alu::evaluate_amo_maxs_raw(col_evals)).mul(&ap));
        ap = ap.mul(alpha);

        // Selector consistency: binary constraints for each selector
        let sel_indices = self.selector_column_indices();
        for &si in &sel_indices {
            let s = &col_evals[si];
            let s_m1 = s.sub(&one);
            result = result.add(&s.mul(&s_m1).mul(&ap));
            ap = ap.mul(alpha);
        }

        // Sum-to-one: exactly one selector must be 1 on every row
        let mut sel_sum = Scalar::zero(curve);
        for &si in &sel_indices {
            sel_sum = sel_sum.add(&col_evals[si]);
        }
        result = result.add(&sel_sum.sub(&one).mul(&ap));

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metavm_zkp::field::{Scalar, CurveType};
    use crate::trace::TraceColumns;

    fn make_valid_trace(steps: usize) -> TraceColumns {
        let mut cols = TraceColumns {
            step: Vec::with_capacity(steps),
            pc: Vec::with_capacity(steps),
            rd: Vec::with_capacity(steps),
            rd_val_before: Vec::with_capacity(steps),
            rd_val_after: Vec::with_capacity(steps),
            rs1: Vec::with_capacity(steps),
            rs1_val: Vec::with_capacity(steps),
            rs2: Vec::with_capacity(steps),
            rs2_val: Vec::with_capacity(steps),
            mem_addr: Vec::with_capacity(steps),
            mem_val: Vec::with_capacity(steps),
            next_pc: Vec::with_capacity(steps),
            privilege_mode: Vec::with_capacity(steps),
            insn_type: Vec::with_capacity(steps),
            funct: Vec::with_capacity(steps),
            immediate: Vec::with_capacity(steps),
            insn_len: Vec::with_capacity(steps),
            aux0: Vec::with_capacity(steps),
            aux1: Vec::with_capacity(steps),
            aux2: Vec::with_capacity(steps),
            sel_r_alu_add: Vec::with_capacity(steps),
            sel_r_alu_sub: Vec::with_capacity(steps),
            sel_r_and: Vec::with_capacity(steps),
            sel_r_or: Vec::with_capacity(steps),
            sel_r_xor: Vec::with_capacity(steps),
            sel_r_sll: Vec::with_capacity(steps),
            sel_r_srl: Vec::with_capacity(steps),
            sel_r_sra: Vec::with_capacity(steps),
            sel_r_compare: Vec::with_capacity(steps),
            sel_i_alu_add: Vec::with_capacity(steps),
            sel_i_and: Vec::with_capacity(steps),
            sel_i_or: Vec::with_capacity(steps),
            sel_i_xor: Vec::with_capacity(steps),
            sel_i_sll: Vec::with_capacity(steps),
            sel_i_srl: Vec::with_capacity(steps),
            sel_i_sra: Vec::with_capacity(steps),
            sel_i_compare: Vec::with_capacity(steps),
            sel_w_alu_add: Vec::with_capacity(steps),
            sel_w_alu_sub: Vec::with_capacity(steps),
            sel_w_alu_addi: Vec::with_capacity(steps),
            sel_w_sll: Vec::with_capacity(steps),
            sel_w_srl: Vec::with_capacity(steps),
            sel_w_sra: Vec::with_capacity(steps),
            sel_w_alu_other: Vec::with_capacity(steps),
            sel_mul: Vec::with_capacity(steps),
            sel_div: Vec::with_capacity(steps),
            sel_rem: Vec::with_capacity(steps),
            sel_muldiv_other: Vec::with_capacity(steps),
            sel_load: Vec::with_capacity(steps),
            sel_store: Vec::with_capacity(steps),
            sel_beq: Vec::with_capacity(steps),
            sel_bne: Vec::with_capacity(steps),
            sel_bltu: Vec::with_capacity(steps),
            sel_bgeu: Vec::with_capacity(steps),
            sel_blt: Vec::with_capacity(steps),
            sel_bge: Vec::with_capacity(steps),
            sel_jal: Vec::with_capacity(steps),
            sel_jalr: Vec::with_capacity(steps),
            sel_lui: Vec::with_capacity(steps),
            sel_auipc: Vec::with_capacity(steps),
            sel_csrrw: Vec::with_capacity(steps),
            sel_csrrs: Vec::with_capacity(steps),
            sel_csrrc: Vec::with_capacity(steps),
            sel_system: Vec::with_capacity(steps),
            sel_lr: Vec::with_capacity(steps),
            sel_sc: Vec::with_capacity(steps),
            sel_amo_swap: Vec::with_capacity(steps),
            sel_amo_add: Vec::with_capacity(steps),
            sel_amo_bitwise: Vec::with_capacity(steps),
            sel_amo_compare: Vec::with_capacity(steps),
            sel_atomic_other: Vec::with_capacity(steps),
            sel_mulhu: Vec::with_capacity(steps),
            sel_csrrw_i: Vec::with_capacity(steps),
            sel_csrrs_i: Vec::with_capacity(steps),
            sel_mulh: Vec::with_capacity(steps),
            sel_mulhsu: Vec::with_capacity(steps),
            sel_mulw: Vec::with_capacity(steps),
            sel_divw: Vec::with_capacity(steps),
            sel_remw: Vec::with_capacity(steps),
            sel_amo_or: Vec::with_capacity(steps),
            sel_amo_xor: Vec::with_capacity(steps),
            sel_amo_maxu: Vec::with_capacity(steps),
            sel_amo_mins: Vec::with_capacity(steps),
            sel_amo_maxs: Vec::with_capacity(steps),
        };

        for i in 0..steps {
            let pc = (i as u64) * 4;
            cols.step.push(i as u64);
            cols.pc.push(pc);
            cols.rd.push(1);
            cols.rd_val_before.push(0);
            cols.rd_val_after.push(15);
            cols.rs1.push(2);
            cols.rs1_val.push(10);
            cols.rs2.push(3);
            cols.rs2_val.push(5);
            cols.mem_addr.push(0);
            cols.mem_val.push(0);
            cols.next_pc.push(pc + 4);
            cols.privilege_mode.push(3);
            cols.insn_type.push(0); // R-type ALU
            cols.funct.push(0);     // ADD
            cols.immediate.push(0);
            cols.insn_len.push(4);
            cols.aux0.push(0);      // no carry for small values
            cols.aux1.push(0);
            cols.aux2.push(0);
            // Selectors: R-type ADD is active
            cols.sel_r_alu_add.push(1);
            cols.sel_r_alu_sub.push(0);
            cols.sel_r_and.push(0);
            cols.sel_r_or.push(0);
            cols.sel_r_xor.push(0);
            cols.sel_r_sll.push(0);
            cols.sel_r_srl.push(0);
            cols.sel_r_sra.push(0);
            cols.sel_r_compare.push(0);
            cols.sel_i_alu_add.push(0);
            cols.sel_i_and.push(0);
            cols.sel_i_or.push(0);
            cols.sel_i_xor.push(0);
            cols.sel_i_sll.push(0);
            cols.sel_i_srl.push(0);
            cols.sel_i_sra.push(0);
            cols.sel_i_compare.push(0);
            cols.sel_w_alu_add.push(0);
            cols.sel_w_alu_sub.push(0);
            cols.sel_w_alu_addi.push(0);
            cols.sel_w_sll.push(0);
            cols.sel_w_srl.push(0);
            cols.sel_w_sra.push(0);
            cols.sel_w_alu_other.push(0);
            cols.sel_mul.push(0);
            cols.sel_div.push(0);
            cols.sel_rem.push(0);
            cols.sel_muldiv_other.push(0);
            cols.sel_load.push(0);
            cols.sel_store.push(0);
            cols.sel_beq.push(0);
            cols.sel_bne.push(0);
            cols.sel_bltu.push(0);
            cols.sel_bgeu.push(0);
            cols.sel_blt.push(0);
            cols.sel_bge.push(0);
            cols.sel_jal.push(0);
            cols.sel_jalr.push(0);
            cols.sel_lui.push(0);
            cols.sel_auipc.push(0);
            cols.sel_csrrw.push(0);
            cols.sel_csrrs.push(0);
            cols.sel_csrrc.push(0);
            cols.sel_system.push(0);
            cols.sel_lr.push(0);
            cols.sel_sc.push(0);
            cols.sel_amo_swap.push(0);
            cols.sel_amo_add.push(0);
            cols.sel_amo_bitwise.push(0);
            cols.sel_amo_compare.push(0);
            cols.sel_atomic_other.push(0);
            cols.sel_mulhu.push(0);
            cols.sel_csrrw_i.push(0);
            cols.sel_csrrs_i.push(0);
            cols.sel_mulh.push(0);
            cols.sel_mulhsu.push(0);
            cols.sel_mulw.push(0);
            cols.sel_divw.push(0);
            cols.sel_remw.push(0);
            cols.sel_amo_or.push(0);
            cols.sel_amo_xor.push(0);
            cols.sel_amo_maxu.push(0);
            cols.sel_amo_mins.push(0);
            cols.sel_amo_maxs.push(0);
        }

        cols
    }

    /// Helper to build constraint polynomial from evaluation vectors.
    fn build_constraint_polynomial(
        evaluations: &[Vec<Scalar>],
        alpha: &Scalar,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        if evaluations.is_empty() {
            return Vec::new();
        }

        let n = evaluations[0].len();
        let mut result = vec![Scalar::zero(curve); n];

        let mut alpha_power = Scalar::one(curve);
        for constraint_evals in evaluations {
            for j in 0..n.min(constraint_evals.len()) {
                let term = alpha_power.mul(&constraint_evals[j]);
                result[j] = result[j].add(&term);
            }
            alpha_power = alpha_power.mul(alpha);
        }

        result
    }

    #[test]
    fn test_legacy_pc_transition_valid() {
        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::new();
        let eval = cs.evaluate_on_trace(&polys);

        for (i, v) in eval[0].iter().enumerate() {
            assert!(v.is_zero(), "PC constraint failed at row {}", i);
        }
    }

    #[test]
    fn test_legacy_x0_constraint() {
        let columns = TraceColumns {
            step: vec![0], pc: vec![0], rd: vec![0],
            rd_val_before: vec![0], rd_val_after: vec![0],
            rs1: vec![0], rs1_val: vec![0], rs2: vec![0], rs2_val: vec![0],
            mem_addr: vec![0], mem_val: vec![0], next_pc: vec![4],
            privilege_mode: vec![3],
            insn_type: vec![0], funct: vec![0], immediate: vec![0], insn_len: vec![4],
            aux0: vec![0], aux1: vec![0], aux2: vec![0],
            sel_r_alu_add: vec![1], sel_r_alu_sub: vec![0], sel_r_and: vec![0], sel_r_or: vec![0], sel_r_xor: vec![0], sel_r_sll: vec![0], sel_r_srl: vec![0], sel_r_sra: vec![0], sel_r_compare: vec![0],
            sel_i_alu_add: vec![0], sel_i_and: vec![0], sel_i_or: vec![0], sel_i_xor: vec![0],
            sel_i_sll: vec![0], sel_i_srl: vec![0], sel_i_sra: vec![0], sel_i_compare: vec![0],
            sel_w_alu_add: vec![0], sel_w_alu_sub: vec![0], sel_w_alu_addi: vec![0],
            sel_w_sll: vec![0], sel_w_srl: vec![0], sel_w_sra: vec![0], sel_w_alu_other: vec![0],
            sel_mul: vec![0], sel_div: vec![0], sel_rem: vec![0], sel_muldiv_other: vec![0],
            sel_load: vec![0], sel_store: vec![0],
            sel_beq: vec![0], sel_bne: vec![0], sel_bltu: vec![0], sel_bgeu: vec![0],
            sel_blt: vec![0], sel_bge: vec![0], sel_jal: vec![0],
            sel_jalr: vec![0], sel_lui: vec![0], sel_auipc: vec![0], sel_csrrw: vec![0], sel_csrrs: vec![0], sel_csrrc: vec![0],
            sel_system: vec![0],
            sel_lr: vec![0], sel_sc: vec![0], sel_amo_swap: vec![0], sel_amo_add: vec![0],
            sel_amo_bitwise: vec![0], sel_amo_compare: vec![0], sel_atomic_other: vec![0],
            sel_mulhu: vec![0],
            sel_csrrw_i: vec![0], sel_csrrs_i: vec![0],
            sel_mulh: vec![0], sel_mulhsu: vec![0], sel_mulw: vec![0], sel_divw: vec![0], sel_remw: vec![0],
            sel_amo_or: vec![0], sel_amo_xor: vec![0],
            sel_amo_maxu: vec![0], sel_amo_mins: vec![0], sel_amo_maxs: vec![0],
        };
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::new();
        let eval = cs.evaluate_on_trace(&polys);
        assert!(eval[1][0].is_zero());
    }

    #[test]
    fn test_legacy_x0_violation() {
        let columns = TraceColumns {
            step: vec![0], pc: vec![0], rd: vec![0],
            rd_val_before: vec![0], rd_val_after: vec![99],
            rs1: vec![0], rs1_val: vec![0], rs2: vec![0], rs2_val: vec![0],
            mem_addr: vec![0], mem_val: vec![0], next_pc: vec![4],
            privilege_mode: vec![3],
            insn_type: vec![0], funct: vec![0], immediate: vec![0], insn_len: vec![4],
            aux0: vec![0], aux1: vec![0], aux2: vec![0],
            sel_r_alu_add: vec![1], sel_r_alu_sub: vec![0], sel_r_and: vec![0], sel_r_or: vec![0], sel_r_xor: vec![0],
            sel_r_sll: vec![0], sel_r_srl: vec![0], sel_r_sra: vec![0], sel_r_compare: vec![0],
            sel_i_alu_add: vec![0], sel_i_and: vec![0], sel_i_or: vec![0], sel_i_xor: vec![0],
            sel_i_sll: vec![0], sel_i_srl: vec![0], sel_i_sra: vec![0], sel_i_compare: vec![0],
            sel_w_alu_add: vec![0], sel_w_alu_sub: vec![0], sel_w_alu_addi: vec![0],
            sel_w_sll: vec![0], sel_w_srl: vec![0], sel_w_sra: vec![0], sel_w_alu_other: vec![0],
            sel_mul: vec![0], sel_div: vec![0], sel_rem: vec![0], sel_muldiv_other: vec![0],
            sel_load: vec![0], sel_store: vec![0],
            sel_beq: vec![0], sel_bne: vec![0], sel_bltu: vec![0], sel_bgeu: vec![0],
            sel_blt: vec![0], sel_bge: vec![0], sel_jal: vec![0],
            sel_jalr: vec![0], sel_lui: vec![0], sel_auipc: vec![0], sel_csrrw: vec![0], sel_csrrs: vec![0], sel_csrrc: vec![0],
            sel_system: vec![0],
            sel_lr: vec![0], sel_sc: vec![0], sel_amo_swap: vec![0], sel_amo_add: vec![0],
            sel_amo_bitwise: vec![0], sel_amo_compare: vec![0], sel_atomic_other: vec![0],
            sel_mulhu: vec![0],
            sel_csrrw_i: vec![0], sel_csrrs_i: vec![0],
            sel_mulh: vec![0], sel_mulhsu: vec![0], sel_mulw: vec![0], sel_divw: vec![0], sel_remw: vec![0],
            sel_amo_or: vec![0], sel_amo_xor: vec![0],
            sel_amo_maxu: vec![0], sel_amo_mins: vec![0], sel_amo_maxs: vec![0],
        };
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::new();
        let eval = cs.evaluate_on_trace(&polys);
        assert!(!eval[1][0].is_zero());
    }

    #[test]
    fn test_legacy_alu_valid() {
        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::new();
        let eval = cs.evaluate_on_trace(&polys);

        for (i, v) in eval[2].iter().enumerate() {
            assert!(v.is_zero(), "ALU constraint failed at row {}", i);
        }
    }

    #[test]
    fn test_legacy_combined_zero() {
        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::new();
        let eval = cs.evaluate_on_trace(&polys);

        let alpha = Scalar::from_u64(7, CurveType::Bls48581);
        let combined = build_constraint_polynomial(&eval, &alpha);

        for (i, v) in combined.iter().enumerate() {
            assert!(v.is_zero(), "Combined should be zero at row {} for valid trace", i);
        }
    }

    #[test]
    fn test_legacy_pc_invalid() {
        let columns = TraceColumns {
            step: vec![0, 1], pc: vec![0, 4],
            rd: vec![1, 2], rd_val_before: vec![0, 0], rd_val_after: vec![42, 43],
            rs1: vec![0, 1], rs1_val: vec![0, 42], rs2: vec![0, 0], rs2_val: vec![0, 0],
            mem_addr: vec![0, 0], mem_val: vec![0, 0],
            next_pc: vec![4, 100], // wrong
            privilege_mode: vec![3, 3],
            insn_type: vec![1, 1], funct: vec![0, 0], immediate: vec![42, 1], insn_len: vec![4, 4],
            aux0: vec![0, 0], aux1: vec![0, 0], aux2: vec![0, 0],
            sel_r_alu_add: vec![0, 0], sel_r_alu_sub: vec![0, 0], sel_r_and: vec![0, 0], sel_r_or: vec![0, 0], sel_r_xor: vec![0, 0],
            sel_r_sll: vec![0, 0], sel_r_srl: vec![0, 0], sel_r_sra: vec![0, 0], sel_r_compare: vec![0, 0],
            sel_i_alu_add: vec![1, 1], sel_i_and: vec![0, 0], sel_i_or: vec![0, 0], sel_i_xor: vec![0, 0],
            sel_i_sll: vec![0, 0], sel_i_srl: vec![0, 0], sel_i_sra: vec![0, 0], sel_i_compare: vec![0, 0],
            sel_w_alu_add: vec![0, 0], sel_w_alu_sub: vec![0, 0], sel_w_alu_addi: vec![0, 0],
            sel_w_sll: vec![0, 0], sel_w_srl: vec![0, 0], sel_w_sra: vec![0, 0], sel_w_alu_other: vec![0, 0],
            sel_mul: vec![0, 0], sel_div: vec![0, 0], sel_rem: vec![0, 0], sel_muldiv_other: vec![0, 0],
            sel_load: vec![0, 0], sel_store: vec![0, 0],
            sel_beq: vec![0, 0], sel_bne: vec![0, 0], sel_bltu: vec![0, 0], sel_bgeu: vec![0, 0],
            sel_blt: vec![0, 0], sel_bge: vec![0, 0], sel_jal: vec![0, 0],
            sel_jalr: vec![0, 0], sel_lui: vec![0, 0], sel_auipc: vec![0, 0], sel_csrrw: vec![0, 0], sel_csrrs: vec![0, 0], sel_csrrc: vec![0, 0],
            sel_system: vec![0, 0],
            sel_lr: vec![0, 0], sel_sc: vec![0, 0], sel_amo_swap: vec![0, 0], sel_amo_add: vec![0, 0],
            sel_amo_bitwise: vec![0, 0], sel_amo_compare: vec![0, 0], sel_atomic_other: vec![0, 0],
            sel_mulhu: vec![0, 0],
            sel_csrrw_i: vec![0, 0], sel_csrrs_i: vec![0, 0],
            sel_mulh: vec![0, 0], sel_mulhsu: vec![0, 0], sel_mulw: vec![0, 0], sel_divw: vec![0, 0], sel_remw: vec![0, 0],
            sel_amo_or: vec![0, 0], sel_amo_xor: vec![0, 0],
            sel_amo_maxu: vec![0, 0], sel_amo_mins: vec![0, 0], sel_amo_maxs: vec![0, 0],
        };
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::new();
        let eval = cs.evaluate_on_trace(&polys);
        assert!(eval[0][0].is_zero());
        assert!(!eval[0][1].is_zero());
    }

    #[test]
    fn test_extended_add_constraint() {
        // R-type ADD: rd_val_after = rs1_val + rs2_val
        let columns = TraceColumns {
            step: vec![0], pc: vec![0],
            rd: vec![1], rd_val_before: vec![0], rd_val_after: vec![15],
            rs1: vec![2], rs1_val: vec![10], rs2: vec![3], rs2_val: vec![5],
            mem_addr: vec![0], mem_val: vec![0],
            next_pc: vec![4], privilege_mode: vec![3],
            insn_type: vec![0], funct: vec![0], immediate: vec![0], insn_len: vec![4],
            aux0: vec![0], aux1: vec![0], aux2: vec![0],
            sel_r_alu_add: vec![1], sel_r_alu_sub: vec![0], sel_r_and: vec![0], sel_r_or: vec![0], sel_r_xor: vec![0],
            sel_r_sll: vec![0], sel_r_srl: vec![0], sel_r_sra: vec![0], sel_r_compare: vec![0],
            sel_i_alu_add: vec![0], sel_i_and: vec![0], sel_i_or: vec![0], sel_i_xor: vec![0],
            sel_i_sll: vec![0], sel_i_srl: vec![0], sel_i_sra: vec![0], sel_i_compare: vec![0],
            sel_w_alu_add: vec![0], sel_w_alu_sub: vec![0], sel_w_alu_addi: vec![0],
            sel_w_sll: vec![0], sel_w_srl: vec![0], sel_w_sra: vec![0], sel_w_alu_other: vec![0],
            sel_mul: vec![0], sel_div: vec![0], sel_rem: vec![0], sel_muldiv_other: vec![0],
            sel_load: vec![0], sel_store: vec![0],
            sel_beq: vec![0], sel_bne: vec![0], sel_bltu: vec![0], sel_bgeu: vec![0],
            sel_blt: vec![0], sel_bge: vec![0], sel_jal: vec![0],
            sel_jalr: vec![0], sel_lui: vec![0], sel_auipc: vec![0], sel_csrrw: vec![0], sel_csrrs: vec![0], sel_csrrc: vec![0],
            sel_system: vec![0],
            sel_lr: vec![0], sel_sc: vec![0], sel_amo_swap: vec![0], sel_amo_add: vec![0],
            sel_amo_bitwise: vec![0], sel_amo_compare: vec![0], sel_atomic_other: vec![0],
            sel_mulhu: vec![0],
            sel_csrrw_i: vec![0], sel_csrrs_i: vec![0],
            sel_mulh: vec![0], sel_mulhsu: vec![0], sel_mulw: vec![0], sel_divw: vec![0], sel_remw: vec![0],
            sel_amo_or: vec![0], sel_amo_xor: vec![0],
            sel_amo_maxu: vec![0], sel_amo_mins: vec![0], sel_amo_maxs: vec![0],
        };
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::full();
        let eval = cs.evaluate_on_trace(&polys);

        // ALU constraint (index 1) should be satisfied
        assert!(eval[1][0].is_zero(), "ADD constraint should be satisfied");
    }

    #[test]
    fn test_extended_lui_constraint() {
        // LUI: rd_val_after = immediate
        let imm: u64 = 0x12345000;
        let columns = TraceColumns {
            step: vec![0], pc: vec![0x100],
            rd: vec![1], rd_val_before: vec![0], rd_val_after: vec![imm],
            rs1: vec![0], rs1_val: vec![0], rs2: vec![0], rs2_val: vec![0],
            mem_addr: vec![0], mem_val: vec![0],
            next_pc: vec![0x104], privilege_mode: vec![3],
            insn_type: vec![9], funct: vec![0], immediate: vec![imm], insn_len: vec![4],
            aux0: vec![0], aux1: vec![0], aux2: vec![0],
            sel_r_alu_add: vec![0], sel_r_alu_sub: vec![0], sel_r_and: vec![0], sel_r_or: vec![0], sel_r_xor: vec![0],
            sel_r_sll: vec![0], sel_r_srl: vec![0], sel_r_sra: vec![0], sel_r_compare: vec![0],
            sel_i_alu_add: vec![0], sel_i_and: vec![0], sel_i_or: vec![0], sel_i_xor: vec![0],
            sel_i_sll: vec![0], sel_i_srl: vec![0], sel_i_sra: vec![0], sel_i_compare: vec![0],
            sel_w_alu_add: vec![0], sel_w_alu_sub: vec![0], sel_w_alu_addi: vec![0],
            sel_w_sll: vec![0], sel_w_srl: vec![0], sel_w_sra: vec![0], sel_w_alu_other: vec![0],
            sel_mul: vec![0], sel_div: vec![0], sel_rem: vec![0], sel_muldiv_other: vec![0],
            sel_load: vec![0], sel_store: vec![0],
            sel_beq: vec![0], sel_bne: vec![0], sel_bltu: vec![0], sel_bgeu: vec![0],
            sel_blt: vec![0], sel_bge: vec![0], sel_jal: vec![0],
            sel_jalr: vec![0], sel_lui: vec![1], sel_auipc: vec![0], sel_csrrw: vec![0], sel_csrrs: vec![0], sel_csrrc: vec![0],
            sel_system: vec![0],
            sel_lr: vec![0], sel_sc: vec![0], sel_amo_swap: vec![0], sel_amo_add: vec![0],
            sel_amo_bitwise: vec![0], sel_amo_compare: vec![0], sel_atomic_other: vec![0],
            sel_mulhu: vec![0],
            sel_csrrw_i: vec![0], sel_csrrs_i: vec![0],
            sel_mulh: vec![0], sel_mulhsu: vec![0], sel_mulw: vec![0], sel_divw: vec![0], sel_remw: vec![0],
            sel_amo_or: vec![0], sel_amo_xor: vec![0],
            sel_amo_maxu: vec![0], sel_amo_mins: vec![0], sel_amo_maxs: vec![0],
        };
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::full();
        let eval = cs.evaluate_on_trace(&polys);

        // LUI body (index 11)
        assert!(eval[11][0].is_zero(), "LUI constraint should be satisfied");
    }

    #[test]
    fn test_extended_branch_not_taken() {
        // BEQ not taken: next_pc = pc + 4, rs1 != rs2 so aux1 = 1 (inequality flag)
        let columns = TraceColumns {
            step: vec![0], pc: vec![0x100],
            rd: vec![0], rd_val_before: vec![0], rd_val_after: vec![0],
            rs1: vec![1], rs1_val: vec![5], rs2: vec![2], rs2_val: vec![10],
            mem_addr: vec![0], mem_val: vec![0],
            next_pc: vec![0x104], privilege_mode: vec![3],
            insn_type: vec![6], funct: vec![0], immediate: vec![0x20], insn_len: vec![4],
            aux0: vec![0], aux1: vec![1], aux2: vec![0],
            sel_r_alu_add: vec![0], sel_r_alu_sub: vec![0], sel_r_and: vec![0], sel_r_or: vec![0], sel_r_xor: vec![0],
            sel_r_sll: vec![0], sel_r_srl: vec![0], sel_r_sra: vec![0], sel_r_compare: vec![0],
            sel_i_alu_add: vec![0], sel_i_and: vec![0], sel_i_or: vec![0], sel_i_xor: vec![0],
            sel_i_sll: vec![0], sel_i_srl: vec![0], sel_i_sra: vec![0], sel_i_compare: vec![0],
            sel_w_alu_add: vec![0], sel_w_alu_sub: vec![0], sel_w_alu_addi: vec![0],
            sel_w_sll: vec![0], sel_w_srl: vec![0], sel_w_sra: vec![0], sel_w_alu_other: vec![0],
            sel_mul: vec![0], sel_div: vec![0], sel_rem: vec![0], sel_muldiv_other: vec![0],
            sel_load: vec![0], sel_store: vec![0],
            sel_beq: vec![1], sel_bne: vec![0], sel_bltu: vec![0], sel_bgeu: vec![0],
            sel_blt: vec![0], sel_bge: vec![0], sel_jal: vec![0],
            sel_jalr: vec![0], sel_lui: vec![0], sel_auipc: vec![0], sel_csrrw: vec![0], sel_csrrs: vec![0], sel_csrrc: vec![0],
            sel_system: vec![0],
            sel_lr: vec![0], sel_sc: vec![0], sel_amo_swap: vec![0], sel_amo_add: vec![0],
            sel_amo_bitwise: vec![0], sel_amo_compare: vec![0], sel_atomic_other: vec![0],
            sel_mulhu: vec![0],
            sel_csrrw_i: vec![0], sel_csrrs_i: vec![0],
            sel_mulh: vec![0], sel_mulhsu: vec![0], sel_mulw: vec![0], sel_divw: vec![0], sel_remw: vec![0],
            sel_amo_or: vec![0], sel_amo_xor: vec![0],
            sel_amo_maxu: vec![0], sel_amo_mins: vec![0], sel_amo_maxs: vec![0],
        };
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::full();
        let eval = cs.evaluate_on_trace(&polys);

        // BEQ condition (index 4) - branch should be zero
        assert!(eval[4][0].is_zero(), "Branch not-taken constraint should be satisfied");
    }

    #[test]
    fn test_trace_to_polynomials() {
        // Build a small trace simulating two ADDI instructions:
        //   step 0: pc=0, ADDI x1, x0, 42  -> rd=1, rd_before=0, rd_after=42, next_pc=4
        //   step 1: pc=4, ADDI x2, x1, 1   -> rd=2, rd_before=0, rd_after=43, next_pc=8
        let columns = TraceColumns {
            step: vec![0, 1],
            pc: vec![0, 4],
            rd: vec![1, 2],
            rd_val_before: vec![0, 0],
            rd_val_after: vec![42, 43],
            rs1: vec![0, 1],
            rs1_val: vec![0, 42],
            rs2: vec![0, 0],
            rs2_val: vec![0, 0],
            mem_addr: vec![0, 0],
            mem_val: vec![0, 0],
            next_pc: vec![4, 8],
            privilege_mode: vec![3, 3],
            insn_type: vec![1, 1],  // I-type ALU
            funct: vec![0, 0],      // ADDI
            immediate: vec![42, 1],
            insn_len: vec![4, 4],
            aux0: vec![0, 0],
            aux1: vec![0, 0],
            aux2: vec![0, 0],
            sel_r_alu_add: vec![0, 0], sel_r_alu_sub: vec![0, 0], sel_r_and: vec![0, 0], sel_r_or: vec![0, 0], sel_r_xor: vec![0, 0],
            sel_r_sll: vec![0, 0], sel_r_srl: vec![0, 0], sel_r_sra: vec![0, 0], sel_r_compare: vec![0, 0],
            sel_i_alu_add: vec![1, 1], sel_i_and: vec![0, 0], sel_i_or: vec![0, 0], sel_i_xor: vec![0, 0],
            sel_i_sll: vec![0, 0], sel_i_srl: vec![0, 0], sel_i_sra: vec![0, 0], sel_i_compare: vec![0, 0],
            sel_w_alu_add: vec![0, 0], sel_w_alu_sub: vec![0, 0], sel_w_alu_addi: vec![0, 0],
            sel_w_sll: vec![0, 0], sel_w_srl: vec![0, 0], sel_w_sra: vec![0, 0], sel_w_alu_other: vec![0, 0],
            sel_mul: vec![0, 0], sel_div: vec![0, 0], sel_rem: vec![0, 0], sel_muldiv_other: vec![0, 0],
            sel_load: vec![0, 0], sel_store: vec![0, 0],
            sel_beq: vec![0, 0], sel_bne: vec![0, 0], sel_bltu: vec![0, 0], sel_bgeu: vec![0, 0],
            sel_blt: vec![0, 0], sel_bge: vec![0, 0], sel_jal: vec![0, 0],
            sel_jalr: vec![0, 0], sel_lui: vec![0, 0], sel_auipc: vec![0, 0], sel_csrrw: vec![0, 0], sel_csrrs: vec![0, 0], sel_csrrc: vec![0, 0],
            sel_system: vec![0, 0],
            sel_lr: vec![0, 0], sel_sc: vec![0, 0], sel_amo_swap: vec![0, 0], sel_amo_add: vec![0, 0],
            sel_amo_bitwise: vec![0, 0], sel_amo_compare: vec![0, 0], sel_atomic_other: vec![0, 0],
            sel_mulhu: vec![0, 0],
            sel_csrrw_i: vec![0, 0], sel_csrrs_i: vec![0, 0],
            sel_mulh: vec![0, 0], sel_mulhsu: vec![0, 0], sel_mulw: vec![0, 0], sel_divw: vec![0, 0], sel_remw: vec![0, 0],
            sel_amo_or: vec![0, 0], sel_amo_xor: vec![0, 0],
            sel_amo_maxu: vec![0, 0], sel_amo_mins: vec![0, 0], sel_amo_maxs: vec![0, 0],
        };

        let polys = super::trace_polys_from_columns(&columns);

        assert_eq!(polys.num_rows, 2);
        assert_eq!(polys.padded_size, 16);

        // Verify pc column: [0, 4]
        assert!(polys.columns[COL_PC].evaluations[0].is_zero());
        let four = Scalar::from_u64(4, CurveType::Bls48581);
        assert!(polys.columns[COL_PC].evaluations[1].sub(&four).is_zero());

        // Verify rd_val_after column: [42, 43]
        let forty_two = Scalar::from_u64(42, CurveType::Bls48581);
        assert!(polys.columns[COL_RD_VAL_AFTER].evaluations[0].sub(&forty_two).is_zero());

        let forty_three = Scalar::from_u64(43, CurveType::Bls48581);
        assert!(polys.columns[COL_RD_VAL_AFTER].evaluations[1].sub(&forty_three).is_zero());

        // Verify next_pc column: [4, 8]
        let eight = Scalar::from_u64(8, CurveType::Bls48581);
        assert!(polys.columns[COL_NEXT_PC].evaluations[1].sub(&eight).is_zero());
    }

    #[test]
    fn test_prove_verify_roundtrip() {
        metavm_zkp::commitment::init();

        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::full();

        let proof = metavm_zkp::prover::prove(&polys, &cs);

        let valid = metavm_zkp::verifier::verify(&proof, &cs);
        assert!(valid, "Valid trace should produce a verifying proof");
    }

    #[test]
    fn test_prove_verify_bls48581_scheme() {
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();

        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::full();

        let proof = metavm_zkp::prover::prove_with_scheme(&polys, &cs, &scheme);
        let valid = metavm_zkp::verifier::verify_with_scheme(
            &proof, &cs, &scheme, metavm_zkp::field::CurveType::Bls48581,
        );
        assert!(valid, "RISC-V BLS48-581 scheme prove/verify should succeed");
    }

    #[test]
    fn test_prove_verify_bls12381_scheme() {
        use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let scheme = Bls12381Scheme::new();
        scheme.init();

        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns_with_curve(
            &columns, metavm_zkp::field::CurveType::Bls12381,
        );
        let cs = RiscvConstraintSystem::full();

        let proof = metavm_zkp::prover::prove_with_scheme(&polys, &cs, &scheme);
        let valid = metavm_zkp::verifier::verify_with_scheme(
            &proof, &cs, &scheme, metavm_zkp::field::CurveType::Bls12381,
        );
        assert!(valid, "RISC-V BLS12-381 scheme prove/verify should succeed");
    }

    /// Regression: real RISC-V ChunkProof must verify through
    /// `begin_chunk_scheme` + `verify_final_scheme` (the recursive
    /// accumulator path the prove-elf / prove-boot CLIs use).
    ///
    /// Companion to the EVM regression in
    /// `metavm-evm/src/constraints/mod.rs::evm_chunk_proof_verifies_through_recursive_accumulator`.
    /// Locks in the bitwise transcript fix in
    /// `recursive::recover_chunk_challenges_scheme`.
    #[test]
    #[ignore = "slow: full prove + recursive verify on RISC-V trace; run with --release --ignored"]
    fn riscv_chunk_proof_verifies_through_recursive_accumulator() {
        use metavm_zkp::prover::prove_chunk_with_scheme;
        use metavm_zkp::recursive::{begin_chunk_scheme, verify_final_scheme};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = metavm_zkp::field::CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::full();

        let zero = [0u8; 32];
        let chunk_proof = prove_chunk_with_scheme(&polys, &cs, 0, &zero, &zero, &scheme);

        let recursive = begin_chunk_scheme(chunk_proof, &scheme, curve);
        let valid = verify_final_scheme(&recursive, &scheme);
        assert!(
            valid,
            "RISC-V ChunkProof must verify through recursive accumulator \
             (regression guard for the bitwise transcript fix in \
             `recover_chunk_challenges_scheme`)",
        );
    }

    /// The recursive scalar accumulator must reject a tampered constraint
    /// evaluation. RISC-V exercises the full set of auxiliary contributions
    /// (logup + memory perm + register perm + bitwise), so this verifies
    /// every branch of `verifier::compute_c_at_z` feeds into the per-chunk
    /// `c_check = Q(z)·Z_H(z) - C(z)` accumulator.
    #[test]
    #[ignore = "slow: full prove + recursive fold on RISC-V trace; run with --release --ignored"]
    fn riscv_recursive_full_scheme_rejects_tampered_evaluation() {
        use metavm_zkp::prover::prove_chunk_with_scheme;
        use metavm_zkp::recursive::{begin_chunk_full_scheme, verify_final_scheme};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = metavm_zkp::field::CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let columns = make_valid_trace(4);
        let polys = super::trace_polys_from_columns(&columns);
        let cs = RiscvConstraintSystem::full();

        let zero = [0u8; 32];
        let chunk_proof = prove_chunk_with_scheme(&polys, &cs, 0, &zero, &zero, &scheme);

        // Sanity: untouched proof must verify with c_check folded in.
        let recursive_ok = begin_chunk_full_scheme(chunk_proof.clone(), &scheme, curve, &cs);
        assert!(
            verify_final_scheme(&recursive_ok, &scheme),
            "untouched RISC-V chunk must verify through full-scheme recursive accumulator",
        );

        // Tamper with a column evaluation. The constraint identity check
        // `Q(z)·Z_H(z) - C(z)` must now be non-zero, so `scalar_acc` is
        // non-zero and `verify_final_scheme` rejects.
        let mut tampered = chunk_proof.clone();
        assert!(
            !tampered.execution_proof.evaluations.is_empty(),
            "RISC-V proof must have evaluations to tamper with",
        );
        // Flip a low-order byte of the first column evaluation.
        let target = &mut tampered.execution_proof.evaluations[0];
        let last = target.len() - 1;
        target[last] ^= 0x01;

        let recursive_bad = begin_chunk_full_scheme(tampered, &scheme, curve, &cs);
        assert!(
            !verify_final_scheme(&recursive_bad, &scheme),
            "RISC-V chunk with tampered evaluation MUST be rejected by the \
             scalar_acc check",
        );
    }
}
