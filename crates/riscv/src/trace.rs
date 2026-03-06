use crate::decode;
use crate::isa::Instruction;
use crate::vm::{Vm, VmError};

/// Classify an instruction into (insn_type, funct) for the trace.
/// Must match the constants in zkp::selectors.
fn classify_instruction(insn: &Instruction) -> (u8, u8) {
    match insn {
        // R-type ALU (type 0)
        Instruction::ADD  { .. } => (0, 0),
        Instruction::SUB  { .. } => (0, 1),
        Instruction::SLL  { .. } => (0, 2),
        Instruction::SLT  { .. } => (0, 3),
        Instruction::SLTU { .. } => (0, 4),
        Instruction::XOR  { .. } => (0, 5),
        Instruction::SRL  { .. } => (0, 6),
        Instruction::SRA  { .. } => (0, 7),
        Instruction::OR   { .. } => (0, 8),
        Instruction::AND  { .. } => (0, 9),
        // I-type ALU (type 1)
        Instruction::ADDI  { .. } => (1, 0),
        Instruction::SLTI  { .. } => (1, 1),
        Instruction::SLTIU { .. } => (1, 2),
        Instruction::XORI  { .. } => (1, 3),
        Instruction::ORI   { .. } => (1, 4),
        Instruction::ANDI  { .. } => (1, 5),
        Instruction::SLLI  { .. } => (1, 6),
        Instruction::SRLI  { .. } => (1, 7),
        Instruction::SRAI  { .. } => (1, 8),
        // W-type ALU (type 2)
        Instruction::ADDW  { .. } => (2, 0),
        Instruction::SUBW  { .. } => (2, 1),
        Instruction::SLLW  { .. } => (2, 2),
        Instruction::SRLW  { .. } => (2, 3),
        Instruction::SRAW  { .. } => (2, 4),
        Instruction::ADDIW { .. } => (2, 5),
        Instruction::SLLIW { .. } => (2, 6),
        Instruction::SRLIW { .. } => (2, 7),
        Instruction::SRAIW { .. } => (2, 8),
        // MUL/DIV (type 3)
        Instruction::MUL    { .. } => (3, 0),
        Instruction::MULH   { .. } => (3, 1),
        Instruction::MULHSU { .. } => (3, 2),
        Instruction::MULHU  { .. } => (3, 3),
        Instruction::DIV    { .. } => (3, 4),
        Instruction::DIVU   { .. } => (3, 5),
        Instruction::REM    { .. } => (3, 6),
        Instruction::REMU   { .. } => (3, 7),
        Instruction::MULW   { .. } => (3, 8),
        Instruction::DIVW   { .. } => (3, 9),
        Instruction::DIVUW  { .. } => (3, 10),
        Instruction::REMW   { .. } => (3, 11),
        Instruction::REMUW  { .. } => (3, 12),
        // Loads (type 4)
        Instruction::LB  { .. } => (4, 0),
        Instruction::LH  { .. } => (4, 1),
        Instruction::LW  { .. } => (4, 2),
        Instruction::LD  { .. } => (4, 3),
        Instruction::LBU { .. } => (4, 4),
        Instruction::LHU { .. } => (4, 5),
        Instruction::LWU { .. } => (4, 6),
        // Stores (type 5)
        Instruction::SB { .. } => (5, 0),
        Instruction::SH { .. } => (5, 1),
        Instruction::SW { .. } => (5, 2),
        Instruction::SD { .. } => (5, 3),
        // Branches (type 6)
        Instruction::BEQ  { .. } => (6, 0),
        Instruction::BNE  { .. } => (6, 1),
        Instruction::BLT  { .. } => (6, 2),
        Instruction::BGE  { .. } => (6, 3),
        Instruction::BLTU { .. } => (6, 4),
        Instruction::BGEU { .. } => (6, 5),
        // Control flow
        Instruction::JAL  { .. } => (7, 0),
        Instruction::JALR { .. } => (8, 0),
        Instruction::LUI  { .. } => (9, 0),
        Instruction::AUIPC { .. } => (10, 0),
        // CSR (type 11)
        Instruction::CSRRW  { .. } => (11, 0),
        Instruction::CSRRS  { .. } => (11, 1),
        Instruction::CSRRC  { .. } => (11, 2),
        Instruction::CSRRWI { .. } => (11, 3),
        Instruction::CSRRSI { .. } => (11, 4),
        Instruction::CSRRCI { .. } => (11, 5),
        // System (type 12)
        Instruction::ECALL         => (12, 0),
        Instruction::EBREAK        => (12, 1),
        Instruction::MRET          => (12, 2),
        Instruction::SRET          => (12, 3),
        Instruction::WFI           => (12, 4),
        Instruction::FENCE { .. }  => (12, 5),
        Instruction::SFENCE_VMA { .. } => (12, 6),
        Instruction::FENCE_I       => (12, 7),
        // Atomics (type 13)
        Instruction::LR_W      { .. } => (13, 0),
        Instruction::LR_D      { .. } => (13, 1),
        Instruction::SC_W      { .. } => (13, 2),
        Instruction::SC_D      { .. } => (13, 3),
        Instruction::AMOSWAP_W { .. } => (13, 4),
        Instruction::AMOSWAP_D { .. } => (13, 5),
        Instruction::AMOADD_W  { .. } => (13, 6),
        Instruction::AMOADD_D  { .. } => (13, 7),
        Instruction::AMOAND_W  { .. } => (13, 8),
        Instruction::AMOAND_D  { .. } => (13, 9),
        Instruction::AMOOR_W   { .. } => (13, 10),
        Instruction::AMOOR_D   { .. } => (13, 11),
        Instruction::AMOXOR_W  { .. } => (13, 12),
        Instruction::AMOXOR_D  { .. } => (13, 13),
        Instruction::AMOMAX_W  { .. } => (13, 14),
        Instruction::AMOMAX_D  { .. } => (13, 15),
        Instruction::AMOMIN_W  { .. } => (13, 16),
        Instruction::AMOMIN_D  { .. } => (13, 17),
        Instruction::AMOMAXU_W { .. } => (13, 18),
        Instruction::AMOMAXU_D { .. } => (13, 19),
        Instruction::AMOMINU_W { .. } => (13, 20),
        Instruction::AMOMINU_D { .. } => (13, 21),
    }
}

/// Extract the immediate from an instruction, sign-extended to i64.
fn extract_immediate(insn: &Instruction) -> i64 {
    match insn {
        Instruction::LUI { imm, .. } | Instruction::AUIPC { imm, .. } => *imm as i64,
        Instruction::JAL { imm, .. } => *imm as i64,
        Instruction::JALR { imm, .. }
        | Instruction::LB { imm, .. } | Instruction::LH { imm, .. }
        | Instruction::LW { imm, .. } | Instruction::LD { imm, .. }
        | Instruction::LBU { imm, .. } | Instruction::LHU { imm, .. }
        | Instruction::LWU { imm, .. }
        | Instruction::ADDI { imm, .. } | Instruction::SLTI { imm, .. }
        | Instruction::SLTIU { imm, .. } | Instruction::XORI { imm, .. }
        | Instruction::ORI { imm, .. } | Instruction::ANDI { imm, .. }
        | Instruction::SLLI { imm, .. } | Instruction::SRLI { imm, .. }
        | Instruction::SRAI { imm, .. }
        | Instruction::ADDIW { imm, .. } | Instruction::SLLIW { imm, .. }
        | Instruction::SRLIW { imm, .. } | Instruction::SRAIW { imm, .. } => *imm as i64,
        Instruction::SB { imm, .. } | Instruction::SH { imm, .. }
        | Instruction::SW { imm, .. } | Instruction::SD { imm, .. } => *imm as i64,
        Instruction::BEQ { imm, .. } | Instruction::BNE { imm, .. }
        | Instruction::BLT { imm, .. } | Instruction::BGE { imm, .. }
        | Instruction::BLTU { imm, .. } | Instruction::BGEU { imm, .. } => *imm as i64,
        Instruction::CSRRWI { uimm, .. }
        | Instruction::CSRRSI { uimm, .. }
        | Instruction::CSRRCI { uimm, .. } => *uimm as i64,
        _ => 0,
    }
}

/// Type of memory operation in a trace row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemOp {
    None,
    Read,
    Write,
}

/// A single row of the execution trace, capturing the full state transition for one step.
#[derive(Clone, Debug)]
pub struct TraceRow {
    /// Step number (0-indexed).
    pub step: u64,
    /// Program counter before this step.
    pub pc: u64,
    /// The decoded instruction.
    pub instruction: Instruction,
    /// Destination register index (0 if none).
    pub rd: u8,
    /// Value of rd before the instruction executed.
    pub rd_val_before: u64,
    /// Value of rd after the instruction executed.
    pub rd_val_after: u64,
    /// First source register index.
    pub rs1: u8,
    /// Value of rs1.
    pub rs1_val: u64,
    /// Second source register index.
    pub rs2: u8,
    /// Value of rs2.
    pub rs2_val: u64,
    /// Memory address accessed (0 if no memory op).
    pub mem_addr: u64,
    /// Memory value read or written (0 if no memory op).
    pub mem_val: u64,
    /// Type of memory operation.
    pub mem_op: MemOp,
    /// PC after the instruction executed.
    pub next_pc: u64,
    /// Privilege mode before the instruction executed (0=U, 1=S, 3=M).
    pub privilege_mode: u8,
    /// Instruction length in bytes (2 or 4).
    pub insn_len: u8,
    /// Primary auxiliary value (carry, borrow, mul_hi, truncation quotient, shift power).
    pub aux0: u64,
    /// Secondary auxiliary value (remainder, sign bit, shift overflow/remainder).
    pub aux1: u64,
    /// Tertiary auxiliary value (SRA overflow correction).
    pub aux2: u64,
}

/// A tracing VM that wraps a regular VM and records each execution step.
pub struct TracingVm {
    pub vm: Vm,
    pub trace: Vec<TraceRow>,
}

impl TracingVm {
    pub fn new(vm: Vm) -> Self {
        TracingVm {
            vm,
            trace: Vec::new(),
        }
    }

    /// Execute one step, recording the trace.
    pub fn step(&mut self) -> Result<(), VmError> {
        if self.vm.halted {
            return Err(VmError::Halted);
        }

        let pc_before = self.vm.cpu.pc;
        let priv_before = self.vm.cpu.priv_mode as u8;

        // Fetch and decode using variable-length fetch
        let fetched = decode::fetch_decode(&self.vm.memory, pc_before)?;
        let instruction = fetched.instruction;
        let insn_len = fetched.len;

        // Capture pre-execution register state
        let rd = instruction.rd().unwrap_or(0);
        let rs1 = instruction.rs1().unwrap_or(0);
        let rs2 = instruction.rs2().unwrap_or(0);
        let rd_val_before = self.vm.cpu.read_reg(rd);
        let rs1_val = self.vm.cpu.read_reg(rs1);
        let rs2_val = self.vm.cpu.read_reg(rs2);

        // Determine memory address for load/store instructions (pre-compute)
        let (mem_addr, mem_op) = compute_mem_info(&instruction, &self.vm);

        // Execute the instruction
        self.vm.step()?;

        // Capture post-execution state
        let rd_val_after = self.vm.cpu.read_reg(rd);
        let next_pc = self.vm.cpu.pc;

        // For memory ops, capture the value
        // AMO ops write the computed RMW result, not rs2_val
        let mem_val = match mem_op {
            MemOp::Read => rd_val_after,
            MemOp::Write => compute_mem_val(&instruction, rd_val_after, rs2_val),
            MemOp::None => 0,
        };

        let (aux0, aux1, aux2) = compute_aux_values(&instruction, rs1_val, rs2_val, rd_val_after);

        let step_num = self.trace.len() as u64;
        self.trace.push(TraceRow {
            step: step_num,
            pc: pc_before,
            instruction,
            rd,
            rd_val_before,
            rd_val_after,
            rs1,
            rs1_val,
            rs2,
            rs2_val,
            mem_addr,
            mem_val,
            mem_op,
            next_pc,
            privilege_mode: priv_before,
            insn_len,
            aux0,
            aux1,
            aux2,
        });

        Ok(())
    }

    /// Run until halted or max_steps reached.
    pub fn run(&mut self, max_steps: u64) -> Result<u64, VmError> {
        let mut steps = 0;
        while !self.vm.halted && steps < max_steps {
            self.step()?;
            steps += 1;
        }
        Ok(steps)
    }

    /// Export trace as column vectors for polynomial commitment.
    pub fn export_columns(&self) -> TraceColumns {
        export_trace_rows_to_columns(&self.trace)
    }
}

/// Column names for the RISC-V trace (matches column order in TraceColumns).
pub const COLUMN_NAMES: [&str; 84] = [
    "step", "pc", "rd", "rd_val_before", "rd_val_after",
    "rs1", "rs1_val", "rs2", "rs2_val",
    "mem_addr", "mem_val", "next_pc", "privilege_mode",
    "insn_type", "funct", "immediate", "insn_len",
    "aux0", "aux1", "aux2",
    "sel_r_alu_add", "sel_r_alu_sub",
    "sel_r_and", "sel_r_or", "sel_r_xor",
    "sel_r_sll", "sel_r_srl", "sel_r_sra", "sel_r_compare",
    "sel_i_alu_add",
    "sel_i_and", "sel_i_or", "sel_i_xor",
    "sel_i_sll", "sel_i_srl", "sel_i_sra", "sel_i_compare",
    "sel_w_alu_add", "sel_w_alu_sub", "sel_w_alu_addi",
    "sel_w_sll", "sel_w_srl", "sel_w_sra", "sel_w_alu_other",
    "sel_mul", "sel_div", "sel_rem", "sel_muldiv_other",
    "sel_load", "sel_store",
    "sel_beq", "sel_bne", "sel_bltu", "sel_bgeu", "sel_blt", "sel_bge",
    "sel_jal", "sel_jalr", "sel_lui", "sel_auipc",
    "sel_csrrw", "sel_csrrs", "sel_csrrc",
    "sel_system",
    "sel_lr", "sel_sc", "sel_amo_swap", "sel_amo_add",
    "sel_amo_bitwise", "sel_amo_compare", "sel_atomic_other",
    "sel_mulhu",
    "sel_csrrw_i", "sel_csrrs_i",
    "sel_mulh", "sel_mulhsu", "sel_mulw", "sel_divw", "sel_remw",
    "sel_amo_or", "sel_amo_xor", "sel_amo_maxu", "sel_amo_mins", "sel_amo_maxs",
];

/// Column-oriented representation of the execution trace for ZK proofs.
pub struct TraceColumns {
    pub step: Vec<u64>,
    pub pc: Vec<u64>,
    pub rd: Vec<u64>,
    pub rd_val_before: Vec<u64>,
    pub rd_val_after: Vec<u64>,
    pub rs1: Vec<u64>,
    pub rs1_val: Vec<u64>,
    pub rs2: Vec<u64>,
    pub rs2_val: Vec<u64>,
    pub mem_addr: Vec<u64>,
    pub mem_val: Vec<u64>,
    pub next_pc: Vec<u64>,
    pub privilege_mode: Vec<u64>,
    pub insn_type: Vec<u64>,
    pub funct: Vec<u64>,
    pub immediate: Vec<u64>,
    pub insn_len: Vec<u64>,
    pub aux0: Vec<u64>,
    pub aux1: Vec<u64>,
    pub aux2: Vec<u64>,
    pub sel_r_alu_add: Vec<u64>,
    pub sel_r_alu_sub: Vec<u64>,
    pub sel_r_and: Vec<u64>,
    pub sel_r_or: Vec<u64>,
    pub sel_r_xor: Vec<u64>,
    pub sel_r_sll: Vec<u64>,
    pub sel_r_srl: Vec<u64>,
    pub sel_r_sra: Vec<u64>,
    pub sel_r_compare: Vec<u64>,
    pub sel_i_alu_add: Vec<u64>,
    pub sel_i_and: Vec<u64>,
    pub sel_i_or: Vec<u64>,
    pub sel_i_xor: Vec<u64>,
    pub sel_i_sll: Vec<u64>,
    pub sel_i_srl: Vec<u64>,
    pub sel_i_sra: Vec<u64>,
    pub sel_i_compare: Vec<u64>,
    pub sel_w_alu_add: Vec<u64>,
    pub sel_w_alu_sub: Vec<u64>,
    pub sel_w_alu_addi: Vec<u64>,
    pub sel_w_sll: Vec<u64>,
    pub sel_w_srl: Vec<u64>,
    pub sel_w_sra: Vec<u64>,
    pub sel_w_alu_other: Vec<u64>,
    pub sel_mul: Vec<u64>,
    pub sel_div: Vec<u64>,
    pub sel_rem: Vec<u64>,
    pub sel_muldiv_other: Vec<u64>,
    pub sel_load: Vec<u64>,
    pub sel_store: Vec<u64>,
    pub sel_beq: Vec<u64>,
    pub sel_bne: Vec<u64>,
    pub sel_bltu: Vec<u64>,
    pub sel_bgeu: Vec<u64>,
    pub sel_blt: Vec<u64>,
    pub sel_bge: Vec<u64>,
    pub sel_jal: Vec<u64>,
    pub sel_jalr: Vec<u64>,
    pub sel_lui: Vec<u64>,
    pub sel_auipc: Vec<u64>,
    pub sel_csrrw: Vec<u64>,
    pub sel_csrrs: Vec<u64>,
    pub sel_csrrc: Vec<u64>,
    pub sel_system: Vec<u64>,
    pub sel_lr: Vec<u64>,
    pub sel_sc: Vec<u64>,
    pub sel_amo_swap: Vec<u64>,
    pub sel_amo_add: Vec<u64>,
    pub sel_amo_bitwise: Vec<u64>,
    pub sel_amo_compare: Vec<u64>,
    pub sel_atomic_other: Vec<u64>,
    pub sel_mulhu: Vec<u64>,
    pub sel_csrrw_i: Vec<u64>,
    pub sel_csrrs_i: Vec<u64>,
    pub sel_mulh: Vec<u64>,
    pub sel_mulhsu: Vec<u64>,
    pub sel_mulw: Vec<u64>,
    pub sel_divw: Vec<u64>,
    pub sel_remw: Vec<u64>,
    pub sel_amo_or: Vec<u64>,
    pub sel_amo_xor: Vec<u64>,
    pub sel_amo_maxu: Vec<u64>,
    pub sel_amo_mins: Vec<u64>,
    pub sel_amo_maxs: Vec<u64>,
}

impl metavm_core::vm_traits::VmTrace for TraceColumns {
    fn num_steps(&self) -> usize {
        self.step.len()
    }

    fn num_columns(&self) -> usize {
        84
    }

    fn column(&self, index: usize) -> &[u64] {
        match index {
            0 => &self.step,
            1 => &self.pc,
            2 => &self.rd,
            3 => &self.rd_val_before,
            4 => &self.rd_val_after,
            5 => &self.rs1,
            6 => &self.rs1_val,
            7 => &self.rs2,
            8 => &self.rs2_val,
            9 => &self.mem_addr,
            10 => &self.mem_val,
            11 => &self.next_pc,
            12 => &self.privilege_mode,
            13 => &self.insn_type,
            14 => &self.funct,
            15 => &self.immediate,
            16 => &self.insn_len,
            17 => &self.aux0,
            18 => &self.aux1,
            19 => &self.aux2,
            20 => &self.sel_r_alu_add,
            21 => &self.sel_r_alu_sub,
            22 => &self.sel_r_and,
            23 => &self.sel_r_or,
            24 => &self.sel_r_xor,
            25 => &self.sel_r_sll,
            26 => &self.sel_r_srl,
            27 => &self.sel_r_sra,
            28 => &self.sel_r_compare,
            29 => &self.sel_i_alu_add,
            30 => &self.sel_i_and,
            31 => &self.sel_i_or,
            32 => &self.sel_i_xor,
            33 => &self.sel_i_sll,
            34 => &self.sel_i_srl,
            35 => &self.sel_i_sra,
            36 => &self.sel_i_compare,
            37 => &self.sel_w_alu_add,
            38 => &self.sel_w_alu_sub,
            39 => &self.sel_w_alu_addi,
            40 => &self.sel_w_sll,
            41 => &self.sel_w_srl,
            42 => &self.sel_w_sra,
            43 => &self.sel_w_alu_other,
            44 => &self.sel_mul,
            45 => &self.sel_div,
            46 => &self.sel_rem,
            47 => &self.sel_muldiv_other,
            48 => &self.sel_load,
            49 => &self.sel_store,
            50 => &self.sel_beq,
            51 => &self.sel_bne,
            52 => &self.sel_bltu,
            53 => &self.sel_bgeu,
            54 => &self.sel_blt,
            55 => &self.sel_bge,
            56 => &self.sel_jal,
            57 => &self.sel_jalr,
            58 => &self.sel_lui,
            59 => &self.sel_auipc,
            60 => &self.sel_csrrw,
            61 => &self.sel_csrrs,
            62 => &self.sel_csrrc,
            63 => &self.sel_system,
            64 => &self.sel_lr,
            65 => &self.sel_sc,
            66 => &self.sel_amo_swap,
            67 => &self.sel_amo_add,
            68 => &self.sel_amo_bitwise,
            69 => &self.sel_amo_compare,
            70 => &self.sel_atomic_other,
            71 => &self.sel_mulhu,
            72 => &self.sel_csrrw_i,
            73 => &self.sel_csrrs_i,
            74 => &self.sel_mulh,
            75 => &self.sel_mulhsu,
            76 => &self.sel_mulw,
            77 => &self.sel_divw,
            78 => &self.sel_remw,
            79 => &self.sel_amo_or,
            80 => &self.sel_amo_xor,
            81 => &self.sel_amo_maxu,
            82 => &self.sel_amo_mins,
            83 => &self.sel_amo_maxs,
            _ => panic!("column index {} out of range (0..84)", index),
        }
    }

    fn columns(&self) -> Vec<&[u64]> {
        vec![
            &self.step, &self.pc, &self.rd, &self.rd_val_before, &self.rd_val_after,
            &self.rs1, &self.rs1_val, &self.rs2, &self.rs2_val,
            &self.mem_addr, &self.mem_val, &self.next_pc, &self.privilege_mode,
            &self.insn_type, &self.funct, &self.immediate, &self.insn_len,
            &self.aux0, &self.aux1, &self.aux2,
            &self.sel_r_alu_add, &self.sel_r_alu_sub,
            &self.sel_r_and, &self.sel_r_or, &self.sel_r_xor,
            &self.sel_r_sll, &self.sel_r_srl, &self.sel_r_sra, &self.sel_r_compare,
            &self.sel_i_alu_add,
            &self.sel_i_and, &self.sel_i_or, &self.sel_i_xor,
            &self.sel_i_sll, &self.sel_i_srl, &self.sel_i_sra, &self.sel_i_compare,
            &self.sel_w_alu_add, &self.sel_w_alu_sub, &self.sel_w_alu_addi,
            &self.sel_w_sll, &self.sel_w_srl, &self.sel_w_sra, &self.sel_w_alu_other,
            &self.sel_mul, &self.sel_div, &self.sel_rem, &self.sel_muldiv_other,
            &self.sel_load, &self.sel_store,
            &self.sel_beq, &self.sel_bne, &self.sel_bltu, &self.sel_bgeu,
            &self.sel_blt, &self.sel_bge,
            &self.sel_jal, &self.sel_jalr, &self.sel_lui, &self.sel_auipc,
            &self.sel_csrrw, &self.sel_csrrs, &self.sel_csrrc,
            &self.sel_system,
            &self.sel_lr, &self.sel_sc, &self.sel_amo_swap, &self.sel_amo_add,
            &self.sel_amo_bitwise, &self.sel_amo_compare, &self.sel_atomic_other,
            &self.sel_mulhu,
            &self.sel_csrrw_i, &self.sel_csrrs_i,
            &self.sel_mulh, &self.sel_mulhsu, &self.sel_mulw, &self.sel_divw, &self.sel_remw,
            &self.sel_amo_or, &self.sel_amo_xor, &self.sel_amo_maxu, &self.sel_amo_mins, &self.sel_amo_maxs,
        ]
    }

    fn column_names(&self) -> Vec<&'static str> {
        COLUMN_NAMES.to_vec()
    }
}

/// Compute auxiliary values for a trace row based on the instruction type.
///
/// Returns (aux0, aux1, aux2) where:
/// - ADD/ADDI: aux0 = carry (overflow from u64 addition)
/// - SUB: aux0 = borrow (1 if rs1 < rs2, else 0)
/// - SLL: aux0 = 2^k (power of 2), aux1 = overflow = (rs1 * 2^k) >> 64
/// - SRL: aux0 = 2^k, aux1 = remainder = rs1 mod 2^k
/// - SRA: aux0 = 2^k, aux1 = remainder = rs1 mod 2^k, aux2 = sign_bit * (2^k - 1)
/// - MUL: aux0 = upper 64 bits of 128-bit product
/// - DIV/DIVU: aux0 = remainder
/// - REM/REMU: aux0 = quotient
/// - SB: aux0 = rs2_val >> 8 (truncation quotient)
/// - SH: aux0 = rs2_val >> 16
/// - SW: aux0 = rs2_val >> 32
/// - W-variants: aux0 = upper 32 bits before truncation
/// - SLTU: aux0 = unsigned_lt (1 if rs1 < rs2, else 0), aux1=0, aux2=0
/// - SLT: aux0 = unsigned_lt, aux1 = sign(rs1), aux2 = sign(rs2)
/// - SLTIU: aux0 = unsigned_lt (1 if rs1 < imm, else 0), aux1=0, aux2=0
/// - SLTI: aux0 = unsigned_lt, aux1 = sign(rs1), aux2 = sign(imm)
/// - CSRRW/CSRRWI: aux0 = new CSR value (rs1_val or uimm)
/// - CSRRS/CSRRSI: aux0 = old | source, aux1 = AND(old, source)
/// - CSRRC/CSRRCI: aux0 = old & ~source, aux1 = AND(old, source)
/// - SC: aux0 = 1 if reservation valid (rd=0), 0 if invalid (rd!=0)
pub fn compute_aux_values(insn: &Instruction, rs1_val: u64, rs2_val: u64, rd_val_after: u64) -> (u64, u64, u64) {
    match insn {
        // R-type ALU
        Instruction::ADD { .. } => {
            let carry = ((rs1_val as u128 + rs2_val as u128) >> 64) as u64;
            (carry, 0, 0)
        }
        Instruction::SUB { .. } => {
            let borrow = if rs1_val < rs2_val { 1u64 } else { 0u64 };
            (borrow, 0, 0)
        }
        Instruction::SLTU { .. } => {
            // aux0 = unsigned_lt (binary), aux1=0, aux2=0
            let unsigned_lt = if rs1_val < rs2_val { 1u64 } else { 0u64 };
            (unsigned_lt, 0, 0)
        }
        Instruction::SLT { .. } => {
            // aux0 = unsigned_lt, aux1 = sign(rs1), aux2 = sign(rs2)
            let unsigned_lt = if rs1_val < rs2_val { 1u64 } else { 0u64 };
            let sign_rs1 = (rs1_val >> 63) & 1;
            let sign_rs2 = (rs2_val >> 63) & 1;
            (unsigned_lt, sign_rs1, sign_rs2)
        }

        // R-type shifts
        Instruction::SLL { .. } => {
            let k = rs2_val & 0x3F; // lower 6 bits
            let power = 1u64 << k;
            let overflow = ((rs1_val as u128 * power as u128) >> 64) as u64;
            (power, overflow, 0)
        }
        Instruction::SRL { .. } => {
            let k = rs2_val & 0x3F;
            let power = 1u64 << k;
            let remainder = if k == 0 { 0 } else { rs1_val % power };
            (power, remainder, 0)
        }
        Instruction::SRA { .. } => {
            let k = rs2_val & 0x3F;
            let power = 1u64 << k;
            let remainder = if k == 0 { 0 } else { rs1_val % power };
            let sign_bit = (rs1_val >> 63) & 1;
            let overflow_correction = sign_bit * (power.wrapping_sub(1));
            (power, remainder, overflow_correction)
        }

        // R-type bitwise: aux0 = AND(rs1, rs2)
        Instruction::AND { .. } | Instruction::OR { .. } | Instruction::XOR { .. } => {
            let and_val = rs1_val & rs2_val;
            (and_val, 0, 0)
        }

        // I-type ALU
        Instruction::ADDI { imm, .. } => {
            let imm_val = *imm as i64 as u64;
            let carry = ((rs1_val as u128 + imm_val as u128) >> 64) as u64;
            (carry, 0, 0)
        }
        Instruction::SLTIU { imm, .. } => {
            // aux0 = unsigned_lt (binary), aux1=0, aux2=0
            let imm_val = *imm as i64 as u64;
            let unsigned_lt = if rs1_val < imm_val { 1u64 } else { 0u64 };
            (unsigned_lt, 0, 0)
        }
        Instruction::SLTI { imm, .. } => {
            // aux0 = unsigned_lt, aux1 = sign(rs1), aux2 = sign(imm)
            let imm_val = *imm as i64 as u64;
            let unsigned_lt = if rs1_val < imm_val { 1u64 } else { 0u64 };
            let sign_rs1 = (rs1_val >> 63) & 1;
            let sign_imm = (imm_val >> 63) & 1;
            (unsigned_lt, sign_rs1, sign_imm)
        }

        // I-type shifts
        Instruction::SLLI { imm, .. } => {
            let k = (*imm as u64) & 0x3F;
            let power = 1u64 << k;
            let overflow = ((rs1_val as u128 * power as u128) >> 64) as u64;
            (power, overflow, 0)
        }
        Instruction::SRLI { imm, .. } => {
            let k = (*imm as u64) & 0x3F;
            let power = 1u64 << k;
            let remainder = if k == 0 { 0 } else { rs1_val % power };
            (power, remainder, 0)
        }
        Instruction::SRAI { imm, .. } => {
            let k = (*imm as u64) & 0x3F;
            let power = 1u64 << k;
            let remainder = if k == 0 { 0 } else { rs1_val % power };
            let sign_bit = (rs1_val >> 63) & 1;
            let overflow_correction = sign_bit * (power.wrapping_sub(1));
            (power, remainder, overflow_correction)
        }

        // I-type bitwise: aux0 = AND(rs1, immediate)
        Instruction::ANDI { imm, .. } | Instruction::ORI { imm, .. } | Instruction::XORI { imm, .. } => {
            let imm_val = *imm as i64 as u64;
            let and_val = rs1_val & imm_val;
            (and_val, 0, 0)
        }

        // W-type ALU
        Instruction::ADDW { .. } => {
            let full = rs1_val.wrapping_add(rs2_val);
            let upper = full >> 32;
            let lower32 = full as u32;
            let sign_bit = (lower32 >> 31) & 1;
            (upper, sign_bit as u64, 0)
        }
        Instruction::SUBW { .. } => {
            let full = rs1_val.wrapping_sub(rs2_val);
            let upper = full >> 32;
            let lower32 = full as u32;
            let sign_bit = (lower32 >> 31) & 1;
            (upper, sign_bit as u64, 0)
        }
        Instruction::ADDIW { imm, .. } => {
            let imm_val = *imm as i64 as u64;
            let full = rs1_val.wrapping_add(imm_val);
            let upper = full >> 32;
            let lower32 = full as u32;
            let sign_bit = (lower32 >> 31) & 1;
            (upper, sign_bit as u64, 0)
        }

        // W-type shifts (32-bit, shift amount from lower 5 bits)
        Instruction::SLLW { .. } => {
            let k = rs2_val & 0x1F;
            let power = 1u32 << k;
            let rs1_low = rs1_val as u32;
            let overflow = ((rs1_low as u64 * power as u64) >> 32) as u64;
            (power as u64, overflow, 0)
        }
        Instruction::SRLW { .. } => {
            let k = rs2_val & 0x1F;
            let power = 1u32 << k;
            let rs1_low = rs1_val as u32;
            let remainder = if k == 0 { 0 } else { (rs1_low % power) as u64 };
            (power as u64, remainder, 0)
        }
        Instruction::SRAW { .. } => {
            let k = rs2_val & 0x1F;
            let power = 1u32 << k;
            let rs1_low = rs1_val as u32;
            let remainder = if k == 0 { 0 } else { (rs1_low % power) as u64 };
            let sign_bit_32 = (rs1_low >> 31) & 1;
            let overflow_correction = sign_bit_32 as u64 * (power.wrapping_sub(1)) as u64;
            (power as u64, remainder, overflow_correction)
        }
        Instruction::SLLIW { imm, .. } => {
            let k = (*imm as u32) & 0x1F;
            let power = 1u32 << k;
            let rs1_low = rs1_val as u32;
            let overflow = ((rs1_low as u64 * power as u64) >> 32) as u64;
            (power as u64, overflow, 0)
        }
        Instruction::SRLIW { imm, .. } => {
            let k = (*imm as u32) & 0x1F;
            let power = 1u32 << k;
            let rs1_low = rs1_val as u32;
            let remainder = if k == 0 { 0 } else { (rs1_low % power) as u64 };
            (power as u64, remainder, 0)
        }
        Instruction::SRAIW { imm, .. } => {
            let k = (*imm as u32) & 0x1F;
            let power = 1u32 << k;
            let rs1_low = rs1_val as u32;
            let remainder = if k == 0 { 0 } else { (rs1_low % power) as u64 };
            let sign_bit_32 = (rs1_low >> 31) & 1;
            let overflow_correction = sign_bit_32 as u64 * (power.wrapping_sub(1)) as u64;
            (power as u64, remainder, overflow_correction)
        }

        // MUL/DIV
        Instruction::MUL { .. } => {
            let product = (rs1_val as u128) * (rs2_val as u128);
            let hi = (product >> 64) as u64;
            (hi, 0, 0)
        }
        Instruction::MULH { .. } => {
            // MULH: signed×signed upper 64. Constraint uses unsigned product + sign correction.
            // aux0 = low 64 bits of unsigned product, aux1 = sign(rs1), aux2 = sign(rs2)
            let product = (rs1_val as u128) * (rs2_val as u128);
            let lo = product as u64;
            let sign_a = (rs1_val >> 63) & 1;
            let sign_b = (rs2_val >> 63) & 1;
            (lo, sign_a, sign_b)
        }
        Instruction::MULHU { .. } => {
            let product = (rs1_val as u128) * (rs2_val as u128);
            let lo = product as u64;
            (lo, 0, 0)
        }
        Instruction::MULHSU { .. } => {
            // MULHSU: signed×unsigned upper 64. Only rs1 is signed.
            // aux0 = low 64 bits of unsigned product, aux1 = sign(rs1), aux2 = 0
            let product = (rs1_val as u128) * (rs2_val as u128);
            let lo = product as u64;
            let sign_a = (rs1_val >> 63) & 1;
            (lo, sign_a, 0)
        }
        Instruction::DIV { .. } => {
            if rs2_val == 0 {
                (0, 0, 0)
            } else {
                let remainder = (rs1_val as i64).wrapping_rem(rs2_val as i64) as u64;
                (remainder, 0, 0)
            }
        }
        Instruction::DIVU { .. } => {
            if rs2_val == 0 {
                (0, 0, 0)
            } else {
                let remainder = rs1_val % rs2_val;
                (remainder, 0, 0)
            }
        }
        Instruction::REM { .. } => {
            if rs2_val == 0 {
                (0, 0, 0)
            } else {
                let quotient = (rs1_val as i64).wrapping_div(rs2_val as i64) as u64;
                (quotient, 0, 0)
            }
        }
        Instruction::REMU { .. } => {
            if rs2_val == 0 {
                (0, 0, 0)
            } else {
                let quotient = rs1_val / rs2_val;
                (quotient, 0, 0)
            }
        }
        Instruction::MULW { .. } => {
            // MULW: aux0 = upper bits quotient, aux1 = sign bit of lower 32
            let product = (rs1_val as u32 as u64) * (rs2_val as u32 as u64);
            let upper = product >> 32;
            let sign = (product >> 31) & 1;
            (upper, sign, 0)
        }
        Instruction::DIVW { .. } => {
            // DIVW: rd * rs2_low + aux0 = rs1_low (mod 2^32)
            // aux0 = remainder, aux1 = correction = (rd*rs2 + aux0 - rs1) / 2^32
            let a = rs1_val as i32;
            let b = rs2_val as i32;
            if b == 0 {
                // DIV by zero: rd = -1 = 0xFFFF...FFFF, remainder = a
                let rd = rd_val_after;
                let rem = a as u64;
                let expr = rd.wrapping_mul(rs2_val).wrapping_add(rem).wrapping_sub(rs1_val);
                let correction = (expr as i64 >> 32) as u64;
                (rem, correction, 0)
            } else {
                let rem = a.wrapping_rem(b) as u64;
                let expr = rd_val_after.wrapping_mul(rs2_val).wrapping_add(rem).wrapping_sub(rs1_val);
                let correction = (expr as i64 >> 32) as u64;
                (rem, correction, 0)
            }
        }
        Instruction::DIVUW { .. } => {
            let a = rs1_val as u32;
            let b = rs2_val as u32;
            if b == 0 {
                let rd = rd_val_after;
                let rem = a as u64;
                let expr = rd.wrapping_mul(rs2_val).wrapping_add(rem).wrapping_sub(rs1_val);
                let correction = (expr as i64 >> 32) as u64;
                (rem, correction, 0)
            } else {
                let rem = (a % b) as u64;
                let expr = rd_val_after.wrapping_mul(rs2_val).wrapping_add(rem).wrapping_sub(rs1_val);
                let correction = (expr as i64 >> 32) as u64;
                (rem, correction, 0)
            }
        }
        Instruction::REMW { .. } => {
            // REMW: aux0 * rs2 + rd = rs1 (mod 2^32)
            // aux0 = quotient, aux1 = correction = (aux0*rs2 + rd - rs1) / 2^32
            let a = rs1_val as i32;
            let b = rs2_val as i32;
            if b == 0 {
                // REM by zero: rd = a
                (0, 0, 0)
            } else {
                let quot = a.wrapping_div(b) as u64;
                let expr = quot.wrapping_mul(rs2_val).wrapping_add(rd_val_after).wrapping_sub(rs1_val);
                let correction = (expr as i64 >> 32) as u64;
                (quot, correction, 0)
            }
        }
        Instruction::REMUW { .. } => {
            let a = rs1_val as u32;
            let b = rs2_val as u32;
            if b == 0 {
                (0, 0, 0)
            } else {
                let quot = (a / b) as u64;
                let expr = quot.wrapping_mul(rs2_val).wrapping_add(rd_val_after).wrapping_sub(rs1_val);
                let correction = (expr as i64 >> 32) as u64;
                (quot, correction, 0)
            }
        }

        // Stores: truncation quotient
        Instruction::SB { .. } => (rs2_val >> 8, 0, 0),
        Instruction::SH { .. } => (rs2_val >> 16, 0, 0),
        Instruction::SW { .. } => (rs2_val >> 32, 0, 0),

        // Branches: aux1 = condition witness
        Instruction::BEQ { .. } | Instruction::BNE { .. } => {
            if rs1_val != rs2_val {
                (0, 1, 0)
            } else {
                (0, 0, 0)
            }
        }
        Instruction::BLTU { .. } | Instruction::BGEU { .. } => {
            let borrow = if rs1_val < rs2_val { 1u64 } else { 0u64 };
            (0, borrow, 0)
        }
        Instruction::BLT { .. } | Instruction::BGE { .. } => {
            let borrow = if rs1_val < rs2_val { 1u64 } else { 0u64 };
            let sign_a = (rs1_val >> 63) & 1;
            let sign_b = (rs2_val >> 63) & 1;
            (borrow, sign_a, sign_b)
        }

        // JALR: aux1 = LSB of (rs1_val + immediate), the bit cleared by &~1
        Instruction::JALR { imm, .. } => {
            let target = rs1_val.wrapping_add(*imm as i64 as u64);
            let lsb = target & 1;
            (0, lsb, 0)
        }

        // CSR instructions
        Instruction::CSRRW { .. } => {
            // aux0 = new CSR value = rs1_val
            (rs1_val, 0, 0)
        }
        Instruction::CSRRWI { uimm, .. } => {
            // aux0 = new CSR value = uimm (zero-extended)
            (*uimm as u64, 0, 0)
        }
        Instruction::CSRRS { .. } => {
            // rd_val_after = old CSR value, source = rs1_val
            // aux0 = old | source, aux1 = AND(old, source)
            let old = rd_val_after;
            let and_val = old & rs1_val;
            let or_val = old | rs1_val;
            (or_val, and_val, 0)
        }
        Instruction::CSRRSI { uimm, .. } => {
            let old = rd_val_after;
            let source = *uimm as u64;
            let and_val = old & source;
            let or_val = old | source;
            (or_val, and_val, 0)
        }
        Instruction::CSRRC { .. } => {
            // rd_val_after = old CSR value, source = rs1_val
            // aux0 = old & ~source, aux1 = AND(old, source)
            let old = rd_val_after;
            let and_val = old & rs1_val;
            let clear_val = old & !rs1_val;
            (clear_val, and_val, 0)
        }
        Instruction::CSRRCI { uimm, .. } => {
            let old = rd_val_after;
            let source = *uimm as u64;
            let and_val = old & source;
            let clear_val = old & !source;
            (clear_val, and_val, 0)
        }

        // SC: aux0 = 1 if reservation valid (rd=0), 0 if invalid (rd!=0)
        Instruction::SC_W { .. } | Instruction::SC_D { .. } => {
            let valid = if rd_val_after == 0 { 1u64 } else { 0u64 };
            (valid, 0, 0)
        }

        // AMO operations: rd_val_after = old memory value, rs2_val = operand
        // aux values provide witnesses for the constraint
        Instruction::AMOSWAP_D { .. } => (0, 0, 0), // D: mem_val = rs2_val, no aux needed
        Instruction::AMOSWAP_W { .. } => {
            // W: mem_val = sign_extend_32(rs2_val[31:0])
            // Constraint: rs2_val - aux0*2^32 - mem_val + aux1*(2^64-2^32) = 0
            let upper = rs2_val >> 32;
            let sign = (rs2_val >> 31) & 1;
            (upper, sign, 0)
        }
        Instruction::AMOADD_D { .. } => {
            // D: mem_val = old + rs2, carry = (old + rs2) >> 64
            let carry = if (rd_val_after as u128 + rs2_val as u128) >= (1u128 << 64) { 1u64 } else { 0u64 };
            (carry, 0, 0) // aux0=carry, aux2=0 (D-flag)
        }
        Instruction::AMOADD_W { .. } => {
            // W: mem_val = sign_extend_32((old_low + rs2_low)[31:0])
            let sum32 = (rd_val_after as u32).wrapping_add(rs2_val as u32);
            let result = sum32 as i32 as i64 as u64;
            // Constraint: old + rs2 - Q*2^32 - mem_val + sign*(2^64-2^32) = 0
            let full = rd_val_after.wrapping_add(rs2_val);
            let q = full.wrapping_sub(result) >> 32;
            let sign = (sum32 >> 31) & 1;
            (q, sign as u64, 1) // aux2=1 (W-flag)
        }
        Instruction::AMOAND_D { .. } => {
            // AND: mem_val = old & rs2. aux0 = OR(old, rs2) = old + rs2 - AND
            let or_val = rd_val_after | rs2_val;
            (or_val, 0, 0)
        }
        Instruction::AMOAND_W { .. } => {
            let and32 = (rd_val_after as u32) & (rs2_val as u32);
            let result = and32 as i32 as i64 as u64;
            let or_val = (rd_val_after as u32 | rs2_val as u32) as u64;
            // Correction: (mem_val + or_val - old - rs2) / 2^32
            let expr = result.wrapping_add(or_val).wrapping_sub(rd_val_after).wrapping_sub(rs2_val);
            let correction = (expr as i64 >> 32) as u64;
            (or_val, correction, 0)
        }
        Instruction::AMOOR_D { .. } => {
            // OR: mem_val = old | rs2. aux0 = AND(old, rs2)
            let and_val = rd_val_after & rs2_val;
            (and_val, 0, 0)
        }
        Instruction::AMOOR_W { .. } => {
            let or32 = (rd_val_after as u32) | (rs2_val as u32);
            let result = or32 as i32 as i64 as u64;
            let and_val = (rd_val_after as u32 & rs2_val as u32) as u64;
            let expr = result.wrapping_add(and_val).wrapping_sub(rd_val_after).wrapping_sub(rs2_val);
            let correction = (expr as i64 >> 32) as u64;
            (and_val, correction, 0)
        }
        Instruction::AMOXOR_D { .. } => {
            // XOR: mem_val = old ^ rs2. aux0 = AND(old, rs2)
            let and_val = rd_val_after & rs2_val;
            (and_val, 0, 0)
        }
        Instruction::AMOXOR_W { .. } => {
            let xor32 = (rd_val_after as u32) ^ (rs2_val as u32);
            let result = xor32 as i32 as i64 as u64;
            let and_val = (rd_val_after as u32 & rs2_val as u32) as u64;
            // XOR = a + b - 2*AND. Correction absorbs upper-bit differences.
            let expr = result.wrapping_add(and_val.wrapping_mul(2)).wrapping_sub(rd_val_after).wrapping_sub(rs2_val);
            let correction = (expr as i64 >> 32) as u64;
            (and_val, correction, 0)
        }
        Instruction::AMOMINU_D { .. } => {
            // MINU: mem_val = min(old, rs2). aux0 = borrow (old < rs2)
            let borrow = if rd_val_after < rs2_val { 1u64 } else { 0u64 };
            (borrow, 0, 0)
        }
        Instruction::AMOMINU_W { .. } => {
            let borrow = if (rd_val_after as u32) < (rs2_val as u32) { 1u64 } else { 0u64 };
            let result = if borrow == 1 { rd_val_after as u32 } else { rs2_val as u32 };
            let result_ext = result as i32 as i64 as u64;
            // Correction for W sign extension
            let expr = result_ext.wrapping_sub(rs2_val).wrapping_sub(
                borrow.wrapping_mul(rd_val_after.wrapping_sub(rs2_val)));
            let correction = (expr as i64 >> 32) as u64;
            (borrow, correction, 0)
        }
        Instruction::AMOMAXU_D { .. } => {
            let borrow = if rd_val_after < rs2_val { 1u64 } else { 0u64 };
            (borrow, 0, 0)
        }
        Instruction::AMOMAXU_W { .. } => {
            let borrow = if (rd_val_after as u32) < (rs2_val as u32) { 1u64 } else { 0u64 };
            let result = if borrow == 1 { rs2_val as u32 } else { rd_val_after as u32 };
            let result_ext = result as i32 as i64 as u64;
            let expr = result_ext.wrapping_sub(rd_val_after).wrapping_add(
                borrow.wrapping_mul(rd_val_after.wrapping_sub(rs2_val)));
            let correction = (expr as i64 >> 32) as u64;
            (borrow, correction, 0)
        }
        Instruction::AMOMIN_D { .. } => {
            // Signed min: aux0=unsigned_borrow, aux1=sign(old), aux2=sign(rs2)
            let borrow = if rd_val_after < rs2_val { 1u64 } else { 0u64 };
            let sign_a = (rd_val_after >> 63) & 1;
            let sign_b = (rs2_val >> 63) & 1;
            (borrow, sign_a, sign_b)
        }
        Instruction::AMOMIN_W { .. } => {
            let borrow = if (rd_val_after as u32) < (rs2_val as u32) { 1u64 } else { 0u64 };
            let sign_a = ((rd_val_after as u32) >> 31) & 1;
            let sign_b = ((rs2_val as u32) >> 31) & 1;
            (borrow, sign_a as u64, sign_b as u64)
        }
        Instruction::AMOMAX_D { .. } => {
            let borrow = if rd_val_after < rs2_val { 1u64 } else { 0u64 };
            let sign_a = (rd_val_after >> 63) & 1;
            let sign_b = (rs2_val >> 63) & 1;
            (borrow, sign_a, sign_b)
        }
        Instruction::AMOMAX_W { .. } => {
            let borrow = if (rd_val_after as u32) < (rs2_val as u32) { 1u64 } else { 0u64 };
            let sign_a = ((rd_val_after as u32) >> 31) & 1;
            let sign_b = ((rs2_val as u32) >> 31) & 1;
            (borrow, sign_a as u64, sign_b as u64)
        }

        // All others: no auxiliary
        _ => (0, 0, 0),
    }
}

/// Export a slice of TraceRows to TraceColumns (shared by TracingVm and TracingVmBorrowed).
pub fn export_trace_rows_to_columns(trace: &[TraceRow]) -> TraceColumns {
    let n = trace.len();
    let mut cols = TraceColumns {
        step: Vec::with_capacity(n),
        pc: Vec::with_capacity(n),
        rd: Vec::with_capacity(n),
        rd_val_before: Vec::with_capacity(n),
        rd_val_after: Vec::with_capacity(n),
        rs1: Vec::with_capacity(n),
        rs1_val: Vec::with_capacity(n),
        rs2: Vec::with_capacity(n),
        rs2_val: Vec::with_capacity(n),
        mem_addr: Vec::with_capacity(n),
        mem_val: Vec::with_capacity(n),
        next_pc: Vec::with_capacity(n),
        privilege_mode: Vec::with_capacity(n),
        insn_type: Vec::with_capacity(n),
        funct: Vec::with_capacity(n),
        immediate: Vec::with_capacity(n),
        insn_len: Vec::with_capacity(n),
        aux0: Vec::with_capacity(n),
        aux1: Vec::with_capacity(n),
        aux2: Vec::with_capacity(n),
        sel_r_alu_add: Vec::with_capacity(n),
        sel_r_alu_sub: Vec::with_capacity(n),
        sel_r_and: Vec::with_capacity(n),
        sel_r_or: Vec::with_capacity(n),
        sel_r_xor: Vec::with_capacity(n),
        sel_r_sll: Vec::with_capacity(n),
        sel_r_srl: Vec::with_capacity(n),
        sel_r_sra: Vec::with_capacity(n),
        sel_r_compare: Vec::with_capacity(n),
        sel_i_alu_add: Vec::with_capacity(n),
        sel_i_and: Vec::with_capacity(n),
        sel_i_or: Vec::with_capacity(n),
        sel_i_xor: Vec::with_capacity(n),
        sel_i_sll: Vec::with_capacity(n),
        sel_i_srl: Vec::with_capacity(n),
        sel_i_sra: Vec::with_capacity(n),
        sel_i_compare: Vec::with_capacity(n),
        sel_w_alu_add: Vec::with_capacity(n),
        sel_w_alu_sub: Vec::with_capacity(n),
        sel_w_alu_addi: Vec::with_capacity(n),
        sel_w_sll: Vec::with_capacity(n),
        sel_w_srl: Vec::with_capacity(n),
        sel_w_sra: Vec::with_capacity(n),
        sel_w_alu_other: Vec::with_capacity(n),
        sel_mul: Vec::with_capacity(n),
        sel_div: Vec::with_capacity(n),
        sel_rem: Vec::with_capacity(n),
        sel_muldiv_other: Vec::with_capacity(n),
        sel_load: Vec::with_capacity(n),
        sel_store: Vec::with_capacity(n),
        sel_beq: Vec::with_capacity(n),
        sel_bne: Vec::with_capacity(n),
        sel_bltu: Vec::with_capacity(n),
        sel_bgeu: Vec::with_capacity(n),
        sel_blt: Vec::with_capacity(n),
        sel_bge: Vec::with_capacity(n),
        sel_jal: Vec::with_capacity(n),
        sel_jalr: Vec::with_capacity(n),
        sel_lui: Vec::with_capacity(n),
        sel_auipc: Vec::with_capacity(n),
        sel_csrrw: Vec::with_capacity(n),
        sel_csrrs: Vec::with_capacity(n),
        sel_csrrc: Vec::with_capacity(n),
        sel_system: Vec::with_capacity(n),
        sel_lr: Vec::with_capacity(n),
        sel_sc: Vec::with_capacity(n),
        sel_amo_swap: Vec::with_capacity(n),
        sel_amo_add: Vec::with_capacity(n),
        sel_amo_bitwise: Vec::with_capacity(n),
        sel_amo_compare: Vec::with_capacity(n),
        sel_atomic_other: Vec::with_capacity(n),
        sel_mulhu: Vec::with_capacity(n),
        sel_csrrw_i: Vec::with_capacity(n),
        sel_csrrs_i: Vec::with_capacity(n),
        sel_mulh: Vec::with_capacity(n),
        sel_mulhsu: Vec::with_capacity(n),
        sel_mulw: Vec::with_capacity(n),
        sel_divw: Vec::with_capacity(n),
        sel_remw: Vec::with_capacity(n),
        sel_amo_or: Vec::with_capacity(n),
        sel_amo_xor: Vec::with_capacity(n),
        sel_amo_maxu: Vec::with_capacity(n),
        sel_amo_mins: Vec::with_capacity(n),
        sel_amo_maxs: Vec::with_capacity(n),
    };

    for row in trace {
        cols.step.push(row.step);
        cols.pc.push(row.pc);
        cols.rd.push(row.rd as u64);
        cols.rd_val_before.push(row.rd_val_before);
        cols.rd_val_after.push(row.rd_val_after);
        cols.rs1.push(row.rs1 as u64);
        cols.rs1_val.push(row.rs1_val);
        cols.rs2.push(row.rs2 as u64);
        cols.rs2_val.push(row.rs2_val);
        cols.mem_addr.push(row.mem_addr);
        cols.mem_val.push(row.mem_val);
        cols.next_pc.push(row.next_pc);
        cols.privilege_mode.push(row.privilege_mode as u64);

        let (itype, funct) = classify_instruction(&row.instruction);
        cols.insn_type.push(itype as u64);
        cols.funct.push(funct as u64);
        cols.immediate.push(extract_immediate(&row.instruction) as u64);
        cols.insn_len.push(row.insn_len as u64);
        cols.aux0.push(row.aux0);
        cols.aux1.push(row.aux1);
        cols.aux2.push(row.aux2);

        // One-hot selector columns from (insn_type, funct)
        let mut sel = [0u64; 64];
        match itype {
            0 => match funct { // R-type ALU
                0 => sel[0] = 1,  // ADD
                1 => sel[1] = 1,  // SUB
                9 => sel[2] = 1,  // AND
                8 => sel[3] = 1,  // OR
                5 => sel[4] = 1,  // XOR
                2 => sel[5] = 1,  // SLL
                6 => sel[6] = 1,  // SRL
                7 => sel[7] = 1,  // SRA
                3 | 4 => sel[8] = 1, // SLT/SLTU
                _ => sel[8] = 1,  // unknown R-type
            },
            1 => match funct { // I-type ALU
                0 => sel[9] = 1,  // ADDI
                5 => sel[10] = 1, // ANDI
                4 => sel[11] = 1, // ORI
                3 => sel[12] = 1, // XORI
                6 => sel[13] = 1, // SLLI
                7 => sel[14] = 1, // SRLI
                8 => sel[15] = 1, // SRAI
                1 | 2 => sel[16] = 1, // SLTI/SLTIU
                _ => sel[16] = 1, // unknown I-type
            },
            2 => match funct { // W-type ALU
                0 => sel[17] = 1,  // ADDW
                1 => sel[18] = 1,  // SUBW
                5 => sel[19] = 1,  // ADDIW
                2 => sel[20] = 1,  // SLLW
                3 => sel[21] = 1,  // SRLW
                4 => sel[22] = 1,  // SRAW
                6 => sel[20] = 1,  // SLLIW (reuse SLLW selector)
                7 => sel[21] = 1,  // SRLIW (reuse SRLW selector)
                8 => sel[22] = 1,  // SRAIW (reuse SRAW selector)
                _ => sel[23] = 1,  // W-type other
            },
            3 => match funct { // MULDIV
                0 => sel[24] = 1,      // MUL
                1 => sel[54] = 1,      // MULH
                2 => sel[55] = 1,      // MULHSU
                3 => sel[51] = 1,      // MULHU
                4 | 5 => sel[25] = 1,  // DIV/DIVU
                6 | 7 => sel[26] = 1,  // REM/REMU
                8 => sel[56] = 1,      // MULW
                9 | 10 => sel[57] = 1, // DIVW/DIVUW
                11 | 12 => sel[58] = 1,// REMW/REMUW
                _ => sel[27] = 1,      // muldiv_other (dead)
            },
            4 => sel[28] = 1,  // LOAD
            5 => sel[29] = 1,  // STORE
            6 => match funct { // BRANCH
                0 => sel[30] = 1, // BEQ
                1 => sel[31] = 1, // BNE
                4 => sel[32] = 1, // BLTU
                5 => sel[33] = 1, // BGEU
                2 => sel[34] = 1, // BLT
                3 => sel[35] = 1, // BGE
                _ => sel[34] = 1, // unknown branch -> BLT
            },
            7 => sel[36] = 1,  // JAL
            8 => sel[37] = 1,  // JALR
            9 => sel[38] = 1,  // LUI
            10 => sel[39] = 1, // AUIPC
            11 => match funct { // CSR
                0 => sel[40] = 1,     // CSRRW (R-variant)
                3 => sel[52] = 1,     // CSRRWI (I-variant)
                1 => sel[41] = 1,     // CSRRS (R-variant)
                4 => sel[53] = 1,     // CSRRSI (I-variant)
                2 | 5 => sel[42] = 1, // CSRRC / CSRRCI
                _ => sel[40] = 1,     // unknown CSR -> CSRRW
            },
            12 => sel[43] = 1, // SYSTEM
            13 => match funct { // ATOMIC
                0 | 1 => sel[44] = 1,       // LR.W / LR.D
                2 | 3 => sel[45] = 1,       // SC.W / SC.D
                4 | 5 => sel[46] = 1,       // AMOSWAP.W / AMOSWAP.D
                6 | 7 => sel[47] = 1,       // AMOADD.W / AMOADD.D
                8 | 9 => sel[48] = 1,       // AMOAND.W / AMOAND.D
                10 | 11 => sel[59] = 1,     // AMOOR.W / AMOOR.D
                12 | 13 => sel[60] = 1,     // AMOXOR.W / AMOXOR.D
                14 | 15 => sel[63] = 1,     // AMOMAX.W / AMOMAX.D (signed)
                16 | 17 => sel[62] = 1,     // AMOMIN.W / AMOMIN.D (signed)
                18 | 19 => sel[61] = 1,     // AMOMAXU.W / AMOMAXU.D (unsigned)
                20 | 21 => sel[49] = 1,     // AMOMINU.W / AMOMINU.D (unsigned)
                _ => sel[50] = 1,            // atomic other (dead)
            },
            _ => sel[43] = 1,  // unknown -> SYSTEM
        }
        cols.sel_r_alu_add.push(sel[0]);
        cols.sel_r_alu_sub.push(sel[1]);
        cols.sel_r_and.push(sel[2]);
        cols.sel_r_or.push(sel[3]);
        cols.sel_r_xor.push(sel[4]);
        cols.sel_r_sll.push(sel[5]);
        cols.sel_r_srl.push(sel[6]);
        cols.sel_r_sra.push(sel[7]);
        cols.sel_r_compare.push(sel[8]);
        cols.sel_i_alu_add.push(sel[9]);
        cols.sel_i_and.push(sel[10]);
        cols.sel_i_or.push(sel[11]);
        cols.sel_i_xor.push(sel[12]);
        cols.sel_i_sll.push(sel[13]);
        cols.sel_i_srl.push(sel[14]);
        cols.sel_i_sra.push(sel[15]);
        cols.sel_i_compare.push(sel[16]);
        cols.sel_w_alu_add.push(sel[17]);
        cols.sel_w_alu_sub.push(sel[18]);
        cols.sel_w_alu_addi.push(sel[19]);
        cols.sel_w_sll.push(sel[20]);
        cols.sel_w_srl.push(sel[21]);
        cols.sel_w_sra.push(sel[22]);
        cols.sel_w_alu_other.push(sel[23]);
        cols.sel_mul.push(sel[24]);
        cols.sel_div.push(sel[25]);
        cols.sel_rem.push(sel[26]);
        cols.sel_muldiv_other.push(sel[27]);
        cols.sel_load.push(sel[28]);
        cols.sel_store.push(sel[29]);
        cols.sel_beq.push(sel[30]);
        cols.sel_bne.push(sel[31]);
        cols.sel_bltu.push(sel[32]);
        cols.sel_bgeu.push(sel[33]);
        cols.sel_blt.push(sel[34]);
        cols.sel_bge.push(sel[35]);
        cols.sel_jal.push(sel[36]);
        cols.sel_jalr.push(sel[37]);
        cols.sel_lui.push(sel[38]);
        cols.sel_auipc.push(sel[39]);
        cols.sel_csrrw.push(sel[40]);
        cols.sel_csrrs.push(sel[41]);
        cols.sel_csrrc.push(sel[42]);
        cols.sel_system.push(sel[43]);
        cols.sel_lr.push(sel[44]);
        cols.sel_sc.push(sel[45]);
        cols.sel_amo_swap.push(sel[46]);
        cols.sel_amo_add.push(sel[47]);
        cols.sel_amo_bitwise.push(sel[48]);
        cols.sel_amo_compare.push(sel[49]);
        cols.sel_atomic_other.push(sel[50]);
        cols.sel_mulhu.push(sel[51]);
        cols.sel_csrrw_i.push(sel[52]);
        cols.sel_csrrs_i.push(sel[53]);
        cols.sel_mulh.push(sel[54]);
        cols.sel_mulhsu.push(sel[55]);
        cols.sel_mulw.push(sel[56]);
        cols.sel_divw.push(sel[57]);
        cols.sel_remw.push(sel[58]);
        cols.sel_amo_or.push(sel[59]);
        cols.sel_amo_xor.push(sel[60]);
        cols.sel_amo_maxu.push(sel[61]);
        cols.sel_amo_mins.push(sel[62]);
        cols.sel_amo_maxs.push(sel[63]);
    }

    cols
}

/// Determine the memory address and operation type for an instruction.
pub fn compute_mem_info(instruction: &Instruction, vm: &Vm) -> (u64, MemOp) {
    compute_mem_info_with(instruction, |r| vm.cpu.read_reg(r))
}

/// Compute the value written to memory for store/AMO operations.
///
/// For regular stores (SB/SH/SW/SD) and SC, returns rs2_val.
/// For AMO operations, computes the read-modify-write result:
///   - rd_val_after = old memory value (loaded by atomic read)
///   - rs2_val = source operand
///   - Returns the new value to be stored
pub fn compute_mem_val(instruction: &Instruction, rd_val_after: u64, rs2_val: u64) -> u64 {
    match instruction {
        // AMOSWAP: store rs2 (D) or sign-extend lower 32 of rs2 (W)
        Instruction::AMOSWAP_D { .. } => rs2_val,
        Instruction::AMOSWAP_W { .. } => (rs2_val as i32) as i64 as u64,

        // AMOADD: store old + rs2
        Instruction::AMOADD_D { .. } => rd_val_after.wrapping_add(rs2_val),
        Instruction::AMOADD_W { .. } => {
            let sum = (rd_val_after as u32).wrapping_add(rs2_val as u32);
            (sum as i32) as i64 as u64
        }

        // AMOAND: store old & rs2
        Instruction::AMOAND_D { .. } => rd_val_after & rs2_val,
        Instruction::AMOAND_W { .. } => {
            let result = (rd_val_after as u32) & (rs2_val as u32);
            (result as i32) as i64 as u64
        }

        // AMOOR: store old | rs2
        Instruction::AMOOR_D { .. } => rd_val_after | rs2_val,
        Instruction::AMOOR_W { .. } => {
            let result = (rd_val_after as u32) | (rs2_val as u32);
            (result as i32) as i64 as u64
        }

        // AMOXOR: store old ^ rs2
        Instruction::AMOXOR_D { .. } => rd_val_after ^ rs2_val,
        Instruction::AMOXOR_W { .. } => {
            let result = (rd_val_after as u32) ^ (rs2_val as u32);
            (result as i32) as i64 as u64
        }

        // AMOMINU: store min(old, rs2) unsigned
        Instruction::AMOMINU_D { .. } => std::cmp::min(rd_val_after, rs2_val),
        Instruction::AMOMINU_W { .. } => {
            let result = std::cmp::min(rd_val_after as u32, rs2_val as u32);
            (result as i32) as i64 as u64
        }

        // AMOMAXU: store max(old, rs2) unsigned
        Instruction::AMOMAXU_D { .. } => std::cmp::max(rd_val_after, rs2_val),
        Instruction::AMOMAXU_W { .. } => {
            let result = std::cmp::max(rd_val_after as u32, rs2_val as u32);
            (result as i32) as i64 as u64
        }

        // AMOMIN: store min(old, rs2) signed
        Instruction::AMOMIN_D { .. } => {
            std::cmp::min(rd_val_after as i64, rs2_val as i64) as u64
        }
        Instruction::AMOMIN_W { .. } => {
            let result = std::cmp::min(rd_val_after as i32, rs2_val as i32);
            (result as i64) as u64
        }

        // AMOMAX: store max(old, rs2) signed
        Instruction::AMOMAX_D { .. } => {
            std::cmp::max(rd_val_after as i64, rs2_val as i64) as u64
        }
        Instruction::AMOMAX_W { .. } => {
            let result = std::cmp::max(rd_val_after as i32, rs2_val as i32);
            (result as i64) as u64
        }

        // Regular stores, SC, and anything else: rs2_val
        _ => rs2_val,
    }
}

/// Compute memory access info using a register-read function.
/// This allows computing mem_info from saved register state (e.g. after vm.step()).
pub fn compute_mem_info_with(instruction: &Instruction, read_reg: impl Fn(u8) -> u64) -> (u64, MemOp) {
    match instruction {
        Instruction::LB { rs1, imm, .. }
        | Instruction::LH { rs1, imm, .. }
        | Instruction::LW { rs1, imm, .. }
        | Instruction::LD { rs1, imm, .. }
        | Instruction::LBU { rs1, imm, .. }
        | Instruction::LHU { rs1, imm, .. }
        | Instruction::LWU { rs1, imm, .. } => {
            let addr = read_reg(*rs1).wrapping_add(*imm as i64 as u64);
            (addr, MemOp::Read)
        }
        Instruction::SB { rs1, imm, .. }
        | Instruction::SH { rs1, imm, .. }
        | Instruction::SW { rs1, imm, .. }
        | Instruction::SD { rs1, imm, .. } => {
            let addr = read_reg(*rs1).wrapping_add(*imm as i64 as u64);
            (addr, MemOp::Write)
        }
        // LR loads from [rs1]
        Instruction::LR_W { rs1, .. }
        | Instruction::LR_D { rs1, .. } => {
            let addr = read_reg(*rs1);
            (addr, MemOp::Read)
        }
        // SC and AMO operations read-modify-write at [rs1]
        Instruction::SC_W { rs1, .. }
        | Instruction::SC_D { rs1, .. }
        | Instruction::AMOSWAP_W { rs1, .. }
        | Instruction::AMOSWAP_D { rs1, .. }
        | Instruction::AMOADD_W  { rs1, .. }
        | Instruction::AMOADD_D  { rs1, .. }
        | Instruction::AMOAND_W  { rs1, .. }
        | Instruction::AMOAND_D  { rs1, .. }
        | Instruction::AMOOR_W   { rs1, .. }
        | Instruction::AMOOR_D   { rs1, .. }
        | Instruction::AMOXOR_W  { rs1, .. }
        | Instruction::AMOXOR_D  { rs1, .. }
        | Instruction::AMOMAX_W  { rs1, .. }
        | Instruction::AMOMAX_D  { rs1, .. }
        | Instruction::AMOMIN_W  { rs1, .. }
        | Instruction::AMOMIN_D  { rs1, .. }
        | Instruction::AMOMAXU_W { rs1, .. }
        | Instruction::AMOMAXU_D { rs1, .. }
        | Instruction::AMOMINU_W { rs1, .. }
        | Instruction::AMOMINU_D { rs1, .. } => {
            let addr = read_reg(*rs1);
            (addr, MemOp::Write)
        }
        _ => (0, MemOp::None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuState;
    use crate::memory::Memory;

    fn make_tracing_vm(program: &[u32]) -> TracingVm {
        let mut memory = Memory::new();
        for (i, &word) in program.iter().enumerate() {
            memory.store_word((i * 4) as u64, word);
        }
        let vm = Vm::new(CpuState::new(), memory);
        TracingVm::new(vm)
    }

    #[test]
    fn test_trace_addi() {
        // ADDI x1, x0, 42 => 0x02A00093
        let mut tvm = make_tracing_vm(&[0x02A00093, 0x00000073]); // ADDI then ECALL
        tvm.step().unwrap();

        assert_eq!(tvm.trace.len(), 1);
        let row = &tvm.trace[0];
        assert_eq!(row.step, 0);
        assert_eq!(row.pc, 0);
        assert_eq!(row.rd, 1);
        assert_eq!(row.rd_val_before, 0);
        assert_eq!(row.rd_val_after, 42);
        assert_eq!(row.mem_op, MemOp::None);
        assert_eq!(row.next_pc, 4);
    }

    #[test]
    fn test_trace_columns() {
        // Two ADDI instructions
        let mut tvm = make_tracing_vm(&[
            0x02A00093, // ADDI x1, x0, 42
            0x00108113, // ADDI x2, x1, 1
            0x00000073, // ECALL
        ]);
        tvm.run(2).unwrap();

        let cols = tvm.export_columns();
        assert_eq!(cols.step.len(), 2);
        assert_eq!(cols.pc, vec![0, 4]);
        assert_eq!(cols.rd_val_after, vec![42, 43]);
    }
}
