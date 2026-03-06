//! Instruction type selectors for the constraint system.
//!
//! Maps each RISC-V instruction variant to an (insn_type, funct) pair
//! used by the selector-based AIR constraint system.

use crate::isa::Instruction;

// Instruction type constants
pub const INSN_R_ALU: u8 = 0;       // R-type ALU (ADD, SUB, SLL, etc.)
pub const INSN_I_ALU: u8 = 1;       // I-type ALU (ADDI, SLTI, etc.)
pub const INSN_W_ALU: u8 = 2;       // W-type ALU (ADDW, SUBW, etc.)
pub const INSN_MULDIV: u8 = 3;      // MUL/DIV
pub const INSN_LOAD: u8 = 4;        // Loads (LB, LH, LW, LD, etc.)
pub const INSN_STORE: u8 = 5;       // Stores (SB, SH, SW, SD)
pub const INSN_BRANCH: u8 = 6;      // Branches (BEQ, BNE, etc.)
pub const INSN_JAL: u8 = 7;         // JAL
pub const INSN_JALR: u8 = 8;        // JALR
pub const INSN_LUI: u8 = 9;         // LUI
pub const INSN_AUIPC: u8 = 10;      // AUIPC
pub const INSN_CSR: u8 = 11;        // CSR instructions
pub const INSN_SYSTEM: u8 = 12;     // ECALL, EBREAK, MRET, SRET, WFI, FENCE
pub const INSN_ATOMIC: u8 = 13;     // LR, SC, AMO*

// Funct codes for distinguishing within a type.
// For R-type ALU:
pub const FUNCT_ADD: u8 = 0;
pub const FUNCT_SUB: u8 = 1;
pub const FUNCT_SLL: u8 = 2;
pub const FUNCT_SLT: u8 = 3;
pub const FUNCT_SLTU: u8 = 4;
pub const FUNCT_XOR: u8 = 5;
pub const FUNCT_SRL: u8 = 6;
pub const FUNCT_SRA: u8 = 7;
pub const FUNCT_OR: u8 = 8;
pub const FUNCT_AND: u8 = 9;

// For I-type ALU: same naming but for immediate variants
pub const FUNCT_ADDI: u8 = 0;
pub const FUNCT_SLTI: u8 = 1;
pub const FUNCT_SLTIU: u8 = 2;
pub const FUNCT_XORI: u8 = 3;
pub const FUNCT_ORI: u8 = 4;
pub const FUNCT_ANDI: u8 = 5;
pub const FUNCT_SLLI: u8 = 6;
pub const FUNCT_SRLI: u8 = 7;
pub const FUNCT_SRAI: u8 = 8;

// For W-type ALU:
pub const FUNCT_ADDW: u8 = 0;
pub const FUNCT_SUBW: u8 = 1;
pub const FUNCT_SLLW: u8 = 2;
pub const FUNCT_SRLW: u8 = 3;
pub const FUNCT_SRAW: u8 = 4;
pub const FUNCT_ADDIW: u8 = 5;
pub const FUNCT_SLLIW: u8 = 6;
pub const FUNCT_SRLIW: u8 = 7;
pub const FUNCT_SRAIW: u8 = 8;

// For branches:
pub const FUNCT_BEQ: u8 = 0;
pub const FUNCT_BNE: u8 = 1;
pub const FUNCT_BLT: u8 = 2;
pub const FUNCT_BGE: u8 = 3;
pub const FUNCT_BLTU: u8 = 4;
pub const FUNCT_BGEU: u8 = 5;

// For loads:
pub const FUNCT_LB: u8 = 0;
pub const FUNCT_LH: u8 = 1;
pub const FUNCT_LW: u8 = 2;
pub const FUNCT_LD: u8 = 3;
pub const FUNCT_LBU: u8 = 4;
pub const FUNCT_LHU: u8 = 5;
pub const FUNCT_LWU: u8 = 6;

// For stores:
pub const FUNCT_SB: u8 = 0;
pub const FUNCT_SH: u8 = 1;
pub const FUNCT_SW: u8 = 2;
pub const FUNCT_SD: u8 = 3;

// For CSR:
pub const FUNCT_CSRRW: u8 = 0;
pub const FUNCT_CSRRS: u8 = 1;
pub const FUNCT_CSRRC: u8 = 2;
pub const FUNCT_CSRRWI: u8 = 3;
pub const FUNCT_CSRRSI: u8 = 4;
pub const FUNCT_CSRRCI: u8 = 5;

// For atomics:
pub const FUNCT_LR_W: u8 = 0;
pub const FUNCT_LR_D: u8 = 1;
pub const FUNCT_SC_W: u8 = 2;
pub const FUNCT_SC_D: u8 = 3;
pub const FUNCT_AMOSWAP_W: u8 = 4;
pub const FUNCT_AMOSWAP_D: u8 = 5;
pub const FUNCT_AMOADD_W: u8 = 6;
pub const FUNCT_AMOADD_D: u8 = 7;
pub const FUNCT_AMOAND_W: u8 = 8;
pub const FUNCT_AMOAND_D: u8 = 9;
pub const FUNCT_AMOOR_W: u8 = 10;
pub const FUNCT_AMOOR_D: u8 = 11;
pub const FUNCT_AMOXOR_W: u8 = 12;
pub const FUNCT_AMOXOR_D: u8 = 13;
pub const FUNCT_AMOMAX_W: u8 = 14;
pub const FUNCT_AMOMAX_D: u8 = 15;
pub const FUNCT_AMOMIN_W: u8 = 16;
pub const FUNCT_AMOMIN_D: u8 = 17;
pub const FUNCT_AMOMAXU_W: u8 = 18;
pub const FUNCT_AMOMAXU_D: u8 = 19;
pub const FUNCT_AMOMINU_W: u8 = 20;
pub const FUNCT_AMOMINU_D: u8 = 21;

/// Classify an instruction into (insn_type, funct) pair.
pub fn classify(insn: &Instruction) -> (u8, u8) {
    match insn {
        // R-type ALU
        Instruction::ADD  { .. } => (INSN_R_ALU, FUNCT_ADD),
        Instruction::SUB  { .. } => (INSN_R_ALU, FUNCT_SUB),
        Instruction::SLL  { .. } => (INSN_R_ALU, FUNCT_SLL),
        Instruction::SLT  { .. } => (INSN_R_ALU, FUNCT_SLT),
        Instruction::SLTU { .. } => (INSN_R_ALU, FUNCT_SLTU),
        Instruction::XOR  { .. } => (INSN_R_ALU, FUNCT_XOR),
        Instruction::SRL  { .. } => (INSN_R_ALU, FUNCT_SRL),
        Instruction::SRA  { .. } => (INSN_R_ALU, FUNCT_SRA),
        Instruction::OR   { .. } => (INSN_R_ALU, FUNCT_OR),
        Instruction::AND  { .. } => (INSN_R_ALU, FUNCT_AND),

        // I-type ALU
        Instruction::ADDI  { .. } => (INSN_I_ALU, FUNCT_ADDI),
        Instruction::SLTI  { .. } => (INSN_I_ALU, FUNCT_SLTI),
        Instruction::SLTIU { .. } => (INSN_I_ALU, FUNCT_SLTIU),
        Instruction::XORI  { .. } => (INSN_I_ALU, FUNCT_XORI),
        Instruction::ORI   { .. } => (INSN_I_ALU, FUNCT_ORI),
        Instruction::ANDI  { .. } => (INSN_I_ALU, FUNCT_ANDI),
        Instruction::SLLI  { .. } => (INSN_I_ALU, FUNCT_SLLI),
        Instruction::SRLI  { .. } => (INSN_I_ALU, FUNCT_SRLI),
        Instruction::SRAI  { .. } => (INSN_I_ALU, FUNCT_SRAI),

        // W-type ALU
        Instruction::ADDW  { .. } => (INSN_W_ALU, FUNCT_ADDW),
        Instruction::SUBW  { .. } => (INSN_W_ALU, FUNCT_SUBW),
        Instruction::SLLW  { .. } => (INSN_W_ALU, FUNCT_SLLW),
        Instruction::SRLW  { .. } => (INSN_W_ALU, FUNCT_SRLW),
        Instruction::SRAW  { .. } => (INSN_W_ALU, FUNCT_SRAW),
        Instruction::ADDIW { .. } => (INSN_W_ALU, FUNCT_ADDIW),
        Instruction::SLLIW { .. } => (INSN_W_ALU, FUNCT_SLLIW),
        Instruction::SRLIW { .. } => (INSN_W_ALU, FUNCT_SRLIW),
        Instruction::SRAIW { .. } => (INSN_W_ALU, FUNCT_SRAIW),

        // MUL/DIV
        Instruction::MUL    { .. } => (INSN_MULDIV, 0),
        Instruction::MULH   { .. } => (INSN_MULDIV, 1),
        Instruction::MULHSU { .. } => (INSN_MULDIV, 2),
        Instruction::MULHU  { .. } => (INSN_MULDIV, 3),
        Instruction::DIV    { .. } => (INSN_MULDIV, 4),
        Instruction::DIVU   { .. } => (INSN_MULDIV, 5),
        Instruction::REM    { .. } => (INSN_MULDIV, 6),
        Instruction::REMU   { .. } => (INSN_MULDIV, 7),
        Instruction::MULW   { .. } => (INSN_MULDIV, 8),
        Instruction::DIVW   { .. } => (INSN_MULDIV, 9),
        Instruction::DIVUW  { .. } => (INSN_MULDIV, 10),
        Instruction::REMW   { .. } => (INSN_MULDIV, 11),
        Instruction::REMUW  { .. } => (INSN_MULDIV, 12),

        // Loads
        Instruction::LB  { .. } => (INSN_LOAD, FUNCT_LB),
        Instruction::LH  { .. } => (INSN_LOAD, FUNCT_LH),
        Instruction::LW  { .. } => (INSN_LOAD, FUNCT_LW),
        Instruction::LD  { .. } => (INSN_LOAD, FUNCT_LD),
        Instruction::LBU { .. } => (INSN_LOAD, FUNCT_LBU),
        Instruction::LHU { .. } => (INSN_LOAD, FUNCT_LHU),
        Instruction::LWU { .. } => (INSN_LOAD, FUNCT_LWU),

        // Stores
        Instruction::SB { .. } => (INSN_STORE, FUNCT_SB),
        Instruction::SH { .. } => (INSN_STORE, FUNCT_SH),
        Instruction::SW { .. } => (INSN_STORE, FUNCT_SW),
        Instruction::SD { .. } => (INSN_STORE, FUNCT_SD),

        // Branches
        Instruction::BEQ  { .. } => (INSN_BRANCH, FUNCT_BEQ),
        Instruction::BNE  { .. } => (INSN_BRANCH, FUNCT_BNE),
        Instruction::BLT  { .. } => (INSN_BRANCH, FUNCT_BLT),
        Instruction::BGE  { .. } => (INSN_BRANCH, FUNCT_BGE),
        Instruction::BLTU { .. } => (INSN_BRANCH, FUNCT_BLTU),
        Instruction::BGEU { .. } => (INSN_BRANCH, FUNCT_BGEU),

        // Control flow
        Instruction::JAL  { .. } => (INSN_JAL, 0),
        Instruction::JALR { .. } => (INSN_JALR, 0),
        Instruction::LUI  { .. } => (INSN_LUI, 0),
        Instruction::AUIPC { .. } => (INSN_AUIPC, 0),

        // CSR
        Instruction::CSRRW  { .. } => (INSN_CSR, 0),
        Instruction::CSRRS  { .. } => (INSN_CSR, 1),
        Instruction::CSRRC  { .. } => (INSN_CSR, 2),
        Instruction::CSRRWI { .. } => (INSN_CSR, 3),
        Instruction::CSRRSI { .. } => (INSN_CSR, 4),
        Instruction::CSRRCI { .. } => (INSN_CSR, 5),

        // System
        Instruction::ECALL     => (INSN_SYSTEM, 0),
        Instruction::EBREAK    => (INSN_SYSTEM, 1),
        Instruction::MRET      => (INSN_SYSTEM, 2),
        Instruction::SRET      => (INSN_SYSTEM, 3),
        Instruction::WFI       => (INSN_SYSTEM, 4),
        Instruction::FENCE { .. } => (INSN_SYSTEM, 5),
        Instruction::SFENCE_VMA { .. } => (INSN_SYSTEM, 6),
        Instruction::FENCE_I   => (INSN_SYSTEM, 7),

        // Atomics
        Instruction::LR_W      { .. } => (INSN_ATOMIC, 0),
        Instruction::LR_D      { .. } => (INSN_ATOMIC, 1),
        Instruction::SC_W      { .. } => (INSN_ATOMIC, 2),
        Instruction::SC_D      { .. } => (INSN_ATOMIC, 3),
        Instruction::AMOSWAP_W { .. } => (INSN_ATOMIC, 4),
        Instruction::AMOSWAP_D { .. } => (INSN_ATOMIC, 5),
        Instruction::AMOADD_W  { .. } => (INSN_ATOMIC, 6),
        Instruction::AMOADD_D  { .. } => (INSN_ATOMIC, 7),
        Instruction::AMOAND_W  { .. } => (INSN_ATOMIC, 8),
        Instruction::AMOAND_D  { .. } => (INSN_ATOMIC, 9),
        Instruction::AMOOR_W   { .. } => (INSN_ATOMIC, 10),
        Instruction::AMOOR_D   { .. } => (INSN_ATOMIC, 11),
        Instruction::AMOXOR_W  { .. } => (INSN_ATOMIC, 12),
        Instruction::AMOXOR_D  { .. } => (INSN_ATOMIC, 13),
        Instruction::AMOMAX_W  { .. } => (INSN_ATOMIC, 14),
        Instruction::AMOMAX_D  { .. } => (INSN_ATOMIC, 15),
        Instruction::AMOMIN_W  { .. } => (INSN_ATOMIC, 16),
        Instruction::AMOMIN_D  { .. } => (INSN_ATOMIC, 17),
        Instruction::AMOMAXU_W { .. } => (INSN_ATOMIC, 18),
        Instruction::AMOMAXU_D { .. } => (INSN_ATOMIC, 19),
        Instruction::AMOMINU_W { .. } => (INSN_ATOMIC, 20),
        Instruction::AMOMINU_D { .. } => (INSN_ATOMIC, 21),
    }
}

/// Extract the immediate value from an instruction (sign-extended to i64).
pub fn extract_immediate(insn: &Instruction) -> i64 {
    match insn {
        // U-type: immediate is upper 20 bits << 12
        Instruction::LUI { imm, .. } | Instruction::AUIPC { imm, .. } => *imm as i64,

        // J-type
        Instruction::JAL { imm, .. } => *imm as i64,

        // I-type with immediate
        Instruction::JALR { imm, .. }
        | Instruction::LB { imm, .. }
        | Instruction::LH { imm, .. }
        | Instruction::LW { imm, .. }
        | Instruction::LD { imm, .. }
        | Instruction::LBU { imm, .. }
        | Instruction::LHU { imm, .. }
        | Instruction::LWU { imm, .. }
        | Instruction::ADDI { imm, .. }
        | Instruction::SLTI { imm, .. }
        | Instruction::SLTIU { imm, .. }
        | Instruction::XORI { imm, .. }
        | Instruction::ORI { imm, .. }
        | Instruction::ANDI { imm, .. }
        | Instruction::SLLI { imm, .. }
        | Instruction::SRLI { imm, .. }
        | Instruction::SRAI { imm, .. }
        | Instruction::ADDIW { imm, .. }
        | Instruction::SLLIW { imm, .. }
        | Instruction::SRLIW { imm, .. }
        | Instruction::SRAIW { imm, .. } => *imm as i64,

        // S-type (stores)
        Instruction::SB { imm, .. }
        | Instruction::SH { imm, .. }
        | Instruction::SW { imm, .. }
        | Instruction::SD { imm, .. } => *imm as i64,

        // B-type (branches)
        Instruction::BEQ { imm, .. }
        | Instruction::BNE { imm, .. }
        | Instruction::BLT { imm, .. }
        | Instruction::BGE { imm, .. }
        | Instruction::BLTU { imm, .. }
        | Instruction::BGEU { imm, .. } => *imm as i64,

        // CSR immediate variants
        Instruction::CSRRWI { uimm, .. }
        | Instruction::CSRRSI { uimm, .. }
        | Instruction::CSRRCI { uimm, .. } => *uimm as i64,

        // All other instructions have no immediate
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_add() {
        let insn = Instruction::ADD { rd: 1, rs1: 2, rs2: 3 };
        assert_eq!(classify(&insn), (INSN_R_ALU, FUNCT_ADD));
    }

    #[test]
    fn test_classify_addi() {
        let insn = Instruction::ADDI { rd: 1, rs1: 2, imm: 42 };
        assert_eq!(classify(&insn), (INSN_I_ALU, FUNCT_ADDI));
    }

    #[test]
    fn test_classify_branch() {
        let insn = Instruction::BEQ { rs1: 1, rs2: 2, imm: 8 };
        assert_eq!(classify(&insn), (INSN_BRANCH, FUNCT_BEQ));
    }

    #[test]
    fn test_classify_lui() {
        let insn = Instruction::LUI { rd: 1, imm: 0x12345 };
        assert_eq!(classify(&insn), (INSN_LUI, 0));
    }

    #[test]
    fn test_classify_load() {
        let insn = Instruction::LD { rd: 1, rs1: 2, imm: 0 };
        assert_eq!(classify(&insn), (INSN_LOAD, FUNCT_LD));
    }

    #[test]
    fn test_classify_store() {
        let insn = Instruction::SD { rs1: 1, rs2: 2, imm: 0 };
        assert_eq!(classify(&insn), (INSN_STORE, FUNCT_SD));
    }

    #[test]
    fn test_classify_ecall() {
        assert_eq!(classify(&Instruction::ECALL), (INSN_SYSTEM, 0));
    }

    #[test]
    fn test_extract_immediate_addi() {
        let insn = Instruction::ADDI { rd: 1, rs1: 0, imm: -5 };
        assert_eq!(extract_immediate(&insn), -5);
    }

    #[test]
    fn test_extract_immediate_lui() {
        let insn = Instruction::LUI { rd: 1, imm: 0x12345000u32 as i32 };
        assert_eq!(extract_immediate(&insn), 0x12345000u32 as i32 as i64);
    }

    #[test]
    fn test_extract_immediate_add_is_zero() {
        let insn = Instruction::ADD { rd: 1, rs1: 2, rs2: 3 };
        assert_eq!(extract_immediate(&insn), 0);
    }
}
