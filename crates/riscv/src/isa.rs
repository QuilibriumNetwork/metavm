/// Instruction formats defined by the RISC-V specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstructionFormat {
    R,
    I,
    S,
    B,
    U,
    J,
}

/// Decoded RV64IM instruction with operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum Instruction {
    // ── U-type ───────────────────────────────────────────────────────────
    LUI  { rd: u8, imm: i32 },
    AUIPC { rd: u8, imm: i32 },

    // ── J-type ───────────────────────────────────────────────────────────
    JAL  { rd: u8, imm: i32 },

    // ── I-type (jump) ────────────────────────────────────────────────────
    JALR { rd: u8, rs1: u8, imm: i32 },

    // ── B-type (branches) ────────────────────────────────────────────────
    BEQ  { rs1: u8, rs2: u8, imm: i32 },
    BNE  { rs1: u8, rs2: u8, imm: i32 },
    BLT  { rs1: u8, rs2: u8, imm: i32 },
    BGE  { rs1: u8, rs2: u8, imm: i32 },
    BLTU { rs1: u8, rs2: u8, imm: i32 },
    BGEU { rs1: u8, rs2: u8, imm: i32 },

    // ── I-type (loads) ───────────────────────────────────────────────────
    LB   { rd: u8, rs1: u8, imm: i32 },
    LH   { rd: u8, rs1: u8, imm: i32 },
    LW   { rd: u8, rs1: u8, imm: i32 },
    LD   { rd: u8, rs1: u8, imm: i32 },
    LBU  { rd: u8, rs1: u8, imm: i32 },
    LHU  { rd: u8, rs1: u8, imm: i32 },
    LWU  { rd: u8, rs1: u8, imm: i32 },

    // ── S-type (stores) ──────────────────────────────────────────────────
    SB   { rs1: u8, rs2: u8, imm: i32 },
    SH   { rs1: u8, rs2: u8, imm: i32 },
    SW   { rs1: u8, rs2: u8, imm: i32 },
    SD   { rs1: u8, rs2: u8, imm: i32 },

    // ── I-type (integer register-immediate) ──────────────────────────────
    ADDI  { rd: u8, rs1: u8, imm: i32 },
    SLTI  { rd: u8, rs1: u8, imm: i32 },
    SLTIU { rd: u8, rs1: u8, imm: i32 },
    XORI  { rd: u8, rs1: u8, imm: i32 },
    ORI   { rd: u8, rs1: u8, imm: i32 },
    ANDI  { rd: u8, rs1: u8, imm: i32 },
    SLLI  { rd: u8, rs1: u8, imm: i32 },
    SRLI  { rd: u8, rs1: u8, imm: i32 },
    SRAI  { rd: u8, rs1: u8, imm: i32 },

    // ── R-type (integer register-register) ───────────────────────────────
    ADD  { rd: u8, rs1: u8, rs2: u8 },
    SUB  { rd: u8, rs1: u8, rs2: u8 },
    SLL  { rd: u8, rs1: u8, rs2: u8 },
    SLT  { rd: u8, rs1: u8, rs2: u8 },
    SLTU { rd: u8, rs1: u8, rs2: u8 },
    XOR  { rd: u8, rs1: u8, rs2: u8 },
    SRL  { rd: u8, rs1: u8, rs2: u8 },
    SRA  { rd: u8, rs1: u8, rs2: u8 },
    OR   { rd: u8, rs1: u8, rs2: u8 },
    AND  { rd: u8, rs1: u8, rs2: u8 },

    // ── Misc ─────────────────────────────────────────────────────────────
    FENCE  { pred: u8, succ: u8 },
    ECALL,
    EBREAK,

    // ── RV64I W-variants (I-type) ────────────────────────────────────────
    ADDIW { rd: u8, rs1: u8, imm: i32 },
    SLLIW { rd: u8, rs1: u8, imm: i32 },
    SRLIW { rd: u8, rs1: u8, imm: i32 },
    SRAIW { rd: u8, rs1: u8, imm: i32 },

    // ── RV64I W-variants (R-type) ────────────────────────────────────────
    ADDW { rd: u8, rs1: u8, rs2: u8 },
    SUBW { rd: u8, rs1: u8, rs2: u8 },
    SLLW { rd: u8, rs1: u8, rs2: u8 },
    SRLW { rd: u8, rs1: u8, rs2: u8 },
    SRAW { rd: u8, rs1: u8, rs2: u8 },

    // ── M extension (R-type) ─────────────────────────────────────────────
    MUL    { rd: u8, rs1: u8, rs2: u8 },
    MULH   { rd: u8, rs1: u8, rs2: u8 },
    MULHSU { rd: u8, rs1: u8, rs2: u8 },
    MULHU  { rd: u8, rs1: u8, rs2: u8 },
    DIV    { rd: u8, rs1: u8, rs2: u8 },
    DIVU   { rd: u8, rs1: u8, rs2: u8 },
    REM    { rd: u8, rs1: u8, rs2: u8 },
    REMU   { rd: u8, rs1: u8, rs2: u8 },

    // ── RV64M W-variants (R-type) ────────────────────────────────────────
    MULW  { rd: u8, rs1: u8, rs2: u8 },
    DIVW  { rd: u8, rs1: u8, rs2: u8 },
    DIVUW { rd: u8, rs1: u8, rs2: u8 },
    REMW  { rd: u8, rs1: u8, rs2: u8 },
    REMUW { rd: u8, rs1: u8, rs2: u8 },

    // ── CSR instructions ───────────────────────────────────────────────
    CSRRW  { rd: u8, rs1: u8, csr: u16 },
    CSRRS  { rd: u8, rs1: u8, csr: u16 },
    CSRRC  { rd: u8, rs1: u8, csr: u16 },
    CSRRWI { rd: u8, uimm: u8, csr: u16 },
    CSRRSI { rd: u8, uimm: u8, csr: u16 },
    CSRRCI { rd: u8, uimm: u8, csr: u16 },

    // ── Privileged instructions ────────────────────────────────────────
    MRET,
    SRET,
    WFI,
    SFENCE_VMA { rs1: u8, rs2: u8 },
    FENCE_I,

    // ── A extension (atomics) ──────────────────────────────────────────
    LR_W      { rd: u8, rs1: u8, aq: bool, rl: bool },
    LR_D      { rd: u8, rs1: u8, aq: bool, rl: bool },
    SC_W      { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    SC_D      { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOSWAP_W { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOSWAP_D { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOADD_W  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOADD_D  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOAND_W  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOAND_D  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOOR_W   { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOOR_D   { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOXOR_W  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOXOR_D  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMAX_W  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMAX_D  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMIN_W  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMIN_D  { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMAXU_W { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMAXU_D { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMINU_W { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AMOMINU_D { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
}

impl Instruction {
    /// Returns the destination register, or `None` for instructions that do
    /// not write to a register (S-type, B-type, FENCE, ECALL, EBREAK).
    pub fn rd(&self) -> Option<u8> {
        match *self {
            // S-type
            Instruction::SB  { .. }
            | Instruction::SH  { .. }
            | Instruction::SW  { .. }
            | Instruction::SD  { .. }
            // B-type
            | Instruction::BEQ  { .. }
            | Instruction::BNE  { .. }
            | Instruction::BLT  { .. }
            | Instruction::BGE  { .. }
            | Instruction::BLTU { .. }
            | Instruction::BGEU { .. }
            // Special
            | Instruction::FENCE  { .. }
            | Instruction::ECALL
            | Instruction::EBREAK
            // Privileged (no rd)
            | Instruction::MRET
            | Instruction::SRET
            | Instruction::WFI
            | Instruction::SFENCE_VMA { .. }
            | Instruction::FENCE_I => None,

            // U-type
            Instruction::LUI  { rd, .. }
            | Instruction::AUIPC { rd, .. }
            // J-type
            | Instruction::JAL  { rd, .. }
            // I-type
            | Instruction::JALR  { rd, .. }
            | Instruction::LB   { rd, .. }
            | Instruction::LH   { rd, .. }
            | Instruction::LW   { rd, .. }
            | Instruction::LD   { rd, .. }
            | Instruction::LBU  { rd, .. }
            | Instruction::LHU  { rd, .. }
            | Instruction::LWU  { rd, .. }
            | Instruction::ADDI  { rd, .. }
            | Instruction::SLTI  { rd, .. }
            | Instruction::SLTIU { rd, .. }
            | Instruction::XORI  { rd, .. }
            | Instruction::ORI   { rd, .. }
            | Instruction::ANDI  { rd, .. }
            | Instruction::SLLI  { rd, .. }
            | Instruction::SRLI  { rd, .. }
            | Instruction::SRAI  { rd, .. }
            | Instruction::ADDIW { rd, .. }
            | Instruction::SLLIW { rd, .. }
            | Instruction::SRLIW { rd, .. }
            | Instruction::SRAIW { rd, .. }
            // R-type
            | Instruction::ADD  { rd, .. }
            | Instruction::SUB  { rd, .. }
            | Instruction::SLL  { rd, .. }
            | Instruction::SLT  { rd, .. }
            | Instruction::SLTU { rd, .. }
            | Instruction::XOR  { rd, .. }
            | Instruction::SRL  { rd, .. }
            | Instruction::SRA  { rd, .. }
            | Instruction::OR   { rd, .. }
            | Instruction::AND  { rd, .. }
            | Instruction::ADDW { rd, .. }
            | Instruction::SUBW { rd, .. }
            | Instruction::SLLW { rd, .. }
            | Instruction::SRLW { rd, .. }
            | Instruction::SRAW { rd, .. }
            | Instruction::MUL    { rd, .. }
            | Instruction::MULH   { rd, .. }
            | Instruction::MULHSU { rd, .. }
            | Instruction::MULHU  { rd, .. }
            | Instruction::DIV    { rd, .. }
            | Instruction::DIVU   { rd, .. }
            | Instruction::REM    { rd, .. }
            | Instruction::REMU   { rd, .. }
            | Instruction::MULW  { rd, .. }
            | Instruction::DIVW  { rd, .. }
            | Instruction::DIVUW { rd, .. }
            | Instruction::REMW  { rd, .. }
            | Instruction::REMUW { rd, .. }
            // CSR instructions
            | Instruction::CSRRW  { rd, .. }
            | Instruction::CSRRS  { rd, .. }
            | Instruction::CSRRC  { rd, .. }
            | Instruction::CSRRWI { rd, .. }
            | Instruction::CSRRSI { rd, .. }
            | Instruction::CSRRCI { rd, .. }
            // A extension
            | Instruction::LR_W      { rd, .. }
            | Instruction::LR_D      { rd, .. }
            | Instruction::SC_W      { rd, .. }
            | Instruction::SC_D      { rd, .. }
            | Instruction::AMOSWAP_W { rd, .. }
            | Instruction::AMOSWAP_D { rd, .. }
            | Instruction::AMOADD_W  { rd, .. }
            | Instruction::AMOADD_D  { rd, .. }
            | Instruction::AMOAND_W  { rd, .. }
            | Instruction::AMOAND_D  { rd, .. }
            | Instruction::AMOOR_W   { rd, .. }
            | Instruction::AMOOR_D   { rd, .. }
            | Instruction::AMOXOR_W  { rd, .. }
            | Instruction::AMOXOR_D  { rd, .. }
            | Instruction::AMOMAX_W  { rd, .. }
            | Instruction::AMOMAX_D  { rd, .. }
            | Instruction::AMOMIN_W  { rd, .. }
            | Instruction::AMOMIN_D  { rd, .. }
            | Instruction::AMOMAXU_W { rd, .. }
            | Instruction::AMOMAXU_D { rd, .. }
            | Instruction::AMOMINU_W { rd, .. }
            | Instruction::AMOMINU_D { rd, .. } => Some(rd),
        }
    }

    /// Returns the first source register, or `None` for instructions that do
    /// not read rs1 (U-type, J-type, FENCE, ECALL, EBREAK).
    pub fn rs1(&self) -> Option<u8> {
        match *self {
            // U-type
            Instruction::LUI  { .. }
            | Instruction::AUIPC { .. }
            // J-type
            | Instruction::JAL  { .. }
            // Special
            | Instruction::FENCE  { .. }
            | Instruction::ECALL
            | Instruction::EBREAK
            // Privileged (no rs1)
            | Instruction::MRET
            | Instruction::SRET
            | Instruction::WFI
            | Instruction::FENCE_I
            // CSR immediate variants (use uimm, not rs1)
            | Instruction::CSRRWI { .. }
            | Instruction::CSRRSI { .. }
            | Instruction::CSRRCI { .. } => None,

            // I-type
            Instruction::JALR  { rs1, .. }
            | Instruction::LB   { rs1, .. }
            | Instruction::LH   { rs1, .. }
            | Instruction::LW   { rs1, .. }
            | Instruction::LD   { rs1, .. }
            | Instruction::LBU  { rs1, .. }
            | Instruction::LHU  { rs1, .. }
            | Instruction::LWU  { rs1, .. }
            | Instruction::ADDI  { rs1, .. }
            | Instruction::SLTI  { rs1, .. }
            | Instruction::SLTIU { rs1, .. }
            | Instruction::XORI  { rs1, .. }
            | Instruction::ORI   { rs1, .. }
            | Instruction::ANDI  { rs1, .. }
            | Instruction::SLLI  { rs1, .. }
            | Instruction::SRLI  { rs1, .. }
            | Instruction::SRAI  { rs1, .. }
            | Instruction::ADDIW { rs1, .. }
            | Instruction::SLLIW { rs1, .. }
            | Instruction::SRLIW { rs1, .. }
            | Instruction::SRAIW { rs1, .. }
            // S-type
            | Instruction::SB  { rs1, .. }
            | Instruction::SH  { rs1, .. }
            | Instruction::SW  { rs1, .. }
            | Instruction::SD  { rs1, .. }
            // B-type
            | Instruction::BEQ  { rs1, .. }
            | Instruction::BNE  { rs1, .. }
            | Instruction::BLT  { rs1, .. }
            | Instruction::BGE  { rs1, .. }
            | Instruction::BLTU { rs1, .. }
            | Instruction::BGEU { rs1, .. }
            // R-type
            | Instruction::ADD  { rs1, .. }
            | Instruction::SUB  { rs1, .. }
            | Instruction::SLL  { rs1, .. }
            | Instruction::SLT  { rs1, .. }
            | Instruction::SLTU { rs1, .. }
            | Instruction::XOR  { rs1, .. }
            | Instruction::SRL  { rs1, .. }
            | Instruction::SRA  { rs1, .. }
            | Instruction::OR   { rs1, .. }
            | Instruction::AND  { rs1, .. }
            | Instruction::ADDW { rs1, .. }
            | Instruction::SUBW { rs1, .. }
            | Instruction::SLLW { rs1, .. }
            | Instruction::SRLW { rs1, .. }
            | Instruction::SRAW { rs1, .. }
            | Instruction::MUL    { rs1, .. }
            | Instruction::MULH   { rs1, .. }
            | Instruction::MULHSU { rs1, .. }
            | Instruction::MULHU  { rs1, .. }
            | Instruction::DIV    { rs1, .. }
            | Instruction::DIVU   { rs1, .. }
            | Instruction::REM    { rs1, .. }
            | Instruction::REMU   { rs1, .. }
            | Instruction::MULW  { rs1, .. }
            | Instruction::DIVW  { rs1, .. }
            | Instruction::DIVUW { rs1, .. }
            | Instruction::REMW  { rs1, .. }
            | Instruction::REMUW { rs1, .. }
            // CSR register variants
            | Instruction::CSRRW  { rs1, .. }
            | Instruction::CSRRS  { rs1, .. }
            | Instruction::CSRRC  { rs1, .. }
            // SFENCE_VMA
            | Instruction::SFENCE_VMA { rs1, .. }
            // A extension
            | Instruction::LR_W      { rs1, .. }
            | Instruction::LR_D      { rs1, .. }
            | Instruction::SC_W      { rs1, .. }
            | Instruction::SC_D      { rs1, .. }
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
            | Instruction::AMOMINU_D { rs1, .. } => Some(rs1),
        }
    }

    /// Returns the second source register, or `None` for instructions that do
    /// not read rs2 (I-type, U-type, J-type, FENCE, ECALL, EBREAK).
    pub fn rs2(&self) -> Option<u8> {
        match *self {
            // R-type
            Instruction::ADD  { rs2, .. }
            | Instruction::SUB  { rs2, .. }
            | Instruction::SLL  { rs2, .. }
            | Instruction::SLT  { rs2, .. }
            | Instruction::SLTU { rs2, .. }
            | Instruction::XOR  { rs2, .. }
            | Instruction::SRL  { rs2, .. }
            | Instruction::SRA  { rs2, .. }
            | Instruction::OR   { rs2, .. }
            | Instruction::AND  { rs2, .. }
            | Instruction::ADDW { rs2, .. }
            | Instruction::SUBW { rs2, .. }
            | Instruction::SLLW { rs2, .. }
            | Instruction::SRLW { rs2, .. }
            | Instruction::SRAW { rs2, .. }
            | Instruction::MUL    { rs2, .. }
            | Instruction::MULH   { rs2, .. }
            | Instruction::MULHSU { rs2, .. }
            | Instruction::MULHU  { rs2, .. }
            | Instruction::DIV    { rs2, .. }
            | Instruction::DIVU   { rs2, .. }
            | Instruction::REM    { rs2, .. }
            | Instruction::REMU   { rs2, .. }
            | Instruction::MULW  { rs2, .. }
            | Instruction::DIVW  { rs2, .. }
            | Instruction::DIVUW { rs2, .. }
            | Instruction::REMW  { rs2, .. }
            | Instruction::REMUW { rs2, .. }
            // S-type
            | Instruction::SB  { rs2, .. }
            | Instruction::SH  { rs2, .. }
            | Instruction::SW  { rs2, .. }
            | Instruction::SD  { rs2, .. }
            // B-type
            | Instruction::BEQ  { rs2, .. }
            | Instruction::BNE  { rs2, .. }
            | Instruction::BLT  { rs2, .. }
            | Instruction::BGE  { rs2, .. }
            | Instruction::BLTU { rs2, .. }
            | Instruction::BGEU { rs2, .. }
            // SFENCE_VMA
            | Instruction::SFENCE_VMA { rs2, .. }
            // A extension (SC and AMO have rs2)
            | Instruction::SC_W      { rs2, .. }
            | Instruction::SC_D      { rs2, .. }
            | Instruction::AMOSWAP_W { rs2, .. }
            | Instruction::AMOSWAP_D { rs2, .. }
            | Instruction::AMOADD_W  { rs2, .. }
            | Instruction::AMOADD_D  { rs2, .. }
            | Instruction::AMOAND_W  { rs2, .. }
            | Instruction::AMOAND_D  { rs2, .. }
            | Instruction::AMOOR_W   { rs2, .. }
            | Instruction::AMOOR_D   { rs2, .. }
            | Instruction::AMOXOR_W  { rs2, .. }
            | Instruction::AMOXOR_D  { rs2, .. }
            | Instruction::AMOMAX_W  { rs2, .. }
            | Instruction::AMOMAX_D  { rs2, .. }
            | Instruction::AMOMIN_W  { rs2, .. }
            | Instruction::AMOMIN_D  { rs2, .. }
            | Instruction::AMOMAXU_W { rs2, .. }
            | Instruction::AMOMAXU_D { rs2, .. }
            | Instruction::AMOMINU_W { rs2, .. }
            | Instruction::AMOMINU_D { rs2, .. } => Some(rs2),

            _ => None,
        }
    }

    /// Returns the instruction format category.
    pub fn format(&self) -> InstructionFormat {
        match *self {
            // R-type
            Instruction::ADD  { .. }
            | Instruction::SUB  { .. }
            | Instruction::SLL  { .. }
            | Instruction::SLT  { .. }
            | Instruction::SLTU { .. }
            | Instruction::XOR  { .. }
            | Instruction::SRL  { .. }
            | Instruction::SRA  { .. }
            | Instruction::OR   { .. }
            | Instruction::AND  { .. }
            | Instruction::ADDW { .. }
            | Instruction::SUBW { .. }
            | Instruction::SLLW { .. }
            | Instruction::SRLW { .. }
            | Instruction::SRAW { .. }
            | Instruction::MUL    { .. }
            | Instruction::MULH   { .. }
            | Instruction::MULHSU { .. }
            | Instruction::MULHU  { .. }
            | Instruction::DIV    { .. }
            | Instruction::DIVU   { .. }
            | Instruction::REM    { .. }
            | Instruction::REMU   { .. }
            | Instruction::MULW  { .. }
            | Instruction::DIVW  { .. }
            | Instruction::DIVUW { .. }
            | Instruction::REMW  { .. }
            | Instruction::REMUW { .. }
            | Instruction::SFENCE_VMA { .. }
            // A extension (R-type format)
            | Instruction::LR_W      { .. }
            | Instruction::LR_D      { .. }
            | Instruction::SC_W      { .. }
            | Instruction::SC_D      { .. }
            | Instruction::AMOSWAP_W { .. }
            | Instruction::AMOSWAP_D { .. }
            | Instruction::AMOADD_W  { .. }
            | Instruction::AMOADD_D  { .. }
            | Instruction::AMOAND_W  { .. }
            | Instruction::AMOAND_D  { .. }
            | Instruction::AMOOR_W   { .. }
            | Instruction::AMOOR_D   { .. }
            | Instruction::AMOXOR_W  { .. }
            | Instruction::AMOXOR_D  { .. }
            | Instruction::AMOMAX_W  { .. }
            | Instruction::AMOMAX_D  { .. }
            | Instruction::AMOMIN_W  { .. }
            | Instruction::AMOMIN_D  { .. }
            | Instruction::AMOMAXU_W { .. }
            | Instruction::AMOMAXU_D { .. }
            | Instruction::AMOMINU_W { .. }
            | Instruction::AMOMINU_D { .. } => InstructionFormat::R,

            // I-type
            Instruction::JALR  { .. }
            | Instruction::LB   { .. }
            | Instruction::LH   { .. }
            | Instruction::LW   { .. }
            | Instruction::LD   { .. }
            | Instruction::LBU  { .. }
            | Instruction::LHU  { .. }
            | Instruction::LWU  { .. }
            | Instruction::ADDI  { .. }
            | Instruction::SLTI  { .. }
            | Instruction::SLTIU { .. }
            | Instruction::XORI  { .. }
            | Instruction::ORI   { .. }
            | Instruction::ANDI  { .. }
            | Instruction::SLLI  { .. }
            | Instruction::SRLI  { .. }
            | Instruction::SRAI  { .. }
            | Instruction::ADDIW { .. }
            | Instruction::SLLIW { .. }
            | Instruction::SRLIW { .. }
            | Instruction::SRAIW { .. }
            | Instruction::FENCE  { .. }
            | Instruction::ECALL
            | Instruction::EBREAK
            | Instruction::FENCE_I
            | Instruction::MRET
            | Instruction::SRET
            | Instruction::WFI
            // CSR instructions (I-type encoding)
            | Instruction::CSRRW  { .. }
            | Instruction::CSRRS  { .. }
            | Instruction::CSRRC  { .. }
            | Instruction::CSRRWI { .. }
            | Instruction::CSRRSI { .. }
            | Instruction::CSRRCI { .. } => InstructionFormat::I,

            // S-type
            Instruction::SB { .. }
            | Instruction::SH { .. }
            | Instruction::SW { .. }
            | Instruction::SD { .. } => InstructionFormat::S,

            // B-type
            Instruction::BEQ  { .. }
            | Instruction::BNE  { .. }
            | Instruction::BLT  { .. }
            | Instruction::BGE  { .. }
            | Instruction::BLTU { .. }
            | Instruction::BGEU { .. } => InstructionFormat::B,

            // U-type
            Instruction::LUI   { .. }
            | Instruction::AUIPC { .. } => InstructionFormat::U,

            // J-type
            Instruction::JAL { .. } => InstructionFormat::J,
        }
    }
}
