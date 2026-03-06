use crate::isa::Instruction;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DecodeError {
    #[error("unknown opcode: 0x{0:02x}")]
    UnknownOpcode(u8),
    #[error("unknown funct3: 0x{funct3:x} for opcode 0x{opcode:02x}")]
    UnknownFunct3 { opcode: u8, funct3: u8 },
    #[error("unknown funct7: 0x{funct7:x} for opcode 0x{opcode:02x} funct3 0x{funct3:x}")]
    UnknownFunct7 { opcode: u8, funct3: u8, funct7: u8 },
    #[error("illegal compressed instruction: 0x{0:04x}")]
    IllegalCompressed(u16),
}

// ---------------------------------------------------------------------------
// Field extraction helpers
// ---------------------------------------------------------------------------

#[inline]
fn opcode(w: u32) -> u8 {
    (w & 0x7F) as u8
}

#[inline]
fn rd(w: u32) -> u8 {
    ((w >> 7) & 0x1F) as u8
}

#[inline]
fn funct3(w: u32) -> u8 {
    ((w >> 12) & 0x7) as u8
}

#[inline]
fn rs1(w: u32) -> u8 {
    ((w >> 15) & 0x1F) as u8
}

#[inline]
fn rs2(w: u32) -> u8 {
    ((w >> 20) & 0x1F) as u8
}

#[inline]
fn funct7(w: u32) -> u8 {
    ((w >> 25) & 0x7F) as u8
}

// ---------------------------------------------------------------------------
// Immediate decoding helpers (sign-extended to i32)
// ---------------------------------------------------------------------------

/// I-type immediate: bits [31:20], sign-extended.
#[inline]
fn imm_i(w: u32) -> i32 {
    (w as i32) >> 20
}

/// S-type immediate: bits [31:25|11:7], sign-extended.
#[inline]
fn imm_s(w: u32) -> i32 {
    let hi = (w & 0xFE00_0000) as i32 >> 20; // bits [31:25] shifted to [11:5]
    let lo = ((w >> 7) & 0x1F) as i32;        // bits [11:7] -> [4:0]
    hi | lo
}

/// B-type immediate: bits [12|10:5|4:1|11], sign-extended, LSB is always 0.
#[inline]
fn imm_b(w: u32) -> i32 {
    let bit12  = ((w >> 31) & 1) as i32;       // bit 31 -> bit 12
    let bit11  = ((w >> 7) & 1) as i32;        // bit 7  -> bit 11
    let bits10_5 = ((w >> 25) & 0x3F) as i32;  // bits [30:25] -> bits [10:5]
    let bits4_1  = ((w >> 8) & 0xF) as i32;    // bits [11:8]  -> bits [4:1]
    let raw = (bits4_1 << 1)
        | (bits10_5 << 5)
        | (bit11 << 11)
        | (bit12 << 12);
    // Sign-extend from bit 12
    sign_extend(raw, 13)
}

/// U-type immediate: bits [31:12] placed in upper 20 bits (shifted left 12).
#[inline]
fn imm_u(w: u32) -> i32 {
    (w & 0xFFFF_F000) as i32
}

/// J-type immediate: bits [20|10:1|11|19:12], sign-extended, LSB is always 0.
#[inline]
fn imm_j(w: u32) -> i32 {
    let bit20    = ((w >> 31) & 1) as i32;         // bit 31 -> bit 20
    let bits19_12 = ((w >> 12) & 0xFF) as i32;     // bits [19:12]
    let bit11    = ((w >> 20) & 1) as i32;         // bit 20 -> bit 11
    let bits10_1 = ((w >> 21) & 0x3FF) as i32;    // bits [30:21] -> bits [10:1]
    let raw = (bits10_1 << 1)
        | (bit11 << 11)
        | (bits19_12 << 12)
        | (bit20 << 20);
    // Sign-extend from bit 20
    sign_extend(raw, 21)
}

/// Sign-extend a value from `bits` width to full i32.
#[inline]
fn sign_extend(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (val << shift) >> shift
}

// ---------------------------------------------------------------------------
// Main decode function
// ---------------------------------------------------------------------------

/// Decode a 32-bit RISC-V instruction word into an `Instruction`.
///
/// Supports the full RV64IM instruction set: base integer instructions plus
/// the M (multiply/divide) standard extension.
pub fn decode(word: u32) -> Result<Instruction, DecodeError> {
    let op = opcode(word);
    match op {
        // ── U-type ──────────────────────────────────────────────────────
        0x37 => Ok(Instruction::LUI {
            rd: rd(word),
            imm: imm_u(word),
        }),
        0x17 => Ok(Instruction::AUIPC {
            rd: rd(word),
            imm: imm_u(word),
        }),

        // ── J-type ──────────────────────────────────────────────────────
        0x6F => Ok(Instruction::JAL {
            rd: rd(word),
            imm: imm_j(word),
        }),

        // ── I-type (JALR) ───────────────────────────────────────────────
        0x67 => {
            let f3 = funct3(word);
            if f3 != 0 {
                return Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 });
            }
            Ok(Instruction::JALR {
                rd: rd(word),
                rs1: rs1(word),
                imm: imm_i(word),
            })
        }

        // ── B-type (branches) ───────────────────────────────────────────
        0x63 => {
            let f3 = funct3(word);
            let r1 = rs1(word);
            let r2 = rs2(word);
            let imm = imm_b(word);
            match f3 {
                0 => Ok(Instruction::BEQ  { rs1: r1, rs2: r2, imm }),
                1 => Ok(Instruction::BNE  { rs1: r1, rs2: r2, imm }),
                4 => Ok(Instruction::BLT  { rs1: r1, rs2: r2, imm }),
                5 => Ok(Instruction::BGE  { rs1: r1, rs2: r2, imm }),
                6 => Ok(Instruction::BLTU { rs1: r1, rs2: r2, imm }),
                7 => Ok(Instruction::BGEU { rs1: r1, rs2: r2, imm }),
                _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
            }
        }

        // ── I-type (loads) ──────────────────────────────────────────────
        0x03 => {
            let f3 = funct3(word);
            let d = rd(word);
            let r1 = rs1(word);
            let imm = imm_i(word);
            match f3 {
                0 => Ok(Instruction::LB  { rd: d, rs1: r1, imm }),
                1 => Ok(Instruction::LH  { rd: d, rs1: r1, imm }),
                2 => Ok(Instruction::LW  { rd: d, rs1: r1, imm }),
                3 => Ok(Instruction::LD  { rd: d, rs1: r1, imm }),
                4 => Ok(Instruction::LBU { rd: d, rs1: r1, imm }),
                5 => Ok(Instruction::LHU { rd: d, rs1: r1, imm }),
                6 => Ok(Instruction::LWU { rd: d, rs1: r1, imm }),
                _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
            }
        }

        // ── S-type (stores) ─────────────────────────────────────────────
        0x23 => {
            let f3 = funct3(word);
            let r1 = rs1(word);
            let r2 = rs2(word);
            let imm = imm_s(word);
            match f3 {
                0 => Ok(Instruction::SB { rs1: r1, rs2: r2, imm }),
                1 => Ok(Instruction::SH { rs1: r1, rs2: r2, imm }),
                2 => Ok(Instruction::SW { rs1: r1, rs2: r2, imm }),
                3 => Ok(Instruction::SD { rs1: r1, rs2: r2, imm }),
                _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
            }
        }

        // ── I-type (OP-IMM, 64-bit) ────────────────────────────────────
        0x13 => {
            let f3 = funct3(word);
            let d = rd(word);
            let r1 = rs1(word);
            match f3 {
                0 => Ok(Instruction::ADDI  { rd: d, rs1: r1, imm: imm_i(word) }),
                2 => Ok(Instruction::SLTI  { rd: d, rs1: r1, imm: imm_i(word) }),
                3 => Ok(Instruction::SLTIU { rd: d, rs1: r1, imm: imm_i(word) }),
                4 => Ok(Instruction::XORI  { rd: d, rs1: r1, imm: imm_i(word) }),
                6 => Ok(Instruction::ORI   { rd: d, rs1: r1, imm: imm_i(word) }),
                7 => Ok(Instruction::ANDI  { rd: d, rs1: r1, imm: imm_i(word) }),
                1 => {
                    // SLLI: shamt = imm[5:0] (6 bits for RV64)
                    let shamt = (imm_i(word) & 0x3F) as i32;
                    Ok(Instruction::SLLI { rd: d, rs1: r1, imm: shamt })
                }
                5 => {
                    // SRLI vs SRAI: bit 30 of word distinguishes
                    let shamt = (imm_i(word) & 0x3F) as i32;
                    if word & (1 << 30) != 0 {
                        Ok(Instruction::SRAI { rd: d, rs1: r1, imm: shamt })
                    } else {
                        Ok(Instruction::SRLI { rd: d, rs1: r1, imm: shamt })
                    }
                }
                _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
            }
        }

        // ── R-type (OP, 64-bit) ─────────────────────────────────────────
        0x33 => decode_r_type(word, op),

        // ── I-type (OP-IMM-32) ──────────────────────────────────────────
        0x1B => {
            let f3 = funct3(word);
            let d = rd(word);
            let r1 = rs1(word);
            match f3 {
                0 => Ok(Instruction::ADDIW { rd: d, rs1: r1, imm: imm_i(word) }),
                1 => {
                    // SLLIW: shamt = imm[4:0] (5 bits for W-variants)
                    let shamt = (imm_i(word) & 0x1F) as i32;
                    Ok(Instruction::SLLIW { rd: d, rs1: r1, imm: shamt })
                }
                5 => {
                    // SRLIW vs SRAIW: bit 30 distinguishes
                    let shamt = (imm_i(word) & 0x1F) as i32;
                    if word & (1 << 30) != 0 {
                        Ok(Instruction::SRAIW { rd: d, rs1: r1, imm: shamt })
                    } else {
                        Ok(Instruction::SRLIW { rd: d, rs1: r1, imm: shamt })
                    }
                }
                _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
            }
        }

        // ── R-type (OP-32) ──────────────────────────────────────────────
        0x3B => decode_r32_type(word, op),

        // ── FENCE / FENCE.I ────────────────────────────────────────────
        0x0F => {
            let f3 = funct3(word);
            match f3 {
                0 => {
                    let pred = ((word >> 24) & 0xF) as u8;
                    let succ = ((word >> 20) & 0xF) as u8;
                    Ok(Instruction::FENCE { pred, succ })
                }
                1 => Ok(Instruction::FENCE_I),
                _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
            }
        }

        // ── SYSTEM (CSR + privileged) ─────────────────────────────────
        0x73 => {
            let f3 = funct3(word);
            match f3 {
                0 => {
                    // ECALL, EBREAK, MRET, SRET, WFI, SFENCE.VMA
                    let f7 = funct7(word);
                    let r2 = rs2(word);
                    match (f7, r2) {
                        (0x00, 0) => Ok(Instruction::ECALL),
                        (0x00, 1) => Ok(Instruction::EBREAK),
                        (0x18, 2) => Ok(Instruction::MRET),
                        (0x08, 2) => Ok(Instruction::SRET),
                        (0x08, 5) => Ok(Instruction::WFI),
                        (0x09, _) => Ok(Instruction::SFENCE_VMA { rs1: rs1(word), rs2: r2 }),
                        _ => Err(DecodeError::UnknownFunct7 { opcode: op, funct3: f3, funct7: f7 }),
                    }
                }
                // CSRRW
                1 => {
                    let csr = (word >> 20) as u16 & 0xFFF;
                    Ok(Instruction::CSRRW { rd: rd(word), rs1: rs1(word), csr })
                }
                // CSRRS
                2 => {
                    let csr = (word >> 20) as u16 & 0xFFF;
                    Ok(Instruction::CSRRS { rd: rd(word), rs1: rs1(word), csr })
                }
                // CSRRC
                3 => {
                    let csr = (word >> 20) as u16 & 0xFFF;
                    Ok(Instruction::CSRRC { rd: rd(word), rs1: rs1(word), csr })
                }
                // CSRRWI
                5 => {
                    let csr = (word >> 20) as u16 & 0xFFF;
                    Ok(Instruction::CSRRWI { rd: rd(word), uimm: rs1(word), csr })
                }
                // CSRRSI
                6 => {
                    let csr = (word >> 20) as u16 & 0xFFF;
                    Ok(Instruction::CSRRSI { rd: rd(word), uimm: rs1(word), csr })
                }
                // CSRRCI
                7 => {
                    let csr = (word >> 20) as u16 & 0xFFF;
                    Ok(Instruction::CSRRCI { rd: rd(word), uimm: rs1(word), csr })
                }
                _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
            }
        }

        // ── A extension (AMO) ─────────────────────────────────────────
        0x2F => decode_amo(word, op),

        _ => Err(DecodeError::UnknownOpcode(op)),
    }
}

/// Decode an R-type instruction with opcode 0x33 (OP, 64-bit).
fn decode_r_type(word: u32, op: u8) -> Result<Instruction, DecodeError> {
    let f3 = funct3(word);
    let f7 = funct7(word);
    let d = rd(word);
    let r1 = rs1(word);
    let r2 = rs2(word);

    match f7 {
        0x00 => match f3 {
            0 => Ok(Instruction::ADD  { rd: d, rs1: r1, rs2: r2 }),
            1 => Ok(Instruction::SLL  { rd: d, rs1: r1, rs2: r2 }),
            2 => Ok(Instruction::SLT  { rd: d, rs1: r1, rs2: r2 }),
            3 => Ok(Instruction::SLTU { rd: d, rs1: r1, rs2: r2 }),
            4 => Ok(Instruction::XOR  { rd: d, rs1: r1, rs2: r2 }),
            5 => Ok(Instruction::SRL  { rd: d, rs1: r1, rs2: r2 }),
            6 => Ok(Instruction::OR   { rd: d, rs1: r1, rs2: r2 }),
            7 => Ok(Instruction::AND  { rd: d, rs1: r1, rs2: r2 }),
            _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
        },
        0x20 => match f3 {
            0 => Ok(Instruction::SUB { rd: d, rs1: r1, rs2: r2 }),
            5 => Ok(Instruction::SRA { rd: d, rs1: r1, rs2: r2 }),
            _ => Err(DecodeError::UnknownFunct7 { opcode: op, funct3: f3, funct7: f7 }),
        },
        0x01 => match f3 {
            0 => Ok(Instruction::MUL    { rd: d, rs1: r1, rs2: r2 }),
            1 => Ok(Instruction::MULH   { rd: d, rs1: r1, rs2: r2 }),
            2 => Ok(Instruction::MULHSU { rd: d, rs1: r1, rs2: r2 }),
            3 => Ok(Instruction::MULHU  { rd: d, rs1: r1, rs2: r2 }),
            4 => Ok(Instruction::DIV    { rd: d, rs1: r1, rs2: r2 }),
            5 => Ok(Instruction::DIVU   { rd: d, rs1: r1, rs2: r2 }),
            6 => Ok(Instruction::REM    { rd: d, rs1: r1, rs2: r2 }),
            7 => Ok(Instruction::REMU   { rd: d, rs1: r1, rs2: r2 }),
            _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
        },
        _ => Err(DecodeError::UnknownFunct7 { opcode: op, funct3: f3, funct7: f7 }),
    }
}

/// Decode an R-type instruction with opcode 0x3B (OP-32).
fn decode_r32_type(word: u32, op: u8) -> Result<Instruction, DecodeError> {
    let f3 = funct3(word);
    let f7 = funct7(word);
    let d = rd(word);
    let r1 = rs1(word);
    let r2 = rs2(word);

    match f7 {
        0x00 => match f3 {
            0 => Ok(Instruction::ADDW { rd: d, rs1: r1, rs2: r2 }),
            1 => Ok(Instruction::SLLW { rd: d, rs1: r1, rs2: r2 }),
            5 => Ok(Instruction::SRLW { rd: d, rs1: r1, rs2: r2 }),
            _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
        },
        0x20 => match f3 {
            0 => Ok(Instruction::SUBW { rd: d, rs1: r1, rs2: r2 }),
            5 => Ok(Instruction::SRAW { rd: d, rs1: r1, rs2: r2 }),
            _ => Err(DecodeError::UnknownFunct7 { opcode: op, funct3: f3, funct7: f7 }),
        },
        0x01 => match f3 {
            0 => Ok(Instruction::MULW  { rd: d, rs1: r1, rs2: r2 }),
            4 => Ok(Instruction::DIVW  { rd: d, rs1: r1, rs2: r2 }),
            5 => Ok(Instruction::DIVUW { rd: d, rs1: r1, rs2: r2 }),
            6 => Ok(Instruction::REMW  { rd: d, rs1: r1, rs2: r2 }),
            7 => Ok(Instruction::REMUW { rd: d, rs1: r1, rs2: r2 }),
            _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
        },
        _ => Err(DecodeError::UnknownFunct7 { opcode: op, funct3: f3, funct7: f7 }),
    }
}

/// Decode an A-extension (atomic) instruction with opcode 0x2F.
fn decode_amo(word: u32, op: u8) -> Result<Instruction, DecodeError> {
    let f3 = funct3(word);
    let d = rd(word);
    let r1 = rs1(word);
    let r2 = rs2(word);
    let funct5 = (word >> 27) & 0x1F;
    let aq = (word >> 26) & 1 != 0;
    let rl = (word >> 25) & 1 != 0;

    match f3 {
        // .W (32-bit)
        2 => match funct5 {
            0x02 => Ok(Instruction::LR_W { rd: d, rs1: r1, aq, rl }),
            0x03 => Ok(Instruction::SC_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x01 => Ok(Instruction::AMOSWAP_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x00 => Ok(Instruction::AMOADD_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x0C => Ok(Instruction::AMOAND_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x08 => Ok(Instruction::AMOOR_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x04 => Ok(Instruction::AMOXOR_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x14 => Ok(Instruction::AMOMAX_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x10 => Ok(Instruction::AMOMIN_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x1C => Ok(Instruction::AMOMAXU_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x18 => Ok(Instruction::AMOMINU_W { rd: d, rs1: r1, rs2: r2, aq, rl }),
            _ => Err(DecodeError::UnknownFunct7 { opcode: op, funct3: f3, funct7: (funct5 as u8) << 2 }),
        },
        // .D (64-bit)
        3 => match funct5 {
            0x02 => Ok(Instruction::LR_D { rd: d, rs1: r1, aq, rl }),
            0x03 => Ok(Instruction::SC_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x01 => Ok(Instruction::AMOSWAP_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x00 => Ok(Instruction::AMOADD_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x0C => Ok(Instruction::AMOAND_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x08 => Ok(Instruction::AMOOR_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x04 => Ok(Instruction::AMOXOR_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x14 => Ok(Instruction::AMOMAX_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x10 => Ok(Instruction::AMOMIN_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x1C => Ok(Instruction::AMOMAXU_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            0x18 => Ok(Instruction::AMOMINU_D { rd: d, rs1: r1, rs2: r2, aq, rl }),
            _ => Err(DecodeError::UnknownFunct7 { opcode: op, funct3: f3, funct7: (funct5 as u8) << 2 }),
        },
        _ => Err(DecodeError::UnknownFunct3 { opcode: op, funct3: f3 }),
    }
}

/// Fetch and decode a variable-length instruction from memory.
/// Returns the decoded instruction and its length in bytes (2 or 4).
pub fn fetch_decode(memory: &crate::memory::Memory, pc: u64) -> Result<FetchedInstruction, DecodeError> {
    let lo = memory.load_half(pc);
    if lo & 0x3 != 0x3 {
        // Compressed (16-bit) instruction
        let insn = decode_compressed(lo)?;
        Ok(FetchedInstruction { instruction: insn, len: 2 })
    } else {
        // Standard 32-bit instruction
        let word = memory.load_word(pc);
        let insn = decode(word)?;
        Ok(FetchedInstruction { instruction: insn, len: 4 })
    }
}

/// A fetched instruction with its byte length.
#[derive(Clone, Copy, Debug)]
pub struct FetchedInstruction {
    pub instruction: Instruction,
    pub len: u8,
}

/// Decode a 16-bit compressed (C extension) instruction.
pub fn decode_compressed(half: u16) -> Result<Instruction, DecodeError> {
    let quadrant = half & 0x3;
    let funct3 = ((half >> 13) & 0x7) as u8;

    match quadrant {
        0b00 => decode_c_q0(half, funct3),
        0b01 => decode_c_q1(half, funct3),
        0b10 => decode_c_q2(half, funct3),
        _ => Err(DecodeError::IllegalCompressed(half)),
    }
}

/// Map 3-bit compressed register field to x8-x15.
#[inline]
fn creg(bits: u16) -> u8 {
    (bits as u8 & 0x7) + 8
}

/// Quadrant 0 compressed instructions.
fn decode_c_q0(half: u16, funct3: u8) -> Result<Instruction, DecodeError> {
    match funct3 {
        // C.ADDI4SPN: addi rd', x2, nzuimm
        0b000 => {
            let rd = creg((half >> 2) & 0x7);
            // nzuimm[5:4|9:6|2|3]
            let nzuimm = (((half >> 6) & 1) << 2)
                | (((half >> 5) & 1) << 3)
                | (((half >> 11) & 0x3) << 4)
                | (((half >> 7) & 0xF) << 6);
            if nzuimm == 0 {
                return Err(DecodeError::IllegalCompressed(half));
            }
            Ok(Instruction::ADDI { rd, rs1: 2, imm: nzuimm as i32 })
        }
        // C.LW: lw rd', offset(rs1')
        0b010 => {
            let rd = creg((half >> 2) & 0x7);
            let rs1 = creg((half >> 7) & 0x7);
            // offset[5:3|2|6]
            let off = (((half >> 6) & 1) << 2)
                | (((half >> 10) & 0x7) << 3)
                | (((half >> 5) & 1) << 6);
            Ok(Instruction::LW { rd, rs1, imm: off as i32 })
        }
        // C.LD: ld rd', offset(rs1')
        0b011 => {
            let rd = creg((half >> 2) & 0x7);
            let rs1 = creg((half >> 7) & 0x7);
            // offset[5:3|7:6]
            let off = (((half >> 10) & 0x7) << 3)
                | (((half >> 5) & 0x3) << 6);
            Ok(Instruction::LD { rd, rs1, imm: off as i32 })
        }
        // C.SW: sw rs2', offset(rs1')
        0b110 => {
            let rs2 = creg((half >> 2) & 0x7);
            let rs1 = creg((half >> 7) & 0x7);
            let off = (((half >> 6) & 1) << 2)
                | (((half >> 10) & 0x7) << 3)
                | (((half >> 5) & 1) << 6);
            Ok(Instruction::SW { rs1, rs2, imm: off as i32 })
        }
        // C.SD: sd rs2', offset(rs1')
        0b111 => {
            let rs2 = creg((half >> 2) & 0x7);
            let rs1 = creg((half >> 7) & 0x7);
            let off = (((half >> 10) & 0x7) << 3)
                | (((half >> 5) & 0x3) << 6);
            Ok(Instruction::SD { rs1, rs2, imm: off as i32 })
        }
        _ => Err(DecodeError::IllegalCompressed(half)),
    }
}

/// Sign-extend a value from `bits` width to i32.
#[inline]
fn c_sign_extend(val: u16, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((val as i32) << shift) >> shift
}

/// Quadrant 1 compressed instructions.
fn decode_c_q1(half: u16, funct3: u8) -> Result<Instruction, DecodeError> {
    match funct3 {
        // C.NOP / C.ADDI
        0b000 => {
            let rd = ((half >> 7) & 0x1F) as u8;
            let imm = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as u16;
            let imm = c_sign_extend(imm, 6);
            Ok(Instruction::ADDI { rd, rs1: rd, imm })
        }
        // C.ADDIW
        0b001 => {
            let rd = ((half >> 7) & 0x1F) as u8;
            let imm = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as u16;
            let imm = c_sign_extend(imm, 6);
            Ok(Instruction::ADDIW { rd, rs1: rd, imm })
        }
        // C.LI: addi rd, x0, imm
        0b010 => {
            let rd = ((half >> 7) & 0x1F) as u8;
            let imm = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as u16;
            let imm = c_sign_extend(imm, 6);
            Ok(Instruction::ADDI { rd, rs1: 0, imm })
        }
        // C.ADDI16SP / C.LUI
        0b011 => {
            let rd = ((half >> 7) & 0x1F) as u8;
            if rd == 2 {
                // C.ADDI16SP: addi x2, x2, nzimm
                let nzimm = (((half >> 6) & 1) << 4)
                    | (((half >> 2) & 1) << 5)
                    | (((half >> 5) & 1) << 6)
                    | (((half >> 3) & 0x3) << 7)
                    | (((half >> 12) & 1) << 9);
                let nzimm = c_sign_extend(nzimm as u16, 10);
                if nzimm == 0 {
                    return Err(DecodeError::IllegalCompressed(half));
                }
                Ok(Instruction::ADDI { rd: 2, rs1: 2, imm: nzimm })
            } else {
                // C.LUI
                let imm = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as u16;
                let imm = c_sign_extend(imm, 6);
                if imm == 0 {
                    return Err(DecodeError::IllegalCompressed(half));
                }
                Ok(Instruction::LUI { rd, imm: imm << 12 })
            }
        }
        // C.SRLI / C.SRAI / C.ANDI / C.SUB / C.XOR / C.OR / C.AND / C.SUBW / C.ADDW
        0b100 => {
            let funct2 = (half >> 10) & 0x3;
            let rd = creg((half >> 7) & 0x7);
            match funct2 {
                // C.SRLI
                0b00 => {
                    let shamt = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as i32;
                    Ok(Instruction::SRLI { rd, rs1: rd, imm: shamt })
                }
                // C.SRAI
                0b01 => {
                    let shamt = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as i32;
                    Ok(Instruction::SRAI { rd, rs1: rd, imm: shamt })
                }
                // C.ANDI
                0b10 => {
                    let imm = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as u16;
                    let imm = c_sign_extend(imm, 6);
                    Ok(Instruction::ANDI { rd, rs1: rd, imm })
                }
                // C.SUB / C.XOR / C.OR / C.AND / C.SUBW / C.ADDW
                0b11 => {
                    let rs2 = creg((half >> 2) & 0x7);
                    let funct1 = (half >> 12) & 1;
                    let funct2b = (half >> 5) & 0x3;
                    match (funct1, funct2b) {
                        (0, 0b00) => Ok(Instruction::SUB { rd, rs1: rd, rs2 }),
                        (0, 0b01) => Ok(Instruction::XOR { rd, rs1: rd, rs2 }),
                        (0, 0b10) => Ok(Instruction::OR { rd, rs1: rd, rs2 }),
                        (0, 0b11) => Ok(Instruction::AND { rd, rs1: rd, rs2 }),
                        (1, 0b00) => Ok(Instruction::SUBW { rd, rs1: rd, rs2 }),
                        (1, 0b01) => Ok(Instruction::ADDW { rd, rs1: rd, rs2 }),
                        _ => Err(DecodeError::IllegalCompressed(half)),
                    }
                }
                _ => unreachable!(),
            }
        }
        // C.J: jal x0, offset
        0b101 => {
            // offset[11|4|9:8|10|6|7|3:1|5]
            let off = (((half >> 3) & 0x7) << 1)
                | (((half >> 11) & 1) << 4)
                | (((half >> 2) & 1) << 5)
                | (((half >> 7) & 1) << 6)
                | (((half >> 6) & 1) << 7)
                | (((half >> 9) & 0x3) << 8)
                | (((half >> 8) & 1) << 10)
                | (((half >> 12) & 1) << 11);
            let off = c_sign_extend(off as u16, 12);
            Ok(Instruction::JAL { rd: 0, imm: off })
        }
        // C.BEQZ: beq rs1', x0, offset
        0b110 => {
            let rs1 = creg((half >> 7) & 0x7);
            let off = (((half >> 3) & 0x3) << 1)
                | (((half >> 10) & 0x3) << 3)
                | (((half >> 2) & 1) << 5)
                | (((half >> 5) & 0x3) << 6)
                | (((half >> 12) & 1) << 8);
            let off = c_sign_extend(off as u16, 9);
            Ok(Instruction::BEQ { rs1, rs2: 0, imm: off })
        }
        // C.BNEZ: bne rs1', x0, offset
        0b111 => {
            let rs1 = creg((half >> 7) & 0x7);
            let off = (((half >> 3) & 0x3) << 1)
                | (((half >> 10) & 0x3) << 3)
                | (((half >> 2) & 1) << 5)
                | (((half >> 5) & 0x3) << 6)
                | (((half >> 12) & 1) << 8);
            let off = c_sign_extend(off as u16, 9);
            Ok(Instruction::BNE { rs1, rs2: 0, imm: off })
        }
        _ => Err(DecodeError::IllegalCompressed(half)),
    }
}

/// Quadrant 2 compressed instructions.
fn decode_c_q2(half: u16, funct3: u8) -> Result<Instruction, DecodeError> {
    match funct3 {
        // C.SLLI
        0b000 => {
            let rd = ((half >> 7) & 0x1F) as u8;
            let shamt = (((half >> 2) & 0x1F) | (((half >> 12) & 1) << 5)) as i32;
            Ok(Instruction::SLLI { rd, rs1: rd, imm: shamt })
        }
        // C.LWSP: lw rd, offset(x2)
        0b010 => {
            let rd = ((half >> 7) & 0x1F) as u8;
            let off = (((half >> 4) & 0x7) << 2)
                | (((half >> 12) & 1) << 5)
                | (((half >> 2) & 0x3) << 6);
            Ok(Instruction::LW { rd, rs1: 2, imm: off as i32 })
        }
        // C.LDSP: ld rd, offset(x2)
        0b011 => {
            let rd = ((half >> 7) & 0x1F) as u8;
            let off = (((half >> 5) & 0x3) << 3)
                | (((half >> 12) & 1) << 5)
                | (((half >> 2) & 0x7) << 6);
            Ok(Instruction::LD { rd, rs1: 2, imm: off as i32 })
        }
        // C.JR / C.MV / C.EBREAK / C.JALR / C.ADD
        0b100 => {
            let bit12 = (half >> 12) & 1;
            let rd = ((half >> 7) & 0x1F) as u8;
            let rs2 = ((half >> 2) & 0x1F) as u8;
            match (bit12, rd, rs2) {
                (0, _, 0) if rd != 0 => {
                    // C.JR: jalr x0, rs1, 0
                    Ok(Instruction::JALR { rd: 0, rs1: rd, imm: 0 })
                }
                (0, _, _) if rs2 != 0 => {
                    // C.MV: add rd, x0, rs2
                    Ok(Instruction::ADD { rd, rs1: 0, rs2 })
                }
                (1, 0, 0) => {
                    // C.EBREAK
                    Ok(Instruction::EBREAK)
                }
                (1, _, 0) if rd != 0 => {
                    // C.JALR: jalr x1, rs1, 0
                    Ok(Instruction::JALR { rd: 1, rs1: rd, imm: 0 })
                }
                (1, _, _) if rs2 != 0 => {
                    // C.ADD: add rd, rd, rs2
                    Ok(Instruction::ADD { rd, rs1: rd, rs2 })
                }
                _ => Err(DecodeError::IllegalCompressed(half)),
            }
        }
        // C.SWSP: sw rs2, offset(x2)
        0b110 => {
            let rs2 = ((half >> 2) & 0x1F) as u8;
            let off = (((half >> 9) & 0xF) << 2)
                | (((half >> 7) & 0x3) << 6);
            Ok(Instruction::SW { rs1: 2, rs2, imm: off as i32 })
        }
        // C.SDSP: sd rs2, offset(x2)
        0b111 => {
            let rs2 = ((half >> 2) & 0x1F) as u8;
            let off = (((half >> 10) & 0x7) << 3)
                | (((half >> 7) & 0x7) << 6);
            Ok(Instruction::SD { rs1: 2, rs2, imm: off as i32 })
        }
        _ => Err(DecodeError::IllegalCompressed(half)),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Encoding helpers -- build raw 32-bit instruction words for testing.
    // -----------------------------------------------------------------------

    fn encode_r(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn encode_i(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm: i32) -> u32 {
        let imm_bits = (imm as u32) & 0xFFF;
        (imm_bits << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn encode_s(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
        let imm_u = imm as u32;
        let imm_11_5 = (imm_u >> 5) & 0x7F;
        let imm_4_0 = imm_u & 0x1F;
        (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode
    }

    fn encode_b(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
        let imm_u = imm as u32;
        let bit12 = (imm_u >> 12) & 1;
        let bit11 = (imm_u >> 11) & 1;
        let bits10_5 = (imm_u >> 5) & 0x3F;
        let bits4_1 = (imm_u >> 1) & 0xF;
        (bit12 << 31)
            | (bits10_5 << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (bits4_1 << 8)
            | (bit11 << 7)
            | opcode
    }

    fn encode_u(opcode: u32, rd: u32, imm: i32) -> u32 {
        // imm already has bits [31:12] placed; mask upper 20 bits.
        ((imm as u32) & 0xFFFF_F000) | (rd << 7) | opcode
    }

    fn encode_j(opcode: u32, rd: u32, imm: i32) -> u32 {
        let imm_u = imm as u32;
        let bit20 = (imm_u >> 20) & 1;
        let bits19_12 = (imm_u >> 12) & 0xFF;
        let bit11 = (imm_u >> 11) & 1;
        let bits10_1 = (imm_u >> 1) & 0x3FF;
        (bit20 << 31)
            | (bits10_1 << 21)
            | (bit11 << 20)
            | (bits19_12 << 12)
            | (rd << 7)
            | opcode
    }

    // -----------------------------------------------------------------------
    // U-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lui() {
        let word = encode_u(0x37, 5, 0x12345_000_u32 as i32);
        let inst = decode(word).unwrap();
        assert_eq!(inst, Instruction::LUI { rd: 5, imm: 0x12345_000_u32 as i32 });
    }

    #[test]
    fn test_auipc() {
        let word = encode_u(0x17, 10, 0xFFFFF_000_u32 as i32);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::AUIPC { rd: 10, imm: 0xFFFFF_000_u32 as i32 }
        );
    }

    #[test]
    fn test_lui_zero_imm() {
        let word = encode_u(0x37, 1, 0);
        let inst = decode(word).unwrap();
        assert_eq!(inst, Instruction::LUI { rd: 1, imm: 0 });
    }

    // -----------------------------------------------------------------------
    // J-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jal_positive() {
        // JAL x1, +100 (offset = 100)
        let word = encode_j(0x6F, 1, 100);
        let inst = decode(word).unwrap();
        assert_eq!(inst, Instruction::JAL { rd: 1, imm: 100 });
    }

    #[test]
    fn test_jal_negative() {
        // JAL x1, -8
        let word = encode_j(0x6F, 1, -8);
        let inst = decode(word).unwrap();
        assert_eq!(inst, Instruction::JAL { rd: 1, imm: -8 });
    }

    #[test]
    fn test_jal_large_positive() {
        let word = encode_j(0x6F, 1, 0xFFE); // 4094
        let inst = decode(word).unwrap();
        assert_eq!(inst, Instruction::JAL { rd: 1, imm: 0xFFE });
    }

    // -----------------------------------------------------------------------
    // I-type (JALR) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jalr() {
        let word = encode_i(0x67, 1, 0, 5, 16);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::JALR { rd: 1, rs1: 5, imm: 16 }
        );
    }

    #[test]
    fn test_jalr_negative_imm() {
        let word = encode_i(0x67, 1, 0, 5, -4);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::JALR { rd: 1, rs1: 5, imm: -4 }
        );
    }

    // -----------------------------------------------------------------------
    // B-type (branch) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_beq() {
        let word = encode_b(0x63, 0, 1, 2, 8);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::BEQ { rs1: 1, rs2: 2, imm: 8 }
        );
    }

    #[test]
    fn test_bne() {
        let word = encode_b(0x63, 1, 3, 4, -12);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::BNE { rs1: 3, rs2: 4, imm: -12 }
        );
    }

    #[test]
    fn test_blt() {
        let word = encode_b(0x63, 4, 5, 6, 256);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::BLT { rs1: 5, rs2: 6, imm: 256 }
        );
    }

    #[test]
    fn test_bge() {
        let word = encode_b(0x63, 5, 7, 8, -256);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::BGE { rs1: 7, rs2: 8, imm: -256 }
        );
    }

    #[test]
    fn test_bltu() {
        let word = encode_b(0x63, 6, 9, 10, 4);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::BLTU { rs1: 9, rs2: 10, imm: 4 }
        );
    }

    #[test]
    fn test_bgeu() {
        let word = encode_b(0x63, 7, 11, 12, -4);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::BGEU { rs1: 11, rs2: 12, imm: -4 }
        );
    }

    #[test]
    fn test_branch_sign_extension() {
        // Test that negative branch offsets are sign-extended correctly
        let word = encode_b(0x63, 0, 1, 2, -2048);
        let inst = decode(word).unwrap();
        assert_eq!(
            inst,
            Instruction::BEQ { rs1: 1, rs2: 2, imm: -2048 }
        );
    }

    // -----------------------------------------------------------------------
    // Load tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lb() {
        let word = encode_i(0x03, 5, 0, 10, 42);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LB { rd: 5, rs1: 10, imm: 42 }
        );
    }

    #[test]
    fn test_lh() {
        let word = encode_i(0x03, 5, 1, 10, -1);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LH { rd: 5, rs1: 10, imm: -1 }
        );
    }

    #[test]
    fn test_lw() {
        let word = encode_i(0x03, 5, 2, 10, 0);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LW { rd: 5, rs1: 10, imm: 0 }
        );
    }

    #[test]
    fn test_ld() {
        let word = encode_i(0x03, 5, 3, 10, 8);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LD { rd: 5, rs1: 10, imm: 8 }
        );
    }

    #[test]
    fn test_lbu() {
        let word = encode_i(0x03, 5, 4, 10, 100);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LBU { rd: 5, rs1: 10, imm: 100 }
        );
    }

    #[test]
    fn test_lhu() {
        let word = encode_i(0x03, 5, 5, 10, -100);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LHU { rd: 5, rs1: 10, imm: -100 }
        );
    }

    #[test]
    fn test_lwu() {
        let word = encode_i(0x03, 5, 6, 10, 2047);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LWU { rd: 5, rs1: 10, imm: 2047 }
        );
    }

    #[test]
    fn test_load_negative_imm() {
        let word = encode_i(0x03, 5, 0, 10, -2048);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LB { rd: 5, rs1: 10, imm: -2048 }
        );
    }

    // -----------------------------------------------------------------------
    // Store tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sb() {
        let word = encode_s(0x23, 0, 10, 5, 42);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SB { rs1: 10, rs2: 5, imm: 42 }
        );
    }

    #[test]
    fn test_sh() {
        let word = encode_s(0x23, 1, 10, 5, -1);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SH { rs1: 10, rs2: 5, imm: -1 }
        );
    }

    #[test]
    fn test_sw() {
        let word = encode_s(0x23, 2, 10, 5, 0);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SW { rs1: 10, rs2: 5, imm: 0 }
        );
    }

    #[test]
    fn test_sd() {
        let word = encode_s(0x23, 3, 10, 5, -2048);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SD { rs1: 10, rs2: 5, imm: -2048 }
        );
    }

    #[test]
    fn test_store_positive_max() {
        let word = encode_s(0x23, 0, 10, 5, 2047);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SB { rs1: 10, rs2: 5, imm: 2047 }
        );
    }

    // -----------------------------------------------------------------------
    // OP-IMM tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_addi() {
        let word = encode_i(0x13, 5, 0, 10, 42);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDI { rd: 5, rs1: 10, imm: 42 }
        );
    }

    #[test]
    fn test_addi_negative() {
        let word = encode_i(0x13, 5, 0, 10, -1);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDI { rd: 5, rs1: 10, imm: -1 }
        );
    }

    #[test]
    fn test_slti() {
        let word = encode_i(0x13, 5, 2, 10, 100);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLTI { rd: 5, rs1: 10, imm: 100 }
        );
    }

    #[test]
    fn test_sltiu() {
        let word = encode_i(0x13, 5, 3, 10, 100);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLTIU { rd: 5, rs1: 10, imm: 100 }
        );
    }

    #[test]
    fn test_xori() {
        let word = encode_i(0x13, 5, 4, 10, 0xFF);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::XORI { rd: 5, rs1: 10, imm: 0xFF }
        );
    }

    #[test]
    fn test_ori() {
        let word = encode_i(0x13, 5, 6, 10, 0xFF);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ORI { rd: 5, rs1: 10, imm: 0xFF }
        );
    }

    #[test]
    fn test_andi() {
        let word = encode_i(0x13, 5, 7, 10, 0xFF);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ANDI { rd: 5, rs1: 10, imm: 0xFF }
        );
    }

    // -----------------------------------------------------------------------
    // Shift immediate tests (OP-IMM)
    // -----------------------------------------------------------------------

    #[test]
    fn test_slli() {
        // SLLI rd, rs1, shamt  -- funct3=1, funct7[6]=0
        // imm[11:0] = 0b000000_shamt (shamt=5 -> 0x005)
        let word = encode_i(0x13, 5, 1, 10, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLLI { rd: 5, rs1: 10, imm: 5 }
        );
    }

    #[test]
    fn test_slli_large_shamt() {
        // RV64 allows 6-bit shift amount (0-63)
        let word = encode_i(0x13, 5, 1, 10, 63);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLLI { rd: 5, rs1: 10, imm: 63 }
        );
    }

    #[test]
    fn test_srli() {
        // SRLI: funct3=5, bit30=0, shamt=7
        let word = encode_i(0x13, 5, 5, 10, 7);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRLI { rd: 5, rs1: 10, imm: 7 }
        );
    }

    #[test]
    fn test_srai() {
        // SRAI: funct3=5, bit30=1, shamt=7
        // imm[11:0] = 0b010000_000111 = 0x407
        let imm = (1 << 10) | 7; // bit 10 sets bit 30 in the word
        let word = encode_i(0x13, 5, 5, 10, imm);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRAI { rd: 5, rs1: 10, imm: 7 }
        );
    }

    #[test]
    fn test_srai_large_shamt() {
        // SRAI with shamt=63 (max for RV64)
        let imm = (1 << 10) | 63;
        let word = encode_i(0x13, 5, 5, 10, imm);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRAI { rd: 5, rs1: 10, imm: 63 }
        );
    }

    // -----------------------------------------------------------------------
    // R-type (OP, funct7=0x00) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add() {
        let word = encode_r(0x33, 5, 0, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADD { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_sll() {
        let word = encode_r(0x33, 5, 1, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLL { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_slt() {
        let word = encode_r(0x33, 5, 2, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLT { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_sltu() {
        let word = encode_r(0x33, 5, 3, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLTU { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_xor() {
        let word = encode_r(0x33, 5, 4, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::XOR { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_srl() {
        let word = encode_r(0x33, 5, 5, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRL { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_or() {
        let word = encode_r(0x33, 5, 6, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::OR { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_and() {
        let word = encode_r(0x33, 5, 7, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AND { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    // -----------------------------------------------------------------------
    // R-type (OP, funct7=0x20) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sub() {
        let word = encode_r(0x33, 5, 0, 10, 11, 0x20);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SUB { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_sra() {
        let word = encode_r(0x33, 5, 5, 10, 11, 0x20);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRA { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    // -----------------------------------------------------------------------
    // M extension (OP, funct7=0x01) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul() {
        let word = encode_r(0x33, 5, 0, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::MUL { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_mulh() {
        let word = encode_r(0x33, 5, 1, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::MULH { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_mulhsu() {
        let word = encode_r(0x33, 5, 2, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::MULHSU { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_mulhu() {
        let word = encode_r(0x33, 5, 3, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::MULHU { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_div() {
        let word = encode_r(0x33, 5, 4, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::DIV { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_divu() {
        let word = encode_r(0x33, 5, 5, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::DIVU { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_rem() {
        let word = encode_r(0x33, 5, 6, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::REM { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_remu() {
        let word = encode_r(0x33, 5, 7, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::REMU { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    // -----------------------------------------------------------------------
    // OP-IMM-32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_addiw() {
        let word = encode_i(0x1B, 5, 0, 10, 42);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDIW { rd: 5, rs1: 10, imm: 42 }
        );
    }

    #[test]
    fn test_addiw_negative() {
        let word = encode_i(0x1B, 5, 0, 10, -100);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDIW { rd: 5, rs1: 10, imm: -100 }
        );
    }

    #[test]
    fn test_slliw() {
        // SLLIW: funct3=1, shamt=5 (5-bit)
        let word = encode_i(0x1B, 5, 1, 10, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLLIW { rd: 5, rs1: 10, imm: 5 }
        );
    }

    #[test]
    fn test_slliw_max_shamt() {
        // SLLIW max shamt = 31
        let word = encode_i(0x1B, 5, 1, 10, 31);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLLIW { rd: 5, rs1: 10, imm: 31 }
        );
    }

    #[test]
    fn test_srliw() {
        // SRLIW: funct3=5, bit30=0
        let word = encode_i(0x1B, 5, 5, 10, 7);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRLIW { rd: 5, rs1: 10, imm: 7 }
        );
    }

    #[test]
    fn test_sraiw() {
        // SRAIW: funct3=5, bit30=1, shamt=7
        let imm = (1 << 10) | 7; // bit 10 sets bit 30
        let word = encode_i(0x1B, 5, 5, 10, imm);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRAIW { rd: 5, rs1: 10, imm: 7 }
        );
    }

    // -----------------------------------------------------------------------
    // OP-32 R-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_addw() {
        let word = encode_r(0x3B, 5, 0, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_subw() {
        let word = encode_r(0x3B, 5, 0, 10, 11, 0x20);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SUBW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_sllw() {
        let word = encode_r(0x3B, 5, 1, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SLLW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_srlw() {
        let word = encode_r(0x3B, 5, 5, 10, 11, 0x00);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRLW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_sraw() {
        let word = encode_r(0x3B, 5, 5, 10, 11, 0x20);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SRAW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    // -----------------------------------------------------------------------
    // OP-32 M extension tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mulw() {
        let word = encode_r(0x3B, 5, 0, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::MULW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_divw() {
        let word = encode_r(0x3B, 5, 4, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::DIVW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_divuw() {
        let word = encode_r(0x3B, 5, 5, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::DIVUW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_remw() {
        let word = encode_r(0x3B, 5, 6, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::REMW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_remuw() {
        let word = encode_r(0x3B, 5, 7, 10, 11, 0x01);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::REMUW { rd: 5, rs1: 10, rs2: 11 }
        );
    }

    // -----------------------------------------------------------------------
    // FENCE test
    // -----------------------------------------------------------------------

    #[test]
    fn test_fence() {
        // FENCE: opcode=0x0F, pred=0b1111, succ=0b0011
        // bits: imm[11:0] = 0b0000_1111_0011 but we don't use encode_i;
        // we construct manually to place pred/succ correctly.
        let pred: u32 = 0xF;
        let succ: u32 = 0x3;
        let word = (pred << 24) | (succ << 20) | 0x0F;
        let inst = decode(word).unwrap();
        assert_eq!(inst, Instruction::FENCE { pred: 0xF, succ: 0x3 });
    }

    #[test]
    fn test_fence_all() {
        let pred: u32 = 0xF;
        let succ: u32 = 0xF;
        let word = (pred << 24) | (succ << 20) | 0x0F;
        let inst = decode(word).unwrap();
        assert_eq!(inst, Instruction::FENCE { pred: 0xF, succ: 0xF });
    }

    // -----------------------------------------------------------------------
    // SYSTEM tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ecall() {
        // ECALL: opcode=0x73, imm=0
        let word = encode_i(0x73, 0, 0, 0, 0);
        assert_eq!(decode(word).unwrap(), Instruction::ECALL);
    }

    #[test]
    fn test_ebreak() {
        // EBREAK: opcode=0x73, imm=1
        let word = encode_i(0x73, 0, 0, 0, 1);
        assert_eq!(decode(word).unwrap(), Instruction::EBREAK);
    }

    // -----------------------------------------------------------------------
    // Error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unknown_opcode() {
        // Opcode 0x02 is not a valid RV64IM opcode
        let word: u32 = 0x02;
        let err = decode(word).unwrap_err();
        assert_eq!(err, DecodeError::UnknownOpcode(0x02));
    }

    #[test]
    fn test_unknown_funct3_branch() {
        // Branch with funct3=2 (invalid)
        let word = encode_b(0x63, 2, 1, 2, 8);
        let err = decode(word).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownFunct3 { opcode: 0x63, funct3: 2 }
        );
    }

    #[test]
    fn test_unknown_funct3_branch_3() {
        let word = encode_b(0x63, 3, 1, 2, 8);
        let err = decode(word).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownFunct3 { opcode: 0x63, funct3: 3 }
        );
    }

    #[test]
    fn test_unknown_funct7_op() {
        // OP with funct7=0x7F (invalid)
        let word = encode_r(0x33, 5, 0, 10, 11, 0x7F);
        let err = decode(word).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownFunct7 { opcode: 0x33, funct3: 0, funct7: 0x7F }
        );
    }

    #[test]
    fn test_unknown_funct7_op32() {
        let word = encode_r(0x3B, 5, 0, 10, 11, 0x7F);
        let err = decode(word).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownFunct7 { opcode: 0x3B, funct3: 0, funct7: 0x7F }
        );
    }

    #[test]
    fn test_unknown_funct3_load() {
        // Load with funct3=7 (invalid)
        let word = encode_i(0x03, 5, 7, 10, 0);
        let err = decode(word).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownFunct3 { opcode: 0x03, funct3: 7 }
        );
    }

    #[test]
    fn test_unknown_funct3_store() {
        // Store with funct3=4 (invalid)
        let word = encode_s(0x23, 4, 10, 5, 0);
        let err = decode(word).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownFunct3 { opcode: 0x23, funct3: 4 }
        );
    }

    #[test]
    fn test_unknown_funct3_jalr() {
        // JALR with funct3=1 (invalid)
        let word = encode_i(0x67, 1, 1, 5, 0);
        let err = decode(word).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnknownFunct3 { opcode: 0x67, funct3: 1 }
        );
    }

    // -----------------------------------------------------------------------
    // Sign extension roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_i_type_sign_extension_max_positive() {
        // I-type max positive immediate: 2047 (0x7FF)
        let word = encode_i(0x13, 5, 0, 10, 2047);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDI { rd: 5, rs1: 10, imm: 2047 }
        );
    }

    #[test]
    fn test_i_type_sign_extension_min_negative() {
        // I-type min negative immediate: -2048 (0x800)
        let word = encode_i(0x13, 5, 0, 10, -2048);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDI { rd: 5, rs1: 10, imm: -2048 }
        );
    }

    #[test]
    fn test_s_type_sign_extension_positive() {
        let word = encode_s(0x23, 0, 10, 5, 2047);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SB { rs1: 10, rs2: 5, imm: 2047 }
        );
    }

    #[test]
    fn test_s_type_sign_extension_negative() {
        let word = encode_s(0x23, 0, 10, 5, -2048);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SB { rs1: 10, rs2: 5, imm: -2048 }
        );
    }

    #[test]
    fn test_b_type_sign_extension_max_positive() {
        // B-type max positive: 4094 (0xFFE) -- must be even
        let word = encode_b(0x63, 0, 1, 2, 4094);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::BEQ { rs1: 1, rs2: 2, imm: 4094 }
        );
    }

    #[test]
    fn test_b_type_sign_extension_min_negative() {
        // B-type min negative: -4096 (0xFFFFF000) -- must be even
        let word = encode_b(0x63, 0, 1, 2, -4096);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::BEQ { rs1: 1, rs2: 2, imm: -4096 }
        );
    }

    #[test]
    fn test_j_type_sign_extension_positive() {
        // J-type positive: 1048574 (0xFFFFE) -- must be even
        let word = encode_j(0x6F, 1, 1048574);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::JAL { rd: 1, imm: 1048574 }
        );
    }

    #[test]
    fn test_j_type_sign_extension_negative() {
        // J-type min negative: -1048576 (0xFFF00000)
        let word = encode_j(0x6F, 1, -1048576);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::JAL { rd: 1, imm: -1048576 }
        );
    }

    #[test]
    fn test_u_type_sign_extension_negative() {
        // U-type with bit 31 set -> negative in i32
        let word = encode_u(0x37, 5, 0x80000000_u32 as i32);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LUI { rd: 5, imm: 0x80000000_u32 as i32 }
        );
    }

    // -----------------------------------------------------------------------
    // Encoding roundtrip: known real instruction encodings
    // -----------------------------------------------------------------------

    #[test]
    fn test_known_addi_x10_x0_1() {
        // ADDI x10, x0, 1 => 0x00100513
        let word: u32 = 0x00100513;
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDI { rd: 10, rs1: 0, imm: 1 }
        );
    }

    #[test]
    fn test_known_add_x5_x6_x7() {
        // ADD x5, x6, x7 => 0x007302B3
        let word: u32 = 0x007302B3;
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADD { rd: 5, rs1: 6, rs2: 7 }
        );
    }

    #[test]
    fn test_known_lui() {
        // LUI x1, 0xDEADB => 0xDEADB0B7
        let word: u32 = 0xDEADB0B7;
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LUI { rd: 1, imm: 0xDEADB000_u32 as i32 }
        );
    }

    #[test]
    fn test_all_registers_r_type() {
        // Test that register encoding works for all 32 registers
        for r in 0..32u32 {
            let word = encode_r(0x33, r, 0, r, r, 0x00);
            let inst = decode(word).unwrap();
            assert_eq!(
                inst,
                Instruction::ADD {
                    rd: r as u8,
                    rs1: r as u8,
                    rs2: r as u8,
                }
            );
        }
    }

    #[test]
    fn test_nop() {
        // NOP is ADDI x0, x0, 0 => 0x00000013
        let word: u32 = 0x00000013;
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDI { rd: 0, rs1: 0, imm: 0 }
        );
    }

    #[test]
    fn test_mv() {
        // MV rd, rs is pseudo for ADDI rd, rs, 0
        // MV x5, x10 => ADDI x5, x10, 0
        let word = encode_i(0x13, 5, 0, 10, 0);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::ADDI { rd: 5, rs1: 10, imm: 0 }
        );
    }

    // -----------------------------------------------------------------------
    // CSR instruction tests
    // -----------------------------------------------------------------------

    fn encode_csr(rd: u32, funct3: u32, rs1: u32, csr: u32) -> u32 {
        (csr << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x73
    }

    #[test]
    fn test_csrrw() {
        let word = encode_csr(5, 1, 10, 0x300);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::CSRRW { rd: 5, rs1: 10, csr: 0x300 }
        );
    }

    #[test]
    fn test_csrrs() {
        let word = encode_csr(5, 2, 10, 0x341);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::CSRRS { rd: 5, rs1: 10, csr: 0x341 }
        );
    }

    #[test]
    fn test_csrrc() {
        let word = encode_csr(5, 3, 10, 0x342);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::CSRRC { rd: 5, rs1: 10, csr: 0x342 }
        );
    }

    #[test]
    fn test_csrrwi() {
        let word = encode_csr(5, 5, 3, 0x300);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::CSRRWI { rd: 5, uimm: 3, csr: 0x300 }
        );
    }

    #[test]
    fn test_csrrsi() {
        let word = encode_csr(5, 6, 7, 0x344);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::CSRRSI { rd: 5, uimm: 7, csr: 0x344 }
        );
    }

    #[test]
    fn test_csrrci() {
        let word = encode_csr(5, 7, 15, 0x300);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::CSRRCI { rd: 5, uimm: 15, csr: 0x300 }
        );
    }

    // -----------------------------------------------------------------------
    // Privileged instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mret() {
        // MRET: funct7=0x18, rs2=2, funct3=0, opcode=0x73
        let word = encode_r(0x73, 0, 0, 0, 2, 0x18);
        assert_eq!(decode(word).unwrap(), Instruction::MRET);
    }

    #[test]
    fn test_sret() {
        // SRET: funct7=0x08, rs2=2, funct3=0, opcode=0x73
        let word = encode_r(0x73, 0, 0, 0, 2, 0x08);
        assert_eq!(decode(word).unwrap(), Instruction::SRET);
    }

    #[test]
    fn test_wfi() {
        // WFI: funct7=0x08, rs2=5, funct3=0, opcode=0x73
        let word = encode_r(0x73, 0, 0, 0, 5, 0x08);
        assert_eq!(decode(word).unwrap(), Instruction::WFI);
    }

    #[test]
    fn test_sfence_vma() {
        // SFENCE.VMA: funct7=0x09, rs1=10, rs2=11, funct3=0, opcode=0x73
        let word = encode_r(0x73, 0, 0, 10, 11, 0x09);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SFENCE_VMA { rs1: 10, rs2: 11 }
        );
    }

    #[test]
    fn test_fence_i() {
        // FENCE.I: opcode=0x0F, funct3=1
        let word = (1u32 << 12) | 0x0F;
        assert_eq!(decode(word).unwrap(), Instruction::FENCE_I);
    }

    #[test]
    fn test_ecall_still_works() {
        let word = encode_i(0x73, 0, 0, 0, 0);
        assert_eq!(decode(word).unwrap(), Instruction::ECALL);
    }

    // -----------------------------------------------------------------------
    // AMO instruction tests
    // -----------------------------------------------------------------------

    fn encode_amo(funct5: u32, aq: bool, rl: bool, rs2: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        let aq_bit = if aq { 1u32 } else { 0 };
        let rl_bit = if rl { 1u32 } else { 0 };
        (funct5 << 27) | (aq_bit << 26) | (rl_bit << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x2F
    }

    #[test]
    fn test_lr_w() {
        let word = encode_amo(0x02, true, false, 0, 10, 2, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LR_W { rd: 5, rs1: 10, aq: true, rl: false }
        );
    }

    #[test]
    fn test_sc_w() {
        let word = encode_amo(0x03, false, true, 11, 10, 2, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SC_W { rd: 5, rs1: 10, rs2: 11, aq: false, rl: true }
        );
    }

    #[test]
    fn test_amoswap_d() {
        let word = encode_amo(0x01, true, true, 11, 10, 3, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOSWAP_D { rd: 5, rs1: 10, rs2: 11, aq: true, rl: true }
        );
    }

    #[test]
    fn test_amoadd_w() {
        let word = encode_amo(0x00, false, false, 11, 10, 2, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOADD_W { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_amoand_d() {
        let word = encode_amo(0x0C, false, false, 11, 10, 3, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOAND_D { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_amoor_w() {
        let word = encode_amo(0x08, false, false, 11, 10, 2, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOOR_W { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_amoxor_d() {
        let word = encode_amo(0x04, false, false, 11, 10, 3, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOXOR_D { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_amomax_w() {
        let word = encode_amo(0x14, false, false, 11, 10, 2, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOMAX_W { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_amomin_d() {
        let word = encode_amo(0x10, false, false, 11, 10, 3, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOMIN_D { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_amomaxu_w() {
        let word = encode_amo(0x1C, false, false, 11, 10, 2, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOMAXU_W { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_amominu_d() {
        let word = encode_amo(0x18, false, false, 11, 10, 3, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::AMOMINU_D { rd: 5, rs1: 10, rs2: 11, aq: false, rl: false }
        );
    }

    #[test]
    fn test_lr_d() {
        let word = encode_amo(0x02, false, false, 0, 10, 3, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::LR_D { rd: 5, rs1: 10, aq: false, rl: false }
        );
    }

    #[test]
    fn test_sc_d() {
        let word = encode_amo(0x03, true, true, 11, 10, 3, 5);
        assert_eq!(
            decode(word).unwrap(),
            Instruction::SC_D { rd: 5, rs1: 10, rs2: 11, aq: true, rl: true }
        );
    }

    // -----------------------------------------------------------------------
    // Compressed instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_c_addi4spn() {
        // C.ADDI4SPN x8, x2, 8
        // nzuimm = 8 -> bits: [5:4]=0b10, rest 0 -> half encoding
        // nzuimm[5:4|9:6|2|3]: nzuimm=8 means bit3=1
        // bit positions in half: bit5->bit6, bit3->bit5
        let half: u16 = 0b000_00000_100_00_000; // nzuimm=8: bit3=1 -> position 5
        let half = half | (0b000 << 2); // rd' = 0 -> x8
        let result = decode_compressed(half);
        // Easier: just encode nzuimm=8 properly
        // nzuimm[5:4] from bits[12:11], nzuimm[9:6] from bits[10:7], nzuimm[2] from bit[6], nzuimm[3] from bit[5]
        // For nzuimm=8 (bit 3 set): bit[5] = 1
        let half: u16 = (1 << 5) | (0b000 << 2) | 0b00; // rd'=x8, quadrant 0, funct3=000
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::ADDI { rd: 8, rs1: 2, imm: 8 });
    }

    #[test]
    fn test_c_nop() {
        // C.NOP: quadrant 01, funct3=000, rd=0, imm=0
        let half: u16 = 0b000_0_00000_00000_01;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::ADDI { rd: 0, rs1: 0, imm: 0 });
    }

    #[test]
    fn test_c_li() {
        // C.LI x10, 5: quadrant 01, funct3=010, rd=10, imm=5
        let half: u16 = (0b010 << 13) | (10 << 7) | (5 << 2) | 0b01;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::ADDI { rd: 10, rs1: 0, imm: 5 });
    }

    #[test]
    fn test_c_j() {
        // C.J offset=0: quadrant 01, funct3=101
        let half: u16 = (0b101 << 13) | 0b01;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::JAL { rd: 0, imm: 0 });
    }

    #[test]
    fn test_c_ebreak() {
        // C.EBREAK: quadrant 10, funct3=100, bit12=1, rd=0, rs2=0
        let half: u16 = (0b100 << 13) | (1 << 12) | 0b10;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::EBREAK);
    }

    #[test]
    fn test_c_slli() {
        // C.SLLI x10, 3: quadrant 10, funct3=000, rd=10, shamt=3
        let half: u16 = (0b000 << 13) | (10 << 7) | (3 << 2) | 0b10;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::SLLI { rd: 10, rs1: 10, imm: 3 });
    }

    #[test]
    fn test_c_addi4spn_zero_is_illegal() {
        // C.ADDI4SPN with nzuimm=0 is illegal
        let half: u16 = 0b000_00000_000_00_00;
        let result = decode_compressed(half);
        assert!(result.is_err());
    }

    #[test]
    fn test_c_mv() {
        // C.MV x1, x10: quadrant 10, funct3=100, bit12=0, rd=1, rs2=10
        let half: u16 = (0b100 << 13) | (1 << 7) | (10 << 2) | 0b10;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::ADD { rd: 1, rs1: 0, rs2: 10 });
    }

    #[test]
    fn test_c_add() {
        // C.ADD x1, x10: quadrant 10, funct3=100, bit12=1, rd=1, rs2=10
        let half: u16 = (0b100 << 13) | (1 << 12) | (1 << 7) | (10 << 2) | 0b10;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::ADD { rd: 1, rs1: 1, rs2: 10 });
    }

    #[test]
    fn test_c_jr() {
        // C.JR x10: quadrant 10, funct3=100, bit12=0, rs1=10, rs2=0
        let half: u16 = (0b100 << 13) | (10 << 7) | 0b10;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::JALR { rd: 0, rs1: 10, imm: 0 });
    }

    #[test]
    fn test_c_jalr() {
        // C.JALR x10: quadrant 10, funct3=100, bit12=1, rs1=10, rs2=0
        let half: u16 = (0b100 << 13) | (1 << 12) | (10 << 7) | 0b10;
        let result = decode_compressed(half).unwrap();
        assert_eq!(result, Instruction::JALR { rd: 1, rs1: 10, imm: 0 });
    }
}
