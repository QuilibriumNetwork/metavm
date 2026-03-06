//! SBF (Solana BPF) execution trace.
//!
//! SBF is register-based with 11 registers (r0-r10), 64-bit values.
//! Very similar to RISC-V in structure.

use metavm_core::vm_traits::VmTrace;
use metavm_zkp::field::CurveType;
use metavm_zkp::trace::TracePolynomials;
use sha3::{Sha3_256, Digest};

/// Number of data columns (excluding step).
pub const NUM_SBF_COLUMNS: usize = 39;

// Column index constants for SBF trace layout.
pub const COL_PC: usize = 0;
pub const COL_OPCODE: usize = 1;
pub const COL_DST_REG: usize = 2;
pub const COL_DST_VAL_BEFORE: usize = 3;
pub const COL_DST_VAL_AFTER: usize = 4;
pub const COL_SRC_REG: usize = 5;
pub const COL_SRC_VAL: usize = 6;
pub const COL_MEM_ADDR: usize = 7;
pub const COL_MEM_VAL: usize = 8;
pub const COL_MEM_SIZE: usize = 9;
pub const COL_NEXT_PC: usize = 10;
pub const COL_INSN_TYPE: usize = 11;
pub const COL_FUNCT: usize = 12;
pub const COL_IMMEDIATE: usize = 13;
pub const COL_AUX0: usize = 14;
pub const COL_AUX1: usize = 15;
pub const COL_AUX2: usize = 16;

// Selector columns: one-hot encoding of (insn_type, funct) groups
pub const COL_SEL_ALU_ADD: usize = 17;
pub const COL_SEL_ALU_SUB: usize = 18;
pub const COL_SEL_ALU_MOV: usize = 19;
pub const COL_SEL_ALU_MUL: usize = 20;
pub const COL_SEL_ALU_DIV: usize = 21;
pub const COL_SEL_ALU_MOD: usize = 22;
pub const COL_SEL_ALU_AND: usize = 23;
pub const COL_SEL_ALU_OR: usize = 24;
pub const COL_SEL_ALU_XOR: usize = 25;
pub const COL_SEL_ALU_LSH: usize = 26;
pub const COL_SEL_ALU_RSH: usize = 27;
pub const COL_SEL_ALU_ARSH: usize = 28;
pub const COL_SEL_ALU_OTHER: usize = 29;
pub const COL_SEL_LOAD: usize = 30;
pub const COL_SEL_STORE: usize = 31;
pub const COL_SEL_BRANCH_EQ: usize = 32;
pub const COL_SEL_BRANCH_NEQ: usize = 33;
pub const COL_SEL_BRANCH_LT: usize = 34;
pub const COL_SEL_BRANCH_GE: usize = 35;
pub const COL_SEL_BRANCH_OTHER: usize = 36;
pub const COL_SEL_CALL: usize = 37;
pub const COL_SEL_EXIT: usize = 38;

/// SBF instruction type selectors.
pub const INSN_ALU64: u8 = 0;
pub const INSN_ALU32: u8 = 1;
pub const INSN_LOAD: u8 = 2;
pub const INSN_STORE: u8 = 3;
pub const INSN_BRANCH: u8 = 4;
pub const INSN_CALL: u8 = 5;
pub const INSN_EXIT: u8 = 6;

/// ALU funct codes (shared by ALU64 and ALU32).
pub const FUNCT_ADD: u8 = 0;
pub const FUNCT_SUB: u8 = 1;
pub const FUNCT_MUL: u8 = 2;
pub const FUNCT_DIV: u8 = 3;
pub const FUNCT_MOD: u8 = 4;
pub const FUNCT_OR: u8 = 5;
pub const FUNCT_AND: u8 = 6;
pub const FUNCT_XOR: u8 = 7;
pub const FUNCT_LSH: u8 = 8;
pub const FUNCT_RSH: u8 = 9;
pub const FUNCT_ARSH: u8 = 10;
pub const FUNCT_NEG: u8 = 11;
pub const FUNCT_MOV: u8 = 12;

/// Branch funct codes.
pub const FUNCT_JA: u8 = 0;
pub const FUNCT_JEQ: u8 = 1;
pub const FUNCT_JNE: u8 = 2;
pub const FUNCT_JLT: u8 = 3;
pub const FUNCT_JLE: u8 = 4;
pub const FUNCT_JGT: u8 = 5;
pub const FUNCT_JGE: u8 = 6;
pub const FUNCT_JSLT: u8 = 7;
pub const FUNCT_JSLE: u8 = 8;
pub const FUNCT_JSGT: u8 = 9;
pub const FUNCT_JSGE: u8 = 10;
pub const FUNCT_JSET: u8 = 11;

/// Load funct codes.
pub const FUNCT_LDDW: u8 = 0;
pub const FUNCT_LDXB: u8 = 1;
pub const FUNCT_LDXH: u8 = 2;
pub const FUNCT_LDXW: u8 = 3;
pub const FUNCT_LDXDW: u8 = 4;

/// Store funct codes.
pub const FUNCT_STB: u8 = 0;
pub const FUNCT_STH: u8 = 1;
pub const FUNCT_STW: u8 = 2;
pub const FUNCT_STDW: u8 = 3;
pub const FUNCT_STXB: u8 = 4;
pub const FUNCT_STXH: u8 = 5;
pub const FUNCT_STXW: u8 = 6;
pub const FUNCT_STXDW: u8 = 7;

/// One row of the SBF execution trace.
#[derive(Clone, Debug)]
pub struct SbfTraceRow {
    pub step: u64,
    pub pc: u64,
    pub opcode: u8,
    pub dst_reg: u8,
    pub dst_val_before: u64,
    pub dst_val_after: u64,
    pub src_reg: u8,
    pub src_val: u64,
    pub mem_addr: u64,
    pub mem_val: u64,
    pub mem_size: u8,
    pub next_pc: u64,
    pub insn_type: u8,
    pub funct: u8,
    pub immediate: u64,
    pub aux0: u64,
    pub aux1: u64,
    pub aux2: u64,
}

/// Column-oriented SBF trace (40 columns: step + 39 data columns).
pub struct SbfTraceColumns {
    pub step: Vec<u64>,
    pub pc: Vec<u64>,
    pub opcode: Vec<u64>,
    pub dst_reg: Vec<u64>,
    pub dst_val_before: Vec<u64>,
    pub dst_val_after: Vec<u64>,
    pub src_reg: Vec<u64>,
    pub src_val: Vec<u64>,
    pub mem_addr: Vec<u64>,
    pub mem_val: Vec<u64>,
    pub mem_size: Vec<u64>,
    pub next_pc: Vec<u64>,
    pub insn_type: Vec<u64>,
    pub funct: Vec<u64>,
    pub immediate: Vec<u64>,
    pub aux0: Vec<u64>,
    pub aux1: Vec<u64>,
    pub aux2: Vec<u64>,
    pub sel_alu_add: Vec<u64>,
    pub sel_alu_sub: Vec<u64>,
    pub sel_alu_mov: Vec<u64>,
    pub sel_alu_mul: Vec<u64>,
    pub sel_alu_div: Vec<u64>,
    pub sel_alu_mod: Vec<u64>,
    pub sel_alu_and: Vec<u64>,
    pub sel_alu_or: Vec<u64>,
    pub sel_alu_xor: Vec<u64>,
    pub sel_alu_lsh: Vec<u64>,
    pub sel_alu_rsh: Vec<u64>,
    pub sel_alu_arsh: Vec<u64>,
    pub sel_alu_other: Vec<u64>,
    pub sel_load: Vec<u64>,
    pub sel_store: Vec<u64>,
    pub sel_branch_eq: Vec<u64>,
    pub sel_branch_neq: Vec<u64>,
    pub sel_branch_lt: Vec<u64>,
    pub sel_branch_ge: Vec<u64>,
    pub sel_branch_other: Vec<u64>,
    pub sel_call: Vec<u64>,
    pub sel_exit: Vec<u64>,
}

impl SbfTraceColumns {
    pub fn new() -> Self {
        SbfTraceColumns {
            step: Vec::new(),
            pc: Vec::new(),
            opcode: Vec::new(),
            dst_reg: Vec::new(),
            dst_val_before: Vec::new(),
            dst_val_after: Vec::new(),
            src_reg: Vec::new(),
            src_val: Vec::new(),
            mem_addr: Vec::new(),
            mem_val: Vec::new(),
            mem_size: Vec::new(),
            next_pc: Vec::new(),
            insn_type: Vec::new(),
            funct: Vec::new(),
            immediate: Vec::new(),
            aux0: Vec::new(),
            aux1: Vec::new(),
            aux2: Vec::new(),
            sel_alu_add: Vec::new(),
            sel_alu_sub: Vec::new(),
            sel_alu_mov: Vec::new(),
            sel_alu_mul: Vec::new(),
            sel_alu_div: Vec::new(),
            sel_alu_mod: Vec::new(),
            sel_alu_and: Vec::new(),
            sel_alu_or: Vec::new(),
            sel_alu_xor: Vec::new(),
            sel_alu_lsh: Vec::new(),
            sel_alu_rsh: Vec::new(),
            sel_alu_arsh: Vec::new(),
            sel_alu_other: Vec::new(),
            sel_load: Vec::new(),
            sel_store: Vec::new(),
            sel_branch_eq: Vec::new(),
            sel_branch_neq: Vec::new(),
            sel_branch_lt: Vec::new(),
            sel_branch_ge: Vec::new(),
            sel_branch_other: Vec::new(),
            sel_call: Vec::new(),
            sel_exit: Vec::new(),
        }
    }

    /// Extract rows `[start..end)` into a new trace with step renumbered from 0.
    pub fn slice_rows(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.step.len());
        assert!(start <= end, "slice_rows: start {} > end {}", start, end);
        let len = end - start;

        macro_rules! slice_vec {
            ($v:expr) => { $v[start..end].to_vec() };
        }

        SbfTraceColumns {
            step: (0..len as u64).collect(),
            pc: slice_vec!(self.pc),
            opcode: slice_vec!(self.opcode),
            dst_reg: slice_vec!(self.dst_reg),
            dst_val_before: slice_vec!(self.dst_val_before),
            dst_val_after: slice_vec!(self.dst_val_after),
            src_reg: slice_vec!(self.src_reg),
            src_val: slice_vec!(self.src_val),
            mem_addr: slice_vec!(self.mem_addr),
            mem_val: slice_vec!(self.mem_val),
            mem_size: slice_vec!(self.mem_size),
            next_pc: slice_vec!(self.next_pc),
            insn_type: slice_vec!(self.insn_type),
            funct: slice_vec!(self.funct),
            immediate: slice_vec!(self.immediate),
            aux0: slice_vec!(self.aux0),
            aux1: slice_vec!(self.aux1),
            aux2: slice_vec!(self.aux2),
            sel_alu_add: slice_vec!(self.sel_alu_add),
            sel_alu_sub: slice_vec!(self.sel_alu_sub),
            sel_alu_mov: slice_vec!(self.sel_alu_mov),
            sel_alu_mul: slice_vec!(self.sel_alu_mul),
            sel_alu_div: slice_vec!(self.sel_alu_div),
            sel_alu_mod: slice_vec!(self.sel_alu_mod),
            sel_alu_and: slice_vec!(self.sel_alu_and),
            sel_alu_or: slice_vec!(self.sel_alu_or),
            sel_alu_xor: slice_vec!(self.sel_alu_xor),
            sel_alu_lsh: slice_vec!(self.sel_alu_lsh),
            sel_alu_rsh: slice_vec!(self.sel_alu_rsh),
            sel_alu_arsh: slice_vec!(self.sel_alu_arsh),
            sel_alu_other: slice_vec!(self.sel_alu_other),
            sel_load: slice_vec!(self.sel_load),
            sel_store: slice_vec!(self.sel_store),
            sel_branch_eq: slice_vec!(self.sel_branch_eq),
            sel_branch_neq: slice_vec!(self.sel_branch_neq),
            sel_branch_lt: slice_vec!(self.sel_branch_lt),
            sel_branch_ge: slice_vec!(self.sel_branch_ge),
            sel_branch_other: slice_vec!(self.sel_branch_other),
            sel_call: slice_vec!(self.sel_call),
            sel_exit: slice_vec!(self.sel_exit),
        }
    }

    pub fn push_row(&mut self, row: &SbfTraceRow) {
        self.step.push(row.step);
        self.pc.push(row.pc);
        self.opcode.push(row.opcode as u64);
        self.dst_reg.push(row.dst_reg as u64);
        self.dst_val_before.push(row.dst_val_before);
        self.dst_val_after.push(row.dst_val_after);
        self.src_reg.push(row.src_reg as u64);
        self.src_val.push(row.src_val);
        self.mem_addr.push(row.mem_addr);
        self.mem_val.push(row.mem_val);
        self.mem_size.push(row.mem_size as u64);
        self.next_pc.push(row.next_pc);
        self.insn_type.push(row.insn_type as u64);
        self.funct.push(row.funct as u64);
        self.immediate.push(row.immediate);
        self.aux0.push(row.aux0);
        self.aux1.push(row.aux1);
        self.aux2.push(row.aux2);

        // Selector columns: one-hot encoding based on (insn_type, funct)
        let mut sel = [0u64; 22];
        match row.insn_type {
            INSN_ALU64 | INSN_ALU32 => match row.funct {
                FUNCT_ADD => sel[0] = 1,
                FUNCT_SUB => sel[1] = 1,
                FUNCT_MOV => sel[2] = 1,
                FUNCT_MUL => sel[3] = 1,
                FUNCT_DIV => sel[4] = 1,
                FUNCT_MOD => sel[5] = 1,
                FUNCT_AND => sel[6] = 1,
                FUNCT_OR => sel[7] = 1,
                FUNCT_XOR => sel[8] = 1,
                FUNCT_LSH => sel[9] = 1,
                FUNCT_RSH => sel[10] = 1,
                FUNCT_ARSH => sel[11] = 1,
                _ => sel[12] = 1, // NEG and any remaining
            },
            INSN_LOAD => sel[13] = 1,
            INSN_STORE => sel[14] = 1,
            INSN_BRANCH => match row.funct {
                FUNCT_JEQ => sel[15] = 1,
                FUNCT_JNE => sel[16] = 1,
                FUNCT_JLT | FUNCT_JSLT => sel[17] = 1,
                FUNCT_JGE | FUNCT_JSGE => sel[18] = 1,
                _ => sel[19] = 1, // JA, JGT, JLE, JSGT, JSLE, JSET
            },
            INSN_CALL => sel[20] = 1,
            _ => sel[21] = 1, // EXIT or unknown
        }
        self.sel_alu_add.push(sel[0]);
        self.sel_alu_sub.push(sel[1]);
        self.sel_alu_mov.push(sel[2]);
        self.sel_alu_mul.push(sel[3]);
        self.sel_alu_div.push(sel[4]);
        self.sel_alu_mod.push(sel[5]);
        self.sel_alu_and.push(sel[6]);
        self.sel_alu_or.push(sel[7]);
        self.sel_alu_xor.push(sel[8]);
        self.sel_alu_lsh.push(sel[9]);
        self.sel_alu_rsh.push(sel[10]);
        self.sel_alu_arsh.push(sel[11]);
        self.sel_alu_other.push(sel[12]);
        self.sel_load.push(sel[13]);
        self.sel_store.push(sel[14]);
        self.sel_branch_eq.push(sel[15]);
        self.sel_branch_neq.push(sel[16]);
        self.sel_branch_lt.push(sel[17]);
        self.sel_branch_ge.push(sel[18]);
        self.sel_branch_other.push(sel[19]);
        self.sel_call.push(sel[20]);
        self.sel_exit.push(sel[21]);
    }
}

impl VmTrace for SbfTraceColumns {
    fn num_steps(&self) -> usize {
        self.step.len()
    }

    fn num_columns(&self) -> usize {
        1 + NUM_SBF_COLUMNS // step + 39 data columns
    }

    fn column(&self, index: usize) -> &[u64] {
        match index {
            0 => &self.step,
            1 => &self.pc,
            2 => &self.opcode,
            3 => &self.dst_reg,
            4 => &self.dst_val_before,
            5 => &self.dst_val_after,
            6 => &self.src_reg,
            7 => &self.src_val,
            8 => &self.mem_addr,
            9 => &self.mem_val,
            10 => &self.mem_size,
            11 => &self.next_pc,
            12 => &self.insn_type,
            13 => &self.funct,
            14 => &self.immediate,
            15 => &self.aux0,
            16 => &self.aux1,
            17 => &self.aux2,
            18 => &self.sel_alu_add,
            19 => &self.sel_alu_sub,
            20 => &self.sel_alu_mov,
            21 => &self.sel_alu_mul,
            22 => &self.sel_alu_div,
            23 => &self.sel_alu_mod,
            24 => &self.sel_alu_and,
            25 => &self.sel_alu_or,
            26 => &self.sel_alu_xor,
            27 => &self.sel_alu_lsh,
            28 => &self.sel_alu_rsh,
            29 => &self.sel_alu_arsh,
            30 => &self.sel_alu_other,
            31 => &self.sel_load,
            32 => &self.sel_store,
            33 => &self.sel_branch_eq,
            34 => &self.sel_branch_neq,
            35 => &self.sel_branch_lt,
            36 => &self.sel_branch_ge,
            37 => &self.sel_branch_other,
            38 => &self.sel_call,
            39 => &self.sel_exit,
            _ => panic!("Column index {} out of range", index),
        }
    }

    fn columns(&self) -> Vec<&[u64]> {
        (0..self.num_columns()).map(|i| self.column(i)).collect()
    }

    fn column_names(&self) -> Vec<&'static str> {
        vec![
            "step", "pc", "opcode", "dst_reg", "dst_val_before", "dst_val_after",
            "src_reg", "src_val", "mem_addr", "mem_val", "mem_size",
            "next_pc", "insn_type", "funct", "immediate", "aux0", "aux1", "aux2",
            "sel_alu_add", "sel_alu_sub", "sel_alu_mov",
            "sel_alu_mul", "sel_alu_div", "sel_alu_mod",
            "sel_alu_and", "sel_alu_or", "sel_alu_xor",
            "sel_alu_lsh", "sel_alu_rsh", "sel_alu_arsh", "sel_alu_other",
            "sel_load", "sel_store",
            "sel_branch_eq", "sel_branch_neq", "sel_branch_lt", "sel_branch_ge",
            "sel_branch_other", "sel_call", "sel_exit",
        ]
    }
}

/// Classify a BPF opcode into (insn_type, funct) pair.
///
/// BPF opcodes use a class field (bits 0-2) and an operation field (bits 4-7).
pub fn classify_instruction(opcode: u8) -> (u8, u8) {
    let class = opcode & 0x07;
    let op = (opcode >> 4) & 0x0F;
    let source = (opcode >> 3) & 0x01; // 0 = immediate, 1 = register

    match class {
        // ALU64 (class 7) and ALU32 (class 4)
        0x07 | 0x04 => {
            let insn_type = if class == 0x07 { INSN_ALU64 } else { INSN_ALU32 };
            let funct = match op {
                0x0 => FUNCT_ADD,
                0x1 => FUNCT_SUB,
                0x2 => FUNCT_MUL,
                0x3 => FUNCT_DIV,
                0x4 => FUNCT_OR,
                0x5 => FUNCT_AND,
                0x6 => FUNCT_LSH,
                0x7 => FUNCT_RSH,
                0x8 => FUNCT_NEG,
                0x9 => FUNCT_MOD,
                0xA => FUNCT_XOR,
                0xB => FUNCT_MOV,
                0xC => FUNCT_ARSH,
                _ => 0xFF,
            };
            (insn_type, funct)
        }
        // Load (class 0, 1, 2, 3)
        0x00 => (INSN_LOAD, FUNCT_LDDW),   // LDDW (64-bit immediate load)
        0x01 => {
            // LDXB/LDXH/LDXW/LDXDW depending on size
            let funct = match (opcode >> 3) & 0x03 {
                0x1 => FUNCT_LDXB,
                0x2 => FUNCT_LDXH,
                0x4 => FUNCT_LDXW,
                0x6 => FUNCT_LDXDW,
                _ => {
                    match op {
                        0x1 => FUNCT_LDXB,
                        0x2 => FUNCT_LDXH,
                        0x4 => FUNCT_LDXW,
                        0x6 => FUNCT_LDXDW,
                        _ => 0xFF,
                    }
                }
            };
            (INSN_LOAD, funct)
        }
        // Store (class 2, 3)
        0x02 | 0x03 => {
            let base_funct = match op & 0x03 {
                0x0 => 0, // byte
                0x1 => 1, // half
                0x2 => 2, // word
                0x3 => 3, // double
                _ => 0xFF,
            };
            let funct = if source == 1 {
                base_funct + 4 // STX variants
            } else {
                base_funct // ST variants
            };
            (INSN_STORE, funct)
        }
        // Jump (class 5)
        0x05 => {
            let funct = match op {
                0x0 => FUNCT_JA,
                0x1 => FUNCT_JEQ,
                0x2 => FUNCT_JGT,
                0x3 => FUNCT_JGE,
                0x4 => FUNCT_JSET,
                0x5 => FUNCT_JNE,
                0x6 => FUNCT_JSGT,
                0x7 => FUNCT_JSGE,
                0xA => FUNCT_JLT,
                0xB => FUNCT_JLE,
                0xC => FUNCT_JSLT,
                0xD => FUNCT_JSLE,
                0x8 => {
                    // CALL
                    return (INSN_CALL, 0);
                }
                0x9 => {
                    // EXIT
                    return (INSN_EXIT, 0);
                }
                _ => 0xFF,
            };
            (INSN_BRANCH, funct)
        }
        _ => (INSN_EXIT, 0xFF),
    }
}

/// Compute auxiliary values for carry/borrow in 64-bit ALU operations,
/// condition indicators for branch instructions, and shift witnesses.
///
/// For ALU ops:
/// - ADD: aux0 = carry (overflow bit)
/// - SUB: aux0 = borrow (1 if dst < src)
/// - MUL: aux0 = high 64 bits of 128-bit product
/// - DIV: aux0 = remainder
/// - MOD: aux0 = quotient
/// - LSH: aux0 = 2^k, aux1 = overflow = ((dst * 2^k) >> 64)
/// - RSH: aux0 = 2^k, aux1 = remainder = dst mod 2^k
/// - ARSH: aux0 = 2^k, aux1 = remainder = dst mod 2^k, aux2 = sign_bit * (2^k - 1)
///
/// For branch ops:
/// - JEQ/JNE: aux1 = 1 if dst != src, 0 if equal
/// - JGT/JGE/JLT/JLE: aux1 = borrow = 1 if dst < src (unsigned)
/// - JSGT/JSGE/JSLT/JSLE: aux1 = 1 if (dst as i64) < (src as i64)
pub fn compute_sbf_aux(opcode: u8, src_val: u64, dst_val_before: u64, _dst_val_after: u64) -> (u64, u64, u64) {
    let class = opcode & 0x07;
    let op = (opcode >> 4) & 0x0F;

    // Handle branches (class 0x05, excluding CALL=0x8 and EXIT=0x9)
    if class == 0x05 {
        match op {
            0x1 | 0x5 => {
                // JEQ/JNE: aux1 = 1 if dst != src, 0 if equal
                let neq = if dst_val_before != src_val { 1u64 } else { 0u64 };
                return (0, neq, 0);
            }
            0x2 | 0x3 | 0xA | 0xB => {
                // JGT/JGE/JLT/JLE (unsigned): aux1 = borrow = 1 if dst < src
                let borrow = if dst_val_before < src_val { 1u64 } else { 0u64 };
                return (0, borrow, 0);
            }
            0x6 | 0x7 | 0xC | 0xD => {
                // JSGT/JSGE/JSLT/JSLE (signed): aux1 = 1 if (dst as i64) < (src as i64)
                let signed_lt = if (dst_val_before as i64) < (src_val as i64) { 1u64 } else { 0u64 };
                return (0, signed_lt, 0);
            }
            _ => return (0, 0, 0), // JA, CALL, EXIT, JSET
        }
    }

    // Only compute ALU aux for ALU64/ALU32
    if class != 0x07 && class != 0x04 {
        return (0, 0, 0);
    }

    match op {
        0x0 => {
            // ADD: carry = (dst_val_before + src_val) overflows
            let (_, overflow) = dst_val_before.overflowing_add(src_val);
            (overflow as u64, 0, 0)
        }
        0x1 => {
            // SUB: borrow = (dst_val_before < src_val)
            let borrow = if dst_val_before < src_val { 1u64 } else { 0u64 };
            (borrow, 0, 0)
        }
        0x2 => {
            // MUL: aux0 = high 64 bits of 128-bit product
            let product = (dst_val_before as u128) * (src_val as u128);
            let hi = (product >> 64) as u64;
            (hi, 0, 0)
        }
        0x3 => {
            // DIV: remainder
            if src_val == 0 {
                (0, 0, 0)
            } else {
                let remainder = dst_val_before % src_val;
                (remainder, 0, 0)
            }
        }
        0x4 | 0x5 | 0xA => {
            // OR (0x4), AND (0x5), XOR (0xA): aux0 = AND(dst_val_before, src_val)
            let and_val = dst_val_before & src_val;
            (and_val, 0, 0)
        }
        0x6 => {
            // LSH: shift left by src_val % 64
            let k = (src_val % 64) as u32;
            let power = 1u64.wrapping_shl(k);
            let full = (dst_val_before as u128) * (power as u128);
            let overflow = (full >> 64) as u64;
            (power, overflow, 0)
        }
        0x7 => {
            // RSH: shift right by src_val % 64
            let k = (src_val % 64) as u32;
            let power = 1u64.wrapping_shl(k);
            let remainder = if k == 0 { 0 } else { dst_val_before % power };
            (power, remainder, 0)
        }
        0x9 => {
            // MOD: quotient
            if src_val == 0 {
                (0, 0, 0)
            } else {
                let quotient = dst_val_before / src_val;
                (quotient, 0, 0)
            }
        }
        0x8 => {
            // NEG: two's complement negation. aux0 = carry (1 if dst_before != 0)
            let carry = if dst_val_before != 0 { 1u64 } else { 0u64 };
            (carry, 0, 0)
        }
        0xC => {
            // ARSH: arithmetic shift right by src_val % 64
            let k = (src_val % 64) as u32;
            let power = 1u64.wrapping_shl(k);
            let remainder = if k == 0 { 0 } else { dst_val_before % power };
            let sign_bit = (dst_val_before >> 63) & 1;
            let sign_correction = sign_bit.wrapping_mul(power.wrapping_sub(1));
            (power, remainder, sign_correction)
        }
        _ => (0, 0, 0),
    }
}

/// Compute a SHA3-256 hash of the SBF execution state at a given row.
/// Used to chain chunk boundaries (initial/final state hashes).
pub fn sbf_state_hash(trace: &SbfTraceColumns, row: usize) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(trace.step[row].to_le_bytes());
    hasher.update(trace.pc[row].to_le_bytes());
    hasher.update(trace.opcode[row].to_le_bytes());
    hasher.update(trace.dst_reg[row].to_le_bytes());
    hasher.update(trace.dst_val_after[row].to_le_bytes());
    hasher.finalize().into()
}

/// Hash for the initial state (before any execution).
pub fn sbf_initial_state_hash() -> [u8; 32] {
    let hasher = Sha3_256::new();
    hasher.finalize().into()
}

/// Hash for the final state after all execution.
pub fn sbf_final_state_hash(trace: &SbfTraceColumns) -> [u8; 32] {
    if trace.step.is_empty() {
        return sbf_initial_state_hash();
    }
    let last = trace.step.len() - 1;
    let mut hasher = Sha3_256::new();
    hasher.update(b"final");
    hasher.update(trace.step[last].to_le_bytes());
    hasher.update(trace.pc[last].to_le_bytes());
    hasher.update(trace.opcode[last].to_le_bytes());
    hasher.update(trace.dst_reg[last].to_le_bytes());
    hasher.update(trace.dst_val_after[last].to_le_bytes());
    hasher.finalize().into()
}

/// Convenience wrapper: build TracePolynomials from an SBF trace with the given curve.
pub fn sbf_trace_polys_with_curve(trace: &SbfTraceColumns, curve: CurveType) -> TracePolynomials {
    TracePolynomials::from_vm_trace(trace, curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_alu64_add() {
        // ALU64 ADD imm: class=7, op=0, src=0 => opcode 0x07
        assert_eq!(classify_instruction(0x07), (INSN_ALU64, FUNCT_ADD));
    }

    #[test]
    fn test_classify_alu64_sub_reg() {
        // ALU64 SUB reg: class=7, op=1, src=1 => opcode 0x1F
        assert_eq!(classify_instruction(0x1F), (INSN_ALU64, FUNCT_SUB));
    }

    #[test]
    fn test_classify_exit() {
        // EXIT: class=5, op=9 => opcode 0x95
        assert_eq!(classify_instruction(0x95), (INSN_EXIT, 0));
    }

    #[test]
    fn test_classify_call() {
        // CALL: class=5, op=8 => opcode 0x85
        assert_eq!(classify_instruction(0x85), (INSN_CALL, 0));
    }

    #[test]
    fn test_vm_trace_columns() {
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

        assert_eq!(trace.num_steps(), 1);
        assert_eq!(trace.num_columns(), 40);
        assert_eq!(trace.column(1)[0], 0);  // pc
    }

    #[test]
    fn test_compute_aux_add_no_overflow() {
        let (aux0, aux1, _) = compute_sbf_aux(0x07, 5, 10, 15);
        assert_eq!(aux0, 0);
        assert_eq!(aux1, 0);
    }

    #[test]
    fn test_compute_aux_add_overflow() {
        let (aux0, _, _) = compute_sbf_aux(0x07, 1, u64::MAX, 0);
        assert_eq!(aux0, 1);
    }

    #[test]
    fn test_compute_aux_div() {
        // DIV: 17 / 5 = 3 remainder 2
        let (aux0, _, _) = compute_sbf_aux(0x37, 5, 17, 3);
        assert_eq!(aux0, 2); // remainder
    }

    #[test]
    fn test_compute_aux_lsh() {
        // ALU64 LSH imm: class=7, op=6 => opcode 0x67
        let (aux0, aux1, aux2) = compute_sbf_aux(0x67, 4, 0x0F, 0xF0);
        assert_eq!(aux0, 16); // 2^4
        assert_eq!(aux1, 0);  // no overflow
        assert_eq!(aux2, 0);
    }

    #[test]
    fn test_compute_aux_lsh_overflow() {
        // LSH with overflow: 0xFFFF_FFFF_FFFF_FFFF << 32
        let (aux0, aux1, _) = compute_sbf_aux(0x67, 32, 0xFFFF_FFFF_FFFF_FFFF, 0);
        assert_eq!(aux0, 1u64 << 32);
        assert_eq!(aux1, 0xFFFF_FFFF); // upper 32 bits overflow
    }

    #[test]
    fn test_compute_aux_rsh() {
        // ALU64 RSH imm: class=7, op=7 => opcode 0x77
        let (aux0, aux1, aux2) = compute_sbf_aux(0x77, 4, 0xFF, 0x0F);
        assert_eq!(aux0, 16);  // 2^4
        assert_eq!(aux1, 15);  // remainder = 0xFF % 16 = 15
        assert_eq!(aux2, 0);
    }

    #[test]
    fn test_compute_aux_arsh_positive() {
        // ALU64 ARSH imm: class=7, op=0xC => opcode 0xC7
        let (aux0, aux1, aux2) = compute_sbf_aux(0xC7, 4, 0x7F00, 0);
        assert_eq!(aux0, 16); // 2^4
        assert_eq!(aux1, 0);  // remainder = 0x7F00 % 16 = 0
        assert_eq!(aux2, 0);  // positive, no sign correction
    }

    #[test]
    fn test_compute_aux_arsh_negative() {
        // ARSH on negative: 0xFFFF_FFFF_FFFF_FFF0 >> 4
        let dst_before = 0xFFFF_FFFF_FFFF_FFF0u64;
        let (aux0, aux1, aux2) = compute_sbf_aux(0xC7, 4, dst_before, 0);
        assert_eq!(aux0, 16); // 2^4
        assert_eq!(aux1, 0);  // remainder = 0xFFF0 % 16 = 0
        assert_eq!(aux2, 15); // sign_bit=1, correction = 1*(16-1) = 15
    }
}
