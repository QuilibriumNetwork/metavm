//! EVM execution trace with 256-bit limb decomposition.
//!
//! Each 256-bit EVM value is decomposed into 4 x u64 limbs (little-endian:
//! limb0 = least significant 64 bits).

use metavm_core::vm_traits::VmTrace;
use metavm_zkp::field::CurveType;
use metavm_zkp::trace::TracePolynomials;
use sha3::{Sha3_256, Digest};

/// Number of data columns (excluding step).
pub const NUM_EVM_COLUMNS: usize = 77;

// Column index constants for the EVM trace layout.
// These index into the data columns (step is column 0 in VmTrace but excluded
// from TracePolynomials).
pub const COL_PC: usize = 0;
pub const COL_OPCODE: usize = 1;
pub const COL_GAS_REMAINING: usize = 2;
pub const COL_STACK_DEPTH: usize = 3;
// input0: U256 as 4 limbs
pub const COL_INPUT0_L0: usize = 4;
pub const COL_INPUT0_L1: usize = 5;
pub const COL_INPUT0_L2: usize = 6;
pub const COL_INPUT0_L3: usize = 7;
// input1: U256 as 4 limbs
pub const COL_INPUT1_L0: usize = 8;
pub const COL_INPUT1_L1: usize = 9;
pub const COL_INPUT1_L2: usize = 10;
pub const COL_INPUT1_L3: usize = 11;
// output0: U256 as 4 limbs
pub const COL_OUTPUT0_L0: usize = 12;
pub const COL_OUTPUT0_L1: usize = 13;
pub const COL_OUTPUT0_L2: usize = 14;
pub const COL_OUTPUT0_L3: usize = 15;
// memory
pub const COL_MEM_OFFSET: usize = 16;
pub const COL_MEM_VALUE_L0: usize = 17;
pub const COL_MEM_VALUE_L1: usize = 18;
pub const COL_MEM_VALUE_L2: usize = 19;
pub const COL_MEM_VALUE_L3: usize = 20;
// selectors
pub const COL_INSN_TYPE: usize = 21;
pub const COL_FUNCT: usize = 22;
// immediate (PUSH value): U256 as 4 limbs
pub const COL_IMMEDIATE_L0: usize = 23;
pub const COL_IMMEDIATE_L1: usize = 24;
pub const COL_IMMEDIATE_L2: usize = 25;
pub const COL_IMMEDIATE_L3: usize = 26;
// aux0: U256 as 4 limbs (overflow, carry, intermediate)
pub const COL_AUX0_L0: usize = 27;
pub const COL_AUX0_L1: usize = 28;
pub const COL_AUX0_L2: usize = 29;
pub const COL_AUX0_L3: usize = 30;
// aux1: U256 as 4 limbs (borrow, remainder)
pub const COL_AUX1_L0: usize = 31;
pub const COL_AUX1_L1: usize = 32;
pub const COL_AUX1_L2: usize = 33;
pub const COL_AUX1_L3: usize = 34;
// Selector columns: one-hot encoding of (insn_type, funct) groups (41 selectors)
pub const COL_SEL_STOP: usize = 35;
pub const COL_SEL_ARITH_ADD: usize = 36;
pub const COL_SEL_ARITH_SUB: usize = 37;
pub const COL_SEL_ARITH_MUL: usize = 38;
pub const COL_SEL_ARITH_DIV: usize = 39;
pub const COL_SEL_MOD: usize = 40;
pub const COL_SEL_SDIV: usize = 41;
pub const COL_SEL_SMOD: usize = 42;
pub const COL_SEL_ADDMOD: usize = 43;
pub const COL_SEL_MULMOD: usize = 44;
pub const COL_SEL_EXP: usize = 45;
pub const COL_SEL_SIGNEXTEND: usize = 46;
pub const COL_SEL_LT: usize = 47;
pub const COL_SEL_GT: usize = 48;
pub const COL_SEL_EQ: usize = 49;
pub const COL_SEL_ISZERO: usize = 50;
pub const COL_SEL_COMPARE_OTHER: usize = 51;
pub const COL_SEL_AND: usize = 52;
pub const COL_SEL_OR: usize = 53;
pub const COL_SEL_XOR: usize = 54;
pub const COL_SEL_BITWISE_OTHER: usize = 55;
pub const COL_SEL_SHL: usize = 56;
pub const COL_SEL_SHR: usize = 57;
pub const COL_SEL_SAR: usize = 58;
pub const COL_SEL_KECCAK: usize = 59;
pub const COL_SEL_ENV: usize = 60;
pub const COL_SEL_BLOCK: usize = 61;
pub const COL_SEL_PUSH: usize = 62;
pub const COL_SEL_DUP: usize = 63;
pub const COL_SEL_POP: usize = 64;
pub const COL_SEL_SWAP: usize = 65;
pub const COL_SEL_STACK_OTHER: usize = 66;
pub const COL_SEL_MLOAD: usize = 67;
pub const COL_SEL_MSTORE: usize = 68;
pub const COL_SEL_MSTORE8: usize = 69;
pub const COL_SEL_MSIZE: usize = 70;
pub const COL_SEL_MEMORY_OTHER: usize = 71;
pub const COL_SEL_STORAGE: usize = 72;
pub const COL_SEL_JUMP: usize = 73;
pub const COL_SEL_LOG: usize = 74;
pub const COL_SEL_CALL: usize = 75;
// Next PC for cross-row PC continuity constraint
pub const COL_NEXT_PC: usize = 76;

/// EVM instruction type selectors.
pub const INSN_STOP: u8 = 0;
pub const INSN_ARITH: u8 = 1;
pub const INSN_COMPARE: u8 = 2;
pub const INSN_BITWISE: u8 = 3;
pub const INSN_KECCAK: u8 = 4;
pub const INSN_ENV: u8 = 5;
pub const INSN_BLOCK: u8 = 6;
pub const INSN_STACK: u8 = 7;
pub const INSN_MEMORY: u8 = 8;
pub const INSN_STORAGE: u8 = 9;
pub const INSN_JUMP: u8 = 10;
pub const INSN_LOG: u8 = 11;
pub const INSN_CALL: u8 = 12;

/// Funct codes for arithmetic sub-variants.
pub const FUNCT_ADD: u8 = 0;
pub const FUNCT_MUL: u8 = 1;
pub const FUNCT_SUB: u8 = 2;
pub const FUNCT_DIV: u8 = 3;
pub const FUNCT_SDIV: u8 = 4;
pub const FUNCT_MOD: u8 = 5;
pub const FUNCT_SMOD: u8 = 6;
pub const FUNCT_ADDMOD: u8 = 7;
pub const FUNCT_MULMOD: u8 = 8;
pub const FUNCT_EXP: u8 = 9;
pub const FUNCT_SIGNEXTEND: u8 = 10;

/// Funct codes for comparison sub-variants.
pub const FUNCT_LT: u8 = 0;
pub const FUNCT_GT: u8 = 1;
pub const FUNCT_SLT: u8 = 2;
pub const FUNCT_SGT: u8 = 3;
pub const FUNCT_EQ: u8 = 4;
pub const FUNCT_ISZERO: u8 = 5;

/// Funct codes for bitwise sub-variants.
pub const FUNCT_AND: u8 = 0;
pub const FUNCT_OR: u8 = 1;
pub const FUNCT_XOR: u8 = 2;
pub const FUNCT_NOT: u8 = 3;
pub const FUNCT_BYTE: u8 = 4;
pub const FUNCT_SHL: u8 = 5;
pub const FUNCT_SHR: u8 = 6;
pub const FUNCT_SAR: u8 = 7;

/// Funct codes for stack operations.
pub const FUNCT_POP: u8 = 0;
pub const FUNCT_PUSH: u8 = 1;
pub const FUNCT_DUP: u8 = 2;
pub const FUNCT_SWAP: u8 = 3;

/// Funct codes for memory operations.
pub const FUNCT_MLOAD: u8 = 0;
pub const FUNCT_MSTORE: u8 = 1;
pub const FUNCT_MSTORE8: u8 = 2;
pub const FUNCT_MSIZE: u8 = 3;

/// Funct codes for jump operations.
pub const FUNCT_JUMP: u8 = 0;
pub const FUNCT_JUMPI: u8 = 1;
pub const FUNCT_JUMPDEST: u8 = 2;
pub const FUNCT_PC: u8 = 3;

/// Decompose a U256 (represented as [u64; 4] in little-endian) into 4 u64 limbs.
pub fn u256_to_limbs(limbs: [u64; 4]) -> [u64; 4] {
    limbs // Already in the right format for revm's Uint<256, 4>
}

/// Classify an EVM opcode into (insn_type, funct) pair.
pub fn classify_opcode(opcode: u8) -> (u8, u8) {
    match opcode {
        0x00 => (INSN_STOP, 0),           // STOP
        0xFE => (INSN_STOP, 1),           // INVALID
        0xFD => (INSN_STOP, 2),           // REVERT
        0xFF => (INSN_STOP, 3),           // SELFDESTRUCT

        0x01 => (INSN_ARITH, FUNCT_ADD),
        0x02 => (INSN_ARITH, FUNCT_MUL),
        0x03 => (INSN_ARITH, FUNCT_SUB),
        0x04 => (INSN_ARITH, FUNCT_DIV),
        0x05 => (INSN_ARITH, FUNCT_SDIV),
        0x06 => (INSN_ARITH, FUNCT_MOD),
        0x07 => (INSN_ARITH, FUNCT_SMOD),
        0x08 => (INSN_ARITH, FUNCT_ADDMOD),
        0x09 => (INSN_ARITH, FUNCT_MULMOD),
        0x0A => (INSN_ARITH, FUNCT_EXP),
        0x0B => (INSN_ARITH, FUNCT_SIGNEXTEND),

        0x10 => (INSN_COMPARE, FUNCT_LT),
        0x11 => (INSN_COMPARE, FUNCT_GT),
        0x12 => (INSN_COMPARE, FUNCT_SLT),
        0x13 => (INSN_COMPARE, FUNCT_SGT),
        0x14 => (INSN_COMPARE, FUNCT_EQ),
        0x15 => (INSN_COMPARE, FUNCT_ISZERO),

        0x16 => (INSN_BITWISE, FUNCT_AND),
        0x17 => (INSN_BITWISE, FUNCT_OR),
        0x18 => (INSN_BITWISE, FUNCT_XOR),
        0x19 => (INSN_BITWISE, FUNCT_NOT),
        0x1A => (INSN_BITWISE, FUNCT_BYTE),
        0x1B => (INSN_BITWISE, FUNCT_SHL),
        0x1C => (INSN_BITWISE, FUNCT_SHR),
        0x1D => (INSN_BITWISE, FUNCT_SAR),

        0x20 => (INSN_KECCAK, 0),         // SHA3

        0x30..=0x3F => (INSN_ENV, opcode - 0x30),
        0x40..=0x48 => (INSN_BLOCK, opcode - 0x40),

        0x50 => (INSN_STACK, FUNCT_POP),
        0x60..=0x7F => (INSN_STACK, FUNCT_PUSH),  // PUSH1-PUSH32
        0x80..=0x8F => (INSN_STACK, FUNCT_DUP),   // DUP1-DUP16
        0x90..=0x9F => (INSN_STACK, FUNCT_SWAP),   // SWAP1-SWAP16

        0x51 => (INSN_MEMORY, FUNCT_MLOAD),
        0x52 => (INSN_MEMORY, FUNCT_MSTORE),
        0x53 => (INSN_MEMORY, FUNCT_MSTORE8),
        0x59 => (INSN_MEMORY, FUNCT_MSIZE),

        0x54 => (INSN_STORAGE, 0),         // SLOAD
        0x55 => (INSN_STORAGE, 1),         // SSTORE
        0x5C => (INSN_STORAGE, 2),         // TLOAD
        0x5D => (INSN_STORAGE, 3),         // TSTORE

        0x56 => (INSN_JUMP, FUNCT_JUMP),
        0x57 => (INSN_JUMP, FUNCT_JUMPI),
        0x5B => (INSN_JUMP, FUNCT_JUMPDEST),
        0x58 => (INSN_JUMP, FUNCT_PC),

        0xA0..=0xA4 => (INSN_LOG, opcode - 0xA0),

        0xF0 | 0xF1 | 0xF2 | 0xF4 | 0xF5 | 0xF3 | 0xFA => (INSN_CALL, opcode - 0xF0),

        _ => (INSN_STOP, 0xFF),  // Unknown opcodes treated as stop
    }
}

/// One row of the EVM execution trace.
#[derive(Clone, Debug)]
pub struct EvmTraceRow {
    pub step: u64,
    pub pc: u64,
    pub opcode: u8,
    pub gas_remaining: u64,
    pub stack_depth: u64,
    pub input0: [u64; 4],
    pub input1: [u64; 4],
    pub output0: [u64; 4],
    pub mem_offset: u64,
    pub mem_value: [u64; 4],
    pub insn_type: u8,
    pub funct: u8,
    pub immediate: [u64; 4],
    pub aux0: [u64; 4],
    pub aux1: [u64; 4],
    /// Expected PC of the next instruction (for PC continuity constraint).
    /// Set by the executor/inspector based on opcode semantics.
    pub next_pc: u64,
}

/// Column-oriented EVM trace.
pub struct EvmTraceColumns {
    pub step: Vec<u64>,
    pub pc: Vec<u64>,
    pub opcode: Vec<u64>,
    pub gas_remaining: Vec<u64>,
    pub stack_depth: Vec<u64>,
    pub input0: [Vec<u64>; 4],
    pub input1: [Vec<u64>; 4],
    pub output0: [Vec<u64>; 4],
    pub mem_offset: Vec<u64>,
    pub mem_value: [Vec<u64>; 4],
    pub insn_type: Vec<u64>,
    pub funct: Vec<u64>,
    pub immediate: [Vec<u64>; 4],
    pub aux0: [Vec<u64>; 4],
    pub aux1: [Vec<u64>; 4],
    pub sel_stop: Vec<u64>,
    pub sel_arith_add: Vec<u64>,
    pub sel_arith_sub: Vec<u64>,
    pub sel_arith_mul: Vec<u64>,
    pub sel_arith_div: Vec<u64>,
    pub sel_mod: Vec<u64>,
    pub sel_sdiv: Vec<u64>,
    pub sel_smod: Vec<u64>,
    pub sel_addmod: Vec<u64>,
    pub sel_mulmod: Vec<u64>,
    pub sel_exp: Vec<u64>,
    pub sel_signextend: Vec<u64>,
    pub sel_lt: Vec<u64>,
    pub sel_gt: Vec<u64>,
    pub sel_eq: Vec<u64>,
    pub sel_iszero: Vec<u64>,
    pub sel_compare_other: Vec<u64>,
    pub sel_and: Vec<u64>,
    pub sel_or: Vec<u64>,
    pub sel_xor: Vec<u64>,
    pub sel_bitwise_other: Vec<u64>,
    pub sel_shl: Vec<u64>,
    pub sel_shr: Vec<u64>,
    pub sel_sar: Vec<u64>,
    pub sel_keccak: Vec<u64>,
    pub sel_env: Vec<u64>,
    pub sel_block: Vec<u64>,
    pub sel_push: Vec<u64>,
    pub sel_dup: Vec<u64>,
    pub sel_pop: Vec<u64>,
    pub sel_swap: Vec<u64>,
    pub sel_stack_other: Vec<u64>,
    pub sel_mload: Vec<u64>,
    pub sel_mstore: Vec<u64>,
    pub sel_mstore8: Vec<u64>,
    pub sel_msize: Vec<u64>,
    pub sel_memory_other: Vec<u64>,
    pub sel_storage: Vec<u64>,
    pub sel_jump: Vec<u64>,
    pub sel_log: Vec<u64>,
    pub sel_call: Vec<u64>,
    pub next_pc: Vec<u64>,
}

impl EvmTraceColumns {
    pub fn new() -> Self {
        EvmTraceColumns {
            step: Vec::new(),
            pc: Vec::new(),
            opcode: Vec::new(),
            gas_remaining: Vec::new(),
            stack_depth: Vec::new(),
            input0: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            input1: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            output0: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            mem_offset: Vec::new(),
            mem_value: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            insn_type: Vec::new(),
            funct: Vec::new(),
            immediate: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            aux0: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            aux1: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            sel_stop: Vec::new(),
            sel_arith_add: Vec::new(),
            sel_arith_sub: Vec::new(),
            sel_arith_mul: Vec::new(),
            sel_arith_div: Vec::new(),
            sel_mod: Vec::new(),
            sel_sdiv: Vec::new(),
            sel_smod: Vec::new(),
            sel_addmod: Vec::new(),
            sel_mulmod: Vec::new(),
            sel_exp: Vec::new(),
            sel_signextend: Vec::new(),
            sel_lt: Vec::new(),
            sel_gt: Vec::new(),
            sel_eq: Vec::new(),
            sel_iszero: Vec::new(),
            sel_compare_other: Vec::new(),
            sel_and: Vec::new(),
            sel_or: Vec::new(),
            sel_xor: Vec::new(),
            sel_bitwise_other: Vec::new(),
            sel_shl: Vec::new(),
            sel_shr: Vec::new(),
            sel_sar: Vec::new(),
            sel_keccak: Vec::new(),
            sel_env: Vec::new(),
            sel_block: Vec::new(),
            sel_push: Vec::new(),
            sel_dup: Vec::new(),
            sel_pop: Vec::new(),
            sel_swap: Vec::new(),
            sel_stack_other: Vec::new(),
            sel_mload: Vec::new(),
            sel_mstore: Vec::new(),
            sel_mstore8: Vec::new(),
            sel_msize: Vec::new(),
            sel_memory_other: Vec::new(),
            sel_storage: Vec::new(),
            sel_jump: Vec::new(),
            sel_log: Vec::new(),
            sel_call: Vec::new(),
            next_pc: Vec::new(),
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
        macro_rules! slice_arr4 {
            ($a:expr) => {
                [slice_vec!($a[0]), slice_vec!($a[1]), slice_vec!($a[2]), slice_vec!($a[3])]
            };
        }

        let mut result = EvmTraceColumns {
            step: (0..len as u64).collect(),
            pc: slice_vec!(self.pc),
            opcode: slice_vec!(self.opcode),
            gas_remaining: slice_vec!(self.gas_remaining),
            stack_depth: slice_vec!(self.stack_depth),
            input0: slice_arr4!(self.input0),
            input1: slice_arr4!(self.input1),
            output0: slice_arr4!(self.output0),
            mem_offset: slice_vec!(self.mem_offset),
            mem_value: slice_arr4!(self.mem_value),
            insn_type: slice_vec!(self.insn_type),
            funct: slice_vec!(self.funct),
            immediate: slice_arr4!(self.immediate),
            aux0: slice_arr4!(self.aux0),
            aux1: slice_arr4!(self.aux1),
            sel_stop: slice_vec!(self.sel_stop),
            sel_arith_add: slice_vec!(self.sel_arith_add),
            sel_arith_sub: slice_vec!(self.sel_arith_sub),
            sel_arith_mul: slice_vec!(self.sel_arith_mul),
            sel_arith_div: slice_vec!(self.sel_arith_div),
            sel_mod: slice_vec!(self.sel_mod),
            sel_sdiv: slice_vec!(self.sel_sdiv),
            sel_smod: slice_vec!(self.sel_smod),
            sel_addmod: slice_vec!(self.sel_addmod),
            sel_mulmod: slice_vec!(self.sel_mulmod),
            sel_exp: slice_vec!(self.sel_exp),
            sel_signextend: slice_vec!(self.sel_signextend),
            sel_lt: slice_vec!(self.sel_lt),
            sel_gt: slice_vec!(self.sel_gt),
            sel_eq: slice_vec!(self.sel_eq),
            sel_iszero: slice_vec!(self.sel_iszero),
            sel_compare_other: slice_vec!(self.sel_compare_other),
            sel_and: slice_vec!(self.sel_and),
            sel_or: slice_vec!(self.sel_or),
            sel_xor: slice_vec!(self.sel_xor),
            sel_bitwise_other: slice_vec!(self.sel_bitwise_other),
            sel_shl: slice_vec!(self.sel_shl),
            sel_shr: slice_vec!(self.sel_shr),
            sel_sar: slice_vec!(self.sel_sar),
            sel_keccak: slice_vec!(self.sel_keccak),
            sel_env: slice_vec!(self.sel_env),
            sel_block: slice_vec!(self.sel_block),
            sel_push: slice_vec!(self.sel_push),
            sel_dup: slice_vec!(self.sel_dup),
            sel_pop: slice_vec!(self.sel_pop),
            sel_swap: slice_vec!(self.sel_swap),
            sel_stack_other: slice_vec!(self.sel_stack_other),
            sel_mload: slice_vec!(self.sel_mload),
            sel_mstore: slice_vec!(self.sel_mstore),
            sel_mstore8: slice_vec!(self.sel_mstore8),
            sel_msize: slice_vec!(self.sel_msize),
            sel_memory_other: slice_vec!(self.sel_memory_other),
            sel_storage: slice_vec!(self.sel_storage),
            sel_jump: slice_vec!(self.sel_jump),
            sel_log: slice_vec!(self.sel_log),
            sel_call: slice_vec!(self.sel_call),
            next_pc: slice_vec!(self.next_pc),
        };
        let _ = &mut result; // suppress unused_mut
        result
    }

    pub fn push_row(&mut self, row: &EvmTraceRow) {
        self.step.push(row.step);
        self.pc.push(row.pc);
        self.opcode.push(row.opcode as u64);
        self.gas_remaining.push(row.gas_remaining);
        self.stack_depth.push(row.stack_depth);
        for j in 0..4 {
            self.input0[j].push(row.input0[j]);
            self.input1[j].push(row.input1[j]);
            self.output0[j].push(row.output0[j]);
            self.mem_value[j].push(row.mem_value[j]);
            self.immediate[j].push(row.immediate[j]);
            self.aux0[j].push(row.aux0[j]);
            self.aux1[j].push(row.aux1[j]);
        }
        self.mem_offset.push(row.mem_offset);
        self.insn_type.push(row.insn_type as u64);
        self.funct.push(row.funct as u64);

        // Set selector columns based on (insn_type, funct)
        let mut sel = [0u64; 41];
        match row.insn_type {
            INSN_STOP => sel[0] = 1,
            INSN_ARITH => match row.funct {
                FUNCT_ADD => sel[1] = 1,
                FUNCT_SUB => sel[2] = 1,
                FUNCT_MUL => sel[3] = 1,
                FUNCT_DIV => sel[4] = 1,
                FUNCT_MOD => sel[5] = 1,
                FUNCT_SDIV => sel[6] = 1,
                FUNCT_SMOD => sel[7] = 1,
                FUNCT_ADDMOD => sel[8] = 1,
                FUNCT_MULMOD => sel[9] = 1,
                FUNCT_EXP => sel[10] = 1,
                FUNCT_SIGNEXTEND => sel[11] = 1,
                _ => sel[0] = 1,
            },
            INSN_COMPARE => match row.funct {
                FUNCT_LT => sel[12] = 1,
                FUNCT_GT => sel[13] = 1,
                FUNCT_EQ => sel[14] = 1,
                FUNCT_ISZERO => sel[15] = 1,
                _ => sel[16] = 1,
            },
            INSN_BITWISE => match row.funct {
                FUNCT_AND => sel[17] = 1,
                FUNCT_OR => sel[18] = 1,
                FUNCT_XOR => sel[19] = 1,
                FUNCT_SHL => sel[21] = 1,
                FUNCT_SHR => sel[22] = 1,
                FUNCT_SAR => sel[23] = 1,
                _ => sel[20] = 1,
            },
            INSN_KECCAK => sel[24] = 1,
            INSN_ENV => sel[25] = 1,
            INSN_BLOCK => sel[26] = 1,
            INSN_STACK => match row.funct {
                FUNCT_PUSH => sel[27] = 1,
                FUNCT_DUP => sel[28] = 1,
                FUNCT_POP => sel[29] = 1,
                FUNCT_SWAP => sel[30] = 1,
                _ => sel[31] = 1,
            },
            INSN_MEMORY => match row.funct {
                FUNCT_MLOAD => sel[32] = 1,
                FUNCT_MSTORE => sel[33] = 1,
                FUNCT_MSTORE8 => sel[34] = 1,
                FUNCT_MSIZE => sel[35] = 1,
                _ => sel[36] = 1,
            },
            INSN_STORAGE => sel[37] = 1,
            INSN_JUMP => sel[38] = 1,
            INSN_LOG => sel[39] = 1,
            INSN_CALL => sel[40] = 1,
            _ => sel[0] = 1,
        }
        self.sel_stop.push(sel[0]);
        self.sel_arith_add.push(sel[1]);
        self.sel_arith_sub.push(sel[2]);
        self.sel_arith_mul.push(sel[3]);
        self.sel_arith_div.push(sel[4]);
        self.sel_mod.push(sel[5]);
        self.sel_sdiv.push(sel[6]);
        self.sel_smod.push(sel[7]);
        self.sel_addmod.push(sel[8]);
        self.sel_mulmod.push(sel[9]);
        self.sel_exp.push(sel[10]);
        self.sel_signextend.push(sel[11]);
        self.sel_lt.push(sel[12]);
        self.sel_gt.push(sel[13]);
        self.sel_eq.push(sel[14]);
        self.sel_iszero.push(sel[15]);
        self.sel_compare_other.push(sel[16]);
        self.sel_and.push(sel[17]);
        self.sel_or.push(sel[18]);
        self.sel_xor.push(sel[19]);
        self.sel_bitwise_other.push(sel[20]);
        self.sel_shl.push(sel[21]);
        self.sel_shr.push(sel[22]);
        self.sel_sar.push(sel[23]);
        self.sel_keccak.push(sel[24]);
        self.sel_env.push(sel[25]);
        self.sel_block.push(sel[26]);
        self.sel_push.push(sel[27]);
        self.sel_dup.push(sel[28]);
        self.sel_pop.push(sel[29]);
        self.sel_swap.push(sel[30]);
        self.sel_stack_other.push(sel[31]);
        self.sel_mload.push(sel[32]);
        self.sel_mstore.push(sel[33]);
        self.sel_mstore8.push(sel[34]);
        self.sel_msize.push(sel[35]);
        self.sel_memory_other.push(sel[36]);
        self.sel_storage.push(sel[37]);
        self.sel_jump.push(sel[38]);
        self.sel_log.push(sel[39]);
        self.sel_call.push(sel[40]);
        self.next_pc.push(row.next_pc);
    }
}

impl VmTrace for EvmTraceColumns {
    fn num_steps(&self) -> usize {
        self.step.len()
    }

    fn num_columns(&self) -> usize {
        1 + NUM_EVM_COLUMNS
    }

    fn column(&self, index: usize) -> &[u64] {
        match index {
            0 => &self.step,
            1 => &self.pc,
            2 => &self.opcode,
            3 => &self.gas_remaining,
            4 => &self.stack_depth,
            5 => &self.input0[0],
            6 => &self.input0[1],
            7 => &self.input0[2],
            8 => &self.input0[3],
            9 => &self.input1[0],
            10 => &self.input1[1],
            11 => &self.input1[2],
            12 => &self.input1[3],
            13 => &self.output0[0],
            14 => &self.output0[1],
            15 => &self.output0[2],
            16 => &self.output0[3],
            17 => &self.mem_offset,
            18 => &self.mem_value[0],
            19 => &self.mem_value[1],
            20 => &self.mem_value[2],
            21 => &self.mem_value[3],
            22 => &self.insn_type,
            23 => &self.funct,
            24 => &self.immediate[0],
            25 => &self.immediate[1],
            26 => &self.immediate[2],
            27 => &self.immediate[3],
            28 => &self.aux0[0],
            29 => &self.aux0[1],
            30 => &self.aux0[2],
            31 => &self.aux0[3],
            32 => &self.aux1[0],
            33 => &self.aux1[1],
            34 => &self.aux1[2],
            35 => &self.aux1[3],
            36 => &self.sel_stop,
            37 => &self.sel_arith_add,
            38 => &self.sel_arith_sub,
            39 => &self.sel_arith_mul,
            40 => &self.sel_arith_div,
            41 => &self.sel_mod,
            42 => &self.sel_sdiv,
            43 => &self.sel_smod,
            44 => &self.sel_addmod,
            45 => &self.sel_mulmod,
            46 => &self.sel_exp,
            47 => &self.sel_signextend,
            48 => &self.sel_lt,
            49 => &self.sel_gt,
            50 => &self.sel_eq,
            51 => &self.sel_iszero,
            52 => &self.sel_compare_other,
            53 => &self.sel_and,
            54 => &self.sel_or,
            55 => &self.sel_xor,
            56 => &self.sel_bitwise_other,
            57 => &self.sel_shl,
            58 => &self.sel_shr,
            59 => &self.sel_sar,
            60 => &self.sel_keccak,
            61 => &self.sel_env,
            62 => &self.sel_block,
            63 => &self.sel_push,
            64 => &self.sel_dup,
            65 => &self.sel_pop,
            66 => &self.sel_swap,
            67 => &self.sel_stack_other,
            68 => &self.sel_mload,
            69 => &self.sel_mstore,
            70 => &self.sel_mstore8,
            71 => &self.sel_msize,
            72 => &self.sel_memory_other,
            73 => &self.sel_storage,
            74 => &self.sel_jump,
            75 => &self.sel_log,
            76 => &self.sel_call,
            77 => &self.next_pc,
            _ => panic!("Column index {} out of range", index),
        }
    }

    fn columns(&self) -> Vec<&[u64]> {
        (0..self.num_columns()).map(|i| self.column(i)).collect()
    }

    fn column_names(&self) -> Vec<&'static str> {
        vec![
            "step", "pc", "opcode", "gas_remaining", "stack_depth",
            "input0_l0", "input0_l1", "input0_l2", "input0_l3",
            "input1_l0", "input1_l1", "input1_l2", "input1_l3",
            "output0_l0", "output0_l1", "output0_l2", "output0_l3",
            "mem_offset",
            "mem_value_l0", "mem_value_l1", "mem_value_l2", "mem_value_l3",
            "insn_type", "funct",
            "immediate_l0", "immediate_l1", "immediate_l2", "immediate_l3",
            "aux0_l0", "aux0_l1", "aux0_l2", "aux0_l3",
            "aux1_l0", "aux1_l1", "aux1_l2", "aux1_l3",
            "sel_stop", "sel_arith_add", "sel_arith_sub",
            "sel_arith_mul", "sel_arith_div",
            "sel_mod", "sel_sdiv", "sel_smod", "sel_addmod",
            "sel_mulmod", "sel_exp", "sel_signextend",
            "sel_lt", "sel_gt", "sel_eq", "sel_iszero", "sel_compare_other",
            "sel_and", "sel_or", "sel_xor", "sel_bitwise_other",
            "sel_shl", "sel_shr", "sel_sar",
            "sel_keccak", "sel_env", "sel_block",
            "sel_push", "sel_dup", "sel_pop", "sel_swap", "sel_stack_other",
            "sel_mload", "sel_mstore", "sel_mstore8", "sel_msize", "sel_memory_other",
            "sel_storage", "sel_jump", "sel_log", "sel_call",
            "next_pc",
        ]
    }
}

/// Extract the effective shift amount from a U256 input1 (clamped to 0..=256).
/// If input1 >= 256 (any upper limb nonzero or limb0 >= 256), returns 256.
fn shift_amount_u256(input1: [u64; 4]) -> u32 {
    if input1[1] != 0 || input1[2] != 0 || input1[3] != 0 || input1[0] >= 256 {
        256
    } else {
        input1[0] as u32
    }
}

/// Compute 2^k as 4 x u64 limbs. If k >= 256, returns [0,0,0,0].
fn power_of_two_limbs(k: u32) -> [u64; 4] {
    if k >= 256 {
        return [0, 0, 0, 0];
    }
    let limb_idx = (k / 64) as usize;
    let bit_idx = k % 64;
    let mut result = [0u64; 4];
    result[limb_idx] = 1u64 << bit_idx;
    result
}

/// Compute the 2^k limb representation for use in shift constraint immediate columns.
/// The shift amount is taken from input0 (top of stack in EVM shift operations).
pub fn shift_power_of_two(shift_amount: [u64; 4]) -> [u64; 4] {
    power_of_two_limbs(shift_amount_u256(shift_amount))
}

/// Compute auxiliary values for an EVM instruction.
///
/// For ADD: aux0 = per-limb carry chain [carry0..carry3], aux1 = 0
/// For SUB: aux0 = per-limb borrow chain [borrow0..borrow3], aux1 = 0
/// For MUL: aux0 = high 256 bits of 512-bit product, aux1[0] = high 64 bits of input0_l0*input1_l0
/// For DIV: aux0 = remainder, aux1[0] = carry from limb 0 of quotient*divisor+remainder
/// For LT (0x10): aux1[0] = lt result (1 if input0 < input1)
/// For GT (0x11): aux1[0] = gt result (1 if input0 > input1)
/// For ISZERO (0x15): aux1 = 0 (no extra witness needed)
/// For MSTORE8 (0x53): aux0[0] = input1_l0 / 256 (quotient for byte extraction)
pub fn compute_evm_aux(
    opcode: u8,
    input0: [u64; 4],
    input1: [u64; 4],
    output: [u64; 4],
) -> ([u64; 4], [u64; 4]) {
    let zero = [0u64; 4];
    match opcode {
        0x01 => {
            // ADD: per-limb carry chain
            let mut aux0 = [0u64; 4];
            let mut carry = 0u64;
            for i in 0..4 {
                let (s1, c1) = input0[i].overflowing_add(input1[i]);
                let (_, c2) = s1.overflowing_add(carry);
                carry = (c1 as u64) + (c2 as u64);
                aux0[i] = carry;
            }
            (aux0, zero)
        }
        0x03 => {
            // SUB: per-limb borrow chain
            let mut aux0 = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (s1, b1) = output[i].overflowing_add(input1[i]);
                let (_, b2) = s1.overflowing_add(borrow);
                borrow = (b1 as u64) + (b2 as u64);
                aux0[i] = borrow;
            }
            (aux0, zero)
        }
        0x02 => {
            // MUL: compute high 256 bits of 512-bit product (stored in aux0)
            // and the full carry chain for schoolbook multiplication (stored in aux1).
            let mut full = [0u128; 8];
            for i in 0..4 {
                let mut carry = 0u128;
                for j in 0..4 {
                    full[i + j] += (input0[i] as u128) * (input1[j] as u128) + carry;
                    carry = full[i + j] >> 64;
                    full[i + j] &= 0xFFFF_FFFF_FFFF_FFFF;
                }
                if i + 4 < 8 {
                    full[i + 4] += carry;
                }
            }
            let mut aux0 = [0u64; 4];
            for i in 0..4 {
                aux0[i] = full[i + 4] as u64;
            }
            let mut aux1 = [0u64; 4];
            let mut acc = (input0[0] as u128) * (input1[0] as u128);
            aux1[0] = (acc >> 64) as u64;
            acc = (input0[0] as u128) * (input1[1] as u128)
                + (input0[1] as u128) * (input1[0] as u128)
                + (aux1[0] as u128);
            aux1[1] = (acc >> 64) as u64;
            acc = (input0[0] as u128) * (input1[2] as u128)
                + (input0[1] as u128) * (input1[1] as u128)
                + (input0[2] as u128) * (input1[0] as u128)
                + (aux1[1] as u128);
            aux1[2] = (acc >> 64) as u64;
            acc = (input0[0] as u128) * (input1[3] as u128)
                + (input0[1] as u128) * (input1[2] as u128)
                + (input0[2] as u128) * (input1[1] as u128)
                + (input0[3] as u128) * (input1[0] as u128)
                + (aux1[2] as u128);
            aux1[3] = (acc >> 64) as u64;
            (aux0, aux1)
        }
        0x04 | 0x05 => {
            // DIV/SDIV: aux0 = remainder = dividend - quotient * divisor (mod 2^256)
            // aux1 = full carry chain for (quotient * divisor + remainder) per limb
            // The algebraic identity quotient*divisor+remainder = dividend holds in two's complement.
            let remainder = compute_remainder(input0, input1, output);
            let mut aux1 = [0u64; 4];
            let mut acc = (output[0] as u128) * (input1[0] as u128) + (remainder[0] as u128);
            aux1[0] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (input1[1] as u128)
                + (output[1] as u128) * (input1[0] as u128)
                + (remainder[1] as u128) + (aux1[0] as u128);
            aux1[1] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (input1[2] as u128)
                + (output[1] as u128) * (input1[1] as u128)
                + (output[2] as u128) * (input1[0] as u128)
                + (remainder[2] as u128) + (aux1[1] as u128);
            aux1[2] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (input1[3] as u128)
                + (output[1] as u128) * (input1[2] as u128)
                + (output[2] as u128) * (input1[1] as u128)
                + (output[3] as u128) * (input1[0] as u128)
                + (remainder[3] as u128) + (aux1[2] as u128);
            aux1[3] = (acc >> 64) as u64;
            (remainder, aux1)
        }
        0x06 | 0x07 => {
            // MOD/SMOD: output = remainder, aux0 = quotient
            // For SMOD: quotient is the signed quotient (two's complement).
            // The algebraic identity quotient*divisor+remainder = dividend holds mod 2^256.
            if input1 == [0, 0, 0, 0] {
                (zero, zero)
            } else {
                // For MOD (0x06), quotient is unsigned. For SMOD (0x07), compute signed quotient.
                let quotient = if opcode == 0x07 {
                    sdiv_u256(input0, input1)
                } else {
                    div_u256(input0, input1)
                };
                let mut aux1 = [0u64; 4];
                let mut acc = (quotient[0] as u128) * (input1[0] as u128) + (output[0] as u128);
                aux1[0] = (acc >> 64) as u64;
                acc = (quotient[0] as u128) * (input1[1] as u128)
                    + (quotient[1] as u128) * (input1[0] as u128)
                    + (output[1] as u128) + (aux1[0] as u128);
                aux1[1] = (acc >> 64) as u64;
                acc = (quotient[0] as u128) * (input1[2] as u128)
                    + (quotient[1] as u128) * (input1[1] as u128)
                    + (quotient[2] as u128) * (input1[0] as u128)
                    + (output[2] as u128) + (aux1[1] as u128);
                aux1[2] = (acc >> 64) as u64;
                acc = (quotient[0] as u128) * (input1[3] as u128)
                    + (quotient[1] as u128) * (input1[2] as u128)
                    + (quotient[2] as u128) * (input1[1] as u128)
                    + (quotient[3] as u128) * (input1[0] as u128)
                    + (output[3] as u128) + (aux1[2] as u128);
                aux1[3] = (acc >> 64) as u64;
                (quotient, aux1)
            }
        }
        0x10 => {
            // LT: subtract input0 - input1 with borrow chain
            let mut aux0 = [0u64; 4];
            let mut aux1 = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = input0[i].overflowing_sub(input1[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                aux1[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
                aux0[i] = borrow;
            }
            (aux0, aux1)
        }
        0x11 => {
            // GT: subtract input1 - input0 with borrow chain
            let mut aux0 = [0u64; 4];
            let mut aux1 = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = input1[i].overflowing_sub(input0[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                aux1[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
                aux0[i] = borrow;
            }
            (aux0, aux1)
        }
        0x14 => {
            // EQ: aux1[0..3] = diff limbs (input0 - input1), aux0 = 0
            let mut aux1 = [0u64; 4];
            for i in 0..4 {
                aux1[i] = input0[i].wrapping_sub(input1[i]);
            }
            (zero, aux1)
        }
        0x15 => {
            // ISZERO: no extra witness needed
            (zero, zero)
        }
        0x16 | 0x17 | 0x18 => {
            // AND/OR/XOR: aux0 = per-limb AND(input0, input1)
            let mut aux0 = [0u64; 4];
            for i in 0..4 {
                aux0[i] = input0[i] & input1[i];
            }
            (aux0, zero)
        }
        0x1B => {
            // SHL: output = input1 << input0 (mod 2^256)
            let k = shift_amount_u256(input0);
            let pow2 = power_of_two_limbs(k);
            let mut aux0 = [0u64; 4];
            let mut acc = (input1[0] as u128) * (pow2[0] as u128);
            aux0[0] = (acc >> 64) as u64;
            acc = (input1[0] as u128) * (pow2[1] as u128)
                + (input1[1] as u128) * (pow2[0] as u128)
                + (aux0[0] as u128);
            aux0[1] = (acc >> 64) as u64;
            acc = (input1[0] as u128) * (pow2[2] as u128)
                + (input1[1] as u128) * (pow2[1] as u128)
                + (input1[2] as u128) * (pow2[0] as u128)
                + (aux0[1] as u128);
            aux0[2] = (acc >> 64) as u64;
            acc = (input1[0] as u128) * (pow2[3] as u128)
                + (input1[1] as u128) * (pow2[2] as u128)
                + (input1[2] as u128) * (pow2[1] as u128)
                + (input1[3] as u128) * (pow2[0] as u128)
                + (aux0[2] as u128);
            aux0[3] = (acc >> 64) as u64;
            (aux0, zero)
        }
        0x1C | 0x1D => {
            // SHR/SAR: output = input1 >> input0 (logical or arithmetic)
            let k = shift_amount_u256(input0);
            let pow2 = power_of_two_limbs(k);
            let product = mul_u256_low(output, pow2);
            let remainder = sub_u256(input1, product);
            let mut aux1 = [0u64; 4];
            let mut acc = (output[0] as u128) * (pow2[0] as u128) + (remainder[0] as u128);
            aux1[0] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (pow2[1] as u128)
                + (output[1] as u128) * (pow2[0] as u128)
                + (remainder[1] as u128) + (aux1[0] as u128);
            aux1[1] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (pow2[2] as u128)
                + (output[1] as u128) * (pow2[1] as u128)
                + (output[2] as u128) * (pow2[0] as u128)
                + (remainder[2] as u128) + (aux1[1] as u128);
            aux1[2] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (pow2[3] as u128)
                + (output[1] as u128) * (pow2[2] as u128)
                + (output[2] as u128) * (pow2[1] as u128)
                + (output[3] as u128) * (pow2[0] as u128)
                + (remainder[3] as u128) + (aux1[2] as u128);
            aux1[3] = (acc >> 64) as u64;
            (remainder, aux1)
        }
        0x53 => {
            // MSTORE8: aux0[0] = input1_l0 / 256 (quotient for byte extraction)
            // Constraint: aux0_l0 * 256 + mem_val_l0 - input1_l0 = 0
            let aux0 = [input1[0] / 256, 0, 0, 0];
            (aux0, zero)
        }
        _ => (zero, zero),
    }
}

/// Add two 256-bit numbers, returning (result, carry).
#[allow(dead_code)]
fn add_u256_with_carry(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        result[i] = s2;
        carry = (c1 as u64) + (c2 as u64);
    }
    (result, carry > 0)
}

/// Compare two u256 values: a < b.
fn lt_u256(a: [u64; 4], b: [u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

/// Compute remainder = dividend - quotient * divisor (for DIV constraint).
fn compute_remainder(dividend: [u64; 4], divisor: [u64; 4], quotient: [u64; 4]) -> [u64; 4] {
    let product = mul_u256_low(quotient, divisor);
    sub_u256(dividend, product)
}

/// Multiply two u256 values, returning only the low 256 bits.
pub fn mul_u256_low(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            if i + j >= 4 { break; }
            let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
            result[i + j] = prod as u64;
            carry = prod >> 64;
        }
    }
    result
}

/// Divide a by b (unsigned 256-bit), returning the quotient.
/// Returns [0,0,0,0] if b == 0.
fn div_u256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    if b == [0, 0, 0, 0] {
        return [0, 0, 0, 0];
    }
    let mut quotient = [0u64; 4];
    let mut remainder = [0u64; 4];
    for bit in (0..256).rev() {
        let carry3 = remainder[3] >> 63;
        remainder[3] = (remainder[3] << 1) | (remainder[2] >> 63);
        remainder[2] = (remainder[2] << 1) | (remainder[1] >> 63);
        remainder[1] = (remainder[1] << 1) | (remainder[0] >> 63);
        remainder[0] <<= 1;
        let _ = carry3;
        let limb_idx = bit / 64;
        let bit_idx = bit % 64;
        remainder[0] |= (a[limb_idx] >> bit_idx) & 1;
        if !lt_u256(remainder, b) {
            remainder = sub_u256(remainder, b);
            quotient[limb_idx] |= 1u64 << bit_idx;
        }
    }
    quotient
}

/// Signed division of two 256-bit two's complement values, returning the signed quotient
/// in two's complement representation (mod 2^256).
fn sdiv_u256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    if b == [0, 0, 0, 0] {
        return [0, 0, 0, 0];
    }
    let a_neg = (a[3] >> 63) != 0;
    let b_neg = (b[3] >> 63) != 0;
    let abs_a = if a_neg { negate_u256(a) } else { a };
    let abs_b = if b_neg { negate_u256(b) } else { b };
    let q = div_u256(abs_a, abs_b);
    if a_neg != b_neg { negate_u256(q) } else { q }
}

/// Negate a 256-bit two's complement value: result = (2^256 - val) mod 2^256 = !val + 1.
fn negate_u256(val: [u64; 4]) -> [u64; 4] {
    let mut result = [!val[0], !val[1], !val[2], !val[3]];
    let mut carry = 1u64;
    for i in 0..4 {
        let (s, c) = result[i].overflowing_add(carry);
        result[i] = s;
        carry = c as u64;
    }
    result
}

/// Subtract b from a (mod 2^256).
fn sub_u256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (s1, b1) = a[i].overflowing_sub(b[i]);
        let (s2, b2) = s1.overflowing_sub(borrow);
        result[i] = s2;
        borrow = (b1 as u64) + (b2 as u64);
    }
    result
}

/// Compute a SHA3-256 hash of the EVM execution state at a given row.
pub fn evm_state_hash(trace: &EvmTraceColumns, row: usize) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(trace.step[row].to_le_bytes());
    hasher.update(trace.pc[row].to_le_bytes());
    hasher.update(trace.opcode[row].to_le_bytes());
    hasher.update(trace.gas_remaining[row].to_le_bytes());
    hasher.update(trace.stack_depth[row].to_le_bytes());
    hasher.finalize().into()
}

/// Hash for the initial state (before any execution).
pub fn evm_initial_state_hash() -> [u8; 32] {
    let hasher = Sha3_256::new();
    hasher.finalize().into()
}

/// Hash for the final state after all execution.
pub fn evm_final_state_hash(trace: &EvmTraceColumns) -> [u8; 32] {
    if trace.step.is_empty() {
        return evm_initial_state_hash();
    }
    let last = trace.step.len() - 1;
    let mut hasher = Sha3_256::new();
    hasher.update(b"final");
    hasher.update(trace.step[last].to_le_bytes());
    hasher.update(trace.pc[last].to_le_bytes());
    hasher.update(trace.opcode[last].to_le_bytes());
    hasher.update(trace.gas_remaining[last].to_le_bytes());
    hasher.update(trace.stack_depth[last].to_le_bytes());
    hasher.finalize().into()
}

/// Convenience wrapper: build TracePolynomials from an EVM trace with the given curve.
pub fn evm_trace_polys_with_curve(trace: &EvmTraceColumns, curve: CurveType) -> TracePolynomials {
    TracePolynomials::from_vm_trace(trace, curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_opcode_add() {
        assert_eq!(classify_opcode(0x01), (INSN_ARITH, FUNCT_ADD));
    }

    #[test]
    fn test_classify_opcode_push1() {
        assert_eq!(classify_opcode(0x60), (INSN_STACK, FUNCT_PUSH));
    }

    #[test]
    fn test_classify_opcode_stop() {
        assert_eq!(classify_opcode(0x00), (INSN_STOP, 0));
    }

    #[test]
    fn test_classify_opcode_jump() {
        assert_eq!(classify_opcode(0x56), (INSN_JUMP, FUNCT_JUMP));
    }

    #[test]
    fn test_u256_add_no_carry() {
        let a = [10, 0, 0, 0];
        let b = [20, 0, 0, 0];
        let (result, carry) = add_u256_with_carry(a, b);
        assert_eq!(result, [30, 0, 0, 0]);
        assert!(!carry);
    }

    #[test]
    fn test_u256_add_with_carry() {
        let a = [u64::MAX, 0, 0, 0];
        let b = [1, 0, 0, 0];
        let (result, carry) = add_u256_with_carry(a, b);
        assert_eq!(result, [0, 1, 0, 0]);
        assert!(!carry);
    }

    #[test]
    fn test_u256_add_overflow() {
        let a = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let b = [1, 0, 0, 0];
        let (result, carry) = add_u256_with_carry(a, b);
        assert_eq!(result, [0, 0, 0, 0]);
        assert!(carry);
    }

    #[test]
    fn test_u256_sub() {
        let a = [30, 0, 0, 0];
        let b = [10, 0, 0, 0];
        let result = sub_u256(a, b);
        assert_eq!(result, [20, 0, 0, 0]);
    }

    #[test]
    fn test_u256_mul() {
        let a = [3, 0, 0, 0];
        let b = [7, 0, 0, 0];
        let result = mul_u256_low(a, b);
        assert_eq!(result, [21, 0, 0, 0]);
    }

    #[test]
    fn test_vm_trace_columns() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x01, // ADD
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [10, 0, 0, 0],
            input1: [20, 0, 0, 0],
            output0: [30, 0, 0, 0],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_ADD,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
        };
        trace.push_row(&row);

        assert_eq!(trace.num_steps(), 1);
        assert_eq!(trace.num_columns(), 78); // step + 77 data columns
        assert_eq!(trace.column(1)[0], 0);  // pc
        assert_eq!(trace.column(2)[0], 0x01);  // opcode
    }

    #[test]
    fn test_compute_aux_add() {
        let input0 = [10, 0, 0, 0];
        let input1 = [20, 0, 0, 0];
        let output = [30, 0, 0, 0];
        let (aux0, aux1) = compute_evm_aux(0x01, input0, input1, output);
        assert_eq!(aux0[0], 0); // no carry
        assert_eq!(aux1, [0; 4]);
    }

    #[test]
    fn test_compute_aux_add_overflow() {
        let input0 = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let input1 = [1, 0, 0, 0];
        let output = [0, 0, 0, 0];
        let (aux0, _) = compute_evm_aux(0x01, input0, input1, output);
        assert_eq!(aux0[0], 1); // carry
    }

    #[test]
    fn test_compute_aux_mstore8() {
        // MSTORE8: stores byte at input1_l0 & 0xFF
        let input0 = [0x100, 0, 0, 0]; // offset
        let input1 = [0x1234, 0, 0, 0]; // value (low byte = 0x34)
        let output = [0; 4];
        let (aux0, _) = compute_evm_aux(0x53, input0, input1, output);
        // aux0[0] = input1_l0 / 256 = 0x1234 / 256 = 0x12
        assert_eq!(aux0[0], 0x12);
    }
}
