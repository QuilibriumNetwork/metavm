//! Custom revm Inspector that records each EVM step for trace generation.
//!
//! The inspector captures pre- and post-execution state at each opcode,
//! producing `EvmTraceRow` entries that populate an `EvmTraceColumns`.

use crate::trace::{EvmTraceRow, EvmTraceColumns, classify_opcode, compute_evm_aux, shift_power_of_two};
use revm::interpreter::{Interpreter, Stack};
use revm::interpreter::interpreter_types::{Jumps, LegacyBytecode, InterpreterTypes};
use revm::Inspector;

/// Pending row state captured in step() before execution.
struct PendingRow {
    step: u64,
    pc: u64,
    opcode: u8,
    gas_remaining: u64,
    stack_depth: u64,
    input0: [u64; 4],
    input1: [u64; 4],
    immediate: [u64; 4],
    insn_type: u8,
    funct: u8,
}

/// Inspector that records EVM execution traces.
pub struct TracingInspector {
    pub trace: EvmTraceColumns,
    pending: Option<PendingRow>,
    step_count: u64,
}

impl TracingInspector {
    pub fn new() -> Self {
        TracingInspector {
            trace: EvmTraceColumns::new(),
            pending: None,
            step_count: 0,
        }
    }
}

/// Extract limbs from a revm U256 value.
fn u256_limbs(val: &revm::primitives::U256) -> [u64; 4] {
    val.as_limbs().clone()
}

/// Peek a stack value safely, returning zero if index is out of range.
fn safe_peek(stack: &Stack, index: usize) -> [u64; 4] {
    if index < stack.len() {
        match stack.peek(index) {
            Ok(val) => u256_limbs(&val),
            Err(_) => [0u64; 4],
        }
    } else {
        [0u64; 4]
    }
}

impl<CTX> Inspector<CTX> for TracingInspector {
    fn step(&mut self, interp: &mut Interpreter, _context: &mut CTX) {
        let pc = interp.bytecode.pc() as u64;
        let opcode = interp.bytecode.opcode();
        let gas_remaining = interp.gas.remaining();
        let stack_depth = interp.stack.len() as u64;
        let (insn_type, funct) = classify_opcode(opcode);

        let input0 = safe_peek(&interp.stack, 0);
        let input1 = safe_peek(&interp.stack, 1);

        // For PUSH instructions, extract the immediate value from bytecode
        let immediate = if opcode >= 0x60 && opcode <= 0x7F {
            let push_size = (opcode - 0x60 + 1) as usize;
            let bytecode = interp.bytecode.bytecode_slice();
            let pc_val = pc as usize;
            let mut imm_bytes = [0u8; 32];
            let start = pc_val + 1;
            let end = (start + push_size).min(bytecode.len());
            if start < bytecode.len() {
                let available = end - start;
                // Big-endian: value bytes are placed at the end
                imm_bytes[32 - available..32].copy_from_slice(&bytecode[start..end]);
            }
            // Convert big-endian bytes to U256 limbs (limb 0 = least significant)
            // imm_bytes is big-endian: [0] is MSB, [31] is LSB
            // Limb 0 = bytes [24..32] (least significant 64 bits)
            // Limb 3 = bytes [0..8]   (most significant 64 bits)
            [
                u64::from_be_bytes([imm_bytes[24], imm_bytes[25], imm_bytes[26], imm_bytes[27],
                                    imm_bytes[28], imm_bytes[29], imm_bytes[30], imm_bytes[31]]),
                u64::from_be_bytes([imm_bytes[16], imm_bytes[17], imm_bytes[18], imm_bytes[19],
                                    imm_bytes[20], imm_bytes[21], imm_bytes[22], imm_bytes[23]]),
                u64::from_be_bytes([imm_bytes[8], imm_bytes[9], imm_bytes[10], imm_bytes[11],
                                    imm_bytes[12], imm_bytes[13], imm_bytes[14], imm_bytes[15]]),
                u64::from_be_bytes([imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3],
                                    imm_bytes[4], imm_bytes[5], imm_bytes[6], imm_bytes[7]]),
            ]
        } else if opcode == 0x1B || opcode == 0x1C || opcode == 0x1D {
            // SHL/SHR/SAR: immediate = 2^k where k = shift amount (input0)
            shift_power_of_two(input0)
        } else {
            [0u64; 4]
        };

        self.pending = Some(PendingRow {
            step: self.step_count,
            pc,
            opcode,
            gas_remaining,
            stack_depth,
            input0,
            input1,
            immediate,
            insn_type,
            funct,
        });
    }

    fn step_end(&mut self, interp: &mut Interpreter, _context: &mut CTX) {
        if let Some(pending) = self.pending.take() {
            // Capture post-execution output (top of stack after execution)
            let output0 = safe_peek(&interp.stack, 0);

            // Memory access: for MLOAD/MSTORE, capture address and value
            let (mem_offset, mem_value) = match pending.opcode {
                0x51 => {
                    // MLOAD: offset was input0, loaded value is output0
                    (pending.input0[0], output0)
                }
                0x52 => {
                    // MSTORE: offset was input0, stored value was input1
                    (pending.input0[0], pending.input1)
                }
                0x53 => {
                    // MSTORE8: offset was input0, value (1 byte) was input1
                    (pending.input0[0], pending.input1)
                }
                _ => (0u64, [0u64; 4]),
            };

            let (aux0, aux1) = compute_evm_aux(
                pending.opcode,
                pending.input0,
                pending.input1,
                output0,
            );

            let next_pc = interp.bytecode.pc() as u64;

            let row = EvmTraceRow {
                step: pending.step,
                pc: pending.pc,
                opcode: pending.opcode,
                gas_remaining: pending.gas_remaining,
                stack_depth: pending.stack_depth,
                input0: pending.input0,
                input1: pending.input1,
                output0,
                mem_offset,
                mem_value,
                insn_type: pending.insn_type,
                funct: pending.funct,
                immediate: pending.immediate,
                aux0,
                aux1,
                next_pc,
            };
            self.trace.push_row(&row);
            self.step_count += 1;
        }
    }
}
