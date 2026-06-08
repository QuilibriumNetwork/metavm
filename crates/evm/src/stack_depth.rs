//! Stack depth consistency oracle.
//!
//! Verifies that the `stack_depth` column in the EVM trace is
//! consistent with each opcode's push/pop behavior.

use crate::trace::EvmTraceColumns;

pub fn stack_delta(opcode: u8) -> Option<(u8, u8)> {
    match opcode {
        0x00 => Some((0, 0)), // STOP
        0x01..=0x0B => Some((2, 1)), // ADD..SIGNEXTEND (2 pop, 1 push)
        0x10..=0x1D => Some((2, 1)), // LT..SAR
        0x20 => Some((2, 1)), // KECCAK256
        0x30 => Some((0, 1)), // ADDRESS
        0x31 => Some((1, 1)), // BALANCE
        0x32 => Some((0, 1)), // ORIGIN
        0x33 => Some((0, 1)), // CALLER
        0x34 => Some((0, 1)), // CALLVALUE
        0x35 => Some((1, 1)), // CALLDATALOAD
        0x36 => Some((0, 1)), // CALLDATASIZE
        0x37 => Some((3, 0)), // CALLDATACOPY
        0x38 => Some((0, 1)), // CODESIZE
        0x39 => Some((3, 0)), // CODECOPY
        0x3A => Some((0, 1)), // GASPRICE
        0x3B => Some((1, 1)), // EXTCODESIZE
        0x3C => Some((4, 0)), // EXTCODECOPY
        0x3D => Some((0, 1)), // RETURNDATASIZE
        0x3E => Some((3, 0)), // RETURNDATACOPY
        0x3F => Some((1, 1)), // EXTCODEHASH
        0x40 => Some((1, 1)), // BLOCKHASH
        0x41..=0x48 => Some((0, 1)), // COINBASE..BASEFEE
        0x50 => Some((1, 0)), // POP
        0x51 => Some((1, 1)), // MLOAD
        0x52 => Some((2, 0)), // MSTORE
        0x53 => Some((2, 0)), // MSTORE8
        0x54 => Some((1, 1)), // SLOAD
        0x55 => Some((2, 0)), // SSTORE
        0x56 => Some((1, 0)), // JUMP
        0x57 => Some((2, 0)), // JUMPI
        0x58 => Some((0, 1)), // PC
        0x59 => Some((0, 1)), // MSIZE
        0x5A => Some((0, 1)), // GAS
        0x5B => Some((0, 0)), // JUMPDEST
        0x5F => Some((0, 1)), // PUSH0
        0x60..=0x7F => Some((0, 1)), // PUSH1..PUSH32
        0x80..=0x8F => { let n = opcode - 0x80 + 1; Some((n, n + 1)) } // DUP1..DUP16
        0x90..=0x9F => { let n = opcode - 0x90 + 2; Some((n, n)) } // SWAP1..SWAP16
        0xA0 => Some((2, 0)), // LOG0
        0xA1 => Some((3, 0)), // LOG1
        0xA2 => Some((4, 0)), // LOG2
        0xA3 => Some((5, 0)), // LOG3
        0xA4 => Some((6, 0)), // LOG4
        0xF0 => Some((3, 1)), // CREATE
        0xF1 => Some((7, 1)), // CALL
        0xF3 => Some((2, 0)), // RETURN
        0xF5 => Some((4, 1)), // CREATE2
        0xFA => Some((6, 1)), // STATICCALL
        0xFD => Some((2, 0)), // REVERT
        0xFE => Some((0, 0)), // INVALID
        0xFF => Some((1, 0)), // SELFDESTRUCT
        _ => None,
    }
}

pub fn verify_stack_depth_consistency(cols: &EvmTraceColumns) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n.saturating_sub(1) {
        let opcode = cols.opcode[r] as u8;
        if let Some((pops, pushes)) = stack_delta(opcode) {
            let current = cols.stack_depth[r];
            let next = cols.stack_depth[r + 1];
            if opcode == 0x00 || opcode == 0xF3 || opcode == 0xFD {
                continue;
            }
            if opcode >= 0xF0 {
                continue;
            }
            let expected = current - pops as u64 + pushes as u64;
            if next != expected {
                return Err(format!(
                    "step {}: opcode 0x{:02x} stack {} -> {} (expected {})",
                    r, opcode, current, next, expected,
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn simple_add_stack_consistent() {
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00]; // PUSH1; PUSH1; ADD; STOP
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_stack_depth_consistency(&cols).unwrap();
    }

    #[test]
    fn push_pop_stack_consistent() {
        let bc = vec![0x60, 0x42, 0x50, 0x00]; // PUSH1 0x42; POP; STOP
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_stack_depth_consistency(&cols).unwrap();
    }

    #[test]
    fn dup_swap_consistent() {
        // PUSH1 1; PUSH1 2; DUP2; SWAP1; POP; POP; STOP
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x81, 0x90, 0x50, 0x50, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_stack_depth_consistency(&cols).unwrap();
    }

    #[test]
    fn stack_delta_push1() {
        assert_eq!(stack_delta(0x60), Some((0, 1)));
    }

    #[test]
    fn stack_delta_add() {
        assert_eq!(stack_delta(0x01), Some((2, 1)));
    }

    #[test]
    fn stack_delta_dup1() {
        assert_eq!(stack_delta(0x80), Some((1, 2)));
    }

    #[test]
    fn stack_delta_log2() {
        assert_eq!(stack_delta(0xA2), Some((4, 0)));
    }
}
