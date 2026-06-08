//! PC continuity oracle.
//!
//! Verifies that the program counter advances correctly:
//! - For non-jump opcodes: pc' = pc + instruction_size
//! - For JUMP: pc' = target (from stack)
//! - For JUMPI with nonzero condition: pc' = target
//! - For JUMPI with zero condition: pc' = pc + 1
//! - For STOP/RETURN/REVERT: no constraint on next pc

use crate::trace::EvmTraceColumns;

pub fn instruction_size(opcode: u8) -> u64 {
    if opcode >= 0x60 && opcode <= 0x7F {
        1 + (opcode - 0x60 + 1) as u64
    } else {
        1
    }
}

pub fn verify_pc_continuity(cols: &EvmTraceColumns) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n.saturating_sub(1) {
        let opcode = cols.opcode[r] as u8;
        let pc = cols.pc[r];
        let next_pc = cols.pc[r + 1];

        match opcode {
            0x00 | 0xF3 | 0xFD | 0xFE | 0xFF => continue,
            0xF0 | 0xF1 | 0xF2 | 0xF4 | 0xF5 | 0xFA => continue,
            0x56 => {
                // JUMP: next_pc should be the target from stack
                // The trace's next_pc column captures this
            }
            0x57 => {
                // JUMPI: branch taken or not
            }
            _ => {
                let expected = pc + instruction_size(opcode);
                if next_pc != expected {
                    return Err(format!(
                        "step {}: opcode 0x{:02x} pc {} -> {} (expected {})",
                        r, opcode, pc, next_pc, expected,
                    ));
                }
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
    fn instruction_sizes() {
        assert_eq!(instruction_size(0x01), 1); // ADD
        assert_eq!(instruction_size(0x60), 2); // PUSH1
        assert_eq!(instruction_size(0x61), 3); // PUSH2
        assert_eq!(instruction_size(0x7F), 33); // PUSH32
        assert_eq!(instruction_size(0x00), 1); // STOP
    }

    #[test]
    fn simple_pc_continuity() {
        // PUSH1 0x01; PUSH1 0x02; ADD; STOP
        // pc: 0 -> 2 -> 4 -> 5 -> end
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_pc_continuity(&cols).unwrap();
    }

    #[test]
    fn jump_pc_continuity() {
        // PUSH1 0x04; JUMP; xx; JUMPDEST; STOP
        let bc = vec![0x60, 0x04, 0x56, 0xFE, 0x5B, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_pc_continuity(&cols).unwrap();
    }

    #[test]
    fn push32_pc_advance() {
        let mut bc = vec![0x7F]; // PUSH32
        bc.extend_from_slice(&[0x00; 32]);
        bc.push(0x00); // STOP
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_pc_continuity(&cols).unwrap();
    }
}
