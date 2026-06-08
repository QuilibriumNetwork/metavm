//! JUMPDEST validity oracle.
//!
//! Verifies that every JUMP/JUMPI target in the EVM trace lands on a
//! valid JUMPDEST position. Uses `jumpdest::valid_jumpdest_positions`
//! to build the valid set, then checks each JUMP row's target.

use crate::jumpdest::valid_jumpdest_positions;
use crate::trace::EvmTraceColumns;
use std::collections::HashSet;

pub fn verify_jump_targets(bytecode: &[u8], cols: &EvmTraceColumns) -> Result<(), String> {
    let valid_positions: HashSet<u64> = valid_jumpdest_positions(bytecode)
        .into_iter()
        .map(|p| p as u64)
        .collect();

    let n = cols.step.len();
    for r in 0..n {
        if cols.sel_jump[r] != 1 { continue; }
        let opcode = cols.opcode[r] as u8;
        match opcode {
            0x56 => {
                // JUMP: target = input0[0] (stack top)
                let target = cols.input0[0][r];
                if !valid_positions.contains(&target) {
                    return Err(format!(
                        "step {}: JUMP target {} is not a valid JUMPDEST",
                        r, target,
                    ));
                }
            }
            0x57 => {
                // JUMPI: target = input0[0], condition = input1[0]
                let target = cols.input0[0][r];
                let condition = cols.input1[0][r];
                if condition != 0 && !valid_positions.contains(&target) {
                    return Err(format!(
                        "step {}: JUMPI target {} (condition={}) is not a valid JUMPDEST",
                        r, target, condition,
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn valid_jump_passes() {
        // PUSH1 0x04; JUMP; INVALID; JUMPDEST; STOP
        let bc = vec![0x60, 0x04, 0x56, 0xFE, 0x5B, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_jump_targets(&bc, &cols).unwrap();
    }

    #[test]
    fn valid_jumpi_taken_passes() {
        // PUSH1 0x01 (condition); PUSH1 0x06; JUMPI; INVALID; INVALID; JUMPDEST; STOP
        // JUMPI is at offset 4. JUMPDEST is at offset 6, so target = 6.
        let bc = vec![0x60, 0x01, 0x60, 0x06, 0x57, 0xFE, 0x5B, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_jump_targets(&bc, &cols).unwrap();
    }

    #[test]
    fn jumpi_not_taken_skips_check() {
        // PUSH1 0x00 (condition=0); PUSH1 0xFF (invalid target); JUMPI; STOP
        let bc = vec![0x60, 0x00, 0x60, 0xFF, 0x57, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_jump_targets(&bc, &cols).unwrap();
    }

    #[test]
    fn no_jumps_passes() {
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00]; // PUSH; PUSH; ADD; STOP
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_jump_targets(&bc, &cols).unwrap();
    }

    #[test]
    fn valid_jumpdest_positions_correct() {
        // PUSH2 0x5B 0x5B; JUMPDEST; STOP
        // Offset 0: PUSH2, offsets 1,2: immediates (0x5B looks like JUMPDEST but isn't)
        // Offset 3: JUMPDEST (valid)
        let bc = vec![0x61, 0x5B, 0x5B, 0x5B, 0x00];
        let positions = valid_jumpdest_positions(&bc);
        assert!(positions.contains(&3));
        assert!(!positions.contains(&1));
        assert!(!positions.contains(&2));
    }
}
