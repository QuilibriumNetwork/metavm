//! CODESIZE/EXTCODESIZE oracle.
//!
//! CODESIZE (0x38) returns the size of the current contract's code.
//! EXTCODESIZE (0x3B) returns the size of an external account's code.
//!
//! This oracle verifies CODESIZE results match the deployed bytecode.

use crate::trace::EvmTraceColumns;

pub fn verify_codesize(
    bytecode: &[u8],
    cols: &EvmTraceColumns,
) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n {
        if cols.opcode[r] != 0x38 { continue; }
        let result = cols.output0[0][r];
        if result != bytecode.len() as u64 {
            return Err(format!(
                "step {}: CODESIZE {} != expected {}",
                r, result, bytecode.len(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn codesize_correct() {
        // CODESIZE; STOP (bytecode = 2 bytes)
        let bc = vec![0x38, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_codesize(&bc, &cols).unwrap();
    }

    #[test]
    fn codesize_larger_bytecode() {
        // PUSH1 0x01; PUSH1 0x02; ADD; CODESIZE; STOP (6 bytes)
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x38, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_codesize(&bc, &cols).unwrap();
        // Find the CODESIZE row and check the result
        let n = cols.step.len();
        for r in 0..n {
            if cols.opcode[r] == 0x38 {
                assert_eq!(cols.output0[0][r], 7); // 7 bytes
            }
        }
    }

    #[test]
    fn no_codesize_passes() {
        let bc = vec![0x60, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_codesize(&bc, &cols).unwrap();
    }
}
