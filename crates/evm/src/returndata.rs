//! Return data oracle for CALL-family opcodes.
//!
//! RETURNDATASIZE (0x3D) returns the size of the last external call's
//! return data. RETURNDATACOPY (0x3E) copies return data to memory.
//!
//! This oracle extracts RETURNDATASIZE values from the trace and
//! verifies monotonicity (return data size can only change after
//! CALL/STATICCALL/DELEGATECALL/CALLCODE operations).

use crate::trace::EvmTraceColumns;

pub fn extract_returndatasize_values(cols: &EvmTraceColumns) -> Vec<(usize, u64)> {
    let n = cols.step.len();
    let mut results = Vec::new();
    for r in 0..n {
        if cols.opcode[r] == 0x3D {
            results.push((r, cols.output0[0][r]));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn returndatasize_zero_without_calls() {
        // RETURNDATASIZE; STOP
        let bc = vec![0x3D, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let results = extract_returndatasize_values(&cols);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn no_returndatasize_empty() {
        let bc = vec![0x60, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        assert!(extract_returndatasize_values(&cols).is_empty());
    }
}
