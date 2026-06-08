//! Calldata oracle for ENV opcodes.
//!
//! CALLDATALOAD (0x35) reads 32 bytes from calldata at a given offset.
//! CALLDATASIZE (0x36) returns the calldata length.
//! CALLDATACOPY (0x37) copies calldata bytes to memory.
//!
//! This oracle verifies that CALLDATALOAD results in the trace match
//! the transaction's input data.

use crate::trace::EvmTraceColumns;

pub fn verify_calldataload_results(
    calldata: &[u8],
    cols: &EvmTraceColumns,
) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n {
        if cols.opcode[r] != 0x35 { continue; }
        let offset = cols.input0[0][r] as usize;
        let mut expected = [0u8; 32];
        for i in 0..32 {
            if offset + i < calldata.len() {
                expected[i] = calldata[offset + i];
            }
        }
        let result_l0 = cols.output0[0][r];
        let result_l1 = cols.output0[1][r];
        let result_l2 = cols.output0[2][r];
        let result_l3 = cols.output0[3][r];

        let exp_l0 = u64::from_be_bytes([
            expected[24], expected[25], expected[26], expected[27],
            expected[28], expected[29], expected[30], expected[31],
        ]);
        let exp_l1 = u64::from_be_bytes([
            expected[16], expected[17], expected[18], expected[19],
            expected[20], expected[21], expected[22], expected[23],
        ]);
        let exp_l2 = u64::from_be_bytes([
            expected[8], expected[9], expected[10], expected[11],
            expected[12], expected[13], expected[14], expected[15],
        ]);
        let exp_l3 = u64::from_be_bytes([
            expected[0], expected[1], expected[2], expected[3],
            expected[4], expected[5], expected[6], expected[7],
        ]);

        if result_l0 != exp_l0 || result_l1 != exp_l1 || result_l2 != exp_l2 || result_l3 != exp_l3 {
            return Err(format!(
                "step {}: CALLDATALOAD at offset {} mismatch",
                r, offset,
            ));
        }
    }
    Ok(())
}

pub fn verify_calldatasize(
    calldata: &[u8],
    cols: &EvmTraceColumns,
) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n {
        if cols.opcode[r] != 0x36 { continue; }
        let result = cols.output0[0][r];
        if result != calldata.len() as u64 {
            return Err(format!(
                "step {}: CALLDATASIZE {} != expected {}",
                r, result, calldata.len(),
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
    fn calldatasize_correct() {
        // CALLDATASIZE; STOP
        let bc = vec![0x36, 0x00];
        let calldata = vec![0x01, 0x02, 0x03, 0x04];
        let cols = execute_bytecode(&bc, &calldata).unwrap();
        verify_calldatasize(&calldata, &cols).unwrap();
    }

    #[test]
    fn calldatasize_empty() {
        let bc = vec![0x36, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_calldatasize(&[], &cols).unwrap();
    }

    #[test]
    fn calldataload_correct() {
        // PUSH1 0x00; CALLDATALOAD; STOP
        let bc = vec![0x60, 0x00, 0x35, 0x00];
        let mut calldata = vec![0u8; 32];
        calldata[31] = 0x42;
        let cols = execute_bytecode(&bc, &calldata).unwrap();
        verify_calldataload_results(&calldata, &cols).unwrap();
    }

    #[test]
    fn calldataload_past_end_pads_zeros() {
        // PUSH1 0x10; CALLDATALOAD; STOP (offset 16, calldata only 4 bytes)
        let bc = vec![0x60, 0x10, 0x35, 0x00];
        let calldata = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let cols = execute_bytecode(&bc, &calldata).unwrap();
        verify_calldataload_results(&calldata, &cols).unwrap();
    }

    #[test]
    fn no_calldataload_passes() {
        let bc = vec![0x60, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        verify_calldataload_results(&[], &cols).unwrap();
    }
}
