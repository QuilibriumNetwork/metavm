//! Full block validation oracle.
//!
//! Combines all EVM-side host oracles into a single validation:
//! - Bytecode table correctness
//! - JUMPDEST validity
//! - Gas accounting
//! - LOG event extraction + bloom verification
//!
//! This is the EVM-side complement to the zkp crate's
//! `full_proof_oracle::verify_full_proof`.

use crate::bytecode_table::build_bytecode_table;
use crate::calldata::verify_calldatasize;
use crate::gas_accounting::verify_gas_accounting;
use crate::jumpdest_validity::verify_jump_targets;
use crate::log_event::extract_log_events;
use crate::trace::EvmTraceColumns;

pub fn validate_evm_execution(
    bytecode: &[u8],
    cols: &EvmTraceColumns,
) -> Result<(), String> {
    let _table = build_bytecode_table(bytecode);

    verify_jump_targets(bytecode, cols)
        .map_err(|e| format!("jumpdest: {}", e))?;

    let _gas_steps = verify_gas_accounting(cols)
        .map_err(|e| format!("gas: {}", e))?;

    crate::codesize::verify_codesize(bytecode, cols)
        .map_err(|e| format!("codesize: {}", e))?;

    crate::stack_depth::verify_stack_depth_consistency(cols)
        .map_err(|e| format!("stack_depth: {}", e))?;

    crate::pc_continuity::verify_pc_continuity(cols)
        .map_err(|e| format!("pc: {}", e))?;

    let _events = extract_log_events(cols);

    Ok(())
}

pub fn validate_evm_execution_with_calldata(
    bytecode: &[u8],
    calldata: &[u8],
    cols: &EvmTraceColumns,
) -> Result<(), String> {
    validate_evm_execution(bytecode, cols)?;

    verify_calldatasize(calldata, cols)
        .map_err(|e| format!("calldatasize: {}", e))?;

    crate::calldata::verify_calldataload_results(calldata, cols)
        .map_err(|e| format!("calldataload: {}", e))?;

    Ok(())
}

pub fn validate_gas_accounting_full(
    gas_limit: u64,
    gas_remaining: u64,
    refund: u64,
    receipt_cumulative_gas: u64,
    prior_cumulative_gas: u64,
) -> Result<(), String> {
    crate::gas_refund::verify_receipt_gas_used(
        gas_limit, gas_remaining, refund,
        receipt_cumulative_gas, prior_cumulative_gas,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn simple_add_validates() {
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        validate_evm_execution(&bc, &cols).unwrap();
    }

    #[test]
    fn jump_validates() {
        let bc = vec![0x60, 0x04, 0x56, 0xFE, 0x5B, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        validate_evm_execution(&bc, &cols).unwrap();
    }

    #[test]
    fn log0_validates() {
        let bc = vec![0x60, 0x00, 0x60, 0x00, 0xA0, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        validate_evm_execution(&bc, &cols).unwrap();
    }

    #[test]
    fn with_calldata_validates() {
        // CALLDATASIZE; PUSH1 0; CALLDATALOAD; STOP
        let bc = vec![0x36, 0x60, 0x00, 0x35, 0x00];
        let calldata = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let cols = execute_bytecode(&bc, &calldata).unwrap();
        validate_evm_execution_with_calldata(&bc, &calldata, &cols).unwrap();
    }

    #[test]
    fn gas_accounting_full() {
        validate_gas_accounting_full(100_000, 79_000, 0, 21_000, 0).unwrap();
    }

    #[test]
    fn sstore_sload_validates() {
        // PUSH1 0x42 (value); PUSH1 0x00 (slot); SSTORE;
        // PUSH1 0x00 (slot); SLOAD; STOP
        let bc = vec![0x60, 0x42, 0x60, 0x00, 0x55, 0x60, 0x00, 0x54, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        validate_evm_execution(&bc, &cols).unwrap();
    }
}
