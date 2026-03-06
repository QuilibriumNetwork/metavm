//! EVM execution wrapper that runs bytecode and captures a traced execution.

use crate::inspector::TracingInspector;
use crate::trace::EvmTraceColumns;
use revm::context::Context;
use revm::context::TxEnv;
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::handler::{ExecuteCommitEvm, ExecuteEvm, MainBuilder, MainContext};
use revm::InspectEvm;
use revm::primitives::{Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};

/// Execute raw EVM bytecode with the given calldata, returning the execution trace.
pub fn execute_bytecode(bytecode: &[u8], calldata: &[u8]) -> Result<EvmTraceColumns, String> {
    let contract_addr = Address::from([0x42; 20]);
    let caller_addr = Address::from([0xCA; 20]);

    let mut db = CacheDB::<EmptyDB>::default();

    // Insert contract with the given bytecode
    db.insert_account_info(
        contract_addr,
        AccountInfo {
            balance: U256::ZERO,
            nonce: 0,
            code_hash: Default::default(),
            account_id: None,
            code: Some(Bytecode::new_legacy(Bytes::copy_from_slice(bytecode))),
        },
    );

    // Give caller some ETH for gas
    db.insert_account_info(
        caller_addr,
        AccountInfo {
            balance: U256::from(1_000_000_000_000_000_000u64),
            nonce: 0,
            code_hash: Default::default(),
            account_id: None,
            code: None,
        },
    );

    let inspector = TracingInspector::new();
    let ctx = Context::mainnet().with_db(db);
    let mut evm = ctx.build_mainnet_with_inspector(inspector);

    let tx = TxEnv::builder()
        .caller(caller_addr)
        .kind(TxKind::Call(contract_addr))
        .gas_limit(10_000_000)
        .gas_price(0)
        .value(U256::ZERO)
        .data(Bytes::copy_from_slice(calldata))
        .nonce(0)
        .build_fill();

    // Use inspect_tx (not transact) so the inspector hooks get invoked
    let result = evm.inspect_tx(tx);

    match result {
        Ok(res) => {
            eprintln!("[evm] Execution result: gas_used={}", res.result.gas_used());
            let inspector = evm.into_inspector();
            Ok(inspector.trace)
        }
        Err(e) => Err(format!("EVM execution failed: {:?}", e)),
    }
}

/// Deploy a contract (init code) then call it, returning the call's trace.
pub fn deploy_and_call(init_code: &[u8], calldata: &[u8]) -> Result<EvmTraceColumns, String> {
    let caller_addr = Address::from([0xCA; 20]);

    let mut db = CacheDB::<EmptyDB>::default();
    db.insert_account_info(
        caller_addr,
        AccountInfo {
            balance: U256::from(1_000_000_000_000_000_000u64),
            nonce: 0,
            code_hash: Default::default(),
            account_id: None,
            code: None,
        },
    );

    // Phase 1: Deploy (no tracing)
    let ctx = Context::mainnet().with_db(db);
    let mut evm = ctx.build_mainnet();

    let deploy_tx = TxEnv::builder()
        .caller(caller_addr)
        .kind(TxKind::Create)
        .gas_limit(10_000_000)
        .gas_price(0)
        .value(U256::ZERO)
        .data(Bytes::copy_from_slice(init_code))
        .nonce(0)
        .build_fill();

    let deploy_result = evm.transact_commit(deploy_tx)
        .map_err(|e| format!("Deploy failed: {:?}", e))?;

    let contract_addr = match deploy_result {
        ExecutionResult::Success { output, .. } => {
            match output {
                Output::Create(_, Some(addr)) => addr,
                _ => return Err("Deploy did not return contract address".to_string()),
            }
        }
        ExecutionResult::Revert { output, .. } => {
            return Err(format!("Deploy reverted: {:?}", output));
        }
        ExecutionResult::Halt { reason, .. } => {
            return Err(format!("Deploy halted: {:?}", reason));
        }
    };

    eprintln!("[evm] Contract deployed at: {:?}", contract_addr);

    // Phase 2: Call with tracing
    let inspector = TracingInspector::new();
    let mut evm = evm.with_inspector(inspector);

    let call_tx = TxEnv::builder()
        .caller(caller_addr)
        .kind(TxKind::Call(contract_addr))
        .gas_limit(10_000_000)
        .gas_price(0)
        .value(U256::ZERO)
        .data(Bytes::copy_from_slice(calldata))
        .nonce(1)
        .build_fill();

    // Use inspect_tx to invoke inspector hooks
    let result = evm.inspect_tx(call_tx);

    match result {
        Ok(res) => {
            eprintln!("[evm] Call result: gas_used={}", res.result.gas_used());
            let inspector = evm.into_inspector();
            Ok(inspector.trace)
        }
        Err(e) => Err(format!("Call failed: {:?}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_simple_add() {
        // PUSH1 10, PUSH1 20, ADD, PUSH1 0, MSTORE, PUSH1 32, PUSH1 0, RETURN
        let bytecode = vec![
            0x60, 0x0A, // PUSH1 10
            0x60, 0x14, // PUSH1 20
            0x01,       // ADD
            0x60, 0x00, // PUSH1 0
            0x52,       // MSTORE
            0x60, 0x20, // PUSH1 32
            0x60, 0x00, // PUSH1 0
            0xF3,       // RETURN
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        assert!(trace.step.len() > 0, "Should have trace rows");
        eprintln!("[test] Traced {} steps", trace.step.len());
    }

    #[test]
    fn test_execute_and_prove() {
        use metavm_zkp::commitment;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove;
        use metavm_zkp::verifier::verify;
        use crate::constraints::EvmConstraintSystem;

        commitment::init();

        // PUSH1 3, PUSH1 5, ADD, STOP
        let bytecode = vec![
            0x60, 0x03, // PUSH1 3
            0x60, 0x05, // PUSH1 5
            0x01,       // ADD
            0x00,       // STOP
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        assert!(trace.step.len() >= 4, "Should have at least 4 steps");

        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove(&polys, &cs);
        let valid = verify(&proof, &cs);
        assert!(valid, "Proof should verify for valid EVM execution");
    }
}
