//! EVM execution wrapper that runs bytecode and captures a traced execution.

use crate::inspector::TracingInspector;
use crate::trace::EvmTraceColumns;
use revm::context::Context;
use revm::context::TxEnv;
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::handler::{ExecuteCommitEvm, MainBuilder, MainContext};
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
        Ok(_res) => {
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
        Ok(_res) => {
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

    /// Standalone diagnostic: run a real CREATE bytecode through the
    /// executor and prove + verify the resulting EVM trace WITHOUT any
    /// cross-AIR linkages. **CURRENTLY FAILS** (~2s) — the EVM main
    /// constraint system has a pre-existing bug in its handling of
    /// CREATE bytecode traces. Specifically, one of the CREATE-
    /// specific shifted constraints (depth_next, callee_next from
    /// hint, caller_next, return_pc_next, static_propagate; all
    /// gated by `sel_create`) doesn't hold on a real revm-produced
    /// CREATE row.
    ///
    /// No existing test in the repo proves a CREATE-bytecode trace
    /// standalone; the bug is exposed by this new test. Resolving
    /// it requires bisecting which CREATE shifted constraint actually
    /// fires — likely culprits: the post-CREATE frame transition's
    /// `caller_next` (might use a different limb encoding than
    /// expected), `return_pc_next` (revm might write a different
    /// return address than `pc + 1`), or `static_propagate` semantics.
    ///
    /// The 3-AIR `joint_prove_evm_create_address_e2e` test in
    /// `crates/evm/src/cross_air_linkage.rs` is blocked on this fix.
    #[test]
    #[ignore = "BLOCKED: pre-existing EVM constraint bug in CREATE bytecode \
                handling; see test doc for diagnosis path"]
    fn test_execute_and_prove_create_bytecode() {
        use metavm_zkp::commitment;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove;
        use metavm_zkp::verifier::verify;
        use crate::constraints::EvmConstraintSystem;

        commitment::init();

        // PUSH1 0, PUSH1 0, PUSH1 0, CREATE, STOP — minimal CREATE.
        let bytecode = vec![
            0x60, 0x00,
            0x60, 0x00,
            0x60, 0x00,
            0xF0,
            0x00,
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        eprintln!("[diag] CREATE trace produced {} rows", trace.step.len());

        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove(&polys, &cs);
        let valid = verify(&proof, &cs);
        assert!(valid, "Standalone EVM proof of CREATE bytecode must verify");
    }

    /// Diagnostic: dump the EVM trace columns at the CREATE row and
    /// the row immediately after, comparing them against the EVM
    /// constraint system's CREATE shifted-constraint expectations:
    ///   - depth_next - depth - 1 = 0
    ///   - callee_next[k] - create_address_hint[k] = 0 for k in 0..4
    ///   - caller_next[k] - frame_callee[k] = 0 for k in 0..4
    ///   - return_pc_next - pc - 1 = 0
    ///   - static_next - static = 0
    /// If any of these is non-zero on a real revm-produced trace, the
    /// EVM constraint system's expectation doesn't match revm's
    /// actual behavior — that's the source of the
    /// `test_execute_and_prove_create_bytecode` failure.
    /// Diagnostic: ADD bytecode constraint vector.
    #[test]
    fn diag_evaluate_constraints_on_add_bytecode_trace() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        use crate::constraints::EvmConstraintSystem;
        let bytecode = vec![0x60, 0x03, 0x60, 0x05, 0x01, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns: Vec<&Vec<_>> = polys.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&columns, polys.num_rows);
        let labels = cs.constraint_labels();
        eprintln!("[diag] ADD: {} bodies, {} rows", bodies.len(), polys.num_rows);
        for (b_idx, body) in bodies.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                if !val.is_zero() {
                    let label = labels.get(b_idx).map(|s| s.as_str()).unwrap_or("?");
                    eprintln!("[diag] FAIL constraint #{} ({}) at row {}", b_idx, label, r);
                }
            }
        }
    }

    /// Runs the EVM constraint system's evaluate_on_domain on a real
    /// CREATE bytecode trace and reports any constraint body that
    /// produces a non-zero scalar — pinpoints which constraint fires
    /// when the standalone CREATE bytecode prove/verify fails.
    /// Bisection probe: even ADD alone through `prove_with_scheme`.
    /// If FAILS → bug is in padding-row handling for ALL short EVM
    /// traces, not specific to any opcode. This is the critical
    /// follow-up since MSTORE8-alone fails.
    #[test]
    #[ignore = "slow: ~10 min release"]
    fn diag_add_alone_through_scheme() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::verifier::verify_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        // PUSH1 3; PUSH1 5; ADD; STOP — same as test_execute_and_prove
        let bytecode = vec![0x60, 0x03, 0x60, 0x05, 0x01, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove_with_scheme(&polys, &cs, &scheme);
        let valid = verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        eprintln!("[diag] ADD-alone scheme verify = {}", valid);
    }

    /// Bisection probe: MSTORE (32-byte word write, not MSTORE8) through
    /// `prove_with_scheme`. MSTORE was always in store_sels — so if this
    /// passes, the bug is MSTORE8-specific (likely in compute_evm_aux's
    /// byte-extraction path or the aux0 LogUp range). If this also
    /// fails, the memory permutation logic itself is broken when
    /// targeting addr=0.
    #[test]
    #[ignore = "slow: ~10 min release"]
    fn diag_mstore_alone_through_scheme() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::verifier::verify_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        // PUSH1 0xff; PUSH1 0x00 (offset=0); MSTORE; STOP (writes 32-byte word)
        let bytecode = vec![0x60, 0xff, 0x60, 0x00, 0x52, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove_with_scheme(&polys, &cs, &scheme);
        let valid = verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        eprintln!("[diag] MSTORE-alone scheme verify = {}", valid);
    }

    /// Bisection probe: MSTORE8 at OFFSET=1 (not 0) through `prove_with_scheme`.
    /// Tests the hypothesis that addr-collision with addr=0 dummies is the
    /// post-fix failure mode. If THIS PASSES → addr-0 collision confirmed,
    /// next fix needed is dummy-sentinel addressing.
    #[test]
    #[ignore = "slow: ~10 min release"]
    fn diag_mstore8_at_offset_1_through_scheme() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::verifier::verify_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        // PUSH1 0xab; PUSH1 0x01 (offset=1!); MSTORE8; STOP
        let bytecode = vec![0x60, 0xab, 0x60, 0x01, 0x53, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove_with_scheme(&polys, &cs, &scheme);
        let valid = verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        eprintln!("[diag] MSTORE8-at-offset-1 scheme verify = {}", valid);
    }

    /// Bisection probe: MSTORE8 alone through `prove_with_scheme`.
    /// If FAILS → bug is in memory permutation byte-write handling.
    #[test]
    #[ignore = "slow: ~10 min release"]
    fn diag_mstore8_alone_through_scheme() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::verifier::verify_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        // PUSH1 0xab; PUSH1 0; MSTORE8; STOP
        let bytecode = vec![0x60, 0xab, 0x60, 0x00, 0x53, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove_with_scheme(&polys, &cs, &scheme);
        let valid = verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        eprintln!("[diag] MSTORE8-alone scheme verify = {}", valid);
    }

    /// Bisection probe: SHA3 over zero-length memory through `prove_with_scheme`.
    /// If FAILS → bug is in SHA3 oracle / LogUp interaction.
    #[test]
    #[ignore = "slow: ~10 min release"]
    fn diag_sha3_alone_through_scheme() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::verifier::verify_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        // PUSH1 0; PUSH1 0; SHA3; STOP (hashing 0 bytes)
        let bytecode = vec![0x60, 0x00, 0x60, 0x00, 0x20, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove_with_scheme(&polys, &cs, &scheme);
        let valid = verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        eprintln!("[diag] SHA3-alone scheme verify = {}", valid);
    }

    /// Diagnostic: byte-diff the proof structures from legacy `prove`
    /// vs `prove_with_scheme` on the SAME SHA3 trace. Localizes which
    /// proof component diverges between the two prover paths.
    #[test]
    #[ignore = "slow: both prove paths on SHA3 bytecode; ~12 min total"]
    fn diag_byte_diff_legacy_vs_scheme_prove_sha3() {
        use metavm_zkp::commitment;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::{prove, prove_with_scheme};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;
        commitment::init();
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();

        let p1 = prove(&polys, &cs);
        let p2 = prove_with_scheme(&polys, &cs, &scheme);

        eprintln!("[diff] num_steps:               legacy={} scheme={}", p1.num_steps, p2.num_steps);
        eprintln!("[diff] domain_size:             legacy={} scheme={}", p1.domain_size, p2.domain_size);
        eprintln!("[diff] num_quotient_chunks:     legacy={} scheme={}", p1.num_quotient_chunks, p2.num_quotient_chunks);
        eprintln!("[diff] column_commitments len:  legacy={} scheme={}", p1.column_commitments.len(), p2.column_commitments.len());
        eprintln!("[diff] evaluations len:         legacy={} scheme={}", p1.evaluations.len(), p2.evaluations.len());
        eprintln!("[diff] shifted_evals len:       legacy={} scheme={}", p1.shifted_evaluations.len(), p2.shifted_evaluations.len());
        eprintln!("[diff] logup_commitments len:   legacy={} scheme={}", p1.logup_commitments.len(), p2.logup_commitments.len());
        eprintln!("[diff] perm_commitments len:    legacy={} scheme={}", p1.perm_commitments.len(), p2.perm_commitments.len());
        eprintln!("[diff] reg_perm_commitments:    legacy={} scheme={}", p1.reg_perm_commitments.len(), p2.reg_perm_commitments.len());
        eprintln!("[diff] frame_perm_commitment:   legacy={} scheme={}",
            p1.frame_perm_commitment.is_some(), p2.frame_perm_commitment.is_some());
        eprintln!("[diff] oracle_data len:         legacy={} scheme={}", p1.oracle_data.len(), p2.oracle_data.len());

        // Per-column commitment diff
        let mut diffs = 0;
        for k in 0..p1.column_commitments.len().min(p2.column_commitments.len()) {
            if p1.column_commitments[k].0 != p2.column_commitments[k].0 {
                if diffs < 5 {
                    eprintln!("[diff] column_commitment[{}] DIFFERS (lengths {} vs {})",
                        k, p1.column_commitments[k].0.len(), p2.column_commitments[k].0.len());
                }
                diffs += 1;
            }
        }
        eprintln!("[diff] total column_commitments that differ: {}", diffs);

        // Per-evaluation diff
        let mut e_diffs = 0;
        for k in 0..p1.evaluations.len().min(p2.evaluations.len()) {
            if p1.evaluations[k] != p2.evaluations[k] {
                if e_diffs < 5 {
                    eprintln!("[diff] evaluations[{}] DIFFERS", k);
                }
                e_diffs += 1;
            }
        }
        eprintln!("[diff] total evaluations that differ: {}", e_diffs);
    }

    /// Diagnostic: cross-verify — prove via LEGACY path, verify via
    /// SCHEME path. Localizes whether the SHA3-bytecode failure is in
    /// `prove_with_scheme` or `verify_with_scheme`. If this PASSES, the
    /// bug is in prove_with_scheme. If FAILS, bug is in verify_with_scheme.
    #[test]
    #[ignore = "slow: BLS48-581 verify takes ~5min"]
    fn diag_cross_verify_legacy_prove_scheme_verify_sha3() {
        use metavm_zkp::commitment;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove;
        use metavm_zkp::verifier::verify_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;
        commitment::init();
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove(&polys, &cs);
        let valid_scheme = verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        eprintln!("[diag] legacy-prove proof + scheme-verify = {}", valid_scheme);
        // No assertion — diagnostic only. The result tells us which path is broken.
    }

    /// Diagnostic: SHA3 bytecode through legacy `prove`+`verify` path
    /// (in-memory commitment, not BLS48-581 scheme). If this PASSES, the
    /// bug is scheme-specific. If FAILS, the regression is in the EVM
    /// constraint system itself. Runs in ~60-90s vs ~10 min for the
    /// scheme path.
    #[test]
    fn diag_legacy_prove_verify_sha3_bytecode() {
        use metavm_zkp::commitment;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove;
        use metavm_zkp::verifier::verify;
        use crate::constraints::EvmConstraintSystem;
        commitment::init();
        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        eprintln!("[diag] legacy SHA3: {} trace rows", trace.step.len());
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove(&polys, &cs);
        let valid = verify(&proof, &cs);
        eprintln!("[diag] legacy verify = {}", valid);
        assert!(valid, "SHA3 bytecode must verify under legacy prove+verify path");
    }

    /// Diagnostic: evaluate EVM constraints on the FULLY PADDED MSTORE8
    /// trace (4 real rows + 252 padding rows = 256), to check if any
    /// row-local constraint fires on a padding row that the basic
    /// constraint diagnostic (which uses num_rows = 16) missed.
    #[test]
    fn diag_evaluate_constraints_on_padded_mstore8_trace() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        use crate::constraints::EvmConstraintSystem;
        let bytecode = vec![0x60, 0xab, 0x60, 0x00, 0x53, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let mut polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        // Pad to 256 manually (matching what the prover does for LogUp).
        let zero = metavm_zkp::field::Scalar::zero(CurveType::Bls48581);
        for poly in polys.columns.iter_mut() {
            poly.evaluations.resize(256, zero.clone());
        }
        polys.padded_size = 256;
        // Apply the same selector / padding fixups the prover applies.
        let mut col_evals: Vec<Vec<_>> = polys.columns.iter()
            .map(|p| p.evaluations.clone()).collect();
        // Set padding_selector_column = 1 on padding rows (sel_stop).
        if let Some(pad_col) = cs.padding_selector_column() {
            let one = metavm_zkp::field::Scalar::one(CurveType::Bls48581);
            for i in polys.num_rows..256 {
                col_evals[pad_col][i] = one.clone();
            }
        }
        cs.fix_trace_padding(&mut col_evals, polys.num_rows, 256);
        let columns: Vec<&Vec<_>> = col_evals.iter().collect();
        let bodies = cs.evaluate_on_domain(&columns, 256);
        let labels = cs.constraint_labels();
        eprintln!("[diag] padded MSTORE8 trace: {} bodies, padded to 256",
            bodies.len());
        let mut any_failed = false;
        for (b_idx, body) in bodies.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                if !val.is_zero() {
                    let label = labels.get(b_idx).map(|s| s.as_str()).unwrap_or("?");
                    let opcode = trace.opcode.get(r).copied().unwrap_or(0);
                    eprintln!("[diag] FAIL #{} ({}) row {} opcode={:#x}",
                        b_idx, label, r, opcode);
                    any_failed = true;
                    if r > polys.num_rows { break; }  // only show first padding failure
                }
            }
        }
        if !any_failed {
            eprintln!("[diag] all row-local constraints vanish on padded MSTORE8 — bug is in cross-row / LogUp / perm / quotient");
        }
    }

    /// Diagnostic: evaluate EVM constraint system on the SHA3 bytecode
    /// that fails standalone prove+verify. Pinpoints which constraint
    /// body has a non-zero scalar (the regression we need to fix for
    /// the A1b chain validation).
    #[test]
    fn diag_evaluate_constraints_on_sha3_bytecode_trace() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        use crate::constraints::EvmConstraintSystem;
        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns: Vec<&Vec<_>> = polys.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&columns, polys.num_rows);
        let labels = cs.constraint_labels();
        eprintln!("[diag] SHA3 trace: {} bodies, {} rows, padded={}",
            bodies.len(), polys.num_rows, polys.padded_size);
        let mut any_failed = false;
        for (b_idx, body) in bodies.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                if !val.is_zero() {
                    let label = labels.get(b_idx).map(|s| s.as_str()).unwrap_or("?");
                    eprintln!("[diag] FAIL constraint #{} ({}) row {} opcode={:#x}",
                        b_idx, label, r, trace.opcode.get(r).copied().unwrap_or(0));
                    any_failed = true;
                }
            }
        }
        if !any_failed {
            eprintln!("[diag] all constraints vanish on honest trace — failure must be in proof/verify protocol (LogUp/perm/shifted/transcript), not row-local constraints");
        }
    }

    #[test]
    fn diag_evaluate_constraints_on_create_bytecode_trace() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        use crate::constraints::EvmConstraintSystem;
        let bytecode = vec![0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF0, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns: Vec<&Vec<_>> = polys.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&columns, polys.num_rows);
        let labels = cs.constraint_labels();
        eprintln!("[diag] {} constraint bodies, {} rows (padded), {} num_rows",
            bodies.len(), polys.padded_size, polys.num_rows);
        for (b_idx, body) in bodies.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                if !val.is_zero() {
                    let label = labels.get(b_idx).map(|s| s.as_str()).unwrap_or("?");
                    eprintln!("[diag] constraint #{} ({}) NON-ZERO at row {}: {:?}",
                        b_idx, label, r, val.to_bytes());
                }
            }
        }
    }

    /// Standalone EVM CREATE bytecode prove/verify via the BLS48-581
    /// commitment scheme path (the same path joint_prove uses
    /// internally). If this passes, the EVM constraints work; any
    /// 3-AIR e2e failure is in cross_air_logup itself.
    #[test]
    #[ignore = "slow: ~15 min full BLS48-581 EVM proof"]
    fn test_execute_and_prove_create_bytecode_bls48581_scheme() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::verifier::verify_with_scheme;
        use crate::constraints::EvmConstraintSystem;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let bytecode = vec![0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF0, 0x00];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let proof = prove_with_scheme(&polys, &cs, &scheme);
        let valid = verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        assert!(valid, "BLS48-581 scheme EVM proof of CREATE bytecode must verify");
    }

    #[test]
    fn diag_dump_create_row_shifted_constraints() {
        let bytecode = vec![
            0x60, 0x00,
            0x60, 0x00,
            0x60, 0x00,
            0xF0,       // CREATE
            0x00,       // STOP
        ];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let n = trace.step.len();
        eprintln!("[diag] trace has {} rows", n);
        let create_row = (0..n)
            .find(|&i| trace.opcode[i] == 0xF0)
            .expect("CREATE row must exist");
        eprintln!("[diag] CREATE row index: {}", create_row);
        assert!(create_row + 1 < n, "must have a row after CREATE");
        let nx = create_row + 1;
        eprintln!("[diag] row {} (CREATE):  pc={} depth={} callee={:?} static={}",
            create_row,
            trace.pc[create_row],
            trace.frame_depth[create_row],
            [trace.frame_callee[0][create_row], trace.frame_callee[1][create_row],
             trace.frame_callee[2][create_row], trace.frame_callee[3][create_row]],
            trace.frame_static[create_row]);
        eprintln!("[diag] row {} (after):   pc={} depth={} callee={:?} caller={:?} static={} return_pc={}",
            nx,
            trace.pc[nx],
            trace.frame_depth[nx],
            [trace.frame_callee[0][nx], trace.frame_callee[1][nx],
             trace.frame_callee[2][nx], trace.frame_callee[3][nx]],
            [trace.frame_caller[0][nx], trace.frame_caller[1][nx],
             trace.frame_caller[2][nx], trace.frame_caller[3][nx]],
            trace.frame_static[nx],
            trace.frame_return_pc[nx]);
        eprintln!("[diag] CREATE row create_address_hint={:?}",
            [trace.create_address_hint[0][create_row], trace.create_address_hint[1][create_row],
             trace.create_address_hint[2][create_row], trace.create_address_hint[3][create_row]]);

        // Check each shifted constraint:
        let depth_diff: i64 = trace.frame_depth[nx] as i64
            - trace.frame_depth[create_row] as i64 - 1;
        eprintln!("[diag] depth_next - depth - 1 = {} (expected 0)", depth_diff);

        for k in 0..4 {
            let cl = trace.frame_callee[k][nx] as i64
                - trace.create_address_hint[k][create_row] as i64;
            eprintln!("[diag] callee_next[{}] - hint[{}] = {} (expected 0)", k, k, cl);
        }
        for k in 0..4 {
            let cl = trace.frame_caller[k][nx] as i64
                - trace.frame_callee[k][create_row] as i64;
            eprintln!("[diag] caller_next[{}] - frame_callee[{}] = {} (expected 0)", k, k, cl);
        }
        let rpc_diff: i64 = trace.frame_return_pc[nx] as i64
            - trace.pc[create_row] as i64 - 1;
        eprintln!("[diag] return_pc_next - pc - 1 = {} (expected 0)", rpc_diff);
        let static_diff: i64 = trace.frame_static[nx] as i64
            - trace.frame_static[create_row] as i64;
        eprintln!("[diag] static_next - static = {} (expected 0)", static_diff);

        // Dump every row's relevant fields for full context.
        eprintln!("[diag] Full row dump:");
        for i in 0..n {
            eprintln!("  row {}: opcode=0x{:02x} pc={} next_pc={} depth={} sel_create={} sel_call={} sel_call_push={} sel_call_return={} sel_stop={}",
                i, trace.opcode[i], trace.pc[i], trace.next_pc[i],
                trace.frame_depth[i],
                trace.sel_create[i], trace.sel_call[i],
                trace.sel_call_push_frame[i], trace.sel_call_return[i],
                trace.sel_stop[i]);
        }
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
