//! Prove correct execution of an EVM smart contract call.
//!
//! Usage: prove-evm [--scheme bls12381|bls48581|bls48581-fast]
//!
//! Deploys a simple counter contract, calls it, and produces a ZK proof.

use metavm_evm::constraints::EvmConstraintSystem;
use metavm_evm::executor::execute_bytecode;
use metavm_evm::trace::evm_trace_polys_with_curve;
use metavm_zkp::field::CurveType;
use metavm_zkp::prover::prove_chunk_with_scheme;
use metavm_zkp::recursive::{begin_chunk_scheme, verify_final_scheme};
use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
use metavm_zkp::scheme::bls48581_scheme::{Bls48581Scheme, Bls48581SchemeFast};
use metavm_zkp::scheme::CommitmentScheme;
use std::sync::Arc;
use std::time::Instant;
use std::{env, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut scheme_name = "bls12381".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scheme" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --scheme");
                    process::exit(1);
                }
                scheme_name = args[i].clone();
                if scheme_name != "bls12381"
                    && scheme_name != "bls48581"
                    && scheme_name != "bls48581-fast"
                {
                    eprintln!(
                        "Invalid scheme '{}', must be 'bls12381', 'bls48581', or 'bls48581-fast'",
                        scheme_name
                    );
                    process::exit(1);
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: prove-evm [--scheme bls12381|bls48581|bls48581-fast]");
                process::exit(0);
            }
            other => {
                eprintln!("Unexpected argument: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("[prove-evm] EVM Smart Contract ZK Proof Demo");
    eprintln!("[prove-evm] ─────────────────────────────────");
    eprintln!("[prove-evm] Scheme: {}", scheme_name);

    // Counter contract: SLOAD slot 0, ADD 1, SSTORE slot 0, STOP
    let contract_bytecode = vec![
        0x60, 0x00, // PUSH1 0 (storage slot)
        0x54, // SLOAD
        0x60, 0x01, // PUSH1 1
        0x01, // ADD
        0x60, 0x00, // PUSH1 0 (storage slot)
        0x55, // SSTORE
        0x00, // STOP
    ];

    eprintln!("[prove-evm] Contract: counter increment (SLOAD, ADD 1, SSTORE)");
    eprintln!("[prove-evm] Bytecode: {} bytes", contract_bytecode.len());

    // Execute
    eprintln!("[prove-evm] Executing contract...");
    let exec_start = Instant::now();
    let trace = execute_bytecode(&contract_bytecode, &[]).unwrap_or_else(|e| {
        eprintln!("[prove-evm] Execution failed: {}", e);
        process::exit(1);
    });
    let exec_elapsed = exec_start.elapsed();
    eprintln!(
        "[prove-evm] Execution: {} steps in {:.3}s",
        trace.step.len(),
        exec_elapsed.as_secs_f64()
    );

    // Initialize commitment scheme
    let (scheme, curve_type): (Arc<dyn CommitmentScheme>, CurveType) = match scheme_name.as_str() {
        "bls48581" => {
            let s = Bls48581Scheme;
            eprintln!("[prove-evm] Initializing BLS48-581...");
            s.init();
            eprintln!("[prove-evm] BLS48-581 ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        "bls48581-fast" => {
            let s = Bls48581SchemeFast;
            eprintln!("[prove-evm] Initializing BLS48-581 (fast)...");
            s.init();
            eprintln!("[prove-evm] BLS48-581 (fast) ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        _ => {
            let s = Bls12381Scheme;
            eprintln!("[prove-evm] Initializing BLS12-381...");
            s.init();
            eprintln!("[prove-evm] BLS12-381 ready");
            (Arc::new(s), CurveType::Bls12381)
        }
    };

    // Build polynomials
    let polys = evm_trace_polys_with_curve(&trace, curve_type);
    let cs = EvmConstraintSystem::new();

    // Prove as single chunk
    eprintln!("[prove-evm] Proving...");
    let prove_start = Instant::now();
    let zero_hash = [0u8; 32];
    let chunk_proof = prove_chunk_with_scheme(&polys, &cs, 0, &zero_hash, &zero_hash, &*scheme);
    let prove_elapsed = prove_start.elapsed();
    eprintln!(
        "[prove-evm] Proof generated in {:.3}s",
        prove_elapsed.as_secs_f64()
    );

    // Serialize the chunk proof + emit a one-line size summary plus a
    // LayerChainProof commitment that binds this proof into the
    // recursive-fold envelope's public-input root. The envelope
    // dispatch (`LayerChainProof::verify_with_layer_verifier`) will
    // recover the same commitment for any caller that consumes the
    // serialized bytes.
    let chunk_bytes = chunk_proof.to_bytes();
    eprintln!(
        "[prove-evm] Serialized ChunkProof: {} bytes",
        chunk_bytes.len()
    );
    {
        use metavm_zkp::layer_chain::{
            ChainBoundaries, LayerChain, LayerChainProof, LayerProof, LayerProofKind,
        };
        // Synthetic boundary values — a real production caller would
        // supply boundary values derived from this transaction's block
        // and the beacon-chain context. Using deterministic synthetic
        // values here just demonstrates the envelope wiring.
        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: 1,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 0 {
                    LayerProof::with_proof(
                        claim,
                        LayerProofKind::VmEvm,
                        chunk_bytes.clone(),
                    )
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);
        if let Some(commitment) = chain_proof.commitment() {
            let mut hex = String::with_capacity(64);
            for b in &commitment {
                hex.push_str(&format!("{:02x}", b));
            }
            eprintln!("[prove-evm] LayerChain commitment: 0x{}", hex);
        } else {
            eprintln!("[prove-evm] LayerChain commitment: <malformed>");
        }
        eprintln!(
            "[prove-evm] LayerChain total proof bytes: {}",
            chain_proof.total_proof_bytes()
        );
    }

    // Wrap in recursive proof and verify.
    let recursive_proof = begin_chunk_scheme(chunk_proof, &*scheme, curve_type);
    let valid = verify_final_scheme(&recursive_proof, &*scheme);
    eprintln!("[prove-evm] Recursive proof verified: {}", valid);
    eprintln!(
        "[prove-evm] Steps: {}, Domain: {}",
        recursive_proof.current_proof.num_steps, recursive_proof.current_proof.domain_size
    );

    let total = exec_start.elapsed();
    eprintln!("[prove-evm] Total: {:.3}s", total.as_secs_f64());
    eprintln!(
        "[prove-evm] Result: {}",
        if valid { "SUCCESS" } else { "FAILED" }
    );
}
