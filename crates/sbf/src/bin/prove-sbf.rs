//! Prove correct execution of an SBF/BPF program.
//!
//! Usage: prove-sbf <elf-file> [--output proof.bin]
//!        prove-sbf --asm "<assembly code>"
//!        [--scheme bls12381|bls48581|bls48581-fast]
//!
//! Executes the program and produces a ZK proof of correct execution.

use metavm_sbf::constraints::SbfConstraintSystem;
use metavm_sbf::executor;
use metavm_sbf::trace::sbf_trace_polys_with_curve;
use metavm_zkp::field::CurveType;
use metavm_zkp::prover::prove_chunk_with_scheme;
use metavm_zkp::recursive::{begin_chunk_scheme, verify_final_scheme};
use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
use metavm_zkp::scheme::bls48581_scheme::{Bls48581Scheme, Bls48581SchemeFast};
use metavm_zkp::scheme::CommitmentScheme;
use std::sync::Arc;
use std::time::Instant;
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: prove-sbf <elf-file> [--output proof.bin] [--scheme bls12381|bls48581|bls48581-fast]");
        eprintln!("       prove-sbf --asm \"<assembly code>\" [--scheme ...]");
        eprintln!();
        eprintln!("Prove correct execution of an SBF/BPF program.");
        process::exit(1);
    }

    let mut elf_path: Option<String> = None;
    let mut asm_code: Option<String> = None;
    let mut _output_path = "sbf_proof.bin".to_string();
    let mut scheme_name = "bls12381".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                _output_path = args[i].clone();
            }
            "--asm" => {
                i += 1;
                asm_code = Some(args[i].clone());
            }
            "--scheme" => {
                i += 1;
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
            path => {
                if elf_path.is_none() {
                    elf_path = Some(path.to_string());
                } else {
                    eprintln!("Unexpected argument: {}", path);
                    process::exit(1);
                }
            }
        }
        i += 1;
    }

    eprintln!("[prove-sbf] SBF/BPF ZK Proof Demo");
    eprintln!("[prove-sbf] ─────────────────────");
    eprintln!("[prove-sbf] Scheme: {}", scheme_name);

    // Execute
    let exec_start = Instant::now();
    let trace = if let Some(asm) = &asm_code {
        eprintln!("[prove-sbf] Assembling and executing: {}", asm);
        executor::execute_sbf_asm(asm, &[]).unwrap_or_else(|e| {
            eprintln!("[prove-sbf] Execution failed: {}", e);
            process::exit(1);
        })
    } else if let Some(path) = &elf_path {
        eprintln!("[prove-sbf] Loading ELF: {}", path);
        let elf_data = fs::read(path).unwrap_or_else(|e| {
            eprintln!("Failed to read ELF: {}", e);
            process::exit(1);
        });
        eprintln!("[prove-sbf] ELF size: {} bytes", elf_data.len());
        executor::execute_sbf_elf(&elf_data, &[]).unwrap_or_else(|e| {
            eprintln!("[prove-sbf] Execution failed: {}", e);
            process::exit(1);
        })
    } else {
        // Default demo: simple add program
        eprintln!("[prove-sbf] Running default demo: r0 = 10 + 20");
        let asm = "mov64 r1, 10\nmov64 r2, 20\nadd64 r1, r2\nmov64 r0, r1\nexit";
        executor::execute_sbf_asm(asm, &[]).unwrap_or_else(|e| {
            eprintln!("[prove-sbf] Execution failed: {}", e);
            process::exit(1);
        })
    };

    let exec_elapsed = exec_start.elapsed();
    eprintln!(
        "[prove-sbf] Execution: {} steps in {:.3}s",
        trace.step.len(),
        exec_elapsed.as_secs_f64()
    );

    // Initialize commitment scheme
    let (scheme, curve_type): (Arc<dyn CommitmentScheme>, CurveType) = match scheme_name.as_str() {
        "bls48581" => {
            let s = Bls48581Scheme;
            eprintln!("[prove-sbf] Initializing BLS48-581...");
            s.init();
            eprintln!("[prove-sbf] BLS48-581 ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        "bls48581-fast" => {
            let s = Bls48581SchemeFast;
            eprintln!("[prove-sbf] Initializing BLS48-581 (fast)...");
            s.init();
            eprintln!("[prove-sbf] BLS48-581 (fast) ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        _ => {
            let s = Bls12381Scheme;
            eprintln!("[prove-sbf] Initializing BLS12-381...");
            s.init();
            eprintln!("[prove-sbf] BLS12-381 ready");
            (Arc::new(s), CurveType::Bls12381)
        }
    };

    // Build polynomials
    let polys = sbf_trace_polys_with_curve(&trace, curve_type);
    let cs = SbfConstraintSystem::new();

    // Prove as single chunk
    eprintln!("[prove-sbf] Proving...");
    let prove_start = Instant::now();
    let zero_hash = [0u8; 32];
    let chunk_proof = prove_chunk_with_scheme(&polys, &cs, 0, &zero_hash, &zero_hash, &*scheme);
    let prove_elapsed = prove_start.elapsed();
    eprintln!(
        "[prove-sbf] Proof generated in {:.3}s",
        prove_elapsed.as_secs_f64()
    );

    // Wrap in recursive proof and verify
    let recursive_proof = begin_chunk_scheme(chunk_proof, &*scheme, curve_type);
    let valid = verify_final_scheme(&recursive_proof, &*scheme);
    eprintln!("[prove-sbf] Proof verified: {}", valid);
    eprintln!(
        "[prove-sbf] Steps: {}, Domain: {}",
        recursive_proof.current_proof.num_steps, recursive_proof.current_proof.domain_size
    );

    let total = exec_start.elapsed();
    eprintln!("[prove-sbf] Total: {:.3}s", total.as_secs_f64());
    eprintln!(
        "[prove-sbf] Result: {}",
        if valid { "SUCCESS" } else { "FAILED" }
    );
}
