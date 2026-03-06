//! Prove correct execution of a bare-metal RISC-V ELF.
//!
//! Usage: prove-elf <elf-file> [--chunk-size N] [--max-steps N] [--output proof.bin] [--workers N]

use metavm_riscv::chunked_trace::StreamingProver;
use metavm_riscv::constraints::{RiscvConstraintSystem, trace_polys_from_columns_with_curve};
use metavm_riscv::loader::load_elf_into_vm;
use metavm_riscv::trace::TraceColumns;
use metavm_zkp::field::CurveType;
use metavm_zkp::prover::{prove_chunk_with_scheme, ChunkProof};
use metavm_zkp::scheme::CommitmentScheme;
use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
use metavm_zkp::scheme::bls48581_scheme::{Bls48581Scheme, Bls48581SchemeFast};
use metavm_zkp::tree_fold::tree_fold_with_progress_scheme;
use metavm_zkp::recursive::verify_final_scheme;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use std::{env, fs, process, thread};

// ── Ctrl+C handling ─────────────────────────────────────────────────────────

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}

extern "C" fn handle_sigint(_sig: i32) {
    if STOP_REQUESTED.load(Ordering::SeqCst) {
        // Second Ctrl+C: abort immediately
        std::process::abort();
    }
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

/// Work item sent from the main thread to worker threads.
struct ProveWork {
    chunk_index: u64,
    columns: TraceColumns,
    initial_state_hash: [u8; 32],
    final_state_hash: [u8; 32],
}

/// Bounded work queue with backpressure.
struct WorkQueue {
    queue: Mutex<VecDeque<Option<ProveWork>>>,
    not_empty: Condvar,
    not_full: Condvar,
    capacity: usize,
}

impl WorkQueue {
    fn new(capacity: usize) -> Self {
        WorkQueue {
            queue: Mutex::new(VecDeque::new()),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            capacity,
        }
    }

    /// Push a work item, blocking if the queue is full (backpressure).
    fn push(&self, item: Option<ProveWork>) {
        let mut q = self.queue.lock().unwrap();
        while q.len() >= self.capacity {
            q = self.not_full.wait(q).unwrap();
        }
        q.push_back(item);
        self.not_empty.notify_one();
    }

    /// Pop a work item, blocking if the queue is empty.
    /// Returns None when a poison pill is received (worker should exit).
    fn pop(&self) -> Option<ProveWork> {
        let mut q = self.queue.lock().unwrap();
        loop {
            if let Some(item) = q.pop_front() {
                self.not_full.notify_one();
                return item;
            }
            q = self.not_empty.wait(q).unwrap();
        }
    }
}

/// Iterator that receives ChunkProofs from workers (possibly out of order)
/// and yields them in sequential chunk_index order.
struct ReorderIterator {
    rx: mpsc::Receiver<ChunkProof>,
    buf: BTreeMap<u64, ChunkProof>,
    next: u64,
}

impl ReorderIterator {
    fn new(rx: mpsc::Receiver<ChunkProof>) -> Self {
        ReorderIterator {
            rx,
            buf: BTreeMap::new(),
            next: 0,
        }
    }
}

impl Iterator for ReorderIterator {
    type Item = ChunkProof;

    fn next(&mut self) -> Option<ChunkProof> {
        loop {
            if let Some(proof) = self.buf.remove(&self.next) {
                self.next += 1;
                return Some(proof);
            }
            match self.rx.recv() {
                Ok(proof) => {
                    self.buf.insert(proof.chunk_index, proof);
                }
                Err(_) => {
                    // Channel closed — drain remaining buffered items in order
                    return self.buf.remove(&self.next).map(|p| {
                        self.next += 1;
                        p
                    });
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: prove-elf <elf-file> [--chunk-size N] [--max-steps N] [--output proof.bin] [--workers N] [--scheme bls12381|bls48581|bls48581-fast]");
        eprintln!();
        eprintln!("Prove correct execution of a bare-metal RISC-V ELF binary.");
        eprintln!("The ELF runs in non-privileged mode (no OS, no devices).");
        eprintln!("Execution halts on ECALL (syscall 0 or 93).");
        process::exit(1);
    }

    let mut elf_path = String::new();
    let mut chunk_size: usize = 128;
    let mut output_path = "elf_proof.bin".to_string();
    let mut max_steps: u64 = 0;
    let mut num_workers: usize = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut scheme_name = "bls12381".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--chunk-size" => {
                i += 1;
                chunk_size = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid chunk size");
                    process::exit(1);
                });
            }
            "--output" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--max-steps" => {
                i += 1;
                max_steps = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid max steps");
                    process::exit(1);
                });
            }
            "--workers" => {
                i += 1;
                num_workers = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid worker count");
                    process::exit(1);
                });
                if num_workers == 0 {
                    num_workers = 1;
                }
            }
            "--scheme" => {
                i += 1;
                scheme_name = args[i].clone();
                if scheme_name != "bls12381" && scheme_name != "bls48581" && scheme_name != "bls48581-fast" {
                    eprintln!("Invalid scheme '{}', must be 'bls12381', 'bls48581', or 'bls48581-fast'", scheme_name);
                    process::exit(1);
                }
            }
            path => {
                if elf_path.is_empty() {
                    elf_path = path.to_string();
                } else {
                    eprintln!("Unexpected argument: {}", path);
                    process::exit(1);
                }
            }
        }
        i += 1;
    }

    if elf_path.is_empty() {
        eprintln!("Error: ELF path is required");
        process::exit(1);
    }

    // Load ELF
    eprintln!("[prove-elf] Loading ELF: {}", elf_path);
    let elf_data = fs::read(&elf_path).unwrap_or_else(|e| {
        eprintln!("Failed to read ELF: {}", e);
        process::exit(1);
    });
    eprintln!("[prove-elf] ELF size: {} bytes", elf_data.len());

    let vm = load_elf_into_vm(&elf_data, None).unwrap_or_else(|e| {
        eprintln!("Failed to load ELF: {}", e);
        process::exit(1);
    });
    eprintln!("[prove-elf] VM initialized, entry=0x{:x}, chunk_size={}", vm.cpu.pc, chunk_size);

    // Initialize commitment scheme
    let (scheme, curve_type): (Arc<dyn CommitmentScheme>, CurveType) = match scheme_name.as_str() {
        "bls48581" => {
            let s = Bls48581Scheme;
            eprintln!("[prove-elf] Initializing BLS48-581...");
            s.init();
            eprintln!("[prove-elf] BLS48-581 ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        "bls48581-fast" => {
            let s = Bls48581SchemeFast;
            eprintln!("[prove-elf] Initializing BLS48-581 (fast)...");
            s.init();
            eprintln!("[prove-elf] BLS48-581 (fast) ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        _ => {
            let s = Bls12381Scheme;
            eprintln!("[prove-elf] Initializing BLS12-381...");
            s.init();
            eprintln!("[prove-elf] BLS12-381 ready");
            (Arc::new(s), CurveType::Bls12381)
        }
    };

    // Clamp chunk size to scheme's max domain size
    let max_domain = scheme.max_domain_size() as usize;
    if chunk_size > max_domain {
        eprintln!("[prove-elf] Warning: chunk_size {} exceeds SRS limit {}, clamping", chunk_size, max_domain);
        chunk_size = max_domain;
    }

    // Create streaming prover
    let mut prover = StreamingProver::new(vm, chunk_size);
    prover.initialize_memory_hash();

    // Install Ctrl+C handler for graceful shutdown
    unsafe { signal(2 /* SIGINT */, handle_sigint); }

    // Execute and collect chunks
    eprintln!("[prove-elf] Executing...");
    let exec_start = Instant::now();
    let mut in_memory_chunks = Vec::new();

    while !prover.is_halted() {
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            eprintln!();
            eprintln!("[prove-elf] Ctrl+C received, stopping execution");
            break;
        }

        if max_steps > 0 && prover.global_step() >= max_steps {
            eprintln!("[prove-elf] Reached max steps limit ({})", max_steps);
            break;
        }

        let chunk = match prover.execute_chunk() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[prove-elf] VM halted at step {}: {}", prover.global_step(), e);
                break;
            }
        };

        if chunk.num_steps > 0 {
            in_memory_chunks.push(chunk);
        }
    }

    let exec_elapsed = exec_start.elapsed();
    eprintln!("[prove-elf] Execution: {} steps, {} chunks in {:.2}s",
        prover.global_step(), in_memory_chunks.len(), exec_elapsed.as_secs_f64());

    if let Some(code) = prover.vm.exit_code {
        eprintln!("[prove-elf] Exit code (a0): 0x{:x} ({})", code, code);
    }

    if in_memory_chunks.is_empty() {
        eprintln!("[prove-elf] No chunks to prove, exiting");
        process::exit(1);
    }

    // =======================================================================
    // Proving phase — parallel worker pool
    // =======================================================================
    let constraints = Arc::new(RiscvConstraintSystem::full());
    let prove_start = Instant::now();
    let chunks_executed = in_memory_chunks.len() as u64;

    // Reset Ctrl+C flag so users can force-abort during proving
    STOP_REQUESTED.store(false, Ordering::SeqCst);

    eprintln!("[prove-elf] ─── Proving {} chunks with {} workers ─────────────────",
        chunks_executed, num_workers);

    // Shared counter for progress reporting
    let chunks_proved = Arc::new(AtomicU64::new(0));
    let total_to_prove = chunks_executed;

    // Work queue: main thread → workers (bounded for backpressure)
    let work_queue = Arc::new(WorkQueue::new(num_workers * 2));

    // Result channel: workers → folder thread
    let (result_tx, result_rx) = mpsc::channel::<ChunkProof>();

    // Spawn worker threads
    let mut worker_handles = Vec::new();
    for worker_id in 0..num_workers {
        let wq = Arc::clone(&work_queue);
        let tx = result_tx.clone();
        let cs = Arc::clone(&constraints);
        let proved_counter = Arc::clone(&chunks_proved);
        let total = total_to_prove;
        let scheme_ref = Arc::clone(&scheme);
        let ct = curve_type;

        let handle = thread::Builder::new()
            .name(format!("prover-{}", worker_id))
            .spawn(move || {
                while let Some(work) = wq.pop() {
                    let polys = trace_polys_from_columns_with_curve(&work.columns, ct);
                    let chunk_proof = prove_chunk_with_scheme(
                        &polys,
                        cs.as_ref(),
                        work.chunk_index,
                        &work.initial_state_hash,
                        &work.final_state_hash,
                        &*scheme_ref,
                    );
                    let done = proved_counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 100 == 0 || done == total {
                        eprintln!("[prove-elf] proved {}/{} chunks ({:.1}%)",
                            done, total, done as f64 / total as f64 * 100.0);
                    }
                    if tx.send(chunk_proof).is_err() {
                        break; // Folder thread gone
                    }
                }
            })
            .unwrap_or_else(|e| {
                eprintln!("[prove-elf] Failed to spawn worker {}: {}", worker_id, e);
                process::exit(1);
            });
        worker_handles.push(handle);
    }
    // Drop the main thread's sender clone so the channel closes when workers finish
    drop(result_tx);

    // Spawn folder thread: reorders results and feeds tree_fold
    let folder_scheme = Arc::clone(&scheme);
    let folder_curve = curve_type;
    let folder_handle = thread::Builder::new()
        .name("folder".to_string())
        .spawn(move || {
            let reorder = ReorderIterator::new(result_rx);
            tree_fold_with_progress_scheme(
                reorder,
                Some(total_to_prove),
                |count, _total| {
                    if count % 100 == 0 {
                        eprintln!("[prove-elf]   folded {} chunks", count);
                    }
                },
                &*folder_scheme,
                folder_curve,
            )
        })
        .unwrap_or_else(|e| {
            eprintln!("[prove-elf] Failed to spawn folder thread: {}", e);
            process::exit(1);
        });

    // Feed chunks to the work queue
    for chunk in in_memory_chunks.drain(..) {
        work_queue.push(Some(ProveWork {
            chunk_index: chunk.chunk_index,
            columns: chunk.columns,
            initial_state_hash: chunk.initial_state_hash,
            final_state_hash: chunk.final_state_hash,
        }));
    }

    // Send poison pills to workers
    for _ in 0..num_workers {
        work_queue.push(None);
    }

    for handle in worker_handles {
        let _ = handle.join();
    }

    eprintln!(
        "[prove-elf] All {} chunks proved",
        chunks_proved.load(Ordering::Relaxed),
    );

    // Wait for the folder thread to complete tree folding
    let fold_result = folder_handle.join().unwrap_or_else(|_| {
        eprintln!("[prove-elf] Folder thread panicked");
        process::exit(1);
    });

    let final_proof = fold_result.unwrap_or_else(|e| {
        eprintln!("[prove-elf] Tree fold failed: {}", e);
        process::exit(1);
    });

    let prove_elapsed = prove_start.elapsed();

    // Verify
    let valid = verify_final_scheme(&final_proof, &*scheme);
    eprintln!("[prove-elf] Proof verified: {}", valid);
    eprintln!("[prove-elf] Depth: {}", final_proof.depth);

    if let Some(ref h) = final_proof.initial_state_hash {
        eprintln!("[prove-elf] Initial state: {}", hex(h));
    }
    if let Some(ref h) = final_proof.final_state_hash {
        eprintln!("[prove-elf] Final state:   {}", hex(h));
    }

    // Serialize proof
    let proof_bytes = serialize_proof(&final_proof);
    fs::write(&output_path, &proof_bytes).unwrap_or_else(|e| {
        eprintln!("[prove-elf] Failed to write proof: {}", e);
        process::exit(1);
    });

    let total = exec_start.elapsed();
    eprintln!("[prove-elf] Proof written to: {} ({} bytes)", output_path, proof_bytes.len());
    eprintln!("[prove-elf] Execution: {:.2}s, Proving: {:.2}s, Total: {:.2}s",
        exec_elapsed.as_secs_f64(), prove_elapsed.as_secs_f64(), total.as_secs_f64());
    eprintln!("[prove-elf] Verified: {}", valid);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn serialize_proof(proof: &metavm_zkp::recursive::RecursiveProof) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"MVMPROOF");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&proof.depth.to_le_bytes());
    buf.extend_from_slice(&proof.initial_state_hash.unwrap_or([0u8; 32]));
    buf.extend_from_slice(&proof.final_state_hash.unwrap_or([0u8; 32]));

    let acc = &proof.accumulator;
    buf.extend_from_slice(&(acc.l_acc.len() as u32).to_le_bytes());
    buf.extend_from_slice(&acc.l_acc);
    buf.extend_from_slice(&(acc.r_acc.len() as u32).to_le_bytes());
    buf.extend_from_slice(&acc.r_acc);
    buf.extend_from_slice(&acc.num_folded.to_le_bytes());

    buf.extend_from_slice(&proof.current_proof.num_steps.to_le_bytes());
    buf.extend_from_slice(&proof.current_proof.domain_size.to_le_bytes());
    buf.extend_from_slice(&(proof.current_proof.column_commitments.len() as u32).to_le_bytes());
    for comm in &proof.current_proof.column_commitments {
        buf.extend_from_slice(&(comm.0.len() as u32).to_le_bytes());
        buf.extend_from_slice(&comm.0);
    }
    buf.extend_from_slice(&(proof.current_proof.num_quotient_chunks as u32).to_le_bytes());
    for qc in &proof.current_proof.quotient_commitments {
        buf.extend_from_slice(&(qc.0.len() as u32).to_le_bytes());
        buf.extend_from_slice(&qc.0);
    }
    buf.extend_from_slice(&(proof.current_proof.evaluations.len() as u32).to_le_bytes());
    for eval in &proof.current_proof.evaluations {
        buf.extend_from_slice(&(eval.len() as u32).to_le_bytes());
        buf.extend_from_slice(eval);
    }
    buf.extend_from_slice(&(proof.current_proof.opening_proof.d.len() as u32).to_le_bytes());
    buf.extend_from_slice(&proof.current_proof.opening_proof.d);
    buf.extend_from_slice(&(proof.current_proof.opening_proof.proof.len() as u32).to_le_bytes());
    buf.extend_from_slice(&proof.current_proof.opening_proof.proof);
    buf
}
