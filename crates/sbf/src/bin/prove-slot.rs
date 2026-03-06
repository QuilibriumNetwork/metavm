//! Replay and prove Solana slot transactions.
//!
//! Usage: prove-slot <rpc_url> <slot> [--chunk-size N] [--workers N]
//!        [--scheme bls12381|bls48581|bls48581-fast] [--output proof.bin]
//!
//! Fetches a slot from a Solana RPC endpoint, downloads BPF programs,
//! executes them with tracing, and produces ZK proofs using parallel
//! chunked proving and tree folding.

use metavm_sbf::constraints::SbfConstraintSystem;
use metavm_sbf::executor;
use metavm_sbf::rpc;
use metavm_sbf::trace::{
    sbf_final_state_hash, sbf_initial_state_hash, sbf_state_hash,
    sbf_trace_polys_with_curve, SbfTraceColumns,
};
use metavm_zkp::field::CurveType;
use metavm_zkp::prover::{prove_chunk_with_scheme, ChunkProof};
use metavm_zkp::recursive::verify_final_scheme;
use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
use metavm_zkp::scheme::bls48581_scheme::{Bls48581Scheme, Bls48581SchemeFast};
use metavm_zkp::scheme::CommitmentScheme;
use metavm_zkp::tree_fold::tree_fold_with_progress_scheme;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use std::{env, process, thread};

// ── Ctrl+C handling ─────────────────────────────────────────────────────────

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}

extern "C" fn handle_sigint(_sig: i32) {
    if STOP_REQUESTED.load(Ordering::SeqCst) {
        std::process::abort();
    }
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

/// Work item sent from the main thread to worker threads.
struct ProveWork {
    chunk_index: u64,
    trace: SbfTraceColumns,
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

    fn push(&self, item: Option<ProveWork>) {
        let mut q = self.queue.lock().unwrap();
        while q.len() >= self.capacity {
            q = self.not_full.wait(q).unwrap();
        }
        q.push_back(item);
        self.not_empty.notify_one();
    }

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
    if args.len() < 3 {
        eprintln!("Usage: prove-slot <rpc_url> <slot> [--chunk-size N] [--workers N] [--scheme bls12381|bls48581|bls48581-fast] [--output proof.bin]");
        eprintln!();
        eprintln!("Replay Solana slot transactions and produce ZK proofs.");
        eprintln!();
        eprintln!("Example: prove-slot https://api.mainnet-beta.solana.com 250000000 --scheme bls12381");
        process::exit(1);
    }

    let mut rpc_url = String::new();
    let mut slot: u64 = 0;
    let mut chunk_size: usize = 256;
    let mut _output_path = "slot_proof.bin".to_string();
    let mut num_workers: usize = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut scheme_name = "bls12381".to_string();
    let mut positional = 0;

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
                _output_path = args[i].clone();
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
            arg => {
                match positional {
                    0 => rpc_url = arg.to_string(),
                    1 => {
                        slot = arg.parse().unwrap_or_else(|_| {
                            eprintln!("Invalid slot number: {}", arg);
                            process::exit(1);
                        });
                    }
                    _ => {
                        eprintln!("Unexpected argument: {}", arg);
                        process::exit(1);
                    }
                }
                positional += 1;
            }
        }
        i += 1;
    }

    if rpc_url.is_empty() || positional < 2 {
        eprintln!("Error: rpc_url and slot are required");
        process::exit(1);
    }

    eprintln!("[prove-slot] Solana Slot ZK Proof");
    eprintln!("[prove-slot] ────────────────────");
    eprintln!("[prove-slot] RPC: {}", rpc_url);
    eprintln!("[prove-slot] Slot: {}", slot);
    eprintln!("[prove-slot] Scheme: {}", scheme_name);
    eprintln!("[prove-slot] Chunk size: {}", chunk_size);
    eprintln!("[prove-slot] Workers: {}", num_workers);

    // Initialize commitment scheme
    let (scheme, curve_type): (Arc<dyn CommitmentScheme>, CurveType) = match scheme_name.as_str() {
        "bls48581" => {
            let s = Bls48581Scheme;
            eprintln!("[prove-slot] Initializing BLS48-581...");
            s.init();
            eprintln!("[prove-slot] BLS48-581 ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        "bls48581-fast" => {
            let s = Bls48581SchemeFast;
            eprintln!("[prove-slot] Initializing BLS48-581 (fast)...");
            s.init();
            eprintln!("[prove-slot] BLS48-581 (fast) ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        _ => {
            let s = Bls12381Scheme;
            eprintln!("[prove-slot] Initializing BLS12-381...");
            s.init();
            eprintln!("[prove-slot] BLS12-381 ready");
            (Arc::new(s), CurveType::Bls12381)
        }
    };

    // Clamp chunk size to scheme's max domain size
    let max_domain = scheme.max_domain_size() as usize;
    if chunk_size > max_domain {
        eprintln!(
            "[prove-slot] Warning: chunk_size {} exceeds SRS limit {}, clamping",
            chunk_size, max_domain
        );
        chunk_size = max_domain;
    }

    // Fetch slot
    eprintln!("[prove-slot] Fetching slot...");
    let slot_data = rpc::fetch_block(&rpc_url, slot).unwrap_or_else(|e| {
        eprintln!("[prove-slot] Failed to fetch slot: {}", e);
        process::exit(1);
    });

    eprintln!(
        "[prove-slot] Transactions with BPF instructions: {}",
        slot_data.transactions.len()
    );

    if slot_data.transactions.is_empty() {
        eprintln!("[prove-slot] No BPF transactions in slot");
        process::exit(0);
    }

    // Install Ctrl+C handler
    unsafe {
        signal(2 /* SIGINT */, handle_sigint);
    }

    // Cache program ELFs and account data to avoid re-downloading
    let mut program_cache: HashMap<String, Vec<u8>> = HashMap::new();
    let mut account_cache: HashMap<String, rpc::AccountData> = HashMap::new();
    let cs = Arc::new(SbfConstraintSystem::new());
    let mut proved = 0;
    let mut failed = 0;

    for (tx_idx, tx) in slot_data.transactions.iter().enumerate() {
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            eprintln!("[prove-slot] Ctrl+C received, stopping");
            break;
        }

        eprintln!(
            "[prove-slot] ── tx {}/{}: program={} sig={}...",
            tx_idx + 1,
            slot_data.transactions.len(),
            &tx.program_id,
            &tx.signature[..16.min(tx.signature.len())]
        );

        // Fetch program ELF (with caching)
        let elf_bytes = if let Some(cached) = program_cache.get(&tx.program_id) {
            cached.clone()
        } else {
            match rpc::fetch_program_elf(&rpc_url, &tx.program_id) {
                Ok(elf) => {
                    eprintln!("[prove-slot]   Fetched program ELF: {} bytes", elf.len());
                    program_cache.insert(tx.program_id.clone(), elf.clone());
                    elf
                }
                Err(e) => {
                    eprintln!("[prove-slot]   Failed to fetch program: {}", e);
                    failed += 1;
                    continue;
                }
            }
        };

        // Fetch account data for all instruction accounts
        let mut accounts_with_data: Vec<(String, rpc::AccountData, bool, bool)> = Vec::new();
        let mut acct_fetch_failed = false;

        for (acct_idx, acct_key) in tx.accounts.iter().enumerate() {
            let acct_data = if let Some(cached) = account_cache.get(acct_key) {
                rpc::AccountData {
                    lamports: cached.lamports,
                    data: cached.data.clone(),
                    owner: cached.owner,
                    executable: cached.executable,
                    rent_epoch: cached.rent_epoch,
                }
            } else {
                match rpc::fetch_account_info(&rpc_url, acct_key) {
                    Ok(data) => {
                        account_cache.insert(
                            acct_key.clone(),
                            rpc::AccountData {
                                lamports: data.lamports,
                                data: data.data.clone(),
                                owner: data.owner,
                                executable: data.executable,
                                rent_epoch: data.rent_epoch,
                            },
                        );
                        data
                    }
                    Err(e) => {
                        eprintln!(
                            "[prove-slot]   Failed to fetch account {}: {}",
                            acct_key, e
                        );
                        acct_fetch_failed = true;
                        break;
                    }
                }
            };

            let is_signer = tx.signer_indices.contains(&acct_idx);
            let is_writable = tx.writable_indices.contains(&acct_idx);
            accounts_with_data.push((acct_key.clone(), acct_data, is_signer, is_writable));
        }

        if acct_fetch_failed {
            failed += 1;
            continue;
        }

        eprintln!(
            "[prove-slot]   Accounts: {} ({} signer, {} writable)",
            tx.accounts.len(),
            tx.signer_indices.len(),
            tx.writable_indices.len()
        );

        // Serialize BPF input with full account data
        let input = rpc::serialize_bpf_input(&accounts_with_data, &tx.data, &tx.program_id);
        eprintln!("[prove-slot]   Serialized input: {} bytes", input.len());

        // Execute with tracing
        let exec_start = Instant::now();
        let trace = match executor::execute_sbf_elf(&elf_bytes, &input) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[prove-slot]   Execution failed: {}", e);
                failed += 1;
                continue;
            }
        };

        let exec_elapsed = exec_start.elapsed();
        let total_steps = trace.step.len();
        eprintln!(
            "[prove-slot]   Executed: {} steps in {:.3}s",
            total_steps,
            exec_elapsed.as_secs_f64()
        );

        if total_steps == 0 {
            eprintln!("[prove-slot]   No steps traced, skipping");
            continue;
        }

        // Prove this transaction's trace using chunked parallel proving
        let prove_start = Instant::now();
        let num_chunks = (total_steps + chunk_size - 1) / chunk_size;

        if num_chunks == 1 {
            // Single chunk — no need for worker pool overhead
            let polys = sbf_trace_polys_with_curve(&trace, curve_type);
            let zero_hash = [0u8; 32];
            let chunk_proof =
                prove_chunk_with_scheme(&polys, cs.as_ref(), 0, &zero_hash, &zero_hash, &*scheme);
            let recursive_proof =
                metavm_zkp::recursive::begin_chunk_scheme(chunk_proof, &*scheme, curve_type);
            let valid = verify_final_scheme(&recursive_proof, &*scheme);
            let prove_elapsed = prove_start.elapsed();
            eprintln!(
                "[prove-slot]   Proof verified: {} ({:.3}s, 1 chunk)",
                valid,
                prove_elapsed.as_secs_f64()
            );
            if valid {
                proved += 1;
            } else {
                failed += 1;
            }
        } else {
            // Multi-chunk — parallel worker pool
            eprintln!(
                "[prove-slot]   Proving {} chunks with {} workers...",
                num_chunks, num_workers
            );

            let chunks_proved = Arc::new(AtomicU64::new(0));
            let chunks_total = num_chunks as u64;
            let work_queue = Arc::new(WorkQueue::new(num_workers * 2));
            let (result_tx, result_rx) = mpsc::channel::<ChunkProof>();

            // Spawn worker threads
            let mut worker_handles = Vec::new();
            for worker_id in 0..num_workers {
                let wq = Arc::clone(&work_queue);
                let tx = result_tx.clone();
                let cs_ref = Arc::clone(&cs);
                let proved_counter = Arc::clone(&chunks_proved);
                let scheme_ref = Arc::clone(&scheme);
                let ct = curve_type;

                let handle = thread::Builder::new()
                    .name(format!("prover-{}", worker_id))
                    .spawn(move || {
                        while let Some(work) = wq.pop() {
                            let polys = sbf_trace_polys_with_curve(&work.trace, ct);
                            let chunk_proof = prove_chunk_with_scheme(
                                &polys,
                                cs_ref.as_ref(),
                                work.chunk_index,
                                &work.initial_state_hash,
                                &work.final_state_hash,
                                &*scheme_ref,
                            );
                            proved_counter.fetch_add(1, Ordering::Relaxed);
                            if tx.send(chunk_proof).is_err() {
                                break;
                            }
                        }
                    })
                    .unwrap_or_else(|e| {
                        eprintln!(
                            "[prove-slot] Failed to spawn worker {}: {}",
                            worker_id, e
                        );
                        process::exit(1);
                    });
                worker_handles.push(handle);
            }
            drop(result_tx);

            // Spawn folder thread
            let folder_scheme = Arc::clone(&scheme);
            let folder_curve = curve_type;
            let folder_handle = thread::Builder::new()
                .name("folder".to_string())
                .spawn(move || {
                    let reorder = ReorderIterator::new(result_rx);
                    tree_fold_with_progress_scheme(
                        reorder,
                        Some(chunks_total),
                        |_count, _total| {},
                        &*folder_scheme,
                        folder_curve,
                    )
                })
                .unwrap_or_else(|e| {
                    eprintln!("[prove-slot] Failed to spawn folder thread: {}", e);
                    process::exit(1);
                });

            // Feed chunks to work queue
            for chunk_idx in 0..num_chunks {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(total_steps);

                let initial_hash = if chunk_idx == 0 {
                    sbf_initial_state_hash()
                } else {
                    sbf_state_hash(&trace, start)
                };
                let final_hash = if end == total_steps {
                    sbf_final_state_hash(&trace)
                } else {
                    sbf_state_hash(&trace, end)
                };

                let chunk_trace = trace.slice_rows(start, end);
                work_queue.push(Some(ProveWork {
                    chunk_index: chunk_idx as u64,
                    trace: chunk_trace,
                    initial_state_hash: initial_hash,
                    final_state_hash: final_hash,
                }));
            }

            // Poison pills
            for _ in 0..num_workers {
                work_queue.push(None);
            }

            for handle in worker_handles {
                let _ = handle.join();
            }

            let fold_result = folder_handle.join().unwrap_or_else(|_| {
                eprintln!("[prove-slot] Folder thread panicked");
                process::exit(1);
            });

            match fold_result {
                Ok(final_proof) => {
                    let valid = verify_final_scheme(&final_proof, &*scheme);
                    let prove_elapsed = prove_start.elapsed();
                    eprintln!(
                        "[prove-slot]   Proof verified: {} ({:.3}s, {} chunks, depth {})",
                        valid,
                        prove_elapsed.as_secs_f64(),
                        num_chunks,
                        final_proof.depth
                    );
                    if valid {
                        proved += 1;
                    } else {
                        failed += 1;
                    }
                }
                Err(e) => {
                    eprintln!("[prove-slot]   Tree fold failed: {}", e);
                    failed += 1;
                }
            }
        }
    }

    eprintln!("[prove-slot] ────────────────────");
    eprintln!(
        "[prove-slot] Results: {} proved, {} failed, {} total",
        proved,
        failed,
        slot_data.transactions.len()
    );
}
