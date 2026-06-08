//! Replay and prove an Ethereum block from RPC.
//!
//! Usage: prove-block <rpc_url> <block_number> [--chunk-size N] [--workers N]
//!        [--scheme bls12381|bls48581|bls48581-fast] [--output proof.bin]
//!
//! Fetches a block from an Ethereum RPC endpoint, replays each transaction
//! with tracing, and produces a ZK proof of the entire block execution
//! using parallel chunked proving and tree folding.

use metavm_evm::constraints::EvmConstraintSystem;
use metavm_evm::inspector::TracingInspector;
use metavm_evm::rpc::{fetch_block, RpcDatabase};
use metavm_evm::trace::{
    evm_final_state_hash, evm_initial_state_hash, evm_state_hash,
    evm_trace_polys_with_curve, EvmTraceColumns, EvmTraceRow,
};
use metavm_zkp::field::CurveType;
use metavm_zkp::prover::{prove_chunk_with_scheme, ChunkProof};
use metavm_zkp::recursive::{verify_final_scheme, RecursiveProof};
use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
use metavm_zkp::scheme::bls48581_scheme::{Bls48581Scheme, Bls48581SchemeFast};
use metavm_zkp::scheme::CommitmentScheme;
use metavm_zkp::tree_fold::tree_fold_with_progress_scheme;
use revm::context::{BlockEnv, Context, TxEnv};
use revm::context_interface::block::blob::BlobExcessGasAndPrice;
use revm::primitives::hardfork::SpecId;
use revm::database::CacheDB;
use revm::handler::{MainBuilder, MainContext};
use revm::primitives::{B256, TxKind, U256};
use revm::InspectCommitEvm;
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
        std::process::abort();
    }
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

/// Work item sent from the main thread to worker threads.
struct ProveWork {
    chunk_index: u64,
    trace: EvmTraceColumns,
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
        eprintln!("Usage: prove-block <rpc_url> <block_number> [--chunk-size N] [--workers N] [--scheme bls12381|bls48581|bls48581-fast] [--output proof.bin]");
        eprintln!();
        eprintln!("Replay an Ethereum block and produce a ZK proof.");
        eprintln!();
        eprintln!("Example: prove-block https://eth.llamarpc.com 19000000 --scheme bls12381");
        process::exit(1);
    }

    let mut rpc_url = String::new();
    let mut block_number: u64 = 0;
    let mut chunk_size: usize = 256;
    let mut output_path = "block_proof.bin".to_string();
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
                output_path = args[i].clone();
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
                        block_number = arg.parse().unwrap_or_else(|_| {
                            eprintln!("Invalid block number: {}", arg);
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
        eprintln!("Error: rpc_url and block_number are required");
        process::exit(1);
    }

    eprintln!("[prove-block] Ethereum Block ZK Proof");
    eprintln!("[prove-block] ───────────────────────");
    eprintln!("[prove-block] RPC: {}", rpc_url);
    eprintln!("[prove-block] Block: {}", block_number);
    eprintln!("[prove-block] Scheme: {}", scheme_name);
    eprintln!("[prove-block] Chunk size: {}", chunk_size);
    eprintln!("[prove-block] Workers: {}", num_workers);

    // Initialize commitment scheme
    let (scheme, curve_type): (Arc<dyn CommitmentScheme>, CurveType) = match scheme_name.as_str() {
        "bls48581" => {
            let s = Bls48581Scheme;
            eprintln!("[prove-block] Initializing BLS48-581...");
            s.init();
            eprintln!("[prove-block] BLS48-581 ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        "bls48581-fast" => {
            let s = Bls48581SchemeFast;
            eprintln!("[prove-block] Initializing BLS48-581 (fast)...");
            s.init();
            eprintln!("[prove-block] BLS48-581 (fast) ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        _ => {
            let s = Bls12381Scheme;
            eprintln!("[prove-block] Initializing BLS12-381...");
            s.init();
            eprintln!("[prove-block] BLS12-381 ready");
            (Arc::new(s), CurveType::Bls12381)
        }
    };

    // Clamp chunk size to scheme's max domain size
    let max_domain = scheme.max_domain_size() as usize;
    if chunk_size > max_domain {
        eprintln!(
            "[prove-block] Warning: chunk_size {} exceeds SRS limit {}, clamping",
            chunk_size, max_domain
        );
        chunk_size = max_domain;
    }

    // Fetch block
    eprintln!("[prove-block] Fetching block...");
    let block = fetch_block(&rpc_url, block_number).unwrap_or_else(|e| {
        eprintln!("[prove-block] Failed to fetch block: {}", e);
        process::exit(1);
    });

    eprintln!(
        "[prove-block] Block hash: 0x{}",
        hex(block.hash.as_slice())
    );
    eprintln!("[prove-block] Transactions: {}", block.transactions.len());
    eprintln!("[prove-block] Gas limit: {}", block.gas_limit);
    eprintln!("[prove-block] Timestamp: {}", block.timestamp);
    eprintln!("[prove-block] Base fee: {} wei", block.base_fee);
    eprintln!(
        "[prove-block] Coinbase: 0x{}",
        hex(block.coinbase.as_slice())
    );

    if block.transactions.is_empty() {
        eprintln!("[prove-block] No transactions in block, nothing to prove");
        process::exit(0);
    }

    // =========================================================================
    // Execution phase — replay transactions and build merged trace
    // =========================================================================
    eprintln!("[prove-block] Replaying transactions...");
    let exec_start = Instant::now();

    let rpc_db = RpcDatabase::new(&rpc_url, block_number - 1);
    let db = CacheDB::new(rpc_db);

    let block_env = BlockEnv {
        number: U256::from(block.number),
        beneficiary: block.coinbase,
        timestamp: U256::from(block.timestamp),
        gas_limit: block.gas_limit,
        basefee: block.base_fee,
        difficulty: U256::ZERO,
        prevrandao: Some(B256::ZERO),
        blob_excess_gas_and_price: Some(BlobExcessGasAndPrice::new_with_spec(
            block.excess_blob_gas.unwrap_or(0),
            SpecId::CANCUN,
        )),
        slot_num: 0,
    };

    let inspector = TracingInspector::new();
    let ctx = Context::mainnet().with_db(db).with_block(block_env);
    let mut evm = ctx.build_mainnet_with_inspector(inspector);

    let mut merged_trace = EvmTraceColumns::new();
    let mut total_steps = 0usize;
    let mut traced_count = 0usize;

    for (tx_idx, tx) in block.transactions.iter().enumerate() {
        let tx_kind = match &tx.to {
            Some(addr) => TxKind::Call(*addr),
            None => TxKind::Create,
        };

        let mut tx_builder = TxEnv::builder()
            .caller(tx.from)
            .kind(tx_kind)
            .gas_limit(tx.gas_limit)
            .value(tx.value)
            .data(tx.input.clone())
            .nonce(tx.nonce);

        match tx.tx_type {
            3 => {
                // EIP-4844 blob transaction
                tx_builder = tx_builder
                    .tx_type(Some(3))
                    .max_fee_per_gas(tx.max_fee_per_gas.unwrap_or(0) as u128)
                    .gas_priority_fee(Some(tx.max_priority_fee_per_gas.unwrap_or(0) as u128))
                    .max_fee_per_blob_gas(tx.max_fee_per_blob_gas.unwrap_or(0) as u128)
                    .blob_hashes(tx.blob_versioned_hashes.clone());
            }
            2 => {
                // EIP-1559
                tx_builder = tx_builder
                    .tx_type(Some(2))
                    .max_fee_per_gas(tx.max_fee_per_gas.unwrap_or(0) as u128)
                    .gas_priority_fee(Some(tx.max_priority_fee_per_gas.unwrap_or(0) as u128));
            }
            _ => {
                // Legacy / EIP-2930
                tx_builder = tx_builder.gas_price(tx.gas_price as u128);
            }
        }

        let tx_env = tx_builder.build_fill();

        evm.inspector.trace = EvmTraceColumns::new();

        match evm.inspect_tx_commit(tx_env) {
            Ok(result) => {
                let gas_used: u64 = result.gas_used();
                let steps = evm.inspector.trace.step.len();

                if steps > 0 {
                    for row_idx in 0..steps {
                        let t = &evm.inspector.trace;
                        let row = EvmTraceRow {
                            step: (total_steps + row_idx) as u64,
                            pc: t.pc[row_idx],
                            opcode: t.opcode[row_idx] as u8,
                            gas_remaining: t.gas_remaining[row_idx],
                            stack_depth: t.stack_depth[row_idx],
                            input0: [
                                t.input0[0][row_idx],
                                t.input0[1][row_idx],
                                t.input0[2][row_idx],
                                t.input0[3][row_idx],
                            ],
                            input1: [
                                t.input1[0][row_idx],
                                t.input1[1][row_idx],
                                t.input1[2][row_idx],
                                t.input1[3][row_idx],
                            ],
                            output0: [
                                t.output0[0][row_idx],
                                t.output0[1][row_idx],
                                t.output0[2][row_idx],
                                t.output0[3][row_idx],
                            ],
                            mem_offset: t.mem_offset[row_idx],
                            mem_value: [
                                t.mem_value[0][row_idx],
                                t.mem_value[1][row_idx],
                                t.mem_value[2][row_idx],
                                t.mem_value[3][row_idx],
                            ],
                            insn_type: t.insn_type[row_idx] as u8,
                            funct: t.funct[row_idx] as u8,
                            immediate: [
                                t.immediate[0][row_idx],
                                t.immediate[1][row_idx],
                                t.immediate[2][row_idx],
                                t.immediate[3][row_idx],
                            ],
                            aux0: [
                                t.aux0[0][row_idx],
                                t.aux0[1][row_idx],
                                t.aux0[2][row_idx],
                                t.aux0[3][row_idx],
                            ],
                            aux1: [
                                t.aux1[0][row_idx],
                                t.aux1[1][row_idx],
                                t.aux1[2][row_idx],
                                t.aux1[3][row_idx],
                            ],
                            next_pc: t.next_pc[row_idx],
                            frame: metavm_evm::trace::FrameState {
                                depth: t.frame_depth[row_idx],
                                caller: [
                                    t.frame_caller[0][row_idx],
                                    t.frame_caller[1][row_idx],
                                    t.frame_caller[2][row_idx],
                                    t.frame_caller[3][row_idx],
                                ],
                                callee: [
                                    t.frame_callee[0][row_idx],
                                    t.frame_callee[1][row_idx],
                                    t.frame_callee[2][row_idx],
                                    t.frame_callee[3][row_idx],
                                ],
                                value: [
                                    t.frame_value[0][row_idx],
                                    t.frame_value[1][row_idx],
                                    t.frame_value[2][row_idx],
                                    t.frame_value[3][row_idx],
                                ],
                                gas: t.frame_gas[row_idx],
                                return_pc: t.frame_return_pc[row_idx],
                                return_offset: t.frame_return_offset[row_idx],
                                return_size: t.frame_return_size[row_idx],
                                is_static: t.frame_static[row_idx],
                            },
                            create_address_hint: [
                                t.create_address_hint[0][row_idx],
                                t.create_address_hint[1][row_idx],
                                t.create_address_hint[2][row_idx],
                                t.create_address_hint[3][row_idx],
                            ],
                            create_nonce_hint: t.create_nonce_hint[row_idx],
                            sel_stop_pop: t.sel_stop_pop[row_idx],
                            create2_salt_hint: [
                                t.create2_salt_hint[0][row_idx],
                                t.create2_salt_hint[1][row_idx],
                                t.create2_salt_hint[2][row_idx],
                                t.create2_salt_hint[3][row_idx],
                            ],
                            create2_initcode_hash_hint: [
                                t.create2_initcode_hash_hint[0][row_idx],
                                t.create2_initcode_hash_hint[1][row_idx],
                                t.create2_initcode_hash_hint[2][row_idx],
                                t.create2_initcode_hash_hint[3][row_idx],
                            ],
                            tx_origin: [0u64; 4],
                            tx_gas_price: 0,
                            tx_calldata_size: 0,
                            tx_code_size: 0,
                            returndata_size: 0,
                        };
                        merged_trace.push_row(&row);
                    }
                    total_steps += steps;
                    traced_count += 1;
                    let tx_type_str = match tx.tx_type {
                        0 => "legacy",
                        1 => "eip2930",
                        2 => "eip1559",
                        _ => "unknown",
                    };
                    let kind_str = if tx.to.is_none() { "create" } else { "call" };
                    eprintln!(
                        "[prove-block] tx {}/{}: {} steps, gas_used={}, type={}, kind={}",
                        tx_idx + 1,
                        block.transactions.len(),
                        steps,
                        gas_used,
                        tx_type_str,
                        kind_str
                    );
                } else {
                    eprintln!(
                        "[prove-block] tx {}/{}: no EVM steps (ETH transfer), gas_used={}",
                        tx_idx + 1,
                        block.transactions.len(),
                        gas_used
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "[prove-block] tx {}/{}: execution error: {:?}",
                    tx_idx + 1,
                    block.transactions.len(),
                    e
                );
            }
        }
    }

    let exec_elapsed = exec_start.elapsed();
    eprintln!(
        "[prove-block] Replayed {}/{} transactions, {} total steps in {:.3}s",
        traced_count,
        block.transactions.len(),
        total_steps,
        exec_elapsed.as_secs_f64()
    );

    if total_steps == 0 {
        eprintln!("[prove-block] No traced steps, nothing to prove");
        process::exit(0);
    }

    // =========================================================================
    // Chunking phase — split merged trace into chunks
    // =========================================================================
    let num_chunks = (total_steps + chunk_size - 1) / chunk_size;
    eprintln!(
        "[prove-block] Splitting {} steps into {} chunks of up to {} rows",
        total_steps, num_chunks, chunk_size
    );

    // Install Ctrl+C handler
    unsafe {
        signal(2 /* SIGINT */, handle_sigint);
    }

    // =========================================================================
    // Proving phase — parallel worker pool
    // =========================================================================
    let constraints = Arc::new(EvmConstraintSystem::new());
    let prove_start = Instant::now();
    let chunks_total = num_chunks as u64;

    STOP_REQUESTED.store(false, Ordering::SeqCst);

    eprintln!(
        "[prove-block] ─── Proving {} chunks with {} workers ───────────────",
        chunks_total, num_workers
    );

    let chunks_proved = Arc::new(AtomicU64::new(0));

    // Work queue: main → workers (bounded for backpressure)
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
        let total = chunks_total;
        let scheme_ref = Arc::clone(&scheme);
        let ct = curve_type;

        let handle = thread::Builder::new()
            .name(format!("prover-{}", worker_id))
            .spawn(move || {
                while let Some(work) = wq.pop() {
                    let polys = evm_trace_polys_with_curve(&work.trace, ct);
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
                        eprintln!(
                            "[prove-block] proved {}/{} chunks ({:.1}%)",
                            done,
                            total,
                            done as f64 / total as f64 * 100.0
                        );
                    }
                    if tx.send(chunk_proof).is_err() {
                        break;
                    }
                }
            })
            .unwrap_or_else(|e| {
                eprintln!("[prove-block] Failed to spawn worker {}: {}", worker_id, e);
                process::exit(1);
            });
        worker_handles.push(handle);
    }
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
                Some(chunks_total),
                |count, _total| {
                    if count % 100 == 0 {
                        eprintln!("[prove-block]   folded {} chunks", count);
                    }
                },
                &*folder_scheme,
                folder_curve,
            )
        })
        .unwrap_or_else(|e| {
            eprintln!("[prove-block] Failed to spawn folder thread: {}", e);
            process::exit(1);
        });

    // Feed chunks to the work queue
    for chunk_idx in 0..num_chunks {
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            eprintln!("[prove-block] Ctrl+C received, stopping");
            break;
        }

        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(total_steps);

        // Compute boundary hashes
        let initial_hash = if chunk_idx == 0 {
            evm_initial_state_hash()
        } else {
            evm_state_hash(&merged_trace, start)
        };
        let final_hash = if end == total_steps {
            evm_final_state_hash(&merged_trace)
        } else {
            evm_state_hash(&merged_trace, end)
        };

        let chunk_trace = merged_trace.slice_rows(start, end);

        work_queue.push(Some(ProveWork {
            chunk_index: chunk_idx as u64,
            trace: chunk_trace,
            initial_state_hash: initial_hash,
            final_state_hash: final_hash,
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
        "[prove-block] All {} chunks proved",
        chunks_proved.load(Ordering::Relaxed),
    );

    // Wait for folder thread
    let fold_result = folder_handle.join().unwrap_or_else(|_| {
        eprintln!("[prove-block] Folder thread panicked");
        process::exit(1);
    });

    let final_proof = fold_result.unwrap_or_else(|e| {
        eprintln!("[prove-block] Tree fold failed: {}", e);
        process::exit(1);
    });

    let prove_elapsed = prove_start.elapsed();

    // Verify
    let valid = verify_final_scheme(&final_proof, &*scheme);
    eprintln!("[prove-block] Proof verified: {}", valid);
    eprintln!("[prove-block] Depth: {}", final_proof.depth);

    if let Some(ref h) = final_proof.initial_state_hash {
        eprintln!("[prove-block] Initial state: {}", hex(h));
    }
    if let Some(ref h) = final_proof.final_state_hash {
        eprintln!("[prove-block] Final state:   {}", hex(h));
    }

    // Serialize proof
    let proof_bytes = serialize_proof(&final_proof);
    fs::write(&output_path, &proof_bytes).unwrap_or_else(|e| {
        eprintln!("[prove-block] Failed to write proof: {}", e);
        process::exit(1);
    });

    let total = exec_start.elapsed();
    eprintln!(
        "[prove-block] Proof written to: {} ({} bytes)",
        output_path,
        proof_bytes.len()
    );
    eprintln!(
        "[prove-block] Execution: {:.2}s, Proving: {:.2}s, Total: {:.2}s",
        exec_elapsed.as_secs_f64(),
        prove_elapsed.as_secs_f64(),
        total.as_secs_f64()
    );
    eprintln!(
        "[prove-block] Result: {}",
        if valid { "SUCCESS" } else { "FAILED" }
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn serialize_proof(proof: &RecursiveProof) -> Vec<u8> {
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
    buf.extend_from_slice(
        &(proof.current_proof.opening_proof.d.len() as u32).to_le_bytes(),
    );
    buf.extend_from_slice(&proof.current_proof.opening_proof.d);
    buf.extend_from_slice(
        &(proof.current_proof.opening_proof.proof.len() as u32).to_le_bytes(),
    );
    buf.extend_from_slice(&proof.current_proof.opening_proof.proof);
    buf
}
