//! End-to-end binary: prove correct execution of a Linux boot.
//!
//! Usage: prove-boot <kernel.elf> [initrd.gz] [--chunk-size N] [--output proof.bin] [--workers N]
//!        [--max-steps N] [--trace-file path]
//!
//! Two-phase architecture:
//!   Phase 1 (Execution): Runs the VM to completion (or Ctrl+C / --max-steps),
//!     collecting execution chunks in memory or serializing to a trace file.
//!   Phase 2 (Proving): Processes collected chunks in parallel worker threads,
//!     folds all chunk proofs into a single recursive proof via tree aggregation.

use metavm_riscv::chunked_trace::{ExecutionChunk, StreamingProver};
use metavm_riscv::constraints::{RiscvConstraintSystem, trace_polys_from_columns_with_curve};
use metavm_riscv::loader::{setup_linux_boot, LinuxBootConfig};
use metavm_riscv::trace::TraceColumns;
use metavm_zkp::field::CurveType;
use metavm_zkp::prover::{prove_chunk_with_scheme, ChunkProof};
use metavm_zkp::scheme::CommitmentScheme;
use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
use metavm_zkp::scheme::bls48581_scheme::{Bls48581Scheme, Bls48581SchemeFast};
use metavm_zkp::tree_fold::tree_fold_with_progress_scheme;
use metavm_zkp::recursive::verify_final_scheme;
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, BufReader, BufWriter, Read as IoRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use std::{env, fs, process, thread};

// ── Trace file format ────────────────────────────────────────────────────────

const TRACE_MAGIC: &[u8; 8] = b"MVMTRACE";
const TRACE_VERSION: u32 = 1;
/// Number of Vec<u64> columns in TraceColumns.
const NUM_COLUMNS: usize = 84;

fn write_trace_header(w: &mut impl Write, chunk_size: u32) -> io::Result<()> {
    w.write_all(TRACE_MAGIC)?;
    w.write_all(&TRACE_VERSION.to_le_bytes())?;
    w.write_all(&chunk_size.to_le_bytes())?;
    Ok(())
}

fn read_trace_header(r: &mut impl IoRead) -> io::Result<u32> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != TRACE_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad trace magic"));
    }
    let mut ver = [0u8; 4];
    r.read_exact(&mut ver)?;
    let version = u32::from_le_bytes(ver);
    if version != TRACE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported trace version {}", version),
        ));
    }
    let mut cs = [0u8; 4];
    r.read_exact(&mut cs)?;
    Ok(u32::from_le_bytes(cs))
}

fn serialize_chunk(chunk: &ExecutionChunk, w: &mut impl Write) -> io::Result<()> {
    w.write_all(&chunk.chunk_index.to_le_bytes())?;
    w.write_all(&chunk.num_steps.to_le_bytes())?;
    w.write_all(&chunk.initial_state_hash)?;
    w.write_all(&chunk.final_state_hash)?;

    // UART bytes
    w.write_all(&(chunk.uart_bytes_this_chunk.len() as u32).to_le_bytes())?;
    w.write_all(&chunk.uart_bytes_this_chunk)?;

    // 84 columns in fixed order
    let cols = &chunk.columns;
    let all_columns: [&Vec<u64>; NUM_COLUMNS] = [
        &cols.step, &cols.pc, &cols.rd, &cols.rd_val_before, &cols.rd_val_after,
        &cols.rs1, &cols.rs1_val, &cols.rs2, &cols.rs2_val,
        &cols.mem_addr, &cols.mem_val, &cols.next_pc, &cols.privilege_mode,
        &cols.insn_type, &cols.funct, &cols.immediate, &cols.insn_len,
        &cols.aux0, &cols.aux1, &cols.aux2,
        &cols.sel_r_alu_add, &cols.sel_r_alu_sub,
        &cols.sel_r_and, &cols.sel_r_or, &cols.sel_r_xor,
        &cols.sel_r_sll, &cols.sel_r_srl, &cols.sel_r_sra, &cols.sel_r_compare,
        &cols.sel_i_alu_add,
        &cols.sel_i_and, &cols.sel_i_or, &cols.sel_i_xor,
        &cols.sel_i_sll, &cols.sel_i_srl, &cols.sel_i_sra, &cols.sel_i_compare,
        &cols.sel_w_alu_add, &cols.sel_w_alu_sub, &cols.sel_w_alu_addi,
        &cols.sel_w_sll, &cols.sel_w_srl, &cols.sel_w_sra, &cols.sel_w_alu_other,
        &cols.sel_mul, &cols.sel_div, &cols.sel_rem, &cols.sel_muldiv_other,
        &cols.sel_load, &cols.sel_store,
        &cols.sel_beq, &cols.sel_bne, &cols.sel_bltu, &cols.sel_bgeu,
        &cols.sel_blt, &cols.sel_bge,
        &cols.sel_jal, &cols.sel_jalr, &cols.sel_lui, &cols.sel_auipc,
        &cols.sel_csrrw, &cols.sel_csrrs, &cols.sel_csrrc,
        &cols.sel_system,
        &cols.sel_lr, &cols.sel_sc, &cols.sel_amo_swap, &cols.sel_amo_add,
        &cols.sel_amo_bitwise, &cols.sel_amo_compare, &cols.sel_atomic_other,
        &cols.sel_mulhu,
        &cols.sel_csrrw_i, &cols.sel_csrrs_i,
        &cols.sel_mulh, &cols.sel_mulhsu, &cols.sel_mulw, &cols.sel_divw, &cols.sel_remw,
        &cols.sel_amo_or, &cols.sel_amo_xor,
        &cols.sel_amo_maxu, &cols.sel_amo_mins, &cols.sel_amo_maxs,
    ];
    for col in &all_columns {
        w.write_all(&(col.len() as u32).to_le_bytes())?;
        for &val in col.iter() {
            w.write_all(&val.to_le_bytes())?;
        }
    }
    Ok(())
}

fn deserialize_chunk(r: &mut impl IoRead) -> io::Result<ExecutionChunk> {
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];
    let mut buf32 = [0u8; 32];

    r.read_exact(&mut buf8)?;
    let chunk_index = u64::from_le_bytes(buf8);
    r.read_exact(&mut buf8)?;
    let num_steps = u64::from_le_bytes(buf8);

    r.read_exact(&mut buf32)?;
    let initial_state_hash = buf32;
    let mut final_hash = [0u8; 32];
    r.read_exact(&mut final_hash)?;

    r.read_exact(&mut buf4)?;
    let uart_len = u32::from_le_bytes(buf4) as usize;
    let mut uart_bytes = vec![0u8; uart_len];
    r.read_exact(&mut uart_bytes)?;

    fn read_col(r: &mut impl IoRead) -> io::Result<Vec<u64>> {
        let mut lb = [0u8; 4];
        r.read_exact(&mut lb)?;
        let len = u32::from_le_bytes(lb) as usize;
        let mut col = Vec::with_capacity(len);
        let mut vb = [0u8; 8];
        for _ in 0..len {
            r.read_exact(&mut vb)?;
            col.push(u64::from_le_bytes(vb));
        }
        Ok(col)
    }

    let columns = TraceColumns {
        step: read_col(r)?,
        pc: read_col(r)?,
        rd: read_col(r)?,
        rd_val_before: read_col(r)?,
        rd_val_after: read_col(r)?,
        rs1: read_col(r)?,
        rs1_val: read_col(r)?,
        rs2: read_col(r)?,
        rs2_val: read_col(r)?,
        mem_addr: read_col(r)?,
        mem_val: read_col(r)?,
        next_pc: read_col(r)?,
        privilege_mode: read_col(r)?,
        insn_type: read_col(r)?,
        funct: read_col(r)?,
        immediate: read_col(r)?,
        insn_len: read_col(r)?,
        aux0: read_col(r)?,
        aux1: read_col(r)?,
        aux2: read_col(r)?,
        sel_r_alu_add: read_col(r)?,
        sel_r_alu_sub: read_col(r)?,
        sel_r_and: read_col(r)?,
        sel_r_or: read_col(r)?,
        sel_r_xor: read_col(r)?,
        sel_r_sll: read_col(r)?,
        sel_r_srl: read_col(r)?,
        sel_r_sra: read_col(r)?,
        sel_r_compare: read_col(r)?,
        sel_i_alu_add: read_col(r)?,
        sel_i_and: read_col(r)?,
        sel_i_or: read_col(r)?,
        sel_i_xor: read_col(r)?,
        sel_i_sll: read_col(r)?,
        sel_i_srl: read_col(r)?,
        sel_i_sra: read_col(r)?,
        sel_i_compare: read_col(r)?,
        sel_w_alu_add: read_col(r)?,
        sel_w_alu_sub: read_col(r)?,
        sel_w_alu_addi: read_col(r)?,
        sel_w_sll: read_col(r)?,
        sel_w_srl: read_col(r)?,
        sel_w_sra: read_col(r)?,
        sel_w_alu_other: read_col(r)?,
        sel_mul: read_col(r)?,
        sel_div: read_col(r)?,
        sel_rem: read_col(r)?,
        sel_muldiv_other: read_col(r)?,
        sel_load: read_col(r)?,
        sel_store: read_col(r)?,
        sel_beq: read_col(r)?,
        sel_bne: read_col(r)?,
        sel_bltu: read_col(r)?,
        sel_bgeu: read_col(r)?,
        sel_blt: read_col(r)?,
        sel_bge: read_col(r)?,
        sel_jal: read_col(r)?,
        sel_jalr: read_col(r)?,
        sel_lui: read_col(r)?,
        sel_auipc: read_col(r)?,
        sel_csrrw: read_col(r)?,
        sel_csrrs: read_col(r)?,
        sel_csrrc: read_col(r)?,
        sel_system: read_col(r)?,
        sel_lr: read_col(r)?,
        sel_sc: read_col(r)?,
        sel_amo_swap: read_col(r)?,
        sel_amo_add: read_col(r)?,
        sel_amo_bitwise: read_col(r)?,
        sel_amo_compare: read_col(r)?,
        sel_atomic_other: read_col(r)?,
        sel_mulhu: read_col(r)?,
        sel_csrrw_i: read_col(r)?,
        sel_csrrs_i: read_col(r)?,
        sel_mulh: read_col(r)?,
        sel_mulhsu: read_col(r)?,
        sel_mulw: read_col(r)?,
        sel_divw: read_col(r)?,
        sel_remw: read_col(r)?,
        sel_amo_or: read_col(r)?,
        sel_amo_xor: read_col(r)?,
        sel_amo_maxu: read_col(r)?,
        sel_amo_mins: read_col(r)?,
        sel_amo_maxs: read_col(r)?,
    };

    Ok(ExecutionChunk {
        chunk_index,
        columns,
        initial_state_hash,
        final_state_hash: final_hash,
        uart_bytes_this_chunk: uart_bytes,
        num_steps,
    })
}

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
        eprintln!("Usage: prove-boot <kernel.elf> [initrd.gz] [--chunk-size N] [--output proof.bin] [--workers N]");
        eprintln!("       [--max-steps N] [--trace-file path] [--scheme bls12381|bls48581|bls48581-fast]");
        eprintln!();
        eprintln!("Prove correct execution of a Linux boot via KZG polynomial commitments.");
        eprintln!();
        eprintln!("Execution and proving are decoupled into two phases:");
        eprintln!("  Phase 1: Execute the VM, collecting chunks (in memory or to --trace-file)");
        eprintln!("  Phase 2: Prove all chunks in parallel, fold into a single recursive proof");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --chunk-size N     Steps per chunk (default 128)");
        eprintln!("  --output FILE      Output proof file (default: boot_proof.bin)");
        eprintln!("  --max-steps N      Maximum steps to execute (0 = unlimited)");
        eprintln!("  --workers N        Number of parallel proving threads (default: available CPUs)");
        eprintln!("  --trace-file PATH  Write execution chunks to disk instead of memory");
        eprintln!("  --scheme NAME      Commitment scheme: bls12381 (default), bls48581, bls48581-fast");
        process::exit(1);
    }

    // Parse arguments
    let mut kernel_path = String::new();
    let mut initrd_path: Option<String> = None;
    let mut chunk_size: usize = 128;
    let mut output_path = "boot_proof.bin".to_string();
    let mut max_steps: u64 = 0;
    let mut num_workers: usize = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut trace_file_path: Option<String> = None;
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
            "--trace-file" => {
                i += 1;
                trace_file_path = Some(args[i].clone());
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
                if kernel_path.is_empty() {
                    kernel_path = path.to_string();
                } else if initrd_path.is_none() {
                    initrd_path = Some(path.to_string());
                } else {
                    eprintln!("Unexpected argument: {}", path);
                    process::exit(1);
                }
            }
        }
        i += 1;
    }

    if kernel_path.is_empty() {
        eprintln!("Error: kernel path is required");
        process::exit(1);
    }

    // -----------------------------------------------------------------------
    // Step 1: Load kernel + initrd, set up VM
    // -----------------------------------------------------------------------
    eprintln!("[prove] Loading kernel: {}", kernel_path);
    let kernel_data = fs::read(&kernel_path).unwrap_or_else(|e| {
        eprintln!("Failed to read kernel: {}", e);
        process::exit(1);
    });
    eprintln!("[prove] Kernel size: {} bytes", kernel_data.len());

    let initrd_data = initrd_path.as_ref().map(|p| {
        eprintln!("[prove] Loading initrd: {}", p);
        fs::read(p).unwrap_or_else(|e| {
            eprintln!("Failed to read initrd: {}", e);
            process::exit(1);
        })
    });

    let config = LinuxBootConfig {
        kernel_data,
        initrd_data,
        bootargs: "console=ttyS0 earlycon=uart8250,mmio,0x10000000,115200n8 nosoftlockup nohz=off nosmp rdinit=/init".to_string(),
        memory_size: 128 * 1024 * 1024,
        disk_image: None,
    };

    let vm = setup_linux_boot(config).unwrap_or_else(|e| {
        eprintln!("Failed to set up VM: {}", e);
        process::exit(1);
    });

    eprintln!("[prove] VM initialized, entry=0x{:x}, chunk_size={}", vm.cpu.pc, chunk_size);

    // -----------------------------------------------------------------------
    // Step 2: Initialize commitment scheme
    // -----------------------------------------------------------------------
    let (scheme, curve_type): (Arc<dyn CommitmentScheme>, CurveType) = match scheme_name.as_str() {
        "bls48581" => {
            let s = Bls48581Scheme;
            eprintln!("[prove] Initializing BLS48-581...");
            s.init();
            eprintln!("[prove] BLS48-581 ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        "bls48581-fast" => {
            let s = Bls48581SchemeFast;
            eprintln!("[prove] Initializing BLS48-581 (fast)...");
            s.init();
            eprintln!("[prove] BLS48-581 (fast) ready");
            (Arc::new(s), CurveType::Bls48581)
        }
        _ => {
            let s = Bls12381Scheme;
            eprintln!("[prove] Initializing BLS12-381...");
            s.init();
            eprintln!("[prove] BLS12-381 ready");
            (Arc::new(s), CurveType::Bls12381)
        }
    };

    // Clamp chunk size to scheme's max domain size
    let max_domain = scheme.max_domain_size() as usize;
    if chunk_size > max_domain {
        eprintln!("[prove] Warning: chunk_size {} exceeds SRS limit {}, clamping", chunk_size, max_domain);
        chunk_size = max_domain;
    }

    // -----------------------------------------------------------------------
    // Step 3: Create streaming prover
    // -----------------------------------------------------------------------
    let mut prover = StreamingProver::new(vm, chunk_size);
    prover.initialize_memory_hash();

    // Install Ctrl+C handler for graceful shutdown
    unsafe { signal(2 /* SIGINT */, handle_sigint); }

    // =======================================================================
    // Phase 1: Execution
    // =======================================================================
    eprintln!("[prove] ─── Executing ──────────────────────────────────────────");
    eprintln!("[prove] Press Ctrl+C to stop execution and proceed to proving");
    let exec_start = Instant::now();

    let mut total_uart_bytes = 0u64;
    let mut chunks_executed = 0u64;
    let mut last_report = exec_start;

    // Either collect in-memory or write to trace file
    let mut in_memory_chunks: Vec<ExecutionChunk> = Vec::new();
    let mut trace_writer: Option<BufWriter<fs::File>> = None;

    if let Some(ref path) = trace_file_path {
        let file = fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("[prove] Failed to create trace file {}: {}", path, e);
            process::exit(1);
        });
        let mut writer = BufWriter::new(file);
        write_trace_header(&mut writer, chunk_size as u32).unwrap_or_else(|e| {
            eprintln!("[prove] Failed to write trace header: {}", e);
            process::exit(1);
        });
        trace_writer = Some(writer);
        eprintln!("[prove] Writing execution chunks to: {}", path);
    } else {
        eprintln!("[prove] Collecting execution chunks in memory");
    }

    while !prover.is_halted() {
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            eprintln!();
            eprintln!("[execute] Ctrl+C received, stopping execution");
            break;
        }

        if max_steps > 0 && prover.global_step() >= max_steps {
            eprintln!("[execute] Reached max steps limit ({})", max_steps);
            break;
        }

        let chunk = match prover.execute_chunk() {
            Ok(c) => c,
            Err(e) => {
                eprintln!();
                eprintln!("[execute] VM error at step {}: {}", prover.global_step(), e);
                break;
            }
        };

        // Flush UART bytes to stderr in real-time
        if !chunk.uart_bytes_this_chunk.is_empty() {
            total_uart_bytes += chunk.uart_bytes_this_chunk.len() as u64;
            let stderr = std::io::stderr();
            let mut lock = stderr.lock();
            let _ = lock.write_all(&chunk.uart_bytes_this_chunk);
            let _ = lock.flush();
        }

        if chunk.num_steps > 0 {
            chunks_executed += 1;

            if let Some(ref mut writer) = trace_writer {
                serialize_chunk(&chunk, writer).unwrap_or_else(|e| {
                    eprintln!("[execute] Failed to write chunk {}: {}", chunk.chunk_index, e);
                    process::exit(1);
                });
            } else {
                in_memory_chunks.push(chunk);
            }
        }

        // Progress report every 10s
        let now = Instant::now();
        if now.duration_since(last_report).as_secs() >= 10 {
            let elapsed = exec_start.elapsed().as_secs_f64();
            let steps = prover.global_step();
            eprintln!(
                "\n[execute] {} steps, {} chunks ({:.1}s, {:.0} steps/s)",
                steps, chunks_executed, elapsed, steps as f64 / elapsed,
            );
            last_report = now;
        }
    }

    // Flush trace file if used
    if let Some(ref mut writer) = trace_writer {
        writer.flush().unwrap_or_else(|e| {
            eprintln!("[execute] Failed to flush trace file: {}", e);
            process::exit(1);
        });
    }
    drop(trace_writer);

    let exec_elapsed = exec_start.elapsed();
    eprintln!();
    eprintln!("[execute] Done: {} chunks, {} steps in {:.1}s ({:.0} steps/s)",
        chunks_executed,
        prover.global_step(),
        exec_elapsed.as_secs_f64(),
        if exec_elapsed.as_secs_f64() > 0.0 { prover.global_step() as f64 / exec_elapsed.as_secs_f64() } else { 0.0 },
    );
    eprintln!("[execute] UART output: {} bytes", total_uart_bytes);

    if chunks_executed == 0 {
        eprintln!("[prove] No chunks to prove, exiting");
        process::exit(1);
    }

    // =======================================================================
    // Phase 2: Proving
    // =======================================================================
    let constraints = Arc::new(RiscvConstraintSystem::full());
    let prove_start = Instant::now();

    // Reset Ctrl+C flag so users can force-abort during proving
    STOP_REQUESTED.store(false, Ordering::SeqCst);

    eprintln!("[prove] ─── Proving {} chunks with {} workers ─────────────────",
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
                        eprintln!("[prove] proved {}/{} chunks ({:.1}%)",
                            done, total, done as f64 / total as f64 * 100.0);
                    }
                    if tx.send(chunk_proof).is_err() {
                        break; // Folder thread gone
                    }
                }
            })
            .unwrap_or_else(|e| {
                eprintln!("[prove] Failed to spawn worker {}: {}", worker_id, e);
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
                    if count % 10000 == 0 {
                        eprintln!("[prove]   folded {} chunks", count);
                    }
                },
                &*folder_scheme,
                folder_curve,
            )
        })
        .unwrap_or_else(|e| {
            eprintln!("[prove] Failed to spawn folder thread: {}", e);
            process::exit(1);
        });

    // Feed chunks to the work queue
    if let Some(ref path) = trace_file_path {
        // Read chunks back from trace file
        let file = fs::File::open(path).unwrap_or_else(|e| {
            eprintln!("[prove] Failed to open trace file {}: {}", path, e);
            process::exit(1);
        });
        let mut reader = BufReader::new(file);
        let _chunk_size = read_trace_header(&mut reader).unwrap_or_else(|e| {
            eprintln!("[prove] Failed to read trace header: {}", e);
            process::exit(1);
        });

        for _ in 0..chunks_executed {
            let chunk = deserialize_chunk(&mut reader).unwrap_or_else(|e| {
                eprintln!("[prove] Failed to deserialize chunk: {}", e);
                process::exit(1);
            });
            work_queue.push(Some(ProveWork {
                chunk_index: chunk.chunk_index,
                columns: chunk.columns,
                initial_state_hash: chunk.initial_state_hash,
                final_state_hash: chunk.final_state_hash,
            }));
        }
    } else {
        // Drain in-memory chunks
        for chunk in in_memory_chunks.drain(..) {
            work_queue.push(Some(ProveWork {
                chunk_index: chunk.chunk_index,
                columns: chunk.columns,
                initial_state_hash: chunk.initial_state_hash,
                final_state_hash: chunk.final_state_hash,
            }));
        }
    }

    // Send poison pills to workers
    for _ in 0..num_workers {
        work_queue.push(None);
    }

    for handle in worker_handles {
        let _ = handle.join();
    }

    eprintln!(
        "[prove] All {} chunks proved",
        chunks_proved.load(Ordering::Relaxed),
    );

    // Wait for the folder thread to complete tree folding
    let fold_result = folder_handle.join().unwrap_or_else(|_| {
        eprintln!("[prove] Folder thread panicked");
        process::exit(1);
    });

    let final_proof = fold_result.unwrap_or_else(|e| {
        eprintln!("[prove] Tree fold failed: {}", e);
        process::exit(1);
    });

    let prove_elapsed = prove_start.elapsed();
    eprintln!(
        "[prove] Folding complete, depth={} ({:.1}s)",
        final_proof.depth,
        prove_elapsed.as_secs_f64(),
    );

    // -----------------------------------------------------------------------
    // Step 7: Verify the final proof
    // -----------------------------------------------------------------------
    let valid = verify_final_scheme(&final_proof, &*scheme);
    eprintln!("[prove] Proof verified: {}", valid);

    // Report UART hash
    let uart_hash = prover.final_uart_hash();
    eprintln!("[prove] Final UART hash: {}", hex_string(&uart_hash));

    // Report state chain
    if let Some(initial) = &final_proof.initial_state_hash {
        eprintln!("[prove] Initial state: {}", hex_string(initial));
    }
    if let Some(final_h) = &final_proof.final_state_hash {
        eprintln!("[prove] Final state:   {}", hex_string(final_h));
    }

    // -----------------------------------------------------------------------
    // Step 8: Serialize and write proof
    // -----------------------------------------------------------------------
    let proof_bytes = serialize_proof(&final_proof);
    fs::write(&output_path, &proof_bytes).unwrap_or_else(|e| {
        eprintln!("[prove] Failed to write proof: {}", e);
        process::exit(1);
    });

    let total_elapsed = exec_start.elapsed();
    eprintln!("[prove] Proof written to: {} ({} bytes)", output_path, proof_bytes.len());
    eprintln!("[prove] Execution: {:.1}s, Proving: {:.1}s, Total: {:.1}s",
        exec_elapsed.as_secs_f64(), prove_elapsed.as_secs_f64(), total_elapsed.as_secs_f64());
    eprintln!("[prove] Verified: {}", valid);
}

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Minimal binary serialization of the recursive proof.
/// Format: [depth:8][initial_state:32][final_state:32][acc_commit:N][acc_proof:N][acc_value:N]
fn serialize_proof(proof: &metavm_zkp::recursive::RecursiveProof) -> Vec<u8> {
    let mut buf = Vec::new();

    // Header
    buf.extend_from_slice(b"MVMPROOF"); // magic
    buf.extend_from_slice(&1u32.to_le_bytes()); // version

    // Depth
    buf.extend_from_slice(&proof.depth.to_le_bytes());

    // State hashes
    buf.extend_from_slice(&proof.initial_state_hash.unwrap_or([0u8; 32]));
    buf.extend_from_slice(&proof.final_state_hash.unwrap_or([0u8; 32]));

    // Accumulator
    let acc = &proof.accumulator;
    buf.extend_from_slice(&(acc.l_acc.len() as u32).to_le_bytes());
    buf.extend_from_slice(&acc.l_acc);
    buf.extend_from_slice(&(acc.r_acc.len() as u32).to_le_bytes());
    buf.extend_from_slice(&acc.r_acc);
    buf.extend_from_slice(&acc.num_folded.to_le_bytes());

    // Current proof summary
    buf.extend_from_slice(&proof.current_proof.num_steps.to_le_bytes());
    buf.extend_from_slice(&proof.current_proof.domain_size.to_le_bytes());
    buf.extend_from_slice(&(proof.current_proof.column_commitments.len() as u32).to_le_bytes());
    for comm in &proof.current_proof.column_commitments {
        buf.extend_from_slice(&(comm.0.len() as u32).to_le_bytes());
        buf.extend_from_slice(&comm.0);
    }

    // Quotient commitments
    buf.extend_from_slice(&(proof.current_proof.num_quotient_chunks as u32).to_le_bytes());
    for qc in &proof.current_proof.quotient_commitments {
        buf.extend_from_slice(&(qc.0.len() as u32).to_le_bytes());
        buf.extend_from_slice(&qc.0);
    }

    // Evaluations
    buf.extend_from_slice(&(proof.current_proof.evaluations.len() as u32).to_le_bytes());
    for eval in &proof.current_proof.evaluations {
        buf.extend_from_slice(&(eval.len() as u32).to_le_bytes());
        buf.extend_from_slice(eval);
    }

    // Opening proof
    buf.extend_from_slice(&(proof.current_proof.opening_proof.d.len() as u32).to_le_bytes());
    buf.extend_from_slice(&proof.current_proof.opening_proof.d);
    buf.extend_from_slice(&(proof.current_proof.opening_proof.proof.len() as u32).to_le_bytes());
    buf.extend_from_slice(&proof.current_proof.opening_proof.proof);

    buf
}
