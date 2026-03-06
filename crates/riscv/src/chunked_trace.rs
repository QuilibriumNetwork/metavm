use sha3::{Sha3_256, Digest};
use crate::state_hash::{self, IncrementalMemoryHash};
use crate::trace::{TracingVm, TraceColumns};
use crate::vm::{Vm, VmError};

/// A single 256-step execution chunk with trace data and boundary state hashes.
pub struct ExecutionChunk {
    pub chunk_index: u64,
    pub columns: TraceColumns,
    pub initial_state_hash: [u8; 32],
    pub final_state_hash: [u8; 32],
    pub uart_bytes_this_chunk: Vec<u8>,
    pub num_steps: u64,
}

/// Streaming prover that executes the VM in fixed-size chunks,
/// capturing trace data and boundary state hashes without accumulating
/// the full trace in memory.
pub struct StreamingProver {
    pub vm: Vm,
    chunk_size: usize,
    global_step: u64,
    chunk_index: u64,
    memory_hasher: IncrementalMemoryHash,
    uart_hasher: Sha3_256,
}

impl StreamingProver {
    pub fn new(vm: Vm, chunk_size: usize) -> Self {
        StreamingProver {
            vm,
            chunk_size,
            global_step: 0,
            chunk_index: 0,
            memory_hasher: IncrementalMemoryHash::new(),
            uart_hasher: Sha3_256::new(),
        }
    }

    /// Initialize the memory hasher by hashing all currently loaded pages.
    /// Must be called once before execute_chunk() to establish the initial
    /// memory hash.
    pub fn initialize_memory_hash(&mut self) {
        self.memory_hasher.initialize(&self.vm.memory);
        // Clear dirty set so subsequent updates are incremental
        self.vm.memory.clear_dirty();
    }

    /// Execute one chunk of `chunk_size` steps, returning the trace data
    /// and boundary state hashes.
    pub fn execute_chunk(&mut self) -> Result<ExecutionChunk, VmError> {
        // Snapshot initial state
        let mem_hash = self.memory_hasher.update(&mut self.vm.memory);
        let uart_hash = self.current_uart_hash();
        let initial_state = state_hash::snapshot_vm(
            &self.vm, mem_hash, uart_hash, self.global_step,
        );
        let initial_state_hash = state_hash::hash_boundary_state(&initial_state);

        // Create a TracingVm to execute steps with trace recording
        let mut tracing = TracingVm::new_borrowed(&mut self.vm);

        let mut steps_executed = 0u64;
        for _ in 0..self.chunk_size {
            if tracing.vm.halted {
                break;
            }
            tracing.step()?;
            steps_executed += 1;
        }

        // Export columns from the tracing wrapper
        let columns = tracing.export_columns();
        drop(tracing);

        // Drain UART bytes written during this chunk (frees memory)
        let uart_bytes_this_chunk: Vec<u8> = self.vm.mmio.uart.output.drain(..).collect();

        // Update UART hasher
        if !uart_bytes_this_chunk.is_empty() {
            self.uart_hasher.update(&uart_bytes_this_chunk);
        }

        self.global_step += steps_executed;

        // Snapshot final state
        let mem_hash = self.memory_hasher.update(&mut self.vm.memory);
        let uart_hash = self.current_uart_hash();
        let final_state = state_hash::snapshot_vm(
            &self.vm, mem_hash, uart_hash, self.global_step,
        );
        let final_state_hash = state_hash::hash_boundary_state(&final_state);

        let chunk = ExecutionChunk {
            chunk_index: self.chunk_index,
            columns,
            initial_state_hash,
            final_state_hash,
            uart_bytes_this_chunk,
            num_steps: steps_executed,
        };

        self.chunk_index += 1;

        Ok(chunk)
    }

    pub fn is_halted(&self) -> bool {
        self.vm.halted
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn chunk_index(&self) -> u64 {
        self.chunk_index
    }

    /// Get the current cumulative UART output hash.
    fn current_uart_hash(&self) -> [u8; 32] {
        self.uart_hasher.clone().finalize().into()
    }

    /// Get the final UART output hash after all chunks are complete.
    pub fn final_uart_hash(&self) -> [u8; 32] {
        self.current_uart_hash()
    }
}

impl TracingVm {
    /// Create a TracingVm that borrows the VM for trace recording.
    /// The trace is stored locally; the VM is modified in place.
    pub fn new_borrowed(vm: &mut Vm) -> TracingVmBorrowed<'_> {
        TracingVmBorrowed {
            vm,
            trace: Vec::new(),
        }
    }
}

/// A TracingVm that borrows the VM rather than owning it.
pub struct TracingVmBorrowed<'a> {
    pub vm: &'a mut Vm,
    pub trace: Vec<crate::trace::TraceRow>,
}

impl<'a> TracingVmBorrowed<'a> {
    pub fn step(&mut self) -> Result<(), VmError> {
        use crate::trace::{MemOp, TraceRow, compute_mem_info_with, compute_aux_values};

        if self.vm.halted {
            return Err(VmError::Halted);
        }

        // Snapshot pre-execution state: all 32 registers, PC, privilege mode.
        let pc_before = self.vm.cpu.pc;
        let priv_before = self.vm.cpu.priv_mode as u8;
        let regs_before: [u64; 32] = std::array::from_fn(|i| self.vm.cpu.read_reg(i as u8));

        // Execute one step (includes MMU translation, device ticking, trap delivery).
        self.vm.step()?;

        // Read the instruction that was actually fetched and executed.
        // last_fetched is None when step_inner delivered a trap (page fault,
        // illegal instruction) without executing an instruction.
        let (instruction, insn_len) = match self.vm.last_fetched.take() {
            Some(fetched) => fetched,
            None => return Ok(()), // trap delivery — no trace row
        };

        let rd = instruction.rd().unwrap_or(0);
        let rs1 = instruction.rs1().unwrap_or(0);
        let rs2 = instruction.rs2().unwrap_or(0);
        let rd_val_before = regs_before[rd as usize];
        let rs1_val = regs_before[rs1 as usize];
        let rs2_val = regs_before[rs2 as usize];
        let rd_val_after = self.vm.cpu.read_reg(rd);
        let next_pc = self.vm.cpu.pc;

        let (mem_addr, mem_op) = compute_mem_info_with(
            &instruction,
            |r| regs_before[r as usize],
        );

        let mem_val = match mem_op {
            MemOp::Read => rd_val_after,
            MemOp::Write => rs2_val,
            MemOp::None => 0,
        };

        let (aux0, aux1, aux2) = compute_aux_values(&instruction, rs1_val, rs2_val, rd_val_after);

        let step_num = self.trace.len() as u64;
        self.trace.push(TraceRow {
            step: step_num,
            pc: pc_before,
            instruction,
            rd,
            rd_val_before,
            rd_val_after,
            rs1,
            rs1_val,
            rs2,
            rs2_val,
            mem_addr,
            mem_val,
            mem_op,
            next_pc,
            privilege_mode: priv_before,
            insn_len,
            aux0,
            aux1,
            aux2,
        });

        Ok(())
    }

    pub fn export_columns(&self) -> TraceColumns {
        // Delegate to the TracingVm's static helper to avoid duplication
        crate::trace::export_trace_rows_to_columns(&self.trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuState;
    use crate::memory::Memory;

    fn make_test_vm(program: &[u32]) -> Vm {
        let mut memory = Memory::new();
        for (i, &word) in program.iter().enumerate() {
            memory.store_word((i * 4) as u64, word);
        }
        Vm::new(CpuState::new(), memory)
    }

    #[test]
    fn test_streaming_prover_basic() {
        // ADDI x1, x0, 42; ADDI x2, x1, 1; ECALL (3 steps: ADDI, ADDI, ECALL halts)
        let vm = make_test_vm(&[0x02A00093, 0x00108113, 0x00000073]);
        let mut prover = StreamingProver::new(vm, 256);
        prover.initialize_memory_hash();

        let chunk = prover.execute_chunk().unwrap();
        assert_eq!(chunk.chunk_index, 0);
        assert_eq!(chunk.num_steps, 3); // ADDI + ADDI + ECALL
        assert!(prover.is_halted());
    }

    #[test]
    fn test_streaming_prover_state_chain() {
        // 4 ADDI instructions then ECALL, chunk_size=2
        let vm = make_test_vm(&[
            0x02A00093, // ADDI x1, x0, 42
            0x00108113, // ADDI x2, x1, 1
            0x00110193, // ADDI x3, x2, 1
            0x00118213, // ADDI x4, x3, 1
            0x00000073, // ECALL
        ]);
        let mut prover = StreamingProver::new(vm, 2);
        prover.initialize_memory_hash();

        let chunk0 = prover.execute_chunk().unwrap();
        assert_eq!(chunk0.num_steps, 2);

        let chunk1 = prover.execute_chunk().unwrap();
        assert_eq!(chunk1.num_steps, 2);

        // State chain: chunk0's final hash == chunk1's initial hash
        assert_eq!(chunk0.final_state_hash, chunk1.initial_state_hash);

        // chunk2 will have ECALL (1 step)
        let chunk2 = prover.execute_chunk().unwrap();
        assert_eq!(chunk2.num_steps, 1);
        assert_eq!(chunk1.final_state_hash, chunk2.initial_state_hash);
        assert!(prover.is_halted());
    }

    #[test]
    fn test_streaming_prover_halts_mid_chunk() {
        // ADDI then ECALL - halts after 2 steps (ADDI + ECALL) in a chunk of 256
        let vm = make_test_vm(&[0x02A00093, 0x00000073]);
        let mut prover = StreamingProver::new(vm, 256);
        prover.initialize_memory_hash();

        let chunk = prover.execute_chunk().unwrap();
        assert_eq!(chunk.num_steps, 2); // ADDI + ECALL
        assert!(prover.is_halted());
    }
}
