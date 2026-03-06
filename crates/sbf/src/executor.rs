//! SBF program execution wrapper with tracing.

use crate::trace::{SbfTraceRow, SbfTraceColumns, classify_instruction, compute_sbf_aux};
use solana_rbpf::aligned_memory::AlignedMemory;
use solana_rbpf::assembler::assemble;
use solana_rbpf::ebpf;
use solana_rbpf::elf::Executable;
use solana_rbpf::memory_region::{MemoryMapping, MemoryRegion};
use solana_rbpf::program::{BuiltinProgram, FunctionRegistry, SBPFVersion};
use solana_rbpf::vm::{Config, ContextObject, EbpfVm, TestContextObject};
use std::sync::Arc;

/// Execute an SBF ELF binary and return the execution trace.
pub fn execute_sbf_elf(elf_bytes: &[u8], input: &[u8]) -> Result<SbfTraceColumns, String> {
    let config = Config {
        enable_instruction_tracing: true,
        enable_instruction_meter: true,
        // Keep sbpf_v1 for wider ELF compatibility
        enable_sbpf_v1: true,
        enable_sbpf_v2: true,
        ..Config::default()
    };

    let mut function_registry = FunctionRegistry::default();
    crate::syscalls::register_syscalls(&mut function_registry)?;
    let loader = Arc::new(BuiltinProgram::new_loader(config, function_registry));

    let executable = Executable::<TestContextObject>::from_elf(elf_bytes, loader.clone())
        .map_err(|e| format!("Failed to load SBF ELF: {:?}", e))?;

    execute_with_config(&executable, &loader, &config, input)
}

/// Execute an assembled SBF program and return the execution trace.
pub fn execute_sbf_asm(asm_source: &str, input: &[u8]) -> Result<SbfTraceColumns, String> {
    let config = Config {
        enable_instruction_tracing: true,
        enable_instruction_meter: true,
        enable_sbpf_v1: true,
        enable_sbpf_v2: true,
        ..Config::default()
    };

    let function_registry = FunctionRegistry::default();
    let loader = Arc::new(BuiltinProgram::new_loader(config, function_registry));

    let executable = assemble::<TestContextObject>(asm_source, loader.clone())
        .map_err(|e| format!("Failed to assemble: {}", e))?;

    execute_with_config(&executable, &loader, &config, input)
}

fn execute_with_config(
    executable: &Executable<TestContextObject>,
    loader: &Arc<BuiltinProgram<TestContextObject>>,
    config: &Config,
    input: &[u8],
) -> Result<SbfTraceColumns, String> {
    let sbpf_version = executable.get_sbpf_version();

    let mut stack = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(config.stack_size());
    let stack_len = stack.len();
    let mut heap = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(32 * 1024); // 32KB heap

    let mut input_data = input.to_vec();

    let regions: Vec<MemoryRegion> = vec![
        executable.get_ro_region(),
        MemoryRegion::new_writable(stack.as_slice_mut(), ebpf::MM_STACK_START),
        MemoryRegion::new_writable(heap.as_slice_mut(), ebpf::MM_HEAP_START),
        MemoryRegion::new_writable(&mut input_data, ebpf::MM_INPUT_START),
    ];

    let memory_mapping = MemoryMapping::new(regions, config, sbpf_version)
        .map_err(|e| format!("Memory mapping failed: {:?}", e))?;

    let mut context = TestContextObject::new(1_000_000);
    let mut vm = EbpfVm::new(
        loader.clone(),
        sbpf_version,
        &mut context,
        memory_mapping,
        stack_len,
    );

    let (insn_count, result) = vm.execute_program(executable, true);

    match &result {
        solana_rbpf::error::ProgramResult::Ok(val) => {
            eprintln!("[sbf] Program returned: {} (0x{:x})", val, val);
        }
        solana_rbpf::error::ProgramResult::Err(e) => {
            eprintln!("[sbf] Program error: {:?}", e);
        }
    }
    eprintln!("[sbf] Instructions executed: {}", insn_count);
    eprintln!("[sbf] Trace entries: {}", context.trace_log.len());

    // Get the raw bytecode text section for instruction decoding
    let (text_vaddr, text_bytes) = executable.get_text_bytes();

    // Convert trace log to SbfTraceColumns
    let mut trace = SbfTraceColumns::new();
    let num_entries = context.trace_log.len();

    for (step, entry) in context.trace_log.iter().enumerate() {
        // entry = [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, pc]
        // pc is in instruction units (multiply by 8 for byte offset)
        let pc_insn = entry[11];
        let pc_byte = pc_insn * 8;

        // Decode instruction from text section
        let text_offset = pc_byte as usize;
        let (opcode, dst_reg, src_reg, _offset, immediate) = if text_offset + 8 <= text_bytes.len() {
            let insn_bytes = &text_bytes[text_offset..text_offset + 8];
            let opc = insn_bytes[0];
            let regs = insn_bytes[1];
            let dst = (regs & 0x0F) as u8;
            let src = ((regs >> 4) & 0x0F) as u8;
            let off = i16::from_le_bytes([insn_bytes[2], insn_bytes[3]]);
            let imm = i32::from_le_bytes([insn_bytes[4], insn_bytes[5], insn_bytes[6], insn_bytes[7]]);
            (opc, dst, src, off, imm as i64 as u64)
        } else {
            (0u8, 0u8, 0u8, 0i16, 0u64)
        };

        let (insn_type, funct) = classify_instruction(opcode);
        let dst_val_before = entry[dst_reg as usize];

        // Compute dst_val_after from the next trace entry's register state
        let dst_val_after = if step + 1 < num_entries {
            context.trace_log[step + 1][dst_reg as usize]
        } else {
            // Last instruction: use current value (likely unchanged for exit)
            entry[dst_reg as usize]
        };

        // For immediate-source instructions (opcode bit 3 = 0), the effective
        // source value is the immediate, not the src register.
        let source_is_imm = (opcode >> 3) & 1 == 0;
        let src_val = if source_is_imm {
            immediate
        } else {
            entry[src_reg as usize]
        };

        // Compute next_pc
        let next_pc = if step + 1 < num_entries {
            context.trace_log[step + 1][11] * 8
        } else {
            pc_byte + 8
        };

        let (aux0, aux1, aux2) = compute_sbf_aux(opcode, src_val, dst_val_before, dst_val_after);

        let row = SbfTraceRow {
            step: step as u64,
            pc: pc_byte,
            opcode,
            dst_reg,
            dst_val_before,
            dst_val_after,
            src_reg,
            src_val,
            mem_addr: 0,
            mem_val: 0,
            mem_size: 0,
            next_pc,
            insn_type,
            funct,
            immediate,
            aux0,
            aux1,
            aux2,
        };
        trace.push_row(&row);
    }

    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_simple_asm() {
        let asm = "mov64 r0, 42\nexit";
        let trace = execute_sbf_asm(asm, &[]).unwrap();
        assert!(trace.step.len() >= 2, "Should have at least 2 steps (mov + exit)");
    }

    #[test]
    fn test_execute_add_asm() {
        let asm = "mov64 r1, 10\nmov64 r2, 20\nadd64 r1, r2\nmov64 r0, r1\nexit";
        let trace = execute_sbf_asm(asm, &[]).unwrap();
        assert!(trace.step.len() >= 5, "Should have at least 5 steps");
    }

    #[test]
    fn test_execute_and_prove() {
        use metavm_zkp::commitment;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove;
        use metavm_zkp::verifier::verify;
        use crate::constraints::SbfConstraintSystem;

        commitment::init();

        let asm = "mov64 r0, 42\nexit";
        let trace = execute_sbf_asm(asm, &[]).unwrap();

        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = SbfConstraintSystem::new();
        let proof = prove(&polys, &cs);
        let valid = verify(&proof, &cs);
        assert!(valid, "Proof should verify for valid SBF execution");
    }

    #[test]
    fn test_execute_multi_insn_and_prove() {
        use metavm_zkp::commitment;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::prover::prove;
        use metavm_zkp::verifier::verify;
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        use crate::constraints::SbfConstraintSystem;

        commitment::init();

        let asm = "mov64 r1, 10\nmov64 r2, 20\nadd64 r1, r2\nmov64 r0, r1\nexit";
        let trace = execute_sbf_asm(asm, &[]).unwrap();

        // Dump trace for diagnosis
        for i in 0..trace.step.len() {
            eprintln!("[trace] step={} pc={} opcode=0x{:02x} dst_reg={} dst_before={} dst_after={} \
                       src_reg={} src_val={} insn_type={} funct={} imm={} next_pc={} aux0={} aux1={}",
                trace.step[i], trace.pc[i], trace.opcode[i],
                trace.dst_reg[i], trace.dst_val_before[i], trace.dst_val_after[i],
                trace.src_reg[i], trace.src_val[i],
                trace.insn_type[i], trace.funct[i], trace.immediate[i],
                trace.next_pc[i], trace.aux0[i], trace.aux1[i]);
        }

        // Verify constraints are satisfied on all rows
        let cs = SbfConstraintSystem::new();

        // Check constraint satisfaction on trace rows
        {
            let mut polys_check = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
            polys_check.fix_selector_padding(&cs);
            let columns = polys_check.columns();
            let eval = cs.evaluate_on_domain(&columns, polys_check.num_rows);
            for (ci, constraint_evals) in eval.iter().enumerate() {
                for (ri, val) in constraint_evals.iter().enumerate() {
                    if !val.is_zero() {
                        eprintln!("[diag] Constraint {} non-zero on row {}", ci, ri);
                    }
                }
            }
        }

        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let proof = prove(&polys, &cs);
        eprintln!("[diag] num_steps={}, domain_size={}, num_q_chunks={}",
            proof.num_steps, proof.domain_size, proof.num_quotient_chunks);
        let valid = verify(&proof, &cs);
        assert!(valid, "Multi-instruction SBF execution should produce a verifying proof");
    }
}
