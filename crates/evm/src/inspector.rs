//! Custom revm Inspector that records each EVM step for trace generation.
//!
//! The inspector captures pre- and post-execution state at each opcode,
//! producing `EvmTraceRow` entries that populate an `EvmTraceColumns`.

use crate::trace::{
    EvmTraceRow, EvmTraceColumns, FrameState, classify_opcode, compute_evm_aux,
    shift_power_of_two,
};
use revm::interpreter::{Interpreter, Stack};
use revm::interpreter::interpreter_types::{Jumps, LegacyBytecode};
use revm::Inspector;
use revm::context_interface::{ContextTr, journaled_state::JournalTr};

/// Pending row state captured in step() before execution.
struct PendingRow {
    step: u64,
    pc: u64,
    opcode: u8,
    gas_remaining: u64,
    stack_depth: u64,
    input0: [u64; 4],
    input1: [u64; 4],
    immediate: [u64; 4],
    insn_type: u8,
    funct: u8,
    /// Snapshot of the current frame at the time the step began.
    frame: FrameState,
    /// Address hint populated on CREATE/CREATE2 rows. We don't know the
    /// address at step()-time; the inspector's `create()` hook (which
    /// fires after step_end()) backfills it on the just-pushed trace row.
    create_address_hint: [u64; 4],
}

/// Inspector that records EVM execution traces.
pub struct TracingInspector {
    pub trace: EvmTraceColumns,
    pending: Option<PendingRow>,
    step_count: u64,
    /// Stack of active frames. The top of the stack is the current frame.
    /// The Inspector::call hook pushes; Inspector::call_end pops.
    frame_stack: Vec<FrameState>,
}

impl TracingInspector {
    pub fn new() -> Self {
        TracingInspector {
            trace: EvmTraceColumns::new(),
            pending: None,
            step_count: 0,
            frame_stack: Vec::new(),
        }
    }

    /// Return a copy of the current top-of-stack frame, or default if empty.
    fn current_frame(&self) -> FrameState {
        self.frame_stack
            .last()
            .cloned()
            .unwrap_or_default()
    }
}

/// Convert an Ethereum 20-byte address into 4 u64 limbs (little-endian).
/// Limb 0 contains the least-significant 64 bits; address fits in 160 bits so
/// limbs 2 and 3 only carry the high 32 bits combined.
fn address_to_limbs(addr: &revm::primitives::Address) -> [u64; 4] {
    let bytes = addr.0.0; // [u8; 20]
    let mut limbs = [0u64; 4];
    // Little-endian: limb[i] is bytes 8*i..8*i+8.
    for i in 0..2 {
        let mut tmp = [0u8; 8];
        tmp.copy_from_slice(&bytes[8 * i..8 * i + 8]);
        limbs[i] = u64::from_le_bytes(tmp);
    }
    // Last limb carries the top 4 bytes of the 20-byte address (zero-extended).
    let mut tmp = [0u8; 8];
    tmp[..4].copy_from_slice(&bytes[16..20]);
    limbs[2] = u64::from_le_bytes(tmp);
    limbs
}

/// Extract limbs from a revm U256 value.
fn u256_limbs(val: &revm::primitives::U256) -> [u64; 4] {
    val.as_limbs().clone()
}

/// Convert a 32-byte hash to 4 little-endian u64 limbs (limb[i] is bytes 8i..8i+8).
fn b256_to_le_limbs(hash: &revm::primitives::B256) -> [u64; 4] {
    let bytes = hash.0; // [u8; 32]
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        let mut tmp = [0u8; 8];
        tmp.copy_from_slice(&bytes[8 * i..8 * i + 8]);
        limbs[i] = u64::from_le_bytes(tmp);
    }
    limbs
}

/// Peek a stack value safely, returning zero if index is out of range.
fn safe_peek(stack: &Stack, index: usize) -> [u64; 4] {
    if index < stack.len() {
        match stack.peek(index) {
            Ok(val) => u256_limbs(&val),
            Err(_) => [0u64; 4],
        }
    } else {
        [0u64; 4]
    }
}

impl<CTX: ContextTr> Inspector<CTX> for TracingInspector {
    fn step(&mut self, interp: &mut Interpreter, _context: &mut CTX) {
        let pc = interp.bytecode.pc() as u64;
        let opcode = interp.bytecode.opcode();
        let gas_remaining = interp.gas.remaining();
        let stack_depth = interp.stack.len() as u64;
        let (insn_type, funct) = classify_opcode(opcode);

        let input0 = safe_peek(&interp.stack, 0);
        let input1 = {
            let raw = safe_peek(&interp.stack, 1);
            // Per #298: CALL-family opcodes encode stack[1] as a 20-byte
            // Ethereum address. The shifted cross-row constraint compares
            // INPUT1_L0 with frame_callee_l0 = address_to_limbs(addr)[0],
            // which is `u64::from_le_bytes(addr.bytes[0..8])` — the
            // low-byte limb of the BE 20-byte address. The default U256
            // `as_limbs()[0]` differs by byte rotation (it's the low 64
            // bits of the integer = bytes [24..32] of the 32-byte BE
            // repr), so the shifted constraint can never hold.
            //
            // Rewrite input1 to address_to_limbs(addr) for CALL family
            // so INPUT1_L0 matches frame_callee_l0 on the next row.
            //
            // CALL = 0xF1, CALLCODE = 0xF2, DELEGATECALL = 0xF4,
            // STATICCALL = 0xFA. stack[1] is the address operand for all
            // four (gas is stack[0]).
            if opcode == 0xF1 || opcode == 0xF2 || opcode == 0xF4 || opcode == 0xFA {
                // Reconstruct an Address from the U256 value: the low
                // 160 bits of the integer are the address. The
                // 32-byte BE repr of a U256 places the address in
                // bytes [12..32].
                if 1 < interp.stack.len() {
                    if let Ok(val) = interp.stack.peek(1) {
                        let be: [u8; 32] = val.to_be_bytes();
                        let mut addr_bytes = [0u8; 20];
                        addr_bytes.copy_from_slice(&be[12..32]);
                        let addr = revm::primitives::Address::from(addr_bytes);
                        address_to_limbs(&addr)
                    } else {
                        raw
                    }
                } else {
                    raw
                }
            } else {
                raw
            }
        };

        // For PUSH instructions, extract the immediate value from bytecode
        let immediate = if opcode >= 0x60 && opcode <= 0x7F {
            let push_size = (opcode - 0x60 + 1) as usize;
            let bytecode = interp.bytecode.bytecode_slice();
            let pc_val = pc as usize;
            let mut imm_bytes = [0u8; 32];
            let start = pc_val + 1;
            let end = (start + push_size).min(bytecode.len());
            if start < bytecode.len() {
                let available = end - start;
                // Big-endian: value bytes are placed at the end
                imm_bytes[32 - available..32].copy_from_slice(&bytecode[start..end]);
            }
            // Convert big-endian bytes to U256 limbs (limb 0 = least significant)
            // imm_bytes is big-endian: [0] is MSB, [31] is LSB
            // Limb 0 = bytes [24..32] (least significant 64 bits)
            // Limb 3 = bytes [0..8]   (most significant 64 bits)
            [
                u64::from_be_bytes([imm_bytes[24], imm_bytes[25], imm_bytes[26], imm_bytes[27],
                                    imm_bytes[28], imm_bytes[29], imm_bytes[30], imm_bytes[31]]),
                u64::from_be_bytes([imm_bytes[16], imm_bytes[17], imm_bytes[18], imm_bytes[19],
                                    imm_bytes[20], imm_bytes[21], imm_bytes[22], imm_bytes[23]]),
                u64::from_be_bytes([imm_bytes[8], imm_bytes[9], imm_bytes[10], imm_bytes[11],
                                    imm_bytes[12], imm_bytes[13], imm_bytes[14], imm_bytes[15]]),
                u64::from_be_bytes([imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3],
                                    imm_bytes[4], imm_bytes[5], imm_bytes[6], imm_bytes[7]]),
            ]
        } else if opcode == 0x1B || opcode == 0x1C || opcode == 0x1D {
            // SHL/SHR/SAR: immediate = 2^k where k = shift amount (input0)
            shift_power_of_two(input0)
        } else if opcode == 0x08 || opcode == 0x09 {
            // ADDMOD/MULMOD: capture the third stack operand (the modulus n).
            safe_peek(&interp.stack, 2)
        } else if opcode >= 0xA1 && opcode <= 0xA4 {
            // LOG1..LOG4: capture topic0 (stack[2]) in immediate.
            safe_peek(&interp.stack, 2)
        } else {
            [0u64; 4]
        };

        self.pending = Some(PendingRow {
            step: self.step_count,
            pc,
            opcode,
            gas_remaining,
            stack_depth,
            input0,
            input1,
            immediate,
            insn_type,
            funct,
            frame: self.current_frame(),
            create_address_hint: [0u64; 4],
        });
    }

    fn step_end(&mut self, interp: &mut Interpreter, _context: &mut CTX) {
        if let Some(pending) = self.pending.take() {
            // Capture post-execution output (top of stack after execution)
            let output0 = safe_peek(&interp.stack, 0);

            // Memory access: for MLOAD/MSTORE, capture address and value
            let (mem_offset, mem_value) = match pending.opcode {
                0x51 => {
                    // MLOAD: offset was input0, loaded value is output0
                    (pending.input0[0], output0)
                }
                0x52 => {
                    // MSTORE: offset was input0, stored value was input1
                    (pending.input0[0], pending.input1)
                }
                0x53 => {
                    // MSTORE8: offset was input0, value (1 byte) was input1
                    (pending.input0[0], pending.input1)
                }
                _ => (0u64, [0u64; 4]),
            };

            let (aux0, aux1) = compute_evm_aux(
                pending.opcode,
                pending.input0,
                pending.input1,
                output0,
            );

            let next_pc = interp.bytecode.pc() as u64;

            let row = EvmTraceRow {
                step: pending.step,
                pc: pending.pc,
                opcode: pending.opcode,
                gas_remaining: pending.gas_remaining,
                stack_depth: pending.stack_depth,
                input0: pending.input0,
                input1: pending.input1,
                output0,
                mem_offset,
                mem_value,
                insn_type: pending.insn_type,
                funct: pending.funct,
                immediate: pending.immediate,
                aux0,
                aux1,
                next_pc,
                frame: pending.frame,
                create_address_hint: pending.create_address_hint,
                // Always pushed as 0; the `create()` hook below backfills
                // the actual nonce on CREATE rows after step_end runs.
                create_nonce_hint: 0,
                // sel_stop_pop is recomputed by push_row from
                // (opcode, frame.depth) — value here is overwritten.
                sel_stop_pop: 0,
                // Always pushed as 0; the `create()` hook below backfills
                // both fields on CREATE2 rows from `inputs.scheme.salt()`
                // and keccak256(`inputs.init_code`).
                create2_salt_hint: [0u64; 4],
                create2_initcode_hash_hint: [0u64; 4],
                tx_origin: [0u64; 4],
                tx_gas_price: 0,
                tx_calldata_size: 0,
                tx_code_size: 0,
                returndata_size: 0,
            };
            self.trace.push_row(&row);
            self.step_count += 1;
        }
    }

    /// Push a new frame onto the frame stack just before revm enters a CALL
    /// (or CALLCODE/DELEGATECALL/STATICCALL). The first call in a transaction
    /// creates the top-level frame.
    fn call(
        &mut self,
        _context: &mut CTX,
        inputs: &mut revm::interpreter::CallInputs,
    ) -> Option<revm::interpreter::CallOutcome> {
        // Capture the PC of the calling row so that on RETURN the caller
        // resumes at pc+1 (which is what revm does internally). For the
        // top-level frame there is no caller row, so return_pc is 0.
        let return_pc = self
            .trace
            .pc
            .last()
            .copied()
            .map(|pc| pc + 1)
            .unwrap_or(0);
        let return_offset = inputs.return_memory_offset.start as u64;
        let return_size = inputs.return_memory_offset.len() as u64;
        let value_limbs: [u64; 4] = match inputs.value {
            revm::interpreter::CallValue::Transfer(v) => *v.as_limbs(),
            revm::interpreter::CallValue::Apparent(v) => *v.as_limbs(),
        };
        // Determine static-mode propagation. The CallInputs scheme distinguishes
        // STATICCALL from regular CALL/CALLCODE/DELEGATECALL. Once a static
        // frame is entered, all nested frames inherit static mode.
        let parent_static = self.frame_stack.last().map(|f| f.is_static).unwrap_or(0);
        let scheme_is_static = matches!(
            inputs.scheme,
            revm::interpreter::CallScheme::StaticCall
        );
        let is_static = if parent_static == 1 || scheme_is_static { 1 } else { 0 };
        // For DELEGATECALL the new frame inherits the parent's caller and
        // value rather than using the inputs' caller / value (which for
        // DELEGATECALL refer to the active address — see revm semantics).
        let scheme_is_delegate = matches!(
            inputs.scheme,
            revm::interpreter::CallScheme::DelegateCall
        );
        let (frame_caller, frame_value) = if scheme_is_delegate {
            let parent = self.frame_stack.last().cloned().unwrap_or_default();
            (parent.caller, parent.value)
        } else {
            (address_to_limbs(&inputs.caller), value_limbs)
        };
        let new_frame = FrameState {
            depth: self.frame_stack.len() as u64,
            caller: frame_caller,
            callee: address_to_limbs(&inputs.target_address),
            value: frame_value,
            gas: inputs.gas_limit,
            return_pc,
            return_offset,
            return_size,
            is_static,
        };
        self.frame_stack.push(new_frame);
        None
    }

    /// Pop the just-completed frame from the frame stack.
    fn call_end(
        &mut self,
        _context: &mut CTX,
        _inputs: &revm::interpreter::CallInputs,
        _outcome: &mut revm::interpreter::CallOutcome,
    ) {
        let _ = self.frame_stack.pop();
    }

    /// Push a new frame onto the frame stack just before revm enters a
    /// CREATE / CREATE2. The frame's callee is the freshly-derived contract
    /// address (computed by revm from sender+nonce or salt+initcode).
    fn create(
        &mut self,
        context: &mut CTX,
        inputs: &mut revm::interpreter::CreateInputs,
    ) -> Option<revm::interpreter::CreateOutcome> {
        // The CREATE opcode row was pushed onto the trace just before this
        // hook fires (step → interp executes CREATE → step_end pushes the
        // row → this hook fires). Read the sender's live nonce from
        // revm's journal — this hook is documented to fire BEFORE revm
        // bumps the sender's nonce, so the value we read is the
        // pre-bump nonce that the canonical CREATE address derivation
        // hashes (per EIP-150: address = keccak256(rlp([sender, nonce]))[12..]
        // with `nonce` = sender's nonce *before* the increment).
        let caller = inputs.caller();
        let nonce = match context.journal_mut().load_account(caller) {
            Ok(state_load) => state_load.data.info.nonce,
            // If the journal can't load the sender (shouldn't happen for
            // a CREATE that revm is about to execute), fall back to 0
            // so the trace at least has a deterministic value. The
            // cross-AIR linkage will then disagree with the address
            // hint, surfacing the bug in tests.
            Err(_) => 0,
        };
        let new_addr = inputs.created_address(nonce);
        let addr_limbs = address_to_limbs(&new_addr);
        // Backfill the create_address_hint AND create_nonce_hint on the
        // just-pushed CREATE row (the last row in the trace). Also
        // backfill the CREATE2-only salt + initcode_hash hints; for plain
        // CREATE these stay zero.
        if let Some(last) = self.trace.create_address_hint[0].len().checked_sub(1) {
            for j in 0..4 {
                self.trace.create_address_hint[j][last] = addr_limbs[j];
            }
            self.trace.create_nonce_hint[last] = nonce;
            if let revm::context_interface::CreateScheme::Create2 { salt } = inputs.scheme() {
                // CREATE2's salt is treated as a 32-byte BE blob inside the
                // keccak256 pre-image (per EIP-1014). To make the salt limbs
                // align byte-for-byte with the gadget's
                // `bytes32_to_le_limbs(salt_be)` decomposition, decompose the
                // BE byte string LE-wise (NOT the U256 numerical limbs from
                // `u256_limbs(&salt)`, which would give a different byte
                // order and break the cross-AIR linkage).
                let salt_be: [u8; 32] = salt.to_be_bytes();
                let salt_limbs = b256_to_le_limbs(&revm::primitives::B256::from(salt_be));
                let initcode_hash = revm::primitives::keccak256(inputs.init_code().as_ref());
                let hash_limbs = b256_to_le_limbs(&initcode_hash);
                for j in 0..4 {
                    self.trace.create2_salt_hint[j][last] = salt_limbs[j];
                    self.trace.create2_initcode_hash_hint[j][last] = hash_limbs[j];
                }
            }
        }

        let return_pc = self
            .trace
            .pc
            .last()
            .copied()
            .map(|pc| pc + 1)
            .unwrap_or(0);
        let value_limbs = *inputs.value().as_limbs();
        let parent_static = self.frame_stack.last().map(|f| f.is_static).unwrap_or(0);
        let new_frame = FrameState {
            depth: self.frame_stack.len() as u64,
            caller: address_to_limbs(&inputs.caller()),
            callee: addr_limbs,
            value: value_limbs,
            gas: inputs.gas_limit(),
            return_pc,
            return_offset: 0,
            return_size: 0,
            // CREATE/CREATE2 cannot be issued from a STATICCALL (revm enforces
            // this), but if it somehow was, static mode would propagate.
            is_static: parent_static,
        };
        self.frame_stack.push(new_frame);
        None
    }

    /// Pop the just-completed CREATE frame.
    fn create_end(
        &mut self,
        _context: &mut CTX,
        _inputs: &revm::interpreter::CreateInputs,
        _outcome: &mut revm::interpreter::CreateOutcome,
    ) {
        let _ = self.frame_stack.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;
    use crate::trace::{FUNCT_ADDMOD, FUNCT_MULMOD, INSN_ARITH};

    /// Find the trace row index where the given opcode was executed.
    fn find_opcode_row(trace: &EvmTraceColumns, opcode: u8) -> Option<usize> {
        for i in 0..trace.opcode.len() {
            if trace.opcode[i] == opcode as u64 {
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn test_addmod_captures_modulus_in_immediate() {
        // ADDMOD computes (a + b) % n where the EVM stack layout is:
        //   top -> a, b, n  (a pushed last, so popped first)
        // Expected: (5 + 7) % 11 = 1
        //
        // Bytecode order pushes n first, then b, then a:
        //   PUSH1 11  (n)
        //   PUSH1 7   (b)
        //   PUSH1 5   (a)
        //   ADDMOD
        //   STOP
        let bytecode = vec![
            0x60, 0x0B, // PUSH1 11  -- modulus n
            0x60, 0x07, // PUSH1 7   -- b
            0x60, 0x05, // PUSH1 5   -- a
            0x08,       // ADDMOD
            0x00,       // STOP
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let row = find_opcode_row(&trace, 0x08).expect("ADDMOD row must be present");

        // Sanity: selector and classification are correct.
        assert_eq!(trace.insn_type[row], INSN_ARITH as u64);
        assert_eq!(trace.funct[row], FUNCT_ADDMOD as u64);
        assert_eq!(trace.sel_addmod[row], 1);

        // Inputs: input0 = a = 5, input1 = b = 7.
        assert_eq!(trace.input0[0][row], 5);
        assert_eq!(trace.input0[1][row], 0);
        assert_eq!(trace.input0[2][row], 0);
        assert_eq!(trace.input0[3][row], 0);
        assert_eq!(trace.input1[0][row], 7);
        assert_eq!(trace.input1[1][row], 0);
        assert_eq!(trace.input1[2][row], 0);
        assert_eq!(trace.input1[3][row], 0);

        // The third operand n must be captured in the immediate columns.
        assert_eq!(trace.immediate[0][row], 11, "modulus n low limb");
        assert_eq!(trace.immediate[1][row], 0);
        assert_eq!(trace.immediate[2][row], 0);
        assert_eq!(trace.immediate[3][row], 0);

        // Output: (5 + 7) % 11 = 1
        assert_eq!(trace.output0[0][row], 1);
        assert_eq!(trace.output0[1][row], 0);
        assert_eq!(trace.output0[2][row], 0);
        assert_eq!(trace.output0[3][row], 0);
    }

    #[test]
    fn test_mulmod_captures_modulus_in_immediate() {
        // MULMOD computes (a * b) % n. Expected: (5 * 7) % 9 = 35 % 9 = 8.
        let bytecode = vec![
            0x60, 0x09, // PUSH1 9  -- modulus n
            0x60, 0x07, // PUSH1 7  -- b
            0x60, 0x05, // PUSH1 5  -- a
            0x09,       // MULMOD
            0x00,       // STOP
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let row = find_opcode_row(&trace, 0x09).expect("MULMOD row must be present");

        assert_eq!(trace.insn_type[row], INSN_ARITH as u64);
        assert_eq!(trace.funct[row], FUNCT_MULMOD as u64);
        assert_eq!(trace.sel_mulmod[row], 1);

        assert_eq!(trace.input0[0][row], 5);
        assert_eq!(trace.input1[0][row], 7);

        // The third operand n must be captured in the immediate columns.
        assert_eq!(trace.immediate[0][row], 9, "modulus n low limb");
        assert_eq!(trace.immediate[1][row], 0);
        assert_eq!(trace.immediate[2][row], 0);
        assert_eq!(trace.immediate[3][row], 0);

        // Output: (5 * 7) % 9 = 8
        assert_eq!(trace.output0[0][row], 8);
    }

    #[test]
    fn test_non_addmod_mulmod_immediate_unchanged() {
        // Verify that a plain ADD (0x01) row does NOT populate immediate from
        // the stack — this would be a regression breaking PUSH semantics.
        let bytecode = vec![
            0x60, 0x0A, // PUSH1 10
            0x60, 0x14, // PUSH1 20
            0x01,       // ADD  (not ADDMOD/MULMOD)
            0x00,       // STOP
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let row = find_opcode_row(&trace, 0x01).expect("ADD row must be present");

        // ADD has no immediate, so all limbs must be zero.
        for j in 0..4 {
            assert_eq!(trace.immediate[j][row], 0,
                "ADD row must not populate immediate (limb {})", j);
        }
    }

    /// Verifies that the inspector's `create()` hook reads the live
    /// sender nonce from revm's journal and stores it in the EVM trace's
    /// new `create_nonce_hint` column. Closes the residual scope of #99
    /// and #95: the gadget linkage now sees a real nonce, not a placeholder.
    #[test]
    fn test_create_populates_nonce_hint_from_live_state() {
        // Bytecode that does CREATE with empty init code:
        //   PUSH1 0  (size = 0)
        //   PUSH1 0  (offset = 0)
        //   PUSH1 0  (value = 0)
        //   CREATE   (opcode 0xF0)
        //   STOP
        let bytecode = vec![
            0x60, 0x00, // PUSH1 0
            0x60, 0x00, // PUSH1 0
            0x60, 0x00, // PUSH1 0
            0xF0,       // CREATE
            0x00,       // STOP
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let row = find_opcode_row(&trace, 0xF0)
            .expect("CREATE opcode row must be present");

        // The contract address used in `execute_bytecode` is initialized
        // with `nonce: 0` and the CREATE is the first state-changing
        // operation against that account. revm's `create()` hook fires
        // BEFORE bumping the sender's nonce, so the nonce we capture
        // should be 0.
        assert_eq!(
            trace.create_nonce_hint[row], 0,
            "CREATE row must capture pre-bump sender nonce (= 0 for fresh contract)"
        );

        // The address hint must now be derived using the same nonce
        // (previously the inspector hard-coded `0` to the placeholder
        // `inputs.created_address(0)` call; we still get 0 here, but
        // by reading from the journal — so the value is now consistent
        // with whatever revm later observes for state changes).
        // Sanity: at least one limb should be nonzero (the address is
        // a hash, not all-zero).
        let addr_nonzero = (0..4).any(|j| trace.create_address_hint[j][row] != 0);
        assert!(addr_nonzero, "create_address_hint must be populated");

        // All non-CREATE rows must leave create_nonce_hint = 0.
        for r in 0..trace.create_nonce_hint.len() {
            if r == row {
                continue;
            }
            assert_eq!(
                trace.create_nonce_hint[r], 0,
                "non-CREATE row {} must leave create_nonce_hint = 0", r
            );
        }
    }

    /// Verifies that the inspector's `create()` hook backfills the
    /// CREATE2 `salt` and `keccak256(initcode)` hints into the new
    /// `create2_salt_hint` / `create2_initcode_hash_hint` columns.
    /// Closes the residual scope of #117 and the EVM-side adapter for
    /// the CREATE2 input cross-AIR linkage.
    #[test]
    fn test_create2_populates_salt_and_initcode_hash() {
        // Bytecode that does CREATE2 with empty init code and salt = 1:
        //   PUSH1 1   (salt)
        //   PUSH1 0   (size)
        //   PUSH1 0   (offset)
        //   PUSH1 0   (value)
        //   CREATE2   (opcode 0xF5)
        //   STOP
        let bytecode = vec![
            0x60, 0x01, // PUSH1 1
            0x60, 0x00, // PUSH1 0
            0x60, 0x00, // PUSH1 0
            0x60, 0x00, // PUSH1 0
            0xF5,       // CREATE2
            0x00,       // STOP
        ];

        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let row = find_opcode_row(&trace, 0xF5)
            .expect("CREATE2 opcode row must be present");

        // salt limbs: PUSH1 1 sets salt = 1; in BE byte order this is
        // [0; 31] || [1]. The LE-byte→limb decomposition gives limb 0..2
        // = 0 and limb 3 = u64::from_le_bytes([0,0,0,0,0,0,0,1]) =
        // 0x0100_0000_0000_0000.
        assert_eq!(trace.create2_salt_hint[0][row], 0, "salt limb 0");
        assert_eq!(trace.create2_salt_hint[1][row], 0, "salt limb 1");
        assert_eq!(trace.create2_salt_hint[2][row], 0, "salt limb 2");
        assert_eq!(
            trace.create2_salt_hint[3][row], 0x0100_0000_0000_0000,
            "salt limb 3 (top byte = 1 in BE encoding)"
        );

        // keccak256("") = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
        // LE u64 limbs: limb[0] = u64::from_le_bytes(bytes[0..8]) = 0x3c23f78601462dc5
        let empty_hash = revm::primitives::keccak256(&[]);
        let expected_l0 = u64::from_le_bytes(empty_hash.0[0..8].try_into().unwrap());
        let expected_l1 = u64::from_le_bytes(empty_hash.0[8..16].try_into().unwrap());
        let expected_l2 = u64::from_le_bytes(empty_hash.0[16..24].try_into().unwrap());
        let expected_l3 = u64::from_le_bytes(empty_hash.0[24..32].try_into().unwrap());
        assert_eq!(trace.create2_initcode_hash_hint[0][row], expected_l0, "initcode_hash limb 0");
        assert_eq!(trace.create2_initcode_hash_hint[1][row], expected_l1, "initcode_hash limb 1");
        assert_eq!(trace.create2_initcode_hash_hint[2][row], expected_l2, "initcode_hash limb 2");
        assert_eq!(trace.create2_initcode_hash_hint[3][row], expected_l3, "initcode_hash limb 3");

        // All non-CREATE2 rows must leave the hints zero.
        for r in 0..trace.create2_salt_hint[0].len() {
            if r == row {
                continue;
            }
            for k in 0..4 {
                assert_eq!(trace.create2_salt_hint[k][r], 0,
                    "non-CREATE2 row {} salt limb {} must be 0", r, k);
                assert_eq!(trace.create2_initcode_hash_hint[k][r], 0,
                    "non-CREATE2 row {} initcode_hash limb {} must be 0", r, k);
            }
        }
    }
}
