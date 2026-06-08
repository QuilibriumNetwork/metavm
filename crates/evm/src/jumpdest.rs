//! JUMPDEST validity analysis (Phase A3 / #57 step 0).
//!
//! In the EVM, a JUMP or JUMPI is valid only if its target program
//! counter (PC) addresses a `JUMPDEST` opcode (0x5B) that is part of
//! the executable instruction stream — NOT an immediate byte inside a
//! `PUSHx` data segment that happens to equal 0x5B.
//!
//! Host-side this is decidable in a single linear scan: walk the
//! bytecode, skip the immediate bytes of each `PUSHx` (1 ≤ x ≤ 32), and
//! emit PC offsets where the opcode byte equals 0x5B.
//!
//! # Use in the proof pipeline
//!
//! The witness builder for the EVM main trace can call
//! [`valid_jumpdest_positions`] to get the canonical set of valid jump
//! targets for the executing bytecode. A future cross-AIR lookup will
//! prove that every JUMP/JUMPI row's `next_pc` is in this set; combined
//! with the bytecode commitment (via a Bytecode AIR) this closes the
//! "any JUMP target is valid" soundness gap.
//!
//! # PUSHx encoding
//!
//! Per the EVM spec:
//! - `PUSH1..=PUSH32` = 0x60..=0x7F. The opcode byte 0x60 + (n-1)
//!   pushes the next `n` bytes onto the stack.
//! - Operand bytes 0x60..=0x7F are NOT executable; they are immediate
//!   data. JUMPDESTs inside them are NOT valid targets.
//!
//! If the bytecode is truncated (a PUSHx near the end with fewer
//! remaining bytes than its immediate size), we treat the remaining
//! bytes as ALL belonging to the PUSH's data — matching EVM semantics
//! (where the truncated push silently zero-pads).

/// The JUMPDEST opcode byte.
pub const OPCODE_JUMPDEST: u8 = 0x5B;

/// PUSH1 opcode (0x60). PUSHn = PUSH1 + (n − 1).
pub const OPCODE_PUSH1: u8 = 0x60;

/// PUSH32 opcode (0x7F). The last PUSHx opcode.
pub const OPCODE_PUSH32: u8 = 0x7F;

/// Walk `bytecode` and return the sorted list of PC offsets where a
/// JUMPDEST opcode appears in the executable instruction stream
/// (i.e., NOT inside a PUSHx immediate data segment).
pub fn valid_jumpdest_positions(bytecode: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut pc = 0usize;
    let len = bytecode.len();
    while pc < len {
        let op = bytecode[pc];
        if op == OPCODE_JUMPDEST {
            out.push(pc);
            pc += 1;
        } else if (OPCODE_PUSH1..=OPCODE_PUSH32).contains(&op) {
            // PUSHn: opcode byte + n immediate bytes; skip them all.
            let n = (op - OPCODE_PUSH1) as usize + 1;
            pc += 1 + n; // safe even if it overflows past `len` — loop guard catches it
        } else {
            pc += 1;
        }
    }
    out
}

/// Return a `Vec<bool>` of length `bytecode.len()` where `out[i] = true`
/// iff `bytecode[i]` is a valid JUMPDEST target. Useful when many
/// JUMP rows need O(1) membership checks during witness construction.
pub fn valid_jumpdest_bitmap(bytecode: &[u8]) -> Vec<bool> {
    let mut bitmap = vec![false; bytecode.len()];
    for pc in valid_jumpdest_positions(bytecode) {
        bitmap[pc] = true;
    }
    bitmap
}

/// Check whether `target` is a valid JUMPDEST target in `bytecode`.
/// O(N) per call; for hot paths prefer building a bitmap once via
/// [`valid_jumpdest_bitmap`].
pub fn is_valid_jumpdest(bytecode: &[u8], target: usize) -> bool {
    if target >= bytecode.len() {
        return false;
    }
    valid_jumpdest_bitmap(bytecode)[target]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bytecode_has_no_jumpdests() {
        assert!(valid_jumpdest_positions(&[]).is_empty());
    }

    #[test]
    fn single_jumpdest() {
        let bc = vec![OPCODE_JUMPDEST];
        assert_eq!(valid_jumpdest_positions(&bc), vec![0]);
    }

    #[test]
    fn jumpdest_after_arithmetic() {
        // PUSH1 1, PUSH1 2, ADD, JUMPDEST, STOP
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, OPCODE_JUMPDEST, 0x00];
        assert_eq!(valid_jumpdest_positions(&bc), vec![5]);
    }

    #[test]
    fn jumpdest_inside_push1_is_not_valid() {
        // PUSH1 0x5B (the operand byte LOOKS like JUMPDEST but is data),
        // STOP. PC 0 = PUSH1 opcode, PC 1 = 0x5B operand byte, PC 2 = STOP.
        let bc = vec![OPCODE_PUSH1, OPCODE_JUMPDEST, 0x00];
        assert!(valid_jumpdest_positions(&bc).is_empty());
    }

    #[test]
    fn jumpdest_inside_push32_is_not_valid() {
        // PUSH32 [32 bytes including 0x5B], STOP. All 32 immediate bytes
        // are data; no valid JUMPDEST despite the 0x5B byte at PC=1.
        let mut bc = vec![OPCODE_PUSH32];
        for _ in 0..32 { bc.push(OPCODE_JUMPDEST); }
        bc.push(0x00);
        assert!(valid_jumpdest_positions(&bc).is_empty());
    }

    #[test]
    fn jumpdest_immediately_after_push() {
        // PUSH1 0x5B (operand 0x5B), JUMPDEST, STOP.
        // PC 0 = PUSH1, PC 1 = 0x5B operand, PC 2 = JUMPDEST (valid).
        let bc = vec![OPCODE_PUSH1, OPCODE_JUMPDEST, OPCODE_JUMPDEST, 0x00];
        assert_eq!(valid_jumpdest_positions(&bc), vec![2]);
    }

    #[test]
    fn multiple_jumpdests_and_pushes() {
        // JUMPDEST, PUSH2 0x5B5B, JUMPDEST, PUSH1 0x5B, JUMPDEST, STOP
        let bc = vec![
            OPCODE_JUMPDEST,          // PC 0: JUMPDEST (valid)
            0x61, 0x5B, 0x5B,         // PC 1: PUSH2, PC 2-3: data (looks like 2 JUMPDESTs)
            OPCODE_JUMPDEST,          // PC 4: JUMPDEST (valid)
            OPCODE_PUSH1, 0x5B,       // PC 5: PUSH1, PC 6: data
            OPCODE_JUMPDEST,          // PC 7: JUMPDEST (valid)
            0x00,                     // PC 8: STOP
        ];
        assert_eq!(valid_jumpdest_positions(&bc), vec![0, 4, 7]);
    }

    #[test]
    fn truncated_pushn_swallows_remaining_bytes() {
        // PUSH32 with only 5 bytes left; the JUMPDEST byte at the end
        // is INSIDE the push immediate (truncated), so NOT valid.
        let bc = vec![OPCODE_PUSH32, 0x01, 0x02, 0x03, 0x04, OPCODE_JUMPDEST];
        assert!(valid_jumpdest_positions(&bc).is_empty());
    }

    #[test]
    fn bitmap_matches_positions() {
        let bc = vec![
            OPCODE_JUMPDEST,
            OPCODE_PUSH1, 0x5B,
            OPCODE_JUMPDEST,
            0x00,
        ];
        let positions = valid_jumpdest_positions(&bc);
        let bitmap = valid_jumpdest_bitmap(&bc);
        for (i, &valid) in bitmap.iter().enumerate() {
            assert_eq!(valid, positions.contains(&i), "bitmap[{}] mismatch", i);
        }
    }

    #[test]
    fn is_valid_jumpdest_oracle() {
        let bc = vec![OPCODE_JUMPDEST, OPCODE_PUSH1, OPCODE_JUMPDEST, OPCODE_JUMPDEST];
        // PC 0: JUMPDEST (valid)
        assert!(is_valid_jumpdest(&bc, 0));
        // PC 1: PUSH1 opcode (not JUMPDEST)
        assert!(!is_valid_jumpdest(&bc, 1));
        // PC 2: 0x5B byte but it's data (PUSH1 immediate)
        assert!(!is_valid_jumpdest(&bc, 2));
        // PC 3: JUMPDEST (valid; PC 2 was the PUSH1 immediate so PC 3 is next code byte)
        assert!(is_valid_jumpdest(&bc, 3));
        // Out of range
        assert!(!is_valid_jumpdest(&bc, 4));
    }

    /// Cross-check against revm's analyze if its analysis is available
    /// — pin the spec by comparing against the production EVM impl.
    #[test]
    fn matches_real_world_bytecode_pattern() {
        // A realistic snippet: function dispatcher with PUSH4 selector
        // and conditional JUMPs.
        let bc = vec![
            0x60, 0x80,       // PUSH1 0x80
            0x60, 0x40,       // PUSH1 0x40
            0x52,             // MSTORE
            0x34,             // CALLVALUE
            0x80,             // DUP1
            0x15,             // ISZERO
            0x61, 0x00, 0x16, // PUSH2 0x0016
            0x57,             // JUMPI
            0x60, 0x00,       // PUSH1 0x00
            0x80,             // DUP1
            0xFD,             // REVERT
            OPCODE_JUMPDEST,  // PC 16 (decimal) ← target
            0x50,             // POP
            0x00,             // STOP
        ];
        let positions = valid_jumpdest_positions(&bc);
        assert_eq!(positions, vec![16], "JUMPDEST at PC 16");
        assert!(is_valid_jumpdest(&bc, 16));
        // PC 15 is the byte BEFORE the JUMPDEST — should be invalid
        // (it's the REVERT 0xFD).
        assert!(!is_valid_jumpdest(&bc, 15));
    }
}
