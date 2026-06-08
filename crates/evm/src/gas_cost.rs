//! Per-opcode EVM gas cost table (Phase A3 / #60 step 0).
//!
//! Returns the gas cost for opcodes whose cost is constant (not
//! state-dependent or operand-dependent). For dynamic-cost opcodes
//! (SLOAD/SSTORE/CALL family/EXP/SHA3/LOG/MLOAD/MSTORE/COPY-class etc.)
//! returns `None`; their costs are computed per-row by the inspector
//! and bound to the trace's `gas_remaining` column via a future cross-
//! row constraint.
//!
//! Reference: Ethereum yellow paper post-Cancun (EIP-3198, EIP-2929,
//! EIP-4844 in force). Costs reflect the BASE component; warm/cold
//! access surcharges and operand-dependent terms are applied dynamically.
//!
//! ## Scope
//!
//! This is the host-side static table consumed by:
//! 1. Witness builders that need to verify `gas_remaining`'s decrement
//!    matches per-opcode cost on constant-gas rows.
//! 2. The future BytecodeAir + cross-AIR LogUp that will bind every
//!    EVM main row's `(opcode, gas_cost)` to a real lookup against
//!    this table, closing the gas-soundness gap on simple opcodes.
//!
//! Dynamic-gas opcodes are deferred — they need per-opcode constraint
//! modules computing their actual cost from row inputs.

/// Static gas cost for `opcode` IF it has a constant cost in the
/// current hardfork. Returns `None` for opcodes whose cost depends on
/// operand values, runtime state, memory expansion, or call depth.
///
/// Costs are from the post-Cancun yellow paper:
/// - W_zero  (0):    STOP, RETURN, REVERT, INVALID, SELFDESTRUCT (handled specially)
/// - W_base  (2):    ADDRESS, ORIGIN, CALLER, CALLVALUE, CALLDATASIZE,
///                   CODESIZE, GASPRICE, COINBASE, TIMESTAMP, NUMBER,
///                   PREVRANDAO, GASLIMIT, CHAINID, RETURNDATASIZE,
///                   POP, PC, MSIZE, GAS, BASEFEE
/// - W_verylow (3):  ADD, SUB, NOT, LT, GT, SLT, SGT, EQ, ISZERO, AND,
///                   OR, XOR, BYTE, SHL, SHR, SAR, CALLDATALOAD, MLOAD,
///                   MSTORE, MSTORE8, PUSH0-32, DUP1-16, SWAP1-16
/// - W_low     (5):  MUL, DIV, SDIV, MOD, SMOD, SIGNEXTEND, SELFBALANCE
/// - W_mid     (8):  ADDMOD, MULMOD, JUMP
/// - W_high   (10):  JUMPI
/// - JUMPDEST   (1):
pub fn static_gas_cost(opcode: u8) -> Option<u64> {
    match opcode {
        // W_zero — these technically terminate; gas cost reflects per-spec value.
        0x00 => Some(0),  // STOP

        // W_verylow (3) — arithmetic, bitwise, stack manipulation, simple memory
        0x01 => Some(3),  // ADD
        0x03 => Some(3),  // SUB
        0x10 => Some(3),  // LT
        0x11 => Some(3),  // GT
        0x12 => Some(3),  // SLT
        0x13 => Some(3),  // SGT
        0x14 => Some(3),  // EQ
        0x15 => Some(3),  // ISZERO
        0x16 => Some(3),  // AND
        0x17 => Some(3),  // OR
        0x18 => Some(3),  // XOR
        0x19 => Some(3),  // NOT
        0x1A => Some(3),  // BYTE
        0x1B => Some(3),  // SHL
        0x1C => Some(3),  // SHR
        0x1D => Some(3),  // SAR
        0x35 => Some(3),  // CALLDATALOAD
        // MLOAD/MSTORE/MSTORE8 have memory-expansion dynamic component;
        // their base is 3 but the total varies. Returned as None below.
        0x50 => Some(2),  // POP (W_base)

        // PUSH0..PUSH32 (0x5F..0x7F)
        0x5F => Some(2),  // PUSH0 (EIP-3855: W_base)
        0x60..=0x7F => Some(3),  // PUSH1..PUSH32 (W_verylow)

        // DUP1..DUP16 (0x80..0x8F), SWAP1..SWAP16 (0x90..0x9F)
        0x80..=0x9F => Some(3),  // DUP/SWAP (W_verylow)

        // W_low (5)
        0x02 => Some(5),  // MUL
        0x04 => Some(5),  // DIV
        0x05 => Some(5),  // SDIV
        0x06 => Some(5),  // MOD
        0x07 => Some(5),  // SMOD
        0x0B => Some(5),  // SIGNEXTEND
        0x47 => Some(5),  // SELFBALANCE (constant — current contract's balance)

        // W_mid (8)
        0x08 => Some(8),  // ADDMOD
        0x09 => Some(8),  // MULMOD
        0x56 => Some(8),  // JUMP

        // W_high (10)
        0x57 => Some(10), // JUMPI

        // W_base (2) — pure context constants
        0x30 => Some(2),  // ADDRESS
        0x32 => Some(2),  // ORIGIN
        0x33 => Some(2),  // CALLER
        0x34 => Some(2),  // CALLVALUE
        0x36 => Some(2),  // CALLDATASIZE
        0x38 => Some(2),  // CODESIZE
        0x3A => Some(2),  // GASPRICE
        0x3D => Some(2),  // RETURNDATASIZE
        0x41 => Some(2),  // COINBASE
        0x42 => Some(2),  // TIMESTAMP
        0x43 => Some(2),  // NUMBER
        0x44 => Some(2),  // PREVRANDAO (post-merge; was DIFFICULTY)
        0x45 => Some(2),  // GASLIMIT
        0x46 => Some(2),  // CHAINID
        0x48 => Some(2),  // BASEFEE
        0x49 => Some(2),  // BLOBHASH (EIP-4844)
        0x4A => Some(2),  // BLOBBASEFEE (EIP-7516)
        0x58 => Some(2),  // PC
        0x59 => Some(2),  // MSIZE
        0x5A => Some(2),  // GAS
        0x5B => Some(1),  // JUMPDEST

        // Dynamic-gas opcodes — None means "compute per-row from operands/state".
        // EXP (0x0A):        10 + 50 × len_bytes(exponent)
        // SHA3 (0x20):       30 + 6 × ceil(size/32) + mem expansion
        // CALLDATACOPY (0x37): 3 + 3 × ceil(size/32) + mem expansion
        // CODECOPY (0x39):   3 + 3 × ceil(size/32) + mem expansion
        // EXTCODESIZE (0x3B): 100 (warm) / 2600 (cold) — EIP-2929
        // EXTCODECOPY (0x3C): 100 / 2600 + 3 × ceil(size/32) + mem
        // RETURNDATACOPY (0x3E): 3 + 3 × ceil(size/32) + mem
        // EXTCODEHASH (0x3F): 100 / 2600
        // BALANCE (0x31):    100 / 2600
        // BLOCKHASH (0x40):  20
        // MLOAD (0x51):      3 + mem
        // MSTORE (0x52):     3 + mem
        // MSTORE8 (0x53):    3 + mem
        // SLOAD (0x54):      100 / 2100 (EIP-2929)
        // SSTORE (0x55):     dynamic (EIP-2200 + EIP-2929 + EIP-3529)
        // MCOPY (0x5E):      3 + 3 × ceil(size/32) + mem (EIP-5656)
        // TLOAD (0x5C):      100 (EIP-1153)
        // TSTORE (0x5D):     100
        // LOG0..LOG4 (0xA0..0xA4): 375 + 8 × size + 375 × topics + mem
        // CREATE (0xF0):     32000 + initcode + mem + ...
        // CALL/CALLCODE/DELEGATECALL/STATICCALL: 100/2600 + value/new-account + mem + child gas
        // CREATE2 (0xF5):    32000 + 6 × ceil(initcode/32) + initcode + ...
        // RETURN (0xF3):     0 + mem
        // REVERT (0xFD):     0 + mem
        // SELFDESTRUCT (0xFF): 5000 + new-account + cold-account
        _ => None,
    }
}

/// True iff `opcode`'s gas cost is fully determined by the opcode byte
/// alone (no operand- or state-dependence). Equivalent to
/// `static_gas_cost(opcode).is_some()`.
pub fn has_static_gas_cost(opcode: u8) -> bool {
    static_gas_cost(opcode).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_arithmetic() {
        assert_eq!(static_gas_cost(0x01), Some(3));  // ADD
        assert_eq!(static_gas_cost(0x02), Some(5));  // MUL
        assert_eq!(static_gas_cost(0x08), Some(8));  // ADDMOD
    }

    #[test]
    fn stack_and_push() {
        assert_eq!(static_gas_cost(0x50), Some(2));  // POP
        assert_eq!(static_gas_cost(0x5F), Some(2));  // PUSH0
        for opcode in 0x60..=0x7Fu8 {
            assert_eq!(static_gas_cost(opcode), Some(3), "PUSHx at {:#x}", opcode);
        }
        for opcode in 0x80..=0x9Fu8 {
            assert_eq!(static_gas_cost(opcode), Some(3), "DUP/SWAP at {:#x}", opcode);
        }
    }

    #[test]
    fn block_context_opcodes() {
        assert_eq!(static_gas_cost(0x41), Some(2));  // COINBASE
        assert_eq!(static_gas_cost(0x42), Some(2));  // TIMESTAMP
        assert_eq!(static_gas_cost(0x43), Some(2));  // NUMBER
        assert_eq!(static_gas_cost(0x46), Some(2));  // CHAINID
        assert_eq!(static_gas_cost(0x48), Some(2));  // BASEFEE
    }

    #[test]
    fn jump_costs() {
        assert_eq!(static_gas_cost(0x56), Some(8));  // JUMP
        assert_eq!(static_gas_cost(0x57), Some(10)); // JUMPI
        assert_eq!(static_gas_cost(0x5B), Some(1));  // JUMPDEST
    }

    #[test]
    fn dynamic_opcodes_return_none() {
        // Hash, memory, storage, log, call family
        assert_eq!(static_gas_cost(0x20), None);  // SHA3
        assert_eq!(static_gas_cost(0x51), None);  // MLOAD (has mem-expansion)
        assert_eq!(static_gas_cost(0x52), None);  // MSTORE
        assert_eq!(static_gas_cost(0x53), None);  // MSTORE8
        assert_eq!(static_gas_cost(0x54), None);  // SLOAD
        assert_eq!(static_gas_cost(0x55), None);  // SSTORE
        for opcode in 0xA0..=0xA4u8 {
            assert_eq!(static_gas_cost(opcode), None, "LOGx at {:#x}", opcode);
        }
        assert_eq!(static_gas_cost(0xF0), None);  // CREATE
        assert_eq!(static_gas_cost(0xF1), None);  // CALL
        assert_eq!(static_gas_cost(0xF5), None);  // CREATE2
        assert_eq!(static_gas_cost(0xFA), None);  // STATICCALL
        assert_eq!(static_gas_cost(0x0A), None);  // EXP
    }

    #[test]
    fn invalid_opcodes_return_none() {
        // 0x21..0x2F (post-SHA3, pre-ADDRESS) — gaps in opcode space
        for opcode in 0x21..=0x2Fu8 {
            assert_eq!(static_gas_cost(opcode), None, "gap at {:#x}", opcode);
        }
    }

    #[test]
    fn has_static_gas_cost_consistent() {
        for opcode in 0u8..=255 {
            assert_eq!(
                has_static_gas_cost(opcode),
                static_gas_cost(opcode).is_some(),
                "consistency check at {:#x}", opcode
            );
        }
    }

    /// Coverage spot-check: ~50% of opcodes should have static gas
    /// costs (rough heuristic — the rest are dynamic or undefined).
    #[test]
    fn static_gas_coverage_is_substantial() {
        let count = (0u8..=255).filter(|&op| has_static_gas_cost(op)).count();
        assert!(count >= 60, "expected ≥ 60 static-gas opcodes, got {}", count);
        assert!(count <= 130, "suspiciously high coverage: {}", count);
    }
}
