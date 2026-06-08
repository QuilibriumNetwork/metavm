//! Bytecode table for cross-AIR LogUp lookups.
//!
//! Builds a table of `(pc, opcode, is_jumpdest)` triples from the
//! deployed bytecode. The EVM main trace's `(pc, opcode)` columns
//! can then be bound to this table via cross-AIR LogUp, ensuring
//! the prover cannot claim a different opcode at a given PC.

use crate::jumpdest::valid_jumpdest_positions;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct BytecodeEntry {
    pub pc: u64,
    pub opcode: u8,
    pub is_jumpdest: bool,
}

pub fn build_bytecode_table(bytecode: &[u8]) -> Vec<BytecodeEntry> {
    let jumpdests: HashSet<usize> = valid_jumpdest_positions(bytecode).into_iter().collect();
    let mut table = Vec::with_capacity(bytecode.len());
    let mut pc = 0usize;
    while pc < bytecode.len() {
        let opcode = bytecode[pc];
        table.push(BytecodeEntry {
            pc: pc as u64,
            opcode,
            is_jumpdest: jumpdests.contains(&pc),
        });
        // Skip PUSH immediates.
        if opcode >= 0x60 && opcode <= 0x7F {
            let push_size = (opcode - 0x60 + 1) as usize;
            for k in 1..=push_size {
                if pc + k < bytecode.len() {
                    table.push(BytecodeEntry {
                        pc: (pc + k) as u64,
                        opcode: bytecode[pc + k],
                        is_jumpdest: false,
                    });
                }
            }
            pc += 1 + push_size;
        } else {
            pc += 1;
        }
    }
    table
}

pub fn make_bytecode_lookup_descriptor(
    evm_layer_index: usize,
    bytecode_table_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_PC, COL_OPCODE};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_bytecode_lookup_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_PC, COL_OPCODE],
        a_selector_column: None, // every real EVM row should look up
        b_layer_index: bytecode_table_layer_index,
        b_columns: vec![0, 1], // pc, opcode in the bytecode table AIR
        b_selector_column: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_bytecode_table() {
        // PUSH1 0x42; ADD; STOP
        let bc = vec![0x60, 0x42, 0x01, 0x00];
        let table = build_bytecode_table(&bc);
        assert_eq!(table.len(), 4);
        assert_eq!(table[0].pc, 0); assert_eq!(table[0].opcode, 0x60);
        assert_eq!(table[1].pc, 1); assert_eq!(table[1].opcode, 0x42); // immediate byte
        assert_eq!(table[2].pc, 2); assert_eq!(table[2].opcode, 0x01); // ADD
        assert_eq!(table[3].pc, 3); assert_eq!(table[3].opcode, 0x00); // STOP
    }

    #[test]
    fn jumpdest_marked() {
        // JUMPDEST; PUSH1 0x00; JUMP
        let bc = vec![0x5B, 0x60, 0x00, 0x56];
        let table = build_bytecode_table(&bc);
        assert!(table[0].is_jumpdest);
        assert!(!table[1].is_jumpdest);
    }

    #[test]
    fn push_immediate_not_jumpdest() {
        // PUSH1 0x5B (looks like JUMPDEST but is immediate); STOP
        let bc = vec![0x60, 0x5B, 0x00];
        let table = build_bytecode_table(&bc);
        assert!(!table[1].is_jumpdest); // 0x5B is immediate, not JUMPDEST
    }

    #[test]
    fn empty_bytecode() {
        let table = build_bytecode_table(&[]);
        assert!(table.is_empty());
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_bytecode_lookup_descriptor(0, 1);
        assert_eq!(d.label, "evm_bytecode_lookup_v1");
        assert_eq!(d.a_columns.len(), 2);
    }

    #[test]
    fn push32_spans_correctly() {
        let mut bc = vec![0x7F]; // PUSH32
        bc.extend_from_slice(&[0xAA; 32]);
        bc.push(0x00); // STOP
        let table = build_bytecode_table(&bc);
        assert_eq!(table.len(), 34); // 1 PUSH32 + 32 immediates + 1 STOP
        assert_eq!(table[0].opcode, 0x7F);
        assert_eq!(table[33].opcode, 0x00);
    }
}
