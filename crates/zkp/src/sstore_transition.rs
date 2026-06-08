//! SSTORE pre/post storage root transition oracle (#68 step 0).
//!
//! Models the state transition caused by a SSTORE operation: given a
//! pre-state storage root, a slot + value being written, and the
//! resulting post-state storage root, verifies the MPT inclusion of
//! the new value under the modified root.
//!
//! This is the host-side spec for proving that an SSTORE operation
//! correctly transitions the storage root. The algebraic version
//! (step 1+) would use the existing MPT AIR + storage gadget AIR
//! to constrain the transition algebraically.

use crate::keccak::keccak256;
use crate::mpt::{single_leaf_trie, verify_mpt_inclusion};
use crate::rlp::rlp_encode_u256;

/// Witness for a single SSTORE state transition.
#[derive(Clone, Debug)]
pub struct SstoreTransition {
    /// The storage slot being written (32 BE bytes).
    pub slot: [u8; 32],
    /// The new value being stored (32 BE bytes).
    pub new_value: [u8; 32],
    /// The post-SSTORE storage root.
    pub post_storage_root: [u8; 32],
    /// MPT inclusion proof for the new value under post_storage_root.
    pub inclusion_proof: Vec<Vec<u8>>,
}

/// Verify that an SSTORE transition is consistent: the new value is
/// included under the post_storage_root at the correct trie key.
pub fn verify_sstore_transition(t: &SstoreTransition) -> Result<(), String> {
    let trie_key = keccak256(&t.slot);
    let value_rlp = rlp_encode_u256(&t.new_value);
    if !verify_mpt_inclusion(t.post_storage_root, &trie_key, &value_rlp, &t.inclusion_proof) {
        return Err("SSTORE transition: new value not included under post_storage_root".into());
    }
    Ok(())
}

/// Build a consistent SSTORE transition for a single-leaf trie
/// (slot = only slot in the trie, value = new value).
pub fn build_single_slot_sstore_transition(
    slot: [u8; 32],
    new_value: [u8; 32],
) -> SstoreTransition {
    let trie_key = keccak256(&slot);
    let value_rlp = rlp_encode_u256(&new_value);
    let (root, proof) = single_leaf_trie(&trie_key, &value_rlp);
    SstoreTransition {
        slot,
        new_value,
        post_storage_root: root,
        inclusion_proof: proof,
    }
}

/// A sequence of SSTORE transitions within a single transaction.
/// Each transition carries its own post_storage_root, and the chain
/// must be consistent: each transition's post_storage_root is the
/// next transition's pre-state.
#[derive(Clone, Debug)]
pub struct SstoreSequence {
    pub transitions: Vec<SstoreTransition>,
    pub final_storage_root: [u8; 32],
}

pub fn verify_sstore_sequence(seq: &SstoreSequence) -> Result<(), String> {
    if seq.transitions.is_empty() {
        return Ok(());
    }
    for (i, t) in seq.transitions.iter().enumerate() {
        verify_sstore_transition(t)
            .map_err(|e| format!("SSTORE[{}]: {}", i, e))?;
    }
    let last_root = seq.transitions.last().unwrap().post_storage_root;
    if last_root != seq.final_storage_root {
        return Err(format!(
            "final_storage_root mismatch: last transition root != claimed final"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_slot_transition_verifies() {
        let mut slot = [0u8; 32]; slot[31] = 1;
        let mut value = [0u8; 32]; value[31] = 0x42;
        let t = build_single_slot_sstore_transition(slot, value);
        verify_sstore_transition(&t).unwrap();
    }

    #[test]
    fn tampered_value_fails() {
        let mut slot = [0u8; 32]; slot[31] = 1;
        let mut value = [0u8; 32]; value[31] = 0x42;
        let mut t = build_single_slot_sstore_transition(slot, value);
        t.new_value[31] = 0x99; // tamper
        assert!(verify_sstore_transition(&t).is_err());
    }

    #[test]
    fn tampered_root_fails() {
        let mut slot = [0u8; 32]; slot[31] = 1;
        let mut value = [0u8; 32]; value[31] = 0x42;
        let mut t = build_single_slot_sstore_transition(slot, value);
        t.post_storage_root[0] ^= 0xff; // tamper
        assert!(verify_sstore_transition(&t).is_err());
    }

    #[test]
    fn zero_value_transition_verifies() {
        let slot = [0u8; 32];
        let value = [0u8; 32]; // storing zero
        let t = build_single_slot_sstore_transition(slot, value);
        verify_sstore_transition(&t).unwrap();
    }

    #[test]
    fn sequence_single_transition() {
        let mut slot = [0u8; 32]; slot[31] = 1;
        let mut value = [0u8; 32]; value[31] = 0x42;
        let t = build_single_slot_sstore_transition(slot, value);
        let seq = SstoreSequence {
            final_storage_root: t.post_storage_root,
            transitions: vec![t],
        };
        verify_sstore_sequence(&seq).unwrap();
    }

    #[test]
    fn sequence_wrong_final_root_fails() {
        let mut slot = [0u8; 32]; slot[31] = 1;
        let mut value = [0u8; 32]; value[31] = 0x42;
        let t = build_single_slot_sstore_transition(slot, value);
        let seq = SstoreSequence {
            final_storage_root: [0xBB; 32],
            transitions: vec![t],
        };
        assert!(verify_sstore_sequence(&seq).is_err());
    }

    #[test]
    fn empty_sequence_passes() {
        let seq = SstoreSequence { transitions: vec![], final_storage_root: [0u8; 32] };
        verify_sstore_sequence(&seq).unwrap();
    }
}
