//! EVM storage slot key derivation (Phase A2 / #52 step 0).
//!
//! In the EVM, storage is a `U256 -> U256` mapping per contract account.
//! Internally, the storage trie is indexed by `keccak256(slot_index)`
//! where `slot_index` is the 32-byte big-endian encoding of the U256
//! slot key the contract code references.
//!
//! ```text
//! contract.storage_root --MPT--> keccak256(slot_index) → value (RLP-encoded U256)
//! ```
//!
//! For `SLOAD slot`:
//!   1. Compute `trie_key = keccak256(slot.to_be_bytes::<32>())`.
//!   2. MPT inclusion proof: `trie_key → leaf` against `storage_root`.
//!   3. RLP-decode the leaf to recover `value`.
//!   4. `output0 = value`.
//!
//! Mappings (`mapping(K => V)` declared at slot p): the slot of value
//! `m[k]` is `keccak256(k || p)`. Arrays add similar derivations.
//!
//! This module exposes the host-side helpers; the full algebraic chain
//! requires:
//!   - EVM↔KeccakExtract linkage on `(slot.to_be_bytes::<32>, trie_key)`.
//!   - EVM↔MPT linkage on `(trie_key, storage_root, value)` via MPT inclusion AIR.
//!   - Storage root ↔ stateRoot via account RLP + MPT inclusion at higher level.

use revm::primitives::U256;

/// Encode `slot` as 32 big-endian bytes. Standard EVM convention for
/// SLOAD/SSTORE input.
pub fn slot_to_be_bytes(slot: U256) -> [u8; 32] {
    let mut out = [0u8; 32];
    let limbs = slot.as_limbs();
    // U256 stores limbs in little-endian: limbs[0] = low 64 bits.
    // BE bytes 24..32 are limb[0] in BE order.
    for k in 0..4 {
        let limb = limbs[k];
        let start = (3 - k) * 8;
        out[start..start + 8].copy_from_slice(&limb.to_be_bytes());
    }
    out
}

/// Compute the trie key for a storage slot:
/// `trie_key = keccak256(slot.to_be_bytes::<32>())`.
pub fn storage_slot_trie_key(slot: U256) -> [u8; 32] {
    metavm_zkp::keccak::keccak256(&slot_to_be_bytes(slot))
}

/// Compute the trie key for `mapping(K => V)` access:
/// `trie_key = keccak256(key_be_bytes || slot_p_be_bytes)`.
///
/// For nested mappings `mapping(K1 => mapping(K2 => V))` declared at p:
/// `keccak256(k2 || keccak256(k1 || p))`. Use this helper recursively.
pub fn mapping_slot_trie_key(key: U256, mapping_slot: U256) -> [u8; 32] {
    let mut preimage = [0u8; 64];
    preimage[..32].copy_from_slice(&slot_to_be_bytes(key));
    preimage[32..].copy_from_slice(&slot_to_be_bytes(mapping_slot));
    metavm_zkp::keccak::keccak256(&preimage)
}

/// Compute the trie key for a dynamic-array element at index `i`,
/// array declared at slot `p`:
/// `trie_key = keccak256(p_be_bytes) + i`.
pub fn array_element_trie_key(array_slot: U256, index: U256) -> [u8; 32] {
    let base_hash = metavm_zkp::keccak::keccak256(&slot_to_be_bytes(array_slot));
    let base = U256::from_be_bytes(base_hash);
    let key_u256 = base.wrapping_add(index);
    slot_to_be_bytes(key_u256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_zero_to_bytes() {
        let bytes = slot_to_be_bytes(U256::ZERO);
        assert_eq!(bytes, [0u8; 32]);
    }

    #[test]
    fn slot_one_to_bytes() {
        let bytes = slot_to_be_bytes(U256::from(1u64));
        let mut expected = [0u8; 32];
        expected[31] = 1; // BE encoding, low byte at index 31
        assert_eq!(bytes, expected);
    }

    #[test]
    fn slot_high_value_to_bytes() {
        // U256 with distinctive byte pattern
        let value = U256::from_be_bytes([
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
        ]);
        let bytes = slot_to_be_bytes(value);
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[31], 0x20);
        for i in 0..32 {
            assert_eq!(bytes[i], (i as u8) + 1);
        }
    }

    #[test]
    fn storage_slot_trie_key_matches_known() {
        // Per Solidity convention: storage at slot 0 has trie key
        // keccak256(0x000...000) — well-known value.
        let key = storage_slot_trie_key(U256::ZERO);
        // keccak256 of 32 zero bytes is the known constant
        // 0x290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563
        let expected: [u8; 32] = [
            0x29, 0x0d, 0xec, 0xd9, 0x54, 0x8b, 0x62, 0xa8,
            0xd6, 0x03, 0x45, 0xa9, 0x88, 0x38, 0x6f, 0xc8,
            0x4b, 0xa6, 0xbc, 0x95, 0x48, 0x40, 0x08, 0xf6,
            0x36, 0x2f, 0x93, 0x16, 0x0e, 0xf3, 0xe5, 0x63,
        ];
        assert_eq!(key, expected);
    }

    #[test]
    fn mapping_slot_trie_key_consistent() {
        // For mapping(K=>V) at slot p, value at m[k] lives at
        // keccak256(k || p). Verify the helper produces the
        // canonical concatenation.
        let p = U256::from(5u64);
        let k = U256::from(42u64);
        let key = mapping_slot_trie_key(k, p);
        // Build expected by hand
        let mut preimage = [0u8; 64];
        preimage[24..32].copy_from_slice(&42u64.to_be_bytes());
        preimage[56..64].copy_from_slice(&5u64.to_be_bytes());
        let expected = metavm_zkp::keccak::keccak256(&preimage);
        assert_eq!(key, expected);
    }

    #[test]
    fn array_element_trie_key_offsets_from_base() {
        // For array at slot p, element i lives at keccak256(p) + i.
        let p = U256::from(7u64);
        let i = U256::from(3u64);
        let key = array_element_trie_key(p, i);
        let base = metavm_zkp::keccak::keccak256(&slot_to_be_bytes(p));
        let base_u256 = U256::from_be_bytes(base);
        let expected_u256 = base_u256.wrapping_add(i);
        let expected_bytes = slot_to_be_bytes(expected_u256);
        assert_eq!(key, expected_bytes);
    }

    /// Sanity: different slots yield different trie keys.
    #[test]
    fn distinct_slots_distinct_trie_keys() {
        let k0 = storage_slot_trie_key(U256::from(0u64));
        let k1 = storage_slot_trie_key(U256::from(1u64));
        let k2 = storage_slot_trie_key(U256::from(2u64));
        assert_ne!(k0, k1);
        assert_ne!(k0, k2);
        assert_ne!(k1, k2);
    }
}
