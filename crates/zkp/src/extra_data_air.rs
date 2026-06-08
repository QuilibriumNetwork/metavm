//! `extra_data` SSZ sub-tree HTR witness scaffolding.
//!
//! Captures the single `sha256_pair` invocation (the mix-in-length
//! step) that `hash_tree_root_list_bytes(&extra_data, 32)` performs
//! for the variable-length `extra_data: ByteList[32]` field inside an
//! `ExecutionPayloadHeader`.
//!
//! # Why this matters
//!
//! Same gap as [`crate::logs_bloom_air`] — the payload-pair AIR
//! currently trusts `extra_data`'s field root opaquely. This witness
//! exposes the one `sha256_pair` invocation that backs it.
//!
//! # Sub-tree shape
//!
//! `extra_data: List[byte, 32]` — variable length up to 32 bytes.
//! `hash_tree_root_list_bytes(&extra_data, 32)` performs:
//!
//! ```text
//! 1. pack_bytes(extra_data) → 0 or 1 chunk
//! 2. merkleize_chunks(&chunks, Some(1)) → packed_root
//!    - 0 chunks: packed_root = ZERO_CHUNK (no hash invocation)
//!    - 1 chunk:  packed_root = chunk[0]   (no hash invocation — single
//!                                          leaf is the root)
//! 3. mix_in_length(packed_root, len_bytes)
//!    = sha256(packed_root || len_bytes_le_32)        ← 1 pair invocation
//! ```
//!
//! Total: **1** `sha256_pair` invocation. The packed root is computed
//! trivially (no hashing) so only the mix_in_length step is captured.

use crate::beacon_block_header_air::Sha256PairInvocation;
use crate::sha256::sha256_pair;
use crate::ssz::{Chunk, ZERO_CHUNK};

/// Number of `sha256_pair` invocations for `extra_data` HTR
/// (mix_in_length only; the inner merkleization of ≤1 chunk doesn't
/// invoke `sha256_pair`).
pub const NUM_PAIR_INVOCATIONS: usize = 1;

/// Maximum byte length of extra_data per Deneb spec.
pub const MAX_EXTRA_DATA_BYTES: u64 = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtraDataHtrWitness {
    /// The actual extra_data bytes (0..=32).
    pub extra_data: Vec<u8>,
    /// Packed root: ZERO_CHUNK if empty, or the single chunk padded
    /// with trailing zeros.
    pub packed_root: Chunk,
    /// The mix_in_length argument: `len(extra_data)` encoded as 32
    /// LE bytes.
    pub length_chunk: Chunk,
    /// The single `sha256_pair(packed_root, length_chunk)` invocation
    /// that mix_in_length performs.
    pub invocation: Sha256PairInvocation,
    /// Computed root (== `hash_tree_root_list_bytes(extra_data, 32)`).
    pub root: Chunk,
}

impl ExtraDataHtrWitness {
    /// Build a witness from extra_data bytes. Panics if `extra_data.len() > 32`.
    pub fn from_extra_data(extra_data: Vec<u8>) -> Self {
        assert!(
            extra_data.len() as u64 <= MAX_EXTRA_DATA_BYTES,
            "extra_data length {} exceeds spec max {}",
            extra_data.len(), MAX_EXTRA_DATA_BYTES,
        );

        // Step 1+2: pack into ≤1 chunk and merkleize. Since max is 32
        // bytes (1 chunk), the packed_root is just the padded chunk or
        // ZERO_CHUNK if empty.
        let packed_root: Chunk = if extra_data.is_empty() {
            ZERO_CHUNK
        } else {
            let mut chunk = [0u8; 32];
            chunk[..extra_data.len()].copy_from_slice(&extra_data);
            chunk
        };

        // Step 3: mix_in_length. Encode `len(extra_data)` as 32-byte LE.
        let mut length_chunk = [0u8; 32];
        let len_le = (extra_data.len() as u64).to_le_bytes();
        length_chunk[..8].copy_from_slice(&len_le);

        let root = sha256_pair(&packed_root, &length_chunk);

        Self {
            extra_data,
            packed_root,
            length_chunk,
            invocation: Sha256PairInvocation {
                left: packed_root,
                right: length_chunk,
                hash: root,
            },
            root,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssz::hash_tree_root_list_bytes;

    /// **Critical correctness oracle**: witness root MUST equal the
    /// canonical `hash_tree_root_list_bytes(&extra_data, 32)`.
    #[test]
    fn witness_root_matches_canonical_empty() {
        let w = ExtraDataHtrWitness::from_extra_data(vec![]);
        let canonical = hash_tree_root_list_bytes(&[], 32);
        assert_eq!(w.root, canonical);
    }

    #[test]
    fn witness_root_matches_canonical_one_byte() {
        let w = ExtraDataHtrWitness::from_extra_data(vec![0xAB]);
        let canonical = hash_tree_root_list_bytes(&[0xAB], 32);
        assert_eq!(w.root, canonical);
    }

    #[test]
    fn witness_root_matches_canonical_full_32_bytes() {
        let bytes: Vec<u8> = (0..32).collect();
        let w = ExtraDataHtrWitness::from_extra_data(bytes.clone());
        let canonical = hash_tree_root_list_bytes(&bytes, 32);
        assert_eq!(w.root, canonical);
    }

    #[test]
    fn witness_root_matches_canonical_mid_length() {
        let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let w = ExtraDataHtrWitness::from_extra_data(bytes.clone());
        let canonical = hash_tree_root_list_bytes(&bytes, 32);
        assert_eq!(w.root, canonical);
    }

    #[test]
    fn invocation_output_equals_witness_root() {
        let w = ExtraDataHtrWitness::from_extra_data(vec![0x42; 16]);
        assert_eq!(w.invocation.hash, w.root);
    }

    #[test]
    fn length_chunk_is_le_encoded_length() {
        let w = ExtraDataHtrWitness::from_extra_data(vec![0x99; 7]);
        assert_eq!(w.length_chunk[0], 7);
        for k in 1..32 {
            assert_eq!(w.length_chunk[k], 0, "length_chunk[{}] should be 0", k);
        }
    }

    #[test]
    fn packed_root_pads_with_zeros() {
        let w = ExtraDataHtrWitness::from_extra_data(vec![0x11, 0x22, 0x33]);
        assert_eq!(w.packed_root[0], 0x11);
        assert_eq!(w.packed_root[1], 0x22);
        assert_eq!(w.packed_root[2], 0x33);
        for k in 3..32 {
            assert_eq!(w.packed_root[k], 0, "packed_root[{}] should be 0", k);
        }
    }

    #[test]
    fn empty_packed_root_is_zero_chunk() {
        let w = ExtraDataHtrWitness::from_extra_data(vec![]);
        assert_eq!(w.packed_root, ZERO_CHUNK);
    }

    #[test]
    fn different_lengths_different_roots() {
        let w1 = ExtraDataHtrWitness::from_extra_data(vec![0xAB]);
        let w2 = ExtraDataHtrWitness::from_extra_data(vec![0xAB, 0x00]);
        // Mix_in_length includes the length, so [0xAB] and [0xAB, 0x00]
        // produce different roots even though the packed payload is
        // the same (a malicious prover can't fake length).
        assert_ne!(w1.root, w2.root, "mix_in_length must distinguish");
    }

    /// Composition pin: payload AIR's field_root for `extra_data`
    /// (field index 10) equals our witness root.
    #[test]
    fn composition_with_execution_payload_air() {
        use crate::execution_payload::ExecutionPayloadHeader;
        use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;

        let extra = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let mut payload = ExecutionPayloadHeader::default();
        payload.extra_data = extra.clone();
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload);

        let extra_w = ExtraDataHtrWitness::from_extra_data(extra);
        // Payload field index 10 is extra_data.
        assert_eq!(
            payload_w.field_roots[10],
            extra_w.root,
            "payload.field_roots[10] (extra_data root) must equal our sub-tree witness root",
        );
    }

    #[test]
    #[should_panic(expected = "exceeds spec max")]
    fn rejects_over_max_length() {
        let too_long = vec![0u8; 33];
        let _ = ExtraDataHtrWitness::from_extra_data(too_long);
    }
}
