//! `logs_bloom` SSZ sub-tree HTR witness scaffolding.
//!
//! Captures the 7 `sha256_pair` invocations that
//! `hash_tree_root_bytes_fixed(&logs_bloom, 8)` performs internally for
//! the 256-byte `logs_bloom` field carried inside an
//! `ExecutionPayloadHeader`.
//!
//! # Why this matters
//!
//! [`crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness`]
//! captures the 20 *container-level* pair invocations of the payload's
//! merkleization, but it treats `logs_bloom`'s field root as opaque
//! input (it just calls `hash_tree_root_bytes_fixed` host-side and
//! pastes the result into the container leaves). A malicious prover
//! could supply ANY 32-byte value as the `logs_bloom` field root and
//! the payload-pair AIR would accept it.
//!
//! This module starts to close that gap by exposing the 7 sub-tree
//! `sha256_pair` invocations as witness data. Step 1+ wires them into
//! an algebraic AIR + cross-AIR LogUp linkage to `Sha256Extract` (for
//! the SHA-256 bindings) and into the payload-pair AIR (for the
//! root-equality binding).
//!
//! # Sub-tree shape
//!
//! `logs_bloom: Bytes32×8` (256 bytes packed into 8 chunks).
//! `merkleize_chunks(&[chunk; 8], Some(8))` performs:
//!
//! ```text
//! Layer 0 (8 → 4 nodes): pair (0,1), (2,3), (4,5), (6,7)
//! Layer 1 (4 → 2 nodes): pair (0,1), (2,3)
//! Layer 2 (2 → 1 root):  pair (0,1)
//! ```
//!
//! Total: 4 + 2 + 1 = 7 pair invocations. No zero padding (input is
//! already a power-of-two count of chunks).

use crate::beacon_block_header_air::Sha256PairInvocation;
use crate::sha256::sha256_pair;
use crate::ssz::Chunk;

/// Number of `sha256_pair` invocations needed to merkleize 8 chunks
/// (256-byte logs_bloom into a single root).
pub const NUM_PAIR_INVOCATIONS: usize = 7;

/// Number of chunks for the 256-byte logs_bloom field.
pub const NUM_CHUNKS: usize = 8;

/// Witness for the `logs_bloom` sub-tree HTR computation. Exposes
/// every `sha256_pair` invocation so the future AIR + cross-AIR LogUp
/// wiring can bind each pair to a SHA-256 invocation row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogsBloomHtrWitness {
    /// The 256-byte logs_bloom field.
    pub logs_bloom: [u8; 256],
    /// The 8 chunks packed from logs_bloom (chunk i = bytes 32i..32i+32).
    pub chunks: [Chunk; NUM_CHUNKS],
    /// All 7 `sha256_pair(left, right) -> hash` invocations in
    /// evaluation order (layer 0 first, root last).
    pub invocations: [Sha256PairInvocation; NUM_PAIR_INVOCATIONS],
    /// The computed root (== `hash_tree_root_bytes_fixed(logs_bloom, 8)`).
    pub root: Chunk,
}

impl LogsBloomHtrWitness {
    /// Build a witness from a 256-byte logs_bloom.
    pub fn from_logs_bloom(logs_bloom: [u8; 256]) -> Self {
        let mut chunks = [[0u8; 32]; NUM_CHUNKS];
        for i in 0..NUM_CHUNKS {
            chunks[i].copy_from_slice(&logs_bloom[i * 32..i * 32 + 32]);
        }

        let mut invocations: Vec<Sha256PairInvocation> = Vec::with_capacity(NUM_PAIR_INVOCATIONS);

        // Layer 0: 8 → 4 nodes.
        let mut layer1 = [[0u8; 32]; 4];
        for i in 0..4 {
            let left = chunks[2 * i];
            let right = chunks[2 * i + 1];
            let hash = sha256_pair(&left, &right);
            invocations.push(Sha256PairInvocation { left, right, hash });
            layer1[i] = hash;
        }

        // Layer 1: 4 → 2 nodes.
        let mut layer2 = [[0u8; 32]; 2];
        for i in 0..2 {
            let left = layer1[2 * i];
            let right = layer1[2 * i + 1];
            let hash = sha256_pair(&left, &right);
            invocations.push(Sha256PairInvocation { left, right, hash });
            layer2[i] = hash;
        }

        // Layer 2: root.
        let root = sha256_pair(&layer2[0], &layer2[1]);
        invocations.push(Sha256PairInvocation {
            left: layer2[0],
            right: layer2[1],
            hash: root,
        });

        let invocations_arr: [Sha256PairInvocation; NUM_PAIR_INVOCATIONS] =
            invocations.try_into().unwrap();

        Self { logs_bloom, chunks, invocations: invocations_arr, root }
    }

    pub fn layer0(&self) -> &[Sha256PairInvocation] { &self.invocations[0..4] }
    pub fn layer1(&self) -> &[Sha256PairInvocation] { &self.invocations[4..6] }
    pub fn root_invocation(&self) -> &Sha256PairInvocation { &self.invocations[6] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssz::hash_tree_root_bytes_fixed;

    /// **Critical correctness oracle**: witness root MUST equal the
    /// canonical `hash_tree_root_bytes_fixed(&logs_bloom, 8)`.
    #[test]
    fn witness_root_matches_canonical_htr() {
        let logs_bloom = [0xAB; 256];
        let w = LogsBloomHtrWitness::from_logs_bloom(logs_bloom);
        let canonical = hash_tree_root_bytes_fixed(&logs_bloom, 8);
        assert_eq!(w.root, canonical);
    }

    #[test]
    fn witness_root_matches_zero_bloom() {
        let logs_bloom = [0u8; 256];
        let w = LogsBloomHtrWitness::from_logs_bloom(logs_bloom);
        let canonical = hash_tree_root_bytes_fixed(&logs_bloom, 8);
        assert_eq!(w.root, canonical);
    }

    #[test]
    fn witness_has_exactly_7_invocations() {
        let w = LogsBloomHtrWitness::from_logs_bloom([0xCD; 256]);
        assert_eq!(w.invocations.len(), 7);
        assert_eq!(w.layer0().len(), 4);
        assert_eq!(w.layer1().len(), 2);
    }

    #[test]
    fn layer_chaining_consistent() {
        let w = LogsBloomHtrWitness::from_logs_bloom([0xEF; 256]);
        // Layer 1 inputs = layer 0 outputs.
        for i in 0..2 {
            assert_eq!(w.layer1()[i].left, w.layer0()[2 * i].hash);
            assert_eq!(w.layer1()[i].right, w.layer0()[2 * i + 1].hash);
        }
        // Root inputs = layer 1 outputs.
        assert_eq!(w.root_invocation().left, w.layer1()[0].hash);
        assert_eq!(w.root_invocation().right, w.layer1()[1].hash);
        assert_eq!(w.root_invocation().hash, w.root);
    }

    #[test]
    fn chunks_pack_logs_bloom_bytes_correctly() {
        let mut logs_bloom = [0u8; 256];
        for i in 0..256 {
            logs_bloom[i] = i as u8;
        }
        let w = LogsBloomHtrWitness::from_logs_bloom(logs_bloom);
        for i in 0..NUM_CHUNKS {
            for j in 0..32 {
                assert_eq!(
                    w.chunks[i][j],
                    logs_bloom[i * 32 + j],
                    "chunk[{}][{}]", i, j,
                );
            }
        }
    }

    #[test]
    fn different_bloom_different_root() {
        let mut bloom1 = [0u8; 256];
        bloom1[0] = 0x11;
        let mut bloom2 = [0u8; 256];
        bloom2[0] = 0x22;
        let w1 = LogsBloomHtrWitness::from_logs_bloom(bloom1);
        let w2 = LogsBloomHtrWitness::from_logs_bloom(bloom2);
        assert_ne!(w1.root, w2.root);
    }

    /// Composition pin: payload AIR's field_root for `logs_bloom`
    /// (field index 4) equals our LogsBloomHtrWitness.root for the
    /// same input bytes.
    #[test]
    fn composition_with_execution_payload_air() {
        use crate::execution_payload::ExecutionPayloadHeader;
        use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;

        let logs_bloom = [0xAB; 256];
        let mut payload = ExecutionPayloadHeader::default();
        payload.logs_bloom = logs_bloom;
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload);

        let logs_w = LogsBloomHtrWitness::from_logs_bloom(logs_bloom);
        // Payload field index 4 is logs_bloom.
        assert_eq!(
            payload_w.field_roots[4],
            logs_w.root,
            "payload.field_roots[4] (logs_bloom_root) must equal our sub-tree witness root",
        );
    }

    /// Spec pin: 7 invocations is the contract for binding to
    /// payload-pair AIR step 1+. Compile-time check via the
    /// const-size array initializer in `from_logs_bloom`.
    #[test]
    fn invocation_count_pinned_to_7() {
        assert_eq!(NUM_PAIR_INVOCATIONS, 7);
        assert_eq!(NUM_CHUNKS, 8);
    }
}
