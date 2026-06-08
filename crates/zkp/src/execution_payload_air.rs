//! ExecutionPayloadHeader SSZ extraction (Phase C↔B bridge step 3 step 0).
//!
//! Witness scaffolding for proving the SSZ `hash_tree_root` of an
//! [`ExecutionPayloadHeader`]. Mirrors [`crate::beacon_block_header_air`]
//! but for the 17-field execution payload container.
//!
//! # Container shape (Deneb)
//!
//! 17 fields → merkleize padded to 32 leaves (depth 5) → 20 pair
//! invocations across 5 layers:
//!
//! ```text
//! Layer 0 (17 → 9 nodes): pair (0,1), (2,3), ..., (14,15), (16, ZH(0))
//! Layer 1 (9  → 5 nodes): pair (0,1), (2,3), (4,5), (6,7), (8, ZH(1))
//! Layer 2 (5  → 3 nodes): pair (0,1), (2,3), (4, ZH(2))
//! Layer 3 (3  → 2 nodes): pair (0,1), (2, ZH(3))
//! Layer 4 (2  → 1 root):  pair (0,1)
//! ```
//!
//! Where `ZH(d)` is the depth-`d` zero hash (subtree of all-zero leaves).
//!
//! # Soundness scope (step 0)
//!
//! Host-side data shape + witness builder only. The witness exposes
//! every container-level `sha256_pair` invocation, so a future AIR
//! (step 1) can constrain them via cross-AIR LogUp to `Sha256Extract`.
//!
//! **Not yet captured**: per-field sub-tree invocations for `logs_bloom`
//! (7 invocations to merkleize 256 bytes → 8 chunks → root) and
//! `extra_data` (variable, plus `mix_in_length`). These are deferred to
//! step 0b — the user of this witness must compute the field roots
//! externally (or rely on the host-side oracle for soundness over
//! those fields).

use crate::beacon_block_header_air::Sha256PairInvocation;
use crate::execution_payload::ExecutionPayloadHeader;
use crate::sha256::sha256_pair;
use crate::ssz::{
    hash_tree_root_bytes_fixed, hash_tree_root_list_bytes, hash_tree_root_uint, Chunk, ZERO_CHUNK,
};

/// Total number of container-level pair invocations for the 17-field
/// ExecutionPayloadHeader (padded to 32 leaves).
pub const NUM_CONTAINER_PAIR_INVOCATIONS: usize = 20;

/// Number of fields in the Deneb-shape ExecutionPayloadHeader.
pub const NUM_FIELDS: usize = 17;

/// Witness for the ExecutionPayloadHeader hash_tree_root computation.
/// Exposes every container-level `sha256_pair` invocation that the
/// merkleization performs, so future AIR + cross-AIR LogUp wiring
/// can bind each pair to a SHA-256 invocation row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPayloadHeaderHtrWitness {
    /// The source payload header.
    pub payload: ExecutionPayloadHeader,
    /// The 17 field roots passed to the container merkleizer.
    pub field_roots: [Chunk; NUM_FIELDS],
    /// All 20 container-level `sha256_pair(left, right) -> hash`
    /// invocations in evaluation order (layer 0 first, then layer 1,
    /// ..., root last).
    pub invocations: [Sha256PairInvocation; NUM_CONTAINER_PAIR_INVOCATIONS],
    /// Computed payload root (== `payload.hash_tree_root()`).
    pub root: Chunk,
}

/// Depth-`d` zero subtree root: `sha256_pair` chained d times starting
/// from `ZERO_CHUNK`. Local replica of `ssz::zero_hash` (which is
/// private).
fn zero_hash(depth: u32) -> Chunk {
    let mut z = ZERO_CHUNK;
    for _ in 0..depth {
        z = sha256_pair(&z, &z);
    }
    z
}

impl ExecutionPayloadHeaderHtrWitness {
    /// Build a witness from an `ExecutionPayloadHeader`. Computes all
    /// 20 container-level `sha256_pair` invocations and the final root.
    pub fn from_payload(payload: ExecutionPayloadHeader) -> Self {
        let field_roots = Self::compute_field_roots(&payload);
        let (invocations, root) = Self::merkleize_container(&field_roots);
        Self {
            payload,
            field_roots,
            invocations,
            root,
        }
    }

    /// Compute the 17 field roots per the SSZ spec.
    fn compute_field_roots(p: &ExecutionPayloadHeader) -> [Chunk; NUM_FIELDS] {
        [
            p.parent_hash,
            hash_tree_root_bytes_fixed(&p.fee_recipient, 1),
            p.state_root,
            p.receipts_root,
            hash_tree_root_bytes_fixed(&p.logs_bloom, 8),
            p.prev_randao,
            hash_tree_root_uint(p.block_number),
            hash_tree_root_uint(p.gas_limit),
            hash_tree_root_uint(p.gas_used),
            hash_tree_root_uint(p.timestamp),
            hash_tree_root_list_bytes(&p.extra_data, crate::execution_payload::MAX_EXTRA_DATA_BYTES),
            p.base_fee_per_gas,
            p.block_hash,
            p.transactions_root,
            p.withdrawals_root,
            hash_tree_root_uint(p.blob_gas_used),
            hash_tree_root_uint(p.excess_blob_gas),
        ]
    }

    /// Run the 5-layer container merkleization, capturing every pair
    /// invocation. Returns `(invocations, root)`.
    fn merkleize_container(
        field_roots: &[Chunk; NUM_FIELDS],
    ) -> ([Sha256PairInvocation; NUM_CONTAINER_PAIR_INVOCATIONS], Chunk) {
        let mut invocations: Vec<Sha256PairInvocation> =
            Vec::with_capacity(NUM_CONTAINER_PAIR_INVOCATIONS);

        // Layer 0: 17 → 9 nodes. Pairs (0,1)..(14,15) + (16, ZH(0)).
        let mut layer: Vec<Chunk> = field_roots.to_vec();
        for depth in 0..5 {
            let zh = zero_hash(depth);
            let mut next: Vec<Chunk> = Vec::with_capacity((layer.len() + 1) / 2);
            let mut i = 0;
            while i < layer.len() {
                let left = layer[i];
                let right = if i + 1 < layer.len() { layer[i + 1] } else { zh };
                let hash = sha256_pair(&left, &right);
                invocations.push(Sha256PairInvocation { left, right, hash });
                next.push(hash);
                i += 2;
            }
            layer = next;
        }

        assert_eq!(layer.len(), 1, "merkleization must end with a single root");
        assert_eq!(
            invocations.len(),
            NUM_CONTAINER_PAIR_INVOCATIONS,
            "container should produce exactly {} pair invocations",
            NUM_CONTAINER_PAIR_INVOCATIONS,
        );
        let invocations_arr: [Sha256PairInvocation; NUM_CONTAINER_PAIR_INVOCATIONS] =
            invocations.try_into().unwrap();
        (invocations_arr, layer[0])
    }

    /// Pair invocations at layer `l` (0..=4).
    pub fn layer(&self, l: usize) -> &[Sha256PairInvocation] {
        let (start, end) = match l {
            0 => (0, 9),
            1 => (9, 14),
            2 => (14, 17),
            3 => (17, 19),
            4 => (19, 20),
            _ => panic!("layer index out of range (must be 0..=4)"),
        };
        &self.invocations[start..end]
    }

    /// The root invocation (layer 4).
    pub fn root_invocation(&self) -> &Sha256PairInvocation {
        &self.invocations[NUM_CONTAINER_PAIR_INVOCATIONS - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_payload::ExecutionPayloadHeader;

    fn sample_payload() -> ExecutionPayloadHeader {
        let mut p = ExecutionPayloadHeader::default();
        p.parent_hash = [0x11u8; 32];
        p.state_root = [0x22u8; 32];
        p.receipts_root = [0x33u8; 32];
        p.prev_randao = [0x44u8; 32];
        p.block_number = 12345;
        p.gas_limit = 30_000_000;
        p.timestamp = 1_700_000_000;
        p.block_hash = [0xabu8; 32];
        p.transactions_root = [0x55u8; 32];
        p.withdrawals_root = [0x66u8; 32];
        p
    }

    /// **Critical correctness oracle**: our witness builder's `root`
    /// MUST equal `ExecutionPayloadHeader::hash_tree_root()`. If this
    /// fires, the merkleization layout is wrong.
    #[test]
    fn witness_root_matches_canonical_hash_tree_root() {
        let p = sample_payload();
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(p.clone());
        assert_eq!(w.root, p.hash_tree_root());
    }

    #[test]
    fn witness_has_exactly_20_invocations() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        assert_eq!(w.invocations.len(), 20);
        assert_eq!(w.layer(0).len(), 9);
        assert_eq!(w.layer(1).len(), 5);
        assert_eq!(w.layer(2).len(), 3);
        assert_eq!(w.layer(3).len(), 2);
        assert_eq!(w.layer(4).len(), 1);
    }

    #[test]
    fn layer_chaining_is_consistent() {
        // Each layer-N pair's inputs must come from layer-(N-1) outputs
        // (with ZH(N-1) on the right of the last pair when the previous
        // layer had odd count).
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        // The zero-hash used as odd-tail right at layer N is ZH(N) —
        // the merkleizer's `zero_hash(depth)` is called with depth=N
        // when producing layer N's pairs.
        for (next_layer, expected_zh) in [(1usize, 1u32), (2, 2), (3, 3), (4, 4)] {
            let prev = w.layer(next_layer - 1);
            let curr = w.layer(next_layer);
            for (i, inv) in curr.iter().enumerate() {
                assert_eq!(inv.left, prev[2 * i].hash, "layer{} pair{} left", next_layer, i);
                let right = if 2 * i + 1 < prev.len() {
                    prev[2 * i + 1].hash
                } else {
                    zero_hash(expected_zh)
                };
                assert_eq!(inv.right, right, "layer{} pair{} right", next_layer, i);
            }
        }
    }

    #[test]
    fn zero_padding_at_correct_positions() {
        // Each layer N's last pair uses ZH(N) as its right when the
        // input layer had an odd-count tail. Layer 0 (17 inputs, odd
        // tail at 16) → ZH(0). Layer 1 (9 inputs from layer 0, odd
        // tail at 8) → ZH(1). And so on through layer 3.
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        assert_eq!(w.layer(0)[8].right, zero_hash(0), "layer 0 last pair tail = ZH(0)");
        assert_eq!(w.layer(1)[4].right, zero_hash(1), "layer 1 last pair tail = ZH(1)");
        assert_eq!(w.layer(2)[2].right, zero_hash(2), "layer 2 last pair tail = ZH(2)");
        assert_eq!(w.layer(3)[1].right, zero_hash(3), "layer 3 last pair tail = ZH(3)");
    }

    #[test]
    fn different_payloads_produce_different_roots() {
        let p1 = sample_payload();
        let mut p2 = p1.clone();
        p2.block_hash[0] ^= 0xff;
        let w1 = ExecutionPayloadHeaderHtrWitness::from_payload(p1);
        let w2 = ExecutionPayloadHeaderHtrWitness::from_payload(p2);
        assert_ne!(w1.root, w2.root);
        assert_ne!(w1.root_invocation().hash, w2.root_invocation().hash);
    }

    #[test]
    fn root_invocation_output_equals_witness_root() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        assert_eq!(w.root_invocation().hash, w.root);
    }

    #[test]
    fn field_roots_match_canonical_per_field() {
        let p = sample_payload();
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(p.clone());
        assert_eq!(w.field_roots[0], p.parent_hash);
        assert_eq!(w.field_roots[2], p.state_root);
        assert_eq!(w.field_roots[3], p.receipts_root);
        assert_eq!(w.field_roots[5], p.prev_randao);
        assert_eq!(w.field_roots[6], hash_tree_root_uint(p.block_number));
        assert_eq!(w.field_roots[11], p.base_fee_per_gas);
        assert_eq!(w.field_roots[12], p.block_hash);
        assert_eq!(w.field_roots[13], p.transactions_root);
        assert_eq!(w.field_roots[14], p.withdrawals_root);
    }

    /// Empty payload should still produce a determinate witness with
    /// 20 invocations. Many field roots will be ZERO_CHUNK; the merkle
    /// tree still has 20 pair hashes (none of them trivially short-
    /// circuited).
    #[test]
    fn default_payload_witness_well_formed() {
        let p = ExecutionPayloadHeader::default();
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(p.clone());
        assert_eq!(w.invocations.len(), 20);
        assert_eq!(w.root, p.hash_tree_root());
    }
}
