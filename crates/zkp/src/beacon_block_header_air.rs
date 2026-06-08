//! Beacon block header SSZ extraction (Phase C3 step 0).
//!
//! Witness scaffolding for proving the SSZ hash_tree_root computation
//! over a 5-field `BeaconBlockHeader`:
//!
//! ```text
//! BeaconBlockHeader {
//!     slot:           uint64,
//!     proposer_index: uint64,
//!     parent_root:    Bytes32,
//!     state_root:     Bytes32,
//!     body_root:      Bytes32,
//! }
//! ```
//!
//! Container HTR pads to the next power of two (8 leaves: 5 real + 3
//! `ZERO_CHUNK`), then merkleizes bottom-up via `sha256_pair`. Total: 7
//! `sha256_pair` invocations across 3 layers:
//!
//! ```text
//! Layer 0 (leaves, depth 3):
//!   L[0] = htr(slot)             L[1] = htr(proposer_index)
//!   L[2] = parent_root           L[3] = state_root
//!   L[4] = body_root             L[5] = ZERO_CHUNK
//!   L[6] = ZERO_CHUNK            L[7] = ZERO_CHUNK
//!
//! Layer 1 (4 nodes):
//!   N[0] = sha256(L[0], L[1])    N[1] = sha256(L[2], L[3])
//!   N[2] = sha256(L[4], L[5])    N[3] = sha256(L[6], L[7])
//!
//! Layer 2 (2 nodes):
//!   M[0] = sha256(N[0], N[1])    M[1] = sha256(N[2], N[3])
//!
//! Layer 3 (root):
//!   root = sha256(M[0], M[1])
//! ```
//!
//! **Soundness state at step 0**: host-side data shape + witness builder
//! only. No AIR constraints; the 7 `sha256_pair` invocations are
//! computed host-side and stored as oracle data. Step 1+ will wire each
//! pair to a `Sha256Extract` row via cross-AIR LogUp, closing the
//! algebraic chain.
//!
//! **Use cases** (motivating Phase C3+):
//! - Bind a beacon block's `execution_payload.block_hash` to Layer B's
//!   block header (gateway from EVM execution to consensus).
//! - Bind `state_root` to the validator registry root (used by Finality
//!   AIR for stake-weighted finalization).
//! - Bind `slot` and `parent_root` to the FFG checkpoint chain.

use crate::beacon::BeaconBlockHeader;
use crate::sha256::sha256_pair;
use crate::ssz::{hash_tree_root_uint, ZERO_CHUNK};

/// One `sha256(left, right) = hash` invocation. Mirrors a row in
/// [`crate::sha256_extract::KeccakExtractWitness`]'s SHA-256 analog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sha256PairInvocation {
    pub left: [u8; 32],
    pub right: [u8; 32],
    pub hash: [u8; 32],
}

/// Witness for the BeaconBlockHeader hash_tree_root computation.
/// Exposes every intermediate `sha256_pair` invocation that the
/// merkleization performs, so future AIR + cross-AIR LogUp wiring
/// can bind each pair to a SHA-256 invocation row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeaconBlockHeaderHtrWitness {
    pub header: BeaconBlockHeader,
    /// 7 `sha256_pair` invocations in evaluation order:
    /// `[L1_0, L1_1, L1_2, L1_3, L2_0, L2_1, root]`.
    pub invocations: [Sha256PairInvocation; 7],
    /// The computed root (== `BeaconBlockHeader::hash_tree_root()`).
    pub root: [u8; 32],
}

impl BeaconBlockHeaderHtrWitness {
    /// Build a witness from a `BeaconBlockHeader`. Computes all 7
    /// `sha256_pair` invocations and the final root.
    pub fn from_header(header: BeaconBlockHeader) -> Self {
        let leaves: [[u8; 32]; 8] = [
            hash_tree_root_uint(header.slot),
            hash_tree_root_uint(header.proposer_index),
            header.parent_root,
            header.state_root,
            header.body_root,
            ZERO_CHUNK,
            ZERO_CHUNK,
            ZERO_CHUNK,
        ];

        let mut invocations: Vec<Sha256PairInvocation> = Vec::with_capacity(7);

        // Layer 1: 4 pairs over 8 leaves.
        let mut layer1 = [[0u8; 32]; 4];
        for i in 0..4 {
            let left = leaves[2 * i];
            let right = leaves[2 * i + 1];
            let hash = sha256_pair(&left, &right);
            invocations.push(Sha256PairInvocation { left, right, hash });
            layer1[i] = hash;
        }

        // Layer 2: 2 pairs over 4 nodes.
        let mut layer2 = [[0u8; 32]; 2];
        for i in 0..2 {
            let left = layer1[2 * i];
            let right = layer1[2 * i + 1];
            let hash = sha256_pair(&left, &right);
            invocations.push(Sha256PairInvocation { left, right, hash });
            layer2[i] = hash;
        }

        // Layer 3: root.
        let root = sha256_pair(&layer2[0], &layer2[1]);
        invocations.push(Sha256PairInvocation {
            left: layer2[0],
            right: layer2[1],
            hash: root,
        });

        Self {
            header,
            invocations: invocations.try_into().unwrap(),
            root,
        }
    }

    /// Layer-1 hash invocations (indices 0..4).
    pub fn layer1(&self) -> &[Sha256PairInvocation] { &self.invocations[0..4] }
    /// Layer-2 hash invocations (indices 4..6).
    pub fn layer2(&self) -> &[Sha256PairInvocation] { &self.invocations[4..6] }
    /// The root invocation (index 6).
    pub fn root_invocation(&self) -> &Sha256PairInvocation { &self.invocations[6] }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> BeaconBlockHeader {
        BeaconBlockHeader {
            slot: 12345,
            proposer_index: 67,
            parent_root: [0x11u8; 32],
            state_root: [0x22u8; 32],
            body_root: [0x33u8; 32],
        }
    }

    /// **Critical correctness oracle**: our witness builder's `root`
    /// MUST equal `BeaconBlockHeader::hash_tree_root()`. If this fires,
    /// the layer-pairing structure is wrong.
    #[test]
    fn witness_root_matches_canonical_hash_tree_root() {
        let h = sample_header();
        let w = BeaconBlockHeaderHtrWitness::from_header(h);
        let canonical = h.hash_tree_root();
        assert_eq!(w.root, canonical);
    }

    #[test]
    fn witness_has_seven_pair_invocations() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        assert_eq!(w.invocations.len(), 7);
        assert_eq!(w.layer1().len(), 4);
        assert_eq!(w.layer2().len(), 2);
    }

    #[test]
    fn layer_chaining_is_consistent() {
        // Each layer-2 input must equal the corresponding layer-1
        // output. Each layer-2 output feeds the root.
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        for i in 0..2 {
            assert_eq!(w.layer2()[i].left, w.layer1()[2 * i].hash);
            assert_eq!(w.layer2()[i].right, w.layer1()[2 * i + 1].hash);
        }
        assert_eq!(w.root_invocation().left, w.layer2()[0].hash);
        assert_eq!(w.root_invocation().right, w.layer2()[1].hash);
        assert_eq!(w.root_invocation().hash, w.root);
    }

    /// The 3 trailing zero-leaves (indices 5..8) imply layer-1 pair 2
    /// has `right = ZERO_CHUNK` and pair 3 has both sides `ZERO_CHUNK`.
    /// This pin protects against silent layer-padding changes.
    #[test]
    fn zero_padding_at_correct_positions() {
        let h = sample_header();
        let w = BeaconBlockHeaderHtrWitness::from_header(h);
        // Pair 2: left = body_root, right = ZERO_CHUNK
        assert_eq!(w.layer1()[2].left, h.body_root);
        assert_eq!(w.layer1()[2].right, ZERO_CHUNK);
        // Pair 3: both ZERO_CHUNK
        assert_eq!(w.layer1()[3].left, ZERO_CHUNK);
        assert_eq!(w.layer1()[3].right, ZERO_CHUNK);
    }

    /// Different headers produce different roots — sanity check the
    /// witness isn't accidentally constant.
    #[test]
    fn different_headers_different_roots() {
        let h1 = sample_header();
        let mut h2 = h1;
        h2.slot = 99999;
        let w1 = BeaconBlockHeaderHtrWitness::from_header(h1);
        let w2 = BeaconBlockHeaderHtrWitness::from_header(h2);
        assert_ne!(w1.root, w2.root);
    }
}
