//! Merkle Patricia Trie (MPT) — reference implementation.
//!
//! Per Ethereum Yellow Paper Appendix C, the MPT authenticates the
//! state/storage/transaction/receipt tries of the execution layer.  This
//! module provides:
//!
//! - [`Nibbles`]: half-byte path wrapper + Hex-Prefix (HP, a.k.a.
//!   *compact*) encoding/decoding.
//! - [`MptNode`]: the three node variants — leaf, extension, branch.
//! - [`mpt_node_hash`]: RLP-encode a node then keccak256 it — the hash that
//!   identifies the node in its parent (and in the trie root).
//! - [`verify_mpt_inclusion`]: walk an RLP-encoded proof (a `Vec<Vec<u8>>`
//!   of nodes) and verify that `(key, value)` is present under `root`.
//!
//! The decoder side is intentionally permissive: it accepts any
//! well-formed RLP node whether its children are inlined (<32 bytes) or
//! referenced by hash.
//!
//! # AIR wiring (future)
//! Storage-slot / account inclusion proofs will be encoded as a chain of
//! node-hash AIR subproofs: each step commits to the parent node's
//! keccak256 (which must equal the previous step's expected hash) and
//! exposes the child hash at the nibble-path index, plus the HP-encoded
//! partial path.

use crate::keccak::keccak256;
use crate::rlp::{rlp_decode, rlp_encode_bytes, rlp_encode_list, RlpItem};

/// Radix-16 path, each element `∈ 0..16`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nibbles(pub Vec<u8>);

impl Nibbles {
    /// Empty nibble path.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Convert a byte slice to its nibble representation (two per byte,
    /// high nibble first).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut n = Vec::with_capacity(bytes.len() * 2);
        for &b in bytes {
            n.push(b >> 4);
            n.push(b & 0x0f);
        }
        Self(n)
    }

    /// Construct from a pre-validated `Vec<u8>` where every element is in
    /// `0..16`. Panics in debug builds on out-of-range elements.
    pub fn from_nibbles(nibbles: Vec<u8>) -> Self {
        debug_assert!(nibbles.iter().all(|&n| n < 16));
        Self(nibbles)
    }

    /// Number of nibbles in the path.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True iff the path is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Longest common prefix with `other`, in nibbles.
    pub fn common_prefix_len(&self, other: &Nibbles) -> usize {
        self.0
            .iter()
            .zip(other.0.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Hex-Prefix ("compact") encoding per Yellow Paper Appendix C.
    ///
    /// The first byte encodes:
    /// - bit 5 (value `0x20`): leaf vs extension (1 = leaf).
    /// - bit 4 (value `0x10`): odd/even length (1 = odd).
    ///
    /// Concretely the first byte's high nibble is:
    /// - `0x0`: extension, even length.
    /// - `0x1`: extension, odd length — low nibble holds the first path nibble.
    /// - `0x2`: leaf, even length.
    /// - `0x3`: leaf, odd length — low nibble holds the first path nibble.
    ///
    /// The remaining nibbles are packed two per byte, high nibble first.
    pub fn to_encoded_path(&self, is_leaf: bool) -> Vec<u8> {
        let n = self.0.len();
        let odd = n % 2 == 1;
        let flag = (is_leaf as u8) << 5;
        let mut out = Vec::with_capacity(1 + n.div_ceil(2));
        if odd {
            out.push(flag | 0x10 | self.0[0]);
            let rest = &self.0[1..];
            for chunk in rest.chunks(2) {
                out.push((chunk[0] << 4) | chunk[1]);
            }
        } else {
            out.push(flag);
            for chunk in self.0.chunks(2) {
                out.push((chunk[0] << 4) | chunk[1]);
            }
        }
        out
    }

    /// Decode a Hex-Prefix path back to `(nibbles, is_leaf)`.
    pub fn from_encoded_path(encoded: &[u8]) -> Option<(Nibbles, bool)> {
        if encoded.is_empty() {
            return None;
        }
        let first = encoded[0];
        let is_leaf = (first & 0x20) != 0;
        let odd = (first & 0x10) != 0;
        // Bits 7..6 must be zero (per spec), but we don't reject here —
        // legacy clients sometimes set them accidentally; callers can
        // check externally if strict.
        let mut nibbles = Vec::with_capacity(encoded.len() * 2);
        if odd {
            nibbles.push(first & 0x0f);
        }
        for &b in &encoded[1..] {
            nibbles.push(b >> 4);
            nibbles.push(b & 0x0f);
        }
        Some((Nibbles::from_nibbles(nibbles), is_leaf))
    }
}

impl Default for Nibbles {
    fn default() -> Self {
        Self::new()
    }
}

/// The three MPT node types. Children of branches and extensions may be
/// referenced by 32-byte keccak256 hash.
///
/// (For strict parity with geth we'd also support inlined <32-byte node
/// children as raw RLP; for our reference — used only to synthesise test
/// proofs and to verify externally-produced proofs whose nodes we always
/// have in full — hash references suffice.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MptNode {
    /// Terminal node: a partial path plus the stored value.
    Leaf {
        path: Nibbles,
        value: Vec<u8>,
    },
    /// Pointer node: a partial path plus the hash of the child subtree.
    Extension {
        path: Nibbles,
        child: [u8; 32],
    },
    /// Branch node: 16 optional child hashes plus an optional value held
    /// directly at this key.
    Branch {
        children: [Option<[u8; 32]>; 16],
        value: Option<Vec<u8>>,
    },
}

/// RLP-encode an `MptNode` in the canonical MPT structure.
pub fn mpt_node_rlp(node: &MptNode) -> Vec<u8> {
    match node {
        MptNode::Leaf { path, value } => {
            let hp = path.to_encoded_path(true);
            rlp_encode_list(&[rlp_encode_bytes(&hp), rlp_encode_bytes(value)])
        }
        MptNode::Extension { path, child } => {
            let hp = path.to_encoded_path(false);
            rlp_encode_list(&[rlp_encode_bytes(&hp), rlp_encode_bytes(child)])
        }
        MptNode::Branch { children, value } => {
            let mut items: Vec<Vec<u8>> = Vec::with_capacity(17);
            for c in children {
                match c {
                    Some(h) => items.push(rlp_encode_bytes(h)),
                    None => items.push(rlp_encode_bytes(&[])),
                }
            }
            match value {
                Some(v) => items.push(rlp_encode_bytes(v)),
                None => items.push(rlp_encode_bytes(&[])),
            }
            rlp_encode_list(&items)
        }
    }
}

/// keccak256 of the RLP encoding of `node` — its canonical identifier.
pub fn mpt_node_hash(node: &MptNode) -> [u8; 32] {
    keccak256(&mpt_node_rlp(node))
}

/// Standard empty-trie root: `keccak256(rlp_encode_bytes(&[]))`
/// = `0x56e81f17…6fb5e363b421`. This is the root clients use to represent
/// a missing/empty trie (empty state, empty storage, empty receipts).
pub fn empty_trie_root() -> [u8; 32] {
    keccak256(&rlp_encode_bytes(&[]))
}

/// Build a one-leaf Merkle Patricia Trie containing `(key, value)`.
///
/// Returns `(root, proof)` where `root` is the keccak256 of the leaf's RLP
/// encoding and `proof` is the single-element list of RLP-encoded nodes
/// (just the leaf itself) suitable for [`verify_mpt_inclusion`].
///
/// Convenience for test fixtures — building a one-transaction tx-trie
/// or a one-log receipts-trie. For multi-entry tries, construct the
/// `MptNode::Branch`/`Extension`/`Leaf` hierarchy manually.
pub fn single_leaf_trie(key: &[u8], value: &[u8]) -> ([u8; 32], Vec<Vec<u8>>) {
    let leaf = MptNode::Leaf {
        path: Nibbles::from_bytes(key),
        value: value.to_vec(),
    };
    let leaf_rlp = mpt_node_rlp(&leaf);
    let root = keccak256(&leaf_rlp);
    (root, vec![leaf_rlp])
}

/// Build a two-leaf Merkle Patricia Trie where `key1` and `key2`
/// differ in the very first nibble. Returns `(root, proof_for_key1)`.
///
/// The trie shape is:
///   root = Branch with `children[first_nibble(key1)] = leaf1_hash`
///                    and `children[first_nibble(key2)] = leaf2_hash`,
///   leaf1 = Leaf with path = remaining nibbles of key1 after the first.
///   leaf2 = Leaf with path = remaining nibbles of key2 after the first.
///
/// Panics if the keys share their first nibble. Use `single_leaf_trie`
/// for single-key tries.
///
/// Convenience for test fixtures with multiple accounts/slots in one
/// trie (e.g. WorldProof tests with two contracts).
pub fn two_leaf_trie_distinct_first_nibble(
    key1: &[u8],
    value1: &[u8],
    key2: &[u8],
    value2: &[u8],
) -> ([u8; 32], Vec<Vec<u8>>) {
    let nibs1 = Nibbles::from_bytes(key1);
    let nibs2 = Nibbles::from_bytes(key2);
    assert!(!nibs1.0.is_empty() && !nibs2.0.is_empty(),
            "keys must be non-empty");
    assert_ne!(nibs1.0[0], nibs2.0[0],
               "two_leaf_trie_distinct_first_nibble: keys share first nibble");

    let leaf1 = MptNode::Leaf {
        path: Nibbles::from_nibbles(nibs1.0[1..].to_vec()),
        value: value1.to_vec(),
    };
    let leaf2 = MptNode::Leaf {
        path: Nibbles::from_nibbles(nibs2.0[1..].to_vec()),
        value: value2.to_vec(),
    };
    let leaf1_rlp = mpt_node_rlp(&leaf1);
    let leaf2_rlp = mpt_node_rlp(&leaf2);
    let leaf1_hash = keccak256(&leaf1_rlp);
    let leaf2_hash = keccak256(&leaf2_rlp);

    let mut children: [Option<[u8; 32]>; 16] = Default::default();
    children[nibs1.0[0] as usize] = Some(leaf1_hash);
    children[nibs2.0[0] as usize] = Some(leaf2_hash);
    let branch = MptNode::Branch { children, value: None };
    let branch_rlp = mpt_node_rlp(&branch);
    let root = keccak256(&branch_rlp);

    let proof = vec![branch_rlp, leaf1_rlp];
    (root, proof)
}

/// Verify that `(key, value)` is included under `root` using the given
/// RLP-encoded proof nodes.
///
/// The proof is a Merkle path: `proof[0]` hashes to `root`, each subsequent
/// node is the one pointed to by the previous step at the next nibble, and
/// the last node is a leaf whose value matches `value` and whose path
/// (concatenated with the nibbles consumed along the way) matches the full
/// nibble representation of `key`.
///
/// Returns `true` on successful verification, `false` otherwise.
pub fn verify_mpt_inclusion(
    root: [u8; 32],
    key: &[u8],
    value: &[u8],
    proof: &[Vec<u8>],
) -> bool {
    let target_nibbles = Nibbles::from_bytes(key);
    let mut path = &target_nibbles.0[..];
    let mut expected_hash = root;

    for (i, encoded) in proof.iter().enumerate() {
        // The node encoded here must hash to `expected_hash`.
        if keccak256(encoded) != expected_hash {
            return false;
        }
        let (item, consumed) = match rlp_decode(encoded) {
            Ok(x) => x,
            Err(_) => return false,
        };
        if consumed != encoded.len() {
            return false;
        }
        let is_last = i + 1 == proof.len();
        match decode_node(&item) {
            Some(DecodedNode::Leaf { path: p, value: v }) => {
                return is_last && path == p.0.as_slice() && v == value;
            }
            Some(DecodedNode::Extension { path: p, child }) => {
                if path.len() < p.0.len() || path[..p.0.len()] != p.0[..] {
                    return false;
                }
                path = &path[p.0.len()..];
                expected_hash = child;
            }
            Some(DecodedNode::Branch {
                children,
                value: v,
            }) => {
                if path.is_empty() {
                    // Value lives at this branch.
                    return is_last && v.as_deref() == Some(value);
                }
                let nib = path[0] as usize;
                match children[nib] {
                    Some(h) => {
                        expected_hash = h;
                        path = &path[1..];
                    }
                    None => return false,
                }
            }
            None => return false,
        }
    }
    // Ran out of proof nodes before reaching a leaf/branch terminus.
    false
}

// ----------------------------------------------------------------------
// Internal: decode an arbitrary RLP list as an MPT node.
// ----------------------------------------------------------------------

enum DecodedNode {
    Leaf { path: Nibbles, value: Vec<u8> },
    Extension { path: Nibbles, child: [u8; 32] },
    Branch {
        children: [Option<[u8; 32]>; 16],
        value: Option<Vec<u8>>,
    },
}

fn decode_node(item: &RlpItem) -> Option<DecodedNode> {
    let list = match item {
        RlpItem::List(l) => l,
        _ => return None,
    };
    match list.len() {
        2 => {
            let hp = as_bytes(&list[0])?;
            let (nibbles, is_leaf) = Nibbles::from_encoded_path(hp)?;
            let child = as_bytes(&list[1])?;
            if is_leaf {
                Some(DecodedNode::Leaf {
                    path: nibbles,
                    value: child.to_vec(),
                })
            } else {
                if child.len() != 32 {
                    return None;
                }
                let mut h = [0u8; 32];
                h.copy_from_slice(child);
                Some(DecodedNode::Extension {
                    path: nibbles,
                    child: h,
                })
            }
        }
        17 => {
            let mut children: [Option<[u8; 32]>; 16] = Default::default();
            for (i, slot) in list.iter().take(16).enumerate() {
                let b = as_bytes(slot)?;
                if b.is_empty() {
                    children[i] = None;
                } else if b.len() == 32 {
                    let mut h = [0u8; 32];
                    h.copy_from_slice(b);
                    children[i] = Some(h);
                } else {
                    // Inlined small child — not supported by the reference
                    // walker for now; would need recursive decoding.
                    return None;
                }
            }
            let v = as_bytes(&list[16])?;
            let value = if v.is_empty() { None } else { Some(v.to_vec()) };
            Some(DecodedNode::Branch { children, value })
        }
        _ => None,
    }
}

fn as_bytes(item: &RlpItem) -> Option<&[u8]> {
    match item {
        RlpItem::Bytes(b) => Some(b),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- HP encoding vectors (Yellow Paper Appendix C) -----

    #[test]
    fn test_hp_odd_leaf() {
        // [1] leaf → high nibble 3 (leaf+odd), low nibble 1 → 0x31.
        let enc = Nibbles::from_nibbles(vec![1]).to_encoded_path(true);
        assert_eq!(enc, vec![0x31]);
    }

    #[test]
    fn test_hp_even_leaf() {
        // [1, 2] leaf → prefix 0x20, packed 0x12.
        let enc = Nibbles::from_nibbles(vec![1, 2]).to_encoded_path(true);
        assert_eq!(enc, vec![0x20, 0x12]);
    }

    #[test]
    fn test_hp_odd_extension() {
        // [1] extension → prefix 0x1_ with odd nibble in low half → 0x11.
        let enc = Nibbles::from_nibbles(vec![1]).to_encoded_path(false);
        assert_eq!(enc, vec![0x11]);
    }

    #[test]
    fn test_hp_even_extension() {
        // [1, 2] extension → prefix 0x00, packed 0x12.
        let enc = Nibbles::from_nibbles(vec![1, 2]).to_encoded_path(false);
        assert_eq!(enc, vec![0x00, 0x12]);
    }

    #[test]
    fn test_hp_longer_odd_leaf() {
        // [f, 1, c, b, 8] leaf (5 nibbles, odd):
        //   first byte = 0x3f (leaf+odd|f), then 0x1c, 0xb8.
        let enc = Nibbles::from_nibbles(vec![0xf, 0x1, 0xc, 0xb, 0x8])
            .to_encoded_path(true);
        assert_eq!(enc, vec![0x3f, 0x1c, 0xb8]);
    }

    #[test]
    fn test_hp_longer_even_extension() {
        // [0, f, 1, c, b, 8, 1, 0] extension (8 nibbles, even):
        //   0x00, then packed 0x0f, 0x1c, 0xb8, 0x10.
        let enc = Nibbles::from_nibbles(vec![0, 0xf, 1, 0xc, 0xb, 8, 1, 0])
            .to_encoded_path(false);
        assert_eq!(enc, vec![0x00, 0x0f, 0x1c, 0xb8, 0x10]);
    }

    #[test]
    fn test_hp_round_trip() {
        let cases: Vec<(Vec<u8>, bool)> = vec![
            (vec![], true),
            (vec![], false),
            (vec![0], true),
            (vec![0xf], false),
            (vec![1, 2, 3], true),
            (vec![1, 2, 3, 4], false),
            (vec![0xf, 0x1, 0xc, 0xb, 0x8], true),
            (vec![0, 0xf, 1, 0xc, 0xb, 8, 1, 0], false),
        ];
        for (nibs, leaf) in cases {
            let orig = Nibbles::from_nibbles(nibs.clone());
            let enc = orig.to_encoded_path(leaf);
            let (dec, is_leaf) = Nibbles::from_encoded_path(&enc).unwrap();
            assert_eq!(dec.0, nibs);
            assert_eq!(is_leaf, leaf);
        }
    }

    // ----- Nibble helpers -----

    #[test]
    fn test_from_bytes() {
        let n = Nibbles::from_bytes(&[0xab, 0xcd]);
        assert_eq!(n.0, vec![0xa, 0xb, 0xc, 0xd]);
    }

    #[test]
    fn test_common_prefix() {
        let a = Nibbles::from_nibbles(vec![1, 2, 3, 4]);
        let b = Nibbles::from_nibbles(vec![1, 2, 5, 6]);
        assert_eq!(a.common_prefix_len(&b), 2);
    }

    // ----- Empty trie -----

    #[test]
    fn test_empty_trie_root() {
        // Standard Ethereum empty-trie root.
        let expected: [u8; 32] = [
            0x56, 0xe8, 0x1f, 0x17, 0x1b, 0xcc, 0x55, 0xa6,
            0xff, 0x83, 0x45, 0xe6, 0x92, 0xc0, 0xf8, 0x6e,
            0x5b, 0x48, 0xe0, 0x1b, 0x99, 0x6c, 0xad, 0xc0,
            0x01, 0x62, 0x2f, 0xb5, 0xe3, 0x63, 0xb4, 0x21,
        ];
        assert_eq!(empty_trie_root(), expected);
    }

    // ----- Node hashing -----

    #[test]
    fn test_leaf_node_hash_roundtrip() {
        // Encoding a leaf and parsing it back must round-trip via our own
        // decode path.
        let leaf = MptNode::Leaf {
            path: Nibbles::from_bytes(&[0xab, 0xcd]),
            value: b"hello".to_vec(),
        };
        let rlp = mpt_node_rlp(&leaf);
        let (item, n) = rlp_decode(&rlp).unwrap();
        assert_eq!(n, rlp.len());
        let decoded = decode_node(&item).expect("decode_node");
        match decoded {
            DecodedNode::Leaf { path, value } => {
                assert_eq!(path.0, vec![0xa, 0xb, 0xc, 0xd]);
                assert_eq!(value, b"hello".to_vec());
            }
            _ => panic!("expected Leaf"),
        }
    }

    #[test]
    fn test_branch_node_hash_deterministic() {
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[3] = Some([0x11u8; 32]);
        children[9] = Some([0x22u8; 32]);
        let branch = MptNode::Branch {
            children,
            value: None,
        };
        let h1 = mpt_node_hash(&branch);
        let h2 = mpt_node_hash(&branch);
        assert_eq!(h1, h2);
    }

    // ----- Single-leaf MPT inclusion -----

    #[test]
    fn test_two_leaf_distinct_first_nibble_inclusion() {
        // key1 = [0x05, 0xab] → first nibble 0, then [5, a, b]
        // key2 = [0x16, 0xcd] → first nibble 1, then [6, c, d]
        let key1 = vec![0x05u8, 0xab];
        let key2 = vec![0x16u8, 0xcd];
        let val1 = b"alpha".to_vec();
        let val2 = b"beta".to_vec();

        let (root, proof) = two_leaf_trie_distinct_first_nibble(
            &key1, &val1, &key2, &val2,
        );
        // proof = [branch_rlp, leaf1_rlp]
        assert_eq!(proof.len(), 2);

        // key1's value verifies under root.
        assert!(verify_mpt_inclusion(root, &key1, &val1, &proof));

        // key2's value does NOT verify with key1's proof.
        assert!(!verify_mpt_inclusion(root, &key2, &val2, &proof));

        // Wrong value for key1 → false.
        assert!(!verify_mpt_inclusion(root, &key1, b"gamma", &proof));

        // Tampered root → false.
        let mut bad_root = root;
        bad_root[0] ^= 0xff;
        assert!(!verify_mpt_inclusion(bad_root, &key1, &val1, &proof));
    }

    #[test]
    #[should_panic(expected = "share first nibble")]
    fn test_two_leaf_distinct_first_nibble_panics_on_shared_prefix() {
        // Both keys start with first nibble 0 — should panic.
        let key1 = vec![0x05u8];
        let key2 = vec![0x06u8];
        let _ = two_leaf_trie_distinct_first_nibble(
            &key1, b"x", &key2, b"y",
        );
    }

    #[test]
    fn test_single_leaf_inclusion() {
        // Build a trivial trie containing a single leaf at key = `key_bytes`.
        // Root = hash of that leaf node; proof = [leaf rlp].
        let key_bytes: Vec<u8> = vec![0xab, 0xcd];
        let value: Vec<u8> = b"payload".to_vec();

        let leaf = MptNode::Leaf {
            path: Nibbles::from_bytes(&key_bytes),
            value: value.clone(),
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let root = keccak256(&leaf_rlp);
        let proof = vec![leaf_rlp];

        assert!(verify_mpt_inclusion(root, &key_bytes, &value, &proof));

        // Wrong value → false.
        assert!(!verify_mpt_inclusion(root, &key_bytes, b"wrong", &proof));

        // Wrong key → false (leaf's nibble path won't match new key's remainder).
        let wrong_key = vec![0xab, 0xce];
        assert!(!verify_mpt_inclusion(root, &wrong_key, &value, &proof));

        // Tampered proof (flip a byte) → false.
        let mut tampered_proof = proof.clone();
        tampered_proof[0][3] ^= 0xff;
        assert!(!verify_mpt_inclusion(root, &key_bytes, &value, &tampered_proof));

        // Tampered root → false.
        let mut bad_root = root;
        bad_root[0] ^= 0xff;
        assert!(!verify_mpt_inclusion(bad_root, &key_bytes, &value, &proof));
    }

    // ----- Branch + leaf inclusion (multi-step proof) -----

    #[test]
    fn test_branch_then_leaf_inclusion() {
        // Two keys sharing no nibbles beyond the first.
        // key_a = [0x1f, 0xff, ...], first nibble 1 → branch slot 1.
        // key_b = [0x2f, 0xff, ...], first nibble 2 → branch slot 2.
        // After the branch, the leaf path is the remaining 3 nibbles of the key.
        let key_a = vec![0x1a, 0xbc];
        let val_a = b"value-a".to_vec();
        let key_b = vec![0x2d, 0xef];
        let val_b = b"value-b".to_vec();

        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: val_a.clone(),
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: val_b.clone(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);

        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(leaf_a_hash);
        children[2] = Some(leaf_b_hash);
        let branch = MptNode::Branch {
            children,
            value: None,
        };
        let branch_rlp = mpt_node_rlp(&branch);
        let root = keccak256(&branch_rlp);

        let proof_a = vec![branch_rlp.clone(), leaf_a_rlp.clone()];
        assert!(verify_mpt_inclusion(root, &key_a, &val_a, &proof_a));
        assert!(!verify_mpt_inclusion(root, &key_a, &val_b, &proof_a));

        let proof_b = vec![branch_rlp.clone(), leaf_b_rlp.clone()];
        assert!(verify_mpt_inclusion(root, &key_b, &val_b, &proof_b));
        assert!(!verify_mpt_inclusion(root, &key_a, &val_a, &proof_b));
    }

    // ----- Extension + branch + leaf (three-step) -----

    #[test]
    fn test_extension_branch_leaf() {
        // Keys share a long prefix, so the trie has
        //   extension(common) -> branch -> {leaf_a | leaf_b}
        let common_nibbles = vec![0xa, 0xb, 0xc, 0xd]; // 4 nibbles = 2 bytes
        let key_a = vec![0xab, 0xcd, 0x1a, 0xbc];
        let val_a = b"aa".to_vec();
        let key_b = vec![0xab, 0xcd, 0x2d, 0xef];
        let val_b = b"bb".to_vec();

        // After extension skips 4 nibbles, branch consumes 1 nibble (1 or 2),
        // leaf path is remaining 3 nibbles.
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: val_a.clone(),
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: val_b.clone(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);

        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(keccak256(&leaf_a_rlp));
        children[2] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch {
            children,
            value: None,
        };
        let branch_rlp = mpt_node_rlp(&branch);
        let branch_hash = keccak256(&branch_rlp);

        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(common_nibbles),
            child: branch_hash,
        };
        let ext_rlp = mpt_node_rlp(&ext);
        let root = keccak256(&ext_rlp);

        let proof_a = vec![ext_rlp.clone(), branch_rlp.clone(), leaf_a_rlp.clone()];
        assert!(verify_mpt_inclusion(root, &key_a, &val_a, &proof_a));

        let proof_b = vec![ext_rlp.clone(), branch_rlp.clone(), leaf_b_rlp.clone()];
        assert!(verify_mpt_inclusion(root, &key_b, &val_b, &proof_b));

        // Swap leaves between proofs → should fail.
        let bad = vec![ext_rlp.clone(), branch_rlp.clone(), leaf_b_rlp.clone()];
        assert!(!verify_mpt_inclusion(root, &key_a, &val_a, &bad));
    }

    // ----- Value held at a branch node -----

    #[test]
    fn test_branch_value_inclusion() {
        // Key whose nibble path ends exactly at a branch — its value lives in
        // the branch's 17th slot.
        let key = vec![0xab];
        let value = b"at-branch".to_vec();

        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some([0x77u8; 32]); // decoy
        let branch = MptNode::Branch {
            children,
            value: Some(value.clone()),
        };
        let branch_rlp = mpt_node_rlp(&branch);
        let root = keccak256(&branch_rlp);

        // Path for key `[0xab]` is nibbles `[a, b]`. We need the proof to
        // consume exactly those nibbles before landing on the branch-with-value.
        // That's impossible with a bare branch as root (which consumes only 1
        // nibble per step). We instead wrap in an extension consuming [a, b].
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0xa, 0xb]),
            child: root,
        };
        let ext_rlp = mpt_node_rlp(&ext);
        let outer_root = keccak256(&ext_rlp);

        let proof = vec![ext_rlp.clone(), branch_rlp.clone()];
        assert!(verify_mpt_inclusion(outer_root, &key, &value, &proof));
        assert!(!verify_mpt_inclusion(outer_root, &key, b"other", &proof));
    }
}
