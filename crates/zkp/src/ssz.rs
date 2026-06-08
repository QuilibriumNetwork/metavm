//! SSZ (Simple Serialize) hash-tree-root reference implementation.
//!
//! This is a minimal, VM-independent reference for the primitives the Ethereum
//! consensus layer uses to commit to state objects. It is intentionally light
//! on abstraction (no traits, no derive macros) — just the building blocks the
//! SSZ AIR and the higher-level type-root helpers need to cross-check their
//! outputs against.
//!
//! The primitives are:
//!
//! * [`pack`] — groups 32-byte leaves into chunks (identity on 32-byte leaves).
//! * [`merkleize_chunks`] — canonical SSZ merkleization, with optional chunk
//!   limit, zero-padding to the next power of two, and sha256 pair-hashing.
//! * [`mix_in_length`] and [`mix_in_selector`] — the length / selector mixer
//!   used for `List[T, N]` and `Union[...]`.
//!
//! Layered on top are the type-specific hashTreeRoot helpers for the basic
//! types (`uint64`, `bool`), byte containers (`Vector[byte, N]`,
//! `List[byte, N]`, `Bitlist[N]`), and a generic `Container` reducer.
//!
//! The sha256 primitive lives in [`crate::sha256`] so that the eventual AIR
//! witness generator exactly matches the arithmetic here.
//!
//! # References
//! * <https://github.com/ethereum/consensus-specs/blob/master/ssz/simple-serialize.md>
//! * <https://github.com/ethereum/consensus-specs/tree/master/tests/core/pyspec_tests/ssz_static>

#[allow(unused_imports)]
use crate::sha256::{sha256, sha256_pair};

/// A 32-byte chunk — the unit of merkleization.
pub type Chunk = [u8; 32];

/// The zero chunk — a 32-byte leaf of all zeros. Equivalent to `zero_hash[0]`
/// in the Ethereum spec.
pub const ZERO_CHUNK: Chunk = [0u8; 32];

/// Round up `n` to the next power of two. `n == 0` rounds to 1, matching the
/// SSZ convention that an empty merkleization tree still has a single root.
fn next_pow_of_two(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        1u64 << (64 - (n - 1).leading_zeros())
    }
}

/// Return the `zero_hash` at depth `d`: `Z[0] = 0^32`, `Z[d+1] = sha256(Z[d] || Z[d])`.
///
/// Precomputed and memoised per-call; for large trees a global LRU would be
/// preferable but this reference is intentionally simple.
fn zero_hash(depth: u32) -> Chunk {
    let mut z = ZERO_CHUNK;
    for _ in 0..depth {
        z = sha256_pair(&z, &z);
    }
    z
}

/// Group a slice of 32-byte leaves into chunks. For 32-byte leaves this is the
/// identity: the chunking boundary already coincides with the leaf boundary.
///
/// The function is kept for symmetry with the spec's `pack(values)` for basic
/// types, where the caller concatenates LE-encoded values and splits into
/// 32-byte chunks. See [`pack_bytes`] for that variant.
pub fn pack(leaves: &[Chunk]) -> Vec<Chunk> {
    leaves.to_vec()
}

/// Pack a byte slice into 32-byte chunks, zero-padding the final chunk. This
/// is the byte-oriented counterpart to [`pack`] used by `Bytes`, `Bitvector`,
/// `Bitlist`, and `List[basic, N]` serialisations.
pub fn pack_bytes(bytes: &[u8]) -> Vec<Chunk> {
    let num_chunks = bytes.len().div_ceil(32);
    let mut out = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let start = i * 32;
        let end = (start + 32).min(bytes.len());
        let mut chunk = [0u8; 32];
        chunk[..end - start].copy_from_slice(&bytes[start..end]);
        out.push(chunk);
    }
    out
}

/// Canonical SSZ merkleization of a chunk list.
///
/// * If `limit` is `None`, pad `chunks` to `next_pow_of_two(chunks.len())`.
/// * If `limit` is `Some(N)`, pad to `next_pow_of_two(max(chunks.len(), N))`.
///   The `max(..)` guard is a convenience: the spec requires `N >= len`, but we
///   tolerate over-full inputs by growing the tree (defensive).
///
/// Empty subtrees use the depth-indexed zero hash, so this runs in
/// `O(chunks.len() + log2(target))` pair-hashes regardless of how sparse the
/// tree is.
pub fn merkleize_chunks(chunks: &[Chunk], limit: Option<u64>) -> Chunk {
    // Determine the padded tree size (always a power of two, >= 1).
    let effective_len = match limit {
        Some(n) => (chunks.len() as u64).max(n),
        None => chunks.len() as u64,
    };
    let padded = next_pow_of_two(effective_len);
    let full_depth = padded.trailing_zeros();

    // Single-leaf tree: the "root" is the leaf itself (or zero).
    if padded == 1 {
        return chunks.first().copied().unwrap_or(ZERO_CHUNK);
    }

    // SPARSE bottom-up merkleization.
    //
    // Key invariant: for large `limit` (e.g. VALIDATOR_REGISTRY_LIMIT = 2^40)
    // with only a handful of real leaves, the tree is overwhelmingly zero.
    // Instead of materializing 2^full_depth leaves, we merkleize only the
    // populated subtree (up to `inner_depth = ceil(log2(chunks.len()))`),
    // then extend to `full_depth` by pairing the accumulated root with
    // `zero_hash(d)` at each subsequent layer — O(N + D) hashes, not O(2^D).
    let n = chunks.len();
    let inner_depth = if n == 0 {
        0
    } else {
        next_pow_of_two(n as u64).trailing_zeros()
    };

    // Root of the populated subtree (the part of the tree covering the actual
    // chunks, padded to its own nearest-pow2 height).
    let mut node = if n == 0 {
        ZERO_CHUNK
    } else {
        let mut layer: Vec<Chunk> = chunks.to_vec();
        for d in 0..inner_depth {
            let zh = zero_hash(d);
            let mut next = Vec::with_capacity((layer.len() + 1) / 2);
            let mut i = 0;
            while i < layer.len() {
                let left = layer[i];
                let right = if i + 1 < layer.len() { layer[i + 1] } else { zh };
                next.push(sha256_pair(&left, &right));
                i += 2;
            }
            layer = next;
        }
        layer[0]
    };

    // Extend up to full_depth by pairing with zero_hash at each layer — one
    // hash per layer, up to `full_depth - inner_depth` layers.
    for d in inner_depth..full_depth {
        let zh = zero_hash(d);
        node = sha256_pair(&node, &zh);
    }
    node
}

/// Mix a 64-bit length into a merkle root: `sha256(root || length_le_bytes_32)`.
///
/// The length occupies the low 8 bytes little-endian; the upper 24 bytes are
/// zero. This is the top-level combinator for `List[T, N]` hashTreeRoot.
pub fn mix_in_length(root: Chunk, length: u64) -> Chunk {
    let mut len_bytes = [0u8; 32];
    len_bytes[..8].copy_from_slice(&length.to_le_bytes());
    sha256_pair(&root, &len_bytes)
}

/// Mix a 64-bit union selector into a merkle root. Shape-identical to
/// [`mix_in_length`] but semantically distinct — used for `Union[...]`.
pub fn mix_in_selector(root: Chunk, selector: u64) -> Chunk {
    let mut sel_bytes = [0u8; 32];
    sel_bytes[..8].copy_from_slice(&selector.to_le_bytes());
    sha256_pair(&root, &sel_bytes)
}

// ---------------------------------------------------------------------------
// Basic type wrappers
// ---------------------------------------------------------------------------

/// hashTreeRoot(uint64) = 8 LE bytes right-padded to 32.
pub fn hash_tree_root_uint(value: u64) -> Chunk {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&value.to_le_bytes());
    out
}

/// hashTreeRoot(bool) = 0x00 or 0x01 in byte 0, remaining 31 bytes zero.
pub fn hash_tree_root_bool(value: bool) -> Chunk {
    let mut out = [0u8; 32];
    out[0] = value as u8;
    out
}

// ---------------------------------------------------------------------------
// Byte-oriented wrappers
// ---------------------------------------------------------------------------

/// hashTreeRoot(Bytes(...)) treating `bytes` as a bare concatenation packed
/// into chunks and merkleized with no limit (i.e. pad to next power of two of
/// the actual chunk count). This is the "variable-length, no max" form used
/// internally — most beacon types pin a maximum and use a Vector/List wrapper.
pub fn hash_tree_root_bytes(bytes: &[u8]) -> Chunk {
    let chunks = pack_bytes(bytes);
    merkleize_chunks(&chunks, None)
}

/// hashTreeRoot(Vector[byte, N]): fixed-length byte vector. The chunk limit is
/// `ceil(N/32)`; the input must be exactly `N` bytes (callers are responsible
/// for that invariant — we merkleize whatever is provided).
pub fn hash_tree_root_bytes_fixed(bytes: &[u8], chunks_limit: u64) -> Chunk {
    let chunks = pack_bytes(bytes);
    merkleize_chunks(&chunks, Some(chunks_limit))
}

/// hashTreeRoot(List[byte, N]): `merkleize(pack(bytes), ceil(N/32)) -> mix_in_length(.., len(bytes))`.
///
/// `max_length_bytes` is the type-level byte cap `N`; it's rounded up to the
/// nearest chunk as the merkleization limit. The mixed-in length is the byte
/// count of `bytes` (not the chunk count).
pub fn hash_tree_root_list_bytes(bytes: &[u8], max_length_bytes: u64) -> Chunk {
    let chunks_limit = max_length_bytes.div_ceil(32);
    let chunks = pack_bytes(bytes);
    let root = merkleize_chunks(&chunks, Some(chunks_limit));
    mix_in_length(root, bytes.len() as u64)
}

/// hashTreeRoot(Bitlist[N]): strip the sentinel bit, merkleize the packed data
/// bits, then mix_in_length(bit_length).
///
/// **Encoding note**: SSZ Bitlist serialisation appends a `1` bit immediately
/// after the last data bit to mark the end of the list. That sentinel is
/// purely a length delimiter and must be zeroed before merkleization.
/// `bits` is the full SSZ-serialized Bitlist (so its last meaningful bit is
/// the sentinel), `bit_length` is the count of data bits only, and `max_bits`
/// is the type-level cap `N`.
pub fn hash_tree_root_bitlist(bits: &[u8], bit_length: u64, max_bits: u64) -> Chunk {
    let data_bytes_len = bit_length.div_ceil(8) as usize;

    // Copy into a local buffer and zero the sentinel bit.
    let mut data = vec![0u8; data_bytes_len];
    if bit_length > 0 {
        // Take at most `data_bytes_len` bytes from `bits`; they already hold
        // the data bits in positions `[0..bit_length)`.
        let copy_n = data_bytes_len.min(bits.len());
        data[..copy_n].copy_from_slice(&bits[..copy_n]);

        // Zero any bits >= bit_length in the final byte. The sentinel, which
        // sits at position `bit_length`, is included in this mask unless it
        // landed in a fresh byte that data_bytes_len didn't reach — which
        // can't happen because bit_length > 0 so bit bit_length-1 is in
        // the last data byte, and bit `bit_length` is in the same byte when
        // bit_length % 8 != 0.
        let tail_bit = (bit_length % 8) as u32;
        if tail_bit != 0 {
            let mask = (1u8 << tail_bit) - 1;
            let idx = data_bytes_len - 1;
            data[idx] &= mask;
        }
    }

    let chunks_limit = max_bits.div_ceil(256);
    let chunks = pack_bytes(&data);
    let root = merkleize_chunks(&chunks, Some(chunks_limit));
    mix_in_length(root, bit_length)
}

// ---------------------------------------------------------------------------
// Container reducer
// ---------------------------------------------------------------------------

/// hashTreeRoot(Container{f1, f2, ...}) = merkleize(field_roots) with no limit
/// (the tree is padded to the next power of two of the field count).
pub fn hash_tree_root_container(field_roots: &[Chunk]) -> Chunk {
    merkleize_chunks(field_roots, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(chunk: &Chunk) -> String {
        let mut s = String::with_capacity(64);
        for b in chunk {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    // -----------------------------------------------------------------------
    // uint / bool
    // -----------------------------------------------------------------------

    #[test]
    fn uint_zero() {
        assert_eq!(hash_tree_root_uint(0), [0u8; 32]);
    }

    #[test]
    fn uint_one() {
        let mut expected = [0u8; 32];
        expected[0] = 1;
        assert_eq!(hash_tree_root_uint(1), expected);
    }

    #[test]
    fn uint_deadbeef() {
        // 0xdeadbeef as uint64 LE = ef be ad de 00 00 00 00 ...
        let mut expected = [0u8; 32];
        expected[..8].copy_from_slice(&0xdeadbeefu64.to_le_bytes());
        assert_eq!(hash_tree_root_uint(0xdeadbeef), expected);
        // Spell out the first bytes for humans.
        assert_eq!(&expected[..8], &[0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn uint_u64_max() {
        let mut expected = [0u8; 32];
        for b in &mut expected[..8] { *b = 0xff; }
        assert_eq!(hash_tree_root_uint(u64::MAX), expected);
    }

    #[test]
    fn bool_false_true() {
        assert_eq!(hash_tree_root_bool(false), [0u8; 32]);
        let mut expected = [0u8; 32];
        expected[0] = 1;
        assert_eq!(hash_tree_root_bool(true), expected);
    }

    // -----------------------------------------------------------------------
    // pack / merkleize_chunks
    // -----------------------------------------------------------------------

    #[test]
    fn next_pow_of_two_edges() {
        assert_eq!(next_pow_of_two(0), 1);
        assert_eq!(next_pow_of_two(1), 1);
        assert_eq!(next_pow_of_two(2), 2);
        assert_eq!(next_pow_of_two(3), 4);
        assert_eq!(next_pow_of_two(4), 4);
        assert_eq!(next_pow_of_two(5), 8);
        assert_eq!(next_pow_of_two(1024), 1024);
        assert_eq!(next_pow_of_two(1025), 2048);
    }

    #[test]
    fn pack_identity_on_chunks() {
        let leaves = vec![[0x11u8; 32], [0x22u8; 32], [0x33u8; 32]];
        assert_eq!(pack(&leaves), leaves);
    }

    #[test]
    fn pack_bytes_partial_chunk_zero_padded() {
        let bytes = [1u8, 2, 3, 4, 5];
        let chunks = pack_bytes(&bytes);
        assert_eq!(chunks.len(), 1);
        let mut expected = [0u8; 32];
        expected[..5].copy_from_slice(&bytes);
        assert_eq!(chunks[0], expected);
    }

    #[test]
    fn pack_bytes_exactly_two_chunks() {
        let bytes = [7u8; 64];
        let chunks = pack_bytes(&bytes);
        assert_eq!(chunks, vec![[7u8; 32], [7u8; 32]]);
    }

    #[test]
    fn pack_bytes_empty() {
        assert!(pack_bytes(&[]).is_empty());
    }

    #[test]
    fn merkleize_empty_with_limit_one_is_zero_chunk() {
        let root = merkleize_chunks(&[], Some(1));
        assert_eq!(root, ZERO_CHUNK);
    }

    #[test]
    fn merkleize_single_zero_leaf_identity() {
        // With no limit and a single leaf, merkleize returns the leaf itself.
        let root = merkleize_chunks(&[ZERO_CHUNK], None);
        assert_eq!(root, ZERO_CHUNK);
    }

    #[test]
    fn merkleize_single_nonzero_leaf_identity() {
        let leaf = [0x42u8; 32];
        assert_eq!(merkleize_chunks(&[leaf], None), leaf);
    }

    #[test]
    fn merkleize_two_identical_leaves() {
        let leaf = [0x42u8; 32];
        let root = merkleize_chunks(&[leaf, leaf], None);
        // = sha256(leaf || leaf)
        let expected = sha256_pair(&leaf, &leaf);
        assert_eq!(root, expected);
    }

    #[test]
    fn merkleize_four_leaves_matches_manual_tree() {
        // Classic Merkle-tree cross-check: root should equal
        //   sha256(sha256(a||b) || sha256(c||d)).
        let a = [0x01u8; 32];
        let b = [0x02u8; 32];
        let c = [0x03u8; 32];
        let d = [0x04u8; 32];
        let root = merkleize_chunks(&[a, b, c, d], None);
        let left = sha256_pair(&a, &b);
        let right = sha256_pair(&c, &d);
        let expected = sha256_pair(&left, &right);
        assert_eq!(root, expected);
    }

    #[test]
    fn merkleize_three_leaves_pads_with_zero() {
        // 3 leaves -> padded to 4, the missing leaf is ZERO_CHUNK.
        let a = [0x01u8; 32];
        let b = [0x02u8; 32];
        let c = [0x03u8; 32];
        let root = merkleize_chunks(&[a, b, c], None);
        let expected = sha256_pair(&sha256_pair(&a, &b), &sha256_pair(&c, &ZERO_CHUNK));
        assert_eq!(root, expected);
    }

    #[test]
    fn merkleize_with_limit_pads_to_limit_tree_size() {
        // 1 leaf, limit 4 -> tree of depth 2 (4 leaves).
        let a = [0x42u8; 32];
        let root = merkleize_chunks(&[a], Some(4));
        // Expected: sha256(sha256(a||0) || sha256(0||0)) where 0 = ZERO_CHUNK.
        let z0 = ZERO_CHUNK;
        let expected = sha256_pair(&sha256_pair(&a, &z0), &sha256_pair(&z0, &z0));
        assert_eq!(root, expected);
    }

    #[test]
    fn merkleize_zero_hash_tree_matches_depth_1() {
        // Two all-zero leaves -> root = sha256(0^64) = Ethereum zero_hash[1].
        let root = merkleize_chunks(&[ZERO_CHUNK, ZERO_CHUNK], None);
        assert_eq!(
            hex(&root),
            "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        );
    }

    #[test]
    fn zero_hash_ladder() {
        // Ethereum consensus-specs canonical zero_hash chain (first few layers).
        // These are the values used throughout beacon-state merkleization.
        assert_eq!(hex(&zero_hash(0)), "0000000000000000000000000000000000000000000000000000000000000000");
        assert_eq!(hex(&zero_hash(1)), "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b");
        assert_eq!(hex(&zero_hash(2)), "db56114e00fdd4c1f85c892bf35ac9a89289aaecb1ebd0a96cde606a748b5d71");
        assert_eq!(hex(&zero_hash(3)), "c78009fdf07fc56a11f122370658a353aaa542ed63e44c4bc15ff4cd105ab33c");
    }

    // -----------------------------------------------------------------------
    // mix_in_length / mix_in_selector
    // -----------------------------------------------------------------------

    #[test]
    fn mix_in_length_zero_is_sha256_zeros() {
        // mix_in_length([0;32], 0) = sha256([0;64])
        let got = mix_in_length(ZERO_CHUNK, 0);
        let expected = sha256(&[0u8; 64]);
        assert_eq!(got, expected);
        // Cross-check: this is zero_hash[1].
        assert_eq!(
            hex(&got),
            "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        );
    }

    #[test]
    fn mix_in_length_encoding_le() {
        // Length 1 -> low byte of the right half = 0x01, rest zero.
        let got = mix_in_length(ZERO_CHUNK, 1);
        let mut buf = [0u8; 64];
        buf[32] = 1; // first byte of the second half
        let expected = sha256(&buf);
        assert_eq!(got, expected);
    }

    #[test]
    fn mix_in_selector_same_shape() {
        // mix_in_selector is structurally identical to mix_in_length.
        let leaf = [0x55u8; 32];
        assert_eq!(mix_in_length(leaf, 7), mix_in_selector(leaf, 7));
    }

    // -----------------------------------------------------------------------
    // byte / list / bitlist roots
    // -----------------------------------------------------------------------

    #[test]
    fn hash_tree_root_bytes_fixed_all_zero_vector_32() {
        // Vector[byte, 32] of all zeros -> single zero chunk, root = ZERO_CHUNK.
        let bytes = [0u8; 32];
        assert_eq!(hash_tree_root_bytes_fixed(&bytes, 1), ZERO_CHUNK);
    }

    #[test]
    fn hash_tree_root_bytes_fixed_n_96() {
        // Vector[byte, 96] -> ceil(96/32) = 3 chunks, padded to 4.
        // A known beacon case: a zero 96-byte BLSPubkey-ish vector roots
        // to zero_hash[2].
        let bytes = [0u8; 96];
        assert_eq!(
            hex(&hash_tree_root_bytes_fixed(&bytes, 3)),
            "db56114e00fdd4c1f85c892bf35ac9a89289aaecb1ebd0a96cde606a748b5d71"
        );
    }

    #[test]
    fn hash_tree_root_list_bytes_empty() {
        // Empty list of bytes, max=32: merkleize 1 zero chunk -> ZERO_CHUNK,
        // mix_in_length(0) = zero_hash[1].
        let root = hash_tree_root_list_bytes(&[], 32);
        assert_eq!(
            hex(&root),
            "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        );
    }

    #[test]
    fn hash_tree_root_list_bytes_mixes_length() {
        // Two lists with the same packed bytes but different max should have
        // different roots iff the chunk limit differs enough to change the
        // tree depth. With identical max, differing bytes must differ.
        let a = hash_tree_root_list_bytes(b"hello", 64);
        let b = hash_tree_root_list_bytes(b"hellp", 64);
        assert_ne!(a, b);

        // Length mixing: "hello" (5 bytes) and "hello\0" (6 bytes) have the
        // same single packed chunk only if the trailing byte pads equally —
        // but the mixed-in length differs, so the roots must differ.
        let c = hash_tree_root_list_bytes(b"hello", 64);
        let d = hash_tree_root_list_bytes(b"hello\0", 64);
        assert_ne!(c, d);
    }

    #[test]
    fn hash_tree_root_bitlist_strips_sentinel() {
        // Bitlist[8] of 5 set data bits (0b11111), serialized with a
        // sentinel at bit index 5 -> byte = 0b00111111 = 0x3f.
        // After stripping the sentinel: byte = 0b00011111 = 0x1f, bit_len=5.
        let serialized = [0x3fu8];
        let got = hash_tree_root_bitlist(&serialized, 5, 8);

        // Manual reference: packed data = [0x1f, 0, .., 0], chunks_limit = 1.
        let mut packed = [0u8; 32];
        packed[0] = 0x1f;
        let merkle_root = packed; // single chunk, limit 1 -> root = leaf.
        let expected = mix_in_length(merkle_root, 5);
        assert_eq!(got, expected);
    }

    #[test]
    fn hash_tree_root_bitlist_empty() {
        // Empty bitlist (no data bits; sentinel is the lone set bit at index 0).
        // Serialization: [0x01]. After stripping the sentinel we have zero
        // data bytes; merkleize pads to chunks_limit chunks of zero.
        let serialized = [0x01u8];
        let got = hash_tree_root_bitlist(&serialized, 0, 64);

        // chunks_limit = ceil(64/256) = 1 -> merkleize(&[], Some(1)) = ZERO_CHUNK.
        // mix_in_length(ZERO_CHUNK, 0) = zero_hash[1].
        assert_eq!(
            hex(&got),
            "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        );
    }

    #[test]
    fn hash_tree_root_bitlist_byte_aligned() {
        // Exactly 8 data bits (one full byte). Serialization then holds a
        // second byte for the sentinel at bit index 8.
        let serialized = [0xffu8, 0x01];
        let got = hash_tree_root_bitlist(&serialized, 8, 16);
        // Packed data is one byte 0xff; no mask needed since bit_length % 8 == 0.
        let mut packed = [0u8; 32];
        packed[0] = 0xff;
        // chunks_limit = ceil(16/256) = 1.
        let expected = mix_in_length(packed, 8);
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Container
    // -----------------------------------------------------------------------

    #[test]
    fn container_single_field_identity() {
        let f = [0x55u8; 32];
        assert_eq!(hash_tree_root_container(&[f]), f);
    }

    #[test]
    fn container_two_fields_is_pair_hash() {
        let f1 = hash_tree_root_uint(1);
        let f2 = hash_tree_root_uint(2);
        let expected = sha256_pair(&f1, &f2);
        assert_eq!(hash_tree_root_container(&[f1, f2]), expected);
    }

    // -----------------------------------------------------------------------
    // End-to-end cross-check: DepositMessage-shaped container.
    //
    // DepositMessage { pubkey: Bytes48, withdrawal_credentials: Bytes32, amount: uint64 }
    // hashTreeRoot = merkleize([
    //     hash_tree_root(pubkey),
    //     hash_tree_root(withdrawal_credentials),
    //     hash_tree_root(amount),
    // ])
    //
    // With all-zero fields this equals zero_hash[2], which is the canonical
    // "empty 3-field container" root.
    // -----------------------------------------------------------------------

    #[test]
    fn deposit_message_all_zero_fields_manual_merkleization() {
        // DepositMessage { pubkey: Bytes48, withdrawal_credentials: Bytes32, amount: uint64 }
        // All-zero fields. The field roots are NOT all equal to ZERO_CHUNK
        // because `Bytes48` merkleizes to `sha256(0^32 || 0^32) = zero_hash[1]`,
        // not ZERO_CHUNK. This is exactly the sort of quirk the tests need to
        // lock in.
        let pubkey_root = hash_tree_root_bytes_fixed(&[0u8; 48], 48u64.div_ceil(32));
        let wcred_root = hash_tree_root_bytes_fixed(&[0u8; 32], 1);
        let amount_root = hash_tree_root_uint(0);

        assert_eq!(pubkey_root, zero_hash(1), "Bytes48 all-zero root = zero_hash[1]");
        assert_eq!(wcred_root, ZERO_CHUNK, "Bytes32 all-zero root = ZERO_CHUNK");
        assert_eq!(amount_root, ZERO_CHUNK, "uint64 zero root = ZERO_CHUNK");

        let root = hash_tree_root_container(&[pubkey_root, wcred_root, amount_root]);
        // Merkleize 3 fields -> pad to 4:
        //   root = sha256(sha256(pubkey_root || wcred_root) || sha256(amount_root || 0))
        let expected = sha256_pair(
            &sha256_pair(&pubkey_root, &wcred_root),
            &sha256_pair(&amount_root, &ZERO_CHUNK),
        );
        assert_eq!(root, expected);
    }

    #[test]
    fn deposit_message_nonzero_matches_manual_merkleization() {
        // pubkey = 48 bytes of 0x11, wcred = 32 bytes of 0x22, amount = 32 ETH (in Gwei).
        let pubkey = [0x11u8; 48];
        let wcred = [0x22u8; 32];
        let amount: u64 = 32_000_000_000;

        let pubkey_root = hash_tree_root_bytes_fixed(&pubkey, 48u64.div_ceil(32));
        let wcred_root = hash_tree_root_bytes_fixed(&wcred, 1);
        let amount_root = hash_tree_root_uint(amount);

        let container_root = hash_tree_root_container(&[pubkey_root, wcred_root, amount_root]);

        // Manual reconstruction:
        // pubkey_root = merkleize_chunks([p[0..32], p[32..48]||0^16], limit=2)
        let mut p0 = [0u8; 32]; p0.copy_from_slice(&pubkey[..32]);
        let mut p1 = [0u8; 32]; p1[..16].copy_from_slice(&pubkey[32..]);
        let pk = sha256_pair(&p0, &p1);
        assert_eq!(pubkey_root, pk);

        // amount_root is uint64 LE padded.
        let mut amt = [0u8; 32];
        amt[..8].copy_from_slice(&amount.to_le_bytes());
        assert_eq!(amount_root, amt);

        // 3 fields padded to 4: root = sha256(sha256(pk||wcred) || sha256(amt||0)).
        let left = sha256_pair(&pk, &wcred);
        let right = sha256_pair(&amt, &ZERO_CHUNK);
        let expected = sha256_pair(&left, &right);
        assert_eq!(container_root, expected);
    }
}
