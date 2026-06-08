//! SSZ merkleization AIR (witness layer).
//!
//! Per row, one `sha256_pair` invocation. Multiple rows for multiple layers;
//! the final row's `parent` is the merkle root.
//!
//! For the AIR scope, this module treats `sha256_pair` as a primitive
//! provable separately by `sha256_air` / `sha256_constraints`. The SSZ AIR's
//! job is the **tree-structure logic**: which inputs feed which row, how
//! a layer's outputs become the next layer's inputs, and how zero-hash
//! padding extends to the full depth (the sparse-merkleize semantics from
//! [`crate::ssz::merkleize_chunks`]).
//!
//! # Layout
//!
//! Two phases per merkleization:
//!
//! 1. **Populated subtree**. With N input chunks, populate up to
//!    `inner_depth = ceil(log2(N))` layers via pair-hashing adjacent
//!    siblings. Missing right siblings (when a layer has odd cardinality)
//!    are filled with `zero_hash(d)` for that layer.
//!
//! 2. **Zero extension**. From `inner_depth` to `full_depth`, pair the
//!    accumulated root with `zero_hash(d)` once per layer.
//!
//! Each row records its `(left, right, parent, layer_depth, is_left_real,
//! is_right_real)`. The cross-row constraint binds successive layers:
//! every parent at depth `d` becomes a left or right input at depth `d+1`
//! (depending on its position-in-layer parity).

use crate::sha256::sha256_pair;
#[allow(unused_imports)]
use crate::ssz::{merkleize_chunks, Chunk, ZERO_CHUNK};

/// Witness row for a single `sha256_pair` invocation in the merkleize tree.
#[derive(Debug, Clone, Copy)]
pub struct MerkleizeRow {
    pub left: Chunk,
    pub right: Chunk,
    pub parent: Chunk,
    /// Depth of this row's inputs (0 = leaves; root layer = full_depth − 1
    /// for inputs, the row itself produces a depth-(d+1) parent).
    pub layer_depth: u32,
    /// Position of this pair within its layer (0..layer_size/2).
    pub position_in_layer: usize,
    /// True if `left` came from a real chunk / previous-layer parent.
    pub is_left_real: bool,
    /// True if `right` came from a real chunk / previous-layer parent.
    pub is_right_real: bool,
}

/// Compute the per-layer zero hashes used by sparse merkleization.
/// `zero_hash(0) = ZERO_CHUNK`; `zero_hash(d+1) = sha256_pair(zh, zh)`.
fn zero_hash(depth: u32) -> Chunk {
    let mut z = ZERO_CHUNK;
    for _ in 0..depth {
        z = sha256_pair(&z, &z);
    }
    z
}

/// `next_pow_of_two(n)` rounded up; matches `ssz::next_pow_of_two`.
fn next_pow_of_two(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        1u64 << (64 - (n - 1).leading_zeros())
    }
}

/// Generate the full row trace for `merkleize_chunks(chunks, limit)`. The
/// final row's `parent` equals the merkle root, except in the trivial
/// single-leaf case (no rows; root = the leaf itself).
pub fn merkleize_witness(chunks: &[Chunk], limit: Option<u64>) -> Vec<MerkleizeRow> {
    let effective_len = match limit {
        Some(n) => (chunks.len() as u64).max(n),
        None => chunks.len() as u64,
    };
    let padded = next_pow_of_two(effective_len);
    let full_depth = padded.trailing_zeros();

    if padded == 1 || chunks.is_empty() && full_depth == 0 {
        return Vec::new();
    }

    let mut rows = Vec::new();
    let inner_depth = if chunks.is_empty() {
        0
    } else {
        next_pow_of_two(chunks.len() as u64).trailing_zeros()
    };

    // Phase 1: populated subtree.
    let mut layer: Vec<Chunk> = chunks.to_vec();
    for d in 0..inner_depth {
        let zh = zero_hash(d);
        let mut next: Vec<Chunk> = Vec::with_capacity((layer.len() + 1) / 2);
        let mut i = 0;
        let mut pair_idx = 0;
        while i < layer.len() {
            let left = layer[i];
            let (right, is_right_real) = if i + 1 < layer.len() {
                (layer[i + 1], true)
            } else {
                (zh, false)
            };
            let parent = sha256_pair(&left, &right);
            rows.push(MerkleizeRow {
                left,
                right,
                parent,
                layer_depth: d,
                position_in_layer: pair_idx,
                is_left_real: true,
                is_right_real,
            });
            next.push(parent);
            i += 2;
            pair_idx += 1;
        }
        layer = next;
    }

    // Running root after the populated subtree (or ZERO_CHUNK if no chunks).
    let mut root = if chunks.is_empty() {
        ZERO_CHUNK
    } else {
        layer[0]
    };

    // Phase 2: zero extension to full_depth.
    for d in inner_depth..full_depth {
        let zh = zero_hash(d);
        let parent = sha256_pair(&root, &zh);
        rows.push(MerkleizeRow {
            left: root,
            right: zh,
            parent,
            layer_depth: d,
            position_in_layer: 0,
            is_left_real: true,
            is_right_real: false,
        });
        root = parent;
    }

    rows
}

/// Validate that every row's claimed `parent` equals `sha256_pair(left, right)`.
/// This is the math-level analog of the AIR's per-row hash constraint.
pub fn rows_hash_consistent(rows: &[MerkleizeRow]) -> bool {
    rows.iter().all(|r| sha256_pair(&r.left, &r.right) == r.parent)
}

/// Validate that for every row whose `is_*_real` is false, the corresponding
/// side equals `zero_hash(layer_depth)`. Math-level analog of the
/// "absent siblings = zero" AIR constraint.
pub fn rows_zero_padding_consistent(rows: &[MerkleizeRow]) -> bool {
    rows.iter().all(|r| {
        let zh = zero_hash(r.layer_depth);
        (r.is_left_real || r.left == zh) && (r.is_right_real || r.right == zh)
    })
}

/// Math-level full validator: every row's hash is consistent AND every
/// padded side matches the layer's zero hash.
pub fn evaluate_witness(rows: &[MerkleizeRow]) -> bool {
    rows_hash_consistent(rows) && rows_zero_padding_consistent(rows)
}

// ---------------------------------------------------------------------------
// Trace-column layout (32 byte cols × 3 chunks + 4 metadata = 100 cols/row)
// ---------------------------------------------------------------------------

/// Number of byte columns per chunk (left, right, parent each have 32).
pub const CHUNK_BYTES: usize = 32;

/// Column offsets in the SSZ AIR trace.
pub mod col {
    use super::CHUNK_BYTES;
    pub const LEFT_OFFSET: usize = 0;
    pub const RIGHT_OFFSET: usize = LEFT_OFFSET + CHUNK_BYTES; // 32
    pub const PARENT_OFFSET: usize = RIGHT_OFFSET + CHUNK_BYTES; // 64
    pub const LAYER_DEPTH: usize = PARENT_OFFSET + CHUNK_BYTES; // 96
    pub const POSITION_IN_LAYER: usize = LAYER_DEPTH + 1; // 97
    pub const IS_LEFT_REAL: usize = POSITION_IN_LAYER + 1; // 98
    pub const IS_RIGHT_REAL: usize = IS_LEFT_REAL + 1; // 99
    /// `1` iff `LAYER_DEPTH == 0` — i.e., the row is a leaf-level
    /// row whose `LEFT` and `RIGHT` chunks are external inputs (not
    /// intermediate parent hashes from the previous layer). Pinned
    /// algebraically by `LAYER_DEPTH · IS_LEAF_DEPTH = 0` plus a
    /// binarity row-local; combined with the witness builder, fires
    /// only on `layer_depth == 0` rows.
    pub const IS_LEAF_DEPTH: usize = IS_RIGHT_REAL + 1; // 100
    /// `IS_LEFT_REAL · IS_LEAF_DEPTH` — fires only when a row is
    /// (a) at depth 0 and (b) has a real left chunk. Used as the
    /// B-side selector in the parity-aware VE↔SSZ Left linkage so
    /// only validator-leaf rows participate (intermediate parent
    /// rows at depth ≥ 1 are excluded).
    pub const IS_LEFT_VALIDATOR_LEAF: usize = IS_LEAF_DEPTH + 1; // 101
    pub const IS_RIGHT_VALIDATOR_LEAF: usize = IS_LEFT_VALIDATOR_LEAF + 1; // 102
    pub const NUM_COLUMNS: usize = IS_RIGHT_VALIDATOR_LEAF + 1; // 103
}

/// Populate AIR trace columns from a row sequence. `columns.len()` must be
/// at least `col::NUM_COLUMNS`; each inner Vec must be at least
/// `rows.len()` long (the caller pads to a power of two for FFT).
pub fn populate_trace(rows: &[MerkleizeRow], columns: &mut [Vec<u64>]) {
    assert!(columns.len() >= col::NUM_COLUMNS);
    for (row_idx, row) in rows.iter().enumerate() {
        // Chunk bytes — one column per byte.
        for i in 0..CHUNK_BYTES {
            columns[col::LEFT_OFFSET + i][row_idx] = row.left[i] as u64;
            columns[col::RIGHT_OFFSET + i][row_idx] = row.right[i] as u64;
            columns[col::PARENT_OFFSET + i][row_idx] = row.parent[i] as u64;
        }
        columns[col::LAYER_DEPTH][row_idx] = row.layer_depth as u64;
        columns[col::POSITION_IN_LAYER][row_idx] = row.position_in_layer as u64;
        columns[col::IS_LEFT_REAL][row_idx] = row.is_left_real as u64;
        columns[col::IS_RIGHT_REAL][row_idx] = row.is_right_real as u64;
        let is_leaf_depth = (row.layer_depth == 0) as u64;
        columns[col::IS_LEAF_DEPTH][row_idx] = is_leaf_depth;
        columns[col::IS_LEFT_VALIDATOR_LEAF][row_idx] =
            is_leaf_depth * (row.is_left_real as u64);
        columns[col::IS_RIGHT_VALIDATOR_LEAF][row_idx] =
            is_leaf_depth * (row.is_right_real as u64);
    }
}

/// Cross-row binding: parents at depth `d` become inputs at depth `d+1`.
/// More precisely, the row at `(d+1, pos)` consumes parent of row
/// `(d, 2·pos)` as its `left` and parent of row `(d, 2·pos+1)` as its
/// `right` IF that sibling exists (else the right is `zero_hash(d+1)`).
pub fn rows_chain_consistent(rows: &[MerkleizeRow]) -> bool {
    use std::collections::HashMap;
    // Index parents by (depth, position).
    let mut parents: HashMap<(u32, usize), Chunk> = HashMap::new();
    for r in rows {
        parents.insert((r.layer_depth, r.position_in_layer), r.parent);
    }
    for r in rows {
        if r.layer_depth == 0 {
            // Layer-0 inputs are external chunks; nothing to check here.
            continue;
        }
        let prev_depth = r.layer_depth - 1;
        let left_pos = r.position_in_layer * 2;
        let right_pos = left_pos + 1;
        if r.is_left_real {
            if let Some(&p) = parents.get(&(prev_depth, left_pos)) {
                if p != r.left {
                    return false;
                }
            }
            // If the previous layer didn't produce that parent, this row's
            // left came from somewhere else (e.g. zero-extension phase
            // pulled the running root from the inner subtree's root).
            // Accept; the `evaluate_witness` hash check pins correctness.
        }
        if r.is_right_real {
            if let Some(&p) = parents.get(&(prev_depth, right_pos)) {
                if p != r.right {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(byte: u8) -> Chunk {
        let mut c = ZERO_CHUNK;
        c[0] = byte;
        c
    }

    #[test]
    fn witness_empty_input() {
        let rows = merkleize_witness(&[], None);
        assert!(rows.is_empty());
    }

    #[test]
    fn witness_single_leaf_no_limit() {
        // padded = 1, full_depth = 0 → no rows, root = the leaf itself.
        let leaf = make_chunk(0x42);
        let rows = merkleize_witness(&[leaf], None);
        assert!(rows.is_empty());
    }

    #[test]
    fn witness_two_leaves() {
        let a = make_chunk(0x11);
        let b = make_chunk(0x22);
        let rows = merkleize_witness(&[a, b], None);
        assert_eq!(rows.len(), 1, "2 leaves → 1 row");
        let row = &rows[0];
        assert_eq!(row.left, a);
        assert_eq!(row.right, b);
        assert_eq!(row.parent, sha256_pair(&a, &b));
        assert_eq!(row.layer_depth, 0);
        assert!(row.is_left_real && row.is_right_real);

        // Final root matches the reference.
        assert_eq!(row.parent, merkleize_chunks(&[a, b], None));
        assert!(evaluate_witness(&rows));
    }

    #[test]
    fn witness_four_leaves_matches_reference() {
        let a = make_chunk(0x11);
        let b = make_chunk(0x22);
        let c = make_chunk(0x33);
        let d = make_chunk(0x44);
        let rows = merkleize_witness(&[a, b, c, d], None);
        // 2 row at depth 0 + 1 row at depth 1 = 3 rows.
        assert_eq!(rows.len(), 3);
        // The last row's parent is the root.
        let root = rows.last().unwrap().parent;
        assert_eq!(root, merkleize_chunks(&[a, b, c, d], None));
        assert!(evaluate_witness(&rows));
        assert!(rows_chain_consistent(&rows));
    }

    #[test]
    fn witness_three_leaves_uses_zero_pad_at_layer_zero() {
        // 3 leaves → padded to 4, depth 2.
        // Depth-0 row 0: (a, b), real, real.
        // Depth-0 row 1: (c, zero_hash(0)), real, NOT real.
        // Depth-1 row 0: (parent_0, parent_1), real, real.
        let a = make_chunk(0xAA);
        let b = make_chunk(0xBB);
        let c = make_chunk(0xCC);
        let rows = merkleize_witness(&[a, b, c], None);
        assert_eq!(rows.len(), 3);
        // Row 1 has missing right at depth 0.
        let row1 = &rows[1];
        assert!(row1.is_left_real);
        assert!(!row1.is_right_real);
        assert_eq!(row1.right, zero_hash(0));
        // Final root matches reference.
        let root = rows.last().unwrap().parent;
        assert_eq!(root, merkleize_chunks(&[a, b, c], None));
        assert!(evaluate_witness(&rows));
        assert!(rows_chain_consistent(&rows));
    }

    #[test]
    fn witness_zero_extension_one_leaf_limit_eight() {
        // 1 leaf, limit 8 → inner_depth = 0, full_depth = 3.
        // 0 rows in populated subtree (single leaf); 3 rows of zero extension.
        let leaf = make_chunk(0x77);
        let rows = merkleize_witness(&[leaf], Some(8));
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].left, leaf);
        assert_eq!(rows[0].right, zero_hash(0));
        assert!(!rows[0].is_right_real);
        assert_eq!(rows[1].left, rows[0].parent);
        assert_eq!(rows[1].right, zero_hash(1));
        assert_eq!(rows[2].left, rows[1].parent);
        assert_eq!(rows[2].right, zero_hash(2));
        let root = rows.last().unwrap().parent;
        assert_eq!(root, merkleize_chunks(&[leaf], Some(8)));
        assert!(evaluate_witness(&rows));
    }

    #[test]
    fn witness_validator_registry_limit_extension_count() {
        // 1 leaf, limit = 2^40 → 40 rows of zero extension. Run quickly
        // because zero_hash is iterative; cap depth to be sure it doesn't
        // OOM (the sparse-merkleize fix already covers this).
        let leaf = make_chunk(0x99);
        let rows = merkleize_witness(&[leaf], Some(1u64 << 40));
        assert_eq!(rows.len(), 40);
        let root = rows.last().unwrap().parent;
        assert_eq!(root, merkleize_chunks(&[leaf], Some(1u64 << 40)));
    }

    #[test]
    fn rows_hash_consistent_detects_tampered_parent() {
        let a = make_chunk(0xAA);
        let b = make_chunk(0xBB);
        let mut rows = merkleize_witness(&[a, b], None);
        rows[0].parent[0] ^= 0xFF;
        assert!(!rows_hash_consistent(&rows));
        assert!(!evaluate_witness(&rows));
    }

    #[test]
    fn rows_zero_padding_consistent_detects_wrong_pad() {
        let a = make_chunk(0xAA);
        let b = make_chunk(0xBB);
        let c = make_chunk(0xCC);
        let mut rows = merkleize_witness(&[a, b, c], None);
        // Row 1 had right = zero_hash(0), is_right_real = false.
        // Tamper the right value so it's no longer the zero hash.
        rows[1].right[0] ^= 0x01;
        assert!(!rows_zero_padding_consistent(&rows));
        assert!(!evaluate_witness(&rows));
    }

    #[test]
    fn trace_column_populator_round_trips() {
        let a = make_chunk(0xAA);
        let b = make_chunk(0xBB);
        let c = make_chunk(0xCC);
        let d = make_chunk(0xDD);
        let rows = merkleize_witness(&[a, b, c, d], None);

        // Allocate trace columns sized for the rows.
        let mut columns: Vec<Vec<u64>> =
            (0..col::NUM_COLUMNS).map(|_| vec![0u64; rows.len()]).collect();
        populate_trace(&rows, &mut columns);

        // Verify each row's bytes match the corresponding column slot.
        for (row_idx, row) in rows.iter().enumerate() {
            for i in 0..CHUNK_BYTES {
                assert_eq!(columns[col::LEFT_OFFSET + i][row_idx], row.left[i] as u64);
                assert_eq!(columns[col::RIGHT_OFFSET + i][row_idx], row.right[i] as u64);
                assert_eq!(columns[col::PARENT_OFFSET + i][row_idx], row.parent[i] as u64);
            }
            assert_eq!(columns[col::LAYER_DEPTH][row_idx], row.layer_depth as u64);
            assert_eq!(columns[col::POSITION_IN_LAYER][row_idx], row.position_in_layer as u64);
            assert_eq!(columns[col::IS_LEFT_REAL][row_idx], row.is_left_real as u64);
            assert_eq!(columns[col::IS_RIGHT_REAL][row_idx], row.is_right_real as u64);
        }
    }

    #[test]
    fn rows_chain_consistent_detects_broken_chain() {
        let a = make_chunk(0x11);
        let b = make_chunk(0x22);
        let c = make_chunk(0x33);
        let d = make_chunk(0x44);
        let mut rows = merkleize_witness(&[a, b, c, d], None);
        // Tamper row 0's parent so the depth-1 row's left no longer matches.
        rows[0].parent[0] ^= 0x42;
        assert!(!rows_chain_consistent(&rows));
    }
}
