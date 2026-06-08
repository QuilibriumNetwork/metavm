//! Host-side cross-AIR consistency checkers.
//!
//! Each wired AIR proves its own slice of a computation; cross-AIR
//! consistency is the missing link that ties them together. For
//! example the SSZ AIR proves "the merkleization tree structure is
//! consistent" but does NOT prove "every parent hash equals
//! sha256(left || right)" — that hash equality is delegated externally
//! today.
//!
//! This module provides **host-side** checkers that verify cross-AIR
//! invariants without cryptographic linkage. They serve two purposes:
//!
//!   1. **Documentation of the data flow**: the function signature is the
//!      contract a future LogUp-style cross-AIR lookup will enforce
//!      cryptographically. The cryptographic version produces exactly
//!      the same accept/reject decision as these host-side functions.
//!   2. **Test affordance**: integration tests can use these to assert
//!      that two trace witnesses really do compose correctly before
//!      attempting the full proof pipeline. Cheap end-to-end smoke check.
//!
//! # Out of scope
//!
//! Cryptographic LogUp/Plookup-style cross-table arguments — those
//! require coordinated Fiat-Shamir randomness across both proofs
//! (γ challenge derived from BOTH commitments) plus auxiliary witness
//! columns in both AIRs. The host-side checks here are the data-flow
//! contract a cryptographic version would mirror.

use crate::keccak::{keccak256, keccak_witness, HashTrace as KeccakHashTrace};
use crate::mpt_air::InclusionRow;
use crate::sha256::{sha256, HashTrace};
use crate::ssz_air::MerkleizeRow;

/// Per-row result of [`check_ssz_sha256_consistency`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SszRowCheck {
    /// `parent == sha256(left || right)` AND a matching SHA-256 trace
    /// (input == left||right, digest == parent) was found.
    OkLinked,
    /// `parent == sha256(left || right)` but no matching SHA-256 trace
    /// was supplied. The hash check passes natively but the cross-AIR
    /// linkage to a SHA-256 proof is incomplete.
    OkNoTrace,
    /// `parent != sha256(left || right)`. The SSZ row's claim is wrong.
    HashMismatch,
}

/// Host-side cross-AIR consistency check: SSZ ↔ SHA-256.
///
/// For each SSZ merkleize row, verify:
///
///   1. `parent == sha256(left || right)` (computed natively via
///      `crate::sha256::sha256`).
///   2. Some supplied [`HashTrace`] in `sha256_traces` covers the
///      same input/output pair (`input_len == 64`, raw input bytes ==
///      `left || right`, `digest == parent`).
///
/// Returns one [`SszRowCheck`] per SSZ row, in order. The cryptographic
/// LogUp version will enforce condition 1 by checking each
/// (left||right, parent) tuple appears in the SHA-256 trace's
/// (input, output) table — i.e. the same accept rule as
/// `OkLinked`.
///
/// Empty `sha256_traces` is allowed; every row will then report
/// `OkNoTrace` (assuming the native hash check passes) or
/// `HashMismatch`.
pub fn check_ssz_sha256_consistency(
    ssz_rows: &[MerkleizeRow],
    sha256_traces: &[HashTrace],
) -> Vec<SszRowCheck> {
    ssz_rows
        .iter()
        .map(|row| {
            let mut input = [0u8; 64];
            input[..32].copy_from_slice(&row.left);
            input[32..].copy_from_slice(&row.right);
            let computed = sha256(&input);
            if computed != row.parent {
                return SszRowCheck::HashMismatch;
            }
            let linked = sha256_traces.iter().any(|t| {
                if t.input_len != 64 || t.digest != row.parent {
                    return false;
                }
                // Reconstruct the 64-byte input from the trace's first
                // block's first 64 bytes (which equal the original input
                // when input_len == 64).
                t.blocks
                    .first()
                    .map(|b| b.block[..64] == input[..])
                    .unwrap_or(false)
            });
            if linked {
                SszRowCheck::OkLinked
            } else {
                SszRowCheck::OkNoTrace
            }
        })
        .collect()
}

/// Convenience: build the SHA-256 traces a future cross-AIR linkage
/// would need to fully cover an SSZ merkleize trace. Produces one
/// [`HashTrace`] per SSZ row covering the same `sha256(left||right)`
/// invocation. Useful for integration tests where you want
/// [`check_ssz_sha256_consistency`] to return `OkLinked` for every row.
pub fn sha256_traces_for_ssz(rows: &[MerkleizeRow]) -> Vec<HashTrace> {
    use crate::sha256::sha256_witness;
    rows.iter()
        .map(|row| {
            let mut input = [0u8; 64];
            input[..32].copy_from_slice(&row.left);
            input[32..].copy_from_slice(&row.right);
            sha256_witness(&input)
        })
        .collect()
}

/// Per-row result of [`check_mpt_keccak_consistency`]. Mirror of
/// [`SszRowCheck`] for the MPT ↔ Keccak pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MptRowCheck {
    /// `node_hash == keccak256(node_rlp)` AND a matching Keccak trace
    /// (input == node_rlp, digest == node_hash) was found.
    OkLinked,
    /// Native hash check passes but no matching Keccak trace was supplied.
    OkNoTrace,
    /// `node_hash != keccak256(node_rlp)`. The MPT row's claim is wrong.
    HashMismatch,
}

/// Host-side cross-AIR consistency check: MPT inclusion ↔ Keccak.
///
/// For each MPT inclusion row, verify:
///
///   1. `node_hash == keccak256(proof_nodes[i])` — i.e. the row's claimed
///      hash equals what `keccak256` natively produces on the supplied
///      RLP-encoded node bytes.
///   2. Some supplied Keccak [`KeccakHashTrace`] covers the same
///      input/output pair (`input_len == proof_nodes[i].len()`, raw
///      input bytes == `proof_nodes[i]`, `digest == node_hash`).
///
/// `proof_nodes.len()` MUST equal `rows.len()` (one RLP-encoded node per
/// inclusion row). Returns one [`MptRowCheck`] per row, in order.
///
/// The cryptographic LogUp version will enforce the same accept rule
/// against the Keccak AIR's (input, output) trace.
pub fn check_mpt_keccak_consistency(
    rows: &[InclusionRow],
    proof_nodes: &[Vec<u8>],
    keccak_traces: &[KeccakHashTrace],
) -> Vec<MptRowCheck> {
    assert_eq!(
        rows.len(),
        proof_nodes.len(),
        "proof_nodes must align with inclusion rows (one RLP blob per row)",
    );
    rows.iter()
        .zip(proof_nodes.iter())
        .map(|(row, rlp)| {
            let computed = keccak256(rlp);
            if computed != row.node_hash {
                return MptRowCheck::HashMismatch;
            }
            let linked = keccak_traces.iter().any(|t| {
                if t.input_len != rlp.len() || t.digest != row.node_hash {
                    return false;
                }
                // Reconstruct the original input from the trace: for
                // input_len <= 135 (single block), it's the first
                // `input_len` bytes of the first block. For longer
                // inputs, span across blocks. Most MPT nodes are <= 135
                // bytes (32-byte hashes + small RLP overhead per entry).
                if t.blocks.is_empty() {
                    return rlp.is_empty();
                }
                let mut reconstructed = Vec::with_capacity(rlp.len());
                let mut remaining = rlp.len();
                for b in &t.blocks {
                    let take = remaining.min(136);
                    reconstructed.extend_from_slice(&b.block[..take]);
                    remaining -= take;
                    if remaining == 0 {
                        break;
                    }
                }
                reconstructed == *rlp
            });
            if linked {
                MptRowCheck::OkLinked
            } else {
                MptRowCheck::OkNoTrace
            }
        })
        .collect()
}

/// Convenience: build Keccak traces covering each row's RLP-encoded
/// node. Useful for integration tests where you want
/// [`check_mpt_keccak_consistency`] to return `OkLinked` everywhere.
pub fn keccak_traces_for_mpt(proof_nodes: &[Vec<u8>]) -> Vec<KeccakHashTrace> {
    proof_nodes.iter().map(|rlp| keccak_witness(rlp)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssz::{Chunk, ZERO_CHUNK};
    use crate::ssz_air::merkleize_witness;

    fn chunk(byte: u8) -> Chunk {
        let mut c = [0u8; 32];
        c[0] = byte;
        c
    }

    #[test]
    fn empty_inputs_return_empty() {
        let result = check_ssz_sha256_consistency(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn valid_witness_with_matching_traces_links_every_row() {
        let rows = merkleize_witness(&[chunk(0x10), chunk(0x20), chunk(0x30), chunk(0x40)], None);
        assert!(!rows.is_empty(), "merkleize_witness must produce rows");

        let traces = sha256_traces_for_ssz(&rows);
        let result = check_ssz_sha256_consistency(&rows, &traces);

        assert_eq!(result.len(), rows.len());
        for (i, r) in result.iter().enumerate() {
            assert_eq!(*r, SszRowCheck::OkLinked, "row {} not OkLinked", i);
        }
    }

    #[test]
    fn valid_witness_no_traces_reports_ok_no_trace() {
        let rows = merkleize_witness(&[chunk(0xAA), chunk(0xBB)], None);
        let result = check_ssz_sha256_consistency(&rows, &[]);
        assert!(!result.is_empty());
        for r in &result {
            assert_eq!(*r, SszRowCheck::OkNoTrace);
        }
    }

    #[test]
    fn tampered_parent_reports_hash_mismatch() {
        let mut rows = merkleize_witness(&[chunk(0xAA), chunk(0xBB)], None);
        // Flip a byte in the first row's parent — the native hash check
        // must fire.
        rows[0].parent[0] ^= 0xFF;

        // Use traces matching the (correct) computation; the row's
        // parent no longer matches sha256(left||right) so the row is
        // HashMismatch regardless of trace presence.
        let traces = sha256_traces_for_ssz(&rows);
        let result = check_ssz_sha256_consistency(&rows, &traces);
        assert_eq!(result[0], SszRowCheck::HashMismatch);
    }

    #[test]
    fn extra_traces_dont_break_linkage() {
        let rows = merkleize_witness(&[chunk(0x10), chunk(0x20), chunk(0x30), chunk(0x40)], None);
        let mut traces = sha256_traces_for_ssz(&rows);
        // Throw in an unrelated trace — should be ignored by the
        // checker and not affect linkage decisions.
        traces.push(crate::sha256::sha256_witness(b"unrelated input"));

        let result = check_ssz_sha256_consistency(&rows, &traces);
        for r in &result {
            assert_eq!(*r, SszRowCheck::OkLinked);
        }
    }

    #[test]
    fn missing_some_traces_marks_those_rows_only() {
        let rows = merkleize_witness(&[chunk(0x10), chunk(0x20), chunk(0x30), chunk(0x40)], None);
        let traces = sha256_traces_for_ssz(&rows);
        // Drop the LAST trace — only the rows whose parent hash matches
        // a remaining trace should be OkLinked; the row whose hash only
        // appeared in the dropped trace should be OkNoTrace.
        let truncated: Vec<_> = traces[..traces.len() - 1].to_vec();
        let result = check_ssz_sha256_consistency(&rows, &truncated);

        let n_linked = result
            .iter()
            .filter(|r| matches!(r, SszRowCheck::OkLinked))
            .count();
        let n_no_trace = result
            .iter()
            .filter(|r| matches!(r, SszRowCheck::OkNoTrace))
            .count();
        assert_eq!(n_no_trace, 1, "exactly one row should be OkNoTrace");
        assert_eq!(n_linked, rows.len() - 1, "rest should be OkLinked");
        assert_eq!(
            n_linked + n_no_trace,
            rows.len(),
            "no row should be HashMismatch on a valid witness",
        );
    }

    #[test]
    fn zero_pad_rows_are_consistent() {
        // 3-leaf merkleization will have a zero-padded right child at
        // some layer. Make sure the consistency check handles those rows
        // (zero parent_hash etc) correctly.
        let _ = ZERO_CHUNK;  // touch the import
        let rows = merkleize_witness(&[chunk(0x01), chunk(0x02), chunk(0x03)], None);
        let traces = sha256_traces_for_ssz(&rows);
        let result = check_ssz_sha256_consistency(&rows, &traces);
        for r in &result {
            assert_eq!(*r, SszRowCheck::OkLinked);
        }
    }

    // ────────────── MPT ↔ Keccak ──────────────────────────────────────

    use crate::mpt::single_leaf_trie;
    use crate::mpt_air::single_leaf_inclusion_witness;

    #[test]
    fn mpt_empty_inputs_return_empty() {
        let result = check_mpt_keccak_consistency(&[], &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn mpt_valid_inclusion_with_matching_traces_links_every_row() {
        let key = b"\x01\x02\x03\x04";
        let value = b"hello";
        let (_, proof_nodes) = single_leaf_trie(key, value);
        let rows = single_leaf_inclusion_witness(key, value);
        assert_eq!(rows.len(), proof_nodes.len(), "rows must align with proof");

        let traces = keccak_traces_for_mpt(&proof_nodes);
        let result = check_mpt_keccak_consistency(&rows, &proof_nodes, &traces);

        assert!(!result.is_empty());
        for r in &result {
            assert_eq!(*r, MptRowCheck::OkLinked);
        }
    }

    #[test]
    fn mpt_valid_inclusion_no_traces_reports_ok_no_trace() {
        let (_, proof_nodes) = single_leaf_trie(b"\xab\xcd", b"x");
        let rows = single_leaf_inclusion_witness(b"\xab\xcd", b"x");
        let result = check_mpt_keccak_consistency(&rows, &proof_nodes, &[]);
        for r in &result {
            assert_eq!(*r, MptRowCheck::OkNoTrace);
        }
    }

    #[test]
    fn mpt_tampered_node_hash_reports_hash_mismatch() {
        let (_, proof_nodes) = single_leaf_trie(b"\xab\xcd", b"x");
        let mut rows = single_leaf_inclusion_witness(b"\xab\xcd", b"x");
        rows[0].node_hash[0] ^= 0xFF;

        let traces = keccak_traces_for_mpt(&proof_nodes);
        let result = check_mpt_keccak_consistency(&rows, &proof_nodes, &traces);
        assert_eq!(result[0], MptRowCheck::HashMismatch);
    }

    #[test]
    fn mpt_extra_traces_dont_break_linkage() {
        let (_, proof_nodes) = single_leaf_trie(b"\xff", b"yy");
        let rows = single_leaf_inclusion_witness(b"\xff", b"yy");
        let mut traces = keccak_traces_for_mpt(&proof_nodes);
        traces.push(crate::keccak::keccak_witness(b"unrelated input"));

        let result = check_mpt_keccak_consistency(&rows, &proof_nodes, &traces);
        for r in &result {
            assert_eq!(*r, MptRowCheck::OkLinked);
        }
    }
}
