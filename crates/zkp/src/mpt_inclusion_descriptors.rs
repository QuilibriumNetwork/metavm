//! Cross-AIR LogUp descriptors binding RLP-encoded values (storage
//! values, account state, transactions, receipts) to `mpt_air`'s
//! `claimed_leaf_value_bytes` columns.
//!
//! ## Background
//!
//! `mpt_air` exposes — on each row whose `IS_TERMINAL` selector fires
//! — a 32-byte slot `CLAIMED_LEAF_VALUE_BYTE_OFFSET[0..32]` carrying
//! the **RLP-encoded leaf value**. The companion algebraic constraint
//! enforces cross-row constancy of this column (so the row-0 view
//! equals the leaf-row view), and the leaf-row constraint binds it to
//! the actual RLP byte stream embedded in the leaf node's payload.
//!
//! Until this module landed, the binding from algebraic-AIR sources
//! (`storage_access_air::COL_VALUE_BE_OFFSET`,
//! `account_state_air` per-field columns, `u256_rlp_air::COL_ENCODED_OFFSET`,
//! `keccak_extract_wide::COL_INPUT_BYTE_OFFSET`) to that MPT leaf-value
//! column was *oracle-only* — host code computed the RLP and compared,
//! but no LogUp / permutation argument enforced the match in-circuit.
//!
//! ## What this module wires
//!
//! Four descriptor builders, one per leaf shape:
//!
//! | Builder | A side (source) | B side (sink) | Status |
//! |---------|-----------------|---------------|--------|
//! | `make_storage_value_to_mpt_leaf_descriptor` | `u256_rlp_air::COL_ENCODED_OFFSET[0..32]` | `mpt::CLAIMED_LEAF_VALUE_BYTE_OFFSET[0..32]` | **fully bound** — `u256_rlp_air` algebraically certifies the RLP encoding; LogUp binds it to MPT |
//! | `make_account_rlp_to_mpt_leaf_descriptor` | `account_state_air` per-field bytes (`COL_BALANCE_OFFSET[0..32]`) | `mpt::CLAIMED_LEAF_VALUE_BYTE_OFFSET[0..32]` | **partial** — binds 32-byte balance field as a proxy; full account-RLP-encoded leaf binding requires a dedicated `account_rlp_air` (deferred) |
//! | `make_tx_rlp_to_mpt_leaf_descriptor` | `keccak_extract_wide::COL_INPUT_BYTE_OFFSET[0..32]` | `mpt::CLAIMED_LEAF_VALUE_BYTE_OFFSET[0..32]` | **partial** — binds the first 32 bytes of the keccak-preimage view; full variable-length tx RLP (>32 B for non-trivial txs) requires a dedicated `tx_rlp_air` or extended leaf-value columns (deferred) |
//! | `make_receipt_rlp_to_mpt_leaf_descriptor` | `keccak_extract_wide::COL_INPUT_BYTE_OFFSET[0..32]` | `mpt::CLAIMED_LEAF_VALUE_BYTE_OFFSET[0..32]` | **partial** — same partial-prefix shape as tx; receipts can be hundreds of bytes due to logs/bloom (deferred) |
//!
//! ## Deferred work
//!
//! 1. Extend `mpt_air` with a longer `claimed_leaf_value_bytes` slot
//!    (current 32B is sufficient for storage but truncates account /
//!    tx / receipt RLPs). This requires widening the cross-row
//!    constancy + leaf-row equality constraints to match.
//! 2. Build dedicated algebraic AIRs for variable-length account /
//!    tx / receipt RLP encoding (analogous to the existing `u256_rlp_air`),
//!    then re-target these descriptors at their `COL_ENCODED_*` outputs.
//! 3. Soundness of these descriptors composes only when the A side AIR
//!    is itself fully bound — currently `account_state_air` lacks an
//!    RLP-encoding constraint, so the account descriptor only enforces
//!    "balance bytes equal leaf-value bytes", not "RLP-encoded account
//!    equals leaf-value bytes".

use crate::cross_air_logup::CrossAirLogUpDescriptor;

// ─────────────────────────────────────────────────────────────────────
// Storage value → MPT leaf value (FULLY BOUND)
// ─────────────────────────────────────────────────────────────────────

/// Bind `u256_rlp_air`'s 32-byte RLP-encoded value slot to `mpt_air`'s
/// `claimed_leaf_value_bytes` gated by `IS_TERMINAL`.
///
/// **Soundness**: `u256_rlp_air` algebraically certifies that
/// `COL_ENCODED_OFFSET[0..32]` is the canonical RLP encoding of the
/// row's `value_be`. Combined with `mpt_constraints`' cross-row
/// constancy + leaf-row equality on `claimed_leaf_value_bytes`, this
/// closes the storage-value-RLP ↔ leaf-row chain algebraically.
///
/// Note that `u256_rlp_air`'s encoded slot is 33 bytes wide (1-byte
/// short-string prefix + up to 32 value bytes). We bind the first 32
/// bytes here, matching the width of `claimed_leaf_value_bytes`. For
/// the common case (32-byte slot values), this captures the entire
/// `[0xa0, b_31, …, b_0]` encoding modulo the leading length byte.
/// The leading length byte is implicitly bound by the MPT leaf's RLP
/// header constraints in the row-RLP gadget AIRs.
pub fn make_storage_value_to_mpt_leaf_descriptor(
    u256_rlp_layer_index: usize,
    mpt_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;
    use crate::u256_rlp_air as urlp;
    let a_columns: Vec<usize> = (0..32).map(|k| urlp::COL_ENCODED_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..32).map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "storage_value_to_mpt_leaf_v1".into(),
        a_layer_index: u256_rlp_layer_index,
        a_columns,
        a_selector_column: Some(urlp::COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_TERMINAL),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Account state → MPT leaf value (PARTIAL)
// ─────────────────────────────────────────────────────────────────────

/// Bind `account_state_air`'s per-field balance bytes
/// (`COL_BALANCE_OFFSET[0..32]`) to `mpt_air`'s `claimed_leaf_value_bytes`
/// gated by `IS_TERMINAL`.
///
/// **Soundness state: PARTIAL.** This is a 32-byte width binding that
/// can carry one field at a time. The account RLP encoding
/// `[nonce, balance, storage_root, code_hash]` is ≥73 bytes — well
/// beyond the 32-byte `claimed_leaf_value_bytes` slot. We commit the
/// **balance field** here because it's already a contiguous 32-byte
/// big-endian sequence in the trace, making it the natural anchor;
/// other fields (nonce u64, storage_root 32B, code_hash 32B) need
/// their own widened-leaf-value bindings.
///
/// **Deferred** (the actual full account RLP ↔ leaf binding):
///   1. Extend `mpt_air`'s `claimed_leaf_value_bytes` to ≥110 bytes,
///      and propagate the cross-row constancy + leaf-row equality
///      constraints.
///   2. Build a dedicated `account_rlp_air` (analogous to
///      `u256_rlp_air`) that algebraically certifies the canonical
///      RLP encoding of `(nonce, balance, storage_root, code_hash)`.
///   3. Re-target this descriptor at the new gadget's
///      `COL_ENCODED_OFFSET`.
///
/// Until then, this descriptor pins one of the four leaf-value
/// constituents and serves as the composition primitive for the full
/// binding.
pub fn make_account_rlp_to_mpt_leaf_descriptor(
    account_state_layer_index: usize,
    mpt_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::account_state_air as as_air;
    use crate::mpt_air::col as mpt_col;
    let a_columns: Vec<usize> =
        (0..32).map(|k| as_air::COL_BALANCE_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..32).map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "account_rlp_to_mpt_leaf_v1".into(),
        a_layer_index: account_state_layer_index,
        a_columns,
        a_selector_column: Some(as_air::COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_TERMINAL),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Transaction RLP → MPT leaf value (PARTIAL — prefix-32)
// ─────────────────────────────────────────────────────────────────────

/// Bind the first 32 bytes of `keccak_extract_wide`'s
/// `COL_INPUT_BYTE_OFFSET` (used as the algebraic carrier of the
/// tx-RLP byte stream feeding the tx hash) to `mpt_air`'s
/// `claimed_leaf_value_bytes` gated by `IS_TERMINAL`.
///
/// **Soundness state: PARTIAL (prefix-32).** A legacy or EIP-1559
/// transaction's RLP is typically 100-1000+ bytes, so this
/// descriptor binds only the leading 32-byte prefix. Two MPT leaves
/// whose first 32 RLP bytes coincide but diverge in the suffix are
/// indistinguishable under this binding alone.
///
/// **Deferred** (the actual full tx RLP ↔ leaf binding):
///   1. Extend `mpt_air`'s `claimed_leaf_value_bytes` to ≥ MAX_TX_RLP
///      (e.g., 1024 bytes) plus a length column; propagate constancy
///      + leaf-row equality constraints.
///   2. Build a dedicated `tx_rlp_air` (analogous to `u256_rlp_air`)
///      that algebraically certifies the canonical RLP encoding of
///      `LegacyTx` / `Eip1559Tx` body fields.
///   3. Re-target this descriptor at the new gadget's `COL_ENCODED_*`
///      output, widened to the new MPT leaf-value width.
///
/// This descriptor uses `keccak_extract_wide` as the byte source
/// because the tx hash (= keccak256(tx_rlp)) is the natural existing
/// algebraic anchor: when the tx is the sole leaf of a single-leaf
/// trie, the trie root equals keccak256(rlp([key, tx_rlp_value])),
/// and the input bytes are already committed in
/// `COL_INPUT_BYTE_OFFSET`.
pub fn make_tx_rlp_to_mpt_leaf_descriptor(
    keccak_extract_wide_layer_index: usize,
    mpt_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::keccak_extract_wide as kew;
    use crate::mpt_air::col as mpt_col;
    let a_columns: Vec<usize> =
        (0..32).map(|k| kew::COL_INPUT_BYTE_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..32).map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "tx_rlp_to_mpt_leaf_v1".into(),
        a_layer_index: keccak_extract_wide_layer_index,
        a_columns,
        a_selector_column: Some(kew::COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_TERMINAL),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Receipt RLP → MPT leaf value (PARTIAL — prefix-32)
// ─────────────────────────────────────────────────────────────────────

/// Bind the first 32 bytes of `keccak_extract_wide`'s
/// `COL_INPUT_BYTE_OFFSET` (used as the algebraic carrier of the
/// receipt-RLP byte stream feeding the receipt hash) to `mpt_air`'s
/// `claimed_leaf_value_bytes` gated by `IS_TERMINAL`.
///
/// **Soundness state: PARTIAL (prefix-32).** Receipts can run into
/// hundreds of bytes once logs + logs-bloom (256 B) are factored in,
/// so this descriptor binds only the leading 32-byte prefix. Same
/// caveat as `make_tx_rlp_to_mpt_leaf_descriptor`.
///
/// **Deferred** (the actual full receipt RLP ↔ leaf binding):
///   1. Extend `mpt_air`'s `claimed_leaf_value_bytes` to ≥ MAX_RCPT_RLP;
///      propagate constancy + leaf-row equality.
///   2. Build a dedicated `receipt_rlp_air` (with a sub-AIR for the
///      logs-bloom 256-byte payload, possibly reusing
///      `rlp_logs_bloom_air`).
///   3. Re-target.
///
/// Distinct label from the tx descriptor so they can co-exist in the
/// same joint protocol with different B-side gating if/when the MPT
/// leaf rows learn to discriminate by trie kind.
pub fn make_receipt_rlp_to_mpt_leaf_descriptor(
    keccak_extract_wide_layer_index: usize,
    mpt_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::keccak_extract_wide as kew;
    use crate::mpt_air::col as mpt_col;
    let a_columns: Vec<usize> =
        (0..32).map(|k| kew::COL_INPUT_BYTE_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..32).map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "receipt_rlp_to_mpt_leaf_v1".into(),
        a_layer_index: keccak_extract_wide_layer_index,
        a_columns,
        a_selector_column: Some(kew::COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_TERMINAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account_state_air as as_air;
    use crate::keccak_extract_wide as kew;
    use crate::mpt_air::col as mpt_col;
    use crate::u256_rlp_air as urlp;

    #[test]
    fn storage_value_to_mpt_leaf_descriptor_well_formed() {
        let desc = make_storage_value_to_mpt_leaf_descriptor(0, 1);
        assert_eq!(desc.label, "storage_value_to_mpt_leaf_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 32);
        assert_eq!(desc.b_columns.len(), 32);
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], urlp::COL_ENCODED_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k,
            );
        }
        assert_eq!(desc.a_selector_column, Some(urlp::COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(mpt_col::IS_TERMINAL));
    }

    #[test]
    fn account_rlp_to_mpt_leaf_descriptor_well_formed() {
        let desc = make_account_rlp_to_mpt_leaf_descriptor(2, 3);
        assert_eq!(desc.label, "account_rlp_to_mpt_leaf_v1");
        assert_eq!(desc.a_layer_index, 2);
        assert_eq!(desc.b_layer_index, 3);
        assert_eq!(desc.a_columns.len(), 32);
        assert_eq!(desc.b_columns.len(), 32);
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], as_air::COL_BALANCE_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k,
            );
        }
        assert_eq!(desc.a_selector_column, Some(as_air::COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(mpt_col::IS_TERMINAL));
    }

    #[test]
    fn tx_rlp_to_mpt_leaf_descriptor_well_formed() {
        let desc = make_tx_rlp_to_mpt_leaf_descriptor(4, 5);
        assert_eq!(desc.label, "tx_rlp_to_mpt_leaf_v1");
        assert_eq!(desc.a_layer_index, 4);
        assert_eq!(desc.b_layer_index, 5);
        assert_eq!(desc.a_columns.len(), 32);
        assert_eq!(desc.b_columns.len(), 32);
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], kew::COL_INPUT_BYTE_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k,
            );
        }
        assert_eq!(desc.a_selector_column, Some(kew::COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(mpt_col::IS_TERMINAL));
    }

    #[test]
    fn receipt_rlp_to_mpt_leaf_descriptor_well_formed() {
        let desc = make_receipt_rlp_to_mpt_leaf_descriptor(6, 7);
        assert_eq!(desc.label, "receipt_rlp_to_mpt_leaf_v1");
        assert_eq!(desc.a_layer_index, 6);
        assert_eq!(desc.b_layer_index, 7);
        assert_eq!(desc.a_columns.len(), 32);
        assert_eq!(desc.b_columns.len(), 32);
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], kew::COL_INPUT_BYTE_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k,
            );
        }
        assert_eq!(desc.a_selector_column, Some(kew::COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(mpt_col::IS_TERMINAL));
    }

    /// All four descriptors must carry **distinct labels** so they can
    /// be added to the same joint LogUp protocol without collisions
    /// (the LogUp orchestrator keys closures by label).
    #[test]
    fn all_descriptor_labels_are_distinct() {
        let labels = [
            make_storage_value_to_mpt_leaf_descriptor(0, 1).label,
            make_account_rlp_to_mpt_leaf_descriptor(0, 1).label,
            make_tx_rlp_to_mpt_leaf_descriptor(0, 1).label,
            make_receipt_rlp_to_mpt_leaf_descriptor(0, 1).label,
        ];
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                assert_ne!(
                    labels[i], labels[j],
                    "descriptor labels {} and {} collide",
                    i, j
                );
            }
        }
    }

    /// All four descriptors share the same B-side AIR (mpt_air) and
    /// the same B-side selector (`IS_TERMINAL`), so the B-side column
    /// indices must fall within mpt_air's column range. Sanity-check
    /// that the chosen offsets don't accidentally collide with other
    /// mpt_air column ranges (a regression guard against future column
    /// renumbering).
    #[test]
    fn b_side_columns_target_mpt_leaf_value_range() {
        for desc in [
            make_storage_value_to_mpt_leaf_descriptor(0, 1),
            make_account_rlp_to_mpt_leaf_descriptor(0, 1),
            make_tx_rlp_to_mpt_leaf_descriptor(0, 1),
            make_receipt_rlp_to_mpt_leaf_descriptor(0, 1),
        ] {
            assert_eq!(desc.b_columns.len(), 32);
            for (k, &col) in desc.b_columns.iter().enumerate() {
                assert_eq!(col, mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k);
                // Must be strictly inside the leaf-value 32-byte slot.
                assert!(col >= mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET);
                assert!(col < mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + 32);
                // Must not overlap the leaf-key slot that immediately
                // follows.
                assert!(col < mpt_col::CLAIMED_LEAF_KEY_BYTE_OFFSET);
            }
            assert_eq!(desc.b_selector_column, Some(mpt_col::IS_TERMINAL));
        }
    }

    /// Column-count parity: every descriptor here is a 32-byte
    /// (32-column) binding. This is a hard invariant — the MPT
    /// leaf-value slot is currently 32 bytes wide, so any descriptor
    /// targeting it must produce a 32-column tuple.
    #[test]
    fn all_descriptors_have_32_column_tuples() {
        for desc in [
            make_storage_value_to_mpt_leaf_descriptor(0, 1),
            make_account_rlp_to_mpt_leaf_descriptor(0, 1),
            make_tx_rlp_to_mpt_leaf_descriptor(0, 1),
            make_receipt_rlp_to_mpt_leaf_descriptor(0, 1),
        ] {
            assert_eq!(desc.a_columns.len(), 32);
            assert_eq!(desc.b_columns.len(), 32);
            assert_eq!(desc.a_columns.len(), desc.b_columns.len());
        }
    }
}
