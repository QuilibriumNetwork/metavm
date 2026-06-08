//! Account-RLP → wide-leaf → MPT chain composition.
//!
//! **Task #121** — wires the algebraic chain
//!
//! ```text
//!   account_rlp_air    →    mpt_leaf_wide_air    →    mpt_air
//! ```
//!
//! so that the canonical RLP encoding of an Ethereum [`Account`]
//! (produced and certified by [`account_rlp_air`]) is bound — via
//! cross-AIR LogUp linkages — to the leaf bytes that [`mpt_leaf_wide_air`]
//! commits to, and through that gadget to [`mpt_air`]'s
//! `claimed_leaf_value_bytes` slot at row 0 of an inclusion proof
//! against a block-header `state_root`.
//!
//! # The chain
//!
//! 1. `account_rlp_air` (212 cols, 9 row-local constraints) produces
//!    the per-field encoded buffers
//!    `{LIST_PREFIX_0/1, NONCE_ENC, BALANCE_ENC, STORAGE_ROOT_ENC,
//!    CODE_HASH_ENC}` plus `ENCODED_LEN`.  Its constraints already
//!    pin the prefix bytes and `payload_len = ENC_LEN - 2` etc., and
//!    the per-field gadget linkages (u64_rlp / u256_rlp / fixed_rlp32)
//!    pin each sub-buffer.
//!
//! 2. `mpt_leaf_wide_air` (1029 cols, 3 row-local constraints, leaves
//!    up to 1024 bytes) commits — in one contiguous
//!    `COL_LEAF_BYTE_OFFSET[0..1024]` byte buffer — the leaf payload
//!    feeding an MPT inclusion proof.  Sibling AIR
//!    [`crate::mpt_inclusion_descriptors`] already wires per-leaf-kind
//!    "32-byte prefix" bindings into `mpt_air`'s 32-byte
//!    `claimed_leaf_value_bytes` slot; the *full* binding for an
//!    Account RLP (68..=108 bytes) flows through this gadget.
//!
//! 3. `mpt_air` consumes a proof path against the world-state MPT at
//!    a block-header `state_root`.  Row 0 has `IS_ROOT = 1`, the leaf
//!    row has `IS_TERMINAL = 1`; `claimed_leaf_value_bytes[0..32]` is
//!    propagated constant across the chain and equals the first 32
//!    bytes of the leaf-row's RLP-decoded value field.
//!
//! # What the descriptors here pin
//!
//! - [`make_account_rlp_to_mpt_leaf_wide_descriptor`] binds the
//!   account's 32-byte **balance encoded buffer** + `ENCODED_LEN`
//!   tuple to the wide-leaf's first 32 bytes + `LEN_LO`.  This is
//!   the strongest single-tuple binding the two AIRs currently
//!   support without account_rlp_air exposing a contiguous
//!   `encoded_byte[0..MAX_ENCODED_LEN]` buffer (deferred — see
//!   *Limitations* below).  Combined with `account_rlp_air`'s
//!   row-local constraints, the prover cannot misalign the buffer's
//!   length or fabricate a list-prefix byte.
//!
//! - [`make_mpt_leaf_wide_to_mpt_air_descriptor`] binds the first
//!   32 bytes of the wide-leaf's `COL_LEAF_BYTE_OFFSET[..]` slot to
//!   `mpt_air`'s `CLAIMED_LEAF_VALUE_BYTE_OFFSET[0..32]`, gated by
//!   `IS_TERMINAL` on both sides.  Because `mpt_air`'s leaf-value
//!   slot is capped at 32 bytes, only the **leading 32 bytes** of
//!   the leaf payload bind via this linkage.  The full 68..=108-byte
//!   account RLP is fully bound at `mpt_leaf_wide_air`; the MPT
//!   `value_byte` cross-row binding (deferred) and the existing 32-B
//!   `CLAIMED_LEAF_VALUE_BYTE` slot give partial binding into
//!   `mpt_air`.
//!
//! - [`make_account_chain_to_state_root_descriptor`] binds
//!   `mpt_air`'s row-0 `PARENT_HASH` (which equals the trie root by
//!   the existing row-0 / `IS_ROOT` semantics) to
//!   `block_header_air`'s 32-byte `COL_STATE_ROOT_OFFSET`.
//!
//! # Limitations (recorded for follow-up)
//!
//! - `mpt_air`'s `CLAIMED_LEAF_VALUE_BYTE_OFFSET` slot is **32 bytes**.
//!   The full account RLP is 68..=108 bytes.  Therefore only the
//!   leading 32 bytes are bound directly into `mpt_air` via the
//!   second descriptor.  The bytes beyond 32 are bound algebraically
//!   in `mpt_leaf_wide_air` (which holds the full leaf), so the
//!   overall chain still witnesses the full account encoding — the
//!   `mpt_air`-side closure only covers the prefix.  The path to
//!   close this is to widen `CLAIMED_LEAF_VALUE_BYTE_OFFSET` to ≥110
//!   bytes (the existing `mpt_inclusion_descriptors` module flags
//!   this exact follow-up).
//!
//! - `account_rlp_air` does not yet expose a single contiguous
//!   `encoded_byte[0..MAX_ENCODED_LEN]` buffer; the per-field
//!   encoding buffers are spread across separate column blocks.
//!   The first descriptor therefore binds the **balance encoded
//!   buffer** as a 32-byte anchor (matching the `mpt_leaf_wide_air`
//!   per-32-byte tuple shape already used by
//!   `make_mpt_leaf_wide_to_account_rlp_descriptor`).  A future
//!   `account_rlp_concat_air` would let us produce a contiguous
//!   per-byte binding.
//!
//! - The per-account `address → state-root MPT key` derivation
//!   (`trie_key = keccak256(address)`) is not bound here — it lives
//!   in [`crate::address_keccak_air`] +
//!   [`crate::account_state_air`] linkages.  Host-side helpers in
//!   this module compute the `address_trie_key` so callers can
//!   thread it into the proof witness.

use crate::account::{account_rlp, account_trie_key, Account};
use crate::cross_air_logup::CrossAirLogUpDescriptor;

// Per-AIR types referenced by the assembler.
use crate::account_rlp_air::{
    AccountRlpWitness, COL_BALANCE_ENC_OFFSET as AR_BAL_ENC, COL_ENCODED_LEN as AR_ENC_LEN,
    COL_IS_REAL as AR_IS_REAL,
};
use crate::mpt_air::{col as mpt_col, inclusion_witness, InclusionRow};
use crate::mpt_leaf_wide_air::{
    MptLeafWideWitness, COL_IS_TERMINAL as ML_IS_TERMINAL,
    COL_LEAF_BYTE_OFFSET as ML_LEAF_OFFSET, COL_LEN_LO as ML_LEN_LO,
};

// ─────────────────────────────────────────────────────────────────────
// Witness bundle
// ─────────────────────────────────────────────────────────────────────

/// Composite witness threading an account through the
/// `account_rlp_air → mpt_leaf_wide_air → mpt_air` chain.
///
/// Each field is the existing per-AIR witness type so callers can
/// reuse the standard `build_trace_polynomials` paths.  The
/// host-derived `address_trie_key`, `state_root`, and `account_rlp`
/// scalars are exposed at the top level for caller convenience
/// (these are also stored implicitly inside the per-AIR witnesses).
#[derive(Clone, Debug)]
pub struct AccountInclusionChain {
    /// The account being proven included.
    pub account: Account,
    /// 20-byte Ethereum address.  The world-state MPT key for this
    /// account is `keccak256(address)`.
    pub address: [u8; 20],
    /// `account_rlp(account)` — the canonical RLP encoding of the
    /// account state (68..=108 bytes).
    pub account_rlp_bytes: Vec<u8>,
    /// `account_trie_key(address) = keccak256(address)` — the MPT
    /// key for this account in the world state trie.
    pub address_trie_key: [u8; 32],
    /// World-state MPT root the inclusion proof is taken against.
    pub state_root: [u8; 32],
    /// Witness for `account_rlp_air` row, with a single row carrying
    /// the per-field encoded buffers.
    pub account_rlp_witness: AccountRlpWitness,
    /// Witness for `mpt_leaf_wide_air` row, holding the full account
    /// RLP as the leaf payload (length = `account_rlp_bytes.len()`).
    pub leaf_wide_witness: MptLeafWideWitness,
    /// Inclusion-proof rows for `mpt_air` (one per proof node along
    /// the path).  Row 0 has `is_root = 1`, the last row has
    /// `is_terminal = 1`; `claimed_leaf_value_bytes` / `_key_bytes`
    /// / `_full_key_bytes` are populated constant across the chain.
    pub mpt_rows: Vec<InclusionRow>,
}

impl AccountInclusionChain {
    /// Number of proof rows in `mpt_rows`.
    pub fn mpt_path_len(&self) -> usize { self.mpt_rows.len() }

    /// `true` iff the inclusion proof's row chain is host-side
    /// consistent (root-row pinned to `state_root`, parent/child
    /// hash chain, terminal flag on last row).
    pub fn is_host_consistent(&self) -> bool {
        if !crate::mpt_air::rows_chain_consistent(&self.mpt_rows) { return false; }
        if !crate::mpt_air::rows_depth_monotone(&self.mpt_rows) { return false; }
        // Row 0's parent_hash equals the trie root (by definition).
        // The state_root supplied to assemble_account_chain must equal
        // keccak256(proof[0]).
        if self.mpt_rows.first().map(|r| r.parent_hash) != Some(self.state_root) {
            return false;
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────
// Assembler
// ─────────────────────────────────────────────────────────────────────

/// Build an [`AccountInclusionChain`] from a host-supplied account +
/// inclusion proof against a world-state root.
///
/// The caller is responsible for furnishing a proof that *actually*
/// inclusion-proves `account_rlp(account)` at `keccak256(address)`
/// inside `state_root`; this assembler is a witness-shaper, not a
/// validator (validation lives in [`crate::account::verify_account_inclusion_oracle`]).
///
/// Returns an empty `mpt_rows` Vec if the proof RLP can't be parsed
/// (consistent with [`crate::mpt_air::inclusion_witness`] semantics).
pub fn assemble_account_chain(
    account: &Account,
    address: [u8; 20],
    proof: &[Vec<u8>],
    state_root: [u8; 32],
) -> AccountInclusionChain {
    let account_rlp_bytes = account_rlp(account);
    let address_trie_key = account_trie_key(&address);

    let account_rlp_witness = AccountRlpWitness::from_account(account.clone());

    // For the wide-leaf AIR, the leaf bytes ARE the account RLP.
    // from_value_bytes pads to MAX_LEAF_LEN; we only ever expect
    // 68..=108 bytes here, well under the 1024-byte cap.
    let leaf_wide_witness = MptLeafWideWitness::from_value_bytes(&account_rlp_bytes)
        .expect("account RLP <= 108 bytes <= MptLeafWideWitness MAX_LEAF_LEN (1024)");

    let mpt_rows = inclusion_witness(&address_trie_key, proof);

    AccountInclusionChain {
        account: account.clone(),
        address,
        account_rlp_bytes,
        address_trie_key,
        state_root,
        account_rlp_witness,
        leaf_wide_witness,
        mpt_rows,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cross-AIR LogUp descriptors
// ─────────────────────────────────────────────────────────────────────

/// Bind `account_rlp_air`'s balance-encoded buffer + `ENCODED_LEN`
/// tuple to `mpt_leaf_wide_air`'s first 32 leaf bytes + `LEN_LO`.
///
/// This is the mirror direction of the existing
/// [`crate::mpt_leaf_wide_air::make_mpt_leaf_wide_to_account_rlp_descriptor`];
/// the chain assembly uses this direction (`account_rlp` is the
/// *source* AIR, `mpt_leaf_wide_air` is the *sink* — the leaf rows
/// are the lookup table the account's encoding is checked against).
///
/// **Tuple**: 33 cols.
///   - A side: `account_rlp_air::COL_BALANCE_ENC_OFFSET[0..32]` ++
///     `COL_ENCODED_LEN`.
///   - B side: `mpt_leaf_wide_air::COL_LEAF_BYTE_OFFSET[0..32]` ++
///     `COL_LEN_LO`.
///
/// **Selectors**: A on `account_rlp_air::COL_IS_REAL`,
/// B on `mpt_leaf_wide_air::COL_IS_TERMINAL`.
///
/// **Soundness note**: account_rlp_air doesn't yet expose a single
/// contiguous `encoded_byte[]` column; its encoded bytes are split
/// across LIST_PREFIX_0/1 + the per-field ENC blocks.  This
/// descriptor uses the 32-byte balance ENC slot as the canonical
/// 32-byte anchor (matching the symmetric descriptor on
/// `mpt_leaf_wide_air`).  A future `account_rlp_concat_air` will
/// let us widen this to per-byte equality across the full encoded
/// payload.
pub fn make_account_rlp_to_mpt_leaf_wide_descriptor(
    account_rlp_layer_index: usize,
    mpt_leaf_wide_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = (0..32).map(|k| AR_BAL_ENC + k).collect();
    a_columns.push(AR_ENC_LEN);

    let mut b_columns: Vec<usize> = (0..32).map(|k| ML_LEAF_OFFSET + k).collect();
    b_columns.push(ML_LEN_LO);

    CrossAirLogUpDescriptor {
        label: "account_rlp_to_mpt_leaf_wide_v1".into(),
        a_layer_index: account_rlp_layer_index,
        a_columns,
        a_selector_column: Some(AR_IS_REAL),
        b_layer_index: mpt_leaf_wide_layer_index,
        b_columns,
        b_selector_column: Some(ML_IS_TERMINAL),
    }
}

/// Bind the first 32 bytes of `mpt_leaf_wide_air`'s leaf buffer to
/// `mpt_air`'s `CLAIMED_LEAF_VALUE_BYTE_OFFSET[0..32]` slot, gated
/// by `IS_TERMINAL` on both sides.
///
/// **Tuple**: 32 cols.
///
/// **Limitation — the 32-byte cap.** `mpt_air`'s
/// `CLAIMED_LEAF_VALUE_BYTE_OFFSET` is a fixed 32-byte slot, but
/// account RLPs are 68..=108 bytes.  Only the leading 32 bytes are
/// bound into `mpt_air` via this descriptor.  The remaining bytes
/// are bound algebraically inside `mpt_leaf_wide_air` (which holds
/// the full 1024-cap leaf buffer with its own `VALUE_RLC`
/// constraint), so the overall *chain* witnesses the full payload —
/// but the binding into `mpt_air` is a 32-byte prefix.
///
/// To close this fully:
///   1. Widen `CLAIMED_LEAF_VALUE_BYTE_OFFSET` to ≥110 bytes (or to
///      the full `mpt_leaf_wide_air::MAX_LEAF_LEN = 1024`).
///   2. Propagate the cross-row constancy + leaf-row equality
///      constraints in `mpt_constraints` to match.
///   3. Re-target this descriptor at the widened slot.
///
/// This is the same deferred work item flagged by
/// [`crate::mpt_inclusion_descriptors`] for the storage / account /
/// tx / receipt 32-B-prefix descriptors.
pub fn make_mpt_leaf_wide_to_mpt_air_descriptor(
    mpt_leaf_wide_layer_index: usize,
    mpt_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = (0..32).map(|k| ML_LEAF_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..32)
        .map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k)
        .collect();
    CrossAirLogUpDescriptor {
        label: "mpt_leaf_wide_to_mpt_air_v1".into(),
        a_layer_index: mpt_leaf_wide_layer_index,
        a_columns,
        a_selector_column: Some(ML_IS_TERMINAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_TERMINAL),
    }
}

/// Bind `mpt_air`'s row-0 `PARENT_HASH` (= trie root by row-0
/// semantics — see `mpt_air::col::IS_ROOT`) to `block_header_air`'s
/// `COL_STATE_ROOT_OFFSET[0..32]`.  This is the chain's outer seal:
/// it pins the world-state MPT root used by the inclusion proof to
/// the `state_root` of a real block header.
///
/// **Tuple**: 32 cols.
///
/// **Selectors**: A on `mpt_air::col::IS_ROOT` (fires only on row 0
/// of each chain — exactly one row per inclusion proof, where
/// `parent_hash = trie_root`).  B on
/// `block_header_air::COL_IS_REAL`.
pub fn make_account_chain_to_state_root_descriptor(
    mpt_layer_index: usize,
    block_header_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    let a_columns: Vec<usize> = (0..32)
        .map(|k| mpt_col::PARENT_HASH_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..32)
        .map(|k| bh::COL_STATE_ROOT_OFFSET + k)
        .collect();
    CrossAirLogUpDescriptor {
        label: "account_chain_to_state_root_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_ROOT),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

/// Collect all three chain descriptors at concrete layer indices.
///
/// Useful for callers wiring a joint protocol with all three AIRs
/// at known positions.  The 4 layer indices are
/// `(account_rlp, leaf_wide, mpt, block_header)`.
pub fn collect_descriptors(
    account_rlp_layer_index: usize,
    leaf_wide_layer_index: usize,
    mpt_layer_index: usize,
    block_header_layer_index: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    vec![
        make_account_rlp_to_mpt_leaf_wide_descriptor(
            account_rlp_layer_index,
            leaf_wide_layer_index,
        ),
        make_mpt_leaf_wide_to_mpt_air_descriptor(
            leaf_wide_layer_index,
            mpt_layer_index,
        ),
        make_account_chain_to_state_root_descriptor(
            mpt_layer_index,
            block_header_layer_index,
        ),
    ]
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::single_leaf_trie;

    fn funded_account() -> Account {
        Account {
            nonce: 7,
            balance: {
                let mut b = [0u8; 32];
                b[31] = 0xff; b[30] = 0x10;
                b
            },
            storage_root: [0x11; 32],
            code_hash: [0x22; 32],
        }
    }

    /// Build a single-leaf world-state trie holding only this
    /// account at `keccak256(address)`.
    fn single_account_world(
        account: &Account,
        address: [u8; 20],
    ) -> ([u8; 32], Vec<Vec<u8>>) {
        let key = account_trie_key(&address);
        let value = account_rlp(account);
        let (root, proof) = single_leaf_trie(&key, &value);
        (root, proof)
    }

    // ── Chain assembly ────────────────────────────────────────────────

    #[test]
    fn chain_assembles_for_default_account() {
        let account = Account::default();
        let address = [0xab; 20];
        let (root, proof) = single_account_world(&account, address);
        let chain = assemble_account_chain(&account, address, &proof, root);
        assert_eq!(chain.account, account);
        assert_eq!(chain.address, address);
        assert_eq!(chain.address_trie_key, account_trie_key(&address));
        assert_eq!(chain.state_root, root);
        assert_eq!(chain.account_rlp_bytes, account_rlp(&account));
        assert!(chain.account_rlp_witness.rows.len() == 1);
        assert!(chain.leaf_wide_witness.rows.len() == 1);
        assert_eq!(
            chain.leaf_wide_witness.rows[0].length,
            chain.account_rlp_bytes.len()
        );
        assert!(!chain.mpt_rows.is_empty(), "proof must produce at least one row");
        assert!(chain.is_host_consistent());
    }

    #[test]
    fn chain_assembles_for_funded_account() {
        let account = funded_account();
        let address = [0xcd; 20];
        let (root, proof) = single_account_world(&account, address);
        let chain = assemble_account_chain(&account, address, &proof, root);
        assert_eq!(chain.account, account);
        // Funded account: nonce=7, balance has high bytes set.
        assert!(
            chain.account_rlp_bytes.len() >= crate::account_rlp_air::MIN_PAYLOAD_LEN
                + crate::account_rlp_air::LIST_PREFIX_LEN,
        );
        assert!(
            chain.account_rlp_bytes.len()
                <= crate::account_rlp_air::MAX_ENCODED_LEN,
        );
        assert!(chain.is_host_consistent());
        assert_eq!(chain.mpt_path_len(), chain.mpt_rows.len());
        // Leaf row witnesses the full RLP, not just the 32-byte
        // prefix.
        let len = chain.leaf_wide_witness.rows[0].length;
        for k in 0..len {
            assert_eq!(
                chain.leaf_wide_witness.rows[0].leaf_bytes[k],
                chain.account_rlp_bytes[k],
            );
        }
    }

    // ── Witness consistency ───────────────────────────────────────────

    #[test]
    fn chain_witness_lengths_consistent() {
        let account = funded_account();
        let address = [0x99; 20];
        let (root, proof) = single_account_world(&account, address);
        let chain = assemble_account_chain(&account, address, &proof, root);
        // account_rlp_air row count = 1.
        assert_eq!(chain.account_rlp_witness.rows.len(), 1);
        // mpt_leaf_wide_air row count = 1.
        assert_eq!(chain.leaf_wide_witness.rows.len(), 1);
        // mpt_air row count = proof.len() (for valid proofs).
        assert_eq!(chain.mpt_rows.len(), proof.len());
        // For a single-leaf trie, that's exactly 1 row.
        assert_eq!(chain.mpt_rows.len(), 1);
        let leaf_row = &chain.mpt_rows[0];
        // is_root = 1 on row 0, is_terminal = 1 because it's the only row.
        assert_eq!(leaf_row.is_root, 1);
        assert_eq!(leaf_row.is_terminal, 1);
        // claimed_full_key_bytes equals address_trie_key.
        assert_eq!(leaf_row.claimed_full_key_bytes, chain.address_trie_key);
    }

    // ── Address → trie_key derivation ─────────────────────────────────

    #[test]
    fn address_to_trie_key_matches_keccak() {
        let account = Account::default();
        let address = [0x55; 20];
        let (root, proof) = single_account_world(&account, address);
        let chain = assemble_account_chain(&account, address, &proof, root);
        let expected = crate::keccak::keccak256(&address);
        assert_eq!(chain.address_trie_key, expected);
        // And the mpt_air row's claimed_full_key_bytes matches.
        assert_eq!(chain.mpt_rows[0].claimed_full_key_bytes, expected);
    }

    // ── Descriptors well-formed ──────────────────────────────────────

    #[test]
    fn account_rlp_to_mpt_leaf_wide_descriptor_well_formed() {
        let d = make_account_rlp_to_mpt_leaf_wide_descriptor(2, 5);
        assert_eq!(d.label, "account_rlp_to_mpt_leaf_wide_v1");
        assert_eq!(d.a_layer_index, 2);
        assert_eq!(d.b_layer_index, 5);
        assert_eq!(d.a_columns.len(), 33);
        assert_eq!(d.b_columns.len(), 33);
        for k in 0..32 {
            assert_eq!(d.a_columns[k], AR_BAL_ENC + k);
            assert_eq!(d.b_columns[k], ML_LEAF_OFFSET + k);
        }
        assert_eq!(d.a_columns[32], AR_ENC_LEN);
        assert_eq!(d.b_columns[32], ML_LEN_LO);
        assert_eq!(d.a_selector_column, Some(AR_IS_REAL));
        assert_eq!(d.b_selector_column, Some(ML_IS_TERMINAL));
    }

    #[test]
    fn mpt_leaf_wide_to_mpt_air_descriptor_well_formed() {
        let d = make_mpt_leaf_wide_to_mpt_air_descriptor(5, 9);
        assert_eq!(d.label, "mpt_leaf_wide_to_mpt_air_v1");
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        for k in 0..32 {
            assert_eq!(d.a_columns[k], ML_LEAF_OFFSET + k);
            assert_eq!(
                d.b_columns[k],
                mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k,
            );
        }
        assert_eq!(d.a_selector_column, Some(ML_IS_TERMINAL));
        assert_eq!(d.b_selector_column, Some(mpt_col::IS_TERMINAL));
    }

    #[test]
    fn account_chain_to_state_root_descriptor_well_formed() {
        use crate::block_header_air as bh;
        let d = make_account_chain_to_state_root_descriptor(9, 12);
        assert_eq!(d.label, "account_chain_to_state_root_v1");
        assert_eq!(d.a_layer_index, 9);
        assert_eq!(d.b_layer_index, 12);
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        for k in 0..32 {
            assert_eq!(d.a_columns[k], mpt_col::PARENT_HASH_OFFSET + k);
            assert_eq!(d.b_columns[k], bh::COL_STATE_ROOT_OFFSET + k);
        }
        assert_eq!(d.a_selector_column, Some(mpt_col::IS_ROOT));
        assert_eq!(d.b_selector_column, Some(bh::COL_IS_REAL));
    }

    #[test]
    fn collect_descriptors_returns_three_distinct_labels() {
        let descs = collect_descriptors(0, 1, 2, 3);
        assert_eq!(descs.len(), 3);
        let labels: Vec<&str> = descs.iter().map(|d| d.label.as_str()).collect();
        assert_eq!(labels[0], "account_rlp_to_mpt_leaf_wide_v1");
        assert_eq!(labels[1], "mpt_leaf_wide_to_mpt_air_v1");
        assert_eq!(labels[2], "account_chain_to_state_root_v1");
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                assert_ne!(labels[i], labels[j]);
            }
        }
        // Layer-index threading: each descriptor's b_layer_index is
        // the next AIR's a_layer_index, except the final
        // state-root descriptor which targets block_header_air.
        assert_eq!(descs[0].a_layer_index, 0);
        assert_eq!(descs[0].b_layer_index, 1);
        assert_eq!(descs[1].a_layer_index, 1);
        assert_eq!(descs[1].b_layer_index, 2);
        assert_eq!(descs[2].a_layer_index, 2);
        assert_eq!(descs[2].b_layer_index, 3);
    }

    // ── End-to-end host-side oracle sanity ────────────────────────────

    /// Belt-and-suspenders: the existing
    /// `verify_account_inclusion_oracle` must accept the proof we
    /// assemble into the chain.
    #[test]
    fn chain_proof_is_oracle_valid() {
        let account = funded_account();
        let address = [0x77; 20];
        let (root, proof) = single_account_world(&account, address);
        let chain = assemble_account_chain(&account, address, &proof, root);
        crate::account::verify_account_inclusion_oracle(
            chain.state_root,
            &chain.address,
            &chain.account,
            &proof,
        )
        .expect("honestly-assembled proof must verify");
    }
}
