//! World-state proof bundle — host-side scaffolding for a full
//! transaction execution proof.
//!
//! Composes all four MPT inclusion oracles + block header binding
//! into a single `WorldProof` that proves, end-to-end and host-side:
//!
//! 1. **Block header hash**: `keccak256(rlp(block_header)) ==
//!    block_hash`.
//! 2. **Account inclusion**: the executing contract's account state
//!    `(nonce, balance, storage_root, code_hash)` is committed in
//!    `block_header.state_root` via the world MPT at
//!    `keccak256(address)`.
//! 3. **Storage inclusion**: each `(slot, value)` pair touched by an
//!    SLOAD/SSTORE is committed in the contract's `storage_root` via
//!    a storage MPT at `keccak256(slot_be)`.
//! 4. **Transaction inclusion**: the transaction is committed in
//!    `block_header.transactions_root` via the transactions MPT at
//!    `rlp(tx_index)`.
//! 5. **Receipt inclusion**: the execution receipt is committed in
//!    `block_header.receipts_root` via the receipts MPT at
//!    `rlp(tx_index)`.
//!
//! This module is the host-side oracle for the algebraic proof chain
//! that the various AIRs (storage_access_air, mpt_air, KeccakExtract,
//! block header AIR, transaction AIR, receipt AIR) will eventually
//! commit via cross-AIR LogUp linkages. Until those AIRs are wired,
//! this oracle validates the witness construction is correct.

use crate::account::{verify_account_inclusion_oracle, Account};
use crate::block_header::{block_header_hash, BlockHeader};
use crate::keccak::keccak256;
use crate::mpt::verify_mpt_inclusion;
use crate::receipt::{verify_receipt_inclusion_oracle, Receipt};
use crate::rlp::rlp_encode_u256;
use crate::transaction::{verify_transaction_inclusion_oracle, Transaction};

/// MPT inclusion proof bundle for one storage slot. The `slot` and
/// `value` are stored as 32 big-endian bytes (the storage trie's
/// canonical encoding). The trie key for the slot is
/// `keccak256(slot)` and the stored value is `rlp(value)` (RLP of
/// the BE-stripped U256).
#[derive(Clone, Debug)]
pub struct StorageProof {
    /// 32-byte BE representation of the slot key.
    pub slot: [u8; 32],
    /// 32-byte BE representation of the slot value.
    pub value: [u8; 32],
    /// MPT proof from `storage_root` to the leaf containing
    /// `(keccak256(slot), rlp(value))`.
    pub proof: Vec<Vec<u8>>,
}

/// Per-contract account inclusion proof + the storage slots touched.
#[derive(Clone, Debug)]
pub struct ContractProof {
    pub address: [u8; 20],
    pub account: Account,
    /// MPT proof from `state_root` to the leaf containing
    /// `(keccak256(address), rlp(account))`.
    pub account_proof: Vec<Vec<u8>>,
    /// Per-slot MPT proofs against `account.storage_root`.
    pub storage_proofs: Vec<StorageProof>,
}

/// Full world-state proof for one transaction execution.
#[derive(Clone, Debug)]
pub struct WorldProof {
    /// Canonical block header containing the four MPT roots.
    pub block_header: BlockHeader,
    /// Expected keccak256(rlp(block_header)) — must match
    /// `block_header_hash(block_header)`.
    pub block_hash: [u8; 32],
    /// Index of this transaction within the block.
    pub tx_index: u64,
    /// The transaction itself.
    pub transaction: Transaction,
    /// MPT proof from `block_header.transactions_root` to
    /// `(rlp(tx_index), wire_encoding(transaction))`.
    pub transaction_proof: Vec<Vec<u8>>,
    /// Execution receipt.
    pub receipt: Receipt,
    /// MPT proof from `block_header.receipts_root` to
    /// `(rlp(tx_index), wire_encoding(receipt))`.
    pub receipt_proof: Vec<Vec<u8>>,
    /// Per-contract account + storage proofs.
    pub contracts: Vec<ContractProof>,
}

/// Verify the full world-state proof bundle.
///
/// On `Ok(())` every piece — block header hash, account inclusion,
/// storage inclusion, transaction inclusion, receipt inclusion — has
/// been validated against the canonical block.
pub fn verify_world_proof_oracle(proof: &WorldProof) -> Result<(), String> {
    // 1. Block header hash.
    let computed = block_header_hash(&proof.block_header);
    if computed != proof.block_hash {
        return Err(format!(
            "Block header hash mismatch: computed={:?} declared={:?}",
            computed, proof.block_hash,
        ));
    }
    // 2. Transaction inclusion.
    verify_transaction_inclusion_oracle(
        proof.block_header.transactions_root,
        proof.tx_index,
        &proof.transaction,
        &proof.transaction_proof,
    )
    .map_err(|e| format!("Transaction inclusion failed: {}", e))?;
    // 3. Receipt inclusion.
    verify_receipt_inclusion_oracle(
        proof.block_header.receipts_root,
        proof.tx_index,
        &proof.receipt,
        &proof.receipt_proof,
    )
    .map_err(|e| format!("Receipt inclusion failed: {}", e))?;
    // 4. Per-contract account + storage inclusion.
    for (i, c) in proof.contracts.iter().enumerate() {
        verify_account_inclusion_oracle(
            proof.block_header.state_root,
            &c.address,
            &c.account,
            &c.account_proof,
        )
        .map_err(|e| format!("Contract {} account inclusion failed: {}", i, e))?;
        for (k, sp) in c.storage_proofs.iter().enumerate() {
            // Storage trie key = keccak256(slot_be).
            let trie_key = keccak256(&sp.slot);
            // Stored value = RLP of BE-stripped slot value.
            let value_rlp = rlp_encode_u256(&sp.value);
            if !verify_mpt_inclusion(c.account.storage_root, &trie_key, &value_rlp, &sp.proof) {
                return Err(format!(
                    "Contract {} storage proof {} (slot={:?}) failed: MPT inclusion failed",
                    i, k, sp.slot,
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_header::block_header_hash;
    use crate::mpt::single_leaf_trie;
    use crate::receipt::ReceiptType;
    use crate::transaction::LegacyTx;

    fn make_legacy_tx() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: 1,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        })
    }

    fn make_receipt() -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    #[test]
    fn world_proof_assembles_and_verifies() {
        let address = [0xabu8; 20];
        let mut slot_be = [0u8; 32];
        slot_be[31] = 7;
        let mut value_be = [0u8; 32];
        value_be[31] = 0x42;

        // Storage trie: single leaf for slot=7 → value=0x42.
        let trie_key = crate::keccak::keccak256(&slot_be);
        let value_rlp = crate::rlp::rlp_encode_u256(&value_be);
        let (storage_root, storage_proof) = single_leaf_trie(&trie_key, &value_rlp);

        // Account state including this storage root.
        let account = Account {
            nonce: 1,
            balance: [0u8; 32],
            storage_root,
            code_hash: crate::account::empty_code_hash(),
        };
        // World state trie: single leaf for address → account.
        let account_trie_key = crate::keccak::keccak256(&address);
        let account_rlp = crate::account::account_rlp(&account);
        let (state_root, account_proof) =
            single_leaf_trie(&account_trie_key, &account_rlp);

        // Transactions root: single leaf for tx_index=0 → tx_wire.
        let tx = make_legacy_tx();
        let tx_wire = tx.wire_encoding();
        let tx_index = 0u64;
        let tx_trie_key = crate::rlp::rlp_encode_uint(tx_index);
        let (transactions_root, tx_proof) = single_leaf_trie(&tx_trie_key, &tx_wire);

        // Receipts root: single leaf for tx_index=0 → receipt_wire.
        let receipt = make_receipt();
        let receipt_wire = receipt.wire_encoding();
        let (receipts_root, receipt_proof) =
            single_leaf_trie(&tx_trie_key, &receipt_wire);

        // Block header with all four roots.
        let block_header = BlockHeader {
            state_root,
            transactions_root,
            receipts_root,
            ..Default::default()
        };
        let block_hash = block_header_hash(&block_header);

        let proof = WorldProof {
            block_header,
            block_hash,
            tx_index,
            transaction: tx,
            transaction_proof: tx_proof,
            receipt,
            receipt_proof,
            contracts: vec![ContractProof {
                address,
                account,
                account_proof,
                storage_proofs: vec![StorageProof {
                    slot: slot_be,
                    value: value_be,
                    proof: storage_proof,
                }],
            }],
        };

        verify_world_proof_oracle(&proof).unwrap();
    }

    /// Build a complete honest WorldProof with one contract, one
    /// storage slot, one tx, one receipt. Returns the proof for the
    /// tampering tests to mutate.
    fn build_honest_world_proof() -> WorldProof {
        let address = [0xabu8; 20];
        let mut slot_be = [0u8; 32];
        slot_be[31] = 7;
        let mut value_be = [0u8; 32];
        value_be[31] = 0x42;

        let trie_key = crate::keccak::keccak256(&slot_be);
        let value_rlp = crate::rlp::rlp_encode_u256(&value_be);
        let (storage_root, storage_proof) = single_leaf_trie(&trie_key, &value_rlp);

        let account = Account {
            nonce: 1,
            balance: [0u8; 32],
            storage_root,
            code_hash: crate::account::empty_code_hash(),
        };
        let account_trie_key = crate::keccak::keccak256(&address);
        let account_rlp = crate::account::account_rlp(&account);
        let (state_root, account_proof) =
            single_leaf_trie(&account_trie_key, &account_rlp);

        let tx = make_legacy_tx();
        let tx_wire = tx.wire_encoding();
        let tx_index = 0u64;
        let tx_trie_key = crate::rlp::rlp_encode_uint(tx_index);
        let (transactions_root, tx_proof) = single_leaf_trie(&tx_trie_key, &tx_wire);

        let receipt = make_receipt();
        let receipt_wire = receipt.wire_encoding();
        let (receipts_root, receipt_proof) =
            single_leaf_trie(&tx_trie_key, &receipt_wire);

        let block_header = BlockHeader {
            state_root,
            transactions_root,
            receipts_root,
            ..Default::default()
        };
        let block_hash = block_header_hash(&block_header);

        WorldProof {
            block_header,
            block_hash,
            tx_index,
            transaction: tx,
            transaction_proof: tx_proof,
            receipt,
            receipt_proof,
            contracts: vec![ContractProof {
                address,
                account,
                account_proof,
                storage_proofs: vec![StorageProof {
                    slot: slot_be,
                    value: value_be,
                    proof: storage_proof,
                }],
            }],
        }
    }

    #[test]
    fn world_proof_rejects_tampered_storage_value() {
        let mut proof = build_honest_world_proof();
        // Tamper: claim a different storage value (proof was for 0x42,
        // now claim 0xff).
        proof.contracts[0].storage_proofs[0].value[31] = 0xff;
        let err = verify_world_proof_oracle(&proof).unwrap_err();
        assert!(
            err.contains("storage proof") && err.contains("failed"),
            "tampered storage value should be caught; got: {}", err,
        );
    }

    #[test]
    fn world_proof_rejects_tampered_account_storage_root() {
        let mut proof = build_honest_world_proof();
        // Tamper: change storage_root in the account claim — the
        // account RLP no longer matches what's in the state MPT.
        proof.contracts[0].account.storage_root = [0xff; 32];
        let err = verify_world_proof_oracle(&proof).unwrap_err();
        assert!(
            err.contains("account inclusion") || err.contains("Account"),
            "tampered storage_root in account should fail account MPT inclusion; got: {}",
            err,
        );
    }

    #[test]
    fn world_proof_rejects_tampered_transaction_index() {
        let mut proof = build_honest_world_proof();
        // Tamper: claim a different tx_index than the one the proof
        // was built for.
        proof.tx_index = 99;
        let err = verify_world_proof_oracle(&proof).unwrap_err();
        assert!(
            err.contains("Transaction inclusion") || err.contains("Transaction MPT"),
            "tampered tx_index should fail tx inclusion; got: {}",
            err,
        );
    }

    #[test]
    fn world_proof_rejects_tampered_receipt_status() {
        let mut proof = build_honest_world_proof();
        // Tamper: flip receipt status — wire_encoding changes — proof
        // built for old wire fails.
        proof.receipt.status = 0;
        let err = verify_world_proof_oracle(&proof).unwrap_err();
        assert!(
            err.contains("Receipt inclusion") || err.contains("Receipt MPT"),
            "tampered receipt status should fail receipt inclusion; got: {}",
            err,
        );
    }

    #[test]
    fn world_proof_rejects_tampered_block_state_root() {
        let mut proof = build_honest_world_proof();
        // Tamper: change state_root in the block header. block_hash
        // mismatch will catch this first.
        proof.block_header.state_root = [0xff; 32];
        let err = verify_world_proof_oracle(&proof).unwrap_err();
        // Block hash check fires first.
        assert!(
            err.contains("Block header hash mismatch"),
            "tampered state_root → block_hash mismatch; got: {}",
            err,
        );
    }

    /// Multi-contract scenario: two contracts with distinct first
    /// nibbles in `keccak256(address)`. Build a 2-leaf state trie
    /// containing both, plus a proof for ONE of them, and verify
    /// the WorldProof bundle accepts.
    #[test]
    fn world_proof_with_multi_contract_state_trie() {
        use crate::mpt::two_leaf_trie_distinct_first_nibble;

        // Need two addresses whose keccak256 differs in first nibble.
        // [0x01; 20] and [0x02; 20] — try them and find a pair that
        // works (or just hardcode known-good addresses).
        let addr_a = [0x11u8; 20];
        let addr_b = [0x22u8; 20];
        let key_a = crate::keccak::keccak256(&addr_a);
        let key_b = crate::keccak::keccak256(&addr_b);
        let nib_a = key_a[0] >> 4;
        let nib_b = key_b[0] >> 4;
        assert_ne!(
            nib_a, nib_b,
            "test fixture: addresses [0x11;20] and [0x22;20] should hash to \
             distinct first nibbles (nib_a={}, nib_b={})",
            nib_a, nib_b,
        );

        // Account A: with a single storage slot.
        let mut slot_be = [0u8; 32];
        slot_be[31] = 0x07;
        let mut value_be = [0u8; 32];
        value_be[31] = 0x42;
        let trie_key_slot = crate::keccak::keccak256(&slot_be);
        let value_rlp = crate::rlp::rlp_encode_u256(&value_be);
        let (storage_root_a, storage_proof_a) =
            single_leaf_trie(&trie_key_slot, &value_rlp);

        let account_a = Account {
            nonce: 1,
            balance: [0u8; 32],
            storage_root: storage_root_a,
            code_hash: crate::account::empty_code_hash(),
        };

        // Account B: empty (default).
        let account_b = Account::default();

        // World state trie: two leaves at keccak256(addr_a) and keccak256(addr_b).
        let account_a_rlp = crate::account::account_rlp(&account_a);
        let account_b_rlp = crate::account::account_rlp(&account_b);
        let (state_root, account_proof_a) = two_leaf_trie_distinct_first_nibble(
            &key_a, &account_a_rlp, &key_b, &account_b_rlp,
        );

        // Transaction + receipt (single).
        let tx = make_legacy_tx();
        let tx_index = 0u64;
        let tx_trie_key = crate::rlp::rlp_encode_uint(tx_index);
        let (transactions_root, tx_proof) =
            single_leaf_trie(&tx_trie_key, &tx.wire_encoding());
        let receipt = make_receipt();
        let (receipts_root, receipt_proof) =
            single_leaf_trie(&tx_trie_key, &receipt.wire_encoding());

        // Block header.
        let block_header = BlockHeader {
            state_root,
            transactions_root,
            receipts_root,
            ..Default::default()
        };
        let block_hash = block_header_hash(&block_header);

        let proof = WorldProof {
            block_header,
            block_hash,
            tx_index,
            transaction: tx,
            transaction_proof: tx_proof,
            receipt,
            receipt_proof,
            // Only contract A is included in this proof bundle (the
            // tx interacted with it). Contract B exists in the state
            // trie but isn't part of the proof for this transaction.
            contracts: vec![ContractProof {
                address: addr_a,
                account: account_a,
                account_proof: account_proof_a,
                storage_proofs: vec![StorageProof {
                    slot: slot_be,
                    value: value_be,
                    proof: storage_proof_a,
                }],
            }],
        };

        verify_world_proof_oracle(&proof).unwrap();
    }

    #[test]
    fn world_proof_rejects_tampered_block_hash() {
        // Build any valid bundle, then tamper block_hash.
        let address = [0xabu8; 20];
        let account = Account::default();
        let account_trie_key = crate::keccak::keccak256(&address);
        let account_rlp = crate::account::account_rlp(&account);
        let (state_root, account_proof) =
            single_leaf_trie(&account_trie_key, &account_rlp);

        let tx = make_legacy_tx();
        let tx_wire = tx.wire_encoding();
        let tx_index = 0u64;
        let tx_trie_key = crate::rlp::rlp_encode_uint(tx_index);
        let (transactions_root, tx_proof) = single_leaf_trie(&tx_trie_key, &tx_wire);

        let receipt = make_receipt();
        let receipt_wire = receipt.wire_encoding();
        let (receipts_root, receipt_proof) =
            single_leaf_trie(&tx_trie_key, &receipt_wire);

        let block_header = BlockHeader {
            state_root,
            transactions_root,
            receipts_root,
            ..Default::default()
        };

        let proof = WorldProof {
            block_header,
            block_hash: [0xff; 32], // wrong
            tx_index,
            transaction: tx,
            transaction_proof: tx_proof,
            receipt,
            receipt_proof,
            contracts: vec![ContractProof {
                address,
                account,
                account_proof,
                storage_proofs: Vec::new(),
            }],
        };
        let err = verify_world_proof_oracle(&proof).unwrap_err();
        assert!(err.contains("Block header hash mismatch"), "got: {}", err);
    }
}
