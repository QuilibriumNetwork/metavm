//! Block execution proof bundle.
//!
//! Ties together the proof of a single transaction's execution within
//! a block to the block's inclusion in the beacon chain.
//!
//! The bundle contains:
//! - The BlockHeader (Layer B) with its block_hash
//! - Transaction MPT inclusion proof (tx in transactionsRoot)
//! - Receipt MPT inclusion proof (receipt in receiptsRoot)
//! - Account MPT inclusion proof (state_root binding)
//! - Storage MPT inclusion proofs (per SLOAD/SSTORE)
//!
//! The `verify_block_execution_proof` function validates all
//! host-side oracles and returns the block_hash for upstream
//! binding to the beacon chain (Layer C).

use crate::account_transition::AccountTransition;
use crate::block_header::BlockHeader;
use crate::receipt::Receipt;
use crate::sstore_transition::SstoreSequence;
use crate::transaction::Transaction;
use crate::withdrawal::Withdrawal;

#[derive(Clone, Debug)]
pub struct BlockExecutionProof {
    pub header: BlockHeader,
    pub tx_index: u64,
    pub tx: Transaction,
    pub tx_proof: Vec<Vec<u8>>,
    pub receipt: Receipt,
    pub receipt_proof: Vec<Vec<u8>>,
}

/// Extended proof that also covers world state mutation.
#[derive(Clone, Debug)]
pub struct ExtendedExecutionProof {
    pub base: BlockExecutionProof,
    pub sender_transition: AccountTransition,
    pub storage_transitions: Vec<SstoreSequence>,
    pub pre_state_accounts: Vec<crate::state_root_transition::AccountInclusion>,
    pub post_state_accounts: Vec<crate::state_root_transition::AccountInclusion>,
    pub withdrawals: Vec<(u64, Withdrawal, Vec<Vec<u8>>)>,
}

pub fn verify_block_execution_proof(
    proof: &BlockExecutionProof,
) -> Result<[u8; 32], String> {
    let block_hash = crate::block_header::block_header_hash(&proof.header);

    crate::transaction::verify_transaction_inclusion_oracle(
        proof.header.transactions_root,
        proof.tx_index,
        &proof.tx,
        &proof.tx_proof,
    ).map_err(|e| format!("tx inclusion: {}", e))?;

    crate::receipt::verify_receipt_inclusion_oracle(
        proof.header.receipts_root,
        proof.tx_index,
        &proof.receipt,
        &proof.receipt_proof,
    ).map_err(|e| format!("receipt inclusion: {}", e))?;

    Ok(block_hash)
}

pub fn verify_extended_execution_proof(
    proof: &ExtendedExecutionProof,
) -> Result<[u8; 32], String> {
    let block_hash = verify_block_execution_proof(&proof.base)?;

    crate::account_transition::verify_nonce_increment(&proof.sender_transition)
        .map_err(|e| format!("sender: {}", e))?;
    crate::account_transition::verify_code_hash_unchanged(&proof.sender_transition)
        .map_err(|e| format!("sender: {}", e))?;

    for (i, seq) in proof.storage_transitions.iter().enumerate() {
        crate::sstore_transition::verify_sstore_sequence(seq)
            .map_err(|e| format!("storage[{}]: {}", i, e))?;
    }

    // Post-state accounts must be included under header.state_root.
    let post_state_root = proof.base.header.state_root;
    for (i, post) in proof.post_state_accounts.iter().enumerate() {
        crate::state_root_transition::verify_post_state_inclusion(post_state_root, post)
            .map_err(|e| format!("post-state[{}]: {}", i, e))?;
    }

    // Withdrawals must be included under header.withdrawals_root (if present).
    if !proof.withdrawals.is_empty() {
        let wr = proof.base.header.withdrawals_root
            .ok_or("block has withdrawals but header.withdrawals_root is None")?;
        for (i, (idx, w, pf)) in proof.withdrawals.iter().enumerate() {
            crate::withdrawal_trie::verify_withdrawal_inclusion(wr, *idx, w, pf)
                .map_err(|e| format!("withdrawal[{}]: {}", i, e))?;
        }
    }

    Ok(block_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt;
    use crate::receipt::{Receipt, ReceiptType};
    use crate::transaction::{LegacyTx, Transaction};

    fn make_proof() -> BlockExecutionProof {
        let tx = Transaction::Legacy(LegacyTx {
            nonce: 0,
            gas_price: [0u8; 32],
            gas_limit: 21000,
            to: Some([0x42; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 27,
            r: [0xAA; 32],
            s: [0xBB; 32],
        });
        let receipt = Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        };

        let tx_key = crate::transaction::tx_trie_key(0);
        let tx_value = tx.wire_encoding();
        let (tx_root, tx_proof) = mpt::single_leaf_trie(&tx_key, &tx_value);

        let rx_key = crate::receipt::receipt_trie_key(0);
        let rx_value = receipt.wire_encoding();
        let (rx_root, rx_proof) = mpt::single_leaf_trie(&rx_key, &rx_value);

        let mut header = BlockHeader::default();
        header.transactions_root = tx_root;
        header.receipts_root = rx_root;
        header.number = 100;
        header.timestamp = 1700000000;
        header.gas_limit = 30_000_000;
        header.gas_used = 21_000;

        BlockExecutionProof {
            header,
            tx_index: 0,
            tx,
            tx_proof,
            receipt,
            receipt_proof: rx_proof,
        }
    }

    #[test]
    fn valid_proof_passes() {
        let p = make_proof();
        let hash = verify_block_execution_proof(&p).unwrap();
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn tampered_tx_root_fails() {
        let mut p = make_proof();
        p.header.transactions_root[0] ^= 0xff;
        assert!(verify_block_execution_proof(&p).is_err());
    }

    #[test]
    fn tampered_receipt_root_fails() {
        let mut p = make_proof();
        p.header.receipts_root[0] ^= 0xff;
        assert!(verify_block_execution_proof(&p).is_err());
    }

    #[test]
    fn block_hash_matches_header() {
        let p = make_proof();
        let hash = verify_block_execution_proof(&p).unwrap();
        let expected = crate::block_header::block_header_hash(&p.header);
        assert_eq!(hash, expected);
    }

    #[test]
    fn extended_proof_with_account_transition() {
        use crate::account::{Account, empty_code_hash, empty_storage_root};

        let base = make_proof();
        let pre = Account {
            nonce: 0, balance: { let mut b = [0u8; 32]; b[24..32].copy_from_slice(&1_000_000u64.to_be_bytes()); b },
            storage_root: empty_storage_root(), code_hash: empty_code_hash(),
        };
        let post = Account {
            nonce: 1, balance: { let mut b = [0u8; 32]; b[24..32].copy_from_slice(&1_000_000u64.to_be_bytes()); b },
            storage_root: empty_storage_root(), code_hash: empty_code_hash(),
        };
        let sender = AccountTransition { address: [0x42; 20], pre, post };
        let ext = ExtendedExecutionProof {
            base,
            sender_transition: sender,
            storage_transitions: vec![],
            pre_state_accounts: vec![],
            post_state_accounts: vec![],
            withdrawals: vec![],
        };
        verify_extended_execution_proof(&ext).unwrap();
    }

    #[test]
    fn extended_proof_wrong_nonce_fails() {
        use crate::account::{Account, empty_code_hash, empty_storage_root};

        let base = make_proof();
        let pre = Account { nonce: 0, balance: [0u8; 32], storage_root: empty_storage_root(), code_hash: empty_code_hash() };
        let post = Account { nonce: 0, balance: [0u8; 32], storage_root: empty_storage_root(), code_hash: empty_code_hash() };
        let sender = AccountTransition { address: [0x42; 20], pre, post };
        let ext = ExtendedExecutionProof { base, sender_transition: sender, storage_transitions: vec![], pre_state_accounts: vec![], post_state_accounts: vec![], withdrawals: vec![] };
        assert!(verify_extended_execution_proof(&ext).is_err());
    }

    #[test]
    fn extended_proof_with_withdrawal() {
        use crate::account::{Account, empty_code_hash, empty_storage_root};
        use crate::withdrawal::Withdrawal;
        use crate::withdrawal_trie::compute_single_withdrawal_root;

        let w = Withdrawal {
            index: 0, validator_index: 42, address: [0x11; 20], amount: 32_000_000_000,
        };
        let (wr, wp) = compute_single_withdrawal_root(0, &w);

        let mut base = make_proof();
        base.header.withdrawals_root = Some(wr);

        let pre = Account {
            nonce: 0,
            balance: { let mut b = [0u8; 32]; b[24..32].copy_from_slice(&1_000_000u64.to_be_bytes()); b },
            storage_root: empty_storage_root(), code_hash: empty_code_hash(),
        };
        let post = Account { nonce: 1, ..pre.clone() };
        let sender = AccountTransition { address: [0x42; 20], pre, post };

        let ext = ExtendedExecutionProof {
            base,
            sender_transition: sender,
            storage_transitions: vec![],
            pre_state_accounts: vec![],
            post_state_accounts: vec![],
            withdrawals: vec![(0, w, wp)],
        };
        verify_extended_execution_proof(&ext).unwrap();
    }
}
