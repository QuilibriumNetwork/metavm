//! Transaction trie composition oracle.
//!
//! Verifies that a transaction is included in the block's
//! transactionsRoot via MPT inclusion.

use crate::mpt;
use crate::transaction::Transaction;

pub fn verify_tx_root_single(
    transactions_root: [u8; 32],
    tx_index: u64,
    tx: &Transaction,
    proof: &[Vec<u8>],
) -> Result<(), String> {
    crate::transaction::verify_transaction_inclusion_oracle(
        transactions_root, tx_index, tx, proof,
    )
}

pub fn compute_single_leaf_tx_root(
    tx_index: u64,
    tx: &Transaction,
) -> ([u8; 32], Vec<Vec<u8>>) {
    let key = crate::transaction::tx_trie_key(tx_index);
    let value = tx.wire_encoding();
    mpt::single_leaf_trie(&key, &value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{LegacyTx, Transaction};

    fn sample_tx() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: 0,
            gas_price: [0u8; 32],
            gas_limit: 21000,
            to: Some([0x42; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 27,
            r: [0xAA; 32],
            s: [0xBB; 32],
        })
    }

    #[test]
    fn single_leaf_root_deterministic() {
        let tx = sample_tx();
        let (r1, _) = compute_single_leaf_tx_root(0, &tx);
        let (r2, _) = compute_single_leaf_tx_root(0, &tx);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_leaf_inclusion_verifies() {
        let tx = sample_tx();
        let (root, proof) = compute_single_leaf_tx_root(0, &tx);
        verify_tx_root_single(root, 0, &tx, &proof).unwrap();
    }

    #[test]
    fn wrong_root_fails() {
        let tx = sample_tx();
        let (mut root, proof) = compute_single_leaf_tx_root(0, &tx);
        root[0] ^= 0xff;
        assert!(verify_tx_root_single(root, 0, &tx, &proof).is_err());
    }

    #[test]
    fn tx_hash_matches_expected() {
        let tx = sample_tx();
        let h1 = tx.hash();
        let h2 = tx.hash();
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 32]);
    }
}
