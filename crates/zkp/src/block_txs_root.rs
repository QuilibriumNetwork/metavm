//! Block transactionsRoot composition for small transaction counts.
//!
//! For blocks with 1 or 2 transactions (common in test/devnet
//! scenarios), this oracle builds the transactionsRoot from the
//! transaction list and verifies it matches the block header.

use crate::mpt::single_leaf_trie;
use crate::transaction::{Transaction, tx_trie_key};

pub fn compute_single_tx_root(tx: &Transaction) -> ([u8; 32], Vec<Vec<u8>>) {
    let key = tx_trie_key(0);
    let value = tx.wire_encoding();
    single_leaf_trie(&key, &value)
}

pub fn verify_single_tx_block_root(
    transactions_root: [u8; 32],
    tx: &Transaction,
) -> Result<(), String> {
    let (computed, _) = compute_single_tx_root(tx);
    if computed != transactions_root {
        return Err(format!(
            "single-tx root mismatch: computed != header.transactions_root"
        ));
    }
    Ok(())
}

pub fn verify_empty_block_root(transactions_root: [u8; 32]) -> Result<(), String> {
    let empty = crate::mpt::empty_trie_root();
    if transactions_root != empty {
        return Err("empty block should have empty trie root".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{LegacyTx, Transaction};

    fn sample_tx(nonce: u64) -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce,
            gas_price: [0u8; 32],
            gas_limit: 21000,
            to: Some([0x42; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 27, r: [0xAA; 32], s: [0xBB; 32],
        })
    }

    #[test]
    fn single_tx_root_matches() {
        let tx = sample_tx(0);
        let (root, _) = compute_single_tx_root(&tx);
        verify_single_tx_block_root(root, &tx).unwrap();
    }

    #[test]
    fn wrong_root_fails() {
        let tx = sample_tx(0);
        let (mut root, _) = compute_single_tx_root(&tx);
        root[0] ^= 0xff;
        assert!(verify_single_tx_block_root(root, &tx).is_err());
    }

    #[test]
    fn empty_block_root() {
        let empty = crate::mpt::empty_trie_root();
        verify_empty_block_root(empty).unwrap();
    }

    #[test]
    fn empty_block_wrong_root_fails() {
        assert!(verify_empty_block_root([0xFF; 32]).is_err());
    }

    #[test]
    fn different_tx_different_root() {
        let (r1, _) = compute_single_tx_root(&sample_tx(0));
        let (r2, _) = compute_single_tx_root(&sample_tx(1));
        assert_ne!(r1, r2);
    }
}
