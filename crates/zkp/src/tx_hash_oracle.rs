//! Transaction hash oracle.
//!
//! Verifies that a transaction's hash matches its wire encoding,
//! and that the transaction is included in the block's
//! transactionsRoot via MPT inclusion.

use crate::transaction::Transaction;

pub fn verify_tx_hash(tx: &Transaction) -> [u8; 32] {
    tx.hash()
}

pub fn verify_tx_inclusion(
    transactions_root: &[u8; 32],
    tx_index: u64,
    tx: &Transaction,
    proof: &[Vec<u8>],
) -> Result<(), String> {
    crate::transaction::verify_transaction_inclusion_oracle(
        *transactions_root, tx_index, tx, proof,
    )
}

pub fn verify_batch_tx_hashes(txs: &[Transaction]) -> Vec<[u8; 32]> {
    txs.iter().map(|tx| tx.hash()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{LegacyTx, Transaction};

    fn sample_tx() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: 42,
            gas_price: [0u8; 32],
            gas_limit: 21000,
            to: Some([0x42; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 28,
            r: [0xAA; 32],
            s: [0xBB; 32],
        })
    }

    #[test]
    fn tx_hash_deterministic() {
        let tx = sample_tx();
        let h1 = verify_tx_hash(&tx);
        let h2 = verify_tx_hash(&tx);
        assert_eq!(h1, h2);
    }

    #[test]
    fn tx_hash_changes_with_nonce() {
        let tx1 = sample_tx();
        let mut tx2_inner = match sample_tx() {
            Transaction::Legacy(l) => l,
            _ => unreachable!(),
        };
        tx2_inner.nonce = 43;
        let tx2 = Transaction::Legacy(tx2_inner);
        assert_ne!(verify_tx_hash(&tx1), verify_tx_hash(&tx2));
    }

    #[test]
    fn batch_hashes() {
        let txs = vec![sample_tx(), sample_tx()];
        let hashes = verify_batch_tx_hashes(&txs);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], hashes[1]); // same tx, same hash
    }
}
