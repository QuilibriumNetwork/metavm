//! Receipt trie composition oracle.
//!
//! Verifies that a set of receipts produces the correct receiptsRoot
//! via MPT construction (single-leaf shortcut for the common case of
//! proving a single transaction's receipt inclusion).

use crate::mpt;
use crate::receipt::Receipt;

pub fn verify_receipts_root_single(
    receipts_root: [u8; 32],
    tx_index: u64,
    receipt: &Receipt,
    proof: &[Vec<u8>],
) -> Result<(), String> {
    crate::receipt::verify_receipt_inclusion_oracle(
        receipts_root, tx_index, receipt, proof,
    )
}

pub fn compute_single_leaf_receipts_root(
    tx_index: u64,
    receipt: &Receipt,
) -> ([u8; 32], Vec<Vec<u8>>) {
    let key = crate::receipt::receipt_trie_key(tx_index);
    let value = receipt.wire_encoding();
    mpt::single_leaf_trie(&key, &value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt::{Receipt, ReceiptType};

    fn sample_receipt() -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    #[test]
    fn single_leaf_root_deterministic() {
        let r = sample_receipt();
        let (root1, _) = compute_single_leaf_receipts_root(0, &r);
        let (root2, _) = compute_single_leaf_receipts_root(0, &r);
        assert_eq!(root1, root2);
    }

    #[test]
    fn different_index_different_root() {
        let r = sample_receipt();
        let (root0, _) = compute_single_leaf_receipts_root(0, &r);
        let (root1, _) = compute_single_leaf_receipts_root(1, &r);
        assert_ne!(root0, root1);
    }

    #[test]
    fn single_leaf_inclusion_verifies() {
        let r = sample_receipt();
        let (root, proof) = compute_single_leaf_receipts_root(0, &r);
        verify_receipts_root_single(root, 0, &r, &proof).unwrap();
    }

    #[test]
    fn wrong_root_fails() {
        let r = sample_receipt();
        let (mut root, proof) = compute_single_leaf_receipts_root(0, &r);
        root[0] ^= 0xff;
        assert!(verify_receipts_root_single(root, 0, &r, &proof).is_err());
    }
}
