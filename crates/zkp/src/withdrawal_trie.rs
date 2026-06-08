//! Withdrawal trie composition oracle.
//!
//! Verifies withdrawal MPT root matches the block header's
//! withdrawals_root via single-leaf MPT inclusion.

use crate::mpt;
use crate::rlp;
use crate::withdrawal::{Withdrawal, withdrawal_rlp};

pub fn withdrawal_trie_key(index: u64) -> Vec<u8> {
    rlp::rlp_encode_uint(index)
}

pub fn compute_single_withdrawal_root(
    index: u64,
    w: &Withdrawal,
) -> ([u8; 32], Vec<Vec<u8>>) {
    let key = withdrawal_trie_key(index);
    let value = withdrawal_rlp(w);
    mpt::single_leaf_trie(&key, &value)
}

pub fn verify_withdrawal_inclusion(
    withdrawals_root: [u8; 32],
    index: u64,
    w: &Withdrawal,
    proof: &[Vec<u8>],
) -> Result<(), String> {
    let key = withdrawal_trie_key(index);
    let value = withdrawal_rlp(w);
    if !mpt::verify_mpt_inclusion(withdrawals_root, &key, &value, proof) {
        return Err("withdrawal not included under withdrawals_root".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_withdrawal() -> Withdrawal {
        Withdrawal {
            index: 0,
            validator_index: 42,
            address: [0x11; 20],
            amount: 32_000_000_000,
        }
    }

    #[test]
    fn single_withdrawal_inclusion() {
        let w = sample_withdrawal();
        let (root, proof) = compute_single_withdrawal_root(0, &w);
        verify_withdrawal_inclusion(root, 0, &w, &proof).unwrap();
    }

    #[test]
    fn wrong_root_fails() {
        let w = sample_withdrawal();
        let (mut root, proof) = compute_single_withdrawal_root(0, &w);
        root[0] ^= 0xff;
        assert!(verify_withdrawal_inclusion(root, 0, &w, &proof).is_err());
    }

    #[test]
    fn different_index_different_root() {
        let w = sample_withdrawal();
        let (r0, _) = compute_single_withdrawal_root(0, &w);
        let (r1, _) = compute_single_withdrawal_root(1, &w);
        assert_ne!(r0, r1);
    }

    #[test]
    fn trie_key_matches_rlp_uint() {
        assert_eq!(withdrawal_trie_key(0), vec![0x80]);
        assert_eq!(withdrawal_trie_key(1), vec![0x01]);
        assert_eq!(withdrawal_trie_key(127), vec![0x7f]);
    }
}
