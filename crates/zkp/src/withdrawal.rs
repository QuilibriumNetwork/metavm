//! Ethereum withdrawal type and RLP encoding.
//!
//! Withdrawals (Shapella/Cancun) are included in the block header
//! via withdrawals_root. Each withdrawal is RLP-encoded as a list:
//! `[index, validator_index, address, amount]`.

use crate::rlp;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Withdrawal {
    pub index: u64,
    pub validator_index: u64,
    pub address: [u8; 20],
    pub amount: u64, // in Gwei
}

pub fn withdrawal_rlp(w: &Withdrawal) -> Vec<u8> {
    let items: Vec<Vec<u8>> = vec![
        crate::u64_rlp_air::rlp_encode_u64(w.index),
        crate::u64_rlp_air::rlp_encode_u64(w.validator_index),
        rlp::rlp_encode_bytes(&w.address),
        crate::u64_rlp_air::rlp_encode_u64(w.amount),
    ];
    rlp::rlp_encode_list(&items)
}

pub fn withdrawal_hash(w: &Withdrawal) -> [u8; 32] {
    crate::keccak::keccak256(&withdrawal_rlp(w))
}

pub fn verify_withdrawal_rlp_composition(w: &Withdrawal) -> Result<Vec<u8>, String> {
    let canonical = withdrawal_rlp(w);
    let fields: Vec<Vec<u8>> = vec![
        crate::u64_rlp_air::rlp_encode_u64(w.index),
        crate::u64_rlp_air::rlp_encode_u64(w.validator_index),
        rlp::rlp_encode_bytes(&w.address),
        crate::u64_rlp_air::rlp_encode_u64(w.amount),
    ];
    let assembled = rlp::rlp_encode_list(&fields);
    if assembled != canonical {
        return Err("withdrawal RLP mismatch".into());
    }
    Ok(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn withdrawal_rlp_deterministic() {
        let w = Withdrawal {
            index: 0, validator_index: 42,
            address: [0x11; 20], amount: 32_000_000_000,
        };
        let r1 = withdrawal_rlp(&w);
        let r2 = withdrawal_rlp(&w);
        assert_eq!(r1, r2);
    }

    #[test]
    fn withdrawal_hash_nonzero() {
        let w = Withdrawal {
            index: 1, validator_index: 100,
            address: [0x22; 20], amount: 1_000_000_000,
        };
        assert_ne!(withdrawal_hash(&w), [0u8; 32]);
    }

    #[test]
    fn composition_matches_canonical() {
        let w = Withdrawal {
            index: 5, validator_index: 200,
            address: [0x33; 20], amount: 64_000_000_000,
        };
        verify_withdrawal_rlp_composition(&w).unwrap();
    }

    #[test]
    fn different_withdrawals_different_rlp() {
        let w1 = Withdrawal { index: 0, validator_index: 1, address: [0; 20], amount: 100 };
        let w2 = Withdrawal { index: 1, validator_index: 1, address: [0; 20], amount: 100 };
        assert_ne!(withdrawal_rlp(&w1), withdrawal_rlp(&w2));
    }

    #[test]
    fn zero_amount_valid() {
        let w = Withdrawal { index: 0, validator_index: 0, address: [0; 20], amount: 0 };
        verify_withdrawal_rlp_composition(&w).unwrap();
    }
}
