//! Transaction signature hash oracle.
//!
//! Computes the keccak256 of the tx's signed payload (without v/r/s
//! for legacy, or chain-id-prefixed for EIP-1559). This is the
//! message the sender's signature commits to. Algebraic binding to
//! the EVM trace's tx_origin column would require ECDSA recovery
//! (deferred); for now this oracle exposes the hash deterministically.

use crate::keccak::keccak256;
use crate::rlp;
use crate::transaction::{Eip1559Tx, LegacyTx, Transaction};

pub fn legacy_signing_hash(tx: &LegacyTx, chain_id: Option<u64>) -> [u8; 32] {
    let mut fields: Vec<Vec<u8>> = vec![
        crate::u64_rlp_air::rlp_encode_u64(tx.nonce),
        crate::u256_rlp_air::rlp_encode_u256_be(&tx.gas_price),
        crate::u64_rlp_air::rlp_encode_u64(tx.gas_limit),
        match &tx.to {
            Some(addr) => rlp::rlp_encode_bytes(addr),
            None => vec![0x80],
        },
        crate::u256_rlp_air::rlp_encode_u256_be(&tx.value),
        rlp::rlp_encode_bytes(&tx.data),
    ];
    if let Some(cid) = chain_id {
        // EIP-155: append (chain_id, 0, 0) for signing
        fields.push(crate::u64_rlp_air::rlp_encode_u64(cid));
        fields.push(vec![0x80]);
        fields.push(vec![0x80]);
    }
    let payload = rlp::rlp_encode_list(&fields);
    keccak256(&payload)
}

pub fn eip1559_signing_hash(tx: &Eip1559Tx) -> [u8; 32] {
    let fields: Vec<Vec<u8>> = vec![
        crate::u64_rlp_air::rlp_encode_u64(tx.chain_id),
        crate::u64_rlp_air::rlp_encode_u64(tx.nonce),
        crate::u256_rlp_air::rlp_encode_u256_be(&tx.max_priority_fee_per_gas),
        crate::u256_rlp_air::rlp_encode_u256_be(&tx.max_fee_per_gas),
        crate::u64_rlp_air::rlp_encode_u64(tx.gas_limit),
        match &tx.to {
            Some(addr) => rlp::rlp_encode_bytes(addr),
            None => vec![0x80],
        },
        crate::u256_rlp_air::rlp_encode_u256_be(&tx.value),
        rlp::rlp_encode_bytes(&tx.data),
        vec![0xc0], // empty access list
    ];
    let payload = rlp::rlp_encode_list(&fields);
    // Type byte 0x02 prefix per EIP-2718
    let mut prefixed = vec![0x02];
    prefixed.extend_from_slice(&payload);
    keccak256(&prefixed)
}

pub fn signing_hash(tx: &Transaction, chain_id: Option<u64>) -> [u8; 32] {
    match tx {
        Transaction::Legacy(legacy) => legacy_signing_hash(legacy, chain_id),
        Transaction::Eip1559(eip1559) => eip1559_signing_hash(eip1559),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::LegacyTx;

    fn sample_legacy() -> LegacyTx {
        LegacyTx {
            nonce: 0,
            gas_price: [0u8; 32],
            gas_limit: 21000,
            to: Some([0x42; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 27, r: [0xAA; 32], s: [0xBB; 32],
        }
    }

    #[test]
    fn legacy_hash_deterministic() {
        let tx = sample_legacy();
        let h1 = legacy_signing_hash(&tx, None);
        let h2 = legacy_signing_hash(&tx, None);
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 32]);
    }

    #[test]
    fn legacy_eip155_changes_hash() {
        let tx = sample_legacy();
        let pre = legacy_signing_hash(&tx, None);
        let post = legacy_signing_hash(&tx, Some(1));
        assert_ne!(pre, post);
    }

    #[test]
    fn different_chain_ids_different_hash() {
        let tx = sample_legacy();
        let h1 = legacy_signing_hash(&tx, Some(1));
        let h5 = legacy_signing_hash(&tx, Some(5));
        assert_ne!(h1, h5);
    }

    #[test]
    fn eip1559_hash_deterministic() {
        let tx = Eip1559Tx {
            chain_id: 1,
            nonce: 0,
            max_priority_fee_per_gas: [0u8; 32],
            max_fee_per_gas: [0u8; 32],
            gas_limit: 21000,
            to: Some([0x42; 20]),
            value: [0u8; 32],
            data: vec![],
            access_list_rlp: vec![],
            y_parity: 0,
            r: [0xAA; 32],
            s: [0xBB; 32],
        };
        let h1 = eip1559_signing_hash(&tx);
        let h2 = eip1559_signing_hash(&tx);
        assert_eq!(h1, h2);
    }

    #[test]
    fn signing_hash_dispatch() {
        let tx = Transaction::Legacy(sample_legacy());
        let h = signing_hash(&tx, Some(1));
        assert_ne!(h, [0u8; 32]);
    }
}
