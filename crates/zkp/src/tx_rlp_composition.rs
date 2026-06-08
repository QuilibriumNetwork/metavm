//! Legacy transaction RLP composition oracle.
//!
//! Decomposes a `LegacyTx` into per-field RLP encodings using the
//! session's gadgets, assembles them into the canonical wire encoding,
//! and verifies the result matches `LegacyTx::rlp_encode()`.

use crate::transaction::{LegacyTx, Eip1559Tx};
use crate::u64_rlp_air::rlp_encode_u64;
use crate::u256_rlp_air::rlp_encode_u256_be;
use crate::rlp_var_bytes_air::rlp_encode_bytes;

/// Verify that our per-field RLP gadgets produce the canonical legacy
/// tx wire encoding when assembled.
pub fn verify_legacy_tx_rlp_composition(tx: &LegacyTx) -> Result<Vec<u8>, String> {
    let canonical = tx.rlp_encode();

    let to_bytes = tx.to.map(|a| a.to_vec()).unwrap_or_default();
    let fields: Vec<Vec<u8>> = vec![
        rlp_encode_u64(tx.nonce),
        rlp_encode_u256_be(&tx.gas_price),
        rlp_encode_u64(tx.gas_limit),
        rlp_encode_bytes(&to_bytes),
        rlp_encode_u256_be(&tx.value),
        rlp_encode_bytes(&tx.data),
        rlp_encode_u64(tx.v),
        rlp_encode_u256_be(&tx.r),
        rlp_encode_u256_be(&tx.s),
    ];

    let payload_len: usize = fields.iter().map(|f| f.len()).sum();
    let mut assembled = Vec::with_capacity(3 + payload_len);
    if payload_len < 56 {
        assembled.push(0xc0 + payload_len as u8);
    } else {
        let mut len_be = Vec::new();
        let mut n = payload_len;
        while n > 0 { len_be.push((n & 0xff) as u8); n >>= 8; }
        len_be.reverse();
        assembled.push(0xf7 + len_be.len() as u8);
        assembled.extend_from_slice(&len_be);
    }
    for f in &fields { assembled.extend_from_slice(f); }

    if assembled != canonical {
        return Err(format!(
            "Legacy tx RLP mismatch: assembled {} bytes vs canonical {} bytes",
            assembled.len(), canonical.len(),
        ));
    }
    Ok(assembled)
}

/// Verify EIP-1559 tx RLP composition. Note: access_list_rlp is
/// passed through as-is (its internal encoding is not decomposed
/// by our gadgets — it needs a dedicated access-list RLP gadget).
pub fn verify_eip1559_tx_rlp_composition(tx: &Eip1559Tx) -> Result<Vec<u8>, String> {
    let canonical = tx.wire_encoding();

    let to_bytes = tx.to.map(|a| a.to_vec()).unwrap_or_default();
    let fields: Vec<Vec<u8>> = vec![
        rlp_encode_u64(tx.chain_id),
        rlp_encode_u64(tx.nonce),
        rlp_encode_u256_be(&tx.max_priority_fee_per_gas),
        rlp_encode_u256_be(&tx.max_fee_per_gas),
        rlp_encode_u64(tx.gas_limit),
        rlp_encode_bytes(&to_bytes),
        rlp_encode_u256_be(&tx.value),
        rlp_encode_bytes(&tx.data),
        tx.access_list_rlp.clone(),
        rlp_encode_u64(tx.y_parity),
        rlp_encode_u256_be(&tx.r),
        rlp_encode_u256_be(&tx.s),
    ];

    let payload_len: usize = fields.iter().map(|f| f.len()).sum();
    let mut body = Vec::with_capacity(3 + payload_len);
    if payload_len < 56 {
        body.push(0xc0 + payload_len as u8);
    } else {
        let mut len_be = Vec::new();
        let mut n = payload_len;
        while n > 0 { len_be.push((n & 0xff) as u8); n >>= 8; }
        len_be.reverse();
        body.push(0xf7 + len_be.len() as u8);
        body.extend_from_slice(&len_be);
    }
    for f in &fields { body.extend_from_slice(f); }

    let mut assembled = Vec::with_capacity(1 + body.len());
    assembled.push(0x02);
    assembled.extend_from_slice(&body);

    if assembled != canonical {
        return Err(format!(
            "EIP-1559 tx RLP mismatch: assembled {} bytes vs canonical {} bytes",
            assembled.len(), canonical.len(),
        ));
    }
    Ok(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak::keccak256;

    fn sample_tx() -> LegacyTx {
        LegacyTx {
            nonce: 42,
            gas_price: { let mut b = [0u8; 32]; b[24..32].copy_from_slice(&20_000_000_000u64.to_be_bytes()); b },
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: { let mut b = [0u8; 32]; b[24..32].copy_from_slice(&1_000_000_000_000_000_000u64.to_be_bytes()); b },
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        }
    }

    #[test]
    fn legacy_tx_composition_matches_canonical() {
        let tx = sample_tx();
        let assembled = verify_legacy_tx_rlp_composition(&tx).unwrap();
        assert_eq!(assembled, tx.rlp_encode());
        assert_eq!(keccak256(&assembled), tx.hash());
    }

    #[test]
    fn contract_creation_tx_composition_matches() {
        let tx = LegacyTx {
            nonce: 0,
            gas_price: [0u8; 32],
            gas_limit: 100_000,
            to: None,
            value: [0u8; 32],
            data: vec![0x60, 0x80, 0x60, 0x40, 0x52],
            v: 27,
            r: [0x33u8; 32],
            s: [0x44u8; 32],
        };
        verify_legacy_tx_rlp_composition(&tx).unwrap();
    }

    #[test]
    fn eip1559_tx_composition_matches() {
        let tx = Eip1559Tx {
            chain_id: 1,
            nonce: 100,
            max_priority_fee_per_gas: { let mut b=[0u8;32]; b[31]=2; b },
            max_fee_per_gas: { let mut b=[0u8;32]; b[24..32].copy_from_slice(&30_000_000_000u64.to_be_bytes()); b },
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            access_list_rlp: vec![0xc0], // empty access list
            y_parity: 1,
            r: [0x55u8; 32],
            s: [0x66u8; 32],
        };
        let assembled = verify_eip1559_tx_rlp_composition(&tx).unwrap();
        assert_eq!(assembled, tx.wire_encoding());
        assert_eq!(crate::keccak::keccak256(&assembled), tx.hash());
    }

    #[test]
    fn zero_value_tx_composition_matches() {
        let tx = LegacyTx {
            nonce: 0, gas_price: [0u8; 32], gas_limit: 21_000,
            to: Some([0xab; 20]), value: [0u8; 32], data: Vec::new(),
            v: 0, r: [0u8; 32], s: [0u8; 32],
        };
        verify_legacy_tx_rlp_composition(&tx).unwrap();
    }
}
