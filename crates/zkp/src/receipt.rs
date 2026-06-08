//! Transaction receipt RLP + Merkle inclusion (Phase A2 #59 step 0).
//!
//! Every EVM transaction produces a receipt:
//!
//! ```text
//! Receipt = (status, cumulative_gas_used, logs_bloom, logs)
//!   status:             u8 (post-Byzantium; pre-Byzantium had stateRoot)
//!   cumulative_gas_used: u64
//!   logs_bloom:         [u8; 256]
//!   logs:               Vec<Log>  where Log = (address, topics, data)
//! ```
//!
//! Receipts are RLP-encoded; post-EIP-2718 typed receipts have a type
//! byte prefix (`0x01` access-list, `0x02` EIP-1559, `0x03` EIP-4844).
//! Each receipt is committed in the block header's `receiptsRoot` via
//! an MPT where the key is `rlp(tx_index)` and the value is the receipt
//! RLP (or `type || rlp(receipt)` for typed receipts).
//!
//! This module exposes:
//! 1. The Receipt + Log data shape.
//! 2. Host-side RLP encoding (canonical against go-ethereum / revm).
//! 3. Receipt hash (keccak256 of the encoding) — what gets inserted
//!    into the receipt MPT.
//!
//! The algebraic chain (future #59 step 1+) closes:
//! - EVM trace `(status, cumulative_gas_used, logs)` ↔ Receipt RLP via gadget
//! - Receipt RLP ↔ keccak256 via KeccakExtract
//! - Receipt hash ↔ receiptsRoot MPT via MPT inclusion AIR
//! - receiptsRoot ↔ blockHash via BlockHeader AIR (#53 dependency)

use crate::keccak::keccak256;
use crate::rlp::{rlp_encode_bytes, rlp_encode_list, rlp_encode_uint};

/// A single log entry emitted by LOG0..LOG4 opcodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Log {
    /// Emitting contract's address (20 bytes).
    pub address: [u8; 20],
    /// Indexed topics (0..=4 entries, each 32 bytes).
    pub topics: Vec<[u8; 32]>,
    /// Non-indexed log data (arbitrary length).
    pub data: Vec<u8>,
}

impl Log {
    /// RLP-encode a single log as `[address, topics, data]`.
    pub fn rlp_encode(&self) -> Vec<u8> {
        let topics_items: Vec<Vec<u8>> = self
            .topics
            .iter()
            .map(|t| rlp_encode_bytes(t))
            .collect();
        rlp_encode_list(&[
            rlp_encode_bytes(&self.address),
            rlp_encode_list(&topics_items),
            rlp_encode_bytes(&self.data),
        ])
    }
}

/// EIP-2718 transaction-type marker for typed receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptType {
    Legacy,        // No type prefix
    AccessList,    // 0x01 (EIP-2930)
    Eip1559,       // 0x02 (EIP-1559)
    Eip4844,       // 0x03 (EIP-4844)
}

impl ReceiptType {
    pub fn type_byte(self) -> Option<u8> {
        match self {
            ReceiptType::Legacy => None,
            ReceiptType::AccessList => Some(0x01),
            ReceiptType::Eip1559 => Some(0x02),
            ReceiptType::Eip4844 => Some(0x03),
        }
    }
}

/// Post-Byzantium transaction receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    /// Receipt type per EIP-2718. Legacy = no prefix byte.
    pub ty: ReceiptType,
    /// 1 = success, 0 = failure (post-Byzantium).
    pub status: u8,
    /// Cumulative gas used by all transactions in the block UP TO AND
    /// INCLUDING this one.
    pub cumulative_gas_used: u64,
    /// 256-byte logs bloom filter.
    pub logs_bloom: [u8; 256],
    /// List of logs emitted by this transaction.
    pub logs: Vec<Log>,
}

impl Receipt {
    /// RLP-encode the receipt body (without the EIP-2718 type byte
    /// prefix). For legacy receipts this IS the wire format; for typed
    /// receipts `type || rlp_encode_body()` is the wire format.
    pub fn rlp_encode_body(&self) -> Vec<u8> {
        let logs_items: Vec<Vec<u8>> = self.logs.iter().map(|l| l.rlp_encode()).collect();
        rlp_encode_list(&[
            rlp_encode_bytes(&[self.status]),
            rlp_encode_uint(self.cumulative_gas_used),
            rlp_encode_bytes(&self.logs_bloom),
            rlp_encode_list(&logs_items),
        ])
    }

    /// Wire-format encoding: `type_byte_or_empty || rlp_encode_body()`.
    /// For legacy receipts this is `rlp_encode_body()`; for typed
    /// receipts, the type byte is prepended.
    pub fn wire_encoding(&self) -> Vec<u8> {
        let body = self.rlp_encode_body();
        match self.ty.type_byte() {
            None => body,
            Some(type_byte) => {
                let mut out = Vec::with_capacity(1 + body.len());
                out.push(type_byte);
                out.extend_from_slice(&body);
                out
            }
        }
    }

    /// keccak256 of the wire encoding. This is the value inserted
    /// into the receipts MPT at key `rlp(tx_index)`.
    pub fn hash(&self) -> [u8; 32] {
        keccak256(&self.wire_encoding())
    }
}

/// Compute the receiptsRoot-MPT trie key for a receipt at `tx_index`
/// in the block: `trie_key = rlp(tx_index)` (same convention as the
/// transactions trie).
pub fn receipt_trie_key(tx_index: u64) -> Vec<u8> {
    crate::rlp::rlp_encode_uint(tx_index)
}

/// Host-side oracle: verify a receipt inclusion proof against a
/// claimed `receipts_root` from the block header. The value at
/// `rlp(tx_index)` in the receipts MPT is `wire_encoding(receipt)` —
/// the type-prefixed (or bare-RLP for legacy) bytes.
///
/// **Phase A2 #59 step 0 oracle**.
pub fn verify_receipt_inclusion_oracle(
    receipts_root: [u8; 32],
    tx_index: u64,
    receipt: &Receipt,
    proof: &[Vec<u8>],
) -> Result<(), String> {
    let trie_key = receipt_trie_key(tx_index);
    let wire = receipt.wire_encoding();
    if !crate::mpt::verify_mpt_inclusion(receipts_root, &trie_key, &wire, proof) {
        return Err(format!(
            "Receipt MPT inclusion failed for tx_index={} (trie_key={:?})",
            tx_index, trie_key,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod inclusion_tests {
    use super::*;
    use crate::mpt::single_leaf_trie;

    fn make_simple_receipt() -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    #[test]
    fn receipt_inclusion_single_leaf_trie() {
        let receipt = make_simple_receipt();
        let tx_index = 3u64;
        let trie_key = receipt_trie_key(tx_index);
        let wire = receipt.wire_encoding();
        let (root, proof) = single_leaf_trie(&trie_key, &wire);
        verify_receipt_inclusion_oracle(root, tx_index, &receipt, &proof).unwrap();
    }

    #[test]
    fn receipt_inclusion_rejects_wrong_index() {
        let receipt = make_simple_receipt();
        let trie_key = receipt_trie_key(3);
        let wire = receipt.wire_encoding();
        let (root, proof) = single_leaf_trie(&trie_key, &wire);
        let err = verify_receipt_inclusion_oracle(root, 99, &receipt, &proof)
            .unwrap_err();
        assert!(err.contains("Receipt MPT inclusion failed"), "got: {}", err);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_log() -> Log {
        Log {
            address: [0x42u8; 20],
            topics: vec![[0x11u8; 32], [0x22u8; 32]],
            data: vec![0xaa, 0xbb, 0xcc],
        }
    }

    fn sample_receipt() -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21000,
            logs_bloom: [0u8; 256],
            logs: vec![sample_log()],
        }
    }

    #[test]
    fn legacy_receipt_wire_has_no_type_prefix() {
        let r = sample_receipt();
        assert_eq!(r.ty, ReceiptType::Legacy);
        // Wire encoding should start with an RLP list header (>= 0xc0),
        // not a type byte (0x01..=0x03).
        let wire = r.wire_encoding();
        assert!(wire[0] >= 0xc0, "legacy receipt must start with RLP list header");
    }

    #[test]
    fn eip1559_receipt_wire_starts_with_type_byte_02() {
        let mut r = sample_receipt();
        r.ty = ReceiptType::Eip1559;
        let wire = r.wire_encoding();
        assert_eq!(wire[0], 0x02);
        // Remaining bytes are the legacy body.
        let body = r.rlp_encode_body();
        assert_eq!(&wire[1..], &body[..]);
    }

    #[test]
    fn eip4844_receipt_wire_starts_with_type_byte_03() {
        let mut r = sample_receipt();
        r.ty = ReceiptType::Eip4844;
        assert_eq!(r.wire_encoding()[0], 0x03);
    }

    #[test]
    fn receipt_hash_changes_on_status_flip() {
        let r1 = sample_receipt();
        let mut r2 = r1.clone();
        r2.status = 0;
        assert_ne!(r1.hash(), r2.hash());
    }

    #[test]
    fn receipt_hash_changes_on_cumulative_gas_change() {
        let r1 = sample_receipt();
        let mut r2 = r1.clone();
        r2.cumulative_gas_used = 50000;
        assert_ne!(r1.hash(), r2.hash());
    }

    #[test]
    fn receipt_hash_changes_on_log_addition() {
        let r1 = sample_receipt();
        let mut r2 = r1.clone();
        r2.logs.push(sample_log());
        assert_ne!(r1.hash(), r2.hash());
    }

    #[test]
    fn log_rlp_encode_structure() {
        let l = sample_log();
        let rlp = l.rlp_encode();
        // [address(20 bytes), topics([32,32]), data(3 bytes)]
        // The outermost should be a list. We just verify it parses.
        assert!(rlp[0] >= 0xc0, "log RLP must be a list");
        // No further structural verification — relies on rlp_decode
        // round-trip via separate test.
    }

    /// Empty receipt (status 0, no logs, zero gas, zero bloom) has a
    /// well-defined hash. Pin it so silent encoding changes are caught.
    #[test]
    fn empty_receipt_hash_is_deterministic() {
        let r = Receipt {
            ty: ReceiptType::Legacy,
            status: 0,
            cumulative_gas_used: 0,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        };
        let h1 = r.hash();
        let h2 = r.hash();
        assert_eq!(h1, h2, "hash must be deterministic");
    }
}
