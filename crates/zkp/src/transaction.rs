//! EVM transaction RLP + hash (Phase A2 #58 step 0).
//!
//! Every EVM transaction has a canonical wire encoding that's hashed
//! to produce `txHash` (committed in the block's `transactionsRoot`
//! MPT at key `rlp(tx_index)`). Post-EIP-2718, typed transactions
//! prefix the encoding with a type byte (0x01, 0x02, 0x03) before the
//! RLP body.
//!
//! Transaction types covered:
//! - **Legacy** (pre-EIP-2718): no type prefix. RLP body =
//!   `[nonce, gasPrice, gasLimit, to, value, data, v, r, s]`.
//! - **EIP-2930** (0x01, access list): RLP body =
//!   `[chainId, nonce, gasPrice, gasLimit, to, value, data, accessList, yParity, r, s]`.
//! - **EIP-1559** (0x02): RLP body =
//!   `[chainId, nonce, maxPriorityFeePerGas, maxFeePerGas, gasLimit, to, value, data, accessList, yParity, r, s]`.
//! - **EIP-4844** (0x03, blob): RLP body =
//!   `[chainId, nonce, maxPriorityFeePerGas, maxFeePerGas, gasLimit, to, value, data, accessList, maxFeePerBlobGas, blobVersionedHashes, yParity, r, s]`.
//!
//! This step-0 module defines the data shape + RLP encoding + hash
//! computation. The algebraic chain (#58 step 1+) closes:
//! - EVM trace fields (`origin`, `gasprice`, `value`, calldata) ↔ Tx RLP via gadget
//! - Tx RLP ↔ keccak256 via KeccakExtract
//! - Tx hash ↔ transactionsRoot MPT via MPT inclusion AIR
//! - transactionsRoot ↔ blockHash via BlockHeader AIR (#53 dependency)

use crate::keccak::keccak256;
use crate::rlp::{rlp_encode_bytes, rlp_encode_list, rlp_encode_u256, rlp_encode_uint};

/// EIP-2718 transaction type marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxType {
    Legacy,
    AccessList,   // 0x01 (EIP-2930)
    Eip1559,      // 0x02
    Eip4844,      // 0x03
}

impl TxType {
    pub fn type_byte(self) -> Option<u8> {
        match self {
            TxType::Legacy => None,
            TxType::AccessList => Some(0x01),
            TxType::Eip1559 => Some(0x02),
            TxType::Eip4844 => Some(0x03),
        }
    }
}

/// Legacy (pre-EIP-2718) transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyTx {
    pub nonce: u64,
    /// Gas price as 32-byte big-endian U256.
    pub gas_price: [u8; 32],
    pub gas_limit: u64,
    /// Recipient address (20 bytes). For contract creation, use an
    /// empty slice in RLP — represented here as `None`.
    pub to: Option<[u8; 20]>,
    /// Value as 32-byte big-endian U256.
    pub value: [u8; 32],
    pub data: Vec<u8>,
    /// Signature v (chain-id-encoded post-EIP-155).
    pub v: u64,
    pub r: [u8; 32],
    pub s: [u8; 32],
}

impl LegacyTx {
    /// RLP-encode the legacy transaction (this IS the wire format).
    pub fn rlp_encode(&self) -> Vec<u8> {
        let to_bytes = self.to.map(|a| a.to_vec()).unwrap_or_default();
        rlp_encode_list(&[
            rlp_encode_uint(self.nonce),
            rlp_encode_u256(&self.gas_price),
            rlp_encode_uint(self.gas_limit),
            rlp_encode_bytes(&to_bytes),
            rlp_encode_u256(&self.value),
            rlp_encode_bytes(&self.data),
            rlp_encode_uint(self.v),
            rlp_encode_u256(&self.r),
            rlp_encode_u256(&self.s),
        ])
    }

    /// keccak256 of the wire encoding = txHash.
    pub fn hash(&self) -> [u8; 32] {
        keccak256(&self.rlp_encode())
    }
}

/// EIP-1559 transaction (type 0x02). This is the dominant post-London
/// transaction type today.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eip1559Tx {
    pub chain_id: u64,
    pub nonce: u64,
    /// Tip (priority fee) as 32-byte BE U256.
    pub max_priority_fee_per_gas: [u8; 32],
    /// Max total fee as 32-byte BE U256.
    pub max_fee_per_gas: [u8; 32],
    pub gas_limit: u64,
    pub to: Option<[u8; 20]>,
    pub value: [u8; 32],
    pub data: Vec<u8>,
    /// Access list (EIP-2930). Each entry: (address, list of storage keys).
    /// For step 0 represented as a simple Vec; structured access-list
    /// encoding can be added in a follow-up.
    pub access_list_rlp: Vec<u8>,
    pub y_parity: u64,
    pub r: [u8; 32],
    pub s: [u8; 32],
}

impl Eip1559Tx {
    /// RLP-encode the body (without the type-byte prefix).
    pub fn rlp_encode_body(&self) -> Vec<u8> {
        let to_bytes = self.to.map(|a| a.to_vec()).unwrap_or_default();
        rlp_encode_list(&[
            rlp_encode_uint(self.chain_id),
            rlp_encode_uint(self.nonce),
            rlp_encode_u256(&self.max_priority_fee_per_gas),
            rlp_encode_u256(&self.max_fee_per_gas),
            rlp_encode_uint(self.gas_limit),
            rlp_encode_bytes(&to_bytes),
            rlp_encode_u256(&self.value),
            rlp_encode_bytes(&self.data),
            self.access_list_rlp.clone(),
            rlp_encode_uint(self.y_parity),
            rlp_encode_u256(&self.r),
            rlp_encode_u256(&self.s),
        ])
    }

    /// Wire encoding: `0x02 || rlp_encode_body()`.
    pub fn wire_encoding(&self) -> Vec<u8> {
        let body = self.rlp_encode_body();
        let mut out = Vec::with_capacity(1 + body.len());
        out.push(0x02);
        out.extend_from_slice(&body);
        out
    }

    /// keccak256 of the wire encoding = txHash.
    pub fn hash(&self) -> [u8; 32] {
        keccak256(&self.wire_encoding())
    }
}

/// Unified transaction enum. For step 0, only Legacy + EIP-1559 are
/// fully supported; AccessList (EIP-2930) and Eip4844 follow the same
/// pattern but with different field sets (deferred to follow-up).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transaction {
    Legacy(LegacyTx),
    Eip1559(Eip1559Tx),
}

impl Transaction {
    pub fn ty(&self) -> TxType {
        match self {
            Transaction::Legacy(_) => TxType::Legacy,
            Transaction::Eip1559(_) => TxType::Eip1559,
        }
    }

    pub fn wire_encoding(&self) -> Vec<u8> {
        match self {
            Transaction::Legacy(tx) => tx.rlp_encode(),
            Transaction::Eip1559(tx) => tx.wire_encoding(),
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        keccak256(&self.wire_encoding())
    }
}

/// Compute the transactionsRoot-MPT trie key for a transaction at
/// `index` in the block: `trie_key = rlp(index)`.
///
/// Ethereum's transactionsRoot is a Merkle Patricia Trie mapping
/// `rlp(uint(tx_index))` to `wire_encoding(tx)`.
pub fn tx_trie_key(index: u64) -> Vec<u8> {
    rlp_encode_uint(index)
}

/// Host-side oracle: verify a transaction inclusion proof against a
/// claimed `transactions_root` from the block header. Returns
/// `Ok(())` if `wire_encoding(tx)` is correctly placed at the MPT key
/// `rlp(tx_index)` under `transactions_root`.
///
/// **Phase A2 #58 step 0 oracle**: the algebraic chain (step 1+) will
/// commit this binding via a Transaction gadget AIR + MPT inclusion
/// AIR + KeccakExtract.
pub fn verify_transaction_inclusion_oracle(
    transactions_root: [u8; 32],
    tx_index: u64,
    tx: &Transaction,
    proof: &[Vec<u8>],
) -> Result<(), String> {
    let trie_key = tx_trie_key(tx_index);
    let wire = tx.wire_encoding();
    if !crate::mpt::verify_mpt_inclusion(transactions_root, &trie_key, &wire, proof) {
        return Err(format!(
            "Transaction MPT inclusion failed for tx_index={} (trie_key={:?})",
            tx_index, trie_key,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod inclusion_tests {
    use super::*;
    use crate::mpt::single_leaf_trie;

    fn make_simple_legacy() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: 7,
            gas_price: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x05,
            ],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        })
    }

    #[test]
    fn tx_trie_key_uses_rlp_of_index() {
        // tx_index = 0 → RLP = [0x80] (empty string per RLP convention
        // for zero).
        assert_eq!(tx_trie_key(0), vec![0x80]);
        // tx_index = 1 → RLP = [0x01].
        assert_eq!(tx_trie_key(1), vec![0x01]);
        // tx_index = 0x7f → RLP = [0x7f].
        assert_eq!(tx_trie_key(0x7f), vec![0x7f]);
        // tx_index = 0x80 → RLP = [0x81, 0x80].
        assert_eq!(tx_trie_key(0x80), vec![0x81, 0x80]);
    }

    #[test]
    fn transaction_inclusion_single_leaf_trie() {
        let tx = make_simple_legacy();
        let tx_index = 5u64;
        let trie_key = tx_trie_key(tx_index);
        let wire = tx.wire_encoding();
        let (root, proof) = single_leaf_trie(&trie_key, &wire);
        verify_transaction_inclusion_oracle(root, tx_index, &tx, &proof).unwrap();
    }

    #[test]
    fn transaction_inclusion_rejects_wrong_index() {
        let tx = make_simple_legacy();
        let trie_key = tx_trie_key(5);
        let wire = tx.wire_encoding();
        let (root, proof) = single_leaf_trie(&trie_key, &wire);
        // Verify against a different index — proof won't match.
        let err = verify_transaction_inclusion_oracle(root, 99, &tx, &proof)
            .unwrap_err();
        assert!(err.contains("Transaction MPT inclusion failed"), "got: {}", err);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_legacy() -> LegacyTx {
        LegacyTx {
            nonce: 7,
            gas_price: u64_to_be32(20_000_000_000), // 20 gwei
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: u64_to_be32(1_000_000_000_000_000_000), // 1 ETH
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        }
    }

    fn sample_eip1559() -> Eip1559Tx {
        Eip1559Tx {
            chain_id: 1,
            nonce: 7,
            max_priority_fee_per_gas: u64_to_be32(1_500_000_000), // 1.5 gwei
            max_fee_per_gas: u64_to_be32(30_000_000_000),         // 30 gwei
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: u64_to_be32(1_000_000_000_000_000_000),
            data: Vec::new(),
            // Empty access list = empty RLP list = [0xc0]
            access_list_rlp: vec![0xc0],
            y_parity: 0,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        }
    }

    fn u64_to_be32(n: u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[24..].copy_from_slice(&n.to_be_bytes());
        out
    }

    #[test]
    fn legacy_tx_wire_is_rlp_list() {
        let tx = sample_legacy();
        let wire = tx.rlp_encode();
        assert!(wire[0] >= 0xc0, "legacy tx wire starts with RLP list header");
    }

    #[test]
    fn eip1559_tx_wire_starts_with_type_byte_02() {
        let tx = sample_eip1559();
        let wire = tx.wire_encoding();
        assert_eq!(wire[0], 0x02);
        // The remainder is an RLP list.
        assert!(wire[1] >= 0xc0, "after type byte, body is RLP list");
    }

    #[test]
    fn tx_hash_is_deterministic() {
        let tx = sample_legacy();
        let h1 = tx.hash();
        let h2 = tx.hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn legacy_and_eip1559_distinct_hashes() {
        let h1 = sample_legacy().hash();
        let mut e = sample_eip1559();
        e.nonce = 7;
        let h2 = e.hash();
        // Wire formats differ entirely (legacy has no type prefix,
        // EIP-1559 starts with 0x02), so hashes must differ.
        assert_ne!(h1, h2);
    }

    #[test]
    fn contract_creation_tx_uses_empty_to() {
        let mut tx = sample_legacy();
        tx.to = None;
        let wire = tx.rlp_encode();
        // Contract creation: `to` is encoded as the empty byte string (0x80).
        // Find the position after the list header.
        // We just verify the wire is well-formed and distinct from
        // the version with a `to` address.
        let with_to = sample_legacy().rlp_encode();
        assert_ne!(wire, with_to);
    }

    #[test]
    fn transaction_enum_dispatch() {
        let legacy = Transaction::Legacy(sample_legacy());
        let eip1559 = Transaction::Eip1559(sample_eip1559());
        assert_eq!(legacy.ty(), TxType::Legacy);
        assert_eq!(eip1559.ty(), TxType::Eip1559);
        let h1 = legacy.hash();
        let h2 = eip1559.hash();
        assert_ne!(h1, h2);
    }

    /// Different `nonce` values yield distinct hashes — sanity check.
    #[test]
    fn distinct_nonces_distinct_hashes() {
        let tx1 = sample_legacy();
        let mut tx2 = tx1.clone();
        tx2.nonce = 8;
        assert_ne!(tx1.hash(), tx2.hash());
    }
}
