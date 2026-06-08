//! Ethereum account state + world state MPT inclusion oracle.
//!
//! Phase A2 step 3 (#69) foundation. The world state MPT lives at the
//! block header's `state_root` and maps `keccak256(address)` → RLP-
//! encoded account state `(nonce, balance, storage_root, code_hash)`.
//! Once the EVM execution proves a storage access against some claimed
//! `storage_root`, this module's MPT chain proves that `storage_root`
//! is the one committed in the contract's account state inside the
//! block's `state_root`.
//!
//! Step 3 will then build a parallel gadget AIR (mirroring
//! `storage_access_air`) that commits to per-account state and links
//! to the existing MPT inclusion infrastructure.

use crate::keccak::keccak256;
use crate::rlp::{rlp_encode_bytes, rlp_encode_list, rlp_encode_u256, rlp_encode_uint};

/// Ethereum account state stored in the world state MPT at
/// `keccak256(address)`. RLP-encoded as a 4-item list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Account {
    /// Number of transactions sent from this account (or, for
    /// contracts, the number of contract-creations performed).
    pub nonce: u64,
    /// Account balance in wei, as 32 big-endian bytes.
    pub balance: [u8; 32],
    /// Root of the contract's storage trie (or empty-trie hash for
    /// EOAs).
    pub storage_root: [u8; 32],
    /// keccak256 of the contract's bytecode (or
    /// `keccak256(<empty>) = 0xc5d2…` for EOAs).
    pub code_hash: [u8; 32],
}

impl Default for Account {
    /// Zero-valued account suitable for tests; storage_root is the
    /// empty-MPT root, code_hash is `keccak256(<empty>)`.
    fn default() -> Self {
        Account {
            nonce: 0,
            balance: [0u8; 32],
            storage_root: empty_storage_root(),
            code_hash: empty_code_hash(),
        }
    }
}

/// The keccak256 hash of an RLP-encoded empty list = empty MPT root.
/// Per the Yellow Paper this is `0x56e81f17fc04c84a3b6f81b2c7ad7b6f7ad7a8d7e6f3e3f3e3a8d4...` —
/// hardcoded below as the standard value.
pub fn empty_storage_root() -> [u8; 32] {
    [
        0x56, 0xe8, 0x1f, 0x17, 0x1b, 0xcc, 0x55, 0xa6, 0xff, 0x83, 0x45,
        0xe6, 0x92, 0xc0, 0xf8, 0x6e, 0x5b, 0x48, 0xe0, 0x1b, 0x99, 0x6c,
        0xad, 0xc0, 0x01, 0x62, 0x2f, 0xb5, 0xe3, 0x63, 0xb4, 0x21,
    ]
}

/// keccak256 of empty bytes (used as default code_hash for EOAs).
pub fn empty_code_hash() -> [u8; 32] {
    keccak256(&[])
}

/// RLP-encode the account state as a 4-item list:
/// `[nonce, balance, storage_root, code_hash]`.
pub fn account_rlp(a: &Account) -> Vec<u8> {
    let items: Vec<Vec<u8>> = vec![
        rlp_encode_uint(a.nonce),
        rlp_encode_u256(&a.balance),
        rlp_encode_bytes(&a.storage_root),
        rlp_encode_bytes(&a.code_hash),
    ];
    rlp_encode_list(&items)
}

/// Compute the world state MPT key for an account address:
/// `trie_key = keccak256(address)`.
pub fn account_trie_key(address: &[u8; 20]) -> [u8; 32] {
    keccak256(address)
}

/// Verify an account inclusion proof against a state root. Returns
/// `Ok(())` if the proof verifies and the RLP-encoded account state
/// at `keccak256(address)` matches the supplied `account`.
pub fn verify_account_inclusion_oracle(
    state_root: [u8; 32],
    address: &[u8; 20],
    account: &Account,
    proof: &[Vec<u8>],
) -> Result<(), String> {
    let trie_key = account_trie_key(address);
    let account_rlp = account_rlp(account);
    if !crate::mpt::verify_mpt_inclusion(state_root, &trie_key, &account_rlp, proof) {
        return Err(format!(
            "Account MPT inclusion failed for address={:?} (trie_key={:?})",
            address, trie_key,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::single_leaf_trie;

    #[test]
    fn empty_storage_root_known_value() {
        // The empty MPT root in Ethereum is keccak256(rlp(b''))
        // = keccak256([0x80]) (empty STRING RLP, not empty list).
        // Standard value: 0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421.
        let want = keccak256(&[0x80]);
        assert_eq!(empty_storage_root(), want);
    }

    #[test]
    fn empty_code_hash_known_value() {
        // keccak256(<empty>) = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
        let expected: [u8; 32] = [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
            0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
            0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
            0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
        ];
        assert_eq!(empty_code_hash(), expected);
    }

    #[test]
    fn account_rlp_round_trip() {
        let account = Account {
            nonce: 5,
            balance: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x00,
            ],
            storage_root: [0xab; 32],
            code_hash: [0xcd; 32],
        };
        let bytes = account_rlp(&account);
        // RLP-decode back as a sanity check.
        let (item, consumed) = crate::rlp::rlp_decode(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        match item {
            crate::rlp::RlpItem::List(items) => {
                assert_eq!(items.len(), 4);
            }
            _ => panic!("expected RLP list"),
        }
    }

    #[test]
    fn account_inclusion_single_leaf_trie() {
        let address = [0xab; 20];
        let account = Account {
            nonce: 42,
            balance: [0u8; 32],
            storage_root: [0x11; 32],
            code_hash: [0x22; 32],
        };
        let trie_key = account_trie_key(&address);
        let account_bytes = account_rlp(&account);
        let (root, proof) = single_leaf_trie(&trie_key, &account_bytes);
        verify_account_inclusion_oracle(root, &address, &account, &proof).unwrap();
    }

    #[test]
    fn account_inclusion_rejects_tampered_storage_root() {
        let address = [0xab; 20];
        let mut account = Account {
            nonce: 42,
            balance: [0u8; 32],
            storage_root: [0x11; 32],
            code_hash: [0x22; 32],
        };
        let trie_key = account_trie_key(&address);
        let account_bytes = account_rlp(&account);
        let (root, proof) = single_leaf_trie(&trie_key, &account_bytes);
        // Tamper storage_root in the claimed account; the RLP no
        // longer matches what's in the trie.
        account.storage_root = [0x99; 32];
        let err = verify_account_inclusion_oracle(root, &address, &account, &proof)
            .unwrap_err();
        assert!(err.contains("Account MPT inclusion failed"), "got: {}", err);
    }

    #[test]
    fn account_trie_key_matches_keccak_of_address() {
        let address = [0xab; 20];
        let computed = account_trie_key(&address);
        let expected = keccak256(&address);
        assert_eq!(computed, expected);
    }

    #[test]
    fn default_account_has_empty_storage_and_code() {
        let a = Account::default();
        assert_eq!(a.nonce, 0);
        assert_eq!(a.balance, [0u8; 32]);
        assert_eq!(a.storage_root, empty_storage_root());
        assert_eq!(a.code_hash, empty_code_hash());
    }
}
