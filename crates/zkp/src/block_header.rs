//! Ethereum execution-layer block header — reference RLP + hash.
//!
//! Per the Yellow Paper §4.3 and subsequent forks, the header is an RLP
//! list whose length has grown with each fork.  This module uses the
//! **post-Cancun (Dencun)** shape — 20 fields — which subsumes every
//! earlier one:
//!
//! |  # | field                        | since     |
//! |---:|------------------------------|-----------|
//! |  0 | parent_hash                  | Frontier  |
//! |  1 | ommers_hash                  | Frontier  |
//! |  2 | beneficiary (coinbase)       | Frontier  |
//! |  3 | state_root                   | Frontier  |
//! |  4 | transactions_root            | Frontier  |
//! |  5 | receipts_root                | Frontier  |
//! |  6 | logs_bloom (256 bytes)       | Frontier  |
//! |  7 | difficulty                   | Frontier  |
//! |  8 | number                       | Frontier  |
//! |  9 | gas_limit                    | Frontier  |
//! | 10 | gas_used                     | Frontier  |
//! | 11 | timestamp                    | Frontier  |
//! | 12 | extra_data                   | Frontier  |
//! | 13 | mix_hash (prev_randao)       | Frontier  |
//! | 14 | nonce (8 bytes)              | Frontier  |
//! | 15 | base_fee_per_gas             | London    |
//! | 16 | withdrawals_root             | Shanghai  |
//! | 17 | blob_gas_used                | Cancun    |
//! | 18 | excess_blob_gas              | Cancun    |
//! | 19 | parent_beacon_block_root     | Cancun    |
//!
//! Fields 15..19 are [`Option`]-typed so this struct can also represent
//! pre-London / pre-Shanghai / pre-Cancun headers (e.g. the genesis block,
//! which has only the first 15). The RLP encoder emits only the fields
//! that are present.

use crate::keccak::keccak256;
use crate::rlp::{rlp_encode_bytes, rlp_encode_list, rlp_encode_u256, rlp_encode_uint};

/// Length of the logs bloom filter in bytes (2048 bits).
pub const LOGS_BLOOM_LEN: usize = 256;

/// Execution-layer block header, post-Cancun shape (Options for the
/// post-Frontier fields so pre-fork headers can be represented too).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockHeader {
    pub parent_hash: [u8; 32],
    pub ommers_hash: [u8; 32],
    pub beneficiary: [u8; 20],
    pub state_root: [u8; 32],
    pub transactions_root: [u8; 32],
    pub receipts_root: [u8; 32],
    pub logs_bloom: [u8; LOGS_BLOOM_LEN],
    /// Difficulty, as a 32-byte big-endian unsigned integer.
    pub difficulty: [u8; 32],
    pub number: u64,
    pub gas_limit: u64,
    pub gas_used: u64,
    pub timestamp: u64,
    pub extra_data: Vec<u8>,
    pub mix_hash: [u8; 32],
    /// Always 8 bytes, even post-merge where it's fixed to zero.
    pub nonce: [u8; 8],
    /// Post-London (EIP-1559). Stored as 32-byte big-endian for consistency
    /// with other U256 fields; RLP-encoded as a minimal big-endian integer.
    pub base_fee_per_gas: Option<[u8; 32]>,
    /// Post-Shanghai (EIP-4895).
    pub withdrawals_root: Option<[u8; 32]>,
    /// Post-Cancun (EIP-4844).
    pub blob_gas_used: Option<u64>,
    /// Post-Cancun (EIP-4844).
    pub excess_blob_gas: Option<u64>,
    /// Post-Cancun (EIP-4788).
    pub parent_beacon_block_root: Option<[u8; 32]>,
}

impl Default for BlockHeader {
    /// Zero-valued header suitable for building a synthetic block in tests
    /// or as a scaffold to mutate via struct-update syntax
    /// (`BlockHeader { number: 1, ..Default::default() }`).
    ///
    /// All byte arrays are zero; `extra_data` is empty; all post-fork
    /// fields are `None`.
    fn default() -> Self {
        BlockHeader {
            parent_hash: [0u8; 32],
            ommers_hash: [0u8; 32],
            beneficiary: [0u8; 20],
            state_root: [0u8; 32],
            transactions_root: [0u8; 32],
            receipts_root: [0u8; 32],
            logs_bloom: [0u8; LOGS_BLOOM_LEN],
            difficulty: [0u8; 32],
            number: 0,
            gas_limit: 0,
            gas_used: 0,
            timestamp: 0,
            extra_data: Vec::new(),
            mix_hash: [0u8; 32],
            nonce: [0u8; 8],
            base_fee_per_gas: None,
            withdrawals_root: None,
            blob_gas_used: None,
            excess_blob_gas: None,
            parent_beacon_block_root: None,
        }
    }
}

/// RLP-encode a block header, emitting only the fields present per fork.
pub fn block_header_rlp(h: &BlockHeader) -> Vec<u8> {
    let mut items: Vec<Vec<u8>> = Vec::with_capacity(20);
    items.push(rlp_encode_bytes(&h.parent_hash));
    items.push(rlp_encode_bytes(&h.ommers_hash));
    items.push(rlp_encode_bytes(&h.beneficiary));
    items.push(rlp_encode_bytes(&h.state_root));
    items.push(rlp_encode_bytes(&h.transactions_root));
    items.push(rlp_encode_bytes(&h.receipts_root));
    items.push(rlp_encode_bytes(&h.logs_bloom));
    items.push(rlp_encode_u256(&h.difficulty));
    items.push(rlp_encode_uint(h.number));
    items.push(rlp_encode_uint(h.gas_limit));
    items.push(rlp_encode_uint(h.gas_used));
    items.push(rlp_encode_uint(h.timestamp));
    items.push(rlp_encode_bytes(&h.extra_data));
    items.push(rlp_encode_bytes(&h.mix_hash));
    items.push(rlp_encode_bytes(&h.nonce));

    // Optional post-fork fields are monotonic: if any one is present, all
    // prior ones must be too. Emit them in order, stopping at the first
    // `None`.
    if let Some(ref base_fee) = h.base_fee_per_gas {
        items.push(rlp_encode_u256(base_fee));
        if let Some(ref wr) = h.withdrawals_root {
            items.push(rlp_encode_bytes(wr));
            if let Some(bgu) = h.blob_gas_used {
                items.push(rlp_encode_uint(bgu));
                if let Some(ebg) = h.excess_blob_gas {
                    items.push(rlp_encode_uint(ebg));
                    if let Some(ref pbbr) = h.parent_beacon_block_root {
                        items.push(rlp_encode_bytes(pbbr));
                    }
                }
            }
        }
    }

    rlp_encode_list(&items)
}

/// keccak256 of the RLP-encoded header — the canonical block hash.
pub fn block_header_hash(h: &BlockHeader) -> [u8; 32] {
    keccak256(&block_header_rlp(h))
}

// -----------------------------------------------------------------------------
// Known constants
// -----------------------------------------------------------------------------

/// RLP of an empty list — the hash of the empty ommers list every
/// execution-layer block has used since Frontier.
pub fn empty_list_hash() -> [u8; 32] {
    // keccak256(rlp_encode_list(&[])) = keccak256(&[0xc0]).
    keccak256(&[0xc0])
}

/// Empty-trie root (used e.g. for transactions_root / receipts_root in
/// the genesis block).
pub fn empty_trie_root() -> [u8; 32] {
    crate::mpt::empty_trie_root()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hex-decode a 32-byte array from a 64-char string.
    fn h32(hex: &str) -> [u8; 32] {
        let hex = hex.trim_start_matches("0x");
        assert_eq!(hex.len(), 64);
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        out
    }
    fn h8(hex: &str) -> [u8; 8] {
        let hex = hex.trim_start_matches("0x");
        assert_eq!(hex.len(), 16);
        let mut out = [0u8; 8];
        for i in 0..8 {
            out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        out
    }

    /// Ethereum mainnet genesis block (block #0).
    ///
    /// Canonical hash:
    ///   `0xd4e56740f876aef8c010b86a40d5f56745a118d0906a34e69aec8c0db1cb8fa3`
    ///
    /// Field values cross-referenced against go-ethereum's
    /// `params/genesis.go` + the hard-coded values geth uses to produce this
    /// hash. The 15-field (pre-London) layout is what the genesis block
    /// actually encodes on-chain.
    fn mainnet_genesis() -> BlockHeader {
        // Zero roots except state_root, which is the big allocation-hash from
        // genesis.
        let empty_uncles = h32("1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347");
        // transactions and receipts roots in the genesis block are the
        // empty-trie root (no txs, no receipts).
        let empty_trie = h32("56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421");
        // Well-known mainnet genesis state root.
        let state_root = h32("d7f8974fb5ac78d9ac099b9ad5018bedc2ce0a72dad1827a1709da30580f0544");

        // Extra data is the famous banker bailout headline.
        // 00 .. 00 11 bnic 11 bank bailout... (geth encodes this as a
        // specific 32-byte string). The canonical genesis extra_data:
        // "11bbe8db4e347b4e8c937c1c8370e4b5ed33adb3db69cbdb7a38e1e50b1b82fa"
        // NOTE: mainnet genesis extra_data is actually 32 bytes:
        let extra_data = hex_decode(
            "11bbe8db4e347b4e8c937c1c8370e4b5ed33adb3db69cbdb7a38e1e50b1b82fa",
        );

        BlockHeader {
            parent_hash: [0u8; 32],
            ommers_hash: empty_uncles,
            beneficiary: [0u8; 20],
            state_root,
            transactions_root: empty_trie,
            receipts_root: empty_trie,
            logs_bloom: [0u8; LOGS_BLOOM_LEN],
            difficulty: {
                let mut d = [0u8; 32];
                // 0x400000000 = 17179869184
                d[32 - 5..].copy_from_slice(&[0x04, 0x00, 0x00, 0x00, 0x00]);
                d
            },
            number: 0,
            gas_limit: 0x1388, // 5000
            gas_used: 0,
            timestamp: 0,
            extra_data,
            mix_hash: [0u8; 32],
            nonce: h8("0000000000000042"),
            base_fee_per_gas: None,
            withdrawals_root: None,
            blob_gas_used: None,
            excess_blob_gas: None,
            parent_beacon_block_root: None,
        }
    }

    fn hex_decode(hex: &str) -> Vec<u8> {
        let hex = hex.trim_start_matches("0x");
        assert!(hex.len() % 2 == 0);
        (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn test_empty_list_hash_is_ommers_hash() {
        // Standard empty uncles hash = keccak256(rlp([])) = 0x1dcc4de8...d49347
        let expected = h32("1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347");
        assert_eq!(empty_list_hash(), expected);
    }

    #[test]
    fn test_empty_trie_root_constant() {
        // Standard Ethereum empty-trie root.
        let expected = h32("56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421");
        assert_eq!(empty_trie_root(), expected);
    }

    /// Cross-check with mainnet genesis hash
    /// `0xd4e56740f876aef8c010b86a40d5f56745a118d0906a34e69aec8c0db1cb8fa3`.
    /// This is the hash every Ethereum client agrees on for block #0.
    #[test]
    fn test_mainnet_genesis_hash() {
        let header = mainnet_genesis();
        let got = block_header_hash(&header);
        let expected = h32("d4e56740f876aef8c010b86a40d5f56745a118d0906a34e69aec8c0db1cb8fa3");
        assert_eq!(
            got, expected,
            "mainnet genesis hash mismatch: got 0x{}, expected 0x{}",
            hex_encode(&got),
            hex_encode(&expected)
        );
    }

    fn hex_encode(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for &x in b {
            s.push_str(&format!("{:02x}", x));
        }
        s
    }

    /// Round-trip: encode + re-encode produces identical bytes.
    #[test]
    fn test_rlp_deterministic() {
        let header = mainnet_genesis();
        let a = block_header_rlp(&header);
        let b = block_header_rlp(&header);
        assert_eq!(a, b);
    }

    /// Post-London header: base_fee_per_gas present, withdrawals/blob fields
    /// absent. Confirms the optional-field chain stops at the first `None`.
    #[test]
    fn test_london_header_rlp_has_16_items() {
        let mut h = mainnet_genesis();
        let mut base = [0u8; 32];
        base[31] = 1; // base fee = 1 wei
        h.base_fee_per_gas = Some(base);
        let enc = block_header_rlp(&h);
        // Decode and check the outer list has exactly 16 items.
        use crate::rlp::{rlp_decode, RlpItem};
        let (item, n) = rlp_decode(&enc).unwrap();
        assert_eq!(n, enc.len());
        match item {
            RlpItem::List(items) => assert_eq!(items.len(), 16),
            _ => panic!("expected list"),
        }
    }

    /// Post-Cancun header: all 20 fields present.
    #[test]
    fn test_cancun_header_rlp_has_20_items() {
        let mut h = mainnet_genesis();
        let mut base = [0u8; 32];
        base[31] = 1;
        h.base_fee_per_gas = Some(base);
        h.withdrawals_root = Some([0u8; 32]);
        h.blob_gas_used = Some(0);
        h.excess_blob_gas = Some(0);
        h.parent_beacon_block_root = Some([0u8; 32]);
        let enc = block_header_rlp(&h);
        use crate::rlp::{rlp_decode, RlpItem};
        let (item, n) = rlp_decode(&enc).unwrap();
        assert_eq!(n, enc.len());
        match item {
            RlpItem::List(items) => assert_eq!(items.len(), 20),
            _ => panic!("expected list"),
        }
    }

    /// Tamper detection: flipping any single byte of the RLP → different hash.
    #[test]
    fn test_hash_changes_on_tamper() {
        let mut h = mainnet_genesis();
        let orig = block_header_hash(&h);
        h.number = 1;
        let new = block_header_hash(&h);
        assert_ne!(orig, new);
    }
}
