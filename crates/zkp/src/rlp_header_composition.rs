//! Block header RLP composition oracle.
//!
//! Decomposes a `BlockHeader` into 20 per-field RLP encodings using
//! the session's gadgets, concatenates them with the RLP list header,
//! and verifies the result matches the canonical `block_header_rlp()`.
//!
//! This is the **host-side spec** for the algebraic RLP list concat
//! gadget. The algebraic version will use the same per-field gadget
//! outputs + a byte-memory-style AIR for the concatenation.

use crate::block_header::BlockHeader;
use crate::fixed_rlp_air::RLP32_PREFIX;
use crate::fixed_rlp20_air::RLP20_PREFIX;
use crate::fixed_rlp8_air::RLP8_PREFIX;
use crate::rlp_logs_bloom_air::PREFIX_BYTES as BLOOM_PREFIX;
use crate::u64_rlp_air::rlp_encode_u64;
use crate::u256_rlp_air::rlp_encode_u256_be;
use crate::rlp_var_bytes_air::rlp_encode_bytes;

/// Encode a block header into RLP using our per-field gadgets, then
/// verify the result matches the canonical `block_header_rlp`.
///
/// Returns `Ok(encoded_bytes)` if the composition matches, or `Err`
/// with a diagnostic if any field encoding disagrees.
pub fn verify_header_rlp_composition(h: &BlockHeader) -> Result<Vec<u8>, String> {
    let canonical = crate::block_header::block_header_rlp(h);

    // Build per-field encodings using our gadgets.
    let mut fields: Vec<Vec<u8>> = Vec::with_capacity(20);

    // Fields 0-5: 32-byte hashes (prefix 0xa0).
    for (_name, bytes) in [
        ("parent_hash", &h.parent_hash[..]),
        ("ommers_hash", &h.ommers_hash[..]),
    ] {
        let mut enc = vec![RLP32_PREFIX];
        enc.extend_from_slice(bytes);
        fields.push(enc);
    }

    // Field 2: beneficiary (20 bytes, prefix 0x94).
    {
        let mut enc = vec![RLP20_PREFIX];
        enc.extend_from_slice(&h.beneficiary);
        fields.push(enc);
    }

    // Fields 3-5: more 32-byte hashes.
    for bytes in [&h.state_root[..], &h.transactions_root[..], &h.receipts_root[..]] {
        let mut enc = vec![RLP32_PREFIX];
        enc.extend_from_slice(bytes);
        fields.push(enc);
    }

    // Field 6: logs_bloom (256 bytes, long string prefix 0xb9 0x01 0x00).
    {
        let mut enc = Vec::with_capacity(259);
        enc.extend_from_slice(&BLOOM_PREFIX);
        enc.extend_from_slice(&h.logs_bloom);
        fields.push(enc);
    }

    // Field 7: difficulty (u256).
    fields.push(rlp_encode_u256_be(&h.difficulty));

    // Fields 8-11: u64 values.
    fields.push(rlp_encode_u64(h.number));
    fields.push(rlp_encode_u64(h.gas_limit));
    fields.push(rlp_encode_u64(h.gas_used));
    fields.push(rlp_encode_u64(h.timestamp));

    // Field 12: extra_data (variable bytes).
    fields.push(rlp_encode_bytes(&h.extra_data));

    // Field 13: mix_hash (32 bytes).
    {
        let mut enc = vec![RLP32_PREFIX];
        enc.extend_from_slice(&h.mix_hash);
        fields.push(enc);
    }

    // Field 14: nonce (8 bytes, prefix 0x88).
    {
        let mut enc = vec![RLP8_PREFIX];
        enc.extend_from_slice(&h.nonce);
        fields.push(enc);
    }

    // Optional post-fork fields (same monotonic nesting as block_header_rlp).
    if let Some(ref base_fee) = h.base_fee_per_gas {
        fields.push(rlp_encode_u256_be(base_fee));
        if let Some(ref wr) = h.withdrawals_root {
            let mut enc = vec![RLP32_PREFIX];
            enc.extend_from_slice(wr);
            fields.push(enc);
            if let Some(bgu) = h.blob_gas_used {
                fields.push(rlp_encode_u64(bgu));
                if let Some(ebg) = h.excess_blob_gas {
                    fields.push(rlp_encode_u64(ebg));
                    if let Some(ref pbbr) = h.parent_beacon_block_root {
                        let mut enc = vec![RLP32_PREFIX];
                        enc.extend_from_slice(pbbr);
                        fields.push(enc);
                    }
                }
            }
        }
    }

    // Assemble with RLP list header.
    let payload_len: usize = fields.iter().map(|f| f.len()).sum();
    let mut assembled = Vec::with_capacity(3 + payload_len);
    if payload_len < 56 {
        assembled.push(0xc0 + payload_len as u8);
    } else {
        let len_be = {
            let mut be = Vec::new();
            let mut n = payload_len;
            while n > 0 { be.push((n & 0xff) as u8); n >>= 8; }
            be.reverse();
            be
        };
        assembled.push(0xf7 + len_be.len() as u8);
        assembled.extend_from_slice(&len_be);
    }
    for f in &fields {
        assembled.extend_from_slice(f);
    }

    if assembled != canonical {
        return Err(format!(
            "RLP composition mismatch: assembled {} bytes vs canonical {} bytes",
            assembled.len(), canonical.len(),
        ));
    }
    Ok(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_header_composition_matches() {
        let h = BlockHeader::default();
        verify_header_rlp_composition(&h).unwrap();
    }

    #[test]
    fn cancun_header_composition_matches() {
        let h = BlockHeader {
            parent_hash: [0x11; 32],
            ommers_hash: [0x22; 32],
            beneficiary: [0x33; 20],
            state_root: [0x44; 32],
            transactions_root: [0x55; 32],
            receipts_root: [0x66; 32],
            logs_bloom: [0x77; 256],
            difficulty: [0; 32],
            number: 18_500_000,
            gas_limit: 30_000_000,
            gas_used: 15_000_000,
            timestamp: 1_700_000_000,
            extra_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            mix_hash: [0x88; 32],
            nonce: [0x00; 8],
            base_fee_per_gas: Some({
                let mut b = [0u8; 32];
                b[24..32].copy_from_slice(&15_000_000_000u64.to_be_bytes());
                b
            }),
            withdrawals_root: Some([0x99; 32]),
            blob_gas_used: Some(393_216),
            excess_blob_gas: Some(786_432),
            parent_beacon_block_root: Some([0xaa; 32]),
        };
        let result = verify_header_rlp_composition(&h).unwrap();
        assert_eq!(result, crate::block_header::block_header_rlp(&h));
        assert_eq!(
            crate::keccak::keccak256(&result),
            crate::block_header::block_header_hash(&h),
        );
    }

    #[test]
    fn pre_london_header_composition_matches() {
        let h = BlockHeader {
            number: 12_000_000,
            gas_limit: 15_000_000,
            gas_used: 10_000_000,
            timestamp: 1_620_000_000,
            ..Default::default()
        };
        verify_header_rlp_composition(&h).unwrap();
    }

    #[test]
    fn field_count_matches_block_header_rlp() {
        let h = BlockHeader {
            base_fee_per_gas: Some([0; 32]),
            withdrawals_root: Some([0; 32]),
            blob_gas_used: Some(0),
            excess_blob_gas: Some(0),
            parent_beacon_block_root: Some([0; 32]),
            ..Default::default()
        };
        let result = verify_header_rlp_composition(&h).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result, crate::block_header::block_header_rlp(&h));
    }
}
