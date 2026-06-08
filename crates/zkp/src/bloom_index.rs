//! Bloom filter bit index extraction.
//!
//! For an item, computes the 3 bit positions it would set in the
//! Ethereum logsBloom filter (M3:2048 - 3 bits over 2048 total bits).
//! Per the Yellow Paper: each item's keccak256 hash provides 3 pairs
//! of bytes; each pair (after masking to 11 bits) gives one bit
//! position 0..2047.

use crate::keccak::keccak256;

/// Compute the 3 bit positions an item sets in a 2048-bit bloom filter.
pub fn bloom_bit_positions(item: &[u8]) -> [u16; 3] {
    let hash = keccak256(item);
    let mut positions = [0u16; 3];
    for i in 0..3 {
        let pair = ((hash[2 * i] as u16) << 8) | (hash[2 * i + 1] as u16);
        positions[i] = pair & 0x07FF; // mask to 11 bits (0..2047)
    }
    positions
}

/// Check if all three bloom bit positions an item maps to are set in the bloom.
pub fn bloom_contains_item(bloom: &[u8; 256], item: &[u8]) -> bool {
    let positions = bloom_bit_positions(item);
    for pos in positions {
        let byte_idx = 255 - (pos as usize / 8);
        let bit_idx = pos as usize % 8;
        if (bloom[byte_idx] >> bit_idx) & 1 == 0 {
            return false;
        }
    }
    true
}

/// Verify that the bit positions are consistent with the standard bloom oracle.
pub fn verify_bit_positions_match_bloom(item: &[u8]) -> Result<(), String> {
    let bloom = crate::bloom::bloom_m3_2048(item);
    let positions = bloom_bit_positions(item);
    let mut bit_count = 0;
    for pos in positions {
        let byte_idx = 255 - (pos as usize / 8);
        let bit_idx = pos as usize % 8;
        if (bloom[byte_idx] >> bit_idx) & 1 == 0 {
            return Err(format!(
                "bit position {} not set in bloom (byte {}, bit {})",
                pos, byte_idx, bit_idx,
            ));
        }
        bit_count += 1;
    }
    if bit_count != 3 {
        return Err(format!("expected 3 bits set, got {}", bit_count));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_positions_returned() {
        let positions = bloom_bit_positions(b"hello");
        assert_eq!(positions.len(), 3);
        for p in positions {
            assert!(p < 2048, "position {} out of range", p);
        }
    }

    #[test]
    fn positions_match_bloom() {
        verify_bit_positions_match_bloom(b"test").unwrap();
        verify_bit_positions_match_bloom(&[0x42u8; 20]).unwrap();
        verify_bit_positions_match_bloom(&[0xAAu8; 32]).unwrap();
    }

    #[test]
    fn contains_self() {
        let item = b"some_event";
        let bloom = crate::bloom::bloom_m3_2048(item);
        assert!(bloom_contains_item(&bloom, item));
    }

    #[test]
    fn empty_bloom_no_item() {
        let bloom = [0u8; 256];
        assert!(!bloom_contains_item(&bloom, b"anything"));
    }

    #[test]
    fn different_items_different_positions() {
        let p1 = bloom_bit_positions(b"item1");
        let p2 = bloom_bit_positions(b"item2");
        assert_ne!(p1, p2);
    }
}
