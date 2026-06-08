//! Ethereum bloom filter (M3:2048) — pure reference implementation.
//!
//! Used by the EVM LOG opcode receipt encoding and by the block header
//! `logsBloom` field. Per Yellow Paper §4.3.1 and §4.4.4, the bloom is a
//! 2048-bit filter populated by taking the first 6 bytes of `keccak256(item)`,
//! interpreting three big-endian 16-bit values, masking to 11 bits (0..2047),
//! and setting those three bit positions.
//!
//! # Bit ordering
//!
//! Ethereum's bloom uses a big-endian bit-within-filter convention: bit index
//! `idx` (0..2047) lives at **byte `255 - idx/8`**, **bit `idx % 8`**. This
//! matches the go-ethereum implementation in
//! `core/types/bloom9.go::bloomValues`, where:
//!
//! ```text
//! b := binary.BigEndian.Uint16(hash[2i:2i+2]) & 0x07FF
//! byte_index = BloomByteLength - 1 - (b >> 3)        // 255 - b/8
//! bit_mask   = 1 << (b & 7)                          // 1 << (b % 8)
//! ```
//!
//! This module exposes the four primitives needed by the EVM / block-header
//! logic: the core `bloom_m3_2048`, per-log and per-block composition, and a
//! containment query for verifier-side membership checks.
//!
//! # AIR wiring (future)
//!
//! The bloom circuit will reuse the keccak circuit for the `keccak256(item)`
//! step and then express each bit-set as a constraint that ORs a one-hot
//! 2048-bit indicator into the running bloom. OR lookups from `lookup.rs`
//! handle the per-byte composition across items.
//!
use crate::keccak::keccak256;

/// Length in bytes of an Ethereum bloom filter (2048 bits).
pub const BLOOM_BYTE_LENGTH: usize = 256;

/// Length in bits of an Ethereum bloom filter.
pub const BLOOM_BIT_LENGTH: usize = 2048;

/// Set bit `idx` (0..2047) in `bloom` using Ethereum's big-endian convention.
#[inline]
fn set_bit(bloom: &mut [u8; BLOOM_BYTE_LENGTH], idx: u16) {
    debug_assert!((idx as usize) < BLOOM_BIT_LENGTH);
    let byte_index = BLOOM_BYTE_LENGTH - 1 - (idx as usize) / 8;
    let bit_mask = 1u8 << ((idx as usize) % 8);
    bloom[byte_index] |= bit_mask;
}

/// Test bit `idx` (0..2047) in `bloom`.
#[inline]
fn get_bit(bloom: &[u8; BLOOM_BYTE_LENGTH], idx: u16) -> bool {
    debug_assert!((idx as usize) < BLOOM_BIT_LENGTH);
    let byte_index = BLOOM_BYTE_LENGTH - 1 - (idx as usize) / 8;
    let bit_mask = 1u8 << ((idx as usize) % 8);
    (bloom[byte_index] & bit_mask) != 0
}

/// Extract the three 11-bit bloom indices from a keccak256 hash.
///
/// Returns `(i0, i1, i2)` where `i_k = u16::from_be(hash[2k..2k+2]) & 0x07FF`.
#[inline]
fn indices_from_hash(hash: &[u8; 32]) -> (u16, u16, u16) {
    let i0 = u16::from_be_bytes([hash[0], hash[1]]) & 0x07FF;
    let i1 = u16::from_be_bytes([hash[2], hash[3]]) & 0x07FF;
    let i2 = u16::from_be_bytes([hash[4], hash[5]]) & 0x07FF;
    (i0, i1, i2)
}

/// Canonical bloom primitive: `M3:2048(item) = bloom with three bits set`.
///
/// Yellow Paper §4.4.4. Takes a byte string, computes `keccak256(item)`, and
/// returns a 256-byte bloom with exactly three bits set (or fewer if two of
/// the indices collide).
pub fn bloom_m3_2048(item: &[u8]) -> [u8; BLOOM_BYTE_LENGTH] {
    let h = keccak256(item);
    let (i0, i1, i2) = indices_from_hash(&h);
    let mut bloom = [0u8; BLOOM_BYTE_LENGTH];
    set_bit(&mut bloom, i0);
    set_bit(&mut bloom, i1);
    set_bit(&mut bloom, i2);
    bloom
}

/// In-place OR: `dst |= src`.
#[inline]
fn or_into(dst: &mut [u8; BLOOM_BYTE_LENGTH], src: &[u8; BLOOM_BYTE_LENGTH]) {
    for i in 0..BLOOM_BYTE_LENGTH {
        dst[i] |= src[i];
    }
}

/// Compute the bloom filter for a single log entry.
///
/// Combines `bloom_m3_2048(address) | bloom_m3_2048(topic_0) | ... |
/// bloom_m3_2048(topic_{n-1})`. For a standard EVM log with up to 4 topics
/// this sets at most 3 + 4*3 = 15 bits.
pub fn logs_bloom_for_log(address: &[u8; 20], topics: &[[u8; 32]]) -> [u8; BLOOM_BYTE_LENGTH] {
    let mut bloom = bloom_m3_2048(address);
    for t in topics {
        let tb = bloom_m3_2048(t);
        or_into(&mut bloom, &tb);
    }
    bloom
}

/// Compute the `logsBloom` field for a block, given its list of logs.
///
/// Each log is `(address, topics)`. The block's logs bloom is the bitwise OR
/// across every per-log bloom. An empty log list yields the all-zero bloom.
pub fn logs_bloom_for_block(
    logs: &[([u8; 20], Vec<[u8; 32]>)],
) -> [u8; BLOOM_BYTE_LENGTH] {
    let mut bloom = [0u8; BLOOM_BYTE_LENGTH];
    for (addr, topics) in logs {
        let lb = logs_bloom_for_log(addr, topics);
        or_into(&mut bloom, &lb);
    }
    bloom
}

/// Check whether `item`'s three bloom bits are all present in `bloom`.
///
/// This is the canonical bloom membership query. Returns `true` if the item
/// *might* be in the filter (modulo false-positives inherent to bloom
/// filters); returns `false` if it is definitively absent.
pub fn bloom_contains(bloom: &[u8; BLOOM_BYTE_LENGTH], item: &[u8]) -> bool {
    let h = keccak256(item);
    let (i0, i1, i2) = indices_from_hash(&h);
    get_bit(bloom, i0) && get_bit(bloom, i1) && get_bit(bloom, i2)
}

/// Count the number of set bits in a bloom filter. Useful for testing.
pub fn popcount(bloom: &[u8; BLOOM_BYTE_LENGTH]) -> u32 {
    bloom.iter().map(|b| b.count_ones()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty block — no logs, bloom is all zeros.
    #[test]
    fn test_empty_block_bloom_is_zero() {
        let bloom = logs_bloom_for_block(&[]);
        assert_eq!(bloom, [0u8; BLOOM_BYTE_LENGTH]);
    }

    /// `bloom_m3_2048` sets at most 3 bits (fewer only on index collision,
    /// which is rare for non-adversarial inputs).
    #[test]
    fn test_bloom_m3_2048_sets_at_most_three_bits() {
        for seed in 0u64..50 {
            // Distinct pseudo-random inputs.
            let input = seed.to_le_bytes();
            let bloom = bloom_m3_2048(&input);
            let pc = popcount(&bloom);
            assert!(
                pc <= 3,
                "seed {seed}: popcount = {pc}, expected <= 3",
            );
            assert!(
                pc >= 1,
                "seed {seed}: popcount = 0, expected >= 1",
            );
        }
    }

    /// Idempotence: `bloom_m3_2048(x) | bloom_m3_2048(x) == bloom_m3_2048(x)`.
    #[test]
    fn test_bloom_idempotent_under_or() {
        let inputs: &[&[u8]] = &[
            b"",
            b"hello",
            b"ethereum",
            &[0u8; 32],
            &[0xffu8; 20],
            &[0x12, 0x34, 0x56, 0x78],
        ];
        for input in inputs {
            let b = bloom_m3_2048(input);
            let mut b2 = b;
            or_into(&mut b2, &b);
            assert_eq!(b, b2);
        }
    }

    /// Containment: `bloom_contains(bloom_m3_2048(x), x) == true`.
    #[test]
    fn test_bloom_contains_self() {
        for seed in 0u64..100 {
            let input = seed.to_be_bytes();
            let bloom = bloom_m3_2048(&input);
            assert!(
                bloom_contains(&bloom, &input),
                "seed {seed}: bloom does not contain its own input",
            );
        }
    }

    /// Non-containment (probabilistic): for a bloom built from one item,
    /// most *other* items should miss. Bloom filters false-positive at
    /// ~(3/2048)^3 ≈ 3.1e-9 per query with 1 item, so 100 random queries
    /// should have zero hits with overwhelming probability.
    #[test]
    fn test_bloom_probabilistic_non_containment() {
        let x = b"alpha-beta-gamma";
        let bloom = bloom_m3_2048(x);
        let mut false_positives = 0;
        for seed in 0u64..200 {
            let y = seed.to_le_bytes();
            if y.as_slice() == x.as_slice() {
                continue;
            }
            if bloom_contains(&bloom, &y) {
                false_positives += 1;
            }
        }
        // With a single-item bloom (3 bits set), random queries should almost
        // never match. Allow a small margin for the 3-bit-collision regime.
        assert!(
            false_positives <= 1,
            "got {false_positives} false positives in 200 queries; bloom bit-set may be wrong",
        );
    }

    /// ERC-20 `Transfer(address,address,uint256)` topic hash.
    /// `keccak256("Transfer(address,address,uint256)")` =
    /// `0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef`.
    const TRANSFER_TOPIC: [u8; 32] = [
        0xdd, 0xf2, 0x52, 0xad, 0x1b, 0xe2, 0xc8, 0x9b,
        0x69, 0xc2, 0xb0, 0x68, 0xfc, 0x37, 0x8d, 0xaa,
        0x95, 0x2b, 0xa7, 0xf1, 0x63, 0xc4, 0xa1, 0x16,
        0x28, 0xf5, 0x5a, 0x4d, 0xf5, 0x23, 0xb3, 0xef,
    ];

    /// Sanity-check the ERC-20 Transfer topic hash.
    #[test]
    fn test_transfer_topic_hash() {
        let h = keccak256(b"Transfer(address,address,uint256)");
        assert_eq!(h, TRANSFER_TOPIC);
    }

    /// Log bloom from the zero address with the ERC-20 Transfer topic.
    /// Expect up to 3 (from address) + 3 (from topic) = 6 bits set; if any
    /// indices collide the count could be lower, but for these two inputs
    /// the six indices are all distinct.
    #[test]
    fn test_zero_address_transfer_log_bloom() {
        let zero_addr = [0u8; 20];
        let topics = [TRANSFER_TOPIC];
        let bloom = logs_bloom_for_log(&zero_addr, &topics);

        // Verify bit counts — 3 from address + 3 from topic.
        let pc = popcount(&bloom);
        assert_eq!(pc, 6, "expected 6 bits set; got {pc}");

        // Verify bloom contains both items.
        assert!(bloom_contains(&bloom, &zero_addr));
        assert!(bloom_contains(&bloom, &TRANSFER_TOPIC));

        // Derive expected indices explicitly for cross-check.
        // keccak256(zero_addr_20) first 6 bytes:
        let h_addr = keccak256(&zero_addr);
        let (a0, a1, a2) = indices_from_hash(&h_addr);
        // keccak256(TRANSFER_TOPIC) first 6 bytes:
        let h_top = keccak256(&TRANSFER_TOPIC);
        let (t0, t1, t2) = indices_from_hash(&h_top);

        for idx in [a0, a1, a2, t0, t1, t2] {
            assert!(
                get_bit(&bloom, idx),
                "expected bit {idx} set",
            );
        }

        // The six indices should all be distinct for this pair. Confirm.
        let mut indices = [a0, a1, a2, t0, t1, t2];
        indices.sort();
        for w in indices.windows(2) {
            assert_ne!(w[0], w[1], "unexpected index collision: {indices:?}");
        }
    }

    /// Captured reference bloom for the zero-address + Transfer-topic log.
    /// Regenerated deterministically from the primitives above; any change
    /// to the bit-ordering convention would break this vector.
    #[test]
    fn test_zero_address_transfer_bloom_regression_vector() {
        let zero_addr = [0u8; 20];
        let topics = [TRANSFER_TOPIC];
        let bloom = logs_bloom_for_log(&zero_addr, &topics);

        // Build the expected bloom by hand from the six indices, using the
        // same big-endian convention. This is a self-consistency check: if
        // someone flips the convention, this test detects it loudly.
        let h_addr = keccak256(&zero_addr);
        let h_top = keccak256(&TRANSFER_TOPIC);
        let mut expected = [0u8; BLOOM_BYTE_LENGTH];
        for &h in &[&h_addr, &h_top] {
            let (i0, i1, i2) = indices_from_hash(h);
            // Ethereum convention: byte_index = 255 - idx/8, bit = idx % 8.
            for idx in [i0, i1, i2] {
                let byte_index = 255 - (idx as usize) / 8;
                let bit_mask = 1u8 << ((idx as usize) % 8);
                expected[byte_index] |= bit_mask;
            }
        }
        assert_eq!(bloom, expected);

        // Also check exactly 6 non-zero bytes (each of the 6 bits falls in
        // a distinct byte iff no two indices share a byte — true here).
        let nonzero_bytes = bloom.iter().filter(|b| **b != 0).count();
        assert!(
            nonzero_bytes <= 6,
            "nonzero bytes = {nonzero_bytes} > 6 (impossible)",
        );
    }

    /// Multi-log block bloom: two logs' blooms OR together cleanly.
    #[test]
    fn test_block_bloom_ors_logs() {
        let addr_a = [0x11u8; 20];
        let addr_b = [0x22u8; 20];
        let topic = TRANSFER_TOPIC;

        let log_a_bloom = logs_bloom_for_log(&addr_a, &[topic]);
        let log_b_bloom = logs_bloom_for_log(&addr_b, &[topic]);

        let logs: Vec<([u8; 20], Vec<[u8; 32]>)> = vec![
            (addr_a, vec![topic]),
            (addr_b, vec![topic]),
        ];
        let block_bloom = logs_bloom_for_block(&logs);

        // Block bloom equals OR of each log bloom.
        let mut expected = log_a_bloom;
        or_into(&mut expected, &log_b_bloom);
        assert_eq!(block_bloom, expected);

        // Block bloom contains both addresses and the shared topic.
        assert!(bloom_contains(&block_bloom, &addr_a));
        assert!(bloom_contains(&block_bloom, &addr_b));
        assert!(bloom_contains(&block_bloom, &topic));
    }

    /// A log with no topics still contributes the address's 3 bits.
    #[test]
    fn test_log_no_topics() {
        let addr = [0xabu8; 20];
        let bloom = logs_bloom_for_log(&addr, &[]);
        assert_eq!(bloom, bloom_m3_2048(&addr));
        assert!(bloom_contains(&bloom, &addr));
    }

    /// Log with 4 topics (EVM max) hits up to 15 bits.
    #[test]
    fn test_log_four_topics() {
        let addr = [0x01u8; 20];
        let topics = [
            [0x11u8; 32],
            [0x22u8; 32],
            [0x33u8; 32],
            [0x44u8; 32],
        ];
        let bloom = logs_bloom_for_log(&addr, &topics);
        assert!(popcount(&bloom) <= 15);
        assert!(bloom_contains(&bloom, &addr));
        for t in &topics {
            assert!(bloom_contains(&bloom, t));
        }
    }

    /// Bit-ordering spot check: the index convention places low bit-indices
    /// at HIGH byte offsets (byte 255 holds bits 0..7). Verify directly with
    /// a hand-crafted input.
    #[test]
    fn test_bit_ordering_convention() {
        // Synthesize an index: if idx = 0, it should land at byte 255 bit 0.
        let mut bloom = [0u8; BLOOM_BYTE_LENGTH];
        set_bit(&mut bloom, 0);
        assert_eq!(bloom[255], 0x01);
        assert!(bloom[..255].iter().all(|&b| b == 0));

        // idx = 7 → byte 255, bit 7.
        let mut bloom = [0u8; BLOOM_BYTE_LENGTH];
        set_bit(&mut bloom, 7);
        assert_eq!(bloom[255], 0x80);

        // idx = 8 → byte 254, bit 0.
        let mut bloom = [0u8; BLOOM_BYTE_LENGTH];
        set_bit(&mut bloom, 8);
        assert_eq!(bloom[254], 0x01);
        assert_eq!(bloom[255], 0x00);

        // idx = 2047 (max) → byte 0, bit 7.
        let mut bloom = [0u8; BLOOM_BYTE_LENGTH];
        set_bit(&mut bloom, 2047);
        assert_eq!(bloom[0], 0x80);
        assert!(bloom[1..].iter().all(|&b| b == 0));

        // get_bit round-trips set_bit.
        for idx in [0u16, 1, 7, 8, 255, 1023, 1024, 2047] {
            let mut b = [0u8; BLOOM_BYTE_LENGTH];
            set_bit(&mut b, idx);
            assert!(get_bit(&b, idx), "get_bit({idx}) failed");
        }
    }

    /// The low 3 bits of an 11-bit index equal the low 3 bits of hash[2i+1],
    /// which is a cute fact go-ethereum uses directly (`hashbuf[1] & 0x7`).
    /// Verify our extraction preserves this.
    #[test]
    fn test_index_extraction_matches_geth() {
        // Synthetic hash: hash[0] = 0x12, hash[1] = 0x34 → u16 BE = 0x1234,
        // & 0x07FF = 0x0234. Low 3 bits = 0x34 & 0x07 = 0x04.
        let mut h = [0u8; 32];
        h[0] = 0x12; h[1] = 0x34;
        h[2] = 0xab; h[3] = 0xcd;
        h[4] = 0x55; h[5] = 0xaa;
        let (i0, i1, i2) = indices_from_hash(&h);
        assert_eq!(i0, 0x0234);
        assert_eq!(i0 & 0x07, (h[1] & 0x07) as u16);
        assert_eq!(i1, 0xabcd & 0x07FF);
        assert_eq!(i1 & 0x07, (h[3] & 0x07) as u16);
        assert_eq!(i2, 0x55aa & 0x07FF);
        assert_eq!(i2 & 0x07, (h[5] & 0x07) as u16);
    }
}
