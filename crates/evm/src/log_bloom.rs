//! LOG bloom contribution oracle.
//!
//! Verifies that each LOG event's address and topics contribute
//! correctly to the receipt's logsBloom field via Ethereum's
//! bloom filter spec (3 bits per keccak256(item), m=2048).

use crate::log_event::LogEvent;

pub fn compute_log_bloom_contribution(
    address: &[u8; 20],
    topics: &[[u8; 32]],
) -> [u8; 256] {
    metavm_zkp::bloom::logs_bloom_for_log(address, topics)
}

pub fn verify_log_bloom_contributions(
    events: &[(/* address */ [u8; 20], /* topics */ Vec<[u8; 32]>)],
    expected_bloom: &[u8; 256],
) -> Result<(), String> {
    let computed = metavm_zkp::bloom::logs_bloom_for_block(events);
    if &computed != expected_bloom {
        return Err("logsBloom mismatch: computed bloom != expected".into());
    }
    Ok(())
}

pub fn verify_log_events_produce_bloom(
    events: &[LogEvent],
    addresses: &[[u8; 20]],
    expected_bloom: &[u8; 256],
) -> Result<(), String> {
    if events.len() != addresses.len() {
        return Err(format!(
            "event/address count mismatch: {} events, {} addresses",
            events.len(), addresses.len(),
        ));
    }
    let pairs: Vec<([u8; 20], Vec<[u8; 32]>)> = events.iter().zip(addresses.iter())
        .map(|(e, a)| (*a, e.topics.clone()))
        .collect();
    verify_log_bloom_contributions(&pairs, expected_bloom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_log_bloom() {
        let addr = [0x42u8; 20];
        let topics = vec![[0xAA; 32]];
        let bloom = compute_log_bloom_contribution(&addr, &topics);
        assert!(metavm_zkp::bloom::bloom_contains(&bloom, &addr));
        assert!(metavm_zkp::bloom::bloom_contains(&bloom, &topics[0]));
    }

    #[test]
    fn verify_bloom_matches() {
        let addr = [0x42u8; 20];
        let topics = vec![[0xAA; 32]];
        let bloom = metavm_zkp::bloom::logs_bloom_for_block(&[(addr, topics.clone())]);
        verify_log_bloom_contributions(&[(addr, topics)], &bloom).unwrap();
    }

    #[test]
    fn wrong_bloom_rejected() {
        let addr = [0x42u8; 20];
        let topics = vec![[0xAA; 32]];
        let bloom = [0u8; 256];
        assert!(verify_log_bloom_contributions(&[(addr, topics)], &bloom).is_err());
    }

    #[test]
    fn empty_events_zero_bloom() {
        let bloom = [0u8; 256];
        verify_log_bloom_contributions(&[], &bloom).unwrap();
    }

    #[test]
    fn multiple_events_combine() {
        let events = vec![
            ([0x11u8; 20], vec![[0xAA; 32]]),
            ([0x22u8; 20], vec![[0xBB; 32], [0xCC; 32]]),
        ];
        let bloom = metavm_zkp::bloom::logs_bloom_for_block(&events);
        verify_log_bloom_contributions(&events, &bloom).unwrap();
        assert!(metavm_zkp::bloom::bloom_contains(&bloom, &[0x11u8; 20]));
        assert!(metavm_zkp::bloom::bloom_contains(&bloom, &[0x22u8; 20]));
        assert!(metavm_zkp::bloom::bloom_contains(&bloom, &[0xAA; 32]));
        assert!(metavm_zkp::bloom::bloom_contains(&bloom, &[0xBB; 32]));
        assert!(metavm_zkp::bloom::bloom_contains(&bloom, &[0xCC; 32]));
    }
}
