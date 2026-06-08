//! Beacon chain epoch/slot utilities.
//!
//! Constants and helpers for computing epochs from slots,
//! epoch boundaries, and related beacon chain time calculations.

pub const SLOTS_PER_EPOCH: u64 = 32;
pub const SECONDS_PER_SLOT: u64 = 12;
pub const GENESIS_TIME: u64 = 1606824023; // mainnet genesis

pub fn slot_to_epoch(slot: u64) -> u64 {
    slot / SLOTS_PER_EPOCH
}

pub fn epoch_start_slot(epoch: u64) -> u64 {
    epoch * SLOTS_PER_EPOCH
}

pub fn epoch_end_slot(epoch: u64) -> u64 {
    epoch_start_slot(epoch + 1) - 1
}

pub fn is_epoch_boundary(slot: u64) -> bool {
    slot % SLOTS_PER_EPOCH == 0
}

pub fn slot_to_timestamp(slot: u64) -> u64 {
    GENESIS_TIME + slot * SECONDS_PER_SLOT
}

pub fn timestamp_to_slot(timestamp: u64) -> Option<u64> {
    if timestamp < GENESIS_TIME { return None; }
    Some((timestamp - GENESIS_TIME) / SECONDS_PER_SLOT)
}

pub fn slots_since_epoch_start(slot: u64) -> u64 {
    slot % SLOTS_PER_EPOCH
}

pub fn verify_slot_epoch_consistency(slot: u64, claimed_epoch: u64) -> Result<(), String> {
    let actual = slot_to_epoch(slot);
    if actual != claimed_epoch {
        return Err(format!("slot {} is in epoch {} not {}", slot, actual, claimed_epoch));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_epoch_conversion() {
        assert_eq!(slot_to_epoch(0), 0);
        assert_eq!(slot_to_epoch(31), 0);
        assert_eq!(slot_to_epoch(32), 1);
        assert_eq!(slot_to_epoch(63), 1);
        assert_eq!(slot_to_epoch(64), 2);
    }

    #[test]
    fn epoch_start_end() {
        assert_eq!(epoch_start_slot(0), 0);
        assert_eq!(epoch_start_slot(1), 32);
        assert_eq!(epoch_end_slot(0), 31);
        assert_eq!(epoch_end_slot(1), 63);
    }

    #[test]
    fn epoch_boundary() {
        assert!(is_epoch_boundary(0));
        assert!(is_epoch_boundary(32));
        assert!(is_epoch_boundary(64));
        assert!(!is_epoch_boundary(1));
        assert!(!is_epoch_boundary(33));
    }

    #[test]
    fn timestamp_roundtrip() {
        let slot = 1000;
        let ts = slot_to_timestamp(slot);
        assert_eq!(timestamp_to_slot(ts), Some(slot));
    }

    #[test]
    fn timestamp_before_genesis() {
        assert_eq!(timestamp_to_slot(0), None);
    }

    #[test]
    fn slot_epoch_consistency() {
        verify_slot_epoch_consistency(100, 3).unwrap();
        assert!(verify_slot_epoch_consistency(100, 4).is_err());
    }

    #[test]
    fn slots_since_start() {
        assert_eq!(slots_since_epoch_start(0), 0);
        assert_eq!(slots_since_epoch_start(5), 5);
        assert_eq!(slots_since_epoch_start(32), 0);
        assert_eq!(slots_since_epoch_start(37), 5);
    }
}
