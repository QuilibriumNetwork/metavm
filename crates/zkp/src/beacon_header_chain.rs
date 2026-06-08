//! BeaconBlockHeader parent_root chain oracle.
//!
//! Verifies that a sequence of BeaconBlockHeaders forms a valid chain
//! via `parent_root` linkage: each header's `parent_root` equals the
//! previous header's `hash_tree_root()`.

use crate::beacon::BeaconBlockHeader;

pub fn verify_beacon_header_chain(headers: &[BeaconBlockHeader]) -> Result<(), String> {
    if headers.is_empty() { return Ok(()); }
    for i in 1..headers.len() {
        let expected_parent = headers[i - 1].hash_tree_root();
        if headers[i].parent_root != expected_parent {
            return Err(format!(
                "header[{}].parent_root != header[{}].hash_tree_root() at slot {}",
                i, i - 1, headers[i].slot,
            ));
        }
        if headers[i].slot <= headers[i - 1].slot {
            return Err(format!(
                "slots not increasing: header[{}].slot={} <= header[{}].slot={}",
                i, headers[i].slot, i - 1, headers[i - 1].slot,
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(n: usize) -> Vec<BeaconBlockHeader> {
        let mut headers = Vec::with_capacity(n);
        let mut parent_root = [0u8; 32];
        for i in 0..n {
            let h = BeaconBlockHeader {
                slot: (i + 1) as u64,
                proposer_index: (i % 100) as u64,
                parent_root,
                state_root: [(i as u8).wrapping_add(0x11); 32],
                body_root: [(i as u8).wrapping_add(0x22); 32],
            };
            parent_root = h.hash_tree_root();
            headers.push(h);
        }
        headers
    }

    #[test]
    fn valid_chain_of_5() {
        let chain = make_chain(5);
        verify_beacon_header_chain(&chain).unwrap();
    }

    #[test]
    fn broken_parent_root_fails() {
        let mut chain = make_chain(3);
        chain[2].parent_root[0] ^= 0xff;
        assert!(verify_beacon_header_chain(&chain).is_err());
    }

    #[test]
    fn non_increasing_slot_fails() {
        let mut chain = make_chain(3);
        chain[2].slot = chain[1].slot;
        // Must also fix parent_root for this to trigger the right error.
        chain[2].parent_root = chain[1].hash_tree_root();
        assert!(verify_beacon_header_chain(&chain).is_err());
    }

    #[test]
    fn single_header_valid() {
        verify_beacon_header_chain(&make_chain(1)).unwrap();
    }
}
