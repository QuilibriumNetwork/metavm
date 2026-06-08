//! Witness builder for proving a chain of BeaconBlockHeaders.
//!
//! Given a sequence of BeaconBlockHeaders forming a valid parent_root
//! chain, builds the witness material for each header's
//! hash_tree_root (via BeaconBlockHeaderHtrWitness) and validates
//! the parent_root linkage.

use crate::beacon::BeaconBlockHeader;
use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
use crate::beacon_header_chain::verify_beacon_header_chain;

pub struct BeaconChainWitness {
    pub headers: Vec<BeaconBlockHeader>,
    pub htr_witnesses: Vec<BeaconBlockHeaderHtrWitness>,
    pub htr_roots: Vec<[u8; 32]>,
}

impl BeaconChainWitness {
    pub fn from_headers(headers: Vec<BeaconBlockHeader>) -> Result<Self, String> {
        verify_beacon_header_chain(&headers)?;
        let mut htr_witnesses = Vec::with_capacity(headers.len());
        let mut htr_roots = Vec::with_capacity(headers.len());
        for h in &headers {
            let w = BeaconBlockHeaderHtrWitness::from_header(h.clone());
            let root = h.hash_tree_root();
            htr_roots.push(root);
            htr_witnesses.push(w);
        }
        // Verify parent_root linkage is consistent with computed HTR.
        for i in 1..headers.len() {
            if headers[i].parent_root != htr_roots[i - 1] {
                return Err(format!(
                    "header[{}].parent_root doesn't match computed HTR of header[{}]",
                    i, i - 1,
                ));
            }
        }
        Ok(Self { headers, htr_witnesses, htr_roots })
    }

    pub fn len(&self) -> usize { self.headers.len() }
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
    fn witness_from_valid_chain() {
        let w = BeaconChainWitness::from_headers(make_chain(5)).unwrap();
        assert_eq!(w.len(), 5);
        assert_eq!(w.htr_roots.len(), 5);
        for i in 1..5 {
            assert_eq!(w.headers[i].parent_root, w.htr_roots[i - 1]);
        }
    }

    #[test]
    fn witness_from_broken_chain_fails() {
        let mut chain = make_chain(3);
        chain[2].parent_root[0] ^= 0xff;
        assert!(BeaconChainWitness::from_headers(chain).is_err());
    }

    #[test]
    fn htr_witnesses_populated() {
        let w = BeaconChainWitness::from_headers(make_chain(3)).unwrap();
        for hw in &w.htr_witnesses {
            assert_eq!(hw.invocations.len(), 7);
        }
    }

    #[test]
    fn single_header() {
        let w = BeaconChainWitness::from_headers(make_chain(1)).unwrap();
        assert_eq!(w.len(), 1);
    }
}
