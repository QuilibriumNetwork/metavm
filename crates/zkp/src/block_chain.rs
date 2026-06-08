//! Block header chain oracle.
//!
//! Verifies that a sequence of execution block headers forms a
//! valid chain: parent_hash linkage, monotonic number/timestamp,
//! and gas limit bounds.

use crate::block_header::{BlockHeader, block_header_hash};

pub fn verify_block_chain(headers: &[BlockHeader]) -> Result<(), String> {
    if headers.is_empty() { return Ok(()); }
    for i in 1..headers.len() {
        let parent_hash = block_header_hash(&headers[i - 1]);
        if headers[i].parent_hash != parent_hash {
            return Err(format!(
                "header[{}].parent_hash != hash(header[{}])",
                i, i - 1,
            ));
        }
        if headers[i].number != headers[i - 1].number + 1 {
            return Err(format!(
                "non-sequential numbers: {} -> {}",
                headers[i - 1].number, headers[i].number,
            ));
        }
        if headers[i].timestamp <= headers[i - 1].timestamp {
            return Err(format!(
                "timestamp not increasing: {} -> {}",
                headers[i - 1].timestamp, headers[i].timestamp,
            ));
        }
    }
    Ok(())
}

pub fn verify_gas_limit_bounds(
    parent_gas_limit: u64,
    current_gas_limit: u64,
) -> Result<(), String> {
    let delta = if current_gas_limit > parent_gas_limit {
        current_gas_limit - parent_gas_limit
    } else {
        parent_gas_limit - current_gas_limit
    };
    let max_delta = parent_gas_limit / 1024;
    if delta >= max_delta {
        return Err(format!(
            "gas_limit delta {} >= max {}",
            delta, max_delta,
        ));
    }
    if current_gas_limit < 5000 {
        return Err(format!("gas_limit {} below minimum 5000", current_gas_limit));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(n: usize) -> Vec<BlockHeader> {
        let mut headers = Vec::with_capacity(n);
        for i in 0..n {
            let mut h = BlockHeader::default();
            h.number = i as u64;
            h.timestamp = 1700000000 + (i as u64) * 12;
            h.gas_limit = 30_000_000;
            if i > 0 {
                h.parent_hash = block_header_hash(&headers[i - 1]);
            }
            headers.push(h);
        }
        headers
    }

    #[test]
    fn valid_chain() {
        verify_block_chain(&make_chain(5)).unwrap();
    }

    #[test]
    fn broken_parent_hash() {
        let mut chain = make_chain(3);
        chain[2].parent_hash[0] ^= 0xff;
        assert!(verify_block_chain(&chain).is_err());
    }

    #[test]
    fn non_sequential_number() {
        let mut chain = make_chain(3);
        chain[2].number = 5;
        chain[2].parent_hash = block_header_hash(&chain[1]);
        assert!(verify_block_chain(&chain).is_err());
    }

    #[test]
    fn timestamp_not_increasing() {
        let mut chain = make_chain(3);
        chain[2].timestamp = chain[1].timestamp;
        chain[2].parent_hash = block_header_hash(&chain[1]);
        assert!(verify_block_chain(&chain).is_err());
    }

    #[test]
    fn gas_limit_valid_same() {
        verify_gas_limit_bounds(30_000_000, 30_000_000).unwrap();
    }

    #[test]
    fn gas_limit_valid_small_increase() {
        verify_gas_limit_bounds(30_000_000, 30_000_000 + 29_000).unwrap();
    }

    #[test]
    fn gas_limit_too_big_jump() {
        assert!(verify_gas_limit_bounds(30_000_000, 31_000_000).is_err());
    }

    #[test]
    fn gas_limit_below_minimum() {
        assert!(verify_gas_limit_bounds(5000, 4999).is_err());
    }

    #[test]
    fn single_header_valid() {
        verify_block_chain(&make_chain(1)).unwrap();
    }

    #[test]
    fn empty_chain_valid() {
        verify_block_chain(&[]).unwrap();
    }
}
