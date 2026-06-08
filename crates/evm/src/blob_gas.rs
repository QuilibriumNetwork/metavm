//! EIP-4844 blob gas oracle.
//!
//! Cancun/Deneb introduced blob transactions with a separate fee
//! market. The block header includes `blob_gas_used` and
//! `excess_blob_gas` fields.

pub const TARGET_BLOB_GAS_PER_BLOCK: u64 = 393216; // 3 blobs * 131072
pub const MAX_BLOB_GAS_PER_BLOCK: u64 = 786432; // 6 blobs * 131072
pub const BLOB_GAS_PER_BLOB: u64 = 131072; // 2^17
pub const MIN_BLOB_BASE_FEE: u64 = 1;
pub const BLOB_BASE_FEE_UPDATE_FRACTION: u64 = 3338477;

pub fn calc_excess_blob_gas(parent_excess: u64, parent_used: u64) -> u64 {
    let total = parent_excess + parent_used;
    if total < TARGET_BLOB_GAS_PER_BLOCK {
        0
    } else {
        total - TARGET_BLOB_GAS_PER_BLOCK
    }
}

pub fn blob_count_from_gas(blob_gas_used: u64) -> u64 {
    blob_gas_used / BLOB_GAS_PER_BLOB
}

pub fn verify_blob_gas_fields(
    parent_excess_blob_gas: u64,
    parent_blob_gas_used: u64,
    current_excess_blob_gas: u64,
    current_blob_gas_used: u64,
) -> Result<(), String> {
    let expected_excess = calc_excess_blob_gas(parent_excess_blob_gas, parent_blob_gas_used);
    if current_excess_blob_gas != expected_excess {
        return Err(format!(
            "excess_blob_gas: expected {} got {}",
            expected_excess, current_excess_blob_gas,
        ));
    }
    if current_blob_gas_used > MAX_BLOB_GAS_PER_BLOCK {
        return Err(format!(
            "blob_gas_used {} exceeds max {}",
            current_blob_gas_used, MAX_BLOB_GAS_PER_BLOCK,
        ));
    }
    if current_blob_gas_used % BLOB_GAS_PER_BLOB != 0 {
        return Err(format!(
            "blob_gas_used {} not a multiple of {}",
            current_blob_gas_used, BLOB_GAS_PER_BLOB,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excess_at_target() {
        assert_eq!(calc_excess_blob_gas(0, TARGET_BLOB_GAS_PER_BLOCK), 0);
    }

    #[test]
    fn excess_above_target() {
        let excess = calc_excess_blob_gas(0, TARGET_BLOB_GAS_PER_BLOCK + BLOB_GAS_PER_BLOB);
        assert_eq!(excess, BLOB_GAS_PER_BLOB);
    }

    #[test]
    fn excess_below_target() {
        assert_eq!(calc_excess_blob_gas(0, 0), 0);
        assert_eq!(calc_excess_blob_gas(100, 0), 0);
    }

    #[test]
    fn excess_accumulates() {
        let e1 = calc_excess_blob_gas(0, MAX_BLOB_GAS_PER_BLOCK);
        assert_eq!(e1, MAX_BLOB_GAS_PER_BLOCK - TARGET_BLOB_GAS_PER_BLOCK);
        let e2 = calc_excess_blob_gas(e1, MAX_BLOB_GAS_PER_BLOCK);
        assert!(e2 > e1);
    }

    #[test]
    fn blob_count() {
        assert_eq!(blob_count_from_gas(0), 0);
        assert_eq!(blob_count_from_gas(BLOB_GAS_PER_BLOB), 1);
        assert_eq!(blob_count_from_gas(3 * BLOB_GAS_PER_BLOB), 3);
    }

    #[test]
    fn verify_valid_fields() {
        verify_blob_gas_fields(0, TARGET_BLOB_GAS_PER_BLOCK, 0, BLOB_GAS_PER_BLOB).unwrap();
    }

    #[test]
    fn verify_wrong_excess_fails() {
        assert!(verify_blob_gas_fields(0, TARGET_BLOB_GAS_PER_BLOCK, 100, 0).is_err());
    }

    #[test]
    fn verify_too_much_blob_gas_fails() {
        assert!(verify_blob_gas_fields(0, 0, 0, MAX_BLOB_GAS_PER_BLOCK + 1).is_err());
    }

    #[test]
    fn verify_non_multiple_fails() {
        assert!(verify_blob_gas_fields(0, 0, 0, 100).is_err());
    }
}
