//! Gas refund oracle.
//!
//! EVM gas refunds from SSTORE (clearing storage) are capped at
//! max_refund = gas_used / 5 (post-London). This oracle computes
//! the effective gas used after refunds.

pub const MAX_REFUND_QUOTIENT: u64 = 5; // EIP-3529 (London)

pub fn effective_gas_used(gas_limit: u64, gas_remaining: u64, refund: u64) -> u64 {
    let gas_used = gas_limit - gas_remaining;
    let max_refund = gas_used / MAX_REFUND_QUOTIENT;
    let actual_refund = refund.min(max_refund);
    gas_used - actual_refund
}

pub fn verify_receipt_gas_used(
    gas_limit: u64,
    gas_remaining: u64,
    refund: u64,
    receipt_cumulative_gas: u64,
    prior_cumulative_gas: u64,
) -> Result<(), String> {
    let effective = effective_gas_used(gas_limit, gas_remaining, refund);
    let tx_gas_used = receipt_cumulative_gas - prior_cumulative_gas;
    if tx_gas_used != effective {
        return Err(format!(
            "gas used mismatch: receipt says {} but computed {}",
            tx_gas_used, effective,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_refund() {
        assert_eq!(effective_gas_used(100_000, 79_000, 0), 21_000);
    }

    #[test]
    fn refund_capped() {
        // gas_used = 50_000, max_refund = 10_000, claimed refund = 20_000
        assert_eq!(effective_gas_used(100_000, 50_000, 20_000), 40_000);
    }

    #[test]
    fn refund_under_cap() {
        // gas_used = 50_000, max_refund = 10_000, claimed refund = 5_000
        assert_eq!(effective_gas_used(100_000, 50_000, 5_000), 45_000);
    }

    #[test]
    fn receipt_gas_matches() {
        verify_receipt_gas_used(100_000, 79_000, 0, 21_000, 0).unwrap();
    }

    #[test]
    fn receipt_gas_mismatch_fails() {
        assert!(verify_receipt_gas_used(100_000, 79_000, 0, 22_000, 0).is_err());
    }

    #[test]
    fn receipt_gas_with_prior() {
        // Prior tx used 50_000, this tx uses 21_000
        verify_receipt_gas_used(100_000, 79_000, 0, 71_000, 50_000).unwrap();
    }
}
