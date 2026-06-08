//! EIP-1559 fee calculation oracle.
//!
//! Computes effective gas price, priority fee (miner tip), and
//! base fee burned for EIP-1559 transactions.

pub fn effective_gas_price(
    base_fee_per_gas: u64,
    max_fee_per_gas: u64,
    max_priority_fee_per_gas: u64,
) -> Result<u64, String> {
    if max_fee_per_gas < base_fee_per_gas {
        return Err(format!(
            "max_fee_per_gas {} < base_fee_per_gas {}",
            max_fee_per_gas, base_fee_per_gas,
        ));
    }
    let priority_fee = max_priority_fee_per_gas.min(max_fee_per_gas - base_fee_per_gas);
    Ok(base_fee_per_gas + priority_fee)
}

pub fn priority_fee(
    base_fee_per_gas: u64,
    effective_gas_price: u64,
) -> u64 {
    effective_gas_price.saturating_sub(base_fee_per_gas)
}

pub fn base_fee_burned(base_fee_per_gas: u64, gas_used: u64) -> u64 {
    base_fee_per_gas * gas_used
}

pub fn miner_tip(priority_fee_per_gas: u64, gas_used: u64) -> u64 {
    priority_fee_per_gas * gas_used
}

pub fn next_base_fee(
    parent_base_fee: u64,
    parent_gas_used: u64,
    parent_gas_target: u64,
) -> u64 {
    if parent_gas_used == parent_gas_target {
        return parent_base_fee;
    }
    if parent_gas_used > parent_gas_target {
        let delta = parent_gas_used - parent_gas_target;
        let fee_delta = (parent_base_fee * delta / parent_gas_target).max(1);
        parent_base_fee + fee_delta
    } else {
        let delta = parent_gas_target - parent_gas_used;
        let fee_delta = parent_base_fee * delta / parent_gas_target;
        parent_base_fee.saturating_sub(fee_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_price_normal() {
        let price = effective_gas_price(10, 20, 5).unwrap();
        assert_eq!(price, 15); // base + min(priority, max-base) = 10 + min(5, 10) = 15
    }

    #[test]
    fn effective_price_priority_capped() {
        let price = effective_gas_price(10, 12, 5).unwrap();
        assert_eq!(price, 12); // base + min(5, 2) = 10 + 2 = 12
    }

    #[test]
    fn effective_price_max_below_base_fails() {
        assert!(effective_gas_price(10, 5, 3).is_err());
    }

    #[test]
    fn priority_fee_calc() {
        assert_eq!(priority_fee(10, 15), 5);
        assert_eq!(priority_fee(10, 10), 0);
    }

    #[test]
    fn burned_and_tip() {
        assert_eq!(base_fee_burned(10, 21_000), 210_000);
        assert_eq!(miner_tip(5, 21_000), 105_000);
    }

    #[test]
    fn next_base_fee_at_target() {
        assert_eq!(next_base_fee(1000, 15_000_000, 15_000_000), 1000);
    }

    #[test]
    fn next_base_fee_above_target() {
        let next = next_base_fee(1000, 20_000_000, 15_000_000);
        assert!(next > 1000);
    }

    #[test]
    fn next_base_fee_below_target() {
        let next = next_base_fee(1000, 10_000_000, 15_000_000);
        assert!(next < 1000);
    }
}
