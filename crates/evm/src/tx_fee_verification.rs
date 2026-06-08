//! Transaction fee verification oracle.
//!
//! Verifies the effective gas price the sender pays per the EVM rules:
//! - Legacy tx: effective_gas_price = gas_price
//! - EIP-1559 tx: effective_gas_price = base_fee + min(priority_fee, max_fee - base_fee)
//!
//! Also verifies the sender balance deduction matches gas_used * gas_price + value.

use crate::eip1559_fee::effective_gas_price;

#[derive(Clone, Debug)]
pub enum TxFeeWitness {
    Legacy { gas_price: u64 },
    Eip1559 { max_fee: u64, max_priority_fee: u64, base_fee: u64 },
}

pub fn compute_effective_gas_price(w: &TxFeeWitness) -> Result<u64, String> {
    match w {
        TxFeeWitness::Legacy { gas_price } => Ok(*gas_price),
        TxFeeWitness::Eip1559 { max_fee, max_priority_fee, base_fee } => {
            effective_gas_price(*base_fee, *max_fee, *max_priority_fee)
        }
    }
}

pub fn verify_sender_debit(
    w: &TxFeeWitness,
    gas_used: u64,
    value_sent: u64,
    sender_balance_delta: u64,
) -> Result<(), String> {
    let price = compute_effective_gas_price(w)?;
    let expected = price.checked_mul(gas_used)
        .ok_or("gas cost overflow")?
        .checked_add(value_sent)
        .ok_or("debit overflow")?;
    if sender_balance_delta != expected {
        return Err(format!(
            "sender debit {} != expected {} (gas_used={}, price={}, value={})",
            sender_balance_delta, expected, gas_used, price, value_sent,
        ));
    }
    Ok(())
}

pub fn miner_fee_amount(
    w: &TxFeeWitness,
    gas_used: u64,
) -> Result<u64, String> {
    let price = compute_effective_gas_price(w)?;
    match w {
        TxFeeWitness::Legacy { .. } => price.checked_mul(gas_used)
            .ok_or("overflow".into()),
        TxFeeWitness::Eip1559 { base_fee, .. } => {
            let priority = price - base_fee;
            priority.checked_mul(gas_used).ok_or("overflow".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_effective_price() {
        let w = TxFeeWitness::Legacy { gas_price: 100 };
        assert_eq!(compute_effective_gas_price(&w).unwrap(), 100);
    }

    #[test]
    fn eip1559_effective_price() {
        let w = TxFeeWitness::Eip1559 { max_fee: 200, max_priority_fee: 50, base_fee: 100 };
        assert_eq!(compute_effective_gas_price(&w).unwrap(), 150);
    }

    #[test]
    fn eip1559_priority_capped() {
        let w = TxFeeWitness::Eip1559 { max_fee: 120, max_priority_fee: 50, base_fee: 100 };
        assert_eq!(compute_effective_gas_price(&w).unwrap(), 120);
    }

    #[test]
    fn sender_debit_legacy() {
        let w = TxFeeWitness::Legacy { gas_price: 100 };
        // gas_used=21000, value=1000, expected debit = 21000*100 + 1000 = 2101000
        verify_sender_debit(&w, 21000, 1000, 2_101_000).unwrap();
    }

    #[test]
    fn sender_debit_mismatch_fails() {
        let w = TxFeeWitness::Legacy { gas_price: 100 };
        assert!(verify_sender_debit(&w, 21000, 1000, 2_000_000).is_err());
    }

    #[test]
    fn miner_fee_legacy() {
        let w = TxFeeWitness::Legacy { gas_price: 100 };
        assert_eq!(miner_fee_amount(&w, 21000).unwrap(), 2_100_000);
    }

    #[test]
    fn miner_fee_eip1559() {
        let w = TxFeeWitness::Eip1559 { max_fee: 200, max_priority_fee: 50, base_fee: 100 };
        // priority = 50, miner fee = 50 * 21000 = 1_050_000
        assert_eq!(miner_fee_amount(&w, 21000).unwrap(), 1_050_000);
    }
}
