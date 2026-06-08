//! Gas accounting summary oracle.
//!
//! Aggregates all gas-related verifications into a single consistent
//! check across a transaction:
//! - Static gas costs per opcode
//! - Memory expansion cost
//! - Refund cap (EIP-3529)
//! - Effective gas price (Legacy/EIP-1559)
//! - Sender debit + miner fee + base fee burned

use crate::eip1559_fee::{base_fee_burned, miner_tip};
use crate::gas_refund::effective_gas_used;
use crate::tx_fee_verification::{TxFeeWitness, compute_effective_gas_price};

#[derive(Clone, Debug)]
pub struct TxGasSummary {
    pub fee_witness: TxFeeWitness,
    pub gas_limit: u64,
    pub gas_remaining: u64,
    pub refund: u64,
    pub value_sent: u64,
}

#[derive(Clone, Debug)]
pub struct GasComputed {
    pub effective_gas_price: u64,
    pub effective_gas_used: u64,
    pub sender_debit: u64,
    pub miner_fee: u64,
    pub base_fee_burned: u64,
}

pub fn compute_gas_summary(s: &TxGasSummary) -> Result<GasComputed, String> {
    let price = compute_effective_gas_price(&s.fee_witness)?;
    let used = effective_gas_used(s.gas_limit, s.gas_remaining, s.refund);
    let debit = price.checked_mul(used).ok_or("debit overflow")?
        .checked_add(s.value_sent).ok_or("debit overflow")?;
    let (base, priority) = match &s.fee_witness {
        TxFeeWitness::Legacy { gas_price } => (0u64, *gas_price),
        TxFeeWitness::Eip1559 { base_fee, .. } => (*base_fee, price - base_fee),
    };
    let burned = base_fee_burned(base, used);
    let tip = miner_tip(priority, used);
    Ok(GasComputed {
        effective_gas_price: price,
        effective_gas_used: used,
        sender_debit: debit,
        miner_fee: tip,
        base_fee_burned: burned,
    })
}

pub fn verify_gas_summary_consistency(s: &TxGasSummary) -> Result<(), String> {
    let g = compute_gas_summary(s)?;
    // Invariant: miner_fee + base_fee_burned = price * used
    let total_paid = g.miner_fee.checked_add(g.base_fee_burned)
        .ok_or("total overflow")?;
    let expected = g.effective_gas_price.checked_mul(g.effective_gas_used)
        .ok_or("expected overflow")?;
    if total_paid != expected {
        return Err(format!(
            "fee invariant: miner+burned {} != price*used {}",
            total_paid, expected,
        ));
    }
    // Invariant: sender_debit = price*used + value
    let expected_debit = expected + s.value_sent;
    if g.sender_debit != expected_debit {
        return Err(format!(
            "sender debit {} != price*used + value {}",
            g.sender_debit, expected_debit,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_tx_summary() {
        let s = TxGasSummary {
            fee_witness: TxFeeWitness::Legacy { gas_price: 100 },
            gas_limit: 100_000,
            gas_remaining: 79_000,
            refund: 0,
            value_sent: 0,
        };
        let g = compute_gas_summary(&s).unwrap();
        assert_eq!(g.effective_gas_price, 100);
        assert_eq!(g.effective_gas_used, 21_000);
        assert_eq!(g.sender_debit, 2_100_000);
        assert_eq!(g.miner_fee, 2_100_000);
        assert_eq!(g.base_fee_burned, 0);
        verify_gas_summary_consistency(&s).unwrap();
    }

    #[test]
    fn eip1559_tx_summary() {
        let s = TxGasSummary {
            fee_witness: TxFeeWitness::Eip1559 { max_fee: 200, max_priority_fee: 50, base_fee: 100 },
            gas_limit: 100_000,
            gas_remaining: 79_000,
            refund: 0,
            value_sent: 1_000,
        };
        let g = compute_gas_summary(&s).unwrap();
        assert_eq!(g.effective_gas_price, 150); // 100 base + 50 priority
        assert_eq!(g.effective_gas_used, 21_000);
        assert_eq!(g.miner_fee, 1_050_000); // 50 * 21000
        assert_eq!(g.base_fee_burned, 2_100_000); // 100 * 21000
        assert_eq!(g.sender_debit, 150 * 21_000 + 1_000);
        verify_gas_summary_consistency(&s).unwrap();
    }

    #[test]
    fn refund_reduces_effective_gas() {
        let s = TxGasSummary {
            fee_witness: TxFeeWitness::Legacy { gas_price: 100 },
            gas_limit: 100_000,
            gas_remaining: 50_000,
            refund: 5_000,
            value_sent: 0,
        };
        let g = compute_gas_summary(&s).unwrap();
        // gas_used = 50_000, max_refund = 10_000, refund = 5_000
        // effective_gas_used = 45_000
        assert_eq!(g.effective_gas_used, 45_000);
        verify_gas_summary_consistency(&s).unwrap();
    }

    #[test]
    fn refund_capped_at_quotient() {
        let s = TxGasSummary {
            fee_witness: TxFeeWitness::Legacy { gas_price: 100 },
            gas_limit: 100_000,
            gas_remaining: 50_000,
            refund: 50_000, // exceeds cap of 10_000
            value_sent: 0,
        };
        let g = compute_gas_summary(&s).unwrap();
        assert_eq!(g.effective_gas_used, 40_000); // 50_000 - 10_000 cap
    }
}
