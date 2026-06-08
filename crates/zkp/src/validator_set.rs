//! Validator set utilities for FFG attestation verification.
//!
//! Provides host-side helpers for working with validator balances
//! and effective balances in the context of Casper FFG finality.

/// Minimum effective balance for attestation inclusion (32 ETH in Gwei).
pub const MIN_EFFECTIVE_BALANCE: u64 = 32_000_000_000;

/// Maximum effective balance (32 ETH in Gwei).
pub const MAX_EFFECTIVE_BALANCE: u64 = 32_000_000_000;

/// A validator's state for FFG purposes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorState {
    pub pubkey: [u8; 48],
    pub effective_balance: u64,
    pub slashed: bool,
    pub activation_epoch: u64,
    pub exit_epoch: u64,
}

impl ValidatorState {
    pub fn is_active_at(&self, epoch: u64) -> bool {
        self.activation_epoch <= epoch && epoch < self.exit_epoch
    }

    pub fn is_eligible_attester(&self, epoch: u64) -> bool {
        self.is_active_at(epoch) && !self.slashed
    }
}

pub fn total_active_balance(validators: &[ValidatorState], epoch: u64) -> u64 {
    validators.iter()
        .filter(|v| v.is_active_at(epoch))
        .map(|v| v.effective_balance)
        .sum()
}

pub fn total_attesting_balance(
    validators: &[ValidatorState],
    participating: &[bool],
    epoch: u64,
) -> u64 {
    assert_eq!(validators.len(), participating.len());
    validators.iter().zip(participating.iter())
        .filter(|(v, &p)| p && v.is_eligible_attester(epoch))
        .map(|(v, _)| v.effective_balance)
        .sum()
}

/// Check if attesting balance meets 2/3 supermajority threshold.
pub fn is_supermajority(attesting: u64, total: u64) -> bool {
    attesting * 3 >= total * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(balance: u64, active: bool) -> ValidatorState {
        ValidatorState {
            pubkey: [0u8; 48],
            effective_balance: balance,
            slashed: false,
            activation_epoch: if active { 0 } else { u64::MAX },
            exit_epoch: u64::MAX,
        }
    }

    #[test]
    fn total_active() {
        let vals = vec![v(32_000_000_000, true), v(32_000_000_000, true), v(32_000_000_000, false)];
        assert_eq!(total_active_balance(&vals, 10), 64_000_000_000);
    }

    #[test]
    fn total_attesting() {
        let vals = vec![v(32_000_000_000, true), v(32_000_000_000, true), v(32_000_000_000, true)];
        let part = vec![true, false, true];
        assert_eq!(total_attesting_balance(&vals, &part, 10), 64_000_000_000);
    }

    #[test]
    fn supermajority_exactly_two_thirds() {
        assert!(is_supermajority(2, 3));
        assert!(!is_supermajority(1, 3));
    }

    #[test]
    fn supermajority_realistic() {
        let total = 100 * 32_000_000_000u64;
        let attesting = 67 * 32_000_000_000u64;
        assert!(is_supermajority(attesting, total));
        let attesting = 66 * 32_000_000_000u64;
        assert!(!is_supermajority(attesting, total));
    }

    #[test]
    fn slashed_validator_not_eligible() {
        let mut val = v(32_000_000_000, true);
        val.slashed = true;
        assert!(!val.is_eligible_attester(10));
        assert!(val.is_active_at(10)); // still active but slashed
    }

    #[test]
    fn exited_validator() {
        let val = ValidatorState {
            pubkey: [0u8; 48],
            effective_balance: 32_000_000_000,
            slashed: false,
            activation_epoch: 0,
            exit_epoch: 5,
        };
        assert!(val.is_active_at(4));
        assert!(!val.is_active_at(5));
    }
}
