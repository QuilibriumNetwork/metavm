//! Epoch transition verification oracle.
//!
//! Verifies the justification and finalization state transitions
//! at epoch boundaries per Casper FFG rules.

use crate::beacon::Checkpoint;
use crate::validator_set::{is_supermajority, total_active_balance, total_attesting_balance, ValidatorState};

#[derive(Clone, Debug)]
pub struct EpochTransitionWitness {
    pub epoch: u64,
    pub previous_justified: Checkpoint,
    pub current_justified: Checkpoint,
    pub finalized: Checkpoint,
    pub target_root: [u8; 32],
    pub attesting_balance: u64,
    pub total_balance: u64,
}

pub fn verify_justification(
    w: &EpochTransitionWitness,
) -> Result<bool, String> {
    if w.total_balance == 0 {
        return Err("zero total balance".into());
    }
    Ok(is_supermajority(w.attesting_balance, w.total_balance))
}

pub fn verify_finalization_from_justification(
    current_epoch: u64,
    justified: &Checkpoint,
    finalized_epoch: u64,
) -> Result<(), String> {
    if justified.epoch + 1 > current_epoch {
        return Err("justified epoch too recent for finalization".into());
    }
    if finalized_epoch > justified.epoch {
        return Err("finalized epoch ahead of justified epoch".into());
    }
    Ok(())
}

pub fn compute_epoch_attesting_balance(
    validators: &[ValidatorState],
    participating: &[bool],
    epoch: u64,
) -> (u64, u64) {
    let total = total_active_balance(validators, epoch);
    let attesting = total_attesting_balance(validators, participating, epoch);
    (attesting, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_witness(attesting: u64, total: u64) -> EpochTransitionWitness {
        EpochTransitionWitness {
            epoch: 10,
            previous_justified: Checkpoint { epoch: 8, root: [0; 32] },
            current_justified: Checkpoint { epoch: 9, root: [1; 32] },
            finalized: Checkpoint { epoch: 8, root: [0; 32] },
            target_root: [2; 32],
            attesting_balance: attesting,
            total_balance: total,
        }
    }

    #[test]
    fn justification_passes_supermajority() {
        let w = make_witness(700, 1000);
        assert!(verify_justification(&w).unwrap());
    }

    #[test]
    fn justification_fails_below_threshold() {
        let w = make_witness(600, 1000);
        assert!(!verify_justification(&w).unwrap());
    }

    #[test]
    fn justification_rejects_zero_balance() {
        let w = make_witness(0, 0);
        assert!(verify_justification(&w).is_err());
    }

    #[test]
    fn finalization_valid() {
        let cp = Checkpoint { epoch: 8, root: [0; 32] };
        verify_finalization_from_justification(10, &cp, 8).unwrap();
    }

    #[test]
    fn finalization_justified_too_recent() {
        let cp = Checkpoint { epoch: 10, root: [0; 32] };
        assert!(verify_finalization_from_justification(10, &cp, 9).is_err());
    }

    #[test]
    fn epoch_balance_computation() {
        let vals: Vec<ValidatorState> = (0..10).map(|i| ValidatorState {
            pubkey: { let mut p = [0u8; 48]; p[0] = i; p },
            effective_balance: 32_000_000_000,
            slashed: false,
            activation_epoch: 0,
            exit_epoch: u64::MAX,
        }).collect();
        let part: Vec<bool> = (0..10).map(|i| i < 7).collect();
        let (att, tot) = compute_epoch_attesting_balance(&vals, &part, 5);
        assert_eq!(tot, 10 * 32_000_000_000);
        assert_eq!(att, 7 * 32_000_000_000);
    }
}
