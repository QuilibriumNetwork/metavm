//! Gasper finality reference — host-side, pre-AIR.
//!
//! Implements Casper FFG justification + finalization on top of the beacon
//! types in [`crate::beacon`]. The goal is to produce a deterministic
//! `(finalized_block_root, total_effective_balance_weight)` output that
//! a future AIR will constrain.
//!
//! Scope:
//! - [`ValidatorRegistry`] with active-set and effective-balance helpers
//! - Casper FFG slashing rules ([`is_slashable_attestation_data`])
//! - [`JustificationStore`] with the phase-0 4-epoch justification bitfield
//! - [`process_justification_and_finalization`] — runs 2/3-supermajority checks
//!   and applies all four FFG finalization rules (2a, 2b, 3a, 3b)
//! - [`stake_weighted_finalization_check`] — the end-to-end prover output
//!
//! Out of scope (deferred): signature verification helpers are exposed in
//! [`crate::bls_sig`] directly; this module does not duplicate them. Fork
//! domain handling is reduced to passing `domain: [u8; 32]` through since the
//! AIR will see it as a public input.

use crate::beacon::{
    AttestationData, Checkpoint, Epoch, Gwei, IndexedAttestation, Validator, ValidatorIndex,
};

/// Beacon-chain BLS pop ciphersuite domain-separation tag.
pub const BEACON_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

/// Errors that [`process_justification_and_finalization`] and related
/// helpers can surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalityError {
    /// The active validator set is empty — 2/3-supermajority is undefined.
    NoActiveValidators,
    /// An attesting-index referenced a slot beyond the registry.
    ValidatorIndexOutOfRange(ValidatorIndex),
}

/// A snapshot of the validator set plus their *current* balances.
///
/// `validators[i].effective_balance` is the slot used for supermajority
/// accounting; `balances[i]` tracks the finer-grained actual balance (which
/// only affects when `effective_balance` gets re-quantized). For this
/// reference the two move together via [`Validator::effective_balance`].
#[derive(Debug, Clone)]
pub struct ValidatorRegistry {
    pub validators: Vec<Validator>,
    pub balances: Vec<Gwei>,
}

impl ValidatorRegistry {
    pub fn new(validators: Vec<Validator>, balances: Vec<Gwei>) -> Self {
        debug_assert_eq!(validators.len(), balances.len());
        ValidatorRegistry {
            validators,
            balances,
        }
    }

    /// All validators active at `epoch`: activated but not exited.
    pub fn active_indices(&self, epoch: Epoch) -> Vec<ValidatorIndex> {
        self.validators
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v.activation_epoch <= epoch && epoch < v.exit_epoch {
                    Some(i as ValidatorIndex)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Sum of `effective_balance` across all active validators at `epoch`.
    pub fn total_active_balance(&self, epoch: Epoch) -> Gwei {
        self.active_indices(epoch)
            .into_iter()
            .map(|i| self.validators[i as usize].effective_balance)
            .sum()
    }

    /// Sum of `effective_balance` across `indices`. Duplicates are counted
    /// once; out-of-range indices are ignored (caller should validate).
    pub fn total_balance(&self, indices: &[ValidatorIndex]) -> Gwei {
        let mut seen = vec![false; self.validators.len()];
        let mut total: Gwei = 0;
        for &idx in indices {
            let i = idx as usize;
            if i >= self.validators.len() || seen[i] {
                continue;
            }
            seen[i] = true;
            total = total.saturating_add(self.validators[i].effective_balance);
        }
        total
    }

    /// Checked lookup.
    pub fn get(&self, idx: ValidatorIndex) -> Result<&Validator, FinalityError> {
        self.validators
            .get(idx as usize)
            .ok_or(FinalityError::ValidatorIndexOutOfRange(idx))
    }
}

/// Casper FFG slashing rules: an attester double-votes or surround-votes.
///
/// The phase-0 spec defines two slashable conditions on a pair of
/// [`AttestationData`] values `a1`, `a2` from the same attester:
/// 1. **Double vote**: `a1.target.epoch == a2.target.epoch` but the two
///    attestations differ (typically on `beacon_block_root`).
/// 2. **Surround vote**: one strictly surrounds the other, i.e.
///    `a1.source.epoch < a2.source.epoch` and `a2.target.epoch < a1.target.epoch`
///    (or vice versa).
pub fn is_slashable_attestation_data(a1: &AttestationData, a2: &AttestationData) -> bool {
    if a1 == a2 {
        return false;
    }
    // Double vote.
    if a1.target.epoch == a2.target.epoch {
        return true;
    }
    // Surround vote (either direction).
    if a1.source.epoch < a2.source.epoch && a2.target.epoch < a1.target.epoch {
        return true;
    }
    if a2.source.epoch < a1.source.epoch && a1.target.epoch < a2.target.epoch {
        return true;
    }
    false
}

/// Phase-0 justification bitfield + checkpoints.
///
/// `justification_bits[0]` is the most recent epoch; bits are shifted left
/// (higher index = older epoch) by [`process_justification_and_finalization`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JustificationStore {
    pub previous_justified: Checkpoint,
    pub current_justified: Checkpoint,
    pub finalized: Checkpoint,
    pub justification_bits: [bool; 4],
}

impl JustificationStore {
    pub fn genesis() -> Self {
        let zero = Checkpoint {
            epoch: 0,
            root: [0u8; 32],
        };
        JustificationStore {
            previous_justified: zero.clone(),
            current_justified: zero.clone(),
            finalized: zero,
            justification_bits: [false; 4],
        }
    }
}

/// Total `effective_balance` of the subset of `active_indices` whose
/// attestation target matched `target_checkpoint`.
fn attesting_balance_for_target(
    registry: &ValidatorRegistry,
    epoch_attestations: &[IndexedAttestation],
    target_checkpoint: &Checkpoint,
) -> Gwei {
    let mut seen = vec![false; registry.validators.len()];
    let mut total: Gwei = 0;
    for att in epoch_attestations {
        if att.data.target != *target_checkpoint {
            continue;
        }
        for &idx in &att.attesting_indices {
            let i = idx as usize;
            if i >= registry.validators.len() || seen[i] {
                continue;
            }
            seen[i] = true;
            total = total.saturating_add(registry.validators[i].effective_balance);
        }
    }
    total
}

/// Per-epoch Casper FFG update.
///
/// Given the current/previous-epoch attestations, shift the justification
/// bitfield, apply 2/3-supermajority rules to justify current/previous
/// checkpoints, and advance `finalized` via the four FFG finalization rules:
///
/// - **Rule 2a**: bits 1,2,3 set and the 2nd-oldest justified checkpoint
///   matches `previous_justified` from two epochs back → finalize.
/// - **Rule 2b**: bits 1,2 set with current justified source matching
///   → finalize the previous-justified (2-epoch rule, older variant).
/// - **Rule 3a**: bits 0,1,2 set with consistent chain → finalize.
/// - **Rule 3b**: bits 0,1 set, 2-epoch chain → finalize current_justified.
///
/// The `current_epoch_target` and `previous_epoch_target` arguments are the
/// checkpoints the honest majority should have voted for in those epochs
/// (the block-root at the epoch-start slot). Callers supply them.
pub fn process_justification_and_finalization(
    store: &mut JustificationStore,
    current_epoch: Epoch,
    current_epoch_target: Checkpoint,
    previous_epoch_target: Checkpoint,
    previous_epoch_attestations: &[IndexedAttestation],
    current_epoch_attestations: &[IndexedAttestation],
    registry: &ValidatorRegistry,
) -> Result<(), FinalityError> {
    let total_active = registry.total_active_balance(current_epoch);
    if total_active == 0 {
        return Err(FinalityError::NoActiveValidators);
    }

    // Save old previous/current justified for FFG rules below.
    let old_previous_justified = store.previous_justified.clone();
    let old_current_justified = store.current_justified.clone();

    // Shift bits one position (the oldest bit falls off).
    let old_bits = store.justification_bits;
    store.justification_bits = [false, old_bits[0], old_bits[1], old_bits[2]];

    // Previous-epoch supermajority → justify previous_epoch_target.
    let prev_balance = attesting_balance_for_target(
        registry,
        previous_epoch_attestations,
        &previous_epoch_target,
    );
    if prev_balance.saturating_mul(3) >= total_active.saturating_mul(2) {
        store.current_justified = previous_epoch_target.clone();
        store.justification_bits[1] = true;
    }

    // Current-epoch supermajority → justify current_epoch_target.
    let curr_balance = attesting_balance_for_target(
        registry,
        current_epoch_attestations,
        &current_epoch_target,
    );
    if curr_balance.saturating_mul(3) >= total_active.saturating_mul(2) {
        store.current_justified = current_epoch_target.clone();
        store.justification_bits[0] = true;
    }

    // The previous_justified slot advances to the value that was
    // current_justified before this epoch's updates.
    store.previous_justified = old_current_justified.clone();

    // FFG finalization rules. `bits` refers to the NEW bits array (post-shift).
    // Spec names: bits indexed from most-recent epoch = index 0.
    //
    // Rule 2a: bits 1,2,3 set + old_previous_justified (3 epochs ago) == finalized target source.
    // In this reference we simplify: if bits[1..=3] are all true, finalize old_previous_justified.
    if store.justification_bits[1] && store.justification_bits[2] && store.justification_bits[3]
        && old_previous_justified.epoch + 3 == current_epoch
    {
        store.finalized = old_previous_justified.clone();
    }
    // Rule 2b: bits 1,2 set + old_previous_justified 2 epochs ago.
    if store.justification_bits[1] && store.justification_bits[2]
        && old_previous_justified.epoch + 2 == current_epoch
    {
        store.finalized = old_previous_justified;
    }
    // Rule 3a: bits 0,1,2 set + old_current_justified 2 epochs ago.
    if store.justification_bits[0] && store.justification_bits[1] && store.justification_bits[2]
        && old_current_justified.epoch + 2 == current_epoch
    {
        store.finalized = old_current_justified.clone();
    }
    // Rule 3b: bits 0,1 set + old_current_justified 1 epoch ago.
    if store.justification_bits[0] && store.justification_bits[1]
        && old_current_justified.epoch + 1 == current_epoch
    {
        store.finalized = old_current_justified;
    }

    Ok(())
}

/// Result of a stake-weighted finality check — the public output a future
/// AIR will commit to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalityOutput {
    /// Block root the attestations are finalizing.
    pub finalized_block_root: [u8; 32],
    /// Total effective balance (Gwei) backing this finalization.
    /// This is the "ETH securing the protocol" output.
    pub total_effective_balance_weight: Gwei,
    /// Number of unique attesting validators.
    pub num_unique_attesters: u64,
}

/// Compute the stake-weighted finality output given a finalized checkpoint
/// and the attestations that justify it. Attestations whose `target` does
/// not match `finalized` are ignored; duplicate attesting indices across
/// attestations are deduplicated.
pub fn stake_weighted_finalization_check(
    finalized: &Checkpoint,
    registry: &ValidatorRegistry,
    attestations: &[IndexedAttestation],
) -> FinalityOutput {
    let mut seen = vec![false; registry.validators.len()];
    let mut total: Gwei = 0;
    let mut count: u64 = 0;
    for att in attestations {
        if att.data.target != *finalized {
            continue;
        }
        for &idx in &att.attesting_indices {
            let i = idx as usize;
            if i >= registry.validators.len() || seen[i] {
                continue;
            }
            seen[i] = true;
            count += 1;
            total = total.saturating_add(registry.validators[i].effective_balance);
        }
    }
    FinalityOutput {
        finalized_block_root: finalized.root,
        total_effective_balance_weight: total,
        num_unique_attesters: count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::AttestationData;

    fn mk_validator(eff_balance: Gwei, activation: Epoch, exit: Epoch) -> Validator {
        Validator {
            pubkey: [0u8; 48],
            withdrawal_credentials: [0u8; 32],
            effective_balance: eff_balance,
            slashed: false,
            activation_eligibility_epoch: activation,
            activation_epoch: activation,
            exit_epoch: exit,
            withdrawable_epoch: exit.saturating_add(256),
        }
    }

    /// Use a large-but-finite `exit_epoch` so `exit + 256` doesn't overflow.
    const FAR_FUTURE_EPOCH: Epoch = 1u64 << 40;

    fn mk_registry(n: usize, eff_balance: Gwei) -> ValidatorRegistry {
        let validators: Vec<Validator> =
            (0..n).map(|_| mk_validator(eff_balance, 0, FAR_FUTURE_EPOCH)).collect();
        let balances = vec![eff_balance; n];
        ValidatorRegistry::new(validators, balances)
    }

    fn checkpoint(epoch: Epoch, tag: u8) -> Checkpoint {
        let mut root = [0u8; 32];
        root[0] = tag;
        Checkpoint { epoch, root }
    }

    fn att_data(source: Checkpoint, target: Checkpoint, tag: u8) -> AttestationData {
        let mut beacon_block_root = [0u8; 32];
        beacon_block_root[0] = tag;
        AttestationData {
            slot: target.epoch * 32,
            index: 0,
            beacon_block_root,
            source,
            target,
        }
    }

    fn ia(data: AttestationData, indices: Vec<ValidatorIndex>) -> IndexedAttestation {
        IndexedAttestation {
            attesting_indices: indices,
            data,
            signature: [0u8; 96],
        }
    }

    // ── Slashing rules ────────────────────────────────────────────────

    #[test]
    fn slashable_double_vote() {
        let source = checkpoint(1, 0);
        let target = checkpoint(2, 0);
        let a1 = att_data(source.clone(), target.clone(), 0x11);
        let a2 = att_data(source, target, 0x22); // different block root
        assert!(is_slashable_attestation_data(&a1, &a2));
    }

    #[test]
    fn slashable_surround_vote() {
        // a1 surrounds a2: source(a1) < source(a2) && target(a2) < target(a1).
        let a1 = att_data(checkpoint(1, 0), checkpoint(5, 0), 0x11);
        let a2 = att_data(checkpoint(2, 0), checkpoint(4, 0), 0x22);
        assert!(is_slashable_attestation_data(&a1, &a2));
        assert!(is_slashable_attestation_data(&a2, &a1)); // symmetric
    }

    #[test]
    fn not_slashable_disjoint() {
        let a1 = att_data(checkpoint(1, 0), checkpoint(2, 0), 0x11);
        let a2 = att_data(checkpoint(3, 0), checkpoint(4, 0), 0x22);
        assert!(!is_slashable_attestation_data(&a1, &a2));
    }

    #[test]
    fn not_slashable_identical() {
        let a = att_data(checkpoint(1, 0), checkpoint(2, 0), 0x11);
        assert!(!is_slashable_attestation_data(&a, &a));
    }

    // ── Registry ──────────────────────────────────────────────────────

    #[test]
    fn registry_active_indices_respects_activation_and_exit() {
        let v_active = mk_validator(32_000_000_000, 0, 100);
        let v_pending = mk_validator(32_000_000_000, 50, 200);
        let v_exited = mk_validator(32_000_000_000, 0, 10);
        let reg = ValidatorRegistry::new(
            vec![v_active, v_pending, v_exited],
            vec![32_000_000_000; 3],
        );
        assert_eq!(reg.active_indices(25), vec![0]); // only first is active at 25
        assert_eq!(reg.active_indices(75), vec![0, 1]);
        assert_eq!(reg.active_indices(5), vec![0, 2]);
    }

    #[test]
    fn registry_total_balance_dedups() {
        let reg = mk_registry(5, 32_000_000_000);
        let idx = vec![0, 1, 2, 0, 1]; // duplicates
        assert_eq!(reg.total_balance(&idx), 3 * 32_000_000_000);
    }

    // ── Justification ─────────────────────────────────────────────────

    #[test]
    fn justify_current_epoch_on_supermajority() {
        let reg = mk_registry(10, 32_000_000_000);
        let target = checkpoint(2, 0xAA);
        let att = ia(
            att_data(checkpoint(1, 0), target.clone(), 0xAA),
            (0..7).collect(), // 7 of 10 attest → 70% ≥ 66.6…%
        );
        let mut store = JustificationStore::genesis();
        process_justification_and_finalization(
            &mut store,
            2,
            target.clone(),
            checkpoint(1, 0x00),
            &[],
            &[att],
            &reg,
        )
        .unwrap();
        assert_eq!(store.current_justified, target);
        assert!(store.justification_bits[0]);
    }

    #[test]
    fn no_justification_below_supermajority() {
        let reg = mk_registry(10, 32_000_000_000);
        let target = checkpoint(2, 0xAA);
        let att = ia(
            att_data(checkpoint(1, 0), target.clone(), 0xAA),
            (0..6).collect(), // 6 of 10 → 60% < 66.6…%
        );
        let mut store = JustificationStore::genesis();
        process_justification_and_finalization(
            &mut store,
            2,
            target,
            checkpoint(1, 0x00),
            &[],
            &[att],
            &reg,
        )
        .unwrap();
        assert_eq!(store.current_justified, Checkpoint { epoch: 0, root: [0u8; 32] });
        assert!(!store.justification_bits[0]);
    }

    // ── Finalization ──────────────────────────────────────────────────

    #[test]
    fn finalize_rule_3b_two_epochs() {
        // Rule 3b: bits[0], bits[1] set + old_current_justified 1 epoch ago finalizes it.
        let reg = mk_registry(10, 32_000_000_000);
        let cp_prev = checkpoint(1, 0xBB);
        let cp_curr = checkpoint(2, 0xCC);

        let mut store = JustificationStore::genesis();
        // Seed: previous epoch had a justified checkpoint at epoch 1.
        store.current_justified = cp_prev.clone();
        store.justification_bits[0] = true;

        // Now epoch 2: supermajority on cp_curr; cp_prev is old_current_justified (epoch 1 = current_epoch - 1).
        let att = ia(
            att_data(cp_prev.clone(), cp_curr.clone(), 0xCC),
            (0..7).collect(),
        );
        process_justification_and_finalization(
            &mut store,
            2,
            cp_curr.clone(),
            cp_prev.clone(),
            &[],
            &[att],
            &reg,
        )
        .unwrap();
        // bits[0] = current justified (cp_curr), bits[1] = old bits[0] = true.
        assert!(store.justification_bits[0] && store.justification_bits[1]);
        // Rule 3b fires: finalize cp_prev (old_current_justified at epoch 1, current_epoch = 2).
        assert_eq!(store.finalized, cp_prev);
    }

    // ── Stake-weighted output ─────────────────────────────────────────

    #[test]
    fn stake_weighted_output_70_of_100() {
        let reg = mk_registry(100, 32_000_000_000);
        let finalized = checkpoint(5, 0xDD);
        let att = ia(
            att_data(checkpoint(4, 0), finalized.clone(), 0xDD),
            (0..70).collect(),
        );
        let out = stake_weighted_finalization_check(&finalized, &reg, &[att]);
        assert_eq!(out.finalized_block_root, finalized.root);
        assert_eq!(out.total_effective_balance_weight, 70 * 32_000_000_000);
        assert_eq!(out.num_unique_attesters, 70);
    }

    #[test]
    fn stake_weighted_output_dedups_across_attestations() {
        let reg = mk_registry(10, 32_000_000_000);
        let finalized = checkpoint(3, 0xEE);
        let att1 = ia(
            att_data(checkpoint(2, 0), finalized.clone(), 0xEE),
            vec![0, 1, 2, 3],
        );
        let att2 = ia(
            att_data(checkpoint(2, 0), finalized.clone(), 0xEE),
            vec![2, 3, 4, 5],
        );
        // Unique set: {0,1,2,3,4,5} — 6 attesters.
        let out = stake_weighted_finalization_check(&finalized, &reg, &[att1, att2]);
        assert_eq!(out.num_unique_attesters, 6);
        assert_eq!(out.total_effective_balance_weight, 6 * 32_000_000_000);
    }

    #[test]
    fn stake_weighted_output_ignores_wrong_target() {
        let reg = mk_registry(10, 32_000_000_000);
        let finalized = checkpoint(3, 0xEE);
        let wrong_target = checkpoint(3, 0xFF);
        let att_ok = ia(
            att_data(checkpoint(2, 0), finalized.clone(), 0xEE),
            vec![0, 1, 2],
        );
        let att_wrong = ia(
            att_data(checkpoint(2, 0), wrong_target, 0xFF),
            vec![3, 4, 5],
        );
        let out = stake_weighted_finalization_check(&finalized, &reg, &[att_ok, att_wrong]);
        assert_eq!(out.num_unique_attesters, 3);
        assert_eq!(out.total_effective_balance_weight, 3 * 32_000_000_000);
    }

    // ── Error paths ───────────────────────────────────────────────────

    #[test]
    fn no_active_validators_errors() {
        let reg = ValidatorRegistry::new(vec![], vec![]);
        let mut store = JustificationStore::genesis();
        let err = process_justification_and_finalization(
            &mut store,
            1,
            checkpoint(1, 0),
            checkpoint(0, 0),
            &[],
            &[],
            &reg,
        );
        assert_eq!(err, Err(FinalityError::NoActiveValidators));
    }
}
