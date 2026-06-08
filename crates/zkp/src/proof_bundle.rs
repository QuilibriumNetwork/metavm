//! Proof bundle metadata oracle.
//!
//! Wraps `full_proof_oracle::FullProof` with metadata about which
//! AIRs are included, what claims are bound, and what's still
//! oracle-only. Used for proof composition tracking and for
//! validating that a proof bundle has all required components
//! before invoking the underlying verification chain.

use crate::full_proof_oracle::FullProof;
use crate::validator_set::ValidatorState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofComponent {
    EvmExecution,
    TxMptInclusion,
    ReceiptMptInclusion,
    AccountMptInclusion,
    StorageMptInclusion,
    BlockHeaderKeccak,
    PayloadBlockHashBridge,
    BeaconBodyHtr,
    BeaconHeaderHtr,
    FfgSupermajority,
    SyncCommitteeBls, // oracle-only
    SecpSenderRecovery, // oracle-only
    BlsValidatorRegistry, // oracle-only
}

#[derive(Clone, Debug)]
pub struct ProofBundleMetadata {
    pub algebraic: Vec<ProofComponent>,
    pub oracle: Vec<ProofComponent>,
}

impl ProofBundleMetadata {
    /// Current state of the proof system as of session: which components
    /// are algebraic vs oracle-only.
    pub fn current_state() -> Self {
        Self {
            algebraic: vec![
                ProofComponent::EvmExecution, // partially — many opcodes are oracle skips
                ProofComponent::BlockHeaderKeccak,
                ProofComponent::PayloadBlockHashBridge,
                ProofComponent::BeaconBodyHtr,
                ProofComponent::BeaconHeaderHtr,
                ProofComponent::FfgSupermajority,
            ],
            oracle: vec![
                ProofComponent::TxMptInclusion,
                ProofComponent::ReceiptMptInclusion,
                ProofComponent::AccountMptInclusion,
                ProofComponent::StorageMptInclusion,
                ProofComponent::SyncCommitteeBls,
                ProofComponent::SecpSenderRecovery,
                ProofComponent::BlsValidatorRegistry,
            ],
        }
    }

    pub fn coverage_ratio(&self) -> f64 {
        let total = self.algebraic.len() + self.oracle.len();
        if total == 0 { return 0.0; }
        self.algebraic.len() as f64 / total as f64
    }

    pub fn is_algebraic(&self, c: &ProofComponent) -> bool {
        self.algebraic.contains(c)
    }
}

pub struct ProofBundle {
    pub proof: FullProof,
    pub validators: Vec<ValidatorState>,
    pub participating: Vec<bool>,
    pub epoch: u64,
    pub metadata: ProofBundleMetadata,
}

pub fn verify_proof_bundle(bundle: &ProofBundle) -> Result<(), String> {
    if bundle.validators.len() != bundle.participating.len() {
        return Err(format!(
            "validator count {} != participating count {}",
            bundle.validators.len(), bundle.participating.len(),
        ));
    }
    if bundle.validators.is_empty() {
        return Err("no validators".into());
    }
    crate::full_proof_oracle::verify_full_proof(
        &bundle.proof,
        &bundle.validators,
        &bundle.participating,
        bundle.epoch,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_coverage_nonzero() {
        let m = ProofBundleMetadata::current_state();
        assert!(m.coverage_ratio() > 0.0);
        assert!(m.coverage_ratio() < 1.0);
    }

    #[test]
    fn metadata_classifies_components() {
        let m = ProofBundleMetadata::current_state();
        assert!(m.is_algebraic(&ProofComponent::EvmExecution));
        assert!(m.is_algebraic(&ProofComponent::FfgSupermajority));
        assert!(!m.is_algebraic(&ProofComponent::SyncCommitteeBls));
        assert!(!m.is_algebraic(&ProofComponent::SecpSenderRecovery));
    }

    #[test]
    fn metadata_components_unique() {
        let m = ProofBundleMetadata::current_state();
        for c in &m.algebraic {
            assert!(!m.oracle.contains(c), "{:?} in both lists", c);
        }
    }
}
