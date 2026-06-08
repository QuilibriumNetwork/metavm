//! Multi-block proof oracle.
//!
//! Composes per-block `full_proof_oracle::verify_full_proof`
//! invocations with the execution chain parent_hash linkage,
//! producing a single verification for a sequence of finalized
//! Ethereum blocks anchored to a FFG-finalized epoch.

use crate::block_chain::verify_block_chain;
use crate::block_header::BlockHeader;
use crate::full_proof_oracle::{FullProof, verify_full_proof};
use crate::validator_set::ValidatorState;

pub fn verify_multi_block_chain(
    proofs: &[FullProof],
    validators: &[ValidatorState],
    participating: &[bool],
    epoch: u64,
) -> Result<(), String> {
    // Verify each proof's individual finality conditions.
    for (i, proof) in proofs.iter().enumerate() {
        verify_full_proof(proof, validators, participating, epoch)
            .map_err(|e| format!("proof[{}]: {}", i, e))?;
    }

    // Verify the execution chain linkage.
    let headers: Vec<BlockHeader> = proofs.iter()
        .map(|p| p.execution.header.clone())
        .collect();
    verify_block_chain(&headers)
        .map_err(|e| format!("chain: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_proofs_pass() {
        let validators: Vec<ValidatorState> = vec![];
        let part: Vec<bool> = vec![];
        verify_multi_block_chain(&[], &validators, &part, 10).unwrap();
    }
}
