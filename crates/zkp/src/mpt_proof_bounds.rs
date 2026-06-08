//! MPT proof bounds checking.
//!
//! Sanity checks on MPT proof structure: depth, node size, and
//! per-proof byte count limits. Prevents DoS via overly large proofs
//! and validates proof structure before expensive MPT verification.

pub const MAX_PROOF_DEPTH: usize = 64;
pub const MAX_NODE_BYTES: usize = 532; // branch node max: 17 * 32 + RLP overhead
pub const MAX_PROOF_BYTES: usize = MAX_PROOF_DEPTH * MAX_NODE_BYTES;

pub fn validate_proof_structure(proof: &[Vec<u8>]) -> Result<(), String> {
    if proof.len() > MAX_PROOF_DEPTH {
        return Err(format!(
            "proof depth {} exceeds max {}",
            proof.len(), MAX_PROOF_DEPTH,
        ));
    }
    for (i, node) in proof.iter().enumerate() {
        if node.is_empty() {
            return Err(format!("proof[{}] is empty", i));
        }
        if node.len() > MAX_NODE_BYTES {
            return Err(format!(
                "proof[{}] size {} exceeds max {}",
                i, node.len(), MAX_NODE_BYTES,
            ));
        }
    }
    let total: usize = proof.iter().map(|n| n.len()).sum();
    if total > MAX_PROOF_BYTES {
        return Err(format!(
            "total proof bytes {} exceeds max {}",
            total, MAX_PROOF_BYTES,
        ));
    }
    Ok(())
}

pub fn validate_proof_batch(proofs: &[Vec<Vec<u8>>]) -> Result<(), String> {
    for (i, proof) in proofs.iter().enumerate() {
        validate_proof_structure(proof)
            .map_err(|e| format!("proof[{}]: {}", i, e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_proof_passes() {
        let proof = vec![vec![0u8; 100], vec![0u8; 200]];
        validate_proof_structure(&proof).unwrap();
    }

    #[test]
    fn empty_proof_passes() {
        validate_proof_structure(&[]).unwrap();
    }

    #[test]
    fn too_deep_fails() {
        let proof: Vec<Vec<u8>> = (0..MAX_PROOF_DEPTH + 1).map(|_| vec![0u8; 10]).collect();
        assert!(validate_proof_structure(&proof).is_err());
    }

    #[test]
    fn empty_node_fails() {
        let proof = vec![vec![0u8; 100], vec![]];
        assert!(validate_proof_structure(&proof).is_err());
    }

    #[test]
    fn oversized_node_fails() {
        let proof = vec![vec![0u8; MAX_NODE_BYTES + 1]];
        assert!(validate_proof_structure(&proof).is_err());
    }

    #[test]
    fn batch_validation() {
        let proofs = vec![
            vec![vec![0u8; 100]],
            vec![vec![0u8; 200], vec![0u8; 300]],
        ];
        validate_proof_batch(&proofs).unwrap();
    }

    #[test]
    fn batch_rejects_one_bad() {
        let proofs = vec![
            vec![vec![0u8; 100]],
            vec![vec![]],
        ];
        assert!(validate_proof_batch(&proofs).is_err());
    }
}
