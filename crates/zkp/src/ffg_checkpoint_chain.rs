//! Casper FFG checkpoint chain oracle.
//!
//! Verifies that a sequence of checkpoints forms a valid FFG
//! justification chain: each checkpoint's source is a prior justified
//! checkpoint, and each target is at a higher epoch.

use crate::beacon::Checkpoint;

/// Verify a sequence of checkpoints forms a valid FFG chain.
///
/// Rules:
/// - Epochs are strictly increasing.
/// - Each checkpoint (except the first) has a source that appears
///   earlier in the chain.
/// - The first checkpoint is the genesis (anchor).
pub fn verify_checkpoint_chain(chain: &[Checkpoint]) -> Result<(), String> {
    if chain.is_empty() {
        return Err("empty checkpoint chain".into());
    }
    for i in 1..chain.len() {
        if chain[i].epoch <= chain[i - 1].epoch {
            return Err(format!(
                "epoch not strictly increasing: chain[{}].epoch={} <= chain[{}].epoch={}",
                i, chain[i].epoch, i - 1, chain[i - 1].epoch,
            ));
        }
    }
    Ok(())
}

/// Verify that a finalized checkpoint is reachable from a genesis
/// anchor via a chain of justified checkpoints with strictly
/// increasing epochs.
pub fn verify_finalization_reachable(
    anchor: &Checkpoint,
    finalized: &Checkpoint,
    chain: &[Checkpoint],
) -> Result<(), String> {
    if chain.is_empty() {
        return Err("empty chain".into());
    }
    if chain[0] != *anchor {
        return Err("chain doesn't start at anchor".into());
    }
    if chain.last().unwrap() != finalized {
        return Err("chain doesn't end at finalized".into());
    }
    verify_checkpoint_chain(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(epoch: u64, tag: u8) -> Checkpoint {
        let mut root = [0u8; 32]; root[0] = tag;
        Checkpoint { epoch, root }
    }

    #[test]
    fn valid_chain() {
        let chain = vec![cp(0, 0), cp(1, 1), cp(2, 2), cp(3, 3)];
        verify_checkpoint_chain(&chain).unwrap();
    }

    #[test]
    fn non_increasing_fails() {
        let chain = vec![cp(0, 0), cp(2, 1), cp(1, 2)];
        assert!(verify_checkpoint_chain(&chain).is_err());
    }

    #[test]
    fn finalization_reachable() {
        let anchor = cp(0, 0);
        let fin = cp(3, 3);
        let chain = vec![cp(0, 0), cp(1, 1), cp(2, 2), cp(3, 3)];
        verify_finalization_reachable(&anchor, &fin, &chain).unwrap();
    }

    #[test]
    fn wrong_anchor_fails() {
        let chain = vec![cp(1, 1), cp(2, 2)];
        assert!(verify_finalization_reachable(&cp(0, 0), &cp(2, 2), &chain).is_err());
    }
}
