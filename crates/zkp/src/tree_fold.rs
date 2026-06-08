//! Tree-structured folding for massive numbers of chunk proofs.
//!
//! For ~1.76M chunks from a full Linux boot, linear folding would produce a
//! deep accumulator chain. Tree folding reduces the depth to O(log N):
//!
//! ```text
//! Level 0: chunk₀  chunk₁  chunk₂  chunk₃  chunk₄  chunk₅  ...
//!            \      /         \      /         \      /
//! Level 1:  fold₀₁           fold₂₃           fold₄₅      ...
//!              \               /                  |
//! Level 2:    fold₀₁₂₃                       fold₄₅₆₇     ...
//!                 \                           /
//! ...            single final proof
//! ```
//!
//! The iterator-based API processes chunks on-the-fly without requiring all
//! proofs in memory simultaneously. Pairs are folded as soon as both halves
//! are available, and partial results are kept on a stack.

use crate::prover::ChunkProof;
use crate::recursive::{RecursiveProof, AccumulatedClaim, begin_chunk};
use bls48581::bls48581::big;
use bls48581::bls48581::ecp;
use bls48581::bls48581::rom;
use metavm_core::transcript::Transcript;

/// Combine two scalar accumulators from a tree fold step.
///
/// Empty side passes through (treats empty as "no scalar accumulation").
/// Otherwise add the two scalars over the BLS48-581 scalar field. The byte
/// format is BIG MODBYTES (big-endian).
fn combine_scalar_accs(left: &[u8], right: &[u8]) -> Vec<u8> {
    if left.is_empty() {
        return right.to_vec();
    }
    if right.is_empty() {
        return left.to_vec();
    }
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    let l = big::BIG::frombytes(left);
    let r = big::BIG::frombytes(right);
    let sum = big::BIG::modadd(&l, &r, &modulus);
    let mut out = vec![0u8; big::MODBYTES];
    sum.tobytes(&mut out);
    out
}

/// Fold an iterator of chunk proofs into a single recursive proof using
/// tree-structured aggregation.
///
/// Chunks must arrive in order (chunk_index 0, 1, 2, ...). State chain
/// continuity is verified at each fold step.
///
/// Returns `Err` if the state chain is broken between any adjacent chunks,
/// or if the iterator is empty.
pub fn tree_fold(
    chunks: impl Iterator<Item = ChunkProof>,
) -> Result<RecursiveProof, String> {
    // Stack of partial results at each tree level.
    // stack[i] holds a proof at level i (folded from 2^i chunks) if present.
    let mut stack: Vec<Option<RecursiveProof>> = Vec::new();

    let mut count = 0u64;

    for chunk in chunks {
        // Convert chunk proof to a level-0 recursive proof
        let mut current = begin_chunk(chunk);
        let mut level = 0;

        // Carry: if there's already a proof at this level, fold them together
        // and promote to the next level (like binary addition carry propagation)
        while level < stack.len() {
            if let Some(existing) = stack[level].take() {
                // Fold existing (left) with current (right)
                // State chain: existing.final_state_hash == current.initial_state_hash
                // This is checked inside fold_recursive_proofs
                current = fold_recursive_proofs(&existing, &current)?;
                level += 1;
            } else {
                break;
            }
        }

        // Place the result at the appropriate level
        if level >= stack.len() {
            stack.push(Some(current));
        } else {
            stack[level] = Some(current);
        }

        count += 1;
    }

    if count == 0 {
        return Err("No chunks to fold".to_string());
    }

    // Fold remaining stack entries. Lower levels represent later (rightmost) chunks,
    // so we fold from bottom to top: the bottom is the most recent partial chunk,
    // and higher levels are older (leftward) aggregates. We collect non-None entries
    // and fold them right-to-left to preserve state chain order.
    let remaining: Vec<RecursiveProof> = stack.into_iter()
        .filter_map(|s| s)
        .collect();

    if remaining.is_empty() {
        return Err("Stack was empty after processing".to_string());
    }

    // remaining[0] is the lowest level (rightmost/latest chunk group),
    // remaining[last] is the highest level (leftmost/earliest chunk group).
    // Fold from highest to lowest to maintain state chain order.
    let mut result = remaining.last().unwrap().clone();
    for i in (0..remaining.len() - 1).rev() {
        result = fold_recursive_proofs(&result, &remaining[i])?;
    }

    Ok(result)
}

/// Fold two recursive proofs together, verifying state chain continuity.
///
/// The left proof's final state must match the right proof's initial state.
/// Both accumulators are merged via Fiat-Shamir random linear combination:
///   L_combined = L_left + r * L_right
///   R_combined = R_left + r * R_right
fn fold_recursive_proofs(
    left: &RecursiveProof,
    right: &RecursiveProof,
) -> Result<RecursiveProof, String> {
    // Verify state chain continuity
    if let (Some(left_final), Some(right_initial)) = (&left.final_state_hash, &right.initial_state_hash) {
        if left_final != right_initial {
            return Err(format!(
                "State chain broken during tree fold: left final hash != right initial hash"
            ));
        }
    }

    // Derive folding challenge from Fiat-Shamir
    let mut transcript = Transcript::new(b"metavm-tree-fold");
    transcript.append_message(b"left-l-acc", &left.accumulator.l_acc);
    transcript.append_message(b"left-r-acc", &left.accumulator.r_acc);
    transcript.append_message(b"right-l-acc", &right.accumulator.l_acc);
    transcript.append_message(b"right-r-acc", &right.accumulator.r_acc);

    let r_field = transcript.challenge(b"fold-challenge");
    let r_bytes = r_field.to_bytes();
    let mut r_big_bytes = [0u8; big::MODBYTES];
    for i in 0..32 {
        r_big_bytes[big::MODBYTES - 1 - i] = r_bytes[i];
    }
    let r = big::BIG::frombytes(&r_big_bytes);

    // L_combined = L_left + r * L_right
    let mut l_left = ecp::ECP::frombytes(&left.accumulator.l_acc);
    let l_right = ecp::ECP::frombytes(&right.accumulator.l_acc);
    l_left.add(&l_right.mul(&r));
    l_left.affine();
    let mut l_bytes = vec![0u8; 74];
    l_left.tobytes(&mut l_bytes, true);

    // R_combined = R_left + r * R_right
    let mut r_left = ecp::ECP::frombytes(&left.accumulator.r_acc);
    let r_right = ecp::ECP::frombytes(&right.accumulator.r_acc);
    r_left.add(&r_right.mul(&r));
    r_left.affine();
    let mut r_bytes_out = vec![0u8; 74];
    r_left.tobytes(&mut r_bytes_out, true);

    let combined_acc = AccumulatedClaim {
        l_acc: l_bytes,
        r_acc: r_bytes_out,
        num_folded: left.accumulator.num_folded + right.accumulator.num_folded,
        scalar_acc: combine_scalar_accs(
            &left.accumulator.scalar_acc,
            &right.accumulator.scalar_acc,
        ),
    };

    Ok(RecursiveProof {
        current_proof: right.current_proof.clone(),
        accumulator: combined_acc,
        depth: left.depth + right.depth,
        initial_state_hash: left.initial_state_hash,
        final_state_hash: right.final_state_hash,
    })
}

/// Streaming tree fold that reports progress via a callback.
///
/// Same as [`tree_fold`] but calls `on_progress(chunk_index, total_chunks)`
/// after each chunk is processed. `total_chunks` may be `None` if unknown.
pub fn tree_fold_with_progress(
    chunks: impl Iterator<Item = ChunkProof>,
    total_chunks: Option<u64>,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<RecursiveProof, String> {
    let mut stack: Vec<Option<RecursiveProof>> = Vec::new();
    let mut count = 0u64;

    for chunk in chunks {
        let mut current = begin_chunk(chunk);
        let mut level = 0;

        while level < stack.len() {
            if let Some(existing) = stack[level].take() {
                current = fold_recursive_proofs(&existing, &current)?;
                level += 1;
            } else {
                break;
            }
        }

        if level >= stack.len() {
            stack.push(Some(current));
        } else {
            stack[level] = Some(current);
        }

        count += 1;
        on_progress(count, total_chunks);
    }

    if count == 0 {
        return Err("No chunks to fold".to_string());
    }

    let remaining: Vec<RecursiveProof> = stack.into_iter()
        .filter_map(|s| s)
        .collect();

    if remaining.is_empty() {
        return Err("Stack was empty after processing".to_string());
    }

    let mut result = remaining.last().unwrap().clone();
    for i in (0..remaining.len() - 1).rev() {
        result = fold_recursive_proofs(&result, &remaining[i])?;
    }

    Ok(result)
}

// =========================================================================
// Scheme-generic versions (work with BLS48-581 or BLS12-381)
// =========================================================================

use crate::scheme::CommitmentScheme;
use crate::field::{Scalar, CurveType};
use crate::recursive::begin_chunk_scheme;

/// Fold two recursive proofs together — scheme-generic version.
fn fold_recursive_proofs_scheme(
    left: &RecursiveProof,
    right: &RecursiveProof,
    scheme: &dyn CommitmentScheme,
    curve: CurveType,
) -> Result<RecursiveProof, String> {
    // Verify state chain continuity
    if let (Some(left_final), Some(right_initial)) = (&left.final_state_hash, &right.initial_state_hash) {
        if left_final != right_initial {
            return Err(format!(
                "State chain broken during tree fold: left final hash != right initial hash"
            ));
        }
    }

    // Derive folding challenge from Fiat-Shamir
    let mut transcript = Transcript::new(b"metavm-tree-fold");
    transcript.append_message(b"left-l-acc", &left.accumulator.l_acc);
    transcript.append_message(b"left-r-acc", &left.accumulator.r_acc);
    transcript.append_message(b"right-l-acc", &right.accumulator.l_acc);
    transcript.append_message(b"right-r-acc", &right.accumulator.r_acc);

    let r_field = transcript.challenge(b"fold-challenge");
    let r_bytes = r_field.to_bytes();
    let challenge = Scalar::from_challenge_bytes(&r_bytes, curve);

    let (l_bytes, r_bytes_out) = scheme.fold_accumulator(
        &left.accumulator.l_acc,
        &left.accumulator.r_acc,
        &right.accumulator.l_acc,
        &right.accumulator.r_acc,
        &challenge,
    );

    let combined_acc = AccumulatedClaim {
        l_acc: l_bytes,
        r_acc: r_bytes_out,
        num_folded: left.accumulator.num_folded + right.accumulator.num_folded,
        scalar_acc: combine_scalar_accs(
            &left.accumulator.scalar_acc,
            &right.accumulator.scalar_acc,
        ),
    };

    Ok(RecursiveProof {
        current_proof: right.current_proof.clone(),
        accumulator: combined_acc,
        depth: left.depth + right.depth,
        initial_state_hash: left.initial_state_hash,
        final_state_hash: right.final_state_hash,
    })
}

/// Streaming tree fold that reports progress — scheme-generic version.
pub fn tree_fold_with_progress_scheme(
    chunks: impl Iterator<Item = ChunkProof>,
    total_chunks: Option<u64>,
    mut on_progress: impl FnMut(u64, Option<u64>),
    scheme: &dyn CommitmentScheme,
    curve: CurveType,
) -> Result<RecursiveProof, String> {
    let mut stack: Vec<Option<RecursiveProof>> = Vec::new();
    let mut count = 0u64;

    for chunk in chunks {
        let mut current = begin_chunk_scheme(chunk, scheme, curve);
        let mut level = 0;

        while level < stack.len() {
            if let Some(existing) = stack[level].take() {
                current = fold_recursive_proofs_scheme(&existing, &current, scheme, curve)?;
                level += 1;
            } else {
                break;
            }
        }

        if level >= stack.len() {
            stack.push(Some(current));
        } else {
            stack[level] = Some(current);
        }

        count += 1;
        on_progress(count, total_chunks);
    }

    if count == 0 {
        return Err("No chunks to fold".to_string());
    }

    let remaining: Vec<RecursiveProof> = stack.into_iter()
        .filter_map(|s| s)
        .collect();

    if remaining.is_empty() {
        return Err("Stack was empty after processing".to_string());
    }

    let mut result = remaining.last().unwrap().clone();
    for i in (0..remaining.len() - 1).rev() {
        result = fold_recursive_proofs_scheme(&result, &remaining[i], scheme, curve)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{Commitment, BatchProof};
    use crate::prover::{ExecutionProof, ChunkProof};

    fn dummy_chunk(id: u8, chunk_index: u64, initial: [u8; 32], final_h: [u8; 32]) -> ChunkProof {
        // First byte must be 0x02 or 0x03 for compressed G1 points to avoid
        // ECP::frombytes trying to parse as uncompressed (0x04) and reading OOB.
        let mut comm_bytes = vec![id; 74];
        comm_bytes[0] = 0x02;
        let mut proof_bytes = vec![id; 74];
        proof_bytes[0] = 0x02;
        ChunkProof {
            execution_proof: ExecutionProof {
                column_commitments: vec![Commitment(comm_bytes.clone())],
                quotient_commitments: vec![Commitment(comm_bytes)],
                evaluations: vec![vec![id; big::MODBYTES]],
                opening_proof: BatchProof {
                    d: vec![],
                    proof: proof_bytes,
                },
                num_steps: 100,
                domain_size: 128,
                num_quotient_chunks: 1,
                shifted_evaluations: Vec::new(),
                shifted_opening_proof: None,
                logup_commitments: Vec::new(),
                logup_evaluations: Vec::new(),
                logup_shifted_evaluations: Vec::new(),
                logup_opening_proof: None,
                logup_shifted_opening_proof: None,
                bitwise_commitments: Vec::new(),
                bitwise_evaluations: Vec::new(),
                bitwise_shifted_evaluations: Vec::new(),
                bitwise_opening_proof: None,
                bitwise_shifted_opening_proof: None,
                perm_commitments: Vec::new(),
                perm_evaluations: Vec::new(),
                perm_shifted_evaluations: Vec::new(),
                perm_opening_proof: None,
                perm_shifted_opening_proof: None,
                oracle_data: Vec::new(),
                reg_perm_commitments: Vec::new(),
                reg_perm_evaluations: Vec::new(),
                reg_perm_shifted_evaluations: Vec::new(),
                reg_perm_opening_proof: None,
                reg_perm_shifted_opening_proof: None,
                frame_perm_commitment: None,
                frame_perm_evaluation: None,
                frame_perm_shifted_evaluation: None,
                frame_perm_opening_proof: None,
                frame_perm_shifted_opening_proof: None,
                frame_perm_pop_shifted_evaluations: Vec::new(),
                frame_perm_pop_shifted_opening_proof: None,
            },
            initial_state_hash: initial,
            final_state_hash: final_h,
            chunk_index,
        }
    }

    fn make_chain(n: usize) -> Vec<ChunkProof> {
        let mut chunks = Vec::with_capacity(n);
        for i in 0..n {
            let mut initial = [0u8; 32];
            initial[0] = i as u8;
            let mut final_h = [0u8; 32];
            final_h[0] = (i + 1) as u8;
            chunks.push(dummy_chunk(i as u8, i as u64, initial, final_h));
        }
        chunks
    }

    #[test]
    fn test_tree_fold_single_chunk() {
        let chunks = make_chain(1);
        let result = tree_fold(chunks.into_iter()).unwrap();
        assert_eq!(result.depth, 1);
        assert_eq!(result.initial_state_hash, Some([0u8; 32]));
        let mut expected_final = [0u8; 32];
        expected_final[0] = 1;
        assert_eq!(result.final_state_hash, Some(expected_final));
    }

    #[test]
    fn test_tree_fold_two_chunks() {
        let chunks = make_chain(2);
        let result = tree_fold(chunks.into_iter()).unwrap();
        assert_eq!(result.depth, 2);
        assert_eq!(result.initial_state_hash, Some([0u8; 32]));
        let mut expected_final = [0u8; 32];
        expected_final[0] = 2;
        assert_eq!(result.final_state_hash, Some(expected_final));
    }

    #[test]
    fn test_tree_fold_power_of_two() {
        let chunks = make_chain(8);
        let result = tree_fold(chunks.into_iter()).unwrap();
        assert_eq!(result.depth, 8);
        let mut expected_final = [0u8; 32];
        expected_final[0] = 8;
        assert_eq!(result.final_state_hash, Some(expected_final));
    }

    #[test]
    fn test_tree_fold_non_power_of_two() {
        let chunks = make_chain(5);
        let result = tree_fold(chunks.into_iter()).unwrap();
        assert_eq!(result.depth, 5);
        let mut expected_final = [0u8; 32];
        expected_final[0] = 5;
        assert_eq!(result.final_state_hash, Some(expected_final));
    }

    #[test]
    fn test_tree_fold_empty() {
        let chunks: Vec<ChunkProof> = vec![];
        let result = tree_fold(chunks.into_iter());
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_fold_broken_chain() {
        let mut chunks = make_chain(3);
        // Break the chain: chunk 1's initial doesn't match chunk 0's final
        chunks[1].initial_state_hash = [99u8; 32];
        let result = tree_fold(chunks.into_iter());
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_fold_with_progress() {
        let chunks = make_chain(4);
        let mut progress_count = 0u64;
        let result = tree_fold_with_progress(
            chunks.into_iter(),
            Some(4),
            |count, total| {
                progress_count = count;
                assert_eq!(total, Some(4));
            },
        ).unwrap();
        assert_eq!(progress_count, 4);
        assert_eq!(result.depth, 4);
    }
}
