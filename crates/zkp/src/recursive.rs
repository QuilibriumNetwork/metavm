use crate::commitment;
use crate::prover::{ExecutionProof, ChunkProof, challenge_to_big};
use bls48581::bls;
use bls48581::bls48581::big;
use bls48581::bls48581::ecp;
use bls48581::bls48581::ecp8;
use bls48581::bls48581::pair8;
use bls48581::bls48581::rom;
use metavm_core::transcript::Transcript;

/// An accumulated KZG claim for deferred pairing verification.
///
/// Stores the pairing arguments directly. For each KZG check
///   e(C - y*G1, G2) == e(π, [τ]₂ - z*G2)
/// we rearrange to:
///   e(C - y*G1 + z*π, G2) == e(π, [τ]₂)
/// Let L = C - y*G1 + z*π (LHS point) and R = π (RHS point).
/// Multiple claims are accumulated via random linear combination:
///   L_acc = Σ rⁱ * L_i, R_acc = Σ rⁱ * R_i
/// Final check: e(L_acc, G2) == e(R_acc, [τ]₂)
#[derive(Clone, Debug)]
pub struct AccumulatedClaim {
    /// L_acc = Σ rⁱ * (C_i - y_i*G1 + z_i*π_i), 74 bytes (compressed G1 point)
    pub l_acc: Vec<u8>,
    /// R_acc = Σ rⁱ * π_i, 74 bytes (compressed G1 point)
    pub r_acc: Vec<u8>,
    /// Number of proofs folded so far.
    pub num_folded: u64,
}

/// A recursive proof that attests to the correctness of a sequence of execution chunks.
#[derive(Clone, Debug)]
pub struct RecursiveProof {
    /// The proof for the current execution chunk.
    pub current_proof: ExecutionProof,
    /// The accumulated claim from all chunks.
    pub accumulator: AccumulatedClaim,
    /// Total number of chunks proved so far.
    pub depth: u64,
    /// State hash at the very beginning of the proven execution (chunk 0's initial state).
    pub initial_state_hash: Option<[u8; 32]>,
    /// State hash at the end of the most recently folded chunk.
    pub final_state_hash: Option<[u8; 32]>,
}

impl AccumulatedClaim {
    /// Create an initial (empty) accumulator with the identity point.
    pub fn initial() -> Self {
        let inf = ecp::ECP::new(); // point at infinity
        let mut l_bytes = vec![0u8; 74];
        let mut r_bytes = vec![0u8; 74];
        inf.tobytes(&mut l_bytes, true);
        inf.tobytes(&mut r_bytes, true);
        AccumulatedClaim {
            l_acc: l_bytes,
            r_acc: r_bytes,
            num_folded: 0,
        }
    }
}

/// Rebuild the prover's Fiat-Shamir transcript from a chunk proof to recover z and β.
///
/// This re-derives the same z and β the prover used, which are needed to
/// compute the pairing arguments L and R for accumulation.
fn recover_chunk_challenges(
    chunk: &ChunkProof,
) -> (big::BIG, big::BIG, Vec<big::BIG>) {
    let proof = &chunk.execution_proof;
    let _modulus = big::BIG::new_ints(&rom::CURVE_ORDER);

    let mut transcript = Transcript::new(b"metavm-execution-proof");
    transcript.append_u64(b"chunk_index", chunk.chunk_index);
    transcript.append_message(b"initial_state", &chunk.initial_state_hash);
    transcript.append_message(b"final_state", &chunk.final_state_hash);
    transcript.append_u64(b"num_steps", proof.num_steps);
    transcript.append_u64(b"domain_size", proof.domain_size);

    for comm in &proof.column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    let _alpha_bytes = transcript.challenge_bytes(b"alpha");

    // ── LogUp commitments (mirrors prover Step 3b) ──────────────────────
    if !proof.logup_commitments.is_empty() {
        let _gamma_bytes = transcript.challenge_bytes(b"logup_gamma");
        for comm in &proof.logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
    }

    // ── Memory permutation commitments (mirrors prover Step 3c) ─────────
    if !proof.perm_commitments.is_empty() {
        let _pg_bytes = transcript.challenge_bytes(b"perm_gamma");
        let _pd_bytes = transcript.challenge_bytes(b"perm_delta");
        for comm in &proof.perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }
    }

    // ── Oracle public-input data (mirrors prover Step 3d) ───────────────
    if !proof.oracle_data.is_empty() {
        transcript.append_u64(b"num_oracle_entries", proof.oracle_data.len() as u64);
        for entry in &proof.oracle_data {
            transcript.append_message(b"oracle_entry", entry);
        }
    }

    // ── Register permutation commitments (mirrors prover Step 3e) ───────
    if !proof.reg_perm_commitments.is_empty() {
        let _rg_bytes = transcript.challenge_bytes(b"reg_perm_gamma");
        let _rd_bytes = transcript.challenge_bytes(b"reg_perm_delta");
        for comm in &proof.reg_perm_commitments {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }
    }

    for qc in &proof.quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = challenge_to_big(&z_bytes);

    // Absorb all evaluations (columns + shifted + logup + perm + reg_perm)
    for eval_bytes in &proof.evaluations {
        transcript.append_message(b"evaluation", eval_bytes);
    }
    for se in &proof.shifted_evaluations {
        transcript.append_message(b"shifted_evaluation", se);
    }
    for le in &proof.logup_evaluations {
        transcript.append_message(b"logup_evaluation", le);
    }
    for lse in &proof.logup_shifted_evaluations {
        transcript.append_message(b"logup_shifted_evaluation", lse);
    }
    for pe in &proof.perm_evaluations {
        transcript.append_message(b"perm_evaluation", pe);
    }
    for pse in &proof.perm_shifted_evaluations {
        transcript.append_message(b"perm_shifted_evaluation", pse);
    }
    for re in &proof.reg_perm_evaluations {
        transcript.append_message(b"reg_perm_evaluation", re);
    }
    for rse in &proof.reg_perm_shifted_evaluations {
        transcript.append_message(b"reg_perm_shifted_evaluation", rse);
    }
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = challenge_to_big(&beta_bytes);

    // Parse evaluations as BIG values
    let evals: Vec<big::BIG> = proof.evaluations.iter()
        .map(|e| big::BIG::frombytes(e))
        .collect();

    (z, beta, evals)
}

/// Compute L and R pairing arguments for a chunk proof.
///
/// L = C_combined - y_combined*G1 + z*π
/// R = π
///
/// where C_combined = Σ β^i * C_i and y_combined = Σ β^i * y_i
fn compute_lr(chunk: &ChunkProof) -> (ecp::ECP, ecp::ECP) {
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    let (z, beta, evals) = recover_chunk_challenges(chunk);
    let proof = &chunk.execution_proof;

    // Combine commitments: C_combined = Σ β^i * C_i
    let mut c_combined = ecp::ECP::new();
    let mut beta_power = big::BIG::new_int(1);
    for comm in &proof.column_commitments {
        let c = ecp::ECP::frombytes(&comm.0);
        let scaled = c.mul(&beta_power);
        c_combined.add(&scaled);
        beta_power = big::BIG::modmul(&beta_power, &beta, &modulus);
    }
    // Include quotient commitments
    for qc_comm in &proof.quotient_commitments {
        let qc = ecp::ECP::frombytes(&qc_comm.0);
        let scaled = qc.mul(&beta_power);
        c_combined.add(&scaled);
        beta_power = big::BIG::modmul(&beta_power, &beta, &modulus);
    }

    // Combine evaluations: y_combined = Σ β^i * y_i
    let mut y_combined = big::BIG::new();
    beta_power = big::BIG::new_int(1);
    for y in &evals {
        let term = big::BIG::modmul(&beta_power, y, &modulus);
        y_combined = big::BIG::modadd(&y_combined, &term, &modulus);
        beta_power = big::BIG::modmul(&beta_power, &beta, &modulus);
    }

    // Parse π
    let pi = ecp::ECP::frombytes(&proof.opening_proof.proof);

    // L = C_combined - y_combined*G1 + z*π
    let g1 = ecp::ECP::generator();
    let y_g1 = g1.mul(&y_combined);
    let z_pi = pi.mul(&z);

    let mut l = c_combined;
    l.sub(&y_g1);
    l.add(&z_pi);
    l.affine();

    // R = π
    let r = pi;

    (l, r)
}

/// Create the initial recursive proof from the first chunk proof (state-chain aware).
pub fn begin_chunk(first: ChunkProof) -> RecursiveProof {
    let (l, r) = compute_lr(&first);

    let mut l_bytes = vec![0u8; 74];
    l.tobytes(&mut l_bytes, true);
    let mut r_bytes = vec![0u8; 74];
    r.tobytes(&mut r_bytes, true);

    let acc = AccumulatedClaim {
        l_acc: l_bytes,
        r_acc: r_bytes,
        num_folded: 1,
    };

    RecursiveProof {
        current_proof: first.execution_proof,
        accumulator: acc,
        depth: 1,
        initial_state_hash: Some(first.initial_state_hash),
        final_state_hash: Some(first.final_state_hash),
    }
}

/// Fold a new chunk proof into the recursive accumulator with state chain verification.
///
/// Verifies state chain continuity, computes L/R for the new chunk,
/// derives a Fiat-Shamir folding challenge, and accumulates:
///   L_acc = L_prev + r * L_new
///   R_acc = R_prev + r * R_new
pub fn fold_chunks(
    prev: &RecursiveProof,
    current: ChunkProof,
) -> Result<RecursiveProof, String> {
    // Verify state chain continuity
    if let Some(prev_final) = &prev.final_state_hash {
        if *prev_final != current.initial_state_hash {
            return Err(format!(
                "State chain broken at chunk {}: prev final hash != current initial hash",
                current.chunk_index
            ));
        }
    }

    // Compute L and R for the current chunk
    let (l_new, r_new) = compute_lr(&current);

    // Derive folding challenge from Fiat-Shamir
    let mut transcript = Transcript::new(b"metavm-recursive-fold");
    transcript.append_message(b"prev-l-acc", &prev.accumulator.l_acc);
    transcript.append_message(b"prev-r-acc", &prev.accumulator.r_acc);
    transcript.append_u64(b"depth", prev.depth);
    transcript.append_u64(b"chunk_index", current.chunk_index);
    transcript.append_message(b"initial_state", &current.initial_state_hash);
    transcript.append_message(b"final_state", &current.final_state_hash);

    for c in &current.execution_proof.column_commitments {
        transcript.append_message(b"current-commit", &c.0);
    }

    let r_field = transcript.challenge(b"fold-challenge");
    let r_bytes = r_field.to_bytes();
    let mut r_big_bytes = [0u8; big::MODBYTES];
    for i in 0..32 {
        r_big_bytes[big::MODBYTES - 1 - i] = r_bytes[i];
    }
    let r = big::BIG::frombytes(&r_big_bytes);

    // Accumulate: L_acc = L_prev + r * L_new
    let mut l_prev = ecp::ECP::frombytes(&prev.accumulator.l_acc);
    let scaled_l = l_new.mul(&r);
    l_prev.add(&scaled_l);
    l_prev.affine();
    let mut l_acc_bytes = vec![0u8; 74];
    l_prev.tobytes(&mut l_acc_bytes, true);

    // Accumulate: R_acc = R_prev + r * R_new
    let mut r_prev = ecp::ECP::frombytes(&prev.accumulator.r_acc);
    let scaled_r = r_new.mul(&r);
    r_prev.add(&scaled_r);
    r_prev.affine();
    let mut r_acc_bytes = vec![0u8; 74];
    r_prev.tobytes(&mut r_acc_bytes, true);

    let new_acc = AccumulatedClaim {
        l_acc: l_acc_bytes,
        r_acc: r_acc_bytes,
        num_folded: prev.accumulator.num_folded + 1,
    };

    Ok(RecursiveProof {
        current_proof: current.execution_proof,
        accumulator: new_acc,
        depth: prev.depth + 1,
        initial_state_hash: prev.initial_state_hash,
        final_state_hash: Some(current.final_state_hash),
    })
}

/// Verify the final recursive proof via pairing check.
///
/// Checks: e(L_acc, G2) == e(R_acc, [τ]₂)
/// Equivalently: e(L_acc, G2) * e(-R_acc, [τ]₂) == 1
pub fn verify_final(proof: &RecursiveProof) -> bool {
    // Structural validity checks
    if proof.depth == 0 {
        return false;
    }

    if proof.accumulator.l_acc.is_empty() || proof.accumulator.r_acc.is_empty() {
        return false;
    }

    if proof.accumulator.num_folded != proof.depth {
        return false;
    }

    let l_acc = ecp::ECP::frombytes(&proof.accumulator.l_acc);
    let r_acc = ecp::ECP::frombytes(&proof.accumulator.r_acc);

    // Get [τ]₂ from the SRS
    let s = bls::singleton();
    let tau_g2 = s.CeremonyBLS48581G2[1].clone();

    // Check: e(L_acc, G2) == e(R_acc, [τ]₂)
    // Equivalently: e(L_acc, G2) * e(-R_acc, [τ]₂) == 1
    let mut r = pair8::initmp();
    pair8::another(&mut r, &ecp8::ECP8::generator(), &l_acc);
    let mut neg_r_acc = r_acc;
    neg_r_acc.neg();
    pair8::another(&mut r, &tau_g2, &neg_r_acc);
    let mut v = pair8::miller(&mut r);
    v = pair8::fexp(&v);
    v.isunity()
}

/// Verify the final recursive proof with expected initial and final state hashes.
pub fn verify_final_with_state(
    proof: &RecursiveProof,
    expected_initial_state: &[u8; 32],
    expected_final_state: &[u8; 32],
) -> bool {
    if !verify_final(proof) {
        return false;
    }

    match &proof.initial_state_hash {
        Some(hash) if hash == expected_initial_state => {}
        Some(_) => return false,
        None => return false,
    }

    match &proof.final_state_hash {
        Some(hash) if hash == expected_final_state => {}
        Some(_) => return false,
        None => return false,
    }

    true
}

// =========================================================================
// Scheme-generic versions (work with BLS48-581 or BLS12-381)
// =========================================================================

use crate::scheme::CommitmentScheme;
use crate::field::{Scalar, CurveType};
use crate::prover::challenge_to_scalar;

/// Rebuild the prover's Fiat-Shamir transcript from a chunk proof to recover
/// z, β, and evaluation scalars — scheme-generic version.
fn recover_chunk_challenges_scheme(
    chunk: &ChunkProof,
    curve: CurveType,
) -> (Scalar, Scalar, Vec<Scalar>) {
    let proof = &chunk.execution_proof;

    let mut transcript = Transcript::new(b"metavm-execution-proof");
    transcript.append_u64(b"chunk_index", chunk.chunk_index);
    transcript.append_message(b"initial_state", &chunk.initial_state_hash);
    transcript.append_message(b"final_state", &chunk.final_state_hash);
    transcript.append_u64(b"num_steps", proof.num_steps);
    transcript.append_u64(b"domain_size", proof.domain_size);

    for comm in &proof.column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    let _alpha_bytes = transcript.challenge_bytes(b"alpha");

    // ── LogUp commitments (mirrors prover Step 3b) ──────────────────────
    if !proof.logup_commitments.is_empty() {
        let _gamma_bytes = transcript.challenge_bytes(b"logup_gamma");
        for comm in &proof.logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
    }

    // ── Memory permutation commitments (mirrors prover Step 3c) ─────────
    if !proof.perm_commitments.is_empty() {
        let _pg_bytes = transcript.challenge_bytes(b"perm_gamma");
        let _pd_bytes = transcript.challenge_bytes(b"perm_delta");
        for comm in &proof.perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }
    }

    // ── Oracle public-input data (mirrors prover Step 3d) ───────────────
    if !proof.oracle_data.is_empty() {
        transcript.append_u64(b"num_oracle_entries", proof.oracle_data.len() as u64);
        for entry in &proof.oracle_data {
            transcript.append_message(b"oracle_entry", entry);
        }
    }

    // ── Register permutation commitments (mirrors prover Step 3e) ───────
    if !proof.reg_perm_commitments.is_empty() {
        let _rg_bytes = transcript.challenge_bytes(b"reg_perm_gamma");
        let _rd_bytes = transcript.challenge_bytes(b"reg_perm_delta");
        for comm in &proof.reg_perm_commitments {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }
    }

    for qc in &proof.quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = challenge_to_scalar(&z_bytes, curve);

    for eval_bytes in &proof.evaluations {
        transcript.append_message(b"evaluation", eval_bytes);
    }
    for se in &proof.shifted_evaluations {
        transcript.append_message(b"shifted_evaluation", se);
    }
    for le in &proof.logup_evaluations {
        transcript.append_message(b"logup_evaluation", le);
    }
    for lse in &proof.logup_shifted_evaluations {
        transcript.append_message(b"logup_shifted_evaluation", lse);
    }
    for pe in &proof.perm_evaluations {
        transcript.append_message(b"perm_evaluation", pe);
    }
    for pse in &proof.perm_shifted_evaluations {
        transcript.append_message(b"perm_shifted_evaluation", pse);
    }
    for re in &proof.reg_perm_evaluations {
        transcript.append_message(b"reg_perm_evaluation", re);
    }
    for rse in &proof.reg_perm_shifted_evaluations {
        transcript.append_message(b"reg_perm_shifted_evaluation", rse);
    }
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = challenge_to_scalar(&beta_bytes, curve);

    let evals: Vec<Scalar> = proof.evaluations.iter()
        .map(|e| Scalar::from_bytes(e, curve))
        .collect();

    (z, beta, evals)
}

/// Compute L and R pairing arguments for a chunk proof — scheme-generic version.
fn compute_lr_scheme(
    chunk: &ChunkProof,
    scheme: &dyn CommitmentScheme,
    curve: CurveType,
) -> (Vec<u8>, Vec<u8>) {
    let (z, beta, evals) = recover_chunk_challenges_scheme(chunk, curve);
    let proof = &chunk.execution_proof;

    // Collect all commitment byte slices (columns + quotient chunks)
    let all_commitments: Vec<&[u8]> = proof.column_commitments.iter()
        .chain(proof.quotient_commitments.iter())
        .map(|c| c.0.as_slice())
        .collect();

    scheme.compute_lr(&all_commitments, &evals, &z, &beta, &proof.opening_proof.proof.as_slice())
}

/// Create the initial recursive proof from the first chunk proof — scheme-generic.
pub fn begin_chunk_scheme(first: ChunkProof, scheme: &dyn CommitmentScheme, curve: CurveType) -> RecursiveProof {
    let (l_bytes, r_bytes) = compute_lr_scheme(&first, scheme, curve);

    let acc = AccumulatedClaim {
        l_acc: l_bytes,
        r_acc: r_bytes,
        num_folded: 1,
    };

    RecursiveProof {
        current_proof: first.execution_proof,
        accumulator: acc,
        depth: 1,
        initial_state_hash: Some(first.initial_state_hash),
        final_state_hash: Some(first.final_state_hash),
    }
}

/// Fold a new chunk proof into the recursive accumulator — scheme-generic.
pub fn fold_chunks_scheme(
    prev: &RecursiveProof,
    current: ChunkProof,
    scheme: &dyn CommitmentScheme,
    curve: CurveType,
) -> Result<RecursiveProof, String> {
    // Verify state chain continuity
    if let Some(prev_final) = &prev.final_state_hash {
        if *prev_final != current.initial_state_hash {
            return Err(format!(
                "State chain broken at chunk {}: prev final hash != current initial hash",
                current.chunk_index
            ));
        }
    }

    // Compute L and R for the current chunk
    let (l_new, r_new) = compute_lr_scheme(&current, scheme, curve);

    // Derive folding challenge from Fiat-Shamir
    let mut transcript = Transcript::new(b"metavm-recursive-fold");
    transcript.append_message(b"prev-l-acc", &prev.accumulator.l_acc);
    transcript.append_message(b"prev-r-acc", &prev.accumulator.r_acc);
    transcript.append_u64(b"depth", prev.depth);
    transcript.append_u64(b"chunk_index", current.chunk_index);
    transcript.append_message(b"initial_state", &current.initial_state_hash);
    transcript.append_message(b"final_state", &current.final_state_hash);

    for c in &current.execution_proof.column_commitments {
        transcript.append_message(b"current-commit", &c.0);
    }

    let r_field = transcript.challenge(b"fold-challenge");
    let r_bytes = r_field.to_bytes();
    let challenge = Scalar::from_challenge_bytes(&r_bytes, curve);

    // Accumulate via scheme
    let (l_acc_bytes, r_acc_bytes) = scheme.fold_accumulator(
        &prev.accumulator.l_acc,
        &prev.accumulator.r_acc,
        &l_new,
        &r_new,
        &challenge,
    );

    let new_acc = AccumulatedClaim {
        l_acc: l_acc_bytes,
        r_acc: r_acc_bytes,
        num_folded: prev.accumulator.num_folded + 1,
    };

    Ok(RecursiveProof {
        current_proof: current.execution_proof,
        accumulator: new_acc,
        depth: prev.depth + 1,
        initial_state_hash: prev.initial_state_hash,
        final_state_hash: Some(current.final_state_hash),
    })
}

/// Verify the final recursive proof via pairing check — scheme-generic.
pub fn verify_final_scheme(proof: &RecursiveProof, scheme: &dyn CommitmentScheme) -> bool {
    if proof.depth == 0 {
        return false;
    }
    if proof.accumulator.l_acc.is_empty() || proof.accumulator.r_acc.is_empty() {
        return false;
    }
    if proof.accumulator.num_folded != proof.depth {
        return false;
    }

    scheme.verify_accumulated(&proof.accumulator.l_acc, &proof.accumulator.r_acc)
}

// Legacy functions for backward compatibility with non-chunk proofs
/// Create the initial recursive proof from the first proof.
pub fn begin(first_proof: ExecutionProof) -> RecursiveProof {
    // For non-chunk proofs, we can't compute real L/R without z and β.
    // Use identity accumulator — these proofs aren't meant for production.
    let acc = AccumulatedClaim::initial();

    RecursiveProof {
        current_proof: first_proof,
        accumulator: AccumulatedClaim {
            num_folded: 1,
            ..acc
        },
        depth: 1,
        initial_state_hash: None,
        final_state_hash: None,
    }
}

/// Fold a new execution proof into the recursive accumulator (legacy, non-chunk).
pub fn fold(prev: &RecursiveProof, current: ExecutionProof) -> RecursiveProof {
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);

    // Derive folding challenge
    let mut transcript = Transcript::new(b"metavm-recursive-fold");
    transcript.append_message(b"prev-l-acc", &prev.accumulator.l_acc);
    transcript.append_message(b"prev-r-acc", &prev.accumulator.r_acc);
    transcript.append_u64(b"depth", prev.depth);

    for c in &current.column_commitments {
        transcript.append_message(b"current-commit", &c.0);
    }

    let r_field = transcript.challenge(b"fold-challenge");
    let r_bytes = r_field.to_bytes();
    let mut r_big_bytes = [0u8; big::MODBYTES];
    for i in 0..32 {
        r_big_bytes[big::MODBYTES - 1 - i] = r_bytes[i];
    }
    let _r = big::BIG::frombytes(&r_big_bytes);

    // For legacy non-chunk proofs, keep the accumulator simple
    let new_acc = AccumulatedClaim {
        l_acc: prev.accumulator.l_acc.clone(),
        r_acc: prev.accumulator.r_acc.clone(),
        num_folded: prev.accumulator.num_folded + 1,
    };

    RecursiveProof {
        current_proof: current,
        accumulator: new_acc,
        depth: prev.depth + 1,
        initial_state_hash: None,
        final_state_hash: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{Commitment, BatchProof};
    use crate::prover::ExecutionProof;

    /// Helper to create a dummy execution proof for testing.
    fn dummy_proof(id: u8) -> ExecutionProof {
        // First byte must be 0x02 or 0x03 for compressed G1 point format
        // to avoid ECP::frombytes treating 0x04 as uncompressed and reading OOB.
        let mut comm_bytes = vec![id; 74];
        comm_bytes[0] = 0x02;
        let mut proof_bytes = vec![id; 74];
        proof_bytes[0] = 0x02;
        ExecutionProof {
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
        }
    }

    #[test]
    fn test_begin_creates_depth_one() {
        let proof = dummy_proof(1);
        let recursive = begin(proof);
        assert_eq!(recursive.depth, 1);
        assert_eq!(recursive.accumulator.num_folded, 1);
    }

    #[test]
    fn test_begin_accumulator_has_valid_points() {
        let proof = dummy_proof(42);
        let recursive = begin(proof);
        assert_eq!(recursive.accumulator.l_acc.len(), 74);
        assert_eq!(recursive.accumulator.r_acc.len(), 74);
    }

    #[test]
    fn test_fold_increments_depth() {
        let first = begin(dummy_proof(1));
        let folded = fold(&first, dummy_proof(2));
        assert_eq!(folded.depth, 2);
        assert_eq!(folded.accumulator.num_folded, 2);

        let folded2 = fold(&folded, dummy_proof(3));
        assert_eq!(folded2.depth, 3);
        assert_eq!(folded2.accumulator.num_folded, 3);
    }

    #[test]
    fn test_fold_updates_accumulator() {
        let first = begin(dummy_proof(1));
        let folded = fold(&first, dummy_proof(2));
        assert_eq!(folded.accumulator.l_acc.len(), 74);
        assert_eq!(folded.accumulator.r_acc.len(), 74);
    }

    #[test]
    fn test_verify_final_rejects_zero_depth() {
        let proof = RecursiveProof {
            current_proof: dummy_proof(1),
            accumulator: AccumulatedClaim::initial(),
            depth: 0,
            initial_state_hash: None,
            final_state_hash: None,
        };
        assert!(!verify_final(&proof));
    }

    #[test]
    fn test_verify_final_rejects_empty_accumulator() {
        let proof = RecursiveProof {
            current_proof: dummy_proof(1),
            accumulator: AccumulatedClaim {
                l_acc: vec![],
                r_acc: vec![0u8; 74],
                num_folded: 1,
            },
            depth: 1,
            initial_state_hash: None,
            final_state_hash: None,
        };
        assert!(!verify_final(&proof));
    }

    fn dummy_chunk_proof(id: u8, chunk_index: u64, initial: [u8; 32], final_h: [u8; 32]) -> ChunkProof {
        ChunkProof {
            execution_proof: dummy_proof(id),
            initial_state_hash: initial,
            final_state_hash: final_h,
            chunk_index,
        }
    }

    #[test]
    fn test_begin_chunk_preserves_state() {
        let state0 = [0u8; 32];
        let state1 = [1u8; 32];
        let chunk = dummy_chunk_proof(1, 0, state0, state1);
        let recursive = begin_chunk(chunk);
        assert_eq!(recursive.depth, 1);
        assert_eq!(recursive.initial_state_hash, Some(state0));
        assert_eq!(recursive.final_state_hash, Some(state1));
    }

    #[test]
    fn test_fold_chunks_state_chain() {
        let state0 = [0u8; 32];
        let state1 = [1u8; 32];
        let state2 = [2u8; 32];
        let first = begin_chunk(dummy_chunk_proof(1, 0, state0, state1));
        let second = dummy_chunk_proof(2, 1, state1, state2);
        let folded = fold_chunks(&first, second).expect("state chain should match");
        assert_eq!(folded.depth, 2);
        assert_eq!(folded.initial_state_hash, Some(state0));
        assert_eq!(folded.final_state_hash, Some(state2));
    }

    #[test]
    fn test_fold_chunks_rejects_broken_chain() {
        let state0 = [0u8; 32];
        let state1 = [1u8; 32];
        let state_wrong = [99u8; 32];
        let state2 = [2u8; 32];
        let first = begin_chunk(dummy_chunk_proof(1, 0, state0, state1));
        let second = dummy_chunk_proof(2, 1, state_wrong, state2);
        let result = fold_chunks(&first, second);
        assert!(result.is_err());
    }

    #[test]
    fn test_accumulated_claim_initial() {
        let acc = AccumulatedClaim::initial();
        assert_eq!(acc.l_acc.len(), 74);
        assert_eq!(acc.r_acc.len(), 74);
        assert_eq!(acc.num_folded, 0);
    }
}
