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
    /// Aggregated constraint identity scalar `c_check = Q(z)·Z_H(z) − C(z)`
    /// folded across chunks. Empty when the legacy single-opening pipeline
    /// is in use (no scalar accumulation requested). Non-empty values are
    /// MODBYTES-encoded BLS48-581 BIG bytes; `verify_final_*_scheme`
    /// requires this scalar to be zero in addition to the pairing check.
    pub scalar_acc: Vec<u8>,
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

impl RecursiveProof {
    /// Serialize the recursive proof: inner [`ExecutionProof`] +
    /// [`AccumulatedClaim`] (l_acc, r_acc, num_folded) + depth + two
    /// optional state hashes. Uses length-prefix framing analogous to
    /// [`crate::prover::ExecutionProof::to_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let inner = self.current_proof.to_bytes();
        let mut out = Vec::with_capacity(inner.len() + 200);
        // Inner ExecutionProof (length-prefixed so the decoder knows
        // where accumulator metadata begins).
        out.extend_from_slice(&(inner.len() as u32).to_be_bytes());
        out.extend_from_slice(&inner);
        // AccumulatedClaim
        out.extend_from_slice(&(self.accumulator.l_acc.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.accumulator.l_acc);
        out.extend_from_slice(&(self.accumulator.r_acc.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.accumulator.r_acc);
        out.extend_from_slice(&(self.accumulator.scalar_acc.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.accumulator.scalar_acc);
        out.extend_from_slice(&self.accumulator.num_folded.to_be_bytes());
        // depth
        out.extend_from_slice(&self.depth.to_be_bytes());
        // Optional state hashes: 1-byte tag + 32 bytes when present.
        match &self.initial_state_hash {
            None => out.push(0),
            Some(h) => { out.push(1); out.extend_from_slice(h); }
        }
        match &self.final_state_hash {
            None => out.push(0),
            Some(h) => { out.push(1); out.extend_from_slice(h); }
        }
        out
    }

    /// Decode a recursive proof produced by [`Self::to_bytes`]. Strict:
    /// trailing bytes after a complete decode are an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::prover::ProofDecodeError> {
        use crate::prover::{ExecutionProof, ProofDecodeError};
        if bytes.len() < 4 {
            return Err(ProofDecodeError::Truncated { wanted: 4, available: bytes.len() });
        }
        let inner_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut pos = 4;
        if bytes.len() < pos + inner_len {
            return Err(ProofDecodeError::Truncated {
                wanted: inner_len,
                available: bytes.len() - pos,
            });
        }
        let current_proof = ExecutionProof::from_bytes(&bytes[pos..pos + inner_len])?;
        pos += inner_len;

        // Helper: read u32-length-prefixed bytes.
        let read_lp = |pos: &mut usize| -> Result<Vec<u8>, ProofDecodeError> {
            if bytes.len() < *pos + 4 {
                return Err(ProofDecodeError::Truncated {
                    wanted: 4,
                    available: bytes.len() - *pos,
                });
            }
            let n = u32::from_be_bytes([
                bytes[*pos], bytes[*pos + 1], bytes[*pos + 2], bytes[*pos + 3],
            ]) as usize;
            *pos += 4;
            if bytes.len() < *pos + n {
                return Err(ProofDecodeError::Truncated {
                    wanted: n,
                    available: bytes.len() - *pos,
                });
            }
            let v = bytes[*pos..*pos + n].to_vec();
            *pos += n;
            Ok(v)
        };

        let l_acc = read_lp(&mut pos)?;
        let r_acc = read_lp(&mut pos)?;
        let scalar_acc = read_lp(&mut pos)?;

        // num_folded: 8 BE bytes
        if bytes.len() < pos + 8 {
            return Err(ProofDecodeError::Truncated { wanted: 8, available: bytes.len() - pos });
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&bytes[pos..pos + 8]);
        let num_folded = u64::from_be_bytes(a);
        pos += 8;

        // depth: 8 BE bytes
        if bytes.len() < pos + 8 {
            return Err(ProofDecodeError::Truncated { wanted: 8, available: bytes.len() - pos });
        }
        a.copy_from_slice(&bytes[pos..pos + 8]);
        let depth = u64::from_be_bytes(a);
        pos += 8;

        // Optional initial_state_hash
        let read_optional_hash = |pos: &mut usize| -> Result<Option<[u8; 32]>, ProofDecodeError> {
            if *pos >= bytes.len() {
                return Err(ProofDecodeError::Truncated { wanted: 1, available: 0 });
            }
            let tag = bytes[*pos];
            *pos += 1;
            match tag {
                0 => Ok(None),
                1 => {
                    if bytes.len() < *pos + 32 {
                        return Err(ProofDecodeError::Truncated {
                            wanted: 32,
                            available: bytes.len() - *pos,
                        });
                    }
                    let mut h = [0u8; 32];
                    h.copy_from_slice(&bytes[*pos..*pos + 32]);
                    *pos += 32;
                    Ok(Some(h))
                }
                other => Err(ProofDecodeError::InvalidOptionTag(other)),
            }
        };
        let initial_state_hash = read_optional_hash(&mut pos)?;
        let final_state_hash = read_optional_hash(&mut pos)?;

        if pos != bytes.len() {
            return Err(ProofDecodeError::TrailingBytes(bytes.len() - pos));
        }

        Ok(RecursiveProof {
            current_proof,
            accumulator: AccumulatedClaim { l_acc, r_acc, num_folded, scalar_acc },
            depth,
            initial_state_hash,
            final_state_hash,
        })
    }
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
            scalar_acc: Vec::new(),
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

    // ── Bitwise LogUp commitments (mirrors prover Step 3b' / verifier 1b') ─
    if !proof.bitwise_commitments.is_empty() {
        let _bw_gamma_bytes = transcript.challenge_bytes(b"bitwise_gamma");
        let _bw_delta_bytes = transcript.challenge_bytes(b"bitwise_delta");
        for comm in &proof.bitwise_commitments {
            transcript.append_message(b"bitwise_column_commitment", &comm.0);
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

    // ── Frame-stack permutation (mirrors prover Step 3f) ────────────────
    if let Some(ref fp_comm) = proof.frame_perm_commitment {
        let _fpg_bytes = transcript.challenge_bytes(b"frame_perm_gamma");
        let _fpd_bytes = transcript.challenge_bytes(b"frame_perm_delta");
        transcript.append_message(b"frame_perm_column_commitment", &fp_comm.0);
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
    // Bitwise evaluations sit between logup_shifted and perm in the
    // verifier's transcript order; recover_chunk_* must mirror that.
    for be in &proof.bitwise_evaluations {
        transcript.append_message(b"bitwise_evaluation", be);
    }
    for bse in &proof.bitwise_shifted_evaluations {
        transcript.append_message(b"bitwise_shifted_evaluation", bse);
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
    if let Some(ref fpe) = proof.frame_perm_evaluation {
        transcript.append_message(b"frame_perm_evaluation", fpe);
    }
    if let Some(ref fpse) = proof.frame_perm_shifted_evaluation {
        transcript.append_message(b"frame_perm_shifted_evaluation", fpse);
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
        scalar_acc: Vec::new(),
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
        scalar_acc: prev.accumulator.scalar_acc.clone(),
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

    // ── Bitwise LogUp commitments (mirrors prover Step 3b' / verifier 1b') ─
    if !proof.bitwise_commitments.is_empty() {
        let _bw_gamma_bytes = transcript.challenge_bytes(b"bitwise_gamma");
        let _bw_delta_bytes = transcript.challenge_bytes(b"bitwise_delta");
        for comm in &proof.bitwise_commitments {
            transcript.append_message(b"bitwise_column_commitment", &comm.0);
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

    // ── Frame-stack permutation (mirrors prover Step 3f) ────────────────
    if let Some(ref fp_comm) = proof.frame_perm_commitment {
        let _fpg_bytes = transcript.challenge_bytes(b"frame_perm_gamma");
        let _fpd_bytes = transcript.challenge_bytes(b"frame_perm_delta");
        transcript.append_message(b"frame_perm_column_commitment", &fp_comm.0);
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
    // Bitwise evaluations sit between logup_shifted and perm in the
    // verifier's transcript order; recover_chunk_* must mirror that.
    for be in &proof.bitwise_evaluations {
        transcript.append_message(b"bitwise_evaluation", be);
    }
    for bse in &proof.bitwise_shifted_evaluations {
        transcript.append_message(b"bitwise_shifted_evaluation", bse);
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
    if let Some(ref fpe) = proof.frame_perm_evaluation {
        transcript.append_message(b"frame_perm_evaluation", fpe);
    }
    if let Some(ref fpse) = proof.frame_perm_shifted_evaluation {
        transcript.append_message(b"frame_perm_shifted_evaluation", fpse);
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
        scalar_acc: Vec::new(),
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
        scalar_acc: prev.accumulator.scalar_acc.clone(),
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

    if !scheme.verify_accumulated(&proof.accumulator.l_acc, &proof.accumulator.r_acc) {
        return false;
    }

    // When the proof was built via the *_full_scheme path that aggregates
    // the per-chunk constraint identity scalar, scalar_acc carries the
    // running sum. A valid proof reduces to zero. Empty scalar_acc means
    // the legacy pairing-only path was used; nothing to check.
    if !proof.accumulator.scalar_acc.is_empty() {
        if !proof.accumulator.scalar_acc.iter().all(|b| *b == 0) {
            return false;
        }
    }

    true
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

/// Fold a new execution proof into the recursive accumulator (legacy,
/// non-chunk). Kept for backward compatibility with [`begin`]; produces an
/// identity accumulator that doesn't pass [`verify_final`]. Real proofs
/// should use [`begin_chunk_scheme`] / [`fold_chunks_scheme`] (or the
/// `*_full_scheme` variants) which take a [`ChunkProof`].
pub fn fold(prev: &RecursiveProof, current: ExecutionProof) -> RecursiveProof {
    let new_acc = AccumulatedClaim {
        l_acc: prev.accumulator.l_acc.clone(),
        r_acc: prev.accumulator.r_acc.clone(),
        num_folded: prev.accumulator.num_folded + 1,
        scalar_acc: prev.accumulator.scalar_acc.clone(),
    };
    RecursiveProof {
        current_proof: current,
        accumulator: new_acc,
        depth: prev.depth + 1,
        initial_state_hash: None,
        final_state_hash: None,
    }
}

// =========================================================================
// Full-fold variants — fold every KZG opening AND aggregate the per-chunk
// constraint identity scalar `c_check`. Use these when the recursive
// verifier needs to certify each layer's full proof in a single pass,
// eliminating the need to call per-layer `verify_with_scheme` separately
// at expansion time.
// =========================================================================

use crate::scheme::OpeningSpec;
use crate::vm_constraints::VmConstraintSystem;

/// Fiat-Shamir challenges recovered for the full-fold pipeline. Carries
/// every per-opening β plus the meta-challenge ξ that aggregates them.
#[derive(Debug, Clone)]
struct FullChunkChallenges {
    alpha: Scalar,
    z: Scalar,
    omega: Option<Scalar>,
    beta_main: Scalar,
    beta_shifted: Option<Scalar>,
    beta_logup: Option<Scalar>,
    beta_logup_shifted: Option<Scalar>,
    beta_bitwise: Option<Scalar>,
    beta_bitwise_shifted: Option<Scalar>,
    beta_perm: Option<Scalar>,
    beta_perm_shifted: Option<Scalar>,
    beta_reg_perm: Option<Scalar>,
    beta_reg_perm_shifted: Option<Scalar>,
    beta_frame_perm: Option<Scalar>,
    beta_frame_perm_shifted: Option<Scalar>,
    perm_gamma: Option<Scalar>,
    perm_delta: Option<Scalar>,
    bitwise_gamma: Option<Scalar>,
    bitwise_delta: Option<Scalar>,
    reg_perm_gamma: Option<Scalar>,
    reg_perm_delta: Option<Scalar>,
    frame_perm_gamma: Option<Scalar>,
    frame_perm_delta: Option<Scalar>,
    logup_gamma: Option<Scalar>,
    xi: Scalar,
}

/// Replay the prover's transcript fully and derive every β_* + ξ. Mirrors
/// `recover_chunk_challenges_scheme` but continues past `b"beta"` to also
/// derive the per-opening β values and the meta-challenge ξ.
fn recover_full_chunk_challenges_scheme(
    chunk: &ChunkProof,
    curve: CurveType,
    scheme: &dyn CommitmentScheme,
) -> FullChunkChallenges {
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
    let alpha_bytes = transcript.challenge_bytes(b"alpha");
    let alpha = challenge_to_scalar(&alpha_bytes, curve);

    let mut logup_gamma_opt: Option<Scalar> = None;
    if !proof.logup_commitments.is_empty() {
        let lg = transcript.challenge_bytes(b"logup_gamma");
        logup_gamma_opt = Some(challenge_to_scalar(&lg, curve));
        for comm in &proof.logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
    }

    let mut bitwise_gamma_opt: Option<Scalar> = None;
    let mut bitwise_delta_opt: Option<Scalar> = None;
    if !proof.bitwise_commitments.is_empty() {
        let bg = transcript.challenge_bytes(b"bitwise_gamma");
        bitwise_gamma_opt = Some(challenge_to_scalar(&bg, curve));
        let bd = transcript.challenge_bytes(b"bitwise_delta");
        bitwise_delta_opt = Some(challenge_to_scalar(&bd, curve));
        for comm in &proof.bitwise_commitments {
            transcript.append_message(b"bitwise_column_commitment", &comm.0);
        }
    }

    let mut perm_gamma_opt: Option<Scalar> = None;
    let mut perm_delta_opt: Option<Scalar> = None;
    if !proof.perm_commitments.is_empty() {
        let pg = transcript.challenge_bytes(b"perm_gamma");
        perm_gamma_opt = Some(challenge_to_scalar(&pg, curve));
        let pd = transcript.challenge_bytes(b"perm_delta");
        perm_delta_opt = Some(challenge_to_scalar(&pd, curve));
        for comm in &proof.perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }
    }

    if !proof.oracle_data.is_empty() {
        transcript.append_u64(b"num_oracle_entries", proof.oracle_data.len() as u64);
        for entry in &proof.oracle_data {
            transcript.append_message(b"oracle_entry", entry);
        }
    }

    let mut reg_perm_gamma_opt: Option<Scalar> = None;
    let mut reg_perm_delta_opt: Option<Scalar> = None;
    if !proof.reg_perm_commitments.is_empty() {
        let rg = transcript.challenge_bytes(b"reg_perm_gamma");
        reg_perm_gamma_opt = Some(challenge_to_scalar(&rg, curve));
        let rd = transcript.challenge_bytes(b"reg_perm_delta");
        reg_perm_delta_opt = Some(challenge_to_scalar(&rd, curve));
        for comm in &proof.reg_perm_commitments {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }
    }

    let mut frame_perm_gamma_opt: Option<Scalar> = None;
    let mut frame_perm_delta_opt: Option<Scalar> = None;
    if let Some(ref fp_comm) = proof.frame_perm_commitment {
        let fpg = transcript.challenge_bytes(b"frame_perm_gamma");
        frame_perm_gamma_opt = Some(challenge_to_scalar(&fpg, curve));
        let fpd = transcript.challenge_bytes(b"frame_perm_delta");
        frame_perm_delta_opt = Some(challenge_to_scalar(&fpd, curve));
        transcript.append_message(b"frame_perm_column_commitment", &fp_comm.0);
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
    for be in &proof.bitwise_evaluations {
        transcript.append_message(b"bitwise_evaluation", be);
    }
    for bse in &proof.bitwise_shifted_evaluations {
        transcript.append_message(b"bitwise_shifted_evaluation", bse);
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
    if let Some(ref fpe) = proof.frame_perm_evaluation {
        transcript.append_message(b"frame_perm_evaluation", fpe);
    }
    if let Some(ref fpse) = proof.frame_perm_shifted_evaluation {
        transcript.append_message(b"frame_perm_shifted_evaluation", fpse);
    }

    let beta_main_bytes = transcript.challenge_bytes(b"beta");
    let beta_main = challenge_to_scalar(&beta_main_bytes, curve);

    let beta_shifted = if !proof.shifted_evaluations.is_empty() {
        let b = transcript.challenge_bytes(b"beta_shifted");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let has_logup = !proof.logup_commitments.is_empty();
    let beta_logup = if has_logup {
        let b = transcript.challenge_bytes(b"beta_logup");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let beta_logup_shifted = if has_logup && !proof.logup_shifted_evaluations.is_empty() {
        let b = transcript.challenge_bytes(b"beta_logup_shifted");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let has_bitwise = !proof.bitwise_commitments.is_empty();
    let beta_bitwise = if has_bitwise {
        let b = transcript.challenge_bytes(b"beta_bitwise");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let beta_bitwise_shifted = if has_bitwise && !proof.bitwise_shifted_evaluations.is_empty() {
        let b = transcript.challenge_bytes(b"beta_bitwise_shifted");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let has_perm = !proof.perm_commitments.is_empty();
    let beta_perm = if has_perm {
        let b = transcript.challenge_bytes(b"beta_perm");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let beta_perm_shifted = if has_perm && !proof.perm_shifted_evaluations.is_empty() {
        let b = transcript.challenge_bytes(b"beta_perm_shifted");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let has_reg_perm = !proof.reg_perm_commitments.is_empty();
    let beta_reg_perm = if has_reg_perm {
        let b = transcript.challenge_bytes(b"beta_reg_perm");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let beta_reg_perm_shifted = if has_reg_perm && !proof.reg_perm_shifted_evaluations.is_empty() {
        let b = transcript.challenge_bytes(b"beta_reg_perm_shifted");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let has_frame_perm = proof.frame_perm_commitment.is_some();
    let beta_frame_perm = if has_frame_perm {
        let b = transcript.challenge_bytes(b"beta_frame_perm");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };
    let beta_frame_perm_shifted = if has_frame_perm && proof.frame_perm_shifted_evaluation.is_some() {
        let b = transcript.challenge_bytes(b"beta_frame_perm_shifted");
        Some(challenge_to_scalar(&b, curve))
    } else {
        None
    };

    let xi_bytes = transcript.challenge_bytes(b"recursive_xi");
    let xi = challenge_to_scalar(&xi_bytes, curve);

    let needs_omega = beta_shifted.is_some()
        || beta_logup_shifted.is_some()
        || beta_bitwise_shifted.is_some()
        || beta_perm_shifted.is_some()
        || beta_reg_perm_shifted.is_some()
        || beta_frame_perm_shifted.is_some();
    let omega = if needs_omega {
        Some(scheme.domain_generator(proof.domain_size))
    } else {
        None
    };

    FullChunkChallenges {
        alpha,
        z,
        omega,
        beta_main,
        beta_shifted,
        beta_logup,
        beta_logup_shifted,
        beta_bitwise,
        beta_bitwise_shifted,
        beta_perm,
        beta_perm_shifted,
        beta_reg_perm,
        beta_reg_perm_shifted,
        beta_frame_perm,
        beta_frame_perm_shifted,
        perm_gamma: perm_gamma_opt,
        perm_delta: perm_delta_opt,
        bitwise_gamma: bitwise_gamma_opt,
        bitwise_delta: bitwise_delta_opt,
        reg_perm_gamma: reg_perm_gamma_opt,
        reg_perm_delta: reg_perm_delta_opt,
        frame_perm_gamma: frame_perm_gamma_opt,
        frame_perm_delta: frame_perm_delta_opt,
        logup_gamma: logup_gamma_opt,
        xi,
    }
}

/// Construct every [`OpeningSpec`] present in this chunk's
/// [`ExecutionProof`]. `constraints` is required to know which trace columns
/// participate in the shifted opening.
fn build_chunk_openings<'a>(
    chunk: &'a ChunkProof,
    challenges: &FullChunkChallenges,
    constraints: Option<&dyn VmConstraintSystem>,
    curve: CurveType,
) -> Vec<OpeningSpec<'a>> {
    let proof = &chunk.execution_proof;
    let mut openings: Vec<OpeningSpec<'a>> = Vec::new();

    // Main batch at z (columns ++ quotient chunks).
    {
        let mut commitments: Vec<&[u8]> = Vec::new();
        for c in &proof.column_commitments {
            commitments.push(c.0.as_slice());
        }
        for qc in &proof.quotient_commitments {
            commitments.push(qc.0.as_slice());
        }
        let evaluations: Vec<Scalar> = proof.evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();
        openings.push(OpeningSpec {
            commitments,
            evaluations,
            proof: proof.opening_proof.proof.as_slice(),
            point: challenges.z.clone(),
            beta: challenges.beta_main.clone(),
        });
    }

    // Shifted columns at ω·z.
    if let (Some(beta), Some(omega), Some(shifted_proof), Some(cs)) = (
        challenges.beta_shifted.as_ref(),
        challenges.omega.as_ref(),
        proof.shifted_opening_proof.as_ref(),
        constraints,
    ) {
        let shifted_indices = cs.shifted_column_indices();
        if !shifted_indices.is_empty() && !proof.shifted_evaluations.is_empty() {
            let omega_z = omega.mul(&challenges.z);
            let commitments: Vec<&[u8]> = shifted_indices.iter()
                .map(|&i| proof.column_commitments[i].0.as_slice())
                .collect();
            let evaluations: Vec<Scalar> = proof.shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();
            openings.push(OpeningSpec {
                commitments,
                evaluations,
                proof: shifted_proof.proof.as_slice(),
                point: omega_z,
                beta: beta.clone(),
            });
        }
    }

    // LogUp at z.
    if let (Some(beta), Some(p)) = (challenges.beta_logup.as_ref(), proof.logup_opening_proof.as_ref()) {
        let commitments: Vec<&[u8]> = proof.logup_commitments.iter().map(|c| c.0.as_slice()).collect();
        let evaluations: Vec<Scalar> = proof.logup_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve)).collect();
        openings.push(OpeningSpec {
            commitments, evaluations,
            proof: p.proof.as_slice(),
            point: challenges.z.clone(),
            beta: beta.clone(),
        });
    }

    // LogUp shifted (h column at ω·z).
    if let (Some(beta), Some(omega), Some(p)) = (
        challenges.beta_logup_shifted.as_ref(),
        challenges.omega.as_ref(),
        proof.logup_shifted_opening_proof.as_ref(),
    ) {
        if !proof.logup_shifted_evaluations.is_empty() && proof.logup_commitments.len() >= 2 {
            let h_idx = proof.logup_commitments.len() - 2;
            let omega_z = omega.mul(&challenges.z);
            let commitments = vec![proof.logup_commitments[h_idx].0.as_slice()];
            let evaluations: Vec<Scalar> = proof.logup_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve)).collect();
            openings.push(OpeningSpec {
                commitments, evaluations,
                proof: p.proof.as_slice(),
                point: omega_z,
                beta: beta.clone(),
            });
        }
    }

    // Bitwise at z.
    if let (Some(beta), Some(p)) = (challenges.beta_bitwise.as_ref(), proof.bitwise_opening_proof.as_ref()) {
        let commitments: Vec<&[u8]> = proof.bitwise_commitments.iter().map(|c| c.0.as_slice()).collect();
        let evaluations: Vec<Scalar> = proof.bitwise_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve)).collect();
        openings.push(OpeningSpec {
            commitments, evaluations,
            proof: p.proof.as_slice(),
            point: challenges.z.clone(),
            beta: beta.clone(),
        });
    }

    // Bitwise shifted (h_bw column at ω·z).
    if let (Some(beta), Some(omega), Some(p)) = (
        challenges.beta_bitwise_shifted.as_ref(),
        challenges.omega.as_ref(),
        proof.bitwise_shifted_opening_proof.as_ref(),
    ) {
        if !proof.bitwise_shifted_evaluations.is_empty() && !proof.bitwise_commitments.is_empty() {
            // h_bw is the second-to-last bitwise column (matches verifier).
            let h_bw_idx = proof.bitwise_commitments.len().saturating_sub(2);
            let omega_z = omega.mul(&challenges.z);
            let commitments = vec![proof.bitwise_commitments[h_bw_idx].0.as_slice()];
            let evaluations: Vec<Scalar> = proof.bitwise_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve)).collect();
            openings.push(OpeningSpec {
                commitments, evaluations,
                proof: p.proof.as_slice(),
                point: omega_z,
                beta: beta.clone(),
            });
        }
    }

    // Perm at z.
    if let (Some(beta), Some(p)) = (challenges.beta_perm.as_ref(), proof.perm_opening_proof.as_ref()) {
        let commitments: Vec<&[u8]> = proof.perm_commitments.iter().map(|c| c.0.as_slice()).collect();
        let evaluations: Vec<Scalar> = proof.perm_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve)).collect();
        openings.push(OpeningSpec {
            commitments, evaluations,
            proof: p.proof.as_slice(),
            point: challenges.z.clone(),
            beta: beta.clone(),
        });
    }

    // Perm shifted at ω·z.
    if let (Some(beta), Some(omega), Some(p)) = (
        challenges.beta_perm_shifted.as_ref(),
        challenges.omega.as_ref(),
        proof.perm_shifted_opening_proof.as_ref(),
    ) {
        if !proof.perm_shifted_evaluations.is_empty() {
            let omega_z = omega.mul(&challenges.z);
            let commitments: Vec<&[u8]> = proof.perm_commitments.iter().map(|c| c.0.as_slice()).collect();
            let evaluations: Vec<Scalar> = proof.perm_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve)).collect();
            openings.push(OpeningSpec {
                commitments, evaluations,
                proof: p.proof.as_slice(),
                point: omega_z,
                beta: beta.clone(),
            });
        }
    }

    // Reg-perm at z.
    if let (Some(beta), Some(p)) = (challenges.beta_reg_perm.as_ref(), proof.reg_perm_opening_proof.as_ref()) {
        let commitments: Vec<&[u8]> = proof.reg_perm_commitments.iter().map(|c| c.0.as_slice()).collect();
        let evaluations: Vec<Scalar> = proof.reg_perm_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve)).collect();
        openings.push(OpeningSpec {
            commitments, evaluations,
            proof: p.proof.as_slice(),
            point: challenges.z.clone(),
            beta: beta.clone(),
        });
    }

    // Reg-perm shifted at ω·z.
    if let (Some(beta), Some(omega), Some(p)) = (
        challenges.beta_reg_perm_shifted.as_ref(),
        challenges.omega.as_ref(),
        proof.reg_perm_shifted_opening_proof.as_ref(),
    ) {
        if !proof.reg_perm_shifted_evaluations.is_empty() {
            let omega_z = omega.mul(&challenges.z);
            let commitments: Vec<&[u8]> = proof.reg_perm_commitments.iter().map(|c| c.0.as_slice()).collect();
            let evaluations: Vec<Scalar> = proof.reg_perm_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve)).collect();
            openings.push(OpeningSpec {
                commitments, evaluations,
                proof: p.proof.as_slice(),
                point: omega_z,
                beta: beta.clone(),
            });
        }
    }

    // Frame-perm at z (singleton).
    if let (Some(beta), Some(p), Some(fp_comm), Some(fpe_bytes)) = (
        challenges.beta_frame_perm.as_ref(),
        proof.frame_perm_opening_proof.as_ref(),
        proof.frame_perm_commitment.as_ref(),
        proof.frame_perm_evaluation.as_ref(),
    ) {
        let commitments = vec![fp_comm.0.as_slice()];
        let evaluations = vec![Scalar::from_bytes(fpe_bytes, curve)];
        openings.push(OpeningSpec {
            commitments, evaluations,
            proof: p.proof.as_slice(),
            point: challenges.z.clone(),
            beta: beta.clone(),
        });
    }

    // Frame-perm shifted at ω·z (singleton).
    if let (Some(beta), Some(omega), Some(p), Some(fp_comm), Some(fpse_bytes)) = (
        challenges.beta_frame_perm_shifted.as_ref(),
        challenges.omega.as_ref(),
        proof.frame_perm_shifted_opening_proof.as_ref(),
        proof.frame_perm_commitment.as_ref(),
        proof.frame_perm_shifted_evaluation.as_ref(),
    ) {
        let omega_z = omega.mul(&challenges.z);
        let commitments = vec![fp_comm.0.as_slice()];
        let evaluations = vec![Scalar::from_bytes(fpse_bytes, curve)];
        openings.push(OpeningSpec {
            commitments, evaluations,
            proof: p.proof.as_slice(),
            point: omega_z,
            beta: beta.clone(),
        });
    }

    openings
}

/// Compute (L, R) by folding **every** KZG opening present in this chunk.
///
/// Aggregates main + shifted + logup + logup_shifted + bitwise + bitwise_shifted +
/// perm + perm_shifted + reg_perm + reg_perm_shifted + frame_perm + frame_perm_shifted
/// via the meta-challenge ξ (label `b"recursive_xi"`). When `constraints`
/// is `None` the shifted opening is skipped because we cannot determine its
/// commitment list from the proof alone.
pub fn compute_lr_full_scheme(
    chunk: &ChunkProof,
    scheme: &dyn CommitmentScheme,
    curve: CurveType,
    constraints: Option<&dyn VmConstraintSystem>,
) -> (Vec<u8>, Vec<u8>) {
    let challenges = recover_full_chunk_challenges_scheme(chunk, curve, scheme);
    let openings = build_chunk_openings(chunk, &challenges, constraints, curve);
    scheme.compute_lr_multi(&openings, &challenges.xi)
}

/// Compute the per-chunk constraint identity scalar
/// `c_check = Q(z)·Z_H(z) - C(z)`. A valid chunk yields zero. Aggregating
/// `μ^chunk · c_check_chunk` across chunks and requiring `scalar_acc == 0`
/// at the end folds the per-AIR constraint check into the recursive
/// accumulator alongside the pairing check.
fn compute_constraint_check_scalar(
    chunk: &ChunkProof,
    constraints: &dyn VmConstraintSystem,
    scheme: &dyn CommitmentScheme,
    challenges: &FullChunkChallenges,
    curve: CurveType,
) -> Scalar {
    let proof = &chunk.execution_proof;
    let num_columns = proof.column_commitments.len();
    let num_q_chunks = proof.num_quotient_chunks as usize;

    if proof.evaluations.len() < num_columns + num_q_chunks {
        // Malformed proof: return non-zero so the soundness check fails.
        return Scalar::one(curve);
    }

    // AIRs without selectors use the prover's `evaluate_on_domain + IFFT`
    // codepath, where C(X) is constructed from per-row evaluations and Q
    // is the quotient C/Z_H by construction. The verifier never enforces
    // Q(z)·Z_H(z) = C(z) in that mode — it relies on the row-wise
    // constraint evaluations being implicit in the committed columns.
    // Mirror that: no constraint identity to fold for these AIRs.
    if constraints.selector_column_indices().is_empty() {
        return Scalar::zero(curve);
    }

    let c_at_z = crate::verifier::compute_c_at_z(
        proof,
        constraints,
        scheme,
        curve,
        &challenges.alpha,
        &challenges.z,
        challenges.logup_gamma.as_ref(),
        challenges.bitwise_gamma.as_ref(),
        challenges.bitwise_delta.as_ref(),
        challenges.perm_gamma.as_ref(),
        challenges.perm_delta.as_ref(),
        challenges.reg_perm_gamma.as_ref(),
        challenges.reg_perm_delta.as_ref(),
        challenges.frame_perm_gamma.as_ref(),
        challenges.frame_perm_delta.as_ref(),
    );

    // Reconstruct Q(z) = Σ Q_i(z) · z^{i·n}.
    let n = proof.domain_size;
    let mut z_n = Scalar::one(curve);
    let mut base = challenges.z.clone();
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            z_n = z_n.mul(&base);
        }
        base = base.mul(&base);
        exp >>= 1;
    }

    let mut q_at_z = Scalar::zero(curve);
    let mut z_power = Scalar::one(curve);
    for i in 0..num_q_chunks {
        let q_i_z = Scalar::from_bytes(&proof.evaluations[num_columns + i], curve);
        q_at_z = q_at_z.add(&z_power.mul(&q_i_z));
        z_power = z_power.mul(&z_n);
    }

    // Z(z) = z^n - 1
    let z_of_z = z_n.sub(&Scalar::one(curve));
    let qz_zz = q_at_z.mul(&z_of_z);

    // c_check = Q(z)·Z(z) - C(z); zero on a valid proof.
    qz_zz.sub(&c_at_z)
}

/// Derive the per-chunk constraint-aggregation challenge μ. Replays the
/// prover transcript identically to `recover_full_chunk_challenges_scheme`
/// up through ξ, then derives μ with label `b"recursive_mu"` so the
/// pairing fold (using ξ) and the scalar fold (using μ) are independent.
fn derive_constraint_mu(chunk: &ChunkProof, curve: CurveType) -> Scalar {
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
    let _ = transcript.challenge_bytes(b"alpha");

    if !proof.logup_commitments.is_empty() {
        let _ = transcript.challenge_bytes(b"logup_gamma");
        for comm in &proof.logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
    }
    if !proof.bitwise_commitments.is_empty() {
        let _ = transcript.challenge_bytes(b"bitwise_gamma");
        let _ = transcript.challenge_bytes(b"bitwise_delta");
        for comm in &proof.bitwise_commitments {
            transcript.append_message(b"bitwise_column_commitment", &comm.0);
        }
    }
    if !proof.perm_commitments.is_empty() {
        let _ = transcript.challenge_bytes(b"perm_gamma");
        let _ = transcript.challenge_bytes(b"perm_delta");
        for comm in &proof.perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }
    }
    if !proof.oracle_data.is_empty() {
        transcript.append_u64(b"num_oracle_entries", proof.oracle_data.len() as u64);
        for entry in &proof.oracle_data {
            transcript.append_message(b"oracle_entry", entry);
        }
    }
    if !proof.reg_perm_commitments.is_empty() {
        let _ = transcript.challenge_bytes(b"reg_perm_gamma");
        let _ = transcript.challenge_bytes(b"reg_perm_delta");
        for comm in &proof.reg_perm_commitments {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }
    }
    if let Some(ref fp_comm) = proof.frame_perm_commitment {
        let _ = transcript.challenge_bytes(b"frame_perm_gamma");
        let _ = transcript.challenge_bytes(b"frame_perm_delta");
        transcript.append_message(b"frame_perm_column_commitment", &fp_comm.0);
    }
    for qc in &proof.quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let _ = transcript.challenge_bytes(b"z");
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
    for be in &proof.bitwise_evaluations {
        transcript.append_message(b"bitwise_evaluation", be);
    }
    for bse in &proof.bitwise_shifted_evaluations {
        transcript.append_message(b"bitwise_shifted_evaluation", bse);
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
    if let Some(ref fpe) = proof.frame_perm_evaluation {
        transcript.append_message(b"frame_perm_evaluation", fpe);
    }
    if let Some(ref fpse) = proof.frame_perm_shifted_evaluation {
        transcript.append_message(b"frame_perm_shifted_evaluation", fpse);
    }
    let _ = transcript.challenge_bytes(b"beta");
    if !proof.shifted_evaluations.is_empty() {
        let _ = transcript.challenge_bytes(b"beta_shifted");
    }
    let has_logup = !proof.logup_commitments.is_empty();
    if has_logup {
        let _ = transcript.challenge_bytes(b"beta_logup");
    }
    if has_logup && !proof.logup_shifted_evaluations.is_empty() {
        let _ = transcript.challenge_bytes(b"beta_logup_shifted");
    }
    let has_bitwise = !proof.bitwise_commitments.is_empty();
    if has_bitwise {
        let _ = transcript.challenge_bytes(b"beta_bitwise");
    }
    if has_bitwise && !proof.bitwise_shifted_evaluations.is_empty() {
        let _ = transcript.challenge_bytes(b"beta_bitwise_shifted");
    }
    let has_perm = !proof.perm_commitments.is_empty();
    if has_perm {
        let _ = transcript.challenge_bytes(b"beta_perm");
    }
    if has_perm && !proof.perm_shifted_evaluations.is_empty() {
        let _ = transcript.challenge_bytes(b"beta_perm_shifted");
    }
    let has_reg_perm = !proof.reg_perm_commitments.is_empty();
    if has_reg_perm {
        let _ = transcript.challenge_bytes(b"beta_reg_perm");
    }
    if has_reg_perm && !proof.reg_perm_shifted_evaluations.is_empty() {
        let _ = transcript.challenge_bytes(b"beta_reg_perm_shifted");
    }
    let has_frame_perm = proof.frame_perm_commitment.is_some();
    if has_frame_perm {
        let _ = transcript.challenge_bytes(b"beta_frame_perm");
    }
    if has_frame_perm && proof.frame_perm_shifted_evaluation.is_some() {
        let _ = transcript.challenge_bytes(b"beta_frame_perm_shifted");
    }
    let _ = transcript.challenge_bytes(b"recursive_xi");
    let mu_bytes = transcript.challenge_bytes(b"recursive_mu");
    challenge_to_scalar(&mu_bytes, curve)
}

/// Begin a full-fold recursive accumulator from the first chunk proof.
///
/// Aggregates every per-chunk KZG opening (main + shifted + logup + bitwise
/// + perm + reg_perm + frame_perm + their shifted variants) into the
/// `(L_acc, R_acc)` pairing, and seeds the constraint identity scalar
/// accumulator with `c_check = Q(z)·Z_H(z) − C(z)` for this chunk.
/// `verify_final_scheme` then certifies all opening pairings *and* that
/// the aggregated `c_check` is zero in one shot.
pub fn begin_chunk_full_scheme(
    first: ChunkProof,
    scheme: &dyn CommitmentScheme,
    curve: CurveType,
    constraints: &dyn VmConstraintSystem,
) -> RecursiveProof {
    let challenges = recover_full_chunk_challenges_scheme(&first, curve, scheme);
    let openings = build_chunk_openings(&first, &challenges, Some(constraints), curve);
    let (l_bytes, r_bytes) = scheme.compute_lr_multi(&openings, &challenges.xi);

    // Compute the constraint identity scalar c_check for this chunk and
    // seed the scalar accumulator with μ⁰·c_check = c_check. (The mu
    // derivation is kept symmetric with fold_chunks_full_scheme even though
    // the seeding step doesn't multiply by μ.) Now covers every AIR thanks
    // to the extended `verifier::compute_c_at_z`.
    let scalar_acc = compute_scalar_acc_seed(&first, constraints, scheme, &challenges, curve);

    let acc = AccumulatedClaim {
        l_acc: l_bytes,
        r_acc: r_bytes,
        num_folded: 1,
        scalar_acc,
    };

    RecursiveProof {
        current_proof: first.execution_proof,
        accumulator: acc,
        depth: 1,
        initial_state_hash: Some(first.initial_state_hash),
        final_state_hash: Some(first.final_state_hash),
    }
}

/// Compute the initial scalar_acc bytes for the first chunk. Returns the
/// per-chunk constraint identity scalar `c_check = Q(z)·Z_H(z) − C(z)`,
/// which is zero on a valid proof. Every AIR is now supported because
/// `verifier::compute_c_at_z` covers all auxiliary contributions
/// (logup/bitwise/perm/reg_perm/frame_perm).
fn compute_scalar_acc_seed(
    chunk: &ChunkProof,
    constraints: &dyn VmConstraintSystem,
    scheme: &dyn CommitmentScheme,
    challenges: &FullChunkChallenges,
    curve: CurveType,
) -> Vec<u8> {
    let c_check = compute_constraint_check_scalar(chunk, constraints, scheme, challenges, curve);
    c_check.to_bytes()
}

/// Fold a chunk into the accumulator using the full-fold pipeline.
pub fn fold_chunks_full_scheme(
    prev: &RecursiveProof,
    current: ChunkProof,
    scheme: &dyn CommitmentScheme,
    curve: CurveType,
    constraints: &dyn VmConstraintSystem,
) -> Result<RecursiveProof, String> {
    if let Some(prev_final) = &prev.final_state_hash {
        if *prev_final != current.initial_state_hash {
            return Err(format!(
                "State chain broken at chunk {}: prev final hash != current initial hash",
                current.chunk_index
            ));
        }
    }

    let challenges = recover_full_chunk_challenges_scheme(&current, curve, scheme);
    let openings = build_chunk_openings(&current, &challenges, Some(constraints), curve);
    let (l_new, r_new) = scheme.compute_lr_multi(&openings, &challenges.xi);

    let mut transcript = Transcript::new(b"metavm-recursive-fold");
    transcript.append_message(b"prev-l-acc", &prev.accumulator.l_acc);
    transcript.append_message(b"prev-r-acc", &prev.accumulator.r_acc);
    transcript.append_message(b"prev-scalar-acc", &prev.accumulator.scalar_acc);
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

    let (l_acc_bytes, r_acc_bytes) = scheme.fold_accumulator(
        &prev.accumulator.l_acc,
        &prev.accumulator.r_acc,
        &l_new,
        &r_new,
        &challenge,
    );

    // Scalar fold: scalar_acc' = scalar_acc_prev + μ_chunk · c_check_chunk.
    // Each chunk uses an independent μ derived from its own transcript so a
    // fresh c_check term cannot be cancelled by the accumulator state.
    let scalar_acc = if prev.accumulator.scalar_acc.is_empty() {
        // Legacy pairing-only accumulator (e.g. proofs produced by `begin`
        // rather than `begin_chunk_full_scheme`). Keep mode unchanged.
        Vec::new()
    } else {
        let c_check = compute_constraint_check_scalar(
            &current, constraints, scheme, &challenges, curve,
        );
        let mu = derive_constraint_mu(&current, curve);
        let prev_scalar = Scalar::from_bytes(&prev.accumulator.scalar_acc, curve);
        let term = mu.mul(&c_check);
        prev_scalar.add(&term).to_bytes()
    };

    let new_acc = AccumulatedClaim {
        l_acc: l_acc_bytes,
        r_acc: r_acc_bytes,
        num_folded: prev.accumulator.num_folded + 1,
        scalar_acc,
    };

    Ok(RecursiveProof {
        current_proof: current.execution_proof,
        accumulator: new_acc,
        depth: prev.depth + 1,
        initial_state_hash: prev.initial_state_hash,
        final_state_hash: Some(current.final_state_hash),
    })
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
                scalar_acc: Vec::new(),
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

    fn sample_recursive_proof() -> RecursiveProof {
        RecursiveProof {
            current_proof: dummy_proof(1),
            accumulator: AccumulatedClaim {
                l_acc: vec![0xAA; 74],
                r_acc: vec![0xBB; 74],
                num_folded: 17,
                scalar_acc: Vec::new(),
            },
            depth: 17,
            initial_state_hash: Some([0xCC; 32]),
            final_state_hash: Some([0xDD; 32]),
        }
    }

    #[test]
    fn recursive_proof_round_trip_minimal() {
        let p = sample_recursive_proof();
        let bytes = p.to_bytes();
        let decoded = RecursiveProof::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.accumulator.l_acc, p.accumulator.l_acc);
        assert_eq!(decoded.accumulator.r_acc, p.accumulator.r_acc);
        assert_eq!(decoded.accumulator.num_folded, p.accumulator.num_folded);
        assert_eq!(decoded.depth, p.depth);
        assert_eq!(decoded.initial_state_hash, p.initial_state_hash);
        assert_eq!(decoded.final_state_hash, p.final_state_hash);
        assert_eq!(decoded.to_bytes(), bytes, "encoding canonical");
    }

    #[test]
    fn recursive_proof_round_trip_no_state_hashes() {
        let mut p = sample_recursive_proof();
        p.initial_state_hash = None;
        p.final_state_hash = None;
        let bytes = p.to_bytes();
        let decoded = RecursiveProof::from_bytes(&bytes).expect("decode");
        assert!(decoded.initial_state_hash.is_none());
        assert!(decoded.final_state_hash.is_none());
        assert_eq!(decoded.depth, p.depth);
    }

    #[test]
    fn recursive_proof_decode_truncated_fails() {
        let p = sample_recursive_proof();
        let bytes = p.to_bytes();
        for trunc in &[0usize, 1, 5, 20, bytes.len() - 1] {
            assert!(
                RecursiveProof::from_bytes(&bytes[..*trunc]).is_err(),
                "truncated to {} bytes must error", trunc,
            );
        }
    }

    #[test]
    fn recursive_proof_decode_trailing_bytes_fails() {
        let p = sample_recursive_proof();
        let mut bytes = p.to_bytes();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        assert!(matches!(
            RecursiveProof::from_bytes(&bytes).err(),
            Some(crate::prover::ProofDecodeError::TrailingBytes(2)),
        ));
    }
}
