//! Execution proof verification.
//!
//! Verifies proofs produced by the prover module. Rebuilds the Fiat-Shamir
//! transcript to re-derive all challenges, recomputes C(z) from column
//! evaluations using the constraint system, checks the quotient identity
//! Q(z)*Z(z) = C(z), and verifies the batch KZG opening proof via pairing.

use crate::commitment;
use crate::prover::{ExecutionProof, ChunkProof, challenge_to_big};
use crate::vm_constraints::VmConstraintSystem;
use bls48581::bls;
use bls48581::bls48581::big;
use bls48581::bls48581::rom;
use metavm_core::transcript::Transcript;

/// Verify an execution proof.
///
/// Rebuilds the Fiat-Shamir transcript from the proof's commitments to re-derive
/// the same challenges the prover used, recomputes C(z) from column evaluations
/// using the constraint system, then:
/// 1. Re-derives z as an arbitrary field element
/// 2. Recomputes C(z) = constraints.evaluate_at_point(col_evals, alpha)
/// 3. Reconstructs Q(z) from quotient chunk evaluations
/// 4. Checks Q(z) * Z(z) == C(z)
/// 5. Verifies the batch KZG opening proof via pairing check
pub fn verify(proof: &ExecutionProof, constraints: &dyn VmConstraintSystem) -> bool {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    verify_inner(proof, constraints, &mut transcript)
}

/// Verify an execution proof using a generic CommitmentScheme.
///
/// Works with any curve type via the [`CommitmentScheme`] trait (BLS48-581 or BLS12-381).
pub fn verify_with_scheme(
    proof: &ExecutionProof,
    constraints: &dyn VmConstraintSystem,
    scheme: &dyn crate::scheme::CommitmentScheme,
    curve: crate::field::CurveType,
) -> bool {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    verify_inner_scheme(proof, constraints, &mut transcript, scheme, curve)
}

/// Verify a chunk proof.
///
/// Rebuilds the Fiat-Shamir transcript with the chunk-specific metadata
/// (chunk_index, initial_state_hash, final_state_hash) bound in, then
/// verifies the underlying execution proof.
pub fn verify_chunk(chunk_proof: &ChunkProof, constraints: &dyn VmConstraintSystem) -> bool {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    transcript.append_u64(b"chunk_index", chunk_proof.chunk_index);
    transcript.append_message(b"initial_state", &chunk_proof.initial_state_hash);
    transcript.append_message(b"final_state", &chunk_proof.final_state_hash);

    verify_inner(&chunk_proof.execution_proof, constraints, &mut transcript)
}

/// Core verification logic shared by `verify()` and `verify_chunk()`.
fn verify_inner(
    proof: &ExecutionProof,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
) -> bool {
    use crate::field::Scalar;

    // Structural validity checks
    if proof.column_commitments.is_empty() {
        return false;
    }
    if proof.num_steps == 0 {
        return false;
    }

    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    let num_columns = proof.column_commitments.len();
    let num_q_chunks = proof.num_quotient_chunks as usize;

    // Evaluations layout: [col_0..col_k, Q_0, Q_1, ...] = num_columns + num_q_chunks
    if proof.evaluations.len() != num_columns + num_q_chunks {
        return false;
    }

    // -----------------------------------------------------------------------
    // 1. Rebuild Fiat-Shamir transcript identically to the prover
    // -----------------------------------------------------------------------
    transcript.append_u64(b"num_steps", proof.num_steps);
    transcript.append_u64(b"domain_size", proof.domain_size);

    for comm in &proof.column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    // Re-derive constraint challenge alpha
    let alpha_bytes = transcript.challenge_bytes(b"alpha");
    let alpha_big = challenge_to_big(&alpha_bytes);
    let alpha = Scalar::Bls48581(big::BIG::new_copy(&alpha_big));

    // LogUp transcript reconstruction (must match prover Step 3b)
    let has_logup = !proof.logup_commitments.is_empty();
    if has_logup {
        let _gamma_bytes = transcript.challenge_bytes(b"logup_gamma");
        for comm in &proof.logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
    }

    // Permutation transcript reconstruction (must match prover Step 3c)
    let has_perm = !proof.perm_commitments.is_empty();
    if has_perm {
        let _perm_gamma_bytes = transcript.challenge_bytes(b"perm_gamma");
        let _perm_delta_bytes = transcript.challenge_bytes(b"perm_delta");
        for comm in &proof.perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }
    }

    // Oracle data transcript reconstruction (must match prover Step 3d)
    if !proof.oracle_data.is_empty() {
        transcript.append_u64(b"num_oracle_entries", proof.oracle_data.len() as u64);
        for entry in &proof.oracle_data {
            transcript.append_message(b"oracle_entry", entry);
        }
    }

    // Register permutation transcript reconstruction (must match prover Step 3e)
    let has_reg_perm = !proof.reg_perm_commitments.is_empty();
    if has_reg_perm {
        let _rg_bytes = transcript.challenge_bytes(b"reg_perm_gamma");
        let _rd_bytes = transcript.challenge_bytes(b"reg_perm_delta");
        for comm in &proof.reg_perm_commitments {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }
    }

    // Frame-stack permutation transcript reconstruction.
    if let Some(ref fp_comm) = proof.frame_perm_commitment {
        let _fpg = transcript.challenge_bytes(b"frame_perm_gamma");
        let _fpd = transcript.challenge_bytes(b"frame_perm_delta");
        transcript.append_message(b"frame_perm_column_commitment", &fp_comm.0);
    }

    // Absorb quotient commitments, re-derive z
    for qc in &proof.quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = challenge_to_big(&z_bytes);

    // -----------------------------------------------------------------------
    // 2. Quotient identity check: Q(z) * Z(z) == C(z)
    //
    // Recompute C(z) from column evaluations at the random point z.
    // Includes VM constraints, cross-row constraints, LogUp decomposition,
    // and memory permutation constraints.
    // -----------------------------------------------------------------------
    // Always check the `Q(z)·Z(z) == C(z)` identity; combined with the
    // prover's matching change forcing `use_build_poly = true`, every
    // AIR's constraints are now actually enforced.
    let has_selectors = true;

    if has_selectors {
        let col_evals: Vec<Scalar> = proof.evaluations[..num_columns]
            .iter()
            .map(|e| Scalar::Bls48581(big::BIG::frombytes(e)))
            .collect();

        let mut c_at_z = constraints.evaluate_at_point(&col_evals, &alpha);

        // Add cross-row constraint contribution if shifted columns exist
        let shifted_indices = constraints.shifted_column_indices();
        if !shifted_indices.is_empty() && !proof.shifted_evaluations.is_empty() {
            // Compute ω^{n-1} from ω
            let z_scalar = Scalar::Bls48581(big::BIG::new_copy(&z));
            // ω = domain generator
            let s = bls::singleton();
            let n = proof.domain_size;
            let omega_big = s.RootsOfUnityBLS48581[&n][1].clone();
            // ω^{n-1} = ω^{-1} (since ω^n = 1)
            let mut omega_n_minus_1_big = omega_big.clone();
            omega_n_minus_1_big.invmodp(&modulus);
            let omega_n_minus_1 = Scalar::Bls48581(omega_n_minus_1_big);

            let shifted_evals: Vec<Scalar> = proof.shifted_evaluations.iter()
                .map(|e| Scalar::Bls48581(big::BIG::frombytes(e)))
                .collect();

            let c_shifted = constraints.evaluate_shifted_at_point(
                &col_evals, &shifted_evals, &z_scalar, &omega_n_minus_1,
                &alpha, constraints.num_constraints(),
            );
            c_at_z = c_at_z.add(&c_shifted);
        }

        let c_eval = big::BIG::new_copy(c_at_z.as_bls48581());

        // Reconstruct Q(z) = Q_0(z) + z^n * Q_1(z) + ...
        let n = proof.domain_size;
        let mut z_n = big::BIG::new_int(1);
        let mut base = big::BIG::new_copy(&z);
        let mut exp = n;
        while exp > 0 {
            if exp & 1 == 1 {
                z_n = big::BIG::modmul(&z_n, &base, &modulus);
            }
            base = big::BIG::modmul(&base, &base, &modulus);
            exp >>= 1;
        }

        let mut q_at_z = big::BIG::new();
        let mut z_power = big::BIG::new_int(1);
        for i in 0..num_q_chunks {
            let q_i_z = big::BIG::frombytes(&proof.evaluations[num_columns + i]);
            let term = big::BIG::modmul(&z_power, &q_i_z, &modulus);
            q_at_z = big::BIG::modadd(&q_at_z, &term, &modulus);
            z_power = big::BIG::modmul(&z_power, &z_n, &modulus);
        }

        // Z(z) = z^n - 1
        let one = big::BIG::new_int(1);
        let z_of_z = big::BIG::modadd(&z_n, &big::BIG::modneg(&one, &modulus), &modulus);

        // Q(z) * Z(z)
        let qz_zz = big::BIG::modmul(&q_at_z, &z_of_z, &modulus);

        // Check Q(z)*Z(z) == C(z)
        let diff = big::BIG::modadd(&qz_zz, &big::BIG::modneg(&c_eval, &modulus), &modulus);
        if !diff.iszilch() {
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // 5. Absorb evaluations and re-derive β (must match prover)
    // -----------------------------------------------------------------------
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
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = challenge_to_big(&beta_bytes);

    // -----------------------------------------------------------------------
    // 6. Verify the batch KZG opening proof via pairing
    //
    // All committed polynomials: [col_0, ..., col_k, Q_0, Q_1, ...]
    // Combine commitments and evaluations with β, then verify:
    // e(C_combined - y_combined*G1, G2) == e(π, [τ]₂ - z*G2)
    // -----------------------------------------------------------------------
    let mut all_commitment_bytes: Vec<&[u8]> = Vec::with_capacity(num_columns + num_q_chunks);
    for comm in &proof.column_commitments {
        all_commitment_bytes.push(&comm.0);
    }
    for qc in &proof.quotient_commitments {
        all_commitment_bytes.push(&qc.0);
    }

    let all_evals: Vec<big::BIG> = proof.evaluations.iter()
        .map(|e| big::BIG::frombytes(e))
        .collect();

    if !commitment::batch_verify_at_point(
        &all_commitment_bytes,
        &all_evals,
        &z,
        &beta,
        &proof.opening_proof.proof,
    ) {
        return false;
    }

    // -----------------------------------------------------------------------
    // 7. Verify shifted opening proof at ω·z (if cross-row constraints exist)
    // -----------------------------------------------------------------------
    let shifted_indices = constraints.shifted_column_indices();
    if !shifted_indices.is_empty() {
        if let Some(ref shifted_proof) = proof.shifted_opening_proof {
            let beta_shifted_bytes = transcript.challenge_bytes(b"beta_shifted");
            let beta_shifted = challenge_to_big(&beta_shifted_bytes);

            let s = bls::singleton();
            let omega_big = s.RootsOfUnityBLS48581[&proof.domain_size][1].clone();
            let omega_z = big::BIG::modmul(&omega_big, &z, &modulus);

            let shifted_commitments: Vec<&[u8]> = shifted_indices.iter()
                .map(|&idx| proof.column_commitments[idx].0.as_slice())
                .collect();

            let shifted_evals: Vec<big::BIG> = proof.shifted_evaluations.iter()
                .map(|e| big::BIG::frombytes(e))
                .collect();

            if !commitment::batch_verify_at_point(
                &shifted_commitments,
                &shifted_evals,
                &omega_z,
                &beta_shifted,
                &shifted_proof.proof,
            ) {
                return false;
            }
        } else {
            return false; // shifted proof required but missing
        }
    }

    // -----------------------------------------------------------------------
    // 8. Verify LogUp opening proofs
    // -----------------------------------------------------------------------
    if has_logup {
        let _beta_logup_bytes = transcript.challenge_bytes(b"beta_logup");
        if !proof.logup_shifted_evaluations.is_empty() {
            let _beta_logup_shifted_bytes = transcript.challenge_bytes(b"beta_logup_shifted");
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // 9. Verify permutation opening proofs
    // -----------------------------------------------------------------------
    if has_perm {
        let _beta_perm_bytes = transcript.challenge_bytes(b"beta_perm");
        if !proof.perm_shifted_evaluations.is_empty() {
            let _beta_perm_shifted_bytes = transcript.challenge_bytes(b"beta_perm_shifted");
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // 10. Verify register permutation opening proofs
    // -----------------------------------------------------------------------
    if has_reg_perm {
        let _beta_rp_bytes = transcript.challenge_bytes(b"beta_reg_perm");
        if !proof.reg_perm_shifted_evaluations.is_empty() {
            let _beta_rps_bytes = transcript.challenge_bytes(b"beta_reg_perm_shifted");
        }
        return false;
    }

    true
}

/// Core verification logic using a generic CommitmentScheme.
fn verify_inner_scheme(
    proof: &ExecutionProof,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
    scheme: &dyn crate::scheme::CommitmentScheme,
    curve: crate::field::CurveType,
) -> bool {
    use crate::field::Scalar;

    // Structural validity checks
    if proof.column_commitments.is_empty() {
        return false;
    }
    if proof.num_steps == 0 {
        return false;
    }

    let num_columns = proof.column_commitments.len();
    let num_q_chunks = proof.num_quotient_chunks as usize;

    // Evaluations layout: [col_0..col_k, Q_0, Q_1, ...] = num_columns + num_q_chunks
    if proof.evaluations.len() != num_columns + num_q_chunks {
        return false;
    }

    // -----------------------------------------------------------------------
    // 1. Rebuild Fiat-Shamir transcript identically to the prover
    // -----------------------------------------------------------------------
    transcript.append_u64(b"num_steps", proof.num_steps);
    transcript.append_u64(b"domain_size", proof.domain_size);

    for comm in &proof.column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    // Re-derive constraint challenge alpha
    let alpha_bytes = transcript.challenge_bytes(b"alpha");
    let alpha = Scalar::from_challenge_bytes(&alpha_bytes, curve);

    // -----------------------------------------------------------------------
    // 1b. LogUp transcript reconstruction (must match prover Step 3b)
    // -----------------------------------------------------------------------
    let has_logup = !proof.logup_commitments.is_empty();
    let logup_gamma = if has_logup {
        let gamma_bytes = transcript.challenge_bytes(b"logup_gamma");
        let gamma = Scalar::from_challenge_bytes(&gamma_bytes, curve);
        for comm in &proof.logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
        Some(gamma)
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // 1b'. Bitwise LogUp transcript reconstruction (matches prover Step 3b')
    // -----------------------------------------------------------------------
    let has_bitwise = !proof.bitwise_commitments.is_empty();
    let (bitwise_gamma, bitwise_delta) = if has_bitwise {
        let gamma_bytes = transcript.challenge_bytes(b"bitwise_gamma");
        let bw_gamma = Scalar::from_challenge_bytes(&gamma_bytes, curve);
        let delta_bytes = transcript.challenge_bytes(b"bitwise_delta");
        let bw_delta = Scalar::from_challenge_bytes(&delta_bytes, curve);
        for comm in &proof.bitwise_commitments {
            transcript.append_message(b"bitwise_column_commitment", &comm.0);
        }
        (Some(bw_gamma), Some(bw_delta))
    } else {
        (None, None)
    };

    // -----------------------------------------------------------------------
    // 1c. Permutation transcript reconstruction (must match prover Step 3c)
    // -----------------------------------------------------------------------
    let has_perm = !proof.perm_commitments.is_empty();
    let (perm_gamma, perm_delta) = if has_perm {
        let perm_gamma_bytes = transcript.challenge_bytes(b"perm_gamma");
        let pg = Scalar::from_challenge_bytes(&perm_gamma_bytes, curve);
        let perm_delta_bytes = transcript.challenge_bytes(b"perm_delta");
        let pd = Scalar::from_challenge_bytes(&perm_delta_bytes, curve);
        for comm in &proof.perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }
        (Some(pg), Some(pd))
    } else {
        (None, None)
    };

    // -----------------------------------------------------------------------
    // 1d. Oracle data transcript reconstruction (must match prover Step 3d)
    // -----------------------------------------------------------------------
    if !proof.oracle_data.is_empty() {
        transcript.append_u64(b"num_oracle_entries", proof.oracle_data.len() as u64);
        for entry in &proof.oracle_data {
            transcript.append_message(b"oracle_entry", entry);
        }
    }

    // -----------------------------------------------------------------------
    // 1e. Register permutation transcript reconstruction (must match prover Step 3e)
    // -----------------------------------------------------------------------
    let has_reg_perm = !proof.reg_perm_commitments.is_empty();
    let (reg_perm_gamma, reg_perm_delta) = if has_reg_perm {
        let rg_bytes = transcript.challenge_bytes(b"reg_perm_gamma");
        let rg = Scalar::from_challenge_bytes(&rg_bytes, curve);
        let rd_bytes = transcript.challenge_bytes(b"reg_perm_delta");
        let rd = Scalar::from_challenge_bytes(&rd_bytes, curve);
        for comm in &proof.reg_perm_commitments {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }
        (Some(rg), Some(rd))
    } else {
        (None, None)
    };

    // -----------------------------------------------------------------------
    // 1f. Frame-stack permutation transcript reconstruction.
    // -----------------------------------------------------------------------
    let has_frame_perm = proof.frame_perm_commitment.is_some();
    let (frame_perm_gamma, frame_perm_delta) = if has_frame_perm {
        let fpg_bytes = transcript.challenge_bytes(b"frame_perm_gamma");
        let fpg = Scalar::from_challenge_bytes(&fpg_bytes, curve);
        let fpd_bytes = transcript.challenge_bytes(b"frame_perm_delta");
        let fpd = Scalar::from_challenge_bytes(&fpd_bytes, curve);
        if let Some(ref comm) = proof.frame_perm_commitment {
            transcript.append_message(b"frame_perm_column_commitment", &comm.0);
        }
        (Some(fpg), Some(fpd))
    } else {
        (None, None)
    };

    // Absorb quotient commitments, re-derive z
    for qc in &proof.quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = Scalar::from_challenge_bytes(&z_bytes, curve);

    // -----------------------------------------------------------------------
    // 2. Quotient identity check: Q(z) * Z(z) == C(z)
    //
    // Recompute C(z) from column evaluations at the random point z.
    // Includes VM constraints, cross-row constraints, LogUp decomposition,
    // and memory permutation constraints.
    // -----------------------------------------------------------------------
    // See the matching note above; this is the scheme-generic
    // counterpart of the same identity check.
    let has_selectors = true;

    if has_selectors {
        let c_at_z = compute_c_at_z(
            proof,
            constraints,
            scheme,
            curve,
            &alpha,
            &z,
            logup_gamma.as_ref(),
            bitwise_gamma.as_ref(),
            bitwise_delta.as_ref(),
            perm_gamma.as_ref(),
            perm_delta.as_ref(),
            reg_perm_gamma.as_ref(),
            reg_perm_delta.as_ref(),
            frame_perm_gamma.as_ref(),
            frame_perm_delta.as_ref(),
        );

        let n = proof.domain_size;
        let one = Scalar::one(curve);

        // Compute z^n via repeated squaring
        let mut z_n = Scalar::one(curve);
        let mut base = z.clone();
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
        let z_of_z = z_n.sub(&one);

        // Q(z) * Z(z)
        let qz_zz = q_at_z.mul(&z_of_z);

        // Check Q(z)*Z(z) == C(z)
        let diff = qz_zz.sub(&c_at_z);
        if !diff.is_zero() {
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // 5. Absorb evaluations and re-derive β (must match prover)
    // -----------------------------------------------------------------------
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
    let _ = &proof.frame_perm_pop_shifted_evaluations; // reserved
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = Scalar::from_challenge_bytes(&beta_bytes, curve);

    // -----------------------------------------------------------------------
    // 6. Verify the batch KZG opening proof via pairing
    // -----------------------------------------------------------------------
    let mut all_commitment_bytes: Vec<&[u8]> = Vec::with_capacity(num_columns + num_q_chunks);
    for comm in &proof.column_commitments {
        all_commitment_bytes.push(&comm.0);
    }
    for qc in &proof.quotient_commitments {
        all_commitment_bytes.push(&qc.0);
    }

    let all_evals: Vec<Scalar> = proof.evaluations.iter()
        .map(|e| Scalar::from_bytes(e, curve))
        .collect();

    if !scheme.batch_verify_at_point(
        &all_commitment_bytes,
        &all_evals,
        &z,
        &beta,
        &proof.opening_proof.proof,
    ) {
        return false;
    }

    // -----------------------------------------------------------------------
    // 7. Verify shifted opening proof at ω·z (if cross-row constraints exist)
    // -----------------------------------------------------------------------
    let shifted_indices = constraints.shifted_column_indices();
    if !shifted_indices.is_empty() {
        if let Some(ref shifted_proof) = proof.shifted_opening_proof {
            let beta_shifted_bytes = transcript.challenge_bytes(b"beta_shifted");
            let beta_shifted = Scalar::from_challenge_bytes(&beta_shifted_bytes, curve);

            let omega = scheme.domain_generator(proof.domain_size);
            let omega_z = omega.mul(&z);

            let shifted_commitments: Vec<&[u8]> = shifted_indices.iter()
                .map(|&idx| proof.column_commitments[idx].0.as_slice())
                .collect();

            let shifted_evals: Vec<Scalar> = proof.shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            if !scheme.batch_verify_at_point(
                &shifted_commitments,
                &shifted_evals,
                &omega_z,
                &beta_shifted,
                &shifted_proof.proof,
            ) {
                return false;
            }
        } else {
            return false; // shifted proof required but missing
        }
    }

    // -----------------------------------------------------------------------
    // 8. Verify LogUp opening proofs (if lookup declarations exist)
    // -----------------------------------------------------------------------
    if has_logup {
        let logup_evals: Vec<Scalar> = proof.logup_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();

        // 8b. Verify LogUp batch opening at z
        if let Some(ref logup_proof) = proof.logup_opening_proof {
            let beta_logup_bytes = transcript.challenge_bytes(b"beta_logup");
            let beta_logup = Scalar::from_challenge_bytes(&beta_logup_bytes, curve);

            let logup_commitment_bytes: Vec<&[u8]> = proof.logup_commitments.iter()
                .map(|c| c.0.as_slice())
                .collect();

            if !scheme.batch_verify_at_point(
                &logup_commitment_bytes,
                &logup_evals,
                &z,
                &beta_logup,
                &logup_proof.proof,
            ) {
                return false;
            }
        } else {
            return false; // LogUp proof required but missing
        }

        // 8c. Verify LogUp shifted opening at ω·z (running sum h)
        if let Some(ref logup_shifted_proof) = proof.logup_shifted_opening_proof {
            let beta_logup_shifted_bytes = transcript.challenge_bytes(b"beta_logup_shifted");
            let beta_logup_shifted = Scalar::from_challenge_bytes(&beta_logup_shifted_bytes, curve);

            let omega = scheme.domain_generator(proof.domain_size);
            let omega_z = omega.mul(&z);

            // h is the second-to-last LogUp column
            let h_commit_idx = proof.logup_commitments.len() - 2;
            let h_commitment = vec![proof.logup_commitments[h_commit_idx].0.as_slice()];

            let h_shifted_evals: Vec<Scalar> = proof.logup_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            if !scheme.batch_verify_at_point(
                &h_commitment,
                &h_shifted_evals,
                &omega_z,
                &beta_logup_shifted,
                &logup_shifted_proof.proof,
            ) {
                return false;
            }
        } else {
            return false; // LogUp shifted proof required but missing
        }

        // 8d. Bind committed range-table column `t` to the canonical preprocessed
        // polynomial: recompute t(z) via Lagrange on {t[ω^i] = i for i<256, else 0}
        // and compare against the prover's claimed t(z). Without this, the prover
        // could commit an arbitrary table and forge lookups.
        {
            let lookup_reqs = constraints.lookup_declarations();
            let logup_groups = crate::lookup::group_declarations(&lookup_reqs);
            if !logup_groups.is_empty() {
                let logup_layout = crate::lookup::extended_logup_column_layout(&logup_groups);
                let logup_evals: Vec<Scalar> = proof.logup_evaluations.iter()
                    .map(|e| Scalar::from_bytes(e, curve))
                    .collect();
                let t_idx = logup_layout.t_column;
                if t_idx < logup_evals.len() {
                    let claimed_t_at_z = &logup_evals[t_idx];
                    let omega = scheme.domain_generator(proof.domain_size);
                    let expected_t_at_z = crate::lookup::evaluate_range_table_at_point(
                        &z, &omega, proof.domain_size as usize, curve,
                    );
                    if !claimed_t_at_z.sub(&expected_t_at_z).is_zero() {
                        return false;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 8'. Verify bitwise LogUp opening proofs (at z and ω·z) + canonical t binding
    // -----------------------------------------------------------------------
    if has_bitwise {
        let bw_evals: Vec<Scalar> = proof.bitwise_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();

        if let Some(ref bw_proof) = proof.bitwise_opening_proof {
            let beta_bw_bytes = transcript.challenge_bytes(b"beta_bitwise");
            let beta_bw = Scalar::from_challenge_bytes(&beta_bw_bytes, curve);
            let bw_commitment_bytes: Vec<&[u8]> = proof.bitwise_commitments.iter()
                .map(|c| c.0.as_slice())
                .collect();
            if !scheme.batch_verify_at_point(
                &bw_commitment_bytes,
                &bw_evals,
                &z,
                &beta_bw,
                &bw_proof.proof,
            ) {
                return false;
            }
        } else {
            return false;
        }

        if let Some(ref bw_shifted_proof) = proof.bitwise_shifted_opening_proof {
            let beta_bw_shifted_bytes = transcript.challenge_bytes(b"beta_bitwise_shifted");
            let beta_bw_shifted = Scalar::from_challenge_bytes(&beta_bw_shifted_bytes, curve);
            let omega = scheme.domain_generator(proof.domain_size);
            let omega_z = omega.mul(&z);
            // h is at bw_layout.h_column — derive the commitment position.
            let bitwise_decls = constraints.bitwise_lookup_declarations();
            let bitwise_groups = crate::lookup::group_bitwise_declarations(&bitwise_decls);
            let bw_layout = crate::lookup::extended_bitwise_column_layout(&bitwise_groups);
            let h_commit_idx = bw_layout.h_column;
            if h_commit_idx >= proof.bitwise_commitments.len() {
                return false;
            }
            let h_commitment = vec![proof.bitwise_commitments[h_commit_idx].0.as_slice()];
            let h_shifted_evals: Vec<Scalar> = proof.bitwise_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();
            if !scheme.batch_verify_at_point(
                &h_commitment,
                &h_shifted_evals,
                &omega_z,
                &beta_bw_shifted,
                &bw_shifted_proof.proof,
            ) {
                return false;
            }
        } else {
            return false;
        }

        // Canonical nibble-AND table binding: verifier recomputes t_a(z),
        // t_b(z), t_c(z) independently and compares with the prover's claims.
        {
            let bitwise_decls = constraints.bitwise_lookup_declarations();
            let bitwise_groups = crate::lookup::group_bitwise_declarations(&bitwise_decls);
            let bw_layout = crate::lookup::extended_bitwise_column_layout(&bitwise_groups);
            let omega = scheme.domain_generator(proof.domain_size);
            for (component, idx) in [
                (crate::lookup::NibbleAndTableComponent::A, bw_layout.t_a_column),
                (crate::lookup::NibbleAndTableComponent::B, bw_layout.t_b_column),
                (crate::lookup::NibbleAndTableComponent::C, bw_layout.t_c_column),
            ] {
                if idx >= bw_evals.len() {
                    return false;
                }
                let claimed = &bw_evals[idx];
                let expected = crate::lookup::evaluate_nibble_and_table_at_point(
                    component, &z, &omega, proof.domain_size as usize, curve,
                );
                if !claimed.sub(&expected).is_zero() {
                    return false;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 9. Verify permutation opening proofs (if memory permutation exists)
    // -----------------------------------------------------------------------
    if has_perm {
        // 9b. Verify permutation batch opening at z
        if let Some(ref perm_proof) = proof.perm_opening_proof {
            let beta_perm_bytes = transcript.challenge_bytes(b"beta_perm");
            let beta_perm = Scalar::from_challenge_bytes(&beta_perm_bytes, curve);

            let perm_commitment_bytes: Vec<&[u8]> = proof.perm_commitments.iter()
                .map(|c| c.0.as_slice())
                .collect();
            let perm_evals: Vec<Scalar> = proof.perm_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            if !scheme.batch_verify_at_point(
                &perm_commitment_bytes,
                &perm_evals,
                &z,
                &beta_perm,
                &perm_proof.proof,
            ) {
                return false;
            }
        } else {
            return false; // Perm proof required but missing
        }

        // 9c. Verify permutation shifted opening at ω·z
        if let Some(ref perm_shifted_proof) = proof.perm_shifted_opening_proof {
            let beta_perm_shifted_bytes = transcript.challenge_bytes(b"beta_perm_shifted");
            let beta_perm_shifted = Scalar::from_challenge_bytes(&beta_perm_shifted_bytes, curve);

            let omega = scheme.domain_generator(proof.domain_size);
            let omega_z = omega.mul(&z);

            let perm_commitment_bytes: Vec<&[u8]> = proof.perm_commitments.iter()
                .map(|c| c.0.as_slice())
                .collect();
            let perm_shifted_evals: Vec<Scalar> = proof.perm_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            if !scheme.batch_verify_at_point(
                &perm_commitment_bytes,
                &perm_shifted_evals,
                &omega_z,
                &beta_perm_shifted,
                &perm_shifted_proof.proof,
            ) {
                return false;
            }
        } else {
            return false; // Perm shifted proof required but missing
        }
    }

    // -----------------------------------------------------------------------
    // 10. Verify register permutation opening proofs (if register ports exist)
    // -----------------------------------------------------------------------
    if has_reg_perm {
        // 10b. Verify register permutation batch opening at z
        if let Some(ref rp_proof) = proof.reg_perm_opening_proof {
            let beta_rp_bytes = transcript.challenge_bytes(b"beta_reg_perm");
            let beta_rp = Scalar::from_challenge_bytes(&beta_rp_bytes, curve);

            let rp_commitment_bytes: Vec<&[u8]> = proof.reg_perm_commitments.iter()
                .map(|c| c.0.as_slice())
                .collect();
            let rp_evals: Vec<Scalar> = proof.reg_perm_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            if !scheme.batch_verify_at_point(
                &rp_commitment_bytes,
                &rp_evals,
                &z,
                &beta_rp,
                &rp_proof.proof,
            ) {
                return false;
            }
        } else {
            return false;
        }

        // 10c. Verify register permutation shifted opening at ω·z
        if let Some(ref rps_proof) = proof.reg_perm_shifted_opening_proof {
            let beta_rps_bytes = transcript.challenge_bytes(b"beta_reg_perm_shifted");
            let beta_rps = Scalar::from_challenge_bytes(&beta_rps_bytes, curve);

            let omega = scheme.domain_generator(proof.domain_size);
            let omega_z = omega.mul(&z);

            let rp_commitment_bytes: Vec<&[u8]> = proof.reg_perm_commitments.iter()
                .map(|c| c.0.as_slice())
                .collect();
            let rps_evals: Vec<Scalar> = proof.reg_perm_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            if !scheme.batch_verify_at_point(
                &rp_commitment_bytes,
                &rps_evals,
                &omega_z,
                &beta_rps,
                &rps_proof.proof,
            ) {
                return false;
            }
        } else {
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // 11. Verify frame-stack permutation openings.
    // -----------------------------------------------------------------------
    if has_frame_perm {
        let fp_commit_bytes = vec![
            proof.frame_perm_commitment.as_ref().unwrap().0.as_slice(),
        ];

        // Z(z)
        if let Some(ref fp_proof) = proof.frame_perm_opening_proof {
            let _ = transcript.challenge_bytes(b"beta_frame_perm");
            let z_at_z = Scalar::from_bytes(
                proof.frame_perm_evaluation.as_ref().unwrap(), curve);
            if !scheme.batch_verify_at_point(
                &fp_commit_bytes,
                std::slice::from_ref(&z_at_z),
                &z,
                &Scalar::one(curve),
                &fp_proof.proof,
            ) {
                return false;
            }
        } else {
            return false;
        }

        // Z(ω·z)
        if let Some(ref fp_proof_s) = proof.frame_perm_shifted_opening_proof {
            let _ = transcript.challenge_bytes(b"beta_frame_perm_shifted");
            let z_at_omega = Scalar::from_bytes(
                proof.frame_perm_shifted_evaluation.as_ref().unwrap(), curve);
            let omega = scheme.domain_generator(proof.domain_size);
            let omega_z = omega.mul(&z);
            if !scheme.batch_verify_at_point(
                &fp_commit_bytes,
                std::slice::from_ref(&z_at_omega),
                &omega_z,
                &Scalar::one(curve),
                &fp_proof_s.proof,
            ) {
                return false;
            }
        } else {
            return false;
        }

    }

    true
}

/// Recompute the constraint identity polynomial value `C(z)` from a proof's
/// evaluations, mirroring the reconstruction inside `verify_inner_scheme`.
///
/// This helper supports the recursive accumulator's per-chunk
/// constraint-identity scalar `c_check = Q(z)·Z_H(z) - C(z)`. It covers
/// every auxiliary AIR contribution: main + shifted + LogUp + bitwise +
/// memory permutation + register permutation + frame-stack permutation.
///
/// All Fiat-Shamir-derived auxiliary challenges must be supplied by the
/// caller — they're recovered via `recover_full_chunk_challenges_scheme`
/// in the recursive path, or via the inline transcript replay inside
/// `verify_inner_scheme`. Required-vs-optional matches the proof's
/// declared aux structure: e.g. `logup_gamma` must be `Some(...)` whenever
/// `proof.logup_commitments` is non-empty.
#[allow(unused_assignments)]
pub fn compute_c_at_z(
    proof: &ExecutionProof,
    constraints: &dyn VmConstraintSystem,
    scheme: &dyn crate::scheme::CommitmentScheme,
    curve: crate::field::CurveType,
    alpha: &crate::field::Scalar,
    z: &crate::field::Scalar,
    logup_gamma: Option<&crate::field::Scalar>,
    bitwise_gamma: Option<&crate::field::Scalar>,
    bitwise_delta: Option<&crate::field::Scalar>,
    perm_gamma: Option<&crate::field::Scalar>,
    perm_delta: Option<&crate::field::Scalar>,
    reg_perm_gamma: Option<&crate::field::Scalar>,
    reg_perm_delta: Option<&crate::field::Scalar>,
    frame_perm_gamma: Option<&crate::field::Scalar>,
    frame_perm_delta: Option<&crate::field::Scalar>,
) -> crate::field::Scalar {
    use crate::field::Scalar;

    let num_columns = proof.column_commitments.len();
    let col_evals: Vec<Scalar> = proof.evaluations[..num_columns]
        .iter()
        .map(|e| Scalar::from_bytes(e, curve))
        .collect();

    let has_logup = !proof.logup_commitments.is_empty();
    let has_bitwise = !proof.bitwise_commitments.is_empty();
    let has_perm = !proof.perm_commitments.is_empty();
    let has_reg_perm = !proof.reg_perm_commitments.is_empty();
    let has_frame_perm = proof.frame_perm_commitment.is_some();

    let mut c_at_z = constraints.evaluate_at_point(&col_evals, alpha);
    let mut alpha_offset = constraints.num_constraints();

    let shifted_indices = constraints.shifted_column_indices();
    let omega = if !shifted_indices.is_empty()
        || has_logup
        || has_bitwise
        || has_perm
        || has_reg_perm
    {
        Some(scheme.domain_generator(proof.domain_size))
    } else {
        None
    };
    let omega_n_minus_1 = if let Some(ref om) = omega {
        let n_minus_1 = proof.domain_size - 1;
        let mut onm1 = Scalar::one(curve);
        let mut base_o = om.clone();
        let mut exp_o = n_minus_1;
        while exp_o > 0 {
            if exp_o & 1 == 1 {
                onm1 = onm1.mul(&base_o);
            }
            base_o = base_o.mul(&base_o);
            exp_o >>= 1;
        }
        Some(onm1)
    } else {
        None
    };

    if !shifted_indices.is_empty() && !proof.shifted_evaluations.is_empty() {
        let shifted_evals: Vec<Scalar> = proof.shifted_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();
        let c_shifted = constraints.evaluate_shifted_at_point(
            &col_evals, &shifted_evals, z, omega_n_minus_1.as_ref().unwrap(),
            alpha, alpha_offset,
        );
        c_at_z = c_at_z.add(&c_shifted);
        alpha_offset += constraints.num_shifted_constraints();
    }

    // Add LogUp constraints to C(z): decomposition + boundary + inverse +
    // table-inverse + running-sum transition (extended Phase-0 layout).
    if has_logup {
        let lookup_reqs = constraints.lookup_declarations();
        let logup_groups = crate::lookup::group_declarations(&lookup_reqs);
        let logup_layout = crate::lookup::extended_logup_column_layout(&logup_groups);

        let logup_evals: Vec<Scalar> = proof.logup_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // Decomposition: sel(z) · (value(z) - Σ limb_k(z)·256^k) = 0
        for (g_idx, group) in logup_groups.iter().enumerate() {
            let (start_offset, num_limbs) = logup_layout.limb_offsets[g_idx];
            let limb_evals: Vec<Scalar> = (0..num_limbs)
                .map(|l| logup_evals[start_offset + l].clone())
                .collect();
            let value_at_z = &col_evals[group.column_index];
            let selector_at_z = if !group.selectors.is_empty() {
                let mut sel_sum = Scalar::zero(curve);
                for &sel_idx in &group.selectors {
                    sel_sum = sel_sum.add(&col_evals[sel_idx]);
                }
                Some(sel_sum)
            } else {
                None
            };
            let decomp = crate::lookup::evaluate_decomposition_at_point(
                value_at_z, &limb_evals, selector_at_z.as_ref(), curve,
            );
            c_at_z = c_at_z.add(&ap.mul(&decomp));
            ap = ap.mul(alpha);
        }
        alpha_offset += logup_groups.len();

        let h_idx = logup_layout.h_column;
        let m_idx = logup_layout.m_column;
        let t_idx = logup_layout.t_column;
        let u_t_idx = logup_layout.u_t_column;

        // Boundary: L_0(z) · h(z) = 0
        if h_idx < logup_evals.len() {
            let h_at_z = &logup_evals[h_idx];
            let n_scalar = Scalar::from_u64(proof.domain_size, curve);
            let z_minus_1 = z.sub(&Scalar::one(curve));
            let z_n = {
                let mut zn = Scalar::one(curve);
                let mut base = z.clone();
                let mut exp = proof.domain_size;
                while exp > 0 {
                    if exp & 1 == 1 { zn = zn.mul(&base); }
                    base = base.mul(&base);
                    exp >>= 1;
                }
                zn
            };
            let z_n_minus_1 = z_n.sub(&Scalar::one(curve));
            let denom_l0 = n_scalar.mul(&z_minus_1);
            let l0_z = if !denom_l0.is_zero() {
                z_n_minus_1.mul(&denom_l0.inverse())
            } else {
                Scalar::one(curve)
            };
            let boundary = l0_z.mul(h_at_z);
            c_at_z = c_at_z.add(&ap.mul(&boundary));
            ap = ap.mul(alpha);
            alpha_offset += 1;
        }

        let gamma = logup_gamma
            .expect("logup gamma must be set when has_logup");

        // (A) Per-limb inverse: f_k(z) · (γ - ℓ_k(z)) - 1 = 0
        for (g_idx, _group) in logup_groups.iter().enumerate() {
            let (limb_start, num_limbs) = logup_layout.limb_offsets[g_idx];
            let (f_start, _) = logup_layout.f_offsets[g_idx];
            for l in 0..num_limbs {
                let limb_at_z = &logup_evals[limb_start + l];
                let f_at_z = &logup_evals[f_start + l];
                let body = crate::lookup::evaluate_inverse_at_point(
                    f_at_z, limb_at_z, gamma, curve,
                );
                c_at_z = c_at_z.add(&ap.mul(&body));
                ap = ap.mul(alpha);
                alpha_offset += 1;
            }
        }

        // (B) Table inverse: u_t(z) · (γ - t(z)) - 1 = 0
        if u_t_idx < logup_evals.len() && t_idx < logup_evals.len() {
            let t_at_z = &logup_evals[t_idx];
            let u_t_at_z = &logup_evals[u_t_idx];
            let body = crate::lookup::evaluate_inverse_at_point(
                u_t_at_z, t_at_z, gamma, curve,
            );
            c_at_z = c_at_z.add(&ap.mul(&body));
            ap = ap.mul(alpha);
            alpha_offset += 1;
        }

        // (C) Running-sum transition (cyclic):
        // h(ω·z) - h(z) - Σ_{g,k} active_g(z)·f_{g,k}(z) + m(z)·u_t(z) = 0
        if h_idx < logup_evals.len()
            && m_idx < logup_evals.len()
            && u_t_idx < logup_evals.len()
        {
            let h_shifted_evals: Vec<Scalar> = proof.logup_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();
            if !h_shifted_evals.is_empty() {
                let h_at_z = logup_evals[h_idx].clone();
                let h_at_omega_z = h_shifted_evals[0].clone();
                let m_at_z = logup_evals[m_idx].clone();
                let u_t_at_z = logup_evals[u_t_idx].clone();
                let mut active_and_fs: Vec<(Scalar, Vec<Scalar>)> = Vec::new();
                for (g_idx, group) in logup_groups.iter().enumerate() {
                    let (f_start, num_limbs) = logup_layout.f_offsets[g_idx];
                    let active_at_z = if group.selectors.is_empty() {
                        Scalar::one(curve)
                    } else {
                        let mut s = Scalar::zero(curve);
                        for &sel_idx in &group.selectors {
                            s = s.add(&col_evals[sel_idx]);
                        }
                        s
                    };
                    let fs: Vec<Scalar> = (0..num_limbs)
                        .map(|l| logup_evals[f_start + l].clone())
                        .collect();
                    active_and_fs.push((active_at_z, fs));
                }
                let body = crate::lookup::evaluate_transition_at_point(
                    &h_at_z, &h_at_omega_z, &active_and_fs, &m_at_z, &u_t_at_z, curve,
                );
                c_at_z = c_at_z.add(&ap.mul(&body));
                ap = ap.mul(alpha);
                alpha_offset += 1;
            }
        }
    }

    // Add bitwise LogUp constraints to C(z)
    if has_bitwise {
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        let bitwise_decls_vec = constraints.bitwise_lookup_declarations();
        let bitwise_groups = crate::lookup::group_bitwise_declarations(&bitwise_decls_vec);
        let bw_layout = crate::lookup::extended_bitwise_column_layout(&bitwise_groups);
        let bw_evals: Vec<Scalar> = proof.bitwise_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();

        let bw_gamma = bitwise_gamma
            .expect("bitwise gamma must be set when has_bitwise");
        let bw_delta = bitwise_delta
            .expect("bitwise delta must be set when has_bitwise");
        let bw_delta_sq = bw_delta.mul(bw_delta);
        let sixteen = Scalar::from_u64(16, curve);

        // (BW-A) Per-group: nibble decomp of A, nibble decomp of B, result derivation.
        for (g_idx, group) in bitwise_groups.iter().enumerate() {
            let (nib_start, num_nibs) = bw_layout.nibble_offsets[g_idx];
            let a_start = nib_start;
            let b_start = nib_start + num_nibs;
            let c_start = nib_start + 2 * num_nibs;

            let sel_at_z: Scalar = if group.selectors.is_empty() {
                Scalar::one(curve)
            } else {
                let mut s = Scalar::zero(curve);
                for &sel_idx in &group.selectors {
                    s = s.add(&col_evals[sel_idx]);
                }
                s
            };

            let recompose_at_z = |start: usize| -> Scalar {
                let mut acc = Scalar::zero(curve);
                let mut power = Scalar::one(curve);
                for k in 0..num_nibs {
                    acc = acc.add(&bw_evals[start + k].mul(&power));
                    power = power.mul(&sixteen);
                }
                acc
            };
            let a_recomp = recompose_at_z(a_start);
            let b_recomp = recompose_at_z(b_start);
            let c_recomp = recompose_at_z(c_start);

            // sel · (operand_a(z) - a_recomp) = 0
            let a_diff = col_evals[group.operand_a_column].sub(&a_recomp);
            c_at_z = c_at_z.add(&ap.mul(&sel_at_z).mul(&a_diff));
            ap = ap.mul(alpha);
            alpha_offset += 1;

            // sel · (operand_b(z) - b_recomp) = 0
            let b_diff = col_evals[group.operand_b_column].sub(&b_recomp);
            c_at_z = c_at_z.add(&ap.mul(&sel_at_z).mul(&b_diff));
            ap = ap.mul(alpha);
            alpha_offset += 1;

            // Result derivation based on op
            let expected_at_z: Scalar = match group.op {
                crate::lookup::BitwiseOp::And => c_recomp.clone(),
                crate::lookup::BitwiseOp::Or => col_evals[group.operand_a_column]
                    .add(&col_evals[group.operand_b_column])
                    .sub(&c_recomp),
                crate::lookup::BitwiseOp::Xor => {
                    let two = Scalar::from_u64(2, curve);
                    col_evals[group.operand_a_column]
                        .add(&col_evals[group.operand_b_column])
                        .sub(&two.mul(&c_recomp))
                }
            };
            let r_diff = col_evals[group.result_column].sub(&expected_at_z);
            c_at_z = c_at_z.add(&ap.mul(&sel_at_z).mul(&r_diff));
            ap = ap.mul(alpha);
            alpha_offset += 1;
        }

        // (BW-B) Per-nibble inverse: f_k · (γ - (a + δ·b + δ²·c)) - 1 = 0
        for (g_idx, _group) in bitwise_groups.iter().enumerate() {
            let (nib_start, num_nibs) = bw_layout.nibble_offsets[g_idx];
            let (f_start, _) = bw_layout.f_offsets[g_idx];
            for k in 0..num_nibs {
                let a = &bw_evals[nib_start + k];
                let b = &bw_evals[nib_start + num_nibs + k];
                let c = &bw_evals[nib_start + 2 * num_nibs + k];
                let f = &bw_evals[f_start + k];
                let q = a.add(&bw_delta.mul(b)).add(&bw_delta_sq.mul(c));
                let prod = f.mul(&bw_gamma.sub(&q));
                let body = prod.sub(&Scalar::one(curve));
                c_at_z = c_at_z.add(&ap.mul(&body));
                ap = ap.mul(alpha);
                alpha_offset += 1;
            }
        }

        // (BW-C) Table inverse
        let t_a_at_z = bw_evals[bw_layout.t_a_column].clone();
        let t_b_at_z = bw_evals[bw_layout.t_b_column].clone();
        let t_c_at_z = bw_evals[bw_layout.t_c_column].clone();
        let u_t_at_z = bw_evals[bw_layout.u_t_column].clone();
        {
            let combined = t_a_at_z
                .add(&bw_delta.mul(&t_b_at_z))
                .add(&bw_delta_sq.mul(&t_c_at_z));
            let prod = u_t_at_z.mul(&bw_gamma.sub(&combined));
            let body = prod.sub(&Scalar::one(curve));
            c_at_z = c_at_z.add(&ap.mul(&body));
            ap = ap.mul(alpha);
            alpha_offset += 1;
        }

        // (BW-D) Transition: h(ωz) - h(z) - Σ active·Σf + m·u_t = 0
        let h_bw_at_z = bw_evals[bw_layout.h_column].clone();
        let m_bw_at_z = bw_evals[bw_layout.m_column].clone();
        let bw_shifted_evals: Vec<Scalar> = proof.bitwise_shifted_evaluations.iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();
        if !bw_shifted_evals.is_empty() {
            let h_bw_at_omega_z = bw_shifted_evals[0].clone();
            let mut lhs = h_bw_at_omega_z.sub(&h_bw_at_z);
            for (g_idx, group) in bitwise_groups.iter().enumerate() {
                let (f_start, num_nibs) = bw_layout.f_offsets[g_idx];
                let active = if group.selectors.is_empty() {
                    Scalar::one(curve)
                } else {
                    let mut s = Scalar::zero(curve);
                    for &sel_idx in &group.selectors {
                        s = s.add(&col_evals[sel_idx]);
                    }
                    s
                };
                let mut fsum = Scalar::zero(curve);
                for k in 0..num_nibs {
                    fsum = fsum.add(&bw_evals[f_start + k]);
                }
                lhs = lhs.sub(&active.mul(&fsum));
            }
            lhs = lhs.add(&m_bw_at_z.mul(&u_t_at_z));
            c_at_z = c_at_z.add(&ap.mul(&lhs));
            ap = ap.mul(alpha);
            alpha_offset += 1;
        }

        // (BW-E) Boundary: L_0(z) · h(z) = 0
        {
            let n_scalar = Scalar::from_u64(proof.domain_size, curve);
            let z_minus_1 = z.sub(&Scalar::one(curve));
            let z_n = {
                let mut zn = Scalar::one(curve);
                let mut base = z.clone();
                let mut exp = proof.domain_size;
                while exp > 0 {
                    if exp & 1 == 1 { zn = zn.mul(&base); }
                    base = base.mul(&base);
                    exp >>= 1;
                }
                zn
            };
            let z_n_minus_1 = z_n.sub(&Scalar::one(curve));
            let denom_l0 = n_scalar.mul(&z_minus_1);
            let l0_z = if !denom_l0.is_zero() {
                z_n_minus_1.mul(&denom_l0.inverse())
            } else {
                Scalar::one(curve)
            };
            let body = l0_z.mul(&h_bw_at_z);
            c_at_z = c_at_z.add(&ap.mul(&body));
            ap = ap.mul(alpha);
            alpha_offset += 1;
        }
    }

    // Add memory permutation constraints to C(z)
    if has_perm {
        let mem_cols = constraints.memory_columns();
        if let Some((addr_col, val_cols, load_sels, store_sels)) = &mem_cols {
            let perm_evals: Vec<Scalar> = proof.perm_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();
            let perm_shifted_evals: Vec<Scalar> = proof.perm_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            let layout = crate::permutation::MemoryPermutationLayout::new(val_cols.len());

            // Sorted consistency constraints (is_same_addr binary, addr continuity, read consistency)
            let perm_c = crate::permutation::evaluate_permutation_constraints_at_point(
                &perm_evals,
                &perm_shifted_evals,
                &layout,
                z,
                omega_n_minus_1.as_ref().unwrap(),
                alpha,
                alpha_offset,
            );
            c_at_z = c_at_z.add(&perm_c);

            // Grand product transition + boundary constraints
            let gp_offset = alpha_offset + 3 + layout.num_val_columns;
            let gp_c = crate::permutation::evaluate_grand_product_at_point(
                &perm_evals,
                &perm_shifted_evals,
                &col_evals,
                &layout,
                *addr_col,
                val_cols,
                load_sels,
                store_sels,
                perm_gamma.expect("perm_gamma must be set when has_perm"),
                perm_delta.expect("perm_delta must be set when has_perm"),
                z,
                proof.domain_size,
                alpha,
                gp_offset,
            );
            c_at_z = c_at_z.add(&gp_c);

            alpha_offset += crate::permutation::num_permutation_constraints(&layout);
        }
    }

    // Add register file permutation constraints to C(z)
    if has_reg_perm {
        let reg_ports = constraints.register_ports();
        if !reg_ports.is_empty() {
            let reg_evals: Vec<Scalar> = proof.reg_perm_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();
            let reg_shifted_evals: Vec<Scalar> = proof.reg_perm_shifted_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            let layout = crate::permutation::RegisterPermutationLayout::new(reg_ports.len());

            let reg_c = crate::permutation::evaluate_register_perm_at_point(
                &reg_evals,
                &reg_shifted_evals,
                &col_evals,
                &layout,
                &reg_ports,
                reg_perm_gamma.expect("reg_perm_gamma must be set when has_reg_perm"),
                reg_perm_delta.expect("reg_perm_delta must be set when has_reg_perm"),
                z,
                omega_n_minus_1.as_ref().unwrap(),
                proof.domain_size,
                alpha,
                alpha_offset,
            );
            c_at_z = c_at_z.add(&reg_c);

            alpha_offset += crate::permutation::num_register_perm_constraints(&layout);
        }
    }

    // Frame-stack permutation contribution to C(z).
    if has_frame_perm {
        if let Some(fp_layout) = constraints.frame_perm_layout() {
            let z_at_z = Scalar::from_bytes(
                proof.frame_perm_evaluation.as_ref().unwrap(), curve);
            let z_at_omega_z = Scalar::from_bytes(
                proof.frame_perm_shifted_evaluation.as_ref().unwrap(), curve);

            // Tuple at z (from main col_evals) and at ω·z (from
            // proof.shifted_evaluations using shifted_indices order).
            let tuple_at_z: Vec<Scalar> = fp_layout.tuple_columns.iter()
                .map(|&c| col_evals[c].clone())
                .collect();
            let shifted_indices_v = constraints.shifted_column_indices();
            let mut tuple_at_omega_z: Vec<Scalar> =
                Vec::with_capacity(fp_layout.tuple_columns.len());
            for &c in &fp_layout.tuple_columns {
                let pos = shifted_indices_v.iter().position(|&x| x == c)
                    .expect("frame-perm tuple column must be in shifted_column_indices");
                let bytes = &proof.shifted_evaluations[pos];
                tuple_at_omega_z.push(Scalar::from_bytes(bytes, curve));
            }

            // is_push(z) = Σ push_sel_k(z); is_pop(z) = Σ pop_sel_k(z).
            let mut is_push_at_z = Scalar::zero(curve);
            for &s in &fp_layout.push_selectors {
                is_push_at_z = is_push_at_z.add(&col_evals[s]);
            }
            let mut is_pop_at_z = Scalar::zero(curve);
            for &s in &fp_layout.pop_selectors {
                is_pop_at_z = is_pop_at_z.add(&col_evals[s]);
            }

            let fp_c = crate::permutation::evaluate_frame_perm_at_point(
                &z_at_z,
                &z_at_omega_z,
                &tuple_at_z,
                &tuple_at_omega_z,
                &is_push_at_z,
                &is_pop_at_z,
                frame_perm_gamma.expect("frame_perm_gamma must be set when has_frame_perm"),
                frame_perm_delta.expect("frame_perm_delta must be set when has_frame_perm"),
                z,
                proof.domain_size,
                alpha,
                alpha_offset,
            );
            c_at_z = c_at_z.add(&fp_c);

            alpha_offset += crate::permutation::FrameStackPermLayout::NUM_CONSTRAINTS;
        }
    }

    let _ = alpha_offset;
    c_at_z
}
