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
    use crate::field::{Scalar, CurveType};

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
    let has_selectors = !constraints.selector_column_indices().is_empty();

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
            let omega_scalar = Scalar::Bls48581(omega_big.clone());
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
    let has_selectors = !constraints.selector_column_indices().is_empty();

    if has_selectors {
        let col_evals: Vec<Scalar> = proof.evaluations[..num_columns]
            .iter()
            .map(|e| Scalar::from_bytes(e, curve))
            .collect();

        let mut c_at_z = constraints.evaluate_at_point(&col_evals, &alpha);

        // Track total alpha offset for LogUp/permutation constraints
        let mut alpha_offset = constraints.num_constraints();

        // Add cross-row constraint contribution if shifted columns exist
        let shifted_indices = constraints.shifted_column_indices();
        let omega = if !shifted_indices.is_empty() || has_logup || has_perm || has_reg_perm {
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
                &col_evals, &shifted_evals, &z, omega_n_minus_1.as_ref().unwrap(),
                &alpha, alpha_offset,
            );
            c_at_z = c_at_z.add(&c_shifted);
            alpha_offset += constraints.num_shifted_constraints();
        }

        // Add LogUp decomposition constraints to C(z)
        if has_logup {
            let lookup_reqs = constraints.lookup_declarations();
            let logup_groups = crate::lookup::group_declarations(&lookup_reqs);
            let logup_layout = crate::lookup::logup_column_layout(&logup_groups);

            let logup_evals: Vec<Scalar> = proof.logup_evaluations.iter()
                .map(|e| Scalar::from_bytes(e, curve))
                .collect();

            // Compute alpha^alpha_offset
            let mut ap = Scalar::one(curve);
            for _ in 0..alpha_offset {
                ap = ap.mul(&alpha);
            }

            // Add decomposition constraint for each group:
            // alpha^k * selector(z) * (value(z) - Σ limb_i(z) * 256^i) = 0
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
                    value_at_z,
                    &limb_evals,
                    selector_at_z.as_ref(),
                    curve,
                );
                c_at_z = c_at_z.add(&ap.mul(&decomp));
                ap = ap.mul(&alpha);
            }

            alpha_offset += logup_groups.len();

            // Running sum boundary: L_0(z) · h(z) = 0 (ensures h(ω^0) = 0)
            let h_idx = logup_layout.h_column;
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
                ap = ap.mul(&alpha);
                alpha_offset += 1;
            }

            // Running sum transition: h(ω·z) - h(z) - row_contribution(z) = 0
            // requires inverse auxiliary columns for full polynomial enforcement.
            // The boundary check above proves h(ω^0) = 0; combined with
            // decomposition constraints this provides partial soundness.
        }

        // Add memory permutation constraints to C(z)
        if has_perm {
            let mem_cols = constraints.memory_columns();
            if let Some((addr_col, val_cols, _load_sels, store_sels)) = &mem_cols {
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
                    &z,
                    omega_n_minus_1.as_ref().unwrap(),
                    &alpha,
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
                    store_sels,
                    perm_gamma.as_ref().unwrap(),
                    perm_delta.as_ref().unwrap(),
                    &z,
                    proof.domain_size,
                    &alpha,
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
                    reg_perm_gamma.as_ref().unwrap(),
                    reg_perm_delta.as_ref().unwrap(),
                    &z,
                    omega_n_minus_1.as_ref().unwrap(),
                    proof.domain_size,
                    &alpha,
                    alpha_offset,
                );
                c_at_z = c_at_z.add(&reg_c);

                alpha_offset += crate::permutation::num_register_perm_constraints(&layout);
            }
        }

        let _ = alpha_offset;

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

    true
}
