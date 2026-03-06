//! Execution proof generation.
//!
//! This module implements the prover side of the MetaVM execution proof
//! protocol. Given an execution trace (as [`TracePolynomials`]) and a
//! VM-specific constraint system (via [`VmConstraintSystem`]), it produces
//! an [`ExecutionProof`] that a verifier can check without re-executing.
//!
//! The prover builds the combined constraint polynomial C(x) at full algebraic
//! degree using `build_constraint_polynomial`, then divides by Z(x) = x^n - 1
//! to get the quotient Q(x). If deg(Q) > n-1, Q is split into chunks.
//! The verifier recomputes C(z) from column evaluations — no constraint
//! commitment is included in the proof.

use crate::commitment::{self, Commitment, BatchProof};
use crate::field::{Scalar, CurveType};
use crate::trace::TracePolynomials;
use crate::vm_constraints::VmConstraintSystem;
use bls48581::bls48581::big;
use bls48581::bls48581::rom;
use metavm_core::transcript::Transcript;
use rayon::prelude::*;

/// A proof of correct execution.
#[derive(Clone, Debug)]
pub struct ExecutionProof {
    /// Commitments to each trace column polynomial.
    pub column_commitments: Vec<Commitment>,
    /// Commitments to the quotient polynomial chunk(s).
    /// Q(x) = Q_0(x) + x^n * Q_1(x) when deg(Q) > n-1.
    pub quotient_commitments: Vec<Commitment>,
    /// Evaluations of all committed polynomials at the challenge point `z`.
    /// Layout: [col_0(z), ..., col_k(z), Q_0(z), Q_1(z), ...]
    /// Each entry is a serialized scalar.
    pub evaluations: Vec<Vec<u8>>,
    /// Batch opening proof covering all committed polynomials at `z`.
    pub opening_proof: BatchProof,
    /// Number of actual execution steps in the trace (before padding).
    pub num_steps: u64,
    /// Padded domain size (power of 2) used for polynomial commitments.
    pub domain_size: u64,
    /// Number of quotient polynomial chunks (1 or 2).
    pub num_quotient_chunks: u8,
    /// Evaluations of shifted columns at ω·z (for cross-row constraints).
    /// Layout: [col_{s0}(ω·z), col_{s1}(ω·z), ...] matching shifted_column_indices().
    /// Empty if no cross-row constraints.
    pub shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for shifted columns at ω·z.
    /// None if no cross-row constraints.
    pub shifted_opening_proof: Option<BatchProof>,
    // ── LogUp range check proof ────────────────────────────────────────
    /// Commitments to LogUp auxiliary columns (byte limbs + h + m).
    /// Empty if no lookup declarations.
    pub logup_commitments: Vec<Commitment>,
    /// Evaluations of LogUp auxiliary columns at z.
    /// Empty if no lookup declarations.
    pub logup_evaluations: Vec<Vec<u8>>,
    /// Evaluations of LogUp shifted columns (h) at ω·z.
    /// Empty if no lookup declarations.
    pub logup_shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for LogUp columns at z.
    pub logup_opening_proof: Option<BatchProof>,
    /// Batch opening proof for LogUp shifted columns at ω·z.
    pub logup_shifted_opening_proof: Option<BatchProof>,
    // ── Memory permutation proof ───────────────────────────────────────
    /// Commitments to permutation auxiliary columns (sorted, Z, is_same_addr, inv_addr_diff).
    /// Empty if no memory permutation.
    pub perm_commitments: Vec<Commitment>,
    /// Evaluations of permutation columns at z.
    pub perm_evaluations: Vec<Vec<u8>>,
    /// Evaluations of permutation shifted columns at ω·z.
    pub perm_shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for permutation columns at z.
    pub perm_opening_proof: Option<BatchProof>,
    /// Batch opening proof for permutation shifted columns at ω·z.
    pub perm_shifted_opening_proof: Option<BatchProof>,
    // ── Oracle public-input data ─────────────────────────────────────
    /// Serialized oracle operation entries for external verification.
    ///
    /// Each entry contains: [row_index (8 bytes LE), selector_col (8 bytes LE),
    /// data_col_0 (8 bytes LE), ..., data_col_k (8 bytes LE)] for rows where
    /// an oracle selector is active. Absorbed into the Fiat-Shamir transcript
    /// to bind the proof to specific oracle values.
    pub oracle_data: Vec<Vec<u8>>,
    // ── Register file permutation proof ──────────────────────────────
    /// Commitments to register permutation auxiliary columns.
    /// Empty if no register ports declared.
    pub reg_perm_commitments: Vec<Commitment>,
    /// Evaluations of register permutation columns at z.
    pub reg_perm_evaluations: Vec<Vec<u8>>,
    /// Evaluations of register permutation shifted columns at ω·z.
    pub reg_perm_shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for register permutation columns at z.
    pub reg_perm_opening_proof: Option<BatchProof>,
    /// Batch opening proof for register permutation shifted columns at ω·z.
    pub reg_perm_shifted_opening_proof: Option<BatchProof>,
}

/// A chunk proof wrapping an execution proof with state chain metadata.
#[derive(Clone, Debug)]
pub struct ChunkProof {
    /// The underlying execution proof for this chunk's trace.
    pub execution_proof: ExecutionProof,
    /// SHA3-256 hash of the VM state at the beginning of this chunk.
    pub initial_state_hash: [u8; 32],
    /// SHA3-256 hash of the VM state at the end of this chunk.
    pub final_state_hash: [u8; 32],
    /// Sequential chunk index (0-based).
    pub chunk_index: u64,
}

/// Convert 32 Fiat-Shamir challenge bytes to a BLS48-581 BIG field element.
pub fn challenge_to_big(challenge_bytes: &[u8; 32]) -> big::BIG {
    let mut z_padded = [0u8; big::MODBYTES];
    z_padded[big::MODBYTES - 32..].copy_from_slice(challenge_bytes);
    let mut z = big::BIG::frombytes(&z_padded);
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    z.rmod(&modulus);
    z
}

/// Convert 32 Fiat-Shamir challenge bytes to a Scalar for a given curve.
pub fn challenge_to_scalar(challenge_bytes: &[u8; 32], curve: CurveType) -> Scalar {
    Scalar::from_challenge_bytes(challenge_bytes, curve)
}

/// Serialize a BIG value to MODBYTES for inclusion in proof evaluations.
fn big_to_eval_bytes(val: &big::BIG) -> Vec<u8> {
    let mut buf = vec![0u8; big::MODBYTES];
    val.tobytes(&mut buf);
    buf
}

/// Core proving logic shared by `prove()` and `prove_chunk()`.
///
/// Currently only supports BLS48-581 for commitment operations.
fn prove_inner(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
) -> ExecutionProof {
    let num_steps = trace.num_steps();
    let domain_size = trace.domain_size();
    let curve = trace.curve;
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);

    assert_eq!(curve, CurveType::Bls48581, "Commitment currently only supports BLS48-581");

    // Get columns as BIG vectors for commitment operations
    let mut big_columns = trace.columns_as_bls48581();

    // Fix selector padding: set the designated "no-op" selector to 1 on
    // padding rows so that the sum-to-one constraint is satisfied everywhere.
    if let Some(padding_col) = constraints.padding_selector_column() {
        let one = big::BIG::new_int(1);
        let num_rows = trace.num_rows;
        let padded = domain_size as usize;
        if padding_col < big_columns.len() {
            for i in num_rows..padded {
                big_columns[padding_col][i] = big::BIG::new_copy(&one);
            }
        }
    }

    // Fix trace padding for cross-row constraint satisfaction (PC continuity).
    {
        let num_rows = trace.num_rows;
        let padded = domain_size as usize;
        let mut scalar_columns: Vec<Vec<Scalar>> = big_columns.iter()
            .map(|col| col.iter().map(|b| Scalar::Bls48581(big::BIG::new_copy(b))).collect())
            .collect();
        constraints.fix_trace_padding(&mut scalar_columns, num_rows, padded);
        // Copy back to big_columns
        for (col_idx, scalar_col) in scalar_columns.iter().enumerate() {
            for (row_idx, s) in scalar_col.iter().enumerate() {
                big_columns[col_idx][row_idx] = big::BIG::new_copy(s.as_bls48581());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 1: Commit to each trace column
    // -----------------------------------------------------------------------
    let mut column_commitments: Vec<Commitment> = Vec::with_capacity(big_columns.len());

    for col in &big_columns {
        let comm = commitment::commit_from_scalars(col, domain_size);
        column_commitments.push(comm);
    }

    // -----------------------------------------------------------------------
    // Step 2: Absorb commitments into Fiat-Shamir transcript
    // -----------------------------------------------------------------------
    transcript.append_u64(b"num_steps", num_steps);
    transcript.append_u64(b"domain_size", domain_size);

    for comm in &column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    // -----------------------------------------------------------------------
    // Step 3: Draw constraint combination challenge alpha
    // -----------------------------------------------------------------------
    let alpha_bytes = transcript.challenge_bytes(b"alpha");
    let alpha_big = big::BIG::frombytes(&alpha_bytes);
    let alpha = Scalar::Bls48581(big::BIG::new_copy(&alpha_big));

    // -----------------------------------------------------------------------
    // Step 4: Build C(x) — either via build_constraint_polynomial (selector-based)
    //         or via evaluate_on_domain + IFFT (fallback)
    //
    // We compute column_coeffs_big here and cache it for reuse in Step 8,
    // avoiding a redundant IFFT pass over all columns.
    // -----------------------------------------------------------------------
    let c_coeffs: Vec<big::BIG>;
    let use_build_poly = !constraints.selector_column_indices().is_empty();

    // Compute column coefficients ONCE — reused in Step 8 for evaluations at z
    let column_coeffs_big: Vec<Vec<big::BIG>> = big_columns.iter()
        .map(|col| commitment::eval_to_coeff(col, domain_size))
        .collect();

    // Check for cross-row constraints
    let shifted_indices = constraints.shifted_column_indices();
    let has_shifts = !shifted_indices.is_empty();

    if use_build_poly {
        // Selector-based VMs: build C(x) algebraically in coefficient form
        let column_coeffs_scalar: Vec<Vec<Scalar>> = column_coeffs_big.iter()
            .map(|coeffs| {
                coeffs.iter().map(|b| Scalar::Bls48581(big::BIG::new_copy(b))).collect()
            })
            .collect();

        let mut c_coeffs_scalar = constraints.build_constraint_polynomial(
            &column_coeffs_scalar, &alpha, domain_size,
        );

        // Add cross-row constraint polynomial if any
        if has_shifts {
            use bls48581::bls;
            let s = bls::singleton();
            let omega_big = s.RootsOfUnityBLS48581[&domain_size][1].clone();
            let omega = Scalar::Bls48581(omega_big);
            let c_shifted = constraints.build_shifted_constraint_polynomial(
                &column_coeffs_scalar, &alpha, domain_size,
                &omega,
                constraints.num_constraints(),
            );
            c_coeffs_scalar = crate::poly_arith::poly_add(&c_coeffs_scalar, &c_shifted, curve);
        }

        c_coeffs = c_coeffs_scalar.iter()
            .map(|s| big::BIG::new_copy(s.as_bls48581()))
            .collect();
    } else {
        // Fallback: evaluate constraints on domain, combine with alpha, IFFT to coeffs
        let columns_refs: Vec<&Vec<Scalar>> = trace.columns().into_iter().collect();
        let constraint_evals = constraints.evaluate_on_domain(&columns_refs, trace.num_rows);
        let n = domain_size as usize;

        // Combine: C_evals[j] = Σ alpha^i * constraint_i_evals[j]
        let mut c_evals_big = vec![big::BIG::new(); n];
        let mut alpha_power = big::BIG::new_int(1);
        for constraint in &constraint_evals {
            for j in 0..n {
                let val = if j < constraint.len() {
                    big::BIG::new_copy(constraint[j].as_bls48581())
                } else {
                    big::BIG::new()
                };
                let term = big::BIG::modmul(&alpha_power, &val, &modulus);
                c_evals_big[j] = big::BIG::modadd(&c_evals_big[j], &term, &modulus);
            }
            alpha_power = big::BIG::modmul(&alpha_power, &alpha_big, &modulus);
        }

        c_coeffs = commitment::eval_to_coeff(&c_evals_big, domain_size);
    };

    // -----------------------------------------------------------------------
    // Step 5: Divide C(x) by Z(x) = x^n - 1 to get Q(x)
    // -----------------------------------------------------------------------
    let n = domain_size as usize;
    let c_len = c_coeffs.len();

    // Pad to at least n if needed
    let mut dividend = c_coeffs.clone();
    while dividend.len() < n {
        dividend.push(big::BIG::new());
    }

    // Q degree = c_deg - n. If c_deg < n, Q = 0 (constraint is trivially divisible).
    let c_deg = if c_len > 0 { c_len - 1 } else { 0 };
    let q_deg = if c_deg >= n { c_deg - n } else { 0 };

    let mut quotient_coeffs_all = vec![big::BIG::new(); if c_deg >= n { q_deg + 1 } else { n }];

    if c_deg >= n {
        // Proper polynomial division: C(x) / (x^n - 1)
        // For i from c_deg down to n: q[i-n] = dividend[i]; dividend[i-n] += dividend[i]
        let mut div = dividend.clone();
        while div.len() <= c_deg {
            div.push(big::BIG::new());
        }
        for i in (n..=c_deg).rev() {
            quotient_coeffs_all[i - n] = big::BIG::new_copy(&div[i]);
            div[i - n] = big::BIG::modadd(&div[i - n], &div[i], &modulus);
        }
    } else {
        // C(x) degree < n: standard division where C has degree n-1
        // This handles the case from evaluate_on_domain (degree n-1 constraints)
        let mut div = dividend.clone();
        while div.len() < 2 * n {
            div.push(big::BIG::new());
        }
        quotient_coeffs_all = vec![big::BIG::new(); n];
        for i in (0..n).rev() {
            quotient_coeffs_all[i] = big::BIG::new_copy(&div[n + i]);
            div[i] = big::BIG::modadd(&div[i], &quotient_coeffs_all[i], &modulus);
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Split Q into chunks if needed, commit each
    // -----------------------------------------------------------------------
    let num_q_chunks = if quotient_coeffs_all.len() > n {
        ((quotient_coeffs_all.len() - 1) / n) + 1
    } else {
        1
    };
    let num_q_chunks = num_q_chunks.min(2) as u8; // At most 2 chunks for our degree budget

    let mut quotient_commitments = Vec::new();
    let mut q_chunk_coeffs: Vec<Vec<big::BIG>> = Vec::new();

    for chunk_idx in 0..num_q_chunks as usize {
        let start = chunk_idx * n;
        let end = (start + n).min(quotient_coeffs_all.len());
        let mut chunk: Vec<big::BIG> = quotient_coeffs_all[start..end].to_vec();
        // Pad chunk to n for commitment
        while chunk.len() < n {
            chunk.push(big::BIG::new());
        }
        // Commit in coefficient form
        let comm_point = bls48581::commit_scalars_monomial(&chunk);
        quotient_commitments.push(Commitment(comm_point));
        q_chunk_coeffs.push(chunk);
    }

    // -----------------------------------------------------------------------
    // Step 7: Absorb quotient commitments, derive z
    // -----------------------------------------------------------------------
    for qc in &quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = challenge_to_big(&z_bytes);

    // -----------------------------------------------------------------------
    // Step 8: Evaluate columns and Q chunks at z
    // (Reuses column_coeffs_big computed in Step 4 — no redundant IFFTs)
    // -----------------------------------------------------------------------
    let mut evaluations: Vec<Vec<u8>> = Vec::with_capacity(big_columns.len() + num_q_chunks as usize);

    for coeffs in &column_coeffs_big {
        let y = commitment::eval_poly_at(coeffs, &z);
        evaluations.push(big_to_eval_bytes(&y));
    }

    // Evaluate each Q chunk at z
    let mut q_chunk_evals = Vec::new();
    for chunk in &q_chunk_coeffs {
        let q_at_z = commitment::eval_poly_at(chunk, &z);
        evaluations.push(big_to_eval_bytes(&q_at_z));
        q_chunk_evals.push(q_at_z);
    }

    // -----------------------------------------------------------------------
    // Step 8b: Evaluate shifted columns at ω·z (for cross-row constraints)
    // -----------------------------------------------------------------------
    let mut shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_shifts {
        use bls48581::bls;
        let s = bls::singleton();
        let omega_big = s.RootsOfUnityBLS48581[&domain_size][1].clone();
        let omega_z = big::BIG::modmul(&omega_big, &z, &modulus);
        for &col_idx in &shifted_indices {
            let y = commitment::eval_poly_at(&column_coeffs_big[col_idx], &omega_z);
            shifted_evaluations.push(big_to_eval_bytes(&y));
        }
    }

    // -----------------------------------------------------------------------
    // Step 9: Absorb evaluations, derive batch opening challenge β
    // -----------------------------------------------------------------------
    for eval_bytes in &evaluations {
        transcript.append_message(b"evaluation", eval_bytes);
    }
    for se in &shifted_evaluations {
        transcript.append_message(b"shifted_evaluation", se);
    }
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = challenge_to_big(&beta_bytes);

    // -----------------------------------------------------------------------
    // Step 10: Batch open all polynomials at z using β
    // -----------------------------------------------------------------------
    let batch_size = n; // All coefficient arrays are padded to n
    let mut combined_coeffs = vec![big::BIG::new(); batch_size];
    let mut combined_y = big::BIG::new();
    let mut beta_power = big::BIG::new_int(1);

    // Combine column polynomials
    for (col_idx, coeffs) in column_coeffs_big.iter().enumerate() {
        let y_bytes = &evaluations[col_idx];
        let y = big::BIG::frombytes(y_bytes);
        let y_term = big::BIG::modmul(&beta_power, &y, &modulus);
        combined_y = big::BIG::modadd(&combined_y, &y_term, &modulus);

        for j in 0..coeffs.len().min(batch_size) {
            let term = big::BIG::modmul(&beta_power, &coeffs[j], &modulus);
            combined_coeffs[j] = big::BIG::modadd(&combined_coeffs[j], &term, &modulus);
        }
        beta_power = big::BIG::modmul(&beta_power, &beta, &modulus);
    }

    // Combine Q chunk polynomials
    for (chunk_idx, chunk) in q_chunk_coeffs.iter().enumerate() {
        let q_eval = &q_chunk_evals[chunk_idx];
        let y_term = big::BIG::modmul(&beta_power, q_eval, &modulus);
        combined_y = big::BIG::modadd(&combined_y, &y_term, &modulus);

        for j in 0..chunk.len().min(batch_size) {
            let term = big::BIG::modmul(&beta_power, &chunk[j], &modulus);
            combined_coeffs[j] = big::BIG::modadd(&combined_coeffs[j], &term, &modulus);
        }
        beta_power = big::BIG::modmul(&beta_power, &beta, &modulus);
    }

    // Subtract combined_y from constant term
    combined_coeffs[0] = big::BIG::modadd(
        &combined_coeffs[0],
        &big::BIG::modneg(&combined_y, &modulus),
        &modulus,
    );

    // Synthetic division by (x - z)
    let q_open_coeffs = commitment::div_by_linear(&combined_coeffs, &z);

    // Commit quotient in coefficient form via monomial SRS
    let proof_point = bls48581::commit_scalars_monomial(&q_open_coeffs);

    let opening_proof = BatchProof {
        d: vec![],
        proof: proof_point,
    };

    // -----------------------------------------------------------------------
    // Step 10b: Batch open shifted columns at ω·z (if any)
    // -----------------------------------------------------------------------
    let shifted_opening_proof = if has_shifts {
        use bls48581::bls;
        let beta_shifted_bytes = transcript.challenge_bytes(b"beta_shifted");
        let beta_shifted = challenge_to_big(&beta_shifted_bytes);
        let s = bls::singleton();
        let omega_big = s.RootsOfUnityBLS48581[&domain_size][1].clone();
        let omega_z = big::BIG::modmul(&omega_big, &z, &modulus);

        let mut combined_shifted = vec![big::BIG::new(); batch_size];
        let mut combined_shifted_y = big::BIG::new();
        let mut bs_power = big::BIG::new_int(1);

        for (i, &col_idx) in shifted_indices.iter().enumerate() {
            let y = big::BIG::frombytes(&shifted_evaluations[i]);
            let y_term = big::BIG::modmul(&bs_power, &y, &modulus);
            combined_shifted_y = big::BIG::modadd(&combined_shifted_y, &y_term, &modulus);

            let coeffs = &column_coeffs_big[col_idx];
            for j in 0..coeffs.len().min(batch_size) {
                let term = big::BIG::modmul(&bs_power, &coeffs[j], &modulus);
                combined_shifted[j] = big::BIG::modadd(&combined_shifted[j], &term, &modulus);
            }
            bs_power = big::BIG::modmul(&bs_power, &beta_shifted, &modulus);
        }

        combined_shifted[0] = big::BIG::modadd(
            &combined_shifted[0],
            &big::BIG::modneg(&combined_shifted_y, &modulus),
            &modulus,
        );
        let q_shifted = commitment::div_by_linear(&combined_shifted, &omega_z);
        let shifted_proof_point = bls48581::commit_scalars_monomial(&q_shifted);

        Some(BatchProof {
            d: vec![],
            proof: shifted_proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 11: Assemble and return the proof
    // -----------------------------------------------------------------------
    ExecutionProof {
        column_commitments,
        quotient_commitments,
        evaluations,
        opening_proof,
        num_steps,
        domain_size,
        num_quotient_chunks: num_q_chunks,
        shifted_evaluations,
        shifted_opening_proof,
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

/// Generate an execution proof from a trace and constraint system.
pub fn prove(trace: &TracePolynomials, constraints: &dyn VmConstraintSystem) -> ExecutionProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    prove_inner(trace, constraints, &mut transcript)
}

/// Generate an execution proof using a generic CommitmentScheme.
///
/// Works with any curve type via the [`CommitmentScheme`] trait (BLS48-581 or BLS12-381).
pub fn prove_with_scheme(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    scheme: &dyn crate::scheme::CommitmentScheme,
) -> ExecutionProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    prove_inner_scheme(trace, constraints, &mut transcript, scheme)
}

/// Core proving logic using a generic CommitmentScheme.
fn prove_inner_scheme(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
    scheme: &dyn crate::scheme::CommitmentScheme,
) -> ExecutionProof {
    use std::time::Instant;
    let prove_start = Instant::now();

    let num_steps = trace.num_steps();
    let domain_size = trace.domain_size();
    let curve = trace.curve;

    // Build evaluation-form column data from trace polynomials
    let mut col_eval_forms: Vec<Vec<Scalar>> = trace.columns.iter()
        .map(|p| p.evaluations.clone())
        .collect();

    // Fix selector padding: set the designated "no-op" selector to 1 on
    // padding rows so that the sum-to-one constraint is satisfied everywhere.
    if let Some(padding_col) = constraints.padding_selector_column() {
        let one = Scalar::one(curve);
        let num_rows = trace.num_rows;
        let padded = domain_size as usize;
        if padding_col < col_eval_forms.len() {
            for i in num_rows..padded {
                col_eval_forms[padding_col][i] = one.clone();
            }
        }
    }

    // Fix trace padding for cross-row constraint satisfaction (PC continuity).
    constraints.fix_trace_padding(
        &mut col_eval_forms,
        trace.num_rows,
        domain_size as usize,
    );

    // Convert each column to coefficient form for commitment.
    let t_ifft = Instant::now();
    let column_coeffs_all: Vec<Vec<Scalar>> = col_eval_forms.par_iter()
        .map(|col| scheme.ifft(col, domain_size))
        .collect();
    let ifft_trace_ms = t_ifft.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 1: Commit to each trace column (coefficient form)
    // -----------------------------------------------------------------------
    let t_commit_trace = Instant::now();
    let column_commitments: Vec<Commitment> = column_coeffs_all.par_iter()
        .map(|coeffs| Commitment(scheme.commit_coefficients(coeffs)))
        .collect();
    let commit_trace_ms = t_commit_trace.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 2: Absorb commitments into Fiat-Shamir transcript
    // -----------------------------------------------------------------------
    transcript.append_u64(b"num_steps", num_steps);
    transcript.append_u64(b"domain_size", domain_size);

    for comm in &column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    // -----------------------------------------------------------------------
    // Step 3: Draw constraint combination challenge alpha
    // -----------------------------------------------------------------------
    let alpha_bytes = transcript.challenge_bytes(b"alpha");
    let alpha = Scalar::from_challenge_bytes(&alpha_bytes, curve);

    // -----------------------------------------------------------------------
    // Step 3b: LogUp range check witness computation
    // -----------------------------------------------------------------------
    let lookup_reqs = constraints.lookup_declarations();
    let logup_groups = crate::lookup::group_declarations(&lookup_reqs);
    let logup_layout = crate::lookup::logup_column_layout(&logup_groups);
    let has_logup = !logup_groups.is_empty();

    let mut logup_commitments: Vec<Commitment> = Vec::new();
    let mut logup_column_coeffs: Vec<Vec<Scalar>> = Vec::new();
    let t_logup = Instant::now();

    if has_logup {
        // Draw gamma challenge for LogUp
        let gamma_bytes = transcript.challenge_bytes(b"logup_gamma");
        let gamma = Scalar::from_challenge_bytes(&gamma_bytes, curve);

        // Compute LogUp witness (byte limb decompositions, running sum, multiplicities)
        let col_refs: Vec<&Vec<Scalar>> = col_eval_forms.iter().collect();
        let witness = crate::lookup::compute_logup_witness(
            &col_refs, &logup_groups, &logup_layout,
            &gamma, trace.num_rows, domain_size as usize, curve,
        );

        // Flatten limb columns + h + m into auxiliary columns
        let mut logup_eval_forms: Vec<Vec<Scalar>> = Vec::new();
        for group_limbs in &witness.limb_columns {
            for limb_col in group_limbs {
                logup_eval_forms.push(limb_col.clone());
            }
        }
        logup_eval_forms.push(witness.h_column.clone());
        logup_eval_forms.push(witness.m_column.clone());

        // Convert to coefficient form and commit (parallel)
        let (comms, coeffs_list): (Vec<_>, Vec<_>) = logup_eval_forms.par_iter()
            .map(|col| {
                let coeffs = scheme.ifft(col, domain_size);
                let comm = Commitment(scheme.commit_coefficients(&coeffs));
                (comm, coeffs)
            })
            .unzip();
        logup_commitments = comms;
        logup_column_coeffs = coeffs_list;

        // Absorb LogUp commitments
        for comm in &logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
    }
    let logup_ms = t_logup.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 3c: Memory permutation witness computation
    // -----------------------------------------------------------------------
    let t_mem_perm = Instant::now();
    let mem_cols = constraints.memory_columns();
    let has_perm = mem_cols.is_some();
    let mut perm_commitments: Vec<Commitment> = Vec::new();
    let mut perm_column_coeffs: Vec<Vec<Scalar>> = Vec::new();
    let mut perm_layout: Option<crate::permutation::MemoryPermutationLayout> = None;
    let mut perm_gamma_opt: Option<Scalar> = None;
    let mut perm_delta_opt: Option<Scalar> = None;

    if let Some((addr_col, val_cols, load_sels, store_sels)) = &mem_cols {
        // Draw permutation challenges
        let perm_gamma_bytes = transcript.challenge_bytes(b"perm_gamma");
        let perm_gamma = Scalar::from_challenge_bytes(&perm_gamma_bytes, curve);
        let perm_delta_bytes = transcript.challenge_bytes(b"perm_delta");
        let perm_delta = Scalar::from_challenge_bytes(&perm_delta_bytes, curve);
        perm_gamma_opt = Some(perm_gamma.clone());
        perm_delta_opt = Some(perm_delta.clone());

        let num_rows = trace.num_rows;
        let n = domain_size as usize;

        // Extract memory accesses from trace columns
        let mut accesses: Vec<crate::permutation::MemoryAccess> = Vec::with_capacity(n);
        for row in 0..n {
            let is_load = if row < num_rows { load_sels.iter().any(|&s| !col_eval_forms[s][row].is_zero()) } else { false };
            let is_store = if row < num_rows { store_sels.iter().any(|&s| !col_eval_forms[s][row].is_zero()) } else { false };

            let (addr, values, rw) = if is_load || is_store {
                let addr = col_eval_forms[*addr_col][row].to_u64();
                let values: Vec<u64> = val_cols.iter().map(|&vc| col_eval_forms[vc][row].to_u64()).collect();
                let rw = if is_store { 1u64 } else { 0u64 };
                (addr, values, rw)
            } else {
                // Non-memory row: dummy entry
                let values: Vec<u64> = vec![0u64; val_cols.len()];
                (0u64, values, 0u64)
            };

            accesses.push(crate::permutation::MemoryAccess {
                addr,
                values,
                timestamp: row as u64,
                rw,
            });
        }

        // Sort and compute auxiliary columns
        let (sorted, is_same_addr, inv_addr_diff) =
            crate::permutation::sort_and_compute_aux(&accesses, curve);
        let z_col = crate::permutation::compute_grand_product(
            &accesses, &sorted, &perm_gamma, &perm_delta, curve,
        );

        let layout = crate::permutation::MemoryPermutationLayout::new(val_cols.len());

        // Build evaluation-form columns for sorted trace + auxiliary
        let mut perm_eval_forms: Vec<Vec<Scalar>> = Vec::with_capacity(layout.num_columns);

        // sorted_addr
        let mut sorted_addr_col = vec![Scalar::zero(curve); n];
        for (i, a) in sorted.iter().enumerate() {
            sorted_addr_col[i] = Scalar::from_u64(a.addr, curve);
        }
        perm_eval_forms.push(sorted_addr_col);

        // sorted_val columns
        for v_idx in 0..val_cols.len() {
            let mut sorted_val_col = vec![Scalar::zero(curve); n];
            for (i, a) in sorted.iter().enumerate() {
                sorted_val_col[i] = Scalar::from_u64(a.values[v_idx], curve);
            }
            perm_eval_forms.push(sorted_val_col);
        }

        // sorted_ts
        let mut sorted_ts_col = vec![Scalar::zero(curve); n];
        for (i, a) in sorted.iter().enumerate() {
            sorted_ts_col[i] = Scalar::from_u64(a.timestamp, curve);
        }
        perm_eval_forms.push(sorted_ts_col);

        // sorted_rw
        let mut sorted_rw_col = vec![Scalar::zero(curve); n];
        for (i, a) in sorted.iter().enumerate() {
            sorted_rw_col[i] = Scalar::from_u64(a.rw, curve);
        }
        perm_eval_forms.push(sorted_rw_col);

        // Z column
        let mut z_eval_col = vec![Scalar::zero(curve); n];
        for (i, z_val) in z_col.iter().enumerate() {
            z_eval_col[i] = z_val.clone();
        }
        perm_eval_forms.push(z_eval_col);

        // is_same_addr
        let mut isa_col = vec![Scalar::zero(curve); n];
        for (i, v) in is_same_addr.iter().enumerate() {
            isa_col[i] = v.clone();
        }
        perm_eval_forms.push(isa_col);

        // inv_addr_diff
        let mut iad_col = vec![Scalar::zero(curve); n];
        for (i, v) in inv_addr_diff.iter().enumerate() {
            iad_col[i] = v.clone();
        }
        perm_eval_forms.push(iad_col);

        // original_ts: row index for each row (needed for grand product constraint)
        let mut orig_ts_col = vec![Scalar::zero(curve); n];
        for i in 0..n {
            orig_ts_col[i] = Scalar::from_u64(i as u64, curve);
        }
        perm_eval_forms.push(orig_ts_col);

        // Convert to coefficient form and commit (parallel)
        let (comms, coeffs_list): (Vec<_>, Vec<_>) = perm_eval_forms.par_iter()
            .map(|col| {
                let coeffs = scheme.ifft(col, domain_size);
                let comm = Commitment(scheme.commit_coefficients(&coeffs));
                (comm, coeffs)
            })
            .unzip();
        perm_commitments = comms;
        perm_column_coeffs = coeffs_list;

        // Absorb permutation commitments
        for comm in &perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }

        perm_layout = Some(layout);
    }
    let mem_perm_ms = t_mem_perm.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 3d: Oracle public-input data collection
    // -----------------------------------------------------------------------
    let oracle_sels = constraints.oracle_selectors();
    let mut oracle_data_entries: Vec<Vec<u8>> = Vec::new();
    if !oracle_sels.is_empty() {
        let num_data_cols = constraints.selector_column_indices().first().copied().unwrap_or(col_eval_forms.len());
        for row in 0..trace.num_rows {
            for &sel_col in &oracle_sels {
                if sel_col < col_eval_forms.len() && !col_eval_forms[sel_col][row].is_zero() {
                    let mut entry = Vec::with_capacity(8 + 8 + 8 * num_data_cols);
                    entry.extend_from_slice(&(row as u64).to_le_bytes());
                    entry.extend_from_slice(&(sel_col as u64).to_le_bytes());
                    for c in 0..num_data_cols {
                        entry.extend_from_slice(&col_eval_forms[c][row].to_u64().to_le_bytes());
                    }
                    oracle_data_entries.push(entry);
                }
            }
        }
        // Absorb oracle data into transcript (only if entries exist)
        if !oracle_data_entries.is_empty() {
            transcript.append_u64(b"num_oracle_entries", oracle_data_entries.len() as u64);
            for entry in &oracle_data_entries {
                transcript.append_message(b"oracle_entry", entry);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3e: Register file permutation witness computation
    // -----------------------------------------------------------------------
    let t_reg_perm = Instant::now();
    let reg_ports = constraints.register_ports();
    let has_reg_perm = !reg_ports.is_empty();
    let mut reg_perm_commitments_vec: Vec<Commitment> = Vec::new();
    let mut reg_perm_column_coeffs: Vec<Vec<Scalar>> = Vec::new();
    let mut reg_perm_layout: Option<crate::permutation::RegisterPermutationLayout> = None;
    let mut reg_perm_gamma_opt: Option<Scalar> = None;
    let mut reg_perm_delta_opt: Option<Scalar> = None;

    if has_reg_perm {
        let num_ports = reg_ports.len();
        let n = domain_size as usize;

        // Draw register permutation challenges
        let rg_bytes = transcript.challenge_bytes(b"reg_perm_gamma");
        let reg_gamma = Scalar::from_challenge_bytes(&rg_bytes, curve);
        let rd_bytes = transcript.challenge_bytes(b"reg_perm_delta");
        let reg_delta = Scalar::from_challenge_bytes(&rd_bytes, curve);
        reg_perm_gamma_opt = Some(reg_gamma.clone());
        reg_perm_delta_opt = Some(reg_delta.clone());

        // Collect all register accesses: num_ports entries per row, interleaved
        let mut all_accesses: Vec<crate::permutation::MemoryAccess> = Vec::with_capacity(num_ports * n);
        for row in 0..n {
            for (port_idx, &(reg_col, val_col, is_write)) in reg_ports.iter().enumerate() {
                let reg_num = if row < trace.num_rows {
                    col_eval_forms[reg_col][row].to_u64()
                } else { 0 };
                let val = if row < trace.num_rows {
                    col_eval_forms[val_col][row].to_u64()
                } else { 0 };
                let ts = (row * num_ports + port_idx) as u64;
                let rw = if is_write { 1u64 } else { 0u64 };
                all_accesses.push(crate::permutation::MemoryAccess {
                    addr: reg_num,
                    values: vec![val],
                    timestamp: ts,
                    rw,
                });
            }
        }

        // Sort all accesses by (register, timestamp)
        let (sorted, _is_same_flat, _inv_diff_flat) =
            crate::permutation::sort_and_compute_aux(&all_accesses, curve);
        let z_all = crate::permutation::compute_grand_product(
            &all_accesses, &sorted, &reg_gamma, &reg_delta, curve,
        );

        let layout = crate::permutation::RegisterPermutationLayout::new(num_ports);
        let mut reg_eval_forms: Vec<Vec<Scalar>> = Vec::with_capacity(layout.num_columns);

        // Sorted columns: num_ports lanes × 4 (reg, val, ts, rw), interleaved
        for lane in 0..num_ports {
            for field_fn in [
                |a: &crate::permutation::MemoryAccess| a.addr,
                |a: &crate::permutation::MemoryAccess| a.values[0],
                |a: &crate::permutation::MemoryAccess| a.timestamp,
                |a: &crate::permutation::MemoryAccess| a.rw,
            ] {
                let mut col = vec![Scalar::zero(curve); n];
                for row in 0..n {
                    let idx = row * num_ports + lane;
                    if idx < sorted.len() {
                        col[row] = Scalar::from_u64(field_fn(&sorted[idx]), curve);
                    }
                }
                reg_eval_forms.push(col);
            }
        }

        // Z accumulator: Z[row] = ∏_{j=0}^{row*P-1} numer_j / denom_j
        // z_all[k] = ∏_{j=0}^{k-1} numer_j/denom_j, so Z[row] = z_all[row * P]
        let mut z_eval = vec![Scalar::zero(curve); n];
        z_eval[0] = Scalar::one(curve); // z_all[0] = 1
        for row in 1..n {
            let idx = row * num_ports;
            if idx < z_all.len() {
                z_eval[row] = z_all[idx].clone();
            }
        }
        reg_eval_forms.push(z_eval);

        // is_same_reg (backward-looking): isa[row] = 1 if sorted[row] same addr as sorted[row-1].
        // inv_reg_diff (forward-looking): inv[row] = 1/(addr[row+1] - addr[row]) when they differ.
        // The constraint accesses isa(ω·X) (shifted) and inv(X) (unshifted), so at X=ω^r:
        //   isa[r+1] tells if rows r+1 and r have same addr
        //   inv[r] must be the inverse of the addr diff between rows r+1 and r
        // Layout expects all is_same_reg columns contiguously, then all inv_reg_diff columns.
        let mut isa_cols: Vec<Vec<Scalar>> = Vec::with_capacity(num_ports);
        let mut inv_cols: Vec<Vec<Scalar>> = Vec::with_capacity(num_ports);
        for lane in 0..num_ports {
            let mut isa_col = vec![Scalar::zero(curve); n];
            let mut inv_col = vec![Scalar::zero(curve); n];
            // is_same_reg: backward-looking (row vs row-1)
            for row in 1..n {
                let curr_idx = row * num_ports + lane;
                let prev_idx = (row - 1) * num_ports + lane;
                if curr_idx < sorted.len() && prev_idx < sorted.len() {
                    if sorted[curr_idx].addr == sorted[prev_idx].addr {
                        isa_col[row] = Scalar::one(curve);
                    }
                }
            }
            // inv_reg_diff: forward-looking (row vs row+1)
            for row in 0..n.saturating_sub(1) {
                let curr_idx = row * num_ports + lane;
                let next_idx = (row + 1) * num_ports + lane;
                if next_idx < sorted.len() && curr_idx < sorted.len() {
                    if sorted[next_idx].addr != sorted[curr_idx].addr {
                        let diff = Scalar::from_u64(
                            sorted[next_idx].addr.wrapping_sub(sorted[curr_idx].addr), curve,
                        );
                        inv_col[row] = diff.inverse();
                    }
                }
            }
            isa_cols.push(isa_col);
            inv_cols.push(inv_col);
        }
        for col in isa_cols {
            reg_eval_forms.push(col);
        }
        for col in inv_cols {
            reg_eval_forms.push(col);
        }

        // Row timestamp column: row_ts[i] = i
        let mut row_ts_col = vec![Scalar::zero(curve); n];
        for i in 0..n {
            row_ts_col[i] = Scalar::from_u64(i as u64, curve);
        }
        reg_eval_forms.push(row_ts_col);

        assert_eq!(reg_eval_forms.len(), layout.num_columns,
            "register permutation column count mismatch");

        // Convert to coefficient form and commit (parallel)
        let (comms, coeffs_list): (Vec<_>, Vec<_>) = reg_eval_forms.par_iter()
            .map(|col| {
                let coeffs = scheme.ifft(col, domain_size);
                let comm = Commitment(scheme.commit_coefficients(&coeffs));
                (comm, coeffs)
            })
            .unzip();
        reg_perm_commitments_vec = comms;
        reg_perm_column_coeffs = coeffs_list;

        // Absorb register permutation commitments
        for comm in &reg_perm_commitments_vec {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }

        reg_perm_layout = Some(layout);
    }
    let reg_perm_ms = t_reg_perm.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 4: Build C(x) — either via build_constraint_polynomial (selector-based)
    //         or via evaluate_on_domain + IFFT (fallback)
    // -----------------------------------------------------------------------
    let t_constraints = Instant::now();
    let c_coeffs: Vec<Scalar>;
    let use_build_poly = !constraints.selector_column_indices().is_empty();

    // Compute domain generator ω for cross-row constraints (if needed)
    let shifted_indices = constraints.shifted_column_indices();
    let has_shifts = !shifted_indices.is_empty();
    let omega = if has_shifts || has_logup || has_perm || has_reg_perm {
        Some(scheme.domain_generator(domain_size))
    } else {
        None
    };

    if use_build_poly {
        // Selector-based VMs: build C(x) algebraically in coefficient form
        let mut c_intra = constraints.build_constraint_polynomial(
            &column_coeffs_all, &alpha, domain_size,
        );

        // Track alpha offset for LogUp/permutation constraints
        let mut alpha_offset = constraints.num_constraints();

        // Add cross-row constraint polynomial if any
        if has_shifts {
            let c_shifted = constraints.build_shifted_constraint_polynomial(
                &column_coeffs_all, &alpha, domain_size,
                omega.as_ref().unwrap(),
                alpha_offset,
            );
            c_intra = crate::poly_arith::poly_add(&c_intra, &c_shifted, curve);
            alpha_offset += constraints.num_shifted_constraints();
        }

        // Add LogUp decomposition constraints to C(x)
        if has_logup {
            // Compute alpha^alpha_offset
            let mut ap = Scalar::one(curve);
            for _ in 0..alpha_offset {
                ap = ap.mul(&alpha);
            }

            let lookup_reqs = constraints.lookup_declarations();
            let logup_groups_for_cx = crate::lookup::group_declarations(&lookup_reqs);
            let logup_layout_for_cx = crate::lookup::logup_column_layout(&logup_groups_for_cx);

            // For each group: build sel(X) * (value(X) - Σ limb_k(X) * 256^k)
            for (g_idx, group) in logup_groups_for_cx.iter().enumerate() {
                let (start_offset, num_limbs) = logup_layout_for_cx.limb_offsets[g_idx];

                // Recompose: Σ limb_k(X) * 256^k in coefficient form
                let two56 = Scalar::from_u64(256, curve);
                let mut recomposed = vec![Scalar::zero(curve); domain_size as usize];
                let mut power = Scalar::one(curve);

                for l in 0..num_limbs {
                    let limb_coeffs = &logup_column_coeffs[start_offset + l];
                    let scaled = crate::poly_arith::poly_scalar_mul(limb_coeffs, &power);
                    recomposed = crate::poly_arith::poly_add(&recomposed, &scaled, curve);
                    power = power.mul(&two56);
                }

                // diff(X) = value(X) - recomposed(X)
                let value_coeffs = &column_coeffs_all[group.column_index];
                let diff = crate::poly_arith::poly_sub(value_coeffs, &recomposed, curve);

                // Gate by selector: combined_sel(X) = Σ sel_k(X)
                let body = if !group.selectors.is_empty() {
                    let mut sel_combined = vec![Scalar::zero(curve); domain_size as usize];
                    for &sel_idx in &group.selectors {
                        sel_combined = crate::poly_arith::poly_add(
                            &sel_combined, &column_coeffs_all[sel_idx], curve,
                        );
                    }
                    // body = sel_combined(X) * diff(X)
                    crate::poly_arith::poly_mul(&sel_combined, &diff, curve)
                } else {
                    diff
                };

                // Accumulate: alpha^k * body(X)
                let scaled_body = crate::poly_arith::poly_scalar_mul(&body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_body, curve);
                ap = ap.mul(&alpha);
            }

            alpha_offset += logup_groups_for_cx.len();

            // Running sum boundary: L_0(X) · h(X) = 0 (ensures h(ω^0) = 0)
            // L_0(X) = (1/n) · (1 + X + X^2 + ... + X^{n-1})
            let h_idx = logup_layout_for_cx.h_column;
            if h_idx < logup_column_coeffs.len() {
                let n = domain_size as usize;
                let n_inv = Scalar::from_u64(n as u64, curve).inverse();
                let l0_coeffs: Vec<Scalar> = vec![n_inv; n];
                let h_coeffs = &logup_column_coeffs[h_idx];
                let body_h_boundary = crate::poly_arith::poly_mul(&l0_coeffs, h_coeffs, curve);
                let scaled_h_boundary = crate::poly_arith::poly_scalar_mul(&body_h_boundary, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_h_boundary, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;
            }
        }

        // Add memory permutation constraints to C(x)
        if has_perm {
            if let Some(ref layout) = perm_layout {
                // Compute alpha^alpha_offset
                let mut ap = Scalar::one(curve);
                for _ in 0..alpha_offset {
                    ap = ap.mul(&alpha);
                }

                let om = omega.as_ref().unwrap();

                // Compute ω^{n-1}
                let n_minus_1 = domain_size - 1;
                let mut omega_n_minus_1 = Scalar::one(curve);
                let mut base_o = om.clone();
                let mut exp_o = n_minus_1;
                while exp_o > 0 {
                    if exp_o & 1 == 1 {
                        omega_n_minus_1 = omega_n_minus_1.mul(&base_o);
                    }
                    base_o = base_o.mul(&base_o);
                    exp_o >>= 1;
                }

                // Constraint 1: is_same_addr * (is_same_addr - 1) = 0
                let isa_coeffs = &perm_column_coeffs[layout.is_same_addr];
                let one_poly = {
                    let mut p = vec![Scalar::zero(curve); domain_size as usize];
                    p[0] = Scalar::one(curve);
                    p
                };
                let isa_minus_one = crate::poly_arith::poly_sub(isa_coeffs, &one_poly, curve);
                let body1 = crate::poly_arith::poly_mul(isa_coeffs, &isa_minus_one, curve);
                let scaled1 = crate::poly_arith::poly_scalar_mul(&body1, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled1, curve);
                ap = ap.mul(&alpha);

                // Exclusion factor polynomial: (X - ω^{n-1})
                // Cross-row constraints are multiplied by this to exclude wrap-around

                // Constraint 2: is_same_addr(ω·X) * (sorted_addr(ω·X) - sorted_addr(X)) = 0
                // Multiply by (X - ω^{n-1})
                let isa_shifted = crate::poly_arith::poly_shift(isa_coeffs, om);
                let sa_coeffs = &perm_column_coeffs[layout.sorted_addr];
                let sa_shifted = crate::poly_arith::poly_shift(sa_coeffs, om);
                let addr_diff = crate::poly_arith::poly_sub(&sa_shifted, sa_coeffs, curve);
                let body2_inner = crate::poly_arith::poly_mul(&isa_shifted, &addr_diff, curve);
                let body2 = crate::poly_arith::poly_mul_linear(&body2_inner, &omega_n_minus_1);
                let scaled2 = crate::poly_arith::poly_scalar_mul(&body2, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled2, curve);
                ap = ap.mul(&alpha);

                // Constraint 3: (1 - is_same_addr(ω·X)) * (1 - addr_diff * inv_addr_diff) = 0
                // Multiply by (X - ω^{n-1})
                let one_minus_isa_shifted = crate::poly_arith::poly_sub(&one_poly, &isa_shifted, curve);
                let inv_diff_coeffs = &perm_column_coeffs[layout.inv_addr_diff];
                let addr_diff_inv_prod = crate::poly_arith::poly_mul(&addr_diff, inv_diff_coeffs, curve);
                let one_minus_prod = crate::poly_arith::poly_sub(&one_poly, &addr_diff_inv_prod, curve);
                let body3_inner = crate::poly_arith::poly_mul(&one_minus_isa_shifted, &one_minus_prod, curve);
                let body3 = crate::poly_arith::poly_mul_linear(&body3_inner, &omega_n_minus_1);
                let scaled3 = crate::poly_arith::poly_scalar_mul(&body3, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled3, curve);
                ap = ap.mul(&alpha);

                // Constraint 4: Read consistency per value column
                // is_same_addr(ω·X) * (1 - sorted_rw(ω·X)) * (sorted_val_k(ω·X) - sorted_val_k(X)) = 0
                let srw_coeffs = &perm_column_coeffs[layout.sorted_rw];
                let srw_shifted = crate::poly_arith::poly_shift(srw_coeffs, om);
                let one_minus_srw_shifted = crate::poly_arith::poly_sub(&one_poly, &srw_shifted, curve);
                let isa_times_rw = crate::poly_arith::poly_mul(&isa_shifted, &one_minus_srw_shifted, curve);

                for v in 0..layout.num_val_columns {
                    let sv_coeffs = &perm_column_coeffs[layout.sorted_val_start + v];
                    let sv_shifted = crate::poly_arith::poly_shift(sv_coeffs, om);
                    let val_diff = crate::poly_arith::poly_sub(&sv_shifted, sv_coeffs, curve);
                    let body4_inner = crate::poly_arith::poly_mul(&isa_times_rw, &val_diff, curve);
                    let body4 = crate::poly_arith::poly_mul_linear(&body4_inner, &omega_n_minus_1);
                    let scaled4 = crate::poly_arith::poly_scalar_mul(&body4, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled4, curve);
                    ap = ap.mul(&alpha);
                }

                // Constraint 5: Grand product transition
                // Z(ω·X) · denom(X) - Z(X) · numer(X) = 0
                // numer(X) = γ + addr(X) + δ·val(X) + δ²·ts(X) + δ³·rw(X)  (original trace)
                // denom(X) = γ + sorted_addr(X) + δ·sorted_val(X) + δ²·sorted_ts(X) + δ³·sorted_rw(X)
                if let (Some((addr_col, val_cols, _load_sel, store_sel)), Some(ref pg), Some(ref pd)) =
                    (&mem_cols, &perm_gamma_opt, &perm_delta_opt)
                {
                    let delta2 = pd.mul(pd);
                    let delta3 = delta2.mul(pd);

                    // Build numer(X) polynomial
                    let gamma_poly = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = pg.clone();
                        p
                    };
                    let mut numer_poly = gamma_poly.clone();
                    numer_poly = crate::poly_arith::poly_add(&numer_poly, &column_coeffs_all[*addr_col], curve);

                    // δ · val_combined(X)
                    if val_cols.len() == 1 {
                        let scaled = crate::poly_arith::poly_scalar_mul(&column_coeffs_all[val_cols[0]], pd);
                        numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled, curve);
                    } else {
                        let delta4 = delta2.mul(&delta2);
                        let mut dp = pd.clone();
                        for (v_idx, &vc) in val_cols.iter().enumerate() {
                            let scaled = crate::poly_arith::poly_scalar_mul(&column_coeffs_all[vc], &dp);
                            numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled, curve);
                            if v_idx < val_cols.len() - 1 {
                                dp = dp.mul(&delta4);
                            }
                        }
                    }

                    // δ² · original_ts(X)
                    let ts_coeffs = &perm_column_coeffs[layout.original_ts];
                    let scaled_ts = crate::poly_arith::poly_scalar_mul(ts_coeffs, &delta2);
                    numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled_ts, curve);

                    // δ³ · rw(X) = sum of store_sel columns
                    let mut rw_coeffs = vec![Scalar::zero(curve)];
                    for &ss in store_sel {
                        rw_coeffs = crate::poly_arith::poly_add(&rw_coeffs, &column_coeffs_all[ss], curve);
                    }
                    let scaled_rw = crate::poly_arith::poly_scalar_mul(&rw_coeffs, &delta3);
                    numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled_rw, curve);

                    // Build denom(X) polynomial from sorted columns
                    let mut denom_poly = gamma_poly;
                    denom_poly = crate::poly_arith::poly_add(
                        &denom_poly, &perm_column_coeffs[layout.sorted_addr], curve,
                    );

                    if layout.num_val_columns == 1 {
                        let scaled = crate::poly_arith::poly_scalar_mul(
                            &perm_column_coeffs[layout.sorted_val_start], pd,
                        );
                        denom_poly = crate::poly_arith::poly_add(&denom_poly, &scaled, curve);
                    } else {
                        let delta4 = delta2.mul(&delta2);
                        let mut dp = pd.clone();
                        for v in 0..layout.num_val_columns {
                            let scaled = crate::poly_arith::poly_scalar_mul(
                                &perm_column_coeffs[layout.sorted_val_start + v], &dp,
                            );
                            denom_poly = crate::poly_arith::poly_add(&denom_poly, &scaled, curve);
                            if v < layout.num_val_columns - 1 {
                                dp = dp.mul(&delta4);
                            }
                        }
                    }

                    let sorted_ts_scaled = crate::poly_arith::poly_scalar_mul(
                        &perm_column_coeffs[layout.sorted_ts], &delta2,
                    );
                    denom_poly = crate::poly_arith::poly_add(&denom_poly, &sorted_ts_scaled, curve);
                    let sorted_rw_scaled = crate::poly_arith::poly_scalar_mul(
                        &perm_column_coeffs[layout.sorted_rw], &delta3,
                    );
                    denom_poly = crate::poly_arith::poly_add(&denom_poly, &sorted_rw_scaled, curve);

                    // Z(ω·X) · denom(X) - Z(X) · numer(X)
                    let z_coeffs = &perm_column_coeffs[layout.z_column];
                    let z_shifted = crate::poly_arith::poly_shift(z_coeffs, om);
                    let term1 = crate::poly_arith::poly_mul(&z_shifted, &denom_poly, curve);
                    let term2 = crate::poly_arith::poly_mul(z_coeffs, &numer_poly, curve);
                    let body_gp = crate::poly_arith::poly_sub(&term1, &term2, curve);
                    let scaled_gp = crate::poly_arith::poly_scalar_mul(&body_gp, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_gp, curve);
                    ap = ap.mul(&alpha);

                    // Constraint 6: Boundary Z(1) = 1
                    // L_0(X) · (Z(X) - 1) = 0, where L_0(X) = (1/n) · Σ X^k
                    let n = domain_size as usize;
                    let n_inv = Scalar::from_u64(n as u64, curve).inverse();
                    let l0_coeffs: Vec<Scalar> = vec![n_inv; n];
                    let z_minus_one = crate::poly_arith::poly_sub(z_coeffs, &one_poly, curve);
                    let body_boundary = crate::poly_arith::poly_mul(&l0_coeffs, &z_minus_one, curve);
                    let scaled_boundary = crate::poly_arith::poly_scalar_mul(&body_boundary, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_boundary, curve);
                    ap = ap.mul(&alpha);
                }

                alpha_offset += crate::permutation::num_permutation_constraints(layout);
            }
        }

        // Add register file permutation constraints to C(x)
        if has_reg_perm {
            if let Some(ref layout) = reg_perm_layout {
                let mut ap = Scalar::one(curve);
                for _ in 0..alpha_offset {
                    ap = ap.mul(&alpha);
                }

                let om = omega.as_ref().unwrap();
                let num_ports = layout.num_ports;

                // Compute ω^{n-1}
                let n_minus_1 = domain_size - 1;
                let mut omega_n_minus_1 = Scalar::one(curve);
                let mut base_o = om.clone();
                let mut exp_o = n_minus_1;
                while exp_o > 0 {
                    if exp_o & 1 == 1 {
                        omega_n_minus_1 = omega_n_minus_1.mul(&base_o);
                    }
                    base_o = base_o.mul(&base_o);
                    exp_o >>= 1;
                }

                let one_poly = {
                    let mut p = vec![Scalar::zero(curve); domain_size as usize];
                    p[0] = Scalar::one(curve);
                    p
                };

                // Per-lane sorted consistency constraints (4 per lane)
                for lane in 0..num_ports {
                    let isa_coeffs = &reg_perm_column_coeffs[layout.is_same_reg_start + lane];
                    let inv_coeffs = &reg_perm_column_coeffs[layout.inv_reg_diff_start + lane];
                    let sr_coeffs = &reg_perm_column_coeffs[layout.sorted_reg(lane)];
                    let sv_coeffs = &reg_perm_column_coeffs[layout.sorted_val(lane)];
                    let srw_coeffs = &reg_perm_column_coeffs[layout.sorted_rw(lane)];

                    // 1. is_same_reg binary: isa * (isa - 1) = 0
                    let isa_minus_one = crate::poly_arith::poly_sub(isa_coeffs, &one_poly, curve);
                    let body_bin = crate::poly_arith::poly_mul(isa_coeffs, &isa_minus_one, curve);
                    let scaled_bin = crate::poly_arith::poly_scalar_mul(&body_bin, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_bin, curve);
                    ap = ap.mul(&alpha);

                    // Shifted versions for cross-row constraints
                    let isa_shifted = crate::poly_arith::poly_shift(isa_coeffs, om);
                    let sr_shifted = crate::poly_arith::poly_shift(sr_coeffs, om);
                    let sv_shifted = crate::poly_arith::poly_shift(sv_coeffs, om);
                    let srw_shifted = crate::poly_arith::poly_shift(srw_coeffs, om);

                    // 2. Address continuity: isa(ω·X) * (sr(ω·X) - sr(X)) * (X - ω^{n-1})
                    let reg_diff = crate::poly_arith::poly_sub(&sr_shifted, sr_coeffs, curve);
                    let body2_inner = crate::poly_arith::poly_mul(&isa_shifted, &reg_diff, curve);
                    let body2 = crate::poly_arith::poly_mul_linear(&body2_inner, &omega_n_minus_1);
                    let scaled2 = crate::poly_arith::poly_scalar_mul(&body2, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled2, curve);
                    ap = ap.mul(&alpha);

                    // 3. Difference inverse: (1 - isa(ω·X)) * (1 - diff * inv(X)) * (X - ω^{n-1})
                    let one_minus_isa = crate::poly_arith::poly_sub(&one_poly, &isa_shifted, curve);
                    let diff_inv_prod = crate::poly_arith::poly_mul(&reg_diff, inv_coeffs, curve);
                    let one_minus_prod = crate::poly_arith::poly_sub(&one_poly, &diff_inv_prod, curve);
                    let body3_inner = crate::poly_arith::poly_mul(&one_minus_isa, &one_minus_prod, curve);
                    let body3 = crate::poly_arith::poly_mul_linear(&body3_inner, &omega_n_minus_1);
                    let scaled3 = crate::poly_arith::poly_scalar_mul(&body3, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled3, curve);
                    ap = ap.mul(&alpha);

                    // 4. Read consistency: isa(ω·X) * (1 - rw(ω·X)) * (val(ω·X) - val(X)) * (X - ω^{n-1})
                    let one_minus_srw = crate::poly_arith::poly_sub(&one_poly, &srw_shifted, curve);
                    let isa_times_rw = crate::poly_arith::poly_mul(&isa_shifted, &one_minus_srw, curve);
                    let val_diff = crate::poly_arith::poly_sub(&sv_shifted, sv_coeffs, curve);
                    let body4_inner = crate::poly_arith::poly_mul(&isa_times_rw, &val_diff, curve);
                    let body4 = crate::poly_arith::poly_mul_linear(&body4_inner, &omega_n_minus_1);
                    let scaled4 = crate::poly_arith::poly_scalar_mul(&body4, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled4, curve);
                    ap = ap.mul(&alpha);
                }

                // Grand product transition: Z(ω·X) * ∏ denom_p(X) - Z(X) * ∏ numer_p(X) = 0
                if let (Some(ref rg), Some(ref rd)) = (&reg_perm_gamma_opt, &reg_perm_delta_opt) {
                    let delta2 = rd.mul(rd);
                    let delta3 = delta2.mul(rd);
                    let p_scalar = Scalar::from_u64(num_ports as u64, curve);

                    let z_coeffs = &reg_perm_column_coeffs[layout.z_column];
                    let z_shifted = crate::poly_arith::poly_shift(z_coeffs, om);
                    let row_ts_coeffs = &reg_perm_column_coeffs[layout.row_ts];

                    let gamma_poly = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = rg.clone();
                        p
                    };

                    // Build product of numer and denom polynomials for all ports
                    let mut numer_product_poly = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = Scalar::one(curve);
                        p
                    };
                    let mut denom_product_poly = numer_product_poly.clone();

                    for (port_idx, &(reg_col, val_col, is_write)) in reg_ports.iter().enumerate() {
                        // numer_p(X) = γ + trace_reg(X) + δ·trace_val(X) + δ²·(P·row_ts(X)+p) + δ³·rw
                        let mut numer_p = gamma_poly.clone();
                        numer_p = crate::poly_arith::poly_add(&numer_p, &column_coeffs_all[reg_col], curve);
                        let val_scaled = crate::poly_arith::poly_scalar_mul(&column_coeffs_all[val_col], rd);
                        numer_p = crate::poly_arith::poly_add(&numer_p, &val_scaled, curve);
                        // δ²·(P·row_ts + port_idx)
                        let p_row_ts = crate::poly_arith::poly_scalar_mul(row_ts_coeffs, &p_scalar);
                        let port_idx_scalar = Scalar::from_u64(port_idx as u64, curve);
                        let mut p_row_ts_plus_p = p_row_ts;
                        p_row_ts_plus_p[0] = p_row_ts_plus_p[0].add(&port_idx_scalar);
                        let ts_scaled = crate::poly_arith::poly_scalar_mul(&p_row_ts_plus_p, &delta2);
                        numer_p = crate::poly_arith::poly_add(&numer_p, &ts_scaled, curve);
                        // δ³·rw (constant)
                        let rw_val = if is_write { Scalar::one(curve) } else { Scalar::zero(curve) };
                        let rw_term = delta3.mul(&rw_val);
                        numer_p[0] = numer_p[0].add(&rw_term);

                        numer_product_poly = crate::poly_arith::poly_mul(&numer_product_poly, &numer_p, curve);

                        // denom_p(X) = γ + sorted_reg_lane(X) + δ·sorted_val_lane(X) + δ²·sorted_ts_lane(X) + δ³·sorted_rw_lane(X)
                        let mut denom_p = gamma_poly.clone();
                        denom_p = crate::poly_arith::poly_add(
                            &denom_p, &reg_perm_column_coeffs[layout.sorted_reg(port_idx)], curve,
                        );
                        let sv_scaled = crate::poly_arith::poly_scalar_mul(
                            &reg_perm_column_coeffs[layout.sorted_val(port_idx)], rd,
                        );
                        denom_p = crate::poly_arith::poly_add(&denom_p, &sv_scaled, curve);
                        let sts_scaled = crate::poly_arith::poly_scalar_mul(
                            &reg_perm_column_coeffs[layout.sorted_ts(port_idx)], &delta2,
                        );
                        denom_p = crate::poly_arith::poly_add(&denom_p, &sts_scaled, curve);
                        let srw_scaled = crate::poly_arith::poly_scalar_mul(
                            &reg_perm_column_coeffs[layout.sorted_rw(port_idx)], &delta3,
                        );
                        denom_p = crate::poly_arith::poly_add(&denom_p, &srw_scaled, curve);

                        denom_product_poly = crate::poly_arith::poly_mul(&denom_product_poly, &denom_p, curve);
                    }

                    // Z(ω·X) · ∏denom - Z(X) · ∏numer
                    let term1 = crate::poly_arith::poly_mul(&z_shifted, &denom_product_poly, curve);
                    let term2 = crate::poly_arith::poly_mul(z_coeffs, &numer_product_poly, curve);
                    let body_gp = crate::poly_arith::poly_sub(&term1, &term2, curve);
                    let scaled_gp = crate::poly_arith::poly_scalar_mul(&body_gp, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_gp, curve);
                    ap = ap.mul(&alpha);

                    // Boundary: L_0(X) · (Z(X) - 1) = 0
                    let n_inv = Scalar::from_u64(domain_size, curve).inverse();
                    let l0_coeffs: Vec<Scalar> = vec![n_inv; domain_size as usize];
                    let z_minus_one = crate::poly_arith::poly_sub(z_coeffs, &one_poly, curve);
                    let body_boundary = crate::poly_arith::poly_mul(&l0_coeffs, &z_minus_one, curve);
                    let scaled_boundary = crate::poly_arith::poly_scalar_mul(&body_boundary, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_boundary, curve);
                    ap = ap.mul(&alpha);
                }

                alpha_offset += crate::permutation::num_register_perm_constraints(layout);
                let _ = ap;
            }
        }

        let _ = alpha_offset;
        c_coeffs = c_intra;
    } else {
        // Fallback: evaluate constraints on domain, combine with alpha, IFFT to coeffs
        let columns_refs: Vec<&Vec<Scalar>> = col_eval_forms.iter().collect();
        let constraint_evals = constraints.evaluate_on_domain(&columns_refs, trace.num_rows);
        let n = domain_size as usize;

        // Combine: C_evals[j] = Σ alpha^i * constraint_i_evals[j]
        let mut c_evals = vec![Scalar::zero(curve); n];
        let mut alpha_power = Scalar::one(curve);
        for constraint in &constraint_evals {
            for j in 0..n {
                let val = if j < constraint.len() {
                    constraint[j].clone()
                } else {
                    Scalar::zero(curve)
                };
                let term = alpha_power.mul(&val);
                c_evals[j] = c_evals[j].add(&term);
            }
            alpha_power = alpha_power.mul(&alpha);
        }

        c_coeffs = scheme.ifft(&c_evals, domain_size);
    };
    let constraints_ms = t_constraints.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 5: Divide C(x) by Z(x) = x^n - 1 to get Q(x)
    // -----------------------------------------------------------------------
    let t_quotient = Instant::now();
    let n = domain_size as usize;
    let c_len = c_coeffs.len();
    let c_deg = if c_len > 0 { c_len - 1 } else { 0 };

    let quotient_coeffs_all: Vec<Scalar>;
    if c_deg >= n {
        // C(x) has degree >= n, do proper division
        let mut div = c_coeffs.clone();
        while div.len() <= c_deg {
            div.push(Scalar::zero(curve));
        }
        let q_deg = c_deg - n;
        let mut q = vec![Scalar::zero(curve); q_deg + 1];
        for i in (n..=c_deg).rev() {
            q[i - n] = div[i].clone();
            div[i - n] = div[i - n].add(&div[i]);
        }
        quotient_coeffs_all = q;
    } else {
        // C(x) degree < n: standard approach
        let mut div = c_coeffs.clone();
        while div.len() < 2 * n {
            div.push(Scalar::zero(curve));
        }
        let mut q = vec![Scalar::zero(curve); n];
        for i in (0..n).rev() {
            q[i] = div[n + i].clone();
            div[i] = div[i].add(&q[i]);
        }
        quotient_coeffs_all = q;
    }

    // -----------------------------------------------------------------------
    // Step 6: Split Q into chunks if needed, commit each
    // -----------------------------------------------------------------------
    let num_q_chunks = if quotient_coeffs_all.len() > n {
        ((quotient_coeffs_all.len() - 1) / n) + 1
    } else {
        1
    };
    let num_q_chunks = num_q_chunks as u8;

    let quotient_div_ms = t_quotient.elapsed().as_millis();

    let t_quotient_commit = Instant::now();
    let (quotient_commitments, q_chunk_coeffs): (Vec<_>, Vec<_>) = (0..num_q_chunks as usize)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * n;
            let end = (start + n).min(quotient_coeffs_all.len());
            let mut chunk: Vec<Scalar> = quotient_coeffs_all[start..end].to_vec();
            chunk.resize(n, Scalar::zero(curve));
            let comm = Commitment(scheme.commit_coefficients(&chunk));
            (comm, chunk)
        })
        .unzip();
    let quotient_commit_ms = t_quotient_commit.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 7: Absorb quotient commitments, derive z
    // -----------------------------------------------------------------------
    for qc in &quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = Scalar::from_challenge_bytes(&z_bytes, curve);

    // -----------------------------------------------------------------------
    // Step 8: Evaluate columns and Q chunks at z
    // -----------------------------------------------------------------------
    let t_evals = Instant::now();
    let mut evaluations: Vec<Vec<u8>> = Vec::with_capacity(col_eval_forms.len() + num_q_chunks as usize);

    for coeffs in &column_coeffs_all {
        let y = scheme.eval_poly_at(coeffs, &z);
        evaluations.push(y.to_bytes());
    }

    let mut q_chunk_evals = Vec::new();
    for chunk in &q_chunk_coeffs {
        let q_at_z = scheme.eval_poly_at(chunk, &z);
        evaluations.push(q_at_z.to_bytes());
        q_chunk_evals.push(q_at_z);
    }

    // -----------------------------------------------------------------------
    // Step 8b: Evaluate shifted columns at ω·z (for cross-row constraints)
    // -----------------------------------------------------------------------
    let mut shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_shifts {
        let omega_z = omega.as_ref().unwrap().mul(&z);
        for &col_idx in &shifted_indices {
            let y = scheme.eval_poly_at(&column_coeffs_all[col_idx], &omega_z);
            shifted_evaluations.push(y.to_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Step 8c: Evaluate LogUp columns at z and h at ω·z
    // -----------------------------------------------------------------------
    let mut logup_evaluations_vec: Vec<Vec<u8>> = Vec::new();
    let mut logup_shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_logup {
        for coeffs in &logup_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &z);
            logup_evaluations_vec.push(y.to_bytes());
        }
        // h column needs shifted evaluation for running sum transition
        let h_idx = logup_column_coeffs.len() - 2; // h is second-to-last
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let h_shifted = scheme.eval_poly_at(&logup_column_coeffs[h_idx], &omega_z);
        logup_shifted_evaluations.push(h_shifted.to_bytes());
    }

    // -----------------------------------------------------------------------
    // Step 8d: Evaluate permutation columns at z and shifted columns at ω·z
    // -----------------------------------------------------------------------
    let mut perm_evaluations_vec: Vec<Vec<u8>> = Vec::new();
    let mut perm_shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_perm {
        for coeffs in &perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &z);
            perm_evaluations_vec.push(y.to_bytes());
        }
        // Shifted evaluations for cross-row constraints: all perm columns need ω·z
        let omega_z = omega.as_ref().unwrap().mul(&z);
        for coeffs in &perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &omega_z);
            perm_shifted_evaluations.push(y.to_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Step 8e: Evaluate register permutation columns at z and ω·z
    // -----------------------------------------------------------------------
    let mut reg_perm_evaluations_vec: Vec<Vec<u8>> = Vec::new();
    let mut reg_perm_shifted_evals_vec: Vec<Vec<u8>> = Vec::new();
    if has_reg_perm {
        for coeffs in &reg_perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &z);
            reg_perm_evaluations_vec.push(y.to_bytes());
        }
        let omega_z = omega.as_ref().unwrap().mul(&z);
        for coeffs in &reg_perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &omega_z);
            reg_perm_shifted_evals_vec.push(y.to_bytes());
        }
    }
    let evals_ms = t_evals.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 9: Absorb evaluations, derive batch opening challenge β
    // -----------------------------------------------------------------------
    for eval_bytes in &evaluations {
        transcript.append_message(b"evaluation", eval_bytes);
    }
    for se in &shifted_evaluations {
        transcript.append_message(b"shifted_evaluation", se);
    }
    for le in &logup_evaluations_vec {
        transcript.append_message(b"logup_evaluation", le);
    }
    for lse in &logup_shifted_evaluations {
        transcript.append_message(b"logup_shifted_evaluation", lse);
    }
    for pe in &perm_evaluations_vec {
        transcript.append_message(b"perm_evaluation", pe);
    }
    for pse in &perm_shifted_evaluations {
        transcript.append_message(b"perm_shifted_evaluation", pse);
    }
    for re in &reg_perm_evaluations_vec {
        transcript.append_message(b"reg_perm_evaluation", re);
    }
    for rse in &reg_perm_shifted_evals_vec {
        transcript.append_message(b"reg_perm_shifted_evaluation", rse);
    }
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = Scalar::from_challenge_bytes(&beta_bytes, curve);

    // -----------------------------------------------------------------------
    // Step 10: Batch open all polynomials at z using β
    // -----------------------------------------------------------------------
    let t_batch_open = Instant::now();
    let mut combined_coeffs_batch = vec![Scalar::zero(curve); n];
    let mut combined_y = Scalar::zero(curve);
    let mut beta_power = Scalar::one(curve);

    // Combine column polynomials
    for (col_idx, coeffs) in column_coeffs_all.iter().enumerate() {
        let y = Scalar::from_bytes(&evaluations[col_idx], curve);
        let y_term = beta_power.mul(&y);
        combined_y = combined_y.add(&y_term);

        for j in 0..coeffs.len().min(n) {
            let term = beta_power.mul(&coeffs[j]);
            combined_coeffs_batch[j] = combined_coeffs_batch[j].add(&term);
        }
        beta_power = beta_power.mul(&beta);
    }

    // Combine Q chunk polynomials
    for (chunk_idx, chunk) in q_chunk_coeffs.iter().enumerate() {
        let y_term = beta_power.mul(&q_chunk_evals[chunk_idx]);
        combined_y = combined_y.add(&y_term);

        for j in 0..chunk.len().min(n) {
            let term = beta_power.mul(&chunk[j]);
            combined_coeffs_batch[j] = combined_coeffs_batch[j].add(&term);
        }
        beta_power = beta_power.mul(&beta);
    }

    // Subtract combined_y from constant term
    combined_coeffs_batch[0] = combined_coeffs_batch[0].sub(&combined_y);

    // Synthetic division by (x - z)
    let q_open_coeffs = scheme.div_by_linear(&combined_coeffs_batch, &z);

    // Commit quotient in coefficient form via monomial SRS
    let proof_point = scheme.commit_coefficients(&q_open_coeffs);

    let opening_proof = BatchProof {
        d: vec![],
        proof: proof_point,
    };

    // -----------------------------------------------------------------------
    // Step 10b: Batch open shifted columns at ω·z (if any)
    // -----------------------------------------------------------------------
    let shifted_opening_proof = if has_shifts {
        let beta_shifted_bytes = transcript.challenge_bytes(b"beta_shifted");
        let beta_shifted = Scalar::from_challenge_bytes(&beta_shifted_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);

        let mut combined_shifted = vec![Scalar::zero(curve); n];
        let mut combined_shifted_y = Scalar::zero(curve);
        let mut bs_power = Scalar::one(curve);

        for (i, &col_idx) in shifted_indices.iter().enumerate() {
            let y = Scalar::from_bytes(&shifted_evaluations[i], curve);
            let y_term = bs_power.mul(&y);
            combined_shifted_y = combined_shifted_y.add(&y_term);

            let coeffs = &column_coeffs_all[col_idx];
            for j in 0..coeffs.len().min(n) {
                let term = bs_power.mul(&coeffs[j]);
                combined_shifted[j] = combined_shifted[j].add(&term);
            }
            bs_power = bs_power.mul(&beta_shifted);
        }

        combined_shifted[0] = combined_shifted[0].sub(&combined_shifted_y);
        let q_shifted = scheme.div_by_linear(&combined_shifted, &omega_z);
        let shifted_proof_point = scheme.commit_coefficients(&q_shifted);

        Some(BatchProof {
            d: vec![],
            proof: shifted_proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10c: Batch open LogUp columns at z and h at ω·z (if any)
    // -----------------------------------------------------------------------
    let logup_opening_proof = if has_logup && !logup_column_coeffs.is_empty() {
        let beta_logup_bytes = transcript.challenge_bytes(b"beta_logup");
        let beta_logup = Scalar::from_challenge_bytes(&beta_logup_bytes, curve);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut bl_power = Scalar::one(curve);

        for (i, coeffs) in logup_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&logup_evaluations_vec[i], curve);
            let y_term = bl_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = bl_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            bl_power = bl_power.mul(&beta_logup);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_logup = scheme.div_by_linear(&combined, &z);
        let logup_proof_point = scheme.commit_coefficients(&q_logup);

        Some(BatchProof {
            d: vec![],
            proof: logup_proof_point,
        })
    } else {
        None
    };

    let logup_shifted_opening_proof = if has_logup && !logup_shifted_evaluations.is_empty() {
        let beta_logup_shifted_bytes = transcript.challenge_bytes(b"beta_logup_shifted");
        let beta_logup_shifted = Scalar::from_challenge_bytes(&beta_logup_shifted_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let h_idx = logup_column_coeffs.len() - 2;

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let bl_power = Scalar::one(curve);

        let y = Scalar::from_bytes(&logup_shifted_evaluations[0], curve);
        combined_y = combined_y.add(&bl_power.mul(&y));
        let coeffs = &logup_column_coeffs[h_idx];
        for j in 0..coeffs.len().min(n) {
            combined[j] = combined[j].add(&bl_power.mul(&coeffs[j]));
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_shifted = scheme.div_by_linear(&combined, &omega_z);
        let proof_point = scheme.commit_coefficients(&q_shifted);

        Some(BatchProof {
            d: vec![],
            proof: proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10d: Batch open permutation columns at z and ω·z (if any)
    // -----------------------------------------------------------------------
    let perm_opening_proof = if has_perm && !perm_column_coeffs.is_empty() {
        let beta_perm_bytes = transcript.challenge_bytes(b"beta_perm");
        let beta_perm = Scalar::from_challenge_bytes(&beta_perm_bytes, curve);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut bp_power = Scalar::one(curve);

        for (i, coeffs) in perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&perm_evaluations_vec[i], curve);
            let y_term = bp_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = bp_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            bp_power = bp_power.mul(&beta_perm);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_perm = scheme.div_by_linear(&combined, &z);
        let perm_proof_point = scheme.commit_coefficients(&q_perm);

        Some(BatchProof {
            d: vec![],
            proof: perm_proof_point,
        })
    } else {
        None
    };

    let perm_shifted_opening_proof = if has_perm && !perm_shifted_evaluations.is_empty() {
        let beta_perm_shifted_bytes = transcript.challenge_bytes(b"beta_perm_shifted");
        let beta_perm_shifted = Scalar::from_challenge_bytes(&beta_perm_shifted_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut bps_power = Scalar::one(curve);

        for (i, coeffs) in perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&perm_shifted_evaluations[i], curve);
            let y_term = bps_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = bps_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            bps_power = bps_power.mul(&beta_perm_shifted);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_perm_shifted = scheme.div_by_linear(&combined, &omega_z);
        let proof_point = scheme.commit_coefficients(&q_perm_shifted);

        Some(BatchProof {
            d: vec![],
            proof: proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10e: Batch open register permutation columns at z and ω·z
    // -----------------------------------------------------------------------
    let reg_perm_opening = if has_reg_perm && !reg_perm_column_coeffs.is_empty() {
        let beta_rp_bytes = transcript.challenge_bytes(b"beta_reg_perm");
        let beta_rp = Scalar::from_challenge_bytes(&beta_rp_bytes, curve);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut brp_power = Scalar::one(curve);

        for (i, coeffs) in reg_perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&reg_perm_evaluations_vec[i], curve);
            let y_term = brp_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = brp_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            brp_power = brp_power.mul(&beta_rp);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_rp = scheme.div_by_linear(&combined, &z);
        let rp_proof_point = scheme.commit_coefficients(&q_rp);

        Some(BatchProof {
            d: vec![],
            proof: rp_proof_point,
        })
    } else {
        None
    };

    let reg_perm_shifted_opening = if has_reg_perm && !reg_perm_shifted_evals_vec.is_empty() {
        let beta_rps_bytes = transcript.challenge_bytes(b"beta_reg_perm_shifted");
        let beta_rps = Scalar::from_challenge_bytes(&beta_rps_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut brps_power = Scalar::one(curve);

        for (i, coeffs) in reg_perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&reg_perm_shifted_evals_vec[i], curve);
            let y_term = brps_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = brps_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            brps_power = brps_power.mul(&beta_rps);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_rps = scheme.div_by_linear(&combined, &omega_z);
        let rps_proof_point = scheme.commit_coefficients(&q_rps);

        Some(BatchProof {
            d: vec![],
            proof: rps_proof_point,
        })
    } else {
        None
    };

    let batch_open_ms = t_batch_open.elapsed().as_millis();
    let total_ms = prove_start.elapsed().as_millis();

    // Print timing breakdown
    let num_logup_cols = logup_column_coeffs.len();
    let num_perm_cols = perm_column_coeffs.len();
    let num_reg_perm_cols = reg_perm_column_coeffs.len();
    eprintln!("[prover] domain={} steps={} cols={}+{}logup+{}perm+{}regperm+{}Q",
        domain_size, num_steps, col_eval_forms.len(),
        num_logup_cols, num_perm_cols, num_reg_perm_cols, num_q_chunks);
    eprintln!("[prover] ifft_trace={}ms commit_trace={}ms logup={}ms mem_perm={}ms reg_perm={}ms",
        ifft_trace_ms, commit_trace_ms, logup_ms, mem_perm_ms, reg_perm_ms);
    eprintln!("[prover] constraints={}ms quot_div={}ms quot_commit={}ms evals={}ms batch_open={}ms total={}ms",
        constraints_ms, quotient_div_ms, quotient_commit_ms, evals_ms, batch_open_ms, total_ms);

    // -----------------------------------------------------------------------
    // Step 11: Assemble and return the proof
    // -----------------------------------------------------------------------
    ExecutionProof {
        column_commitments,
        quotient_commitments,
        evaluations,
        opening_proof,
        num_steps,
        domain_size,
        num_quotient_chunks: num_q_chunks,
        shifted_evaluations,
        shifted_opening_proof,
        logup_commitments,
        logup_evaluations: logup_evaluations_vec,
        logup_shifted_evaluations,
        logup_opening_proof,
        logup_shifted_opening_proof,
        perm_commitments,
        perm_evaluations: perm_evaluations_vec,
        perm_shifted_evaluations,
        perm_opening_proof,
        perm_shifted_opening_proof,
        oracle_data: oracle_data_entries,
        reg_perm_commitments: reg_perm_commitments_vec,
        reg_perm_evaluations: reg_perm_evaluations_vec,
        reg_perm_shifted_evaluations: reg_perm_shifted_evals_vec,
        reg_perm_opening_proof: reg_perm_opening,
        reg_perm_shifted_opening_proof: reg_perm_shifted_opening,
    }
}

/// Generate a chunk proof from a trace, constraint system, and state chain metadata.
pub fn prove_chunk(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    chunk_index: u64,
    initial_state_hash: &[u8; 32],
    final_state_hash: &[u8; 32],
) -> ChunkProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    transcript.append_u64(b"chunk_index", chunk_index);
    transcript.append_message(b"initial_state", initial_state_hash);
    transcript.append_message(b"final_state", final_state_hash);

    let execution_proof = prove_inner(trace, constraints, &mut transcript);

    ChunkProof {
        execution_proof,
        initial_state_hash: *initial_state_hash,
        final_state_hash: *final_state_hash,
        chunk_index,
    }
}

/// Generate a chunk proof using a generic CommitmentScheme.
///
/// Same as [`prove_chunk`] but works with any curve type via the
/// [`CommitmentScheme`] trait (BLS48-581 or BLS12-381).
pub fn prove_chunk_with_scheme(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    chunk_index: u64,
    initial_state_hash: &[u8; 32],
    final_state_hash: &[u8; 32],
    scheme: &dyn crate::scheme::CommitmentScheme,
) -> ChunkProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    transcript.append_u64(b"chunk_index", chunk_index);
    transcript.append_message(b"initial_state", initial_state_hash);
    transcript.append_message(b"final_state", final_state_hash);

    let execution_proof = prove_inner_scheme(trace, constraints, &mut transcript, scheme);

    ChunkProof {
        execution_proof,
        initial_state_hash: *initial_state_hash,
        final_state_hash: *final_state_hash,
        chunk_index,
    }
}
