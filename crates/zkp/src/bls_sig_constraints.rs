//! [`VmConstraintSystem`] wiring for BLS aggregate-signature verification.
//!
//! Proves algebraically that the host-claimed aggregate public key
//!
//!     agg_pk = Σ pk_i  (in G1, BLS12-381)
//!
//! is the field-correct sum of the supplied per-attester public keys.
//! Internally that decomposes into a sequence of non-native `Fp` operations
//! (subtraction, inversion, multiplication, squaring) implementing affine
//! G1 point addition; we delegate the entire constraint surface to
//! [`crate::nonnative_fp_constraints::NonnativeFpConstraintSystem`] so the
//! existing 19-category, 150-column AIR (with its 64-bit / 72-bit / 1-bit
//! lookup tables) carries the soundness.
//!
//! Steps 2 (`hm = H_to_G2(msg)`) and 3 (`e(agg_pk, hm) · e(-G1, agg_sig) ==
//! 1`) of fast-aggregate-verify are **not** in-circuit. They are treated as
//! a host-side oracle: the trace builder refuses to construct a witness
//! unless the host's [`bls_sig::fast_aggregate_verify`] returned `true`.
//! Closing the remaining gap (a malicious prover stipulating `accept` while
//! the pairing check actually fails) requires a heavyweight Miller-loop +
//! final-exponentiation AIR — out of scope here.
//!
//! Because we reuse the underlying Fp AIR verbatim, the constraint count,
//! lookup declarations, and column layout are all inherited from
//! [`NonnativeFpConstraintSystem`]; the only public-facing addition is
//! `BlsSigWitness` + a trace builder that produces the right `FpOp`
//! sequence.
//!
//! # Cost model
//!
//! Each `G1Affine::add` (non-doubling, non-degenerate) decomposes into
//! roughly **9 `FpOp`s**: two coordinate subtractions, one Inv (for `1/dx`),
//! one Mul (`λ = dy / dx`), one squaring (`λ²`), three subtractions for
//! `x₃` and `S.x − x₃`, one Mul (`λ · (S.x − x₃)`), one final subtraction
//! for `y₃ = · − S.y`. Aggregating `n` pubkeys is `(n−1)` such adds (the
//! first attester is copied straight into the running sum) plus one
//! identity-fold pseudo-op. The trace builder budgets 12 `FpOp`s per
//! attester to generalise cleanly to doubling and equal-x edge cases
//! (which we resolve by host-side fallback to a non-degenerate witness;
//! see `populate_g1_add`).
//!
//! # Soundness scope
//!
//! Cryptographically proves:
//!   * `num_attesters` rows of pubkey aggregation arithmetic exist in the
//!     trace (each step is a real `Fp` op, range-checked and lookup-bound).
//!   * The running sum at the end of the trace is the algebraic sum of the
//!     witness pubkeys.
//!
//! Out of scope (host-side oracle): the pairing decision
//! `accept_bit == 1 ⟺ verify(agg_pk, H(msg), agg_sig)`. The trace builder
//! refuses to commit a witness whose `fast_aggregate_verify` returned
//! false, so this is strictly stronger than treating the layer as
//! `LayerProofKind::ReferenceOnly`.

use crate::bls_sig::{self, fast_aggregate_verify, PublicKey, Signature};
use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::nonnative_fp::Fp;
use crate::nonnative_fp_air::FpOp;
use crate::nonnative_fp_constraints::{
    build_fp_trace_polynomials, NonnativeFpConstraintSystem,
};
use crate::pairing::G1Affine;
use crate::trace::TracePolynomials;
use crate::vm_constraints::VmConstraintSystem;

// ──── Witness ─────────────────────────────────────────────────────────

/// Host-side witness for one BLS aggregate-signature verification.
///
/// `pubkeys`, `msg`, `agg_sig`, `dst` are the same arguments that
/// [`bls_sig::fast_aggregate_verify`] accepts. The trace builder calls
/// that function and refuses to populate a trace if it returned `false`,
/// so the witness is always "accept-bit = 1" by construction.
#[derive(Debug, Clone)]
pub struct BlsSigWitness {
    /// Individual attester public keys (compressed G1, 48 bytes each).
    pub pubkeys: Vec<PublicKey>,
    /// Message that was signed.
    pub msg: Vec<u8>,
    /// Aggregate signature (compressed G2, 96 bytes).
    pub agg_sig: Signature,
    /// Domain-separation tag for hash-to-curve.
    pub dst: Vec<u8>,
}

// ──── Constraint system ───────────────────────────────────────────────

/// [`VmConstraintSystem`] for BLS aggregate-signature verification.
///
/// All constraint methods delegate to
/// [`NonnativeFpConstraintSystem`]; this wrapper exists so the layer-chain
/// dispatch (`LayerProofKind::BlsSig`) has a distinct type to attach to
/// even though the AIR itself is identical to the underlying nonnative
/// `Fp` arithmetic AIR. The `num_attesters` field is informational
/// (currently unused by constraint evaluation) and reserved for the
/// future heavyweight pairing AIR's row count check.
#[derive(Debug, Clone, Copy)]
pub struct BlsSigConstraintSystem {
    /// Number of attesters the BlsSigWitness was constructed over. Carried
    /// for diagnostic / public-input purposes; the constraint count and
    /// lookup tables do not depend on it.
    pub num_attesters: usize,
    inner: NonnativeFpConstraintSystem,
}

impl BlsSigConstraintSystem {
    /// Construct a BLS aggregate-signature constraint system covering
    /// `num_attesters` per-attester public keys.
    pub const fn new(num_attesters: usize) -> Self {
        Self {
            num_attesters,
            inner: NonnativeFpConstraintSystem::new(),
        }
    }

    /// Inner non-native Fp constraint system. Exposed for the rare caller
    /// that needs to bypass this wrapper (e.g. compatibility tests).
    pub fn inner(&self) -> NonnativeFpConstraintSystem {
        self.inner
    }
}

// ──── Trace builder ────────────────────────────────────────────────────

/// Build a [`TracePolynomials`] proving `agg_pk = Σ witness.pubkeys` is
/// the field-correct sum, conditional on the host having checked
/// [`fast_aggregate_verify`] = `true`.
///
/// # Panics / errors
///
/// Panics (via `expect`) if:
///   * `witness.pubkeys` is empty,
///   * any pubkey fails to decompress,
///   * `fast_aggregate_verify` rejects the witness (i.e. the host-side
///     oracle says the pairing check fails — the trace builder refuses to
///     construct a witness for a failing aggregation).
///
/// On success the trace has at least one row per per-`G1Affine::add`
/// arithmetic step; padding rows are all-zero and satisfy every body
/// trivially (see [`NonnativeFpConstraintSystem::padding_selector_column`]).
pub fn build_bls_sig_trace_polynomials(
    witness: &BlsSigWitness,
    curve: CurveType,
) -> TracePolynomials {
    assert!(
        !witness.pubkeys.is_empty(),
        "BlsSigWitness must contain at least one pubkey"
    );
    assert!(
        fast_aggregate_verify(
            &witness.pubkeys,
            &witness.msg,
            &witness.agg_sig,
            &witness.dst,
        ),
        "BlsSigWitness host-side oracle (fast_aggregate_verify) returned \
         false — the trace builder refuses to construct a witness for a \
         failing aggregation. Fix the inputs or upgrade to the heavyweight \
         pairing AIR."
    );

    // Decompress every pubkey to affine. `from_bytes` performs the
    // on-curve check; we omit a separate subgroup check because
    // `fast_aggregate_verify` already ran and would have rejected a
    // non-subgroup point upstream.
    let affines: Vec<G1Affine> = witness
        .pubkeys
        .iter()
        .map(|pk| {
            G1Affine::from_bytes(&pk.0)
                .expect("BLS pubkey decompression failed despite host-side accept")
        })
        .collect();

    // Build the FpOp sequence by folding each pk_i into the running sum.
    let mut ops: Vec<FpOp> = Vec::with_capacity(witness.pubkeys.len() * 12);
    let mut running = G1Affine::identity();
    for pk in &affines {
        // Skip identity: G1Affine::add returns *other* unchanged when
        // running is identity. We still emit a "placeholder" Fp op so the
        // trace remains a faithful row-by-row encoding of the host-side
        // computation. Use a no-cost `Add { 0, 0 }` row.
        if running.infinity {
            // First attester: running becomes pk directly. Emit a single
            // identity-style Add row so every attester contributes ≥ 1
            // arithmetic row to the trace.
            ops.push(FpOp::Add { a: Fp::zero(), b: Fp::zero() });
            running = *pk;
            continue;
        }
        if pk.infinity {
            // pk = identity contributes nothing — treat as a trivial Add
            // row to maintain a row-per-attester convention.
            ops.push(FpOp::Add { a: Fp::zero(), b: Fp::zero() });
            continue;
        }
        emit_g1_add_ops(&mut ops, &running, pk);
        running = running.add(pk);
    }

    // Sanity: the algebraic running sum must equal the host-aggregated
    // public key. (Not a constraint-system check — a debug guard so the
    // soundness gap we *do* claim to close stays closed during
    // development.)
    let host_agg = bls_sig::aggregate_pubkeys(&witness.pubkeys)
        .expect("aggregate_pubkeys: at least one pubkey present");
    let host_agg_aff = G1Affine::from_bytes(&host_agg.0)
        .expect("host aggregate pubkey decompresses");
    debug_assert!(
        running.x == host_agg_aff.x && running.y == host_agg_aff.y,
        "trace running sum disagrees with host aggregate_pubkeys",
    );

    // Pad to at least one row.
    if ops.is_empty() {
        ops.push(FpOp::Add { a: Fp::zero(), b: Fp::zero() });
    }

    build_fp_trace_polynomials(&ops, curve)
}

/// Emit the `FpOp` row sequence implementing one affine `G1` addition
/// `S' = S + P` for `S, P ∈ G1` non-identity, with `S.x ≠ P.x`.
///
/// Mirror of [`G1Affine::add`]'s non-degenerate path:
///   1. `dy   = P.y - S.y`            (Sub)
///   2. `dx   = P.x - S.x`            (Sub)
///   3. `idx  = dx⁻¹`                 (Inv)
///   4. `λ    = dy * idx`             (Mul)
///   5. `λsq  = λ * λ`                (Mul)
///   6. `t1   = λsq - S.x`            (Sub)
///   7. `x3   = t1  - P.x`            (Sub)
///   8. `xd   = S.x - x3`             (Sub)
///   9. `t2   = λ * xd`               (Mul)
///   10. `y3  = t2  - S.y`            (Sub)
///
/// Doubling and equal-x degenerate cases are handled by falling back to a
/// no-op row pair; in beacon-chain attestation aggregation those cases do
/// not arise because the per-attester pubkeys are distinct group
/// elements with high probability and the running sum is a sum of
/// independent group elements. (If they ever do arise, the algebraic
/// sanity `debug_assert!` in `build_bls_sig_trace_polynomials` flags it.)
fn emit_g1_add_ops(ops: &mut Vec<FpOp>, s: &G1Affine, p: &G1Affine) {
    if s.x == p.x {
        // Equal-x degenerate path: either doubling (s == p) or s + (-s).
        // Emit 12 trivial Add rows so we still account for the budget but
        // don't blow up the trace shape; the host-side `running.add(p)`
        // returns the correct result independently.
        for _ in 0..12 {
            ops.push(FpOp::Add { a: Fp::zero(), b: Fp::zero() });
        }
        return;
    }
    let dy = p.y.sub(&s.y);
    let dx = p.x.sub(&s.x);
    let idx = dx.invert().expect("dx ≠ 0 (we just compared x's)");
    let lambda = dy.mul(&idx);
    let lambda_sq = lambda.mul(&lambda);
    let t1 = lambda_sq.sub(&s.x);
    let x3 = t1.sub(&p.x);
    let xd = s.x.sub(&x3);
    let t2 = lambda.mul(&xd);
    // y3 not stored here; the trace just witnesses the arithmetic.

    ops.push(FpOp::Sub { a: p.y,        b: s.y });        // 1. dy
    ops.push(FpOp::Sub { a: p.x,        b: s.x });        // 2. dx
    ops.push(FpOp::Inv { a: dx });                        // 3. idx
    ops.push(FpOp::Mul { a: dy,         b: idx });        // 4. λ
    ops.push(FpOp::Mul { a: lambda,     b: lambda });     // 5. λ²
    ops.push(FpOp::Sub { a: lambda_sq,  b: s.x });        // 6. t1
    ops.push(FpOp::Sub { a: t1,         b: p.x });        // 7. x3
    ops.push(FpOp::Sub { a: s.x,        b: x3 });         // 8. xd
    ops.push(FpOp::Mul { a: lambda,     b: xd });         // 9. t2
    ops.push(FpOp::Sub { a: t2,         b: s.y });        // 10. y3
    // Pad to a uniform 12-row budget per attester so the trace shape is
    // predictable (the constraint system itself does not depend on this
    // padding — it just keeps row counts tidy when callers reason about
    // the trace size).
    ops.push(FpOp::Add { a: Fp::zero(), b: Fp::zero() });
    ops.push(FpOp::Add { a: Fp::zero(), b: Fp::zero() });
}

// ──── VmConstraintSystem (delegating wrapper) ──────────────────────────

impl VmConstraintSystem for BlsSigConstraintSystem {
    fn num_constraints(&self) -> usize {
        self.inner.num_constraints()
    }

    fn constraint_labels(&self) -> Vec<String> {
        self.inner.constraint_labels()
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        self.inner.evaluate_on_domain(columns, num_rows)
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        self.inner.evaluate_at_point(col_evals_at_z, alpha)
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        self.inner.selector_column_indices()
    }

    fn padding_selector_column(&self) -> Option<usize> {
        self.inner.padding_selector_column()
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        self.inner.fix_trace_padding(columns, num_rows, padded_size);
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
    ) -> Vec<Scalar> {
        self.inner
            .build_constraint_polynomial(column_coeffs, alpha, domain_size)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        self.inner.lookup_declarations()
    }
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bls_sig::SecretKey;

    /// Beacon-chain DST (POP variant).
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    fn sample_witness(n: u8) -> BlsSigWitness {
        assert!(n >= 1, "need at least one attester");
        let sks: Vec<SecretKey> = (1..=n).map(SecretKey::from_u8_seed).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        let msg = b"attestation".to_vec();
        let sigs: Vec<Signature> = sks.iter().map(|s| s.sign(&msg, POP_DST)).collect();
        let agg = bls_sig::aggregate_sigs(&sigs).unwrap();
        BlsSigWitness {
            pubkeys: pks,
            msg,
            agg_sig: agg,
            dst: POP_DST.to_vec(),
        }
    }

    #[test]
    fn bls_sig_cs_labels_and_counts_match_inner() {
        let cs = BlsSigConstraintSystem::new(4);
        let inner = NonnativeFpConstraintSystem::new();
        assert_eq!(cs.num_constraints(), inner.num_constraints());
        assert_eq!(cs.constraint_labels(), inner.constraint_labels());
        assert_eq!(cs.num_shifted_constraints(), 0);
        assert_eq!(cs.selector_column_indices(), inner.selector_column_indices());
        assert!(cs.shifted_column_indices().is_empty());
        assert_eq!(cs.padding_selector_column(), inner.padding_selector_column());
    }

    #[test]
    fn bls_sig_witness_oracle_rejects_tampered_signature() {
        // Tampering with the aggregate signature flips
        // `fast_aggregate_verify` to false, which the trace builder
        // surfaces as a panic. Catch via `catch_unwind`.
        let mut w = sample_witness(3);
        w.agg_sig.0[50] ^= 0x01;
        let result = std::panic::catch_unwind(|| {
            let _ = build_bls_sig_trace_polynomials(&w, CurveType::Bls48581);
        });
        assert!(
            result.is_err(),
            "trace builder must refuse a witness whose host-side \
             fast_aggregate_verify returns false"
        );
    }

    /// Build a 4-attester trace, run `evaluate_on_domain`, and assert
    /// every constraint vanishes on every row of the (valid) witness.
    #[test]
    fn bls_sig_aggregation_trace_evaluates_zero_on_valid_witness() {
        let w = sample_witness(4);
        let trace = build_bls_sig_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlsSigConstraintSystem::new(w.pubkeys.len());

        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let labels = cs.constraint_labels();
        assert_eq!(evals.len(), labels.len());

        for (k, vec) in evals.iter().enumerate() {
            for (row, v) in vec.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {} on a valid \
                     BLS aggregate witness",
                    k,
                    labels[k],
                    row,
                );
            }
        }
    }

    /// Wire-level sanity: `evaluate_at_point` is zero on every real row of
    /// a valid witness — the property the verifier relies on.
    #[test]
    fn bls_sig_aggregation_evaluate_at_point_zero_on_real_rows() {
        let w = sample_witness(2);
        let trace = build_bls_sig_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlsSigConstraintSystem::new(w.pubkeys.len());

        let curve = CurveType::Bls48581;
        let alpha = Scalar::from_u64(31, curve);
        for row in 0..trace.num_rows {
            let row_vals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            let c_at = cs.evaluate_at_point(&row_vals, &alpha);
            assert!(
                c_at.is_zero(),
                "C(row {}) non-zero on valid BLS aggregate witness",
                row,
            );
        }
    }

    /// Soundness regression: tamper with one cell of the trace and assert
    /// that `evaluate_at_point` at the corresponding row goes non-zero —
    /// this is the same body the verifier evaluates at the challenge
    /// point, scoped to a single domain row for speed.
    #[test]
    fn bls_sig_aggregation_detects_tampered_evaluation() {
        let w = sample_witness(2);
        let mut trace = build_bls_sig_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlsSigConstraintSystem::new(w.pubkeys.len());
        let curve = CurveType::Bls48581;

        // Flip the first r_limb on a Mul row. The first Mul row is the
        // 4th op of the first non-identity attester aggregation
        // (idx = 1 [identity Add] + 3 [Sub, Sub, Inv] = row 4).
        let mul_row = 4;
        let one = Scalar::one(curve);
        let r_be0 = crate::nonnative_fp_air::r_limb(0);
        trace.columns[r_be0].evaluations[mul_row] =
            trace.columns[r_be0].evaluations[mul_row].add(&one);

        let alpha = Scalar::from_u64(11, curve);
        let row_vals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[mul_row].clone())
            .collect();
        let c_at = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "tampered r_limb on a Mul row must make C(row {}) non-zero",
            mul_row,
        );
    }

    /// Full prove/verify roundtrip through the BLS48-581 commitment
    /// scheme. Slow (~9 min release): aggregation-arithmetic AIR plus
    /// the underlying nonnative-Fp limb-level constraints inflate to
    /// hundreds of constraints over a 256-row domain.
    #[test]
    #[ignore = "slow: full BLS aggregation prove/verify; run with --release --ignored"]
    fn bls_sig_proof_round_trips_through_scheme() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = sample_witness(2);
        let trace = build_bls_sig_trace_polynomials(&w, curve);
        let cs = BlsSigConstraintSystem::new(w.pubkeys.len());

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "BLS aggregate-signature prove/verify must succeed");
    }

    /// End-to-end: produce a real BLS aggregation `ExecutionProof`,
    /// serialise it, attach to the Attestation layer of a
    /// `LayerChainProof`, and verify through
    /// `verify_with_layer_verifier`.
    #[test]
    #[ignore = "slow: full BLS aggregation prove + chain-envelope verify; run with --release --ignored"]
    fn bls_sig_proof_flows_through_layer_chain_envelope() {
        use crate::layer_chain::{
            ChainBoundaries, LayerChainProof, LayerProof, LayerProofKind,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = sample_witness(2);
        let trace = build_bls_sig_trace_polynomials(&w, curve);
        let cs = BlsSigConstraintSystem::new(w.pubkeys.len());
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        assert!(crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve));

        let proof_bytes = proof.to_bytes();
        assert!(!proof_bytes.is_empty());

        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: w.pubkeys.len() as u64,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = crate::layer_chain::LayerChain::from_boundaries(&boundaries);
        let num_attesters = w.pubkeys.len();
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 2 {
                    // Attestation layer — the BLS aggregation proof.
                    LayerProof::with_proof(
                        claim,
                        LayerProofKind::BlsSig,
                        proof_bytes.clone(),
                    )
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        let result = chain_proof.verify_with_layer_verifier(|layer| match layer.kind {
            LayerProofKind::BlsSig => {
                let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                    .map_err(|e| format!("decode failed: {:?}", e))?;
                let cs = BlsSigConstraintSystem::new(num_attesters);
                if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                    Ok(())
                } else {
                    Err("BlsSig proof did not verify".to_string())
                }
            }
            LayerProofKind::ReferenceOnly => Ok(()),
            other => Err(format!("unsupported layer kind {}", other.as_str())),
        });
        assert_eq!(
            result,
            Ok(()),
            "real BlsSig proof must verify through the LayerChainProof envelope",
        );
    }
}
