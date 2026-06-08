//! Commitment scheme abstraction.
//!
//! Defines the [`CommitmentScheme`] trait that abstracts KZG polynomial
//! commitment operations. Implementations exist for:
//! - BLS48-581 (via the `bls48581` crate's ceremony SRS)
//! - BLS12-381 (via `blst` with Ethereum KZG trusted setup)

pub mod bls48581_scheme;
pub mod bls12381_scheme;

use crate::field::Scalar;

/// Serialized G1 point (commitment or proof).
#[derive(Clone, Debug)]
pub struct CommitmentPoint(pub Vec<u8>);

/// Result of a batch opening operation.
#[derive(Clone, Debug)]
pub struct BatchOpenResult {
    pub evaluations: Vec<Scalar>,
    pub proof: Vec<u8>,
}

/// Accumulated claim for recursive proof composition.
#[derive(Clone, Debug)]
pub struct AccumulatedClaim {
    /// L_acc compressed G1 point bytes.
    pub l_acc: Vec<u8>,
    /// R_acc compressed G1 point bytes.
    pub r_acc: Vec<u8>,
    /// Number of proofs folded so far.
    pub num_folded: u64,
}

/// Description of a single batched KZG opening within an
/// [`crate::prover::ExecutionProof`]. Each opening conceptually verifies
/// `e(combined_C - combined_y · G1, G2) == e(π, [τ]₂ - point · G2)` where
/// `combined_C = Σ β^i · C_i` and `combined_y = Σ β^i · y_i` over the
/// listed `commitments` / `evaluations`.
///
/// The cross-opening fold combines all openings of one chunk into a
/// single `(L_chunk, R_chunk)` via a meta-challenge ξ.
#[derive(Clone, Debug)]
pub struct OpeningSpec<'a> {
    /// Commitments contributing to this opening (compressed G1 bytes).
    pub commitments: Vec<&'a [u8]>,
    /// Per-commitment evaluations at `point`.
    pub evaluations: Vec<Scalar>,
    /// KZG opening proof π (compressed G1 bytes).
    pub proof: &'a [u8],
    /// The point at which the opening is taken — typically `z` or `ω·z`.
    pub point: Scalar,
    /// The per-opening RLC challenge β used to combine the entries.
    pub beta: Scalar,
}

/// Abstract polynomial commitment scheme.
///
/// Implementations provide curve-specific KZG operations while the proving
/// pipeline remains generic. All byte-serialized points use compressed format.
pub trait CommitmentScheme: Send + Sync {
    /// Initialize global state (SRS loading, precomputation).
    fn init(&self);

    /// Size of a compressed G1 point in bytes.
    fn g1_compressed_size(&self) -> usize;

    /// Maximum supported domain size (number of evaluation points).
    fn max_domain_size(&self) -> u64;

    // FFT operations

    /// Inverse FFT: evaluation form → coefficient form.
    fn ifft(&self, evals: &[Scalar], domain_size: u64) -> Vec<Scalar>;

    /// Forward FFT: coefficient form → evaluation form.
    fn fft(&self, coeffs: &[Scalar], domain_size: u64) -> Vec<Scalar>;

    // Commit operations

    /// Commit to a polynomial in evaluation form.
    fn commit_evaluations(&self, evals: &[Scalar], domain_size: u64) -> Vec<u8>;

    /// Commit to a polynomial in coefficient form (monomial SRS).
    fn commit_coefficients(&self, coeffs: &[Scalar]) -> Vec<u8>;

    // Opening operations

    /// Evaluate polynomial (in coefficient form) at arbitrary point z via Horner.
    fn eval_poly_at(&self, coeffs: &[Scalar], z: &Scalar) -> Scalar;

    /// Synthetic division: (p(x) - p(z)) / (x - z).
    fn div_by_linear(&self, coeffs: &[Scalar], z: &Scalar) -> Vec<Scalar>;

    /// Open polynomial at arbitrary point z.
    /// Takes evaluation-form polynomial.
    /// Returns (y = p(z), proof as compressed G1 bytes).
    fn open_at_point(&self, eval_form: &[Scalar], z: &Scalar, domain_size: u64) -> (Scalar, Vec<u8>);

    /// Verify single opening: e(C - y*G1, G2) == e(π, [τ]₂ - z*G2).
    fn verify_at_point(&self, commitment: &[u8], z: &Scalar, y: &Scalar, proof: &[u8]) -> bool;

    /// Batch verify: combine commitments/evaluations with β, single pairing check.
    fn batch_verify_at_point(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> bool;

    // Recursive accumulator operations

    /// Create an identity (empty) accumulator.
    fn initial_accumulator(&self) -> AccumulatedClaim;

    /// Compute L and R pairing arguments from commitments, evaluations, z, β, proof.
    fn compute_lr(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> (Vec<u8>, Vec<u8>);

    /// Aggregate `(L, R)` across multiple batched KZG openings into one
    /// pair via the meta-challenge ξ. For each opening k:
    /// `L_k = combined_C_k − combined_y_k · G1 + point_k · π_k`,
    /// `R_k = π_k`. Then `L_chunk = Σ ξ^k · L_k`, `R_chunk = Σ ξ^k · R_k`.
    ///
    /// Default implementation builds on [`Self::compute_lr`] +
    /// [`Self::fold_accumulator`]; concrete schemes may override for
    /// constant-factor speedups.
    fn compute_lr_multi(
        &self,
        openings: &[OpeningSpec<'_>],
        xi: &Scalar,
    ) -> (Vec<u8>, Vec<u8>) {
        let init = self.initial_accumulator();
        let mut l_acc = init.l_acc;
        let mut r_acc = init.r_acc;
        let mut xi_power = Scalar::one(xi.curve_type());
        for opening in openings {
            let (l_k, r_k) = self.compute_lr(
                &opening.commitments,
                &opening.evaluations,
                &opening.point,
                &opening.beta,
                opening.proof,
            );
            let (new_l, new_r) = self.fold_accumulator(
                &l_acc, &r_acc, &l_k, &r_k, &xi_power,
            );
            l_acc = new_l;
            r_acc = new_r;
            xi_power = xi_power.mul(xi);
        }
        (l_acc, r_acc)
    }

    /// Fold two accumulators: L_new = L_left + challenge * L_right, etc.
    fn fold_accumulator(
        &self,
        l_left: &[u8],
        r_left: &[u8],
        l_right: &[u8],
        r_right: &[u8],
        challenge: &Scalar,
    ) -> (Vec<u8>, Vec<u8>);

    /// Final verification: e(L_acc, G2) == e(R_acc, [τ]₂).
    fn verify_accumulated(&self, l_acc: &[u8], r_acc: &[u8]) -> bool;

    /// Return the primitive n-th root of unity ω for the given domain size.
    ///
    /// ω generates the multiplicative subgroup of order `domain_size`:
    /// ω^domain_size = 1 and ω^k ≠ 1 for 0 < k < domain_size.
    /// Used for cross-row (shifted) constraints where column values at ω·X
    /// represent the "next row" in the evaluation domain.
    fn domain_generator(&self, domain_size: u64) -> Scalar;
}
