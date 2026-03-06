//! Wrapper around BLS48-581 KZG commitment functions from the `bls48581` crate.
//!
//! This module provides a typed interface over the raw KZG polynomial commitment
//! operations implemented by the `bls48581` crate. All commitment, opening, and
//! verification functions delegate to the underlying BLS48-581 pairing-based
//! cryptographic primitives.
//!
//! # Initialization
//!
//! [`init()`] **must** be called exactly once before any other function in this
//! module. It sets up the global BLS48-581 constants (roots of unity, ceremony
//! points, etc.) that every subsequent operation depends on.

use bls48581;
use bls48581::bls;
use bls48581::bls48581::big;
use bls48581::bls48581::ecp;
use bls48581::bls48581::ecp8;
use bls48581::bls48581::pair8;
use bls48581::bls48581::rom;

/// A KZG polynomial commitment (74 bytes, compressed BLS48-581 G1 point).
#[derive(Clone, Debug)]
pub struct Commitment(pub Vec<u8>);

/// A KZG opening proof.
#[derive(Clone, Debug)]
pub struct OpeningProof(pub Vec<u8>);

/// Batch opening proof for multiple polynomials at multiple points.
#[derive(Clone, Debug)]
pub struct BatchProof {
    pub d: Vec<u8>,
    pub proof: Vec<u8>,
}

/// Initialize the BLS48-581 commitment scheme.
///
/// This sets up the global tables (roots of unity, ceremony SRS points, etc.)
/// required by every other function in this module. It **must** be called once
/// before any commitment, opening, or verification operation.
pub fn init() {
    bls48581::init();
}

/// Commit to a polynomial given as evaluation data (full-width MODBYTES per scalar).
///
/// `data` contains the polynomial evaluations serialized as bytes (MODBYTES = 73
/// bytes per scalar, big-endian). `poly_size` is the domain size (must be a
/// power of 2 and must match a precomputed FFT width from the ceremony SRS).
///
/// Returns a [`Commitment`] wrapping the 74-byte compressed G1 point.
pub fn commit(data: &[u8], poly_size: u64) -> Commitment {
    Commitment(bls48581::commit_raw_full(data, poly_size))
}

/// Commit to a polynomial given directly as BIG scalars in evaluation form.
///
/// No byte serialization — uses the full-precision BIG values directly.
pub fn commit_from_scalars(scalars: &[big::BIG], poly_size: u64) -> Commitment {
    Commitment(bls48581::commit_scalars(scalars, poly_size))
}

/// Commit to a polynomial given as BIG scalars in coefficient form.
///
/// Uses the monomial-basis SRS directly, avoiding an FFT to convert
/// to evaluation form.
pub fn commit_from_coeffs(coeffs: &[big::BIG]) -> Commitment {
    Commitment(bls48581::commit_scalars_monomial(coeffs))
}

/// Create an opening proof at a given evaluation index.
///
/// `data` is the same evaluation-form byte blob passed to [`commit`].
/// `index` is the position in the evaluation domain (0-based).
/// `poly_size` is the domain size used when committing.
///
/// Returns an [`OpeningProof`] wrapping a 74-byte compressed G1 point.
pub fn open(data: &[u8], index: u64, poly_size: u64) -> OpeningProof {
    OpeningProof(bls48581::prove_raw(data, index, poly_size))
}

/// Verify a single opening.
///
/// `data` is the claimed evaluation value (as raw bytes).
/// `commitment` is the polynomial commitment to verify against.
/// `index` is the evaluation-domain index that was opened.
/// `proof` is the opening proof produced by [`open`].
/// `poly_size` is the domain size.
///
/// Returns `true` if the proof is valid.
pub fn verify_opening(
    data: &[u8],
    commitment: &Commitment,
    index: u64,
    proof: &OpeningProof,
    poly_size: u64,
) -> bool {
    bls48581::verify_raw(data, &commitment.0, index, &proof.0, poly_size)
}

/// Create a batch opening proof for multiple polynomials at multiple indices.
///
/// This implements the multi-opening protocol from the `bls48581` crate which
/// uses a Fiat-Shamir combined quotient approach.
///
/// * `commitments` - one commitment per polynomial.
/// * `polys` - the evaluation-form byte blobs (one per polynomial).
/// * `indices` - the evaluation-domain index to open for each polynomial.
/// * `poly_size` - the shared domain size for all polynomials.
pub fn batch_open(
    commitments: &[Commitment],
    polys: &[Vec<u8>],
    indices: &[u64],
    poly_size: u64,
) -> BatchProof {
    let commit_refs: Vec<Vec<u8>> = commitments.iter().map(|c| c.0.clone()).collect();
    let mp = bls48581::prove_multiple(&commit_refs, &polys.to_vec(), &indices.to_vec(), poly_size);
    BatchProof {
        d: mp.d,
        proof: mp.proof,
    }
}

/// Verify a batch opening proof.
///
/// * `commitments` - the commitments that were batch-opened.
/// * `y_values` - the claimed evaluation values (one byte blob per polynomial).
/// * `indices` - the evaluation-domain indices that were opened.
/// * `poly_size` - the shared domain size.
/// * `batch_proof` - the proof produced by [`batch_open`].
///
/// Returns `true` if all openings are valid.
pub fn batch_verify(
    commitments: &[Commitment],
    y_values: &[Vec<u8>],
    indices: &[u64],
    poly_size: u64,
    batch_proof: &BatchProof,
) -> bool {
    let commit_refs: Vec<Vec<u8>> = commitments.iter().map(|c| c.0.clone()).collect();
    bls48581::verify_multiple(
        &commit_refs,
        &y_values.to_vec(),
        &indices.to_vec(),
        poly_size,
        &batch_proof.d,
        &batch_proof.proof,
    )
}

/// Serialize a BIG scalar to full-width bytes (MODBYTES = 73).
pub fn big_to_bytes(val: &big::BIG) -> [u8; big::MODBYTES] {
    let mut buf = [0u8; big::MODBYTES];
    val.tobytes(&mut buf);
    buf
}

/// Serialize a vector of BIG scalars to bytes (MODBYTES per scalar).
pub fn poly_to_bytes(poly: &[big::BIG]) -> Vec<u8> {
    let mut result = Vec::with_capacity(poly.len() * big::MODBYTES);
    for scalar in poly {
        let mut buf = [0u8; big::MODBYTES];
        scalar.tobytes(&mut buf);
        result.extend_from_slice(&buf);
    }
    result
}

/// Convert evaluation-form polynomial to coefficient form via inverse FFT.
pub fn eval_to_coeff(eval_form: &[big::BIG], domain_size: u64) -> Vec<big::BIG> {
    bls48581::fft(eval_form, domain_size, true)
        .expect("inverse FFT should succeed")
}

/// Evaluate polynomial at an arbitrary field element z using Horner's method.
/// Takes coefficient-form polynomial.
pub fn eval_poly_at(coeffs: &[big::BIG], z: &big::BIG) -> big::BIG {
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    // Horner's method: p(z) = c_0 + z*(c_1 + z*(c_2 + ...))
    let mut result = big::BIG::new();
    for i in (0..coeffs.len()).rev() {
        result = big::BIG::modmul(&result, z, &modulus);
        result = big::BIG::modadd(&result, &coeffs[i], &modulus);
    }
    result
}

/// Compute (p(x) - p(z)) / (x - z) in coefficient form via synthetic division.
/// Returns the quotient polynomial coefficients.
pub fn div_by_linear(coeffs: &[big::BIG], z: &big::BIG) -> Vec<big::BIG> {
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    let n = coeffs.len();
    if n == 0 {
        return vec![];
    }
    // After subtracting p(z), the constant term becomes 0 when dividing by (x - z).
    // Synthetic division: q_{n-2} = c_{n-1}, then q_{i-1} = c_i + z * q_i
    let mut quotient = vec![big::BIG::new(); n - 1];
    if n == 1 {
        return vec![];
    }
    quotient[n - 2] = big::BIG::new_copy(&coeffs[n - 1]);
    for i in (0..n - 2).rev() {
        // q[i] = c[i+1] + z * q[i+1]
        let zq = big::BIG::modmul(z, &quotient[i + 1], &modulus);
        quotient[i] = big::BIG::modadd(&coeffs[i + 1], &zq, &modulus);
    }
    quotient
}

/// Open polynomial at an arbitrary field element z.
/// Takes evaluation-form polynomial. Returns (y = p(z), proof π as 74-byte G1 point).
pub fn open_at_point(
    eval_form: &[big::BIG],
    z: &big::BIG,
    domain_size: u64,
) -> (big::BIG, Vec<u8>) {
    // Convert to coefficient form
    let coeffs = eval_to_coeff(eval_form, domain_size);

    // Evaluate at z
    let y = eval_poly_at(&coeffs, z);

    // Compute (p(x) - y) / (x - z) = (p(x) - p(z)) / (x - z)
    // First subtract y from the constant term
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    let mut shifted = coeffs.clone();
    shifted[0] = big::BIG::modadd(&shifted[0], &big::BIG::modneg(&y, &modulus), &modulus);

    // Synthetic division by (x - z)
    let quotient_coeffs = div_by_linear(&shifted, z);

    // Commit quotient polynomial in coefficient form via monomial SRS.
    // This avoids an FFT round-trip and uses full-precision scalars.
    let proof = bls48581::commit_scalars_monomial(&quotient_coeffs);

    (y, proof)
}

/// Verify a KZG opening at an arbitrary field element z.
///
/// Checks: e(C - y*G1, G2) == e(π, [τ]₂ - z*G2)
/// Equivalently: e(C - y*G1, G2) * e(-π, [τ]₂ - z*G2) == 1
pub fn verify_at_point(
    commitment: &[u8],
    z: &big::BIG,
    y: &big::BIG,
    proof: &[u8],
) -> bool {
    let s = bls::singleton();

    // Parse commitment and proof as G1 points
    let c = ecp::ECP::frombytes(commitment);
    let pi = ecp::ECP::frombytes(proof);

    if c.is_infinity() || pi.is_infinity() {
        return false;
    }

    // Compute C - y * G1
    let g1 = ecp::ECP::generator();
    let y_g1 = g1.mul(y);
    let mut lhs_g1 = c;
    lhs_g1.sub(&y_g1);
    lhs_g1.affine();

    // Compute [τ]₂ - z * G2
    let tau_g2 = s.CeremonyBLS48581G2[1].clone();
    let g2_gen = ecp8::ECP8::generator();
    let z_g2 = g2_gen.mul(z);
    let mut rhs_g2 = tau_g2;
    rhs_g2.sub(&z_g2);
    rhs_g2.affine();

    // Pairing check: e(C - y*G1, G2) == e(π, [τ]₂ - z*G2)
    // Equivalently: e(C - y*G1, G2) * e(-π, [τ]₂ - z*G2) == 1
    let mut r = pair8::initmp();
    pair8::another(&mut r, &ecp8::ECP8::generator(), &lhs_g1);
    let mut neg_pi = pi;
    neg_pi.neg();
    pair8::another(&mut r, &rhs_g2, &neg_pi);
    let mut v = pair8::miller(&mut r);
    v = pair8::fexp(&v);
    v.isunity()
}

/// Batch open multiple polynomials at the same point z.
///
/// Uses random linear combination: draw β from transcript,
/// combine f(x) = Σ β^i * p_i(x), compute combined opening.
/// Returns (evaluations, combined_proof).
pub fn batch_open_at_point(
    eval_forms: &[&Vec<big::BIG>],
    z: &big::BIG,
    domain_size: u64,
    beta: &big::BIG,
) -> (Vec<big::BIG>, Vec<u8>) {
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    let n = eval_forms.len();

    // Evaluate each polynomial at z and combine coefficients
    let mut evaluations = Vec::with_capacity(n);
    let mut combined_coeffs = vec![big::BIG::new(); domain_size as usize];
    let mut beta_power = big::BIG::new_int(1);

    for poly_evals in eval_forms {
        let coeffs = eval_to_coeff(poly_evals, domain_size);
        let y = eval_poly_at(&coeffs, z);
        evaluations.push(y);

        // Accumulate β^i * p_i(x) in coefficient form
        for j in 0..coeffs.len() {
            let term = big::BIG::modmul(&beta_power, &coeffs[j], &modulus);
            combined_coeffs[j] = big::BIG::modadd(&combined_coeffs[j], &term, &modulus);
        }
        beta_power = big::BIG::modmul(&beta_power, beta, &modulus);
    }

    // Combined evaluation y_f = Σ β^i * y_i
    let mut combined_y = big::BIG::new();
    beta_power = big::BIG::new_int(1);
    for y in &evaluations {
        let term = big::BIG::modmul(&beta_power, y, &modulus);
        combined_y = big::BIG::modadd(&combined_y, &term, &modulus);
        beta_power = big::BIG::modmul(&beta_power, beta, &modulus);
    }

    // Compute (f(x) - f(z)) / (x - z)
    combined_coeffs[0] = big::BIG::modadd(
        &combined_coeffs[0],
        &big::BIG::modneg(&combined_y, &modulus),
        &modulus,
    );
    let quotient_coeffs = div_by_linear(&combined_coeffs, z);

    // Commit quotient polynomial in coefficient form via monomial SRS.
    let proof = bls48581::commit_scalars_monomial(&quotient_coeffs);

    (evaluations, proof)
}

/// Verify a batch opening at an arbitrary field element z.
///
/// Uses the same random linear combination as batch_open_at_point:
/// C_combined = Σ β^i * C_i, y_combined = Σ β^i * y_i, single pairing check.
pub fn batch_verify_at_point(
    commitments: &[&[u8]],
    evaluations: &[big::BIG],
    z: &big::BIG,
    beta: &big::BIG,
    proof: &[u8],
) -> bool {
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);

    // Combine commitments: C_f = Σ β^i * C_i
    let mut combined_c = ecp::ECP::new();
    let mut beta_power = big::BIG::new_int(1);
    for comm_bytes in commitments {
        let c = ecp::ECP::frombytes(comm_bytes);
        let scaled = c.mul(&beta_power);
        combined_c.add(&scaled);
        beta_power = big::BIG::modmul(&beta_power, beta, &modulus);
    }
    combined_c.affine();

    // Combine evaluations: y_f = Σ β^i * y_i
    let mut combined_y = big::BIG::new();
    beta_power = big::BIG::new_int(1);
    for y in evaluations {
        let term = big::BIG::modmul(&beta_power, y, &modulus);
        combined_y = big::BIG::modadd(&combined_y, &term, &modulus);
        beta_power = big::BIG::modmul(&beta_power, beta, &modulus);
    }

    // Serialize combined commitment
    let mut combined_c_bytes = vec![0u8; 74];
    combined_c.tobytes(&mut combined_c_bytes, true);

    // Verify using the single-point verifier
    verify_at_point(&combined_c_bytes, z, &combined_y, proof)
}
