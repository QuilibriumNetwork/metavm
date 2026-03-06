//! BLS12-381 implementation of the CommitmentScheme trait.
//!
//! Uses the `blst` crate for BLS12-381 arithmetic and the Ethereum KZG
//! trusted setup ceremony data (4096 G1 powers of tau, 65 G2 powers).
//!
//! The trusted setup is embedded as a binary file at compile time and
//! deserialized on first use via `init()`.

use crate::field::{Scalar, CurveType};
use crate::scheme::{CommitmentScheme, AccumulatedClaim};
use blst::*;

/// Binary trusted setup data (Ethereum KZG ceremony).
/// Format: [4B num_g1 LE][4B num_g2 LE][48*num_g1 G1 monomial][48*num_g1 G1 lagrange][96*num_g2 G2 monomial]
static TRUSTED_SETUP_BIN: &[u8] = include_bytes!("trusted_setup.bin");

/// Global SRS state, initialized once.
static SRS: std::sync::OnceLock<Srs> = std::sync::OnceLock::new();

/// Structured Reference String (SRS) loaded from the Ethereum KZG ceremony.
struct Srs {
    /// G1 monomial-form points: [G1, τ·G1, τ²·G1, ...] in affine form.
    /// Used for committing to polynomials in coefficient form.
    g1_monomial: Vec<blst_p1_affine>,
    /// G1 Lagrange-form points: [L_0(τ)·G1, L_1(τ)·G1, ...] in affine form.
    /// Used for committing to polynomials in evaluation form.
    g1_lagrange: Vec<blst_p1_affine>,
    /// G2 points: [G2, τ·G2, ...] in affine form (monomial form).
    g2_affine: Vec<blst_p2_affine>,
    /// Number of G1 points (4096 for Ethereum).
    num_g1: usize,
}

fn load_srs() -> Srs {
    let data = TRUSTED_SETUP_BIN;
    assert!(data.len() >= 8, "Trusted setup too small");

    let num_g1 = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let num_g2 = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

    let expected_len = 8 + 48 * num_g1 * 2 + 96 * num_g2;
    assert_eq!(data.len(), expected_len, "Trusted setup size mismatch: got {} expected {}", data.len(), expected_len);

    // G1 monomial points come first
    let mut g1_monomial = Vec::with_capacity(num_g1);
    let g1_mono_start = 8;
    for i in 0..num_g1 {
        let offset = g1_mono_start + i * 48;
        let mut pt = blst_p1_affine::default();
        let err = unsafe { blst_p1_uncompress(&mut pt, data[offset..].as_ptr()) };
        assert_eq!(err, BLST_ERROR::BLST_SUCCESS, "G1 monomial point {} decompression failed", i);
        g1_monomial.push(pt);
    }

    // G1 Lagrange points come next
    let mut g1_lagrange = Vec::with_capacity(num_g1);
    let g1_lagr_start = g1_mono_start + num_g1 * 48;
    for i in 0..num_g1 {
        let offset = g1_lagr_start + i * 48;
        let mut pt = blst_p1_affine::default();
        let err = unsafe { blst_p1_uncompress(&mut pt, data[offset..].as_ptr()) };
        assert_eq!(err, BLST_ERROR::BLST_SUCCESS, "G1 Lagrange point {} decompression failed", i);
        g1_lagrange.push(pt);
    }

    // G2 monomial points
    let g2_start = g1_lagr_start + num_g1 * 48;
    let mut g2_affine = Vec::with_capacity(num_g2);
    for i in 0..num_g2 {
        let offset = g2_start + i * 96;
        let mut pt = blst_p2_affine::default();
        let err = unsafe { blst_p2_uncompress(&mut pt, data[offset..].as_ptr()) };
        assert_eq!(err, BLST_ERROR::BLST_SUCCESS, "G2 point {} decompression failed", i);
        g2_affine.push(pt);
    }

    Srs { g1_monomial, g1_lagrange, g2_affine, num_g1 }
}

fn srs() -> &'static Srs {
    SRS.get_or_init(load_srs)
}

// ============================================================================
// FFT over BLS12-381 scalar field
// ============================================================================

/// BLS12-381 scalar field order r - 1 has a factor of 2^32.
/// This means we have 2^32-th roots of unity.
/// Generator for the 2^32 subgroup: ω = 7^((r-1)/2^32) mod r
/// (Ethereum consensus specs use PRIMITIVE_ROOT_OF_UNITY = 7)
const MAX_LOG_DOMAIN: u32 = 32;

/// Compute ω = generator of the 2^k subgroup of the BLS12-381 scalar field.
/// Uses the fact that 7 is the primitive root specified by Ethereum.
pub(crate) fn root_of_unity(log_n: u32) -> blst_fr {
    assert!(log_n <= MAX_LOG_DOMAIN, "Domain size 2^{} exceeds max 2^{}", log_n, MAX_LOG_DOMAIN);

    // r - 1 = 2^32 * t where t is odd.
    // ω_{2^32} = 7^t mod r is a primitive 2^32-th root of unity.
    // ω_{2^k} = ω_{2^32}^(2^(32-k)) is a primitive 2^k-th root.

    // The 2^32-th root of unity for BLS12-381:
    // This is 7^((r-1)/2^32) mod r, where r is the BLS12-381 scalar field order.
    // Precomputed value (hex, big-endian):
    // 0x16a2a19edfe81f20d09b681922c813b4b63683508c2280b93829971f439f0d2b
    let root_2_32_bytes: [u8; 32] = [
        0x16, 0xa2, 0xa1, 0x9e, 0xdf, 0xe8, 0x1f, 0x20,
        0xd0, 0x9b, 0x68, 0x19, 0x22, 0xc8, 0x13, 0xb4,
        0xb6, 0x36, 0x83, 0x50, 0x8c, 0x22, 0x80, 0xb9,
        0x38, 0x29, 0x97, 0x1f, 0x43, 0x9f, 0x0d, 0x2b,
    ];

    let mut scalar = blst_scalar::default();
    unsafe { blst_scalar_from_bendian(&mut scalar, root_2_32_bytes.as_ptr()); }
    let mut omega = blst_fr::default();
    unsafe { blst_fr_from_scalar(&mut omega, &scalar); }

    // Square (32 - log_n) times to get the 2^log_n root
    for _ in 0..(MAX_LOG_DOMAIN - log_n) {
        let mut tmp = blst_fr::default();
        unsafe { blst_fr_sqr(&mut tmp, &omega); }
        omega = tmp;
    }

    omega
}

/// In-place radix-2 DIT FFT over BLS12-381 scalars.
pub(crate) fn fft_in_place(vals: &mut [blst_fr], inverse: bool) {
    let n = vals.len();
    assert!(n.is_power_of_two(), "FFT size must be power of 2, got {}", n);
    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros();
    let omega = root_of_unity(log_n);

    // If inverse, use ω^{-1}
    let omega = if inverse {
        fr_inverse(&omega)
    } else {
        omega
    };

    // Bit-reversal permutation
    {
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                vals.swap(i, j);
            }
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // w_len = omega^(n/len) = step through roots
        let step_exp = n / len;
        let w_len = fr_pow(&omega, step_exp as u64);

        let mut k = 0;
        while k < n {
            let mut w = fr_one();
            for j in 0..half {
                let u = vals[k + j];
                let mut v = blst_fr::default();
                unsafe { blst_fr_mul(&mut v, &vals[k + j + half], &w); }

                let mut sum = blst_fr::default();
                let mut diff = blst_fr::default();
                unsafe {
                    blst_fr_add(&mut sum, &u, &v);
                    blst_fr_sub(&mut diff, &u, &v);
                }
                vals[k + j] = sum;
                vals[k + j + half] = diff;

                let mut next_w = blst_fr::default();
                unsafe { blst_fr_mul(&mut next_w, &w, &w_len); }
                w = next_w;
            }
            k += len;
        }
        len <<= 1;
    }

    // If inverse, divide by n
    if inverse {
        let n_inv = fr_inverse(&fr_from_u64(n as u64));
        for v in vals.iter_mut() {
            let mut tmp = blst_fr::default();
            unsafe { blst_fr_mul(&mut tmp, v, &n_inv); }
            *v = tmp;
        }
    }
}

// ============================================================================
// blst_fr helper functions
// ============================================================================

fn fr_one() -> blst_fr {
    fr_from_u64(1)
}

fn fr_from_u64(val: u64) -> blst_fr {
    let mut scalar = blst_scalar::default();
    let mut fr = blst_fr::default();
    unsafe {
        blst_scalar_from_uint64(&mut scalar, [val, 0, 0, 0].as_ptr());
        blst_fr_from_scalar(&mut fr, &scalar);
    }
    fr
}

fn fr_pow(base: &blst_fr, exp: u64) -> blst_fr {
    let mut result = fr_one();
    let mut b = *base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            let mut tmp = blst_fr::default();
            unsafe { blst_fr_mul(&mut tmp, &result, &b); }
            result = tmp;
        }
        let mut tmp = blst_fr::default();
        unsafe { blst_fr_sqr(&mut tmp, &b); }
        b = tmp;
        e >>= 1;
    }
    result
}

fn fr_inverse(a: &blst_fr) -> blst_fr {
    let mut result = blst_fr::default();
    unsafe { blst_fr_inverse(&mut result, a); }
    result
}

fn fr_to_scalar(fr: &blst_fr) -> blst_scalar {
    let mut scalar = blst_scalar::default();
    unsafe { blst_scalar_from_fr(&mut scalar, fr); }
    scalar
}

#[cfg(test)]
fn fr_is_zero(a: &blst_fr) -> bool {
    let s = fr_to_scalar(a);
    s.b == [0u8; 32]
}

// ============================================================================
// BLS12-381 KZG Commitment Scheme
// ============================================================================

/// BLS12-381 KZG commitment scheme using the Ethereum trusted setup.
pub struct Bls12381Scheme;

impl Bls12381Scheme {
    pub fn new() -> Self {
        Bls12381Scheme
    }

    /// Convert Scalar to blst_fr.
    fn scalar_to_fr(s: &Scalar) -> blst_fr {
        *s.as_bls12381()
    }

    /// Convert slice of Scalars to blst_fr vec.
    fn scalars_to_frs(scalars: &[Scalar]) -> Vec<blst_fr> {
        scalars.iter().map(|s| Self::scalar_to_fr(s)).collect()
    }

    /// Convert blst_fr vec to Scalars.
    fn frs_to_scalars(frs: &[blst_fr]) -> Vec<Scalar> {
        frs.iter().map(|fr| Scalar::Bls12381(*fr)).collect()
    }

    /// Multi-scalar multiplication: Σ scalar_i * g1_i
    fn msm(points: &[blst_p1_affine], scalars_fr: &[blst_fr]) -> blst_p1 {
        let n = points.len().min(scalars_fr.len());
        if n == 0 {
            return blst_p1::default(); // point at infinity
        }

        // Convert blst_fr to byte arrays for the MSM API
        let scalar_bytes: Vec<u8> = scalars_fr[..n].iter().flat_map(|fr| {
            let s = fr_to_scalar(fr);
            s.b.to_vec()
        }).collect();

        // Use blst's high-level MultiPoint trait
        points[..n].mult(&scalar_bytes, 255)
    }

    /// Compress a blst_p1 to 48 bytes.
    fn compress_g1(p: &blst_p1) -> Vec<u8> {
        let mut out = [0u8; 48];
        unsafe { blst_p1_compress(out.as_mut_ptr(), p); }
        out.to_vec()
    }

    /// Decompress 48 bytes to blst_p1.
    fn decompress_g1(bytes: &[u8]) -> blst_p1 {
        assert!(bytes.len() >= 48, "G1 compressed point needs 48 bytes");
        let mut affine = blst_p1_affine::default();
        let err = unsafe { blst_p1_uncompress(&mut affine, bytes.as_ptr()) };
        assert_eq!(err, BLST_ERROR::BLST_SUCCESS, "G1 decompression failed");
        let mut p = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut p, &affine); }
        p
    }
}

impl CommitmentScheme for Bls12381Scheme {
    fn init(&self) {
        // Force SRS loading
        let _ = srs();
    }

    fn g1_compressed_size(&self) -> usize {
        48
    }

    fn max_domain_size(&self) -> u64 {
        // Limited by SRS size (4096 G1 points)
        srs().num_g1 as u64
    }

    fn ifft(&self, evals: &[Scalar], domain_size: u64) -> Vec<Scalar> {
        let n = domain_size as usize;
        let mut frs = Self::scalars_to_frs(evals);
        frs.resize(n, blst_fr::default());
        fft_in_place(&mut frs, true);
        Self::frs_to_scalars(&frs)
    }

    fn fft(&self, coeffs: &[Scalar], domain_size: u64) -> Vec<Scalar> {
        let n = domain_size as usize;
        let mut frs = Self::scalars_to_frs(coeffs);
        frs.resize(n, blst_fr::default());
        fft_in_place(&mut frs, false);
        Self::frs_to_scalars(&frs)
    }

    fn commit_evaluations(&self, evals: &[Scalar], domain_size: u64) -> Vec<u8> {
        // Use Lagrange-form SRS for direct commitment in evaluation form
        let s = srs();
        let n = evals.len().min(domain_size as usize).min(s.num_g1);
        let frs = Self::scalars_to_frs(&evals[..n]);
        let result = Self::msm(&s.g1_lagrange[..n], &frs);
        Self::compress_g1(&result)
    }

    fn commit_coefficients(&self, coeffs: &[Scalar]) -> Vec<u8> {
        // Use monomial-form SRS for commitment in coefficient form
        let s = srs();
        let n = coeffs.len().min(s.num_g1);
        let frs = Self::scalars_to_frs(&coeffs[..n]);
        let result = Self::msm(&s.g1_monomial[..n], &frs);
        Self::compress_g1(&result)
    }

    fn eval_poly_at(&self, coeffs: &[Scalar], z: &Scalar) -> Scalar {
        // Horner's method
        let mut result = Scalar::zero(CurveType::Bls12381);
        for i in (0..coeffs.len()).rev() {
            result = result.mul(z);
            result = result.add(&coeffs[i]);
        }
        result
    }

    fn div_by_linear(&self, coeffs: &[Scalar], z: &Scalar) -> Vec<Scalar> {
        // Synthetic division of p(x) by (x - z)
        let n = coeffs.len();
        if n <= 1 {
            return vec![];
        }
        let mut quotient = vec![Scalar::zero(CurveType::Bls12381); n - 1];
        quotient[n - 2] = coeffs[n - 1].clone();
        for i in (0..n - 2).rev() {
            let zq = z.mul(&quotient[i + 1]);
            quotient[i] = coeffs[i + 1].add(&zq);
        }
        quotient
    }

    fn open_at_point(&self, eval_form: &[Scalar], z: &Scalar, domain_size: u64) -> (Scalar, Vec<u8>) {
        let coeffs = self.ifft(eval_form, domain_size);
        let y = self.eval_poly_at(&coeffs, z);

        // Subtract y from constant term: p(x) - y
        let mut shifted = coeffs;
        shifted[0] = shifted[0].sub(&y);

        // Divide by (x - z) to get quotient polynomial
        let quotient_coeffs = self.div_by_linear(&shifted, z);

        // Commit to quotient polynomial as proof
        let proof = self.commit_coefficients(&quotient_coeffs);

        (y, proof)
    }

    fn verify_at_point(&self, commitment: &[u8], z: &Scalar, y: &Scalar, proof: &[u8]) -> bool {
        let s = srs();

        let c = Self::decompress_g1(commitment);
        let pi = Self::decompress_g1(proof);

        // Check: e(C - y·G1, G2) = e(π, [τ]₂ - z·G2)
        // Equivalently: e(C - y·G1, G2) · e(-π, [τ]₂ - z·G2) = 1

        // Compute C - y·G1
        let y_scalar = y.to_blst_scalar();
        let mut g1 = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut g1, &s.g1_monomial[0]); }
        let mut y_g1 = blst_p1::default();
        unsafe { blst_p1_mult(&mut y_g1, &g1, y_scalar.b.as_ptr(), 255); }

        let mut lhs = c;
        let mut neg_y_g1 = y_g1;
        unsafe { blst_p1_cneg(&mut neg_y_g1, true); }
        unsafe { blst_p1_add_or_double(&mut lhs, &lhs, &neg_y_g1); }

        // Compute [τ]₂ - z·G2
        let z_scalar = z.to_blst_scalar();
        let mut g2 = blst_p2::default();
        unsafe { blst_p2_from_affine(&mut g2, &s.g2_affine[0]); }
        let mut z_g2 = blst_p2::default();
        unsafe { blst_p2_mult(&mut z_g2, &g2, z_scalar.b.as_ptr(), 255); }

        let mut tau_g2 = blst_p2::default();
        unsafe { blst_p2_from_affine(&mut tau_g2, &s.g2_affine[1]); }
        let mut rhs_g2 = tau_g2;
        let mut neg_z_g2 = z_g2;
        unsafe { blst_p2_cneg(&mut neg_z_g2, true); }
        unsafe { blst_p2_add_or_double(&mut rhs_g2, &rhs_g2, &neg_z_g2); }

        // Convert to affine for pairing
        let mut lhs_aff = blst_p1_affine::default();
        unsafe { blst_p1_to_affine(&mut lhs_aff, &lhs); }
        let mut rhs_g2_aff = blst_p2_affine::default();
        unsafe { blst_p2_to_affine(&mut rhs_g2_aff, &rhs_g2); }
        let mut pi_aff = blst_p1_affine::default();
        unsafe { blst_p1_to_affine(&mut pi_aff, &pi); }
        let g2_aff = s.g2_affine[0]; // G2 generator

        // e(lhs, G2) · e(-π, rhs_g2) == 1
        let ml1 = blst_fp12::miller_loop(&g2_aff, &lhs_aff);
        let mut neg_pi = pi;
        unsafe { blst_p1_cneg(&mut neg_pi, true); }
        let mut neg_pi_aff = blst_p1_affine::default();
        unsafe { blst_p1_to_affine(&mut neg_pi_aff, &neg_pi); }
        let ml2 = blst_fp12::miller_loop(&rhs_g2_aff, &neg_pi_aff);

        let product = ml1 * ml2;
        let mut final_exp = blst_fp12::default();
        unsafe { blst_final_exp(&mut final_exp, &product); }

        unsafe { blst_fp12_is_equal(&final_exp, blst_fp12_one()) }
    }

    fn batch_verify_at_point(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> bool {
        // Combine commitments: C_f = Σ β^i * C_i
        let mut combined_c = blst_p1::default(); // infinity
        let mut beta_power = Scalar::one(CurveType::Bls12381);
        for comm_bytes in commitments {
            let c = Self::decompress_g1(comm_bytes);
            let bp_scalar = beta_power.to_blst_scalar();
            let mut scaled = blst_p1::default();
            unsafe { blst_p1_mult(&mut scaled, &c, bp_scalar.b.as_ptr(), 255); }
            unsafe { blst_p1_add_or_double(&mut combined_c, &combined_c, &scaled); }
            beta_power = beta_power.mul(beta);
        }

        // Combine evaluations: y_f = Σ β^i * y_i
        let mut combined_y = Scalar::zero(CurveType::Bls12381);
        beta_power = Scalar::one(CurveType::Bls12381);
        for y in evaluations {
            let term = beta_power.mul(y);
            combined_y = combined_y.add(&term);
            beta_power = beta_power.mul(beta);
        }

        let combined_c_bytes = Self::compress_g1(&combined_c);
        self.verify_at_point(&combined_c_bytes, z, &combined_y, proof)
    }

    fn initial_accumulator(&self) -> AccumulatedClaim {
        // Point at infinity (compressed)
        let inf = blst_p1::default();
        let l_bytes = Self::compress_g1(&inf);
        let r_bytes = Self::compress_g1(&inf);
        AccumulatedClaim {
            l_acc: l_bytes,
            r_acc: r_bytes,
            num_folded: 0,
        }
    }

    fn compute_lr(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> (Vec<u8>, Vec<u8>) {
        let s = srs();

        // C_combined = Σ β^i * C_i
        let mut c_combined = blst_p1::default();
        let mut beta_power = Scalar::one(CurveType::Bls12381);
        for comm_bytes in commitments {
            let c = Self::decompress_g1(comm_bytes);
            let bp_scalar = beta_power.to_blst_scalar();
            let mut scaled = blst_p1::default();
            unsafe { blst_p1_mult(&mut scaled, &c, bp_scalar.b.as_ptr(), 255); }
            unsafe { blst_p1_add_or_double(&mut c_combined, &c_combined, &scaled); }
            beta_power = beta_power.mul(beta);
        }

        // y_combined = Σ β^i * y_i
        let mut y_combined = Scalar::zero(CurveType::Bls12381);
        beta_power = Scalar::one(CurveType::Bls12381);
        for y in evaluations {
            let term = beta_power.mul(y);
            y_combined = y_combined.add(&term);
            beta_power = beta_power.mul(beta);
        }

        let pi = Self::decompress_g1(proof);

        // L = C_combined - y_combined * G1 + z * π
        let mut g1 = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut g1, &s.g1_monomial[0]); }
        let y_scalar = y_combined.to_blst_scalar();
        let mut y_g1 = blst_p1::default();
        unsafe { blst_p1_mult(&mut y_g1, &g1, y_scalar.b.as_ptr(), 255); }

        let z_scalar = z.to_blst_scalar();
        let mut z_pi = blst_p1::default();
        unsafe { blst_p1_mult(&mut z_pi, &pi, z_scalar.b.as_ptr(), 255); }

        let mut l = c_combined;
        let mut neg_y_g1 = y_g1;
        unsafe { blst_p1_cneg(&mut neg_y_g1, true); }
        unsafe { blst_p1_add_or_double(&mut l, &l, &neg_y_g1); }
        unsafe { blst_p1_add_or_double(&mut l, &l, &z_pi); }

        let l_bytes = Self::compress_g1(&l);
        let r_bytes = Self::compress_g1(&pi);

        (l_bytes, r_bytes)
    }

    fn fold_accumulator(
        &self,
        l_left: &[u8],
        r_left: &[u8],
        l_right: &[u8],
        r_right: &[u8],
        challenge: &Scalar,
    ) -> (Vec<u8>, Vec<u8>) {
        let r_scalar = challenge.to_blst_scalar();

        let mut l_l = Self::decompress_g1(l_left);
        let l_r = Self::decompress_g1(l_right);
        let mut scaled = blst_p1::default();
        unsafe { blst_p1_mult(&mut scaled, &l_r, r_scalar.b.as_ptr(), 255); }
        unsafe { blst_p1_add_or_double(&mut l_l, &l_l, &scaled); }

        let mut r_l = Self::decompress_g1(r_left);
        let r_r = Self::decompress_g1(r_right);
        unsafe { blst_p1_mult(&mut scaled, &r_r, r_scalar.b.as_ptr(), 255); }
        unsafe { blst_p1_add_or_double(&mut r_l, &r_l, &scaled); }

        (Self::compress_g1(&l_l), Self::compress_g1(&r_l))
    }

    fn verify_accumulated(&self, l_acc: &[u8], r_acc: &[u8]) -> bool {
        let s = srs();

        let l = Self::decompress_g1(l_acc);
        let r = Self::decompress_g1(r_acc);

        // Check: e(L, G2) = e(R, [τ]₂)
        // i.e.: e(L, G2) · e(-R, [τ]₂) = 1
        let mut l_aff = blst_p1_affine::default();
        unsafe { blst_p1_to_affine(&mut l_aff, &l); }
        let mut neg_r = r;
        unsafe { blst_p1_cneg(&mut neg_r, true); }
        let mut neg_r_aff = blst_p1_affine::default();
        unsafe { blst_p1_to_affine(&mut neg_r_aff, &neg_r); }

        let g2_aff = s.g2_affine[0];
        let tau_g2_aff = s.g2_affine[1];

        let ml1 = blst_fp12::miller_loop(&g2_aff, &l_aff);
        let ml2 = blst_fp12::miller_loop(&tau_g2_aff, &neg_r_aff);

        let product = ml1 * ml2;
        let mut final_exp = blst_fp12::default();
        unsafe { blst_final_exp(&mut final_exp, &product); }

        unsafe { blst_fp12_is_equal(&final_exp, blst_fp12_one()) }
    }

    fn domain_generator(&self, domain_size: u64) -> Scalar {
        assert!(domain_size.is_power_of_two(), "domain_size must be power of 2");
        let log_n = domain_size.trailing_zeros();
        let omega = root_of_unity(log_n);
        Scalar::Bls12381(omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Scalar, CurveType};
    use crate::scheme::CommitmentScheme;

    fn curve() -> CurveType {
        CurveType::Bls12381
    }

    #[test]
    fn test_monomial_g1_is_generator() {
        // g1_monomial[0] from the Ethereum SRS should be the BLS12-381 G1 generator.
        let scheme = Bls12381Scheme::new();
        scheme.init();
        let s = srs();

        let mut gen = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut gen, blst_p1_affine_generator()); }
        let gen_bytes = Bls12381Scheme::compress_g1(&gen);

        let mut mono0 = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut mono0, &s.g1_monomial[0]); }
        let mono0_bytes = Bls12381Scheme::compress_g1(&mono0);

        assert_eq!(mono0_bytes, gen_bytes, "monomial[0] should be G1 generator");
    }

    #[test]
    fn test_srs_pairing_consistency() {
        // Verify using commit_evaluations == commit_coefficients for a known poly.
        let scheme = Bls12381Scheme::new();
        scheme.init();

        let coeffs = vec![
            Scalar::from_u64(1, curve()),
            Scalar::from_u64(2, curve()),
        ];
        let c_mono = scheme.commit_coefficients(&coeffs);
        let evals = scheme.fft(&coeffs, 4096);
        let c_lagr = scheme.commit_evaluations(&evals, 4096);
        assert_eq!(c_mono, c_lagr, "Monomial and Lagrange commits should match for same polynomial");
    }

    #[test]
    fn test_pairing_sanity() {
        // Directly verify e(G1, [τ]₂) = e([τ]₁, G2) via pairing
        let scheme = Bls12381Scheme::new();
        scheme.init();
        let s = srs();

        // Use monomial SRS: g1_monomial[0] = G1, g1_monomial[1] = τ·G1
        // G2 SRS: g2_affine[0] = G2, g2_affine[1] = τ·G2

        let g1 = s.g1_monomial[0];
        let tau_g1 = s.g1_monomial[1];
        let g2 = s.g2_affine[0];
        let tau_g2 = s.g2_affine[1];

        // e(G1, τ·G2)
        let f1 = blst_fp12::miller_loop(&tau_g2, &g1);
        let mut gt1 = blst_fp12::default();
        unsafe { blst_final_exp(&mut gt1, &f1); }

        // e(τ·G1, G2)
        let f2 = blst_fp12::miller_loop(&g2, &tau_g1);
        let mut gt2 = blst_fp12::default();
        unsafe { blst_final_exp(&mut gt2, &f2); }

        assert_eq!(gt1, gt2, "Pairing check e(G1,τG2) = e(τG1,G2) failed");
    }

    #[test]
    fn test_trivial_verify() {
        // Verify for the trivial polynomial p(x) = c (constant)
        // C = c·G1, π = 0 (infinity), y = c at any z
        // Check: e(C - y·G1, G2) = e(0, [τ]₂ - z·G2) = 1
        // Since C = y·G1, we get e(0, G2) = 1, which is trivially true.
        let scheme = Bls12381Scheme::new();
        scheme.init();

        let c_val = 42u64;
        let coeffs = vec![Scalar::from_u64(c_val, curve())];
        let commitment = scheme.commit_coefficients(&coeffs);

        let z = Scalar::from_u64(7, curve());
        let y = Scalar::from_u64(c_val, curve());

        // For constant polynomial, quotient (p(x) - y) / (x - z) = 0
        // So proof is point at infinity
        let inf = blst_p1::default();
        let proof = Bls12381Scheme::compress_g1(&inf);

        let valid = scheme.verify_at_point(&commitment, &z, &y, &proof);
        assert!(valid, "Constant polynomial verify should succeed");
    }

    #[test]
    fn test_linear_verify() {
        // p(x) = x, so C = [τ]₁ = SRS[1]
        // At z: y = z, quotient = 1, proof = G1 = SRS[0]
        // Check: e(C - z·G1, G2) = e(G1, [τ]₂ - z·G2)
        let scheme = Bls12381Scheme::new();
        scheme.init();
        let s = srs();

        // p(x) = 0 + 1*x  →  coeff[0] = 0, coeff[1] = 1
        let coeffs = vec![
            Scalar::from_u64(0, curve()),
            Scalar::from_u64(1, curve()),
        ];
        let commitment = scheme.commit_coefficients(&coeffs);

        // Verify commitment == [τ]₁ = SRS[1]
        let mut expected_c = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut expected_c, &s.g1_monomial[1]); }
        let expected_bytes = Bls12381Scheme::compress_g1(&expected_c);
        assert_eq!(commitment, expected_bytes, "C for p(x)=x should be SRS[1]");

        // y = z = 5
        let z = Scalar::from_u64(5, curve());
        let y = Scalar::from_u64(5, curve());

        // quotient = (p(x) - y) / (x - z) = (x - 5) / (x - 5) = 1
        // proof = commit([1]) = 1 * G1 = SRS[0]
        let proof_coeffs = vec![Scalar::from_u64(1, curve())];
        let proof = scheme.commit_coefficients(&proof_coeffs);

        let valid = scheme.verify_at_point(&commitment, &z, &y, &proof);
        assert!(valid, "Linear polynomial verify should succeed");
    }

    #[test]
    fn test_srs_loads() {
        let scheme = Bls12381Scheme::new();
        scheme.init();
        let s = srs();
        assert_eq!(s.num_g1, 4096);
        assert!(s.g2_affine.len() >= 2);
    }

    #[test]
    fn test_root_of_unity() {
        // ω^n should equal 1
        for log_n in [1, 2, 4, 8, 12] {
            let omega = root_of_unity(log_n);
            let n = 1u64 << log_n;
            let omega_n = fr_pow(&omega, n);
            let one = fr_one();
            let mut diff = blst_fr::default();
            unsafe { blst_fr_sub(&mut diff, &omega_n, &one); }
            assert!(fr_is_zero(&diff), "ω^{} should be 1 for log_n={}", n, log_n);

            // ω^(n/2) should NOT be 1 (it should be -1)
            if n > 1 {
                let omega_half = fr_pow(&omega, n / 2);
                let mut diff = blst_fr::default();
                unsafe { blst_fr_sub(&mut diff, &omega_half, &one); }
                assert!(!fr_is_zero(&diff), "ω^(n/2) should not be 1");
            }
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let scheme = Bls12381Scheme::new();
        scheme.init();

        let n = 16u64;
        let orig: Vec<Scalar> = (0..n).map(|i| Scalar::from_u64(i + 1, curve())).collect();

        let evaled = scheme.fft(&orig, n);
        let recovered = scheme.ifft(&evaled, n);

        for i in 0..n as usize {
            let diff = orig[i].sub(&recovered[i]);
            assert!(diff.is_zero(), "FFT roundtrip failed at index {}", i);
        }
    }

    #[test]
    fn test_commit_open_verify() {
        let scheme = Bls12381Scheme::new();
        scheme.init();

        // Create a polynomial p(x) = 1 + 2x + 3x^2 + 4x^3 in coefficient form
        let coeffs = vec![
            Scalar::from_u64(1, curve()),
            Scalar::from_u64(2, curve()),
            Scalar::from_u64(3, curve()),
            Scalar::from_u64(4, curve()),
        ];

        // Commit to it
        let commitment = scheme.commit_coefficients(&coeffs);
        assert_eq!(commitment.len(), 48);

        // Evaluate at z = 5: p(5) = 1 + 10 + 75 + 500 = 586
        let z = Scalar::from_u64(5, curve());
        let y = scheme.eval_poly_at(&coeffs, &z);
        assert_eq!(y.to_u64(), 586);

        // Convert to evaluation form on domain of size 16
        let domain_size = 16u64;
        let eval_form = scheme.fft(&coeffs, domain_size);

        // Open at z
        let (y_open, proof) = scheme.open_at_point(&eval_form, &z, domain_size);
        assert_eq!(y_open.to_u64(), 586);
        assert_eq!(proof.len(), 48);

        // Verify
        let valid = scheme.verify_at_point(&commitment, &z, &y_open, &proof);
        assert!(valid, "Valid opening should verify");

        // Verify with wrong y should fail
        let wrong_y = Scalar::from_u64(999, curve());
        let invalid = scheme.verify_at_point(&commitment, &z, &wrong_y, &proof);
        assert!(!invalid, "Wrong evaluation should not verify");
    }

    #[test]
    fn test_commit_evaluations() {
        // Lagrange SRS is for domain size 4096, so evaluation domain must match.
        let scheme = Bls12381Scheme::new();
        scheme.init();

        let coeffs = vec![
            Scalar::from_u64(1, curve()),
            Scalar::from_u64(2, curve()),
            Scalar::from_u64(3, curve()),
            Scalar::from_u64(4, curve()),
        ];

        let domain_size = 4096u64;
        let eval_form = scheme.fft(&coeffs, domain_size);

        // commit_evaluations should produce the same commitment as commit_coefficients
        // (since they represent the same polynomial, just in different forms)
        let c1 = scheme.commit_coefficients(&coeffs);
        let c2 = scheme.commit_evaluations(&eval_form, domain_size);
        assert_eq!(c1, c2, "Coefficient and evaluation commits should match");
    }

    #[test]
    fn test_batch_verify() {
        let scheme = Bls12381Scheme::new();
        scheme.init();

        let domain_size = 16u64;

        // Polynomial 1: p(x) = 1 + x
        let coeffs1 = vec![
            Scalar::from_u64(1, curve()),
            Scalar::from_u64(1, curve()),
        ];
        let c1 = scheme.commit_coefficients(&coeffs1);

        // Polynomial 2: p(x) = 2 + 3x
        let coeffs2 = vec![
            Scalar::from_u64(2, curve()),
            Scalar::from_u64(3, curve()),
        ];
        let c2 = scheme.commit_coefficients(&coeffs2);

        let z = Scalar::from_u64(7, curve());

        // Evaluate both at z
        let y1 = scheme.eval_poly_at(&coeffs1, &z); // 1 + 7 = 8
        let y2 = scheme.eval_poly_at(&coeffs2, &z); // 2 + 21 = 23
        assert_eq!(y1.to_u64(), 8);
        assert_eq!(y2.to_u64(), 23);

        // Create combined proof manually using β
        let beta = Scalar::from_u64(13, curve());

        // Combined polynomial: f(x) = p1(x) + β * p2(x)
        let mut combined_coeffs = vec![Scalar::zero(curve()); domain_size as usize];
        let mut beta_power = Scalar::one(curve());
        for coeffs in [&coeffs1, &coeffs2] {
            for (j, c) in coeffs.iter().enumerate() {
                let term = beta_power.mul(c);
                combined_coeffs[j] = combined_coeffs[j].add(&term);
            }
            beta_power = beta_power.mul(&beta);
        }

        // Open combined at z
        let combined_eval_form = scheme.fft(&combined_coeffs, domain_size);
        let (_, combined_proof) = scheme.open_at_point(&combined_eval_form, &z, domain_size);

        // Batch verify
        let commitments: Vec<&[u8]> = vec![&c1, &c2];
        let evaluations = vec![y1, y2];
        let valid = scheme.batch_verify_at_point(&commitments, &evaluations, &z, &beta, &combined_proof);
        assert!(valid, "Batch verify should succeed");
    }

    #[test]
    fn test_recursive_accumulator() {
        let scheme = Bls12381Scheme::new();
        scheme.init();

        let domain_size = 16u64;

        // Create two polynomials and their proofs
        let coeffs1 = vec![Scalar::from_u64(1, curve()), Scalar::from_u64(2, curve())];
        let c1 = scheme.commit_coefficients(&coeffs1);

        let coeffs2 = vec![Scalar::from_u64(3, curve()), Scalar::from_u64(4, curve())];
        let c2 = scheme.commit_coefficients(&coeffs2);

        let z = Scalar::from_u64(5, curve());
        let beta = Scalar::from_u64(7, curve());

        let y1 = scheme.eval_poly_at(&coeffs1, &z);
        let y2 = scheme.eval_poly_at(&coeffs2, &z);

        // Create combined proofs for each "chunk"
        let mut combined1_coeffs = coeffs1.clone();
        combined1_coeffs.resize(domain_size as usize, Scalar::zero(curve()));
        let combined1_evals = scheme.fft(&combined1_coeffs, domain_size);
        let (_, proof1) = scheme.open_at_point(&combined1_evals, &z, domain_size);

        let mut combined2_coeffs = coeffs2.clone();
        combined2_coeffs.resize(domain_size as usize, Scalar::zero(curve()));
        let combined2_evals = scheme.fft(&combined2_coeffs, domain_size);
        let (_, proof2) = scheme.open_at_point(&combined2_evals, &z, domain_size);

        // Compute L, R for each
        let (l1, r1) = scheme.compute_lr(&[&c1], &[y1.clone()], &z, &beta, &proof1);
        let (l2, r2) = scheme.compute_lr(&[&c2], &[y2.clone()], &z, &beta, &proof2);

        // Each individual (L, R) should verify
        assert!(scheme.verify_accumulated(&l1, &r1), "First LR should verify");
        assert!(scheme.verify_accumulated(&l2, &r2), "Second LR should verify");

        // Fold them together
        let challenge = Scalar::from_u64(11, curve());
        let (l_fold, r_fold) = scheme.fold_accumulator(&l1, &r1, &l2, &r2, &challenge);

        // Folded accumulator should also verify
        assert!(scheme.verify_accumulated(&l_fold, &r_fold), "Folded accumulator should verify");
    }
}
