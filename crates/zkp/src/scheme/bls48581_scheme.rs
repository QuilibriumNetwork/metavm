//! BLS48-581 implementation of the CommitmentScheme trait.
//!
//! Delegates to the existing `bls48581` crate and `commitment.rs` module.

use crate::field::{Scalar, CurveType};
use crate::scheme::{CommitmentScheme, AccumulatedClaim};
use bls48581::bls;
use bls48581::bls48581::big;
use bls48581::bls48581::ecp;
use bls48581::bls48581::ecp8;
use bls48581::bls48581::pair8;
use bls48581::bls48581::rom;

/// BLS48-581 KZG commitment scheme using the ceremony SRS.
pub struct Bls48581Scheme;

/// Optimized BLS48-581 KZG commitment scheme.
///
/// Same as [`Bls48581Scheme`] but uses the `_fast` variants:
/// - `fft_fast`: iterative in-place FFT with cached modulus
/// - `muln_fast`: signed-digit adaptive-window Pippenger with mixed addition
///
/// The original [`Bls48581Scheme`] is preserved for side-by-side benchmarking.
pub struct Bls48581SchemeFast;

impl Bls48581Scheme {
    pub fn new() -> Self {
        Bls48581Scheme
    }

    fn scalar_to_big(s: &Scalar) -> big::BIG {
        big::BIG::new_copy(s.as_bls48581())
    }

    fn scalars_to_bigs(scalars: &[Scalar]) -> Vec<big::BIG> {
        scalars.iter().map(|s| Self::scalar_to_big(s)).collect()
    }

    fn bigs_to_scalars(bigs: &[big::BIG]) -> Vec<Scalar> {
        bigs.iter().map(|b| Scalar::Bls48581(big::BIG::new_copy(b))).collect()
    }
}

impl CommitmentScheme for Bls48581Scheme {
    fn init(&self) {
        bls48581::init();
    }

    fn g1_compressed_size(&self) -> usize {
        74
    }

    fn max_domain_size(&self) -> u64 {
        256
    }

    fn ifft(&self, evals: &[Scalar], domain_size: u64) -> Vec<Scalar> {
        let bigs = Self::scalars_to_bigs(evals);
        let result = bls48581::fft(&bigs, domain_size, true)
            .expect("inverse FFT should succeed");
        Self::bigs_to_scalars(&result)
    }

    fn fft(&self, coeffs: &[Scalar], domain_size: u64) -> Vec<Scalar> {
        let bigs = Self::scalars_to_bigs(coeffs);
        let result = bls48581::fft(&bigs, domain_size, false)
            .expect("forward FFT should succeed");
        Self::bigs_to_scalars(&result)
    }

    fn commit_evaluations(&self, evals: &[Scalar], domain_size: u64) -> Vec<u8> {
        let bigs = Self::scalars_to_bigs(evals);
        bls48581::commit_scalars(&bigs, domain_size)
    }

    fn commit_coefficients(&self, coeffs: &[Scalar]) -> Vec<u8> {
        let bigs = Self::scalars_to_bigs(coeffs);
        bls48581::commit_scalars_monomial(&bigs)
    }

    fn eval_poly_at(&self, coeffs: &[Scalar], z: &Scalar) -> Scalar {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let z_big = Self::scalar_to_big(z);
        let mut result = big::BIG::new();
        for i in (0..coeffs.len()).rev() {
            result = big::BIG::modmul(&result, &z_big, &modulus);
            result = big::BIG::modadd(&result, coeffs[i].as_bls48581(), &modulus);
        }
        Scalar::Bls48581(result)
    }

    fn div_by_linear(&self, coeffs: &[Scalar], z: &Scalar) -> Vec<Scalar> {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let z_big = Self::scalar_to_big(z);
        let n = coeffs.len();
        if n <= 1 {
            return vec![];
        }
        let mut quotient = vec![Scalar::zero(CurveType::Bls48581); n - 1];
        quotient[n - 2] = coeffs[n - 1].clone();
        for i in (0..n - 2).rev() {
            let zq = big::BIG::modmul(&z_big, quotient[i + 1].as_bls48581(), &modulus);
            let val = big::BIG::modadd(coeffs[i + 1].as_bls48581(), &zq, &modulus);
            quotient[i] = Scalar::Bls48581(val);
        }
        quotient
    }

    fn open_at_point(&self, eval_form: &[Scalar], z: &Scalar, domain_size: u64) -> (Scalar, Vec<u8>) {
        let coeffs = self.ifft(eval_form, domain_size);
        let y = self.eval_poly_at(&coeffs, z);

        // Subtract y from constant term
        let mut shifted = coeffs;
        shifted[0] = shifted[0].sub(&y);

        let quotient_coeffs = self.div_by_linear(&shifted, z);
        let proof = self.commit_coefficients(&quotient_coeffs);

        (y, proof)
    }

    fn verify_at_point(&self, commitment: &[u8], z: &Scalar, y: &Scalar, proof: &[u8]) -> bool {
        let s = bls::singleton();
        let z_big = Self::scalar_to_big(z);
        let y_big = Self::scalar_to_big(y);

        let c = ecp::ECP::frombytes(commitment);
        let pi = ecp::ECP::frombytes(proof);

        if c.is_infinity() || pi.is_infinity() {
            return false;
        }

        // C - y * G1
        let g1 = ecp::ECP::generator();
        let y_g1 = g1.mul(&y_big);
        let mut lhs_g1 = c;
        lhs_g1.sub(&y_g1);
        lhs_g1.affine();

        // [τ]₂ - z * G2
        let tau_g2 = s.CeremonyBLS48581G2[1].clone();
        let g2_gen = ecp8::ECP8::generator();
        let z_g2 = g2_gen.mul(&z_big);
        let mut rhs_g2 = tau_g2;
        rhs_g2.sub(&z_g2);
        rhs_g2.affine();

        let mut r = pair8::initmp();
        pair8::another(&mut r, &ecp8::ECP8::generator(), &lhs_g1);
        let mut neg_pi = pi;
        neg_pi.neg();
        pair8::another(&mut r, &rhs_g2, &neg_pi);
        let mut v = pair8::miller(&mut r);
        v = pair8::fexp(&v);
        v.isunity()
    }

    fn batch_verify_at_point(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> bool {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let beta_big = Self::scalar_to_big(beta);

        // Combine commitments: C_f = Σ β^i * C_i
        let mut combined_c = ecp::ECP::new();
        let mut beta_power = big::BIG::new_int(1);
        for comm_bytes in commitments {
            let c = ecp::ECP::frombytes(comm_bytes);
            let scaled = c.mul(&beta_power);
            combined_c.add(&scaled);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }
        combined_c.affine();

        // Combine evaluations: y_f = Σ β^i * y_i
        let mut combined_y = big::BIG::new();
        beta_power = big::BIG::new_int(1);
        for y in evaluations {
            let term = big::BIG::modmul(&beta_power, y.as_bls48581(), &modulus);
            combined_y = big::BIG::modadd(&combined_y, &term, &modulus);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }

        let mut combined_c_bytes = vec![0u8; 74];
        combined_c.tobytes(&mut combined_c_bytes, true);

        self.verify_at_point(
            &combined_c_bytes,
            z,
            &Scalar::Bls48581(combined_y),
            proof,
        )
    }

    fn initial_accumulator(&self) -> AccumulatedClaim {
        let inf = ecp::ECP::new();
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

    fn compute_lr(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> (Vec<u8>, Vec<u8>) {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let z_big = Self::scalar_to_big(z);
        let beta_big = Self::scalar_to_big(beta);

        // C_combined = Σ β^i * C_i
        let mut c_combined = ecp::ECP::new();
        let mut beta_power = big::BIG::new_int(1);
        for comm_bytes in commitments {
            let c = ecp::ECP::frombytes(comm_bytes);
            let scaled = c.mul(&beta_power);
            c_combined.add(&scaled);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }

        // y_combined = Σ β^i * y_i
        let mut y_combined = big::BIG::new();
        beta_power = big::BIG::new_int(1);
        for y in evaluations {
            let term = big::BIG::modmul(&beta_power, y.as_bls48581(), &modulus);
            y_combined = big::BIG::modadd(&y_combined, &term, &modulus);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }

        let pi = ecp::ECP::frombytes(proof);

        // L = C_combined - y_combined*G1 + z*π
        let g1 = ecp::ECP::generator();
        let y_g1 = g1.mul(&y_combined);
        let z_pi = pi.mul(&z_big);
        let mut l = c_combined;
        l.sub(&y_g1);
        l.add(&z_pi);
        l.affine();

        let mut l_bytes = vec![0u8; 74];
        l.tobytes(&mut l_bytes, true);
        let mut r_bytes = vec![0u8; 74];
        pi.tobytes(&mut r_bytes, true);

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
        let r = Self::scalar_to_big(challenge);

        let mut l_l = ecp::ECP::frombytes(l_left);
        let l_r = ecp::ECP::frombytes(l_right);
        l_l.add(&l_r.mul(&r));
        l_l.affine();
        let mut l_bytes = vec![0u8; 74];
        l_l.tobytes(&mut l_bytes, true);

        let mut r_l = ecp::ECP::frombytes(r_left);
        let r_r = ecp::ECP::frombytes(r_right);
        r_l.add(&r_r.mul(&r));
        r_l.affine();
        let mut r_bytes = vec![0u8; 74];
        r_l.tobytes(&mut r_bytes, true);

        (l_bytes, r_bytes)
    }

    fn verify_accumulated(&self, l_acc: &[u8], r_acc: &[u8]) -> bool {
        let l = ecp::ECP::frombytes(l_acc);
        let r = ecp::ECP::frombytes(r_acc);

        let s = bls::singleton();
        let tau_g2 = s.CeremonyBLS48581G2[1].clone();

        let mut multi = pair8::initmp();
        pair8::another(&mut multi, &ecp8::ECP8::generator(), &l);
        let mut neg_r = r;
        neg_r.neg();
        pair8::another(&mut multi, &tau_g2, &neg_r);
        let mut v = pair8::miller(&mut multi);
        v = pair8::fexp(&v);
        v.isunity()
    }

    fn domain_generator(&self, domain_size: u64) -> Scalar {
        let s = bls::singleton();
        let roots = &s.RootsOfUnityBLS48581[&domain_size];
        // roots[0] = 1, roots[1] = ω, roots[2] = ω², ...
        Scalar::Bls48581(roots[1].clone())
    }
}

// ── Bls48581SchemeFast ──────────────────────────────────────────────────────

impl Bls48581SchemeFast {
    pub fn new() -> Self {
        Bls48581SchemeFast
    }

    fn scalar_to_big(s: &Scalar) -> big::BIG {
        big::BIG::new_copy(s.as_bls48581())
    }

    fn scalars_to_bigs(scalars: &[Scalar]) -> Vec<big::BIG> {
        scalars.iter().map(|s| Self::scalar_to_big(s)).collect()
    }

    fn bigs_to_scalars(bigs: &[big::BIG]) -> Vec<Scalar> {
        bigs.iter().map(|b| Scalar::Bls48581(big::BIG::new_copy(b))).collect()
    }
}

impl CommitmentScheme for Bls48581SchemeFast {
    fn init(&self) {
        bls48581::init();
    }

    fn g1_compressed_size(&self) -> usize {
        74
    }

    fn max_domain_size(&self) -> u64 {
        256
    }

    fn ifft(&self, evals: &[Scalar], domain_size: u64) -> Vec<Scalar> {
        let bigs = Self::scalars_to_bigs(evals);
        let result = bls48581::fft_fast(&bigs, domain_size, true)
            .expect("inverse FFT should succeed");
        Self::bigs_to_scalars(&result)
    }

    fn fft(&self, coeffs: &[Scalar], domain_size: u64) -> Vec<Scalar> {
        let bigs = Self::scalars_to_bigs(coeffs);
        let result = bls48581::fft_fast(&bigs, domain_size, false)
            .expect("forward FFT should succeed");
        Self::bigs_to_scalars(&result)
    }

    fn commit_evaluations(&self, evals: &[Scalar], domain_size: u64) -> Vec<u8> {
        let bigs = Self::scalars_to_bigs(evals);
        bls48581::commit_scalars_fast(&bigs, domain_size)
    }

    fn commit_coefficients(&self, coeffs: &[Scalar]) -> Vec<u8> {
        let bigs = Self::scalars_to_bigs(coeffs);
        bls48581::commit_scalars_monomial_fast(&bigs)
    }

    fn eval_poly_at(&self, coeffs: &[Scalar], z: &Scalar) -> Scalar {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let z_big = Self::scalar_to_big(z);
        let mut result = big::BIG::new();
        for i in (0..coeffs.len()).rev() {
            result = big::BIG::modmul(&result, &z_big, &modulus);
            result = big::BIG::modadd(&result, coeffs[i].as_bls48581(), &modulus);
        }
        Scalar::Bls48581(result)
    }

    fn div_by_linear(&self, coeffs: &[Scalar], z: &Scalar) -> Vec<Scalar> {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let z_big = Self::scalar_to_big(z);
        let n = coeffs.len();
        if n <= 1 {
            return vec![];
        }
        let mut quotient = vec![Scalar::zero(CurveType::Bls48581); n - 1];
        quotient[n - 2] = coeffs[n - 1].clone();
        for i in (0..n - 2).rev() {
            let zq = big::BIG::modmul(&z_big, quotient[i + 1].as_bls48581(), &modulus);
            let val = big::BIG::modadd(coeffs[i + 1].as_bls48581(), &zq, &modulus);
            quotient[i] = Scalar::Bls48581(val);
        }
        quotient
    }

    fn open_at_point(&self, eval_form: &[Scalar], z: &Scalar, domain_size: u64) -> (Scalar, Vec<u8>) {
        let coeffs = self.ifft(eval_form, domain_size);
        let y = self.eval_poly_at(&coeffs, z);

        let mut shifted = coeffs;
        shifted[0] = shifted[0].sub(&y);

        let quotient_coeffs = self.div_by_linear(&shifted, z);
        let proof = self.commit_coefficients(&quotient_coeffs);

        (y, proof)
    }

    fn verify_at_point(&self, commitment: &[u8], z: &Scalar, y: &Scalar, proof: &[u8]) -> bool {
        let s = bls::singleton();
        let z_big = Self::scalar_to_big(z);
        let y_big = Self::scalar_to_big(y);

        let c = ecp::ECP::frombytes(commitment);
        let pi = ecp::ECP::frombytes(proof);

        if c.is_infinity() || pi.is_infinity() {
            return false;
        }

        let g1 = ecp::ECP::generator();
        let y_g1 = g1.mul(&y_big);
        let mut lhs_g1 = c;
        lhs_g1.sub(&y_g1);
        lhs_g1.affine();

        let tau_g2 = s.CeremonyBLS48581G2[1].clone();
        let g2_gen = ecp8::ECP8::generator();
        let z_g2 = g2_gen.mul(&z_big);
        let mut rhs_g2 = tau_g2;
        rhs_g2.sub(&z_g2);
        rhs_g2.affine();

        let mut r = pair8::initmp();
        pair8::another(&mut r, &ecp8::ECP8::generator(), &lhs_g1);
        let mut neg_pi = pi;
        neg_pi.neg();
        pair8::another(&mut r, &rhs_g2, &neg_pi);
        let mut v = pair8::miller(&mut r);
        v = pair8::fexp(&v);
        v.isunity()
    }

    fn batch_verify_at_point(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> bool {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let beta_big = Self::scalar_to_big(beta);

        let mut combined_c = ecp::ECP::new();
        let mut beta_power = big::BIG::new_int(1);
        for comm_bytes in commitments {
            let c = ecp::ECP::frombytes(comm_bytes);
            let scaled = c.mul(&beta_power);
            combined_c.add(&scaled);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }
        combined_c.affine();

        let mut combined_y = big::BIG::new();
        beta_power = big::BIG::new_int(1);
        for y in evaluations {
            let term = big::BIG::modmul(&beta_power, y.as_bls48581(), &modulus);
            combined_y = big::BIG::modadd(&combined_y, &term, &modulus);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }

        let mut combined_c_bytes = vec![0u8; 74];
        combined_c.tobytes(&mut combined_c_bytes, true);

        self.verify_at_point(
            &combined_c_bytes,
            z,
            &Scalar::Bls48581(combined_y),
            proof,
        )
    }

    fn initial_accumulator(&self) -> AccumulatedClaim {
        let inf = ecp::ECP::new();
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

    fn compute_lr(
        &self,
        commitments: &[&[u8]],
        evaluations: &[Scalar],
        z: &Scalar,
        beta: &Scalar,
        proof: &[u8],
    ) -> (Vec<u8>, Vec<u8>) {
        let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
        let z_big = Self::scalar_to_big(z);
        let beta_big = Self::scalar_to_big(beta);

        let mut c_combined = ecp::ECP::new();
        let mut beta_power = big::BIG::new_int(1);
        for comm_bytes in commitments {
            let c = ecp::ECP::frombytes(comm_bytes);
            let scaled = c.mul(&beta_power);
            c_combined.add(&scaled);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }

        let mut y_combined = big::BIG::new();
        beta_power = big::BIG::new_int(1);
        for y in evaluations {
            let term = big::BIG::modmul(&beta_power, y.as_bls48581(), &modulus);
            y_combined = big::BIG::modadd(&y_combined, &term, &modulus);
            beta_power = big::BIG::modmul(&beta_power, &beta_big, &modulus);
        }

        let pi = ecp::ECP::frombytes(proof);

        let g1 = ecp::ECP::generator();
        let y_g1 = g1.mul(&y_combined);
        let z_pi = pi.mul(&z_big);
        let mut l = c_combined;
        l.sub(&y_g1);
        l.add(&z_pi);
        l.affine();

        let mut l_bytes = vec![0u8; 74];
        l.tobytes(&mut l_bytes, true);
        let mut r_bytes = vec![0u8; 74];
        pi.tobytes(&mut r_bytes, true);

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
        let r = Self::scalar_to_big(challenge);

        let mut l_l = ecp::ECP::frombytes(l_left);
        let l_r = ecp::ECP::frombytes(l_right);
        l_l.add(&l_r.mul(&r));
        l_l.affine();
        let mut l_bytes = vec![0u8; 74];
        l_l.tobytes(&mut l_bytes, true);

        let mut r_l = ecp::ECP::frombytes(r_left);
        let r_r = ecp::ECP::frombytes(r_right);
        r_l.add(&r_r.mul(&r));
        r_l.affine();
        let mut r_bytes = vec![0u8; 74];
        r_l.tobytes(&mut r_bytes, true);

        (l_bytes, r_bytes)
    }

    fn verify_accumulated(&self, l_acc: &[u8], r_acc: &[u8]) -> bool {
        let l = ecp::ECP::frombytes(l_acc);
        let r = ecp::ECP::frombytes(r_acc);

        let s = bls::singleton();
        let tau_g2 = s.CeremonyBLS48581G2[1].clone();

        let mut multi = pair8::initmp();
        pair8::another(&mut multi, &ecp8::ECP8::generator(), &l);
        let mut neg_r = r;
        neg_r.neg();
        pair8::another(&mut multi, &tau_g2, &neg_r);
        let mut v = pair8::miller(&mut multi);
        v = pair8::fexp(&v);
        v.isunity()
    }

    fn domain_generator(&self, domain_size: u64) -> Scalar {
        let s = bls::singleton();
        let roots = &s.RootsOfUnityBLS48581[&domain_size];
        Scalar::Bls48581(roots[1].clone())
    }
}
