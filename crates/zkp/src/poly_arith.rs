//! Coefficient-form polynomial arithmetic utilities.
//!
//! Operations on polynomials represented as coefficient vectors:
//! `[a0, a1, ..., an]` represents `a0 + a1*x + ... + an*x^n`.

use crate::field::{Scalar, CurveType};

/// Add two polynomials in coefficient form.
pub fn poly_add(a: &[Scalar], b: &[Scalar], curve: CurveType) -> Vec<Scalar> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let ai = if i < a.len() { a[i].clone() } else { Scalar::zero(curve) };
        let bi = if i < b.len() { b[i].clone() } else { Scalar::zero(curve) };
        result.push(ai.add(&bi));
    }
    result
}

/// Subtract two polynomials in coefficient form: a - b.
pub fn poly_sub(a: &[Scalar], b: &[Scalar], curve: CurveType) -> Vec<Scalar> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let ai = if i < a.len() { a[i].clone() } else { Scalar::zero(curve) };
        let bi = if i < b.len() { b[i].clone() } else { Scalar::zero(curve) };
        result.push(ai.sub(&bi));
    }
    result
}

/// Threshold above which we use NTT-based multiplication for BLS12-381.
const NTT_THRESHOLD: usize = 64;

/// Input size at/above which Karatsuba is preferred over the naive O(n*m)
/// convolution. Small constants favor naive due to the extra additions and
/// recursion overhead; for BLS48-581 the per-mul cost is high enough that
/// 32 is a good crossover point.
const KARATSUBA_CUTOFF: usize = 32;

/// Multiply two polynomials in coefficient form.
///
/// Uses O(n log n) NTT-based multiplication for BLS12-381 when either input
/// has degree >= NTT_THRESHOLD. Uses O(n^1.585) Karatsuba for BLS48-581 (and
/// for BLS12-381 below the NTT threshold) when either input has length
/// >= KARATSUBA_CUTOFF. Falls back to O(n*m) naive convolution for the
/// smallest inputs.
pub fn poly_mul(a: &[Scalar], b: &[Scalar], curve: CurveType) -> Vec<Scalar> {
    if a.is_empty() || b.is_empty() {
        return vec![Scalar::zero(curve)];
    }
    let max_len = a.len().max(b.len());
    // NTT path for BLS12-381 when either input is large enough
    if matches!(curve, CurveType::Bls12381) && max_len >= NTT_THRESHOLD {
        return ntt_mul_bls12381(a, b);
    }
    // Karatsuba path when inputs are non-trivial
    if max_len >= KARATSUBA_CUTOFF {
        return karatsuba_mul(a, b, curve);
    }
    // Naive O(n*m) convolution for small inputs
    naive_mul(a, b, curve)
}

/// Naive O(n*m) convolution. Precondition: `a` and `b` are non-empty.
fn naive_mul(a: &[Scalar], b: &[Scalar], curve: CurveType) -> Vec<Scalar> {
    let result_len = a.len() + b.len() - 1;
    let mut result = vec![Scalar::zero(curve); result_len];
    for i in 0..a.len() {
        for j in 0..b.len() {
            let term = a[i].mul(&b[j]);
            result[i + j] = result[i + j].add(&term);
        }
    }
    result
}

/// Karatsuba polynomial multiplication: O(n^log2(3)) ≈ O(n^1.585).
///
/// Splits each input into low/high halves at `k = ceil(max_len / 2)` and
/// computes three sub-products via the identity
///   a·b = z0 + (z1 − z0 − z2)·x^k + z2·x^{2k}
/// with `z1 = (a0+a1)·(b0+b1)`.
///
/// Handles unequal-length inputs: a slice shorter than `k` becomes an empty
/// high half (so the corresponding `z2` or `z1` term degenerates to a lower
/// product, which we handle inline).
///
/// Preconditions: both inputs non-empty.
fn karatsuba_mul(a: &[Scalar], b: &[Scalar], curve: CurveType) -> Vec<Scalar> {
    let max_len = a.len().max(b.len());
    // Base case: either recursed down far enough, or one input is trivially small.
    if max_len < KARATSUBA_CUTOFF || a.len() == 1 || b.len() == 1 {
        return naive_mul(a, b, curve);
    }

    let k = (max_len + 1) / 2;

    // Split a into low/high halves at position k.
    let (a_lo, a_hi): (&[Scalar], &[Scalar]) = if a.len() <= k {
        (a, &[])
    } else {
        a.split_at(k)
    };
    let (b_lo, b_hi): (&[Scalar], &[Scalar]) = if b.len() <= k {
        (b, &[])
    } else {
        b.split_at(k)
    };

    // z0 = a_lo * b_lo (always non-empty since a, b are non-empty)
    let z0 = karatsuba_mul(a_lo, b_lo, curve);

    // z2 = a_hi * b_hi; if either high half is empty, z2 is zero polynomial.
    let z2 = if a_hi.is_empty() || b_hi.is_empty() {
        Vec::new()
    } else {
        karatsuba_mul(a_hi, b_hi, curve)
    };

    // z1 = (a_lo + a_hi) * (b_lo + b_hi) − z0 − z2
    // When a high half is empty, (a_lo + a_hi) = a_lo and similarly for b.
    let a_sum = if a_hi.is_empty() {
        a_lo.to_vec()
    } else {
        poly_add(a_lo, a_hi, curve)
    };
    let b_sum = if b_hi.is_empty() {
        b_lo.to_vec()
    } else {
        poly_add(b_lo, b_hi, curve)
    };
    let mut z1 = karatsuba_mul(&a_sum, &b_sum, curve);
    // z1 -= z0
    for (i, c) in z0.iter().enumerate() {
        if i < z1.len() {
            z1[i] = z1[i].sub(c);
        } else {
            // z0 should never extend beyond z1 because deg(z0) <= deg(z1).
            z1.push(Scalar::zero(curve).sub(c));
        }
    }
    // z1 -= z2
    for (i, c) in z2.iter().enumerate() {
        if i < z1.len() {
            z1[i] = z1[i].sub(c);
        } else {
            z1.push(Scalar::zero(curve).sub(c));
        }
    }

    // Assemble: result = z0 + z1·x^k + z2·x^{2k}
    // Output length is a.len() + b.len() - 1.
    let result_len = a.len() + b.len() - 1;
    let mut result = vec![Scalar::zero(curve); result_len];
    for (i, c) in z0.iter().enumerate() {
        result[i] = result[i].add(c);
    }
    for (i, c) in z1.iter().enumerate() {
        let idx = i + k;
        if idx < result_len {
            result[idx] = result[idx].add(c);
        }
        // else: trailing zeros from (a_sum * b_sum) beyond the true product;
        // invariant is that the true sum has no nonzero contribution here
        // since z1 = a_lo·b_hi + a_hi·b_lo, which has degree <= (k-1)+(max_len-k-1) = max_len-2.
    }
    for (i, c) in z2.iter().enumerate() {
        let idx = i + 2 * k;
        if idx < result_len {
            result[idx] = result[idx].add(c);
        }
    }
    result
}

/// NTT-based polynomial multiplication for BLS12-381.
/// O(n log n) via FFT: pad to 2n, forward FFT, pointwise multiply, inverse FFT.
fn ntt_mul_bls12381(a: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
    use blst::*;
    use crate::scheme::bls12381_scheme::fft_in_place;

    let result_len = a.len() + b.len() - 1;
    let domain = result_len.next_power_of_two();

    // Extract blst_fr values, zero-pad to domain size
    let mut a_fr: Vec<blst_fr> = vec![blst_fr::default(); domain];
    for (i, s) in a.iter().enumerate() {
        a_fr[i] = *s.as_bls12381();
    }
    let mut b_fr: Vec<blst_fr> = vec![blst_fr::default(); domain];
    for (i, s) in b.iter().enumerate() {
        b_fr[i] = *s.as_bls12381();
    }

    // Forward FFT both
    fft_in_place(&mut a_fr, false);
    fft_in_place(&mut b_fr, false);

    // Pointwise multiply
    for i in 0..domain {
        let mut tmp = blst_fr::default();
        unsafe { blst_fr_mul(&mut tmp, &a_fr[i], &b_fr[i]); }
        a_fr[i] = tmp;
    }

    // Inverse FFT
    fft_in_place(&mut a_fr, true);

    // Convert back to Scalar, truncate to actual result length
    a_fr.truncate(result_len);
    a_fr.into_iter().map(|fr| Scalar::Bls12381(fr)).collect()
}

/// Multiply a polynomial by a scalar constant.
pub fn poly_scalar_mul(a: &[Scalar], s: &Scalar) -> Vec<Scalar> {
    a.iter().map(|c| c.mul(s)).collect()
}

/// Negate a polynomial: -p(x).
pub fn poly_neg(a: &[Scalar], curve: CurveType) -> Vec<Scalar> {
    let zero = Scalar::zero(curve);
    a.iter().map(|c| zero.sub(c)).collect()
}

/// Shift a polynomial by ω: if p(x) = Σ c_i x^i, then p(ω·x) = Σ (c_i · ω^i) x^i.
///
/// This is used for cross-row constraints where we need column values at ω·X
/// (the "next row" in the evaluation domain).
pub fn poly_shift(coeffs: &[Scalar], omega: &Scalar) -> Vec<Scalar> {
    let mut result = Vec::with_capacity(coeffs.len());
    let mut omega_power = Scalar::one(omega.curve_type());
    for c in coeffs {
        result.push(c.mul(&omega_power));
        omega_power = omega_power.mul(omega);
    }
    result
}

/// Multiply polynomial by the linear factor (x - c).
///
/// If p(x) = Σ a_i x^i, then p(x)·(x - c) = -c·a_0 + (a_0 - c·a_1)x + ... + a_n x^{n+1}.
pub fn poly_mul_linear(coeffs: &[Scalar], c: &Scalar) -> Vec<Scalar> {
    if coeffs.is_empty() {
        return vec![Scalar::zero(c.curve_type())];
    }
    let curve = c.curve_type();
    let neg_c = Scalar::zero(curve).sub(c);
    let n = coeffs.len();
    let mut result = vec![Scalar::zero(curve); n + 1];
    for (i, coeff) in coeffs.iter().enumerate() {
        // coefficient i contributes: coeff * (-c) to result[i] and coeff to result[i+1]
        result[i] = result[i].add(&coeff.mul(&neg_c));
        result[i + 1] = result[i + 1].add(coeff);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_add_same_len() {
        let curve = CurveType::Bls48581;
        let a = vec![Scalar::from_u64(1, curve), Scalar::from_u64(2, curve)];
        let b = vec![Scalar::from_u64(3, curve), Scalar::from_u64(4, curve)];
        let c = poly_add(&a, &b, curve);
        assert_eq!(c.len(), 2);
        assert!(c[0].sub(&Scalar::from_u64(4, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(6, curve)).is_zero());
    }

    #[test]
    fn test_poly_add_diff_len() {
        let curve = CurveType::Bls48581;
        let a = vec![Scalar::from_u64(1, curve)];
        let b = vec![Scalar::from_u64(2, curve), Scalar::from_u64(3, curve)];
        let c = poly_add(&a, &b, curve);
        assert_eq!(c.len(), 2);
        assert!(c[0].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(3, curve)).is_zero());
    }

    #[test]
    fn test_poly_sub() {
        let curve = CurveType::Bls48581;
        let a = vec![Scalar::from_u64(5, curve), Scalar::from_u64(3, curve)];
        let b = vec![Scalar::from_u64(2, curve), Scalar::from_u64(1, curve)];
        let c = poly_sub(&a, &b, curve);
        assert!(c[0].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(2, curve)).is_zero());
    }

    #[test]
    fn test_poly_mul_linear() {
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let curve = CurveType::Bls48581;
        let a = vec![Scalar::from_u64(1, curve), Scalar::from_u64(2, curve)];
        let b = vec![Scalar::from_u64(3, curve), Scalar::from_u64(4, curve)];
        let c = poly_mul(&a, &b, curve);
        assert_eq!(c.len(), 3);
        assert!(c[0].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(10, curve)).is_zero());
        assert!(c[2].sub(&Scalar::from_u64(8, curve)).is_zero());
    }

    #[test]
    fn test_poly_mul_constant() {
        // (5) * (2 + 3x) = 10 + 15x
        let curve = CurveType::Bls48581;
        let a = vec![Scalar::from_u64(5, curve)];
        let b = vec![Scalar::from_u64(2, curve), Scalar::from_u64(3, curve)];
        let c = poly_mul(&a, &b, curve);
        assert_eq!(c.len(), 2);
        assert!(c[0].sub(&Scalar::from_u64(10, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(15, curve)).is_zero());
    }

    #[test]
    fn test_poly_scalar_mul() {
        let curve = CurveType::Bls48581;
        let a = vec![Scalar::from_u64(2, curve), Scalar::from_u64(3, curve)];
        let s = Scalar::from_u64(5, curve);
        let c = poly_scalar_mul(&a, &s);
        assert!(c[0].sub(&Scalar::from_u64(10, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(15, curve)).is_zero());
    }

    #[test]
    fn test_poly_neg() {
        let curve = CurveType::Bls48581;
        let a = vec![Scalar::from_u64(5, curve), Scalar::from_u64(3, curve)];
        let neg = poly_neg(&a, curve);
        let sum = poly_add(&a, &neg, curve);
        assert!(sum[0].is_zero());
        assert!(sum[1].is_zero());
    }

    #[test]
    fn test_poly_mul_empty() {
        let curve = CurveType::Bls48581;
        let a: Vec<Scalar> = vec![];
        let b = vec![Scalar::from_u64(1, curve)];
        let c = poly_mul(&a, &b, curve);
        assert_eq!(c.len(), 1);
        assert!(c[0].is_zero());
    }

    #[test]
    fn test_poly_add_bls12381() {
        let curve = CurveType::Bls12381;
        let a = vec![Scalar::from_u64(10, curve), Scalar::from_u64(20, curve)];
        let b = vec![Scalar::from_u64(30, curve), Scalar::from_u64(40, curve)];
        let c = poly_add(&a, &b, curve);
        assert!(c[0].sub(&Scalar::from_u64(40, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(60, curve)).is_zero());
    }

    #[test]
    fn test_poly_mul_bls12381() {
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let curve = CurveType::Bls12381;
        let a = vec![Scalar::from_u64(1, curve), Scalar::from_u64(2, curve)];
        let b = vec![Scalar::from_u64(3, curve), Scalar::from_u64(4, curve)];
        let c = poly_mul(&a, &b, curve);
        assert_eq!(c.len(), 3);
        assert!(c[0].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(c[1].sub(&Scalar::from_u64(10, curve)).is_zero());
        assert!(c[2].sub(&Scalar::from_u64(8, curve)).is_zero());
    }

    /// Helper: naive O(n^2) convolution for cross-checking NTT results.
    fn naive_poly_mul(a: &[Scalar], b: &[Scalar], curve: CurveType) -> Vec<Scalar> {
        if a.is_empty() || b.is_empty() {
            return vec![Scalar::zero(curve)];
        }
        let result_len = a.len() + b.len() - 1;
        let mut result = vec![Scalar::zero(curve); result_len];
        for i in 0..a.len() {
            for j in 0..b.len() {
                let term = a[i].mul(&b[j]);
                result[i + j] = result[i + j].add(&term);
            }
        }
        result
    }

    #[test]
    fn test_ntt_mul_matches_naive_small() {
        // Force NTT path by calling ntt_mul_bls12381 directly on a small polynomial
        let curve = CurveType::Bls12381;
        let a: Vec<Scalar> = (1..=8).map(|i| Scalar::from_u64(i, curve)).collect();
        let b: Vec<Scalar> = (10..=17).map(|i| Scalar::from_u64(i, curve)).collect();

        let naive = naive_poly_mul(&a, &b, curve);
        let ntt = ntt_mul_bls12381(&a, &b);

        assert_eq!(naive.len(), ntt.len());
        for i in 0..naive.len() {
            assert!(naive[i].sub(&ntt[i]).is_zero(), "NTT mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ntt_mul_matches_naive_large() {
        // Test above NTT_THRESHOLD to confirm poly_mul dispatches correctly
        let curve = CurveType::Bls12381;
        let n = 128;
        let a: Vec<Scalar> = (0..n).map(|i| Scalar::from_u64(i + 1, curve)).collect();
        let b: Vec<Scalar> = (0..n).map(|i| Scalar::from_u64(i * 3 + 7, curve)).collect();

        let naive = naive_poly_mul(&a, &b, curve);
        let result = poly_mul(&a, &b, curve);

        assert_eq!(naive.len(), result.len());
        for i in 0..naive.len() {
            assert!(naive[i].sub(&result[i]).is_zero(), "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ntt_mul_asymmetric() {
        // One side below threshold, one above — should still use NTT
        let curve = CurveType::Bls12381;
        let a: Vec<Scalar> = (0..3).map(|i| Scalar::from_u64(i + 1, curve)).collect();
        let b: Vec<Scalar> = (0..100).map(|i| Scalar::from_u64(i + 1, curve)).collect();

        let naive = naive_poly_mul(&a, &b, curve);
        let result = poly_mul(&a, &b, curve);

        assert_eq!(naive.len(), result.len());
        for i in 0..naive.len() {
            assert!(naive[i].sub(&result[i]).is_zero(), "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ntt_mul_degree_zero() {
        // Constant * large polynomial
        let curve = CurveType::Bls12381;
        let a = vec![Scalar::from_u64(42, curve)];
        let b: Vec<Scalar> = (0..100).map(|i| Scalar::from_u64(i + 1, curve)).collect();

        let naive = naive_poly_mul(&a, &b, curve);
        let result = poly_mul(&a, &b, curve);

        assert_eq!(naive.len(), result.len());
        for i in 0..naive.len() {
            assert!(naive[i].sub(&result[i]).is_zero(), "Mismatch at index {}", i);
        }
    }

    /// Assert two coefficient vectors represent the same polynomial.
    fn assert_poly_eq(got: &[Scalar], want: &[Scalar], ctx: &str) {
        assert_eq!(got.len(), want.len(), "{}: length mismatch", ctx);
        for i in 0..got.len() {
            assert!(
                got[i].sub(&want[i]).is_zero(),
                "{}: mismatch at index {}",
                ctx,
                i
            );
        }
    }

    /// A tiny deterministic LCG for reproducible pseudo-random inputs.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state
    }

    fn random_poly(n: usize, curve: CurveType, seed: &mut u64) -> Vec<Scalar> {
        (0..n).map(|_| Scalar::from_u64(lcg_next(seed), curve)).collect()
    }

    #[test]
    fn test_karatsuba_matches_naive_bls48581() {
        // BLS48-581 scalar mul is expensive in debug mode; keep the grid modest.
        // Sizes still cover the cutoff (32), odd boundaries (33, 63, 65),
        // and one level above (65, 128 means k=64, recursed sub-muls are 32 < cutoff).
        let curve = CurveType::Bls48581;
        let sizes = [32usize, 33, 40, 48, 63, 64, 65, 80];
        let mut seed: u64 = 0xDEADBEEF;
        for &na in &sizes {
            for &nb in &sizes {
                let a = random_poly(na, curve, &mut seed);
                let b = random_poly(nb, curve, &mut seed);
                let got = karatsuba_mul(&a, &b, curve);
                let want = naive_poly_mul(&a, &b, curve);
                assert_poly_eq(&got, &want, &format!("bls48581 na={} nb={}", na, nb));
            }
        }
    }

    #[test]
    fn test_karatsuba_matches_naive_bls12381() {
        // BLS12-381 is fast; run the full grid requested in the spec.
        let curve = CurveType::Bls12381;
        let sizes = [32usize, 33, 40, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256, 257];
        let mut seed: u64 = 0xC0FFEE;
        for &na in &sizes {
            for &nb in &sizes {
                let a = random_poly(na, curve, &mut seed);
                let b = random_poly(nb, curve, &mut seed);
                let got = karatsuba_mul(&a, &b, curve);
                let want = naive_poly_mul(&a, &b, curve);
                assert_poly_eq(&got, &want, &format!("bls12381 na={} nb={}", na, nb));
            }
        }
    }

    #[test]
    fn test_poly_mul_property_random_lengths_bls12381() {
        // Property test: random pairs of lengths in [0..300] for BLS12-381.
        // Covers NTT (>= 64), Karatsuba (32..=63), and naive (< 32) dispatch paths.
        let curve = CurveType::Bls12381;
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        for _trial in 0..60 {
            let na = (lcg_next(&mut seed) as usize) % 301;
            let nb = (lcg_next(&mut seed) as usize) % 301;
            let a = random_poly(na, curve, &mut seed);
            let b = random_poly(nb, curve, &mut seed);
            let got = poly_mul(&a, &b, curve);
            let want = naive_poly_mul(&a, &b, curve);
            assert_poly_eq(&got, &want, &format!("bls12381 na={} nb={}", na, nb));
        }
    }

    #[test]
    fn test_poly_mul_property_random_lengths_bls48581() {
        // Property test for BLS48-581. Capped to length 96 because scalar mul
        // on BLS48-581 is very slow in debug mode; 96 still exercises Karatsuba
        // recursion (k=48, sub-muls hit the naive cutoff).
        let curve = CurveType::Bls48581;
        let mut seed: u64 = 0x1234_5678_9ABC_DEF1;
        for _trial in 0..20 {
            let na = (lcg_next(&mut seed) as usize) % 97;
            let nb = (lcg_next(&mut seed) as usize) % 97;
            let a = random_poly(na, curve, &mut seed);
            let b = random_poly(nb, curve, &mut seed);
            let got = poly_mul(&a, &b, curve);
            let want = naive_poly_mul(&a, &b, curve);
            assert_poly_eq(&got, &want, &format!("bls48581 na={} nb={}", na, nb));
        }
    }

    #[test]
    fn test_poly_mul_karatsuba_edge_sizes() {
        // Targeted boundaries: just below, at, and just above the cutoff and
        // powers of two. BLS48-581 uses a reduced upper bound for speed.
        let mut seed: u64 = 42;
        let bls48_sizes = [1usize, 2, 16, 31, 32, 33, 63, 64, 65, 96];
        for &n in &bls48_sizes {
            let curve = CurveType::Bls48581;
            let a = random_poly(n, curve, &mut seed);
            let b = random_poly(n, curve, &mut seed);
            let got = poly_mul(&a, &b, curve);
            let want = naive_poly_mul(&a, &b, curve);
            assert_poly_eq(&got, &want, &format!("bls48581 n={}", n));
        }
        let bls12_sizes = [1usize, 2, 16, 31, 32, 33, 63, 64, 127, 128, 255, 256, 257];
        for &n in &bls12_sizes {
            let curve = CurveType::Bls12381;
            let a = random_poly(n, curve, &mut seed);
            let b = random_poly(n, curve, &mut seed);
            let got = poly_mul(&a, &b, curve);
            let want = naive_poly_mul(&a, &b, curve);
            assert_poly_eq(&got, &want, &format!("bls12381 n={}", n));
        }
    }

    #[test]
    fn test_poly_mul_asymmetric_karatsuba() {
        // Highly unequal lengths — exercises the "short split" path where one
        // high half is empty.
        let mut seed: u64 = 0xABCD;
        let bls48_pairs = [(1usize, 64usize), (3, 64), (5, 96), (33, 65), (40, 60)];
        for &(na, nb) in &bls48_pairs {
            let curve = CurveType::Bls48581;
            let a = random_poly(na, curve, &mut seed);
            let b = random_poly(nb, curve, &mut seed);
            let got = poly_mul(&a, &b, curve);
            let want = naive_poly_mul(&a, &b, curve);
            assert_poly_eq(&got, &want, &format!("bls48581 {}x{}", na, nb));
            let got_swap = poly_mul(&b, &a, curve);
            let want_swap = naive_poly_mul(&b, &a, curve);
            assert_poly_eq(&got_swap, &want_swap, &format!("bls48581 {}x{} swap", na, nb));
        }
        let bls12_pairs = [
            (1usize, 64usize),
            (3, 100),
            (5, 256),
            (33, 257),
            (40, 60),
            (127, 300),
        ];
        for &(na, nb) in &bls12_pairs {
            let curve = CurveType::Bls12381;
            let a = random_poly(na, curve, &mut seed);
            let b = random_poly(nb, curve, &mut seed);
            let got = poly_mul(&a, &b, curve);
            let want = naive_poly_mul(&a, &b, curve);
            assert_poly_eq(&got, &want, &format!("bls12381 {}x{}", na, nb));
            let got_swap = poly_mul(&b, &a, curve);
            let want_swap = naive_poly_mul(&b, &a, curve);
            assert_poly_eq(&got_swap, &want_swap, &format!("bls12381 {}x{} swap", na, nb));
        }
    }

    #[test]
    fn test_poly_mul_empty_and_single_coeff() {
        // Must match naive path for edge inputs.
        for curve in [CurveType::Bls48581, CurveType::Bls12381] {
            // Empty * non-empty
            let a: Vec<Scalar> = vec![];
            let b = vec![Scalar::from_u64(7, curve), Scalar::from_u64(11, curve)];
            let got = poly_mul(&a, &b, curve);
            assert_eq!(got.len(), 1);
            assert!(got[0].is_zero());

            // Non-empty * empty
            let got = poly_mul(&b, &a, curve);
            assert_eq!(got.len(), 1);
            assert!(got[0].is_zero());

            // Empty * empty
            let got = poly_mul(&a, &a, curve);
            assert_eq!(got.len(), 1);
            assert!(got[0].is_zero());

            // Single coefficient * larger polynomial (across cutoff)
            let c = vec![Scalar::from_u64(13, curve)];
            let d: Vec<Scalar> = (0..100).map(|i| Scalar::from_u64(i + 1, curve)).collect();
            let got = poly_mul(&c, &d, curve);
            let want = naive_poly_mul(&c, &d, curve);
            assert_poly_eq(&got, &want, "single x 100");
        }
    }
}
