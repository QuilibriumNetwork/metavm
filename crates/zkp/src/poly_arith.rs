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

/// Multiply two polynomials in coefficient form.
///
/// Uses O(n log n) NTT-based multiplication for BLS12-381 when either input
/// has degree >= NTT_THRESHOLD. Falls back to O(n*m) naive convolution for
/// small inputs or BLS48-581.
pub fn poly_mul(a: &[Scalar], b: &[Scalar], curve: CurveType) -> Vec<Scalar> {
    if a.is_empty() || b.is_empty() {
        return vec![Scalar::zero(curve)];
    }
    // NTT path for BLS12-381 when either input is large enough
    if matches!(curve, CurveType::Bls12381) && a.len().max(b.len()) >= NTT_THRESHOLD {
        return ntt_mul_bls12381(a, b);
    }
    // Naive O(n*m) convolution for small inputs or BLS48-581
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
}
