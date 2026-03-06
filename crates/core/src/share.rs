use crate::field::FieldElement;
use rand::RngCore;

/// A single share in a Shamir secret sharing scheme.
#[derive(Clone, Debug)]
pub struct Share {
    /// The evaluation point (party index, 1-based).
    pub id: u64,
    /// The share value.
    pub value: FieldElement,
}

/// Split a secret into `n` shares with threshold `t` (need `t` shares to reconstruct).
/// Uses a random polynomial of degree `t-1` where `poly(0) = secret`.
pub fn share<R: RngCore + rand::CryptoRng>(
    secret: &FieldElement,
    n: usize,
    t: usize,
    rng: &mut R,
) -> Vec<Share> {
    assert!(t >= 1, "threshold must be at least 1");
    assert!(n >= t, "share count must be >= threshold");

    // Build polynomial coefficients: a_0 = secret, a_1..a_{t-1} random
    let mut coeffs = Vec::with_capacity(t);
    coeffs.push(*secret);
    for _ in 1..t {
        coeffs.push(FieldElement::random(rng));
    }

    // Evaluate polynomial at points 1, 2, ..., n
    let mut shares = Vec::with_capacity(n);
    for i in 1..=n {
        let x = FieldElement::from_u64(i as u64);
        let mut result = coeffs[0];
        let mut x_pow = x;
        for j in 1..t {
            result += coeffs[j] * x_pow;
            x_pow *= x;
        }
        shares.push(Share {
            id: i as u64,
            value: result,
        });
    }

    shares
}

/// Reconstruct the secret from a set of shares using Lagrange interpolation.
/// Requires at least `t` shares (the threshold used during sharing).
pub fn reconstruct(shares: &[Share]) -> FieldElement {
    let n = shares.len();
    let mut result = FieldElement::ZERO;

    for j in 0..n {
        let mut num = FieldElement::ONE;
        let mut den = FieldElement::ONE;

        let xj = FieldElement::from_u64(shares[j].id);

        for k in 0..n {
            if j != k {
                let xk = FieldElement::from_u64(shares[k].id);
                num *= xk;
                den *= xk - xj;
            }
        }

        let lagrange_coeff = num * den.invert();
        result += lagrange_coeff * shares[j].value;
    }

    result
}

/// Add two sets of shares element-wise (local operation, no communication).
/// Both share vectors must have the same length and matching IDs.
pub fn share_add(a: &[Share], b: &[Share]) -> Vec<Share> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            assert_eq!(ai.id, bi.id);
            Share {
                id: ai.id,
                value: ai.value + bi.value,
            }
        })
        .collect()
}

/// Subtract two sets of shares element-wise (local operation, no communication).
pub fn share_sub(a: &[Share], b: &[Share]) -> Vec<Share> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            assert_eq!(ai.id, bi.id);
            Share {
                id: ai.id,
                value: ai.value - bi.value,
            }
        })
        .collect()
}

/// Multiply all shares by a public constant (local operation).
pub fn share_mul_const(shares: &[Share], constant: FieldElement) -> Vec<Share> {
    shares
        .iter()
        .map(|s| Share {
            id: s.id,
            value: s.value * constant,
        })
        .collect()
}

/// Add a public constant to Shamir shares. Since shares are evaluations of a
/// polynomial p(x), adding c to every evaluation yields evaluations of p(x)+c,
/// so p'(0) = p(0) + c = secret + c.
pub fn share_add_const(shares: &[Share], constant: FieldElement) -> Vec<Share> {
    shares
        .iter()
        .map(|s| Share {
            id: s.id,
            value: s.value + constant,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_reconstruct_roundtrip() {
        let mut rng = rand::thread_rng();
        let secret = FieldElement::from_u64(42);
        let shares = share(&secret, 5, 3, &mut rng);
        assert_eq!(shares.len(), 5);

        // Reconstruct with exactly t=3 shares
        let reconstructed = reconstruct(&shares[..3]);
        assert_eq!(reconstructed, secret);

        // Reconstruct with different subset
        let subset = vec![shares[0].clone(), shares[2].clone(), shares[4].clone()];
        let reconstructed2 = reconstruct(&subset);
        assert_eq!(reconstructed2, secret);
    }

    #[test]
    fn test_share_reconstruct_all() {
        let mut rng = rand::thread_rng();
        let secret = FieldElement::from_u64(12345);
        let shares = share(&secret, 10, 3, &mut rng);
        let reconstructed = reconstruct(&shares);
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_additive_homomorphism() {
        let mut rng = rand::thread_rng();
        let a = FieldElement::from_u64(100);
        let b = FieldElement::from_u64(200);

        let shares_a = share(&a, 5, 3, &mut rng);
        let shares_b = share(&b, 5, 3, &mut rng);

        let shares_sum = share_add(&shares_a, &shares_b);
        let sum = reconstruct(&shares_sum[..3]);
        assert_eq!(sum, a + b);
    }

    #[test]
    fn test_sub_homomorphism() {
        let mut rng = rand::thread_rng();
        let a = FieldElement::from_u64(300);
        let b = FieldElement::from_u64(100);

        let shares_a = share(&a, 5, 3, &mut rng);
        let shares_b = share(&b, 5, 3, &mut rng);

        let shares_diff = share_sub(&shares_a, &shares_b);
        let diff = reconstruct(&shares_diff[..3]);
        assert_eq!(diff, a - b);
    }

    #[test]
    fn test_mul_const() {
        let mut rng = rand::thread_rng();
        let secret = FieldElement::from_u64(7);
        let constant = FieldElement::from_u64(6);

        let shares = share(&secret, 5, 3, &mut rng);
        let scaled = share_mul_const(&shares, constant);
        let result = reconstruct(&scaled[..3]);
        assert_eq!(result, secret * constant);
    }

    #[test]
    fn test_add_const() {
        let mut rng = rand::thread_rng();
        let secret = FieldElement::from_u64(40);
        let constant = FieldElement::from_u64(2);

        let shares = share(&secret, 5, 3, &mut rng);
        let shifted = share_add_const(&shares, constant);
        let result = reconstruct(&shifted[..3]);
        assert_eq!(result, secret + constant);
    }

    #[test]
    fn test_threshold_1() {
        let mut rng = rand::thread_rng();
        let secret = FieldElement::from_u64(99);
        let shares = share(&secret, 3, 1, &mut rng);
        // With t=1, any single share should reconstruct the secret
        let reconstructed = reconstruct(&shares[..1]);
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_share_zero() {
        let mut rng = rand::thread_rng();
        let secret = FieldElement::ZERO;
        let shares = share(&secret, 5, 3, &mut rng);
        let reconstructed = reconstruct(&shares[..3]);
        assert_eq!(reconstructed, FieldElement::ZERO);
    }
}
