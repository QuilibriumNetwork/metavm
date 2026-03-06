//! Curve-agnostic scalar field abstraction.
//!
//! Provides a `Scalar` enum that dispatches field operations to either
//! BLS48-581 (via `bls48581::big::BIG`) or BLS12-381 (via `blst::blst_fr`).
//! This allows the proving pipeline to work with either curve at runtime.

use bls48581::bls48581::big;
use bls48581::bls48581::rom;

/// Which pairing-friendly curve to use for commitments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CurveType {
    Bls48581,
    Bls12381,
}

/// A scalar field element that dispatches to the appropriate curve implementation.
#[derive(Clone)]
pub enum Scalar {
    Bls48581(big::BIG),
    Bls12381(blst::blst_fr),
}

impl std::fmt::Debug for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Bls48581(_) => write!(f, "Scalar::Bls48581(...)"),
            Scalar::Bls12381(_) => write!(f, "Scalar::Bls12381(...)"),
        }
    }
}

impl Scalar {
    /// Create a zero scalar for the given curve.
    pub fn zero(curve: CurveType) -> Self {
        match curve {
            CurveType::Bls48581 => Scalar::Bls48581(big::BIG::new()),
            CurveType::Bls12381 => {
                let fr = blst::blst_fr::default();
                Scalar::Bls12381(fr)
            }
        }
    }

    /// Create a one scalar for the given curve.
    pub fn one(curve: CurveType) -> Self {
        match curve {
            CurveType::Bls48581 => Scalar::Bls48581(big::BIG::new_int(1)),
            CurveType::Bls12381 => {
                let mut fr = blst::blst_fr::default();
                let mut scalar = blst::blst_scalar::default();
                let val: u64 = 1;
                unsafe {
                    blst::blst_scalar_from_uint64(&mut scalar, [val, 0, 0, 0].as_ptr());
                    blst::blst_fr_from_scalar(&mut fr, &scalar);
                }
                Scalar::Bls12381(fr)
            }
        }
    }

    /// Create a scalar from a u64 value.
    pub fn from_u64(val: u64, curve: CurveType) -> Self {
        match curve {
            CurveType::Bls48581 => {
                let mut buf = [0u8; big::MODBYTES];
                let start = big::MODBYTES - 8;
                buf[start..].copy_from_slice(&val.to_be_bytes());
                Scalar::Bls48581(big::BIG::frombytes(&buf))
            }
            CurveType::Bls12381 => {
                let mut fr = blst::blst_fr::default();
                let mut scalar = blst::blst_scalar::default();
                unsafe {
                    blst::blst_scalar_from_uint64(&mut scalar, [val, 0, 0, 0].as_ptr());
                    blst::blst_fr_from_scalar(&mut fr, &scalar);
                }
                Scalar::Bls12381(fr)
            }
        }
    }

    /// Check if this scalar is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Scalar::Bls48581(b) => b.iszilch(),
            Scalar::Bls12381(fr) => {
                // Convert back to scalar to check
                let mut scalar = blst::blst_scalar::default();
                unsafe {
                    blst::blst_scalar_from_fr(&mut scalar, fr);
                }
                scalar.b == [0u8; 32]
            }
        }
    }

    /// Field addition.
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Scalar::Bls48581(a), Scalar::Bls48581(b)) => {
                let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
                Scalar::Bls48581(big::BIG::modadd(a, b, &modulus))
            }
            (Scalar::Bls12381(a), Scalar::Bls12381(b)) => {
                let mut result = blst::blst_fr::default();
                unsafe { blst::blst_fr_add(&mut result, a, b); }
                Scalar::Bls12381(result)
            }
            _ => panic!("Cannot mix curve types in field operations"),
        }
    }

    /// Field subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        match (self, other) {
            (Scalar::Bls48581(a), Scalar::Bls48581(b)) => {
                let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
                let neg_b = big::BIG::modneg(b, &modulus);
                Scalar::Bls48581(big::BIG::modadd(a, &neg_b, &modulus))
            }
            (Scalar::Bls12381(a), Scalar::Bls12381(b)) => {
                let mut result = blst::blst_fr::default();
                unsafe { blst::blst_fr_sub(&mut result, a, b); }
                Scalar::Bls12381(result)
            }
            _ => panic!("Cannot mix curve types in field operations"),
        }
    }

    /// Field multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Scalar::Bls48581(a), Scalar::Bls48581(b)) => {
                let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
                Scalar::Bls48581(big::BIG::modmul(a, b, &modulus))
            }
            (Scalar::Bls12381(a), Scalar::Bls12381(b)) => {
                let mut result = blst::blst_fr::default();
                unsafe { blst::blst_fr_mul(&mut result, a, b); }
                Scalar::Bls12381(result)
            }
            _ => panic!("Cannot mix curve types in field operations"),
        }
    }

    /// Field negation.
    pub fn neg(&self) -> Self {
        match self {
            Scalar::Bls48581(a) => {
                let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
                Scalar::Bls48581(big::BIG::modneg(a, &modulus))
            }
            Scalar::Bls12381(a) => {
                // negate by computing 0 - a
                let mut result = blst::blst_fr::default();
                let zero = blst::blst_fr::default();
                unsafe { blst::blst_fr_sub(&mut result, &zero, a); }
                Scalar::Bls12381(result)
            }
        }
    }

    /// Serialize to bytes (big-endian).
    /// BLS48-581: MODBYTES (73) bytes. BLS12-381: 32 bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Scalar::Bls48581(b) => {
                let mut buf = vec![0u8; big::MODBYTES];
                b.tobytes(&mut buf);
                buf
            }
            Scalar::Bls12381(fr) => {
                let mut scalar = blst::blst_scalar::default();
                unsafe {
                    blst::blst_scalar_from_fr(&mut scalar, fr);
                }
                scalar.b.to_vec()
            }
        }
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8], curve: CurveType) -> Self {
        match curve {
            CurveType::Bls48581 => {
                let mut buf = [0u8; big::MODBYTES];
                let copy_len = bytes.len().min(big::MODBYTES);
                // Right-align for big-endian
                let start = big::MODBYTES - copy_len;
                buf[start..start + copy_len].copy_from_slice(&bytes[..copy_len]);
                Scalar::Bls48581(big::BIG::frombytes(&buf))
            }
            CurveType::Bls12381 => {
                let mut scalar = blst::blst_scalar::default();
                let copy_len = bytes.len().min(32);
                scalar.b[..copy_len].copy_from_slice(&bytes[..copy_len]);
                let mut fr = blst::blst_fr::default();
                unsafe {
                    blst::blst_fr_from_scalar(&mut fr, &scalar);
                }
                Scalar::Bls12381(fr)
            }
        }
    }

    /// Convert to u64 (for small values like instruction type/funct selectors).
    pub fn to_u64(&self) -> u64 {
        match self {
            Scalar::Bls48581(b) => {
                let mut buf = [0u8; big::MODBYTES];
                b.tobytes(&mut buf);
                let start = big::MODBYTES - 8;
                u64::from_be_bytes([
                    buf[start], buf[start + 1], buf[start + 2], buf[start + 3],
                    buf[start + 4], buf[start + 5], buf[start + 6], buf[start + 7],
                ])
            }
            Scalar::Bls12381(fr) => {
                let mut scalar = blst::blst_scalar::default();
                unsafe {
                    blst::blst_scalar_from_fr(&mut scalar, fr);
                }
                // blst scalar bytes are little-endian
                u64::from_le_bytes([
                    scalar.b[0], scalar.b[1], scalar.b[2], scalar.b[3],
                    scalar.b[4], scalar.b[5], scalar.b[6], scalar.b[7],
                ])
            }
        }
    }

    /// Return the byte length of a serialized scalar for this curve.
    pub fn byte_len(&self) -> usize {
        match self {
            Scalar::Bls48581(_) => big::MODBYTES,
            Scalar::Bls12381(_) => 32,
        }
    }

    /// Return which curve this scalar belongs to.
    pub fn curve_type(&self) -> CurveType {
        match self {
            Scalar::Bls48581(_) => CurveType::Bls48581,
            Scalar::Bls12381(_) => CurveType::Bls12381,
        }
    }

    /// Unwrap as a BLS48-581 BIG. Panics if wrong variant.
    pub fn as_bls48581(&self) -> &big::BIG {
        match self {
            Scalar::Bls48581(b) => b,
            _ => panic!("Expected Bls48581 scalar"),
        }
    }

    /// Unwrap as a mutable BLS48-581 BIG. Panics if wrong variant.
    pub fn as_bls48581_mut(&mut self) -> &mut big::BIG {
        match self {
            Scalar::Bls48581(b) => b,
            _ => panic!("Expected Bls48581 scalar"),
        }
    }

    /// Unwrap as a BLS12-381 blst_fr. Panics if wrong variant.
    pub fn as_bls12381(&self) -> &blst::blst_fr {
        match self {
            Scalar::Bls12381(fr) => fr,
            _ => panic!("Expected Bls12381 scalar"),
        }
    }

    /// Convert BLS12-381 scalar to blst_scalar (for point multiplication).
    /// Panics if wrong variant.
    pub fn to_blst_scalar(&self) -> blst::blst_scalar {
        match self {
            Scalar::Bls12381(fr) => {
                let mut scalar = blst::blst_scalar::default();
                unsafe { blst::blst_scalar_from_fr(&mut scalar, fr); }
                scalar
            }
            _ => panic!("Expected Bls12381 scalar"),
        }
    }

    /// Check if the scalar is one.
    pub fn is_one(&self) -> bool {
        match self {
            Scalar::Bls48581(b) => b.isunity(),
            Scalar::Bls12381(_) => {
                let one = Self::one(CurveType::Bls12381);
                self.sub(&one).is_zero()
            }
        }
    }

    /// Field multiplicative inverse: a^{-1} such that a * a^{-1} = 1.
    ///
    /// Panics if invoked on zero (zero has no inverse).
    pub fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "Cannot invert zero");
        match self {
            Scalar::Bls48581(a) => {
                let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
                let mut result = big::BIG::new_copy(a);
                result.invmodp(&modulus);
                Scalar::Bls48581(result)
            }
            Scalar::Bls12381(a) => {
                let mut result = blst::blst_fr::default();
                unsafe { blst::blst_fr_inverse(&mut result, a); }
                Scalar::Bls12381(result)
            }
        }
    }

    /// Reduce modulo the curve order (for BLS48-581; no-op for BLS12-381 since
    /// blst_fr is always reduced).
    pub fn reduce(&mut self) {
        match self {
            Scalar::Bls48581(b) => {
                let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
                b.rmod(&modulus);
            }
            Scalar::Bls12381(_) => {} // always reduced
        }
    }

    /// Convert 32 Fiat-Shamir challenge bytes to a scalar.
    pub fn from_challenge_bytes(challenge_bytes: &[u8; 32], curve: CurveType) -> Self {
        match curve {
            CurveType::Bls48581 => {
                let mut z_padded = [0u8; big::MODBYTES];
                z_padded[big::MODBYTES - 32..].copy_from_slice(challenge_bytes);
                let mut z = big::BIG::frombytes(&z_padded);
                let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
                z.rmod(&modulus);
                Scalar::Bls48581(z)
            }
            CurveType::Bls12381 => {
                // Interpret as little-endian scalar and reduce
                let mut scalar = blst::blst_scalar::default();
                scalar.b[..32].copy_from_slice(challenge_bytes);
                let mut fr = blst::blst_fr::default();
                unsafe {
                    blst::blst_fr_from_scalar(&mut fr, &scalar);
                }
                Scalar::Bls12381(fr)
            }
        }
    }

    /// Return the scalar field byte length for a given curve type.
    pub fn byte_len_for_curve(curve: CurveType) -> usize {
        match curve {
            CurveType::Bls48581 => big::MODBYTES,
            CurveType::Bls12381 => 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_is_zero() {
        let z48 = Scalar::zero(CurveType::Bls48581);
        assert!(z48.is_zero());
        let z12 = Scalar::zero(CurveType::Bls12381);
        assert!(z12.is_zero());
    }

    #[test]
    fn test_one_is_not_zero() {
        let o48 = Scalar::one(CurveType::Bls48581);
        assert!(!o48.is_zero());
        assert!(o48.is_one());
        let o12 = Scalar::one(CurveType::Bls12381);
        assert!(!o12.is_zero());
        assert!(o12.is_one());
    }

    #[test]
    fn test_from_u64_roundtrip() {
        for &val in &[0u64, 1, 42, 1000, u64::MAX] {
            let s48 = Scalar::from_u64(val, CurveType::Bls48581);
            assert_eq!(s48.to_u64(), val, "BLS48-581 roundtrip failed for {}", val);

            let s12 = Scalar::from_u64(val, CurveType::Bls12381);
            assert_eq!(s12.to_u64(), val, "BLS12-381 roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_add() {
        let a = Scalar::from_u64(10, CurveType::Bls48581);
        let b = Scalar::from_u64(20, CurveType::Bls48581);
        let c = a.add(&b);
        assert_eq!(c.to_u64(), 30);

        let a12 = Scalar::from_u64(10, CurveType::Bls12381);
        let b12 = Scalar::from_u64(20, CurveType::Bls12381);
        let c12 = a12.add(&b12);
        assert_eq!(c12.to_u64(), 30);
    }

    #[test]
    fn test_sub() {
        let a = Scalar::from_u64(30, CurveType::Bls48581);
        let b = Scalar::from_u64(10, CurveType::Bls48581);
        let c = a.sub(&b);
        assert_eq!(c.to_u64(), 20);

        let a12 = Scalar::from_u64(30, CurveType::Bls12381);
        let b12 = Scalar::from_u64(10, CurveType::Bls12381);
        let c12 = a12.sub(&b12);
        assert_eq!(c12.to_u64(), 20);
    }

    #[test]
    fn test_mul() {
        let a = Scalar::from_u64(6, CurveType::Bls48581);
        let b = Scalar::from_u64(7, CurveType::Bls48581);
        let c = a.mul(&b);
        assert_eq!(c.to_u64(), 42);

        let a12 = Scalar::from_u64(6, CurveType::Bls12381);
        let b12 = Scalar::from_u64(7, CurveType::Bls12381);
        let c12 = a12.mul(&b12);
        assert_eq!(c12.to_u64(), 42);
    }

    #[test]
    fn test_neg_add_is_zero() {
        let a = Scalar::from_u64(42, CurveType::Bls48581);
        let neg_a = a.neg();
        let sum = a.add(&neg_a);
        assert!(sum.is_zero());

        let a12 = Scalar::from_u64(42, CurveType::Bls12381);
        let neg_a12 = a12.neg();
        let sum12 = a12.add(&neg_a12);
        assert!(sum12.is_zero());
    }

    #[test]
    fn test_inverse() {
        let a48 = Scalar::from_u64(7, CurveType::Bls48581);
        let inv48 = a48.inverse();
        let prod48 = a48.mul(&inv48);
        assert!(prod48.is_one(), "BLS48-581: 7 * 7^-1 should be 1");

        let a12 = Scalar::from_u64(7, CurveType::Bls12381);
        let inv12 = a12.inverse();
        let prod12 = a12.mul(&inv12);
        assert!(prod12.is_one(), "BLS12-381: 7 * 7^-1 should be 1");
    }

    #[test]
    fn test_inverse_large() {
        let a48 = Scalar::from_u64(123456789, CurveType::Bls48581);
        let inv48 = a48.inverse();
        let prod48 = a48.mul(&inv48);
        assert!(prod48.is_one());

        let a12 = Scalar::from_u64(123456789, CurveType::Bls12381);
        let inv12 = a12.inverse();
        let prod12 = a12.mul(&inv12);
        assert!(prod12.is_one());
    }

    #[test]
    fn test_inverse_one() {
        let one48 = Scalar::one(CurveType::Bls48581);
        let inv48 = one48.inverse();
        assert!(inv48.is_one(), "1^-1 should be 1");

        let one12 = Scalar::one(CurveType::Bls12381);
        let inv12 = one12.inverse();
        assert!(inv12.is_one(), "1^-1 should be 1");
    }

    #[test]
    #[should_panic(expected = "Cannot invert zero")]
    fn test_inverse_zero_panics_48() {
        let z = Scalar::zero(CurveType::Bls48581);
        let _ = z.inverse();
    }

    #[test]
    #[should_panic(expected = "Cannot invert zero")]
    fn test_inverse_zero_panics_12() {
        let z = Scalar::zero(CurveType::Bls12381);
        let _ = z.inverse();
    }

    #[test]
    fn test_curve_type() {
        let s48 = Scalar::zero(CurveType::Bls48581);
        assert_eq!(s48.curve_type(), CurveType::Bls48581);
        let s12 = Scalar::zero(CurveType::Bls12381);
        assert_eq!(s12.curve_type(), CurveType::Bls12381);
    }

    #[test]
    fn test_byte_len() {
        let s48 = Scalar::zero(CurveType::Bls48581);
        assert_eq!(s48.byte_len(), big::MODBYTES); // 73
        let s12 = Scalar::zero(CurveType::Bls12381);
        assert_eq!(s12.byte_len(), 32);
    }
}
