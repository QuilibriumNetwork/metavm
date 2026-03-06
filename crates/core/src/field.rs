use curve25519_dalek::scalar::Scalar;
use rand::RngCore;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Field element over the Curve25519 scalar field (order ~2^252).
/// Wraps `curve25519_dalek::Scalar` for use in secret sharing and MPC protocols.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FieldElement(pub Scalar);

impl FieldElement {
    pub const ZERO: Self = FieldElement(Scalar::ZERO);
    pub const ONE: Self = FieldElement(Scalar::ONE);

    pub fn from_u64(val: u64) -> Self {
        FieldElement(Scalar::from(val))
    }

    pub fn random<R: RngCore + rand::CryptoRng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        FieldElement(Scalar::from_bytes_mod_order(bytes))
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        FieldElement(Scalar::from_bytes_mod_order(bytes))
    }

    pub fn from_bytes_mod_order(bytes: [u8; 32]) -> Self {
        FieldElement(Scalar::from_bytes_mod_order(bytes))
    }

    pub fn invert(&self) -> Self {
        FieldElement(self.0.invert())
    }

    pub fn inner(&self) -> &Scalar {
        &self.0
    }

    pub fn into_inner(self) -> Scalar {
        self.0
    }
}

impl fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FieldElement({:?})", &self.0.to_bytes()[..8])
    }
}

impl fmt::Display for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.0.to_bytes();
        write!(f, "0x")?;
        for b in bytes.iter().rev().take(8) {
            write!(f, "{:02x}", b)?;
        }
        write!(f, "...")
    }
}

impl From<u64> for FieldElement {
    fn from(val: u64) -> Self {
        FieldElement::from_u64(val)
    }
}

impl From<Scalar> for FieldElement {
    fn from(s: Scalar) -> Self {
        FieldElement(s)
    }
}

impl Add for FieldElement {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        FieldElement(self.0 + rhs.0)
    }
}

impl AddAssign for FieldElement {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for FieldElement {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        FieldElement(self.0 - rhs.0)
    }
}

impl SubAssign for FieldElement {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for FieldElement {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        FieldElement(self.0 * rhs.0)
    }
}

impl MulAssign for FieldElement {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Neg for FieldElement {
    type Output = Self;
    fn neg(self) -> Self {
        FieldElement(-self.0)
    }
}

impl std::iter::Sum for FieldElement {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(FieldElement::ZERO, |acc, x| acc + x)
    }
}

/// Maps a u64 VM register value into a field element.
/// Since the Curve25519 scalar field is ~2^252, all u64 values fit without reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Word(pub u64);

impl Word {
    pub fn to_field(&self) -> FieldElement {
        FieldElement::from_u64(self.0)
    }

    pub fn from_field(f: &FieldElement) -> Self {
        let bytes = f.to_bytes();
        let val = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        Word(val)
    }
}

impl From<u64> for Word {
    fn from(val: u64) -> Self {
        Word(val)
    }
}

impl From<Word> for u64 {
    fn from(w: Word) -> u64 {
        w.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_arithmetic() {
        let a = FieldElement::from_u64(42);
        let b = FieldElement::from_u64(58);
        let c = a + b;
        assert_eq!(c, FieldElement::from_u64(100));
    }

    #[test]
    fn test_field_sub() {
        let a = FieldElement::from_u64(100);
        let b = FieldElement::from_u64(42);
        let c = a - b;
        assert_eq!(c, FieldElement::from_u64(58));
    }

    #[test]
    fn test_field_mul() {
        let a = FieldElement::from_u64(6);
        let b = FieldElement::from_u64(7);
        let c = a * b;
        assert_eq!(c, FieldElement::from_u64(42));
    }

    #[test]
    fn test_field_invert() {
        let a = FieldElement::from_u64(7);
        let a_inv = a.invert();
        let product = a * a_inv;
        assert_eq!(product, FieldElement::ONE);
    }

    #[test]
    fn test_field_bytes_roundtrip() {
        let a = FieldElement::from_u64(123456789);
        let bytes = a.to_bytes();
        let b = FieldElement::from_bytes(bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_word_field_roundtrip() {
        let w = Word(0xDEADBEEFCAFEBABE);
        let f = w.to_field();
        let w2 = Word::from_field(&f);
        assert_eq!(w, w2);
    }

    #[test]
    fn test_field_random() {
        let mut rng = rand::thread_rng();
        let a = FieldElement::random(&mut rng);
        let b = FieldElement::random(&mut rng);
        assert_ne!(a, b);
    }

    #[test]
    fn test_field_zero_one() {
        let z = FieldElement::ZERO;
        let o = FieldElement::ONE;
        assert_eq!(z + o, o);
        assert_eq!(o * z, z);
        assert_eq!(o * o, o);
    }

    #[test]
    fn test_field_neg() {
        let a = FieldElement::from_u64(42);
        let neg_a = -a;
        assert_eq!(a + neg_a, FieldElement::ZERO);
    }
}
