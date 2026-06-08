//! Non-native field arithmetic reference module for BLS12-381 Fp and Fp2.
//!
//! # Purpose
//!
//! Foundation for the in-circuit BLS12-381 pairing. Provides a pure-Rust,
//! *algebraic* reference implementation of the field operations that a
//! future AIR will have to constrain, modelled as multi-precision limb
//! arithmetic over the native proving field.
//!
//! Every operation is written so that its logic maps directly onto per-row
//! algebraic constraints: limbs are explicit, carries are explicit, and the
//! "quotient" of the reduction step is materialized as a witness rather than
//! hidden inside a big-integer routine.
//!
//! # Representation
//!
//! The BLS12-381 base field prime `p` is 381 bits. We represent Fp elements as
//! `[u64; 6]` in **big-endian limb order** (limb 0 = most significant), matching
//! the IETF presentation and the existing `bls12381_scheme` module. The top limb
//! uses at most 58 bits for non-reduced intermediates; reduced elements always
//! satisfy `self < p`.
//!
//! Fp2 elements are defined by the quadratic extension `u^2 = -1`.
//!
//! # Reduction algorithm
//!
//! Multiplication: schoolbook 6×6 -> 12 limbs, then reduce modulo `p` via
//! quotient-remainder decomposition. The quotient `q` (at most 6 limbs) and
//! remainder `r` (6 limbs, with `r < p`) are computed so that `a*b = q*p + r`.
//!
//! This is the "trace-row" shape an AIR will constrain:
//!
//!   prover writes:  a, b, q, r, plus limb-level cross-product witness
//!   verifier checks: 12-limb product of (a,b) matches 12-limb product of
//!                    (q,p) plus (r, 0, 0, 0, 0, 0, 0), and r < p.
//!
//! Internally, to *find* `q` from the product, we run schoolbook long division
//! at base `2^64` — that is a deterministic procedure over native `u64` lanes
//! that would itself be constrainable (but is NOT the AIR's job; the AIR just
//! checks the algebraic relation on the witnessed `q` and `r`).
//!
//! # blst cross-validation
//!
//! `#[cfg(test)]` paths call into `blst_fp_*` to sanity-check that our
//! non-native ops agree with the canonical BLS12-381 implementation. This
//! catches reduction bugs.
//!
//! # Not yet implemented
//!
//! * Fp6 / Fp12 tower (Fp6 uses mul_by_nonresidue = (1+u); Fp12 is where the
//!   Miller loop lives).
//! * Actual AIR wiring — the `*_witness` helpers just return the data; no
//!   constraint system consumes them yet.

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// BLS12-381 base field modulus, big-endian limbs (limb[0] = MSB).
///
///   p = 0x1a0111ea397fe69a 4b1ba7b6434bacd7 64774b84f38512bf
///       6730d2a0f6b0f624 1eabfffeb153ffff b9feffffffffaaab
pub const P_LIMBS: [u64; 6] = [
    0x1a0111ea397fe69a,
    0x4b1ba7b6434bacd7,
    0x64774b84f38512bf,
    0x6730d2a0f6b0f624,
    0x1eabfffeb153ffff,
    0xb9feffffffffaaab,
];

/// BLS12-381 base field modulus as an [`Fp`] constant.
pub const P: Fp = Fp { limbs: P_LIMBS };

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonnativeError {
    /// Byte encoding decoded to a value >= p.
    NotCanonical,
}

impl fmt::Display for NonnativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonnativeError::NotCanonical => {
                write!(f, "non-canonical Fp encoding: value >= p")
            }
        }
    }
}

impl std::error::Error for NonnativeError {}

// ---------------------------------------------------------------------------
// Fp: BLS12-381 base field
// ---------------------------------------------------------------------------

/// Element of the BLS12-381 base field, stored as 6 big-endian u64 limbs.
///
/// Invariant for "reduced" elements (i.e. everything returned by the public
/// API): `self < p`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp {
    /// Big-endian limbs: `limbs[0]` is the most significant u64.
    pub limbs: [u64; 6],
}

// --- construction / conversion ---------------------------------------------

impl Fp {
    /// Zero-extend a `u64` to an `Fp` element.
    #[inline]
    pub fn from_u64(x: u64) -> Self {
        Fp { limbs: [0, 0, 0, 0, 0, x] }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Fp { limbs: [0; 6] }
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Fp::from_u64(1)
    }

    /// Decode a big-endian 48-byte encoding. Rejects values `>= p`.
    pub fn from_bytes_be(b: &[u8; 48]) -> Result<Self, NonnativeError> {
        let mut limbs = [0u64; 6];
        for i in 0..6 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&b[i * 8..i * 8 + 8]);
            limbs[i] = u64::from_be_bytes(buf);
        }
        let out = Fp { limbs };
        if out.cmp_limbs(&P) != std::cmp::Ordering::Less {
            return Err(NonnativeError::NotCanonical);
        }
        Ok(out)
    }

    /// Big-endian 48-byte encoding.
    pub fn to_bytes_be(&self) -> [u8; 48] {
        let mut out = [0u8; 48];
        for i in 0..6 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.limbs[i].to_be_bytes());
        }
        out
    }

    /// Whether this element equals zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs == [0u64; 6]
    }

    /// Compare limbs as an unsigned big-endian integer.
    #[inline]
    fn cmp_limbs(&self, other: &Self) -> std::cmp::Ordering {
        for i in 0..6 {
            match self.limbs[i].cmp(&other.limbs[i]) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}

// --- primitive limb arithmetic ---------------------------------------------

/// Add two 6-limb big-endian values (no mod reduction). Returns (sum, carry_out).
/// Carry chain is from least-significant limb (index 5) up to most (index 0).
#[allow(dead_code)]
fn add6(a: &[u64; 6], b: &[u64; 6]) -> ([u64; 6], u64) {
    let mut out = [0u64; 6];
    let mut carry: u64 = 0;
    for i in (0..6).rev() {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        out[i] = s2;
        carry = (c1 as u64) + (c2 as u64);
    }
    (out, carry)
}

/// Subtract `b` from `a` (6 big-endian limbs each). Returns (diff, borrow_out).
fn sub6(a: &[u64; 6], b: &[u64; 6]) -> ([u64; 6], u64) {
    let mut out = [0u64; 6];
    let mut borrow: u64 = 0;
    for i in (0..6).rev() {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        out[i] = d2;
        borrow = (b1 as u64) + (b2 as u64);
    }
    (out, borrow)
}

/// Compare as little-endian u64 arrays (most-significant-first for our layout
/// means iterating 0..n).
fn cmp_be_slice(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        match a[i].cmp(&b[i]) {
            std::cmp::Ordering::Equal => continue,
            ord => return ord,
        }
    }
    std::cmp::Ordering::Equal
}

// --- add / sub / neg -------------------------------------------------------

/// Witness data for `Fp::sub`.
#[derive(Debug, Clone)]
pub struct SubWitness {
    /// Inputs.
    pub a: [u64; 6],
    pub b: [u64; 6],
    /// Un-reduced difference d = a - b (6 limbs, potentially underflowed).
    pub diff: [u64; 6],
    /// Borrow chain from the 6-limb subtraction. `sub_borrows[i]` is the
    /// borrow out of limb `5 - i` (LSB-first); `sub_borrows[6]` is the
    /// final borrow (equals 1 iff underflow).
    pub sub_borrows: [u64; 7],
    /// Trial sum (diff + p), 6 limbs with carry chain.
    pub trial_sum: [u64; 6],
    /// Carry chain from the trial add-back.
    pub add_carries: [u64; 7],
    /// Whether the add-back correction was taken (i.e. a < b).
    /// `1 iff sub_borrows[6] == 1`.
    pub add_back_flag: u64,
    /// Final reduced result.
    pub result: [u64; 6],
}

/// Witness data for `Fp::add`.
#[derive(Debug, Clone)]
pub struct AddWitness {
    /// Inputs.
    pub a: [u64; 6],
    pub b: [u64; 6],
    /// Un-reduced sum s = a + b (6 limbs + 1 carry bit).
    pub sum: [u64; 6],
    /// Carry chain from the 6-limb addition. `add_carries[i]` is the carry
    /// out of limb `5 - i` (LSB-first), and `add_carries[6]` is the final
    /// carry out of the most-significant limb.
    pub add_carries: [u64; 7],
    /// Trial difference (s - p), 6 limbs + borrow.
    pub trial_diff: [u64; 6],
    /// Borrow chain from the trial subtraction.
    pub sub_borrows: [u64; 7],
    /// Whether the reduction subtraction was taken (i.e. s >= p). This is
    /// 1 iff `add_carries[6] == 1 OR sub_borrows[6] == 0`.
    pub reduce_flag: u64,
    /// Final reduced result.
    pub result: [u64; 6],
}

impl Fp {
    /// `self + other mod p`.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let w = self.add_witness(other);
        Fp { limbs: w.result }
    }

    /// Produce the AIR witness data for `self + other`.
    pub fn add_witness(&self, other: &Self) -> AddWitness {
        let a = self.limbs;
        let b = other.limbs;

        // Step 1: 6-limb add with full carry chain.
        let mut sum = [0u64; 6];
        let mut add_carries = [0u64; 7];
        let mut carry: u64 = 0;
        for i in (0..6).rev() {
            let (s1, c1) = a[i].overflowing_add(b[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            sum[i] = s2;
            let cout = (c1 as u64) + (c2 as u64);
            add_carries[5 - i] = cout;
            carry = cout;
        }
        add_carries[6] = carry;
        let overflow = carry;

        // Step 2: trial subtract p with borrow chain.
        let mut trial_diff = [0u64; 6];
        let mut sub_borrows = [0u64; 7];
        let mut borrow: u64 = 0;
        for i in (0..6).rev() {
            let (d1, b1) = sum[i].overflowing_sub(P_LIMBS[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            trial_diff[i] = d2;
            let bout = (b1 as u64) + (b2 as u64);
            sub_borrows[5 - i] = bout;
            borrow = bout;
        }
        sub_borrows[6] = borrow;

        // Reduction: take trial_diff iff the unreduced sum is >= p. Overflow=1
        // means sum >= 2^384 > p, so take it. Otherwise borrow=0 means sum >= p.
        let reduce_flag = if overflow == 1 || borrow == 0 { 1 } else { 0 };
        let result = if reduce_flag == 1 { trial_diff } else { sum };

        AddWitness {
            a,
            b,
            sum,
            add_carries,
            trial_diff,
            sub_borrows,
            reduce_flag,
            result,
        }
    }

    /// `self - other mod p`.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let w = self.sub_witness(other);
        Fp { limbs: w.result }
    }

    /// Produce the AIR witness data for `self - other`.
    pub fn sub_witness(&self, other: &Self) -> SubWitness {
        let a = self.limbs;
        let b = other.limbs;

        // Step 1: 6-limb sub with full borrow chain.
        let mut diff = [0u64; 6];
        let mut sub_borrows = [0u64; 7];
        let mut borrow: u64 = 0;
        for i in (0..6).rev() {
            let (d1, b1) = a[i].overflowing_sub(b[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            diff[i] = d2;
            let bout = (b1 as u64) + (b2 as u64);
            sub_borrows[5 - i] = bout;
            borrow = bout;
        }
        sub_borrows[6] = borrow;
        let underflow = borrow;

        // Step 2: trial add p with carry chain.
        let mut trial_sum = [0u64; 6];
        let mut add_carries = [0u64; 7];
        let mut carry: u64 = 0;
        for i in (0..6).rev() {
            let (s1, c1) = diff[i].overflowing_add(P_LIMBS[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            trial_sum[i] = s2;
            let cout = (c1 as u64) + (c2 as u64);
            add_carries[5 - i] = cout;
            carry = cout;
        }
        add_carries[6] = carry;

        // If underflow, take trial_sum; else keep diff.
        let add_back_flag = underflow;
        let result = if add_back_flag == 1 { trial_sum } else { diff };

        SubWitness {
            a,
            b,
            diff,
            sub_borrows,
            trial_sum,
            add_carries,
            add_back_flag,
            result,
        }
    }

    /// Additive inverse.
    #[inline]
    pub fn neg(&self) -> Self {
        if self.is_zero() {
            Fp::zero()
        } else {
            let (diff, _b) = sub6(&P_LIMBS, &self.limbs);
            Fp { limbs: diff }
        }
    }
}

// --- multiplication --------------------------------------------------------

/// Witness data for `Fp::mul`.
#[derive(Debug, Clone)]
pub struct MulWitness {
    /// Inputs (already reduced).
    pub a: [u64; 6],
    pub b: [u64; 6],
    /// Full schoolbook product, big-endian 12 limbs.
    /// `product[0]` is the most significant u64, `product[11]` is LSB.
    pub product: [u64; 12],
    /// Quotient `q` such that `a*b = q*p + r`, big-endian 6 limbs.
    /// Since `a, b < p < 2^381`, we have `a*b < 2^762` and `q < 2^381`,
    /// so 6 limbs suffice for q.
    pub q: [u64; 6],
    /// Remainder `r = a*b mod p`, the final result.
    pub r: [u64; 6],
    /// 12-limb product `q * p` (for the checker to compare against `product`).
    pub q_times_p: [u64; 12],
}

impl Fp {
    /// `self * other mod p` (schoolbook multiplication + long-division reduction).
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let w = self.mul_witness(other);
        Fp { limbs: w.r }
    }

    /// Produce the AIR witness data for `self * other`.
    pub fn mul_witness(&self, other: &Self) -> MulWitness {
        let a = self.limbs;
        let b = other.limbs;

        // Step 1: schoolbook 6x6 -> 12-limb product, big-endian.
        // Indexing: a[i] is the i-th big-endian limb of a (weight 2^(64*(5-i))).
        // Similarly for b. The product has 12 limbs with `product[k]` at weight
        // 2^(64*(11-k)).
        //
        // contribution of a[i]*b[j] lands at big-endian position i+j (since
        // weights multiply: (5-i)+(5-j) = 10-(i+j), matching product index
        // 11-(10-(i+j)) = 1 + (i+j) ... wait, need to be careful.
        //
        // Simpler: convert to little-endian indices internally for the mul.
        let a_le: [u64; 6] = [a[5], a[4], a[3], a[2], a[1], a[0]];
        let b_le: [u64; 6] = [b[5], b[4], b[3], b[2], b[1], b[0]];
        let mut prod_le = [0u64; 12];
        for i in 0..6 {
            let mut carry: u64 = 0;
            for j in 0..6 {
                let (lo, hi) = mul_add_u64(a_le[i], b_le[j], prod_le[i + j], carry);
                prod_le[i + j] = lo;
                carry = hi;
            }
            prod_le[i + 6] = carry;
        }
        // Convert back to big-endian for witness exposure.
        let mut product = [0u64; 12];
        for i in 0..12 {
            product[i] = prod_le[11 - i];
        }

        // Step 2: long division of `product` (big-endian) by `P_LIMBS`. We
        // compute `q` (6 big-endian limbs) and `r` (6 big-endian limbs, < p).
        //
        // Implementation: schoolbook bit-by-bit division from the most
        // significant bit of `product` down. Slow but deterministic and
        // sufficient as a reference.
        let (q, r) = divmod_12_by_6(&product, &P_LIMBS);

        // Step 3: recompute q*p as 12 big-endian limbs (for the AIR check).
        let q_times_p = mul_6_by_6_be(&q, &P_LIMBS);

        MulWitness { a, b, product, q, r, q_times_p }
    }

    /// `self^2 mod p`. Implemented via `mul(self, self)`; a specialized
    /// squaring could save ~half the limb multiplies but has identical
    /// algebraic shape for AIR purposes.
    #[inline]
    pub fn square(&self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse via Fermat's little theorem: `x^(p-2) mod p`.
    /// Returns `None` iff `self == 0`.
    pub fn invert(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        // p - 2, big-endian limbs
        let p_minus_2 = {
            let (d, _b) = sub6(&P_LIMBS, &[0, 0, 0, 0, 0, 2]);
            d
        };
        Some(self.pow(&p_minus_2))
    }

    /// Compute `self^exp mod p` where `exp` is a 6-limb big-endian exponent.
    /// Square-and-multiply, scanning bits most-significant-first.
    pub fn pow(&self, exp: &[u64; 6]) -> Self {
        let mut result = Fp::one();
        let mut started = false;
        for i in 0..6 {
            let limb = exp[i];
            for bit in (0..64).rev() {
                if started {
                    result = result.square();
                }
                if ((limb >> bit) & 1) == 1 {
                    if !started {
                        result = *self;
                        started = true;
                    } else {
                        result = result.mul(self);
                    }
                }
            }
        }
        if !started {
            // exp == 0 => result is 1 by convention.
            return Fp::one();
        }
        result
    }
}

/// Schoolbook-style single-step: computes `a*b + c + carry = lo + (hi << 64)`.
#[inline]
fn mul_add_u64(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    // Use u128 to get the 128-bit product, then add c and carry.
    let wide = (a as u128) * (b as u128) + (c as u128) + (carry as u128);
    (wide as u64, (wide >> 64) as u64)
}

/// Big-endian 6x6 -> 12-limb multiplication.
fn mul_6_by_6_be(a: &[u64; 6], b: &[u64; 6]) -> [u64; 12] {
    let a_le: [u64; 6] = [a[5], a[4], a[3], a[2], a[1], a[0]];
    let b_le: [u64; 6] = [b[5], b[4], b[3], b[2], b[1], b[0]];
    let mut prod_le = [0u64; 12];
    for i in 0..6 {
        let mut carry: u64 = 0;
        for j in 0..6 {
            let (lo, hi) = mul_add_u64(a_le[i], b_le[j], prod_le[i + j], carry);
            prod_le[i + j] = lo;
            carry = hi;
        }
        prod_le[i + 6] = carry;
    }
    let mut out = [0u64; 12];
    for i in 0..12 {
        out[i] = prod_le[11 - i];
    }
    out
}

/// Divide 12-limb big-endian `numerator` by 6-limb big-endian `modulus`,
/// returning `(quotient_6_limbs_be, remainder_6_limbs_be)`.
///
/// Used to derive the Fp reduction witness. The algorithm is schoolbook
/// bit-by-bit division from the most-significant bit down; it's slow but
/// directly mirrors what a per-row AIR constraint would look like.
fn divmod_12_by_6(numerator: &[u64; 12], modulus: &[u64; 6]) -> ([u64; 6], [u64; 6]) {
    // We'll represent the running remainder as 12 big-endian limbs (same
    // layout as `numerator`) so that we can do in-place 12-limb subtractions
    // of `modulus` (padded on the left with 6 zeros).
    let mut rem: [u64; 12] = *numerator;
    let mut quot_bits = [0u8; 6 * 64]; // most-significant bit first

    // Modulus extended to 12 limbs, left-padded with zeros.
    // Since quotient has at most 6 limbs = 384 bits, and we scan the top 384
    // bits of the 768-bit numerator, the modulus shift `s` ranges over
    // 0..=383 bit positions.
    //
    // For each shift position s (from high to low), we check whether the
    // remainder is >= (modulus << s). If so, subtract and set the quotient bit.
    for s in (0..384).rev() {
        // Build `modulus << s` as 12 big-endian limbs.
        // Bit position s counts from the LSB of the 12-limb container (which
        // has 12*64 = 768 bits). We want modulus placed such that its LSB
        // sits at bit s.
        //
        // A simpler construction: shift the 6-limb modulus left by `s` bits
        // into a 12-limb buffer. The modulus is 381 bits, so the result
        // occupies at most `381 + s` bits (<= 768 when s <= 383 since
        // 381 + 383 = 764 < 768 — so never overflows).
        let shifted = shift_left_6_into_12(modulus, s);

        // Compare rem to shifted (12-limb big-endian, unsigned).
        if cmp_be_slice(&rem, &shifted) != std::cmp::Ordering::Less {
            // Subtract in place.
            let mut borrow: u64 = 0;
            for i in (0..12).rev() {
                let (d1, b1) = rem[i].overflowing_sub(shifted[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                rem[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
            }
            // Set quotient bit at position s.
            // quot_bits is indexed most-significant-bit-first: index 0 = bit 383.
            let idx = 383 - s;
            quot_bits[idx] = 1;
        }
    }

    // Pack quot_bits (MSB-first) into 6 big-endian limbs.
    let mut q = [0u64; 6];
    for i in 0..384 {
        if quot_bits[i] == 1 {
            let limb_index = i / 64; // big-endian limb index (0 = MSB)
            let bit_in_limb = 63 - (i % 64);
            q[limb_index] |= 1u64 << bit_in_limb;
        }
    }

    // Remainder is the low 6 big-endian limbs of `rem`. (High 6 must be zero
    // since r < p < 2^381 < 2^(64*6).)
    let mut r = [0u64; 6];
    r.copy_from_slice(&rem[6..12]);
    debug_assert!(
        rem[..6].iter().all(|&x| x == 0),
        "division remainder overflowed 6 limbs: {:?}",
        rem
    );

    (q, r)
}

/// Shift a 6-limb big-endian value left by `s` bits into a 12-limb big-endian
/// buffer. Caller ensures `s + 381 < 768` so no overflow.
fn shift_left_6_into_12(src: &[u64; 6], s: usize) -> [u64; 12] {
    let mut out = [0u64; 12];
    // Place `src` into the low 6 limbs of `out`.
    out[6..12].copy_from_slice(src);
    // Now shift the 12-limb value left by `s` bits.
    let limb_shift = s / 64;
    let bit_shift = s % 64;

    // First, shift by whole limbs (move bytes left = toward lower big-endian
    // index).
    if limb_shift > 0 {
        let mut shifted = [0u64; 12];
        for i in 0..12 {
            let src_idx = i + limb_shift;
            if src_idx < 12 {
                shifted[i] = out[src_idx];
            }
        }
        out = shifted;
    }

    // Then, shift by `bit_shift` bits within the 12-limb buffer (big-endian).
    if bit_shift > 0 {
        let mut shifted = [0u64; 12];
        for i in 0..12 {
            let hi = out[i] << bit_shift;
            let lo = if i + 1 < 12 {
                out[i + 1] >> (64 - bit_shift)
            } else {
                0
            };
            shifted[i] = hi | lo;
        }
        out = shifted;
    }

    out
}

// ---------------------------------------------------------------------------
// Fp2: quadratic extension with u^2 = -1
// ---------------------------------------------------------------------------

/// Element of `Fp2 = Fp[u] / (u^2 + 1)`. Written as `c0 + c1 * u`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp2 {
    pub c0: Fp,
    pub c1: Fp,
}

impl Fp2 {
    #[inline]
    pub fn zero() -> Self {
        Fp2 { c0: Fp::zero(), c1: Fp::zero() }
    }

    #[inline]
    pub fn one() -> Self {
        Fp2 { c0: Fp::one(), c1: Fp::zero() }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Fp2 {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
        }
    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Fp2 {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
        }
    }

    #[inline]
    pub fn neg(&self) -> Self {
        Fp2 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    /// Complex conjugate: `c0 + c1*u -> c0 - c1*u`.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Fp2 {
            c0: self.c0,
            c1: self.c1.neg(),
        }
    }

    /// `(c0 + c1 u)(d0 + d1 u) = (c0 d0 - c1 d1) + (c0 d1 + c1 d0) u`
    /// Karatsuba form: save one Fp multiply at the cost of two Fp additions.
    ///    t0 = c0 * d0
    ///    t1 = c1 * d1
    ///    t2 = (c0 + c1)(d0 + d1) - t0 - t1 = c0 d1 + c1 d0
    ///    result = (t0 - t1) + t2 * u
    pub fn mul(&self, other: &Self) -> Self {
        let t0 = self.c0.mul(&other.c0);
        let t1 = self.c1.mul(&other.c1);
        let sum_a = self.c0.add(&self.c1);
        let sum_b = other.c0.add(&other.c1);
        let t_cross = sum_a.mul(&sum_b);
        let t2 = t_cross.sub(&t0).sub(&t1);
        let new_c0 = t0.sub(&t1);
        Fp2 { c0: new_c0, c1: t2 }
    }

    /// Specialized squaring. `(c0 + c1 u)^2 = (c0 + c1)(c0 - c1) + 2 c0 c1 u`.
    pub fn square(&self) -> Self {
        let a = self.c0.add(&self.c1);
        let b = self.c0.sub(&self.c1);
        let c0 = a.mul(&b);
        let c0c1 = self.c0.mul(&self.c1);
        let c1 = c0c1.add(&c0c1);
        Fp2 { c0, c1 }
    }

    /// `self^{-1}`. For `u^2 = -1`, `(c0 + c1 u)^{-1} = (c0 - c1 u)/(c0^2 + c1^2)`.
    /// Returns `None` iff `self == 0`.
    pub fn invert(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let norm = self.c0.square().add(&self.c1.square());
        let norm_inv = norm.invert()?;
        Some(Fp2 {
            c0: self.c0.mul(&norm_inv),
            c1: self.c1.neg().mul(&norm_inv),
        })
    }

    /// Multiply by the Fp2 non-residue `u`: `(c0 + c1 u) * u = -c1 + c0 u`.
    ///
    /// This is the Fp2->Fp2 non-residue multiplication. The Fp6/Fp12 tower
    /// uses a *different* mul_by_nonresidue with `(1 + u)`; we don't need
    /// that one here.
    #[inline]
    pub fn mul_by_nonresidue(&self) -> Self {
        Fp2 {
            c0: self.c1.neg(),
            c1: self.c0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (including blst cross-validation)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use blst::*;

    // ---- utilities to bridge Fp <-> blst_fp ------------------------------

    /// Convert our `Fp` into `blst_fp`. We hand big-endian bytes to
    /// `blst_fp_from_bendian`, which parses canonical encodings and internally
    /// converts to Montgomery form.
    fn fp_to_blst(fp: &Fp) -> blst_fp {
        let bytes = fp.to_bytes_be();
        let mut out = blst_fp::default();
        unsafe { blst_fp_from_bendian(&mut out, bytes.as_ptr()); }
        out
    }

    /// Convert a `blst_fp` back to our `Fp`.
    fn blst_to_fp(bfp: &blst_fp) -> Fp {
        let mut bytes = [0u8; 48];
        unsafe { blst_bendian_from_fp(bytes.as_mut_ptr(), bfp); }
        Fp::from_bytes_be(&bytes).expect("blst output must be canonical")
    }

    fn sample_fps() -> Vec<Fp> {
        vec![
            Fp::zero(),
            Fp::one(),
            Fp::from_u64(2),
            Fp::from_u64(0xffffffffffffffff),
            // Hand-picked non-trivial 6-limb values, all strictly less than p.
            Fp { limbs: [
                0x0000000000000001,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ]},
            Fp { limbs: [
                0x0123456789abcdef,
                0xfedcba9876543210,
                0xdeadbeefcafebabe,
                0x0f0e0d0c0b0a0908,
                0x0706050403020100,
                0x8080808080808080,
            ]},
            Fp { limbs: [
                0x19ffffffffffffff,
                0xffffffffffffffff,
                0xffffffffffffffff,
                0xffffffffffffffff,
                0xffffffffffffffff,
                0xffffffffffffffff,
            ]},
            // p - 1
            Fp { limbs: [
                P_LIMBS[0], P_LIMBS[1], P_LIMBS[2],
                P_LIMBS[3], P_LIMBS[4], P_LIMBS[5] - 1,
            ]},
        ]
    }

    // ---- required tests --------------------------------------------------

    #[test]
    fn roundtrip_parse_zero_and_one() {
        let zero_bytes = [0u8; 48];
        let parsed_zero = Fp::from_bytes_be(&zero_bytes).unwrap();
        assert_eq!(parsed_zero, Fp::zero());
        assert_eq!(Fp::zero().to_bytes_be(), zero_bytes);

        let mut one_bytes = [0u8; 48];
        one_bytes[47] = 1;
        assert_eq!(Fp::one().to_bytes_be(), one_bytes);
        let parsed_one = Fp::from_bytes_be(&one_bytes).unwrap();
        assert_eq!(parsed_one, Fp::one());
    }

    #[test]
    fn overflow_rejection() {
        let all_ff = [0xffu8; 48];
        let err = Fp::from_bytes_be(&all_ff).unwrap_err();
        assert_eq!(err, NonnativeError::NotCanonical);

        // Also check exactly p is rejected.
        let p_bytes = P.to_bytes_be();
        let err = Fp::from_bytes_be(&p_bytes).unwrap_err();
        assert_eq!(err, NonnativeError::NotCanonical);

        // p - 1 is accepted.
        let mut pm1_bytes = P.to_bytes_be();
        pm1_bytes[47] -= 1;
        let pm1 = Fp::from_bytes_be(&pm1_bytes).unwrap();
        assert_eq!(pm1.limbs[5], P_LIMBS[5] - 1);
    }

    #[test]
    fn addition_identity() {
        for a in sample_fps() {
            let sum = a.add(&Fp::zero());
            assert_eq!(sum, a, "a + 0 != a for a = {:?}", a);
            let sum = Fp::zero().add(&a);
            assert_eq!(sum, a, "0 + a != a for a = {:?}", a);
        }
    }

    #[test]
    fn subtraction_cancels_addition() {
        let samples = sample_fps();
        for a in &samples {
            for b in &samples {
                let s = a.add(b);
                let back = s.sub(b);
                assert_eq!(back, *a, "(a + b) - b != a for a={:?} b={:?}", a, b);
            }
        }
    }

    #[test]
    fn multiplication_by_zero_and_one() {
        for a in sample_fps() {
            assert_eq!(a.mul(&Fp::zero()), Fp::zero());
            assert_eq!(Fp::zero().mul(&a), Fp::zero());
            assert_eq!(a.mul(&Fp::one()), a, "a * 1 != a for a = {:?}", a);
            assert_eq!(Fp::one().mul(&a), a);
        }
    }

    // ---- blst cross-checks ----------------------------------------------

    #[test]
    fn cross_check_add() {
        for a in sample_fps() {
            for b in sample_fps() {
                let ours = a.add(&b);

                let ba = fp_to_blst(&a);
                let bb = fp_to_blst(&b);
                let mut bc = blst_fp::default();
                unsafe { blst_fp_add(&mut bc, &ba, &bb); }
                let theirs = blst_to_fp(&bc);

                assert_eq!(
                    ours, theirs,
                    "add mismatch: a={:?} b={:?}, ours={:?} theirs={:?}",
                    a, b, ours, theirs
                );
            }
        }
    }

    #[test]
    fn cross_check_sub() {
        for a in sample_fps() {
            for b in sample_fps() {
                let ours = a.sub(&b);

                let ba = fp_to_blst(&a);
                let bb = fp_to_blst(&b);
                let mut bc = blst_fp::default();
                unsafe { blst_fp_sub(&mut bc, &ba, &bb); }
                let theirs = blst_to_fp(&bc);

                assert_eq!(
                    ours, theirs,
                    "sub mismatch: a={:?} b={:?}",
                    a, b
                );
            }
        }
    }

    #[test]
    fn cross_check_neg() {
        for a in sample_fps() {
            let ours = a.neg();

            let ba = fp_to_blst(&a);
            let mut bc = blst_fp::default();
            unsafe { blst_fp_cneg(&mut bc, &ba, true); }
            let theirs = blst_to_fp(&bc);

            // Note: blst_fp_cneg(x, true) = -x; for x = 0 we also expect 0.
            assert_eq!(ours, theirs, "neg mismatch for a = {:?}", a);
        }
    }

    #[test]
    fn cross_check_mul() {
        for a in sample_fps() {
            for b in sample_fps() {
                let ours = a.mul(&b);

                let ba = fp_to_blst(&a);
                let bb = fp_to_blst(&b);
                let mut bc = blst_fp::default();
                unsafe { blst_fp_mul(&mut bc, &ba, &bb); }
                let theirs = blst_to_fp(&bc);

                assert_eq!(
                    ours, theirs,
                    "mul mismatch: a={:?} b={:?}",
                    a, b
                );
            }
        }
    }

    #[test]
    fn cross_check_square() {
        for a in sample_fps() {
            let ours = a.square();

            let ba = fp_to_blst(&a);
            let mut bc = blst_fp::default();
            unsafe { blst_fp_sqr(&mut bc, &ba); }
            let theirs = blst_to_fp(&bc);

            assert_eq!(ours, theirs, "square mismatch for a = {:?}", a);
        }
    }

    #[test]
    fn cross_check_invert() {
        for a in sample_fps() {
            if a.is_zero() {
                assert!(a.invert().is_none());
                continue;
            }
            let ours = a.invert().expect("non-zero element inverts");

            let ba = fp_to_blst(&a);
            let mut bc = blst_fp::default();
            unsafe { blst_fp_inverse(&mut bc, &ba); }
            let theirs = blst_to_fp(&bc);

            assert_eq!(ours, theirs, "invert mismatch for a = {:?}", a);

            // And sanity: a * a^{-1} == 1
            let prod = a.mul(&ours);
            assert_eq!(prod, Fp::one());
        }
    }

    #[test]
    fn mul_witness_relation() {
        // Verify that q*p + r reconstructs the 12-limb product exactly.
        let a = Fp { limbs: [
            0x0123456789abcdef,
            0xfedcba9876543210,
            0xdeadbeefcafebabe,
            0x0f0e0d0c0b0a0908,
            0x0706050403020100,
            0x8080808080808080,
        ]};
        let b = Fp { limbs: [
            0x0000000100000001,
            0x0000000200000002,
            0x0000000300000003,
            0x0000000400000004,
            0x0000000500000005,
            0x0000000600000006,
        ]};
        let w = a.mul_witness(&b);

        // Compute q*p + r as 12 big-endian limbs.
        // q*p is already 12 big-endian limbs in w.q_times_p.
        // r is 6 big-endian limbs; extend to 12 by zero-padding on the high end.
        let mut r_extended = [0u64; 12];
        r_extended[6..12].copy_from_slice(&w.r);

        // Add q_times_p + r_extended.
        let mut sum = [0u64; 12];
        let mut carry: u64 = 0;
        for i in (0..12).rev() {
            let (s1, c1) = w.q_times_p[i].overflowing_add(r_extended[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            sum[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        assert_eq!(carry, 0, "q*p + r overflowed 12 limbs");
        assert_eq!(sum, w.product, "q*p + r != a*b");
        assert!(cmp_be_slice(&w.r, &P_LIMBS) == std::cmp::Ordering::Less,
                "remainder r must be < p");
    }

    #[test]
    fn add_witness_carries() {
        // p - 1 + 1 = p, should reduce to 0.
        let pm1 = Fp { limbs: [
            P_LIMBS[0], P_LIMBS[1], P_LIMBS[2],
            P_LIMBS[3], P_LIMBS[4], P_LIMBS[5] - 1,
        ]};
        let w = pm1.add_witness(&Fp::one());
        assert_eq!(w.result, [0u64; 6]);
        assert_eq!(w.reduce_flag, 1);

        // 1 + 2 = 3, no reduction.
        let w = Fp::from_u64(1).add_witness(&Fp::from_u64(2));
        assert_eq!(w.result, [0, 0, 0, 0, 0, 3]);
        assert_eq!(w.reduce_flag, 0);
    }

    // ---- Fp2 tests -------------------------------------------------------

    fn sample_fp2s() -> Vec<Fp2> {
        let samples = sample_fps();
        let mut out = Vec::new();
        // Just pair some Fps. Avoid the full n^2 explosion.
        for (i, c0) in samples.iter().enumerate() {
            let c1 = &samples[(i + 3) % samples.len()];
            out.push(Fp2 { c0: *c0, c1: *c1 });
        }
        out
    }

    #[test]
    fn fp2_difference_of_squares() {
        // (a + b)(a - b) == a^2 - b^2
        for a in sample_fp2s() {
            for b in &sample_fp2s() {
                let lhs = a.add(b).mul(&a.sub(b));
                let rhs = a.square().sub(&b.square());
                assert_eq!(lhs, rhs, "(a+b)(a-b) != a^2 - b^2 for a={:?} b={:?}", a, b);
            }
        }
    }

    #[test]
    fn fp2_binomial_square() {
        // (a + b)^2 == a^2 + 2ab + b^2
        for a in sample_fp2s() {
            for b in &sample_fp2s() {
                let lhs = a.add(b).square();
                let ab = a.mul(b);
                let two_ab = ab.add(&ab);
                let rhs = a.square().add(&two_ab).add(&b.square());
                assert_eq!(lhs, rhs, "(a+b)^2 != a^2 + 2ab + b^2");
            }
        }
    }

    #[test]
    fn fp2_invert_is_inverse() {
        for a in sample_fp2s() {
            match a.invert() {
                Some(inv) => {
                    let prod = a.mul(&inv);
                    assert_eq!(prod, Fp2::one(), "a * a^-1 != 1 for a = {:?}", a);
                }
                None => {
                    assert!(a.is_zero(), "invert returned None for nonzero {:?}", a);
                }
            }
        }
    }

    #[test]
    fn fp2_mul_by_nonresidue_is_mul_by_u() {
        // u = Fp2 { c0: 0, c1: 1 }; mul_by_nonresidue should equal *u.
        let u = Fp2 { c0: Fp::zero(), c1: Fp::one() };
        for a in sample_fp2s() {
            let via_method = a.mul_by_nonresidue();
            let via_mul = a.mul(&u);
            assert_eq!(via_method, via_mul, "mul_by_nonresidue != *u for a={:?}", a);
        }
    }

    #[test]
    fn fp2_conjugate_involutive() {
        for a in sample_fp2s() {
            assert_eq!(a.conjugate().conjugate(), a);
        }
    }

    #[test]
    fn p_constant_matches_bls12_381() {
        // Sanity: P, serialized big-endian, equals the canonical BLS12-381 p.
        let bytes = P.to_bytes_be();
        // First byte should be 0x1a.
        assert_eq!(bytes[0], 0x1a);
        // Last byte should be 0xab.
        assert_eq!(bytes[47], 0xab);
        // P is not canonical as an Fp encoding (it equals the modulus).
        assert!(Fp::from_bytes_be(&bytes).is_err());
    }
}
