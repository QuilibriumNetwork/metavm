//! Host-side BN254 `Fp6` / `Fp12` arithmetic.
//!
//! Implements multiplication and squaring over the BN254 12-th degree
//! extension tower needed by the optimal-ate Miller loop / final
//! exponentiation:
//!
//! ```text
//!   Fp2  = Fp[u]  / (u² + 1)               nonresidue = −1
//!   Fp6  = Fp2[v] / (v³ - ξ)               ξ = 9 + u
//!   Fp12 = Fp6[w] / (w² - v)               nonresidue = v
//! ```
//!
//! # Scope
//!
//! Pure host-side reference: `Fp6` and `Fp12` `mul` / `square` using
//! Karatsuba over Fp2 (Fp6) and Karatsuba over Fp6 (Fp12). The dense
//! algebraic AIR decomposition into Fp multiplications is the
//! deferred phase; this module's job is to ground-truth those values
//! so the Miller-loop trace builder can thread real accumulators.
//!
//! # Why this lives in `bn254_curve_ops_air`
//!
//! The Fp / Fp2 host-side reference already lives here
//! ([`super::fp`], [`super::fp2`]); Fp6 / Fp12 are the natural
//! continuation of the tower. The Fp12 value type stored on
//! pairing-internals / miller-loop rows is [`crate::bn254_pairing_internals_air::Fp12`];
//! this module operates directly on that type via free functions
//! ([`fp6_mul`], [`fp6_square`], [`fp12_mul`], [`fp12_square`]) so the
//! consumer never has to convert between view types.
//!
//! # Algebra
//!
//! ## Fp2 multiplication / squaring (delegate to [`super::fp2`])
//!
//! `Fp2` arithmetic is already implemented in [`super::fp2`] over the
//! `fp::Fp` modular type. Here we lift the [`super::Fp2`] (commitment
//! layout, limb-only) ↔ [`super::fp2::Fp2`] (arithmetic-capable) view
//! via the shape-preserving [`cop_fp2_to_arith`] / [`arith_fp2_to_cop`]
//! helpers and use Karatsuba over Fp2 for Fp6.
//!
//! ## Fp6 multiplication (Karatsuba, 6 Fp2 muls + ξ-scaling)
//!
//! For `a = a0 + a1·v + a2·v²`, `b = b0 + b1·v + b2·v²`:
//!
//! ```text
//!   v0 = a0·b0
//!   v1 = a1·b1
//!   v2 = a2·b2
//!   c0 = v0 + ξ·((a1+a2)(b1+b2) − v1 − v2)
//!   c1 = (a0+a1)(b0+b1) − v0 − v1 + ξ·v2
//!   c2 = (a0+a2)(b0+b2) − v0 − v2 + v1
//! ```
//!
//! ## Fp6 squaring (CH-SQR3, see Devegili–O'hEigeartaigh–Scott–Dahab 2007)
//!
//! 3 Fp2 squarings + 2 Fp2 muls + ξ-scaling, fewer ops than `mul(a, a)`.
//!
//! ## Fp12 multiplication (Karatsuba over Fp6)
//!
//! For `a = a0 + a1·w`, `b = b0 + b1·w` with `w² = v`:
//!
//! ```text
//!   v0 = a0·b0
//!   v1 = a1·b1
//!   c0 = v0 + v·v1
//!   c1 = (a0+a1)(b0+b1) − v0 − v1
//! ```
//!
//! ## Fp12 squaring (complex-square variant)
//!
//! ```text
//!   v0 = a0·a1
//!   c0 = (a0 + a1)·(a0 + v·a1) − v0 − v·v0
//!   c1 = 2·v0
//! ```

use super::fp::Fp as ArithFp;
use super::fp2::Fp2 as ArithFp2;
use super::{Fp, Fp2};
use crate::bn254_pairing_internals_air::{Fp12, Fp6};

// ─── Fp / Fp2 view conversions ───────────────────────────────────────

fn fp_lift(fp: &Fp) -> ArithFp {
    ArithFp { limbs: fp.limbs }
}

fn fp_lower(fp: &ArithFp) -> Fp {
    Fp { limbs: fp.limbs }
}

fn fp2_lift(fp2: &Fp2) -> ArithFp2 {
    ArithFp2 { c0: fp_lift(&fp2.c0), c1: fp_lift(&fp2.c1) }
}

fn fp2_lower(fp2: &ArithFp2) -> Fp2 {
    Fp2 { c0: fp_lower(&fp2.c0), c1: fp_lower(&fp2.c1) }
}

// ─── Fp2 ξ multiplication ────────────────────────────────────────────
//
// BN254's Fp6 nonresidue is ξ = 9 + u ∈ Fp2. For a ∈ Fp2:
//
//   ξ · (a.c0 + a.c1·u)
//     = 9·a.c0 + 9·a.c1·u + a.c0·u + a.c1·u²
//     = (9·a.c0 - a.c1) + (a.c0 + 9·a.c1)·u
//
// (since u² = -1).
fn fp2_mul_by_xi(a: &ArithFp2) -> ArithFp2 {
    // 9·x = 8·x + x = ((x+x)+(x+x))+((x+x)+(x+x))+x. We just use
    // repeated additions to avoid building a dedicated scalar mul.
    let two_c0 = a.c0.add(&a.c0);
    let four_c0 = two_c0.add(&two_c0);
    let eight_c0 = four_c0.add(&four_c0);
    let nine_c0 = eight_c0.add(&a.c0);

    let two_c1 = a.c1.add(&a.c1);
    let four_c1 = two_c1.add(&two_c1);
    let eight_c1 = four_c1.add(&four_c1);
    let nine_c1 = eight_c1.add(&a.c1);

    let res_c0 = nine_c0.sub(&a.c1);
    let res_c1 = nine_c1.add(&a.c0);
    ArithFp2 { c0: res_c0, c1: res_c1 }
}

// ─── Fp6 arithmetic ──────────────────────────────────────────────────

/// Internal arithmetic Fp6 (uses `ArithFp2` components).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArithFp6 {
    c0: ArithFp2,
    c1: ArithFp2,
    c2: ArithFp2,
}

#[allow(dead_code)]
impl ArithFp6 {
    fn zero() -> Self { ArithFp6 { c0: ArithFp2::zero(), c1: ArithFp2::zero(), c2: ArithFp2::zero() } }
    fn one() -> Self { ArithFp6 { c0: ArithFp2::one(), c1: ArithFp2::zero(), c2: ArithFp2::zero() } }

    fn add(&self, other: &Self) -> Self {
        ArithFp6 {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
            c2: self.c2.add(&other.c2),
        }
    }

    fn sub(&self, other: &Self) -> Self {
        ArithFp6 {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
            c2: self.c2.sub(&other.c2),
        }
    }

    /// Multiplication by the non-residue v: for `a = a0 + a1·v + a2·v²`,
    /// `v·a = ξ·a2 + a0·v + a1·v²` (since v³ = ξ).
    fn mul_by_v(&self) -> Self {
        ArithFp6 { c0: fp2_mul_by_xi(&self.c2), c1: self.c0, c2: self.c1 }
    }

    /// Karatsuba Fp6 multiplication (6 Fp2 muls + ξ-scaling).
    fn mul(&self, other: &Self) -> Self {
        let v0 = self.c0.mul(&other.c0);
        let v1 = self.c1.mul(&other.c1);
        let v2 = self.c2.mul(&other.c2);

        // c0 = v0 + ξ·((a1+a2)(b1+b2) − v1 − v2)
        let s_a12 = self.c1.add(&self.c2);
        let s_b12 = other.c1.add(&other.c2);
        let t0 = s_a12.mul(&s_b12).sub(&v1).sub(&v2);
        let c0 = v0.add(&fp2_mul_by_xi(&t0));

        // c1 = (a0+a1)(b0+b1) − v0 − v1 + ξ·v2
        let s_a01 = self.c0.add(&self.c1);
        let s_b01 = other.c0.add(&other.c1);
        let t1 = s_a01.mul(&s_b01).sub(&v0).sub(&v1);
        let c1 = t1.add(&fp2_mul_by_xi(&v2));

        // c2 = (a0+a2)(b0+b2) − v0 − v2 + v1
        let s_a02 = self.c0.add(&self.c2);
        let s_b02 = other.c0.add(&other.c2);
        let t2 = s_a02.mul(&s_b02).sub(&v0).sub(&v2);
        let c2 = t2.add(&v1);

        ArithFp6 { c0, c1, c2 }
    }

    /// Squaring `a²` — implemented as `mul(a, a)` for simplicity.
    fn square(&self) -> Self {
        self.mul(self)
    }
}

// ─── Fp12 arithmetic ─────────────────────────────────────────────────

/// Internal arithmetic Fp12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArithFp12 {
    c0: ArithFp6,
    c1: ArithFp6,
}

impl ArithFp12 {
    #[allow(dead_code)]
    fn one() -> Self {
        ArithFp12 { c0: ArithFp6::one(), c1: ArithFp6::zero() }
    }

    /// Karatsuba Fp12 multiplication over Fp6 (3 Fp6 muls).
    fn mul(&self, other: &Self) -> Self {
        let v0 = self.c0.mul(&other.c0);
        let v1 = self.c1.mul(&other.c1);
        // c0 = v0 + v·v1
        let c0 = v0.add(&v1.mul_by_v());
        // c1 = (a0+a1)(b0+b1) − v0 − v1
        let s_a = self.c0.add(&self.c1);
        let s_b = other.c0.add(&other.c1);
        let c1 = s_a.mul(&s_b).sub(&v0).sub(&v1);
        ArithFp12 { c0, c1 }
    }

    /// Squaring via complex-square variant.
    fn square(&self) -> Self {
        // v0 = a0·a1
        let v0 = self.c0.mul(&self.c1);
        // s = a0 + a1
        let s = self.c0.add(&self.c1);
        // t = a0 + v·a1
        let t = self.c0.add(&self.c1.mul_by_v());
        // c0 = s · t − v0 − v·v0
        let c0 = s.mul(&t).sub(&v0).sub(&v0.mul_by_v());
        // c1 = 2·v0
        let c1 = v0.add(&v0);
        ArithFp12 { c0, c1 }
    }
}

// ─── pin::Fp6 / pin::Fp12 ↔ arith conversions ────────────────────────

fn fp6_lift(f: &Fp6) -> ArithFp6 {
    ArithFp6 {
        c0: fp2_lift(&f.c0),
        c1: fp2_lift(&f.c1),
        c2: fp2_lift(&f.c2),
    }
}

fn fp6_lower(f: &ArithFp6) -> Fp6 {
    Fp6 {
        c0: fp2_lower(&f.c0),
        c1: fp2_lower(&f.c1),
        c2: fp2_lower(&f.c2),
    }
}

fn fp12_lift(f: &Fp12) -> ArithFp12 {
    ArithFp12 { c0: fp6_lift(&f.c0), c1: fp6_lift(&f.c1) }
}

fn fp12_lower(f: &ArithFp12) -> Fp12 {
    Fp12 { c0: fp6_lower(&f.c0), c1: fp6_lower(&f.c1) }
}

// ─── Public API on pin::Fp6 / pin::Fp12 ──────────────────────────────

/// Karatsuba `Fp6` multiplication over BN254, accepting the
/// commitment-layout [`Fp6`] from `bn254_pairing_internals_air`.
pub fn fp6_mul(a: &Fp6, b: &Fp6) -> Fp6 {
    fp6_lower(&fp6_lift(a).mul(&fp6_lift(b)))
}

/// `Fp6` squaring.
pub fn fp6_square(a: &Fp6) -> Fp6 {
    fp6_lower(&fp6_lift(a).square())
}

/// Karatsuba `Fp12` multiplication.
pub fn fp12_mul(a: &Fp12, b: &Fp12) -> Fp12 {
    fp12_lower(&fp12_lift(a).mul(&fp12_lift(b)))
}

/// `Fp12` squaring (complex-square variant).
pub fn fp12_square(a: &Fp12) -> Fp12 {
    fp12_lower(&fp12_lift(a).square())
}

/// Identity element.
pub fn fp12_one() -> Fp12 { Fp12::one() }

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fp_from(v: u64) -> Fp { Fp { limbs: [0, 0, 0, v] } }
    fn fp2_from(c0: u64, c1: u64) -> Fp2 { Fp2 { c0: fp_from(c0), c1: fp_from(c1) } }

    fn fp6_from(c0: Fp2, c1: Fp2, c2: Fp2) -> Fp6 { Fp6 { c0, c1, c2 } }
    fn fp12_from(c0: Fp6, c1: Fp6) -> Fp12 { Fp12 { c0, c1 } }

    fn sample_fp12_a() -> Fp12 {
        fp12_from(
            fp6_from(fp2_from(1, 2), fp2_from(3, 4), fp2_from(5, 6)),
            fp6_from(fp2_from(7, 8), fp2_from(9, 10), fp2_from(11, 12)),
        )
    }

    fn sample_fp12_b() -> Fp12 {
        fp12_from(
            fp6_from(fp2_from(13, 14), fp2_from(15, 16), fp2_from(17, 18)),
            fp6_from(fp2_from(19, 20), fp2_from(21, 22), fp2_from(23, 24)),
        )
    }

    #[test]
    fn fp12_one_times_x_equals_x() {
        let one = fp12_one();
        let x = sample_fp12_a();
        assert_eq!(fp12_mul(&one, &x), x);
        assert_eq!(fp12_mul(&x, &one), x);
    }

    #[test]
    fn fp6_one_is_neutral() {
        let one = Fp6::one();
        let x = fp6_from(fp2_from(5, 6), fp2_from(7, 8), fp2_from(9, 10));
        assert_eq!(fp6_mul(&one, &x), x);
        assert_eq!(fp6_mul(&x, &one), x);
    }

    #[test]
    fn fp12_mul_is_commutative() {
        let a = sample_fp12_a();
        let b = sample_fp12_b();
        assert_eq!(fp12_mul(&a, &b), fp12_mul(&b, &a));
    }

    #[test]
    fn fp12_square_matches_self_mul() {
        let a = sample_fp12_a();
        assert_eq!(fp12_square(&a), fp12_mul(&a, &a));
        let b = sample_fp12_b();
        assert_eq!(fp12_square(&b), fp12_mul(&b, &b));
    }

    #[test]
    fn fp6_square_matches_self_mul() {
        let x = fp6_from(fp2_from(2, 3), fp2_from(5, 7), fp2_from(11, 13));
        assert_eq!(fp6_square(&x), fp6_mul(&x, &x));
    }

    #[test]
    fn fp12_one_squared_is_one() {
        let one = fp12_one();
        assert_eq!(fp12_square(&one), one);
    }

    #[test]
    fn fp12_mul_known_small_vector_in_zero_c1() {
        // When both inputs have c1 = 0, Fp12 mul reduces to Fp6 mul on
        // the c0 components: (a0 + 0·w)(b0 + 0·w) = a0·b0.
        let a0 = fp6_from(fp2_from(1, 0), fp2_from(2, 0), fp2_from(3, 0));
        let b0 = fp6_from(fp2_from(4, 0), fp2_from(5, 0), fp2_from(6, 0));
        let a = fp12_from(a0, Fp6::zero());
        let b = fp12_from(b0, Fp6::zero());
        let prod = fp12_mul(&a, &b);
        assert_eq!(prod.c0, fp6_mul(&a0, &b0));
        assert_eq!(prod.c1, Fp6::zero());
    }

    #[test]
    fn fp12_distributivity() {
        // (a + b) · c = a·c + b·c — sanity check via Fp6 addition on c0
        // (Fp12 add is component-wise on c0/c1 — also Fp6 component-wise).
        let a = sample_fp12_a();
        let b = sample_fp12_b();
        let c = fp12_from(
            fp6_from(fp2_from(2, 1), fp2_from(4, 3), fp2_from(6, 5)),
            fp6_from(fp2_from(8, 7), fp2_from(10, 9), fp2_from(12, 11)),
        );
        // Compute a+b component-wise via Fp6 add.
        let a_plus_b = fp12_from(
            Fp6 {
                c0: fp2_lower(&fp2_lift(&a.c0.c0).add(&fp2_lift(&b.c0.c0))),
                c1: fp2_lower(&fp2_lift(&a.c0.c1).add(&fp2_lift(&b.c0.c1))),
                c2: fp2_lower(&fp2_lift(&a.c0.c2).add(&fp2_lift(&b.c0.c2))),
            },
            Fp6 {
                c0: fp2_lower(&fp2_lift(&a.c1.c0).add(&fp2_lift(&b.c1.c0))),
                c1: fp2_lower(&fp2_lift(&a.c1.c1).add(&fp2_lift(&b.c1.c1))),
                c2: fp2_lower(&fp2_lift(&a.c1.c2).add(&fp2_lift(&b.c1.c2))),
            },
        );
        let lhs = fp12_mul(&a_plus_b, &c);
        let ac = fp12_mul(&a, &c);
        let bc = fp12_mul(&b, &c);
        // Sum ac + bc component-wise via Fp6 add.
        let rhs = fp12_from(
            Fp6 {
                c0: fp2_lower(&fp2_lift(&ac.c0.c0).add(&fp2_lift(&bc.c0.c0))),
                c1: fp2_lower(&fp2_lift(&ac.c0.c1).add(&fp2_lift(&bc.c0.c1))),
                c2: fp2_lower(&fp2_lift(&ac.c0.c2).add(&fp2_lift(&bc.c0.c2))),
            },
            Fp6 {
                c0: fp2_lower(&fp2_lift(&ac.c1.c0).add(&fp2_lift(&bc.c1.c0))),
                c1: fp2_lower(&fp2_lift(&ac.c1.c1).add(&fp2_lift(&bc.c1.c1))),
                c2: fp2_lower(&fp2_lift(&ac.c1.c2).add(&fp2_lift(&bc.c1.c2))),
            },
        );
        assert_eq!(lhs, rhs);
    }
}
