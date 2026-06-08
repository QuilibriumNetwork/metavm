//! Non-native tower extension reference for BLS12-381: Fp6 and Fp12.
//!
//! # Purpose
//!
//! Follow-on to `nonnative_fp` (which provides Fp, Fp2). This module builds
//! the cubic extension Fp6 and the quadratic extension Fp12 on top, giving us
//! the field arithmetic that lives inside the Miller loop and final
//! exponentiation of the BLS12-381 optimal-ate pairing.
//!
//! # Tower shape (BLS12-381 convention, matching blst / pairing-friendly RFC)
//!
//! ```text
//!   Fp2  = Fp[u]  / (u^2 + 1)           // nonresidue = -1
//!   Fp6  = Fp2[v] / (v^3 − ξ)           // ξ = 1 + u  ∈ Fp2
//!   Fp12 = Fp6[w] / (w^2 − v)           // nonresidue on this level = v
//! ```
//!
//! All operations are plain field arithmetic built on `Fp` and `Fp2`
//! primitives. Like `nonnative_fp`, the code is deliberately written in a way
//! that mirrors an AIR constraint trace: multiplications use Karatsuba-style
//! expansions with explicit sub-product temporaries.
//!
//! # What's here
//!
//! * Fp6 add/sub/neg, full mul, specialized square, inverse, mul_by_nonresidue,
//!   and two sparse-mul helpers (mul_by_01, mul_by_1) used by the Miller loop.
//! * Fp12 add/sub/neg, full mul, specialized square, inverse, conjugate, and
//!   the sparse mul_by_014 helper used when accumulating a line evaluation.
//!
//! # What's stubbed / not here
//!
//! * `Fp12::cyclotomic_square` — currently implemented as a direct alias to
//!   `Fp12::square` (which is always correct; Granger–Scott is just an
//!   optimization on the cyclotomic subgroup). The AIR version, when built,
//!   should use the specialized 3-Fp2-mul formula from Granger–Scott 2010 for
//!   cost. Correctness on cyclotomic inputs is cross-checked in tests.
//! * Final exponentiation / Miller loop / pairing — intentionally out of
//!   scope for this file.

use crate::nonnative_fp::{Fp, Fp2, P_LIMBS};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Helpers on top of Fp2: nonresidue for the Fp6 tower is (1 + u).
// ---------------------------------------------------------------------------

/// Multiply an Fp2 element by the Fp6 tower nonresidue `ξ = 1 + u`.
///
/// `(c0 + c1 u)(1 + u) = (c0 − c1) + (c0 + c1) u`   (since `u^2 = −1`).
///
/// NB: `Fp2::mul_by_nonresidue()` already exists in `nonnative_fp`, but that
/// one multiplies by `u`, which is the Fp2→Fp2 quadratic-extension nonresidue.
/// The tower uses `ξ = 1 + u` as the nonresidue for the cubic step, which is
/// what this helper computes.
#[inline]
fn fp2_mul_by_xi(a: &Fp2) -> Fp2 {
    let c0 = a.c0.sub(&a.c1);
    let c1 = a.c0.add(&a.c1);
    Fp2 { c0, c1 }
}

// ---------------------------------------------------------------------------
// Fp6 = Fp2[v] / (v^3 − ξ),  represented as c0 + c1·v + c2·v^2
// ---------------------------------------------------------------------------

/// Element of the cubic extension Fp6. The basis is `{1, v, v^2}` with
/// `v^3 = ξ = 1 + u`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp6 {
    pub c0: Fp2,
    pub c1: Fp2,
    pub c2: Fp2,
}

impl Fp6 {
    // ----- constants --------------------------------------------------------

    #[inline]
    pub fn zero() -> Self {
        Fp6 { c0: Fp2::zero(), c1: Fp2::zero(), c2: Fp2::zero() }
    }

    #[inline]
    pub fn one() -> Self {
        Fp6 { c0: Fp2::one(), c1: Fp2::zero(), c2: Fp2::zero() }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero() && self.c2.is_zero()
    }

    // ----- additive ops -----------------------------------------------------

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Fp6 {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
            c2: self.c2.add(&other.c2),
        }
    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Fp6 {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
            c2: self.c2.sub(&other.c2),
        }
    }

    #[inline]
    pub fn neg(&self) -> Self {
        Fp6 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
            c2: self.c2.neg(),
        }
    }

    // ----- multiplicative ops ----------------------------------------------

    /// Full Fp6 multiplication using the Karatsuba/Toom-Cook-style layout
    /// standard for cubic towers (Aranha et al., blst `sqt.c`).
    ///
    /// Let a = a0 + a1 v + a2 v^2, b = b0 + b1 v + b2 v^2. Then the product
    /// in Fp6 (with `v^3 = ξ`) is:
    ///
    ///   c0 = a0 b0 + ξ (a1 b2 + a2 b1)
    ///   c1 = a0 b1 + a1 b0 + ξ a2 b2
    ///   c2 = a0 b2 + a1 b1 + a2 b0
    ///
    /// Using Karatsuba we compute only 6 Fp2 mults:
    ///   v0 = a0 b0,  v1 = a1 b1,  v2 = a2 b2
    ///   c0 = v0 + ξ * ((a1+a2)(b1+b2) − v1 − v2)
    ///   c1 = (a0+a1)(b0+b1) − v0 − v1 + ξ * v2
    ///   c2 = (a0+a2)(b0+b2) − v0 − v2 + v1
    pub fn mul(&self, other: &Self) -> Self {
        let v0 = self.c0.mul(&other.c0);
        let v1 = self.c1.mul(&other.c1);
        let v2 = self.c2.mul(&other.c2);

        // c0 = v0 + ξ * ((a1+a2)(b1+b2) − v1 − v2)
        let sum_a12 = self.c1.add(&self.c2);
        let sum_b12 = other.c1.add(&other.c2);
        let t = sum_a12.mul(&sum_b12).sub(&v1).sub(&v2);
        let c0 = v0.add(&fp2_mul_by_xi(&t));

        // c1 = (a0+a1)(b0+b1) − v0 − v1 + ξ * v2
        let sum_a01 = self.c0.add(&self.c1);
        let sum_b01 = other.c0.add(&other.c1);
        let t = sum_a01.mul(&sum_b01).sub(&v0).sub(&v1);
        let c1 = t.add(&fp2_mul_by_xi(&v2));

        // c2 = (a0+a2)(b0+b2) − v0 − v2 + v1
        let sum_a02 = self.c0.add(&self.c2);
        let sum_b02 = other.c0.add(&other.c2);
        let t = sum_a02.mul(&sum_b02).sub(&v0).sub(&v2);
        let c2 = t.add(&v1);

        Fp6 { c0, c1, c2 }
    }

    /// Specialized cubic-extension squaring ("CH-SQR3" from Chung-Hasan),
    /// 5 Fp2 multiplications (or 2 squarings + 3 mults) vs 6 for generic mul.
    ///
    /// Using the following witness temporaries:
    ///   s0 = a0^2
    ///   s1 = 2 a0 a1
    ///   s2 = (a0 − a1 + a2)^2
    ///   s3 = 2 a1 a2
    ///   s4 = a2^2
    ///
    /// Output:
    ///   c0 = s0 + ξ s3
    ///   c1 = s1 + ξ s4
    ///   c2 = s1 + s2 + s3 − s0 − s4
    pub fn square(&self) -> Self {
        let s0 = self.c0.square();
        let ab = self.c0.mul(&self.c1);
        let s1 = ab.add(&ab);
        let s2 = self.c0.sub(&self.c1).add(&self.c2).square();
        let bc = self.c1.mul(&self.c2);
        let s3 = bc.add(&bc);
        let s4 = self.c2.square();

        let c0 = s0.add(&fp2_mul_by_xi(&s3));
        let c1 = s1.add(&fp2_mul_by_xi(&s4));
        let c2 = s1.add(&s2).add(&s3).sub(&s0).sub(&s4);

        Fp6 { c0, c1, c2 }
    }

    /// Multiplication by `v`: `(c0 + c1 v + c2 v^2) * v = c2 ξ + c0 v + c1 v^2`.
    /// This is the tower-level nonresidue multiplier used by Fp12.
    #[inline]
    pub fn mul_by_nonresidue(&self) -> Self {
        Fp6 {
            c0: fp2_mul_by_xi(&self.c2),
            c1: self.c0,
            c2: self.c1,
        }
    }

    /// Sparse multiplication: `self * (c0 + c1 v)` where c2 = 0.
    ///
    /// Derivation (Aranha §3 / RFC pairing-friendly-curves §4.3.3):
    ///   v0 = a0 c0,  v1 = a1 c1
    ///   out.c0 = v0 + ξ (a2 c1)                        ( = v0 + ξ ((a1+a2)(0+c1) − a1 c1) simplified )
    ///   out.c1 = (a0 + a1)(c0 + c1) − v0 − v1
    ///   out.c2 = (a0 + a2) c0 − v0 + v1                 ( = (a0+a2)(c0+0) − v0 + v1 )
    pub fn mul_by_01(&self, c0: &Fp2, c1: &Fp2) -> Self {
        let a_a = self.c0.mul(c0);
        let b_b = self.c1.mul(c1);

        // out.c0 = ξ * (a2 * c1) + a_a
        let t1 = self.c2.mul(c1);
        let out_c0 = fp2_mul_by_xi(&t1).add(&a_a);

        // out.c1 = (a0 + a1)(c0 + c1) − a_a − b_b
        let sum_a01 = self.c0.add(&self.c1);
        let sum_c01 = c0.add(c1);
        let t2 = sum_a01.mul(&sum_c01);
        let out_c1 = t2.sub(&a_a).sub(&b_b);

        // out.c2 = (a0 + a2) * c0 − a_a + b_b
        let sum_a02 = self.c0.add(&self.c2);
        let t3 = sum_a02.mul(c0);
        let out_c2 = t3.sub(&a_a).add(&b_b);

        Fp6 { c0: out_c0, c1: out_c1, c2: out_c2 }
    }

    /// Sparse multiplication: `self * (c1 v)` where c0 = c2 = 0.
    ///
    /// Derivation:
    ///   self * c1 v = (a0 + a1 v + a2 v^2) * c1 v
    ///              = a0 c1 v + a1 c1 v^2 + a2 c1 v^3
    ///              = ξ (a2 c1) + (a0 c1) v + (a1 c1) v^2
    pub fn mul_by_1(&self, c1: &Fp2) -> Self {
        let b_b = self.c1.mul(c1);
        let t1 = self.c2.mul(c1);
        let t2 = self.c0.mul(c1);

        Fp6 {
            c0: fp2_mul_by_xi(&t1),
            c1: t2,
            c2: b_b,
        }
    }

    /// Multiplicative inverse for Fp6. Cubic-extension inversion via the
    /// "Chung-Hasan" trick:
    ///
    ///   For a = a0 + a1 v + a2 v^2, define
    ///     t0 = a0^2 − ξ a1 a2
    ///     t1 = ξ a2^2 − a0 a1
    ///     t2 = a1^2 − a0 a2
    ///   Then denom = a0 t0 + ξ a2 t1 + ξ a1 t2   (an Fp2 element)
    ///   and a^{-1} = (t0 + t1 v + t2 v^2) / denom.
    ///
    /// Returns `None` iff `self == 0`.
    pub fn invert(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let a0_sq = self.c0.square();
        let a1_sq = self.c1.square();
        let a2_sq = self.c2.square();
        let a0_a1 = self.c0.mul(&self.c1);
        let a0_a2 = self.c0.mul(&self.c2);
        let a1_a2 = self.c1.mul(&self.c2);

        let t0 = a0_sq.sub(&fp2_mul_by_xi(&a1_a2));
        let t1 = fp2_mul_by_xi(&a2_sq).sub(&a0_a1);
        let t2 = a1_sq.sub(&a0_a2);

        // denom = a0 t0 + ξ * (a2 t1 + a1 t2)
        let inner = self.c2.mul(&t1).add(&self.c1.mul(&t2));
        let denom = self.c0.mul(&t0).add(&fp2_mul_by_xi(&inner));

        let denom_inv = denom.invert()?;

        Some(Fp6 {
            c0: t0.mul(&denom_inv),
            c1: t1.mul(&denom_inv),
            c2: t2.mul(&denom_inv),
        })
    }
}

// ---------------------------------------------------------------------------
// Fp12 = Fp6[w] / (w^2 − v),  represented as c0 + c1·w
// ---------------------------------------------------------------------------

/// Element of the quadratic extension Fp12 over Fp6. The basis is `{1, w}`
/// with `w^2 = v` (where `v` is the Fp6 indeterminate / nonresidue).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp12 {
    pub c0: Fp6,
    pub c1: Fp6,
}

impl Fp12 {
    // ----- constants --------------------------------------------------------

    #[inline]
    pub fn zero() -> Self {
        Fp12 { c0: Fp6::zero(), c1: Fp6::zero() }
    }

    #[inline]
    pub fn one() -> Self {
        Fp12 { c0: Fp6::one(), c1: Fp6::zero() }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    // ----- additive ops -----------------------------------------------------

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Fp12 {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
        }
    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Fp12 {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
        }
    }

    #[inline]
    pub fn neg(&self) -> Self {
        Fp12 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    /// `c0 + c1 w -> c0 − c1 w`. Over the cyclotomic subgroup this is the
    /// same as raising to the power `p^6`.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Fp12 { c0: self.c0, c1: self.c1.neg() }
    }

    // ----- multiplicative ops ----------------------------------------------

    /// Full Fp12 multiplication. Karatsuba in the quadratic extension plus
    /// `Fp6::mul_by_nonresidue` to fold `w^2 = v`.
    ///
    ///   (a0 + a1 w)(b0 + b1 w)
    ///     = a0 b0 + a1 b1 v        (from w^2 = v)
    ///     + ((a0 + a1)(b0 + b1) − a0 b0 − a1 b1) w
    pub fn mul(&self, other: &Self) -> Self {
        let v0 = self.c0.mul(&other.c0);
        let v1 = self.c1.mul(&other.c1);
        let sum_a = self.c0.add(&self.c1);
        let sum_b = other.c0.add(&other.c1);
        let t = sum_a.mul(&sum_b).sub(&v0).sub(&v1);
        let c0 = v0.add(&v1.mul_by_nonresidue());
        Fp12 { c0, c1: t }
    }

    /// Specialized squaring. `(a0 + a1 w)^2 = (a0+a1)(a0 + v a1) − a0 a1 − v a0 a1 + 2 a0 a1 w`.
    /// Equivalent to Karatsuba: 2 Fp6 mults instead of 3 for generic mul.
    pub fn square(&self) -> Self {
        // (a0 + a1 w)^2 = (a0 + a1)(a0 + v * a1) − v a0 a1 − a0 a1 + 2 a0 a1 w
        //
        // Standard formula ("complex squaring"):
        //   v0 = a0 * a1
        //   c0 = (a0 + a1)(a0 + v * a1) − v0 − v * v0
        //   c1 = 2 v0
        let v0 = self.c0.mul(&self.c1);
        let a0_plus_a1 = self.c0.add(&self.c1);
        let a0_plus_v_a1 = self.c0.add(&self.c1.mul_by_nonresidue());
        let c0_tmp = a0_plus_a1.mul(&a0_plus_v_a1);
        let v_v0 = v0.mul_by_nonresidue();
        let c0 = c0_tmp.sub(&v0).sub(&v_v0);
        let c1 = v0.add(&v0);
        Fp12 { c0, c1 }
    }

    /// Inverse via `a^{-1} = (c0 − c1 w) / (c0^2 − v c1^2)`.
    ///
    /// Returns `None` iff `self == 0`.
    pub fn invert(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let c0_sq = self.c0.square();
        let c1_sq = self.c1.square();
        let norm = c0_sq.sub(&c1_sq.mul_by_nonresidue());
        let norm_inv = norm.invert()?;
        Some(Fp12 {
            c0: self.c0.mul(&norm_inv),
            c1: self.c1.neg().mul(&norm_inv),
        })
    }

    /// Sparse multiplication used by the Miller loop to fold in a line
    /// evaluation. The line has only three non-zero Fp2 coefficients at
    /// positions (0,0), (0,1), and (1,4) of the Fp12 basis {1, v, v^2, w,
    /// v w, v^2 w}, so the "multiplier" is `c0 + c1 v + c4 v w`.
    ///
    ///   aa = self.c0 * (c0, c1)   via Fp6::mul_by_01
    ///   bb = self.c1 * c4         via Fp6::mul_by_1
    ///   out.c0 = bb.mul_by_nonresidue() + aa
    ///   out.c1 = (self.c0 + self.c1) * (c0, c1, c4).mul_by_01_plus_w_mul_by_1 − aa − bb
    ///          = (self.c0 + self.c1).mul_by_01(c0, c1 + c4) − aa − bb
    ///            -- but this last form only works if we refactor carefully.
    ///
    /// Standard published formula (pairing-friendly-curves RFC §4.3.4):
    ///   aa = self.c0.mul_by_01(c0, c1)
    ///   bb = self.c1.mul_by_1(c4)
    ///   o = c1 + c4                                    ∈ Fp2
    ///   new.c1 = (self.c0 + self.c1).mul_by_01(c0, &o) − aa − bb
    ///   new.c0 = bb.mul_by_nonresidue() + aa
    pub fn mul_by_014(&self, c0: &Fp2, c1: &Fp2, c4: &Fp2) -> Self {
        let aa = self.c0.mul_by_01(c0, c1);
        let bb = self.c1.mul_by_1(c4);
        let o = c1.add(c4);
        let sum = self.c0.add(&self.c1);
        let cross = sum.mul_by_01(c0, &o);
        let new_c1 = cross.sub(&aa).sub(&bb);
        let new_c0 = bb.mul_by_nonresidue().add(&aa);
        Fp12 { c0: new_c0, c1: new_c1 }
    }

    /// Frobenius map π^p applied `power` times: returns `self^(p^power)`.
    ///
    /// On the tower, Frobenius acts coordinate-wise using the precomputed
    /// γ constants (see `frobenius_constants`). For an element
    /// `x = c0 + c1·w ∈ Fp12` with `c0, c1 ∈ Fp6 = Fp2[v]/(v³-ξ)` and
    /// `w² = v`:
    ///
    /// ```text
    ///     π^k(x) = F6(c0) + F6(c1) · γ_{k,6} · w
    ///     F6(a0 + a1 v + a2 v²)
    ///         = π^k_Fp2(a0)
    ///         + π^k_Fp2(a1) · γ_{k,2} · v
    ///         + π^k_Fp2(a2) · γ_{k,4} · v²
    /// ```
    ///
    /// where π_Fp2 is complex conjugation (since `p ≡ 3 mod 4`), and the γ
    /// constants are powers of the nonresidue ξ derived from `(p^k - 1)/6`.
    /// `power` is reduced modulo 12 (π has order dividing 12 on Fp12).
    pub fn frobenius_map(&self, power: usize) -> Self {
        let k = power % 12;
        if k == 0 {
            return *self;
        }
        let consts = frobenius_constants();

        // F6 applied to self.c0 and self.c1.
        let f6 = |a: &Fp6| -> Fp6 {
            // Apply π^k to each Fp2 coefficient — conjugate iff k is odd.
            // (Fp2 has order 2, so π^k|Fp2 = π^{k mod 2}.)
            let (a0_f, a1_f, a2_f) = if k % 2 == 0 {
                (a.c0, a.c1, a.c2)
            } else {
                (a.c0.conjugate(), a.c1.conjugate(), a.c2.conjugate())
            };
            Fp6 {
                c0: a0_f,
                c1: a1_f.mul(&consts[k].gamma_k_2),
                c2: a2_f.mul(&consts[k].gamma_k_4),
            }
        };

        let new_c0 = f6(&self.c0);
        let mut new_c1 = f6(&self.c1);
        // Multiply each Fp2 coefficient of new_c1 by γ_{k,6}.
        let g = &consts[k].gamma_k_6;
        new_c1.c0 = new_c1.c0.mul(g);
        new_c1.c1 = new_c1.c1.mul(g);
        new_c1.c2 = new_c1.c2.mul(g);

        Fp12 { c0: new_c0, c1: new_c1 }
    }

    /// Cyclotomic-subgroup squaring.
    ///
    /// Valid only for elements in the cyclotomic subgroup `G_φ12(Fp) ⊂ Fp12`,
    /// reached after the "easy part" of final exponentiation (`x^{(p^6-1)(p^2+1)}`).
    /// Granger–Scott (2010) give a 3-Fp2-multiplication formula that's faster
    /// than the generic 2-Fp6-mul squaring, but requires the input to satisfy
    /// `x^(p^4 - p^2 + 1) = 1`.
    ///
    /// **This reference implementation delegates to `self.square()`**, which is
    /// always correct (and agrees with the specialized formula on cyclotomic
    /// inputs). The in-circuit AIR version, when built, should use the
    /// optimized 3-mul form from Granger–Scott Table 1 for gate count.
    pub fn cyclotomic_square(&self) -> Self {
        self.square()
    }
}

// ---------------------------------------------------------------------------
// Frobenius γ constants for Fp12
// ---------------------------------------------------------------------------
//
// For each power k ∈ 0..12 we cache three Fp2 elements:
//
//   γ_{k,2} = ξ^((p^k - 1)/3)     — used on the v coefficient of Fp6
//   γ_{k,4} = ξ^(2(p^k - 1)/3)    — used on the v² coefficient of Fp6
//                                    (equivalently γ_{k,2}²)
//   γ_{k,6} = ξ^((p^k - 1)/6)     — used on the w coefficient of Fp12
//
// where ξ = 1 + u is the Fp6 tower nonresidue.
//
// Derivation strategy: compute γ_{1,6} = ξ^((p-1)/6) once via Fermat pow,
// then derive γ_{1,2} = γ_{1,6}² and γ_{1,4} = γ_{1,6}⁴. Higher powers k > 1
// are obtained via
//
//   γ_{k+1,j} = γ_{k,j} · π(γ_{k,j}) · ... = γ_{1,j}^{(p^k + p^{k-1} + ... + 1)}
//             = γ_{1,j} · γ_{k,j}^p
//
// which in our tower means: γ_{k+1,j} = γ_{1,j} · γ_{k,j}.conjugate() when
// the exponent on γ_{1,j} is odd-total... actually, the cleanest formula
// uses the multiplicative accumulation
//
//   γ_{k,j} = ∏_{i=0}^{k-1} π^i(γ_{1,j})                           (1)
//
// where π on Fp2 is complex conjugation. Equivalently:
//
//   γ_{k,j} = γ_{k-1,j}.conjugate() · γ_{1,j}     for k ≥ 2.        (2)
//
// We use (2) below since it's clean.

#[derive(Debug, Clone, Copy)]
struct FrobeniusCoeffs {
    gamma_k_2: Fp2,
    gamma_k_4: Fp2,
    gamma_k_6: Fp2,
}

fn frobenius_constants() -> &'static [FrobeniusCoeffs; 12] {
    static TABLE: OnceLock<[FrobeniusCoeffs; 12]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let xi = Fp2 { c0: Fp::one(), c1: Fp::one() };

        // Exponent (p - 1) / 6 as a 6-limb big-endian Fp exponent.
        let exp_p_minus_1_over_6 = p_minus_1_over_6();

        // γ_{1,6} = ξ^((p-1)/6)
        let g16 = fp2_pow(&xi, &exp_p_minus_1_over_6);
        // γ_{1,2} = γ_{1,6}^2 = ξ^((p-1)/3)
        let g12 = g16.square();
        // γ_{1,4} = γ_{1,6}^4 = ξ^(2(p-1)/3)
        let g14 = g12.square();

        // Identity row (k = 0): γ_{0,*} = 1.
        let mut out: [FrobeniusCoeffs; 12] = [FrobeniusCoeffs {
            gamma_k_2: Fp2::one(),
            gamma_k_4: Fp2::one(),
            gamma_k_6: Fp2::one(),
        }; 12];

        // k = 1.
        out[1] = FrobeniusCoeffs {
            gamma_k_2: g12,
            gamma_k_4: g14,
            gamma_k_6: g16,
        };

        // k >= 2: γ_{k,j} = γ_{k-1,j}.conjugate() · γ_{1,j}.
        for k in 2..12 {
            let prev = out[k - 1];
            out[k] = FrobeniusCoeffs {
                gamma_k_2: prev.gamma_k_2.conjugate().mul(&g12),
                gamma_k_4: prev.gamma_k_4.conjugate().mul(&g14),
                gamma_k_6: prev.gamma_k_6.conjugate().mul(&g16),
            };
        }

        out
    })
}

/// Public accessor for the cached Frobenius γ constants for power `k`.
///
/// Returns `(γ_{k,2}, γ_{k,4}, γ_{k,6})` — the three Fp2 multipliers used by
/// `Fp12::frobenius_map(k)` on the `v`, `v²`, and `w` coefficients of the
/// tower. `power` is reduced modulo 12; `power = 0` returns three `Fp2::one`s
/// (Frobenius has order dividing 12 on Fp12).
///
/// Exposed so the AIR-side compiler in `nonnative_fp2_compile` can multiply
/// in-circuit values by these constants without re-deriving them.
pub fn frobenius_gamma_constants(power: usize) -> (Fp2, Fp2, Fp2) {
    let k = power % 12;
    let consts = frobenius_constants();
    (consts[k].gamma_k_2, consts[k].gamma_k_4, consts[k].gamma_k_6)
}

/// Compute `(p - 1) / 6` as a 6-limb big-endian exponent suitable for
/// `Fp::pow` / `fp2_pow`.
fn p_minus_1_over_6() -> [u64; 6] {
    // (p-1)/6: divide p-1 by 6 via schoolbook long division over base 2^64.
    // p - 1 = P_LIMBS with the last limb decremented by 1.
    let mut pm1 = P_LIMBS;
    pm1[5] -= 1; // p's low limb is 0xaaab ≠ 0, so no borrow.

    let mut quot = [0u64; 6];
    let mut rem: u128 = 0;
    for i in 0..6 {
        let cur = (rem << 64) | (pm1[i] as u128);
        quot[i] = (cur / 6) as u64;
        rem = cur % 6;
    }
    debug_assert_eq!(rem, 0, "(p-1) must be divisible by 6 for BLS12-381");
    quot
}

/// Fp2 square-and-multiply exponentiation with a 6-limb big-endian exponent.
fn fp2_pow(base: &Fp2, exp: &[u64; 6]) -> Fp2 {
    let mut result = Fp2::one();
    let mut started = false;
    for i in 0..6 {
        let limb = exp[i];
        for bit in (0..64).rev() {
            if started {
                result = result.square();
            }
            if ((limb >> bit) & 1) == 1 {
                if !started {
                    result = *base;
                    started = true;
                } else {
                    result = result.mul(base);
                }
            }
        }
    }
    if !started {
        return Fp2::one();
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonnative_fp::Fp;
    use blst::*;

    // ----- sample generators -----------------------------------------------

    /// A handful of non-trivial Fps, diverse in structure (zero, one, small,
    /// dense, near-p).
    fn sample_fps() -> Vec<Fp> {
        vec![
            Fp::zero(),
            Fp::one(),
            Fp::from_u64(2),
            Fp::from_u64(0xdeadbeef),
            Fp { limbs: [
                0x0123456789abcdef,
                0xfedcba9876543210,
                0xdeadbeefcafebabe,
                0x0f0e0d0c0b0a0908,
                0x0706050403020100,
                0x8080808080808080,
            ]},
            Fp { limbs: [
                0x00abcdef01234567,
                0x1111111111111111,
                0x2222222222222222,
                0x3333333333333333,
                0x4444444444444444,
                0x5555555555555555,
            ]},
        ]
    }

    fn sample_fp2s() -> Vec<Fp2> {
        let s = sample_fps();
        let mut out = Vec::new();
        for (i, a) in s.iter().enumerate() {
            let b = &s[(i + 2) % s.len()];
            out.push(Fp2 { c0: *a, c1: *b });
        }
        out
    }

    fn sample_fp6s() -> Vec<Fp6> {
        let s = sample_fp2s();
        let mut out = Vec::new();
        for i in 0..s.len() {
            out.push(Fp6 {
                c0: s[i],
                c1: s[(i + 1) % s.len()],
                c2: s[(i + 2) % s.len()],
            });
        }
        out
    }

    fn sample_fp12s() -> Vec<Fp12> {
        let s = sample_fp6s();
        let mut out = Vec::new();
        for i in 0..s.len() {
            out.push(Fp12 {
                c0: s[i],
                c1: s[(i + 1) % s.len()],
            });
        }
        out
    }

    // ----- Fp6 basic identity tests ----------------------------------------

    #[test]
    fn fp6_zero_is_additive_identity() {
        for a in sample_fp6s() {
            assert_eq!(a.add(&Fp6::zero()), a);
            assert_eq!(Fp6::zero().add(&a), a);
            assert_eq!(a.sub(&a), Fp6::zero());
        }
    }

    #[test]
    fn fp6_one_is_multiplicative_identity() {
        for a in sample_fp6s() {
            assert_eq!(a.mul(&Fp6::one()), a, "a * 1 != a");
            assert_eq!(Fp6::one().mul(&a), a, "1 * a != a");
        }
    }

    #[test]
    fn fp6_binomial_square() {
        // (a + b)^2 == a^2 + 2ab + b^2
        let samples = sample_fp6s();
        for a in &samples {
            for b in &samples {
                let lhs = a.add(b).square();
                let ab = a.mul(b);
                let two_ab = ab.add(&ab);
                let rhs = a.square().add(&two_ab).add(&b.square());
                assert_eq!(lhs, rhs, "(a+b)^2 != a^2 + 2ab + b^2 for Fp6");
            }
        }
    }

    #[test]
    fn fp6_square_matches_mul_self() {
        for a in sample_fp6s() {
            assert_eq!(a.square(), a.mul(&a), "Fp6::square != a*a");
        }
    }

    #[test]
    fn fp6_invert_is_inverse() {
        for a in sample_fp6s() {
            match a.invert() {
                Some(inv) => {
                    assert_eq!(a.mul(&inv), Fp6::one(), "a * a^{{-1}} != 1 for Fp6");
                    assert_eq!(inv.mul(&a), Fp6::one(), "a^{{-1}} * a != 1 for Fp6");
                }
                None => {
                    assert!(a.is_zero(), "Fp6::invert returned None for nonzero");
                }
            }
        }
    }

    #[test]
    fn fp6_mul_by_nonresidue_equals_times_v() {
        // v = Fp6 { c0: 0, c1: 1, c2: 0 }
        let v = Fp6 { c0: Fp2::zero(), c1: Fp2::one(), c2: Fp2::zero() };
        for a in sample_fp6s() {
            let via_method = a.mul_by_nonresidue();
            let via_mul = a.mul(&v);
            assert_eq!(via_method, via_mul, "Fp6::mul_by_nonresidue != a*v");
        }
    }

    #[test]
    fn fp6_mul_by_01_matches_full_mul() {
        let samples = sample_fp6s();
        for a in &samples {
            for b in &samples {
                // Build a candidate with c2 = 0.
                let b01 = Fp6 { c0: b.c0, c1: b.c1, c2: Fp2::zero() };
                let full = a.mul(&b01);
                let sparse = a.mul_by_01(&b.c0, &b.c1);
                assert_eq!(full, sparse, "mul_by_01 mismatch");
            }
        }
    }

    #[test]
    fn fp6_mul_by_1_matches_full_mul() {
        let samples = sample_fp6s();
        for a in &samples {
            for b in &samples {
                // Build a candidate with c0 = c2 = 0.
                let b1 = Fp6 { c0: Fp2::zero(), c1: b.c1, c2: Fp2::zero() };
                let full = a.mul(&b1);
                let sparse = a.mul_by_1(&b.c1);
                assert_eq!(full, sparse, "mul_by_1 mismatch");
            }
        }
    }

    // ----- Fp12 basic identity tests ---------------------------------------

    #[test]
    fn fp12_zero_is_additive_identity() {
        for a in sample_fp12s() {
            assert_eq!(a.add(&Fp12::zero()), a);
            assert_eq!(Fp12::zero().add(&a), a);
            assert_eq!(a.sub(&a), Fp12::zero());
        }
    }

    #[test]
    fn fp12_one_is_multiplicative_identity() {
        for a in sample_fp12s() {
            assert_eq!(a.mul(&Fp12::one()), a);
            assert_eq!(Fp12::one().mul(&a), a);
        }
    }

    #[test]
    fn fp12_binomial_square() {
        let samples = sample_fp12s();
        for a in &samples {
            for b in &samples {
                let lhs = a.add(b).square();
                let ab = a.mul(b);
                let two_ab = ab.add(&ab);
                let rhs = a.square().add(&two_ab).add(&b.square());
                assert_eq!(lhs, rhs, "(a+b)^2 != a^2 + 2ab + b^2 for Fp12");
            }
        }
    }

    #[test]
    fn fp12_square_matches_mul_self() {
        for a in sample_fp12s() {
            assert_eq!(a.square(), a.mul(&a), "Fp12::square != a*a");
        }
    }

    #[test]
    fn fp12_invert_is_inverse() {
        for a in sample_fp12s() {
            match a.invert() {
                Some(inv) => {
                    assert_eq!(a.mul(&inv), Fp12::one(), "a * a^-1 != 1 for Fp12");
                }
                None => {
                    assert!(a.is_zero());
                }
            }
        }
    }

    #[test]
    fn fp12_conjugate_involutive() {
        for a in sample_fp12s() {
            assert_eq!(a.conjugate().conjugate(), a, "conjugate not involutive");
        }
    }

    #[test]
    fn fp12_mul_by_014_matches_full_mul() {
        // Build `other` with only (c0.c0, c0.c1, c1.c1) populated, rest zero.
        // In terms of Fp12 = Fp6[w]: other = (c0 + c1*v + 0*v^2) + (0 + c4*v + 0*v^2)*w
        //                                  = c0 + c1 v + c4 v w
        let samples = sample_fp12s();
        for a in &samples {
            for b in &samples {
                let other = Fp12 {
                    c0: Fp6 { c0: b.c0.c0, c1: b.c0.c1, c2: Fp2::zero() },
                    c1: Fp6 { c0: Fp2::zero(), c1: b.c1.c1, c2: Fp2::zero() },
                };
                let full = a.mul(&other);
                let sparse = a.mul_by_014(&b.c0.c0, &b.c0.c1, &b.c1.c1);
                assert_eq!(full, sparse, "mul_by_014 mismatch");
            }
        }
    }

    // ----- blst cross-checks (Fp12) ---------------------------------------
    //
    // blst exposes Fp12 as `blst_fp12` with fields fp6[2], which are
    // `blst_fp6 { fp2: [blst_fp2; 3] }`, which are `blst_fp2 { fp: [blst_fp; 2] }`.
    // The Fp ordering within blst_fp2 is (fp[0], fp[1]) == (c0, c1) in our
    // notation.
    //
    // Conversion goes Fp <-> bytes <-> blst_fp via blst_fp_{from,to}_bendian.
    //
    // For Fp6 ops, blst does NOT expose an API — we exercise them by
    // embedding into Fp12 with c1 = 0 and comparing the Fp6 result to the
    // Fp12 result's c0 component.

    fn fp_to_blst_fp(fp: &Fp) -> blst_fp {
        let bytes = fp.to_bytes_be();
        let mut out = blst_fp::default();
        unsafe { blst_fp_from_bendian(&mut out, bytes.as_ptr()); }
        out
    }

    fn blst_fp_to_fp(bfp: &blst_fp) -> Fp {
        let mut bytes = [0u8; 48];
        unsafe { blst_bendian_from_fp(bytes.as_mut_ptr(), bfp); }
        Fp::from_bytes_be(&bytes).expect("blst output must be canonical")
    }

    fn fp2_to_blst(a: &Fp2) -> blst_fp2 {
        blst_fp2 { fp: [fp_to_blst_fp(&a.c0), fp_to_blst_fp(&a.c1)] }
    }

    fn blst_to_fp2(b: &blst_fp2) -> Fp2 {
        Fp2 { c0: blst_fp_to_fp(&b.fp[0]), c1: blst_fp_to_fp(&b.fp[1]) }
    }

    fn fp6_to_blst(a: &Fp6) -> blst_fp6 {
        blst_fp6 { fp2: [fp2_to_blst(&a.c0), fp2_to_blst(&a.c1), fp2_to_blst(&a.c2)] }
    }

    fn blst_to_fp6(b: &blst_fp6) -> Fp6 {
        Fp6 {
            c0: blst_to_fp2(&b.fp2[0]),
            c1: blst_to_fp2(&b.fp2[1]),
            c2: blst_to_fp2(&b.fp2[2]),
        }
    }

    fn fp12_to_blst(a: &Fp12) -> blst_fp12 {
        blst_fp12 { fp6: [fp6_to_blst(&a.c0), fp6_to_blst(&a.c1)] }
    }

    fn blst_to_fp12(b: &blst_fp12) -> Fp12 {
        Fp12 { c0: blst_to_fp6(&b.fp6[0]), c1: blst_to_fp6(&b.fp6[1]) }
    }

    #[test]
    fn fp12_cross_check_mul() {
        // Compare a few a*b against blst_fp12_mul.
        let samples = sample_fp12s();
        let mut count = 0;
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                let a = &samples[i];
                let b = &samples[j];
                let ours = a.mul(b);

                let ba = fp12_to_blst(a);
                let bb = fp12_to_blst(b);
                let mut bc = blst_fp12::default();
                unsafe { blst_fp12_mul(&mut bc, &ba, &bb); }
                let theirs = blst_to_fp12(&bc);

                assert_eq!(ours, theirs, "blst mul mismatch (i={},j={})", i, j);
                count += 1;
            }
        }
        assert!(count > 0);
    }

    #[test]
    fn fp12_cross_check_sqr() {
        for a in sample_fp12s() {
            let ours = a.square();

            let ba = fp12_to_blst(&a);
            let mut bc = blst_fp12::default();
            unsafe { blst_fp12_sqr(&mut bc, &ba); }
            let theirs = blst_to_fp12(&bc);

            assert_eq!(ours, theirs, "blst sqr mismatch");
        }
    }

    #[test]
    fn fp12_cross_check_inverse() {
        for a in sample_fp12s() {
            if a.is_zero() {
                assert!(a.invert().is_none());
                continue;
            }
            let ours = a.invert().expect("nonzero Fp12 has an inverse");

            let ba = fp12_to_blst(&a);
            let mut bc = blst_fp12::default();
            unsafe { blst_fp12_inverse(&mut bc, &ba); }
            let theirs = blst_to_fp12(&bc);

            assert_eq!(ours, theirs, "blst inverse mismatch");

            // And sanity on our side.
            assert_eq!(a.mul(&ours), Fp12::one());
        }
    }

    // ----- Fp6 via Fp12 round-trip cross-check -----------------------------
    //
    // For an Fp6 element x, embed as `X = x + 0 w ∈ Fp12`. Then blst's
    // `blst_fp12_mul(X, Y)` with Y similarly embedded produces `x*y` in the
    // c0 slot. Same for sqr, inverse (inverse of X = x + 0 w = x^{-1} + 0 w).

    #[test]
    fn fp6_mul_via_fp12_cross_check() {
        let samples = sample_fp6s();
        for a in &samples {
            for b in &samples {
                let ours = a.mul(b);

                let a12 = Fp12 { c0: *a, c1: Fp6::zero() };
                let b12 = Fp12 { c0: *b, c1: Fp6::zero() };
                let ba = fp12_to_blst(&a12);
                let bb = fp12_to_blst(&b12);
                let mut bc = blst_fp12::default();
                unsafe { blst_fp12_mul(&mut bc, &ba, &bb); }
                let theirs12 = blst_to_fp12(&bc);

                assert_eq!(theirs12.c1, Fp6::zero(), "embedded Fp6*Fp6 leaked into c1");
                assert_eq!(ours, theirs12.c0, "Fp6 mul disagreement via Fp12 embedding");
            }
        }
    }

    #[test]
    fn fp6_sqr_via_fp12_cross_check() {
        for a in sample_fp6s() {
            let ours = a.square();

            let a12 = Fp12 { c0: a, c1: Fp6::zero() };
            let ba = fp12_to_blst(&a12);
            let mut bc = blst_fp12::default();
            unsafe { blst_fp12_sqr(&mut bc, &ba); }
            let theirs12 = blst_to_fp12(&bc);

            assert_eq!(theirs12.c1, Fp6::zero());
            assert_eq!(ours, theirs12.c0, "Fp6 sqr disagreement via Fp12 embedding");
        }
    }

    #[test]
    fn fp6_inverse_via_fp12_cross_check() {
        for a in sample_fp6s() {
            if a.is_zero() {
                continue;
            }
            let ours = a.invert().expect("nonzero Fp6 inverts");

            let a12 = Fp12 { c0: a, c1: Fp6::zero() };
            let ba = fp12_to_blst(&a12);
            let mut bc = blst_fp12::default();
            unsafe { blst_fp12_inverse(&mut bc, &ba); }
            let theirs12 = blst_to_fp12(&bc);

            assert_eq!(theirs12.c1, Fp6::zero());
            assert_eq!(ours, theirs12.c0, "Fp6 inverse disagreement via Fp12 embedding");
        }
    }

    // ----- helpers / smoke -------------------------------------------------

    #[test]
    fn fp6_helper_mul_by_xi_matches_mul_by_fp2_one_plus_u() {
        // ξ = 1 + u
        let xi = Fp2 { c0: Fp::one(), c1: Fp::one() };
        for a in sample_fp2s() {
            let via_helper = super::fp2_mul_by_xi(&a);
            let via_full = a.mul(&xi);
            assert_eq!(via_helper, via_full, "fp2_mul_by_xi != a*(1+u)");
        }
    }

    // ----- Frobenius tests -------------------------------------------------

    #[test]
    fn frobenius_power_zero_is_identity() {
        for a in sample_fp12s() {
            assert_eq!(a.frobenius_map(0), a, "π^0 != id");
        }
    }

    #[test]
    fn frobenius_power_twelve_is_identity() {
        // π has order 12 on Fp12: x^(p^12) = x.
        for a in sample_fp12s() {
            assert_eq!(a.frobenius_map(12), a, "π^12 != id");
        }
    }

    #[test]
    fn frobenius_six_equals_conjugate() {
        // π^6 on Fp12 is complex conjugation (c0 + c1·w → c0 - c1·w) since
        // w^(p^6) = -w.
        for a in sample_fp12s() {
            assert_eq!(a.frobenius_map(6), a.conjugate(), "π^6 != conjugate");
        }
    }

    #[test]
    fn frobenius_is_homomorphism_over_product() {
        // π(a·b) = π(a)·π(b), for any power k.
        let samples = sample_fp12s();
        for k in [1usize, 2, 3, 4, 6] {
            for a in &samples {
                for b in &samples {
                    let lhs = a.mul(b).frobenius_map(k);
                    let rhs = a.frobenius_map(k).mul(&b.frobenius_map(k));
                    assert_eq!(lhs, rhs, "π^{} not a ring hom on a*b", k);
                }
            }
        }
    }

    #[test]
    fn frobenius_powers_compose() {
        // π^i ∘ π^j = π^{i+j}  (for i+j ≤ 12).
        for a in sample_fp12s() {
            for i in 0..=6 {
                for j in 0..=6 {
                    let lhs = a.frobenius_map(i).frobenius_map(j);
                    let rhs = a.frobenius_map(i + j);
                    assert_eq!(lhs, rhs, "π^{} ∘ π^{} != π^{}", i, j, i + j);
                }
            }
        }
    }

    #[test]
    fn frobenius_on_fp_is_identity() {
        // π(x) = x for x ∈ Fp. Embed sample Fps as Fp12 via c0 = x + 0v + 0v².
        for x_fp in sample_fps() {
            let x_fp2 = Fp2 { c0: x_fp, c1: Fp::zero() };
            let x_fp6 = Fp6 { c0: x_fp2, c1: Fp2::zero(), c2: Fp2::zero() };
            let x_fp12 = Fp12 { c0: x_fp6, c1: Fp6::zero() };
            for k in 1..=12 {
                assert_eq!(x_fp12.frobenius_map(k), x_fp12, "π^{}(Fp element) != self", k);
            }
        }
    }

    #[test]
    fn frobenius_cross_check_blst() {
        // Cross-check against blst_fp12_frobenius_map. Note blst's API
        // semantics: `blst_fp12_frobenius_map(&mut out, &in, n)` computes
        // out = in^(p^n) — same convention we use.
        //
        // Note: blst only supports n ∈ {1, 2, 3} (reduces `n mod 3` internally).
        // We verify higher powers algebraically in separate tests.
        let samples = sample_fp12s();
        for k in [1usize, 2, 3] {
            for a in &samples {
                let ours = a.frobenius_map(k);

                let ba = fp12_to_blst(a);
                let mut bc = blst_fp12::default();
                unsafe { blst_fp12_frobenius_map(&mut bc, &ba, k); }
                let theirs = blst_to_fp12(&bc);

                assert_eq!(ours, theirs, "π^{} disagrees with blst", k);
            }
        }
    }

    #[test]
    fn frobenius_constants_hardcoded_spot_check() {
        // The first-row Frobenius constants for BLS12-381 are well-known.
        // γ_{1,6} = ξ^((p-1)/6) is the primitive 6th root of unity shaping
        // the tower. One algebraic property: (γ_{1,6})^6 = ξ^(p-1) = 1 (by
        // Fermat on Fp2 = Fp[u]/(u²+1), since (p-1) annihilates every nonzero
        // Fp2 element's order... actually (γ_{1,6})^6 ∈ Fp and ξ^(p-1) in Fp2
        // may not be 1 in Fp2 — let's just check multiplicative consistency.)
        let consts = frobenius_constants();

        // γ_{k,2} = γ_{k,6}^2 for every k.
        for k in 0..12 {
            assert_eq!(
                consts[k].gamma_k_2,
                consts[k].gamma_k_6.square(),
                "γ_{{{},2}} != γ_{{{},6}}^2",
                k,
                k,
            );
            assert_eq!(
                consts[k].gamma_k_4,
                consts[k].gamma_k_6.square().square(),
                "γ_{{{},4}} != γ_{{{},6}}^4",
                k,
                k,
            );
        }

        // γ_{0,*} = 1.
        assert_eq!(consts[0].gamma_k_2, Fp2::one());
        assert_eq!(consts[0].gamma_k_4, Fp2::one());
        assert_eq!(consts[0].gamma_k_6, Fp2::one());

        // γ_{6,6}^2 = γ_{6,2} and γ_{6,6} is a 12th root of unity; in
        // particular γ_{6,6}^12 = 1.
        let g66_12 = {
            let s = consts[6].gamma_k_6.square();
            let q = s.square();         // γ^4
            let o = q.square();         // γ^8
            o.mul(&q)                   // γ^12
        };
        assert_eq!(g66_12, Fp2::one(), "γ_{{6,6}}^12 != 1");
    }

    // ----- Cyclotomic subgroup tests --------------------------------------

    #[test]
    fn cyclotomic_square_on_one_is_one() {
        assert_eq!(Fp12::one().cyclotomic_square(), Fp12::one());
    }

    #[test]
    fn cyclotomic_square_agrees_with_square_generally() {
        // Our reference cyclotomic_square delegates to square(), so this is
        // trivially true. But exercise a range of inputs to make sure nothing
        // subtle is off if we ever swap in the optimized formula.
        for a in sample_fp12s() {
            assert_eq!(a.cyclotomic_square(), a.square(),
                       "cyclotomic_square disagrees with square");
        }
    }

    #[test]
    fn cyclotomic_square_on_subgroup_elements() {
        // Construct elements in the cyclotomic subgroup:
        //   b = a^{p^6 - 1} = a^(p^6) · a^{-1} = conjugate(a) · a^{-1}
        // Then b^{p^6} = a^{(p^6 - 1) · p^6} = a^{p^{12} - p^6} = (a · a^{-p^6})
        // which simplifies — more usefully, any nonzero a^{p^6 - 1} satisfies
        // (a^{p^6-1})^{p^6+1} = a^{p^{12}-1} = 1, so it lands in the kernel of
        // x ↦ x^{p^6+1}, a superset of the cyclotomic subgroup. The cyclotomic
        // subgroup specifically is { x : x^{Φ_12(p)} = 1 } where
        // Φ_12(p) = p^4 - p^2 + 1, reachable via a^{(p^6-1)(p^2+1)}.
        //
        // For this test we just sanity-check that cyclotomic_square agrees
        // with square() on these (trivially true with fallback, but keeps the
        // test in place for the eventual optimized version).
        for a in sample_fp12s() {
            if a.is_zero() { continue; }
            let inv_a = match a.invert() {
                Some(i) => i,
                None => continue,
            };
            let b = a.conjugate().mul(&inv_a);         // ∈ ker(x ↦ x^(p^6+1))
            let b_p2_plus_1 = b.frobenius_map(2).mul(&b);
            // b_p2_plus_1 is in the cyclotomic subgroup.
            assert_eq!(
                b_p2_plus_1.cyclotomic_square(),
                b_p2_plus_1.square(),
                "cyclotomic_square disagrees with square on cyclotomic input",
            );
        }
    }
}
