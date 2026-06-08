//! Algebraic limb-level AIR for **BLS12-381** non-native `Fp` arithmetic.
//!
//! Mirrors [`crate::bn254_curve_ops_air::fp`] for ergonomic parity, but
//! specialised to the BLS12-381 base prime
//!
//! ```text
//! p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf
//!     6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
//! ```
//!
//! All field elements are 381-bit, held as **6 big-endian u64 limbs**.
//!
//! # Implementation note
//!
//! The BLS12-381 base-field Fp arithmetic AIR already exists in this
//! crate as [`crate::nonnative_fp_air`] (the 6-limb / 12-limb-product
//! analogue of [`crate::bn254_curve_ops_air::fp`]) and the host-side
//! Fp type in [`crate::nonnative_fp`]. This submodule simply
//! **re-exports** those pieces under names matching the BN254 layout,
//! so [`crate::bls12_381_curve_ops_air`] can refer to `fp::Fp`,
//! `fp::COL_A_OFFSET`, etc., exactly as the BN254 module does.
//!
//! # Cross-AIR LogUp tuple shape
//!
//! Each `(a, b, c)` Fp triple committed in the curve-ops AIR is bound
//! to the BLS12-381 Fp AIR's `(COL_A, COL_B, COL_R)` triple via a
//! 18-column LogUp tuple (3 × 6 BE u64 limbs). Selector columns are
//! `COL_SEL_ADD`, `COL_SEL_SUB`, `COL_SEL_MUL`, `COL_SEL_INV`
//! (re-exported below).

// ─── Re-exports from the existing BLS12-381 Fp AIR ─────────────────────

pub use crate::nonnative_fp::{Fp, P_LIMBS, P};
pub use crate::nonnative_fp_air::{
    LIMBS_PER_FP,
    LIMBS_PER_PROD,
    COL_A_OFFSET,
    COL_B_OFFSET,
    // The Fp AIR calls the output column `COL_R_OFFSET`; the BN254
    // mirror calls it `COL_C_OFFSET`. Provide both names.
    COL_R_OFFSET as COL_C_OFFSET,
    COL_SEL_ADD,
    COL_SEL_SUB,
    COL_SEL_MUL,
    COL_SEL_INV,
    NUM_NONNATIVE_FP_COLUMNS,
};

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bls12_381_fp_air_layout_consistency() {
        assert_eq!(LIMBS_PER_FP, 6);
        assert_eq!(LIMBS_PER_PROD, 12);
        assert_eq!(COL_A_OFFSET, 0);
        assert_eq!(COL_B_OFFSET, 6);
        assert_eq!(COL_C_OFFSET, 12);
        // Sel cols come right after the slack borrows section.
        assert!(COL_SEL_ADD < COL_SEL_SUB);
        assert!(COL_SEL_SUB < COL_SEL_MUL);
        assert!(COL_SEL_MUL < COL_SEL_INV);
        assert_eq!(COL_SEL_INV + 1, NUM_NONNATIVE_FP_COLUMNS);
    }

    #[test]
    fn bls12_381_fp_modulus_top_bit_is_zero() {
        // BLS12-381 p has 381 bits — top limb's high 3 bits must be zero.
        assert_eq!(P_LIMBS[0] >> 61, 0);
        assert_eq!(P_LIMBS[0], 0x1a0111ea397fe69a);
        assert_eq!(P_LIMBS[5], 0xb9feffffffffaaab);
    }

    #[test]
    fn bls12_381_fp_add_sub_mul_inv_smoke() {
        // 1 + 2 = 3 over BLS12-381 Fp.
        let a = Fp::from_u64(1);
        let b = Fp::from_u64(2);
        assert_eq!(a.add(&b), Fp::from_u64(3));
        // 7 - 4 = 3.
        assert_eq!(Fp::from_u64(7).sub(&Fp::from_u64(4)), Fp::from_u64(3));
        // 6 · 7 = 42.
        assert_eq!(Fp::from_u64(6).mul(&Fp::from_u64(7)), Fp::from_u64(42));
        // 2 · 2^{-1} = 1.
        let two = Fp::from_u64(2);
        let inv = two.invert().expect("2 is nonzero");
        assert_eq!(two.mul(&inv), Fp::one());
    }
}
