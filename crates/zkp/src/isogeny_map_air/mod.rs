//! BLS12-381 3-isogeny rational map AIR (#125 — the missing piece
//! between [`crate::hash_to_g2_air`] (SSWU on the isogeny curve `E'`)
//! and [`crate::g2_cofactor_clear_air`] (ψ + cofactor on BLS12-381
//! `G2`)).
//!
//! # Purpose
//!
//! RFC 9380 `hash_to_curve(BLS12381G2_XMD:SHA-256_SSWU_RO_)` proceeds in
//! four stages:
//!
//!   1. `expand_message_xmd → mod p` to obtain `u₀, u₁ ∈ Fp2`.
//!   2. **SSWU** on the 3-isogenous curve `E'(Fp2) : y² = x³ + 240·i·x
//!      + (1012 + 1012·u)` — produces `(x', y') ∈ E'`.
//!   3. **3-isogeny rational map** `iso_E'_to_E` taking
//!      `(x', y') ∈ E'` to `(x, y) ∈ BLS12-381 G2 : y² = x³ + 4(1+u)`.
//!   4. ψ-based cofactor clearing (Wahby–Boneh / Budroni–Pintore).
//!
//! [`crate::hash_to_g2_air`] covers stages 1–2, and
//! [`crate::g2_cofactor_clear_air`] covers stage 4. **This AIR fills
//! the gap at stage 3.**
//!
//! # Rational map shape
//!
//! From RFC 9380 §E.3 the 3-isogeny is
//!
//! ```text
//!   x = N_x(x') / D_x(x')   with deg N_x = 3, deg D_x = 2
//!   y = y' · N_y(x') / D_y(x')   with deg N_y = 3, deg D_y = 3
//! ```
//!
//! where the four polynomials `N_x, D_x, N_y, D_y` are tabulated `Fp2`
//! polynomials in `x'`. Horner evaluation of a degree-3 poly costs 3
//! Fp2 multiplications (= 9 Fp muls under Karatsuba); each Fp2
//! inversion is ~1 Fp inversion + a few Fp muls. Adding it all up the
//! isogeny is ~25 Fp2 multiplications + 2 Fp2 inversions per input
//! point.
//!
//! This module commits the **shape** of the witness (input on `E'`,
//! output on `G2`, plus one row of polynomial-evaluation intermediates)
//! plus the **cross-AIR LogUp descriptors** that bind the input from
//! [`crate::hash_to_g2_air`], the output into
//! [`crate::g2_cofactor_clear_air`] (replacing the cofactor AIR's
//! current direct-from-SSWU input descriptor), and at least 5
//! representative Fp-mul steps into [`crate::nonnative_fp_air`]. The
//! host-side witness builder uses blst's `blst_hash_to_g2` to obtain
//! the post-isogeny / pre-cofactor point — i.e. for the scaffold we
//! treat the isogeny as the **identity passthrough** for column shape
//! purposes (the same approach the cofactor AIR takes for its ψ stage),
//! and rely on the cross-AIR LogUp links into `nonnative_fp_air` to
//! carry the eventual algebraic binding.
//!
//! # Witness shape
//!
//! Per row this AIR commits one `E' → G2` rational-map evaluation:
//!
//!   * `in_point_e_prime` — the 4 Fp2 components `(x'.c0, x'.c1,
//!     y'.c0, y'.c1)` of the input on `E'`. 4 × 6 = 24 BE u64 limb
//!     columns. **Source**: the SSWU output of [`crate::hash_to_g2_air`]
//!     once that AIR is extended with explicit `pre_isogeny_*`
//!     columns; for now we point at the existing `(xd, n1)` slice as a
//!     target-shape proxy (same convention as `g2_cofactor_clear_air`).
//!   * `out_point_g2` — the 4 Fp2 components `(x.c0, x.c1, y.c0,
//!     y.c1)` of the output on G2. 4 × 6 = 24 BE u64 limb columns.
//!   * `nx_eval, dx_eval, ny_eval, dy_eval` — the 4 Fp2 polynomial
//!     evaluations at `x'`. 4 × 2 × 6 = 48 BE u64 limb columns.
//!   * `dx_inv, dy_inv` — Fp2 inverses of `D_x(x')` and `D_y(x')`. 2 ×
//!     2 × 6 = 24 BE u64 limb columns.
//!   * `is_real` — selector (1 on real rows, 0 on padding).
//!
//! # What is algebraically enforced
//!
//!   1. `is_real ∈ {0, 1}` — selector binarity.
//!   2. **One representative Fp limb-level polynomial-term equality**:
//!      the limb-by-limb identity `out_x.c0.limbs[j] = nx_eval.c0.limbs[j]`
//!      (i.e. for the scaffolding passthrough where `D_x(x') = 1` and
//!      the input/output points coincide). The 6 constraints have the
//!      right shape for the eventual `out_x = N_x(x') / D_x(x')` Fp2
//!      multiplication binding once `nonnative_fp_air` rows are
//!      committed via the cross-AIR LogUp descriptors below.
//!
//! Total row-local algebraic constraints: `1 + 6 = 7`.
//!
//! # Cross-AIR linkages
//!
//! - [`make_isogeny_input_from_hash_to_g2_descriptor`] — binds the
//!   24-limb `in_point_e_prime` columns of this AIR to the (eventual)
//!   `pre_isogeny_*` columns of [`crate::hash_to_g2_air`]. As with
//!   `g2_cofactor_clear_air`, the descriptor currently points at the
//!   `(xd, n1)` Fp2 slice as the *shape proxy*.
//! - [`make_isogeny_output_to_cofactor_descriptor`] — binds the
//!   24-limb `out_point_g2` columns of this AIR to the
//!   `in_point_e_prime` columns of [`crate::g2_cofactor_clear_air`].
//!   This is the critical wiring that **renames** the cofactor AIR's
//!   "input on E'" to "input on G2" semantically; algebraically the
//!   column shape is unchanged (both are 24 limbs of an Fp2 point),
//!   only the meaning of those limbs shifts to "post-isogeny G2 point"
//!   once this AIR is in the chain.
//! - [`make_isogeny_step_descriptors`] — returns 5 representative
//!   Fp-mul descriptors covering the polynomial-evaluation Horner
//!   steps for `N_x`, `D_x`, `N_y`, `D_y`, plus the final `y' · N_y`
//!   multiplication. Each points at a single [`crate::nonnative_fp_air`]
//!   row.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupRequirements, LookupTable};
use crate::nonnative_fp::{Fp, Fp2};
use crate::pairing::G2Affine;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of 64-bit limbs in one Fp element.
pub const LIMBS_PER_FP: usize = 6;

// ─── Column layout ────────────────────────────────────────────────────
//
// Row layout (per E'→G2 rational-map row):
//
//   in_x_c0_limbs   : 6 BE u64       (x' on E')
//   in_x_c1_limbs   : 6
//   in_y_c0_limbs   : 6
//   in_y_c1_limbs   : 6
//   out_x_c0_limbs  : 6              (x on G2)
//   out_x_c1_limbs  : 6
//   out_y_c0_limbs  : 6
//   out_y_c1_limbs  : 6
//   nx_c0_limbs     : 6              (N_x(x'))
//   nx_c1_limbs     : 6
//   dx_c0_limbs     : 6              (D_x(x'))
//   dx_c1_limbs     : 6
//   ny_c0_limbs     : 6              (N_y(x'))
//   ny_c1_limbs     : 6
//   dy_c0_limbs     : 6              (D_y(x'))
//   dy_c1_limbs     : 6
//   dx_inv_c0_limbs : 6              (D_x(x')^{-1})
//   dx_inv_c1_limbs : 6
//   dy_inv_c0_limbs : 6              (D_y(x')^{-1})
//   dy_inv_c1_limbs : 6
//   is_real         : 1
//
// total = 20 * 6 + 1 = 121

pub const COL_IN_X_C0_LIMB_OFFSET: usize = 0;
pub const COL_IN_X_C1_LIMB_OFFSET: usize = COL_IN_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_IN_Y_C0_LIMB_OFFSET: usize = COL_IN_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_IN_Y_C1_LIMB_OFFSET: usize = COL_IN_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_OUT_X_C0_LIMB_OFFSET: usize = COL_IN_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_X_C1_LIMB_OFFSET: usize = COL_OUT_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_Y_C0_LIMB_OFFSET: usize = COL_OUT_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_Y_C1_LIMB_OFFSET: usize = COL_OUT_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_NX_C0_LIMB_OFFSET: usize = COL_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_NX_C1_LIMB_OFFSET: usize = COL_NX_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_DX_C0_LIMB_OFFSET: usize = COL_NX_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_DX_C1_LIMB_OFFSET: usize = COL_DX_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_NY_C0_LIMB_OFFSET: usize = COL_DX_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_NY_C1_LIMB_OFFSET: usize = COL_NY_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_DY_C0_LIMB_OFFSET: usize = COL_NY_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_DY_C1_LIMB_OFFSET: usize = COL_DY_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_DX_INV_C0_LIMB_OFFSET: usize = COL_DY_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_DX_INV_C1_LIMB_OFFSET: usize = COL_DX_INV_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_DY_INV_C0_LIMB_OFFSET: usize = COL_DX_INV_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_DY_INV_C1_LIMB_OFFSET: usize = COL_DY_INV_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_IS_REAL: usize = COL_DY_INV_C1_LIMB_OFFSET + LIMBS_PER_FP;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// Row-local constraints:
///   0:     is_real ∈ {0, 1}
///   1..7:  6 Fp limb-equality constraints
///          `out_x.c0.limbs[j] = nx_eval.c0.limbs[j]`
///          (representative polynomial-term binding — for the
///          scaffolding passthrough where `D_x(x') = 1` this holds
///          honestly; the general case becomes the LogUp Fp-mul row
///          `out_x.c0 = nx.c0 * dx_inv.c0 - nx.c1 * dx_inv.c1`).
pub const NUM_POLY_TERM_CONSTRAINTS: usize = LIMBS_PER_FP;
pub const NUM_ROW_CONSTRAINTS: usize = 1 + NUM_POLY_TERM_CONSTRAINTS;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct IsogenyMapRow {
    /// Input point on `E'` (the SSWU isogeny curve), as 4 Fp2 components.
    pub in_x_c0: Fp,
    pub in_x_c1: Fp,
    pub in_y_c0: Fp,
    pub in_y_c1: Fp,
    /// Output point on BLS12-381 `G2`, as 4 Fp2 components.
    pub out_x_c0: Fp,
    pub out_x_c1: Fp,
    pub out_y_c0: Fp,
    pub out_y_c1: Fp,
    /// Polynomial evaluations: `N_x(x'), D_x(x'), N_y(x'), D_y(x')`.
    pub nx_c0: Fp,
    pub nx_c1: Fp,
    pub dx_c0: Fp,
    pub dx_c1: Fp,
    pub ny_c0: Fp,
    pub ny_c1: Fp,
    pub dy_c0: Fp,
    pub dy_c1: Fp,
    /// Inverses of `D_x(x')` and `D_y(x')`.
    pub dx_inv_c0: Fp,
    pub dx_inv_c1: Fp,
    pub dy_inv_c0: Fp,
    pub dy_inv_c1: Fp,
}

#[derive(Clone, Debug, Default)]
pub struct IsogenyMapWitness {
    pub rows: Vec<IsogenyMapRow>,
}

impl IsogenyMapWitness {
    /// Build a single-row witness from a host-side `E'` point.
    ///
    /// Because the algebraic 3-isogeny rational-map evaluation is
    /// deferred to the cross-AIR LogUp Fp-mul links into
    /// [`crate::nonnative_fp_air`], this constructor wires the input
    /// point straight through into the output columns (treating the
    /// isogeny as the identity for scaffolding purposes). Concretely:
    ///
    ///   * `out_point_g2 := in_point_e_prime`
    ///   * `nx_eval     := in_x` (so `out_x.c0 = nx.c0` honestly holds
    ///     limb-for-limb under the representative algebraic constraint)
    ///   * `dx_eval     := Fp2::one()` (so the `out = N_x / D_x`
    ///     identity collapses to `out = N_x`)
    ///   * `ny_eval     := Fp2::one()`
    ///   * `dy_eval     := Fp2::one()`
    ///   * `dx_inv      := Fp2::one()`
    ///   * `dy_inv      := Fp2::one()`
    ///
    /// Returns `None` if the input point is the identity (real
    /// hash-to-curve never produces the identity; rejecting it
    /// matches the convention of [`crate::g2_cofactor_clear_air`]).
    pub fn from_e_prime_point(e_prime: G2Affine) -> Option<Self> {
        if e_prime.infinity {
            return None;
        }
        let one_fp = Fp::one();
        let zero_fp = Fp::zero();
        Some(Self {
            rows: vec![IsogenyMapRow {
                in_x_c0: e_prime.x.c0,
                in_x_c1: e_prime.x.c1,
                in_y_c0: e_prime.y.c0,
                in_y_c1: e_prime.y.c1,
                out_x_c0: e_prime.x.c0,
                out_x_c1: e_prime.x.c1,
                out_y_c0: e_prime.y.c0,
                out_y_c1: e_prime.y.c1,
                // N_x(x') passthrough: same as in_x so the
                // representative constraint `out_x.c0 = nx.c0` holds.
                nx_c0: e_prime.x.c0,
                nx_c1: e_prime.x.c1,
                // D_x(x') = 1 + 0u.
                dx_c0: one_fp,
                dx_c1: zero_fp,
                // N_y(x') = 1 + 0u (so y · N_y / D_y = y' under D_y=1).
                ny_c0: one_fp,
                ny_c1: zero_fp,
                // D_y(x') = 1 + 0u.
                dy_c0: one_fp,
                dy_c1: zero_fp,
                // Inverses of 1 + 0u are 1 + 0u.
                dx_inv_c0: one_fp,
                dx_inv_c1: zero_fp,
                dy_inv_c0: one_fp,
                dy_inv_c1: zero_fp,
            }],
        })
    }

    /// Build a single-row witness from a real signature point obtained
    /// from `hash_to_curve(msg, dst)` via blst. blst performs the full
    /// pipeline internally; only the final G2 point is returned. The
    /// scaffold commits that point into the `in` and `out` column
    /// groups (treating the isogeny as the identity for scaffolding,
    /// same convention as [`crate::g2_cofactor_clear_air`]).
    pub fn from_message(msg: &[u8], dst: &[u8]) -> Option<Self> {
        let aff = crate::bls_sig::hash_to_g2_affine(msg, dst);
        let mut compressed = [0u8; 96];
        unsafe {
            blst::blst_p2_affine_compress(compressed.as_mut_ptr(), &aff);
        }
        let g2 = G2Affine::from_bytes(&compressed).ok()?;
        Self::from_e_prime_point(g2)
    }

    /// Append a raw row (used by tampering tests / fixtures).
    pub fn push_raw(&mut self, row: IsogenyMapRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &IsogenyMapWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for (off, fp) in [
            (COL_IN_X_C0_LIMB_OFFSET, &row.in_x_c0),
            (COL_IN_X_C1_LIMB_OFFSET, &row.in_x_c1),
            (COL_IN_Y_C0_LIMB_OFFSET, &row.in_y_c0),
            (COL_IN_Y_C1_LIMB_OFFSET, &row.in_y_c1),
            (COL_OUT_X_C0_LIMB_OFFSET, &row.out_x_c0),
            (COL_OUT_X_C1_LIMB_OFFSET, &row.out_x_c1),
            (COL_OUT_Y_C0_LIMB_OFFSET, &row.out_y_c0),
            (COL_OUT_Y_C1_LIMB_OFFSET, &row.out_y_c1),
            (COL_NX_C0_LIMB_OFFSET, &row.nx_c0),
            (COL_NX_C1_LIMB_OFFSET, &row.nx_c1),
            (COL_DX_C0_LIMB_OFFSET, &row.dx_c0),
            (COL_DX_C1_LIMB_OFFSET, &row.dx_c1),
            (COL_NY_C0_LIMB_OFFSET, &row.ny_c0),
            (COL_NY_C1_LIMB_OFFSET, &row.ny_c1),
            (COL_DY_C0_LIMB_OFFSET, &row.dy_c0),
            (COL_DY_C1_LIMB_OFFSET, &row.dy_c1),
            (COL_DX_INV_C0_LIMB_OFFSET, &row.dx_inv_c0),
            (COL_DX_INV_C1_LIMB_OFFSET, &row.dx_inv_c1),
            (COL_DY_INV_C0_LIMB_OFFSET, &row.dy_inv_c0),
            (COL_DY_INV_C1_LIMB_OFFSET, &row.dy_inv_c1),
        ] {
            for j in 0..LIMBS_PER_FP {
                columns[off + j][r] = Scalar::from_u64(fp.limbs[j], curve);
            }
        }
        columns[COL_IS_REAL][r] = one.clone();
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct IsogenyMapConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl IsogenyMapConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for IsogenyMapConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("poly_term_out_x_c0_limb_{}_eq_nx", j));
        }
        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1..7: representative polynomial-term equality
        //       out_x.c0.limbs[j] = nx_eval.c0.limbs[j].
        for j in 0..LIMBS_PER_FP {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = &columns[COL_OUT_X_C0_LIMB_OFFSET + j][r];
                let rhs = &columns[COL_NX_C0_LIMB_OFFSET + j][r];
                c[r] = lhs.sub(rhs);
            }
            out.push(c);
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 1..7: poly-term equality.
        for j in 0..LIMBS_PER_FP {
            let body = col_evals[COL_OUT_X_C0_LIMB_OFFSET + j]
                .sub(&col_evals[COL_NX_C0_LIMB_OFFSET + j]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 1..7: poly-term equality.
        for j in 0..LIMBS_PER_FP {
            let body = poly_sub(
                &col_coeffs[COL_OUT_X_C0_LIMB_OFFSET + j],
                &col_coeffs[COL_NX_C0_LIMB_OFFSET + j],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls12381);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // Limb columns are 64-bit-bounded by construction (u64 from the
        // witness); no byte columns to range-check here (those live in
        // `hash_to_g2_air` on the input side).
        LookupRequirements {
            tables: vec![LookupTable::range(256)],
            declarations: Vec::new(),
        }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Cross-AIR LogUp: bind the 24-limb `in_point_e_prime` columns of this
/// AIR to the (eventual) `pre_isogeny_*` columns of
/// [`crate::hash_to_g2_air`].
///
/// A side = this AIR (selected by [`COL_IS_REAL`]).
/// B side = [`crate::hash_to_g2_air`] (selected by its `IS_REAL`).
///
/// # Caveat (target shape only)
///
/// `hash_to_g2_air` does **not** currently expose a 24-limb
/// `pre_isogeny_*` group of columns. The closest existing alignment is
/// the SSWU intermediate `(xd, n1)` Fp2 pairs (2 Fp2 = 4 Fp = 24
/// limbs). The descriptor below targets that 24-limb slice as the
/// *shape proxy* — a follow-up round on `hash_to_g2_air` (#119 step
/// N+1) will add explicit `pre_isogeny_x_c0_limbs`,
/// `pre_isogeny_x_c1_limbs`, `pre_isogeny_y_c0_limbs`,
/// `pre_isogeny_y_c1_limbs` columns aligned 1-for-1 with this AIR's
/// `in_point_e_prime` columns.
pub fn make_isogeny_input_from_hash_to_g2_descriptor(
    isogeny_layer_index: usize,
    hash_to_g2_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::hash_to_g2_air as h2g2;
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| COL_IN_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| COL_IN_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_IN_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_IN_Y_C1_LIMB_OFFSET + j))
        .collect();
    // Target-shape proxy: bind to the SSWU intermediate (xd, n1) Fp2
    // limb columns of hash_to_g2_air until that AIR is extended with
    // explicit pre_isogeny_* columns.
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| h2g2::COL_XD_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| h2g2::COL_XD_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| h2g2::COL_N1_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| h2g2::COL_N1_C1_LIMB_OFFSET + j))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "isogeny_input_from_hash_to_g2_v1".into(),
        a_layer_index: isogeny_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: hash_to_g2_layer_index,
        b_columns,
        b_selector_column: Some(h2g2::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp: bind the 24-limb `out_point_g2` columns of this AIR
/// to the 24-limb input columns of [`crate::g2_cofactor_clear_air`].
///
/// A side = [`crate::g2_cofactor_clear_air`] (selected by its `IS_REAL`).
/// B side = this AIR (selected by [`COL_IS_REAL`]).
///
/// # Semantic rename note
///
/// The cofactor AIR's input columns are currently named
/// `in_point_e_prime` (i.e. "the SSWU output on the isogeny curve"),
/// reflecting the historical assumption that the cofactor AIR
/// consumed an E' point directly and silently absorbed the isogeny
/// stage. With this AIR in the chain those columns are correctly
/// reinterpreted as "the post-isogeny BLS12-381 G2 point" — the
/// limb shape is unchanged (24 Fp limbs of an Fp2 point) but the
/// *meaning* shifts. A follow-up rename pass on `g2_cofactor_clear_air`
/// would clarify the column names; for the cross-AIR LogUp descriptor
/// only the limb layout matters.
pub fn make_isogeny_output_to_cofactor_descriptor(
    isogeny_layer_index: usize,
    cofactor_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::g2_cofactor_clear_air as gcc;
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| gcc::COL_IN_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| gcc::COL_IN_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| gcc::COL_IN_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| gcc::COL_IN_Y_C1_LIMB_OFFSET + j))
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| COL_OUT_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_Y_C1_LIMB_OFFSET + j))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "isogeny_output_to_g2_cofactor_v1".into(),
        a_layer_index: cofactor_layer_index,
        a_columns,
        a_selector_column: Some(gcc::COL_IS_REAL),
        b_layer_index: isogeny_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Build a vector of cross-AIR LogUp descriptors that wire
/// representative Fp multiplication steps of the 3-isogeny
/// polynomial-evaluation pipeline to rows of
/// [`crate::nonnative_fp_air`]. Each descriptor uses the
/// `nonnative_fp_air`'s `(a, b, r)` 18-limb layout as the B side and
/// an `(a_limbs, b_limbs, r_limbs)` triple drawn from this AIR's
/// committed columns as the A side.
///
/// # Scaffolding shape
///
/// The current 5 descriptors point each Horner step to a *single
/// representative* Fp-mul row, treating each multi-Fp-mul Fp2
/// polynomial-evaluation step as if it were one Fp multiplication.
/// The actual decomposition (degree-3 Horner over Fp2 = 3 Fp2 muls = 9
/// Fp muls per polynomial; 4 polynomials = 36 Fp muls + 2 Fp2
/// inversions + 2 Fp2 muls = ~48 Fp muls total per isogeny row)
/// requires committing intermediate scratch columns and adding ~48
/// descriptors per row. Both extensions preserve the descriptor-set
/// shape this function returns.
///
/// Returns 5 descriptors:
///   - `isogeny_nx_eval_mul_v1` — (in_x.c0, in_x.c0, nx.c0)
///     representative `N_x(x')` Horner Fp mul.
///   - `isogeny_dx_eval_mul_v1` — (in_x.c0, in_x.c1, dx.c0)
///     representative `D_x(x')` Horner Fp mul.
///   - `isogeny_ny_eval_mul_v1` — (in_x.c1, in_x.c1, ny.c0)
///     representative `N_y(x')` Horner Fp mul.
///   - `isogeny_dy_eval_mul_v1` — (in_x.c0, dy.c0, dy.c1)
///     representative `D_y(x')` Horner Fp mul.
///   - `isogeny_y_times_ny_mul_v1` — (in_y.c0, ny.c0, out_y.c0)
///     representative `y' · N_y(x')` final Fp mul.
pub fn make_isogeny_step_descriptors(
    isogeny_layer_index: usize,
    nonnative_fp_layer_index: usize,
) -> Vec<crate::cross_air_logup::CrossAirLogUpDescriptor> {
    use crate::nonnative_fp_air as nfp;

    // B-side template: the (a, b, r) limb columns of a single nfp row.
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| nfp::COL_A_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_B_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_R_OFFSET + j))
        .collect();

    // Helper to assemble an A-side (a_off, b_off, r_off) tuple of
    // committed limb columns.
    let triple = |a_off: usize, b_off: usize, r_off: usize| -> Vec<usize> {
        (0..LIMBS_PER_FP)
            .map(move |j| a_off + j)
            .chain((0..LIMBS_PER_FP).map(move |j| b_off + j))
            .chain((0..LIMBS_PER_FP).map(move |j| r_off + j))
            .collect()
    };

    let make_desc =
        |label: &str, a_columns: Vec<usize>| crate::cross_air_logup::CrossAirLogUpDescriptor {
            label: label.into(),
            a_layer_index: isogeny_layer_index,
            a_columns,
            a_selector_column: Some(COL_IS_REAL),
            b_layer_index: nonnative_fp_layer_index,
            b_columns: b_columns.clone(),
            b_selector_column: Some(nfp::COL_SEL_MUL),
        };

    vec![
        make_desc(
            "isogeny_nx_eval_mul_v1",
            triple(
                COL_IN_X_C0_LIMB_OFFSET,
                COL_IN_X_C0_LIMB_OFFSET,
                COL_NX_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "isogeny_dx_eval_mul_v1",
            triple(
                COL_IN_X_C0_LIMB_OFFSET,
                COL_IN_X_C1_LIMB_OFFSET,
                COL_DX_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "isogeny_ny_eval_mul_v1",
            triple(
                COL_IN_X_C1_LIMB_OFFSET,
                COL_IN_X_C1_LIMB_OFFSET,
                COL_NY_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "isogeny_dy_eval_mul_v1",
            triple(
                COL_IN_X_C0_LIMB_OFFSET,
                COL_DY_C0_LIMB_OFFSET,
                COL_DY_C1_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "isogeny_y_times_ny_mul_v1",
            triple(
                COL_IN_Y_C0_LIMB_OFFSET,
                COL_NY_C0_LIMB_OFFSET,
                COL_OUT_Y_C0_LIMB_OFFSET,
            ),
        ),
    ]
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Beacon-chain ciphersuite DST (POP variant) for the real-signature
    /// path test.
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    /// Build a synthetic non-identity `E'` "point" with deterministic
    /// limb values. This is **not** on the isogeny curve; the scaffold
    /// constraints do not verify curve membership, so any non-zero
    /// Fp2 quadruple suffices for column-layout / passthrough tests.
    fn synthetic_e_prime_point() -> G2Affine {
        G2Affine {
            x: Fp2 {
                c0: Fp { limbs: [1, 2, 3, 4, 5, 6] },
                c1: Fp { limbs: [7, 8, 9, 10, 11, 12] },
            },
            y: Fp2 {
                c0: Fp { limbs: [13, 14, 15, 16, 17, 18] },
                c1: Fp { limbs: [19, 20, 21, 22, 23, 24] },
            },
            infinity: false,
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: identity-point input is rejected
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn identity_input_is_rejected() {
        assert!(IsogenyMapWitness::from_e_prime_point(G2Affine::identity()).is_none());
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: synthetic point builds & passthrough holds
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn synthetic_point_passes_through() {
        let p = synthetic_e_prime_point();
        let w = IsogenyMapWitness::from_e_prime_point(p).expect("non-identity accepted");
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        // Passthrough: in == out across all four Fp2 components.
        assert_eq!(row.in_x_c0, row.out_x_c0);
        assert_eq!(row.in_x_c1, row.out_x_c1);
        assert_eq!(row.in_y_c0, row.out_y_c0);
        assert_eq!(row.in_y_c1, row.out_y_c1);
        // N_x = in_x (so the representative algebraic constraint
        // `out_x.c0 = nx.c0` holds limb-for-limb).
        assert_eq!(row.nx_c0, row.in_x_c0);
        assert_eq!(row.nx_c1, row.in_x_c1);
        // D_x = D_y = N_y = 1 + 0u.
        assert_eq!(row.dx_c0, Fp::one());
        assert_eq!(row.dx_c1, Fp::zero());
        assert_eq!(row.dy_c0, Fp::one());
        assert_eq!(row.dy_c1, Fp::zero());
        assert_eq!(row.ny_c0, Fp::one());
        assert_eq!(row.ny_c1, Fp::zero());
        // Inverses also 1 + 0u.
        assert_eq!(row.dx_inv_c0, Fp::one());
        assert_eq!(row.dy_inv_c0, Fp::one());
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: witness builds for a real signature point via blst
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_from_real_signature_point() {
        let w = IsogenyMapWitness::from_message(b"isogeny-map-test", POP_DST)
            .expect("hash_to_curve must produce a non-identity point");
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        // Output coordinates are non-zero (hash-to-curve never lands
        // on the identity).
        assert!(row.out_x_c0.limbs.iter().any(|&l| l != 0)
            || row.out_x_c1.limbs.iter().any(|&l| l != 0));
        assert!(row.out_y_c0.limbs.iter().any(|&l| l != 0)
            || row.out_y_c1.limbs.iter().any(|&l| l != 0));
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: constraints zero on honest witness; tampering fires
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let p = synthetic_e_prime_point();
        let w = IsogenyMapWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = IsogenyMapConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn poly_term_fires_on_tampered_out_x_c0() {
        let p = synthetic_e_prime_point();
        let w = IsogenyMapWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Tamper out_x.c0 limb 0.
        let original = cols[COL_OUT_X_C0_LIMB_OFFSET][0].clone();
        cols[COL_OUT_X_C0_LIMB_OFFSET][0] = original.add(&Scalar::one(curve));
        let cs = IsogenyMapConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Poly-term constraint indices start at 1; limb 0 = index 1.
        let poly_term_limb0 = 1;
        assert!(
            !results[poly_term_limb0][0].is_zero(),
            "tampering out_x.c0 limb 0 must fire the polynomial-term constraint",
        );
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_selector() {
        let p = synthetic_e_prime_point();
        let w = IsogenyMapWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(5, CurveType::Bls12381);
        let cs = IsogenyMapConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "non-binary is_real must fire");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: cross-AIR descriptors are well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn input_from_hash_to_g2_descriptor_well_formed() {
        let d = make_isogeny_input_from_hash_to_g2_descriptor(0, 1);
        assert_eq!(d.label, "isogeny_input_from_hash_to_g2_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), 4 * LIMBS_PER_FP, "4 Fp × 6 limbs = 24");
        assert_eq!(d.b_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::hash_to_g2_air::COL_IS_REAL),
        );
        // First A column = this AIR's in_x.c0 limb 0.
        assert_eq!(d.a_columns[0], COL_IN_X_C0_LIMB_OFFSET);
        // Last A column = in_y.c1 limb 5.
        assert_eq!(
            d.a_columns[4 * LIMBS_PER_FP - 1],
            COL_IN_Y_C1_LIMB_OFFSET + LIMBS_PER_FP - 1,
        );
        // First B column = hash_to_g2_air's xd.c0 limb 0 (target-shape proxy).
        assert_eq!(
            d.b_columns[0],
            crate::hash_to_g2_air::COL_XD_C0_LIMB_OFFSET,
        );
    }

    #[test]
    fn output_to_cofactor_descriptor_well_formed() {
        let d = make_isogeny_output_to_cofactor_descriptor(0, 1);
        assert_eq!(d.label, "isogeny_output_to_g2_cofactor_v1");
        // A side = cofactor (layer 1), B side = this AIR (layer 0).
        assert_eq!(d.a_layer_index, 1);
        assert_eq!(d.b_layer_index, 0);
        assert_eq!(d.a_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(d.b_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(
            d.a_selector_column,
            Some(crate::g2_cofactor_clear_air::COL_IS_REAL),
        );
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));
        // First B column = this AIR's out_x.c0 limb 0.
        assert_eq!(d.b_columns[0], COL_OUT_X_C0_LIMB_OFFSET);
        // First A column = cofactor AIR's in_x.c0 limb 0.
        assert_eq!(
            d.a_columns[0],
            crate::g2_cofactor_clear_air::COL_IN_X_C0_LIMB_OFFSET,
        );
    }

    #[test]
    fn isogeny_step_descriptor_set_has_five_entries() {
        let descriptors = make_isogeny_step_descriptors(3, 7);
        assert!(
            descriptors.len() >= 5,
            "isogeny step descriptor set must include >= 5 Fp-mul links \
             (got {}); each polynomial-evaluation step contributes at \
             least one nfp link",
            descriptors.len(),
        );
        for d in &descriptors {
            assert_eq!(d.a_layer_index, 3);
            assert_eq!(d.b_layer_index, 7);
            assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
            assert_eq!(
                d.b_selector_column,
                Some(crate::nonnative_fp_air::COL_SEL_MUL),
            );
            // Each descriptor binds an (a, b, r) tuple of 6 limbs each
            // → 18 columns per side.
            assert_eq!(d.a_columns.len(), 3 * LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_FP);
        }
        // Labels are unique.
        let mut labels: Vec<&str> =
            descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), descriptors.len(), "descriptor labels must be unique");
        // B-side tuple is the (a, b, r) layout of nfp.
        let d0 = &descriptors[0];
        assert_eq!(d0.b_columns[0], crate::nonnative_fp_air::COL_A_OFFSET);
        assert_eq!(
            d0.b_columns[LIMBS_PER_FP],
            crate::nonnative_fp_air::COL_B_OFFSET,
        );
        assert_eq!(
            d0.b_columns[2 * LIMBS_PER_FP],
            crate::nonnative_fp_air::COL_R_OFFSET,
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: column layout pinned
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_IN_X_C0_LIMB_OFFSET, 0);
        assert_eq!(COL_IN_X_C1_LIMB_OFFSET, LIMBS_PER_FP);
        assert_eq!(COL_IN_Y_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_OUT_X_C0_LIMB_OFFSET);
        assert_eq!(COL_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_NX_C0_LIMB_OFFSET);
        assert_eq!(COL_NX_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_DX_C0_LIMB_OFFSET);
        assert_eq!(COL_DX_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_NY_C0_LIMB_OFFSET);
        assert_eq!(COL_NY_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_DY_C0_LIMB_OFFSET);
        assert_eq!(COL_DY_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_DX_INV_C0_LIMB_OFFSET);
        assert_eq!(COL_DX_INV_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_DY_INV_C0_LIMB_OFFSET);
        assert_eq!(COL_DY_INV_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_IS_REAL);
        assert_eq!(COL_IS_REAL + 1, NUM_COLUMNS);
        // 5 Fp2 groups × 2 Fp × 6 limbs + 1 selector = 121.
        // (in, out, [nx,dx,ny,dy], [dx_inv, dy_inv]) = 2 + 2 + 4 + 2 = 10 Fp2 pairs
        // × 2 Fp × 6 limbs = 120, + 1 selector = 121.
        assert_eq!(NUM_COLUMNS, 10 * 2 * LIMBS_PER_FP + 1);
        assert_eq!(NUM_COLUMNS, 121);
        assert_eq!(NUM_ROW_CONSTRAINTS, 1 + LIMBS_PER_FP);
        assert_eq!(NUM_ROW_CONSTRAINTS, 7);
    }
}
