//! BLS12-381 G2 cofactor clearing AIR scaffold (#124 — post-SSWU stage
//! that maps points from `E'` to BLS12-381 `G2` and multiplies by the
//! cofactor for the correct subgroup).
//!
//! # Purpose
//!
//! `hash_to_curve(BLS12381G2_XMD:SHA-256_SSWU_RO_)` ends with two
//! arithmetic-heavy stages **after** [`crate::hash_to_g2_air`] finishes
//! the SSWU pre-image:
//!
//!   1. **3-isogeny evaluation** `iso_E'_to_E` mapping the SSWU output
//!      point on the isogeny curve `E'(Fp2) : y² = x³ + 240·i·x + (1012 + 1012·u)`
//!      to a point on BLS12-381's `G2 : y² = x³ + 4(1+u)`. The isogeny
//!      is a rational map of degree 3 whose numerator/denominator
//!      polynomials in `x_E'` are tabulated in RFC 9380 §E.3 and have
//!      degrees 3 / 2 for `x` and 3 / 3 for `y/y_E'`.
//!
//!   2. **Cofactor clearing**. After the isogeny the point lies on
//!      BLS12-381 `G2(Fp2)` but **not necessarily** in the
//!      prime-order-`r` subgroup. The BLS-family-friendly clearing
//!      formula (Wahby–Boneh / Budroni–Pintore) uses the `ψ` twisted
//!      Frobenius endomorphism and the BLS12-381 parameter
//!      `x = -0xd201000000010000`:
//!
//!      `clear_cofactor(P) = (x² - x - 1)·P + (x - 1)·ψ(P) + ψ²(P)`.
//!
//! Both stages are very heavy algebraically (the isogeny alone is ~25
//! Fp2 multiplications + 1 Fp2 inversion; cofactor clearing is ~600
//! Fp2 doublings + ~100 Fp2 additions + 2 ψ evaluations). This
//! scaffold gets the **column layout**, the **trace builder**, the
//! **cross-AIR LogUp descriptors** to [`crate::hash_to_g2_air`]
//! (input side) and to [`crate::bls_pairing_air`] (output side), and
//! **one substantive Fp limb-level binding** (`out_x.c0 = isogeny_x.c0`
//! limb-by-limb) into the tree so downstream isogeny/cofactor rows can
//! be added incrementally without re-litigating the row/column shape.
//!
//! # Witness shape
//!
//! Per row this AIR commits one
//! `E'-point → G2-point` transformation:
//!
//!   * `in_point_e_prime` — the 4 Fp2 components `(x_E'.c0, x_E'.c1,
//!     y_E'.c0, y_E'.c1)` of the input point on the SSWU isogeny curve
//!     `E'`. Stored as 4 × 6 = 24 BE u64 limb columns. **Source**: the
//!     output of [`crate::hash_to_g2_air`]'s SSWU stage once that AIR
//!     is extended with `pre_isogeny_*` columns (see
//!     [`make_cofactor_input_from_sswu_descriptor`]). For now the host-
//!     side trace builder threads the same `G2Affine` into both
//!     `in_point_e_prime` and `out_point_g2`, treating the isogeny + ψ
//!     stages as the identity for scaffolding purposes; the
//!     representative algebraic constraint pinned below
//!     (`out_x.c0 limb_j = isogeny_intermediate_x.c0 limb_j`) is
//!     therefore satisfied without committing the full isogeny
//!     rational-map evaluation.
//!   * `isogeny_intermediate` — 4 Fp2 components representing the
//!     **output of stage 1** (the 3-isogeny `E' → G2`), pre-cofactor
//!     clearing. 4 × 6 = 24 limb columns.
//!   * `out_point_g2` — 4 Fp2 components of the **final** subgroup-
//!     correct G2 point. 4 × 6 = 24 limb columns.
//!   * `is_real` — selector (1 on real rows, 0 on padding).
//!
//! # What is algebraically enforced
//!
//!   1. `is_real ∈ {0, 1}` — selector binarity.
//!   2. Output G2 limb-decomposition mirrors of [`crate::hash_to_g2_air`]
//!      are deferred until canonical-byte columns are added (the output
//!      bytes are already committed by `hash_to_g2_air` on its side, so
//!      this AIR only needs the limb form). **Six Fp limb-equality
//!      constraints** bind `out_x.c0` to the isogeny intermediate's
//!      `x.c0` limbs (i.e. the cofactor-clearing stage preserves the
//!      `x` coordinate's `c0` component when ψ acts as the identity on
//!      `x.c0` — which is true for the trivial-passthrough test point
//!      but **NOT** for general points). The constraint shape is the
//!      right shape for the eventual cross-AIR LogUp binding into
//!      `nonnative_fp_air` once the algebraic ψ evaluation lands; the
//!      current scaffold uses it as a "pass-through" so the witness
//!      threads correctly and the constraint is non-vacuous.
//!
//! Total row-local algebraic constraints: 1 + 6 = **7**.
//!
//! # What is NOT yet algebraically enforced (deferred)
//!
//! - **Stage 1: 3-isogeny rational-map evaluation**
//!   `(x_E', y_E') → (n_x(x_E')/d_x(x_E'), y_E' · n_y(x_E')/d_y(x_E'))`.
//!   The four polynomial evaluations (`n_x`, `d_x`, `n_y`, `d_y`) plus
//!   two Fp2 inversions plus 2 Fp2 multiplications would consume ~25
//!   Fp-mul rows of [`crate::nonnative_fp_air`] via cross-AIR LogUp.
//!   The host-side trace builder uses blst's `blst_hash_to_g2` which
//!   already performs the isogeny internally; the scaffold commits
//!   the post-isogeny point and the post-cofactor-clearing point but
//!   defers the algebraic check that one was actually computed from
//!   the other.
//! - **Stage 2: ψ endomorphism evaluation**
//!   `ψ(x, y) = (x^p · constant, y^p · constant)` where the constants
//!   are the IETF-fixed `ψ_x = u^{(p-1)/3}` and `ψ_y = u^{(p-1)/2}` in
//!   `Fp2`. Each ψ application requires 2 Fp2 Frobenius (1 conjugation
//!   on `Fp` × 2 components) + 2 Fp2 muls.
//! - **Stage 2: scalar multiplication** by `(x² - x - 1)` and `(x - 1)`.
//!   The BLS parameter `x = -0xd201000000010000` has 64-bit width;
//!   double-and-add requires ~64 G2 doublings + ~50 G2 adds (each ~12
//!   Fp2 muls per double, ~16 per add).
//! - **Stage 2: final G2 addition** `(x² - x - 1)·P + (x - 1)·ψ(P) + ψ²(P)`.
//!
//! # Cross-AIR linkages
//!
//! - [`make_cofactor_input_from_sswu_descriptor`] — binds the 24-limb
//!   `in_point_e_prime` columns to the (eventual) `pre_isogeny_point`
//!   columns of [`crate::hash_to_g2_air`]. **NOTE**: `hash_to_g2_air`
//!   does not currently expose `pre_isogeny_*` columns; the descriptor
//!   here points at the closest existing alignment — the SSWU
//!   intermediates' final `(xd, n1)`-style outputs — as a *target
//!   shape*. The full `pre_isogeny_*` columns will be added to
//!   `hash_to_g2_air` in a follow-up round (#119 step N+1).
//! - [`make_cofactor_output_to_pairing_descriptor`] — binds the 24-limb
//!   `out_point_g2` columns to [`crate::bls_pairing_air`]'s
//!   `(sig_x_c0, sig_x_c1, sig_y_c0, sig_y_c1)` limb columns. This
//!   replaces the [`crate::hash_to_g2_air::make_hash_to_g2_to_bls_pairing_descriptor`]
//!   descriptor in the eventual full chain: the pairing AIR's signature
//!   point flows from **here**, not directly from `hash_to_g2_air`
//!   (because `hash_to_g2_air` commits the post-cofactor-clearing
//!   point via host-side oracle, so the binding into `bls_pairing_air`
//!   currently bypasses the algebraic cofactor-clearing step).
//! - [`make_cofactor_step_descriptors`] — returns 5 representative
//!   Fp-mul descriptors covering one isogeny rational-evaluation row
//!   and a few representative ψ-evaluation Fp-muls. Each points at a
//!   single [`crate::nonnative_fp_air`] row.
//!
//! # Notes on the curve choice
//!
//! As with the rest of the BLS12-381 algebraic stack, this AIR is
//! built on `CurveType::Bls12381`. The witness builder calls
//! [`crate::bls_sig::hash_to_g2_affine`] (blst) to produce the
//! output G2 bytes for a real signature point, or accepts a
//! caller-supplied [`crate::pairing::G2Affine`] for the
//! identity / generator-style tests.

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

/// BLS12-381 parameter `x = -0xd201000000010000` (committed here as a
/// constant for future cofactor-clearing scalar-mul rows; not used by
/// the present scaffold constraints).
pub const BLS_X_ABS: u64 = 0xd201_0000_0001_0000;

// ─── Column layout ────────────────────────────────────────────────────
//
// Row layout (per E'-point → G2 row):
//
//   in_x_c0_limbs       : 6 BE u64
//   in_x_c1_limbs       : 6
//   in_y_c0_limbs       : 6
//   in_y_c1_limbs       : 6
//   iso_x_c0_limbs      : 6
//   iso_x_c1_limbs      : 6
//   iso_y_c0_limbs      : 6
//   iso_y_c1_limbs      : 6
//   out_x_c0_limbs      : 6
//   out_x_c1_limbs      : 6
//   out_y_c0_limbs      : 6
//   out_y_c1_limbs      : 6
//   is_real             : 1
//   ── ψ endomorphism witness (added by step #262) ──
//   frob_x_c0_limbs     : 6   ← `conjugate(iso_x).c0` = iso_x.c0
//   frob_x_c1_limbs     : 6   ← `conjugate(iso_x).c1` = -iso_x.c1 (mod p)
//   frob_y_c0_limbs     : 6   ← `conjugate(iso_y).c0` = iso_y.c0
//   frob_y_c1_limbs     : 6   ← `conjugate(iso_y).c1` = -iso_y.c1 (mod p)
//   psi_x_c0_limbs      : 6   ← ξ⁻¹ · frob_x in c0
//   psi_x_c1_limbs      : 6   ← ξ⁻¹ · frob_x in c1
//   psi_y_c0_limbs      : 6   ← ξ⁻² · frob_y in c0
//   psi_y_c1_limbs      : 6   ← ξ⁻² · frob_y in c1
//
// total = 12 * 6 + 1 + 8 * 6 = 73 + 48 = 121

pub const COL_IN_X_C0_LIMB_OFFSET: usize = 0;
pub const COL_IN_X_C1_LIMB_OFFSET: usize = COL_IN_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_IN_Y_C0_LIMB_OFFSET: usize = COL_IN_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_IN_Y_C1_LIMB_OFFSET: usize = COL_IN_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_ISO_X_C0_LIMB_OFFSET: usize = COL_IN_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_ISO_X_C1_LIMB_OFFSET: usize = COL_ISO_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_ISO_Y_C0_LIMB_OFFSET: usize = COL_ISO_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_ISO_Y_C1_LIMB_OFFSET: usize = COL_ISO_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_OUT_X_C0_LIMB_OFFSET: usize = COL_ISO_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_X_C1_LIMB_OFFSET: usize = COL_OUT_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_Y_C0_LIMB_OFFSET: usize = COL_OUT_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_OUT_Y_C1_LIMB_OFFSET: usize = COL_OUT_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_IS_REAL: usize = COL_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;

// ψ endomorphism witness columns (Frobenius + twist-constant product).
pub const COL_FROB_X_C0_LIMB_OFFSET: usize = COL_IS_REAL + 1;
pub const COL_FROB_X_C1_LIMB_OFFSET: usize = COL_FROB_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_FROB_Y_C0_LIMB_OFFSET: usize = COL_FROB_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_FROB_Y_C1_LIMB_OFFSET: usize = COL_FROB_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_PSI_X_C0_LIMB_OFFSET: usize = COL_FROB_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PSI_X_C1_LIMB_OFFSET: usize = COL_PSI_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PSI_Y_C0_LIMB_OFFSET: usize = COL_PSI_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_PSI_Y_C1_LIMB_OFFSET: usize = COL_PSI_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const NUM_COLUMNS: usize = COL_PSI_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;

/// Row-local constraints:
///   0:           is_real ∈ {0, 1}
///   1..7:        6 limb-equality constraints
///                `out_x.c0.limbs[j] = iso_x.c0.limbs[j]`
///                (representative "cofactor-clearing preserves x.c0
///                component when ψ is the identity on x.c0" — used as
///                a scaffold pass-through; real ψ generalizes this).
///   7..13:       6 ψ-Frobenius c0 passthrough constraints
///                `frob_x.c0.limbs[j] = iso_x.c0.limbs[j]`
///                (algebraic statement that the p-power Frobenius on
///                Fp2 = `(a + b·u) ↦ (a − b·u)` leaves the c0 component
///                invariant — natively-provable since limbs are equal).
///   13..19:      6 ψ-Frobenius c0 passthrough constraints
///                `frob_y.c0.limbs[j] = iso_y.c0.limbs[j]`.
///
/// The companion **c1 = −iso.c1 (mod p)** and **ψ = ξ⁻¹ · frob (mod p)**
/// statements involve modular arithmetic and are pinned via cross-AIR
/// LogUp into [`crate::nonnative_fp_air`] (see
/// [`make_psi_frobenius_neg_descriptors`] and
/// [`make_psi_xi_mul_descriptors`]).
pub const NUM_PASSTHROUGH_CONSTRAINTS: usize = LIMBS_PER_FP;
pub const NUM_FROB_C0_CONSTRAINTS: usize = 2 * LIMBS_PER_FP;
pub const NUM_ROW_CONSTRAINTS: usize =
    1 + NUM_PASSTHROUGH_CONSTRAINTS + NUM_FROB_C0_CONSTRAINTS;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct G2CofactorClearRow {
    /// Input point on `E'` (the SSWU isogeny curve), as 4 Fp2 components.
    pub in_x_c0: Fp,
    pub in_x_c1: Fp,
    pub in_y_c0: Fp,
    pub in_y_c1: Fp,
    /// Post-isogeny intermediate point on `BLS12-381 G2` (pre-cofactor
    /// clearing).
    pub iso_x_c0: Fp,
    pub iso_x_c1: Fp,
    pub iso_y_c0: Fp,
    pub iso_y_c1: Fp,
    /// Final cofactor-cleared output point in the r-torsion subgroup.
    pub out_x_c0: Fp,
    pub out_x_c1: Fp,
    pub out_y_c0: Fp,
    pub out_y_c1: Fp,
    /// p-power Frobenius applied to `iso` (Fp2 conjugation):
    /// `frob_x = (iso_x.c0, -iso_x.c1 mod p)`,
    /// `frob_y = (iso_y.c0, -iso_y.c1 mod p)`.
    pub frob_x_c0: Fp,
    pub frob_x_c1: Fp,
    pub frob_y_c0: Fp,
    pub frob_y_c1: Fp,
    /// ψ output: `psi_x = ξ⁻¹ · frob_x`, `psi_y = ξ⁻² · frob_y` (Fp2 mul).
    pub psi_x_c0: Fp,
    pub psi_x_c1: Fp,
    pub psi_y_c0: Fp,
    pub psi_y_c1: Fp,
}

#[derive(Clone, Debug, Default)]
pub struct G2CofactorClearWitness {
    pub rows: Vec<G2CofactorClearRow>,
}

impl G2CofactorClearWitness {
    /// Build a single-row witness from a host-side `E'` point. Because
    /// the algebraic isogeny and ψ stages are deferred (see module
    /// docs), this constructor wires the *same* point into the
    /// `in_point_e_prime`, `isogeny_intermediate`, and `out_point_g2`
    /// columns — i.e. it treats both stages as the identity. The
    /// representative `out_x.c0 = iso_x.c0` constraint is satisfied
    /// trivially by this pass-through.
    ///
    /// Returns `None` if the input point is the identity (the
    /// cofactor-clearing scaffold rejects the identity to keep the
    /// downstream `bls_pairing_air` binding non-vacuous; the real
    /// hash-to-curve flow never produces the identity).
    pub fn from_e_prime_point(e_prime: G2Affine) -> Option<Self> {
        if e_prime.infinity {
            return None;
        }
        // Scaffold ψ:
        //   * Frobenius (p-power) on Fp2 = complex conjugation:
        //     `frob(x) = (x.c0, −x.c1 mod p)`.
        //   * The twist constants `ξ⁻¹` and `ξ⁻²` are committed
        //     implicitly via the cross-AIR LogUp into nonnative_fp_air
        //     (see `make_psi_xi_mul_descriptors`); the host-side scaffold
        //     here uses the identity element `(1, 0)` for both, so
        //     `psi_x = frob_x` and `psi_y = frob_y` value-wise. The
        //     algebraic c0-passthrough constraints are satisfied
        //     trivially; the c1-negation is the binding statement
        //     deferred to nonnative_fp_air.
        let iso = G2Affine {
            x: e_prime.x,
            y: e_prime.y,
            infinity: false,
        };
        let frob_x = iso.x.conjugate();
        let frob_y = iso.y.conjugate();
        Some(Self {
            rows: vec![G2CofactorClearRow {
                in_x_c0: e_prime.x.c0,
                in_x_c1: e_prime.x.c1,
                in_y_c0: e_prime.y.c0,
                in_y_c1: e_prime.y.c1,
                iso_x_c0: iso.x.c0,
                iso_x_c1: iso.x.c1,
                iso_y_c0: iso.y.c0,
                iso_y_c1: iso.y.c1,
                out_x_c0: e_prime.x.c0,
                out_x_c1: e_prime.x.c1,
                out_y_c0: e_prime.y.c0,
                out_y_c1: e_prime.y.c1,
                frob_x_c0: frob_x.c0,
                frob_x_c1: frob_x.c1,
                frob_y_c0: frob_y.c0,
                frob_y_c1: frob_y.c1,
                // Scaffold twist constant = identity → ψ = frob value-wise.
                psi_x_c0: frob_x.c0,
                psi_x_c1: frob_x.c1,
                psi_y_c0: frob_y.c0,
                psi_y_c1: frob_y.c1,
            }],
        })
    }

    /// Build a single-row witness for a real signature point obtained
    /// from `hash_to_curve(msg, dst)` via blst. The blst routine
    /// performs the full pipeline (`expand_message_xmd → mod p → SSWU
    /// → isogeny → cofactor clear`) internally, so we obtain only the
    /// final `G2` point. The scaffold commits that point into all
    /// three (`in`, `iso`, `out`) column groups; the cross-AIR LogUp
    /// links into [`crate::hash_to_g2_air`] and
    /// [`crate::bls_pairing_air`] then carry the real binding once
    /// the full algebraic chain lands.
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
    pub fn push_raw(&mut self, row: G2CofactorClearRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &G2CofactorClearWitness,
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
            (COL_ISO_X_C0_LIMB_OFFSET, &row.iso_x_c0),
            (COL_ISO_X_C1_LIMB_OFFSET, &row.iso_x_c1),
            (COL_ISO_Y_C0_LIMB_OFFSET, &row.iso_y_c0),
            (COL_ISO_Y_C1_LIMB_OFFSET, &row.iso_y_c1),
            (COL_OUT_X_C0_LIMB_OFFSET, &row.out_x_c0),
            (COL_OUT_X_C1_LIMB_OFFSET, &row.out_x_c1),
            (COL_OUT_Y_C0_LIMB_OFFSET, &row.out_y_c0),
            (COL_OUT_Y_C1_LIMB_OFFSET, &row.out_y_c1),
            (COL_FROB_X_C0_LIMB_OFFSET, &row.frob_x_c0),
            (COL_FROB_X_C1_LIMB_OFFSET, &row.frob_x_c1),
            (COL_FROB_Y_C0_LIMB_OFFSET, &row.frob_y_c0),
            (COL_FROB_Y_C1_LIMB_OFFSET, &row.frob_y_c1),
            (COL_PSI_X_C0_LIMB_OFFSET, &row.psi_x_c0),
            (COL_PSI_X_C1_LIMB_OFFSET, &row.psi_x_c1),
            (COL_PSI_Y_C0_LIMB_OFFSET, &row.psi_y_c0),
            (COL_PSI_Y_C1_LIMB_OFFSET, &row.psi_y_c1),
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

pub struct G2CofactorClearConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl G2CofactorClearConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for G2CofactorClearConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("passthrough_out_x_c0_limb_{}_eq_iso", j));
        }
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("psi_frob_x_c0_limb_{}_eq_iso", j));
        }
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("psi_frob_y_c0_limb_{}_eq_iso", j));
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

        // 1..7: passthrough — out_x.c0.limbs[j] = iso_x.c0.limbs[j].
        for j in 0..LIMBS_PER_FP {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = &columns[COL_OUT_X_C0_LIMB_OFFSET + j][r];
                let rhs = &columns[COL_ISO_X_C0_LIMB_OFFSET + j][r];
                c[r] = lhs.sub(rhs);
            }
            out.push(c);
        }

        // 7..13: ψ Frobenius c0 passthrough — frob_x.c0.limbs[j] = iso_x.c0.limbs[j].
        // Algebraic statement: the p-power Frobenius on Fp2 is complex
        // conjugation `(a + b·u) ↦ (a − b·u)`, so the c0 component is
        // preserved limb-by-limb. The companion c1 = −iso.c1 (mod p)
        // statement requires modular arithmetic and is deferred to the
        // nonnative_fp_air SUB rows linked via
        // `make_psi_frobenius_neg_descriptors`.
        for j in 0..LIMBS_PER_FP {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = &columns[COL_FROB_X_C0_LIMB_OFFSET + j][r];
                let rhs = &columns[COL_ISO_X_C0_LIMB_OFFSET + j][r];
                c[r] = lhs.sub(rhs);
            }
            out.push(c);
        }

        // 13..19: ψ Frobenius c0 passthrough — frob_y.c0.limbs[j] = iso_y.c0.limbs[j].
        for j in 0..LIMBS_PER_FP {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = &columns[COL_FROB_Y_C0_LIMB_OFFSET + j][r];
                let rhs = &columns[COL_ISO_Y_C0_LIMB_OFFSET + j][r];
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

        // 1..7: passthrough.
        for j in 0..LIMBS_PER_FP {
            let body = col_evals[COL_OUT_X_C0_LIMB_OFFSET + j]
                .sub(&col_evals[COL_ISO_X_C0_LIMB_OFFSET + j]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..13: ψ Frobenius x.c0 passthrough.
        for j in 0..LIMBS_PER_FP {
            let body = col_evals[COL_FROB_X_C0_LIMB_OFFSET + j]
                .sub(&col_evals[COL_ISO_X_C0_LIMB_OFFSET + j]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 13..19: ψ Frobenius y.c0 passthrough.
        for j in 0..LIMBS_PER_FP {
            let body = col_evals[COL_FROB_Y_C0_LIMB_OFFSET + j]
                .sub(&col_evals[COL_ISO_Y_C0_LIMB_OFFSET + j]);
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

        // 1..7: passthrough.
        for j in 0..LIMBS_PER_FP {
            let body = poly_sub(
                &col_coeffs[COL_OUT_X_C0_LIMB_OFFSET + j],
                &col_coeffs[COL_ISO_X_C0_LIMB_OFFSET + j],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..13: ψ Frobenius x.c0 passthrough.
        for j in 0..LIMBS_PER_FP {
            let body = poly_sub(
                &col_coeffs[COL_FROB_X_C0_LIMB_OFFSET + j],
                &col_coeffs[COL_ISO_X_C0_LIMB_OFFSET + j],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 13..19: ψ Frobenius y.c0 passthrough.
        for j in 0..LIMBS_PER_FP {
            let body = poly_sub(
                &col_coeffs[COL_FROB_Y_C0_LIMB_OFFSET + j],
                &col_coeffs[COL_ISO_Y_C0_LIMB_OFFSET + j],
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
        // Limb columns are 64-bit-bounded by construction (u64 from
        // the witness) and pinned to specific values by the
        // passthrough constraint; no byte columns to range-check
        // (those live in `hash_to_g2_air`).
        LookupRequirements {
            tables: vec![LookupTable::range(256)],
            declarations: Vec::new(),
        }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Cross-AIR LogUp: bind the 24-limb `in_point_e_prime` columns of this
/// AIR to the 24-limb (eventual) `pre_isogeny_point` columns of
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
/// `in_point_e_prime` columns. Once those columns land, the
/// `b_columns` list below will need to be updated to point at the
/// new offsets (the column count of 24 stays the same).
pub fn make_cofactor_input_from_sswu_descriptor(
    cofactor_layer_index: usize,
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
        label: "g2_cofactor_input_from_sswu_v1".into(),
        a_layer_index: cofactor_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: hash_to_g2_layer_index,
        b_columns,
        b_selector_column: Some(h2g2::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp: bind the 24-limb `out_point_g2` columns of this AIR
/// to the matching `(sig_x_c0, sig_x_c1, sig_y_c0, sig_y_c1)` limb
/// columns of [`crate::bls_pairing_air`]. This makes the pairing AIR's
/// signature point a *consequence* of the full hash-to-curve →
/// isogeny → cofactor-clear pipeline rather than a host-side oracle.
///
/// A side = [`crate::bls_pairing_air`] (selected by its `IS_REAL`).
/// B side = this AIR (selected by [`COL_IS_REAL`]).
///
/// This descriptor *supersedes*
/// [`crate::hash_to_g2_air::make_hash_to_g2_to_bls_pairing_descriptor`]
/// in the eventual full chain: once this AIR is wired into the joint
/// LogUp protocol, the pairing AIR's `sig_*` columns are bound to
/// **this AIR's `out_*` columns**, not directly to `hash_to_g2_air`'s
/// `out_*` columns (which are the post-cofactor-cleared point arriving
/// via a host-side oracle and would skip the algebraic isogeny + ψ
/// stages).
pub fn make_cofactor_output_to_pairing_descriptor(
    cofactor_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| bp::COL_SIG_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| bp::COL_SIG_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| bp::COL_SIG_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| bp::COL_SIG_Y_C1_LIMB_OFFSET + j))
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| COL_OUT_X_C0_LIMB_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_X_C1_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_Y_C0_LIMB_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| COL_OUT_Y_C1_LIMB_OFFSET + j))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "g2_cofactor_output_to_bls_pairing_v1".into(),
        a_layer_index: bls_pairing_layer_index,
        a_columns,
        a_selector_column: Some(bp::COL_IS_REAL),
        b_layer_index: cofactor_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Build a vector of cross-AIR LogUp descriptors that wire
/// representative Fp multiplication steps of the isogeny + ψ pipeline
/// to rows of [`crate::nonnative_fp_air`]. Each descriptor uses the
/// `nonnative_fp_air`'s `(a, b, r)` limb layout (18 columns) as the B
/// side, and an `(a_limbs, b_limbs, r_limbs)` triple drawn from this
/// AIR's committed columns as the A side.
///
/// # Scaffolding shape
///
/// The current 5 descriptors point each isogeny / ψ step to a *single
/// representative* Fp-mul row, treating the multi-Fp-mul rational-map
/// evaluation as if it were one Fp multiplication. The actual
/// decomposition (Karatsuba Fp2 mul = 3 Fp muls, polynomial
/// evaluation = degree-3 Horner = 3 Fp2 muls = 9 Fp muls) requires
/// committing intermediate scratch columns + adding ~25 descriptors
/// per isogeny evaluation. Both extensions preserve the descriptor-set
/// shape this function returns; the follow-up "isogeny scratch" round
/// will add the missing scratch columns and replace each of these
/// placeholder Fp-mul links with the proper triples.
///
/// Returns 5 descriptors:
///   - `g2_cofactor_iso_x_c0_mul_v1` — (in_x.c0, in_x.c0, iso_x.c0)
///     representative isogeny `n_x(x)` polynomial-eval Fp mul.
///   - `g2_cofactor_iso_y_c0_mul_v1` — (in_y.c0, in_y.c0, iso_y.c0)
///     representative isogeny `y · n_y(x)/d_y(x)` Fp mul.
///   - `g2_cofactor_psi_x_c0_mul_v1` — (iso_x.c0, iso_x.c1, out_x.c0)
///     representative ψ `x · constant` Fp mul.
///   - `g2_cofactor_psi_y_c0_mul_v1` — (iso_y.c0, iso_y.c1, out_y.c0)
///     representative ψ `y · constant` Fp mul.
///   - `g2_cofactor_scalar_x_mul_v1` — (iso_x.c0, out_x.c1, out_x.c0)
///     representative scalar-mul-by-`(x-1)` doubling-row Fp mul.
pub fn make_cofactor_step_descriptors(
    cofactor_layer_index: usize,
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
            a_layer_index: cofactor_layer_index,
            a_columns,
            a_selector_column: Some(COL_IS_REAL),
            b_layer_index: nonnative_fp_layer_index,
            b_columns: b_columns.clone(),
            b_selector_column: Some(nfp::COL_SEL_MUL),
        };

    vec![
        make_desc(
            "g2_cofactor_iso_x_c0_mul_v1",
            triple(
                COL_IN_X_C0_LIMB_OFFSET,
                COL_IN_X_C0_LIMB_OFFSET,
                COL_ISO_X_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_iso_y_c0_mul_v1",
            triple(
                COL_IN_Y_C0_LIMB_OFFSET,
                COL_IN_Y_C0_LIMB_OFFSET,
                COL_ISO_Y_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_psi_x_c0_mul_v1",
            triple(
                COL_ISO_X_C0_LIMB_OFFSET,
                COL_ISO_X_C1_LIMB_OFFSET,
                COL_OUT_X_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_psi_y_c0_mul_v1",
            triple(
                COL_ISO_Y_C0_LIMB_OFFSET,
                COL_ISO_Y_C1_LIMB_OFFSET,
                COL_OUT_Y_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_scalar_x_mul_v1",
            triple(
                COL_ISO_X_C0_LIMB_OFFSET,
                COL_OUT_X_C1_LIMB_OFFSET,
                COL_OUT_X_C0_LIMB_OFFSET,
            ),
        ),
    ]
}

/// Cross-AIR LogUp: bind the ψ-Frobenius c1-negation steps to
/// [`crate::nonnative_fp_air`] SUB rows. The p-power Frobenius on Fp2 is
/// complex conjugation: `(a + b·u) ↦ (a − b·u)`. The c0 component is
/// preserved (handled by row-local constraints 7..19 above); the c1
/// component requires `frob.c1 = (0 − iso.c1) mod p`, which is exactly
/// a nonnative_fp_air SUB row of the form `r = a − b` with `a = 0` and
/// `b = iso.c1`, `r = frob.c1`.
///
/// Returns 2 descriptors (one per Fp2 component — x and y).
///
/// **Caveat**: the A-side `a` slot points at the same 6 columns as
/// `b` (a 0-valued "constant" slot is **not** yet committed as a
/// dedicated zero-column in this AIR; this is a scaffold placeholder
/// that the joint-prove orchestrator will need to align with an actual
/// nonnative_fp_air row whose `a` limbs are zero on the matched row).
/// The descriptor shape and column count (18) is correct.
pub fn make_psi_frobenius_neg_descriptors(
    cofactor_layer_index: usize,
    nonnative_fp_layer_index: usize,
) -> Vec<crate::cross_air_logup::CrossAirLogUpDescriptor> {
    use crate::nonnative_fp_air as nfp;

    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| nfp::COL_A_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_B_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_R_OFFSET + j))
        .collect();

    // A-side triple: (0-stand-in, iso.c1, frob.c1). The 0-stand-in
    // reuses iso.c0 columns as a placeholder; the joint-prove
    // orchestrator must inject a real zero column on the A side
    // (or use a dedicated `psi_zero_c0_limbs` group) once wired into
    // a joint_prove test. The COL_COUNT (18) is the correct shape.
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
            a_layer_index: cofactor_layer_index,
            a_columns,
            a_selector_column: Some(COL_IS_REAL),
            b_layer_index: nonnative_fp_layer_index,
            b_columns: b_columns.clone(),
            b_selector_column: Some(nfp::COL_SEL_SUB),
        };

    vec![
        make_desc(
            "g2_cofactor_psi_frob_x_c1_neg_v1",
            triple(
                COL_ISO_X_C0_LIMB_OFFSET, // placeholder for the 0 a-slot
                COL_ISO_X_C1_LIMB_OFFSET,
                COL_FROB_X_C1_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_psi_frob_y_c1_neg_v1",
            triple(
                COL_ISO_Y_C0_LIMB_OFFSET, // placeholder for the 0 a-slot
                COL_ISO_Y_C1_LIMB_OFFSET,
                COL_FROB_Y_C1_LIMB_OFFSET,
            ),
        ),
    ]
}

/// Cross-AIR LogUp: bind the ψ-twist-constant multiplication steps to
/// [`crate::nonnative_fp_air`] MUL rows.
///
/// The ψ map computes `psi.x = ξ⁻¹ · frob_x` and `psi.y = ξ⁻² · frob_y`
/// as Fp2 multiplications. Each Fp2 mul = 3 Fp muls (Karatsuba) plus
/// some Fp adds/subs. The 4 descriptors here are representative
/// Fp-mul placeholders binding one limb-tuple per component to a
/// single nonnative_fp_air row. Full Karatsuba expansion would add
/// 3 × 2 = 6 descriptors per Fp2 mul (one per Fp sub-product) plus
/// scratch columns; that is deferred.
///
/// Returns 4 descriptors covering the (c0, c1) of psi.x and psi.y.
pub fn make_psi_xi_mul_descriptors(
    cofactor_layer_index: usize,
    nonnative_fp_layer_index: usize,
) -> Vec<crate::cross_air_logup::CrossAirLogUpDescriptor> {
    use crate::nonnative_fp_air as nfp;

    let b_columns: Vec<usize> = (0..LIMBS_PER_FP)
        .map(|j| nfp::COL_A_OFFSET + j)
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_B_OFFSET + j))
        .chain((0..LIMBS_PER_FP).map(|j| nfp::COL_R_OFFSET + j))
        .collect();

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
            a_layer_index: cofactor_layer_index,
            a_columns,
            a_selector_column: Some(COL_IS_REAL),
            b_layer_index: nonnative_fp_layer_index,
            b_columns: b_columns.clone(),
            b_selector_column: Some(nfp::COL_SEL_MUL),
        };

    vec![
        make_desc(
            "g2_cofactor_psi_xi_inv_x_c0_mul_v1",
            triple(
                COL_FROB_X_C0_LIMB_OFFSET,
                COL_FROB_X_C1_LIMB_OFFSET, // placeholder ξ⁻¹ slot
                COL_PSI_X_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_psi_xi_inv_x_c1_mul_v1",
            triple(
                COL_FROB_X_C1_LIMB_OFFSET,
                COL_FROB_X_C0_LIMB_OFFSET, // placeholder ξ⁻¹ slot
                COL_PSI_X_C1_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_psi_xi_inv2_y_c0_mul_v1",
            triple(
                COL_FROB_Y_C0_LIMB_OFFSET,
                COL_FROB_Y_C1_LIMB_OFFSET, // placeholder ξ⁻² slot
                COL_PSI_Y_C0_LIMB_OFFSET,
            ),
        ),
        make_desc(
            "g2_cofactor_psi_xi_inv2_y_c1_mul_v1",
            triple(
                COL_FROB_Y_C1_LIMB_OFFSET,
                COL_FROB_Y_C0_LIMB_OFFSET, // placeholder ξ⁻² slot
                COL_PSI_Y_C1_LIMB_OFFSET,
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
    // Test 1: identity-point input is rejected; synthetic point passes
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn identity_input_is_rejected() {
        assert!(G2CofactorClearWitness::from_e_prime_point(G2Affine::identity()).is_none());
    }

    #[test]
    fn synthetic_point_passes_through() {
        let p = synthetic_e_prime_point();
        let w = G2CofactorClearWitness::from_e_prime_point(p).expect("non-identity accepted");
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        // Passthrough: in == iso == out across all four Fp2 components.
        assert_eq!(row.in_x_c0, row.iso_x_c0);
        assert_eq!(row.iso_x_c0, row.out_x_c0);
        assert_eq!(row.in_x_c1, row.out_x_c1);
        assert_eq!(row.in_y_c0, row.out_y_c0);
        assert_eq!(row.in_y_c1, row.out_y_c1);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: witness builds for a real signature point via blst
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_from_real_signature_point() {
        let w = G2CofactorClearWitness::from_message(b"cofactor-clear-test", POP_DST)
            .expect("hash_to_curve must produce a non-identity point");
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        // Output point is not the identity.
        assert!(!row.out_x_c0.is_zero() || !row.out_x_c1.is_zero());
        assert!(!row.out_y_c0.is_zero() || !row.out_y_c1.is_zero());
        // Round-trip the bytes through G2Affine::from_bytes (the
        // pairing module's curve-membership check) — the scaffold's
        // input came from blst → compressed → G2Affine, so by
        // transitivity the output coordinates must be on the curve.
        let g2 = G2Affine {
            x: Fp2 { c0: row.out_x_c0, c1: row.out_x_c1 },
            y: Fp2 { c0: row.out_y_c0, c1: row.out_y_c1 },
            infinity: false,
        };
        // We don't have an `is_on_curve` predicate exposed; instead
        // check that x and y limbs are non-zero (real hash-to-curve
        // never produces zero coordinates).
        assert!(g2.x.c0.limbs.iter().any(|&l| l != 0));
        assert!(g2.y.c0.limbs.iter().any(|&l| l != 0));
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: constraints zero on honest witness; tampering fires
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let p = synthetic_e_prime_point();
        let w = G2CofactorClearWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = G2CofactorClearConstraintSystem::new(trace.num_rows);
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
    fn passthrough_fires_on_tampered_out_x_c0() {
        let p = synthetic_e_prime_point();
        let w = G2CofactorClearWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Tamper out_x.c0 limb 0.
        let original = cols[COL_OUT_X_C0_LIMB_OFFSET][0].clone();
        cols[COL_OUT_X_C0_LIMB_OFFSET][0] = original.add(&Scalar::one(curve));
        let cs = G2CofactorClearConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Passthrough constraint indices start at 1; limb 0 = index 1.
        let passthrough_limb0 = 1;
        assert!(
            !results[passthrough_limb0][0].is_zero(),
            "tampering out_x.c0 limb 0 must fire the passthrough constraint",
        );
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_selector() {
        let p = synthetic_e_prime_point();
        let w = G2CofactorClearWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(5, CurveType::Bls12381);
        let cs = G2CofactorClearConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "non-binary is_real must fire");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: cross-AIR descriptors are well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn input_from_sswu_descriptor_well_formed() {
        let d = make_cofactor_input_from_sswu_descriptor(0, 1);
        assert_eq!(d.label, "g2_cofactor_input_from_sswu_v1");
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
    fn output_to_pairing_descriptor_well_formed() {
        let d = make_cofactor_output_to_pairing_descriptor(0, 1);
        assert_eq!(d.label, "g2_cofactor_output_to_bls_pairing_v1");
        // A side = bls_pairing (layer 1), B side = this AIR (layer 0).
        assert_eq!(d.a_layer_index, 1);
        assert_eq!(d.b_layer_index, 0);
        assert_eq!(d.a_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(d.b_columns.len(), 4 * LIMBS_PER_FP);
        assert_eq!(
            d.a_selector_column,
            Some(crate::bls_pairing_air::COL_IS_REAL),
        );
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));
        // First B column = this AIR's out_x.c0 limb 0.
        assert_eq!(d.b_columns[0], COL_OUT_X_C0_LIMB_OFFSET);
        // First A column = bls_pairing's sig_x.c0 limb 0.
        assert_eq!(
            d.a_columns[0],
            crate::bls_pairing_air::COL_SIG_X_C0_LIMB_OFFSET,
        );
    }

    #[test]
    fn cofactor_step_descriptor_set_has_five_entries() {
        let descriptors = make_cofactor_step_descriptors(3, 7);
        assert!(
            descriptors.len() >= 5,
            "cofactor-clearing step descriptor set must include >= 5 \
             Fp-mul links (got {}); each isogeny / ψ / scalar-mul stage \
             contributes at least one nfp link",
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
    // Test 5: column layout & helpers
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_IN_X_C0_LIMB_OFFSET, 0);
        assert_eq!(COL_IN_X_C1_LIMB_OFFSET, LIMBS_PER_FP);
        assert_eq!(COL_IN_Y_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_ISO_X_C0_LIMB_OFFSET);
        assert_eq!(COL_ISO_Y_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_OUT_X_C0_LIMB_OFFSET);
        assert_eq!(COL_OUT_Y_C1_LIMB_OFFSET + LIMBS_PER_FP, COL_IS_REAL);
        // ψ-block packed immediately after the IS_REAL selector.
        assert_eq!(COL_IS_REAL + 1, COL_FROB_X_C0_LIMB_OFFSET);
        assert_eq!(COL_FROB_X_C1_LIMB_OFFSET, COL_FROB_X_C0_LIMB_OFFSET + LIMBS_PER_FP);
        assert_eq!(COL_FROB_Y_C0_LIMB_OFFSET, COL_FROB_X_C1_LIMB_OFFSET + LIMBS_PER_FP);
        assert_eq!(COL_FROB_Y_C1_LIMB_OFFSET, COL_FROB_Y_C0_LIMB_OFFSET + LIMBS_PER_FP);
        assert_eq!(COL_PSI_X_C0_LIMB_OFFSET, COL_FROB_Y_C1_LIMB_OFFSET + LIMBS_PER_FP);
        assert_eq!(COL_PSI_Y_C1_LIMB_OFFSET + LIMBS_PER_FP, NUM_COLUMNS);
        // 3 point groups × 4 Fp components × 6 limbs + 1 selector
        //   + 2 ψ groups (frob + psi) × 4 Fp components × 6 limbs.
        assert_eq!(NUM_COLUMNS, 3 * 4 * LIMBS_PER_FP + 1 + 2 * 4 * LIMBS_PER_FP);
        assert_eq!(NUM_COLUMNS, 121);
        assert_eq!(
            NUM_ROW_CONSTRAINTS,
            1 + LIMBS_PER_FP + 2 * LIMBS_PER_FP,
        );
        assert_eq!(NUM_ROW_CONSTRAINTS, 19);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: ψ Frobenius witness & constraints
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn psi_witness_populates_frobenius_and_psi_columns() {
        let p = synthetic_e_prime_point();
        let w = G2CofactorClearWitness::from_e_prime_point(p).unwrap();
        let row = &w.rows[0];
        // Frobenius preserves c0 limb-by-limb.
        assert_eq!(row.frob_x_c0, row.iso_x_c0);
        assert_eq!(row.frob_y_c0, row.iso_y_c0);
        // Frobenius negates c1 mod p: frob.c1 = neg(iso.c1).
        assert_eq!(row.frob_x_c1, row.iso_x_c1.neg());
        assert_eq!(row.frob_y_c1, row.iso_y_c1.neg());
        // Scaffold twist constant = identity → ψ = frob value-wise.
        assert_eq!(row.psi_x_c0, row.frob_x_c0);
        assert_eq!(row.psi_x_c1, row.frob_x_c1);
        assert_eq!(row.psi_y_c0, row.frob_y_c0);
        assert_eq!(row.psi_y_c1, row.frob_y_c1);
    }

    #[test]
    fn psi_frobenius_c0_passthrough_fires_on_tamper() {
        let p = synthetic_e_prime_point();
        let w = G2CofactorClearWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Tamper frob_x.c0 limb 3.
        let original = cols[COL_FROB_X_C0_LIMB_OFFSET + 3][0].clone();
        cols[COL_FROB_X_C0_LIMB_OFFSET + 3][0] = original.add(&Scalar::one(curve));
        let cs = G2CofactorClearConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Frobenius x.c0 block starts at index 1 + LIMBS_PER_FP = 7;
        // limb 3 → index 10.
        let frob_x_c0_limb3 = 1 + LIMBS_PER_FP + 3;
        assert!(
            !results[frob_x_c0_limb3][0].is_zero(),
            "tampering frob_x.c0 limb 3 must fire its passthrough constraint",
        );
    }

    #[test]
    fn psi_frobenius_y_c0_passthrough_fires_on_tamper() {
        let p = synthetic_e_prime_point();
        let w = G2CofactorClearWitness::from_e_prime_point(p).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        let original = cols[COL_FROB_Y_C0_LIMB_OFFSET + 5][0].clone();
        cols[COL_FROB_Y_C0_LIMB_OFFSET + 5][0] = original.add(&Scalar::one(curve));
        let cs = G2CofactorClearConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let frob_y_c0_limb5 = 1 + LIMBS_PER_FP + LIMBS_PER_FP + 5;
        assert!(
            !results[frob_y_c0_limb5][0].is_zero(),
            "tampering frob_y.c0 limb 5 must fire its passthrough constraint",
        );
    }

    #[test]
    fn psi_frobenius_neg_descriptors_well_formed() {
        let descriptors = make_psi_frobenius_neg_descriptors(0, 5);
        assert_eq!(descriptors.len(), 2);
        for d in &descriptors {
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 5);
            assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
            assert_eq!(
                d.b_selector_column,
                Some(crate::nonnative_fp_air::COL_SEL_SUB),
            );
            assert_eq!(d.a_columns.len(), 3 * LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_FP);
        }
        let mut labels: Vec<&str> =
            descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn psi_xi_mul_descriptors_well_formed() {
        let descriptors = make_psi_xi_mul_descriptors(0, 5);
        assert_eq!(descriptors.len(), 4);
        for d in &descriptors {
            assert_eq!(d.a_layer_index, 0);
            assert_eq!(d.b_layer_index, 5);
            assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
            assert_eq!(
                d.b_selector_column,
                Some(crate::nonnative_fp_air::COL_SEL_MUL),
            );
            assert_eq!(d.a_columns.len(), 3 * LIMBS_PER_FP);
            assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_FP);
        }
        let mut labels: Vec<&str> =
            descriptors.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn bls_x_constant_matches_iref() {
        // Spec value of the BLS12-381 parameter abs(x).
        // The cofactor clearing scalar mul uses x²-x-1 and x-1 with
        // x = -0xd201000000010000.
        assert_eq!(BLS_X_ABS, 0xd201_0000_0001_0000);
    }
}
