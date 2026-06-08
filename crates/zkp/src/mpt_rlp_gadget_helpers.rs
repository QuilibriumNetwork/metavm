//! Shared evaluation- and polynomial-form constraint-body helpers for the
//! MPT RLP-decoding gadget AIRs.
//!
//! Every gadget AIR in the `mpt_*_rlp_air` family (Phase 1, Phase 3,
//! Phase 4, Phase 6 — and any future shape) uses the same handful of
//! row-local body shapes:
//!
//! 1. **byte_pinned**: `IS_REAL · (col[byte_col] − constant) = 0`
//! 2. **byte_slice_binding**: `IS_REAL · β-RLC_k (RLP_BYTE[rlp_offset + k]
//!    − col[decoded_offset + k])` for k ∈ 0..n
//! 3. **zero_tail**: `IS_REAL · β-RLC_k RLP_BYTE[k]` for k ∈ start..end
//!    (pins the RLP zero-pad region to zero)
//! 4. **decoded_zero_tail**: `IS_REAL · β-RLC_c col[c]` for c ∈
//!    start_col..end_col (pins decoded-field zero-pad bytes, e.g.
//!    KEY_PATH_BYTE bytes 2..32 on Phase 3 / Phase 4 / Phase 6 rows)
//!
//! Each pattern has an eval-form (used in `evaluate_at_point` /
//! `evaluate_on_domain`) and a polynomial-form (used in
//! `build_constraint_polynomial`). Total 8 helpers.
//!
//! Each helper takes `is_real_col` and `rlp_byte_offset` as explicit
//! parameters since they vary slightly between gadgets (though
//! `rlp_byte_offset` is currently 0 in every gadget — kept explicit
//! for forward compatibility).

use crate::field::{CurveType, Scalar};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};

// ─── Eval-form helpers ────────────────────────────────────────────────

/// `IS_REAL · (col_evals[byte_col] − constant)`
pub fn eval_byte_pinned(
    col_evals: &[Scalar],
    byte_col: usize,
    constant: u64,
    is_real_col: usize,
) -> Scalar {
    let curve = col_evals[is_real_col].curve_type();
    let body = col_evals[byte_col].sub(&Scalar::from_u64(constant, curve));
    col_evals[is_real_col].mul(&body)
}

/// `IS_REAL · Σ_{k=0..n} α^k · (RLP_BYTE[rlp_offset + k] − col[decoded_offset + k])`
pub fn eval_byte_slice_binding(
    col_evals: &[Scalar],
    alpha: &Scalar,
    rlp_byte_offset: usize,
    rlp_offset: usize,
    decoded_offset: usize,
    n: usize,
    is_real_col: usize,
) -> Scalar {
    let curve = alpha.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for k in 0..n {
        let body = col_evals[rlp_byte_offset + rlp_offset + k]
            .sub(&col_evals[decoded_offset + k]);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[is_real_col].mul(&acc)
}

/// `IS_REAL · Σ_{k=start..end} α^(k-start) · RLP_BYTE[k]`
pub fn eval_zero_tail(
    col_evals: &[Scalar],
    alpha: &Scalar,
    rlp_byte_offset: usize,
    start: usize,
    end: usize,
    is_real_col: usize,
) -> Scalar {
    let curve = alpha.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for k in start..end {
        let body = col_evals[rlp_byte_offset + k].clone();
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[is_real_col].mul(&acc)
}

/// `IS_REAL · Σ_{c=start_col..end_col} α^(c-start_col) · col[c]`
pub fn eval_decoded_zero_tail(
    col_evals: &[Scalar],
    alpha: &Scalar,
    start_col: usize,
    end_col: usize,
    is_real_col: usize,
) -> Scalar {
    let curve = alpha.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for c in start_col..end_col {
        let body = col_evals[c].clone();
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[is_real_col].mul(&acc)
}

// ─── Polynomial-form helpers ──────────────────────────────────────────

/// Polynomial form of [`eval_byte_pinned`].
pub fn build_byte_pinned_poly(
    col_coeffs: &[Vec<Scalar>],
    byte_col: usize,
    constant: u64,
    is_real_col: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let const_poly = vec![Scalar::from_u64(constant, curve)];
    let body = poly_sub(&col_coeffs[byte_col], &const_poly, curve);
    poly_mul(&col_coeffs[is_real_col], &body, curve)
}

/// Polynomial form of [`eval_byte_slice_binding`].
pub fn build_byte_slice_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    rlp_byte_offset: usize,
    rlp_offset: usize,
    decoded_offset: usize,
    n: usize,
    is_real_col: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for k in 0..n {
        let body = poly_sub(
            &col_coeffs[rlp_byte_offset + rlp_offset + k],
            &col_coeffs[decoded_offset + k],
            curve,
        );
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[is_real_col], &acc, curve)
}

/// Polynomial form of [`eval_zero_tail`].
pub fn build_zero_tail_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    rlp_byte_offset: usize,
    start: usize,
    end: usize,
    is_real_col: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for k in start..end {
        let body = col_coeffs[rlp_byte_offset + k].clone();
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[is_real_col], &acc, curve)
}

/// Polynomial form of [`eval_decoded_zero_tail`].
pub fn build_decoded_zero_tail_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    start_col: usize,
    end_col: usize,
    is_real_col: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for c in start_col..end_col {
        let body = col_coeffs[c].clone();
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[is_real_col], &acc, curve)
}
