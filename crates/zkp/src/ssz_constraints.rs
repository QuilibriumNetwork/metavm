//! [`VmConstraintSystem`] wiring for the SSZ merkleization AIR.
//!
//! This module adapts the SSZ tree-structure AIR defined in
//! [`crate::ssz_air`] into the shape required by the generic
//! [`prove_with_scheme`] / [`verify_with_scheme`] pipeline. It provides:
//!
//! - `SszConstraintSystem`: the trait implementer. Construct with
//!   `SszConstraintSystem::new(num_rows)` where `num_rows` is the
//!   real, pre-padding trace height (= `merkleize_witness(...).len()`).
//! - `build_trace_polynomials_from_rows`: helper that lifts a
//!   `Vec<MerkleizeRow>` into the `TracePolynomials` the pipeline expects.
//!
//! # Constraint layout
//!
//! 4 row-local consolidated categories (label → index):
//!
//!   0. `is_left_real_binary`            — `is_left_real · (is_left_real − 1) = 0`
//!   1. `is_right_real_binary`           — `is_right_real · (is_right_real − 1) = 0`
//!   2. `byte_validity_lookup`           — placeholder (always zero); the actual
//!      per-byte range check is enforced by the lookup declarations below.
//!   3. `is_right_real_implies_left_real` — `is_right_real · (is_left_real − 1) = 0`.
//!      Pins the `(is_left_real, is_right_real)` pair to the three valid
//!      combinations `{(0,0), (1,0), (1,1)}` — `merkleize_witness` only ever
//!      emits real rows with `is_left_real = 1`, so the (0,1) configuration
//!      is invalid.
//!
//! 1 shifted (cross-row) constraint: `layer_depth_increment_01`.
//!   The witness produced by [`merkleize_witness`] emits rows in layer order:
//!   within a layer the depth stays the same, between layers it increments
//!   by exactly 1. So `δ := layer_depth(ω·X) − layer_depth(X)` lies in
//!   {0, 1} on every real row, encoded as `δ · (δ − 1) = 0`.
//!
//! # Padding strategy
//!
//! [`VmConstraintSystem::padding_selector_column`] returns `None`. Padding
//! rows carry all-zero columns: every row-local body trivially vanishes
//! (`0 · (0 − 1) = 0` for each binary check). The cross-row body uses
//! boundary exclusion to skip the last real row (where the next row's
//! `layer_depth = 0` would yield a negative δ) and the domain wrap.
//!
//! # Soundness gaps (deferred)
//!
//! This is the **structural** AIR; several semantic constraints are stubbed
//! out and listed here so reviewers see them at a glance:
//!
//! 1. **Per-row hash check** `parent == sha256_pair(left, right)` — out of
//!    scope; delegated to a separate `sha256_constraints` AIR via cross-AIR
//!    linkage (future task).
//! 2. **Zero-padding** when `is_right_real == 0`, `right` must equal
//!    `zero_hash(layer_depth)`. Requires a `(depth, byte_idx, value)`
//!    lookup table; pending. Host-side check: [`rows_zero_padding_consistent`].
//! 3. **Chain consistency**: parents at depth `d` reappear as inputs of
//!    rows at depth `d+1`. Requires a permutation argument over the parent
//!    column versus the (left, right) columns of the next layer; pending.
//!    Host-side check: [`rows_chain_consistent`].
//! 4. **Position upper bound** `position_in_layer < 2^layer_depth` —
//!    relaxed to `position_in_layer < 2^64` (declared as a 64-bit range
//!    check); the per-depth bound would need a depth-keyed lookup table.
//!
//! # Lookup declarations
//!
//! - Each of the 96 chunk byte columns (`left`, `right`, `parent` × 32) is
//!   declared as an 8-bit range check.
//! - `LAYER_DEPTH` is declared as an 8-bit range check (beacon-chain merkle
//!   trees never exceed depth ≈ 40).
//! - `POSITION_IN_LAYER` is declared as a 64-bit range check (relaxed from
//!   the tighter `< 2^layer_depth` bound).

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use crate::ssz::Chunk;
use crate::ssz_air::{col, merkleize_witness, MerkleizeRow, CHUNK_BYTES};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Number of consolidated row-local constraint categories.
/// Original 4 + 4 new for IS_LEAF_DEPTH / leaf-only validator selectors:
///   4. `is_leaf_depth_binary` — IS_LEAF_DEPTH · (IS_LEAF_DEPTH − 1) = 0
///   5. `is_leaf_depth_zeroes_at_higher_depth` — LAYER_DEPTH · IS_LEAF_DEPTH = 0
///   6. `is_left_validator_leaf_binding` —
///      IS_LEFT_VALIDATOR_LEAF − IS_LEFT_REAL · IS_LEAF_DEPTH = 0
///   7. `is_right_validator_leaf_binding` — same on RIGHT side.
pub const NUM_ROW_CONSTRAINTS: usize = 8;

/// Number of consolidated cross-row (shifted) constraints.
pub const NUM_SHIFTED: usize = 1;

// ──── Constraint system ────────────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the SSZ merkleization AIR.
pub struct SszConstraintSystem {
    /// Number of real trace rows (before padding). The padded domain size
    /// is determined by [`TracePolynomials`] to the next power of two ≥
    /// `num_rows`.
    pub num_rows: usize,
    /// The domain generator ω for the trace's padded domain. When `Some`,
    /// the verifier-side `evaluate_shifted_at_point` excludes the last real
    /// row (`X − ω^{num_rows − 1}`) in addition to the wrap row
    /// (`X − ω^{n−1}`). When `None`, only the wrap-around factor is
    /// excluded — sound only when `num_rows == domain_size`.
    pub omega: Option<Scalar>,
    /// The padded domain size (power of two ≥ `num_rows`). Used together
    /// with `omega` to build the boundary-row exclusion product.
    pub domain_size: Option<u64>,
}

impl SszConstraintSystem {
    /// Construct an SSZ constraint system for a trace of `num_rows` real
    /// rows (= `merkleize_witness(...).len()`).
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    /// Attach the domain generator `omega` and `domain_size` for the
    /// scheme/trace this constraint system is paired with. The verifier
    /// needs these to replicate the prover's boundary exclusion product
    /// `Π_{r ∈ boundary_rows} (z − ω^r)`.
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

// ──── Trace construction helpers ───────────────────────────────────────

/// Allocate a zeroed scalar trace with `padded_size` rows and
/// `col::NUM_COLUMNS` columns, targeting `curve`.
fn alloc_trace(padded_size: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..col::NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded_size])
        .collect()
}

/// Populate a scalar trace from a `MerkleizeRow` slice. Mirrors
/// [`populate_trace`] but writes directly into the field rather than into
/// a `u64` staging buffer (saves one Scalar::from_u64 per cell).
fn populate_scalar_trace(rows: &[MerkleizeRow], columns: &mut [Vec<Scalar>], curve: CurveType) {
    debug_assert!(columns.len() >= col::NUM_COLUMNS);
    for (row_idx, row) in rows.iter().enumerate() {
        for i in 0..CHUNK_BYTES {
            columns[col::LEFT_OFFSET + i][row_idx] = Scalar::from_u64(row.left[i] as u64, curve);
            columns[col::RIGHT_OFFSET + i][row_idx] = Scalar::from_u64(row.right[i] as u64, curve);
            columns[col::PARENT_OFFSET + i][row_idx] = Scalar::from_u64(row.parent[i] as u64, curve);
        }
        columns[col::LAYER_DEPTH][row_idx] = Scalar::from_u64(row.layer_depth as u64, curve);
        columns[col::POSITION_IN_LAYER][row_idx] = Scalar::from_u64(row.position_in_layer as u64, curve);
        columns[col::IS_LEFT_REAL][row_idx] = Scalar::from_u64(row.is_left_real as u64, curve);
        columns[col::IS_RIGHT_REAL][row_idx] = Scalar::from_u64(row.is_right_real as u64, curve);
        let is_leaf_depth = (row.layer_depth == 0) as u64;
        columns[col::IS_LEAF_DEPTH][row_idx] = Scalar::from_u64(is_leaf_depth, curve);
        columns[col::IS_LEFT_VALIDATOR_LEAF][row_idx] =
            Scalar::from_u64(is_leaf_depth * (row.is_left_real as u64), curve);
        columns[col::IS_RIGHT_VALIDATOR_LEAF][row_idx] =
            Scalar::from_u64(is_leaf_depth * (row.is_right_real as u64), curve);
    }
}

/// Build a [`TracePolynomials`] wrapping the SSZ merkleization trace
/// produced by populating `rows` in order. Each row populates exactly one
/// row of the trace; the trace pads up to the next power of two with
/// zeros (consistent with the "no selector on padding" strategy).
pub fn build_trace_polynomials_from_rows(
    rows: &[MerkleizeRow],
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let mut columns = alloc_trace(padded, curve);
    populate_scalar_trace(rows, &mut columns, curve);
    into_trace_polynomials(columns, num_rows, padded, curve)
}

/// Variant for direct chunk inputs: builds the witness then lifts it.
pub fn build_trace_polynomials_from_chunks(
    chunks: &[Chunk],
    limit: Option<u64>,
    curve: CurveType,
) -> TracePolynomials {
    let rows = merkleize_witness(chunks, limit);
    build_trace_polynomials_from_rows(&rows, curve)
}

fn into_trace_polynomials(
    columns: Vec<Vec<Scalar>>,
    num_rows: usize,
    padded: usize,
    curve: CurveType,
) -> TracePolynomials {
    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial {
            evaluations: evals,
            degree: num_rows,
        })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ──── Helpers for scalar-point evaluation ──────────────────────────────

/// Fast-exponentiate a scalar by a `u64` exponent.
fn scalar_pow(base: &Scalar, exp: u64) -> Scalar {
    let mut result = Scalar::one(base.curve_type());
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&b);
        }
        b = b.mul(&b);
        e >>= 1;
    }
    result
}

// ──── Scalar-point evaluation of each category body ────────────────────

/// 1. `is_left_real_binary`: `is_left_real · (is_left_real − 1) = 0`.
fn eval_is_left_real_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_LEFT_REAL];
    v.mul(&v.sub(&one))
}

/// 2. `is_right_real_binary`: `is_right_real · (is_right_real − 1) = 0`.
fn eval_is_right_real_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_RIGHT_REAL];
    v.mul(&v.sub(&one))
}

/// 3. `byte_validity_lookup`: the algebraic body is always zero — the
/// actual per-byte range check is enforced by the lookup declarations
/// returned from [`SszConstraintSystem::lookup_declarations`].
fn eval_byte_validity_lookup_at_point(cols: &[Scalar]) -> Scalar {
    Scalar::zero(cols[0].curve_type())
}

/// 4. `is_right_real_implies_left_real`: `is_right_real · (is_left_real − 1) = 0`.
/// Pins the (is_left_real, is_right_real) pair to {(0,0), (1,0), (1,1)} —
/// rejecting (is_left_real=0, is_right_real=1), which is the only
/// invalid combination per [`merkleize_witness`]'s output (which always
/// emits `is_left_real = true` on real rows; padding rows are all zero).
/// On padding rows: `0 · (0 − 1) = 0`. On real rows with `is_left_real = 1`:
/// `is_right_real · 0 = 0`. The forged combination (0,1) yields `1·(0−1) = −1`.
fn eval_is_right_real_implies_left_real_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let l = &cols[col::IS_LEFT_REAL];
    let r = &cols[col::IS_RIGHT_REAL];
    r.mul(&l.sub(&one))
}

// ──── Polynomial-form builders for each category body ──────────────────

fn build_is_left_real_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_LEFT_REAL];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_right_real_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_RIGHT_REAL];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_byte_validity_lookup_poly(curve: CurveType) -> Vec<Scalar> {
    vec![Scalar::zero(curve)]
}

fn build_is_right_real_implies_left_real_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let l = &cols[col::IS_LEFT_REAL];
    let r = &cols[col::IS_RIGHT_REAL];
    let l_m1 = poly_sub(l, &one_poly, curve);
    poly_mul(r, &l_m1, curve)
}

/// 5. `is_leaf_depth_binary`: `IS_LEAF_DEPTH · (IS_LEAF_DEPTH − 1) = 0`.
fn eval_is_leaf_depth_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_LEAF_DEPTH];
    v.mul(&v.sub(&one))
}

fn build_is_leaf_depth_binary_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_LEAF_DEPTH];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

/// 6. `is_leaf_depth_zeroes_at_higher_depth`:
/// `LAYER_DEPTH · IS_LEAF_DEPTH = 0`. Forces `IS_LEAF_DEPTH = 0`
/// whenever `LAYER_DEPTH > 0`. Combined with the multiset-equality
/// closure in the VE↔SSZ linkage, this pins the pair
/// `(LAYER_DEPTH = 0) ↔ (IS_LEAF_DEPTH = 1)` end-to-end.
fn eval_is_leaf_depth_zeroes_at_higher_depth_at_point(cols: &[Scalar]) -> Scalar {
    let depth = &cols[col::LAYER_DEPTH];
    let leaf = &cols[col::IS_LEAF_DEPTH];
    depth.mul(leaf)
}

fn build_is_leaf_depth_zeroes_at_higher_depth_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    poly_mul(&cols[col::LAYER_DEPTH], &cols[col::IS_LEAF_DEPTH], curve)
}

/// 7. `is_left_validator_leaf_binding`:
/// `IS_LEFT_VALIDATOR_LEAF − IS_LEFT_REAL · IS_LEAF_DEPTH = 0`.
fn eval_is_left_validator_leaf_binding_at_point(cols: &[Scalar]) -> Scalar {
    let left_leaf = &cols[col::IS_LEFT_VALIDATOR_LEAF];
    let l = &cols[col::IS_LEFT_REAL];
    let leaf = &cols[col::IS_LEAF_DEPTH];
    left_leaf.sub(&l.mul(leaf))
}

fn build_is_left_validator_leaf_binding_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let product = poly_mul(&cols[col::IS_LEFT_REAL], &cols[col::IS_LEAF_DEPTH], curve);
    poly_sub(&cols[col::IS_LEFT_VALIDATOR_LEAF], &product, curve)
}

/// 8. `is_right_validator_leaf_binding`:
/// `IS_RIGHT_VALIDATOR_LEAF − IS_RIGHT_REAL · IS_LEAF_DEPTH = 0`.
fn eval_is_right_validator_leaf_binding_at_point(cols: &[Scalar]) -> Scalar {
    let right_leaf = &cols[col::IS_RIGHT_VALIDATOR_LEAF];
    let r = &cols[col::IS_RIGHT_REAL];
    let leaf = &cols[col::IS_LEAF_DEPTH];
    right_leaf.sub(&r.mul(leaf))
}

fn build_is_right_validator_leaf_binding_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let product = poly_mul(&cols[col::IS_RIGHT_REAL], &cols[col::IS_LEAF_DEPTH], curve);
    poly_sub(&cols[col::IS_RIGHT_VALIDATOR_LEAF], &product, curve)
}

// ──── Cross-row helpers ────────────────────────────────────────────────

/// Boundary rows whose cross-row transition must be excluded from
/// vanishing. Two boundaries:
///   - `num_rows − 1` (last real row — next row is the first padding row
///     where `layer_depth = 0`, which would yield negative δ).
///   - `domain_size − 1` (domain wrap-around).
///
/// Returns a sorted, deduplicated list (modulo `domain_size`).
fn boundary_rows(num_rows: usize, domain_size: usize) -> Vec<usize> {
    let mut set: std::collections::BTreeSet<usize> = Default::default();
    if num_rows > 0 {
        set.insert(num_rows - 1);
    }
    if domain_size > 0 {
        set.insert(domain_size - 1);
    }
    set.into_iter().collect()
}

// ──── VmConstraintSystem implementation ────────────────────────────────

impl VmConstraintSystem for SszConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_left_real_binary".into(),
            "is_right_real_binary".into(),
            "byte_validity_lookup".into(),
            "is_right_real_implies_left_real".into(),
            "is_leaf_depth_binary".into(),
            "is_leaf_depth_zeroes_at_higher_depth".into(),
            "is_left_validator_leaf_binding".into(),
            "is_right_validator_leaf_binding".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(
            columns.len() >= col::NUM_COLUMNS,
            "ssz AIR expects at least {} columns",
            col::NUM_COLUMNS
        );
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        let n = columns[0].len();

        let mut is_left_evals = vec![zero.clone(); n];
        let mut is_right_evals = vec![zero.clone(); n];
        let byte_validity_evals = vec![zero.clone(); n];
        let mut implies_evals = vec![zero.clone(); n];
        let mut leaf_bin = vec![zero.clone(); n];
        let mut leaf_zero_higher = vec![zero.clone(); n];
        let mut left_leaf_bind = vec![zero.clone(); n];
        let mut right_leaf_bind = vec![zero.clone(); n];
        for row in 0..n {
            let l = &columns[col::IS_LEFT_REAL][row];
            is_left_evals[row] = l.mul(&l.sub(&one));
            let r = &columns[col::IS_RIGHT_REAL][row];
            is_right_evals[row] = r.mul(&r.sub(&one));
            // is_right_real · (is_left_real − 1) = 0
            implies_evals[row] = r.mul(&l.sub(&one));

            let leaf = &columns[col::IS_LEAF_DEPTH][row];
            let depth = &columns[col::LAYER_DEPTH][row];
            let left_leaf = &columns[col::IS_LEFT_VALIDATOR_LEAF][row];
            let right_leaf = &columns[col::IS_RIGHT_VALIDATOR_LEAF][row];
            // IS_LEAF_DEPTH binarity.
            leaf_bin[row] = leaf.mul(&leaf.sub(&one));
            // LAYER_DEPTH · IS_LEAF_DEPTH = 0 (forces leaf=0 when depth>0).
            leaf_zero_higher[row] = depth.mul(leaf);
            // IS_LEFT_VALIDATOR_LEAF − IS_LEFT_REAL · IS_LEAF_DEPTH = 0
            left_leaf_bind[row] = left_leaf.sub(&l.mul(leaf));
            // IS_RIGHT_VALIDATOR_LEAF − IS_RIGHT_REAL · IS_LEAF_DEPTH = 0
            right_leaf_bind[row] = right_leaf.sub(&r.mul(leaf));
        }
        vec![
            is_left_evals,
            is_right_evals,
            byte_validity_evals,
            implies_evals,
            leaf_bin,
            leaf_zero_higher,
            left_leaf_bind,
            right_leaf_bind,
        ]
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < col::NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            eval_is_left_real_binary_at_point(col_evals_at_z),
            eval_is_right_real_binary_at_point(col_evals_at_z),
            eval_byte_validity_lookup_at_point(col_evals_at_z),
            eval_is_right_real_implies_left_real_at_point(col_evals_at_z),
            eval_is_leaf_depth_binary_at_point(col_evals_at_z),
            eval_is_leaf_depth_zeroes_at_higher_depth_at_point(col_evals_at_z),
            eval_is_left_validator_leaf_binding_at_point(col_evals_at_z),
            eval_is_right_validator_leaf_binding_at_point(col_evals_at_z),
        ];
        let curve = alpha.curve_type();
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        // No selectors in this AIR — the SSZ structural constraints are
        // ungated row-local checks. Returning empty triggers the
        // `evaluate_on_domain + IFFT` codepath in the prover, which is
        // appropriate here.
        Vec::new()
    }

    /// Return `None`: padding rows carry all-zero data columns, which
    /// makes both binary checks vanish (`0·(0−1) = 0`).
    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        // Defensive: zero every column on padding rows.
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < col::NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col_v in columns.iter_mut().take(col::NUM_COLUMNS) {
            for cell in col_v.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let bodies: Vec<Vec<Scalar>> = vec![
            build_is_left_real_binary_poly(column_coeffs, curve),
            build_is_right_real_binary_poly(column_coeffs, curve),
            build_byte_validity_lookup_poly(curve),
            build_is_right_real_implies_left_real_poly(column_coeffs, curve),
            build_is_leaf_depth_binary_poly(column_coeffs, curve),
            build_is_leaf_depth_zeroes_at_higher_depth_poly(column_coeffs, curve),
            build_is_left_validator_leaf_binding_poly(column_coeffs, curve),
            build_is_right_validator_leaf_binding_poly(column_coeffs, curve),
        ];
        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            let scaled = poly_scalar_mul(body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    // ── Cross-row support ──────────────────────────────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Only `LAYER_DEPTH` is referenced shifted (at ω·X) for the
        // monotone-step constraint.
        vec![col::LAYER_DEPTH]
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        if shifted_evals.is_empty() || col_evals_at_z.len() < col::NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        // δ := layer_depth(ω·z) − layer_depth(z); body = δ · (δ − 1)
        let depth_curr = &col_evals_at_z[col::LAYER_DEPTH];
        let depth_next = &shifted_evals[0];
        let delta = depth_next.sub(depth_curr);
        let body = delta.mul(&delta.sub(&one));

        // α^alpha_offset.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // Boundary-row exclusion product Π_{r ∈ boundary_rows} (z − ω^r).
        //
        // We derive ω from `omega_n_minus_1` rather than `self.omega`:
        // the actual proof domain may be larger than the trace's natural
        // padded size when downstream machinery (LogUp byte tables)
        // inflates it, and `self.omega` would be stale in that case.
        // ω · ω^(n-1) = ω^n = 1 ⟹ ω = inverse(ω^(n-1)).
        let exclusion = if self.num_rows == 0 {
            // Degenerate: no real rows. Fall back to wrap-only exclusion.
            z.sub(omega_n_minus_1)
        } else {
            let omega = omega_n_minus_1.inverse();
            // Always exclude num_rows-1 (last real row → first padding row,
            // where layer_depth resets and δ goes negative). Wrap-around
            // is excluded automatically because `omega_n_minus_1` IS ω^(n-1).
            let mut prod = z.sub(omega_n_minus_1);
            if self.num_rows > 0 {
                let omega_r = scalar_pow(&omega, (self.num_rows - 1) as u64);
                let last_real = z.sub(&omega_r);
                // Avoid double-multiplying when num_rows == domain_size.
                if !last_real.sub(&prod).is_zero() {
                    prod = prod.mul(&last_real);
                }
            }
            prod
        };
        ap.mul(&body).mul(&exclusion)
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        // δ(X) := layer_depth(ω·X) − layer_depth(X)
        let depth = &column_coeffs[col::LAYER_DEPTH];
        let depth_shift = poly_shift(depth, omega);
        let delta = poly_sub(&depth_shift, depth, curve);

        // body(X) = δ(X) · (δ(X) − 1)
        let delta_m1 = poly_sub(&delta, &one_poly, curve);
        let body = poly_mul(&delta, &delta_m1, curve);

        // Multiply by (X − ω^r) for every boundary row r.
        let rows = boundary_rows(self.num_rows, domain_size as usize);
        let mut excluded = body;
        for r in &rows {
            let omega_r = scalar_pow(omega, *r as u64);
            excluded = poly_mul_linear(&excluded, &omega_r);
        }

        // α^alpha_offset.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        poly_scalar_mul(&excluded, &ap)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // Tables: 8-bit range and 64-bit range.
        let table8 = LookupTable::range(8);
        let table64 = LookupTable::range(64);
        let tables = vec![table8, table64];
        const TBL_8: usize = 0;
        const TBL_64: usize = 1;

        let mut declarations: Vec<(LookupDeclaration, usize)> = Vec::new();

        // Each chunk byte column (left, right, parent × 32) is in [0, 256).
        for i in 0..CHUNK_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("left_byte_{}_range_8", i),
                    column_index: col::LEFT_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }
        for i in 0..CHUNK_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("right_byte_{}_range_8", i),
                    column_index: col::RIGHT_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }
        for i in 0..CHUNK_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("parent_byte_{}_range_8", i),
                    column_index: col::PARENT_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }

        // Layer depth: ≤ ~64 in beacon-chain merkleization → 8-bit fits.
        declarations.push((
            LookupDeclaration {
                label: "layer_depth_range_8".into(),
                column_index: col::LAYER_DEPTH,
                max_bits: 8,
                selector_column: None,
            },
            TBL_8,
        ));

        // Position-in-layer: relaxed to 64-bit (tighter `< 2^layer_depth`
        // bound is a deferred soundness improvement).
        declarations.push((
            LookupDeclaration {
                label: "position_in_layer_range_64".into(),
                column_index: col::POSITION_IN_LAYER,
                max_bits: 64,
                selector_column: None,
            },
            TBL_64,
        ));

        LookupRequirements { tables, declarations }
    }
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;
    use crate::ssz::ZERO_CHUNK;
    use crate::ssz_air::populate_trace;

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    fn make_chunk(byte: u8) -> Chunk {
        let mut c = ZERO_CHUNK;
        c[0] = byte;
        c
    }

    /// Build the same four-leaf merkleize trace shared with the AIR
    /// tests. Returns `(rows, padded scalar columns)`.
    fn four_leaf_columns() -> (Vec<MerkleizeRow>, Vec<Vec<Scalar>>) {
        let a = make_chunk(0x11);
        let b = make_chunk(0x22);
        let c = make_chunk(0x33);
        let d = make_chunk(0x44);
        let rows = merkleize_witness(&[a, b, c, d], None);
        let padded = crate::trace::nearest_power_of_two(rows.len().max(1));
        let mut columns = alloc_trace(padded, CurveType::Bls48581);
        populate_scalar_trace(&rows, &mut columns, CurveType::Bls48581);
        (rows, columns)
    }

    #[test]
    fn ssz_cs_labels_and_counts() {
        let (rows, _cols) = four_leaf_columns();
        let cs = SszConstraintSystem::new(rows.len());
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_constraints(), 8);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), 1);
        assert!(cs.selector_column_indices().is_empty());
        assert_eq!(cs.shifted_column_indices(), vec![col::LAYER_DEPTH]);
        assert!(cs.padding_selector_column().is_none());
    }

    #[test]
    fn ssz_cs_evaluate_on_domain_matches_witness() {
        let (rows, columns) = four_leaf_columns();
        let cs = SszConstraintSystem::new(rows.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, vec_) in evals.iter().enumerate() {
            for (row, v) in vec_.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {} on valid witness",
                    k,
                    cs.constraint_labels()[k],
                    row
                );
            }
        }
    }

    #[test]
    fn ssz_cs_evaluate_at_point_zero_on_real_rows() {
        let (rows, columns) = four_leaf_columns();
        let cs = SszConstraintSystem::new(rows.len());
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);
        for row in 0..rows.len() {
            let col_vals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let c_at_row = cs.evaluate_at_point(&col_vals, &alpha);
            assert!(
                c_at_row.is_zero(),
                "combined constraint C(row {}) nonzero on valid witness",
                row
            );
        }
    }

    #[test]
    fn ssz_cs_evaluate_at_point_zero_on_all_zero_row() {
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let col_vals = vec![zero.clone(); col::NUM_COLUMNS];
        let cs = SszConstraintSystem::new(8);
        let alpha = Scalar::from_u64(23, curve);
        let c_at = cs.evaluate_at_point(&col_vals, &alpha);
        assert!(
            c_at.is_zero(),
            "all-zero padding row must evaluate to zero (otherwise \
             padding_selector_column must be set)"
        );
    }

    #[test]
    fn ssz_cs_constraints_reject_tampered_is_left_real() {
        let (rows, mut columns) = four_leaf_columns();
        // Set is_left_real = 2 on row 0 — breaks the binary constraint
        // (2·(2−1) = 2 ≠ 0).
        let curve = CurveType::Bls48581;
        columns[col::IS_LEFT_REAL][0] = Scalar::from_u64(2, curve);

        let cs = SszConstraintSystem::new(rows.len());
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered IS_LEFT_REAL should make C(row 0) non-zero"
        );

        // Domain-form check should also fire.
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(!evals[0][0].is_zero(), "is_left_real_binary must fire on row 0");
    }

    #[test]
    fn ssz_cs_constraints_reject_tampered_is_right_real() {
        let (rows, mut columns) = four_leaf_columns();
        let curve = CurveType::Bls48581;
        // Set is_right_real = 5 on row 0 → 5·4 = 20 ≠ 0.
        columns[col::IS_RIGHT_REAL][0] = Scalar::from_u64(5, curve);
        let cs = SszConstraintSystem::new(rows.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(!evals[1][0].is_zero(), "is_right_real_binary must fire on row 0");
    }

    #[test]
    fn ssz_cs_constraints_reject_invalid_real_flag_combination() {
        // Forge (is_left_real, is_right_real) = (0, 1) on row 0.
        // The binary checks pass (both are 0/1) but the new
        // `is_right_real_implies_left_real` constraint fires:
        // 1 · (0 − 1) = −1 ≠ 0.
        let (rows, mut columns) = four_leaf_columns();
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        columns[col::IS_LEFT_REAL][0] = zero.clone();
        columns[col::IS_RIGHT_REAL][0] = one.clone();

        let cs = SszConstraintSystem::new(rows.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        // Binary checks must remain satisfied (both flags are 0 or 1).
        assert!(evals[0][0].is_zero(), "is_left_real_binary stays satisfied for 0/1 flag");
        assert!(evals[1][0].is_zero(), "is_right_real_binary stays satisfied for 0/1 flag");
        // The new implication constraint at index 3 must fire.
        assert!(
            !evals[3][0].is_zero(),
            "is_right_real_implies_left_real must fire on (left=0, right=1) — \
             only this constraint catches the forgery"
        );
    }

    #[test]
    fn ssz_cs_constraints_allow_valid_real_flag_combinations() {
        // The three valid (is_left_real, is_right_real) pairs are
        // {(0,0), (1,0), (1,1)}. All must satisfy every row-local check
        // (binary, byte_validity, implication).
        let curve = CurveType::Bls48581;
        let cs = SszConstraintSystem::new(1);
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        let alpha = Scalar::from_u64(13, curve);

        for (left, right) in [
            (zero.clone(), zero.clone()),
            (one.clone(), zero.clone()),
            (one.clone(), one.clone()),
        ] {
            let mut row_vals = vec![zero.clone(); col::NUM_COLUMNS];
            row_vals[col::IS_LEFT_REAL] = left.clone();
            row_vals[col::IS_RIGHT_REAL] = right.clone();
            let c_at = cs.evaluate_at_point(&row_vals, &alpha);
            assert!(
                c_at.is_zero(),
                "valid flag pair (left={:?}, right={:?}) must satisfy all row-local constraints",
                left.to_bytes(),
                right.to_bytes()
            );
        }
    }

    #[test]
    fn ssz_cs_lookup_declarations_are_well_formed() {
        let cs = SszConstraintSystem::new(4);
        let reqs = cs.lookup_declarations();

        // Two distinct tables: 8-bit and 64-bit range.
        assert_eq!(reqs.tables.len(), 2);
        assert_eq!(reqs.tables[0].bits, 8);
        assert_eq!(reqs.tables[1].bits, 64);

        // Declarations: 96 byte columns + 1 layer_depth (8-bit) + 1
        // position_in_layer (64-bit) = 98 declarations total.
        assert_eq!(reqs.declarations.len(), 96 + 1 + 1);

        // Every declaration's column index must be in range.
        for (decl, table_idx) in &reqs.declarations {
            assert!(
                decl.column_index < col::NUM_COLUMNS,
                "declaration {} references out-of-range column {}",
                decl.label,
                decl.column_index
            );
            assert!(*table_idx < reqs.tables.len());
        }

        // The 64-bit declaration is the position column.
        let pos_decl = reqs
            .declarations
            .iter()
            .find(|(d, t)| *t == 1 && d.column_index == col::POSITION_IN_LAYER)
            .expect("position_in_layer must use the 64-bit table");
        assert_eq!(pos_decl.0.max_bits, 64);
    }

    #[test]
    fn ssz_cs_boundary_rows_basic() {
        // num_rows = 3, domain_size = 16 → boundaries {2, 15}.
        let br = boundary_rows(3, 16);
        assert_eq!(br, vec![2, 15]);
    }

    #[test]
    fn ssz_cs_cross_row_layer_depth_monotone_holds_on_witness() {
        // Use the four-leaf trace: row 0 depth=0, row 1 depth=0, row 2
        // depth=1. δ values are {0, 1, …}. Both fall in {0, 1}.
        let (rows, columns) = four_leaf_columns();
        let curve = CurveType::Bls48581;
        let one = Scalar::one(curve);
        for r in 0..rows.len().saturating_sub(1) {
            let d_curr = &columns[col::LAYER_DEPTH][r];
            let d_next = &columns[col::LAYER_DEPTH][r + 1];
            let delta = d_next.sub(d_curr);
            let body = delta.mul(&delta.sub(&one));
            assert!(
                body.is_zero(),
                "cross-row monotone failed between rows {} and {} (δ outside {{0,1}})",
                r,
                r + 1
            );
        }
    }

    /// Marked `#[ignore]`: full prove/verify roundtrip is expensive (KZG
    /// commitments + schoolbook builds across 100 columns under
    /// BLS48-581). Run manually with:
    ///     cargo test --release -p metavm-zkp --lib \
    ///         ssz_cs_prove_verify_small -- --ignored --nocapture
    #[test]
    #[ignore = "slow: full prover roundtrip; run with --release --ignored"]
    fn ssz_cs_prove_verify_small() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let a = make_chunk(0x11);
        let b = make_chunk(0x22);
        let c = make_chunk(0x33);
        let d = make_chunk(0x44);
        let rows = merkleize_witness(&[a, b, c, d], None);
        let trace = build_trace_polynomials_from_rows(&rows, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = SszConstraintSystem::new(rows.len())
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "ssz four-leaf proof must verify");
    }

    /// End-to-end: produce a real SSZ ExecutionProof, serialize it, attach
    /// to the Finality layer (validator-registry merkleization is SSZ-shaped),
    /// and verify through `LayerChainProof::verify_with_layer_verifier`.
    /// Validates the LayerProofKind::Ssz dispatch path.
    #[test]
    #[ignore = "slow: produces a real SSZ proof; run with --release --ignored"]
    fn ssz_proof_flows_through_layer_chain_envelope() {
        use crate::layer_chain::{
            ChainBoundaries, LayerChainProof, LayerProof, LayerProofKind,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 1. Generate a real SSZ proof on a 4-leaf merkleization.
        let a = make_chunk(0xA0);
        let b = make_chunk(0xB0);
        let c = make_chunk(0xC0);
        let d = make_chunk(0xD0);
        let rows = merkleize_witness(&[a, b, c, d], None);
        let trace = build_trace_polynomials_from_rows(&rows, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = SszConstraintSystem::new(rows.len())
            .with_omega_and_domain(omega.clone(), domain_size);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        assert!(crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve));

        // 2. Serialize.
        let proof_bytes = proof.to_bytes();
        assert!(!proof_bytes.is_empty());

        // 3. Build a chain proof with the SSZ proof attached at the Finality
        //    layer (validator-registry merkleization).
        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: 1,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = crate::layer_chain::LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 3 {
                    // Finality layer carries the real SSZ proof.
                    LayerProof::with_proof(claim, LayerProofKind::Ssz, proof_bytes.clone())
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // 4. Closure-based dispatch: kind=Ssz triggers SSZ verification.
        let num_rows = rows.len();
        let result = chain_proof.verify_with_layer_verifier(|layer| match layer.kind {
            LayerProofKind::Ssz => {
                let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                    .map_err(|e| format!("decode failed: {:?}", e))?;
                let cs = SszConstraintSystem::new(num_rows)
                    .with_omega_and_domain(omega.clone(), domain_size);
                if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                    Ok(())
                } else {
                    Err("Ssz proof did not verify".to_string())
                }
            }
            LayerProofKind::ReferenceOnly => Ok(()),
            other => Err(format!("unsupported layer kind {}", other.as_str())),
        });
        assert_eq!(
            result,
            Ok(()),
            "real SSZ proof must verify through the LayerChainProof envelope",
        );
    }

    /// Use [`populate_trace`] (the public AIR populator) to populate via
    /// the `u64` staging buffer, then convert to scalars; sanity that
    /// both populators agree.
    #[test]
    fn ssz_cs_scalar_populator_matches_u64_populator() {
        let a = make_chunk(0xAA);
        let b = make_chunk(0xBB);
        let c = make_chunk(0xCC);
        let rows = merkleize_witness(&[a, b, c], None);
        let n = rows.len();

        let mut u64_cols: Vec<Vec<u64>> =
            (0..col::NUM_COLUMNS).map(|_| vec![0u64; n]).collect();
        populate_trace(&rows, &mut u64_cols);

        let curve = CurveType::Bls48581;
        let mut scalar_cols = alloc_trace(n, curve);
        populate_scalar_trace(&rows, &mut scalar_cols, curve);

        for c_idx in 0..col::NUM_COLUMNS {
            for r in 0..n {
                let expected = Scalar::from_u64(u64_cols[c_idx][r], curve);
                let diff = scalar_cols[c_idx][r].sub(&expected);
                assert!(
                    diff.is_zero(),
                    "scalar/u64 populator disagree at col {} row {}",
                    c_idx, r
                );
            }
        }
    }
}
