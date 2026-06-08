//! Multi-row precompile I/O chunking AIR.
//!
//! # Purpose
//!
//! Several Ethereum precompiles take input lengths that exceed the
//! [`CHUNK_BYTES`] bound the existing single-row precompile AIRs
//! (`ripemd160_precompile_air`, `blake2f_precompile_air`, …) use to
//! commit their input bytes. For example:
//!
//!   * KZG point evaluation (`0x0a`): **192 bytes** = 3 × 64-byte
//!     chunks (versioned_hash || z || y || commitment || proof).
//!   * ECRECOVER (`0x01`): **128 bytes** = 2 × 64-byte chunks (hash ||
//!     v || r || s).
//!   * Identity (`0x04`) / SHA-256 (`0x02`) / MODEXP (`0x05`): arbitrary
//!     length.
//!
//! This AIR provides the **canonical multi-row chunked I/O surface**:
//! one row per 64-byte chunk, with row-locked cross-row continuity
//! constraints and a wired descriptor binding each chunked row to one
//! row of the existing [`MAX_INPUT_LENGTH`]-shaped precompile I/O
//! surface for backward compatibility.
//!
//! # Witness shape (per row)
//!
//!   * `chunk_index` (u8 effectively, but committed as a u64 for the
//!     LE-byte decomposition).
//!   * `chunk_bytes[0..64]` — the 64 bytes of this chunk (zero-padded
//!     when the precompile input length is not a multiple of 64).
//!   * `total_length` (u64) — invocation-wide input length. Constant
//!     across every row of the same invocation (algebraically enforced
//!     by a shifted constancy body).
//!   * `is_last_chunk` (binary) — marks the terminal row.
//!   * `is_real` (binary) — selector.
//!
//! # Algebraic constraints (≥ 5 row-local + 4 shifted)
//!
//! ## Row-local
//!
//!   0. `is_real_binary`            — `IS_REAL · (IS_REAL − 1) = 0`.
//!   1. `is_last_binary`            — `IS_LAST · (IS_LAST − 1) = 0`.
//!   2. `is_last_implies_real`      — `IS_LAST · (1 − IS_REAL) = 0`.
//!   3. `chunk_index_le_decomp`     — LE byte decomposition of CHUNK_INDEX.
//!   4. `total_length_le_decomp`    — LE byte decomposition of TOTAL_LENGTH.
//!
//! ## Cross-row (shifted)
//!
//!   0. `chunk_index_inc` —
//!      `IS_REAL(X) · IS_REAL(ωX) · (CHUNK_INDEX(ωX) − CHUNK_INDEX(X) − 1) = 0`.
//!   1. `total_length_const` —
//!      `IS_REAL(X) · IS_REAL(ωX) · (TOTAL_LENGTH(ωX) − TOTAL_LENGTH(X)) = 0`.
//!   2. `is_last_terminates` —
//!      `IS_REAL(X) · IS_LAST(X) · IS_REAL(ωX) = 0` — nothing real
//!      after the last chunk.
//!
//! Both shifted bodies are multiplied by `(X − ω^{n−1})` to exclude
//! the wrap-around row.
//!
//! # Cross-AIR LogUp descriptor
//!
//! [`make_chunked_to_precompile_io_descriptor`] binds each chunked row
//! `(chunk_index, chunk_bytes[0..64])` to one row of the **existing**
//! single-row precompile I/O column layout. The B-side column indices
//! are placeholder sentinels until a dedicated `precompile_io_air`
//! lands; in the meantime the descriptor's structure pins the
//! tuple-alignment contract.
//!
//! # Backward compatibility
//!
//! A 1-chunk invocation (input length ≤ 64) reduces to a single row
//! whose `chunk_bytes[0..64]` column block is exactly the existing
//! `MAX_INPUT_LENGTH = 64` shape that
//! [`crate::ripemd160_precompile_air`] et al. use. The chunked AIR is
//! a strict generalization: every existing precompile I/O witness can
//! be lifted into a 1-row chunked witness with no semantic change.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of input bytes per chunk row. Matches the single-row
/// `MAX_INPUT_LENGTH` constant in existing precompile AIRs (e.g.
/// [`crate::ripemd160_precompile_air::MAX_INPUT_LENGTH`]).
pub const CHUNK_BYTES: usize = 64;

/// LE byte width of u64 columns range-checked via the 8-bit table.
pub const U64_BYTES: usize = 8;

/// Maximum chunks any single invocation can span. Chosen so that the
/// composer's largest expected precompile (modular-exponentiation
/// scratch buffer) still fits.
pub const MAX_CHUNKS: usize = 64;

/// Sentinel column index used for the B-side of the
/// `make_chunked_to_precompile_io_descriptor` until a dedicated
/// `precompile_io_air` lands.
pub const PRECOMPILE_IO_PLACEHOLDER: usize = usize::MAX;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_CHUNK_INDEX: usize = 0;
pub const COL_CHUNK_INDEX_BYTE_OFFSET: usize = COL_CHUNK_INDEX + 1; // 1..9

pub const COL_CHUNK_BYTES_OFFSET: usize =
    COL_CHUNK_INDEX_BYTE_OFFSET + U64_BYTES; // 9..73

pub const COL_TOTAL_LENGTH: usize = COL_CHUNK_BYTES_OFFSET + CHUNK_BYTES; // 73
pub const COL_TOTAL_LENGTH_BYTE_OFFSET: usize = COL_TOTAL_LENGTH + 1; // 74..82

pub const COL_IS_LAST: usize = COL_TOTAL_LENGTH_BYTE_OFFSET + U64_BYTES; // 82
pub const COL_IS_REAL: usize = COL_IS_LAST + 1; // 83

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 84

pub const NUM_ROW_CONSTRAINTS: usize = 5;
pub const NUM_SHIFTED: usize = 3;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecompileIoChunkedRow {
    pub chunk_index: u64,
    pub chunk_bytes: [u8; CHUNK_BYTES],
    pub total_length: u64,
    pub is_last_chunk: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrecompileIoChunkedWitness {
    pub rows: Vec<PrecompileIoChunkedRow>,
}

impl PrecompileIoChunkedWitness {
    /// Build a chunked witness from arbitrary input bytes. Input is
    /// split into [`CHUNK_BYTES`]-byte rows; the last row is
    /// zero-padded if `input.len()` is not a multiple of [`CHUNK_BYTES`].
    /// Empty input still produces a single all-zero row with
    /// `is_last_chunk = true` and `total_length = 0`.
    ///
    /// Panics if `chunks > MAX_CHUNKS`.
    pub fn from_input(input: &[u8]) -> Self {
        let total_length = input.len() as u64;
        let num_chunks = if input.is_empty() {
            1
        } else {
            input.len().div_ceil(CHUNK_BYTES)
        };
        assert!(
            num_chunks <= MAX_CHUNKS,
            "chunked precompile input chunks {} exceeds MAX_CHUNKS={}",
            num_chunks,
            MAX_CHUNKS,
        );
        let mut rows = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let mut chunk_bytes = [0u8; CHUNK_BYTES];
            let start = i * CHUNK_BYTES;
            let end = ((i + 1) * CHUNK_BYTES).min(input.len());
            if start < end {
                chunk_bytes[..end - start].copy_from_slice(&input[start..end]);
            }
            rows.push(PrecompileIoChunkedRow {
                chunk_index: i as u64,
                chunk_bytes,
                total_length,
                is_last_chunk: i + 1 == num_chunks,
            });
        }
        Self { rows }
    }

    pub fn from_rows(rows: Vec<PrecompileIoChunkedRow>) -> Self {
        Self { rows }
    }

    pub fn num_chunks(&self) -> usize {
        self.rows.len()
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

fn eval_le_decomp(target: &Scalar, byte_off: usize, col_evals: &[Scalar]) -> Scalar {
    let curve = target.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[byte_off + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target.sub(&sum)
}

fn build_le_decomp_poly(
    target_poly: &[Scalar],
    byte_off: usize,
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[byte_off + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    poly_sub(target_poly, &sum, curve)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &PrecompileIoChunkedWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_CHUNK_INDEX][i] = Scalar::from_u64(row.chunk_index, curve);
        let ci_bytes = row.chunk_index.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_CHUNK_INDEX_BYTE_OFFSET + b][i] =
                Scalar::from_u64(ci_bytes[b] as u64, curve);
        }
        for k in 0..CHUNK_BYTES {
            columns[COL_CHUNK_BYTES_OFFSET + k][i] =
                Scalar::from_u64(row.chunk_bytes[k] as u64, curve);
        }
        columns[COL_TOTAL_LENGTH][i] = Scalar::from_u64(row.total_length, curve);
        let tl_bytes = row.total_length.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_TOTAL_LENGTH_BYTE_OFFSET + b][i] =
                Scalar::from_u64(tl_bytes[b] as u64, curve);
        }
        if row.is_last_chunk {
            columns[COL_IS_LAST][i] = one.clone();
        }
        columns[COL_IS_REAL][i] = one.clone();
    }

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

// ─── Constraint system ────────────────────────────────────────────────

pub struct PrecompileIoChunkedConstraintSystem {
    pub num_rows: usize,
}

impl PrecompileIoChunkedConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows }
    }
}

impl VmConstraintSystem for PrecompileIoChunkedConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_last_binary".into(),
            "is_last_implies_real".into(),
            "chunk_index_le_decomp".into(),
            "total_length_le_decomp".into(),
        ]
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
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_last = &row_evals[COL_IS_LAST];
            let ci = &row_evals[COL_CHUNK_INDEX];
            let tl = &row_evals[COL_TOTAL_LENGTH];
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_last.mul(&is_last.sub(&one));
            bodies[2][row] = is_last.mul(&one.sub(is_real));
            bodies[3][row] =
                eval_le_decomp(ci, COL_CHUNK_INDEX_BYTE_OFFSET, &row_evals);
            bodies[4][row] =
                eval_le_decomp(tl, COL_TOTAL_LENGTH_BYTE_OFFSET, &row_evals);
        }
        bodies
    }

    fn evaluate_at_point(
        &self,
        col_evals: &[Scalar],
        alpha: &Scalar,
    ) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];
        let is_last = &col_evals[COL_IS_LAST];
        let ci = &col_evals[COL_CHUNK_INDEX];
        let tl = &col_evals[COL_TOTAL_LENGTH];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_last.mul(&is_last.sub(&one)),
            is_last.mul(&one.sub(is_real)),
            eval_le_decomp(ci, COL_CHUNK_INDEX_BYTE_OFFSET, col_evals),
            eval_le_decomp(tl, COL_TOTAL_LENGTH_BYTE_OFFSET, col_evals),
        ];

        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
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

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_last = &col_coeffs[COL_IS_LAST];
        let ci = &col_coeffs[COL_CHUNK_INDEX];
        let tl = &col_coeffs[COL_TOTAL_LENGTH];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_last_m1 = poly_sub(is_last, &one_poly, curve);
        let is_last_binary = poly_mul(is_last, &is_last_m1, curve);

        let one_minus_real = poly_sub(&one_poly, is_real, curve);
        let is_last_implies_real = poly_mul(is_last, &one_minus_real, curve);

        let ci_decomp =
            build_le_decomp_poly(ci, COL_CHUNK_INDEX_BYTE_OFFSET, col_coeffs, curve);
        let tl_decomp =
            build_le_decomp_poly(tl, COL_TOTAL_LENGTH_BYTE_OFFSET, col_coeffs, curve);

        let bodies: [Vec<Scalar>; NUM_ROW_CONSTRAINTS] = [
            is_real_binary,
            is_last_binary,
            is_last_implies_real,
            ci_decomp,
            tl_decomp,
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

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        // chunk_index byte decomp.
        for k in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("chunked_chunk_index_byte_{}_8bit", k),
                    column_index: COL_CHUNK_INDEX_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // total_length byte decomp.
        for k in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("chunked_total_length_byte_{}_8bit", k),
                    column_index: COL_TOTAL_LENGTH_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // chunk_bytes (64 byte cols).
        for k in 0..CHUNK_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("chunked_chunk_byte_{}_8bit", k),
                    column_index: COL_CHUNK_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }

        LookupRequirements { tables, declarations }
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // [0] IS_REAL_NEXT, [1] CHUNK_INDEX_NEXT, [2] TOTAL_LENGTH_NEXT.
        vec![COL_IS_REAL, COL_CHUNK_INDEX, COL_TOTAL_LENGTH]
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
        if shifted_evals.len() != 3 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals_at_z[COL_IS_REAL];
        let is_last = &col_evals_at_z[COL_IS_LAST];
        let ci = &col_evals_at_z[COL_CHUNK_INDEX];
        let tl = &col_evals_at_z[COL_TOTAL_LENGTH];
        let is_real_next = &shifted_evals[0];
        let ci_next = &shifted_evals[1];
        let tl_next = &shifted_evals[2];

        let gating = is_real.mul(is_real_next);

        // body 0: chunk_index increment.
        let body0 = gating.mul(&ci_next.sub(ci).sub(&one));
        // body 1: total_length constancy.
        let body1 = gating.mul(&tl_next.sub(tl));
        // body 2: is_last terminates (no real row after the last
        // chunk). gating is `is_real(X) · is_last(X) · is_real(ωX)`.
        let body2 = is_real.mul(is_last).mul(is_real_next);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = ap.mul(&body0);
        let ap1 = ap.mul(alpha);
        let term1 = ap1.mul(&body1);
        let ap2 = ap1.mul(alpha);
        let term2 = ap2.mul(&body2);
        let total = term0.add(&term1).add(&term2);

        total.mul(&z.sub(omega_n_minus_1))
    }

    fn build_shifted_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_real_next = poly_shift(is_real, omega);
        let is_last = &col_coeffs[COL_IS_LAST];
        let ci = &col_coeffs[COL_CHUNK_INDEX];
        let ci_next = poly_shift(ci, omega);
        let tl = &col_coeffs[COL_TOTAL_LENGTH];
        let tl_next = poly_shift(tl, omega);

        let gating = poly_mul(is_real, &is_real_next, curve);

        // body 0: chunk_index_next - chunk_index - 1.
        let ci_diff_no_const = poly_sub(&ci_next, ci, curve);
        let ci_diff = poly_sub(&ci_diff_no_const, &one_poly, curve);
        let body0 = poly_mul(&gating, &ci_diff, curve);

        // body 1: total_length_next - total_length.
        let tl_diff = poly_sub(&tl_next, tl, curve);
        let body1 = poly_mul(&gating, &tl_diff, curve);

        // body 2: is_real(X) · is_last(X) · is_real(ωX).
        let pre = poly_mul(is_real, is_last, curve);
        let body2 = poly_mul(&pre, &is_real_next, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = poly_scalar_mul(&body0, &ap);
        let ap1 = ap.mul(alpha);
        let term1 = poly_scalar_mul(&body1, &ap1);
        let ap2 = ap1.mul(alpha);
        let term2 = poly_scalar_mul(&body2, &ap2);
        let mut total = poly_add(&term0, &term1, curve);
        total = poly_add(&total, &term2, curve);

        // Multiply by (X - ω^{n-1}) to vanish on the full domain.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let factor = vec![neg, Scalar::one(curve)];
        total = poly_mul(&total, &factor, curve);
        total
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Per-row binding `(chunk_index, chunk_bytes[0..64])` ↔ the
/// corresponding row on a single-row precompile I/O AIR (e.g. the
/// existing per-precompile gadgets like
/// [`crate::ripemd160_precompile_air::COL_INPUT_BYTES_OFFSET`] ..+
/// [`crate::ripemd160_precompile_air::MAX_INPUT_LENGTH`]).
///
/// **Stub**: the B-side column indices are
/// [`PRECOMPILE_IO_PLACEHOLDER`] until a dedicated `precompile_io_air`
/// row layout lands. The descriptor's structure pins the tuple
/// alignment (chunk_index + 64 bytes = 65 columns).
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_chunked_to_precompile_io_descriptor(
    chunked_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + CHUNK_BYTES);
    a_columns.push(COL_CHUNK_INDEX);
    for k in 0..CHUNK_BYTES {
        a_columns.push(COL_CHUNK_BYTES_OFFSET + k);
    }
    let b_columns: Vec<usize> = vec![PRECOMPILE_IO_PLACEHOLDER; 1 + CHUNK_BYTES];
    CrossAirLogUpDescriptor {
        label: "precompile_io_chunked_to_precompile_io_v1_stub".into(),
        a_layer_index: chunked_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn body_eval_at(
        cs: &PrecompileIoChunkedConstraintSystem,
        witness: &PrecompileIoChunkedWitness,
    ) -> Vec<Vec<Scalar>> {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(witness, curve);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    fn shifted_body_at(
        cols: &[Vec<Scalar>],
        transition: usize,
    ) -> (Scalar, Scalar, Scalar) {
        let curve = cols[0][0].curve_type();
        let one = Scalar::one(curve);
        let is_real = &cols[COL_IS_REAL][transition];
        let is_real_next = &cols[COL_IS_REAL][transition + 1];
        let is_last = &cols[COL_IS_LAST][transition];
        let ci = &cols[COL_CHUNK_INDEX][transition];
        let ci_next = &cols[COL_CHUNK_INDEX][transition + 1];
        let tl = &cols[COL_TOTAL_LENGTH][transition];
        let tl_next = &cols[COL_TOTAL_LENGTH][transition + 1];

        let gating = is_real.mul(is_real_next);
        let body0 = gating.mul(&ci_next.sub(ci).sub(&one));
        let body1 = gating.mul(&tl_next.sub(tl));
        let body2 = is_real.mul(is_last).mul(is_real_next);
        (body0, body1, body2)
    }

    // ─── Test 1: 64-byte input (single chunk) ─────────────────────────

    #[test]
    fn single_chunk_64_byte_input() {
        let input: Vec<u8> = (0..64u8).collect();
        let w = PrecompileIoChunkedWitness::from_input(&input);
        assert_eq!(w.num_chunks(), 1);
        assert_eq!(w.rows[0].chunk_index, 0);
        assert_eq!(w.rows[0].total_length, 64);
        assert!(w.rows[0].is_last_chunk);
        // chunk_bytes equals the input.
        for k in 0..64 {
            assert_eq!(w.rows[0].chunk_bytes[k], input[k]);
        }
        let cs = PrecompileIoChunkedConstraintSystem::new(w.num_chunks());
        let bodies = body_eval_at(&cs, &w);
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) must vanish on row {}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    // ─── Test 2: 192-byte input (3 chunks, KZG point-eval shape) ──────

    #[test]
    fn three_chunk_192_byte_input_kzg_shape() {
        let input: Vec<u8> = (0..192u8).collect();
        let w = PrecompileIoChunkedWitness::from_input(&input);
        assert_eq!(w.num_chunks(), 3);
        assert_eq!(w.rows[0].total_length, 192);
        assert_eq!(w.rows[1].total_length, 192);
        assert_eq!(w.rows[2].total_length, 192);
        assert_eq!(w.rows[0].chunk_index, 0);
        assert_eq!(w.rows[1].chunk_index, 1);
        assert_eq!(w.rows[2].chunk_index, 2);
        assert!(!w.rows[0].is_last_chunk);
        assert!(!w.rows[1].is_last_chunk);
        assert!(w.rows[2].is_last_chunk);
        // Each row's chunk_bytes maps to the correct slice.
        for i in 0..3 {
            for k in 0..CHUNK_BYTES {
                assert_eq!(w.rows[i].chunk_bytes[k], input[i * CHUNK_BYTES + k]);
            }
        }
        let cs = PrecompileIoChunkedConstraintSystem::new(w.num_chunks());
        let bodies = body_eval_at(&cs, &w);
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) must vanish on row {}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }

        // Shifted bodies vanish at transitions 0→1, 1→2.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        for t in [0usize, 1] {
            let (b0, b1, b2) = shifted_body_at(&cols, t);
            assert!(b0.is_zero(), "chunk_index_inc body must vanish at transition {}", t);
            assert!(b1.is_zero(), "total_length_const body must vanish at transition {}", t);
            assert!(b2.is_zero(), "is_last_terminates body must vanish at transition {}", t);
        }
    }

    // ─── Test 3: 128-byte input (2 chunks, ECRECOVER shape) ───────────

    #[test]
    fn two_chunk_128_byte_input_ecrecover_shape() {
        let input: Vec<u8> = (0..128u8).collect();
        let w = PrecompileIoChunkedWitness::from_input(&input);
        assert_eq!(w.num_chunks(), 2);
        assert_eq!(w.rows[0].total_length, 128);
        assert_eq!(w.rows[1].total_length, 128);
        assert_eq!(w.rows[0].chunk_index, 0);
        assert_eq!(w.rows[1].chunk_index, 1);
        assert!(!w.rows[0].is_last_chunk);
        assert!(w.rows[1].is_last_chunk);
        let cs = PrecompileIoChunkedConstraintSystem::new(w.num_chunks());
        let bodies = body_eval_at(&cs, &w);
        for body in &bodies {
            for v in body {
                assert!(v.is_zero());
            }
        }
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let (b0, b1, b2) = shifted_body_at(&cols, 0);
        assert!(b0.is_zero());
        assert!(b1.is_zero());
        assert!(b2.is_zero());
    }

    // ─── Test 4: tampered cross-row continuity detected ───────────────

    #[test]
    fn tampered_chunk_index_continuity_detected() {
        let input: Vec<u8> = (0..192u8).collect();
        let w = PrecompileIoChunkedWitness::from_input(&input);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper chunk_index on row 1: bump it to 5 (so transition 0→1
        // has ci_next - ci - 1 = 4, body fires).
        cols[COL_CHUNK_INDEX][1] = Scalar::from_u64(5, curve);
        let (body0, body1, body2) = shifted_body_at(&cols, 0);
        assert!(!body0.is_zero(), "chunk_index_inc body must fire on tampered chunk_index");
        // total_length untouched → body1 still vanishes.
        assert!(body1.is_zero());
        // is_last_terminates untouched → body2 vanishes.
        assert!(body2.is_zero());
    }

    #[test]
    fn tampered_total_length_constancy_detected() {
        let input: Vec<u8> = (0..192u8).collect();
        let w = PrecompileIoChunkedWitness::from_input(&input);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper total_length on row 1: change to 200.
        cols[COL_TOTAL_LENGTH][1] = Scalar::from_u64(200, curve);
        let (body0, body1, body2) = shifted_body_at(&cols, 0);
        // chunk_index untouched → body0 vanishes.
        assert!(body0.is_zero());
        // total_length differs → body1 fires.
        assert!(!body1.is_zero());
        assert!(body2.is_zero());
    }

    // ─── Test 5: descriptor well-formed + backward compat ─────────────

    #[test]
    fn descriptor_well_formed_and_backward_compat() {
        let d = make_chunked_to_precompile_io_descriptor(0, 1);
        assert_eq!(d.label, "precompile_io_chunked_to_precompile_io_v1_stub");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        // tuple shape: chunk_index + 64 chunk bytes = 65 columns.
        assert_eq!(d.a_columns.len(), 1 + CHUNK_BYTES);
        assert_eq!(d.b_columns.len(), 1 + CHUNK_BYTES);
        assert_eq!(d.a_columns[0], COL_CHUNK_INDEX);
        for k in 0..CHUNK_BYTES {
            assert_eq!(d.a_columns[1 + k], COL_CHUNK_BYTES_OFFSET + k);
            assert_eq!(d.b_columns[1 + k], PRECOMPILE_IO_PLACEHOLDER);
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, None);

        // Backward compatibility: 1-row chunked witness reduces to the
        // existing single-row precompile-IO shape — the
        // `chunk_bytes[0..64]` block IS the existing
        // `MAX_INPUT_LENGTH = 64` input_bytes column block. We confirm
        // this by checking that the chunk_bytes column range exactly
        // matches the canonical single-precompile size.
        assert_eq!(CHUNK_BYTES, 64);
        let w = PrecompileIoChunkedWitness::from_input(&[0x42u8; 32]);
        assert_eq!(w.num_chunks(), 1);
        assert_eq!(w.rows[0].chunk_bytes[0], 0x42);
        assert_eq!(w.rows[0].chunk_bytes[31], 0x42);
        // Trailing bytes 32..64 are zero-padded.
        assert_eq!(w.rows[0].chunk_bytes[32], 0);
    }

    // ─── Test 6: column layout pinned ─────────────────────────────────

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_CHUNK_INDEX, 0);
        assert_eq!(COL_CHUNK_INDEX_BYTE_OFFSET, 1);
        assert_eq!(COL_CHUNK_BYTES_OFFSET, 9);
        assert_eq!(COL_TOTAL_LENGTH, 73);
        assert_eq!(COL_TOTAL_LENGTH_BYTE_OFFSET, 74);
        assert_eq!(COL_IS_LAST, 82);
        assert_eq!(COL_IS_REAL, 83);
        assert_eq!(NUM_COLUMNS, 84);
        assert_eq!(NUM_ROW_CONSTRAINTS, 5);
        assert_eq!(NUM_SHIFTED, 3);
        assert_eq!(CHUNK_BYTES, 64);
        assert_eq!(MAX_CHUNKS, 64);

        // Shifted column indices ordering matches `evaluate_shifted_at_point`.
        let cs = PrecompileIoChunkedConstraintSystem::new(1);
        let s = cs.shifted_column_indices();
        assert_eq!(s, vec![COL_IS_REAL, COL_CHUNK_INDEX, COL_TOTAL_LENGTH]);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);

        // Lookup declarations cover all byte columns.
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 8 chunk_index + 8 total_length + 64 chunk_bytes = 80 cols.
        assert_eq!(reqs.declarations.len(), 2 * U64_BYTES + CHUNK_BYTES);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }

    // ─── Test 7: empty-input edge case ────────────────────────────────

    #[test]
    fn empty_input_single_padding_chunk() {
        let w = PrecompileIoChunkedWitness::from_input(&[]);
        // Empty input still produces a single sentinel row.
        assert_eq!(w.num_chunks(), 1);
        assert_eq!(w.rows[0].chunk_index, 0);
        assert_eq!(w.rows[0].total_length, 0);
        assert!(w.rows[0].is_last_chunk);
        for k in 0..CHUNK_BYTES {
            assert_eq!(w.rows[0].chunk_bytes[k], 0);
        }
        let cs = PrecompileIoChunkedConstraintSystem::new(w.num_chunks());
        let bodies = body_eval_at(&cs, &w);
        for body in &bodies {
            for v in body {
                assert!(v.is_zero());
            }
        }
    }
}
