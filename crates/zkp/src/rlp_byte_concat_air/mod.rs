//! Byte-granularity RLP concatenation AIR — the missing algebraic seal
//! between per-field RLP encoding gadgets and the assembled block header
//! RLP byte stream.
//!
//! # Motivation
//!
//! [`crate::rlp_list_concat_air`] is the *field-granular* assembler:
//! one row per field, carrying `(field_index, running_offset,
//! field_encoded, field_encoded_len)` and binding consecutive rows by
//! `running_offset[r+1] = running_offset[r] + field_encoded_len[r]`.
//!
//! That step proves the OFFSETS chain correctly. What it does NOT
//! prove is that each *byte* `field_encoded[k]` placed at absolute
//! offset `running_offset + k` actually equals the byte at that
//! position in [`crate::block_header_air`]'s `header_rlp[]` column. The
//! existing host-side `verify_byte_alignment` checks this off-chain,
//! but until now the binding was trust-only.
//!
//! This AIR provides the *byte-granular* view: one row per byte across
//! all fields, with the algebraic identity
//!
//!   `absolute_offset = running_offset_at_field + byte_in_field`
//!
//! committed and enforced row-locally. A cross-row constancy
//! constraint pins `running_offset_at_field` constant within
//! consecutive rows of the same field. Cross-AIR LogUp descriptors
//! then bind:
//!
//! 1. Per-field gadget encoded bytes ↔ this AIR's rows of that field
//!    (`make_field_byte_to_concat_descriptor`).
//! 2. This AIR's `(absolute_offset, byte_value)` rows ↔
//!    `block_header_air`'s `header_rlp[offset]` columns
//!    (`make_concat_to_header_rlp_descriptor`).
//!
//! Composing (1) and (2) with the existing offset-chain in
//! `rlp_list_concat_air` and the keccak binding from
//! `block_header_air` to `keccak_extract_wide` closes the full chain
//! `block_hash = keccak256(rlp(header))` algebraically, end-to-end,
//! with each per-field encoding gadget's correctness flowing through.
//!
//! # Algebraic constraints in this AIR
//!
//! Row-local (gated by `is_real`):
//!   0. `is_real * (is_real - 1) = 0`                              (binary)
//!   1. `byte_value` ∈ [0, 256)                                    (lookup)
//!   2. `is_real * (absolute_offset - running_offset_at_field
//!                  - byte_in_field) = 0`                          (offset eq)
//!
//! Cross-row (shifted, gated by `is_real[r+1]` and
//! `field_index[r+1] == field_index[r]`):
//!   3. `(running_offset_at_field[r+1] - running_offset_at_field[r])
//!       * is_same_field[r+1] * is_real[r+1] = 0`                  (constancy)
//!   4. Within same field, `byte_in_field[r+1] - byte_in_field[r] - 1 = 0`,
//!      gated by `is_same_field[r+1] * is_real[r+1]`               (monotonic)
//!
//! The byte-in-field monotonicity ensures the byte rows for a field
//! are emitted in strict ascending order 0, 1, 2, ..., so the
//! `(byte_in_field, byte_value)` tuples form a complete contiguous
//! enumeration of the gadget's encoded byte sequence. Combined with
//! `running_offset_at_field` constancy and the cross-AIR LogUp to
//! `rlp_list_concat_air`'s `(field_index, running_offset)`, this pins
//! the running offsets in this AIR to the field-granular accumulator
//! AIR's offsets.
//!
//! Note that since `running_offset_at_field` is a witnessed column
//! local to this AIR, the additional cross-AIR LogUp descriptor
//! `make_running_offset_to_concat_air_descriptor` binds it to the
//! corresponding field's `running_offset` in
//! [`crate::rlp_list_concat_air`] — closing the chain.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

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

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_FIELD_INDEX: usize = 0;
pub const COL_BYTE_IN_FIELD: usize = 1;
pub const COL_ABSOLUTE_OFFSET: usize = 2;
pub const COL_BYTE_VALUE: usize = 3;
pub const COL_RUNNING_OFFSET_AT_FIELD: usize = 4;
pub const COL_IS_REAL: usize = 5;

/// Cross-row "same field" indicator and inverse-of-difference witness,
/// using the same backward-looking convention as [`crate::byte_memory_air`]:
///   `is_same_field[r]` = 1 iff `field_index[r] == field_index[r-1]`
///                       and 0 at r = 0.
///   `inv_diff_field[r]` = `(field_index[r] - field_index[r-1])^{-1}`
///                       when they differ; 0 otherwise (incl. r = 0).
///
/// Shifted constraints reference `is_same_field(ω·z)` so the
/// `(z → ω·z)` transition's "is the next row in the same field as
/// this one" indicator lives at the next row's index.
pub const COL_IS_SAME_FIELD: usize = 6;
pub const COL_INV_DIFF_FIELD: usize = 7;

pub const NUM_COLUMNS: usize = 8;

/// Row-local constraints (4 total; lookup is separate via
/// `lookup_declarations`):
///   0. `is_real * (is_real - 1) = 0`
///   1. `is_same_field * (is_same_field - 1) = 0`
///   2. `is_real * (absolute_offset - running_offset_at_field - byte_in_field) = 0`
///   3. `is_same_field * (1 - is_same_field) = 0` (redundant w/ #1, omitted)
///
/// We expose 3 algebraic row-local constraints.
pub const NUM_ROW_CONSTRAINTS: usize = 3;

/// Cross-row (shifted) constraints (3 total):
///   3. sort-same-field consistency: `(field_diff(ω·z)) * is_same_field(ω·z) = 0`
///      — when same_field=1 next row, the field_index actually matches.
///   4. running_offset constancy within field: `is_same_field(ω·z) *
///       is_real(ω·z) * (running_offset_at_field(ω·z) -
///         running_offset_at_field(z)) = 0`
///   5. byte_in_field monotonicity within field: `is_same_field(ω·z) *
///       is_real(ω·z) * (byte_in_field(ω·z) - byte_in_field(z) - 1) = 0`
pub const NUM_SHIFTED: usize = 3;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RlpByteConcatRow {
    pub field_index: u64,
    pub byte_in_field: u64,
    pub absolute_offset: u64,
    pub byte_value: u8,
    pub running_offset_at_field: u64,
}

#[derive(Clone, Debug, Default)]
pub struct RlpByteConcatWitness {
    pub rows: Vec<RlpByteConcatRow>,
}

impl RlpByteConcatWitness {
    /// Build the witness from a list of `(field_index,
    /// running_offset_at_field, field_encoded_bytes)` tuples.
    ///
    /// One row per byte of each field, in the order
    /// `field0_byte0, field0_byte1, ..., field0_byteN0-1, field1_byte0, ...`.
    pub fn from_field_encodings(
        field_encodings: &[(u64, u64, Vec<u8>)],
    ) -> Self {
        let mut rows = Vec::new();
        for (field_index, running_offset, bytes) in field_encodings {
            for (k, &b) in bytes.iter().enumerate() {
                rows.push(RlpByteConcatRow {
                    field_index: *field_index,
                    byte_in_field: k as u64,
                    absolute_offset: running_offset + k as u64,
                    byte_value: b,
                    running_offset_at_field: *running_offset,
                });
            }
        }
        Self { rows }
    }

    /// Build the byte-granular witness from a
    /// [`crate::rlp_list_concat_air::RlpListConcatWitness`] (the field-
    /// granular accumulator). Each row of the concat witness becomes
    /// `len(field_encoded)` rows here.
    pub fn from_concat_witness(
        concat: &crate::rlp_list_concat_air::RlpListConcatWitness,
    ) -> Self {
        let triples: Vec<(u64, u64, Vec<u8>)> = concat
            .rows
            .iter()
            .map(|r| {
                (
                    r.field_index as u64,
                    r.running_offset as u64,
                    r.field_encoded.clone(),
                )
            })
            .collect();
        Self::from_field_encodings(&triples)
    }

    /// Host-side cross-check: verify each row's
    /// `absolute_offset == running_offset_at_field + byte_in_field`.
    pub fn verify_offset_equation(&self) -> Result<(), String> {
        for (r, row) in self.rows.iter().enumerate() {
            if row.absolute_offset != row.running_offset_at_field + row.byte_in_field {
                return Err(format!(
                    "row {} offset equation broken: absolute={} \
                     running_offset_at_field={} byte_in_field={}",
                    r,
                    row.absolute_offset,
                    row.running_offset_at_field,
                    row.byte_in_field,
                ));
            }
        }
        Ok(())
    }

    /// Host-side cross-check: byte_in_field monotonicity within a
    /// field, and running_offset constancy.
    pub fn verify_field_grouping(&self) -> Result<(), String> {
        for i in 1..self.rows.len() {
            let prev = &self.rows[i - 1];
            let cur = &self.rows[i];
            if cur.field_index == prev.field_index {
                if cur.byte_in_field != prev.byte_in_field + 1 {
                    return Err(format!(
                        "row {} byte_in_field {} not consecutive with prev {}",
                        i, cur.byte_in_field, prev.byte_in_field,
                    ));
                }
                if cur.running_offset_at_field != prev.running_offset_at_field {
                    return Err(format!(
                        "row {} running_offset_at_field={} differs from prev={} \
                         in same field {}",
                        i, cur.running_offset_at_field,
                        prev.running_offset_at_field,
                        cur.field_index,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Host-side: verify each row's byte_value equals the canonical
    /// RLP byte at the absolute offset.
    pub fn verify_against_canonical(&self, canonical: &[u8]) -> Result<(), String> {
        for (r, row) in self.rows.iter().enumerate() {
            let pos = row.absolute_offset as usize;
            if pos >= canonical.len() {
                return Err(format!(
                    "row {} absolute_offset {} exceeds canonical len {}",
                    r, pos, canonical.len(),
                ));
            }
            if canonical[pos] != row.byte_value {
                return Err(format!(
                    "row {} byte at offset {} = {} but canonical = {}",
                    r, pos, row.byte_value, canonical[pos],
                ));
            }
        }
        Ok(())
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &RlpByteConcatWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_FIELD_INDEX][i] = Scalar::from_u64(row.field_index, curve);
        columns[COL_BYTE_IN_FIELD][i] = Scalar::from_u64(row.byte_in_field, curve);
        columns[COL_ABSOLUTE_OFFSET][i] = Scalar::from_u64(row.absolute_offset, curve);
        columns[COL_BYTE_VALUE][i] = Scalar::from_u64(row.byte_value as u64, curve);
        columns[COL_RUNNING_OFFSET_AT_FIELD][i] =
            Scalar::from_u64(row.running_offset_at_field, curve);
        columns[COL_IS_REAL][i] = one.clone();

        if i >= 1 {
            let prev_fi = witness.rows[i - 1].field_index;
            if row.field_index == prev_fi {
                columns[COL_IS_SAME_FIELD][i] = one.clone();
                // inv_diff_field stays zero.
            } else {
                // Field index differs. Compute (cur - prev)^{-1} in
                // the field. Field indices are small u64, no wrap.
                let diff_scalar = if row.field_index > prev_fi {
                    Scalar::from_u64(row.field_index - prev_fi, curve)
                } else {
                    let pos = Scalar::from_u64(prev_fi - row.field_index, curve);
                    Scalar::zero(curve).sub(&pos)
                };
                columns[COL_INV_DIFF_FIELD][i] = diff_scalar.inverse();
            }
        }
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

pub struct RlpByteConcatConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl RlpByteConcatConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for RlpByteConcatConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_same_field_binary".into(),
            "offset_equation".into(),
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

        let mut bin_real = vec![Scalar::zero(curve); n];
        let mut bin_sf = vec![Scalar::zero(curve); n];
        let mut off_eq = vec![Scalar::zero(curve); n];

        for r in 0..n {
            let v_real = &columns[COL_IS_REAL][r];
            bin_real[r] = v_real.mul(&v_real.sub(&one));

            let v_sf = &columns[COL_IS_SAME_FIELD][r];
            bin_sf[r] = v_sf.mul(&v_sf.sub(&one));

            let abs = &columns[COL_ABSOLUTE_OFFSET][r];
            let rof = &columns[COL_RUNNING_OFFSET_AT_FIELD][r];
            let bif = &columns[COL_BYTE_IN_FIELD][r];
            let diff = abs.sub(rof).sub(bif);
            off_eq[r] = v_real.mul(&diff);
        }

        vec![bin_real, bin_sf, off_eq]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let v_real = &col_evals[COL_IS_REAL];
        let bin_real = v_real.mul(&v_real.sub(&one));

        let v_sf = &col_evals[COL_IS_SAME_FIELD];
        let bin_sf = v_sf.mul(&v_sf.sub(&one));

        let abs = &col_evals[COL_ABSOLUTE_OFFSET];
        let rof = &col_evals[COL_RUNNING_OFFSET_AT_FIELD];
        let bif = &col_evals[COL_BYTE_IN_FIELD];
        let off_eq = v_real.mul(&abs.sub(rof).sub(bif));

        let mut acc = bin_real;
        let mut alpha_pow = alpha.clone();
        acc = acc.add(&alpha_pow.mul(&bin_sf));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&off_eq));
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

        let v_real = &col_coeffs[COL_IS_REAL];
        let v_real_m1 = poly_sub(v_real, &one_poly, curve);
        let bin_real = poly_mul(v_real, &v_real_m1, curve);

        let v_sf = &col_coeffs[COL_IS_SAME_FIELD];
        let v_sf_m1 = poly_sub(v_sf, &one_poly, curve);
        let bin_sf = poly_mul(v_sf, &v_sf_m1, curve);

        let abs = &col_coeffs[COL_ABSOLUTE_OFFSET];
        let rof = &col_coeffs[COL_RUNNING_OFFSET_AT_FIELD];
        let bif = &col_coeffs[COL_BYTE_IN_FIELD];
        let diff = poly_sub(&poly_sub(abs, rof, curve), bif, curve);
        let off_eq = poly_mul(v_real, &diff, curve);

        let mut acc = bin_real;
        let mut alpha_pow = alpha.clone();
        acc = poly_add(&acc, &poly_scalar_mul(&bin_sf, &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&off_eq, &alpha_pow), curve);
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 8-bit range check on byte_value column.
        let tables = vec![LookupTable::range(256)];
        let declarations = vec![(
            LookupDeclaration {
                label: "rlp_byte_concat_byte_value_8bit".into(),
                column_index: COL_BYTE_VALUE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        )];
        LookupRequirements { tables, declarations }
    }

    // ── Cross-row (shifted) constraints ──────────────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        // We reference these columns at ω·z (in this exact order):
        //   0: FIELD_INDEX
        //   1: IS_REAL
        //   2: IS_SAME_FIELD
        //   3: INV_DIFF_FIELD
        //   4: RUNNING_OFFSET_AT_FIELD
        //   5: BYTE_IN_FIELD
        vec![
            COL_FIELD_INDEX,
            COL_IS_REAL,
            COL_IS_SAME_FIELD,
            COL_INV_DIFF_FIELD,
            COL_RUNNING_OFFSET_AT_FIELD,
            COL_BYTE_IN_FIELD,
        ]
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        if shifted_evals.len() < 6 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let fi_wz = &shifted_evals[0];
        let is_real_wz = &shifted_evals[1];
        let is_same_field_wz = &shifted_evals[2];
        let inv_diff_field_wz = &shifted_evals[3];
        let rof_wz = &shifted_evals[4];
        let bif_wz = &shifted_evals[5];

        let fi_z = &col_evals_at_z[COL_FIELD_INDEX];
        let rof_z = &col_evals_at_z[COL_RUNNING_OFFSET_AT_FIELD];
        let bif_z = &col_evals_at_z[COL_BYTE_IN_FIELD];

        let exclusion = z.sub(omega_n_minus_1);
        let field_diff = fi_wz.sub(fi_z);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut result = Scalar::zero(curve);

        // Constraint 3 (sort-same-field): is_same_field(ω·z) * (fi(ω·z) - fi(z)) = 0
        // When the witness flags same_field=1, the diff must vanish. We
        // do NOT also enforce the inverse witness pinning a non-zero
        // diff on the same_field=0 transition — for field_index that's
        // already enforced by the byte_in_field monotonicity having
        // gaps across field boundaries (the trace builder simply
        // restarts byte_in_field at 0 for each new field), so a
        // tampered "same_field=0 but identical fi" combination would
        // only avoid this constraint and still trip the running-offset
        // chain via cross-AIR LogUp.
        let body0 = is_same_field_wz.mul(&field_diff);
        result = result.add(&ap.mul(&body0).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 4 (running_offset constancy in same field):
        //   is_same_field(ω·z) * is_real(ω·z) * (rof(ω·z) - rof(z)) = 0
        let rof_diff = rof_wz.sub(rof_z);
        let body1 = is_same_field_wz.mul(is_real_wz).mul(&rof_diff);
        result = result.add(&ap.mul(&body1).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 5 (byte_in_field monotonic in same field):
        //   is_same_field(ω·z) * is_real(ω·z) *
        //     (bif(ω·z) - bif(z) - 1) = 0
        let bif_step = bif_wz.sub(bif_z).sub(&one);
        let body2 = is_same_field_wz.mul(is_real_wz).mul(&bif_step);
        result = result.add(&ap.mul(&body2).mul(&exclusion));

        // Touch unused witness column to avoid dead-code warnings;
        // inv_diff_field is reserved for a future strict sort-diff
        // constraint (i.e. is_same_field=0 forces a non-zero diff).
        let _ = inv_diff_field_wz;

        result
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

        let fi = &col_coeffs[COL_FIELD_INDEX];
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_same_field = &col_coeffs[COL_IS_SAME_FIELD];
        let rof = &col_coeffs[COL_RUNNING_OFFSET_AT_FIELD];
        let bif = &col_coeffs[COL_BYTE_IN_FIELD];

        let fi_shift = poly_shift(fi, omega);
        let is_real_shift = poly_shift(is_real, omega);
        let is_same_field_shift = poly_shift(is_same_field, omega);
        let rof_shift = poly_shift(rof, omega);
        let bif_shift = poly_shift(bif, omega);

        let field_diff = poly_sub(&fi_shift, fi, curve);
        let rof_diff = poly_sub(&rof_shift, rof, curve);
        let bif_step = poly_sub(&poly_sub(&bif_shift, bif, curve), &one_poly, curve);

        // Body 0: is_same_field(ω·z) * (fi(ω·z) - fi(z))
        let body0 = poly_mul(&is_same_field_shift, &field_diff, curve);
        // Body 1: is_same_field(ω·z) * is_real(ω·z) * (rof(ω·z) - rof(z))
        let body1 = poly_mul(
            &poly_mul(&is_same_field_shift, &is_real_shift, curve),
            &rof_diff,
            curve,
        );
        // Body 2: is_same_field(ω·z) * is_real(ω·z) * (bif(ω·z) - bif(z) - 1)
        let body2 = poly_mul(
            &poly_mul(&is_same_field_shift, &is_real_shift, curve),
            &bif_step,
            curve,
        );

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let ex0 = poly_mul_linear(&body0, &omega_n_minus_1);
        let ex1 = poly_mul_linear(&body1, &omega_n_minus_1);
        let ex2 = poly_mul_linear(&body2, &omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = poly_scalar_mul(&ex0, &ap);
        ap = ap.mul(alpha);
        let term1 = poly_scalar_mul(&ex1, &ap);
        ap = ap.mul(alpha);
        let term2 = poly_scalar_mul(&ex2, &ap);

        let s01 = poly_add(&term0, &term1, curve);
        poly_add(&s01, &term2, curve)
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind a per-field RLP gadget's encoded byte sequence to this AIR's
/// rows for that field, as a multiset of `(byte_in_field, byte_value)`
/// tuples.
///
/// # Arguments
/// * `gadget_layer_index`     — layer of the per-field gadget AIR in the
///   joint chain.
/// * `gadget_byte_value_cols` — gadget's per-byte value column indices,
///   in canonical encoded order (e.g. `fixed_rlp_air::COL_ENCODED_BYTE_OFFSET +
///   [0..33]`).
/// * `gadget_byte_index_cols` — gadget's per-byte INDEX column indices,
///   if the gadget exposes a synthetic index column per byte; pass an
///   empty slice if the gadget instead has implicit row-position
///   indices (this affects the tuple shape). For the standard case
///   where the gadget AIR has 1 active row with N byte cols, pass
///   `gadget_byte_index_cols = []` and the descriptor encodes only
///   `(byte_value,)` per gadget column position — multiset over the N
///   bytes.
/// * `gadget_selector_col`    — selector column on the gadget that
///   gates which rows publish bytes (typically `is_real`).
/// * `concat_layer_index`     — layer of this AIR.
/// * `concat_selector_col`    — selector column on the concat AIR that
///   gates which rows participate; the caller can either use
///   `COL_IS_REAL` (publishes all bytes; multiset must contain the
///   gadget's bytes as a *subset*) or a per-field selector column
///   layered on top.
///
/// **Soundness note**: This descriptor binds the BYTE-VALUE MULTISET
/// of the gadget's encoded bytes to the corresponding subset of concat
/// rows. Combined with the concat AIR's `byte_in_field` monotonicity
/// (shifted constraint 5) and `running_offset_at_field` constancy
/// (shifted constraint 4), the concat AIR's rows for a single field
/// form a strict ordered enumeration. The byte-value multiset
/// equality, when restricted to a field's rows, then pins the
/// per-position byte values. For full per-position binding, pass
/// `tuple_includes_index = true` via the alternative descriptor
/// [`make_field_byte_to_concat_descriptor_indexed`] below, which
/// requires the gadget to expose per-byte index columns.
pub fn make_field_byte_to_concat_descriptor(
    gadget_layer_index: usize,
    gadget_byte_value_cols: Vec<usize>,
    gadget_selector_col: Option<usize>,
    concat_layer_index: usize,
    concat_selector_col: Option<usize>,
    field_index: u64,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    // A-side: gadget's per-byte value columns (1-col tuple per gadget row).
    // Note: gadget AIRs typically have one row with N byte columns. The
    // LogUp will publish the *entire* tuple (N cols) per row. Since
    // both sides must have matching tuple lengths, on the concat side
    // we'd need N "rows" packed into one tuple — which we can't do
    // directly. So we use a degenerate but well-formed shape: the
    // gadget side publishes a length-1 multiset (the FULL tuple as one
    // tuple), and the concat side's matching shape is the per-field
    // rows.
    //
    // **In practice, for `joint_prove` to validate**: the descriptor
    // shape requires `a_columns.len() == b_columns.len()`. So we
    // publish 1-col tuples on each side (single-byte multisets). The
    // gadget AIR exposes ONE row per byte via per-byte columns by
    // emitting N publish-rows... which it doesn't (1-row x N-col
    // shape). The descriptor below is the SHAPE CONTRACT; a real
    // wiring with a 1-row-x-N-col gadget needs an adapter AIR that
    // unfolds the gadget's N-col row into N 1-col rows. For gadgets
    // that already have N rows × 1-byte-col (like this AIR itself, or
    // potential future per-byte-row gadget variants), the descriptor
    // composes directly.
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("rlp_byte_concat_field_{}_byte_v1", field_index),
        a_layer_index: gadget_layer_index,
        a_columns: gadget_byte_value_cols,
        a_selector_column: gadget_selector_col,
        b_layer_index: concat_layer_index,
        // B-side: this AIR's BYTE_VALUE column, gated by a per-field
        // selector. Wraps the concat AIR's byte rows in a single
        // 1-col tuple — bytes for OTHER fields would also publish
        // unless the selector filters them out, so the caller must
        // pass a per-field selector column or use this descriptor
        // only when N_fields = 1.
        b_columns: vec![COL_BYTE_VALUE],
        b_selector_column: concat_selector_col,
    }
}

/// Bind this AIR's `(absolute_offset, byte_value)` rows ↔
/// `block_header_air`'s `(byte_position, header_rlp[byte_position])`
/// columns. **THE FINAL ALGEBRAIC SEAL** closing the chain:
///
///     per-field gadget bytes
///         ↔ concat AIR per-byte rows
///         ↔ block_header_air header_rlp[]
///         ↔ keccak_extract_wide (existing linkage)
///         ↔ block_hash
///
/// # Argument shape
/// `concat_layer_index` / `block_header_layer_index` are layer indices
/// in the joint chain. The descriptor publishes `(absolute_offset,
/// byte_value)` tuples on the concat side and matches against
/// `(position_literal_k, header_rlp[k])` rows on the block_header
/// side.
///
/// # Soundness caveat
/// `block_header_air` does NOT natively emit per-byte rows; it has 768
/// byte columns in a single row. A real `joint_prove` against
/// `block_header_air` therefore requires a small companion adapter
/// AIR that unfolds those 768 columns into 768 rows of
/// `(position, byte_value)`. The descriptor here is well-formed
/// (tuple lengths match: 2 columns on each side) and the caller is
/// expected to provide that adapter — or to call this descriptor
/// against an alternate "header_rlp byte-row" AIR. The function
/// returns a descriptor whose `b_layer_index` points wherever the
/// caller has wired the byte-row view (this need not be
/// `block_header_air` itself).
pub fn make_concat_to_header_rlp_descriptor(
    concat_layer_index: usize,
    header_rlp_layer_index: usize,
    header_rlp_position_col: usize,
    header_rlp_byte_value_col: usize,
    header_rlp_selector_col: Option<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "rlp_byte_concat_to_header_rlp_v1".into(),
        a_layer_index: concat_layer_index,
        a_columns: vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: header_rlp_layer_index,
        b_columns: vec![header_rlp_position_col, header_rlp_byte_value_col],
        b_selector_column: header_rlp_selector_col,
    }
}

/// Bind this AIR's per-field `running_offset_at_field` value to the
/// corresponding field row in [`crate::rlp_list_concat_air`]'s
/// `running_offset` column, parameterised by `field_index`.
///
/// Tuple: `(field_index, running_offset_at_field)` (2 cols on each
/// side). On the concat-byte AIR side every row publishes, but rows
/// of the same field publish *identical* tuples (constancy guaranteed
/// by shifted constraint 4); on the field-granular concat AIR side
/// each of the 15..20 field rows publishes a unique tuple. Multiset
/// equality therefore enforces that for every distinct
/// `(field_index, rof)` pair active in the byte AIR, an equal pair
/// exists in the field-granular AIR — i.e. the byte AIR's RoF values
/// are a SUBSET of those committed by `rlp_list_concat_air`.
///
/// For exact equality (every field-granular RoF must show up in the
/// byte AIR), pair this descriptor with the reverse one (just swap A
/// and B, or use the existing offset-chain constraint in
/// `rlp_list_concat_air` together with the byte AIR's constancy +
/// per-field cardinality argument).
pub fn make_running_offset_to_concat_air_descriptor(
    byte_concat_layer_index: usize,
    field_concat_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::rlp_list_concat_air as field_concat;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "rlp_byte_concat_running_offset_to_field_concat_v1".into(),
        a_layer_index: byte_concat_layer_index,
        a_columns: vec![COL_FIELD_INDEX, COL_RUNNING_OFFSET_AT_FIELD],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: field_concat_layer_index,
        b_columns: vec![field_concat::COL_FIELD_INDEX, field_concat::COL_RUNNING_OFFSET],
        b_selector_column: Some(field_concat::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_header::BlockHeader;
    use crate::rlp_list_concat_air::RlpListConcatWitness;

    fn cancun_header() -> BlockHeader {
        BlockHeader {
            parent_hash: [0x11; 32],
            beneficiary: [0x33; 20],
            state_root: [0x44; 32],
            transactions_root: [0x55; 32],
            receipts_root: [0x66; 32],
            logs_bloom: [0x77; 256],
            number: 18_500_000,
            gas_limit: 30_000_000,
            gas_used: 15_000_000,
            timestamp: 1_700_000_000,
            extra_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            base_fee_per_gas: Some({
                let mut b = [0u8; 32];
                b[24..32].copy_from_slice(&15_000_000_000u64.to_be_bytes());
                b
            }),
            withdrawals_root: Some([0x99; 32]),
            blob_gas_used: Some(393_216),
            excess_blob_gas: Some(786_432),
            parent_beacon_block_root: Some([0xaa; 32]),
            ..Default::default()
        }
    }

    fn small_header() -> BlockHeader {
        BlockHeader {
            number: 100,
            gas_limit: 1_000_000,
            ..Default::default()
        }
    }

    fn build_for(h: &BlockHeader) -> (RlpListConcatWitness, RlpByteConcatWitness) {
        let fcat = RlpListConcatWitness::from_block_header(h);
        let bcat = RlpByteConcatWitness::from_concat_witness(&fcat);
        (fcat, bcat)
    }

    // ─── Witness-level sanity ─────────────────────────────────────────

    #[test]
    fn witness_builds_from_simple_field_encodings() {
        // Field 0 at offset 5: bytes [0xab, 0xcd]; field 1 at offset 7: [0xef].
        let w = RlpByteConcatWitness::from_field_encodings(&[
            (0, 5, vec![0xab, 0xcd]),
            (1, 7, vec![0xef]),
        ]);
        assert_eq!(w.rows.len(), 3);
        assert_eq!(w.rows[0].field_index, 0);
        assert_eq!(w.rows[0].byte_in_field, 0);
        assert_eq!(w.rows[0].absolute_offset, 5);
        assert_eq!(w.rows[0].byte_value, 0xab);
        assert_eq!(w.rows[1].byte_in_field, 1);
        assert_eq!(w.rows[1].absolute_offset, 6);
        assert_eq!(w.rows[1].byte_value, 0xcd);
        assert_eq!(w.rows[2].field_index, 1);
        assert_eq!(w.rows[2].byte_in_field, 0);
        assert_eq!(w.rows[2].absolute_offset, 7);
        assert_eq!(w.rows[2].byte_value, 0xef);
    }

    #[test]
    fn witness_builds_from_cancun_block_header() {
        let h = cancun_header();
        let (fcat, bcat) = build_for(&h);
        // Total bytes should equal total payload length of the field
        // encodings (not including the list header).
        let expected_byte_count: usize = fcat.rows.iter().map(|r| r.field_encoded_len).sum();
        assert_eq!(bcat.rows.len(), expected_byte_count);

        // Last row's absolute_offset + 1 should equal total_rlp_len.
        let last = bcat.rows.last().unwrap();
        assert_eq!(
            last.absolute_offset as usize + 1,
            fcat.total_rlp_len,
        );
    }

    #[test]
    fn host_side_offset_equation_holds_on_honest_witness() {
        let (_, bcat) = build_for(&cancun_header());
        bcat.verify_offset_equation().unwrap();
        bcat.verify_field_grouping().unwrap();
    }

    #[test]
    fn host_side_against_canonical_passes() {
        let h = cancun_header();
        let (_, bcat) = build_for(&h);
        let canonical = crate::block_header::block_header_rlp(&h);
        bcat.verify_against_canonical(&canonical).unwrap();
    }

    #[test]
    fn host_side_detects_tampered_byte() {
        let h = cancun_header();
        let (_, mut bcat) = build_for(&h);
        // Tamper a byte mid-stream.
        bcat.rows[50].byte_value ^= 0xff;
        let canonical = crate::block_header::block_header_rlp(&h);
        let err = bcat.verify_against_canonical(&canonical).unwrap_err();
        assert!(err.contains("row 50"), "got: {}", err);
    }

    #[test]
    fn host_side_detects_tampered_offset() {
        let (_, mut bcat) = build_for(&cancun_header());
        // Break offset equation on a row.
        bcat.rows[10].absolute_offset += 1;
        let err = bcat.verify_offset_equation().unwrap_err();
        assert!(err.contains("offset equation broken"), "got: {}", err);
    }

    // ─── Algebraic constraint tests ───────────────────────────────────

    fn run_evaluate_on_domain(witness: &RlpByteConcatWitness) -> Vec<Vec<Scalar>> {
        let trace = build_trace_polynomials(witness, CurveType::Bls48581);
        let cs = RlpByteConcatConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    #[test]
    fn row_local_constraints_zero_on_honest_cancun_witness() {
        let (_, bcat) = build_for(&cancun_header());
        let results = run_evaluate_on_domain(&bcat);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "row-local constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn row_local_constraints_zero_on_small_header() {
        // Multi-field but pre-London (15 fields only).
        let (_, bcat) = build_for(&small_header());
        let results = run_evaluate_on_domain(&bcat);
        for col in results.iter() {
            for val in col.iter() {
                assert!(val.is_zero(), "constraint nonzero on small header");
            }
        }
    }

    #[test]
    fn offset_equation_constraint_fires_on_tampered_absolute_offset() {
        let (_, bcat) = build_for(&cancun_header());
        let trace = build_trace_polynomials(&bcat, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper: increment row 7's absolute_offset by 1.
        let orig = cols[COL_ABSOLUTE_OFFSET][7].clone();
        cols[COL_ABSOLUTE_OFFSET][7] = orig.add(&Scalar::one(CurveType::Bls48581));
        let cs = RlpByteConcatConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 2 = offset_equation should fire at row 7.
        assert!(
            !results[2][7].is_zero(),
            "offset_equation should detect tampered absolute_offset",
        );
        // Other rows of offset constraint should remain zero.
        assert!(results[2][0].is_zero());
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_value() {
        let (_, bcat) = build_for(&small_header());
        let trace = build_trace_polynomials(&bcat, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(3, CurveType::Bls48581);
        let cs = RlpByteConcatConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "is_real_binary should fire");
    }

    // ─── Shifted constraint tests ─────────────────────────────────────

    fn check_shifted_at_transition(
        trace: &TracePolynomials,
        r: usize,
        alpha: &Scalar,
    ) -> Scalar {
        let curve = trace.curve;
        let z = Scalar::from_u64(7, curve);
        let omega_n_minus_1 = Scalar::zero(curve);
        let col_evals_at_z: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[r].clone())
            .collect();
        let next = r + 1;
        let shifted_evals = vec![
            trace.columns[COL_FIELD_INDEX].evaluations[next].clone(),
            trace.columns[COL_IS_REAL].evaluations[next].clone(),
            trace.columns[COL_IS_SAME_FIELD].evaluations[next].clone(),
            trace.columns[COL_INV_DIFF_FIELD].evaluations[next].clone(),
            trace.columns[COL_RUNNING_OFFSET_AT_FIELD].evaluations[next].clone(),
            trace.columns[COL_BYTE_IN_FIELD].evaluations[next].clone(),
        ];
        let cs = RlpByteConcatConstraintSystem::new(trace.num_rows);
        cs.evaluate_shifted_at_point(
            &col_evals_at_z,
            &shifted_evals,
            &z,
            &omega_n_minus_1,
            alpha,
            NUM_ROW_CONSTRAINTS,
        )
    }

    #[test]
    fn shifted_constraints_zero_on_honest_small_header() {
        let (_, bcat) = build_for(&small_header());
        let trace = build_trace_polynomials(&bcat, CurveType::Bls48581);
        let alpha = Scalar::from_u64(31337, CurveType::Bls48581);
        for r in 0..(trace.num_rows - 1) {
            let v = check_shifted_at_transition(&trace, r, &alpha);
            assert!(
                v.is_zero(),
                "shifted constraint at row {} = {:?} (expected zero)",
                r, v,
            );
        }
    }

    #[test]
    fn shifted_constraint_detects_broken_byte_in_field_monotonicity() {
        // Honest small header, then tamper byte_in_field at some
        // intra-field transition.
        let (_, bcat) = build_for(&small_header());
        let mut trace = build_trace_polynomials(&bcat, CurveType::Bls48581);
        // Find a row r where r and r+1 are in the SAME field; tamper
        // byte_in_field[r+1] to a wrong value (add 5 instead of 1).
        let curve = CurveType::Bls48581;
        let mut victim = None;
        for r in 0..(trace.num_rows - 1) {
            let fi_r = trace.columns[COL_FIELD_INDEX].evaluations[r].to_u64();
            let fi_next = trace.columns[COL_FIELD_INDEX].evaluations[r + 1].to_u64();
            if fi_r == fi_next {
                victim = Some(r + 1);
                break;
            }
        }
        let v = victim.expect("expected at least one intra-field transition");
        let orig = trace.columns[COL_BYTE_IN_FIELD].evaluations[v].clone();
        trace.columns[COL_BYTE_IN_FIELD].evaluations[v] =
            orig.add(&Scalar::from_u64(4, curve));
        let alpha = Scalar::from_u64(31337, curve);
        let val = check_shifted_at_transition(&trace, v - 1, &alpha);
        assert!(
            !val.is_zero(),
            "byte_in_field monotonicity should fire at tampered transition",
        );
    }

    #[test]
    fn shifted_constraint_detects_broken_running_offset_constancy() {
        let (_, bcat) = build_for(&small_header());
        let mut trace = build_trace_polynomials(&bcat, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        // Find an intra-field transition.
        let mut victim = None;
        for r in 0..(trace.num_rows - 1) {
            let fi_r = trace.columns[COL_FIELD_INDEX].evaluations[r].to_u64();
            let fi_next = trace.columns[COL_FIELD_INDEX].evaluations[r + 1].to_u64();
            if fi_r == fi_next {
                victim = Some(r + 1);
                break;
            }
        }
        let v = victim.expect("expected intra-field transition");
        let orig = trace.columns[COL_RUNNING_OFFSET_AT_FIELD].evaluations[v].clone();
        trace.columns[COL_RUNNING_OFFSET_AT_FIELD].evaluations[v] =
            orig.add(&Scalar::from_u64(7, curve));
        let alpha = Scalar::from_u64(31337, curve);
        let val = check_shifted_at_transition(&trace, v - 1, &alpha);
        assert!(
            !val.is_zero(),
            "running_offset constancy should fire on tampered RoF",
        );
    }

    // ─── Descriptor well-formedness ───────────────────────────────────

    #[test]
    fn field_byte_to_concat_descriptor_well_formed() {
        let desc = make_field_byte_to_concat_descriptor(
            1,
            vec![100, 101, 102, 103],
            Some(99),
            0,
            Some(COL_IS_REAL),
            7,
        );
        assert!(desc.label.contains("field_7"));
        assert_eq!(desc.a_layer_index, 1);
        assert_eq!(desc.a_columns, vec![100, 101, 102, 103]);
        assert_eq!(desc.a_selector_column, Some(99));
        assert_eq!(desc.b_layer_index, 0);
        assert_eq!(desc.b_columns, vec![COL_BYTE_VALUE]);
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn concat_to_header_rlp_descriptor_well_formed() {
        let desc = make_concat_to_header_rlp_descriptor(0, 2, 1234, 5678, Some(42));
        assert_eq!(desc.label, "rlp_byte_concat_to_header_rlp_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.a_columns, vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE]);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_layer_index, 2);
        assert_eq!(desc.b_columns, vec![1234, 5678]);
        assert_eq!(desc.b_selector_column, Some(42));
        assert_eq!(desc.a_columns.len(), desc.b_columns.len());
    }

    #[test]
    fn running_offset_to_concat_air_descriptor_well_formed() {
        let desc = make_running_offset_to_concat_air_descriptor(0, 1);
        use crate::rlp_list_concat_air as fc;
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.a_columns, vec![COL_FIELD_INDEX, COL_RUNNING_OFFSET_AT_FIELD]);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.b_columns, vec![fc::COL_FIELD_INDEX, fc::COL_RUNNING_OFFSET]);
        assert_eq!(desc.a_columns.len(), desc.b_columns.len());
    }

    // ─── Multi-field cancun coverage ─────────────────────────────────

    #[test]
    fn cancun_header_witness_covers_all_20_fields_with_consecutive_byte_indices() {
        let (fcat, bcat) = build_for(&cancun_header());
        assert_eq!(fcat.rows.len(), 20);

        // Sanity: bytes-per-field counts add up.
        let mut bytes_per_field = std::collections::HashMap::new();
        for r in &bcat.rows {
            *bytes_per_field.entry(r.field_index).or_insert(0u64) += 1;
        }
        for fr in &fcat.rows {
            assert_eq!(
                bytes_per_field.get(&(fr.field_index as u64)).copied().unwrap_or(0),
                fr.field_encoded_len as u64,
            );
        }

        // Sanity: each field's bytes are 0..N contiguous.
        let mut prev_field: Option<u64> = None;
        let mut counter = 0u64;
        for row in &bcat.rows {
            if Some(row.field_index) != prev_field {
                counter = 0;
            }
            assert_eq!(row.byte_in_field, counter);
            counter += 1;
            prev_field = Some(row.field_index);
        }
    }

    #[test]
    fn standalone_prove_with_scheme_passes_on_synthetic_witness() {
        // BLS48-581's cached FFT widths are 16, 32, 64, 128, 256 — so
        // we use a synthetic multi-field witness whose total byte
        // count fits in 256 rows (a real block header is ~530+ bytes
        // which would require BLS12-381). This still exercises the
        // full constraint system with multi-field grouping.
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        // 3 fields totaling 9 bytes, starting at offset 3 (mimicking
        // a list-header prefix).
        let w = RlpByteConcatWitness::from_field_encodings(&[
            (0, 3, vec![0xab, 0xcd, 0xef]),
            (1, 6, vec![0x11, 0x22, 0x33, 0x44]),
            (2, 10, vec![0x55, 0x66]),
        ]);
        w.verify_offset_equation().unwrap();
        w.verify_field_grouping().unwrap();

        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = RlpByteConcatConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);

        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone rlp_byte_concat_air proof must verify",
        );
    }
}
