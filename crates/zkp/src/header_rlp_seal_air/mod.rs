//! Header-RLP byte-seal AIR — the closing variable-byte seal between
//! [`crate::rlp_byte_concat_air`] (per-field per-byte view assembled
//! from the per-field RLP gadgets) and
//! [`crate::block_header_air`]'s `header_rlp[]` byte columns.
//!
//! # Motivation
//!
//! The full algebraic chain for `block_hash = keccak256(rlp(header))`
//! is composed of:
//!
//!   per-field RLP gadgets  (u64_rlp_air, u256_rlp_air, fixed_rlp_air,
//!                           rlp_logs_bloom_air, rlp_var_bytes_air, …)
//!     ↓ (per-field byte multiset)
//!   rlp_byte_concat_air      (per-byte rows with running_offset)
//!     ↓
//!   (THIS AIR — header_rlp_seal_air)
//!     ↓
//!   block_header_air.header_rlp[0..768] byte columns
//!     ↓
//!   keccak_extract_wide       (already wired)
//!     ↓
//!   block_hash
//!
//! What was missing was a *byte-row view* whose rows can be matched
//! against `block_header_air`'s 768 *column-positional* header_rlp
//! bytes. `rlp_byte_concat_air` already commits per-byte rows with
//! `(field_index, byte_in_field, absolute_offset, byte_value,
//! running_offset_at_field)`, but its tuple shape carries field-local
//! information that has no counterpart in `block_header_air`. This
//! seal AIR strips that away and exposes the *flat* per-byte view
//! `(absolute_offset, byte_value, is_real)` that the
//! variable-length header_rlp window publishes per position.
//!
//! # Algebraic constraints
//!
//! Row-local:
//!   0. `is_real * (is_real - 1) = 0`                          (binary)
//!   1. `byte_value ∈ [0, 256)`                                (lookup)
//!
//! Cross-row (shifted):
//!   2. `is_real(z) * is_real(ω·z) *
//!         (absolute_offset(ω·z) - absolute_offset(z) - 1) = 0` (strict
//!         monotonicity by 1 — bytes form a contiguous 0..len enumeration)
//!
//! Combined with the boundary condition `is_real[0] = 1 ⇒
//! absolute_offset[0] = 0` (enforced host-side by the trace builder;
//! optional algebraic enforcement deferred to a follow-up), this pins
//! the row enumeration to be the canonical sequence
//! `(0, byte0), (1, byte1), …, (len-1, byte_{len-1})` of the actual
//! header_rlp byte stream.
//!
//! # Cross-AIR LogUp descriptors
//!
//! 1. [`make_seal_from_concat_descriptor`]: A-side =
//!    `rlp_byte_concat_air` `(absolute_offset, byte_value)` rows;
//!    B-side = this AIR's `(absolute_offset, byte_value)` rows. Multi-
//!    set equality binds the seal's per-byte view to the concat's
//!    per-byte view across all fields.
//! 2. [`make_seal_to_header_rlp_descriptor`]: A-side = this AIR's
//!    `(absolute_offset, byte_value)` rows; B-side =
//!    `block_header_air`'s `(position_literal_k, header_rlp[k])`
//!    columns for `k ∈ [0, HEADER_RLP_MAX_LEN)`. Like the existing
//!    `make_concat_to_header_rlp_descriptor`, this requires a small
//!    companion "header_rlp byte-row" adapter that unfolds the 768
//!    columns of `block_header_air` into 768 rows of `(position,
//!    byte)`; the descriptor returns the canonical tuple shape and
//!    the caller wires `b_layer_index` to the adapter.

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

pub const COL_ABSOLUTE_OFFSET: usize = 0;
pub const COL_BYTE_VALUE: usize = 1;
pub const COL_IS_REAL: usize = 2;

pub const NUM_COLUMNS: usize = 3;

/// Row-local constraints exposed by [`HeaderRlpSealConstraintSystem`].
///   0. `is_real * (is_real - 1) = 0`     (binary)
///
/// The 8-bit byte range check is published separately via
/// `lookup_declarations`.
pub const NUM_ROW_CONSTRAINTS: usize = 1;

/// Cross-row (shifted) constraints (1 total):
///   1. `is_real(z) * is_real(ω·z) *
///        (absolute_offset(ω·z) - absolute_offset(z) - 1) = 0`
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct HeaderRlpSealRow {
    pub absolute_offset: u64,
    pub byte_value: u8,
}

#[derive(Clone, Debug, Default)]
pub struct HeaderRlpSealWitness {
    pub rows: Vec<HeaderRlpSealRow>,
}

impl HeaderRlpSealWitness {
    /// Build the seal witness from the canonical header_rlp byte
    /// stream. Emits one row per byte at positions
    /// `(0, bytes[0]), (1, bytes[1]), …, (len-1, bytes[len-1])`.
    ///
    /// Maximum supported length: `block_header_air::HEADER_RLP_MAX_LEN`
    /// (768). Longer inputs are accepted but only the leading
    /// `HEADER_RLP_MAX_LEN` bytes are emitted; the caller is responsible
    /// for validating that the actual header fits.
    pub fn from_header_rlp_bytes(bytes: &[u8]) -> Self {
        let max = crate::block_header_air::HEADER_RLP_MAX_LEN;
        let len = bytes.len().min(max);
        let rows = bytes
            .iter()
            .take(len)
            .enumerate()
            .map(|(i, &b)| HeaderRlpSealRow {
                absolute_offset: i as u64,
                byte_value: b,
            })
            .collect();
        Self { rows }
    }

    /// Host-side cross-check: the rows enumerate consecutive offsets
    /// starting at 0, and their byte values match the canonical input.
    pub fn verify_against_canonical(&self, canonical: &[u8]) -> Result<(), String> {
        for (r, row) in self.rows.iter().enumerate() {
            if row.absolute_offset as usize != r {
                return Err(format!(
                    "row {} absolute_offset = {} (expected {})",
                    r, row.absolute_offset, r,
                ));
            }
            let pos = row.absolute_offset as usize;
            if pos >= canonical.len() {
                return Err(format!(
                    "row {} absolute_offset {} exceeds canonical len {}",
                    r, pos, canonical.len(),
                ));
            }
            if canonical[pos] != row.byte_value {
                return Err(format!(
                    "row {} byte at offset {} = {:#04x} but canonical = {:#04x}",
                    r, pos, row.byte_value, canonical[pos],
                ));
            }
        }
        Ok(())
    }

    /// Host-side cross-check: strict monotonicity of `absolute_offset`
    /// — each row's offset is exactly one greater than the previous.
    pub fn verify_strict_monotonic(&self) -> Result<(), String> {
        for r in 1..self.rows.len() {
            let prev = self.rows[r - 1].absolute_offset;
            let cur = self.rows[r].absolute_offset;
            if cur != prev + 1 {
                return Err(format!(
                    "row {} absolute_offset = {} not consecutive with prev = {}",
                    r, cur, prev,
                ));
            }
        }
        Ok(())
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &HeaderRlpSealWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_ABSOLUTE_OFFSET][i] = Scalar::from_u64(row.absolute_offset, curve);
        columns[COL_BYTE_VALUE][i] = Scalar::from_u64(row.byte_value as u64, curve);
        columns[COL_IS_REAL][i] = one.clone();
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

pub struct HeaderRlpSealConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl HeaderRlpSealConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for HeaderRlpSealConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec!["is_real_binary".into()]
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
        for r in 0..n {
            let v_real = &columns[COL_IS_REAL][r];
            bin_real[r] = v_real.mul(&v_real.sub(&one));
        }
        vec![bin_real]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v_real = &col_evals[COL_IS_REAL];
        let _ = alpha;
        v_real.mul(&v_real.sub(&one))
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
        poly_mul(v_real, &v_real_m1, curve)
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
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
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
                label: "header_rlp_seal_byte_value_8bit".into(),
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
        // Referenced at ω·z in this exact order:
        //   0: IS_REAL
        //   1: ABSOLUTE_OFFSET
        vec![COL_IS_REAL, COL_ABSOLUTE_OFFSET]
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
        if shifted_evals.len() < 2 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let is_real_wz = &shifted_evals[0];
        let off_wz = &shifted_evals[1];

        let is_real_z = &col_evals_at_z[COL_IS_REAL];
        let off_z = &col_evals_at_z[COL_ABSOLUTE_OFFSET];

        // Strict monotonicity: gated by is_real(z) * is_real(ω·z) so
        // the constraint is excused when either side is a padding row.
        let off_step = off_wz.sub(off_z).sub(&one);
        let body = is_real_z.mul(is_real_wz).mul(&off_step);

        let exclusion = z.sub(omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&body).mul(&exclusion)
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
        let off = &col_coeffs[COL_ABSOLUTE_OFFSET];

        let is_real_shift = poly_shift(is_real, omega);
        let off_shift = poly_shift(off, omega);

        let off_step = poly_sub(&poly_sub(&off_shift, off, curve), &one_poly, curve);
        let body = poly_mul(
            &poly_mul(is_real, &is_real_shift, curve),
            &off_step,
            curve,
        );

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let ex = poly_mul_linear(&body, &omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term = poly_scalar_mul(&ex, &ap);
        poly_add(&term, &vec![Scalar::zero(curve)], curve)
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind this AIR's `(absolute_offset, byte_value)` rows ↔
/// [`crate::rlp_byte_concat_air`]'s `(COL_ABSOLUTE_OFFSET,
/// COL_BYTE_VALUE)` rows. Multiset equality at the byte level: every
/// `(offset, byte)` tuple emitted by the per-field byte concat must
/// appear (with equal multiplicity) in the seal AIR, and vice versa.
///
/// Composed with the rlp_byte_concat AIR's `running_offset` constancy
/// and `byte_in_field` monotonicity (and its own running-offset chain
/// to `rlp_list_concat_air`), this propagates each per-field gadget's
/// encoded bytes through to the seal AIR at the correct absolute
/// position.
pub fn make_seal_from_concat_descriptor(
    seal_layer_index: usize,
    concat_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::rlp_byte_concat_air as cat;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "header_rlp_seal_from_byte_concat_v1".into(),
        a_layer_index: concat_layer_index,
        a_columns: vec![cat::COL_ABSOLUTE_OFFSET, cat::COL_BYTE_VALUE],
        a_selector_column: Some(cat::COL_IS_REAL),
        b_layer_index: seal_layer_index,
        b_columns: vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE],
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Bind this AIR's `(absolute_offset, byte_value)` rows ↔
/// [`crate::block_header_air`]'s `(position_literal_k, header_rlp[k])`
/// columns for `k ∈ [0, HEADER_RLP_MAX_LEN)`.
///
/// # Shape contract
///
/// The descriptor publishes a 2-column tuple on each side. On the
/// block_header side, the natural shape is "768 columns in one row";
/// to match the seal AIR's per-row tuple shape, the caller wires
/// `b_layer_index` to a small companion "header_rlp byte-row" adapter
/// AIR whose rows are `(position, byte_value, is_real)` derived
/// row-by-row from the 768 header_rlp columns of `block_header_air`.
///
/// The function returns the canonical 2-column descriptor; callers
/// are expected to provide that adapter (or wire `b_columns` to
/// equivalent columns on a custom AIR — the `header_rlp_position_col`
/// and `header_rlp_byte_value_col` parameters expose the relevant
/// column indices).
///
/// **Soundness note**: combined with the seal AIR's strict offset
/// monotonicity by 1 (shifted constraint #1) and the boundary
/// `absolute_offset[0] = 0` enforced host-side by the trace builder,
/// the seal rows form a deterministic 0..len enumeration. Multiset
/// equality on `(offset, byte)` against the header_rlp row-view
/// adapter then pins each byte at the correct absolute position in
/// the header_rlp window.
pub fn make_seal_to_header_rlp_descriptor(
    seal_layer_index: usize,
    header_rlp_byte_row_layer_index: usize,
    header_rlp_position_col: usize,
    header_rlp_byte_value_col: usize,
    header_rlp_selector_col: Option<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "header_rlp_seal_to_header_rlp_v1".into(),
        a_layer_index: seal_layer_index,
        a_columns: vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: header_rlp_byte_row_layer_index,
        b_columns: vec![header_rlp_position_col, header_rlp_byte_value_col],
        b_selector_column: header_rlp_selector_col,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_header::{block_header_rlp, BlockHeader};

    fn small_header() -> BlockHeader {
        BlockHeader {
            number: 100,
            gas_limit: 1_000_000,
            ..Default::default()
        }
    }

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

    // ─── Witness-level sanity ─────────────────────────────────────────

    #[test]
    fn witness_builds_from_small_header_rlp() {
        let rlp = block_header_rlp(&small_header());
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        assert_eq!(w.rows.len(), rlp.len());
        for (r, row) in w.rows.iter().enumerate() {
            assert_eq!(row.absolute_offset, r as u64);
            assert_eq!(row.byte_value, rlp[r]);
        }
        w.verify_against_canonical(&rlp).unwrap();
        w.verify_strict_monotonic().unwrap();
    }

    #[test]
    fn witness_builds_from_full_cancun_header_rlp() {
        let rlp = block_header_rlp(&cancun_header());
        // Real Cancun header should be a few hundred bytes (well above
        // the small_header case, exercising larger-row paths).
        assert!(rlp.len() > 400, "cancun rlp suspiciously short: {}", rlp.len());
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        assert_eq!(w.rows.len(), rlp.len());
        w.verify_against_canonical(&rlp).unwrap();
        w.verify_strict_monotonic().unwrap();
    }

    #[test]
    fn witness_caps_at_header_rlp_max_len() {
        let bytes: Vec<u8> = (0..1000u16).map(|i| (i & 0xff) as u8).collect();
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&bytes);
        assert_eq!(w.rows.len(), crate::block_header_air::HEADER_RLP_MAX_LEN);
        for (r, row) in w.rows.iter().enumerate() {
            assert_eq!(row.absolute_offset, r as u64);
            assert_eq!(row.byte_value, bytes[r]);
        }
    }

    #[test]
    fn host_side_detects_tampered_byte() {
        let rlp = block_header_rlp(&cancun_header());
        let mut w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        // Pick a row guaranteed to be inside the witness (rlp.len() > 100).
        let victim = 73usize;
        w.rows[victim].byte_value ^= 0xff;
        let err = w.verify_against_canonical(&rlp).unwrap_err();
        assert!(err.contains(&format!("row {}", victim)), "got: {}", err);
    }

    #[test]
    fn host_side_detects_tampered_offset() {
        let rlp = block_header_rlp(&small_header());
        let mut w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        // Tamper monotonicity: skip an offset.
        w.rows[5].absolute_offset += 1;
        let err = w.verify_strict_monotonic().unwrap_err();
        assert!(err.contains("row 5"), "got: {}", err);
    }

    // ─── Algebraic constraint tests ───────────────────────────────────

    fn run_evaluate_on_domain(witness: &HeaderRlpSealWitness) -> Vec<Vec<Scalar>> {
        let trace = build_trace_polynomials(witness, CurveType::Bls48581);
        let cs = HeaderRlpSealConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    #[test]
    fn row_local_constraints_zero_on_honest_small_header() {
        let rlp = block_header_rlp(&small_header());
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        let results = run_evaluate_on_domain(&w);
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
    fn row_local_constraints_zero_on_honest_cancun_header() {
        let rlp = block_header_rlp(&cancun_header());
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        let results = run_evaluate_on_domain(&w);
        for col in results.iter() {
            for val in col.iter() {
                assert!(val.is_zero(), "constraint nonzero on cancun header");
            }
        }
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_value() {
        let rlp = block_header_rlp(&small_header());
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(3, CurveType::Bls48581);
        let cs = HeaderRlpSealConstraintSystem::new(trace.num_rows);
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
            trace.columns[COL_IS_REAL].evaluations[next].clone(),
            trace.columns[COL_ABSOLUTE_OFFSET].evaluations[next].clone(),
        ];
        let cs = HeaderRlpSealConstraintSystem::new(trace.num_rows);
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
        let rlp = block_header_rlp(&small_header());
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
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
    fn shifted_constraint_detects_broken_offset_monotonicity() {
        let rlp = block_header_rlp(&small_header());
        let w = HeaderRlpSealWitness::from_header_rlp_bytes(&rlp);
        let mut trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        // Tamper row 4's absolute_offset by adding 7 so the
        // transition (3 → 4) breaks the strict-by-1 rule.
        let orig = trace.columns[COL_ABSOLUTE_OFFSET].evaluations[4].clone();
        trace.columns[COL_ABSOLUTE_OFFSET].evaluations[4] =
            orig.add(&Scalar::from_u64(7, curve));
        let alpha = Scalar::from_u64(31337, curve);
        let v = check_shifted_at_transition(&trace, 3, &alpha);
        assert!(
            !v.is_zero(),
            "monotonicity-by-1 should fire on tampered offset",
        );
    }

    // ─── Descriptor well-formedness ───────────────────────────────────

    #[test]
    fn seal_from_concat_descriptor_well_formed() {
        use crate::rlp_byte_concat_air as cat;
        let desc = make_seal_from_concat_descriptor(0, 1);
        assert_eq!(desc.label, "header_rlp_seal_from_byte_concat_v1");
        assert_eq!(desc.a_layer_index, 1);
        assert_eq!(
            desc.a_columns,
            vec![cat::COL_ABSOLUTE_OFFSET, cat::COL_BYTE_VALUE],
        );
        assert_eq!(desc.a_selector_column, Some(cat::COL_IS_REAL));
        assert_eq!(desc.b_layer_index, 0);
        assert_eq!(desc.b_columns, vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE]);
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.a_columns.len(), desc.b_columns.len());
    }

    #[test]
    fn seal_to_header_rlp_descriptor_well_formed() {
        let desc = make_seal_to_header_rlp_descriptor(0, 2, 1234, 5678, Some(42));
        assert_eq!(desc.label, "header_rlp_seal_to_header_rlp_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.a_columns, vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE]);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_layer_index, 2);
        assert_eq!(desc.b_columns, vec![1234, 5678]);
        assert_eq!(desc.b_selector_column, Some(42));
        assert_eq!(desc.a_columns.len(), desc.b_columns.len());
    }

    // ─── Column layout pinning ────────────────────────────────────────

    #[test]
    fn column_layout_pinned() {
        // Pin the column indices so downstream descriptors (and
        // joint_prove wirings) catch accidental column-layout changes.
        assert_eq!(COL_ABSOLUTE_OFFSET, 0);
        assert_eq!(COL_BYTE_VALUE, 1);
        assert_eq!(COL_IS_REAL, 2);
        assert_eq!(NUM_COLUMNS, 3);
        assert_eq!(NUM_ROW_CONSTRAINTS, 1);
        assert_eq!(NUM_SHIFTED, 1);
    }
}
