//! Data-availability sampling (DAS) / blob propagation AIR.
//!
//! Proves a single DAS cell witness: a (commitment, cell_index,
//! cell_value, cell_proof) tuple committing that
//!
//! ```text
//! KZG.verify(commitment, z = cell_index, y = cell_value, proof) = true
//! ```
//!
//! where the underlying curve is BLS12-381 and the polynomial
//! represented by `commitment` evaluates to `cell_value` at the
//! (canonicalized) root-of-unity index `cell_index`.
//!
//! In EIP-7594 / PeerDAS terminology each blob is split into
//! `CELLS_PER_EXT_BLOB = 128` cells of `FIELD_ELEMENTS_PER_CELL =
//! 64` field elements each. The KZG point-eval precompile that
//! checks any one cell is exactly the EIP-4844 precompile in
//! [`crate::kzg_point_eval_air`] with `z` set to the cell's
//! coset-shifted root-of-unity and `y` set to one element of the
//! cell — so a single DAS cell witness can be discharged by N
//! KZG point-eval AIR rows. This AIR commits the per-cell tuple
//! at the DAS level and binds it via cross-AIR LogUp to the
//! point-eval AIR which actually checks the equation.
//!
//! ## Per-row witness
//!
//! * `blob_commitment[0..48]`  — KZG commitment to the blob's
//!   polynomial (BLS12-381 G1, compressed).
//! * `cell_index`              — u64 index of the cell within the
//!   extended blob (0..=`2 * CELLS_PER_EXT_BLOB` − 1).
//! * `cell_value[0..32]`       — the field element at this cell
//!   index (`y`), BE-encoded.
//! * `cell_proof[0..48]`       — KZG proof π that
//!   `f(z_index) = y_value` (BLS12-381 G1, compressed).
//! * `is_valid`                — host-side selector; 1 iff
//!   `KZG.verify` accepted this tuple.
//!
//! ## Constraints
//!
//! 0. `is_real_binary`         — `is_real (is_real − 1) = 0`.
//! 1. `is_valid_binary`        — `is_valid (is_valid − 1) = 0`.
//! 2. `is_valid_gates_real`    — `is_valid (1 − is_real) = 0`.
//! 3. `cell_index_le_decomp`   — `cell_index − Σ CI_BYTE[b] · 2^(8b) = 0`.
//!
//! Bytes are 8-bit range-checked.
//!
//! ## Cross-AIR linkage
//!
//! * `(commitment, cell_index_be, cell_value, cell_proof)` ↔
//!   [`crate::kzg_point_eval_air`]. This AIR exposes the four
//!   per-cell field/group elements in the order the point-eval AIR
//!   commits them via its `commitment / z / y / proof` columns, so
//!   a single LogUp closure pins the entire tuple to a real KZG
//!   point-eval row whose internal pairing equation is already
//!   algebraically enforced.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

pub const COMMITMENT_LEN: usize = 48;
pub const PROOF_LEN: usize = 48;
pub const CELL_VALUE_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

/// EIP-7594 cells per extended blob.
pub const CELLS_PER_EXT_BLOB: u64 = 128;
/// EIP-7594 field elements per cell.
pub const FIELD_ELEMENTS_PER_CELL: u64 = 64;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_BLOB_COMMITMENT_OFFSET: usize = 0; // 0..48
pub const COL_CELL_VALUE_OFFSET: usize = COL_BLOB_COMMITMENT_OFFSET + COMMITMENT_LEN; // 48..80
pub const COL_CELL_PROOF_OFFSET: usize = COL_CELL_VALUE_OFFSET + CELL_VALUE_LEN; // 80..128
pub const COL_CELL_INDEX_BE_OFFSET: usize = COL_CELL_PROOF_OFFSET + PROOF_LEN; // 128..160

pub const COL_CELL_INDEX: usize = COL_CELL_INDEX_BE_OFFSET + CELL_VALUE_LEN; // 160
pub const COL_CELL_INDEX_BYTE_OFFSET: usize = COL_CELL_INDEX + 1; // 161..169

pub const COL_IS_VALID: usize = COL_CELL_INDEX_BYTE_OFFSET + U64_BYTES; // 169
pub const COL_IS_REAL: usize = COL_IS_VALID + 1; // 170

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 171
pub const NUM_ROW_CONSTRAINTS: usize = 4;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct DasCellRow {
    pub blob_commitment: [u8; COMMITMENT_LEN],
    pub cell_index: u64,
    pub cell_value: [u8; CELL_VALUE_LEN],
    pub cell_proof: [u8; PROOF_LEN],
    pub is_valid: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DasCellWitness {
    pub rows: Vec<DasCellRow>,
}

impl DasCellWitness {
    pub fn from_rows(rows: Vec<DasCellRow>) -> Self {
        Self { rows }
    }
}

/// Host-side single-row builder. Performs only a domain check on
/// `cell_index` (must be in the extended-blob range).
pub fn from_cell(
    blob_commitment: [u8; COMMITMENT_LEN],
    cell_index: u64,
    cell_value: [u8; CELL_VALUE_LEN],
    cell_proof: [u8; PROOF_LEN],
    is_valid: bool,
) -> DasCellWitness {
    assert!(
        cell_index < 2 * CELLS_PER_EXT_BLOB,
        "DAS cell_index {} out of range (max {})",
        cell_index,
        2 * CELLS_PER_EXT_BLOB - 1,
    );
    DasCellWitness {
        rows: vec![DasCellRow {
            blob_commitment,
            cell_index,
            cell_value,
            cell_proof,
            is_valid,
            is_real: true,
        }],
    }
}

/// Encode `cell_index` as a 32-byte big-endian scalar (the `z`
/// input of the point-eval precompile).
pub fn cell_index_to_z_be(cell_index: u64) -> [u8; CELL_VALUE_LEN] {
    let mut z = [0u8; CELL_VALUE_LEN];
    z[CELL_VALUE_LEN - U64_BYTES..].copy_from_slice(&cell_index.to_be_bytes());
    z
}

// ─── Trace builder ────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < 8);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

fn write_le_bytes(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    value: u64,
    row: usize,
    curve: CurveType,
) {
    let bytes = value.to_le_bytes();
    for b in 0..U64_BYTES {
        columns[offset + b][row] = Scalar::from_u64(bytes[b] as u64, curve);
    }
}

pub fn build_trace_polynomials(witness: &DasCellWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..COMMITMENT_LEN {
            columns[COL_BLOB_COMMITMENT_OFFSET + k][i] =
                Scalar::from_u64(row.blob_commitment[k] as u64, curve);
            columns[COL_CELL_PROOF_OFFSET + k][i] =
                Scalar::from_u64(row.cell_proof[k] as u64, curve);
        }
        for k in 0..CELL_VALUE_LEN {
            columns[COL_CELL_VALUE_OFFSET + k][i] =
                Scalar::from_u64(row.cell_value[k] as u64, curve);
        }
        let z_be = cell_index_to_z_be(row.cell_index);
        for k in 0..CELL_VALUE_LEN {
            columns[COL_CELL_INDEX_BE_OFFSET + k][i] = Scalar::from_u64(z_be[k] as u64, curve);
        }
        columns[COL_CELL_INDEX][i] = Scalar::from_u64(row.cell_index, curve);
        write_le_bytes(&mut columns, COL_CELL_INDEX_BYTE_OFFSET, row.cell_index, i, curve);
        columns[COL_IS_VALID][i] = if row.is_valid { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = if row.is_real { one.clone() } else { zero.clone() };
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct DasCellConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl DasCellConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn sum_le_bytes(col_evals: &[Scalar], offset: usize, curve: CurveType) -> Scalar {
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[offset + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    sum
}

fn sum_le_bytes_poly(
    col_coeffs: &[Vec<Scalar>],
    offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[offset + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    sum
}

impl VmConstraintSystem for DasCellConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_valid_binary".into(),
            "is_valid_gates_real".into(),
            "cell_index_le_decomp".into(),
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
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_valid = &row_evals[COL_IS_VALID];
            let cell_index = &row_evals[COL_CELL_INDEX];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_valid.mul(&is_valid.sub(&one));
            bodies[2][row] = is_valid.mul(&one.sub(is_real));
            let ci_sum = sum_le_bytes(&row_evals, COL_CELL_INDEX_BYTE_OFFSET, curve);
            bodies[3][row] = is_real.mul(&cell_index.sub(&ci_sum));
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];
        let is_valid = &col_evals[COL_IS_VALID];
        let cell_index = &col_evals[COL_CELL_INDEX];
        let ci_sum = sum_le_bytes(col_evals, COL_CELL_INDEX_BYTE_OFFSET, curve);

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_valid.mul(&is_valid.sub(&one)),
            is_valid.mul(&one.sub(is_real)),
            is_real.mul(&cell_index.sub(&ci_sum)),
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
        let is_valid = &col_coeffs[COL_IS_VALID];
        let cell_index = &col_coeffs[COL_CELL_INDEX];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve));
        bodies.push(poly_mul(is_valid, &poly_sub(is_valid, &one_poly, curve), curve));
        bodies.push(poly_mul(is_valid, &poly_sub(&one_poly, is_real, curve), curve));
        let ci_sum = sum_le_bytes_poly(col_coeffs, COL_CELL_INDEX_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(cell_index, &ci_sum, curve), curve));

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
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for (off, len, label) in [
            (COL_BLOB_COMMITMENT_OFFSET, COMMITMENT_LEN, "blob_commitment_byte"),
            (COL_CELL_VALUE_OFFSET, CELL_VALUE_LEN, "cell_value_byte"),
            (COL_CELL_PROOF_OFFSET, PROOF_LEN, "cell_proof_byte"),
            (COL_CELL_INDEX_BE_OFFSET, CELL_VALUE_LEN, "cell_index_be_byte"),
            (COL_CELL_INDEX_BYTE_OFFSET, U64_BYTES, "cell_index_le_byte"),
        ] {
            for k in 0..len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptor ───────────────────────────────────────

/// Bind `(commitment, z = cell_index_be, y = cell_value, proof)` of
/// this AIR against [`crate::kzg_point_eval_air`]'s
/// `(COL_COMMITMENT_OFFSET, COL_Z_OFFSET, COL_Y_OFFSET,
/// COL_PROOF_OFFSET)`. Gated by `IS_VALID` on this side and
/// `(COL_IS_REAL ∧ COL_IS_SUCCESS)` on the point-eval side. The
/// resulting LogUp tuple has 48 + 32 + 32 + 48 = 160 columns.
///
/// This is the cryptographic seal of DAS sampling: tampering any
/// of the four DAS witness fields will not match a real point-eval
/// AIR row, breaking the LogUp closure.
pub fn make_das_to_kzg_point_eval_descriptor(
    das_layer_index: usize,
    kzg_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::kzg_point_eval_air as kpe;

    let mut a_columns: Vec<usize> = Vec::with_capacity(
        COMMITMENT_LEN + CELL_VALUE_LEN + CELL_VALUE_LEN + PROOF_LEN,
    );
    a_columns.extend((0..COMMITMENT_LEN).map(|k| COL_BLOB_COMMITMENT_OFFSET + k));
    a_columns.extend((0..CELL_VALUE_LEN).map(|k| COL_CELL_INDEX_BE_OFFSET + k));
    a_columns.extend((0..CELL_VALUE_LEN).map(|k| COL_CELL_VALUE_OFFSET + k));
    a_columns.extend((0..PROOF_LEN).map(|k| COL_CELL_PROOF_OFFSET + k));

    let mut b_columns: Vec<usize> = Vec::with_capacity(a_columns.len());
    b_columns.extend((0..kpe::G1_LEN).map(|k| kpe::COL_COMMITMENT_OFFSET + k));
    b_columns.extend((0..kpe::SCALAR_LEN).map(|k| kpe::COL_Z_OFFSET + k));
    b_columns.extend((0..kpe::SCALAR_LEN).map(|k| kpe::COL_Y_OFFSET + k));
    b_columns.extend((0..kpe::G1_LEN).map(|k| kpe::COL_PROOF_OFFSET + k));

    CrossAirLogUpDescriptor {
        label: "das_cell_to_kzg_point_eval_v1".into(),
        a_layer_index: das_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_VALID),
        b_layer_index: kzg_layer_index,
        b_columns,
        b_selector_column: Some(kpe::COL_IS_SUCCESS),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_commitment(seed: u8) -> [u8; COMMITMENT_LEN] {
        let mut c = [0u8; COMMITMENT_LEN];
        for k in 0..COMMITMENT_LEN {
            c[k] = seed.wrapping_add(k as u8);
        }
        c
    }

    fn make_value(seed: u8) -> [u8; CELL_VALUE_LEN] {
        let mut v = [0u8; CELL_VALUE_LEN];
        for k in 0..CELL_VALUE_LEN {
            v[k] = seed.wrapping_add(k as u8);
        }
        v
    }

    fn make_proof(seed: u8) -> [u8; PROOF_LEN] {
        let mut p = [0u8; PROOF_LEN];
        for k in 0..PROOF_LEN {
            p[k] = seed.wrapping_add(k as u8);
        }
        p
    }

    fn evaluate_bodies(
        witness: &DasCellWitness,
        curve: CurveType,
    ) -> (TracePolynomials, Vec<Vec<Scalar>>) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = DasCellConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        (trace, bodies)
    }

    fn assert_all_vanish(bodies: &[Vec<Scalar>]) {
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} should vanish at row {} (got {:?})",
                    k,
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    #[test]
    fn build_poly_matches_evaluate_at_point() {
        use crate::commitment;
        use bls48581::bls48581::big;
        let w = from_cell(
            make_commitment(0x11),
            17,
            make_value(0x22),
            make_proof(0x33),
            true,
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = DasCellConstraintSystem::new(trace.num_rows);
        let domain_size = trace.padded_size;

        let col_coeffs: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| {
            let big_evals: Vec<big::BIG> = p.evaluations.iter()
                .map(|s| big::BIG::new_copy(s.as_bls48581()))
                .collect();
            let coeffs = commitment::eval_to_coeff(&big_evals, domain_size);
            coeffs.into_iter().map(Scalar::Bls48581).collect()
        }).collect();

        let z_big = big::BIG::new_int(123456789);
        let alpha_big = big::BIG::new_int(987654321);
        let alpha = Scalar::Bls48581(big::BIG::new_copy(&alpha_big));

        let col_evals_at_z: Vec<Scalar> = col_coeffs.iter().map(|coeffs| {
            let big_coeffs: Vec<big::BIG> = coeffs.iter()
                .map(|s| big::BIG::new_copy(s.as_bls48581()))
                .collect();
            Scalar::Bls48581(commitment::eval_poly_at(&big_coeffs, &z_big))
        }).collect();

        let c_at_z_eval = cs.evaluate_at_point(&col_evals_at_z, &alpha);

        let c_poly = cs.build_constraint_polynomial(&col_coeffs, &alpha, domain_size);
        let c_poly_big: Vec<big::BIG> = c_poly.iter()
            .map(|s| big::BIG::new_copy(s.as_bls48581()))
            .collect();
        let c_at_z_poly = Scalar::Bls48581(commitment::eval_poly_at(&c_poly_big, &z_big));

        assert_eq!(
            c_at_z_eval.as_bls48581().tostring(),
            c_at_z_poly.as_bls48581().tostring(),
            "data_availability_sampling_air: build_constraint_polynomial(z) must equal evaluate_at_point",
        );
    }

    #[test]
    fn honest_das_cell_vanishes() {
        let curve = CurveType::Bls48581;
        let w = from_cell(
            make_commitment(0x11),
            17,
            make_value(0x22),
            make_proof(0x33),
            true,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn cell_index_be_matches_le_decomp() {
        let curve = CurveType::Bls48581;
        let w = from_cell(
            make_commitment(0),
            0xab,
            make_value(0),
            make_proof(0),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        // Last byte of BE should be low byte of u64.
        let last_be = trace.columns[COL_CELL_INDEX_BE_OFFSET + CELL_VALUE_LEN - 1]
            .evaluations[0]
            .to_u64();
        let low_le =
            trace.columns[COL_CELL_INDEX_BYTE_OFFSET].evaluations[0].to_u64();
        assert_eq!(last_be, low_le);
        assert_eq!(low_le, 0xab);
    }

    #[test]
    fn tampered_cell_index_decomp_detected() {
        let curve = CurveType::Bls48581;
        let w = from_cell(
            make_commitment(0),
            42,
            make_value(0),
            make_proof(0),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_CELL_INDEX][0] = Scalar::from_u64(99, curve);
        let cs = DasCellConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "cell_index_le_decomp must fire on mismatched scalar"
        );
    }

    #[test]
    fn is_valid_gating_is_real_detected() {
        let curve = CurveType::Bls48581;
        let w = from_cell(
            make_commitment(0),
            0,
            make_value(0),
            make_proof(0),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::zero(curve);
        cols[COL_IS_VALID][0] = Scalar::one(curve);
        let cs = DasCellConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "is_valid_gates_real should fire when is_valid=1 but is_real=0"
        );
    }

    #[test]
    fn descriptor_columns_aligned_with_kzg_point_eval() {
        let d = make_das_to_kzg_point_eval_descriptor(0, 1);
        assert_eq!(d.label, "das_cell_to_kzg_point_eval_v1");
        // 48 (commitment) + 32 (z) + 32 (y) + 48 (proof) = 160.
        assert_eq!(d.a_columns.len(), 160);
        assert_eq!(d.b_columns.len(), 160);
        assert_eq!(d.a_selector_column, Some(COL_IS_VALID));
        assert_eq!(
            d.b_selector_column,
            Some(crate::kzg_point_eval_air::COL_IS_SUCCESS),
        );
        // First column should be commitment[0] on both sides.
        assert_eq!(d.a_columns[0], COL_BLOB_COMMITMENT_OFFSET);
        assert_eq!(
            d.b_columns[0],
            crate::kzg_point_eval_air::COL_COMMITMENT_OFFSET,
        );
    }

    #[test]
    fn column_layout_pinned() {
        // 48 (commitment) + 32 (value) + 48 (proof) + 32 (z_be)
        // + 1+8 (cell_index) + 1 (is_valid) + 1 (is_real) = 171.
        assert_eq!(NUM_COLUMNS, 171);
        assert_eq!(NUM_ROW_CONSTRAINTS, 4);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(CELLS_PER_EXT_BLOB, 128);
        assert_eq!(FIELD_ELEMENTS_PER_CELL, 64);
    }

    #[test]
    fn evaluate_at_point_matches_for_honest() {
        let curve = CurveType::Bls48581;
        let w = from_cell(
            make_commitment(1),
            5,
            make_value(2),
            make_proof(3),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let cs = DasCellConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(29, curve);
        let row0_evals: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    #[test]
    #[ignore = "slow: data_availability_sampling_air standalone prove_with_scheme + verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let w = from_cell(
            make_commitment(0x11),
            17,
            make_value(0x22),
            make_proof(0x33),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let mut cs = DasCellConstraintSystem::new(trace.num_rows);
        cs.omega = Some(omega);
        cs.domain_size = Some(trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone data_availability_sampling_air proof must verify",
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn host_panics_on_out_of_range_cell_index() {
        let _ = from_cell(
            make_commitment(0),
            2 * CELLS_PER_EXT_BLOB, // first invalid index
            make_value(0),
            make_proof(0),
            true,
        );
    }
}
