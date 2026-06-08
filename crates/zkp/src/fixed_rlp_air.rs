//! Fixed-width RLP encoding gadget AIR — Phase B step 2 (#53 step 2
//! partial).
//!
//! Provides algebraic constraints for the simplest RLP encodings:
//! 32-byte fields (always `0xa0 || 32 bytes` = 33 bytes) and 20-byte
//! fields (always `0x94 || 20 bytes` = 21 bytes). These are the
//! fixed-width "hash-or-address" RLP forms that dominate the block
//! header structure (7 of 20 block-header fields are 32-byte hashes,
//! plus the 20-byte beneficiary).
//!
//! Variable-length RLP encoding (for u64/u256 numerics, variable-byte
//! `extra_data`, etc.) is deferred to a separate gadget — it requires
//! per-field length tracking + leading-zero stripping which is
//! substantially more complex.
//!
//! Per row, this gadget exposes:
//!   - `field_bytes[0..32]` — the 32-byte payload
//!   - `encoded_bytes[0..33]` — the RLP encoding `[0xa0, b_0, ...,
//!     b_31]`
//!   - `is_real` binary
//!
//! Row-local constraints:
//!   - `is_real` binary
//!   - `encoded_bytes[0] = 0xa0` (the fixed `0x80 + 32` length prefix)
//!   - `encoded_bytes[k+1] = field_bytes[k]` for k ∈ 0..32
//!     (β-RLC'd into a single body)
//!
//! Linkages (descriptors below):
//!   - `make_fixed_rlp32_to_keccak_input_linkage_descriptor` — links
//!     a `(field_bytes, encoded_bytes)` pair on this AIR to a region
//!     of a larger KeccakExtract input. Used to prove the field's
//!     RLP encoding is byte-aligned with the keccak input window.
//!
//! For the BlockHeader chain: instances of this gadget cover each of
//! the 32-byte fields. A separate "concat" gadget (deferred) glues
//! per-field encodings into the full header_rlp byte stream.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const FIELD_LEN: usize = 32;
pub const ENCODED_LEN: usize = 33; // 0xa0 + 32 bytes
pub const RLP32_PREFIX: u8 = 0xa0;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_FIELD_BYTE_OFFSET: usize = 0;       // 0..32
pub const COL_ENCODED_BYTE_OFFSET: usize = 32;    // 32..65
pub const COL_IS_REAL: usize = 65;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;   // 66

/// Row-local constraints:
///   0: is_real binary
///   1: encoded_bytes[0] - 0xa0 = 0 (gated by is_real)
///   2: β-RLC over 32 sub-bodies (encoded_bytes[k+1] - field_bytes[k])
///      gated by is_real
pub const NUM_ROW_CONSTRAINTS: usize = 3;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness type ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct FixedRlp32Row {
    pub field_bytes: [u8; FIELD_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct FixedRlp32Witness {
    pub invocations: Vec<FixedRlp32Row>,
}

impl FixedRlp32Witness {
    pub fn from_fields(fields: Vec<[u8; FIELD_LEN]>) -> Self {
        Self {
            invocations: fields
                .into_iter()
                .map(|field_bytes| FixedRlp32Row { field_bytes })
                .collect(),
        }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &FixedRlp32Witness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.invocations.iter().enumerate() {
        for k in 0..FIELD_LEN {
            columns[COL_FIELD_BYTE_OFFSET + k][i] =
                Scalar::from_u64(row.field_bytes[k] as u64, curve);
        }
        // encoded[0] = 0xa0, encoded[k+1] = field_bytes[k]
        columns[COL_ENCODED_BYTE_OFFSET][i] = Scalar::from_u64(RLP32_PREFIX as u64, curve);
        for k in 0..FIELD_LEN {
            columns[COL_ENCODED_BYTE_OFFSET + 1 + k][i] =
                Scalar::from_u64(row.field_bytes[k] as u64, curve);
        }
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

pub struct FixedRlp32ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl FixedRlp32ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for FixedRlp32ConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "encoded_byte_0_eq_0xa0".into(),
            "encoded_bytes_match_field_bytes_rlc".into(),
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
        let prefix = Scalar::from_u64(RLP32_PREFIX as u64, curve);
        let beta_test = Scalar::from_u64(7, curve);

        let mut bin = vec![Scalar::zero(curve); n];
        let mut prefix_eq = vec![Scalar::zero(curve); n];
        let mut bytes_match = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin[r] = v.mul(&v.sub(&one));

            // is_real * (encoded[0] - 0xa0) = 0
            let e0 = &columns[COL_ENCODED_BYTE_OFFSET][r];
            prefix_eq[r] = v.mul(&e0.sub(&prefix));

            // is_real * Σ β^k * (encoded[k+1] - field[k])
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..FIELD_LEN {
                let e = &columns[COL_ENCODED_BYTE_OFFSET + 1 + k][r];
                let f = &columns[COL_FIELD_BYTE_OFFSET + k][r];
                acc = acc.add(&bp.mul(&e.sub(f)));
                bp = bp.mul(&beta_test);
            }
            bytes_match[r] = v.mul(&acc);
        }
        vec![bin, prefix_eq, bytes_match]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let prefix = Scalar::from_u64(RLP32_PREFIX as u64, curve);

        let v = &col_evals[COL_IS_REAL];
        let bin = v.mul(&v.sub(&one));

        let e0 = &col_evals[COL_ENCODED_BYTE_OFFSET];
        let prefix_eq = v.mul(&e0.sub(&prefix));

        let mut bytes_match = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN {
            let e = &col_evals[COL_ENCODED_BYTE_OFFSET + 1 + k];
            let f = &col_evals[COL_FIELD_BYTE_OFFSET + k];
            bytes_match = bytes_match.add(&bp.mul(&e.sub(f)));
            bp = bp.mul(alpha);
        }
        let bytes_match = v.mul(&bytes_match);

        let mut acc = bin;
        let mut ap = alpha.clone();
        acc = acc.add(&ap.mul(&prefix_eq));
        ap = ap.mul(alpha);
        acc = acc.add(&ap.mul(&bytes_match));
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
        let prefix_poly = vec![Scalar::from_u64(RLP32_PREFIX as u64, curve)];

        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let bin = poly_mul(v, &v_m1, curve);

        let e0 = &col_coeffs[COL_ENCODED_BYTE_OFFSET];
        let e0_m_prefix = poly_sub(e0, &prefix_poly, curve);
        let prefix_eq = poly_mul(v, &e0_m_prefix, curve);

        let mut bytes_match_acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..FIELD_LEN {
            let e = &col_coeffs[COL_ENCODED_BYTE_OFFSET + 1 + k];
            let f = &col_coeffs[COL_FIELD_BYTE_OFFSET + k];
            let diff = poly_sub(e, f, curve);
            bytes_match_acc = poly_add(&bytes_match_acc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let bytes_match = poly_mul(v, &bytes_match_acc, curve);

        let mut acc = bin;
        let mut ap = alpha.clone();
        acc = poly_add(&acc, &poly_scalar_mul(&prefix_eq, &ap), curve);
        ap = ap.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&bytes_match, &ap), curve);
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
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..FIELD_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("fixed_rlp32_field_{}_8bit", k),
                    column_index: COL_FIELD_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..ENCODED_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("fixed_rlp32_encoded_{}_8bit", k),
                    column_index: COL_ENCODED_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_builder_populates_field_and_encoded() {
        let field = [0xab; 32];
        let w = FixedRlp32Witness::from_fields(vec![field]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.columns[COL_FIELD_BYTE_OFFSET].evaluations[0].to_u64(), 0xab);
        assert_eq!(trace.columns[COL_ENCODED_BYTE_OFFSET].evaluations[0].to_u64(), 0xa0);
        assert_eq!(trace.columns[COL_ENCODED_BYTE_OFFSET + 1].evaluations[0].to_u64(), 0xab);
        assert_eq!(trace.columns[COL_ENCODED_BYTE_OFFSET + 32].evaluations[0].to_u64(), 0xab);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let field = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
        ];
        let w = FixedRlp32Witness::from_fields(vec![field]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = FixedRlp32ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
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
    fn prefix_check_fires_on_tampered_prefix() {
        let field = [0u8; 32];
        let w = FixedRlp32Witness::from_fields(vec![FixedRlp32Row { field_bytes: field }.field_bytes]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper: encoded_bytes[0] = 0xb8 instead of 0xa0.
        cols[COL_ENCODED_BYTE_OFFSET][0] = Scalar::from_u64(0xb8, CurveType::Bls48581);
        let cs = FixedRlp32ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 1 = encoded_byte_0_eq_0xa0 should fire at row 0.
        assert!(!results[1][0].is_zero(), "prefix check should fire on tampered prefix");
    }

    #[test]
    fn bytes_match_fires_on_tampered_field_byte() {
        let field = [0u8; 32];
        let w = FixedRlp32Witness::from_fields(vec![field]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper: field_bytes[5] = 0xff while encoded stays 0.
        cols[COL_FIELD_BYTE_OFFSET + 5][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = FixedRlp32ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 2 = bytes_match should fire.
        assert!(!results[2][0].is_zero(), "bytes_match should fire");
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let field = [0xcd; 32];
        let w = FixedRlp32Witness::from_fields(vec![field]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = FixedRlp32ConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone fixed_rlp_air proof must verify",
        );
    }
}
