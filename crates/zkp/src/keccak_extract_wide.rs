//! Wide-input Keccak-256 extraction AIR (`MAX_INPUT_LEN = 768`).
//!
//! Variant of [`crate::keccak_extract`] that supports inputs up to 768
//! bytes, covering real Ethereum block headers (~600 bytes due to the
//! 256-byte `logs_bloom` field). The original 256-byte variant remains
//! for MPT leaves, SHA3 preimages, and other short inputs.
//!
//! Column layout: `INPUT_BYTE[0..768] + INPUT_LEN + OUTPUT_BYTE[0..32]
//! + IS_REAL = 802 cols`.
//!
//! Omits the address-limb and keccak-output-limb aggregator columns
//! from `keccak_extract` (those are CREATE/CREATE2-specific and not
//! needed for block header keccak binding).

use crate::field::{CurveType, Scalar};
use crate::keccak::keccak256;
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const MAX_INPUT_LEN: usize = 768;
pub const OUTPUT_LEN: usize = 32;

pub const COL_INPUT_BYTE_OFFSET: usize = 0;
pub const COL_INPUT_LEN: usize = COL_INPUT_BYTE_OFFSET + MAX_INPUT_LEN;
pub const COL_OUTPUT_BYTE_OFFSET: usize = COL_INPUT_LEN + 1;
pub const COL_IS_REAL: usize = COL_OUTPUT_BYTE_OFFSET + OUTPUT_LEN;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 802

pub const NUM_ROW_CONSTRAINTS: usize = 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakExtractWideRow {
    pub input: [u8; MAX_INPUT_LEN],
    pub input_len: usize,
    pub output: [u8; OUTPUT_LEN],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakExtractWideWitness {
    pub invocations: Vec<KeccakExtractWideRow>,
}

impl KeccakExtractWideWitness {
    pub fn from_inputs(inputs: &[&[u8]]) -> Result<Self, String> {
        let mut invocations = Vec::with_capacity(inputs.len());
        for inp in inputs {
            if inp.len() > MAX_INPUT_LEN {
                return Err(format!(
                    "input exceeds MAX_INPUT_LEN={}: got {}",
                    MAX_INPUT_LEN, inp.len(),
                ));
            }
            let mut padded = [0u8; MAX_INPUT_LEN];
            padded[..inp.len()].copy_from_slice(inp);
            let output = keccak256(inp);
            invocations.push(KeccakExtractWideRow {
                input: padded,
                input_len: inp.len(),
                output,
            });
        }
        Ok(Self { invocations })
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &KeccakExtractWideWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.invocations.iter().enumerate() {
        for b in 0..MAX_INPUT_LEN {
            columns[COL_INPUT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(row.input[b] as u64, curve);
        }
        columns[COL_INPUT_LEN][i] = Scalar::from_u64(row.input_len as u64, curve);
        for b in 0..OUTPUT_LEN {
            columns[COL_OUTPUT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(row.output[b] as u64, curve);
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

pub struct KeccakExtractWideConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl KeccakExtractWideConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for KeccakExtractWideConstraintSystem {
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
        let mut bin = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin[r] = v.mul(&v.sub(&one));
        }
        vec![bin]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let one = Scalar::one(alpha.curve_type());
        let v = &col_evals[COL_IS_REAL];
        v.mul(&v.sub(&one))
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let _ = alpha;
        let curve = col_coeffs[0][0].curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        poly_mul(v, &v_m1, curve)
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
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp linkage ──────────────────────────────────────────

/// Linkage descriptor for block_header_air → this wide KeccakExtract.
/// Matches the 768-byte header_rlp + 32-byte block_hash tuple to the
/// wide variant's INPUT_BYTE[0..768] + OUTPUT_BYTE[0..32] + INPUT_LEN
/// columns.
///
/// Note: the original `make_block_header_to_keccak_extract_linkage_descriptor`
/// uses the 256-byte variant. This descriptor replaces it for real
/// Ethereum headers that exceed 256 bytes.
pub fn make_block_header_to_keccak_extract_wide_linkage_descriptor(
    block_header_layer_index: usize,
    keccak_extract_wide_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;

    // A side (block_header_air): header_rlp[0..HEADER_RLP_MAX_LEN] + block_hash[0..32].
    // block_header_air has been widened to 768-byte header_rlp, matching
    // the wide keccak's input window, so the full RLP body is bound here.
    let mut a_columns: Vec<usize> = Vec::with_capacity(bh::HEADER_RLP_MAX_LEN + 32);
    for b in 0..bh::HEADER_RLP_MAX_LEN {
        a_columns.push(bh::COL_HEADER_RLP_OFFSET + b);
    }
    for b in 0..32 {
        a_columns.push(bh::COL_BLOCK_HASH_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(bh::HEADER_RLP_MAX_LEN + 32);
    for b in 0..bh::HEADER_RLP_MAX_LEN {
        b_columns.push(COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..OUTPUT_LEN {
        b_columns.push(COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_header_to_keccak_extract_wide_v1".into(),
        a_layer_index: block_header_layer_index,
        a_columns,
        a_selector_column: Some(bh::COL_IS_REAL),
        b_layer_index: keccak_extract_wide_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_builds_for_short_input() {
        let input = vec![0xABu8; 64];
        let w = KeccakExtractWideWitness::from_inputs(&[&input]).unwrap();
        assert_eq!(w.invocations.len(), 1);
        assert_eq!(w.invocations[0].input_len, 64);
        assert_eq!(w.invocations[0].output, keccak256(&input));
    }

    #[test]
    fn witness_builds_for_600_byte_input() {
        let input = vec![0xCDu8; 600];
        let w = KeccakExtractWideWitness::from_inputs(&[&input]).unwrap();
        assert_eq!(w.invocations[0].input_len, 600);
        assert_eq!(w.invocations[0].output, keccak256(&input));
    }

    #[test]
    fn witness_rejects_over_max() {
        let input = vec![0u8; 769];
        assert!(KeccakExtractWideWitness::from_inputs(&[&input]).is_err());
    }

    #[test]
    fn trace_populates_correctly() {
        let input = vec![0x42u8; 100];
        let w = KeccakExtractWideWitness::from_inputs(&[&input]).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.num_rows, 1);
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_INPUT_LEN].evaluations[0].to_u64(), 100);
        assert_eq!(trace.columns[COL_INPUT_BYTE_OFFSET].evaluations[0].to_u64(), 0x42);
        assert!(trace.columns[COL_INPUT_BYTE_OFFSET + 100].evaluations[0].is_zero());
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 802);
        assert_eq!(MAX_INPUT_LEN, 768);
    }

    #[test]
    fn linkage_descriptor_shape() {
        let d = make_block_header_to_keccak_extract_wide_linkage_descriptor(0, 1);
        let bh_rlp_len = crate::block_header_air::HEADER_RLP_MAX_LEN;
        assert_eq!(d.a_columns.len(), bh_rlp_len + 32);
        assert_eq!(d.b_columns.len(), bh_rlp_len + 32);
        assert_eq!(d.label, "block_header_to_keccak_extract_wide_v1");
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
        let input = vec![0xABu8; 600];
        let w = KeccakExtractWideWitness::from_inputs(&[&input]).unwrap();
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = KeccakExtractWideConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }
}
