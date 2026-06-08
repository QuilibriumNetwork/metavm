//! SHA3 input gadget AIR (Phase A1b of the EVM coverage roadmap).
//!
//! Closes the residual gap in Phase A1a: A1a binds the EVM SHA3 row's
//! `output0` (4 BE-packed U256 limbs) to **some** KeccakExtract row's
//! `KECCAK_OUTPUT_LIMB` tuple, but the linked extract row's
//! `(INPUT_BYTE, INPUT_LEN)` was free — the prover never committed to
//! the actual preimage. A malicious prover with infinite computation
//! could in principle fabricate any `(input, output)` pair whose hash
//! matches a desired output (preimage resistance).
//!
//! This gadget AIR introduces an explicit per-SHA3-invocation row that
//! exposes `(input_byte[0..256], input_len, output_limb[0..4])` as
//! algebraic columns. Combined with two cross-AIR LogUps:
//!
//!   L1: EVM main `output0[0..4]` ↔ this gadget `OUTPUT_LIMB[0..4]`
//!       (gated by SEL_KECCAK / IS_REAL)
//!   L2: this gadget `(INPUT_BYTE, INPUT_LEN, OUTPUT_LIMB)` ↔
//!       KeccakExtract `(INPUT_BYTE, INPUT_LEN, KECCAK_OUTPUT_LIMB)`
//!       (gated by IS_REAL / IS_REAL)
//!
//! the linked KeccakExtract row's `INPUT_BYTE` is pinned to equal the
//! gadget's `INPUT_BYTE`, which the prover commits to publicly. The
//! result: the proof now algebraically exposes the SHA3 preimage as
//! committed witness data, not just an unspecified existence claim.
//!
//! **What this gadget does NOT bind (deferred)**: the gadget's
//! `INPUT_BYTE` is still an oracle relative to the EVM memory model.
//! A complete A1b-mem follow-up will add a cross-AIR LogUp from this
//! gadget's `INPUT_BYTE` columns to the EVM memory-access trace,
//! forcing those bytes to equal the actual `interp.shared_memory`
//! contents at the SHA3 row's `(offset, size)`. That requires byte-
//! level memory access tracking which the current EVM memory
//! permutation (U256-aligned) doesn't expose.
//!
//! **Soundness comparison**:
//! - A1a alone (output-side only): preimage resistance; prover claims
//!   "output is some keccak result" without revealing the preimage.
//! - A1b (this gadget, without memory binding): preimage committed —
//!   prover algebraically commits to specific (input, output) preimage
//!   pair. Output match is forced by L1; input match across L1↔L2 is
//!   forced by the gadget's OUTPUT_LIMB being on the same row as the
//!   INPUT_BYTE columns. External observer can audit the preimage.
//! - A1b-mem (future): full algebraic — bytes pinned to memory.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum input length supported. Matches
/// [`crate::keccak_extract::MAX_INPUT_LEN`] so the cross-AIR LogUp
/// tuple shapes align byte-for-byte.
pub const INPUT_BYTE_WIDTH: usize = 256;

/// Output is 4 BE-packed u64 limbs (matches `U256::from_be_bytes` of
/// the 32-byte keccak digest — same convention as
/// [`crate::keccak_extract::keccak_output_limbs_from_output`]).
pub const NUM_OUTPUT_LIMBS: usize = 4;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_INPUT_BYTE_OFFSET: usize = 0;
pub const COL_INPUT_LEN: usize = COL_INPUT_BYTE_OFFSET + INPUT_BYTE_WIDTH;
pub const COL_OUTPUT_LIMB_OFFSET: usize = COL_INPUT_LEN + 1;
pub const COL_IS_REAL: usize = COL_OUTPUT_LIMB_OFFSET + NUM_OUTPUT_LIMBS;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;
pub const NUM_ROW_CONSTRAINTS: usize = 1;
pub const NUM_SHIFTED: usize = 0;

const _: () = assert!(NUM_COLUMNS == 262);

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sha3InputRow {
    /// Padded to [`INPUT_BYTE_WIDTH`] with trailing zeros.
    pub input: [u8; INPUT_BYTE_WIDTH],
    /// Actual input length.
    pub input_len: usize,
    /// `U256::from_be_bytes(keccak256(input[..input_len])).as_limbs()`.
    pub output_limb: [u64; NUM_OUTPUT_LIMBS],
}

#[derive(Clone, Debug, Default)]
pub struct Sha3InputWitness {
    pub invocations: Vec<Sha3InputRow>,
}

impl Sha3InputWitness {
    /// Build a witness from raw input byte slices. Each input is
    /// keccak-hashed using [`crate::keccak::keccak256`] and the result
    /// is BE-packed via [`crate::keccak_extract::keccak_output_limbs_from_output`].
    pub fn from_inputs(inputs: &[Vec<u8>]) -> Result<Self, &'static str> {
        let mut invocations = Vec::with_capacity(inputs.len());
        for inp in inputs {
            if inp.len() > INPUT_BYTE_WIDTH {
                return Err(
                    "sha3_input_air: input exceeds INPUT_BYTE_WIDTH \
                     (pad or use a larger-bound AIR variant)",
                );
            }
            let mut padded = [0u8; INPUT_BYTE_WIDTH];
            padded[..inp.len()].copy_from_slice(inp);
            let digest = crate::keccak::keccak256(inp);
            let output_limb = crate::keccak_extract::keccak_output_limbs_from_output(&digest);
            invocations.push(Sha3InputRow {
                input: padded,
                input_len: inp.len(),
                output_limb,
            });
        }
        Ok(Self { invocations })
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Sha3InputWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (i, inv) in witness.invocations.iter().enumerate() {
        for (b, &v) in inv.input.iter().enumerate() {
            columns[COL_INPUT_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_INPUT_LEN][i] = Scalar::from_u64(inv.input_len as u64, curve);
        for (k, &limb) in inv.output_limb.iter().enumerate() {
            columns[COL_OUTPUT_LIMB_OFFSET + k][i] = Scalar::from_u64(limb, curve);
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

pub struct Sha3InputConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Sha3InputConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Sha3InputConstraintSystem {
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
        for row in 0..n {
            let v = &columns[COL_IS_REAL][row];
            bin[row] = v.mul(&v.sub(&one));
        }
        vec![bin]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let body = v.mul(&v.sub(&one));
        let _ = alpha;
        body
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        let _ = (poly_add::<>, poly_scalar_mul::<>);
        body
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
        let tables = vec![
            LookupTable::range(256),  // tbl 0: 8-bit
            LookupTable::range(2),    // tbl 1: 1-bit
            LookupTable::range(512),  // tbl 2: 9-bit for input_len ∈ [0..256]
        ];
        let mut declarations = Vec::new();
        for b in 0..INPUT_BYTE_WIDTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("sha3_input_byte_{}_8bit", b),
                    column_index: COL_INPUT_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "sha3_input_len_9bit".into(),
                column_index: COL_INPUT_LEN,
                max_bits: 9,
                selector_column: None,
            },
            2,
        ));
        declarations.push((
            LookupDeclaration {
                label: "sha3_is_real_1bit".into(),
                column_index: COL_IS_REAL,
                max_bits: 1,
                selector_column: None,
            },
            1,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkage descriptors ──────────────────────────────

/// L2 of the SHA3 input-binding chain: this gadget ↔ KeccakExtract on
/// the full `(INPUT_BYTE[0..256], INPUT_LEN, OUTPUT_LIMB[0..4])` 261-
/// element tuple. Forces every gadget row's claimed `(input, output)`
/// pair to equal a KeccakExtract row's, where KeccakExtract has its
/// own algebraic binding `OUTPUT_BYTE = keccak256(INPUT_BYTE[..LEN])`
/// (closed by the #91 KeccakExtract↔Keccak chain) and
/// `OUTPUT_LIMB = U256::from_be_bytes(OUTPUT_BYTE).as_limbs()`
/// (closed by `keccak_output_limb_*_binding` row-locals in
/// `crate::keccak_extract`).
pub fn make_sha3_input_keccak_extract_linkage_descriptor(
    gadget_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    debug_assert_eq!(INPUT_BYTE_WIDTH, ke::MAX_INPUT_LEN);
    debug_assert_eq!(NUM_OUTPUT_LIMBS, ke::KECCAK_OUTPUT_LIMB_LEN);

    let mut a_columns: Vec<usize> = (0..INPUT_BYTE_WIDTH)
        .map(|b| COL_INPUT_BYTE_OFFSET + b)
        .collect();
    a_columns.push(COL_INPUT_LEN);
    for k in 0..NUM_OUTPUT_LIMBS {
        a_columns.push(COL_OUTPUT_LIMB_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = (0..ke::MAX_INPUT_LEN)
        .map(|b| ke::COL_INPUT_BYTE_OFFSET + b)
        .collect();
    b_columns.push(ke::COL_INPUT_LEN);
    for k in 0..ke::KECCAK_OUTPUT_LIMB_LEN {
        b_columns.push(ke::COL_KECCAK_OUTPUT_LIMB_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sha3_input_keccak_extract_v1".into(),
        a_layer_index: gadget_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak::keccak256;
    use crate::keccak_extract::keccak_output_limbs_from_output;

    #[test]
    fn from_inputs_computes_keccak_and_be_limbs() {
        let inputs = vec![
            b"hello world".to_vec(),
            b"".to_vec(),
            (0..200u8).collect::<Vec<u8>>(),
        ];
        let w = Sha3InputWitness::from_inputs(&inputs).unwrap();
        assert_eq!(w.invocations.len(), 3);
        for (i, inp) in inputs.iter().enumerate() {
            let expected_digest = keccak256(inp);
            let expected_limbs = keccak_output_limbs_from_output(&expected_digest);
            assert_eq!(
                w.invocations[i].output_limb, expected_limbs,
                "invocation {} output_limb mismatch", i
            );
            assert_eq!(w.invocations[i].input_len, inp.len());
            assert_eq!(&w.invocations[i].input[..inp.len()], inp.as_slice());
        }
    }

    #[test]
    fn from_inputs_rejects_oversize() {
        let big = vec![0u8; INPUT_BYTE_WIDTH + 1];
        assert!(Sha3InputWitness::from_inputs(&[big]).is_err());
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = Sha3InputWitness::from_inputs(&[b"hello".to_vec(), b"world".to_vec()]).unwrap();
        let trace = build_trace_polynomials(&w, curve);
        let cs = Sha3InputConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        for row in 0..trace.padded_size as usize {
            let cols: Vec<Scalar> = trace.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            assert!(cs.evaluate_at_point(&cols, &alpha).is_zero());
        }
    }

    #[test]
    fn keccak_extract_linkage_descriptor_well_formed() {
        use crate::keccak_extract as ke;
        let d = make_sha3_input_keccak_extract_linkage_descriptor(0, 1);
        assert_eq!(d.label, "sha3_input_keccak_extract_v1");
        let expected = INPUT_BYTE_WIDTH + 1 + NUM_OUTPUT_LIMBS;
        assert_eq!(d.a_columns.len(), expected);
        assert_eq!(d.b_columns.len(), expected);
        assert_eq!(d.a_columns[0], COL_INPUT_BYTE_OFFSET);
        assert_eq!(d.a_columns[INPUT_BYTE_WIDTH], COL_INPUT_LEN);
        assert_eq!(d.a_columns[INPUT_BYTE_WIDTH + 1], COL_OUTPUT_LIMB_OFFSET);
        assert_eq!(d.b_columns[0], ke::COL_INPUT_BYTE_OFFSET);
        assert_eq!(d.b_columns[INPUT_BYTE_WIDTH], ke::COL_INPUT_LEN);
        assert_eq!(d.b_columns[INPUT_BYTE_WIDTH + 1], ke::COL_KECCAK_OUTPUT_LIMB_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(ke::COL_IS_REAL));
    }

    /// Critical correctness oracle: this gadget's `(INPUT_BYTE,
    /// INPUT_LEN, OUTPUT_LIMB)` row matches KeccakExtract's
    /// `(INPUT_BYTE, INPUT_LEN, KECCAK_OUTPUT_LIMB)` row byte-for-byte
    /// when built from the same inputs. Validates the cross-AIR LogUp
    /// L2 tuple shape end-to-end on the gadget+extract witnesses.
    #[test]
    fn gadget_byte_tuples_match_keccak_extract_witness() {
        let curve = CurveType::Bls48581;
        let inputs = vec![b"the quick brown fox jumps over the lazy dog".to_vec()];

        let gadget_w = Sha3InputWitness::from_inputs(&inputs).unwrap();
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        let extract_w =
            crate::keccak_extract::KeccakExtractWitness::from_inputs(&inputs).unwrap();
        let extract_trace =
            crate::keccak_extract::build_trace_polynomials(&extract_w, curve);

        for b in 0..INPUT_BYTE_WIDTH {
            let g = gadget_trace.columns[COL_INPUT_BYTE_OFFSET + b].evaluations[0].to_u64();
            let e = extract_trace.columns[crate::keccak_extract::COL_INPUT_BYTE_OFFSET + b]
                .evaluations[0].to_u64();
            assert_eq!(g, e, "input_byte {} mismatch", b);
        }
        let g_len = gadget_trace.columns[COL_INPUT_LEN].evaluations[0].to_u64();
        let e_len = extract_trace.columns[crate::keccak_extract::COL_INPUT_LEN]
            .evaluations[0].to_u64();
        assert_eq!(g_len, e_len);
        for k in 0..NUM_OUTPUT_LIMBS {
            let g = gadget_trace.columns[COL_OUTPUT_LIMB_OFFSET + k].evaluations[0].to_u64();
            let e = extract_trace.columns[
                crate::keccak_extract::COL_KECCAK_OUTPUT_LIMB_OFFSET + k
            ].evaluations[0].to_u64();
            assert_eq!(g, e, "output_limb {} mismatch", k);
        }
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn sha3_input_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = Sha3InputWitness::from_inputs(&[
            b"hello".to_vec(),
            (0..32u8).collect::<Vec<u8>>(),
        ]).unwrap();
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = Sha3InputConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid);
    }
}
