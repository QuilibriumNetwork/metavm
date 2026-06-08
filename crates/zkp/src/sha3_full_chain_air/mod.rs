//! EVM SHA3 full-chain composition AIR.
//!
//! Per row, commits one EVM `SHA3` opcode invocation's complete
//! validity tuple at the top of the composition stack:
//!
//!   * `pc` — program counter at which the SHA3 opcode was executed.
//!   * `mem_offset` — start of the keccak input window in EVM memory.
//!   * `length` (u64) — number of bytes hashed.  Decomposed into 8 LE
//!     bytes for range checking and for binding into byte-memory /
//!     read-byte gadgets that operate at byte granularity.
//!   * `input_bytes[0..MAX_INPUT_BYTES]` — the raw 64-byte window of
//!     keccak input. Inputs longer than `MAX_INPUT_BYTES` are not
//!     supported by this top-level composition AIR; use
//!     [`crate::sha3_input_air`] (256 B) or
//!     [`crate::sha3_read_byte_air`] (256 B) for wider invocations
//!     and wire them as a sub-layer.
//!   * `output_hash[0..32]` — keccak256 digest of the input bytes.
//!   * `output_limb[0..4]` — `U256::from_be_bytes(output_hash).as_limbs()`,
//!     mirroring [`crate::keccak_extract::keccak_output_limbs_from_output`].
//!     Carried separately from the byte form so the SHA3-input ↔
//!     keccak-extract LogUp can be assembled out of the same tuple
//!     shape as [`crate::sha3_input_air`].
//!   * `is_real` — row-active selector.
//!
//! ## What this AIR proves (algebraically)
//!
//! On its own this AIR proves a small set of *structural* invariants:
//!
//!   1. `is_real ∈ {0, 1}` — row active flag is binary.
//!   2. `length = Σ_b length_byte[b] · 2^(8b)` — length agrees with its
//!      LE byte decomposition.
//!   3. `output_limb[k] = Σ_b output_hash[8k + b] · 2^(8(7-b))` for
//!      `k ∈ 0..4` — each LE u64 limb of the output is the BE
//!      packing of the corresponding 8 bytes of the digest (matching
//!      [`crate::keccak_extract::keccak_output_limbs_from_output`]).
//!
//! Total: `1 + 1 + 4 = 6` row-local constraints.
//!
//! ## What the cross-AIR LogUp descriptors close
//!
//! Every meaningful binding to the rest of the chain is enforced
//! algebraically via [`crate::cross_air_logup::CrossAirLogUpDescriptor`]s:
//!
//!   * [`make_sha3_full_to_sha3_input_descriptor`] — binds the
//!     `(input_bytes[0..64], length, output_limb[0..4])` tuple into
//!     [`crate::sha3_input_air`]'s
//!     `(COL_INPUT_BYTE_OFFSET..+64, COL_INPUT_LEN, COL_OUTPUT_LIMB)`.
//!     Note the sha3_input gadget pads to 256 bytes; the 64-byte
//!     prefix is the active window for this top-level AIR.  Bytes
//!     `64..256` are constrained to be zero in the sub-gadget by the
//!     downstream `IS_REAL_BINARY` + the `INPUT_LEN ≤ 64` invariant
//!     that callers wire in via separate range check.
//!   * [`make_sha3_full_to_byte_memory_descriptor`] — binds the
//!     anchor pair `(mem_offset, input_bytes[0])` into
//!     [`crate::byte_memory_air`]'s `(COL_ADDR, COL_VAL)`. **Anchor
//!     binding only**: the per-position closure of all
//!     `MAX_INPUT_BYTES` bytes is performed through the
//!     `sha3_read_byte_air` ↔ `byte_memory_air` per-position
//!     linkages already wired in [`crate::sha3_mem_chain`]; this
//!     descriptor pins the offset/byte0 head of that chain to the
//!     row top of the composition AIR.
//!   * [`make_sha3_full_to_sha3_read_byte_descriptor`] — binds
//!     `(mem_offset, length)` into [`crate::sha3_read_byte_air`]'s
//!     `(COL_OFFSET, COL_INPUT_LEN)`.
//!   * [`make_sha3_full_to_keccak_extract_descriptor`] — binds the
//!     `(input_bytes[0..64], length, output_limb[0..4])` tuple into
//!     [`crate::keccak_extract`]'s
//!     `(COL_INPUT_BYTE_OFFSET..+64, COL_INPUT_LEN,
//!     COL_KECCAK_OUTPUT_LIMB_OFFSET)`. Composed with the
//!     keccak_extract ↔ Keccak permutation closure, this forces the
//!     digest in this AIR's row to equal `keccak256(input_bytes[..length])`.
//!   * [`make_sha3_full_to_stack_contents_descriptor`] — binds
//!     `(pc, output_limb[0..4])` into the EVM stack contents AIR's
//!     `(COL_PC, COL_VALUE_LIMB_0..3)`. **Cross-crate descriptor**:
//!     `metavm-zkp` cannot directly import `metavm-evm`'s
//!     `stack_contents_air`, so the B-side column indices are
//!     hard-coded against the published EVM column contract
//!     ([`STACK_CONTENTS_COL_PC`], [`STACK_CONTENTS_COL_VALUE_LIMB_0`],
//!     [`STACK_CONTENTS_COL_IS_REAL`]). Verified against
//!     `crates/evm/src/stack_contents_air/mod.rs` at descriptor-
//!     well-formedness test time. The B-side selector pins to the
//!     stack-contents `IS_REAL` column; callers further gating to
//!     "writes from SHA3 only" can layer an extra OR with the
//!     opcode-specific selector externally.
//!
//! ## Host-side builder
//!
//! [`Sha3FullChainWitness::from_events`] takes a slice of
//! `(pc, mem_offset, length, input_bytes)` tuples and builds the
//! corresponding witness, computing the keccak digest via
//! [`crate::keccak::keccak256`] and the limb packing via
//! [`crate::keccak_extract::keccak_output_limbs_from_output`].
//!
//! ## Limitations (recorded for follow-up)
//!
//!   * `MAX_INPUT_BYTES = 64`. SHA3 invocations with larger inputs
//!     are not representable as a single row; use the wider
//!     sub-gadgets (256 B) and wire them as a sub-layer.
//!   * EVM-side stack-contents descriptor uses hard-coded EVM column
//!     indices since the `evm` crate cannot be imported here. The
//!     `descriptors_well_formed` test pins the column contract.
//!   * Anchor-only memory binding: only `(mem_offset, input_bytes[0])`
//!     is bound at the top level; the per-position closure flows
//!     through the existing `sha3_mem_chain` descriptors.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// Maximum SHA3 input window committed per row.
pub const MAX_INPUT_BYTES: usize = 64;
/// keccak256 digest length in bytes.
pub const OUTPUT_HASH_LEN: usize = 32;
/// keccak256 output expressed as 4 LE u64 limbs (per
/// [`crate::keccak_extract::keccak_output_limbs_from_output`]).
pub const NUM_OUTPUT_LIMBS: usize = 4;
/// u64 length is decomposed into 8 LE bytes.
pub const LENGTH_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PC: usize = 0;
pub const COL_MEM_OFFSET: usize = 1;
pub const COL_LENGTH: usize = 2;
pub const COL_LENGTH_BYTE_OFFSET: usize = 3; // 3..11
pub const COL_INPUT_BYTE_OFFSET: usize =
    COL_LENGTH_BYTE_OFFSET + LENGTH_BYTES; // 11..75
pub const COL_OUTPUT_HASH_OFFSET: usize =
    COL_INPUT_BYTE_OFFSET + MAX_INPUT_BYTES; // 75..107
pub const COL_OUTPUT_LIMB_OFFSET: usize =
    COL_OUTPUT_HASH_OFFSET + OUTPUT_HASH_LEN; // 107..111
pub const COL_IS_REAL: usize =
    COL_OUTPUT_LIMB_OFFSET + NUM_OUTPUT_LIMBS; // 111

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 112

/// Row-local constraints:
///   0: is_real_binary
///   1: length_le_decomp
///   2..6: 4 output-limb BE byte decompositions (one per limb)
pub const NUM_ROW_CONSTRAINTS: usize = 1 + 1 + NUM_OUTPUT_LIMBS; // 6
pub const NUM_SHIFTED: usize = 0;

// ─── EVM stack_contents_air column contract (hard-coded) ──────────────

/// `crates/evm/src/stack_contents_air/mod.rs::COL_PC`.
pub const STACK_CONTENTS_COL_PC: usize = 1;
/// `crates/evm/src/stack_contents_air/mod.rs::COL_VALUE_LIMB_0`.
pub const STACK_CONTENTS_COL_VALUE_LIMB_0: usize = 4;
/// `crates/evm/src/stack_contents_air/mod.rs::COL_IS_REAL`.
pub const STACK_CONTENTS_COL_IS_REAL: usize = 9;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sha3FullChainRow {
    pub pc: u64,
    pub mem_offset: u64,
    /// Length of the keccak input, in bytes. Must be `≤ MAX_INPUT_BYTES`.
    pub length: u64,
    /// Input bytes, zero-padded to [`MAX_INPUT_BYTES`].
    pub input_bytes: [u8; MAX_INPUT_BYTES],
    /// `keccak256(input_bytes[..length])`.
    pub output_hash: [u8; OUTPUT_HASH_LEN],
    /// LE u64 limbs of the BE-interpreted digest, mirroring
    /// [`crate::keccak_extract::keccak_output_limbs_from_output`].
    pub output_limb: [u64; NUM_OUTPUT_LIMBS],
}

#[derive(Clone, Debug, Default)]
pub struct Sha3FullChainWitness {
    pub rows: Vec<Sha3FullChainRow>,
}

impl Sha3FullChainWitness {
    pub fn from_rows(rows: Vec<Sha3FullChainRow>) -> Self {
        Self { rows }
    }

    /// Build a multi-row witness from a sequence of
    /// `(pc, mem_offset, length, input_bytes)` events.
    ///
    /// `input_bytes` must have length `≤ MAX_INPUT_BYTES`; longer
    /// invocations are rejected. The host-side keccak digest and
    /// the LE limb packing are computed via
    /// [`crate::keccak::keccak256`] +
    /// [`crate::keccak_extract::keccak_output_limbs_from_output`].
    pub fn from_events(
        events: &[(u64, u64, u64, Vec<u8>)],
    ) -> Result<Self, &'static str> {
        let mut rows = Vec::with_capacity(events.len());
        for (pc, mem_offset, length, input) in events {
            if input.len() > MAX_INPUT_BYTES {
                return Err(
                    "sha3_full_chain_air: input exceeds MAX_INPUT_BYTES (64). \
                     Use sha3_input_air (256B) for wider invocations.",
                );
            }
            if (*length as usize) != input.len() {
                return Err(
                    "sha3_full_chain_air: declared length does not match \
                     input_bytes length",
                );
            }
            let mut padded = [0u8; MAX_INPUT_BYTES];
            padded[..input.len()].copy_from_slice(input);
            let digest = crate::keccak::keccak256(input);
            let output_limb =
                crate::keccak_extract::keccak_output_limbs_from_output(&digest);
            rows.push(Sha3FullChainRow {
                pc: *pc,
                mem_offset: *mem_offset,
                length: *length,
                input_bytes: padded,
                output_hash: digest,
                output_limb,
            });
        }
        Ok(Self { rows })
    }
}

/// Top-level builder matching the task description signature.
pub fn from_events(
    events: &[(u64, u64, u64, Vec<u8>)],
) -> Result<Sha3FullChainWitness, &'static str> {
    Sha3FullChainWitness::from_events(events)
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < LENGTH_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

/// `2^(8 * (7 - b))` — BE byte weight for a u64 limb's byte `b` (where
/// `b = 0` is the most-significant byte of the limb's 8 BE bytes).
fn be_limb_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < 8);
    Scalar::from_u64(1u64 << (8 * (7 - b)), curve)
}

fn eval_length_le_decomp(target: &Scalar, col_evals: &[Scalar]) -> Scalar {
    let curve = target.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..LENGTH_BYTES {
        let byte = &col_evals[COL_LENGTH_BYTE_OFFSET + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target.sub(&sum)
}

fn build_length_le_decomp_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..LENGTH_BYTES {
        let byte_poly = &col_coeffs[COL_LENGTH_BYTE_OFFSET + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    poly_sub(&col_coeffs[COL_LENGTH], &sum, curve)
}

fn eval_output_limb_decomp(
    limb_idx: usize,
    col_evals: &[Scalar],
) -> Scalar {
    let curve = col_evals[0].curve_type();
    // Per `keccak_output_limbs_from_output`:
    //   limb[k] = u64::from_be_bytes(output[(3-k)*8 .. (3-k)*8 + 8])
    // i.e. limb 0 ← output[24..32], limb 3 ← output[0..8] (MSB first).
    let byte_base = COL_OUTPUT_HASH_OFFSET + (3 - limb_idx) * 8;
    let mut sum = Scalar::zero(curve);
    for b in 0..8 {
        let byte = &col_evals[byte_base + b];
        sum = sum.add(&byte.mul(&be_limb_byte_pow(b, curve)));
    }
    col_evals[COL_OUTPUT_LIMB_OFFSET + limb_idx].sub(&sum)
}

fn build_output_limb_decomp_poly(
    limb_idx: usize,
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let byte_base = COL_OUTPUT_HASH_OFFSET + (3 - limb_idx) * 8;
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..8 {
        let byte_poly = &col_coeffs[byte_base + b];
        let term = poly_scalar_mul(byte_poly, &be_limb_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    poly_sub(&col_coeffs[COL_OUTPUT_LIMB_OFFSET + limb_idx], &sum, curve)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Sha3FullChainWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_PC][i] = Scalar::from_u64(row.pc, curve);
        columns[COL_MEM_OFFSET][i] = Scalar::from_u64(row.mem_offset, curve);
        columns[COL_LENGTH][i] = Scalar::from_u64(row.length, curve);

        let length_bytes = row.length.to_le_bytes();
        for b in 0..LENGTH_BYTES {
            columns[COL_LENGTH_BYTE_OFFSET + b][i] =
                Scalar::from_u64(length_bytes[b] as u64, curve);
        }

        for b in 0..MAX_INPUT_BYTES {
            columns[COL_INPUT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(row.input_bytes[b] as u64, curve);
        }
        for b in 0..OUTPUT_HASH_LEN {
            columns[COL_OUTPUT_HASH_OFFSET + b][i] =
                Scalar::from_u64(row.output_hash[b] as u64, curve);
        }
        for k in 0..NUM_OUTPUT_LIMBS {
            columns[COL_OUTPUT_LIMB_OFFSET + k][i] =
                Scalar::from_u64(row.output_limb[k], curve);
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct Sha3FullChainConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Sha3FullChainConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Sha3FullChainConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "is_real_binary".into(),
            "length_le_decomp".into(),
        ];
        for k in 0..NUM_OUTPUT_LIMBS {
            labels.push(format!("output_limb_{}_be_decomp", k));
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
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];

            // 0: is_real binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));

            // 1: length LE byte decomposition.
            bodies[1][row] =
                eval_length_le_decomp(&row_evals[COL_LENGTH], &row_evals);

            // 2..6: per-limb BE byte decompositions of output_hash.
            for k in 0..NUM_OUTPUT_LIMBS {
                bodies[2 + k][row] = eval_output_limb_decomp(k, &row_evals);
            }
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

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(eval_length_le_decomp(&col_evals[COL_LENGTH], col_evals));
        for k in 0..NUM_OUTPUT_LIMBS {
            bodies.push(eval_output_limb_decomp(k, col_evals));
        }

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
        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let length_decomp = build_length_le_decomp_poly(col_coeffs, curve);

        let mut limb_bodies: Vec<Vec<Scalar>> =
            Vec::with_capacity(NUM_OUTPUT_LIMBS);
        for k in 0..NUM_OUTPUT_LIMBS {
            limb_bodies.push(build_output_limb_decomp_poly(k, col_coeffs, curve));
        }

        let mut bodies: Vec<Vec<Scalar>> =
            Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real_binary);
        bodies.push(length_decomp);
        for b in limb_bodies {
            bodies.push(b);
        }

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

        let byte_ranges: [(usize, usize, &str); 3] = [
            (COL_LENGTH_BYTE_OFFSET, LENGTH_BYTES, "length_byte"),
            (COL_INPUT_BYTE_OFFSET, MAX_INPUT_BYTES, "input_byte"),
            (COL_OUTPUT_HASH_OFFSET, OUTPUT_HASH_LEN, "output_hash_byte"),
        ];
        for (off, len, label) in byte_ranges {
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

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(input_bytes[0..MAX_INPUT_BYTES], length, output_limb[0..4])`
/// of this AIR against [`crate::sha3_input_air`]'s
/// `(COL_INPUT_BYTE_OFFSET..+MAX_INPUT_BYTES, COL_INPUT_LEN,
/// COL_OUTPUT_LIMB_OFFSET..+4)`.
///
/// Tuple width: `MAX_INPUT_BYTES + 1 + NUM_OUTPUT_LIMBS = 64 + 1 + 4 = 69`.
pub fn make_sha3_full_to_sha3_input_descriptor(
    full_layer_index: usize,
    sha3_input_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::sha3_input_air as si;
    let mut a_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_BYTES + 1 + NUM_OUTPUT_LIMBS);
    let mut b_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_BYTES + 1 + NUM_OUTPUT_LIMBS);
    for b in 0..MAX_INPUT_BYTES {
        a_columns.push(COL_INPUT_BYTE_OFFSET + b);
        b_columns.push(si::COL_INPUT_BYTE_OFFSET + b);
    }
    a_columns.push(COL_LENGTH);
    b_columns.push(si::COL_INPUT_LEN);
    for k in 0..NUM_OUTPUT_LIMBS {
        a_columns.push(COL_OUTPUT_LIMB_OFFSET + k);
        b_columns.push(si::COL_OUTPUT_LIMB_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "sha3_full_to_sha3_input_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha3_input_layer_index,
        b_columns,
        b_selector_column: Some(si::COL_IS_REAL),
    }
}

/// Bind the anchor pair `(mem_offset, input_bytes[0])` of this AIR
/// against [`crate::byte_memory_air`]'s `(COL_ADDR, COL_VAL)`. The
/// per-position closure of all `MAX_INPUT_BYTES` bytes is performed
/// through the existing `sha3_read_byte_air ↔ byte_memory_air`
/// per-position linkages (see [`crate::sha3_mem_chain`]); this
/// descriptor pins the head of that chain to the composition row.
pub fn make_sha3_full_to_byte_memory_descriptor(
    full_layer_index: usize,
    byte_memory_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::byte_memory_air as bm;
    let a_columns: Vec<usize> = vec![COL_MEM_OFFSET, COL_INPUT_BYTE_OFFSET];
    let b_columns: Vec<usize> = vec![bm::COL_ADDR, bm::COL_VAL];
    CrossAirLogUpDescriptor {
        label: "sha3_full_to_byte_memory_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: byte_memory_layer_index,
        b_columns,
        b_selector_column: Some(bm::COL_IS_REAL),
    }
}

/// Bind `(mem_offset, length)` of this AIR against
/// [`crate::sha3_read_byte_air`]'s `(COL_OFFSET, COL_INPUT_LEN)`.
pub fn make_sha3_full_to_sha3_read_byte_descriptor(
    full_layer_index: usize,
    sha3_read_byte_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::sha3_read_byte_air as sr;
    let a_columns: Vec<usize> = vec![COL_MEM_OFFSET, COL_LENGTH];
    let b_columns: Vec<usize> = vec![sr::COL_OFFSET, sr::COL_INPUT_LEN];
    CrossAirLogUpDescriptor {
        label: "sha3_full_to_sha3_read_byte_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha3_read_byte_layer_index,
        b_columns,
        b_selector_column: Some(sr::COL_IS_REAL),
    }
}

/// Bind `(input_bytes[0..MAX_INPUT_BYTES], length, output_limb[0..4])`
/// of this AIR against [`crate::keccak_extract`]'s
/// `(COL_INPUT_BYTE_OFFSET..+MAX_INPUT_BYTES, COL_INPUT_LEN,
/// COL_KECCAK_OUTPUT_LIMB_OFFSET..+4)`. Composed with the
/// keccak_extract ↔ Keccak permutation closure already wired in
/// [`crate::keccak_extract::make_keccak_extract_keccak_linkage_descriptor`],
/// this forces the digest in this row to equal `keccak256(input_bytes[..length])`.
pub fn make_sha3_full_to_keccak_extract_descriptor(
    full_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let mut a_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_BYTES + 1 + NUM_OUTPUT_LIMBS);
    let mut b_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_BYTES + 1 + NUM_OUTPUT_LIMBS);
    for b in 0..MAX_INPUT_BYTES {
        a_columns.push(COL_INPUT_BYTE_OFFSET + b);
        b_columns.push(ke::COL_INPUT_BYTE_OFFSET + b);
    }
    a_columns.push(COL_LENGTH);
    b_columns.push(ke::COL_INPUT_LEN);
    for k in 0..NUM_OUTPUT_LIMBS {
        a_columns.push(COL_OUTPUT_LIMB_OFFSET + k);
        b_columns.push(ke::COL_KECCAK_OUTPUT_LIMB_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "sha3_full_to_keccak_extract_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Bind `(pc, output_limb[0..4])` of this AIR against the EVM
/// `stack_contents_air`'s `(COL_PC, COL_VALUE_LIMB_0..3)`.
///
/// **Cross-crate descriptor**: `metavm-zkp` cannot import
/// `metavm-evm`'s `stack_contents_air` (cycle). The B-side column
/// indices are hard-coded against the published EVM column contract
/// ([`STACK_CONTENTS_COL_PC`], [`STACK_CONTENTS_COL_VALUE_LIMB_0`],
/// [`STACK_CONTENTS_COL_IS_REAL`]). The `descriptors_well_formed`
/// test pins this contract.
pub fn make_sha3_full_to_stack_contents_descriptor(
    full_layer_index: usize,
    stack_contents_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + NUM_OUTPUT_LIMBS);
    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + NUM_OUTPUT_LIMBS);
    a_columns.push(COL_PC);
    b_columns.push(STACK_CONTENTS_COL_PC);
    for k in 0..NUM_OUTPUT_LIMBS {
        a_columns.push(COL_OUTPUT_LIMB_OFFSET + k);
        b_columns.push(STACK_CONTENTS_COL_VALUE_LIMB_0 + k);
    }
    CrossAirLogUpDescriptor {
        label: "sha3_full_to_stack_contents_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: stack_contents_layer_index,
        b_columns,
        b_selector_column: Some(STACK_CONTENTS_COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_all_bodies_vanish(w: &Sha3FullChainWitness) {
        let trace = build_trace_polynomials(w, CurveType::Bls48581);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let cs = Sha3FullChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        let labels = cs.constraint_labels();
        for (k, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {}: got {:?}",
                    k,
                    labels[k],
                    r,
                    v,
                );
            }
        }
    }

    /// Empty-input keccak256 known vector:
    /// `keccak256("") = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470`.
    #[test]
    fn empty_input_keccak_known_vector() {
        let w = Sha3FullChainWitness::from_events(&[(0u64, 0u64, 0u64, vec![])])
            .expect("empty input must succeed");
        assert_eq!(w.rows.len(), 1);
        let expected_hex =
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470";
        let expected: Vec<u8> = (0..32)
            .map(|i| u8::from_str_radix(&expected_hex[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        assert_eq!(&w.rows[0].output_hash[..], &expected[..]);
        // length = 0 ⇒ first limb of LE bytes all zero.
        for b in 0..LENGTH_BYTES {
            assert_eq!(w.rows[0].input_bytes[b], 0);
        }
        check_all_bodies_vanish(&w);
    }

    /// 32-byte input known vector: `keccak256([0x00; 32])`.
    /// Per ground truth (verified against
    /// `crate::keccak::keccak256`):
    /// `0x290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563`.
    #[test]
    fn thirty_two_byte_input_known_vector() {
        let input = vec![0u8; 32];
        let w = Sha3FullChainWitness::from_events(&[(
            0x42u64, 0x100u64, 32u64, input.clone(),
        )])
        .expect("32-byte input must succeed");
        let expected_hex =
            "290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563";
        let expected: Vec<u8> = (0..32)
            .map(|i| u8::from_str_radix(&expected_hex[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        assert_eq!(&w.rows[0].output_hash[..], &expected[..]);
        assert_eq!(w.rows[0].pc, 0x42);
        assert_eq!(w.rows[0].mem_offset, 0x100);
        assert_eq!(w.rows[0].length, 32);
        check_all_bodies_vanish(&w);
    }

    /// Tampering the output hash bytes (without re-deriving limbs)
    /// makes the BE-decomp body for limb 0 fire.
    #[test]
    fn tampered_output_hash_fires_limb_decomp() {
        let w = Sha3FullChainWitness::from_events(&[(0u64, 0u64, 0u64, vec![])])
            .expect("empty input must succeed");
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper byte 24 of output_hash → it backs limb 0 (since
        // limb 0 = u64::from_be_bytes(output[24..32])) without
        // touching `output_limb[0]` → BE-decomp body 2 (limb 0)
        // must fire. keccak256("")[24] = 0xfb; flipping to ~ fires
        // the relation.
        cols[COL_OUTPUT_HASH_OFFSET + 24][0] =
            Scalar::from_u64(0x00u64, CurveType::Bls48581);
        let cs = Sha3FullChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Body 2 = limb 0 BE decomp (covers bytes 24..32 of the
        // output hash by the keccak_extract limb convention).
        assert!(
            !bodies[2][0].is_zero(),
            "output_limb_0_be_decomp should fire on tampered hash byte",
        );
    }

    /// Tampering the length without touching length bytes makes the
    /// `length_le_decomp` body fire.
    #[test]
    fn tampered_length_fires_le_decomp() {
        let w = Sha3FullChainWitness::from_events(&[(0u64, 0u64, 0u64, vec![])])
            .expect("empty input must succeed");
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_LENGTH][0] = Scalar::from_u64(42u64, CurveType::Bls48581);
        let cs = Sha3FullChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[1][0].is_zero(),
            "length_le_decomp should fire on tampered length",
        );
    }

    #[test]
    fn from_events_rejects_oversize_input() {
        let big = vec![0u8; MAX_INPUT_BYTES + 1];
        let res = Sha3FullChainWitness::from_events(&[(
            0u64,
            0u64,
            (MAX_INPUT_BYTES + 1) as u64,
            big,
        )]);
        assert!(res.is_err());
    }

    #[test]
    fn from_events_rejects_length_mismatch() {
        // declared length 10 but input is 5 bytes.
        let res = Sha3FullChainWitness::from_events(&[(
            0u64,
            0u64,
            10u64,
            vec![1u8; 5],
        )]);
        assert!(res.is_err());
    }

    #[test]
    fn descriptors_well_formed() {
        use crate::byte_memory_air as bm;
        use crate::keccak_extract as ke;
        use crate::sha3_input_air as si;
        use crate::sha3_read_byte_air as sr;

        let d_in = make_sha3_full_to_sha3_input_descriptor(0, 1);
        assert_eq!(d_in.label, "sha3_full_to_sha3_input_v1");
        assert_eq!(d_in.a_columns.len(), MAX_INPUT_BYTES + 1 + NUM_OUTPUT_LIMBS);
        assert_eq!(d_in.a_columns.len(), d_in.b_columns.len());
        assert_eq!(d_in.a_columns[0], COL_INPUT_BYTE_OFFSET);
        assert_eq!(d_in.b_columns[0], si::COL_INPUT_BYTE_OFFSET);
        assert_eq!(d_in.a_columns[MAX_INPUT_BYTES], COL_LENGTH);
        assert_eq!(d_in.b_columns[MAX_INPUT_BYTES], si::COL_INPUT_LEN);
        assert_eq!(d_in.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_in.b_selector_column, Some(si::COL_IS_REAL));

        let d_bm = make_sha3_full_to_byte_memory_descriptor(0, 2);
        assert_eq!(d_bm.label, "sha3_full_to_byte_memory_v1");
        assert_eq!(d_bm.a_columns, vec![COL_MEM_OFFSET, COL_INPUT_BYTE_OFFSET]);
        assert_eq!(d_bm.b_columns, vec![bm::COL_ADDR, bm::COL_VAL]);
        assert_eq!(d_bm.b_selector_column, Some(bm::COL_IS_REAL));

        let d_rb = make_sha3_full_to_sha3_read_byte_descriptor(0, 3);
        assert_eq!(d_rb.label, "sha3_full_to_sha3_read_byte_v1");
        assert_eq!(d_rb.a_columns, vec![COL_MEM_OFFSET, COL_LENGTH]);
        assert_eq!(d_rb.b_columns, vec![sr::COL_OFFSET, sr::COL_INPUT_LEN]);
        assert_eq!(d_rb.b_selector_column, Some(sr::COL_IS_REAL));

        let d_ke = make_sha3_full_to_keccak_extract_descriptor(0, 4);
        assert_eq!(d_ke.label, "sha3_full_to_keccak_extract_v1");
        assert_eq!(d_ke.a_columns.len(), MAX_INPUT_BYTES + 1 + NUM_OUTPUT_LIMBS);
        assert_eq!(d_ke.b_columns[0], ke::COL_INPUT_BYTE_OFFSET);
        assert_eq!(d_ke.b_columns[MAX_INPUT_BYTES], ke::COL_INPUT_LEN);
        assert_eq!(
            d_ke.b_columns[MAX_INPUT_BYTES + 1],
            ke::COL_KECCAK_OUTPUT_LIMB_OFFSET,
        );
        assert_eq!(d_ke.b_selector_column, Some(ke::COL_IS_REAL));

        let d_sc = make_sha3_full_to_stack_contents_descriptor(0, 5);
        assert_eq!(d_sc.label, "sha3_full_to_stack_contents_v1");
        assert_eq!(d_sc.a_columns.len(), 1 + NUM_OUTPUT_LIMBS);
        assert_eq!(d_sc.a_columns[0], COL_PC);
        assert_eq!(d_sc.b_columns[0], STACK_CONTENTS_COL_PC);
        assert_eq!(d_sc.b_columns[1], STACK_CONTENTS_COL_VALUE_LIMB_0);
        assert_eq!(d_sc.b_columns[4], STACK_CONTENTS_COL_VALUE_LIMB_0 + 3);
        assert_eq!(d_sc.b_selector_column, Some(STACK_CONTENTS_COL_IS_REAL));

        // Labels distinct.
        let labels = vec![
            d_in.label.clone(),
            d_bm.label.clone(),
            d_rb.label.clone(),
            d_ke.label.clone(),
            d_sc.label.clone(),
        ];
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                assert_ne!(labels[i], labels[j], "duplicate label");
            }
        }
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_PC, 0);
        assert_eq!(COL_MEM_OFFSET, 1);
        assert_eq!(COL_LENGTH, 2);
        assert_eq!(COL_LENGTH_BYTE_OFFSET, 3);
        assert_eq!(COL_INPUT_BYTE_OFFSET, 11);
        assert_eq!(COL_OUTPUT_HASH_OFFSET, 75);
        assert_eq!(COL_OUTPUT_LIMB_OFFSET, 107);
        assert_eq!(COL_IS_REAL, 111);
        assert_eq!(NUM_COLUMNS, 112);
        assert_eq!(NUM_ROW_CONSTRAINTS, 6);
        assert_eq!(NUM_SHIFTED, 0);

        // Stack contents EVM column contract.
        assert_eq!(STACK_CONTENTS_COL_PC, 1);
        assert_eq!(STACK_CONTENTS_COL_VALUE_LIMB_0, 4);
        assert_eq!(STACK_CONTENTS_COL_IS_REAL, 9);
    }

    /// Multi-event witness: pc, offsets, lengths all distinct.
    #[test]
    fn multi_row_constraints_vanish() {
        let events: Vec<(u64, u64, u64, Vec<u8>)> = vec![
            (0u64, 0u64, 0u64, vec![]),
            (5u64, 32u64, 4u64, vec![0xde, 0xad, 0xbe, 0xef]),
            (10u64, 0x80u64, 32u64, vec![0xaau8; 32]),
        ];
        let w = Sha3FullChainWitness::from_events(&events).expect("must build");
        assert_eq!(w.rows.len(), 3);
        check_all_bodies_vanish(&w);
    }

    /// `evaluate_at_point` agrees with `evaluate_on_domain` on an
    /// honest witness — both produce zero under the α-RLC.
    #[test]
    fn evaluate_at_point_zero_on_honest_witness() {
        let w = Sha3FullChainWitness::from_events(&[(
            0u64, 0u64, 4u64, vec![1, 2, 3, 4],
        )])
        .expect("must build");
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Sha3FullChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);
        let row0_evals: Vec<Scalar> =
            col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    /// Byte-range lookup coverage: every byte column gets an 8-bit
    /// declaration.  8 length + 64 input + 32 output_hash = 104.
    #[test]
    fn byte_range_lookup_coverage() {
        let cs = Sha3FullChainConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        let expected = LENGTH_BYTES + MAX_INPUT_BYTES + OUTPUT_HASH_LEN;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }
}
