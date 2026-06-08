//! RIPEMD-160 precompile AIR (Ethereum precompile address `0x03`).
//!
//! # Purpose
//!
//! The RIPEMD-160 precompile at Ethereum address `0x03` takes an
//! arbitrary-length byte string `input` and returns
//!
//! ```text
//!     output[0..12]  = 0x00 ... 0x00      (12 leading zero bytes)
//!     output[12..32] = ripemd160(input)   (20-byte RIPEMD-160 digest)
//! ```
//!
//! i.e. the 20-byte RIPEMD-160 hash is left-padded with twelve zero
//! bytes to fit a 32-byte EVM word.
//!
//! # Algebraic surface (this AIR)
//!
//! This AIR commits the precompile invocation as one row and proves the
//! **shape** of the output — that the first twelve bytes are zero and
//! the remaining twenty bytes equal a committed `ripemd_output` column.
//! Per row:
//!
//!   0.        `is_real ∈ {0, 1}`                              — selector binarity.
//!   1..13.    `output[0..12] = 0`                             — 12 zero-pad equalities.
//!   13..33.   `output[12 + i] = ripemd_output[i]` for `i ∈ 0..20` — 20 byte equalities.
//!   33.       LE byte decomp of `input_length` over 8 bytes:
//!             `Σ_b 2^(8b) * input_length_byte[b] - input_length = 0`.
//!
//! Total: `34` row-local constraints. No shifted constraints. 8-bit
//! range checks on every byte column (`input_bytes`, `input_length_byte`,
//! `ripemd_output`, `output`).
//!
//! # Algebraic RIPEMD-160 is **deferred**
//!
//! The RIPEMD-160 compression function is an MD-family construction
//! built from 5 rounds of 16 boolean/shift operations over two parallel
//! lines and a final combiner. Decomposing it into a constraint system
//! akin to the SHA-256 AIR is a substantial effort and is **deferred**.
//! Until that lands this AIR's `ripemd_output` column is **a witness
//! commitment only**: the host-side trace builder fills it from the
//! `ripemd` crate, and the soundness of `ripemd_output =
//! ripemd160(input)` is taken on faith by this AIR. The shape
//! constraints above DO algebraically pin the precompile-output layout
//! (`output = 0^12 || ripemd_output`).
//!
//! # Cross-AIR LogUp descriptors
//!
//!   - [`make_ripemd_to_precompile_dispatch_descriptor`] —
//!     binds `(sel_ripemd, input_length, output_length=32, gas_cost)`
//!     on this AIR's real rows to the matching `precompile_air`
//!     dispatch row for callee `0x03`. The dispatch row enforces the
//!     RIPEMD gas formula `gas = 600 + 120 * ceil(input_length / 32)`.
//!   - [`make_ripemd_to_precompile_io_descriptor`] —
//!     binds `(input_bytes[0..MAX_INPUT_LENGTH], output[0..32])` to
//!     the corresponding `precompile_io_air` row (memory glue).
//!
//! Both descriptors use the same placeholder-sentinel convention as
//! [`crate::kzg_point_eval_air`] for the EVM-side B columns, since
//! `metavm-zkp` does not depend on the EVM crate. The joint-prover
//! orchestrator substitutes the real indices when wiring layers.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum input length supported by this AIR (matches
/// `precompile_io_air::MAX_INPUT_LENGTH`).
pub const MAX_INPUT_LENGTH: usize = 64;
/// RIPEMD-160 digest length (bytes).
pub const RIPEMD_OUTPUT_LENGTH: usize = 20;
/// EVM word size — the precompile returns a 32-byte word.
pub const OUTPUT_LENGTH: usize = 32;
/// Number of leading zero pad bytes in the 32-byte output.
pub const OUTPUT_PAD_LENGTH: usize = OUTPUT_LENGTH - RIPEMD_OUTPUT_LENGTH; // 12
/// Number of LE bytes used to range-check `input_length`.
pub const INPUT_LENGTH_BYTES: usize = 8;

/// EIP precompile id for RIPEMD-160. Mirrors
/// `metavm_evm::precompile_air::PC_RIPEMD160` (= 3) — kept here so
/// the zkp crate doesn't depend on the EVM crate.
pub const PC_RIPEMD160: u64 = 0x03;

/// Per-row precompile output length (fixed at 32).
pub const PRECOMPILE_OUTPUT_LENGTH_U64: u64 = OUTPUT_LENGTH as u64;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_INPUT_BYTES_OFFSET: usize = 0; // 0..64
pub const COL_INPUT_LENGTH: usize = COL_INPUT_BYTES_OFFSET + MAX_INPUT_LENGTH; // 64
pub const COL_INPUT_LENGTH_BYTE_OFFSET: usize = COL_INPUT_LENGTH + 1; // 65..73
pub const COL_RIPEMD_OUTPUT_OFFSET: usize =
    COL_INPUT_LENGTH_BYTE_OFFSET + INPUT_LENGTH_BYTES; // 73..93
pub const COL_OUTPUT_OFFSET: usize = COL_RIPEMD_OUTPUT_OFFSET + RIPEMD_OUTPUT_LENGTH; // 93..125
pub const COL_IS_REAL: usize = COL_OUTPUT_OFFSET + OUTPUT_LENGTH; // 125
pub const COL_GAS_COST: usize = COL_IS_REAL + 1; // 126
pub const NUM_COLUMNS: usize = COL_GAS_COST + 1; // 127

/// Row-local constraints:
///   0:                                is_real ∈ {0, 1}
///   1..1+OUTPUT_PAD_LENGTH:           output[0..12] = 0
///   1+OUTPUT_PAD_LENGTH..
///     1+OUTPUT_PAD_LENGTH+RIPEMD_OUTPUT_LENGTH:
///                                     output[12+i] = ripemd_output[i]
///   last:                             LE byte-decomp of input_length
pub const NUM_ROW_CONSTRAINTS: usize =
    1 + OUTPUT_PAD_LENGTH + RIPEMD_OUTPUT_LENGTH + 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// One RIPEMD-160 precompile invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ripemd160PrecompileWitness {
    /// Input bytes (zero-padded to `MAX_INPUT_LENGTH`).
    pub input_bytes: [u8; MAX_INPUT_LENGTH],
    /// True input length in bytes.
    pub input_length: u64,
    /// `ripemd160(input)` — 20 bytes.
    pub ripemd_output: [u8; RIPEMD_OUTPUT_LENGTH],
    /// Full 32-byte precompile output `0x00..00 || ripemd_output`.
    pub output: [u8; OUTPUT_LENGTH],
    /// Gas cost for the invocation: `600 + 120 * ceil(input_length / 32)`.
    pub gas_cost: u64,
}

impl Ripemd160PrecompileWitness {
    /// Construct an honest witness from arbitrary input bytes by running
    /// the `ripemd::Ripemd160` digest. Inputs longer than
    /// [`MAX_INPUT_LENGTH`] are truncated (scaffold limitation matching
    /// `precompile_io_air`'s single-row cap).
    pub fn from_input(input: &[u8]) -> Self {
        use ripemd::{Digest, Ripemd160};
        let mut hasher = Ripemd160::new();
        hasher.update(input);
        let digest = hasher.finalize();
        let mut ripemd_output = [0u8; RIPEMD_OUTPUT_LENGTH];
        ripemd_output.copy_from_slice(&digest);

        let mut input_bytes = [0u8; MAX_INPUT_LENGTH];
        let n = input.len().min(MAX_INPUT_LENGTH);
        input_bytes[..n].copy_from_slice(&input[..n]);

        let mut output = [0u8; OUTPUT_LENGTH];
        output[OUTPUT_PAD_LENGTH..].copy_from_slice(&ripemd_output);

        let input_length = input.len() as u64;
        let ceil_chunks = input_length.div_ceil(32);
        let gas_cost = 600 + 120 * ceil_chunks;

        Self {
            input_bytes,
            input_length,
            ripemd_output,
            output,
            gas_cost,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Ripemd160PrecompileTraceWitness {
    pub rows: Vec<Ripemd160PrecompileWitness>,
}

impl Ripemd160PrecompileTraceWitness {
    pub fn from_rows(rows: Vec<Ripemd160PrecompileWitness>) -> Self {
        Self { rows }
    }
    pub fn push(&mut self, row: Ripemd160PrecompileWitness) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Ripemd160PrecompileTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..MAX_INPUT_LENGTH {
            columns[COL_INPUT_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.input_bytes[k] as u64, curve);
        }
        columns[COL_INPUT_LENGTH][r] = Scalar::from_u64(row.input_length, curve);

        let len_bytes = row.input_length.to_le_bytes();
        for b in 0..INPUT_LENGTH_BYTES {
            columns[COL_INPUT_LENGTH_BYTE_OFFSET + b][r] =
                Scalar::from_u64(len_bytes[b] as u64, curve);
        }

        for k in 0..RIPEMD_OUTPUT_LENGTH {
            columns[COL_RIPEMD_OUTPUT_OFFSET + k][r] =
                Scalar::from_u64(row.ripemd_output[k] as u64, curve);
        }
        for k in 0..OUTPUT_LENGTH {
            columns[COL_OUTPUT_OFFSET + k][r] =
                Scalar::from_u64(row.output[k] as u64, curve);
        }

        columns[COL_IS_REAL][r] = one.clone();
        columns[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
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

pub struct Ripemd160PrecompileConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Ripemd160PrecompileConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Ripemd160PrecompileConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        for k in 0..OUTPUT_PAD_LENGTH {
            labels.push(format!("output_pad_zero_byte_{}", k));
        }
        for k in 0..RIPEMD_OUTPUT_LENGTH {
            labels.push(format!("output_eq_ripemd_output_byte_{}", k));
        }
        labels.push("input_length_le_byte_decomp".into());
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
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1..1+OUTPUT_PAD_LENGTH: output[0..12] = 0.
        for k in 0..OUTPUT_PAD_LENGTH {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_OUTPUT_OFFSET + k][r].clone();
            }
            out.push(c);
        }

        // 1+OUTPUT_PAD_LENGTH..: output[12+i] = ripemd_output[i].
        for i in 0..RIPEMD_OUTPUT_LENGTH {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = &columns[COL_OUTPUT_OFFSET + OUTPUT_PAD_LENGTH + i][r];
                let rhs = &columns[COL_RIPEMD_OUTPUT_OFFSET + i][r];
                c[r] = lhs.sub(rhs);
            }
            out.push(c);
        }

        // last: LE byte-decomp of input_length.
        // Σ_b 2^(8b) * input_length_byte[b] - input_length = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for b in 0..INPUT_LENGTH_BYTES {
                    let w = Scalar::from_u64(1u64 << (8 * b), curve);
                    let term = columns[COL_INPUT_LENGTH_BYTE_OFFSET + b][r].mul(&w);
                    sum = sum.add(&term);
                }
                c[r] = sum.sub(&columns[COL_INPUT_LENGTH][r]);
            }
            out.push(c);
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output pad zeros.
        for k in 0..OUTPUT_PAD_LENGTH {
            let body = col_evals[COL_OUTPUT_OFFSET + k].clone();
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output[12+i] = ripemd_output[i].
        for i in 0..RIPEMD_OUTPUT_LENGTH {
            let body = col_evals[COL_OUTPUT_OFFSET + OUTPUT_PAD_LENGTH + i]
                .sub(&col_evals[COL_RIPEMD_OUTPUT_OFFSET + i]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // LE byte-decomp of input_length.
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..INPUT_LENGTH_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                sum = sum.add(&col_evals[COL_INPUT_LENGTH_BYTE_OFFSET + b].mul(&w));
            }
            let body = sum.sub(&col_evals[COL_INPUT_LENGTH]);
            acc = acc.add(&alpha_pow.mul(&body));
            // alpha_pow not needed beyond this point.
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

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output pad zeros.
        for k in 0..OUTPUT_PAD_LENGTH {
            let body = col_coeffs[COL_OUTPUT_OFFSET + k].clone();
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output[12+i] = ripemd_output[i].
        for i in 0..RIPEMD_OUTPUT_LENGTH {
            let body = poly_sub(
                &col_coeffs[COL_OUTPUT_OFFSET + OUTPUT_PAD_LENGTH + i],
                &col_coeffs[COL_RIPEMD_OUTPUT_OFFSET + i],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // LE byte-decomp of input_length.
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..INPUT_LENGTH_BYTES {
                let w = Scalar::from_u64(1u64 << (8 * b), curve);
                let term = poly_scalar_mul(&col_coeffs[COL_INPUT_LENGTH_BYTE_OFFSET + b], &w);
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&sum, &col_coeffs[COL_INPUT_LENGTH], curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
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
        for k in 0..MAX_INPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_input_byte_{}_8bit", k),
                    column_index: COL_INPUT_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..INPUT_LENGTH_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_input_length_byte_{}_8bit", b),
                    column_index: COL_INPUT_LENGTH_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..RIPEMD_OUTPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_ripemd_output_byte_{}_8bit", k),
                    column_index: COL_RIPEMD_OUTPUT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..OUTPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("ripemd160_output_byte_{}_8bit", k),
                    column_index: COL_OUTPUT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Placeholder sentinel for `precompile_air` column indices.
///
/// Same convention as [`crate::kzg_point_eval_air`]: the zkp crate does
/// not depend on the EVM crate, so the EVM-side B-column indices are
/// stubbed with `usize::MAX` and substituted by the joint-prover
/// orchestrator when the layers are wired.
pub const PRECOMPILE_DISPATCH_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds
/// `(sel_ripemd = 1, input_length, output_length = 32, gas_cost)` on
/// this AIR's `IS_REAL` rows to the matching `precompile_air` dispatch
/// row for callee `0x03`. The A side commits one synthetic literal
/// column (the constant `sel_ripemd = 1`, represented here by the
/// `IS_REAL` column itself — every real row of this AIR IS a
/// RIPEMD-160 row) plus `input_length`, a synthetic constant
/// `output_length = 32` (placeholder), and `gas_cost`.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B-side
/// columns are filled with [`PRECOMPILE_DISPATCH_PLACEHOLDER`]. The
/// A-side `output_length` placeholder records the **shape** of the
/// binding — the joint-prover orchestrator substitutes the real
/// indices.
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_ripemd_to_precompile_dispatch_descriptor(
    ripemd_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // A side: (sel_ripemd=is_real, input_length, output_length_placeholder, gas_cost)
    let a_columns: Vec<usize> = vec![
        COL_IS_REAL,
        COL_INPUT_LENGTH,
        PRECOMPILE_DISPATCH_PLACEHOLDER, // synthetic output_length = 32
        COL_GAS_COST,
    ];
    let b_columns: Vec<usize> = vec![
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_SEL_RIPEMD (= 12)
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_INPUT_LENGTH (= 4)
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_OUTPUT_LENGTH (= 5)
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_GAS_COST (= 6)
    ];
    CrossAirLogUpDescriptor {
        label: "ripemd_to_precompile_dispatch_v1_stub".into(),
        a_layer_index: ripemd_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        // precompile_air::COL_SEL_RIPEMD on the B side; orchestrator
        // substitutes.
        b_selector_column: None,
    }
}

/// Placeholder sentinel for `precompile_io_air` column indices.
pub const PRECOMPILE_IO_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds the precompile-row
/// `input_bytes[0..MAX_INPUT_LENGTH]` and full padded `output[0..32]`
/// byte arrays to the corresponding `precompile_io_air` row that
/// records the EVM-side CALL memory I/O.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B side is
/// filled with [`PRECOMPILE_IO_PLACEHOLDER`]. The A side is the full
/// input + output column range on this AIR; the orchestrator
/// substitutes the real B-side indices once `precompile_io_air` gains
/// a dedicated RIPEMD output column (today it carries only SHA256 +
/// IDENTITY output channels).
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_ripemd_to_precompile_io_descriptor(
    ripemd_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_LENGTH + OUTPUT_LENGTH);
    for k in 0..MAX_INPUT_LENGTH {
        a_columns.push(COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..OUTPUT_LENGTH {
        a_columns.push(COL_OUTPUT_OFFSET + k);
    }
    let b_columns: Vec<usize> =
        vec![PRECOMPILE_IO_PLACEHOLDER; MAX_INPUT_LENGTH + OUTPUT_LENGTH];
    CrossAirLogUpDescriptor {
        label: "ripemd_to_precompile_io_v1_stub".into(),
        a_layer_index: ripemd_layer_index,
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

    /// RIPEMD-160 of the empty string. Standard test vector:
    /// `0x9c1185a5c5e9fc54612808977ee8f548b2258d31`.
    const RIPEMD_EMPTY: [u8; RIPEMD_OUTPUT_LENGTH] = [
        0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9, 0xfc, 0x54,
        0x61, 0x28, 0x08, 0x97, 0x7e, 0xe8, 0xf5, 0x48,
        0xb2, 0x25, 0x8d, 0x31,
    ];

    /// RIPEMD-160 of `b"abc"`. Standard test vector:
    /// `0x8eb208f7e05d987a9b044a8e98c6b087f15a0bfc`.
    const RIPEMD_ABC: [u8; RIPEMD_OUTPUT_LENGTH] = [
        0x8e, 0xb2, 0x08, 0xf7, 0xe0, 0x5d, 0x98, 0x7a,
        0x9b, 0x04, 0x4a, 0x8e, 0x98, 0xc6, 0xb0, 0x87,
        0xf1, 0x5a, 0x0b, 0xfc,
    ];

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} row {} nonzero: {:?}",
                    i, r, v,
                );
            }
        }
    }

    #[test]
    fn ripemd160_precompile_empty_input_known_hash() {
        let w = Ripemd160PrecompileWitness::from_input(b"");
        assert_eq!(w.input_length, 0);
        assert_eq!(w.ripemd_output, RIPEMD_EMPTY);
        // First 12 bytes of output are zero.
        for k in 0..OUTPUT_PAD_LENGTH {
            assert_eq!(w.output[k], 0);
        }
        // Remaining 20 bytes equal the digest.
        assert_eq!(&w.output[OUTPUT_PAD_LENGTH..], &RIPEMD_EMPTY[..]);
        // Empty input → ceil(0/32) = 0 → gas = 600.
        assert_eq!(w.gas_cost, 600);

        // Constraints zero on honest witness.
        let trace = build_trace_polynomials(
            &Ripemd160PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Ripemd160PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_precompile_abc_known_hash() {
        let w = Ripemd160PrecompileWitness::from_input(b"abc");
        assert_eq!(w.input_length, 3);
        assert_eq!(w.ripemd_output, RIPEMD_ABC);
        assert_eq!(&w.output[OUTPUT_PAD_LENGTH..], &RIPEMD_ABC[..]);
        // 3 bytes → ceil(3/32) = 1 → gas = 600 + 120 = 720.
        assert_eq!(w.gas_cost, 720);

        let trace = build_trace_polynomials(
            &Ripemd160PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Ripemd160PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_precompile_tampered_output_detected() {
        let w = Ripemd160PrecompileWitness::from_input(b"abc");
        let trace = build_trace_polynomials(
            &Ripemd160PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        // Clone columns and tamper with output[15] (which should equal
        // ripemd_output[3] = 0xf7).
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_OUTPUT_OFFSET + 15][0] =
            Scalar::from_u64(0xab, CurveType::Bls48581);

        let cs = Ripemd160PrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index for output[12+i] = ripemd_output[i] at i=3:
        //   1 (is_real) + 12 (pad zeros) + 3 = 16.
        let tampered_idx = 1 + OUTPUT_PAD_LENGTH + 3;
        assert!(
            !bodies[tampered_idx][0].is_zero(),
            "expected output==ripemd_output constraint at index {} to fire",
            tampered_idx,
        );

        // Also tamper a pad byte: should fire pad constraint.
        let mut cols2: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols2[COL_OUTPUT_OFFSET + 5][0] =
            Scalar::from_u64(0x42, CurveType::Bls48581);
        let col_refs2: Vec<&Vec<Scalar>> = cols2.iter().collect();
        let bodies2 = cs.evaluate_on_domain(&col_refs2, trace.num_rows);
        // Pad constraint at k=5: index 1 + 5 = 6.
        assert!(
            !bodies2[1 + 5][0].is_zero(),
            "expected pad-zero constraint at index 6 to fire",
        );
    }

    #[test]
    fn ripemd160_precompile_input_length_tamper_detected() {
        let w = Ripemd160PrecompileWitness::from_input(b"abc");
        let trace = build_trace_polynomials(
            &Ripemd160PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper input_length scalar without touching the LE bytes:
        // byte-decomp constraint should fire.
        cols[COL_INPUT_LENGTH][0] = Scalar::from_u64(999, CurveType::Bls48581);

        let cs = Ripemd160PrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let last = NUM_ROW_CONSTRAINTS - 1;
        assert!(
            !bodies[last][0].is_zero(),
            "expected input_length byte-decomp constraint to fire",
        );
    }

    #[test]
    fn ripemd160_precompile_descriptors_well_formed() {
        let d1 = make_ripemd_to_precompile_dispatch_descriptor(0, 1);
        assert_eq!(d1.label, "ripemd_to_precompile_dispatch_v1_stub");
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_eq!(d1.a_columns.len(), 4);
        assert_eq!(d1.b_columns.len(), 4);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        // A side carries (sel_ripemd=is_real, input_length, _, gas_cost).
        assert_eq!(d1.a_columns[0], COL_IS_REAL);
        assert_eq!(d1.a_columns[1], COL_INPUT_LENGTH);
        assert_eq!(d1.a_columns[2], PRECOMPILE_DISPATCH_PLACEHOLDER);
        assert_eq!(d1.a_columns[3], COL_GAS_COST);
        for b in &d1.b_columns {
            assert_eq!(*b, PRECOMPILE_DISPATCH_PLACEHOLDER);
        }

        let d2 = make_ripemd_to_precompile_io_descriptor(0, 2);
        assert_eq!(d2.label, "ripemd_to_precompile_io_v1_stub");
        assert_eq!(d2.a_columns.len(), MAX_INPUT_LENGTH + OUTPUT_LENGTH);
        assert_eq!(d2.b_columns.len(), MAX_INPUT_LENGTH + OUTPUT_LENGTH);
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
        // First MAX_INPUT_LENGTH A cols are input bytes.
        for k in 0..MAX_INPUT_LENGTH {
            assert_eq!(d2.a_columns[k], COL_INPUT_BYTES_OFFSET + k);
        }
        // Next OUTPUT_LENGTH A cols are output bytes.
        for k in 0..OUTPUT_LENGTH {
            assert_eq!(
                d2.a_columns[MAX_INPUT_LENGTH + k],
                COL_OUTPUT_OFFSET + k,
            );
        }
        // All B cols are placeholders.
        for b in &d2.b_columns {
            assert_eq!(*b, PRECOMPILE_IO_PLACEHOLDER);
        }
    }

    #[test]
    fn ripemd160_precompile_column_layout_pinned() {
        // Pin numeric offsets so future column additions trip this test.
        assert_eq!(COL_INPUT_BYTES_OFFSET, 0);
        assert_eq!(COL_INPUT_LENGTH, 64);
        assert_eq!(COL_INPUT_LENGTH_BYTE_OFFSET, 65);
        assert_eq!(COL_RIPEMD_OUTPUT_OFFSET, 73);
        assert_eq!(COL_OUTPUT_OFFSET, 93);
        assert_eq!(COL_IS_REAL, 125);
        assert_eq!(COL_GAS_COST, 126);
        assert_eq!(NUM_COLUMNS, 127);

        // Constraints: 1 + 12 + 20 + 1 = 34.
        assert_eq!(NUM_ROW_CONSTRAINTS, 34);
        assert_eq!(NUM_SHIFTED, 0);

        // Constants.
        assert_eq!(MAX_INPUT_LENGTH, 64);
        assert_eq!(RIPEMD_OUTPUT_LENGTH, 20);
        assert_eq!(OUTPUT_LENGTH, 32);
        assert_eq!(OUTPUT_PAD_LENGTH, 12);
        assert_eq!(INPUT_LENGTH_BYTES, 8);
        assert_eq!(PC_RIPEMD160, 0x03);

        // Labels match wired constraint count.
        let cs = Ripemd160PrecompileConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
    }

    #[test]
    fn ripemd160_precompile_multi_row_passes() {
        let rows = vec![
            Ripemd160PrecompileWitness::from_input(b""),
            Ripemd160PrecompileWitness::from_input(b"abc"),
            Ripemd160PrecompileWitness::from_input(b"the quick brown fox"),
        ];
        let trace = build_trace_polynomials(
            &Ripemd160PrecompileTraceWitness::from_rows(rows),
            CurveType::Bls48581,
        );
        let cs = Ripemd160PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn ripemd160_precompile_long_input_truncates_to_max_length() {
        // 96-byte input — longer than MAX_INPUT_LENGTH. The witness's
        // `input_bytes` field is bounded at 64 bytes (scaffold limit);
        // input_length still records the true length so the dispatch
        // gas binding works. Honest constraints still pass since the
        // ripemd output is computed over the FULL input.
        let input: Vec<u8> = (0..96u8).collect();
        let w = Ripemd160PrecompileWitness::from_input(&input);
        assert_eq!(w.input_length, 96);
        // input_bytes truncated to first 64 bytes.
        for k in 0..MAX_INPUT_LENGTH {
            assert_eq!(w.input_bytes[k], input[k]);
        }
        // gas = 600 + 120 * ceil(96/32) = 600 + 360 = 960.
        assert_eq!(w.gas_cost, 960);
        // Output[12..32] is the digest over the FULL 96 bytes; the AIR
        // doesn't algebraically pin RIPEMD itself so this is taken on
        // faith (see module doc).
        let trace = build_trace_polynomials(
            &Ripemd160PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Ripemd160PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }
}
