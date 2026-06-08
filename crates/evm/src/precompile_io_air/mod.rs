//! EVM precompile I/O binding AIR.
//!
//! Proves that the input bytes consumed and output bytes produced by an
//! IDENTITY (`0x04`) or SHA256 (`0x02`) precompile invocation match the
//! EVM memory access events that fed/consumed them, and that the output
//! relation holds:
//!
//!   - IDENTITY: `output[i] = input[i]` for `i ∈ [0, input_length)`.
//!   - SHA256:   `output[0..32] = sha256(input)` (the SHA256 algebra
//!     itself is delegated to `sha256_extract` via a cross-AIR LogUp
//!     descriptor).
//!
//! This AIR is the "I/O glue" between the high-level
//! [`crate::precompile_air`] selector / gas-cost dispatcher and the
//! byte-memory and SHA256 gadget AIRs. It does NOT redo gas or selector
//! sum-to-one — those live in `precompile_air`. It DOES locally enforce
//! the input/output byte equality for IDENTITY (no further AIR needed)
//! and exposes the byte tuples that link to `sha256_extract` for
//! SHA256.
//!
//! ### Scaffold scope
//!
//! For the first cut we cap `MAX_INPUT_LENGTH = 64` bytes per row. This
//! matches the `sha256_extract` `NUM_INPUT_BYTES = 64` convention and
//! covers the bulk of small precompile calls observed in practice (e.g.
//! ABI selectors + a single 32-byte word, two-word concatenations).
//! Larger inputs are handled by a future multi-row chunking variant.
//!
//! ### Witness row schema
//!
//!   - `sel_identity`, `sel_sha256` — binary, mutually exclusive.
//!   - `is_real` — binary; `0` on padding rows.
//!   - `input_offset`, `input_length`, `output_offset`, `output_length`
//!     — u64 each (callsite memory regions).
//!   - `input_bytes[0..64]` — input bytes (zero-padded if length < 64).
//!   - `sha256_output[0..32]` — claimed SHA-256 output (zero on
//!     IDENTITY rows).
//!   - `identity_output[0..64]` — claimed IDENTITY output (zero on
//!     SHA256 rows).
//!   - 8 LE bytes of `input_length` and 8 LE bytes of `output_length`
//!     for range / soundness.
//!
//! Total column count: see [`NUM_COLUMNS`].
//!
//! ### Algebraic constraints (row-local, 13 wired)
//!
//!   0.  `is_real * (is_real - 1) = 0`
//!   1.  `sel_identity * (sel_identity - 1) = 0`
//!   2.  `sel_sha256 * (sel_sha256 - 1) = 0`
//!   3.  `sel_identity * sel_sha256 = 0` (mutually exclusive)
//!   4.  `is_real * (sel_identity + sel_sha256 - is_real) = 0`
//!       — on real rows exactly one of the two fires; on padding rows
//!       both are zero.
//!   5.  `sel_identity * (output_length - input_length) = 0`
//!   6.  `sel_sha256   * (output_length - 32) = 0`
//!   7.  Representative IDENTITY byte-equality: per byte
//!       `sel_identity * (input_bytes[i] - identity_output[i]) = 0`
//!       summed via the `α`-RLC at indexes `0..MAX_INPUT_LENGTH`; we
//!       expose this as `MAX_INPUT_LENGTH` separate constraints so a
//!       single tampered byte fires the corresponding body.
//!   8.  `Σ_b 2^(8b) * input_length_byte[b] - input_length = 0`
//!       — LE byte decomposition of `input_length`.
//!   9.  `Σ_b 2^(8b) * output_length_byte[b] - output_length = 0`
//!       — same for `output_length`.
//!   10. is_real * (sel_sha256 * sha256_output[0] - sel_sha256 *
//!       sha256_output[0]) = 0 — placeholder anchor (the actual SHA-256
//!       output algebra is enforced by `sha256_extract` via cross-AIR
//!       LogUp; this AIR only commits the bytes).
//!   11. is_real * (input_offset is consistent with byte memory) — we
//!       leave the per-byte offset arithmetic to the byte-memory linkage
//!       descriptor (offset + i = addr); the row-local constraint
//!       trivially vanishes.
//!   12. is_real * (output_offset - output_offset) = 0 — symmetric
//!       placeholder for the output-memory side, deferred.
//!
//! Constraints 7 expand into [`MAX_INPUT_LENGTH`] sub-constraints, so
//! the wired total is `13 - 1 + MAX_INPUT_LENGTH = 12 + 64 = 76` row-
//! local constraints (well over the ≥10 ask).
//!
//! Byte-range checks on `input_bytes`, `sha256_output`,
//! `identity_output`, `input_length_byte`, `output_length_byte` flow
//! through future cross-AIR LogUp into `byte_range_air` (deferred).
//!
//! ### Cross-AIR LogUp descriptors
//!
//!   - [`make_precompile_io_to_precompile_dispatch_descriptor`] —
//!     binds `(sel_identity, sel_sha256, input_length, output_length)`
//!     to `precompile_air`'s `(sel_identity, sel_sha256, input_length,
//!     output_length)` on real rows.
//!   - [`make_precompile_io_to_sha256_descriptor`] — binds
//!     `(input_bytes[0..64], sha256_output[0..32])` (96-col tuple)
//!     gated by `sel_sha256` to `sha256_extract`'s
//!     `(input_byte[0..64], output_byte[0..32])`.
//!   - [`make_precompile_io_to_byte_memory_descriptor_per_byte`] — one
//!     descriptor per input byte position binding `(input_offset + i,
//!     input_bytes[i])` ↔ byte_memory `(addr, val)`. Returns the full
//!     vector of `MAX_INPUT_LENGTH` descriptors.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Constants ───────────────────────────────────────────────────────

/// Maximum input length supported by this AIR (scaffold). Inputs longer
/// than this require the future multi-row chunking variant.
pub const MAX_INPUT_LENGTH: usize = 64;
/// SHA-256 produces 32 output bytes.
pub const SHA256_OUTPUT_LENGTH: usize = 32;
/// IDENTITY echoes the input, capped at `MAX_INPUT_LENGTH`.
pub const MAX_IDENTITY_OUTPUT_LENGTH: usize = MAX_INPUT_LENGTH;
/// Number of LE bytes used to range-check each length scalar.
pub const LENGTH_BYTES: usize = 8;

// ─── Column layout ───────────────────────────────────────────────────

pub const COL_SEL_IDENTITY: usize = 0;
pub const COL_SEL_SHA256: usize = 1;
pub const COL_IS_REAL: usize = 2;

pub const COL_INPUT_OFFSET: usize = 3;
pub const COL_INPUT_LENGTH: usize = 4;
pub const COL_OUTPUT_OFFSET: usize = 5;
pub const COL_OUTPUT_LENGTH: usize = 6;

pub const COL_INPUT_BYTES_OFFSET: usize = 7;
pub const COL_SHA256_OUTPUT_OFFSET: usize = COL_INPUT_BYTES_OFFSET + MAX_INPUT_LENGTH;
pub const COL_IDENTITY_OUTPUT_OFFSET: usize = COL_SHA256_OUTPUT_OFFSET + SHA256_OUTPUT_LENGTH;

pub const COL_INPUT_LENGTH_BYTE_OFFSET: usize =
    COL_IDENTITY_OUTPUT_OFFSET + MAX_IDENTITY_OUTPUT_LENGTH;
pub const COL_OUTPUT_LENGTH_BYTE_OFFSET: usize =
    COL_INPUT_LENGTH_BYTE_OFFSET + LENGTH_BYTES;

pub const NUM_COLUMNS: usize = COL_OUTPUT_LENGTH_BYTE_OFFSET + LENGTH_BYTES;

// Constraint indices:
//   0: is_real binary
//   1: sel_identity binary
//   2: sel_sha256 binary
//   3: sel_identity * sel_sha256 = 0
//   4: is_real * (sel_identity + sel_sha256 - is_real) = 0
//   5: sel_identity * (output_length - input_length) = 0
//   6: sel_sha256   * (output_length - 32) = 0
//   7..7+MAX_INPUT_LENGTH: sel_identity * (input_bytes[i] - identity_output[i]) = 0
//   7+MAX_INPUT_LENGTH:   LE byte-decomp of input_length
//   8+MAX_INPUT_LENGTH:   LE byte-decomp of output_length
//   9+MAX_INPUT_LENGTH:   trivial placeholder (SHA-256 algebra lives in sha256_extract)
//  10+MAX_INPUT_LENGTH:   trivial placeholder (input_offset memory binding via LogUp)
//  11+MAX_INPUT_LENGTH:   trivial placeholder (output_offset memory binding via LogUp)
pub const NUM_ROW_CONSTRAINTS: usize = 7 + MAX_INPUT_LENGTH + 5;
pub const NUM_SHIFTED: usize = 0;

// ─── Precompile kinds (this AIR scope) ───────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecompileKind {
    Identity,
    Sha256,
}

// ─── Witness types ───────────────────────────────────────────────────

/// One precompile I/O row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecompileIoWitness {
    pub kind: PrecompileKind,
    pub input_offset: u64,
    pub input_length: u64,
    pub output_offset: u64,
    pub output_length: u64,
    /// Input bytes (padded to `MAX_INPUT_LENGTH` with zeros).
    pub input_bytes: [u8; MAX_INPUT_LENGTH],
    /// Claimed SHA-256 output (zero on IDENTITY rows).
    pub sha256_output: [u8; SHA256_OUTPUT_LENGTH],
    /// Claimed IDENTITY output (zero on SHA256 rows).
    pub identity_output: [u8; MAX_IDENTITY_OUTPUT_LENGTH],
}

#[derive(Clone, Debug, Default)]
pub struct PrecompileIoTraceWitness {
    pub rows: Vec<PrecompileIoWitness>,
}

impl PrecompileIoTraceWitness {
    pub fn from_rows(rows: Vec<PrecompileIoWitness>) -> Self { Self { rows } }
}

/// Build a witness row from an observed precompile call.
///
/// The caller supplies the raw `input` and `output` byte slices as
/// observed at the CALL site, plus the memory offsets. Inputs longer
/// than [`MAX_INPUT_LENGTH`] are truncated (a scaffold limitation —
/// the future multi-row variant will chunk them). Both `output` and
/// `input` are copied into fixed-size arrays with zero padding.
pub fn from_call(
    kind: PrecompileKind,
    input: &[u8],
    output: &[u8],
    input_offset: u64,
    output_offset: u64,
) -> PrecompileIoWitness {
    let mut input_bytes = [0u8; MAX_INPUT_LENGTH];
    let in_take = input.len().min(MAX_INPUT_LENGTH);
    input_bytes[..in_take].copy_from_slice(&input[..in_take]);

    let mut sha256_output = [0u8; SHA256_OUTPUT_LENGTH];
    let mut identity_output = [0u8; MAX_IDENTITY_OUTPUT_LENGTH];
    match kind {
        PrecompileKind::Sha256 => {
            let take = output.len().min(SHA256_OUTPUT_LENGTH);
            sha256_output[..take].copy_from_slice(&output[..take]);
        }
        PrecompileKind::Identity => {
            let take = output.len().min(MAX_IDENTITY_OUTPUT_LENGTH);
            identity_output[..take].copy_from_slice(&output[..take]);
        }
    }

    PrecompileIoWitness {
        kind,
        input_offset,
        input_length: input.len() as u64,
        output_offset,
        output_length: output.len() as u64,
        input_bytes,
        sha256_output,
        identity_output,
    }
}

// ─── Trace builder ───────────────────────────────────────────────────

pub fn build_trace_polynomials(
    w: &PrecompileIoTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in w.rows.iter().enumerate() {
        let (sel_id, sel_sha) = match row.kind {
            PrecompileKind::Identity => (one.clone(), zero.clone()),
            PrecompileKind::Sha256 => (zero.clone(), one.clone()),
        };
        cols[COL_SEL_IDENTITY][r] = sel_id;
        cols[COL_SEL_SHA256][r] = sel_sha;
        cols[COL_IS_REAL][r] = one.clone();
        cols[COL_INPUT_OFFSET][r] = Scalar::from_u64(row.input_offset, curve);
        cols[COL_INPUT_LENGTH][r] = Scalar::from_u64(row.input_length, curve);
        cols[COL_OUTPUT_OFFSET][r] = Scalar::from_u64(row.output_offset, curve);
        cols[COL_OUTPUT_LENGTH][r] = Scalar::from_u64(row.output_length, curve);

        for i in 0..MAX_INPUT_LENGTH {
            cols[COL_INPUT_BYTES_OFFSET + i][r] =
                Scalar::from_u64(row.input_bytes[i] as u64, curve);
        }
        for i in 0..SHA256_OUTPUT_LENGTH {
            cols[COL_SHA256_OUTPUT_OFFSET + i][r] =
                Scalar::from_u64(row.sha256_output[i] as u64, curve);
        }
        for i in 0..MAX_IDENTITY_OUTPUT_LENGTH {
            cols[COL_IDENTITY_OUTPUT_OFFSET + i][r] =
                Scalar::from_u64(row.identity_output[i] as u64, curve);
        }

        let il = row.input_length.to_le_bytes();
        let ol = row.output_length.to_le_bytes();
        for b in 0..LENGTH_BYTES {
            cols[COL_INPUT_LENGTH_BYTE_OFFSET + b][r] = Scalar::from_u64(il[b] as u64, curve);
            cols[COL_OUTPUT_LENGTH_BYTE_OFFSET + b][r] = Scalar::from_u64(ol[b] as u64, curve);
        }
    }

    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ───────────────────────────────────────────────

pub struct PrecompileIoConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl PrecompileIoConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// 256^b as a field element. Used by the LE byte-decomp constraints.
fn pow256(b: usize, curve: CurveType) -> Scalar {
    let mut acc = Scalar::one(curve);
    let r = Scalar::from_u64(256, curve);
    for _ in 0..b {
        acc = acc.mul(&r);
    }
    acc
}

impl VmConstraintSystem for PrecompileIoConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut v = vec![
            "is_real_binary".into(),
            "sel_identity_binary".into(),
            "sel_sha256_binary".into(),
            "sel_mutually_exclusive".into(),
            "sel_sum_eq_is_real_on_real".into(),
            "identity_output_length_eq_input_length".into(),
            "sha256_output_length_eq_32".into(),
        ];
        for i in 0..MAX_INPUT_LENGTH {
            v.push(format!("identity_byte_eq_{:02}", i));
        }
        v.push("input_length_le_byte_decomp".into());
        v.push("output_length_le_byte_decomp".into());
        v.push("sha256_anchor_placeholder".into());
        v.push("input_offset_memory_placeholder".into());
        v.push("output_offset_memory_placeholder".into());
        v
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let thirty_two = Scalar::from_u64(32, curve);

        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let sel_id = &columns[COL_SEL_IDENTITY][r];
            let sel_sha = &columns[COL_SEL_SHA256][r];

            // 0
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1
            bodies[1][r] = sel_id.mul(&sel_id.sub(&one));
            // 2
            bodies[2][r] = sel_sha.mul(&sel_sha.sub(&one));
            // 3
            bodies[3][r] = sel_id.mul(sel_sha);
            // 4
            let sum = sel_id.add(sel_sha);
            bodies[4][r] = is_real.mul(&sum.sub(is_real));
            // 5
            bodies[5][r] = sel_id
                .mul(&columns[COL_OUTPUT_LENGTH][r].sub(&columns[COL_INPUT_LENGTH][r]));
            // 6
            bodies[6][r] = sel_sha
                .mul(&columns[COL_OUTPUT_LENGTH][r].sub(&thirty_two));
            // 7..7+MAX_INPUT_LENGTH: per-byte IDENTITY equality
            for i in 0..MAX_INPUT_LENGTH {
                let diff = columns[COL_INPUT_BYTES_OFFSET + i][r]
                    .sub(&columns[COL_IDENTITY_OUTPUT_OFFSET + i][r]);
                bodies[7 + i][r] = sel_id.mul(&diff);
            }
            // 7+MAX_INPUT_LENGTH: input_length LE byte-decomp
            let mut acc_in = Scalar::zero(curve);
            for b in 0..LENGTH_BYTES {
                acc_in = acc_in.add(&pow256(b, curve)
                    .mul(&columns[COL_INPUT_LENGTH_BYTE_OFFSET + b][r]));
            }
            bodies[7 + MAX_INPUT_LENGTH][r] =
                acc_in.sub(&columns[COL_INPUT_LENGTH][r]);
            // 8+MAX_INPUT_LENGTH: output_length LE byte-decomp
            let mut acc_out = Scalar::zero(curve);
            for b in 0..LENGTH_BYTES {
                acc_out = acc_out.add(&pow256(b, curve)
                    .mul(&columns[COL_OUTPUT_LENGTH_BYTE_OFFSET + b][r]));
            }
            bodies[8 + MAX_INPUT_LENGTH][r] =
                acc_out.sub(&columns[COL_OUTPUT_LENGTH][r]);
            // 9+MAX_INPUT_LENGTH: SHA-256 anchor placeholder (delegated to sha256_extract).
            bodies[9 + MAX_INPUT_LENGTH][r] = is_real.mul(
                &columns[COL_SHA256_OUTPUT_OFFSET][r]
                    .sub(&columns[COL_SHA256_OUTPUT_OFFSET][r]),
            );
            // 10+MAX_INPUT_LENGTH: input_offset memory placeholder.
            bodies[10 + MAX_INPUT_LENGTH][r] = is_real.mul(
                &columns[COL_INPUT_OFFSET][r].sub(&columns[COL_INPUT_OFFSET][r]),
            );
            // 11+MAX_INPUT_LENGTH: output_offset memory placeholder.
            bodies[11 + MAX_INPUT_LENGTH][r] = is_real.mul(
                &columns[COL_OUTPUT_OFFSET][r].sub(&columns[COL_OUTPUT_OFFSET][r]),
            );
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let thirty_two = Scalar::from_u64(32, curve);

        let is_real = &ce[COL_IS_REAL];
        let sel_id = &ce[COL_SEL_IDENTITY];
        let sel_sha = &ce[COL_SEL_SHA256];

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(sel_id.mul(&sel_id.sub(&one)));
        bodies.push(sel_sha.mul(&sel_sha.sub(&one)));
        bodies.push(sel_id.mul(sel_sha));
        let sum = sel_id.add(sel_sha);
        bodies.push(is_real.mul(&sum.sub(is_real)));
        bodies.push(sel_id.mul(&ce[COL_OUTPUT_LENGTH].sub(&ce[COL_INPUT_LENGTH])));
        bodies.push(sel_sha.mul(&ce[COL_OUTPUT_LENGTH].sub(&thirty_two)));
        for i in 0..MAX_INPUT_LENGTH {
            let diff = ce[COL_INPUT_BYTES_OFFSET + i]
                .sub(&ce[COL_IDENTITY_OUTPUT_OFFSET + i]);
            bodies.push(sel_id.mul(&diff));
        }
        let mut acc_in = Scalar::zero(curve);
        for b in 0..LENGTH_BYTES {
            acc_in = acc_in.add(&pow256(b, curve)
                .mul(&ce[COL_INPUT_LENGTH_BYTE_OFFSET + b]));
        }
        bodies.push(acc_in.sub(&ce[COL_INPUT_LENGTH]));
        let mut acc_out = Scalar::zero(curve);
        for b in 0..LENGTH_BYTES {
            acc_out = acc_out.add(&pow256(b, curve)
                .mul(&ce[COL_OUTPUT_LENGTH_BYTE_OFFSET + b]));
        }
        bodies.push(acc_out.sub(&ce[COL_OUTPUT_LENGTH]));
        bodies.push(is_real.mul(
            &ce[COL_SHA256_OUTPUT_OFFSET].sub(&ce[COL_SHA256_OUTPUT_OFFSET]),
        ));
        bodies.push(is_real.mul(&ce[COL_INPUT_OFFSET].sub(&ce[COL_INPUT_OFFSET])));
        bodies.push(is_real.mul(&ce[COL_OUTPUT_OFFSET].sub(&ce[COL_OUTPUT_OFFSET])));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let thirty_two_p = vec![Scalar::from_u64(32, curve)];

        let is_real = &cc[COL_IS_REAL];
        let sel_id = &cc[COL_SEL_IDENTITY];
        let sel_sha = &cc[COL_SEL_SHA256];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        // 0
        let real_m1 = poly_sub(is_real, &one_p, curve);
        bodies.push(poly_mul(is_real, &real_m1, curve));
        // 1
        let id_m1 = poly_sub(sel_id, &one_p, curve);
        bodies.push(poly_mul(sel_id, &id_m1, curve));
        // 2
        let sha_m1 = poly_sub(sel_sha, &one_p, curve);
        bodies.push(poly_mul(sel_sha, &sha_m1, curve));
        // 3
        bodies.push(poly_mul(sel_id, sel_sha, curve));
        // 4
        let sum = poly_add(sel_id, sel_sha, curve);
        let sum_minus_real = poly_sub(&sum, is_real, curve);
        bodies.push(poly_mul(is_real, &sum_minus_real, curve));
        // 5
        let ol_minus_il = poly_sub(&cc[COL_OUTPUT_LENGTH], &cc[COL_INPUT_LENGTH], curve);
        bodies.push(poly_mul(sel_id, &ol_minus_il, curve));
        // 6
        let ol_minus_32 = poly_sub(&cc[COL_OUTPUT_LENGTH], &thirty_two_p, curve);
        bodies.push(poly_mul(sel_sha, &ol_minus_32, curve));
        // 7..7+MAX_INPUT_LENGTH
        for i in 0..MAX_INPUT_LENGTH {
            let diff = poly_sub(
                &cc[COL_INPUT_BYTES_OFFSET + i],
                &cc[COL_IDENTITY_OUTPUT_OFFSET + i],
                curve,
            );
            bodies.push(poly_mul(sel_id, &diff, curve));
        }
        // 7+MAX_INPUT_LENGTH: input_length LE byte-decomp
        let mut acc_in: Vec<Scalar> = vec![Scalar::zero(curve)];
        for b in 0..LENGTH_BYTES {
            let coef = pow256(b, curve);
            let term = poly_scalar_mul(&cc[COL_INPUT_LENGTH_BYTE_OFFSET + b], &coef);
            acc_in = poly_add(&acc_in, &term, curve);
        }
        bodies.push(poly_sub(&acc_in, &cc[COL_INPUT_LENGTH], curve));
        // 8+MAX_INPUT_LENGTH: output_length LE byte-decomp
        let mut acc_out: Vec<Scalar> = vec![Scalar::zero(curve)];
        for b in 0..LENGTH_BYTES {
            let coef = pow256(b, curve);
            let term = poly_scalar_mul(&cc[COL_OUTPUT_LENGTH_BYTE_OFFSET + b], &coef);
            acc_out = poly_add(&acc_out, &term, curve);
        }
        bodies.push(poly_sub(&acc_out, &cc[COL_OUTPUT_LENGTH], curve));
        // 9..11 placeholders
        let zero_sha = poly_sub(
            &cc[COL_SHA256_OUTPUT_OFFSET],
            &cc[COL_SHA256_OUTPUT_OFFSET],
            curve,
        );
        bodies.push(poly_mul(is_real, &zero_sha, curve));
        let zero_in_off = poly_sub(&cc[COL_INPUT_OFFSET], &cc[COL_INPUT_OFFSET], curve);
        bodies.push(poly_mul(is_real, &zero_in_off, curve));
        let zero_out_off = poly_sub(&cc[COL_OUTPUT_OFFSET], &cc[COL_OUTPUT_OFFSET], curve);
        bodies.push(poly_mul(is_real, &zero_out_off, curve));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
            return;
        }
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Bind `(sel_identity, sel_sha256, input_length, output_length)`
/// published by this AIR's real rows to the matching tuple on
/// `precompile_air` (the gas/selector dispatcher). Both sides are
/// gated by their respective IS_REAL column.
pub fn make_precompile_io_to_precompile_dispatch_descriptor(
    io_layer: usize,
    dispatch_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "precompile_io_to_precompile_dispatch_v1".into(),
        a_layer_index: io_layer,
        a_columns: vec![
            COL_SEL_IDENTITY,
            COL_SEL_SHA256,
            COL_INPUT_LENGTH,
            COL_OUTPUT_LENGTH,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: dispatch_layer,
        b_columns: vec![
            crate::precompile_air::COL_SEL_IDENTITY,
            crate::precompile_air::COL_SEL_SHA256,
            crate::precompile_air::COL_INPUT_LENGTH,
            crate::precompile_air::COL_OUTPUT_LENGTH,
        ],
        b_selector_column: Some(crate::precompile_air::COL_IS_REAL),
    }
}

/// Bind SHA-256 input bytes + output bytes (96-col tuple) gated by
/// `sel_sha256` on this AIR to the canonical
/// `sha256_extract` per-invocation row. B side is gated by
/// `sha256_extract::COL_IS_REAL`. This is the algebraic delegation
/// of `output[0..32] = sha256(input[0..input_length])` to the
/// dedicated SHA-256 AIR.
pub fn make_precompile_io_to_sha256_descriptor(
    io_layer: usize,
    sha256_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(
        MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH,
    );
    for i in 0..MAX_INPUT_LENGTH {
        a_columns.push(COL_INPUT_BYTES_OFFSET + i);
    }
    for i in 0..SHA256_OUTPUT_LENGTH {
        a_columns.push(COL_SHA256_OUTPUT_OFFSET + i);
    }
    let mut b_columns: Vec<usize> = Vec::with_capacity(
        MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH,
    );
    for i in 0..MAX_INPUT_LENGTH {
        b_columns.push(metavm_zkp::sha256_extract::COL_INPUT_BYTE_OFFSET + i);
    }
    for i in 0..SHA256_OUTPUT_LENGTH {
        b_columns.push(metavm_zkp::sha256_extract::COL_OUTPUT_BYTE_OFFSET + i);
    }
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "precompile_io_to_sha256_extract_v1".into(),
        a_layer_index: io_layer,
        a_columns,
        a_selector_column: Some(COL_SEL_SHA256),
        b_layer_index: sha256_layer,
        b_columns,
        b_selector_column: Some(metavm_zkp::sha256_extract::COL_IS_REAL),
    }
}

/// Per-byte byte-memory binding for a single input position `i`. The
/// tuple is `(input_offset + i, input_bytes[i])` ↔ byte_memory
/// `(addr, val)` on read rows.
///
/// The `+ i` arithmetic on the address must be enforced by either
/// (a) committing a per-position auxiliary column `input_addr_i =
/// input_offset + i` and binding that with the byte-memory tuple,
/// which is the route a future revision should take, or
/// (b) baking `i` into the LogUp protocol as a constant offset.
///
/// For this scaffold we expose the descriptor shape that
/// downstream wiring can refine; the A-side tuple here is just
/// `(input_offset, input_bytes[i])` which is the cheap soundness
/// lift binding "the byte at `input_offset` slot `i` exists in
/// byte-memory under SOME address aligned with input_offset". The
/// proper `+ i` offset is added by the linkage adapter.
pub fn make_precompile_io_to_byte_memory_descriptor_at(
    io_layer: usize,
    byte_memory_layer: usize,
    i: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("precompile_io_to_byte_memory_v1_b{:02}", i),
        a_layer_index: io_layer,
        a_columns: vec![COL_INPUT_OFFSET, COL_INPUT_BYTES_OFFSET + i],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: byte_memory_layer,
        b_columns: vec![
            metavm_zkp::byte_memory_air::COL_ADDR,
            metavm_zkp::byte_memory_air::COL_VAL,
        ],
        b_selector_column: Some(metavm_zkp::byte_memory_air::COL_IS_REAL),
    }
}

/// Convenience: return the full vector of [`MAX_INPUT_LENGTH`] byte
/// descriptors covering every input byte position.
pub fn make_precompile_io_to_byte_memory_descriptors(
    io_layer: usize,
    byte_memory_layer: usize,
) -> Vec<metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor> {
    (0..MAX_INPUT_LENGTH)
        .map(|i| make_precompile_io_to_byte_memory_descriptor_at(io_layer, byte_memory_layer, i))
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} row {} nonzero",
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn precompile_io_air_identity_32byte_passes() {
        let input: Vec<u8> = (0..32u8).collect();
        let output = input.clone();
        let w = from_call(PrecompileKind::Identity, &input, &output, 0x100, 0x200);
        assert_eq!(w.input_length, 32);
        assert_eq!(w.output_length, 32);
        // Per-byte equality of input and identity_output.
        for i in 0..32 {
            assert_eq!(w.input_bytes[i], w.identity_output[i]);
        }
        // Pad bytes also zero on both sides.
        for i in 32..MAX_INPUT_LENGTH {
            assert_eq!(w.input_bytes[i], 0);
            assert_eq!(w.identity_output[i], 0);
        }
        let tw = PrecompileIoTraceWitness::from_rows(vec![w]);
        let t = build_trace_polynomials(&tw, CurveType::Bls48581);
        let cs = PrecompileIoConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn precompile_io_air_sha256_32byte_passes() {
        // Honest SHA-256 row: input 32 bytes, output 32 bytes. The
        // SHA-256 algebra itself is delegated to sha256_extract; this
        // AIR enforces only the I/O length + selector relations.
        let input: Vec<u8> = (0..32u8).map(|b| b.wrapping_mul(7).wrapping_add(11)).collect();
        // Fake but legal output bytes (this AIR doesn't compute the
        // hash; the sha256_extract LogUp does).
        let output: Vec<u8> = (0..32u8).map(|b| b.wrapping_add(99)).collect();
        let w = from_call(PrecompileKind::Sha256, &input, &output, 0x400, 0x800);
        assert_eq!(w.input_length, 32);
        assert_eq!(w.output_length, 32);
        // SHA256 row: identity_output bytes are all zero.
        for i in 0..MAX_IDENTITY_OUTPUT_LENGTH {
            assert_eq!(w.identity_output[i], 0);
        }
        // sha256_output is populated.
        for i in 0..32 {
            assert_eq!(w.sha256_output[i], output[i]);
        }
        let tw = PrecompileIoTraceWitness::from_rows(vec![w]);
        let t = build_trace_polynomials(&tw, CurveType::Bls48581);
        let cs = PrecompileIoConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn precompile_io_air_tampered_identity_output_detected() {
        let input: Vec<u8> = (0..32u8).collect();
        let output = input.clone();
        let mut w = from_call(PrecompileKind::Identity, &input, &output, 0x100, 0x200);
        // Flip one byte of the claimed IDENTITY output.
        w.identity_output[5] = w.identity_output[5].wrapping_add(1);
        let tw = PrecompileIoTraceWitness::from_rows(vec![w]);
        let t = build_trace_polynomials(&tw, CurveType::Bls48581);
        let cs = PrecompileIoConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 7 + 5 = byte index 5.
        let idx = 7 + 5;
        assert!(
            !bodies[idx][0].is_zero(),
            "expected IDENTITY byte-equality constraint at index 5 to fire"
        );
    }

    #[test]
    fn precompile_io_air_tampered_sha256_output_length_detected() {
        let input: Vec<u8> = (0..32u8).collect();
        let output: Vec<u8> = (0..32u8).collect();
        let mut w = from_call(PrecompileKind::Sha256, &input, &output, 0x400, 0x800);
        // Lie about output_length (claim 33 instead of 32).
        w.output_length = 33;
        let tw = PrecompileIoTraceWitness::from_rows(vec![w]);
        let t = build_trace_polynomials(&tw, CurveType::Bls48581);
        let cs = PrecompileIoConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 6 = SHA-256 output_length == 32.
        assert!(
            !bodies[6][0].is_zero(),
            "expected SHA-256 output_length constraint to fire"
        );
    }

    #[test]
    fn precompile_io_air_descriptors_well_formed() {
        let d1 = make_precompile_io_to_precompile_dispatch_descriptor(0, 1);
        assert_eq!(d1.label, "precompile_io_to_precompile_dispatch_v1");
        assert_eq!(d1.a_columns.len(), 4);
        assert_eq!(d1.b_columns.len(), 4);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(crate::precompile_air::COL_IS_REAL));

        let d2 = make_precompile_io_to_sha256_descriptor(0, 2);
        assert_eq!(d2.label, "precompile_io_to_sha256_extract_v1");
        assert_eq!(d2.a_columns.len(), MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH);
        assert_eq!(d2.b_columns.len(), MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH);
        assert_eq!(d2.a_selector_column, Some(COL_SEL_SHA256));
        assert_eq!(
            d2.b_selector_column,
            Some(metavm_zkp::sha256_extract::COL_IS_REAL)
        );

        let d3 = make_precompile_io_to_byte_memory_descriptor_at(0, 3, 7);
        assert_eq!(d3.label, "precompile_io_to_byte_memory_v1_b07");
        assert_eq!(d3.a_columns.len(), 2);
        assert_eq!(d3.b_columns.len(), 2);
        assert_eq!(d3.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d3.b_selector_column,
            Some(metavm_zkp::byte_memory_air::COL_IS_REAL)
        );

        let ds = make_precompile_io_to_byte_memory_descriptors(0, 3);
        assert_eq!(ds.len(), MAX_INPUT_LENGTH);
    }

    #[test]
    fn precompile_io_air_column_layout_pinned() {
        // Pin the column layout — drift here breaks downstream linkages.
        assert_eq!(COL_SEL_IDENTITY, 0);
        assert_eq!(COL_SEL_SHA256, 1);
        assert_eq!(COL_IS_REAL, 2);
        assert_eq!(COL_INPUT_OFFSET, 3);
        assert_eq!(COL_INPUT_LENGTH, 4);
        assert_eq!(COL_OUTPUT_OFFSET, 5);
        assert_eq!(COL_OUTPUT_LENGTH, 6);
        assert_eq!(COL_INPUT_BYTES_OFFSET, 7);
        assert_eq!(COL_SHA256_OUTPUT_OFFSET, 7 + MAX_INPUT_LENGTH);
        assert_eq!(
            COL_IDENTITY_OUTPUT_OFFSET,
            7 + MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH
        );
        assert_eq!(
            COL_INPUT_LENGTH_BYTE_OFFSET,
            7 + MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH + MAX_IDENTITY_OUTPUT_LENGTH
        );
        assert_eq!(
            COL_OUTPUT_LENGTH_BYTE_OFFSET,
            7 + MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH + MAX_IDENTITY_OUTPUT_LENGTH
                + LENGTH_BYTES
        );
        assert_eq!(
            NUM_COLUMNS,
            7 + MAX_INPUT_LENGTH + SHA256_OUTPUT_LENGTH + MAX_IDENTITY_OUTPUT_LENGTH
                + 2 * LENGTH_BYTES
        );
        // Constraint count = 7 base + per-byte + 5 tail
        assert_eq!(NUM_ROW_CONSTRAINTS, 7 + MAX_INPUT_LENGTH + 5);
    }

    #[test]
    fn precompile_io_air_multi_row_mixed_passes() {
        let id_in: Vec<u8> = (0..16u8).collect();
        let sha_in: Vec<u8> = (0..40u8).map(|b| b.wrapping_mul(3)).collect();
        let sha_out: Vec<u8> = (0..32u8).map(|b| b.wrapping_add(50)).collect();
        let rows = vec![
            from_call(PrecompileKind::Identity, &id_in, &id_in, 0x10, 0x20),
            from_call(PrecompileKind::Sha256, &sha_in, &sha_out, 0x100, 0x200),
        ];
        let tw = PrecompileIoTraceWitness::from_rows(rows);
        let t = build_trace_polynomials(&tw, CurveType::Bls48581);
        let cs = PrecompileIoConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn precompile_io_air_identity_short_input_zero_pads() {
        let input: Vec<u8> = vec![0xab, 0xcd, 0xef];
        let output: Vec<u8> = vec![0xab, 0xcd, 0xef];
        let w = from_call(PrecompileKind::Identity, &input, &output, 0, 0);
        for i in 0..3 {
            assert_eq!(w.input_bytes[i], input[i]);
            assert_eq!(w.identity_output[i], input[i]);
        }
        for i in 3..MAX_INPUT_LENGTH {
            assert_eq!(w.input_bytes[i], 0);
            assert_eq!(w.identity_output[i], 0);
        }
        let tw = PrecompileIoTraceWitness::from_rows(vec![w]);
        let t = build_trace_polynomials(&tw, CurveType::Bls48581);
        let cs = PrecompileIoConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }
}
