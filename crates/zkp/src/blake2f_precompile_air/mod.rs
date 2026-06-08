//! BLAKE2F precompile AIR (Ethereum precompile address `0x09`, EIP-152).
//!
//! # Purpose
//!
//! The BLAKE2F precompile at Ethereum address `0x09` (defined in EIP-152)
//! exposes the BLAKE2b compression function `F` so that EVM contracts can
//! perform BLAKE2 hashing in a gas-bounded, deterministic way. The
//! precompile input is a fixed 213-byte layout:
//!
//! ```text
//!   offset  size  field    description
//!   0       4     rounds   number of rounds, big-endian u32
//!   4       64    h        state vector, 8 little-endian u64 limbs
//!   68      128   m        message block, 16 little-endian u64 limbs
//!   196     16    t        offset counter, 2 little-endian u64 limbs
//!   212     1     f        final-block flag (must be 0x00 or 0x01)
//!   213          (total)
//! ```
//!
//! The output is the 64-byte updated state `h_new` (8 LE u64 limbs).
//!
//! Gas cost: `rounds × 1` per round (no constant overhead).
//!
//! # Algebraic surface (this AIR)
//!
//! This AIR commits the precompile invocation as one row and proves the
//! **shape** of the input and output bindings. Per row it commits:
//!
//!   - the full 213-byte `input` array;
//!   - the structured projection of `input` into `rounds` (u32),
//!     `h_in[0..64]`, `m[0..128]`, `t[0..16]`, `f_byte`;
//!   - the 64-byte `h_out` result of running the BLAKE2b F compression
//!     for the specified number of rounds (witness-only — the F itself
//!     is **not** algebraically constrained, see "Algebraic F deferred");
//!   - the 64-byte `output` array (the precompile return data, equal to
//!     `h_out`);
//!   - the precompile `gas_cost = rounds × 1`;
//!   - the `is_real` selector binary.
//!
//! Row-local constraints (213 input slice equalities + 64 output
//! equalities + 4-byte LE/BE decompositions + gas formula = 285 total):
//!
//!   0.                     `is_real ∈ {0, 1}`.
//!   1..=4.                 `input[i] = rounds_byte_be[i]` for i ∈ 0..4
//!                          (the input rounds field is BIG-endian).
//!   5..=68.                `input[4 + i] = h_in[i]` for i ∈ 0..64.
//!   69..=196.              `input[68 + i] = m[i]` for i ∈ 0..128.
//!   197..=212.             `input[196 + i] = t[i]` for i ∈ 0..16.
//!   213.                   `input[212] = f_byte`.
//!   214..=277.             `output[i] = h_out[i]` for i ∈ 0..64.
//!   278.                   `f_byte * (f_byte - 1) = 0` — f flag binary.
//!   279.                   `is_real * (gas_cost - rounds) = 0` —
//!                          gas formula (cost equals round count).
//!   280.                   `rounds = Σ_b 256^(3-b) * rounds_byte_be[b]`
//!                          (big-endian decomposition into 4 bytes).
//!
//! Total: `281` row-local constraints. No shifted constraints. 8-bit
//! range checks on every byte column.
//!
//! # Algebraic BLAKE2b F is deferred
//!
//! The BLAKE2b F compression function consists of `rounds` mixing rounds
//! over an 8-u64 state with 12 invocations of the G mixing function per
//! round, each involving 64-bit XORs, additions, and rotations. Building
//! a constraint system for this in the field-arithmetic AIR model is a
//! substantial effort and is **deferred**. Until that lands, this AIR's
//! `h_out` column is a witness commitment only: the host-side trace
//! builder computes it via the reference BLAKE2b F implementation (see
//! [`blake2b_f_compress`] below), and the soundness of
//! `h_out = F(rounds, h_in, m, t, f)` is taken on faith by this AIR.
//! The shape constraints above DO algebraically pin the input parsing,
//! output marshalling, and gas formula.
//!
//! # Cross-AIR LogUp descriptors
//!
//!   - [`make_blake2f_to_precompile_dispatch_descriptor`] —
//!     binds `(sel_blake2f = 1, gas_cost)` on this AIR's real rows to
//!     the matching `precompile_air` dispatch row for callee `0x09`.
//!   - [`make_blake2f_to_precompile_io_descriptor`] —
//!     binds `(input[0..213], output[0..64])` to the corresponding
//!     `precompile_io_air` row (memory glue).
//!
//! Both descriptors use the same placeholder-sentinel convention as
//! [`crate::ripemd160_precompile_air`]: the zkp crate does not depend on
//! the EVM crate, so the EVM-side B-column indices are stubbed with
//! `usize::MAX` and substituted by the joint-prover orchestrator when
//! the layers are wired.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// EIP-152 precompile id for BLAKE2F.
pub const PC_BLAKE2F: u64 = 0x09;

/// Total BLAKE2F precompile input length (rounds[4] + h[64] + m[128]
/// + t[16] + f[1]).
pub const INPUT_LENGTH: usize = 213;
/// BLAKE2b state vector length in bytes (8 × u64 little-endian).
pub const H_LENGTH: usize = 64;
/// BLAKE2b message block length in bytes (16 × u64 little-endian).
pub const M_LENGTH: usize = 128;
/// BLAKE2b offset counter length in bytes (2 × u64 little-endian).
pub const T_LENGTH: usize = 16;
/// Number of bytes in the BE `rounds` field.
pub const ROUNDS_BYTES: usize = 4;
/// Output length = state vector length.
pub const OUTPUT_LENGTH: usize = H_LENGTH;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_INPUT_OFFSET: usize = 0; // 0..213
pub const COL_ROUNDS: usize = COL_INPUT_OFFSET + INPUT_LENGTH; // 213
pub const COL_ROUNDS_BYTE_BE_OFFSET: usize = COL_ROUNDS + 1; // 214..218
pub const COL_H_IN_OFFSET: usize =
    COL_ROUNDS_BYTE_BE_OFFSET + ROUNDS_BYTES; // 218..282
pub const COL_M_OFFSET: usize = COL_H_IN_OFFSET + H_LENGTH; // 282..410
pub const COL_T_OFFSET: usize = COL_M_OFFSET + M_LENGTH; // 410..426
pub const COL_F_BYTE: usize = COL_T_OFFSET + T_LENGTH; // 426
pub const COL_H_OUT_OFFSET: usize = COL_F_BYTE + 1; // 427..491
pub const COL_OUTPUT_OFFSET: usize = COL_H_OUT_OFFSET + H_LENGTH; // 491..555
pub const COL_IS_REAL: usize = COL_OUTPUT_OFFSET + OUTPUT_LENGTH; // 555
pub const COL_GAS_COST: usize = COL_IS_REAL + 1; // 556
pub const NUM_COLUMNS: usize = COL_GAS_COST + 1; // 557

/// Row-local constraint count:
///   1 (is_real binary)
/// + ROUNDS_BYTES (4: input rounds slice = rounds_byte_be)
/// + H_LENGTH (64: input h slice = h_in)
/// + M_LENGTH (128: input m slice = m)
/// + T_LENGTH (16: input t slice = t)
/// + 1 (input[212] = f_byte)
/// + OUTPUT_LENGTH (64: output = h_out)
/// + 1 (f_byte binary)
/// + 1 (is_real * (gas_cost - rounds))
/// + 1 (rounds = Σ 256^(3-b) * rounds_byte_be[b])
/// = 281
pub const NUM_ROW_CONSTRAINTS: usize = 1
    + ROUNDS_BYTES
    + H_LENGTH
    + M_LENGTH
    + T_LENGTH
    + 1
    + OUTPUT_LENGTH
    + 1
    + 1
    + 1;
pub const NUM_SHIFTED: usize = 0;

// Convenience constraint-index landmarks (for tests).
pub const CIDX_IS_REAL_BINARY: usize = 0;
pub const CIDX_INPUT_ROUNDS_BEGIN: usize = 1;
pub const CIDX_INPUT_H_BEGIN: usize = CIDX_INPUT_ROUNDS_BEGIN + ROUNDS_BYTES; // 5
pub const CIDX_INPUT_M_BEGIN: usize = CIDX_INPUT_H_BEGIN + H_LENGTH; // 69
pub const CIDX_INPUT_T_BEGIN: usize = CIDX_INPUT_M_BEGIN + M_LENGTH; // 197
pub const CIDX_INPUT_F_BYTE: usize = CIDX_INPUT_T_BEGIN + T_LENGTH; // 213
pub const CIDX_OUTPUT_BEGIN: usize = CIDX_INPUT_F_BYTE + 1; // 214
pub const CIDX_F_BYTE_BINARY: usize = CIDX_OUTPUT_BEGIN + OUTPUT_LENGTH; // 278
pub const CIDX_GAS_FORMULA: usize = CIDX_F_BYTE_BINARY + 1; // 279
pub const CIDX_ROUNDS_BE_DECOMP: usize = CIDX_GAS_FORMULA + 1; // 280

// ─── Reference BLAKE2b F compression ──────────────────────────────────

/// BLAKE2b IV (RFC 7693 §2.6).
const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// BLAKE2b SIGMA permutations (RFC 7693 §2.7).
const BLAKE2B_SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[inline]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

/// EIP-152 BLAKE2b F compression: take in-place state `h[0..8]`, message
/// block `m[0..16]`, offset counter `t[0..2]`, final-block flag `f`, and
/// `rounds`, and produce the updated state. Reference implementation per
/// RFC 7693 §3.2 with the EIP-152 round-count generalisation.
pub fn blake2b_f_compress(
    rounds: u32,
    h: &mut [u64; 8],
    m: &[u64; 16],
    t: &[u64; 2],
    f: bool,
) {
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&BLAKE2B_IV);
    v[12] ^= t[0];
    v[13] ^= t[1];
    if f {
        v[14] = !v[14];
    }

    for i in 0..rounds as usize {
        let s = &BLAKE2B_SIGMA[i % 10];
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }

    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

// ─── Witness ──────────────────────────────────────────────────────────

/// One BLAKE2F precompile invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blake2fPrecompileWitness {
    /// Full 213-byte EIP-152 input layout.
    pub input: [u8; INPUT_LENGTH],
    /// Parsed big-endian rounds count.
    pub rounds: u32,
    /// Parsed input h slice (64 bytes; 8 LE u64 limbs).
    pub h_in: [u8; H_LENGTH],
    /// Parsed input message block (128 bytes; 16 LE u64 limbs).
    pub m: [u8; M_LENGTH],
    /// Parsed input offset counter (16 bytes; 2 LE u64 limbs).
    pub t: [u8; T_LENGTH],
    /// Parsed input final-block flag byte (must be 0 or 1).
    pub f_byte: u8,
    /// Output state after running F (64 bytes; 8 LE u64 limbs).
    pub h_out: [u8; H_LENGTH],
    /// Precompile output payload (== `h_out`).
    pub output: [u8; OUTPUT_LENGTH],
    /// Gas cost — equals `rounds` per EIP-152.
    pub gas_cost: u64,
}

impl Blake2fPrecompileWitness {
    /// Build an honest witness from the full 213-byte EIP-152 input. The
    /// host-side reference BLAKE2b F (see [`blake2b_f_compress`]) is run
    /// to derive `h_out` and `output`.
    pub fn from_input(input: [u8; INPUT_LENGTH]) -> Self {
        // Parse rounds (big-endian u32).
        let rounds = u32::from_be_bytes(input[0..4].try_into().expect("4 bytes"));

        // Parse h_in, m, t, f_byte (raw byte slices).
        let mut h_in = [0u8; H_LENGTH];
        h_in.copy_from_slice(&input[4..4 + H_LENGTH]);
        let mut m_bytes = [0u8; M_LENGTH];
        m_bytes.copy_from_slice(&input[4 + H_LENGTH..4 + H_LENGTH + M_LENGTH]);
        let mut t_bytes = [0u8; T_LENGTH];
        t_bytes.copy_from_slice(
            &input[4 + H_LENGTH + M_LENGTH..4 + H_LENGTH + M_LENGTH + T_LENGTH],
        );
        let f_byte = input[INPUT_LENGTH - 1];

        // Decode to u64 limbs for the reference compression.
        let mut h_limbs = [0u64; 8];
        for i in 0..8 {
            h_limbs[i] = u64::from_le_bytes(
                h_in[i * 8..(i + 1) * 8].try_into().expect("8 bytes"),
            );
        }
        let mut m_limbs = [0u64; 16];
        for i in 0..16 {
            m_limbs[i] = u64::from_le_bytes(
                m_bytes[i * 8..(i + 1) * 8].try_into().expect("8 bytes"),
            );
        }
        let mut t_limbs = [0u64; 2];
        for i in 0..2 {
            t_limbs[i] = u64::from_le_bytes(
                t_bytes[i * 8..(i + 1) * 8].try_into().expect("8 bytes"),
            );
        }
        let f = f_byte != 0;

        // Run reference F.
        blake2b_f_compress(rounds, &mut h_limbs, &m_limbs, &t_limbs, f);

        // Re-encode h_out as 64 LE bytes.
        let mut h_out = [0u8; H_LENGTH];
        for i in 0..8 {
            h_out[i * 8..(i + 1) * 8].copy_from_slice(&h_limbs[i].to_le_bytes());
        }
        let mut output = [0u8; OUTPUT_LENGTH];
        output.copy_from_slice(&h_out);

        Self {
            input,
            rounds,
            h_in,
            m: m_bytes,
            t: t_bytes,
            f_byte,
            h_out,
            output,
            gas_cost: rounds as u64,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Blake2fPrecompileTraceWitness {
    pub rows: Vec<Blake2fPrecompileWitness>,
}

impl Blake2fPrecompileTraceWitness {
    pub fn from_rows(rows: Vec<Blake2fPrecompileWitness>) -> Self {
        Self { rows }
    }
    pub fn push(&mut self, row: Blake2fPrecompileWitness) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Blake2fPrecompileTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..INPUT_LENGTH {
            columns[COL_INPUT_OFFSET + k][r] =
                Scalar::from_u64(row.input[k] as u64, curve);
        }
        columns[COL_ROUNDS][r] = Scalar::from_u64(row.rounds as u64, curve);
        let rb = row.rounds.to_be_bytes();
        for b in 0..ROUNDS_BYTES {
            columns[COL_ROUNDS_BYTE_BE_OFFSET + b][r] =
                Scalar::from_u64(rb[b] as u64, curve);
        }
        for k in 0..H_LENGTH {
            columns[COL_H_IN_OFFSET + k][r] =
                Scalar::from_u64(row.h_in[k] as u64, curve);
        }
        for k in 0..M_LENGTH {
            columns[COL_M_OFFSET + k][r] =
                Scalar::from_u64(row.m[k] as u64, curve);
        }
        for k in 0..T_LENGTH {
            columns[COL_T_OFFSET + k][r] =
                Scalar::from_u64(row.t[k] as u64, curve);
        }
        columns[COL_F_BYTE][r] = Scalar::from_u64(row.f_byte as u64, curve);
        for k in 0..H_LENGTH {
            columns[COL_H_OUT_OFFSET + k][r] =
                Scalar::from_u64(row.h_out[k] as u64, curve);
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct Blake2fPrecompileConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Blake2fPrecompileConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Blake2fPrecompileConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        for b in 0..ROUNDS_BYTES {
            labels.push(format!("input_rounds_byte_{}_eq", b));
        }
        for k in 0..H_LENGTH {
            labels.push(format!("input_h_byte_{}_eq", k));
        }
        for k in 0..M_LENGTH {
            labels.push(format!("input_m_byte_{}_eq", k));
        }
        for k in 0..T_LENGTH {
            labels.push(format!("input_t_byte_{}_eq", k));
        }
        labels.push("input_f_byte_eq".into());
        for k in 0..OUTPUT_LENGTH {
            labels.push(format!("output_byte_{}_eq_h_out", k));
        }
        labels.push("f_byte_binary".into());
        labels.push("gas_cost_eq_rounds".into());
        labels.push("rounds_be_byte_decomp".into());
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
        // input rounds slice equalities.
        for b in 0..ROUNDS_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_INPUT_OFFSET + b][r]
                    .sub(&columns[COL_ROUNDS_BYTE_BE_OFFSET + b][r]);
            }
            out.push(c);
        }
        // input h slice equalities.
        for k in 0..H_LENGTH {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_INPUT_OFFSET + ROUNDS_BYTES + k][r]
                    .sub(&columns[COL_H_IN_OFFSET + k][r]);
            }
            out.push(c);
        }
        // input m slice equalities.
        for k in 0..M_LENGTH {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_INPUT_OFFSET + ROUNDS_BYTES + H_LENGTH + k][r]
                    .sub(&columns[COL_M_OFFSET + k][r]);
            }
            out.push(c);
        }
        // input t slice equalities.
        for k in 0..T_LENGTH {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns
                    [COL_INPUT_OFFSET + ROUNDS_BYTES + H_LENGTH + M_LENGTH + k][r]
                    .sub(&columns[COL_T_OFFSET + k][r]);
            }
            out.push(c);
        }
        // input f byte equality.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_INPUT_OFFSET + INPUT_LENGTH - 1][r]
                    .sub(&columns[COL_F_BYTE][r]);
            }
            out.push(c);
        }
        // output[i] = h_out[i].
        for k in 0..OUTPUT_LENGTH {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_OUTPUT_OFFSET + k][r]
                    .sub(&columns[COL_H_OUT_OFFSET + k][r]);
            }
            out.push(c);
        }
        // f_byte binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_F_BYTE][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // gas_cost = rounds (gated by is_real).
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let diff =
                    columns[COL_GAS_COST][r].sub(&columns[COL_ROUNDS][r]);
                c[r] = columns[COL_IS_REAL][r].mul(&diff);
            }
            out.push(c);
        }
        // rounds = Σ_b 256^(3-b) * rounds_byte_be[b].
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for b in 0..ROUNDS_BYTES {
                    let shift = (ROUNDS_BYTES - 1 - b) * 8;
                    let w = Scalar::from_u64(1u64 << shift, curve);
                    let term =
                        columns[COL_ROUNDS_BYTE_BE_OFFSET + b][r].mul(&w);
                    sum = sum.add(&term);
                }
                c[r] = sum.sub(&columns[COL_ROUNDS][r]);
            }
            out.push(c);
        }

        debug_assert_eq!(out.len(), NUM_ROW_CONSTRAINTS);
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
        let push = |body: Scalar, ap: &mut Scalar, acc: &mut Scalar| {
            *acc = acc.add(&ap.mul(&body));
            *ap = ap.mul(alpha);
        };

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            push(v.mul(&v.sub(&one)), &mut alpha_pow, &mut acc);
        }
        for b in 0..ROUNDS_BYTES {
            push(
                col_evals[COL_INPUT_OFFSET + b]
                    .sub(&col_evals[COL_ROUNDS_BYTE_BE_OFFSET + b]),
                &mut alpha_pow,
                &mut acc,
            );
        }
        for k in 0..H_LENGTH {
            push(
                col_evals[COL_INPUT_OFFSET + ROUNDS_BYTES + k]
                    .sub(&col_evals[COL_H_IN_OFFSET + k]),
                &mut alpha_pow,
                &mut acc,
            );
        }
        for k in 0..M_LENGTH {
            push(
                col_evals[COL_INPUT_OFFSET + ROUNDS_BYTES + H_LENGTH + k]
                    .sub(&col_evals[COL_M_OFFSET + k]),
                &mut alpha_pow,
                &mut acc,
            );
        }
        for k in 0..T_LENGTH {
            push(
                col_evals[COL_INPUT_OFFSET + ROUNDS_BYTES + H_LENGTH + M_LENGTH + k]
                    .sub(&col_evals[COL_T_OFFSET + k]),
                &mut alpha_pow,
                &mut acc,
            );
        }
        push(
            col_evals[COL_INPUT_OFFSET + INPUT_LENGTH - 1]
                .sub(&col_evals[COL_F_BYTE]),
            &mut alpha_pow,
            &mut acc,
        );
        for k in 0..OUTPUT_LENGTH {
            push(
                col_evals[COL_OUTPUT_OFFSET + k]
                    .sub(&col_evals[COL_H_OUT_OFFSET + k]),
                &mut alpha_pow,
                &mut acc,
            );
        }
        // f_byte binary.
        {
            let v = &col_evals[COL_F_BYTE];
            push(v.mul(&v.sub(&one)), &mut alpha_pow, &mut acc);
        }
        // gas formula.
        {
            let diff = col_evals[COL_GAS_COST].sub(&col_evals[COL_ROUNDS]);
            push(col_evals[COL_IS_REAL].mul(&diff), &mut alpha_pow, &mut acc);
        }
        // rounds BE byte decomp.
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..ROUNDS_BYTES {
                let shift = (ROUNDS_BYTES - 1 - b) * 8;
                let w = Scalar::from_u64(1u64 << shift, curve);
                sum = sum.add(&col_evals[COL_ROUNDS_BYTE_BE_OFFSET + b].mul(&w));
            }
            push(sum.sub(&col_evals[COL_ROUNDS]), &mut alpha_pow, &mut acc);
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

        let push =
            |body: Vec<Scalar>, ap: &mut Scalar, acc: &mut Vec<Scalar>| {
                *acc = poly_add(acc, &poly_scalar_mul(&body, ap), curve);
                *ap = ap.mul(alpha);
            };

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            push(poly_mul(v, &v_m1, curve), &mut alpha_pow, &mut acc);
        }
        for b in 0..ROUNDS_BYTES {
            let body = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + b],
                &col_coeffs[COL_ROUNDS_BYTE_BE_OFFSET + b],
                curve,
            );
            push(body, &mut alpha_pow, &mut acc);
        }
        for k in 0..H_LENGTH {
            let body = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + ROUNDS_BYTES + k],
                &col_coeffs[COL_H_IN_OFFSET + k],
                curve,
            );
            push(body, &mut alpha_pow, &mut acc);
        }
        for k in 0..M_LENGTH {
            let body = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + ROUNDS_BYTES + H_LENGTH + k],
                &col_coeffs[COL_M_OFFSET + k],
                curve,
            );
            push(body, &mut alpha_pow, &mut acc);
        }
        for k in 0..T_LENGTH {
            let body = poly_sub(
                &col_coeffs
                    [COL_INPUT_OFFSET + ROUNDS_BYTES + H_LENGTH + M_LENGTH + k],
                &col_coeffs[COL_T_OFFSET + k],
                curve,
            );
            push(body, &mut alpha_pow, &mut acc);
        }
        // input f byte equality.
        {
            let body = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + INPUT_LENGTH - 1],
                &col_coeffs[COL_F_BYTE],
                curve,
            );
            push(body, &mut alpha_pow, &mut acc);
        }
        for k in 0..OUTPUT_LENGTH {
            let body = poly_sub(
                &col_coeffs[COL_OUTPUT_OFFSET + k],
                &col_coeffs[COL_H_OUT_OFFSET + k],
                curve,
            );
            push(body, &mut alpha_pow, &mut acc);
        }
        // f_byte binary.
        {
            let v = &col_coeffs[COL_F_BYTE];
            let v_m1 = poly_sub(v, &one_poly, curve);
            push(poly_mul(v, &v_m1, curve), &mut alpha_pow, &mut acc);
        }
        // gas formula.
        {
            let diff =
                poly_sub(&col_coeffs[COL_GAS_COST], &col_coeffs[COL_ROUNDS], curve);
            push(
                poly_mul(&col_coeffs[COL_IS_REAL], &diff, curve),
                &mut alpha_pow,
                &mut acc,
            );
        }
        // rounds BE byte decomp.
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..ROUNDS_BYTES {
                let shift = (ROUNDS_BYTES - 1 - b) * 8;
                let w = Scalar::from_u64(1u64 << shift, curve);
                let term = poly_scalar_mul(
                    &col_coeffs[COL_ROUNDS_BYTE_BE_OFFSET + b],
                    &w,
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&sum, &col_coeffs[COL_ROUNDS], curve);
            push(body, &mut alpha_pow, &mut acc);
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
        // Byte range checks on every committed byte column.
        for k in 0..INPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2f_input_byte_{}_8bit", k),
                    column_index: COL_INPUT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..ROUNDS_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2f_rounds_byte_be_{}_8bit", b),
                    column_index: COL_ROUNDS_BYTE_BE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..H_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2f_h_in_byte_{}_8bit", k),
                    column_index: COL_H_IN_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..M_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2f_m_byte_{}_8bit", k),
                    column_index: COL_M_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..T_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2f_t_byte_{}_8bit", k),
                    column_index: COL_T_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "blake2f_f_byte_8bit".into(),
                column_index: COL_F_BYTE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        for k in 0..H_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2f_h_out_byte_{}_8bit", k),
                    column_index: COL_H_OUT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..OUTPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("blake2f_output_byte_{}_8bit", k),
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

/// Placeholder sentinel for `precompile_air` / `precompile_io_air`
/// column indices. Same convention as
/// [`crate::ripemd160_precompile_air::PRECOMPILE_DISPATCH_PLACEHOLDER`].
pub const PRECOMPILE_DISPATCH_PLACEHOLDER: usize = usize::MAX;
pub const PRECOMPILE_IO_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds
/// `(sel_blake2f = is_real, gas_cost)` on this AIR's real rows to the
/// matching `precompile_air` dispatch row for callee `0x09`. The
/// dispatch row enforces existence of the BLAKE2F invocation under
/// `precompile_air::COL_SEL_BLAKE2F`.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B-side
/// columns are filled with [`PRECOMPILE_DISPATCH_PLACEHOLDER`]. The
/// joint-prover orchestrator substitutes the real indices when wiring
/// the layers (`precompile_air::COL_SEL_BLAKE2F` = 18,
/// `precompile_air::COL_GAS_COST` = 6).
pub fn make_blake2f_to_precompile_dispatch_descriptor(
    blake2f_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = vec![COL_IS_REAL, COL_GAS_COST];
    let b_columns: Vec<usize> = vec![
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_SEL_BLAKE2F
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_GAS_COST
    ];
    CrossAirLogUpDescriptor {
        label: "blake2f_to_precompile_dispatch_v1_stub".into(),
        a_layer_index: blake2f_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

/// Cross-AIR LogUp descriptor (stub): binds
/// `(input[0..213], output[0..64])` on this AIR's real rows to the
/// corresponding `precompile_io_air` row that records the EVM-side
/// CALL memory I/O for the BLAKE2F invocation.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B side is
/// filled with [`PRECOMPILE_IO_PLACEHOLDER`].
pub fn make_blake2f_to_precompile_io_descriptor(
    blake2f_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(INPUT_LENGTH + OUTPUT_LENGTH);
    for k in 0..INPUT_LENGTH {
        a_columns.push(COL_INPUT_OFFSET + k);
    }
    for k in 0..OUTPUT_LENGTH {
        a_columns.push(COL_OUTPUT_OFFSET + k);
    }
    let b_columns: Vec<usize> =
        vec![PRECOMPILE_IO_PLACEHOLDER; INPUT_LENGTH + OUTPUT_LENGTH];
    CrossAirLogUpDescriptor {
        label: "blake2f_to_precompile_io_v1_stub".into(),
        a_layer_index: blake2f_layer_index,
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

    /// EIP-152 reference test vector (test case "8" in the spec). With:
    ///   rounds = 12
    ///   h = IV-based personalization (BLAKE2b parameter block XOR)
    ///   m = "abc" || zero pad
    ///   t = (3, 0)
    ///   f = true
    /// the output is the BLAKE2b("abc") digest:
    ///   0xba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1
    ///   0x7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923
    fn eip152_vector_8_input() -> [u8; INPUT_LENGTH] {
        let hex_str = concat!(
            "0000000c",                                                         // rounds = 12 (BE)
            "48c9bdf267e6096a3ba7ca8485ae67bb",                                 // h[0]
            "2bf894fe72f36e3cf1361d5f3af54fa5",                                 // h[1]
            "d182e6ad7f520e511f6c3e2b8c68059b",                                 // h[2]
            "6bbd41fbabd9831f79217e1319cde05b",                                 // h[3]
            "6162630000000000000000000000000000000000000000000000000000000000", // m[0..32] ("abc"||pad)
            "0000000000000000000000000000000000000000000000000000000000000000", // m[32..64]
            "0000000000000000000000000000000000000000000000000000000000000000", // m[64..96]
            "0000000000000000000000000000000000000000000000000000000000000000", // m[96..128]
            "03000000000000000000000000000000",                                 // t (3, 0)
            "01",                                                               // f = true
        );
        let bytes = hex_decode(hex_str);
        assert_eq!(bytes.len(), INPUT_LENGTH);
        let mut out = [0u8; INPUT_LENGTH];
        out.copy_from_slice(&bytes);
        out
    }

    fn eip152_vector_8_expected_output() -> [u8; OUTPUT_LENGTH] {
        let bytes = hex_decode(concat!(
            "ba80a53f981c4d0d6a2797b69f12f6e9",
            "4c212f14685ac4b74b12bb6fdbffa2d1",
            "7d87c5392aab792dc252d5de4533cc95",
            "18d38aa8dbf1925ab92386edd4009923",
        ));
        let mut out = [0u8; OUTPUT_LENGTH];
        out.copy_from_slice(&bytes);
        out
    }

    fn hex_decode(s: &str) -> Vec<u8> {
        let s = s.trim();
        assert!(s.len() % 2 == 0, "hex string must have even length");
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
            .collect()
    }

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} row {} nonzero: {:?}",
                    i,
                    r,
                    v,
                );
            }
        }
    }

    #[test]
    fn blake2f_precompile_air_eip152_vector_8_passes() {
        let input = eip152_vector_8_input();
        let expected = eip152_vector_8_expected_output();
        let w = Blake2fPrecompileWitness::from_input(input);
        assert_eq!(w.rounds, 12);
        assert_eq!(w.f_byte, 1);
        assert_eq!(w.gas_cost, 12);
        assert_eq!(w.output, expected, "EIP-152 test vector 8 output mismatch");
        assert_eq!(w.h_out, expected);
        // Input slice matches the structured fields.
        assert_eq!(&w.input[0..4], &12u32.to_be_bytes());
        assert_eq!(&w.input[4..68], &w.h_in[..]);
        assert_eq!(&w.input[68..196], &w.m[..]);
        assert_eq!(&w.input[196..212], &w.t[..]);
        assert_eq!(w.input[212], 1);

        let trace = build_trace_polynomials(
            &Blake2fPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Blake2fPrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn blake2f_precompile_air_tampered_output_detected() {
        let input = eip152_vector_8_input();
        let w = Blake2fPrecompileWitness::from_input(input);
        let trace = build_trace_polynomials(
            &Blake2fPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper output[7] without touching h_out[7] — should fire the
        // output==h_out equality at constraint index CIDX_OUTPUT_BEGIN + 7.
        cols[COL_OUTPUT_OFFSET + 7][0] = Scalar::from_u64(0xab, CurveType::Bls48581);

        let cs = Blake2fPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let tampered_idx = CIDX_OUTPUT_BEGIN + 7;
        assert!(
            !bodies[tampered_idx][0].is_zero(),
            "expected output==h_out constraint at index {} to fire",
            tampered_idx,
        );
    }

    #[test]
    fn blake2f_precompile_air_tampered_rounds_and_gas_detected() {
        let input = eip152_vector_8_input();
        let w = Blake2fPrecompileWitness::from_input(input);
        let trace = build_trace_polynomials(
            &Blake2fPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        // Case A: tamper gas_cost only — gas formula constraint fires.
        {
            let mut cols: Vec<Vec<Scalar>> =
                trace.columns.iter().map(|p| p.evaluations.clone()).collect();
            cols[COL_GAS_COST][0] = Scalar::from_u64(99, CurveType::Bls48581);
            let cs = Blake2fPrecompileConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert!(
                !bodies[CIDX_GAS_FORMULA][0].is_zero(),
                "expected gas formula constraint to fire",
            );
        }
        // Case B: tamper rounds scalar without touching rounds_byte_be —
        // the BE decomp constraint fires AND the gas-formula constraint
        // fires (since gas_cost still = 12).
        {
            let mut cols: Vec<Vec<Scalar>> =
                trace.columns.iter().map(|p| p.evaluations.clone()).collect();
            cols[COL_ROUNDS][0] = Scalar::from_u64(13, CurveType::Bls48581);
            let cs = Blake2fPrecompileConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert!(
                !bodies[CIDX_ROUNDS_BE_DECOMP][0].is_zero(),
                "expected rounds BE decomp constraint to fire",
            );
            assert!(
                !bodies[CIDX_GAS_FORMULA][0].is_zero(),
                "expected gas formula constraint to fire on rounds tamper",
            );
        }
    }

    #[test]
    fn blake2f_precompile_air_tampered_input_slice_detected() {
        // Tamper an input byte (h slice) without updating h_in — the
        // corresponding slice equality should fire.
        let input = eip152_vector_8_input();
        let w = Blake2fPrecompileWitness::from_input(input);
        let trace = build_trace_polynomials(
            &Blake2fPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Touch input[10] (which is in the h slice at h offset 6).
        cols[COL_INPUT_OFFSET + 10][0] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = Blake2fPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let idx = CIDX_INPUT_H_BEGIN + 6;
        assert!(
            !bodies[idx][0].is_zero(),
            "expected input h slice constraint {} to fire",
            idx,
        );
    }

    #[test]
    fn blake2f_precompile_air_descriptors_well_formed() {
        let d1 = make_blake2f_to_precompile_dispatch_descriptor(0, 1);
        assert_eq!(d1.label, "blake2f_to_precompile_dispatch_v1_stub");
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_eq!(d1.a_columns.len(), 2);
        assert_eq!(d1.b_columns.len(), 2);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_columns[0], COL_IS_REAL);
        assert_eq!(d1.a_columns[1], COL_GAS_COST);
        for b in &d1.b_columns {
            assert_eq!(*b, PRECOMPILE_DISPATCH_PLACEHOLDER);
        }

        let d2 = make_blake2f_to_precompile_io_descriptor(0, 2);
        assert_eq!(d2.label, "blake2f_to_precompile_io_v1_stub");
        assert_eq!(d2.a_columns.len(), INPUT_LENGTH + OUTPUT_LENGTH);
        assert_eq!(d2.b_columns.len(), INPUT_LENGTH + OUTPUT_LENGTH);
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
        // First INPUT_LENGTH A cols are input bytes.
        for k in 0..INPUT_LENGTH {
            assert_eq!(d2.a_columns[k], COL_INPUT_OFFSET + k);
        }
        // Next OUTPUT_LENGTH A cols are output bytes.
        for k in 0..OUTPUT_LENGTH {
            assert_eq!(d2.a_columns[INPUT_LENGTH + k], COL_OUTPUT_OFFSET + k);
        }
        for b in &d2.b_columns {
            assert_eq!(*b, PRECOMPILE_IO_PLACEHOLDER);
        }
    }

    #[test]
    fn blake2f_precompile_air_column_layout_pinned() {
        // Pin numeric offsets so accidental column additions trip this test.
        assert_eq!(COL_INPUT_OFFSET, 0);
        assert_eq!(COL_ROUNDS, 213);
        assert_eq!(COL_ROUNDS_BYTE_BE_OFFSET, 214);
        assert_eq!(COL_H_IN_OFFSET, 218);
        assert_eq!(COL_M_OFFSET, 282);
        assert_eq!(COL_T_OFFSET, 410);
        assert_eq!(COL_F_BYTE, 426);
        assert_eq!(COL_H_OUT_OFFSET, 427);
        assert_eq!(COL_OUTPUT_OFFSET, 491);
        assert_eq!(COL_IS_REAL, 555);
        assert_eq!(COL_GAS_COST, 556);
        assert_eq!(NUM_COLUMNS, 557);

        // Constraint count: 1 + 4 + 64 + 128 + 16 + 1 + 64 + 1 + 1 + 1 = 281.
        assert_eq!(NUM_ROW_CONSTRAINTS, 281);
        assert_eq!(NUM_SHIFTED, 0);

        // Constraint landmarks.
        assert_eq!(CIDX_IS_REAL_BINARY, 0);
        assert_eq!(CIDX_INPUT_ROUNDS_BEGIN, 1);
        assert_eq!(CIDX_INPUT_H_BEGIN, 5);
        assert_eq!(CIDX_INPUT_M_BEGIN, 69);
        assert_eq!(CIDX_INPUT_T_BEGIN, 197);
        assert_eq!(CIDX_INPUT_F_BYTE, 213);
        assert_eq!(CIDX_OUTPUT_BEGIN, 214);
        assert_eq!(CIDX_F_BYTE_BINARY, 278);
        assert_eq!(CIDX_GAS_FORMULA, 279);
        assert_eq!(CIDX_ROUNDS_BE_DECOMP, 280);

        // Constants.
        assert_eq!(INPUT_LENGTH, 213);
        assert_eq!(H_LENGTH, 64);
        assert_eq!(M_LENGTH, 128);
        assert_eq!(T_LENGTH, 16);
        assert_eq!(ROUNDS_BYTES, 4);
        assert_eq!(OUTPUT_LENGTH, 64);
        assert_eq!(PC_BLAKE2F, 0x09);

        // Labels match wired constraint count.
        let cs = Blake2fPrecompileConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
    }

    #[test]
    fn blake2f_precompile_air_zero_rounds_is_identity_on_state_xor() {
        // rounds=0: F reduces to h ^= v ^ (v[8..]) where v[0..8]=h,
        // v[8..]=IV (with v[12..14] XOR'd by t and v[14] possibly NOT'd).
        // We just verify constraints pass on the honest witness — not
        // that the output is structurally meaningful.
        let mut input = [0u8; INPUT_LENGTH];
        // rounds = 0 (BE).
        input[0..4].copy_from_slice(&0u32.to_be_bytes());
        // h = zero, m = zero, t = zero, f = 0 → all-zero input apart from rounds.
        let w = Blake2fPrecompileWitness::from_input(input);
        assert_eq!(w.rounds, 0);
        assert_eq!(w.gas_cost, 0);
        // For h_in=0, t=0, f=0, rounds=0:
        //   v[0..8] = 0; v[8..] = IV; final h[i] = 0 ^ 0 ^ IV[i] = IV[i].
        let mut expected = [0u8; OUTPUT_LENGTH];
        for i in 0..8 {
            expected[i * 8..(i + 1) * 8]
                .copy_from_slice(&BLAKE2B_IV[i].to_le_bytes());
        }
        assert_eq!(w.h_out, expected);
        assert_eq!(w.output, expected);

        let trace = build_trace_polynomials(
            &Blake2fPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Blake2fPrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn blake2f_precompile_air_f_byte_must_be_binary() {
        // Honest witness has f_byte = 1; constraint passes.
        let input = eip152_vector_8_input();
        let w = Blake2fPrecompileWitness::from_input(input);
        let trace = build_trace_polynomials(
            &Blake2fPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper f_byte to 2 — binary constraint fires.
        cols[COL_F_BYTE][0] = Scalar::from_u64(2, CurveType::Bls48581);
        // Also tamper input[212] so the input slice equality doesn't
        // mask the f-binary check (otherwise the input slice constraint
        // would also fire).
        cols[COL_INPUT_OFFSET + INPUT_LENGTH - 1][0] =
            Scalar::from_u64(2, CurveType::Bls48581);
        let cs = Blake2fPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[CIDX_F_BYTE_BINARY][0].is_zero(),
            "expected f_byte binary constraint to fire",
        );
    }
}
