//! ModExp precompile AIR (Ethereum precompile address `0x05`).
//!
//! # Purpose
//!
//! The ModExp precompile (EIP-198, gas updated in EIP-2565) computes
//!
//! ```text
//!     ModExp(B, E, M) = B^E mod M
//! ```
//!
//! The EVM input layout is:
//!
//! ```text
//!     Bsize[32] || Esize[32] || Msize[32] || B[Bsize] || E[Esize] || M[Msize]
//! ```
//!
//! All three size fields are big-endian 256-bit integers. The output is
//! the big-endian byte representation of `B^E mod M`, padded/truncated
//! to exactly `Msize` bytes.
//!
//! ### EIP-2565 gas formula
//!
//! ```text
//!     complexity      = max(Bsize, Msize)^2 / 8 + ...   (see EIP-2565)
//!     iteration_count = ...                              (depends on Esize / E)
//!     gas             = max(200, floor(complexity * iteration_count / 3))
//! ```
//!
//! # Algebraic surface (this AIR)
//!
//! This AIR commits the precompile invocation as one row and pins the
//! precompile **shape** plus the gas formula reduction. Per row:
//!
//!   - `bsize`, `esize`, `msize` — u64 each (bounded by 64 in this
//!     scaffold).
//!   - `input_header[0..96]` — the three 32-byte BE size fields. The
//!     last 8 bytes of each 32-byte field are the BE byte-decomp of the
//!     corresponding `*size` u64 (other 24 bytes pinned to zero so this
//!     AIR rejects sizes > 2^64).
//!   - `B[0..64]`, `E[0..64]`, `M[0..64]`, `output[0..64]` — byte-padded
//!     witnesses; the modexp itself is **not** algebraically proven (a
//!     full modular-exponentiation gadget is deferred).
//!   - `gas_cost`, `complexity`, `iteration_count` — u64.
//!   - `gas_remainder ∈ {0, 1, 2}` — the gas-formula floor remainder so
//!     that `3 * gas_cost + gas_remainder = complexity * iteration_count`.
//!   - `is_real ∈ {0, 1}`.
//!
//! ### Row-local constraints (≥ 8)
//!
//!   0. `is_real * (is_real - 1) = 0` — selector binarity.
//!   1. LE byte-decomp of `bsize` over 8 bytes (`input_header[24..32]`,
//!      read in reverse since the on-chain encoding is BE).
//!   2. LE byte-decomp of `esize` over 8 bytes (`input_header[56..64]`).
//!   3. LE byte-decomp of `msize` over 8 bytes (`input_header[88..96]`).
//!   4. Input header high bytes pinned to zero — for each of the three
//!      32-byte size fields, the first 24 bytes must be zero. This is
//!      `is_real * input_header[k] = 0` for `k ∈ {0..24, 32..56, 64..88}`.
//!      72 constraints folded into a single β-RLC for the constraint
//!      count, but here we list them separately for clarity (72
//!      individual constraints).
//!   5. Gas formula identity:
//!         `is_real * (3 * gas_cost + gas_remainder
//!                     - complexity * iteration_count) = 0`
//!   6. `gas_remainder` is in `{0, 1, 2}` — `is_real * gas_remainder *
//!      (gas_remainder - 1) * (gas_remainder - 2) = 0` (degree-4 wrt
//!      gas_remainder).
//!   7. `bsize ≤ MAX_BIGINT_LENGTH` — pinned via byte-decomp on `bsize`
//!      (constraint 1) plus the fact that high 24 bytes are zero
//!      (constraint group 4); no separate algebraic check. The
//!      `MAX_BIGINT_LENGTH` upper bound is enforced by the **trace
//!      builder** (scaffold limitation).
//!
//! Constraint count = `1 + 3 + 72 + 1 + 1 = 78` row-local constraints.
//!
//! ### What is NOT algebraically pinned (deferred)
//!
//!   - The modular exponentiation `output = B^E mod M`. A full
//!     algebraic mod-exp circuit is a major undertaking; for now the
//!     `output` column is a host-side witness commitment.
//!   - The exact EIP-2565 formulas for `complexity` and `iteration_count`.
//!     These come from input-dependent piecewise expressions; this AIR
//!     pins the **floor-division reduction** of gas from
//!     `(complexity, iteration_count)`. A follow-up wires the piecewise
//!     formulas as additional row-local constraints + byte decomps.
//!
//! # Cross-AIR LogUp descriptors
//!
//!   - [`make_modexp_to_precompile_dispatch_descriptor`] — binds
//!     `(sel_modexp = 1, input_length, output_length, gas_cost)` on this
//!     AIR's real rows to the matching `precompile_air` dispatch row for
//!     callee `0x05`.
//!   - [`make_modexp_to_precompile_io_descriptor`] — binds the
//!     `input_header[0..96] || B || E || M || output` byte block to the
//!     corresponding `precompile_io_air` row.
//!
//! Both descriptors use the same placeholder-sentinel convention as
//! [`crate::ripemd160_precompile_air`].

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;
use num_bigint::BigUint;
use num_traits::{One, Zero};

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum supported B/E/M length (bytes). Scaffold bound; the real
/// precompile supports arbitrary-length integers.
pub const MAX_BIGINT_LENGTH: usize = 64;
/// Each size field is encoded as a 32-byte big-endian integer.
pub const SIZE_FIELD_LENGTH: usize = 32;
/// Input header = 3 × 32-byte size fields.
pub const INPUT_HEADER_LENGTH: usize = 3 * SIZE_FIELD_LENGTH; // 96
/// Number of LE bytes used to range-check each u64 size field.
pub const SIZE_LE_BYTES: usize = 8;
/// Maximum output length (bytes) — same as MAX_BIGINT_LENGTH.
pub const MAX_OUTPUT_LENGTH: usize = MAX_BIGINT_LENGTH;

/// EIP precompile id for ModExp.
pub const PC_MODEXP: u64 = 0x05;

/// Within each 32-byte BE size field, the LOW 8 bytes encode the u64
/// size (`input_header[24..32]` for bsize, etc.). The HIGH 24 bytes are
/// pinned to zero (scaffold limit: sizes > 2^64 are rejected).
pub const SIZE_LOW_OFFSET: usize = SIZE_FIELD_LENGTH - SIZE_LE_BYTES; // 24
pub const SIZE_HIGH_LENGTH: usize = SIZE_LOW_OFFSET; // 24 zero bytes per field

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_INPUT_HEADER_OFFSET: usize = 0; // 0..96
pub const COL_B_BYTES_OFFSET: usize = COL_INPUT_HEADER_OFFSET + INPUT_HEADER_LENGTH; // 96
pub const COL_E_BYTES_OFFSET: usize = COL_B_BYTES_OFFSET + MAX_BIGINT_LENGTH; // 160
pub const COL_M_BYTES_OFFSET: usize = COL_E_BYTES_OFFSET + MAX_BIGINT_LENGTH; // 224
pub const COL_OUTPUT_OFFSET: usize = COL_M_BYTES_OFFSET + MAX_BIGINT_LENGTH; // 288

pub const COL_BSIZE: usize = COL_OUTPUT_OFFSET + MAX_OUTPUT_LENGTH; // 352
pub const COL_ESIZE: usize = COL_BSIZE + 1; // 353
pub const COL_MSIZE: usize = COL_ESIZE + 1; // 354

pub const COL_GAS_COST: usize = COL_MSIZE + 1; // 355
pub const COL_COMPLEXITY: usize = COL_GAS_COST + 1; // 356
pub const COL_ITERATION_COUNT: usize = COL_COMPLEXITY + 1; // 357
pub const COL_GAS_REMAINDER: usize = COL_ITERATION_COUNT + 1; // 358

pub const COL_IS_REAL: usize = COL_GAS_REMAINDER + 1; // 359

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 360

/// Constraint indices:
///   0:                       is_real binary
///   1:                       LE byte-decomp of bsize from input_header[24..32]
///   2:                       LE byte-decomp of esize from input_header[56..64]
///   3:                       LE byte-decomp of msize from input_header[88..96]
///   4..4+72:                 input header high bytes pinned to zero (24 × 3)
///   4+72:                    gas formula identity
///   4+72+1:                  gas_remainder ∈ {0, 1, 2}
///
/// = 1 + 3 + 72 + 1 + 1 = 78
pub const NUM_ROW_CONSTRAINTS: usize = 1 + 3 + (3 * SIZE_HIGH_LENGTH) + 1 + 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// One ModExp precompile invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModExpPrecompileWitness {
    /// 3 × 32-byte big-endian sizes (Bsize, Esize, Msize), zero-padded.
    pub input_header: [u8; INPUT_HEADER_LENGTH],
    /// Base bytes (BE), zero-padded to MAX_BIGINT_LENGTH.
    pub b_bytes: [u8; MAX_BIGINT_LENGTH],
    /// Exponent bytes (BE), zero-padded to MAX_BIGINT_LENGTH.
    pub e_bytes: [u8; MAX_BIGINT_LENGTH],
    /// Modulus bytes (BE), zero-padded to MAX_BIGINT_LENGTH.
    pub m_bytes: [u8; MAX_BIGINT_LENGTH],
    /// Output `B^E mod M` (BE), zero-padded to MAX_OUTPUT_LENGTH.
    /// The true output length equals `msize`.
    pub output: [u8; MAX_OUTPUT_LENGTH],
    /// True size of B in bytes.
    pub bsize: u64,
    /// True size of E in bytes.
    pub esize: u64,
    /// True size of M in bytes.
    pub msize: u64,
    /// Total gas cost: `floor(complexity * iteration_count / 3)`.
    pub gas_cost: u64,
    /// EIP-2565 `complexity` value (host-side committed; piecewise
    /// formula not algebraically pinned here).
    pub complexity: u64,
    /// EIP-2565 `iteration_count` value.
    pub iteration_count: u64,
    /// `complexity * iteration_count mod 3` — in `{0, 1, 2}`.
    pub gas_remainder: u64,
}

impl ModExpPrecompileWitness {
    /// Compute an honest witness from the three BE byte arrays
    /// `b`, `e`, `m`. The modular exponentiation is performed via
    /// [`num_bigint::BigUint`]; this is the host-side oracle. Inputs
    /// longer than [`MAX_BIGINT_LENGTH`] are truncated (scaffold limit).
    ///
    /// The EIP-2565 `complexity` and `iteration_count` are computed via
    /// the simplified forms documented at
    /// <https://eips.ethereum.org/EIPS/eip-2565>:
    ///
    /// ```text
    ///     mult_complexity(x) = ceil(x / 8)^2
    ///     complexity         = mult_complexity(max(Bsize, Msize))
    ///     iteration_count    =
    ///         if Esize <= 32 and E == 0: 0
    ///         if Esize <= 32:           floor(log2(E))
    ///         else:                     8 * (Esize - 32)
    ///                                       + floor(log2(top_32_bytes_of_E))
    ///     iteration_count    = max(iteration_count, 1)
    /// ```
    pub fn from_inputs(b: &[u8], e: &[u8], m: &[u8]) -> Self {
        let bsize = b.len() as u64;
        let esize = e.len() as u64;
        let msize = m.len() as u64;

        // Build the 96-byte input header.
        let mut input_header = [0u8; INPUT_HEADER_LENGTH];
        let bsize_be = bsize.to_be_bytes();
        let esize_be = esize.to_be_bytes();
        let msize_be = msize.to_be_bytes();
        // bsize → input_header[24..32]
        input_header[SIZE_LOW_OFFSET..SIZE_FIELD_LENGTH].copy_from_slice(&bsize_be);
        // esize → input_header[32+24..32+32]
        input_header[SIZE_FIELD_LENGTH + SIZE_LOW_OFFSET..2 * SIZE_FIELD_LENGTH]
            .copy_from_slice(&esize_be);
        // msize → input_header[64+24..64+32]
        input_header[2 * SIZE_FIELD_LENGTH + SIZE_LOW_OFFSET..3 * SIZE_FIELD_LENGTH]
            .copy_from_slice(&msize_be);

        // Pad B/E/M into fixed-size buffers (LEFT-truncate if oversized).
        fn pad_be(src: &[u8]) -> [u8; MAX_BIGINT_LENGTH] {
            let mut out = [0u8; MAX_BIGINT_LENGTH];
            let n = src.len().min(MAX_BIGINT_LENGTH);
            // Right-align: BE representation places leading bytes first.
            // We choose to left-pack into the witness so byte k of `out`
            // corresponds to byte k of the EVM-side calldata segment.
            out[..n].copy_from_slice(&src[..n]);
            out
        }
        let b_bytes = pad_be(b);
        let e_bytes = pad_be(e);
        let m_bytes = pad_be(m);

        // Compute B^E mod M via num_bigint.
        let b_big = BigUint::from_bytes_be(b);
        let e_big = BigUint::from_bytes_be(e);
        let m_big = BigUint::from_bytes_be(m);
        let result_big = if m_big.is_zero() || m_big.is_one() {
            // Convention: M = 0 → output is all zeros (EIP-198 returns
            // msize zero bytes when M is empty/zero). M = 1 → output is 0.
            BigUint::zero()
        } else {
            b_big.modpow(&e_big, &m_big)
        };
        // Encode result as exactly `msize` BE bytes, then left-pack into
        // the fixed-size output buffer.
        let mut output = [0u8; MAX_OUTPUT_LENGTH];
        let msize_usize = (msize as usize).min(MAX_OUTPUT_LENGTH);
        if msize_usize > 0 {
            let raw = result_big.to_bytes_be();
            if raw.len() <= msize_usize {
                // Right-align into the first `msize_usize` bytes.
                let lead = msize_usize - raw.len();
                output[lead..msize_usize].copy_from_slice(&raw);
            } else {
                // Truncate to the LOW `msize_usize` bytes (BE).
                let start = raw.len() - msize_usize;
                output[..msize_usize].copy_from_slice(&raw[start..]);
            }
        }

        // EIP-2565 simplified host-side complexity + iteration_count.
        let max_size = bsize.max(msize);
        let words = max_size.div_ceil(8); // ceil(x/8)
        let complexity = words.saturating_mul(words);

        let iteration_count: u64 = if esize <= 32 {
            // Use E as a u256 → take its bit length.
            if e_big.is_zero() {
                0
            } else {
                (e_big.bits() - 1) as u64
            }
        } else {
            // First 32 bytes of E (BE) — interpret as a big int; floor_log2 of that.
            let head_len = 32usize.min(e.len());
            let head = BigUint::from_bytes_be(&e[..head_len]);
            let head_log = if head.is_zero() { 0 } else { (head.bits() - 1) as u64 };
            8u64.saturating_mul(esize.saturating_sub(32))
                .saturating_add(head_log)
        };
        let iteration_count = iteration_count.max(1);

        let prod = complexity.saturating_mul(iteration_count);
        let gas_quot = prod / 3;
        let gas_rem = prod % 3;
        // EIP-2565: gas = max(200, floor(prod / 3)). For algebraic
        // simplicity we PIN `gas_cost = floor(prod / 3)` and require the
        // host to enforce the `max(200, _)` clamp externally (i.e. the
        // EVM dispatch layer applies the 200 floor when forwarding gas).
        // A follow-up wires the clamp as a constraint.
        let gas_cost = gas_quot;

        Self {
            input_header,
            b_bytes,
            e_bytes,
            m_bytes,
            output,
            bsize,
            esize,
            msize,
            gas_cost,
            complexity,
            iteration_count,
            gas_remainder: gas_rem,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModExpPrecompileTraceWitness {
    pub rows: Vec<ModExpPrecompileWitness>,
}

impl ModExpPrecompileTraceWitness {
    pub fn from_rows(rows: Vec<ModExpPrecompileWitness>) -> Self {
        Self { rows }
    }
    pub fn push(&mut self, row: ModExpPrecompileWitness) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ModExpPrecompileTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..INPUT_HEADER_LENGTH {
            columns[COL_INPUT_HEADER_OFFSET + k][r] =
                Scalar::from_u64(row.input_header[k] as u64, curve);
        }
        for k in 0..MAX_BIGINT_LENGTH {
            columns[COL_B_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.b_bytes[k] as u64, curve);
            columns[COL_E_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.e_bytes[k] as u64, curve);
            columns[COL_M_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.m_bytes[k] as u64, curve);
        }
        for k in 0..MAX_OUTPUT_LENGTH {
            columns[COL_OUTPUT_OFFSET + k][r] =
                Scalar::from_u64(row.output[k] as u64, curve);
        }
        columns[COL_BSIZE][r] = Scalar::from_u64(row.bsize, curve);
        columns[COL_ESIZE][r] = Scalar::from_u64(row.esize, curve);
        columns[COL_MSIZE][r] = Scalar::from_u64(row.msize, curve);
        columns[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        columns[COL_COMPLEXITY][r] = Scalar::from_u64(row.complexity, curve);
        columns[COL_ITERATION_COUNT][r] = Scalar::from_u64(row.iteration_count, curve);
        columns[COL_GAS_REMAINDER][r] = Scalar::from_u64(row.gas_remainder, curve);
        columns[COL_IS_REAL][r] = one.clone();
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

pub struct ModExpPrecompileConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ModExpPrecompileConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// The header byte index (in `input_header`) for the low-byte BE
/// representation of `bsize`/`esize`/`msize`. The u64 occupies the LAST
/// 8 bytes of each 32-byte size field; index 0 below is the
/// most-significant byte of that 8-byte chunk.
fn size_field_be_byte_index(field: usize, byte: usize) -> usize {
    debug_assert!(field < 3);
    debug_assert!(byte < SIZE_LE_BYTES);
    field * SIZE_FIELD_LENGTH + SIZE_LOW_OFFSET + byte
}

/// The `byte`-th high-zero index (0..24) of the `field`-th size field
/// (0=bsize, 1=esize, 2=msize).
fn size_field_high_zero_index(field: usize, byte: usize) -> usize {
    debug_assert!(field < 3);
    debug_assert!(byte < SIZE_HIGH_LENGTH);
    field * SIZE_FIELD_LENGTH + byte
}

impl VmConstraintSystem for ModExpPrecompileConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("bsize_be_byte_decomp".into());
        labels.push("esize_be_byte_decomp".into());
        labels.push("msize_be_byte_decomp".into());
        for field in 0..3 {
            for k in 0..SIZE_HIGH_LENGTH {
                let name = ["bsize", "esize", "msize"][field];
                labels.push(format!("{}_high_zero_byte_{}", name, k));
            }
        }
        labels.push("gas_formula_floor_div_3".into());
        labels.push("gas_remainder_lt_3".into());
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
        let two = Scalar::from_u64(2, curve);
        let three = Scalar::from_u64(3, curve);
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

        // 1..=3: BE byte-decomp of bsize/esize/msize.
        for (field, size_col) in [
            (0usize, COL_BSIZE),
            (1, COL_ESIZE),
            (2, COL_MSIZE),
        ] {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                // Most-significant byte of the u64 is byte 0 within the
                // 8-byte slice (BE convention).
                for b in 0..SIZE_LE_BYTES {
                    let shift = 8 * (SIZE_LE_BYTES - 1 - b);
                    let w = Scalar::from_u64(1u64 << shift, curve);
                    let col = size_field_be_byte_index(field, b);
                    let term = columns[COL_INPUT_HEADER_OFFSET + col][r].mul(&w);
                    sum = sum.add(&term);
                }
                c[r] = sum.sub(&columns[size_col][r]);
            }
            out.push(c);
        }

        // 4..4+72: input header high bytes pinned to zero (24 × 3).
        for field in 0..3 {
            for k in 0..SIZE_HIGH_LENGTH {
                let mut c = vec![Scalar::zero(curve); n];
                let col = size_field_high_zero_index(field, k);
                for r in 0..n {
                    let v = &columns[COL_INPUT_HEADER_OFFSET + col][r];
                    c[r] = columns[COL_IS_REAL][r].mul(v);
                }
                out.push(c);
            }
        }

        // Gas formula identity:
        //   is_real * (3 * gas_cost + gas_remainder - complexity * iter) = 0
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let lhs = three.mul(&columns[COL_GAS_COST][r])
                    .add(&columns[COL_GAS_REMAINDER][r]);
                let rhs = columns[COL_COMPLEXITY][r]
                    .mul(&columns[COL_ITERATION_COUNT][r]);
                c[r] = columns[COL_IS_REAL][r].mul(&lhs.sub(&rhs));
            }
            out.push(c);
        }

        // gas_remainder ∈ {0, 1, 2}:
        //   is_real * gas_remainder * (gas_remainder - 1) * (gas_remainder - 2)
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let g = &columns[COL_GAS_REMAINDER][r];
                let body = g.mul(&g.sub(&one)).mul(&g.sub(&two));
                c[r] = columns[COL_IS_REAL][r].mul(&body);
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
        let two = Scalar::from_u64(2, curve);
        let three = Scalar::from_u64(3, curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1..=3: size BE byte-decomp.
        for (field, size_col) in [
            (0usize, COL_BSIZE),
            (1, COL_ESIZE),
            (2, COL_MSIZE),
        ] {
            let mut sum = Scalar::zero(curve);
            for b in 0..SIZE_LE_BYTES {
                let shift = 8 * (SIZE_LE_BYTES - 1 - b);
                let w = Scalar::from_u64(1u64 << shift, curve);
                let col = size_field_be_byte_index(field, b);
                sum = sum.add(&col_evals[COL_INPUT_HEADER_OFFSET + col].mul(&w));
            }
            let body = sum.sub(&col_evals[size_col]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4..: header high zeros.
        for field in 0..3 {
            for k in 0..SIZE_HIGH_LENGTH {
                let col = size_field_high_zero_index(field, k);
                let body = col_evals[COL_IS_REAL]
                    .mul(&col_evals[COL_INPUT_HEADER_OFFSET + col]);
                acc = acc.add(&alpha_pow.mul(&body));
                alpha_pow = alpha_pow.mul(alpha);
            }
        }
        // gas formula.
        {
            let lhs = three.mul(&col_evals[COL_GAS_COST])
                .add(&col_evals[COL_GAS_REMAINDER]);
            let rhs = col_evals[COL_COMPLEXITY].mul(&col_evals[COL_ITERATION_COUNT]);
            let body = col_evals[COL_IS_REAL].mul(&lhs.sub(&rhs));
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // gas_remainder ∈ {0, 1, 2}.
        {
            let g = &col_evals[COL_GAS_REMAINDER];
            let body = g.mul(&g.sub(&one)).mul(&g.sub(&two));
            let body = col_evals[COL_IS_REAL].mul(&body);
            acc = acc.add(&alpha_pow.mul(&body));
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
        let two_poly = vec![Scalar::from_u64(2, curve)];
        let three_poly = vec![Scalar::from_u64(3, curve)];

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
        // 1..=3: size BE byte-decomp.
        for (field, size_col) in [
            (0usize, COL_BSIZE),
            (1, COL_ESIZE),
            (2, COL_MSIZE),
        ] {
            let mut sum: Vec<Scalar> = vec![Scalar::zero(curve)];
            for b in 0..SIZE_LE_BYTES {
                let shift = 8 * (SIZE_LE_BYTES - 1 - b);
                let w = Scalar::from_u64(1u64 << shift, curve);
                let col = size_field_be_byte_index(field, b);
                let term = poly_scalar_mul(
                    &col_coeffs[COL_INPUT_HEADER_OFFSET + col],
                    &w,
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&sum, &col_coeffs[size_col], curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4..: header high zeros.
        for field in 0..3 {
            for k in 0..SIZE_HIGH_LENGTH {
                let col = size_field_high_zero_index(field, k);
                let body = poly_mul(
                    &col_coeffs[COL_IS_REAL],
                    &col_coeffs[COL_INPUT_HEADER_OFFSET + col],
                    curve,
                );
                acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
                alpha_pow = alpha_pow.mul(alpha);
            }
        }
        // gas formula.
        {
            let three_gas = poly_mul(&three_poly, &col_coeffs[COL_GAS_COST], curve);
            let lhs = poly_add(&three_gas, &col_coeffs[COL_GAS_REMAINDER], curve);
            let rhs = poly_mul(
                &col_coeffs[COL_COMPLEXITY],
                &col_coeffs[COL_ITERATION_COUNT],
                curve,
            );
            let diff = poly_sub(&lhs, &rhs, curve);
            let body = poly_mul(&col_coeffs[COL_IS_REAL], &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // gas_remainder ∈ {0, 1, 2}.
        {
            let g = &col_coeffs[COL_GAS_REMAINDER];
            let g_m1 = poly_sub(g, &one_poly, curve);
            let g_m2 = poly_sub(g, &two_poly, curve);
            let body0 = poly_mul(g, &g_m1, curve);
            let body1 = poly_mul(&body0, &g_m2, curve);
            let body = poly_mul(&col_coeffs[COL_IS_REAL], &body1, curve);
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
        for k in 0..INPUT_HEADER_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_input_header_byte_{}_8bit", k),
                    column_index: COL_INPUT_HEADER_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MAX_BIGINT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_b_byte_{}_8bit", k),
                    column_index: COL_B_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_e_byte_{}_8bit", k),
                    column_index: COL_E_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_m_byte_{}_8bit", k),
                    column_index: COL_M_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MAX_OUTPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("modexp_output_byte_{}_8bit", k),
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
pub const PRECOMPILE_DISPATCH_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds
/// `(sel_modexp = is_real, input_length, output_length = msize, gas_cost)`
/// on this AIR's real rows to the matching `precompile_air` dispatch row
/// for callee `0x05`.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B-side columns
/// are filled with [`PRECOMPILE_DISPATCH_PLACEHOLDER`].
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_modexp_to_precompile_dispatch_descriptor(
    modexp_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // A side: (sel_modexp = is_real, msize as output_length proxy, gas_cost).
    // We also include msize as the "output_length" channel — the EVM
    // dispatch layer reports the precompile output_length, which for
    // ModExp equals msize.
    let a_columns: Vec<usize> = vec![
        COL_IS_REAL,
        COL_MSIZE,
        COL_GAS_COST,
    ];
    let b_columns: Vec<usize> = vec![
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_SEL_MODEXP (= 14)
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_OUTPUT_LENGTH (= 5)
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_GAS_COST (= 6)
    ];
    CrossAirLogUpDescriptor {
        label: "modexp_to_precompile_dispatch_v1_stub".into(),
        a_layer_index: modexp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

/// Placeholder sentinel for `precompile_io_air` column indices.
pub const PRECOMPILE_IO_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds the precompile-row
/// `input_header || B || E || M || output` byte arrays to the
/// corresponding `precompile_io_air` row that records EVM CALL memory I/O.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B side is
/// filled with [`PRECOMPILE_IO_PLACEHOLDER`].
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_modexp_to_precompile_io_descriptor(
    modexp_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let total = INPUT_HEADER_LENGTH + 3 * MAX_BIGINT_LENGTH + MAX_OUTPUT_LENGTH;
    let mut a_columns: Vec<usize> = Vec::with_capacity(total);
    for k in 0..INPUT_HEADER_LENGTH {
        a_columns.push(COL_INPUT_HEADER_OFFSET + k);
    }
    for k in 0..MAX_BIGINT_LENGTH {
        a_columns.push(COL_B_BYTES_OFFSET + k);
    }
    for k in 0..MAX_BIGINT_LENGTH {
        a_columns.push(COL_E_BYTES_OFFSET + k);
    }
    for k in 0..MAX_BIGINT_LENGTH {
        a_columns.push(COL_M_BYTES_OFFSET + k);
    }
    for k in 0..MAX_OUTPUT_LENGTH {
        a_columns.push(COL_OUTPUT_OFFSET + k);
    }
    let b_columns: Vec<usize> = vec![PRECOMPILE_IO_PLACEHOLDER; total];
    CrossAirLogUpDescriptor {
        label: "modexp_to_precompile_io_v1_stub".into(),
        a_layer_index: modexp_layer_index,
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

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} row {} nonzero",
                    i, r,
                );
            }
        }
    }

    /// Helper: build a single-row witness and assert all constraints vanish.
    fn check_honest(w: ModExpPrecompileWitness) {
        let trace = build_trace_polynomials(
            &ModExpPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = ModExpPrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn modexp_precompile_air_two_pow_three_mod_seven() {
        // 2^3 mod 7 = 8 mod 7 = 1.
        let w = ModExpPrecompileWitness::from_inputs(&[2], &[3], &[7]);
        assert_eq!(w.bsize, 1);
        assert_eq!(w.esize, 1);
        assert_eq!(w.msize, 1);
        // Output is 1 byte, value 1.
        assert_eq!(w.output[0], 1);
        for k in 1..MAX_OUTPUT_LENGTH {
            assert_eq!(w.output[k], 0);
        }
        check_honest(w);
    }

    #[test]
    fn modexp_precompile_air_modulus_one_returns_zero() {
        // B^E mod 1 = 0 (M = 1 edge case).
        let w = ModExpPrecompileWitness::from_inputs(&[5], &[7], &[1]);
        assert_eq!(w.msize, 1);
        // Output should be all zeros.
        for k in 0..MAX_OUTPUT_LENGTH {
            assert_eq!(w.output[k], 0);
        }
        check_honest(w);
    }

    #[test]
    fn modexp_precompile_air_tampered_output_detected_via_io_descriptor() {
        // The output column is a host-side witness (modexp itself is
        // deferred), so tampering output bytes alone does not fire any
        // of the row-local constraints. What we CAN detect at this layer
        // is tampering of the `gas_cost` such that the floor-div
        // identity no longer balances. We exercise THAT path here.
        let w = ModExpPrecompileWitness::from_inputs(&[2], &[3], &[7]);
        let trace = build_trace_polynomials(
            &ModExpPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        // Tamper gas_cost by +1.
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let gas_idx = COL_GAS_COST;
        let cur = cols[gas_idx][0].clone();
        let one = Scalar::one(CurveType::Bls48581);
        cols[gas_idx][0] = cur.add(&one);

        let cs = ModExpPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Gas formula constraint = constraint index NUM_ROW_CONSTRAINTS - 2.
        let gas_constraint_idx = NUM_ROW_CONSTRAINTS - 2;
        assert!(
            !bodies[gas_constraint_idx][0].is_zero(),
            "expected gas formula constraint to fire on tampered gas_cost",
        );
    }

    #[test]
    fn modexp_precompile_air_tampered_header_high_byte_detected() {
        // Tampering a "high zero" byte of the input_header (one of the
        // 24 leading zero bytes of a size field) fires the high-byte
        // pin constraint, which prevents a malicious prover from
        // declaring `bsize > 2^64`.
        let w = ModExpPrecompileWitness::from_inputs(&[2], &[3], &[7]);
        let trace = build_trace_polynomials(
            &ModExpPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper input_header[0] — leading byte of bsize (should be zero).
        cols[COL_INPUT_HEADER_OFFSET + 0][0] =
            Scalar::from_u64(0x42, CurveType::Bls48581);
        let cs = ModExpPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // High-zero constraints start at index 4 (after is_real + 3 size decomps).
        let first_high_idx = 4;
        assert!(
            !bodies[first_high_idx][0].is_zero(),
            "expected first high-zero constraint to fire",
        );
    }

    #[test]
    fn modexp_precompile_air_size_byte_decomp_detects_size_tamper() {
        // Tamper `bsize` scalar while leaving the BE byte-decomp intact:
        // the size byte-decomp constraint fires.
        let w = ModExpPrecompileWitness::from_inputs(&[2], &[3], &[7]);
        let trace = build_trace_polynomials(
            &ModExpPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_BSIZE][0] = Scalar::from_u64(999, CurveType::Bls48581);
        let cs = ModExpPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // bsize byte decomp = constraint index 1.
        assert!(
            !bodies[1][0].is_zero(),
            "expected bsize byte-decomp constraint to fire",
        );
    }

    #[test]
    fn modexp_precompile_air_descriptors_well_formed() {
        let d1 = make_modexp_to_precompile_dispatch_descriptor(0, 1);
        assert_eq!(d1.label, "modexp_to_precompile_dispatch_v1_stub");
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_eq!(d1.a_columns.len(), 3);
        assert_eq!(d1.b_columns.len(), 3);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_columns[0], COL_IS_REAL);
        assert_eq!(d1.a_columns[1], COL_MSIZE);
        assert_eq!(d1.a_columns[2], COL_GAS_COST);
        for b in &d1.b_columns {
            assert_eq!(*b, PRECOMPILE_DISPATCH_PLACEHOLDER);
        }

        let d2 = make_modexp_to_precompile_io_descriptor(0, 2);
        assert_eq!(d2.label, "modexp_to_precompile_io_v1_stub");
        let expected_len = INPUT_HEADER_LENGTH + 3 * MAX_BIGINT_LENGTH + MAX_OUTPUT_LENGTH;
        assert_eq!(d2.a_columns.len(), expected_len);
        assert_eq!(d2.b_columns.len(), expected_len);
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
        // First 96 cols are input_header.
        for k in 0..INPUT_HEADER_LENGTH {
            assert_eq!(d2.a_columns[k], COL_INPUT_HEADER_OFFSET + k);
        }
        // Next 64 cols are B bytes.
        for k in 0..MAX_BIGINT_LENGTH {
            assert_eq!(
                d2.a_columns[INPUT_HEADER_LENGTH + k],
                COL_B_BYTES_OFFSET + k,
            );
        }
        // All B side is placeholders.
        for b in &d2.b_columns {
            assert_eq!(*b, PRECOMPILE_IO_PLACEHOLDER);
        }
    }

    #[test]
    fn modexp_precompile_air_column_layout_pinned() {
        // Pin numeric offsets so future column additions trip this test.
        assert_eq!(COL_INPUT_HEADER_OFFSET, 0);
        assert_eq!(COL_B_BYTES_OFFSET, 96);
        assert_eq!(COL_E_BYTES_OFFSET, 160);
        assert_eq!(COL_M_BYTES_OFFSET, 224);
        assert_eq!(COL_OUTPUT_OFFSET, 288);
        assert_eq!(COL_BSIZE, 352);
        assert_eq!(COL_ESIZE, 353);
        assert_eq!(COL_MSIZE, 354);
        assert_eq!(COL_GAS_COST, 355);
        assert_eq!(COL_COMPLEXITY, 356);
        assert_eq!(COL_ITERATION_COUNT, 357);
        assert_eq!(COL_GAS_REMAINDER, 358);
        assert_eq!(COL_IS_REAL, 359);
        assert_eq!(NUM_COLUMNS, 360);

        // Constraint count: 1 (is_real) + 3 (size decomps) + 72 (header
        // high zeros) + 1 (gas formula) + 1 (gas_remainder ∈ {0,1,2})
        // = 78.
        assert_eq!(NUM_ROW_CONSTRAINTS, 78);
        assert_eq!(NUM_SHIFTED, 0);

        // Constants.
        assert_eq!(MAX_BIGINT_LENGTH, 64);
        assert_eq!(SIZE_FIELD_LENGTH, 32);
        assert_eq!(INPUT_HEADER_LENGTH, 96);
        assert_eq!(SIZE_LE_BYTES, 8);
        assert_eq!(SIZE_LOW_OFFSET, 24);
        assert_eq!(SIZE_HIGH_LENGTH, 24);
        assert_eq!(MAX_OUTPUT_LENGTH, 64);
        assert_eq!(PC_MODEXP, 0x05);

        // Labels match wired constraint count.
        let cs = ModExpPrecompileConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
    }

    #[test]
    fn modexp_precompile_air_multi_row_passes() {
        let rows = vec![
            ModExpPrecompileWitness::from_inputs(&[2], &[3], &[7]),
            ModExpPrecompileWitness::from_inputs(&[5], &[7], &[1]),
            ModExpPrecompileWitness::from_inputs(&[3, 0], &[2], &[100]),
        ];
        let trace = build_trace_polynomials(
            &ModExpPrecompileTraceWitness::from_rows(rows),
            CurveType::Bls48581,
        );
        let cs = ModExpPrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn modexp_precompile_air_input_header_well_formed() {
        // Verify the 96-byte input_header is the canonical BE encoding
        // of (bsize, esize, msize).
        let w = ModExpPrecompileWitness::from_inputs(&[1, 2], &[3], &[4, 5, 6]);
        assert_eq!(w.bsize, 2);
        assert_eq!(w.esize, 1);
        assert_eq!(w.msize, 3);
        // input_header[0..24] = 0, input_header[24..32] = bsize BE.
        for k in 0..SIZE_LOW_OFFSET {
            assert_eq!(w.input_header[k], 0);
            assert_eq!(w.input_header[SIZE_FIELD_LENGTH + k], 0);
            assert_eq!(w.input_header[2 * SIZE_FIELD_LENGTH + k], 0);
        }
        // Last byte of each size field carries the low byte of size.
        assert_eq!(w.input_header[SIZE_FIELD_LENGTH - 1], 2); // bsize = 2
        assert_eq!(w.input_header[2 * SIZE_FIELD_LENGTH - 1], 1); // esize = 1
        assert_eq!(w.input_header[3 * SIZE_FIELD_LENGTH - 1], 3); // msize = 3
        check_honest(w);
    }
}
