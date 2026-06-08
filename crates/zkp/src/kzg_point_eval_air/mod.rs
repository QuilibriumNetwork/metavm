//! EIP-4844 **KZG point evaluation precompile** AIR (address `0x0A`).
//!
//! # Purpose
//!
//! The KZG point evaluation precompile at Ethereum address `0x0a`
//! validates a single KZG opening proof against a versioned blob
//! commitment:
//!
//! ```text
//!     input  = versioned_hash[32] || z[32] || y[32] ||
//!              commitment[48]   || proof[48]                  (192 bytes)
//!     output = BLS_MODULUS[32] || FIELD_ELEMENTS_PER_BLOB[32] (64 bytes)
//! ```
//!
//! The precompile returns the success constant iff:
//!
//!   1. `versioned_hash[0]   = VERSIONED_HASH_VERSION_KZG = 0x01`
//!   2. `versioned_hash[1..] = sha256(commitment)[1..]`
//!   3. The KZG opening verifies:
//!      `e(commitment - [y]_1, [1]_2) == e(proof, [s - z]_2)`.
//!
//! This AIR is the **precompile-level dispatch row** for `0x0a`:
//! it commits one full call witness per row and exposes the inputs
//! as named slice columns so cross-AIR LogUp descriptors can bind
//!
//!   - `(commitment, versioned_hash)` ↔ [`versioned_hash_air`]
//!     (which in turn bounces off [`sha256_extract`] → SHA-256 AIR
//!     to bind the version-byte SHA-256 condition);
//!   - `(commitment, z, y, proof)` ↔ [`blob_kzg_air`]
//!     (which in turn bounces off [`bls_pairing_air`] →
//!     [`miller_loop_air`] → [`final_exp_air`] to bind the KZG
//!     pairing equation);
//!   - `(input_length=192, output_length=64)` ↔ EVM-side
//!     `precompile_air` dispatch row (placeholder — that AIR lives
//!     in the EVM crate, the column indices are filled with sentinel
//!     `usize::MAX` until the joint-prover orchestration wires them).
//!   - `(input_bytes, output_bytes)` ↔ EVM-side `precompile_io_air`
//!     (placeholder, same reasoning).
//!
//! # Algebraic surface
//!
//! Per row:
//!
//!   1. `is_real    ∈ {0, 1}`               — selector binarity.
//!   2. `is_success ∈ {0, 1}`               — success-flag binarity.
//!   3. `is_real * (is_success - 1) = 0`    — every real row witnesses
//!      a successful precompile call. (Failure cases are out of scope
//!      for this AIR; an EVM-side dispatch will gate the row off when
//!      the precompile returns nothing.)
//!   4. `input[0..32]    = versioned_hash[0..32]`    (32 byte equalities)
//!   5. `input[32..64]   = z[0..32]`                 (32 byte equalities)
//!   6. `input[64..96]   = y[0..32]`                 (32 byte equalities)
//!   7. `input[96..144]  = commitment[0..48]`        (48 byte equalities)
//!   8. `input[144..192] = proof[0..48]`             (48 byte equalities)
//!   9. `is_success * (output[k] - BLS_MODULUS_BE[k])               = 0`
//!      for `k = 0..32`                               (32 byte equalities)
//!  10. `is_success * (output[32+k] - FIELD_ELEMENTS_PER_BLOB_BE[k]) = 0`
//!      for `k = 0..32`                               (32 byte equalities)
//!
//! Total: `3 + 192 + 64 = 259` row-local constraints. No shifted
//! constraints. 8-bit range checks on every byte column
//! (`192 + 32 + 32 + 32 + 48 + 48 + 64 = 448` declarations).
//!
//! # Soundness scope (per-AIR)
//!
//! - The `input` field is the canonical concatenation of the five
//!   named slice columns.
//! - On a successful row, `output` is byte-for-byte the EIP-4844
//!   success constant.
//!
//! **Not** proven here:
//!
//! - That `sha256(commitment)[1..] = versioned_hash[1..]` (the
//!   `kzg_eval_to_versioned_hash` descriptor + the
//!   `versioned_hash_air` ↔ `sha256_extract` chain closes this).
//! - That `e(C - [y]_1, [1]_2) = e(π, [s - z]_2)` (the
//!   `kzg_eval_to_blob_kzg` descriptor + the
//!   `blob_kzg_air` ↔ `bls_pairing_air` chain closes this).
//! - That this row is actually triggered by an EVM `CALL` to
//!   address `0x0a` with matching memory I/O — the EVM-side
//!   `precompile_air` and `precompile_io_air` descriptors close this
//!   once the joint-prover wiring lands.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// EIP-4844 versioned-hash length (= SHA-256 output length).
pub const VERSIONED_HASH_LEN: usize = 32;
/// BLS12-381 scalar field element BE-encoded length.
pub const SCALAR_LEN: usize = 32;
/// BLS12-381 G1 compressed point length.
pub const G1_LEN: usize = 48;

/// Total precompile input length (`versioned_hash || z || y ||
/// commitment || proof`).
pub const INPUT_LEN: usize = VERSIONED_HASH_LEN + SCALAR_LEN + SCALAR_LEN + G1_LEN + G1_LEN;
/// Total precompile output length (`BLS_MODULUS || FIELD_ELEMENTS_PER_BLOB`).
pub const OUTPUT_LEN: usize = SCALAR_LEN + SCALAR_LEN;

/// Byte offsets of the five named slices inside the `input` field.
pub const INPUT_OFF_VERSIONED_HASH: usize = 0;
pub const INPUT_OFF_Z: usize = INPUT_OFF_VERSIONED_HASH + VERSIONED_HASH_LEN;
pub const INPUT_OFF_Y: usize = INPUT_OFF_Z + SCALAR_LEN;
pub const INPUT_OFF_COMMITMENT: usize = INPUT_OFF_Y + SCALAR_LEN;
pub const INPUT_OFF_PROOF: usize = INPUT_OFF_COMMITMENT + G1_LEN;

/// EIP-4844 BLS scalar field modulus, big-endian 32 bytes:
/// `0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001`.
pub const BLS_MODULUS_BE: [u8; SCALAR_LEN] = [
    0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48,
    0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1, 0xd8, 0x05,
    0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
];

/// EIP-4844 `FIELD_ELEMENTS_PER_BLOB = 4096`, big-endian 32 bytes:
/// `0x0000...0001000`.
pub const FIELD_ELEMENTS_PER_BLOB_BE: [u8; SCALAR_LEN] = {
    let mut v = [0u8; SCALAR_LEN];
    // 4096 = 0x1000 → high byte at index 30, low byte at index 31.
    v[30] = 0x10;
    v[31] = 0x00;
    v
};

/// EIP-4844 input length used by the precompile-dispatch binding.
pub const PRECOMPILE_INPUT_LENGTH: u64 = INPUT_LEN as u64;
/// EIP-4844 output length used by the precompile-dispatch binding.
pub const PRECOMPILE_OUTPUT_LENGTH: u64 = OUTPUT_LEN as u64;

// ─── Column layout ────────────────────────────────────────────────────
//
//   input[0..192]            : 192 cols — raw concatenated input
//   versioned_hash[0..32]    :  32 cols — slice copy
//   z[0..32]                 :  32 cols — slice copy
//   y[0..32]                 :  32 cols — slice copy
//   commitment[0..48]        :  48 cols — slice copy
//   proof[0..48]             :  48 cols — slice copy
//   output[0..64]            :  64 cols — precompile return bytes
//   is_real                  :   1 col  — selector
//   is_success               :   1 col  — success flag
//
// total: 192 + 32 + 32 + 32 + 48 + 48 + 64 + 1 + 1 = 450 columns.

pub const COL_INPUT_OFFSET: usize = 0;
pub const COL_VERSIONED_HASH_OFFSET: usize = COL_INPUT_OFFSET + INPUT_LEN;
pub const COL_Z_OFFSET: usize = COL_VERSIONED_HASH_OFFSET + VERSIONED_HASH_LEN;
pub const COL_Y_OFFSET: usize = COL_Z_OFFSET + SCALAR_LEN;
pub const COL_COMMITMENT_OFFSET: usize = COL_Y_OFFSET + SCALAR_LEN;
pub const COL_PROOF_OFFSET: usize = COL_COMMITMENT_OFFSET + G1_LEN;
pub const COL_OUTPUT_OFFSET: usize = COL_PROOF_OFFSET + G1_LEN;
pub const COL_IS_REAL: usize = COL_OUTPUT_OFFSET + OUTPUT_LEN;
pub const COL_IS_SUCCESS: usize = COL_IS_REAL + 1;
pub const NUM_COLUMNS: usize = COL_IS_SUCCESS + 1;

/// Row-local constraints:
///   0:                 is_real ∈ {0, 1}
///   1:                 is_success ∈ {0, 1}
///   2:                 is_real * (is_success - 1) = 0
///   3..35:             input[0..32]    = versioned_hash[0..32]
///   35..67:            input[32..64]   = z[0..32]
///   67..99:            input[64..96]   = y[0..32]
///   99..147:           input[96..144]  = commitment[0..48]
///   147..195:          input[144..192] = proof[0..48]
///   195..227:          is_success * (output[0..32]  - BLS_MODULUS_BE[k])
///   227..259:          is_success * (output[32..64] - FIELD_ELEMENTS_PER_BLOB_BE[k])
pub const NUM_INPUT_SLICE_CONSTRAINTS: usize = INPUT_LEN;
pub const NUM_OUTPUT_CONSTRAINTS: usize = OUTPUT_LEN;
pub const NUM_ROW_CONSTRAINTS: usize =
    3 + NUM_INPUT_SLICE_CONSTRAINTS + NUM_OUTPUT_CONSTRAINTS;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// One KZG point-evaluation precompile call.
#[derive(Clone, Copy, Debug)]
pub struct KzgPointEvalRow {
    pub input: [u8; INPUT_LEN],
    pub versioned_hash: [u8; VERSIONED_HASH_LEN],
    pub z: [u8; SCALAR_LEN],
    pub y: [u8; SCALAR_LEN],
    pub commitment: [u8; G1_LEN],
    pub proof: [u8; G1_LEN],
    pub output: [u8; OUTPUT_LEN],
    pub is_success: bool,
}

#[derive(Clone, Debug, Default)]
pub struct KzgPointEvalWitness {
    pub rows: Vec<KzgPointEvalRow>,
}

impl KzgPointEvalWitness {
    /// Build the canonical success-row witness from the five precompile
    /// input slices. Assembles `input` by concatenation and stamps the
    /// EIP-4844 success constant into `output`.
    pub fn from_inputs(
        versioned_hash: [u8; VERSIONED_HASH_LEN],
        z: [u8; SCALAR_LEN],
        y: [u8; SCALAR_LEN],
        commitment: [u8; G1_LEN],
        proof: [u8; G1_LEN],
    ) -> Self {
        let mut input = [0u8; INPUT_LEN];
        input[INPUT_OFF_VERSIONED_HASH..INPUT_OFF_VERSIONED_HASH + VERSIONED_HASH_LEN]
            .copy_from_slice(&versioned_hash);
        input[INPUT_OFF_Z..INPUT_OFF_Z + SCALAR_LEN].copy_from_slice(&z);
        input[INPUT_OFF_Y..INPUT_OFF_Y + SCALAR_LEN].copy_from_slice(&y);
        input[INPUT_OFF_COMMITMENT..INPUT_OFF_COMMITMENT + G1_LEN]
            .copy_from_slice(&commitment);
        input[INPUT_OFF_PROOF..INPUT_OFF_PROOF + G1_LEN].copy_from_slice(&proof);

        let mut output = [0u8; OUTPUT_LEN];
        output[..SCALAR_LEN].copy_from_slice(&BLS_MODULUS_BE);
        output[SCALAR_LEN..].copy_from_slice(&FIELD_ELEMENTS_PER_BLOB_BE);

        Self {
            rows: vec![KzgPointEvalRow {
                input,
                versioned_hash,
                z,
                y,
                commitment,
                proof,
                output,
                is_success: true,
            }],
        }
    }

    pub fn push(&mut self, row: KzgPointEvalRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &KzgPointEvalWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..INPUT_LEN {
            columns[COL_INPUT_OFFSET + k][r] =
                Scalar::from_u64(row.input[k] as u64, curve);
        }
        for k in 0..VERSIONED_HASH_LEN {
            columns[COL_VERSIONED_HASH_OFFSET + k][r] =
                Scalar::from_u64(row.versioned_hash[k] as u64, curve);
        }
        for k in 0..SCALAR_LEN {
            columns[COL_Z_OFFSET + k][r] = Scalar::from_u64(row.z[k] as u64, curve);
            columns[COL_Y_OFFSET + k][r] = Scalar::from_u64(row.y[k] as u64, curve);
        }
        for k in 0..G1_LEN {
            columns[COL_COMMITMENT_OFFSET + k][r] =
                Scalar::from_u64(row.commitment[k] as u64, curve);
            columns[COL_PROOF_OFFSET + k][r] =
                Scalar::from_u64(row.proof[k] as u64, curve);
        }
        for k in 0..OUTPUT_LEN {
            columns[COL_OUTPUT_OFFSET + k][r] =
                Scalar::from_u64(row.output[k] as u64, curve);
        }
        columns[COL_IS_REAL][r] = one.clone();
        columns[COL_IS_SUCCESS][r] = if row.is_success {
            one.clone()
        } else {
            zero.clone()
        };
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

pub struct KzgPointEvalConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl KzgPointEvalConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            omega: None,
            domain_size: None,
        }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Returns the byte-equality pairs `(input_col, slice_col)` for all
/// 192 input bytes, in declaration order:
///   - versioned_hash[0..32]
///   - z[0..32]
///   - y[0..32]
///   - commitment[0..48]
///   - proof[0..48]
fn input_slice_pairs() -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(INPUT_LEN);
    for k in 0..VERSIONED_HASH_LEN {
        pairs.push((COL_INPUT_OFFSET + INPUT_OFF_VERSIONED_HASH + k,
                    COL_VERSIONED_HASH_OFFSET + k));
    }
    for k in 0..SCALAR_LEN {
        pairs.push((COL_INPUT_OFFSET + INPUT_OFF_Z + k, COL_Z_OFFSET + k));
    }
    for k in 0..SCALAR_LEN {
        pairs.push((COL_INPUT_OFFSET + INPUT_OFF_Y + k, COL_Y_OFFSET + k));
    }
    for k in 0..G1_LEN {
        pairs.push((COL_INPUT_OFFSET + INPUT_OFF_COMMITMENT + k,
                    COL_COMMITMENT_OFFSET + k));
    }
    for k in 0..G1_LEN {
        pairs.push((COL_INPUT_OFFSET + INPUT_OFF_PROOF + k, COL_PROOF_OFFSET + k));
    }
    pairs
}

impl VmConstraintSystem for KzgPointEvalConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("is_success_binary".into());
        labels.push("is_real_implies_success".into());
        for k in 0..VERSIONED_HASH_LEN {
            labels.push(format!("input_eq_versioned_hash_byte_{}", k));
        }
        for k in 0..SCALAR_LEN {
            labels.push(format!("input_eq_z_byte_{}", k));
        }
        for k in 0..SCALAR_LEN {
            labels.push(format!("input_eq_y_byte_{}", k));
        }
        for k in 0..G1_LEN {
            labels.push(format!("input_eq_commitment_byte_{}", k));
        }
        for k in 0..G1_LEN {
            labels.push(format!("input_eq_proof_byte_{}", k));
        }
        for k in 0..SCALAR_LEN {
            labels.push(format!("output_eq_bls_modulus_byte_{}", k));
        }
        for k in 0..SCALAR_LEN {
            labels.push(format!("output_eq_field_elements_per_blob_byte_{}", k));
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
        // 1: is_success binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_SUCCESS][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 2: is_real * (is_success - 1) = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let s = &columns[COL_IS_REAL][r];
                let ok = &columns[COL_IS_SUCCESS][r];
                c[r] = s.mul(&ok.sub(&one));
            }
            out.push(c);
        }

        // 3..195: 192 byte-equality constraints binding input slice
        //         bytes to the named slice columns. Ungated — on
        //         padding rows both sides are zero.
        let pairs = input_slice_pairs();
        for (in_col, slice_col) in pairs {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let a = &columns[in_col][r];
                let b = &columns[slice_col][r];
                c[r] = a.sub(b);
            }
            out.push(c);
        }

        // 195..227: is_success * (output[k] - BLS_MODULUS_BE[k]) = 0.
        for k in 0..SCALAR_LEN {
            let lit = Scalar::from_u64(BLS_MODULUS_BE[k] as u64, curve);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let s = &columns[COL_IS_SUCCESS][r];
                let o = &columns[COL_OUTPUT_OFFSET + k][r];
                c[r] = s.mul(&o.sub(&lit));
            }
            out.push(c);
        }
        // 227..259: is_success * (output[32+k] - FIELD_ELEMENTS_PER_BLOB_BE[k]) = 0.
        for k in 0..SCALAR_LEN {
            let lit = Scalar::from_u64(FIELD_ELEMENTS_PER_BLOB_BE[k] as u64, curve);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let s = &columns[COL_IS_SUCCESS][r];
                let o = &columns[COL_OUTPUT_OFFSET + SCALAR_LEN + k][r];
                c[r] = s.mul(&o.sub(&lit));
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
        // 1: is_success binary.
        {
            let v = &col_evals[COL_IS_SUCCESS];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: is_real * (is_success - 1) = 0.
        {
            let s = &col_evals[COL_IS_REAL];
            let ok = &col_evals[COL_IS_SUCCESS];
            acc = acc.add(&alpha_pow.mul(&s.mul(&ok.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Input slice byte equalities.
        for (in_col, slice_col) in input_slice_pairs() {
            let body = col_evals[in_col].sub(&col_evals[slice_col]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Output BLS_MODULUS bytes.
        for k in 0..SCALAR_LEN {
            let lit = Scalar::from_u64(BLS_MODULUS_BE[k] as u64, curve);
            let s = &col_evals[COL_IS_SUCCESS];
            let o = &col_evals[COL_OUTPUT_OFFSET + k];
            acc = acc.add(&alpha_pow.mul(&s.mul(&o.sub(&lit))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Output FIELD_ELEMENTS_PER_BLOB bytes.
        for k in 0..SCALAR_LEN {
            let lit = Scalar::from_u64(FIELD_ELEMENTS_PER_BLOB_BE[k] as u64, curve);
            let s = &col_evals[COL_IS_SUCCESS];
            let o = &col_evals[COL_OUTPUT_OFFSET + SCALAR_LEN + k];
            acc = acc.add(&alpha_pow.mul(&s.mul(&o.sub(&lit))));
            alpha_pow = alpha_pow.mul(alpha);
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
        // 1: is_success binary.
        {
            let v = &col_coeffs[COL_IS_SUCCESS];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: is_real * (is_success - 1) = 0.
        {
            let s = &col_coeffs[COL_IS_REAL];
            let ok = &col_coeffs[COL_IS_SUCCESS];
            let ok_m1 = poly_sub(ok, &one_poly, curve);
            let body = poly_mul(s, &ok_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Input slice byte equalities.
        for (in_col, slice_col) in input_slice_pairs() {
            let body = poly_sub(&col_coeffs[in_col], &col_coeffs[slice_col], curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Output BLS_MODULUS bytes.
        for k in 0..SCALAR_LEN {
            let lit_poly = vec![Scalar::from_u64(BLS_MODULUS_BE[k] as u64, curve)];
            let s = &col_coeffs[COL_IS_SUCCESS];
            let o = &col_coeffs[COL_OUTPUT_OFFSET + k];
            let body_inner = poly_sub(o, &lit_poly, curve);
            let body = poly_mul(s, &body_inner, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Output FIELD_ELEMENTS_PER_BLOB bytes.
        for k in 0..SCALAR_LEN {
            let lit_poly = vec![Scalar::from_u64(FIELD_ELEMENTS_PER_BLOB_BE[k] as u64, curve)];
            let s = &col_coeffs[COL_IS_SUCCESS];
            let o = &col_coeffs[COL_OUTPUT_OFFSET + SCALAR_LEN + k];
            let body_inner = poly_sub(o, &lit_poly, curve);
            let body = poly_mul(s, &body_inner, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_IS_SUCCESS]
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
        // 8-bit range checks on every byte column.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..INPUT_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("kzg_point_eval_input_byte_{}_8bit", k),
                    column_index: COL_INPUT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..VERSIONED_HASH_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("kzg_point_eval_vh_byte_{}_8bit", k),
                    column_index: COL_VERSIONED_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SCALAR_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("kzg_point_eval_z_byte_{}_8bit", k),
                    column_index: COL_Z_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("kzg_point_eval_y_byte_{}_8bit", k),
                    column_index: COL_Y_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..G1_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("kzg_point_eval_commitment_byte_{}_8bit", k),
                    column_index: COL_COMMITMENT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("kzg_point_eval_proof_byte_{}_8bit", k),
                    column_index: COL_PROOF_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..OUTPUT_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("kzg_point_eval_output_byte_{}_8bit", k),
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

/// Cross-AIR LogUp descriptor: this AIR's
/// `(commitment[0..48] || versioned_hash[0..32])` (gated by `IS_REAL`)
/// ↔ `versioned_hash_air`'s `(commitment[0..48] || versioned_hash[0..32])`
/// (gated by its `IS_REAL`). 80-byte tuple.
///
/// Binds the precompile-row commitment and versioned-hash to a row of
/// `versioned_hash_air`, which then closes the
/// `versioned_hash[1..] = sha256(commitment)[1..]` condition via its
/// downstream `sha256_extract` ↔ SHA-256 chain.
pub fn make_kzg_eval_to_versioned_hash_descriptor(
    kzg_eval_layer_index: usize,
    versioned_hash_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::versioned_hash_air as vh;
    let mut a_columns: Vec<usize> = Vec::with_capacity(G1_LEN + VERSIONED_HASH_LEN);
    for k in 0..G1_LEN {
        a_columns.push(COL_COMMITMENT_OFFSET + k);
    }
    for k in 0..VERSIONED_HASH_LEN {
        a_columns.push(COL_VERSIONED_HASH_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(G1_LEN + VERSIONED_HASH_LEN);
    for k in 0..vh::COMMITMENT_LEN {
        b_columns.push(vh::COL_COMMITMENT_OFFSET + k);
    }
    for k in 0..vh::DIGEST_LEN {
        b_columns.push(vh::COL_VERSIONED_HASH_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "kzg_eval_to_versioned_hash_v1".into(),
        a_layer_index: kzg_eval_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: versioned_hash_layer_index,
        b_columns,
        b_selector_column: Some(vh::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor: this AIR's
/// `(commitment[0..48] || z[0..32] || y[0..32] || proof[0..48])` (gated
/// by `IS_REAL`) ↔ `blob_kzg_air`'s
/// `(commitment_bytes[0..48] || z_bytes[0..32] || y_bytes[0..32] ||
/// proof_bytes[0..48])` (gated by its `IS_REAL`). 160-byte tuple.
///
/// Binds the precompile-row KZG inputs to a row of `blob_kzg_air`,
/// which then closes the pairing equation
/// `e(C - [y]_1, [1]_2) = e(π, [s - z]_2)` via its downstream
/// `bls_pairing_air` ↔ `miller_loop_air` ↔ `final_exp_air` chain.
pub fn make_kzg_eval_to_blob_kzg_descriptor(
    kzg_eval_layer_index: usize,
    blob_kzg_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::blob_kzg_air as bk;
    let mut a_columns: Vec<usize> =
        Vec::with_capacity(G1_LEN + SCALAR_LEN + SCALAR_LEN + G1_LEN);
    for k in 0..G1_LEN {
        a_columns.push(COL_COMMITMENT_OFFSET + k);
    }
    for k in 0..SCALAR_LEN {
        a_columns.push(COL_Z_OFFSET + k);
    }
    for k in 0..SCALAR_LEN {
        a_columns.push(COL_Y_OFFSET + k);
    }
    for k in 0..G1_LEN {
        a_columns.push(COL_PROOF_OFFSET + k);
    }

    let mut b_columns: Vec<usize> =
        Vec::with_capacity(G1_LEN + SCALAR_LEN + SCALAR_LEN + G1_LEN);
    for k in 0..bk::G1_BYTES {
        b_columns.push(bk::COL_COMMITMENT_BYTES_OFFSET + k);
    }
    for k in 0..bk::SCALAR_BYTES {
        b_columns.push(bk::COL_Z_BYTES_OFFSET + k);
    }
    for k in 0..bk::SCALAR_BYTES {
        b_columns.push(bk::COL_Y_BYTES_OFFSET + k);
    }
    for k in 0..bk::G1_BYTES {
        b_columns.push(bk::COL_PROOF_BYTES_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "kzg_eval_to_blob_kzg_v1".into(),
        a_layer_index: kzg_eval_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: blob_kzg_layer_index,
        b_columns,
        b_selector_column: Some(bk::COL_IS_REAL),
    }
}

/// Placeholder sentinel for `precompile_air` column indices.
///
/// `precompile_air` lives in the EVM crate, so this zkp-side AIR can
/// not reference its column constants directly. The descriptor below
/// uses this sentinel on the B side; the joint prover orchestration
/// (which links zkp and evm AIRs together) substitutes the real
/// indices when wiring the descriptor.
pub const PRECOMPILE_DISPATCH_PLACEHOLDER: usize = usize::MAX;

/// A-side payload column synthesised by the descriptor for the
/// `input_length = 192` and `output_length = 64` literals. Since the
/// dispatch row commits scalar lengths (not byte arrays), the binding
/// is via a synthetic 2-column tuple `(input_length, output_length)`.
/// Until a dedicated dispatch row is built in zkp, this descriptor
/// records the **shape** of the binding (a 2-column tuple gated by
/// `IS_REAL`); the A-side columns are reserved placeholders.
pub const KZG_EVAL_DISPATCH_INPUT_LENGTH_PLACEHOLDER: usize = usize::MAX;
pub const KZG_EVAL_DISPATCH_OUTPUT_LENGTH_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds
/// `(input_length = 192, output_length = 64)` on this AIR's row to
/// the corresponding `precompile_air` dispatch row for callee `0x0a`.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B-side
/// columns are filled with [`PRECOMPILE_DISPATCH_PLACEHOLDER`]. The
/// A-side currently does not commit dedicated length scalars (the
/// lengths are implied by the fixed-width input/output column ranges);
/// for the well-formedness contract we record placeholder A-side cols
/// too. The joint-prover orchestration substitutes the real indices.
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_kzg_eval_to_precompile_dispatch_descriptor(
    kzg_eval_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // 2-column tuple: (input_length, output_length).
    let a_columns: Vec<usize> = vec![
        KZG_EVAL_DISPATCH_INPUT_LENGTH_PLACEHOLDER,
        KZG_EVAL_DISPATCH_OUTPUT_LENGTH_PLACEHOLDER,
    ];
    let b_columns: Vec<usize> = vec![
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_INPUT_LENGTH
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_OUTPUT_LENGTH
    ];
    CrossAirLogUpDescriptor {
        label: "kzg_eval_to_precompile_dispatch_v1_stub".into(),
        a_layer_index: kzg_eval_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: None, // precompile_air::COL_IS_REAL is filled by orchestrator
    }
}

/// Placeholder sentinel for `precompile_io_air` column indices.
pub const PRECOMPILE_IO_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds the precompile-row
/// `input[0..192]` and `output[0..64]` byte arrays to the
/// `precompile_io_air` row that records the matching EVM CALL.
///
/// **Stub**: zkp does not depend on the EVM crate. The A side is the
/// full input + output column range on this AIR; the B side is filled
/// with [`PRECOMPILE_IO_PLACEHOLDER`] until the joint-prover
/// orchestration substitutes the real indices.
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_kzg_eval_to_precompile_io_descriptor(
    kzg_eval_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(INPUT_LEN + OUTPUT_LEN);
    for k in 0..INPUT_LEN {
        a_columns.push(COL_INPUT_OFFSET + k);
    }
    for k in 0..OUTPUT_LEN {
        a_columns.push(COL_OUTPUT_OFFSET + k);
    }
    let b_columns: Vec<usize> = vec![PRECOMPILE_IO_PLACEHOLDER; INPUT_LEN + OUTPUT_LEN];
    CrossAirLogUpDescriptor {
        label: "kzg_eval_to_precompile_io_v1_stub".into(),
        a_layer_index: kzg_eval_layer_index,
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

    fn sample_inputs() -> (
        [u8; VERSIONED_HASH_LEN],
        [u8; SCALAR_LEN],
        [u8; SCALAR_LEN],
        [u8; G1_LEN],
        [u8; G1_LEN],
    ) {
        let mut commitment = [0u8; G1_LEN];
        for k in 0..G1_LEN {
            commitment[k] = (k as u8).wrapping_mul(3).wrapping_add(7);
        }
        commitment[0] &= 0x1f;
        let digest = crate::sha256::sha256(&commitment);
        let mut versioned_hash = digest;
        versioned_hash[0] = 0x01;

        let mut z = [0u8; SCALAR_LEN];
        for k in 0..SCALAR_LEN {
            z[k] = (k as u8).wrapping_mul(2).wrapping_add(1);
        }
        z[0] &= 0x7f;
        let mut y = [0u8; SCALAR_LEN];
        for k in 0..SCALAR_LEN {
            y[k] = (k as u8).wrapping_mul(4).wrapping_add(3);
        }
        y[0] &= 0x7f;
        let mut proof = [0u8; G1_LEN];
        for k in 0..G1_LEN {
            proof[k] = (k as u8).wrapping_mul(5).wrapping_add(11);
        }
        proof[0] &= 0x1f;

        (versioned_hash, z, y, commitment, proof)
    }

    #[test]
    fn build_poly_matches_evaluate_at_point() {
        use crate::commitment;
        use bls48581::bls48581::big;
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = KzgPointEvalConstraintSystem::new(trace.num_rows);
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
            "kzg_point_eval_air: build_constraint_polynomial(z) must equal evaluate_at_point",
        );
    }

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_INPUT_OFFSET, 0);
        assert_eq!(COL_VERSIONED_HASH_OFFSET, 192);
        assert_eq!(COL_Z_OFFSET, 192 + 32);
        assert_eq!(COL_Y_OFFSET, 192 + 64);
        assert_eq!(COL_COMMITMENT_OFFSET, 192 + 96);
        assert_eq!(COL_PROOF_OFFSET, 192 + 96 + 48);
        assert_eq!(COL_OUTPUT_OFFSET, 192 + 96 + 96);
        assert_eq!(COL_IS_REAL, 192 + 96 + 96 + 64);
        assert_eq!(COL_IS_SUCCESS, COL_IS_REAL + 1);
        assert_eq!(NUM_COLUMNS, 450);
        assert_eq!(NUM_ROW_CONSTRAINTS, 3 + 192 + 64);
        assert_eq!(NUM_ROW_CONSTRAINTS, 259);
        // BLS_MODULUS literal sanity.
        assert_eq!(BLS_MODULUS_BE[0], 0x73);
        assert_eq!(BLS_MODULUS_BE[31], 0x01);
        // FIELD_ELEMENTS_PER_BLOB = 4096 sanity.
        assert_eq!(FIELD_ELEMENTS_PER_BLOB_BE[30], 0x10);
        assert_eq!(FIELD_ELEMENTS_PER_BLOB_BE[31], 0x00);
        let mut total: u64 = 0;
        for &b in &FIELD_ELEMENTS_PER_BLOB_BE {
            total = (total << 8) | (b as u64);
        }
        assert_eq!(total, 4096);
        assert_eq!(PRECOMPILE_INPUT_LENGTH, 192);
        assert_eq!(PRECOMPILE_OUTPUT_LENGTH, 64);
    }

    #[test]
    fn witness_from_inputs_concatenates_input_and_stamps_output() {
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        // input slice composition.
        assert_eq!(&row.input[0..32], &vh[..]);
        assert_eq!(&row.input[32..64], &z[..]);
        assert_eq!(&row.input[64..96], &y[..]);
        assert_eq!(&row.input[96..144], &c[..]);
        assert_eq!(&row.input[144..192], &p[..]);
        // success constant.
        assert!(row.is_success);
        assert_eq!(&row.output[..32], &BLS_MODULUS_BE[..]);
        assert_eq!(&row.output[32..], &FIELD_ELEMENTS_PER_BLOB_BE[..]);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = KzgPointEvalConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
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
    fn tampered_input_slice_fires_byte_equality() {
        // Bump input[100] (lives in the commitment slice). The
        // matching byte-equality constraint must fire while the
        // is_real / is_success binarity stays intact.
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls48581;
        // input[100] = commitment[4] — bump.
        let i_col = COL_INPUT_OFFSET + 100;
        let orig = cols[i_col][0].clone();
        cols[i_col][0] = orig.add(&Scalar::one(curve));

        let cs = KzgPointEvalConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // 192 input-slice constraints start at index 3. input[100]
        // is the 100th slice byte (versioned_hash=32 + z=32 + y=32 +
        // commitment_byte_4) → 100. So constraint index 3 + 100.
        let fired = &results[3 + 100][0];
        assert!(
            !fired.is_zero(),
            "input[100] tamper must fire byte-eq constraint 3+100",
        );
        // is_real / is_success binary constraints stay zero.
        assert!(results[0][0].is_zero());
        assert!(results[1][0].is_zero());
        assert!(results[2][0].is_zero());
    }

    #[test]
    fn tampered_output_byte_fires_success_constant() {
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls48581;
        // Replace output[0] with 0x00 (correct value is 0x73).
        cols[COL_OUTPUT_OFFSET + 0][0] = Scalar::from_u64(0x00, curve);
        let cs = KzgPointEvalConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Output BLS_MODULUS constraints start at index 3 + 192 = 195.
        let idx = 3 + INPUT_LEN + 0;
        assert!(
            !results[idx][0].is_zero(),
            "output[0] tamper must fire BLS_MODULUS byte-0 constraint",
        );
        // Tampering the second half (FIELD_ELEMENTS_PER_BLOB) too.
        cols[COL_OUTPUT_OFFSET + SCALAR_LEN + 30][0] = Scalar::from_u64(0x00, curve);
        let col_refs2: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results2 = cs.evaluate_on_domain(&col_refs2, trace.num_rows);
        let idx2 = 3 + INPUT_LEN + SCALAR_LEN + 30;
        assert!(
            !results2[idx2][0].is_zero(),
            "output[32+30] tamper must fire FIELD_ELEMENTS_PER_BLOB byte-30 constraint",
        );
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_selector() {
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(5, CurveType::Bls48581);
        let cs = KzgPointEvalConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[0][0].is_zero(),
            "non-binary is_real must fire constraint 0",
        );
    }

    #[test]
    fn versioned_hash_descriptor_well_formed() {
        let d = make_kzg_eval_to_versioned_hash_descriptor(0, 1);
        assert_eq!(d.label, "kzg_eval_to_versioned_hash_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), G1_LEN + VERSIONED_HASH_LEN);
        assert_eq!(d.b_columns.len(), G1_LEN + VERSIONED_HASH_LEN);
        // A side: commitment (48), versioned_hash (32).
        for k in 0..G1_LEN {
            assert_eq!(d.a_columns[k], COL_COMMITMENT_OFFSET + k);
        }
        for k in 0..VERSIONED_HASH_LEN {
            assert_eq!(d.a_columns[G1_LEN + k], COL_VERSIONED_HASH_OFFSET + k);
        }
        // B side: vh::commitment (48), vh::versioned_hash (32).
        for k in 0..G1_LEN {
            assert_eq!(
                d.b_columns[k],
                crate::versioned_hash_air::COL_COMMITMENT_OFFSET + k,
            );
        }
        for k in 0..VERSIONED_HASH_LEN {
            assert_eq!(
                d.b_columns[G1_LEN + k],
                crate::versioned_hash_air::COL_VERSIONED_HASH_OFFSET + k,
            );
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::versioned_hash_air::COL_IS_REAL),
        );
    }

    #[test]
    fn blob_kzg_descriptor_well_formed() {
        let d = make_kzg_eval_to_blob_kzg_descriptor(0, 1);
        assert_eq!(d.label, "kzg_eval_to_blob_kzg_v1");
        let expected_width = G1_LEN + SCALAR_LEN + SCALAR_LEN + G1_LEN;
        assert_eq!(d.a_columns.len(), expected_width);
        assert_eq!(d.b_columns.len(), expected_width);
        // A side ordering: commitment, z, y, proof.
        let mut idx = 0;
        for k in 0..G1_LEN {
            assert_eq!(d.a_columns[idx], COL_COMMITMENT_OFFSET + k);
            idx += 1;
        }
        for k in 0..SCALAR_LEN {
            assert_eq!(d.a_columns[idx], COL_Z_OFFSET + k);
            idx += 1;
        }
        for k in 0..SCALAR_LEN {
            assert_eq!(d.a_columns[idx], COL_Y_OFFSET + k);
            idx += 1;
        }
        for k in 0..G1_LEN {
            assert_eq!(d.a_columns[idx], COL_PROOF_OFFSET + k);
            idx += 1;
        }
        // B side: blob_kzg_air columns in same order.
        let mut idx = 0;
        for k in 0..G1_LEN {
            assert_eq!(
                d.b_columns[idx],
                crate::blob_kzg_air::COL_COMMITMENT_BYTES_OFFSET + k,
            );
            idx += 1;
        }
        for k in 0..SCALAR_LEN {
            assert_eq!(
                d.b_columns[idx],
                crate::blob_kzg_air::COL_Z_BYTES_OFFSET + k,
            );
            idx += 1;
        }
        for k in 0..SCALAR_LEN {
            assert_eq!(
                d.b_columns[idx],
                crate::blob_kzg_air::COL_Y_BYTES_OFFSET + k,
            );
            idx += 1;
        }
        for k in 0..G1_LEN {
            assert_eq!(
                d.b_columns[idx],
                crate::blob_kzg_air::COL_PROOF_BYTES_OFFSET + k,
            );
            idx += 1;
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::blob_kzg_air::COL_IS_REAL),
        );
    }

    #[test]
    fn precompile_dispatch_descriptor_stub_well_formed() {
        let d = make_kzg_eval_to_precompile_dispatch_descriptor(0, 1);
        assert_eq!(d.label, "kzg_eval_to_precompile_dispatch_v1_stub");
        assert_eq!(d.a_columns.len(), 2);
        assert_eq!(d.b_columns.len(), 2);
        // A side currently placeholder (lengths are implicit in the
        // fixed-width input/output column ranges).
        assert_eq!(d.a_columns[0], KZG_EVAL_DISPATCH_INPUT_LENGTH_PLACEHOLDER);
        assert_eq!(d.a_columns[1], KZG_EVAL_DISPATCH_OUTPUT_LENGTH_PLACEHOLDER);
        // B side placeholder until joint-prover orchestration wires
        // EVM-crate precompile_air column indices.
        assert_eq!(d.b_columns[0], PRECOMPILE_DISPATCH_PLACEHOLDER);
        assert_eq!(d.b_columns[1], PRECOMPILE_DISPATCH_PLACEHOLDER);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn precompile_io_descriptor_stub_well_formed() {
        let d = make_kzg_eval_to_precompile_io_descriptor(0, 1);
        assert_eq!(d.label, "kzg_eval_to_precompile_io_v1_stub");
        let expected_width = INPUT_LEN + OUTPUT_LEN;
        assert_eq!(d.a_columns.len(), expected_width);
        assert_eq!(d.b_columns.len(), expected_width);
        // A side: full input (192) then full output (64).
        for k in 0..INPUT_LEN {
            assert_eq!(d.a_columns[k], COL_INPUT_OFFSET + k);
        }
        for k in 0..OUTPUT_LEN {
            assert_eq!(d.a_columns[INPUT_LEN + k], COL_OUTPUT_OFFSET + k);
        }
        // B side fully placeholder.
        for &col in &d.b_columns {
            assert_eq!(col, PRECOMPILE_IO_PLACEHOLDER);
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn lookup_declarations_cover_all_byte_columns() {
        let cs = KzgPointEvalConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        assert_eq!(req.tables.len(), 1);
        // input(192) + vh(32) + z(32) + y(32) + commitment(48) +
        // proof(48) + output(64) = 448.
        let expected = INPUT_LEN
            + VERSIONED_HASH_LEN
            + SCALAR_LEN
            + SCALAR_LEN
            + G1_LEN
            + G1_LEN
            + OUTPUT_LEN;
        assert_eq!(expected, 448);
        assert_eq!(req.declarations.len(), expected);
        // All declarations are 8-bit.
        for (decl, _) in &req.declarations {
            assert_eq!(decl.max_bits, 8);
        }
        // Distinct column indices — no double-coverage.
        let mut cols: Vec<usize> = req
            .declarations
            .iter()
            .map(|(d, _)| d.column_index)
            .collect();
        cols.sort();
        let n_before = cols.len();
        cols.dedup();
        assert_eq!(cols.len(), n_before, "byte cols should be distinct");
        // Constraint count and label count agree.
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
    }

    #[test]
    fn is_real_implies_success_fires_when_unsuccessful_real_row() {
        // Construct a row by hand with is_real=1, is_success=0;
        // constraint 2 must fire.
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_SUCCESS][0] = Scalar::zero(CurveType::Bls48581);
        let cs = KzgPointEvalConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[2][0].is_zero(),
            "is_real * (is_success - 1) must fire when is_success=0 on a real row",
        );
        // Binary constraints still hold.
        assert!(results[0][0].is_zero());
        assert!(results[1][0].is_zero());
    }

    #[test]
    #[ignore = "slow: kzg_point_eval_air standalone prove_with_scheme + verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let (vh, z, y, c, p) = sample_inputs();
        let w = KzgPointEvalWitness::from_inputs(vh, z, y, c, p);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = KzgPointEvalConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone kzg_point_eval_air proof must verify",
        );
    }
}
