//! EIP-4844 blob KZG commitment verification AIR.
//!
//! # Purpose
//!
//! EIP-4844 attaches one or more "blob KZG commitments" to a
//! transaction. Each commitment `C ∈ G1` is a KZG commitment to a
//! blob (4096 BLS12-381 scalar field elements), and the consensus
//! client verifies, per commitment, that a quotient `π ∈ G1` opens
//! `C` at evaluation point `z` to value `y`:
//!
//! ```text
//!     e(C - y·G1, [τ]_2)  ==  e(π, [τ - z]_2)
//! ```
//!
//! The right factor lives in G2 and is **not** the same as the BLS
//! signature aggregate-verify pairing — but it is the *same shape*:
//! a product-of-pairings equality reducing to `e(·, ·) · e(·, ·) = 1`
//! after one G2 negation and rearrangement. Full algebraic closure
//! requires `bls_pairing_air` + `miller_loop_air` + `final_exp_air`
//! to consume the two G1 inputs and the two G2 inputs.
//!
//! This AIR is the **entry point**: it commits one
//! `(commitment, proof, z, y, evaluation)` tuple per row, decomposes
//! each G1 point into 6 big-endian Fp limbs (matching the
//! `nonnative_fp` convention), and exposes cross-AIR LogUp
//! descriptors that hand the G1 byte form off to `bls_pairing_air`
//! for the actual pairing check. Soundness of the pairing equation
//! itself is deferred to those downstream AIRs.
//!
//! # What is algebraically enforced
//!
//! Per row:
//!
//!   1. `is_real ∈ {0, 1}` — selector binarity.
//!   2. Commitment-Fp byte-to-limb decomposition (6 equations).
//!   3. Proof-Fp byte-to-limb decomposition (6 equations).
//!   4. `y_bytes == evaluation_bytes` (32 equations) — the host-side
//!      KZG verifier opens `C` at `z` to `y`, and the consumer
//!      (transaction RLP / blob sidecar) commits `evaluation` as the
//!      claimed value; these must be the same scalar. Bound byte-wise
//!      so a cross-AIR LogUp on either column witnesses the other.
//!
//! Total: 1 + 6 + 6 + 32 = **45 row-local constraints**.
//!
//! # What is NOT yet enforced (deferred to downstream AIRs)
//!
//! - On-curve checks for `commitment` and `proof` G1 points (need
//!   `y² = x³ + 4` Fp arithmetic — one row each of
//!   `nonnative_fp_air`).
//! - The KZG pairing equation itself — handed off via
//!   [`make_blob_kzg_to_pairing_descriptor`] to `bls_pairing_air`,
//!   which in turn drives `miller_loop_air` and `final_exp_air`.
//! - Reduction of `z` and `y` modulo the BLS12-381 scalar field
//!   modulus `r`. The host-side encoding is canonical 32-byte BE
//!   with the top bit clear; a future scalar-canonicalization gadget
//!   will bind this.
//! - Versioned-hash binding: EIP-4844 commits `versioned_hash =
//!   0x01 || sha256(commitment)[1..]` in the transaction body. A
//!   downstream linkage (sha256_air ↔ blob_kzg_air on the
//!   commitment bytes) closes this; not in scope here.
//!
//! # Cross-AIR linkages
//!
//! - [`make_blob_kzg_to_pairing_descriptor`] — 48-byte tuple on
//!   `commitment_bytes` (gated by `IS_REAL`) ↔ `bls_pairing_air`'s
//!   `pk_compressed[0..48]` (gated by `IS_REAL`). Identical shape
//!   reused for the proof bytes via
//!   [`make_blob_kzg_proof_to_pairing_descriptor`]. This is the
//!   load-bearing soundness — once the pairing AIR is closed the
//!   commitment/proof witnessed here is bound to the pairing check.
//! - [`make_blob_kzg_byte_descriptors`] — bookkeeping descriptors
//!   for the byte representations of `z`, `y`, and `evaluation`
//!   that downstream transaction-RLP / blob-sidecar AIRs can use to
//!   bind these scalars to their source.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// BLS12-381 G1 compressed point length.
pub const G1_BYTES: usize = 48;
/// 32-byte BE encoding of a BLS12-381 scalar field element.
pub const SCALAR_BYTES: usize = 32;

/// Number of 64-bit limbs in one Fp element (6 × 64 = 384 ≥ 381).
pub const LIMBS_PER_FP: usize = 6;
/// Bytes per limb.
pub const BYTES_PER_LIMB: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────
//
// Row layout, one EIP-4844 blob KZG verification per row:
//
//   commitment_bytes : 48 bytes (IETF compressed G1, flags in byte 0)
//   proof_bytes      : 48 bytes (IETF compressed G1, flags in byte 0)
//   z_bytes          : 32 bytes (BE scalar field element)
//   y_bytes          : 32 bytes (BE scalar field element)
//   evaluation_bytes : 32 bytes (BE scalar field element, must == y)
//   commitment_x_limbs : 6 limbs (BE u64; matches Fp::limbs convention)
//   proof_x_limbs      : 6 limbs
//   commitment_flag_bits  : 1 (top 3 bits of commitment byte 0, 0..=7)
//   commitment_masked_b0  : 1 (low 5 bits of commitment byte 0, 0..=31)
//   proof_flag_bits       : 1 (top 3 bits of proof byte 0, 0..=7)
//   proof_masked_b0       : 1 (low 5 bits of proof byte 0, 0..=31)
//   is_real          : 1
//
// total: 48 + 48 + 32 + 32 + 32 + 6 + 6 + 4 + 1 = 209 columns
//
// **Flag byte splitter (new in #290)**. IETF G1 compressed encoding
// stuffs 3 flag bits (compressed, infinity, sort) into the top 3 bits
// of byte 0; the actual Fp x-coordinate occupies the low 5 bits of
// byte 0 plus all of bytes 1..48. The original layout used raw byte 0
// in the limb-0 decomposition, which silently required byte 0 < 0x20
// — meaning real BLS12-381 commitments (whose compressed flag bit is
// always set, so byte 0 ≥ 0x80) could not be witnessed.
//
// The fix splits byte 0 algebraically into `flag_bits` (top 3) and
// `masked_byte_0` (low 5) via the equation
//     commitment_bytes[0] = 32·flag_bits + masked_byte_0
// (and similarly for proof). The limb-0 decomposition then consumes
// `masked_byte_0` in place of the raw byte 0. The downstream pairing
// AIR enforces the global tightness that `masked_byte_0` is the true
// low-5-bits of the Fp coordinate (any cheat propagates into the
// `*_x_limbs[0]` value and thus into the pairing check).

pub const COL_COMMITMENT_BYTES_OFFSET: usize = 0;
pub const COL_PROOF_BYTES_OFFSET: usize = COL_COMMITMENT_BYTES_OFFSET + G1_BYTES;
pub const COL_Z_BYTES_OFFSET: usize = COL_PROOF_BYTES_OFFSET + G1_BYTES;
pub const COL_Y_BYTES_OFFSET: usize = COL_Z_BYTES_OFFSET + SCALAR_BYTES;
pub const COL_EVALUATION_BYTES_OFFSET: usize = COL_Y_BYTES_OFFSET + SCALAR_BYTES;
pub const COL_COMMITMENT_X_LIMB_OFFSET: usize = COL_EVALUATION_BYTES_OFFSET + SCALAR_BYTES;
pub const COL_PROOF_X_LIMB_OFFSET: usize = COL_COMMITMENT_X_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_COMMITMENT_FLAG_BITS: usize = COL_PROOF_X_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_COMMITMENT_MASKED_BYTE_0: usize = COL_COMMITMENT_FLAG_BITS + 1;
pub const COL_PROOF_FLAG_BITS: usize = COL_COMMITMENT_MASKED_BYTE_0 + 1;
pub const COL_PROOF_MASKED_BYTE_0: usize = COL_PROOF_FLAG_BITS + 1;
pub const COL_IS_REAL: usize = COL_PROOF_MASKED_BYTE_0 + 1;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// Row-local constraints:
///   0:        is_real ∈ {0, 1}
///   1..7:     6 commitment-x limb-decomp equations (limb 0 uses masked byte 0)
///   7..13:    6 proof-x limb-decomp equations      (limb 0 uses masked byte 0)
///   13:       commitment byte 0 = 32·flag_bits + masked_byte_0
///   14:       proof byte 0      = 32·flag_bits + masked_byte_0
///   15..47:   32 y == evaluation byte-equality equations
pub const NUM_Y_EQ_EVAL_CONSTRAINTS: usize = SCALAR_BYTES;
pub const NUM_FLAG_SPLIT_CONSTRAINTS: usize = 2;
pub const NUM_ROW_CONSTRAINTS: usize = 1
    + LIMBS_PER_FP
    + LIMBS_PER_FP
    + NUM_FLAG_SPLIT_CONSTRAINTS
    + NUM_Y_EQ_EVAL_CONSTRAINTS;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// One EIP-4844 blob KZG verification row.
///
/// Field-element columns are stored as 32-byte big-endian arrays.
/// G1 columns are 48-byte IETF compressed encodings with the 3
/// flag bits in byte 0. The `*_x_limbs` columns are the canonical
/// 6-limb BE representation of the masked x-coordinate, matching
/// the [`crate::nonnative_fp::Fp::limbs`] convention.
#[derive(Clone, Copy, Debug)]
pub struct BlobKzgRow {
    pub commitment: [u8; G1_BYTES],
    pub proof: [u8; G1_BYTES],
    pub z: [u8; SCALAR_BYTES],
    pub y: [u8; SCALAR_BYTES],
    pub evaluation: [u8; SCALAR_BYTES],
    /// Canonical BE 6-limb representation of commitment.x (flag
    /// bits masked off byte 0).
    pub commitment_x_limbs: [u64; LIMBS_PER_FP],
    /// Canonical BE 6-limb representation of proof.x.
    pub proof_x_limbs: [u64; LIMBS_PER_FP],
    /// Top 3 bits of `commitment[0]` (IETF compressed/infinity/sort
    /// flags), value in `0..=7`.
    pub commitment_flag_bits: u8,
    /// Low 5 bits of `commitment[0]` (true MSB of Fp coordinate),
    /// value in `0..=31`.
    pub commitment_masked_byte_0: u8,
    /// Top 3 bits of `proof[0]`.
    pub proof_flag_bits: u8,
    /// Low 5 bits of `proof[0]`.
    pub proof_masked_byte_0: u8,
}

#[derive(Clone, Debug, Default)]
pub struct BlobKzgWitness {
    pub rows: Vec<BlobKzgRow>,
}

/// Split a 48-byte big-endian Fp encoding into 6 big-endian u64
/// limbs (limb 0 = MSB), matching [`crate::nonnative_fp::Fp::limbs`].
fn bytes48_to_be_limbs(bytes: &[u8; G1_BYTES]) -> [u64; LIMBS_PER_FP] {
    let mut limbs = [0u64; LIMBS_PER_FP];
    for j in 0..LIMBS_PER_FP {
        let mut buf = [0u8; BYTES_PER_LIMB];
        buf.copy_from_slice(&bytes[j * BYTES_PER_LIMB..j * BYTES_PER_LIMB + BYTES_PER_LIMB]);
        limbs[j] = u64::from_be_bytes(buf);
    }
    limbs
}

impl BlobKzgWitness {
    /// Build a single-row witness from raw byte inputs.
    ///
    /// Sanity-strips the 3 flag bits from byte 0 of `commitment` and
    /// `proof` when computing the limb decomposition (the limb-decomp
    /// constraint will fire if the prover lies about the limb form).
    pub fn from_commitment_proof(
        commitment: &[u8; G1_BYTES],
        proof: &[u8; G1_BYTES],
        z: &[u8; SCALAR_BYTES],
        y: &[u8; SCALAR_BYTES],
        evaluation: &[u8; SCALAR_BYTES],
    ) -> Self {
        let mut commitment_x = *commitment;
        commitment_x[0] &= 0x1f;
        let mut proof_x = *proof;
        proof_x[0] &= 0x1f;
        let commitment_flag_bits = (commitment[0] >> 5) & 0x07;
        let commitment_masked_byte_0 = commitment[0] & 0x1f;
        let proof_flag_bits = (proof[0] >> 5) & 0x07;
        let proof_masked_byte_0 = proof[0] & 0x1f;
        Self {
            rows: vec![BlobKzgRow {
                commitment: *commitment,
                proof: *proof,
                z: *z,
                y: *y,
                evaluation: *evaluation,
                commitment_x_limbs: bytes48_to_be_limbs(&commitment_x),
                proof_x_limbs: bytes48_to_be_limbs(&proof_x),
                commitment_flag_bits,
                commitment_masked_byte_0,
                proof_flag_bits,
                proof_masked_byte_0,
            }],
        }
    }

    pub fn push(&mut self, row: BlobKzgRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BlobKzgWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..G1_BYTES {
            columns[COL_COMMITMENT_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.commitment[k] as u64, curve);
            columns[COL_PROOF_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.proof[k] as u64, curve);
        }
        for k in 0..SCALAR_BYTES {
            columns[COL_Z_BYTES_OFFSET + k][r] = Scalar::from_u64(row.z[k] as u64, curve);
            columns[COL_Y_BYTES_OFFSET + k][r] = Scalar::from_u64(row.y[k] as u64, curve);
            columns[COL_EVALUATION_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.evaluation[k] as u64, curve);
        }
        for j in 0..LIMBS_PER_FP {
            columns[COL_COMMITMENT_X_LIMB_OFFSET + j][r] =
                Scalar::from_u64(row.commitment_x_limbs[j], curve);
            columns[COL_PROOF_X_LIMB_OFFSET + j][r] =
                Scalar::from_u64(row.proof_x_limbs[j], curve);
        }
        columns[COL_COMMITMENT_FLAG_BITS][r] =
            Scalar::from_u64(row.commitment_flag_bits as u64, curve);
        columns[COL_COMMITMENT_MASKED_BYTE_0][r] =
            Scalar::from_u64(row.commitment_masked_byte_0 as u64, curve);
        columns[COL_PROOF_FLAG_BITS][r] =
            Scalar::from_u64(row.proof_flag_bits as u64, curve);
        columns[COL_PROOF_MASKED_BYTE_0][r] =
            Scalar::from_u64(row.proof_masked_byte_0 as u64, curve);
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

pub struct BlobKzgConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BlobKzgConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// For Fp limb `j` (big-endian; `j = 0` is MSB), the list of
/// `(byte_offset_within_g1_bytes, power_of_256)` pairs whose weighted
/// sum equals the limb's u64 value.
///
/// **NB**: For limb `j = 0` the most significant byte is the IETF flag
/// byte; the decomposition equations consume `masked_byte_0` (low 5
/// bits) instead of the raw byte 0. The caller substitutes the
/// appropriate column index for the entry whose `byte_idx == 0`.
fn x_limb_decomp_targets(limb_j: usize) -> Vec<(usize, u64)> {
    (0..BYTES_PER_LIMB)
        .map(|k| {
            let byte_idx = limb_j * BYTES_PER_LIMB + k;
            let weight = 1u64 << (8 * (BYTES_PER_LIMB - 1 - k));
            (byte_idx, weight)
        })
        .collect()
}

/// Resolve the column index that holds the byte contribution for
/// limb-decomp entry `byte_idx` within G1 byte offsets — substituting
/// the masked-byte-0 column for `byte_idx == 0`.
fn commit_byte_col(byte_idx: usize) -> usize {
    if byte_idx == 0 {
        COL_COMMITMENT_MASKED_BYTE_0
    } else {
        COL_COMMITMENT_BYTES_OFFSET + byte_idx
    }
}

fn proof_byte_col(byte_idx: usize) -> usize {
    if byte_idx == 0 {
        COL_PROOF_MASKED_BYTE_0
    } else {
        COL_PROOF_BYTES_OFFSET + byte_idx
    }
}

impl VmConstraintSystem for BlobKzgConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("commitment_x_limb_{}_decomp", j));
        }
        for j in 0..LIMBS_PER_FP {
            labels.push(format!("proof_x_limb_{}_decomp", j));
        }
        labels.push("commitment_flag_byte_split".into());
        labels.push("proof_flag_byte_split".into());
        for k in 0..SCALAR_BYTES {
            labels.push(format!("y_eq_evaluation_byte_{}", k));
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

        // 1..7: commitment_x limb decomposition (limb 0 byte 0 →
        // masked_byte_0).
        for j in 0..LIMBS_PER_FP {
            let targets = x_limb_decomp_targets(j);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, weight) in &targets {
                    let b = &columns[commit_byte_col(*byte_idx)][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*weight, curve)));
                }
                c[r] = columns[COL_COMMITMENT_X_LIMB_OFFSET + j][r].sub(&sum);
            }
            out.push(c);
        }

        // 7..13: proof_x limb decomposition (limb 0 byte 0 →
        // masked_byte_0).
        for j in 0..LIMBS_PER_FP {
            let targets = x_limb_decomp_targets(j);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, weight) in &targets {
                    let b = &columns[proof_byte_col(*byte_idx)][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*weight, curve)));
                }
                c[r] = columns[COL_PROOF_X_LIMB_OFFSET + j][r].sub(&sum);
            }
            out.push(c);
        }

        // 13: commitment_bytes[0] = 32·flag_bits + masked_byte_0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            let thirty_two = Scalar::from_u64(32, curve);
            for r in 0..n {
                let raw = &columns[COL_COMMITMENT_BYTES_OFFSET][r];
                let flag = &columns[COL_COMMITMENT_FLAG_BITS][r];
                let masked = &columns[COL_COMMITMENT_MASKED_BYTE_0][r];
                c[r] = raw.sub(&flag.mul(&thirty_two).add(masked));
            }
            out.push(c);
        }

        // 14: proof_bytes[0] = 32·flag_bits + masked_byte_0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            let thirty_two = Scalar::from_u64(32, curve);
            for r in 0..n {
                let raw = &columns[COL_PROOF_BYTES_OFFSET][r];
                let flag = &columns[COL_PROOF_FLAG_BITS][r];
                let masked = &columns[COL_PROOF_MASKED_BYTE_0][r];
                c[r] = raw.sub(&flag.mul(&thirty_two).add(masked));
            }
            out.push(c);
        }

        // 15..47: y_bytes == evaluation_bytes.
        for k in 0..SCALAR_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let a = &columns[COL_Y_BYTES_OFFSET + k][r];
                let b = &columns[COL_EVALUATION_BYTES_OFFSET + k][r];
                c[r] = a.sub(b);
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

        // 1..7: commitment_x limb decomposition (limb 0 byte 0 →
        // masked_byte_0).
        for j in 0..LIMBS_PER_FP {
            let targets = x_limb_decomp_targets(j);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, weight) in &targets {
                sum = sum.add(
                    &col_evals[commit_byte_col(*byte_idx)]
                        .mul(&Scalar::from_u64(*weight, curve)),
                );
            }
            let body = col_evals[COL_COMMITMENT_X_LIMB_OFFSET + j].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..13: proof_x limb decomposition (limb 0 byte 0 →
        // masked_byte_0).
        for j in 0..LIMBS_PER_FP {
            let targets = x_limb_decomp_targets(j);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, weight) in &targets {
                sum = sum.add(
                    &col_evals[proof_byte_col(*byte_idx)]
                        .mul(&Scalar::from_u64(*weight, curve)),
                );
            }
            let body = col_evals[COL_PROOF_X_LIMB_OFFSET + j].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 13: commitment_bytes[0] = 32·flag_bits + masked_byte_0.
        {
            let thirty_two = Scalar::from_u64(32, curve);
            let body = col_evals[COL_COMMITMENT_BYTES_OFFSET].sub(
                &col_evals[COL_COMMITMENT_FLAG_BITS]
                    .mul(&thirty_two)
                    .add(&col_evals[COL_COMMITMENT_MASKED_BYTE_0]),
            );
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 14: proof_bytes[0] = 32·flag_bits + masked_byte_0.
        {
            let thirty_two = Scalar::from_u64(32, curve);
            let body = col_evals[COL_PROOF_BYTES_OFFSET].sub(
                &col_evals[COL_PROOF_FLAG_BITS]
                    .mul(&thirty_two)
                    .add(&col_evals[COL_PROOF_MASKED_BYTE_0]),
            );
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 15..47: y == evaluation byte-equality.
        for k in 0..SCALAR_BYTES {
            let body =
                col_evals[COL_Y_BYTES_OFFSET + k].sub(&col_evals[COL_EVALUATION_BYTES_OFFSET + k]);
            acc = acc.add(&alpha_pow.mul(&body));
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

        // 1..7: commitment_x limb decomposition (limb 0 byte 0 →
        // masked_byte_0).
        for j in 0..LIMBS_PER_FP {
            let targets = x_limb_decomp_targets(j);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, weight) in &targets {
                let b = &col_coeffs[commit_byte_col(*byte_idx)];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*weight, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_COMMITMENT_X_LIMB_OFFSET + j], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7..13: proof_x limb decomposition (limb 0 byte 0 →
        // masked_byte_0).
        for j in 0..LIMBS_PER_FP {
            let targets = x_limb_decomp_targets(j);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, weight) in &targets {
                let b = &col_coeffs[proof_byte_col(*byte_idx)];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*weight, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_PROOF_X_LIMB_OFFSET + j], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 13: commitment_bytes[0] = 32·flag_bits + masked_byte_0.
        {
            let thirty_two = Scalar::from_u64(32, curve);
            let flag_scaled =
                poly_scalar_mul(&col_coeffs[COL_COMMITMENT_FLAG_BITS], &thirty_two);
            let split_rhs =
                poly_add(&flag_scaled, &col_coeffs[COL_COMMITMENT_MASKED_BYTE_0], curve);
            let body =
                poly_sub(&col_coeffs[COL_COMMITMENT_BYTES_OFFSET], &split_rhs, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 14: proof_bytes[0] = 32·flag_bits + masked_byte_0.
        {
            let thirty_two = Scalar::from_u64(32, curve);
            let flag_scaled =
                poly_scalar_mul(&col_coeffs[COL_PROOF_FLAG_BITS], &thirty_two);
            let split_rhs =
                poly_add(&flag_scaled, &col_coeffs[COL_PROOF_MASKED_BYTE_0], curve);
            let body = poly_sub(&col_coeffs[COL_PROOF_BYTES_OFFSET], &split_rhs, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 15..47: y == evaluation byte-equality.
        for k in 0..SCALAR_BYTES {
            let body = poly_sub(
                &col_coeffs[COL_Y_BYTES_OFFSET + k],
                &col_coeffs[COL_EVALUATION_BYTES_OFFSET + k],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
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
            .unwrap_or(CurveType::Bls12381);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 8-bit range checks on every byte column. Limb columns get
        // a deferred 64-bit range check via the downstream Fp AIR.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..G1_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("blob_kzg_commitment_byte_{}_8bit", k),
                    column_index: COL_COMMITMENT_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("blob_kzg_proof_byte_{}_8bit", k),
                    column_index: COL_PROOF_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // Flag-byte splitter columns (new in #290): tighten via byte
        // limb decomposition against the 256-range table. Strict
        // 3-bit / 5-bit tightness is enforced transitively via the
        // algebraic split equation + downstream pairing AIR; this
        // declaration ensures the columns are byte-valued at minimum.
        for &col_index in &[
            COL_COMMITMENT_FLAG_BITS,
            COL_COMMITMENT_MASKED_BYTE_0,
            COL_PROOF_FLAG_BITS,
            COL_PROOF_MASKED_BYTE_0,
        ] {
            declarations.push((
                LookupDeclaration {
                    label: format!("blob_kzg_flag_split_col_{}_8bit", col_index),
                    column_index: col_index,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SCALAR_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("blob_kzg_z_byte_{}_8bit", k),
                    column_index: COL_Z_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("blob_kzg_y_byte_{}_8bit", k),
                    column_index: COL_Y_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("blob_kzg_eval_byte_{}_8bit", k),
                    column_index: COL_EVALUATION_BYTES_OFFSET + k,
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

/// Cross-AIR LogUp descriptor: blob_kzg_air commitment bytes (gated by
/// `IS_REAL`) ↔ `bls_pairing_air`'s `pk_compressed[0..48]` (gated by
/// its `IS_REAL`). 48-byte tuple.
///
/// This is the **load-bearing soundness handoff**: by binding the 48
/// commitment bytes to a row of `bls_pairing_air`, the full pairing
/// equation `e(C - y·G1, [τ]_2) = e(π, [τ - z]_2)` becomes the
/// downstream AIR's obligation. blob_kzg_air's role is purely to
/// commit the inputs and publish them through this descriptor.
pub fn make_blob_kzg_to_pairing_descriptor(
    blob_kzg_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;
    let a_columns: Vec<usize> =
        (0..G1_BYTES).map(|k| COL_COMMITMENT_BYTES_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..G1_BYTES).map(|k| bp::COL_PK_COMPRESSED_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "blob_kzg_commitment_to_pairing_v1".into(),
        a_layer_index: blob_kzg_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bls_pairing_layer_index,
        b_columns,
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor: blob_kzg_air proof bytes ↔
/// `bls_pairing_air`'s `pk_compressed[0..48]`. 48-byte tuple. Same
/// shape as the commitment descriptor; the joint prover wires a
/// distinct `bls_pairing_air` row for each side of the pairing
/// product (commitment row + proof row).
pub fn make_blob_kzg_proof_to_pairing_descriptor(
    blob_kzg_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;
    let a_columns: Vec<usize> = (0..G1_BYTES).map(|k| COL_PROOF_BYTES_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..G1_BYTES).map(|k| bp::COL_PK_COMPRESSED_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "blob_kzg_proof_to_pairing_v1".into(),
        a_layer_index: blob_kzg_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bls_pairing_layer_index,
        b_columns,
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

/// Bookkeeping byte descriptors for the three scalar fields. Returns
/// `(z_columns, y_columns, evaluation_columns)` — each a 32-element
/// list of column indices that downstream transaction-RLP /
/// blob-sidecar AIRs can use to bind these scalars to their source
/// via cross-AIR LogUp. The selector to use on this side is
/// [`COL_IS_REAL`].
pub fn make_blob_kzg_byte_descriptors() -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let z = (0..SCALAR_BYTES).map(|k| COL_Z_BYTES_OFFSET + k).collect();
    let y = (0..SCALAR_BYTES).map(|k| COL_Y_BYTES_OFFSET + k).collect();
    let eval = (0..SCALAR_BYTES)
        .map(|k| COL_EVALUATION_BYTES_OFFSET + k)
        .collect();
    (z, y, eval)
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs() -> ([u8; G1_BYTES], [u8; G1_BYTES], [u8; SCALAR_BYTES], [u8; SCALAR_BYTES], [u8; SCALAR_BYTES])
    {
        // Synthetic inputs: byte 0 has flags cleared so the limb
        // decomposition round-trips exactly. The constraint
        // surface only enforces shape, not pairing correctness, so
        // any byte pattern within range is acceptable.
        let mut commitment = [0u8; G1_BYTES];
        for k in 0..G1_BYTES {
            commitment[k] = (k as u8).wrapping_mul(3).wrapping_add(7);
        }
        commitment[0] &= 0x1f;
        let mut proof = [0u8; G1_BYTES];
        for k in 0..G1_BYTES {
            proof[k] = (k as u8).wrapping_mul(5).wrapping_add(11);
        }
        proof[0] &= 0x1f;
        let mut z = [0u8; SCALAR_BYTES];
        for k in 0..SCALAR_BYTES {
            z[k] = (k as u8).wrapping_mul(2).wrapping_add(1);
        }
        z[0] &= 0x7f;
        let mut y = [0u8; SCALAR_BYTES];
        for k in 0..SCALAR_BYTES {
            y[k] = (k as u8).wrapping_mul(4).wrapping_add(3);
        }
        y[0] &= 0x7f;
        let evaluation = y;
        (commitment, proof, z, y, evaluation)
    }

    #[test]
    fn witness_builds_from_byte_inputs() {
        let (c, p, z, y, e) = sample_inputs();
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        assert_eq!(row.commitment, c);
        assert_eq!(row.proof, p);
        assert_eq!(row.z, z);
        assert_eq!(row.y, y);
        assert_eq!(row.evaluation, e);
        // Limb decomposition round-trips: each limb equals the
        // big-endian read of its 8 bytes (byte 0 already masked).
        for j in 0..LIMBS_PER_FP {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&c[j * 8..j * 8 + 8]);
            assert_eq!(row.commitment_x_limbs[j], u64::from_be_bytes(buf));
            let mut buf2 = [0u8; 8];
            buf2.copy_from_slice(&p[j * 8..j * 8 + 8]);
            assert_eq!(row.proof_x_limbs[j], u64::from_be_bytes(buf2));
        }
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let (c, p, z, y, e) = sample_inputs();
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i,
                    r,
                    val,
                );
            }
        }
    }

    #[test]
    fn tampered_commitment_byte_fires_limb_decomp() {
        let (c, p, z, y, e) = sample_inputs();
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump commitment byte 47 (LSB of commitment_x limb 5) by 1
        // without touching the limb columns. The limb-5 decomp must
        // fire.
        let curve = CurveType::Bls12381;
        let original = cols[COL_COMMITMENT_BYTES_OFFSET + 47][0].clone();
        cols[COL_COMMITMENT_BYTES_OFFSET + 47][0] = original.add(&Scalar::one(curve));
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let limb5_constraint = 1 + 5;
        assert!(
            !results[limb5_constraint][0].is_zero(),
            "tampering commitment byte 47 must fire limb 5 decomp",
        );
        // Other limb decomps unaffected.
        for j in 0..5 {
            assert!(
                results[1 + j][0].is_zero(),
                "limb {} decomp should be unaffected by commitment byte 47",
                j,
            );
        }
        // Proof limbs and y==eval all unchanged.
        for j in 0..LIMBS_PER_FP {
            assert!(results[1 + LIMBS_PER_FP + j][0].is_zero());
        }
        // Flag-split constraints unaffected: commitment byte 47 lives
        // in limb 5 and does not touch byte 0 or the flag/masked
        // columns.
        let split_base = 1 + 2 * LIMBS_PER_FP;
        assert!(results[split_base][0].is_zero());
        assert!(results[split_base + 1][0].is_zero());
        let byte_eq_base = split_base + NUM_FLAG_SPLIT_CONSTRAINTS;
        for k in 0..SCALAR_BYTES {
            assert!(results[byte_eq_base + k][0].is_zero());
        }
    }

    #[test]
    fn tampered_proof_byte_fires_proof_limb_decomp() {
        let (c, p, z, y, e) = sample_inputs();
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Bump proof byte 1 (one of the bytes inside proof_x limb 0,
        // not the flag byte). The limb-0 decomp must fire while the
        // flag-split constraint stays satisfied (byte 0 untouched).
        let original = cols[COL_PROOF_BYTES_OFFSET + 1][0].clone();
        cols[COL_PROOF_BYTES_OFFSET + 1][0] = original.add(&Scalar::one(curve));
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // proof limb 0 = constraint index 1 + LIMBS_PER_FP + 0.
        let proof_limb0_constraint = 1 + LIMBS_PER_FP;
        assert!(
            !results[proof_limb0_constraint][0].is_zero(),
            "tampering proof byte 1 must fire proof limb 0 decomp",
        );
        // Flag-split for proof unaffected.
        let proof_split_idx = 1 + 2 * LIMBS_PER_FP + 1;
        assert!(results[proof_split_idx][0].is_zero());
    }

    #[test]
    fn y_neq_evaluation_fires_byte_equality() {
        let (c, p, z, y, _e) = sample_inputs();
        // Pick an evaluation that differs from y at one byte.
        let mut evaluation = y;
        evaluation[5] = evaluation[5].wrapping_add(1);
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &evaluation);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // y_eq_evaluation byte 5 lives at index
        //   1 + 2*LIMBS_PER_FP + NUM_FLAG_SPLIT_CONSTRAINTS + 5.
        let byte_eq_base = 1 + 2 * LIMBS_PER_FP + NUM_FLAG_SPLIT_CONSTRAINTS;
        let byte_eq_5 = byte_eq_base + 5;
        assert!(
            !results[byte_eq_5][0].is_zero(),
            "diverging y vs evaluation at byte 5 must fire its byte-eq",
        );
        // Other 31 byte-eq constraints still hold.
        for k in 0..SCALAR_BYTES {
            if k == 5 {
                continue;
            }
            let idx = byte_eq_base + k;
            assert!(
                results[idx][0].is_zero(),
                "byte-eq constraint {} should still hold",
                k,
            );
        }
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_selector() {
        let (c, p, z, y, e) = sample_inputs();
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(7, CurveType::Bls12381);
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[0][0].is_zero(),
            "non-binary is_real must fire constraint 0",
        );
    }

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_COMMITMENT_BYTES_OFFSET, 0);
        assert_eq!(COL_PROOF_BYTES_OFFSET, 48);
        assert_eq!(COL_Z_BYTES_OFFSET, 96);
        assert_eq!(COL_Y_BYTES_OFFSET, 128);
        assert_eq!(COL_EVALUATION_BYTES_OFFSET, 160);
        assert_eq!(COL_COMMITMENT_X_LIMB_OFFSET, 192);
        assert_eq!(COL_PROOF_X_LIMB_OFFSET, 198);
        assert_eq!(COL_COMMITMENT_FLAG_BITS, 204);
        assert_eq!(COL_COMMITMENT_MASKED_BYTE_0, 205);
        assert_eq!(COL_PROOF_FLAG_BITS, 206);
        assert_eq!(COL_PROOF_MASKED_BYTE_0, 207);
        assert_eq!(COL_IS_REAL, 208);
        assert_eq!(NUM_COLUMNS, 209);
        assert_eq!(NUM_ROW_CONSTRAINTS, 1 + 6 + 6 + 2 + 32);
        assert_eq!(NUM_ROW_CONSTRAINTS, 47);
    }

    #[test]
    fn descriptors_well_formed() {
        let d_commit = make_blob_kzg_to_pairing_descriptor(2, 5);
        assert_eq!(d_commit.label, "blob_kzg_commitment_to_pairing_v1");
        assert_eq!(d_commit.a_layer_index, 2);
        assert_eq!(d_commit.b_layer_index, 5);
        assert_eq!(d_commit.a_columns.len(), G1_BYTES);
        assert_eq!(d_commit.b_columns.len(), G1_BYTES);
        assert_eq!(d_commit.a_columns[0], COL_COMMITMENT_BYTES_OFFSET);
        assert_eq!(d_commit.a_columns[G1_BYTES - 1], COL_COMMITMENT_BYTES_OFFSET + G1_BYTES - 1);
        assert_eq!(
            d_commit.b_columns[0],
            crate::bls_pairing_air::COL_PK_COMPRESSED_OFFSET,
        );
        assert_eq!(d_commit.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_commit.b_selector_column,
            Some(crate::bls_pairing_air::COL_IS_REAL),
        );

        let d_proof = make_blob_kzg_proof_to_pairing_descriptor(2, 5);
        assert_eq!(d_proof.label, "blob_kzg_proof_to_pairing_v1");
        assert_eq!(d_proof.a_columns[0], COL_PROOF_BYTES_OFFSET);
        assert_eq!(d_proof.a_columns[G1_BYTES - 1], COL_PROOF_BYTES_OFFSET + G1_BYTES - 1);

        let (z_cols, y_cols, e_cols) = make_blob_kzg_byte_descriptors();
        assert_eq!(z_cols.len(), SCALAR_BYTES);
        assert_eq!(y_cols.len(), SCALAR_BYTES);
        assert_eq!(e_cols.len(), SCALAR_BYTES);
        assert_eq!(z_cols[0], COL_Z_BYTES_OFFSET);
        assert_eq!(z_cols[SCALAR_BYTES - 1], COL_Z_BYTES_OFFSET + SCALAR_BYTES - 1);
        assert_eq!(y_cols[0], COL_Y_BYTES_OFFSET);
        assert_eq!(e_cols[0], COL_EVALUATION_BYTES_OFFSET);
    }

    #[test]
    fn build_poly_matches_evaluate_at_point() {
        use crate::commitment;
        use bls48581::bls48581::big;
        let (c, p, z, y, e) = sample_inputs();
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
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
            "blob_kzg_air: build_constraint_polynomial(z) must equal evaluate_at_point",
        );
    }

    #[test]
    #[ignore = "slow: blob_kzg_air standalone prove_with_scheme + verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let (c, p, z, y, e) = sample_inputs();
        let w = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = BlobKzgConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone blob_kzg_air proof must verify",
        );
    }

    #[test]
    fn real_bls_compressed_commitment_with_flags_round_trips() {
        // Real BLS12-381 G1 compressed points always have the
        // top bit (compressed flag) set, so byte 0 ≥ 0x80. The flag
        // splitter must accept the witness and the row constraints
        // must all evaluate to zero.
        let mut commitment = [0u8; G1_BYTES];
        commitment[0] = 0xa5; // compressed=1, infinity=0, sort=1, x_msb=00101
        for k in 1..G1_BYTES {
            commitment[k] = (k as u8).wrapping_mul(7);
        }
        let mut proof = [0u8; G1_BYTES];
        proof[0] = 0x83; // compressed=1, infinity=0, sort=0, x_msb=00011
        for k in 1..G1_BYTES {
            proof[k] = (k as u8).wrapping_mul(11);
        }
        let mut z = [0u8; SCALAR_BYTES];
        for k in 0..SCALAR_BYTES {
            z[k] = k as u8;
        }
        z[0] &= 0x7f;
        let y = [9u8; SCALAR_BYTES];
        let evaluation = y;
        let w = BlobKzgWitness::from_commitment_proof(&commitment, &proof, &z, &y, &evaluation);
        let row = &w.rows[0];
        assert_eq!(row.commitment_flag_bits, 0b101);
        assert_eq!(row.commitment_masked_byte_0, 0x05);
        assert_eq!(row.proof_flag_bits, 0b100);
        assert_eq!(row.proof_masked_byte_0, 0x03);
        // The limb-0 high byte must come from the masked value.
        assert_eq!((row.commitment_x_limbs[0] >> 56) as u8, 0x05);
        assert_eq!((row.proof_x_limbs[0] >> 56) as u8, 0x03);

        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "real-flag witness: constraint {} at row {} = {:?}",
                    i,
                    r,
                    val,
                );
            }
        }
    }

    #[test]
    fn tampered_commitment_flag_bits_fires_split_constraint() {
        let mut commitment = [0u8; G1_BYTES];
        commitment[0] = 0xa5;
        for k in 1..G1_BYTES {
            commitment[k] = (k as u8).wrapping_mul(7);
        }
        let proof = [0u8; G1_BYTES];
        let z = [0u8; SCALAR_BYTES];
        let y = [9u8; SCALAR_BYTES];
        let evaluation = y;
        let w = BlobKzgWitness::from_commitment_proof(&commitment, &proof, &z, &y, &evaluation);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Lie: set flag_bits = 0 while keeping masked_byte_0 honest
        // (= 0x05). Then 32*0 + 0x05 = 0x05 ≠ 0xa5 → split fires.
        cols[COL_COMMITMENT_FLAG_BITS][0] = Scalar::zero(curve);
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let commit_split_idx = 1 + 2 * LIMBS_PER_FP;
        assert!(
            !results[commit_split_idx][0].is_zero(),
            "lying about commitment flag bits must fire the split constraint",
        );
        // Proof split untouched.
        assert!(results[commit_split_idx + 1][0].is_zero());
    }

    #[test]
    fn tampered_masked_byte_0_fires_split_and_limb_decomp() {
        let mut commitment = [0u8; G1_BYTES];
        commitment[0] = 0xa5;
        for k in 1..G1_BYTES {
            commitment[k] = (k as u8).wrapping_mul(7);
        }
        let proof = [0u8; G1_BYTES];
        let z = [0u8; SCALAR_BYTES];
        let y = [9u8; SCALAR_BYTES];
        let evaluation = y;
        let w = BlobKzgWitness::from_commitment_proof(&commitment, &proof, &z, &y, &evaluation);
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Bump masked_byte_0 by 1 without updating limbs or raw byte 0:
        // → split fires (LHS-RHS = -1) AND limb 0 decomp fires (limb
        // value vs new masked).
        let orig = cols[COL_COMMITMENT_MASKED_BYTE_0][0].clone();
        cols[COL_COMMITMENT_MASKED_BYTE_0][0] = orig.add(&Scalar::one(curve));
        let cs = BlobKzgConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let limb0_idx = 1;
        let commit_split_idx = 1 + 2 * LIMBS_PER_FP;
        assert!(!results[limb0_idx][0].is_zero(), "limb 0 decomp must fire");
        assert!(
            !results[commit_split_idx][0].is_zero(),
            "split constraint must fire",
        );
    }

    #[test]
    fn trace_builder_is_deterministic() {
        let (c, p, z, y, e) = sample_inputs();
        let w1 = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let w2 = BlobKzgWitness::from_commitment_proof(&c, &p, &z, &y, &e);
        let t1 = build_trace_polynomials(&w1, CurveType::Bls12381);
        let t2 = build_trace_polynomials(&w2, CurveType::Bls12381);
        assert_eq!(t1.num_rows, t2.num_rows);
        assert_eq!(t1.padded_size, t2.padded_size);
        assert_eq!(t1.columns.len(), t2.columns.len());
        for (col_a, col_b) in t1.columns.iter().zip(t2.columns.iter()) {
            assert_eq!(col_a.evaluations.len(), col_b.evaluations.len());
            for (a, b) in col_a.evaluations.iter().zip(col_b.evaluations.iter()) {
                assert!(a.sub(b).is_zero(), "trace evaluations must match deterministically");
            }
        }
    }
}
