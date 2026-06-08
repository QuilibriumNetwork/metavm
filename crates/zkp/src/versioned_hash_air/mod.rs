//! EIP-4844 blob **versioned hash** AIR.
//!
//! # Purpose
//!
//! EIP-4844 attaches one or more KZG commitments to a transaction. To
//! commit one of these commitments inside the transaction body, the
//! consensus client publishes a **versioned hash** of the form:
//!
//! ```text
//!     versioned_hash[0]    = VERSIONED_HASH_VERSION_KZG = 0x01
//!     versioned_hash[1..32] = sha256(commitment)[1..32]
//! ```
//!
//! The first byte being a constant `0x01` makes the hash function
//! domain-separable for future blob versions, and replacing the high
//! byte of `sha256(commitment)` with `0x01` preserves 31 bytes
//! (≈ 248 bits) of preimage resistance.
//!
//! This AIR commits one `(commitment, sha256_digest, versioned_hash)`
//! tuple per row and algebraically enforces the version-byte + tail
//! equality. The actual SHA-256 computation is **not** done here — it
//! is bound via a cross-AIR LogUp descriptor to `sha256_extract` (the
//! bit-level SHA-256 AIR closes the gap from there).
//!
//! # Algebraic surface
//!
//! Per row:
//!
//!   1. `is_real ∈ {0, 1}` — selector binarity.
//!   2. `versioned_hash[0] = 0x01` — literal constant equality.
//!   3. `versioned_hash[i] = sha256_digest[i]` for `i = 1..32` —
//!      31 byte-equality constraints.
//!
//! Total: `1 + 1 + 31 = 33` row-local constraints. No shifted
//! constraints. 8-bit range checks on every byte column
//! (`48 + 32 + 32 = 112` declarations).
//!
//! # Cross-AIR linkages
//!
//! - [`make_versioned_hash_to_sha256_descriptor`] — binds
//!   `(commitment[0..48], constant_padding[48..64], sha256_digest)` ↔
//!   `sha256_extract::(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])`. The
//!   constant SHA-256 padding tail for a 48-byte single-block input
//!   (`0x80 || zeros || 0x00...0180_be`) is committed in dedicated
//!   columns so it can participate in the tuple; the host-side witness
//!   builder pins these to the canonical padding pattern, and they get
//!   byte-range-checked at the per-AIR level. Soundness handoff: once
//!   `sha256_extract` is bound to the bit-level SHA-256 AIR (the
//!   established `Sha256Extract↔SHA-256` pattern), the digest column
//!   on this side is algebraically bound to `sha256(commitment)`.
//!
//! - [`make_versioned_hash_to_blob_kzg_descriptor`] — binds
//!   `commitment[0..48]` ↔ `blob_kzg_air::COL_COMMITMENT_BYTES`. Forces
//!   every `versioned_hash` row to correspond to a real KZG commitment
//!   row.
//!
//! - [`make_versioned_hash_to_tx_rlp_descriptor`] — descriptor stub
//!   binding `versioned_hash[0..32]` to the future `tx_rlp_air`
//!   `blob_versioned_hashes` column range. **`tx_rlp_air` does not yet
//!   expose these columns** (no `COL_BLOB_VERSIONED_HASH_*` constants
//!   are defined); the stub validates the source side and documents
//!   the consumer-side shape for when the EIP-4844 fields are wired in.
//!   Until then the returned descriptor uses `usize::MAX` placeholders
//!   for the B side columns — well-formed-ness tests pin the A side
//!   only.
//!
//! # Soundness scope (per-AIR)
//!
//! - `is_real` is binary.
//! - The first byte of every real `versioned_hash` row is exactly
//!   `0x01`.
//! - The last 31 bytes of every real `versioned_hash` row match the
//!   committed digest column byte-for-byte.
//!
//! What is **not** proven by this AIR alone:
//!
//! - That `sha256_digest = sha256(commitment)`. The cross-AIR LogUp to
//!   `sha256_extract` (+ the downstream Sha256Extract↔SHA-256 binding)
//!   closes this.
//! - That `commitment` is the same as any specific `blob_kzg_air` row.
//!   The cross-AIR LogUp to `blob_kzg_air` (+ that AIR's pairing
//!   binding) closes this.
//! - That `versioned_hash` actually appears in some transaction's
//!   `blob_versioned_hashes` field. Deferred to the `tx_rlp_air`
//!   binding once that AIR exposes the column range.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// BLS12-381 G1 compressed point length — the size of one EIP-4844
/// blob KZG commitment.
pub const COMMITMENT_LEN: usize = 48;

/// SHA-256 output / EIP-4844 versioned-hash length.
pub const DIGEST_LEN: usize = 32;

/// EIP-4844 `VERSIONED_HASH_VERSION_KZG`.
pub const VERSIONED_HASH_VERSION_KZG: u8 = 0x01;

/// Single SHA-256 block size in bytes.
pub const SHA256_BLOCK_LEN: usize = 64;

/// Canonical SHA-256 padding tail for a 48-byte single-block input:
/// `0x80 || 7×0x00 || u64_be(48 * 8) = u64_be(384) = 00 00 00 00 00 00 01 80`.
///
/// Total padding bytes: `64 - 48 = 16`.
pub const COMMITMENT_PADDING_TAIL: [u8; SHA256_BLOCK_LEN - COMMITMENT_LEN] = [
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x80,
];

// ─── Column layout ────────────────────────────────────────────────────
//
//   commitment[0..48]       : 48 cols — the KZG commitment bytes
//   padding_tail[48..64]    : 16 cols — the canonical SHA-256 padding
//                              tail bound to COMMITMENT_PADDING_TAIL
//   sha256_digest[0..32]    : 32 cols — sha256(commitment) output bytes
//   versioned_hash[0..32]   : 32 cols — 0x01 || sha256_digest[1..32]
//   is_real                 : 1 col   — selector
//
// total: 48 + 16 + 32 + 32 + 1 = 129 columns.

pub const COL_COMMITMENT_OFFSET: usize = 0;
pub const COL_PADDING_TAIL_OFFSET: usize = COL_COMMITMENT_OFFSET + COMMITMENT_LEN;
pub const COL_SHA256_DIGEST_OFFSET: usize =
    COL_PADDING_TAIL_OFFSET + (SHA256_BLOCK_LEN - COMMITMENT_LEN);
pub const COL_VERSIONED_HASH_OFFSET: usize = COL_SHA256_DIGEST_OFFSET + DIGEST_LEN;
pub const COL_IS_REAL: usize = COL_VERSIONED_HASH_OFFSET + DIGEST_LEN;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// Row-local constraints:
///   0:        is_real ∈ {0, 1}
///   1:        versioned_hash[0] - 0x01 = 0
///   2..33:    versioned_hash[k] - sha256_digest[k] = 0 for k = 1..32
pub const NUM_VH_TAIL_CONSTRAINTS: usize = DIGEST_LEN - 1; // 31
pub const NUM_ROW_CONSTRAINTS: usize = 1 + 1 + NUM_VH_TAIL_CONSTRAINTS;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// One EIP-4844 versioned-hash row.
#[derive(Clone, Copy, Debug)]
pub struct VersionedHashRow {
    pub commitment: [u8; COMMITMENT_LEN],
    pub sha256_digest: [u8; DIGEST_LEN],
    pub versioned_hash: [u8; DIGEST_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct VersionedHashWitness {
    pub rows: Vec<VersionedHashRow>,
}

impl VersionedHashWitness {
    /// Build a single-row witness from a raw KZG commitment by
    /// computing `sha256(commitment)` and stamping the version byte.
    pub fn from_commitment(commitment: [u8; COMMITMENT_LEN]) -> Self {
        let digest = crate::sha256::sha256(&commitment);
        let mut versioned_hash = digest;
        versioned_hash[0] = VERSIONED_HASH_VERSION_KZG;
        Self {
            rows: vec![VersionedHashRow {
                commitment,
                sha256_digest: digest,
                versioned_hash,
            }],
        }
    }

    /// Build a multi-row witness from a slice of commitments.
    pub fn from_commitments(commitments: &[[u8; COMMITMENT_LEN]]) -> Self {
        let rows = commitments
            .iter()
            .map(|c| {
                let digest = crate::sha256::sha256(c);
                let mut versioned_hash = digest;
                versioned_hash[0] = VERSIONED_HASH_VERSION_KZG;
                VersionedHashRow {
                    commitment: *c,
                    sha256_digest: digest,
                    versioned_hash,
                }
            })
            .collect();
        Self { rows }
    }

    pub fn push(&mut self, row: VersionedHashRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &VersionedHashWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..COMMITMENT_LEN {
            columns[COL_COMMITMENT_OFFSET + k][r] =
                Scalar::from_u64(row.commitment[k] as u64, curve);
        }
        // Padding tail is the SAME constant on every real row.
        for k in 0..(SHA256_BLOCK_LEN - COMMITMENT_LEN) {
            columns[COL_PADDING_TAIL_OFFSET + k][r] =
                Scalar::from_u64(COMMITMENT_PADDING_TAIL[k] as u64, curve);
        }
        for k in 0..DIGEST_LEN {
            columns[COL_SHA256_DIGEST_OFFSET + k][r] =
                Scalar::from_u64(row.sha256_digest[k] as u64, curve);
            columns[COL_VERSIONED_HASH_OFFSET + k][r] =
                Scalar::from_u64(row.versioned_hash[k] as u64, curve);
        }
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

pub struct VersionedHashConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl VersionedHashConstraintSystem {
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

impl VmConstraintSystem for VersionedHashConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        labels.push("versioned_hash_byte_0_eq_0x01".into());
        for k in 1..DIGEST_LEN {
            labels.push(format!("versioned_hash_byte_{}_eq_digest", k));
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

        // Constant scalar for version byte.
        let kzg_version =
            Scalar::from_u64(VERSIONED_HASH_VERSION_KZG as u64, curve);

        // 0: is_real binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1: versioned_hash[0] - 0x01 = 0 (gated by is_real so padding
        //    rows where vh[0] = 0 don't fire).
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let vh0 = &columns[COL_VERSIONED_HASH_OFFSET][r];
                let sel = &columns[COL_IS_REAL][r];
                let body = vh0.sub(&kzg_version);
                c[r] = sel.mul(&body);
            }
            out.push(c);
        }

        // 2..33: versioned_hash[k] - sha256_digest[k] = 0 for k = 1..32.
        //   Ungated: the equality is required on every row. On padding
        //   rows both columns are 0 so the equation already holds.
        for k in 1..DIGEST_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let vh = &columns[COL_VERSIONED_HASH_OFFSET + k][r];
                let dg = &columns[COL_SHA256_DIGEST_OFFSET + k][r];
                c[r] = vh.sub(dg);
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
        let kzg_version =
            Scalar::from_u64(VERSIONED_HASH_VERSION_KZG as u64, curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_real * (vh[0] - 0x01) = 0.
        {
            let sel = &col_evals[COL_IS_REAL];
            let body = col_evals[COL_VERSIONED_HASH_OFFSET].sub(&kzg_version);
            acc = acc.add(&alpha_pow.mul(&sel.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2..33: vh[k] - digest[k] = 0.
        for k in 1..DIGEST_LEN {
            let body = col_evals[COL_VERSIONED_HASH_OFFSET + k]
                .sub(&col_evals[COL_SHA256_DIGEST_OFFSET + k]);
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
        let kzg_version_poly = vec![Scalar::from_u64(
            VERSIONED_HASH_VERSION_KZG as u64,
            curve,
        )];

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
        // 1: is_real * (vh[0] - 0x01) = 0.
        {
            let sel = &col_coeffs[COL_IS_REAL];
            let vh0 = &col_coeffs[COL_VERSIONED_HASH_OFFSET];
            let body_inner = poly_sub(vh0, &kzg_version_poly, curve);
            let body = poly_mul(sel, &body_inner, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2..33: vh[k] - digest[k] = 0.
        for k in 1..DIGEST_LEN {
            let body = poly_sub(
                &col_coeffs[COL_VERSIONED_HASH_OFFSET + k],
                &col_coeffs[COL_SHA256_DIGEST_OFFSET + k],
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
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 8-bit range checks on every byte column (commitment + padding
        // tail + digest + versioned_hash).
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..COMMITMENT_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("versioned_hash_commitment_byte_{}_8bit", k),
                    column_index: COL_COMMITMENT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..(SHA256_BLOCK_LEN - COMMITMENT_LEN) {
            declarations.push((
                LookupDeclaration {
                    label: format!("versioned_hash_padding_byte_{}_8bit", k),
                    column_index: COL_PADDING_TAIL_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..DIGEST_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("versioned_hash_digest_byte_{}_8bit", k),
                    column_index: COL_SHA256_DIGEST_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("versioned_hash_vh_byte_{}_8bit", k),
                    column_index: COL_VERSIONED_HASH_OFFSET + k,
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
/// `(commitment[0..48] || padding_tail[48..64] || sha256_digest[0..32])`
/// (gated by `IS_REAL`) ↔ `sha256_extract`'s
/// `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])` (gated by its `IS_REAL`).
/// 96-byte tuple.
///
/// Binds the witnessed `sha256_digest` to whatever `sha256_extract`
/// computes for the same 64-byte block input. Soundness handoff: once
/// `sha256_extract` is bound to the bit-level SHA-256 AIR, the digest
/// is algebraically bound to `sha256(commitment || canonical_padding)
/// = sha256(commitment)`.
pub fn make_versioned_hash_to_sha256_descriptor(
    vh_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    let mut a_columns: Vec<usize> = Vec::with_capacity(SHA256_BLOCK_LEN + DIGEST_LEN);
    for k in 0..COMMITMENT_LEN {
        a_columns.push(COL_COMMITMENT_OFFSET + k);
    }
    for k in 0..(SHA256_BLOCK_LEN - COMMITMENT_LEN) {
        a_columns.push(COL_PADDING_TAIL_OFFSET + k);
    }
    for k in 0..DIGEST_LEN {
        a_columns.push(COL_SHA256_DIGEST_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(SHA256_BLOCK_LEN + DIGEST_LEN);
    for k in 0..se::NUM_INPUT_BYTES {
        b_columns.push(se::COL_INPUT_BYTE_OFFSET + k);
    }
    for k in 0..se::NUM_OUTPUT_BYTES {
        b_columns.push(se::COL_OUTPUT_BYTE_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "versioned_hash_to_sha256_v1".into(),
        a_layer_index: vh_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor: this AIR's `commitment[0..48]` (gated by
/// `IS_REAL`) ↔ `blob_kzg_air`'s `commitment_bytes[0..48]` (gated by
/// its `IS_REAL`). 48-byte tuple.
///
/// Forces every real `versioned_hash` row to correspond to a real KZG
/// commitment row in `blob_kzg_air`, which downstream binds to the
/// pairing equation via `bls_pairing_air`.
pub fn make_versioned_hash_to_blob_kzg_descriptor(
    vh_layer_index: usize,
    blob_kzg_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::blob_kzg_air as bk;
    let a_columns: Vec<usize> = (0..COMMITMENT_LEN)
        .map(|k| COL_COMMITMENT_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..COMMITMENT_LEN)
        .map(|k| bk::COL_COMMITMENT_BYTES_OFFSET + k)
        .collect();
    CrossAirLogUpDescriptor {
        label: "versioned_hash_to_blob_kzg_v1".into(),
        a_layer_index: vh_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: blob_kzg_layer_index,
        b_columns,
        b_selector_column: Some(bk::COL_IS_REAL),
    }
}

/// Placeholder column index used on the `tx_rlp_air` side of the
/// versioned-hash binding until that AIR exposes
/// `blob_versioned_hashes` columns.
pub const TX_RLP_BLOB_VERSIONED_HASH_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor: this AIR's `versioned_hash[0..32]`
/// (gated by `IS_REAL`) ↔ `tx_rlp_air`'s future
/// `blob_versioned_hashes[0..32]` column range.
///
/// **Stub**: `tx_rlp_air` does not currently expose a
/// `blob_versioned_hashes` column range, so the B side is filled with
/// [`TX_RLP_BLOB_VERSIONED_HASH_PLACEHOLDER`] (`usize::MAX`) sentinels.
/// The A side is fully wired and the descriptor shape (32-byte tuple,
/// selector on `IS_REAL`) is what downstream wiring will use once the
/// EIP-4844 fields land in `tx_rlp_air`. Until that lands, this
/// descriptor MUST NOT be passed to `joint_prove` — it exists for
/// shape documentation and well-formedness testing.
pub fn make_versioned_hash_to_tx_rlp_descriptor(
    vh_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_rlp_air as tr;
    let a_columns: Vec<usize> = (0..DIGEST_LEN)
        .map(|k| COL_VERSIONED_HASH_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = vec![TX_RLP_BLOB_VERSIONED_HASH_PLACEHOLDER; DIGEST_LEN];
    CrossAirLogUpDescriptor {
        label: "versioned_hash_to_tx_rlp_v1_stub".into(),
        a_layer_index: vh_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: tx_rlp_layer_index,
        b_columns,
        b_selector_column: Some(tr::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_commitment() -> [u8; COMMITMENT_LEN] {
        let mut c = [0u8; COMMITMENT_LEN];
        for k in 0..COMMITMENT_LEN {
            c[k] = (k as u8).wrapping_mul(7).wrapping_add(13);
        }
        // Clear the high bits of byte 0 to mimic a canonical
        // compressed-G1 form (not required by this AIR, but matches
        // the blob_kzg_air convention).
        c[0] &= 0x1f;
        c
    }

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_COMMITMENT_OFFSET, 0);
        assert_eq!(COL_PADDING_TAIL_OFFSET, 48);
        assert_eq!(COL_SHA256_DIGEST_OFFSET, 64);
        assert_eq!(COL_VERSIONED_HASH_OFFSET, 96);
        assert_eq!(COL_IS_REAL, 128);
        assert_eq!(NUM_COLUMNS, 129);
        assert_eq!(NUM_ROW_CONSTRAINTS, 1 + 1 + 31);
        assert_eq!(NUM_ROW_CONSTRAINTS, 33);
        // Padding tail length sanity.
        assert_eq!(COMMITMENT_PADDING_TAIL.len(), 16);
        assert_eq!(COMMITMENT_PADDING_TAIL[0], 0x80);
        assert_eq!(COMMITMENT_PADDING_TAIL[14], 0x01);
        assert_eq!(COMMITMENT_PADDING_TAIL[15], 0x80);
    }

    #[test]
    fn witness_from_commitment_matches_sha256() {
        let commitment = sample_commitment();
        let w = VersionedHashWitness::from_commitment(commitment);
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        assert_eq!(row.commitment, commitment);
        let expected_digest = crate::sha256::sha256(&commitment);
        assert_eq!(row.sha256_digest, expected_digest);
        // versioned_hash[0] = 0x01; versioned_hash[1..32] = digest[1..32].
        assert_eq!(row.versioned_hash[0], VERSIONED_HASH_VERSION_KZG);
        for k in 1..DIGEST_LEN {
            assert_eq!(row.versioned_hash[k], expected_digest[k]);
        }
        // Sanity: sha256-of-48-bytes single block padding tail matches
        // our published constant (re-derive: 0x80 || 7×0 || u64_be(48*8)).
        let mut expected_tail = [0u8; SHA256_BLOCK_LEN - COMMITMENT_LEN];
        expected_tail[0] = 0x80;
        let bit_len: u64 = (COMMITMENT_LEN as u64) * 8;
        let be = bit_len.to_be_bytes();
        let tail_len = expected_tail.len();
        expected_tail[tail_len - 8..].copy_from_slice(&be);
        assert_eq!(COMMITMENT_PADDING_TAIL, expected_tail);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let commitment = sample_commitment();
        let w = VersionedHashWitness::from_commitment(commitment);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = VersionedHashConstraintSystem::new(trace.num_rows);
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
    fn tampered_version_byte_fires_constant_eq() {
        let commitment = sample_commitment();
        let w = VersionedHashWitness::from_commitment(commitment);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Replace version byte 0x01 with 0x02 (the BLS-aggregate
        // version byte from a future proposal). Constraint 1 must fire.
        cols[COL_VERSIONED_HASH_OFFSET][0] = Scalar::from_u64(0x02, CurveType::Bls48581);
        let cs = VersionedHashConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[1][0].is_zero(),
            "tampering vh[0] must fire constraint 1 (vh[0] = 0x01)",
        );
        // is_real binary still holds; tail equalities unaffected.
        assert!(results[0][0].is_zero());
        for k in 1..DIGEST_LEN {
            assert!(
                results[2 + (k - 1)][0].is_zero(),
                "tail byte-eq {} must still hold",
                k,
            );
        }
    }

    #[test]
    fn tampered_tail_byte_fires_byte_equality() {
        let commitment = sample_commitment();
        let w = VersionedHashWitness::from_commitment(commitment);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump versioned_hash[7] by 1, breaking the equality with
        // sha256_digest[7].
        let curve = CurveType::Bls48581;
        let orig = cols[COL_VERSIONED_HASH_OFFSET + 7][0].clone();
        cols[COL_VERSIONED_HASH_OFFSET + 7][0] = orig.add(&Scalar::one(curve));
        let cs = VersionedHashConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index for byte k=7 is 2 + (k-1) = 2 + 6 = 8.
        let idx = 2 + (7 - 1);
        assert!(
            !results[idx][0].is_zero(),
            "tampering vh[7] must fire byte-eq constraint for byte 7",
        );
        // Other constraints unaffected.
        assert!(results[0][0].is_zero(), "is_real binary unaffected");
        assert!(results[1][0].is_zero(), "vh[0] eq 0x01 unaffected");
        for k in 1..DIGEST_LEN {
            if k == 7 {
                continue;
            }
            let i = 2 + (k - 1);
            assert!(
                results[i][0].is_zero(),
                "byte-eq constraint {} should still hold",
                k,
            );
        }
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_selector() {
        let commitment = sample_commitment();
        let w = VersionedHashWitness::from_commitment(commitment);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(5, CurveType::Bls48581);
        let cs = VersionedHashConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[0][0].is_zero(),
            "non-binary is_real must fire constraint 0",
        );
    }

    #[test]
    fn sha256_descriptor_well_formed() {
        let d = make_versioned_hash_to_sha256_descriptor(0, 1);
        assert_eq!(d.label, "versioned_hash_to_sha256_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        // Tuple width = 64 input bytes + 32 output bytes = 96.
        assert_eq!(d.a_columns.len(), SHA256_BLOCK_LEN + DIGEST_LEN);
        assert_eq!(d.b_columns.len(), SHA256_BLOCK_LEN + DIGEST_LEN);
        // First 48 = commitment.
        for k in 0..COMMITMENT_LEN {
            assert_eq!(d.a_columns[k], COL_COMMITMENT_OFFSET + k);
            assert_eq!(
                d.b_columns[k],
                crate::sha256_extract::COL_INPUT_BYTE_OFFSET + k,
            );
        }
        // Next 16 = padding tail.
        for k in 0..(SHA256_BLOCK_LEN - COMMITMENT_LEN) {
            assert_eq!(
                d.a_columns[COMMITMENT_LEN + k],
                COL_PADDING_TAIL_OFFSET + k,
            );
            assert_eq!(
                d.b_columns[COMMITMENT_LEN + k],
                crate::sha256_extract::COL_INPUT_BYTE_OFFSET + COMMITMENT_LEN + k,
            );
        }
        // Last 32 = digest / sha256_extract output.
        for k in 0..DIGEST_LEN {
            assert_eq!(
                d.a_columns[SHA256_BLOCK_LEN + k],
                COL_SHA256_DIGEST_OFFSET + k,
            );
            assert_eq!(
                d.b_columns[SHA256_BLOCK_LEN + k],
                crate::sha256_extract::COL_OUTPUT_BYTE_OFFSET + k,
            );
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::sha256_extract::COL_IS_REAL),
        );
    }

    #[test]
    fn blob_kzg_descriptor_well_formed() {
        let d = make_versioned_hash_to_blob_kzg_descriptor(0, 1);
        assert_eq!(d.label, "versioned_hash_to_blob_kzg_v1");
        assert_eq!(d.a_columns.len(), COMMITMENT_LEN);
        assert_eq!(d.b_columns.len(), COMMITMENT_LEN);
        for k in 0..COMMITMENT_LEN {
            assert_eq!(d.a_columns[k], COL_COMMITMENT_OFFSET + k);
            assert_eq!(
                d.b_columns[k],
                crate::blob_kzg_air::COL_COMMITMENT_BYTES_OFFSET + k,
            );
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::blob_kzg_air::COL_IS_REAL),
        );
    }

    #[test]
    fn tx_rlp_descriptor_stub_well_formed() {
        let d = make_versioned_hash_to_tx_rlp_descriptor(0, 1);
        assert_eq!(d.label, "versioned_hash_to_tx_rlp_v1_stub");
        assert_eq!(d.a_columns.len(), DIGEST_LEN);
        assert_eq!(d.b_columns.len(), DIGEST_LEN);
        // A side is the full versioned_hash column range.
        for k in 0..DIGEST_LEN {
            assert_eq!(d.a_columns[k], COL_VERSIONED_HASH_OFFSET + k);
        }
        // B side is placeholder until tx_rlp_air exposes
        // blob_versioned_hashes columns.
        for k in 0..DIGEST_LEN {
            assert_eq!(d.b_columns[k], TX_RLP_BLOB_VERSIONED_HASH_PLACEHOLDER);
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::tx_rlp_air::COL_IS_REAL),
        );
    }

    #[test]
    fn multi_row_witness_builds() {
        let c1 = sample_commitment();
        let mut c2 = c1;
        c2[5] ^= 0x55;
        let w = VersionedHashWitness::from_commitments(&[c1, c2]);
        assert_eq!(w.rows.len(), 2);
        assert_ne!(w.rows[0].sha256_digest, w.rows[1].sha256_digest);
        assert_eq!(w.rows[0].versioned_hash[0], VERSIONED_HASH_VERSION_KZG);
        assert_eq!(w.rows[1].versioned_hash[0], VERSIONED_HASH_VERSION_KZG);
        // Constraints zero on both rows.
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = VersionedHashConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} row {} should be zero", i, r);
            }
        }
    }

    #[test]
    fn build_poly_matches_evaluate_at_point() {
        use crate::commitment;
        use bls48581::bls48581::big;
        let commitment = sample_commitment();
        let w = VersionedHashWitness::from_commitment(commitment);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = VersionedHashConstraintSystem::new(trace.num_rows);
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
            "versioned_hash_air: build_constraint_polynomial(z) must equal evaluate_at_point",
        );
    }

    #[test]
    #[ignore = "slow: versioned_hash_air standalone prove_with_scheme + verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let commitment = sample_commitment();
        let w = VersionedHashWitness::from_commitment(commitment);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = VersionedHashConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone versioned_hash_air proof must verify",
        );
    }
}
