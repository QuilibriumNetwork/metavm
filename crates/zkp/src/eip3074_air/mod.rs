//! EIP-3074 AUTH / AUTHCALL AIR.
//!
//! EIP-3074 introduces two new opcodes:
//!
//! * `AUTH (0xF6)` — Verifies a secp256k1 ECDSA signature over
//!
//!   ```text
//!   auth_msg = keccak256( MAGIC || chainId || nonce || invoker || commit )
//!   ```
//!
//!   where
//!     - `MAGIC = 0x04` (single domain-separator byte),
//!     - `chainId` is the 32-byte big-endian active chain id,
//!     - `nonce` is the 32-byte big-endian authorized account nonce,
//!     - `invoker` is the address that called `AUTH`, zero-padded
//!       on the left to 32 bytes,
//!     - `commit` is a 32-byte free-form application-defined commitment.
//!
//!   The recovered signer (`ecrecover(auth_msg, signature)`) becomes
//!   the row's `authorized_address`. Total preimage length is
//!   `1 + 32 + 32 + 32 + 32 = 129` bytes.
//!
//! * `AUTHCALL (0xF7)` — Performs a `CALL` whose `msg.sender` is the
//!   previously-authorized address rather than the caller. This AIR
//!   does NOT bind the AUTHCALL state-transition itself; it only
//!   commits the per-AUTH witness data + cross-AIR linkages that an
//!   AUTHCALL-aware downstream gadget needs.
//!
//! ## Per-AUTH witness
//!
//! * `authorized_address[0..20]` — recovered signer (claimed equal
//!   to `ecrecover(auth_msg_hash, signature)`).
//! * `invoker[0..20]`            — caller of AUTH (the contract
//!   verifying delegations on its behalf).
//! * `commit[0..32]`             — application-supplied 32-byte
//!   commitment digest.
//! * `nonce`                     — u64 expected nonce of
//!   `authorized_address`.
//! * `chain_id`                  — u64 currently active chain id.
//! * `signature[0..65]`          — `r (32) || s (32) || y_parity (1)`.
//! * `auth_msg_hash[0..32]`      — `keccak256(preimage)`; matches
//!   the `msg_hash` of the cross-AIR secp256k1_recovery row, and the
//!   keccak `output` of the cross-AIR keccak_extract row that owns
//!   the 129-byte preimage.
//! * `is_active`                 — host-side flag; 1 iff AUTH
//!   succeeded (signature verified, recovered address pinned).
//!
//! ## Constraints
//!
//! 0. `is_real_binary`        — `is_real · (is_real − 1) = 0`.
//! 1. `is_active_binary`      — `is_active · (is_active − 1) = 0`.
//! 2. `is_active_gates_real`  — `is_active · (1 − is_real) = 0`.
//! 3. `y_parity_binary`       — last byte of `signature` is `0` or `1`.
//! 4. `chain_id_le_decomp`    — `is_real · (chain_id − Σ_b CHAIN_ID_BYTE[b] · 2^(8b)) = 0`.
//! 5. `nonce_le_decomp`       — `is_real · (nonce − Σ_b NONCE_BYTE[b] · 2^(8b)) = 0`.
//!
//! All address / commit / signature / hash bytes are 8-bit
//! range-checked via lookup declarations.
//!
//! ## Cross-AIR linkages
//!
//! * `(auth_msg_hash, r, s, y_parity, authorized_address)` of this
//!   AIR ↔ `(msg_hash, r, s, v, recovered_address)` of
//!   [`crate::secp256k1_recovery`] — algebraically seals the signature
//!   verification. Gated by `is_active`.
//! * `auth_msg_hash` of this AIR ↔ keccak `output` of
//!   [`crate::keccak_extract`] over the canonical 129-byte preimage
//!   `MAGIC || chainId || nonce || invoker || commit`. Gated by
//!   `is_active`.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// EIP-3074 domain-separator magic byte (`AUTH` preimage prefix).
pub const EIP3074_MAGIC: u8 = 0x04;

pub const ADDR_LEN: usize = 20;
pub const SIG_LEN: usize = 65;
pub const COMMIT_LEN: usize = 32;
pub const HASH_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

/// Canonical AUTH preimage length: `MAGIC (1) || chainId (32) ||
/// nonce (32) || invoker (32, left-padded) || commit (32)`.
pub const AUTH_PREIMAGE_LEN: usize = 1 + 32 + 32 + 32 + 32;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_AUTHORIZED_ADDR_OFFSET: usize = 0; // 0..20
pub const COL_INVOKER_OFFSET: usize = COL_AUTHORIZED_ADDR_OFFSET + ADDR_LEN; // 20..40
pub const COL_COMMIT_OFFSET: usize = COL_INVOKER_OFFSET + ADDR_LEN; // 40..72
pub const COL_SIGNATURE_OFFSET: usize = COL_COMMIT_OFFSET + COMMIT_LEN; // 72..137
pub const COL_AUTH_MSG_HASH_OFFSET: usize = COL_SIGNATURE_OFFSET + SIG_LEN; // 137..169

pub const COL_CHAIN_ID: usize = COL_AUTH_MSG_HASH_OFFSET + HASH_LEN; // 169
pub const COL_CHAIN_ID_BYTE_OFFSET: usize = COL_CHAIN_ID + 1; // 170..178
pub const COL_NONCE: usize = COL_CHAIN_ID_BYTE_OFFSET + U64_BYTES; // 178
pub const COL_NONCE_BYTE_OFFSET: usize = COL_NONCE + 1; // 179..187

pub const COL_IS_ACTIVE: usize = COL_NONCE_BYTE_OFFSET + U64_BYTES; // 187
pub const COL_IS_REAL: usize = COL_IS_ACTIVE + 1; // 188

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 189
pub const NUM_ROW_CONSTRAINTS: usize = 6;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Eip3074AuthRow {
    pub authorized_address: [u8; ADDR_LEN],
    pub invoker: [u8; ADDR_LEN],
    pub commit: [u8; COMMIT_LEN],
    pub nonce: u64,
    pub chain_id: u64,
    pub signature: [u8; SIG_LEN],
    pub auth_msg_hash: [u8; HASH_LEN],
    pub is_active: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Eip3074AuthWitness {
    pub rows: Vec<Eip3074AuthRow>,
}

impl Eip3074AuthWitness {
    pub fn from_rows(rows: Vec<Eip3074AuthRow>) -> Self {
        Self { rows }
    }
}

/// Compute the canonical EIP-3074 AUTH preimage:
/// `MAGIC || chainId (32 BE) || nonce (32 BE) || invoker (32, left-padded) || commit (32)`.
///
/// Host-side helper; the AIR commits the resulting `auth_msg_hash`
/// and binds it to keccak_extract via cross-AIR LogUp rather than
/// recomputing the preimage algebraically.
pub fn auth_message_preimage(
    chain_id: u64,
    nonce: u64,
    invoker: &[u8; ADDR_LEN],
    commit: &[u8; COMMIT_LEN],
) -> [u8; AUTH_PREIMAGE_LEN] {
    let mut out = [0u8; AUTH_PREIMAGE_LEN];
    out[0] = EIP3074_MAGIC;
    // chainId: 32-byte BE; left-pad u64.
    out[1 + 32 - 8..1 + 32].copy_from_slice(&chain_id.to_be_bytes());
    // nonce: 32-byte BE; left-pad u64.
    out[1 + 32 + 32 - 8..1 + 32 + 32].copy_from_slice(&nonce.to_be_bytes());
    // invoker: 32-byte left-padded address.
    out[1 + 32 + 32 + 32 - 20..1 + 32 + 32 + 32].copy_from_slice(invoker);
    // commit: 32 bytes verbatim.
    out[1 + 32 + 32 + 32..AUTH_PREIMAGE_LEN].copy_from_slice(commit);
    out
}

/// Compute the keccak256 of the canonical AUTH preimage. Convenience
/// wrapper that uses the workspace keccak implementation.
pub fn auth_message_hash(
    chain_id: u64,
    nonce: u64,
    invoker: &[u8; ADDR_LEN],
    commit: &[u8; COMMIT_LEN],
) -> [u8; HASH_LEN] {
    let preimage = auth_message_preimage(chain_id, nonce, invoker, commit);
    crate::keccak::keccak256(&preimage)
}

/// Single-row host builder. Minimal host-side validation: y_parity ∈ {0,1}.
/// `auth_msg_hash` is computed from the canonical preimage.
pub fn from_authorization(
    authorized_address: [u8; ADDR_LEN],
    invoker: [u8; ADDR_LEN],
    commit: [u8; COMMIT_LEN],
    chain_id: u64,
    nonce: u64,
    signature: [u8; SIG_LEN],
    is_active: bool,
) -> Eip3074AuthWitness {
    assert!(
        signature[SIG_LEN - 1] <= 1,
        "eip3074: y_parity byte must be 0 or 1 (got {})",
        signature[SIG_LEN - 1],
    );
    let auth_msg_hash = auth_message_hash(chain_id, nonce, &invoker, &commit);
    Eip3074AuthWitness {
        rows: vec![Eip3074AuthRow {
            authorized_address,
            invoker,
            commit,
            nonce,
            chain_id,
            signature,
            auth_msg_hash,
            is_active,
            is_real: true,
        }],
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < 8);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

fn write_le_bytes(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    value: u64,
    row: usize,
    curve: CurveType,
) {
    let bytes = value.to_le_bytes();
    for b in 0..U64_BYTES {
        columns[offset + b][row] = Scalar::from_u64(bytes[b] as u64, curve);
    }
}

pub fn build_trace_polynomials(
    witness: &Eip3074AuthWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..ADDR_LEN {
            columns[COL_AUTHORIZED_ADDR_OFFSET + k][i] =
                Scalar::from_u64(row.authorized_address[k] as u64, curve);
            columns[COL_INVOKER_OFFSET + k][i] =
                Scalar::from_u64(row.invoker[k] as u64, curve);
        }
        for k in 0..COMMIT_LEN {
            columns[COL_COMMIT_OFFSET + k][i] =
                Scalar::from_u64(row.commit[k] as u64, curve);
        }
        for k in 0..SIG_LEN {
            columns[COL_SIGNATURE_OFFSET + k][i] =
                Scalar::from_u64(row.signature[k] as u64, curve);
        }
        for k in 0..HASH_LEN {
            columns[COL_AUTH_MSG_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.auth_msg_hash[k] as u64, curve);
        }
        columns[COL_CHAIN_ID][i] = Scalar::from_u64(row.chain_id, curve);
        write_le_bytes(&mut columns, COL_CHAIN_ID_BYTE_OFFSET, row.chain_id, i, curve);
        columns[COL_NONCE][i] = Scalar::from_u64(row.nonce, curve);
        write_le_bytes(&mut columns, COL_NONCE_BYTE_OFFSET, row.nonce, i, curve);
        columns[COL_IS_ACTIVE][i] = if row.is_active { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = if row.is_real { one.clone() } else { zero.clone() };
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct Eip3074AuthConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Eip3074AuthConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
}

fn sum_le_bytes(col_evals: &[Scalar], offset: usize, curve: CurveType) -> Scalar {
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[offset + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    sum
}

fn sum_le_bytes_poly(
    col_coeffs: &[Vec<Scalar>],
    offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[offset + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    sum
}

impl VmConstraintSystem for Eip3074AuthConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_active_binary".into(),
            "is_active_gates_real".into(),
            "y_parity_binary".into(),
            "chain_id_le_decomp".into(),
            "nonce_le_decomp".into(),
        ]
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
            let row_evals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_active = &row_evals[COL_IS_ACTIVE];
            let chain_id = &row_evals[COL_CHAIN_ID];
            let nonce = &row_evals[COL_NONCE];
            let y_parity = &row_evals[COL_SIGNATURE_OFFSET + SIG_LEN - 1];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_active.mul(&is_active.sub(&one));
            bodies[2][row] = is_active.mul(&one.sub(is_real));
            bodies[3][row] = y_parity.mul(&y_parity.sub(&one));
            let ci_sum = sum_le_bytes(&row_evals, COL_CHAIN_ID_BYTE_OFFSET, curve);
            bodies[4][row] = is_real.mul(&chain_id.sub(&ci_sum));
            let n_sum = sum_le_bytes(&row_evals, COL_NONCE_BYTE_OFFSET, curve);
            bodies[5][row] = is_real.mul(&nonce.sub(&n_sum));
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
        let is_active = &col_evals[COL_IS_ACTIVE];
        let chain_id = &col_evals[COL_CHAIN_ID];
        let nonce = &col_evals[COL_NONCE];
        let y_parity = &col_evals[COL_SIGNATURE_OFFSET + SIG_LEN - 1];
        let ci_sum = sum_le_bytes(col_evals, COL_CHAIN_ID_BYTE_OFFSET, curve);
        let n_sum = sum_le_bytes(col_evals, COL_NONCE_BYTE_OFFSET, curve);

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_active.mul(&is_active.sub(&one)),
            is_active.mul(&one.sub(is_real)),
            y_parity.mul(&y_parity.sub(&one)),
            is_real.mul(&chain_id.sub(&ci_sum)),
            is_real.mul(&nonce.sub(&n_sum)),
        ];

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
        let is_active = &col_coeffs[COL_IS_ACTIVE];
        let chain_id = &col_coeffs[COL_CHAIN_ID];
        let nonce = &col_coeffs[COL_NONCE];
        let y_parity = &col_coeffs[COL_SIGNATURE_OFFSET + SIG_LEN - 1];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve));
        bodies.push(poly_mul(is_active, &poly_sub(is_active, &one_poly, curve), curve));
        bodies.push(poly_mul(is_active, &poly_sub(&one_poly, is_real, curve), curve));
        bodies.push(poly_mul(y_parity, &poly_sub(y_parity, &one_poly, curve), curve));
        let ci_sum = sum_le_bytes_poly(col_coeffs, COL_CHAIN_ID_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(chain_id, &ci_sum, curve), curve));
        let n_sum = sum_le_bytes_poly(col_coeffs, COL_NONCE_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(nonce, &n_sum, curve), curve));

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
        for (off, len, label) in [
            (COL_AUTHORIZED_ADDR_OFFSET, ADDR_LEN, "authorized_addr_byte"),
            (COL_INVOKER_OFFSET, ADDR_LEN, "invoker_byte"),
            (COL_COMMIT_OFFSET, COMMIT_LEN, "commit_byte"),
            (COL_SIGNATURE_OFFSET, SIG_LEN, "signature_byte"),
            (COL_AUTH_MSG_HASH_OFFSET, HASH_LEN, "auth_msg_hash_byte"),
            (COL_CHAIN_ID_BYTE_OFFSET, U64_BYTES, "chain_id_byte"),
            (COL_NONCE_BYTE_OFFSET, U64_BYTES, "nonce_byte"),
        ] {
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

/// Bind this AIR's signature-verification tuple
/// `(auth_msg_hash[0..32], r[0..32], s[0..32], y_parity,
/// authorized_address[0..20])` to
/// [`crate::secp256k1_recovery`]'s
/// `(COL_MSG_HASH_OFFSET, COL_R_OFFSET, COL_S_OFFSET, COL_V,
/// COL_RECOVERED_ADDR_OFFSET)`. The recovery AIR proves
/// `authorized_address = ecrecover(auth_msg_hash, r||s||v)`; this
/// descriptor algebraically commits the EIP-3074 row's claimed
/// signature to that recovery. Gated by `IS_ACTIVE` on the AUTH side
/// and the recovery AIR's `IS_REAL` on the verifier side.
///
/// Total tuple width: 32 (hash) + 32 (r) + 32 (s) + 1 (parity) + 20
/// (addr) = 117 columns per side.
pub fn make_eip3074_to_secp256k1_recovery_descriptor(
    auth_layer_index: usize,
    recovery_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::secp256k1_recovery as sr;
    let mut a_columns: Vec<usize> = Vec::with_capacity(HASH_LEN + 32 + 32 + 1 + ADDR_LEN);
    let mut b_columns: Vec<usize> = Vec::with_capacity(HASH_LEN + 32 + 32 + 1 + ADDR_LEN);

    // msg_hash: 32 bytes.
    for k in 0..HASH_LEN {
        a_columns.push(COL_AUTH_MSG_HASH_OFFSET + k);
        b_columns.push(sr::COL_MSG_HASH_OFFSET + k);
    }
    // r: signature[0..32].
    for k in 0..32 {
        a_columns.push(COL_SIGNATURE_OFFSET + k);
        b_columns.push(sr::COL_R_OFFSET + k);
    }
    // s: signature[32..64].
    for k in 0..32 {
        a_columns.push(COL_SIGNATURE_OFFSET + 32 + k);
        b_columns.push(sr::COL_S_OFFSET + k);
    }
    // y_parity / v: signature[64] ↔ COL_V (recovery AIR pre-decodes v to
    // the y_parity ∈ {0,1} canonical form; EIP-3074's signature is
    // already in y_parity convention so the byte matches directly).
    a_columns.push(COL_SIGNATURE_OFFSET + SIG_LEN - 1);
    b_columns.push(sr::COL_V);
    // authorized_address: 20 bytes.
    for k in 0..ADDR_LEN {
        a_columns.push(COL_AUTHORIZED_ADDR_OFFSET + k);
        b_columns.push(sr::COL_RECOVERED_ADDR_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "eip3074_auth_to_secp256k1_recovery_v1".into(),
        a_layer_index: auth_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ACTIVE),
        b_layer_index: recovery_layer_index,
        b_columns,
        b_selector_column: Some(sr::COL_IS_REAL),
    }
}

/// Bind this AIR's `auth_msg_hash[0..32]` to the keccak `output_byte`
/// columns of [`crate::keccak_extract`] (the keccak gadget that
/// hashes the canonical 129-byte AUTH preimage). The preimage-side
/// linkage (that the keccak input is indeed `MAGIC || chainId ||
/// nonce || invoker || commit`) is committed via the keccak_extract
/// AIR's own input byte columns and an upstream preimage-construction
/// gadget; here we seal only the output hash → AUTH row binding.
///
/// Gated by `IS_ACTIVE` on the AUTH side and keccak_extract's
/// `IS_REAL` on the gadget side.
pub fn make_eip3074_to_keccak_extract_descriptor(
    auth_layer_index: usize,
    keccak_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let a_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| COL_AUTH_MSG_HASH_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| ke::COL_OUTPUT_BYTE_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "eip3074_auth_to_keccak_extract_v1".into(),
        a_layer_index: auth_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ACTIVE),
        b_layer_index: keccak_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sig(y_parity: u8) -> [u8; SIG_LEN] {
        let mut sig = [0u8; SIG_LEN];
        for k in 0..32 {
            sig[k] = (k + 1) as u8;
            sig[32 + k] = (k + 2) as u8;
        }
        sig[SIG_LEN - 1] = y_parity;
        sig
    }

    fn make_invoker() -> [u8; ADDR_LEN] {
        let mut a = [0u8; ADDR_LEN];
        a[0] = 0xc0;
        a[19] = 0xde;
        a
    }

    fn make_commit() -> [u8; COMMIT_LEN] {
        let mut c = [0u8; COMMIT_LEN];
        for (i, b) in c.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(11);
        }
        c
    }

    fn evaluate_bodies(
        witness: &Eip3074AuthWitness,
        curve: CurveType,
    ) -> (TracePolynomials, Vec<Vec<Scalar>>) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = Eip3074AuthConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        (trace, bodies)
    }

    fn assert_all_vanish(bodies: &[Vec<Scalar>]) {
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} should vanish at row {} (got {:?})",
                    k,
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    #[test]
    fn honest_auth_row_vanishes() {
        let curve = CurveType::Bls48581;
        let mut authorized = [0u8; ADDR_LEN];
        authorized[0] = 0xab;
        authorized[19] = 0x42;
        let invoker = make_invoker();
        let commit = make_commit();
        let sig = make_sig(0);
        let w = from_authorization(authorized, invoker, commit, 1, 17, sig, true);
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn tampered_y_parity_detected() {
        let curve = CurveType::Bls48581;
        let authorized = [0u8; ADDR_LEN];
        let invoker = make_invoker();
        let commit = make_commit();
        let sig = make_sig(1);
        let w = from_authorization(authorized, invoker, commit, 1, 1, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_SIGNATURE_OFFSET + SIG_LEN - 1][0] = Scalar::from_u64(3, curve);
        let cs = Eip3074AuthConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "y_parity_binary body must fire on invalid parity"
        );
    }

    #[test]
    fn tampered_chain_id_decomp_detected() {
        let curve = CurveType::Bls48581;
        let authorized = [0u8; ADDR_LEN];
        let invoker = make_invoker();
        let commit = make_commit();
        let sig = make_sig(0);
        let w = from_authorization(authorized, invoker, commit, 0xdeadbeef, 1, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper chain_id scalar without updating the byte decomposition.
        cols[COL_CHAIN_ID][0] = Scalar::from_u64(0xc0ffee, curve);
        let cs = Eip3074AuthConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[4][0].is_zero(),
            "chain_id_le_decomp must fire on mismatched scalar"
        );
    }

    #[test]
    fn is_active_gates_real_fires_on_padding_active() {
        let curve = CurveType::Bls48581;
        let authorized = [0u8; ADDR_LEN];
        let invoker = make_invoker();
        let commit = make_commit();
        let sig = make_sig(0);
        let w = from_authorization(authorized, invoker, commit, 1, 0, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // is_active=1 but is_real=0 — would let a padding row carry
        // a "successful" AUTH.
        cols[COL_IS_REAL][0] = Scalar::zero(curve);
        cols[COL_IS_ACTIVE][0] = Scalar::one(curve);
        let cs = Eip3074AuthConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "is_active_gates_real should fire when is_active=1 but is_real=0",
        );
    }

    #[test]
    fn preimage_layout_pins_magic_and_lengths() {
        let invoker = make_invoker();
        let commit = make_commit();
        let chain_id = 0x1234_5678_9abc_def0u64;
        let nonce = 0x0102_0304_0506_0708u64;
        let pre = auth_message_preimage(chain_id, nonce, &invoker, &commit);
        assert_eq!(pre.len(), AUTH_PREIMAGE_LEN);
        assert_eq!(pre[0], EIP3074_MAGIC);
        // chainId BE in the low 8 bytes of the first 32-byte field.
        assert_eq!(&pre[1 + 24..1 + 32], &chain_id.to_be_bytes());
        // Upper 24 bytes of chainId zero-padded.
        assert!(pre[1..1 + 24].iter().all(|b| *b == 0));
        // nonce BE in the low 8 bytes of the second 32-byte field.
        assert_eq!(&pre[1 + 32 + 24..1 + 32 + 32], &nonce.to_be_bytes());
        // invoker left-padded to 32 bytes.
        assert!(pre[1 + 32 + 32..1 + 32 + 32 + 12].iter().all(|b| *b == 0));
        assert_eq!(&pre[1 + 32 + 32 + 12..1 + 32 + 32 + 32], &invoker[..]);
        // commit verbatim in trailing 32 bytes.
        assert_eq!(&pre[1 + 32 + 32 + 32..AUTH_PREIMAGE_LEN], &commit[..]);
    }

    #[test]
    fn auth_msg_hash_matches_keccak_of_preimage() {
        let invoker = make_invoker();
        let commit = make_commit();
        let chain_id = 7u64;
        let nonce = 42u64;
        let pre = auth_message_preimage(chain_id, nonce, &invoker, &commit);
        let expected = crate::keccak::keccak256(&pre);
        let got = auth_message_hash(chain_id, nonce, &invoker, &commit);
        assert_eq!(got, expected);

        // The trace column should commit the same bytes.
        let sig = make_sig(0);
        let w = from_authorization([0u8; ADDR_LEN], invoker, commit, chain_id, nonce, sig, true);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for k in 0..HASH_LEN {
            assert_eq!(
                trace.columns[COL_AUTH_MSG_HASH_OFFSET + k].evaluations[0].to_u64(),
                expected[k] as u64,
            );
        }
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_eip3074_to_secp256k1_recovery_descriptor(0, 1);
        assert_eq!(d1.label, "eip3074_auth_to_secp256k1_recovery_v1");
        // 32 hash + 32 r + 32 s + 1 parity + 20 addr = 117.
        assert_eq!(d1.a_columns.len(), HASH_LEN + 32 + 32 + 1 + ADDR_LEN);
        assert_eq!(d1.b_columns.len(), d1.a_columns.len());
        assert_eq!(d1.a_selector_column, Some(COL_IS_ACTIVE));
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);

        let d2 = make_eip3074_to_keccak_extract_descriptor(0, 2);
        assert_eq!(d2.label, "eip3074_auth_to_keccak_extract_v1");
        assert_eq!(d2.a_columns.len(), HASH_LEN);
        assert_eq!(d2.b_columns.len(), HASH_LEN);
        assert_eq!(
            d2.a_columns,
            (0..HASH_LEN).map(|k| COL_AUTH_MSG_HASH_OFFSET + k).collect::<Vec<_>>(),
        );
        assert_eq!(d2.a_selector_column, Some(COL_IS_ACTIVE));
    }

    #[test]
    fn column_layout_pinned() {
        // 20 + 20 + 32 + 65 + 32 + 1 + 8 + 1 + 8 + 1 + 1 = 189.
        assert_eq!(COL_AUTHORIZED_ADDR_OFFSET, 0);
        assert_eq!(COL_INVOKER_OFFSET, 20);
        assert_eq!(COL_COMMIT_OFFSET, 40);
        assert_eq!(COL_SIGNATURE_OFFSET, 72);
        assert_eq!(COL_AUTH_MSG_HASH_OFFSET, 137);
        assert_eq!(COL_CHAIN_ID, 169);
        assert_eq!(COL_NONCE, 178);
        assert_eq!(COL_IS_ACTIVE, 187);
        assert_eq!(COL_IS_REAL, 188);
        assert_eq!(NUM_COLUMNS, 189);
        assert_eq!(NUM_ROW_CONSTRAINTS, 6);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(EIP3074_MAGIC, 0x04);
        assert_eq!(AUTH_PREIMAGE_LEN, 129);
    }

    #[test]
    fn evaluate_at_point_matches_for_honest_row() {
        let curve = CurveType::Bls48581;
        let invoker = make_invoker();
        let commit = make_commit();
        let sig = make_sig(1);
        let w =
            from_authorization([0u8; ADDR_LEN], invoker, commit, 1, 1, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let cs = Eip3074AuthConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(29, curve);
        let row0_evals: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    #[test]
    fn lookup_declarations_cover_all_byte_columns() {
        let cs = Eip3074AuthConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        // Expected byte columns: 20 + 20 + 32 + 65 + 32 + 8 + 8 = 185.
        let expected = ADDR_LEN + ADDR_LEN + COMMIT_LEN + SIG_LEN + HASH_LEN + U64_BYTES + U64_BYTES;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, _) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
        }
        // Sanity: range table 256.
        assert_eq!(reqs.tables.len(), 1);
    }
}
