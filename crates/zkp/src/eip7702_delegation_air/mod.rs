//! EIP-7702 set-code (delegation) AIR.
//!
//! EIP-7702 (Pectra) introduces a transaction type that lets an EOA
//! authorize a smart-contract delegate. The authorization is a
//! signed tuple
//!
//! ```text
//! auth_msg = keccak256( MAGIC || rlp([chain_id, address, nonce]) )
//! signature = secp256k1_sign(auth_msg, authority_priv)
//! ```
//!
//! where `MAGIC = 0x05` is the EIP-7702 domain-separator byte, and
//! the recovered signer address MUST equal `authority_address`.
//!
//! ## Per-authorization witness
//!
//! * `authority_address[0..20]` — the EOA performing the
//!   delegation (claimed equal to `ecrecover(auth_msg, signature)`).
//! * `chain_id`                  — u64. The authorization is valid only
//!   on this chain (or for any chain when `chain_id = 0`).
//! * `nonce`                     — u64. Must match the authority's
//!   on-chain account nonce at the time of the transaction's
//!   inclusion.
//! * `delegated_to[0..20]`       — the contract address the
//!   authority is delegating execution to. A zero address removes
//!   the delegation.
//! * `signature[0..65]`          — `r (32) || s (32) || y_parity (1)`.
//! * `is_active`                 — host-side flag; 1 iff this row
//!   represents an authorization that the transaction actually
//!   applied (i.e. the nonce matched and `chain_id` matched).
//!
//! ## Constraints
//!
//! 0. `is_real_binary`        — `is_real · (is_real − 1) = 0`.
//! 1. `is_active_binary`      — `is_active · (is_active − 1) = 0`.
//! 2. `is_active_gates_real`  — `is_active · (1 − is_real) = 0`.
//! 3. `y_parity_binary`       — the last byte of `signature` is
//!    `0` or `1`.
//! 4. `chain_id_le_decomp`    — `chain_id − Σ_b CHAIN_ID_BYTE[b] · 2^(8b) = 0`.
//! 5. `nonce_le_decomp`       — `nonce − Σ_b NONCE_BYTE[b] · 2^(8b) = 0`.
//!
//! All committed byte columns are 8-bit range-checked.
//!
//! ## Cross-AIR linkages
//!
//! * Output of `secp256k1_recovery::RecoveryAir` (recovered address
//!   limbs) ↔ this AIR's `authority_address` bytes packed into 4 ×
//!   u64 limbs.
//! * `(chain_id, nonce)` of this AIR ↔ `(COL_CHAIN_ID, COL_NONCE)`
//!   of [`crate::tx_rlp_air`], which exposes the per-transaction
//!   chain_id and nonce. (EIP-7702's authorization_list lives inside
//!   the tx; this descriptor binds the chain/nonce inputs to the
//!   surrounding transaction's chain_id field.)

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// EIP-7702 domain-separator magic byte.
pub const EIP7702_MAGIC: u8 = 0x05;

pub const ADDR_LEN: usize = 20;
pub const SIG_LEN: usize = 65;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_AUTHORITY_ADDR_OFFSET: usize = 0; // 0..20
pub const COL_DELEGATED_TO_OFFSET: usize = COL_AUTHORITY_ADDR_OFFSET + ADDR_LEN; // 20..40
pub const COL_SIGNATURE_OFFSET: usize = COL_DELEGATED_TO_OFFSET + ADDR_LEN; // 40..105
pub const COL_CHAIN_ID: usize = COL_SIGNATURE_OFFSET + SIG_LEN; // 105
pub const COL_CHAIN_ID_BYTE_OFFSET: usize = COL_CHAIN_ID + 1; // 106..114
pub const COL_NONCE: usize = COL_CHAIN_ID_BYTE_OFFSET + U64_BYTES; // 114
pub const COL_NONCE_BYTE_OFFSET: usize = COL_NONCE + 1; // 115..123

pub const COL_IS_ACTIVE: usize = COL_NONCE_BYTE_OFFSET + U64_BYTES; // 123
pub const COL_IS_REAL: usize = COL_IS_ACTIVE + 1; // 124

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 125
pub const NUM_ROW_CONSTRAINTS: usize = 6;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Eip7702DelegationRow {
    pub authority_address: [u8; ADDR_LEN],
    pub chain_id: u64,
    pub nonce: u64,
    pub delegated_to: [u8; ADDR_LEN],
    pub signature: [u8; SIG_LEN],
    pub is_active: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Eip7702DelegationWitness {
    pub rows: Vec<Eip7702DelegationRow>,
}

impl Eip7702DelegationWitness {
    pub fn from_rows(rows: Vec<Eip7702DelegationRow>) -> Self {
        Self { rows }
    }
}

/// Single-row host builder.
///
/// Performs minimal host-side validation: y_parity is 0 or 1.
pub fn from_authorization(
    authority_address: [u8; ADDR_LEN],
    chain_id: u64,
    nonce: u64,
    delegated_to: [u8; ADDR_LEN],
    signature: [u8; SIG_LEN],
    is_active: bool,
) -> Eip7702DelegationWitness {
    assert!(
        signature[SIG_LEN - 1] <= 1,
        "eip7702: y_parity byte must be 0 or 1 (got {})",
        signature[SIG_LEN - 1],
    );
    Eip7702DelegationWitness {
        rows: vec![Eip7702DelegationRow {
            authority_address,
            chain_id,
            nonce,
            delegated_to,
            signature,
            is_active,
            is_real: true,
        }],
    }
}

/// Compute the EIP-7702 authorization message preimage used by
/// `keccak256` to produce the secp256k1 signing hash. This is
/// `MAGIC (1) || rlp([chain_id, address, nonce])`. Host-side helper
/// for tests and oracle composition; the AIR does not enforce the
/// RLP encoding itself (that is delegated to a future
/// `eip7702_rlp_air` gadget).
pub fn auth_message_preimage(
    chain_id: u64,
    address: &[u8; ADDR_LEN],
    nonce: u64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 1 + 32 + 1 + 20 + 1 + 8);
    out.push(EIP7702_MAGIC);
    // Minimal-RLP encoder for the (u64, [u8;20], u64) tuple.
    fn rlp_u64(n: u64, dst: &mut Vec<u8>) {
        if n == 0 {
            dst.push(0x80);
            return;
        }
        let be = n.to_be_bytes();
        let start = be.iter().position(|b| *b != 0).unwrap();
        let bytes = &be[start..];
        if bytes.len() == 1 && bytes[0] < 0x80 {
            dst.push(bytes[0]);
        } else {
            dst.push(0x80 + bytes.len() as u8);
            dst.extend_from_slice(bytes);
        }
    }
    let mut body = Vec::new();
    rlp_u64(chain_id, &mut body);
    body.push(0x80 + ADDR_LEN as u8);
    body.extend_from_slice(address);
    rlp_u64(nonce, &mut body);
    if body.len() < 56 {
        out.push(0xc0 + body.len() as u8);
    } else {
        let lb = (body.len() as u64).to_be_bytes();
        let start = lb.iter().position(|b| *b != 0).unwrap();
        let bytes = &lb[start..];
        out.push(0xf7 + bytes.len() as u8);
        out.extend_from_slice(bytes);
    }
    out.extend_from_slice(&body);
    out
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
    witness: &Eip7702DelegationWitness,
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
            columns[COL_AUTHORITY_ADDR_OFFSET + k][i] =
                Scalar::from_u64(row.authority_address[k] as u64, curve);
            columns[COL_DELEGATED_TO_OFFSET + k][i] =
                Scalar::from_u64(row.delegated_to[k] as u64, curve);
        }
        for k in 0..SIG_LEN {
            columns[COL_SIGNATURE_OFFSET + k][i] =
                Scalar::from_u64(row.signature[k] as u64, curve);
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

pub struct Eip7702DelegationConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Eip7702DelegationConstraintSystem {
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

impl VmConstraintSystem for Eip7702DelegationConstraintSystem {
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
        // All address + signature bytes are 8-bit.
        for (off, len, label) in [
            (COL_AUTHORITY_ADDR_OFFSET, ADDR_LEN, "authority_addr_byte"),
            (COL_DELEGATED_TO_OFFSET, ADDR_LEN, "delegated_to_byte"),
            (COL_SIGNATURE_OFFSET, SIG_LEN, "signature_byte"),
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

/// Bind this AIR's `authority_address` bytes to
/// [`crate::secp256k1_recovery`]'s `COL_RECOVERED_ADDR_OFFSET` over
/// the same 20 bytes. Gated by `IS_ACTIVE` on the delegation side
/// (active authorizations must have an algebraically recovered
/// signer matching the claimed authority) and by `COL_IS_REAL` on
/// the recovery side.
///
/// This is the algebraic seal between the EIP-7702 authorization
/// and the secp256k1 ECDSA recovery: tampering the
/// `authority_address` column of an active row will break the
/// LogUp closure.
pub fn make_eip7702_to_secp256k1_recovery_descriptor(
    delegation_layer_index: usize,
    recovery_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::secp256k1_recovery as sr;
    let a_columns: Vec<usize> =
        (0..ADDR_LEN).map(|k| COL_AUTHORITY_ADDR_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..ADDR_LEN).map(|k| sr::COL_RECOVERED_ADDR_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "eip7702_delegation_to_secp256k1_recovery_v1".into(),
        a_layer_index: delegation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ACTIVE),
        b_layer_index: recovery_layer_index,
        b_columns,
        b_selector_column: Some(sr::COL_IS_REAL),
    }
}

/// Bind `(chain_id, nonce)` of this AIR to
/// [`crate::tx_rlp_air`]'s `(COL_CHAIN_ID, COL_NONCE)`. The
/// authorization's chain_id MUST match the surrounding tx's
/// chain_id (or be 0 for "any chain"). The nonce binding pins
/// the on-chain authority nonce the tx was executed at.
pub fn make_eip7702_to_tx_rlp_descriptor(
    delegation_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_rlp_air as txr;
    let a_columns = vec![COL_CHAIN_ID, COL_NONCE];
    let b_columns = vec![txr::COL_CHAIN_ID, txr::COL_NONCE];
    CrossAirLogUpDescriptor {
        label: "eip7702_delegation_to_tx_rlp_v1".into(),
        a_layer_index: delegation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ACTIVE),
        b_layer_index: tx_rlp_layer_index,
        b_columns,
        b_selector_column: Some(txr::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sig(y_parity: u8) -> [u8; SIG_LEN] {
        let mut sig = [0u8; SIG_LEN];
        for k in 0..32 {
            sig[k] = (k + 1) as u8; // r
            sig[32 + k] = (k + 2) as u8; // s
        }
        sig[SIG_LEN - 1] = y_parity;
        sig
    }

    fn evaluate_bodies(
        witness: &Eip7702DelegationWitness,
        curve: CurveType,
    ) -> (TracePolynomials, Vec<Vec<Scalar>>) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = Eip7702DelegationConstraintSystem::new(trace.num_rows);
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
    fn honest_authorization_vanishes() {
        let curve = CurveType::Bls48581;
        let mut authority = [0u8; ADDR_LEN];
        authority[0] = 0xab;
        let mut delegated = [0u8; ADDR_LEN];
        delegated[19] = 0xee;
        let sig = make_sig(0);
        let w = from_authorization(authority, 1, 17, delegated, sig, true);
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn tampered_y_parity_detected() {
        let curve = CurveType::Bls48581;
        let authority = [0u8; ADDR_LEN];
        let delegated = [0u8; ADDR_LEN];
        let sig = make_sig(0);
        let w = from_authorization(authority, 1, 1, delegated, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper y_parity byte to 2 (invalid).
        cols[COL_SIGNATURE_OFFSET + SIG_LEN - 1][0] = Scalar::from_u64(2, curve);
        let cs = Eip7702DelegationConstraintSystem::new(trace.num_rows);
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
        let authority = [0u8; ADDR_LEN];
        let delegated = [0u8; ADDR_LEN];
        let sig = make_sig(1);
        let w = from_authorization(authority, 0xdeadbeef, 1, delegated, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper chain_id scalar without updating bytes.
        cols[COL_CHAIN_ID][0] = Scalar::from_u64(0xc0ffee, curve);
        let cs = Eip7702DelegationConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[4][0].is_zero(),
            "chain_id_le_decomp must fire on mismatched scalar"
        );
    }

    #[test]
    fn is_active_gating_is_real() {
        let curve = CurveType::Bls48581;
        let authority = [0u8; ADDR_LEN];
        let delegated = [0u8; ADDR_LEN];
        let sig = make_sig(0);
        let w = from_authorization(authority, 1, 0, delegated, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: is_active=1 but is_real=0 (would let a padding-only
        // row carry an active authorization).
        cols[COL_IS_REAL][0] = Scalar::zero(curve);
        cols[COL_IS_ACTIVE][0] = Scalar::one(curve);
        let cs = Eip7702DelegationConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "is_active_gates_real should fire when is_active=1 but is_real=0"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_eip7702_to_secp256k1_recovery_descriptor(0, 1);
        assert_eq!(d1.label, "eip7702_delegation_to_secp256k1_recovery_v1");
        assert_eq!(d1.a_columns.len(), ADDR_LEN);
        assert_eq!(d1.b_columns.len(), ADDR_LEN);
        assert_eq!(d1.a_selector_column, Some(COL_IS_ACTIVE));

        let d2 = make_eip7702_to_tx_rlp_descriptor(0, 2);
        assert_eq!(d2.label, "eip7702_delegation_to_tx_rlp_v1");
        assert_eq!(d2.a_columns, vec![COL_CHAIN_ID, COL_NONCE]);
        assert_eq!(
            d2.b_columns,
            vec![
                crate::tx_rlp_air::COL_CHAIN_ID,
                crate::tx_rlp_air::COL_NONCE,
            ],
        );
    }

    #[test]
    fn column_layout_pinned() {
        // 20 (auth) + 20 (delegated) + 65 (sig) + 1 + 8 (chain_id)
        // + 1 + 8 (nonce) + 1 (is_active) + 1 (is_real) = 125.
        assert_eq!(NUM_COLUMNS, 125);
        assert_eq!(NUM_ROW_CONSTRAINTS, 6);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(EIP7702_MAGIC, 0x05);
    }

    #[test]
    fn auth_message_preimage_starts_with_magic() {
        let address = [0xaau8; ADDR_LEN];
        let pre = auth_message_preimage(1, &address, 5);
        assert_eq!(pre[0], EIP7702_MAGIC);
        // RLP list prefix follows.
        assert!(pre[1] >= 0xc0);
    }

    #[test]
    fn evaluate_at_point_matches_for_honest() {
        let curve = CurveType::Bls48581;
        let authority = [0u8; ADDR_LEN];
        let delegated = [0u8; ADDR_LEN];
        let sig = make_sig(1);
        let w = from_authorization(authority, 1, 1, delegated, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let cs = Eip7702DelegationConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(23, curve);
        let row0_evals: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    #[test]
    #[ignore = "slow: standalone prove+verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let mut authority = [0u8; ADDR_LEN];
        authority[0] = 0xab;
        let mut delegated = [0u8; ADDR_LEN];
        delegated[19] = 0xee;
        let sig = make_sig(0);
        let w = from_authorization(authority, 1, 17, delegated, sig, true);
        let trace = build_trace_polynomials(&w, curve);
        let cs = Eip7702DelegationConstraintSystem::new(trace.num_rows);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone eip7702_delegation_air proof must verify",
        );
    }
}
