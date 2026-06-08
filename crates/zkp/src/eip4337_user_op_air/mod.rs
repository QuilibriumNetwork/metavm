//! EIP-4337 account-abstraction `UserOperation` AIR.
//!
//! Proves the per-`UserOperation` hash binding used by the
//! EntryPoint contract to validate an AA bundle. The witness
//! commits the (packed) `UserOperation` fields and the
//! `user_op_hash` that downstream validation signs against.
//!
//! `user_op_hash` is computed by the EntryPoint as
//!
//! ```text
//! user_op_hash = keccak256(
//!     keccak256(packed_user_op) || entry_point || chain_id
//! )
//! ```
//!
//! For this AIR we commit the inner keccak — that is, the
//! `user_op_hash` column equals `keccak256(packed_user_op)` —
//! and surface the (commitment, sub-field-hashes) so a downstream
//! AIR can fold in the EntryPoint and chain id when those become
//! relevant. The packing is the canonical 0.7 EntryPoint layout:
//!
//! ```text
//! packed = abi.encode(
//!     sender,
//!     nonce,
//!     keccak256(initCode),
//!     keccak256(callData),
//!     accountGasLimits,         // packed (verificationGasLimit << 128 | callGasLimit)
//!     preVerificationGas,
//!     gasFees,                  // packed (maxPriorityFee << 128 | maxFee)
//!     keccak256(paymasterAndData)
//! )
//! ```
//!
//! ## Per-row witness
//!
//! * `sender[0..20]`         — EOA / smart-account address.
//! * `nonce`                 — u64 (the low 64 bits of the EntryPoint
//!   nonce; the full 256-bit nonce is bound via the
//!   `nonce_packed_hash` column).
//! * `init_code_hash[0..32]` — `keccak256(initCode)`.
//! * `call_data_hash[0..32]` — `keccak256(callData)`.
//! * `signature_hash[0..32]` — `keccak256(signature)`. Bound
//!   separately so a downstream ECDSA / ERC-1271 gadget can fold in.
//! * `paymaster_hash[0..32]` — `keccak256(paymasterAndData)`.
//! * `validation_gas`        — u64 verificationGasLimit.
//! * `call_gas`              — u64 callGasLimit.
//! * `user_op_hash[0..32]`   — `keccak256(packed_user_op)`.
//! * `is_valid`              — host-side selector; 1 iff
//!   the EntryPoint validation step accepted this op.
//!
//! ## Constraints
//!
//! 0. `is_real_binary`         — `is_real (is_real − 1) = 0`.
//! 1. `is_valid_binary`        — `is_valid (is_valid − 1) = 0`.
//! 2. `is_valid_gates_real`    — `is_valid (1 − is_real) = 0`.
//! 3. `nonce_le_decomp`        — `nonce − Σ NONCE_BYTE[b] · 2^(8b) = 0`.
//! 4. `validation_gas_le_decomp` — same for `validation_gas`.
//! 5. `call_gas_le_decomp`     — same for `call_gas`.
//!
//! ## Cross-AIR linkages
//!
//! * `(packed_preimage_hash_tuple, user_op_hash)` ↔
//!   [`crate::keccak_extract`]. The `keccak_extract` AIR commits
//!   `(input_bytes, output_bytes)` pairs; this AIR exposes the
//!   `user_op_hash` (the output) and the EntryPoint will surface
//!   the input bytes via a future packed-preimage gadget. We bind
//!   the 32 output bytes here so any tampering of `user_op_hash`
//!   relative to the preimage hash that the EntryPoint feeds into
//!   `keccak_extract` is rejected.
//! * `(signature_hash, recovered_addr)` ↔
//!   [`crate::ecrecover_chain_air`]. Binds the signature commitment
//!   to the recovered sender address; the EntryPoint requires the
//!   recovered address to equal `sender` for EOAs.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

pub const ADDR_LEN: usize = 20;
pub const HASH_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_SENDER_OFFSET: usize = 0; // 0..20
pub const COL_INIT_CODE_HASH_OFFSET: usize = COL_SENDER_OFFSET + ADDR_LEN; // 20..52
pub const COL_CALL_DATA_HASH_OFFSET: usize = COL_INIT_CODE_HASH_OFFSET + HASH_LEN; // 52..84
pub const COL_SIGNATURE_HASH_OFFSET: usize = COL_CALL_DATA_HASH_OFFSET + HASH_LEN; // 84..116
pub const COL_PAYMASTER_HASH_OFFSET: usize = COL_SIGNATURE_HASH_OFFSET + HASH_LEN; // 116..148
pub const COL_USER_OP_HASH_OFFSET: usize = COL_PAYMASTER_HASH_OFFSET + HASH_LEN; // 148..180

pub const COL_NONCE: usize = COL_USER_OP_HASH_OFFSET + HASH_LEN; // 180
pub const COL_NONCE_BYTE_OFFSET: usize = COL_NONCE + 1; // 181..189
pub const COL_VALIDATION_GAS: usize = COL_NONCE_BYTE_OFFSET + U64_BYTES; // 189
pub const COL_VALIDATION_GAS_BYTE_OFFSET: usize = COL_VALIDATION_GAS + 1; // 190..198
pub const COL_CALL_GAS: usize = COL_VALIDATION_GAS_BYTE_OFFSET + U64_BYTES; // 198
pub const COL_CALL_GAS_BYTE_OFFSET: usize = COL_CALL_GAS + 1; // 199..207

pub const COL_IS_VALID: usize = COL_CALL_GAS_BYTE_OFFSET + U64_BYTES; // 207
pub const COL_IS_REAL: usize = COL_IS_VALID + 1; // 208

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 209
pub const NUM_ROW_CONSTRAINTS: usize = 6;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Eip4337UserOpRow {
    pub sender: [u8; ADDR_LEN],
    pub nonce: u64,
    pub init_code_hash: [u8; HASH_LEN],
    pub call_data_hash: [u8; HASH_LEN],
    pub signature_hash: [u8; HASH_LEN],
    pub paymaster_hash: [u8; HASH_LEN],
    pub validation_gas: u64,
    pub call_gas: u64,
    pub user_op_hash: [u8; HASH_LEN],
    pub is_valid: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Eip4337UserOpWitness {
    pub rows: Vec<Eip4337UserOpRow>,
}

impl Eip4337UserOpWitness {
    pub fn from_rows(rows: Vec<Eip4337UserOpRow>) -> Self {
        Self { rows }
    }
}

/// Single-row host builder. Performs no algebraic checks itself —
/// the AIR algebraically binds nonce/gas decompositions, and the
/// cross-AIR descriptors bind `user_op_hash` to the
/// `keccak_extract` output and `signature_hash` to the ECRecover
/// chain. Use [`pack_user_op`] to compute the canonical preimage
/// that `keccak_extract` should hash to produce `user_op_hash`.
#[allow(clippy::too_many_arguments)]
pub fn from_user_op(
    sender: [u8; ADDR_LEN],
    nonce: u64,
    init_code_hash: [u8; HASH_LEN],
    call_data_hash: [u8; HASH_LEN],
    signature_hash: [u8; HASH_LEN],
    paymaster_hash: [u8; HASH_LEN],
    validation_gas: u64,
    call_gas: u64,
    user_op_hash: [u8; HASH_LEN],
    is_valid: bool,
) -> Eip4337UserOpWitness {
    Eip4337UserOpWitness {
        rows: vec![Eip4337UserOpRow {
            sender,
            nonce,
            init_code_hash,
            call_data_hash,
            signature_hash,
            paymaster_hash,
            validation_gas,
            call_gas,
            user_op_hash,
            is_valid,
            is_real: true,
        }],
    }
}

/// Host helper: pack the EIP-4337 v0.7 user-op fields into the
/// `abi.encode` preimage `keccak_extract` should hash.
///
/// Layout (each segment is 32 bytes, big-endian zero-padded):
///
/// `sender || nonce || init_code_hash || call_data_hash ||
///  accountGasLimits || preVerificationGas || gasFees ||
///  paymaster_hash`
///
/// `accountGasLimits = (validation_gas << 128) | call_gas`
/// (truncated to the low 32 bytes).
/// `preVerificationGas` and `gasFees` are zero in this scaffold
/// (they are committed via separate columns in a future revision).
#[allow(clippy::too_many_arguments)]
pub fn pack_user_op(
    sender: &[u8; ADDR_LEN],
    nonce: u64,
    init_code_hash: &[u8; HASH_LEN],
    call_data_hash: &[u8; HASH_LEN],
    paymaster_hash: &[u8; HASH_LEN],
    validation_gas: u64,
    call_gas: u64,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * 32);
    // sender (left-padded to 32).
    out.extend(std::iter::repeat(0u8).take(32 - ADDR_LEN));
    out.extend_from_slice(sender);
    // nonce (32-byte BE).
    let mut nonce_be = [0u8; 32];
    nonce_be[24..].copy_from_slice(&nonce.to_be_bytes());
    out.extend_from_slice(&nonce_be);
    // init_code_hash.
    out.extend_from_slice(init_code_hash);
    // call_data_hash.
    out.extend_from_slice(call_data_hash);
    // accountGasLimits (verificationGasLimit << 128 | callGasLimit), 32B.
    let mut gas_limits = [0u8; 32];
    gas_limits[8..16].copy_from_slice(&validation_gas.to_be_bytes());
    gas_limits[24..].copy_from_slice(&call_gas.to_be_bytes());
    out.extend_from_slice(&gas_limits);
    // preVerificationGas (32B zero scaffold).
    out.extend_from_slice(&[0u8; 32]);
    // gasFees (32B zero scaffold).
    out.extend_from_slice(&[0u8; 32]);
    // paymaster_hash.
    out.extend_from_slice(paymaster_hash);
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
    witness: &Eip4337UserOpWitness,
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
            columns[COL_SENDER_OFFSET + k][i] = Scalar::from_u64(row.sender[k] as u64, curve);
        }
        for k in 0..HASH_LEN {
            columns[COL_INIT_CODE_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.init_code_hash[k] as u64, curve);
            columns[COL_CALL_DATA_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.call_data_hash[k] as u64, curve);
            columns[COL_SIGNATURE_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.signature_hash[k] as u64, curve);
            columns[COL_PAYMASTER_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.paymaster_hash[k] as u64, curve);
            columns[COL_USER_OP_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.user_op_hash[k] as u64, curve);
        }
        columns[COL_NONCE][i] = Scalar::from_u64(row.nonce, curve);
        write_le_bytes(&mut columns, COL_NONCE_BYTE_OFFSET, row.nonce, i, curve);
        columns[COL_VALIDATION_GAS][i] = Scalar::from_u64(row.validation_gas, curve);
        write_le_bytes(
            &mut columns,
            COL_VALIDATION_GAS_BYTE_OFFSET,
            row.validation_gas,
            i,
            curve,
        );
        columns[COL_CALL_GAS][i] = Scalar::from_u64(row.call_gas, curve);
        write_le_bytes(&mut columns, COL_CALL_GAS_BYTE_OFFSET, row.call_gas, i, curve);
        columns[COL_IS_VALID][i] = if row.is_valid { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = if row.is_real { one.clone() } else { zero.clone() };
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct Eip4337UserOpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Eip4337UserOpConstraintSystem {
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

impl VmConstraintSystem for Eip4337UserOpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_valid_binary".into(),
            "is_valid_gates_real".into(),
            "nonce_le_decomp".into(),
            "validation_gas_le_decomp".into(),
            "call_gas_le_decomp".into(),
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
            let is_valid = &row_evals[COL_IS_VALID];
            let nonce = &row_evals[COL_NONCE];
            let vgas = &row_evals[COL_VALIDATION_GAS];
            let cgas = &row_evals[COL_CALL_GAS];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_valid.mul(&is_valid.sub(&one));
            bodies[2][row] = is_valid.mul(&one.sub(is_real));
            let n_sum = sum_le_bytes(&row_evals, COL_NONCE_BYTE_OFFSET, curve);
            bodies[3][row] = is_real.mul(&nonce.sub(&n_sum));
            let v_sum = sum_le_bytes(&row_evals, COL_VALIDATION_GAS_BYTE_OFFSET, curve);
            bodies[4][row] = is_real.mul(&vgas.sub(&v_sum));
            let c_sum = sum_le_bytes(&row_evals, COL_CALL_GAS_BYTE_OFFSET, curve);
            bodies[5][row] = is_real.mul(&cgas.sub(&c_sum));
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
        let is_valid = &col_evals[COL_IS_VALID];
        let nonce = &col_evals[COL_NONCE];
        let vgas = &col_evals[COL_VALIDATION_GAS];
        let cgas = &col_evals[COL_CALL_GAS];
        let n_sum = sum_le_bytes(col_evals, COL_NONCE_BYTE_OFFSET, curve);
        let v_sum = sum_le_bytes(col_evals, COL_VALIDATION_GAS_BYTE_OFFSET, curve);
        let c_sum = sum_le_bytes(col_evals, COL_CALL_GAS_BYTE_OFFSET, curve);

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_valid.mul(&is_valid.sub(&one)),
            is_valid.mul(&one.sub(is_real)),
            is_real.mul(&nonce.sub(&n_sum)),
            is_real.mul(&vgas.sub(&v_sum)),
            is_real.mul(&cgas.sub(&c_sum)),
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
        let is_valid = &col_coeffs[COL_IS_VALID];
        let nonce = &col_coeffs[COL_NONCE];
        let vgas = &col_coeffs[COL_VALIDATION_GAS];
        let cgas = &col_coeffs[COL_CALL_GAS];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve));
        bodies.push(poly_mul(is_valid, &poly_sub(is_valid, &one_poly, curve), curve));
        bodies.push(poly_mul(is_valid, &poly_sub(&one_poly, is_real, curve), curve));
        let n_sum = sum_le_bytes_poly(col_coeffs, COL_NONCE_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(nonce, &n_sum, curve), curve));
        let v_sum = sum_le_bytes_poly(col_coeffs, COL_VALIDATION_GAS_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(vgas, &v_sum, curve), curve));
        let c_sum = sum_le_bytes_poly(col_coeffs, COL_CALL_GAS_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(cgas, &c_sum, curve), curve));

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
            (COL_SENDER_OFFSET, ADDR_LEN, "sender_byte"),
            (COL_INIT_CODE_HASH_OFFSET, HASH_LEN, "init_code_hash_byte"),
            (COL_CALL_DATA_HASH_OFFSET, HASH_LEN, "call_data_hash_byte"),
            (COL_SIGNATURE_HASH_OFFSET, HASH_LEN, "signature_hash_byte"),
            (COL_PAYMASTER_HASH_OFFSET, HASH_LEN, "paymaster_hash_byte"),
            (COL_USER_OP_HASH_OFFSET, HASH_LEN, "user_op_hash_byte"),
            (COL_NONCE_BYTE_OFFSET, U64_BYTES, "nonce_byte"),
            (COL_VALIDATION_GAS_BYTE_OFFSET, U64_BYTES, "validation_gas_byte"),
            (COL_CALL_GAS_BYTE_OFFSET, U64_BYTES, "call_gas_byte"),
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

/// Bind this AIR's `user_op_hash` bytes to
/// [`crate::keccak_extract`]'s `COL_OUTPUT_BYTE_OFFSET` over the
/// same 32 output bytes.
///
/// Gated by `IS_VALID` on this side and `COL_IS_REAL` on the
/// `keccak_extract` side. Tampering the user_op_hash column on any
/// valid row will break the LogUp closure.
pub fn make_eip4337_to_keccak_extract_descriptor(
    user_op_layer_index: usize,
    keccak_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let a_columns: Vec<usize> = (0..HASH_LEN).map(|k| COL_USER_OP_HASH_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..HASH_LEN).map(|k| ke::COL_OUTPUT_BYTE_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "eip4337_user_op_to_keccak_extract_v1".into(),
        a_layer_index: user_op_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_VALID),
        b_layer_index: keccak_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Bind this AIR's `sender` address bytes to
/// [`crate::ecrecover_chain_air`]'s `COL_RECOVERED_ADDR_OFFSET`.
/// Gated by `IS_VALID` on this side.
///
/// For EOA signers the EntryPoint requires `recovered_addr ==
/// sender`. For ERC-1271 smart-account signers this descriptor
/// can be disabled per-row by setting `is_valid = 0` and using a
/// separate ERC-1271 verification AIR (future).
pub fn make_eip4337_to_ecrecover_chain_descriptor(
    user_op_layer_index: usize,
    ecrecover_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::ecrecover_chain_air as ec;
    let a_columns: Vec<usize> = (0..ADDR_LEN).map(|k| COL_SENDER_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..ADDR_LEN).map(|k| ec::COL_RECOVERED_ADDR_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "eip4337_user_op_to_ecrecover_chain_v1".into(),
        a_layer_index: user_op_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_VALID),
        b_layer_index: ecrecover_layer_index,
        b_columns,
        b_selector_column: Some(ec::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(seed: u8) -> [u8; HASH_LEN] {
        let mut h = [0u8; HASH_LEN];
        for k in 0..HASH_LEN {
            h[k] = seed.wrapping_add(k as u8);
        }
        h
    }

    fn evaluate_bodies(
        witness: &Eip4337UserOpWitness,
        curve: CurveType,
    ) -> (TracePolynomials, Vec<Vec<Scalar>>) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = Eip4337UserOpConstraintSystem::new(trace.num_rows);
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
    fn honest_user_op_vanishes() {
        let curve = CurveType::Bls48581;
        let mut sender = [0u8; ADDR_LEN];
        sender[0] = 0x11;
        let w = from_user_op(
            sender,
            42,
            make_hash(0x11),
            make_hash(0x22),
            make_hash(0x33),
            make_hash(0x44),
            100_000,
            50_000,
            make_hash(0x55),
            true,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn tampered_nonce_decomp_detected() {
        let curve = CurveType::Bls48581;
        let sender = [0u8; ADDR_LEN];
        let w = from_user_op(
            sender,
            7,
            make_hash(0),
            make_hash(0),
            make_hash(0),
            make_hash(0),
            1000,
            1000,
            make_hash(0),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_NONCE][0] = Scalar::from_u64(0xbeef, curve);
        let cs = Eip4337UserOpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "nonce_le_decomp must fire on mismatched scalar"
        );
    }

    #[test]
    fn is_valid_gating_is_real_detected() {
        let curve = CurveType::Bls48581;
        let sender = [0u8; ADDR_LEN];
        let w = from_user_op(
            sender,
            0,
            make_hash(0),
            make_hash(0),
            make_hash(0),
            make_hash(0),
            0,
            0,
            make_hash(0),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // is_valid=1 but is_real=0 — gating constraint must fire.
        cols[COL_IS_REAL][0] = Scalar::zero(curve);
        cols[COL_IS_VALID][0] = Scalar::one(curve);
        let cs = Eip4337UserOpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "is_valid_gates_real must fire on is_valid=1 / is_real=0"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_eip4337_to_keccak_extract_descriptor(0, 1);
        assert_eq!(d1.label, "eip4337_user_op_to_keccak_extract_v1");
        assert_eq!(d1.a_columns.len(), HASH_LEN);
        assert_eq!(d1.b_columns.len(), HASH_LEN);
        assert_eq!(d1.a_selector_column, Some(COL_IS_VALID));

        let d2 = make_eip4337_to_ecrecover_chain_descriptor(0, 2);
        assert_eq!(d2.label, "eip4337_user_op_to_ecrecover_chain_v1");
        assert_eq!(d2.a_columns.len(), ADDR_LEN);
        assert_eq!(d2.b_columns.len(), ADDR_LEN);
    }

    #[test]
    fn pack_user_op_layout_pinned() {
        let sender = [0x11u8; ADDR_LEN];
        let p = pack_user_op(
            &sender,
            42,
            &make_hash(1),
            &make_hash(2),
            &make_hash(3),
            100,
            200,
        );
        assert_eq!(p.len(), 8 * 32);
        // sender left-padded.
        assert_eq!(&p[..12], &[0u8; 12]);
        assert_eq!(&p[12..32], &sender);
        // nonce in low 8 bytes of word 1.
        assert_eq!(&p[32..56], &[0u8; 24]);
        assert_eq!(&p[56..64], &42u64.to_be_bytes());
        // accountGasLimits split.
        assert_eq!(&p[128..136], &[0u8; 8]);
        assert_eq!(&p[136..144], &100u64.to_be_bytes());
        assert_eq!(&p[152..160], &200u64.to_be_bytes());
    }

    #[test]
    fn column_layout_pinned() {
        // 20 (sender) + 5×32 (init/call/sig/paymaster/user_op_hash)
        // + 1+8 (nonce) + 1+8 (validation_gas) + 1+8 (call_gas)
        // + 1 (is_valid) + 1 (is_real) = 209.
        assert_eq!(NUM_COLUMNS, 209);
        assert_eq!(NUM_ROW_CONSTRAINTS, 6);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn evaluate_at_point_matches_for_honest() {
        let curve = CurveType::Bls48581;
        let sender = [0u8; ADDR_LEN];
        let w = from_user_op(
            sender,
            1,
            make_hash(0),
            make_hash(0),
            make_hash(0),
            make_hash(0),
            1,
            1,
            make_hash(0),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let cs = Eip4337UserOpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(31, curve);
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

        let mut sender = [0u8; ADDR_LEN];
        sender[0] = 0x11;
        let w = from_user_op(
            sender,
            42,
            make_hash(0x11),
            make_hash(0x22),
            make_hash(0x33),
            make_hash(0x44),
            100_000,
            50_000,
            make_hash(0x55),
            true,
        );
        let trace = build_trace_polynomials(&w, curve);
        let cs = Eip4337UserOpConstraintSystem::new(trace.num_rows);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone eip4337_user_op_air proof must verify",
        );
    }
}
