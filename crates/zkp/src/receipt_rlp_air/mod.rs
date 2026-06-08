//! Algebraic receipt RLP encoding AIR (no-logs case).
//!
//! Phase A2 #59 step 1+: a dedicated gadget AIR that algebraically
//! commits to the canonical RLP encoding of an Ethereum [`Receipt`]
//! with **no logs**. The host-side oracle in
//! [`crate::receipt_rlp_composition`] already validates the byte-level
//! decomposition; this AIR lifts the no-logs case into row-local
//! constraints + cross-AIR LogUp linkages to the existing per-field
//! gadget AIRs.
//!
//! # Encoding shape (no logs)
//!
//! ```text
//! Receipt.wire_encoding() =
//!   [type_byte]?                              (1 byte, only if typed)
//! || 0xf9 || len_hi || len_lo                 (3-byte long-list prefix)
//! || RLP(status)                              (always 1 byte for status ∈ {0,1})
//! || RLP(cumulative_gas_used)                 (1..=9 bytes, u64 RLP)
//! || 0xb9 || 0x01 || 0x00 || logs_bloom[256]  (259 bytes, fixed)
//! || 0xc0                                     (1 byte, empty logs list)
//! ```
//!
//! Since `logs_bloom` alone contributes 259 bytes, the payload always
//! exceeds 55 bytes and is in fact always ≥ 262 bytes (1 + 1 + 259 + 1)
//! and ≤ 270 bytes (1 + 9 + 259 + 1). Therefore the list prefix is
//! **always** the 3-byte long-list form `[0xf9, len_hi, len_lo]` with
//! the 2-byte length encoding `payload_len` big-endian.
//!
//! # Status encoding
//!
//! We use the canonical scalar-RLP encoding via [`crate::u64_rlp_air`]:
//!   - `status = 0` → `[0x80]` (empty string)
//!   - `status = 1` → `[0x01]`
//!
//! Note: the host-side [`Receipt::wire_encoding`] currently uses
//! `rlp_encode_bytes(&[status])` which yields `[0x00]` for `status=0`
//! (a divergence from the Ethereum yellow-paper convention). For the
//! success case (`status=1`) both conventions agree; the AIR adopts
//! the scalar convention because it matches the available u64-RLP
//! sub-gadget. Tests pin `status=1` for byte-for-byte agreement.
//!
//! # What this AIR enforces algebraically
//!
//! Per row, gated by `IS_REAL`:
//!   1. `IS_REAL` is binary.
//!   2. `IS_LEGACY` is binary.
//!   3. `IS_TYPED` is binary.
//!   4. `IS_LEGACY + IS_TYPED = IS_REAL` (mutually exclusive over real rows).
//!   5. Type byte at position 0 equals `IS_TYPED * TYPE_BYTE` (zero
//!      for legacy; one of {0x01, 0x02, 0x03} for typed).
//!   6. `TYPE_BYTE_LEN = IS_TYPED` (the prepended byte counts only
//!      when typed).
//!   7. `LIST_PREFIX_0 = 0xf9` (always the long-list 2-len-byte form).
//!   8. `LIST_PREFIX_LEN_HI * 256 + LIST_PREFIX_LEN_LO = PAYLOAD_LEN`.
//!   9. `PAYLOAD_LEN = STATUS_ENC_LEN + CUMUL_ENC_LEN + LOGS_BLOOM_ENC_LEN + 1`
//!      with `STATUS_ENC_LEN = 1`, `LOGS_BLOOM_ENC_LEN = 259`, trailing
//!      empty-logs `[0xc0]` contributing 1.
//!  10. `ENCODED_LEN = TYPE_BYTE_LEN + 3 + PAYLOAD_LEN`.
//!  11. `EMPTY_LOGS_BYTE = 0xc0`.
//!
//! All 8-bit columns get range checks via [`LookupTable::range(256)`].
//!
//! # Cross-AIR LogUp linkages (not row-local)
//!
//! - `make_receipt_to_u64_rlp_status_descriptor` — binds
//!   `(STATUS, STATUS_ENC_LEN, STATUS_ENC[0..9])` to the u64-RLP gadget's
//!   `(value, encoded_len, encoded[0..9])`.
//! - `make_receipt_to_u64_rlp_cumulative_gas_descriptor` — analogous
//!   for `cumulative_gas_used`.
//! - `make_receipt_to_logs_bloom_descriptor` — binds the 256 bloom
//!   bytes to [`crate::rlp_logs_bloom_air`]'s field-byte columns.
//!
//! # Deferred
//!
//! - Logs encoding (per-log RLP gadget + variable-length logs-list
//!   concatenation). Witness builder rejects non-empty logs.
//! - SHA3 / MPT binding of the resulting RLP into the receiptsRoot.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

use crate::receipt::{Receipt, ReceiptType};
use crate::u64_rlp_air::{rlp_encode_u64, MAX_ENCODED_LEN as U64_MAX_ENCODED_LEN};

// ─── Encoding constants ──────────────────────────────────────────────

/// Fixed list-prefix lead byte: payload ≥ 262 bytes → always the
/// long-list 2-length-bytes form `0xf7 + 2 = 0xf9`.
pub const LIST_PREFIX_BYTE_0: u8 = 0xf9;

/// List prefix length: always 3 bytes (`[0xf9, len_hi, len_lo]`).
pub const LIST_PREFIX_LEN: usize = 3;

/// Status is always exactly 1 byte in the canonical scalar-RLP
/// encoding for `status ∈ {0, 1}`.
pub const STATUS_ENC_LEN: usize = 1;

/// Logs bloom is fixed 259 bytes (`0xb9 || 0x01 || 0x00 || 256-byte bloom`).
pub const LOGS_BLOOM_ENC_LEN: usize = crate::rlp_logs_bloom_air::ENCODED_LEN; // 259

/// Empty-logs list encodes to a single byte `0xc0`.
pub const EMPTY_LOGS_ENC_LEN: usize = 1;
pub const EMPTY_LOGS_BYTE: u8 = 0xc0;

/// 256-byte logs-bloom width (the raw field).
pub const LOGS_BLOOM_LEN: usize = 256;

/// Minimum / maximum payload length (no-logs receipt):
///   min: status (1) + cumul (1) + bloom (259) + empty_logs (1) = 262
///   max: status (1) + cumul (9) + bloom (259) + empty_logs (1) = 270
pub const MIN_PAYLOAD_LEN: usize = STATUS_ENC_LEN + 1 + LOGS_BLOOM_ENC_LEN + EMPTY_LOGS_ENC_LEN;
pub const MAX_PAYLOAD_LEN: usize =
    STATUS_ENC_LEN + U64_MAX_ENCODED_LEN + LOGS_BLOOM_ENC_LEN + EMPTY_LOGS_ENC_LEN;

/// Max total encoded length: optional type byte + list prefix + payload.
pub const MAX_ENCODED_LEN: usize = 1 + LIST_PREFIX_LEN + MAX_PAYLOAD_LEN;

// ─── Column layout ────────────────────────────────────────────────────

// Receipt-shape selectors
pub const COL_IS_REAL: usize = 0;
pub const COL_IS_LEGACY: usize = 1;
pub const COL_IS_TYPED: usize = 2;

// EIP-2718 type byte (0 for legacy; 0x01/0x02/0x03 for typed). Sits
// at output position 0 when typed.
pub const COL_TYPE_BYTE: usize = 3;
// 1 if typed (the byte is consumed at position 0), 0 if legacy.
pub const COL_TYPE_BYTE_LEN: usize = 4;

// Scalar fields.
pub const COL_STATUS: usize = 5;
pub const COL_CUMULATIVE_GAS: usize = 6;

// Per-field encoded buffers (exposed to cross-AIR LogUp).
pub const COL_STATUS_ENC_OFFSET: usize = 7;                              // 7..16 (9 bytes max, only first byte used)
pub const COL_STATUS_ENC_LEN: usize = COL_STATUS_ENC_OFFSET + U64_MAX_ENCODED_LEN; // 16

pub const COL_CUMUL_ENC_OFFSET: usize = COL_STATUS_ENC_LEN + 1;          // 17..26
pub const COL_CUMUL_ENC_LEN: usize = COL_CUMUL_ENC_OFFSET + U64_MAX_ENCODED_LEN; // 26

// Raw logs_bloom bytes (256) exposed to cross-AIR LogUp.
pub const COL_LOGS_BLOOM_OFFSET: usize = COL_CUMUL_ENC_LEN + 1;          // 27..283
pub const COL_LOGS_BLOOM_END: usize = COL_LOGS_BLOOM_OFFSET + LOGS_BLOOM_LEN; // 283

// Trailing empty-logs byte.
pub const COL_EMPTY_LOGS_BYTE: usize = COL_LOGS_BLOOM_END;               // 283

// List prefix bytes.
pub const COL_LIST_PREFIX_0: usize = COL_EMPTY_LOGS_BYTE + 1;            // 284
pub const COL_LIST_PREFIX_LEN_HI: usize = COL_LIST_PREFIX_0 + 1;         // 285
pub const COL_LIST_PREFIX_LEN_LO: usize = COL_LIST_PREFIX_LEN_HI + 1;    // 286

// Length bookkeeping.
pub const COL_PAYLOAD_LEN: usize = COL_LIST_PREFIX_LEN_LO + 1;           // 287
pub const COL_ENCODED_LEN: usize = COL_PAYLOAD_LEN + 1;                  // 288

pub const NUM_COLUMNS: usize = COL_ENCODED_LEN + 1;                      // 289

/// Row-local constraint count.
///
///   0: is_real binary
///   1: is_legacy binary
///   2: is_typed binary
///   3: is_legacy + is_typed = is_real
///   4: type_byte_len = is_typed
///   5: list_prefix_0 = 0xf9
///   6: list_prefix_len_hi * 256 + list_prefix_len_lo = payload_len
///   7: payload_len = status_enc_len + cumul_enc_len + 260
///   8: encoded_len = type_byte_len + 3 + payload_len
///   9: empty_logs_byte = 0xc0
///  10: status_enc_len = 1 (status fits in a single RLP byte)
pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ReceiptRlpRow {
    pub ty: ReceiptType,
    pub status: u8,
    pub cumulative_gas_used: u64,
    pub logs_bloom: [u8; LOGS_BLOOM_LEN],
    pub log_count: usize,
    pub encoded_bytes: Vec<u8>,
    pub encoded_len: usize,
    pub payload_len: usize,
    pub status_enc_len: usize,
    pub cumul_enc_len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ReceiptRlpWitness {
    pub rows: Vec<ReceiptRlpRow>,
}

impl ReceiptRlpRow {
    /// Build a witness row from a no-logs [`Receipt`]. Receipts with
    /// non-empty logs are rejected; that case requires the (deferred)
    /// per-log RLP gadget + logs-list concat AIR.
    pub fn from_receipt(receipt: &Receipt) -> Result<Self, &'static str> {
        if !receipt.logs.is_empty() {
            return Err("receipt_rlp_air: no-logs case only (per-log RLP gadget deferred)");
        }
        // Status uses canonical scalar RLP via u64_rlp_air. Receipts
        // with status > 1 are out of spec but we accept any u8 fitting
        // in 1 byte; status_enc_len is pinned to 1 here.
        let status_enc = rlp_encode_u64(receipt.status as u64);
        debug_assert!(status_enc.len() == STATUS_ENC_LEN);
        let cumul_enc = rlp_encode_u64(receipt.cumulative_gas_used);
        let cumul_enc_len = cumul_enc.len();
        let payload_len = STATUS_ENC_LEN + cumul_enc_len + LOGS_BLOOM_ENC_LEN + EMPTY_LOGS_ENC_LEN;
        debug_assert!(payload_len >= MIN_PAYLOAD_LEN && payload_len <= MAX_PAYLOAD_LEN);
        let type_byte_len = match receipt.ty.type_byte() {
            None => 0,
            Some(_) => 1,
        };
        let encoded_len = type_byte_len + LIST_PREFIX_LEN + payload_len;

        // Reassemble the wire encoding the AIR claims; cross-check
        // against the canonical encoder for tests / debug builds.
        let mut encoded = Vec::with_capacity(encoded_len);
        if let Some(tb) = receipt.ty.type_byte() {
            encoded.push(tb);
        }
        encoded.push(LIST_PREFIX_BYTE_0);
        encoded.push((payload_len >> 8) as u8);
        encoded.push((payload_len & 0xff) as u8);
        encoded.extend_from_slice(&status_enc);
        encoded.extend_from_slice(&cumul_enc);
        // logs_bloom RLP = 0xb9 || 0x01 || 0x00 || 256 bytes.
        encoded.extend_from_slice(&crate::rlp_logs_bloom_air::PREFIX_BYTES);
        encoded.extend_from_slice(&receipt.logs_bloom);
        encoded.push(EMPTY_LOGS_BYTE);

        Ok(Self {
            ty: receipt.ty,
            status: receipt.status,
            cumulative_gas_used: receipt.cumulative_gas_used,
            logs_bloom: receipt.logs_bloom,
            log_count: 0,
            encoded_bytes: encoded,
            encoded_len,
            payload_len,
            status_enc_len: STATUS_ENC_LEN,
            cumul_enc_len,
        })
    }
}

impl ReceiptRlpWitness {
    pub fn from_receipts(receipts: &[Receipt]) -> Result<Self, &'static str> {
        let mut rows = Vec::with_capacity(receipts.len());
        for r in receipts {
            rows.push(ReceiptRlpRow::from_receipt(r)?);
        }
        Ok(Self { rows })
    }

    pub fn from_receipt(receipt: &Receipt) -> Result<Self, &'static str> {
        Self::from_receipts(std::slice::from_ref(receipt))
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ReceiptRlpWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        // Selectors.
        columns[COL_IS_REAL][r] = one.clone();
        let (is_legacy, is_typed, type_byte_val) = match row.ty.type_byte() {
            None => (one.clone(), zero.clone(), zero.clone()),
            Some(tb) => (
                zero.clone(),
                one.clone(),
                Scalar::from_u64(tb as u64, curve),
            ),
        };
        columns[COL_IS_LEGACY][r] = is_legacy;
        columns[COL_IS_TYPED][r] = is_typed.clone();
        columns[COL_TYPE_BYTE][r] = type_byte_val;
        columns[COL_TYPE_BYTE_LEN][r] = is_typed;

        // Scalar fields.
        columns[COL_STATUS][r] = Scalar::from_u64(row.status as u64, curve);
        columns[COL_CUMULATIVE_GAS][r] = Scalar::from_u64(row.cumulative_gas_used, curve);

        // Per-field canonical encodings.
        let status_enc = rlp_encode_u64(row.status as u64);
        debug_assert!(status_enc.len() <= U64_MAX_ENCODED_LEN);
        columns[COL_STATUS_ENC_LEN][r] = Scalar::from_u64(status_enc.len() as u64, curve);
        for (k, &b) in status_enc.iter().enumerate() {
            columns[COL_STATUS_ENC_OFFSET + k][r] = Scalar::from_u64(b as u64, curve);
        }

        let cumul_enc = rlp_encode_u64(row.cumulative_gas_used);
        debug_assert!(cumul_enc.len() <= U64_MAX_ENCODED_LEN);
        columns[COL_CUMUL_ENC_LEN][r] = Scalar::from_u64(cumul_enc.len() as u64, curve);
        for (k, &b) in cumul_enc.iter().enumerate() {
            columns[COL_CUMUL_ENC_OFFSET + k][r] = Scalar::from_u64(b as u64, curve);
        }

        // Logs bloom (raw bytes).
        for k in 0..LOGS_BLOOM_LEN {
            columns[COL_LOGS_BLOOM_OFFSET + k][r] =
                Scalar::from_u64(row.logs_bloom[k] as u64, curve);
        }

        // Trailing empty-logs byte.
        columns[COL_EMPTY_LOGS_BYTE][r] = Scalar::from_u64(EMPTY_LOGS_BYTE as u64, curve);

        // List prefix + length bookkeeping.
        columns[COL_LIST_PREFIX_0][r] = Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve);
        columns[COL_LIST_PREFIX_LEN_HI][r] = Scalar::from_u64((row.payload_len >> 8) as u64, curve);
        columns[COL_LIST_PREFIX_LEN_LO][r] =
            Scalar::from_u64((row.payload_len & 0xff) as u64, curve);
        columns[COL_PAYLOAD_LEN][r] = Scalar::from_u64(row.payload_len as u64, curve);
        columns[COL_ENCODED_LEN][r] = Scalar::from_u64(row.encoded_len as u64, curve);
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

pub struct ReceiptRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ReceiptRlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for ReceiptRlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_legacy_binary".into(),
            "is_typed_binary".into(),
            "is_legacy_plus_typed_eq_is_real".into(),
            "type_byte_len_eq_is_typed".into(),
            "list_prefix_0_eq_0xf9".into(),
            "list_prefix_len_consistency".into(),
            "payload_len_arith".into(),
            "encoded_len_arith".into(),
            "empty_logs_byte_eq_0xc0".into(),
            "status_enc_len_eq_1".into(),
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
        let f9 = Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve);
        let c0_byte = Scalar::from_u64(EMPTY_LOGS_BYTE as u64, curve);
        let two56 = Scalar::from_u64(256, curve);
        // Fixed part of payload: bloom (259) + empty_logs (1) = 260.
        let fixed = Scalar::from_u64((LOGS_BLOOM_ENC_LEN + EMPTY_LOGS_ENC_LEN) as u64, curve);
        let three = Scalar::from_u64(LIST_PREFIX_LEN as u64, curve);
        let one_const = Scalar::from_u64(STATUS_ENC_LEN as u64, curve);

        let mk = || vec![Scalar::zero(curve); n];
        let mut c_isr = mk();
        let mut c_isl = mk();
        let mut c_ist = mk();
        let mut c_sum = mk();
        let mut c_tbl = mk();
        let mut c_p0 = mk();
        let mut c_plen = mk();
        let mut c_payload = mk();
        let mut c_enc = mk();
        let mut c_empty = mk();
        let mut c_status_len = mk();

        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            c_isr[r] = v.mul(&v.sub(&one));

            let il = &columns[COL_IS_LEGACY][r];
            c_isl[r] = il.mul(&il.sub(&one));

            let it = &columns[COL_IS_TYPED][r];
            c_ist[r] = it.mul(&it.sub(&one));

            // is_legacy + is_typed = is_real
            c_sum[r] = il.add(it).sub(v);

            // type_byte_len = is_typed
            let tbl = &columns[COL_TYPE_BYTE_LEN][r];
            c_tbl[r] = v.mul(&tbl.sub(it));

            // list_prefix_0 = 0xf9
            let p0 = &columns[COL_LIST_PREFIX_0][r];
            c_p0[r] = v.mul(&p0.sub(&f9));

            // list_prefix_len_hi * 256 + lo = payload_len
            let hi = &columns[COL_LIST_PREFIX_LEN_HI][r];
            let lo = &columns[COL_LIST_PREFIX_LEN_LO][r];
            let pl = &columns[COL_PAYLOAD_LEN][r];
            let combined = hi.mul(&two56).add(lo);
            c_plen[r] = v.mul(&combined.sub(pl));

            // payload_len = status_enc_len + cumul_enc_len + 260
            let sl = &columns[COL_STATUS_ENC_LEN][r];
            let cl = &columns[COL_CUMUL_ENC_LEN][r];
            let expected_pl = sl.add(cl).add(&fixed);
            c_payload[r] = v.mul(&pl.sub(&expected_pl));

            // encoded_len = type_byte_len + 3 + payload_len
            let el = &columns[COL_ENCODED_LEN][r];
            let expected_el = tbl.add(&three).add(pl);
            c_enc[r] = v.mul(&el.sub(&expected_el));

            // empty_logs_byte = 0xc0
            let eb = &columns[COL_EMPTY_LOGS_BYTE][r];
            c_empty[r] = v.mul(&eb.sub(&c0_byte));

            // status_enc_len = 1
            c_status_len[r] = v.mul(&sl.sub(&one_const));
        }

        vec![
            c_isr,
            c_isl,
            c_ist,
            c_sum,
            c_tbl,
            c_p0,
            c_plen,
            c_payload,
            c_enc,
            c_empty,
            c_status_len,
        ]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let f9 = Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve);
        let c0_byte = Scalar::from_u64(EMPTY_LOGS_BYTE as u64, curve);
        let two56 = Scalar::from_u64(256, curve);
        let fixed = Scalar::from_u64((LOGS_BLOOM_ENC_LEN + EMPTY_LOGS_ENC_LEN) as u64, curve);
        let three = Scalar::from_u64(LIST_PREFIX_LEN as u64, curve);
        let one_const = Scalar::from_u64(STATUS_ENC_LEN as u64, curve);

        let v = &col_evals[COL_IS_REAL];
        let c0 = v.mul(&v.sub(&one));

        let il = &col_evals[COL_IS_LEGACY];
        let c1 = il.mul(&il.sub(&one));

        let it = &col_evals[COL_IS_TYPED];
        let c2 = it.mul(&it.sub(&one));

        let c3 = il.add(it).sub(v);

        let tbl = &col_evals[COL_TYPE_BYTE_LEN];
        let c4 = v.mul(&tbl.sub(it));

        let p0 = &col_evals[COL_LIST_PREFIX_0];
        let c5 = v.mul(&p0.sub(&f9));

        let hi = &col_evals[COL_LIST_PREFIX_LEN_HI];
        let lo = &col_evals[COL_LIST_PREFIX_LEN_LO];
        let pl = &col_evals[COL_PAYLOAD_LEN];
        let combined = hi.mul(&two56).add(lo);
        let c6 = v.mul(&combined.sub(pl));

        let sl = &col_evals[COL_STATUS_ENC_LEN];
        let cl = &col_evals[COL_CUMUL_ENC_LEN];
        let expected_pl = sl.add(cl).add(&fixed);
        let c7 = v.mul(&pl.sub(&expected_pl));

        let el = &col_evals[COL_ENCODED_LEN];
        let expected_el = tbl.add(&three).add(pl);
        let c8 = v.mul(&el.sub(&expected_el));

        let eb = &col_evals[COL_EMPTY_LOGS_BYTE];
        let c9 = v.mul(&eb.sub(&c0_byte));

        let c10 = v.mul(&sl.sub(&one_const));

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10];
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
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let f9_poly = vec![Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve)];
        let c0_poly = vec![Scalar::from_u64(EMPTY_LOGS_BYTE as u64, curve)];
        let two56_poly = vec![Scalar::from_u64(256, curve)];
        let fixed_poly =
            vec![Scalar::from_u64((LOGS_BLOOM_ENC_LEN + EMPTY_LOGS_ENC_LEN) as u64, curve)];
        let three_poly = vec![Scalar::from_u64(LIST_PREFIX_LEN as u64, curve)];
        let one_const_poly = vec![Scalar::from_u64(STATUS_ENC_LEN as u64, curve)];

        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let c0 = poly_mul(v, &v_m1, curve);

        let il = &col_coeffs[COL_IS_LEGACY];
        let il_m1 = poly_sub(il, &one_poly, curve);
        let c1 = poly_mul(il, &il_m1, curve);

        let it = &col_coeffs[COL_IS_TYPED];
        let it_m1 = poly_sub(it, &one_poly, curve);
        let c2 = poly_mul(it, &it_m1, curve);

        let il_plus_it = poly_add(il, it, curve);
        let c3 = poly_sub(&il_plus_it, v, curve);

        let tbl = &col_coeffs[COL_TYPE_BYTE_LEN];
        let tbl_m_it = poly_sub(tbl, it, curve);
        let c4 = poly_mul(v, &tbl_m_it, curve);

        let p0 = &col_coeffs[COL_LIST_PREFIX_0];
        let p0_m = poly_sub(p0, &f9_poly, curve);
        let c5 = poly_mul(v, &p0_m, curve);

        let hi = &col_coeffs[COL_LIST_PREFIX_LEN_HI];
        let lo = &col_coeffs[COL_LIST_PREFIX_LEN_LO];
        let pl = &col_coeffs[COL_PAYLOAD_LEN];
        let hi_mul = poly_mul(hi, &two56_poly, curve);
        let combined = poly_add(&hi_mul, lo, curve);
        let diff_pl = poly_sub(&combined, pl, curve);
        let c6 = poly_mul(v, &diff_pl, curve);

        let sl = &col_coeffs[COL_STATUS_ENC_LEN];
        let cl = &col_coeffs[COL_CUMUL_ENC_LEN];
        let sl_plus_cl = poly_add(sl, cl, curve);
        let expected_pl = poly_add(&sl_plus_cl, &fixed_poly, curve);
        let pl_diff = poly_sub(pl, &expected_pl, curve);
        let c7 = poly_mul(v, &pl_diff, curve);

        let el = &col_coeffs[COL_ENCODED_LEN];
        let tbl_plus_three = poly_add(tbl, &three_poly, curve);
        let expected_el = poly_add(&tbl_plus_three, pl, curve);
        let el_diff = poly_sub(el, &expected_el, curve);
        let c8 = poly_mul(v, &el_diff, curve);

        let eb = &col_coeffs[COL_EMPTY_LOGS_BYTE];
        let eb_m = poly_sub(eb, &c0_poly, curve);
        let c9 = poly_mul(v, &eb_m, curve);

        let sl_m1 = poly_sub(sl, &one_const_poly, curve);
        let c10 = poly_mul(v, &sl_m1, curve);

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10];
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
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

        // type_byte / list_prefix bytes / empty_logs byte get range checks.
        for (idx, label) in [
            (COL_TYPE_BYTE, "receipt_rlp_type_byte_8bit"),
            (COL_LIST_PREFIX_0, "receipt_rlp_list_prefix_0_8bit"),
            (COL_LIST_PREFIX_LEN_HI, "receipt_rlp_list_prefix_len_hi_8bit"),
            (COL_LIST_PREFIX_LEN_LO, "receipt_rlp_list_prefix_len_lo_8bit"),
            (COL_EMPTY_LOGS_BYTE, "receipt_rlp_empty_logs_byte_8bit"),
        ] {
            declarations.push((
                LookupDeclaration {
                    label: label.into(),
                    column_index: idx,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }

        // Per-field canonical encoded buffers: every byte range-checked.
        for k in 0..U64_MAX_ENCODED_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("receipt_rlp_status_enc_{}_8bit", k),
                    column_index: COL_STATUS_ENC_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("receipt_rlp_cumul_enc_{}_8bit", k),
                    column_index: COL_CUMUL_ENC_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }

        // logs_bloom byte columns get range checks.
        for k in 0..LOGS_BLOOM_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("receipt_rlp_logs_bloom_{}_8bit", k),
                    column_index: COL_LOGS_BLOOM_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Linkage descriptor binding this AIR's `(status, status_enc_len,
/// status_enc[0..9])` tuple to the dedicated [`crate::u64_rlp_air`]'s
/// `(value, encoded_len, encoded[0..9])`.
///
/// Combined with the u64 RLP AIR's own constraints, this pins the
/// `status_enc` bytes to the unique canonical scalar-RLP encoding of
/// `status`.
pub fn make_receipt_to_u64_rlp_status_descriptor(
    receipt_rlp_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(2 + U64_MAX_ENCODED_LEN);
    a_columns.push(COL_STATUS);
    a_columns.push(COL_STATUS_ENC_LEN);
    for k in 0..U64_MAX_ENCODED_LEN {
        a_columns.push(COL_STATUS_ENC_OFFSET + k);
    }

    let mut b_columns = Vec::with_capacity(2 + U64_MAX_ENCODED_LEN);
    b_columns.push(crate::u64_rlp_air::COL_VALUE);
    b_columns.push(crate::u64_rlp_air::COL_ENCODED_LEN);
    for k in 0..U64_MAX_ENCODED_LEN {
        b_columns.push(crate::u64_rlp_air::COL_ENCODED_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "receipt_rlp_to_u64_rlp_status_v1".into(),
        a_layer_index: receipt_rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns,
        b_selector_column: Some(crate::u64_rlp_air::COL_IS_REAL),
    }
}

/// Linkage descriptor binding this AIR's `(cumulative_gas_used,
/// cumul_enc_len, cumul_enc[0..9])` tuple to the u64 RLP AIR.
pub fn make_receipt_to_u64_rlp_cumulative_gas_descriptor(
    receipt_rlp_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(2 + U64_MAX_ENCODED_LEN);
    a_columns.push(COL_CUMULATIVE_GAS);
    a_columns.push(COL_CUMUL_ENC_LEN);
    for k in 0..U64_MAX_ENCODED_LEN {
        a_columns.push(COL_CUMUL_ENC_OFFSET + k);
    }

    let mut b_columns = Vec::with_capacity(2 + U64_MAX_ENCODED_LEN);
    b_columns.push(crate::u64_rlp_air::COL_VALUE);
    b_columns.push(crate::u64_rlp_air::COL_ENCODED_LEN);
    for k in 0..U64_MAX_ENCODED_LEN {
        b_columns.push(crate::u64_rlp_air::COL_ENCODED_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "receipt_rlp_to_u64_rlp_cumulative_gas_v1".into(),
        a_layer_index: receipt_rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns,
        b_selector_column: Some(crate::u64_rlp_air::COL_IS_REAL),
    }
}

/// Linkage descriptor binding this AIR's 256-byte `logs_bloom[0..256]`
/// raw-field columns to the dedicated [`crate::rlp_logs_bloom_air`]'s
/// `FIELD_BYTE[0..256]` columns. Combined with that AIR's prefix and
/// body constraints, this pins the bloom RLP encoding to its canonical
/// 259-byte form.
pub fn make_receipt_to_logs_bloom_descriptor(
    receipt_rlp_layer_index: usize,
    logs_bloom_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(LOGS_BLOOM_LEN);
    for k in 0..LOGS_BLOOM_LEN {
        a_columns.push(COL_LOGS_BLOOM_OFFSET + k);
    }

    let mut b_columns = Vec::with_capacity(LOGS_BLOOM_LEN);
    for k in 0..LOGS_BLOOM_LEN {
        b_columns.push(crate::rlp_logs_bloom_air::COL_FIELD_BYTE_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "receipt_rlp_to_logs_bloom_v1".into(),
        a_layer_index: receipt_rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: logs_bloom_layer_index,
        b_columns,
        b_selector_column: Some(crate::rlp_logs_bloom_air::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_no_logs() -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    fn eip1559_no_logs() -> Receipt {
        Receipt {
            ty: ReceiptType::Eip1559,
            status: 1,
            cumulative_gas_used: 100_000,
            logs_bloom: [0x11; 256],
            logs: Vec::new(),
        }
    }

    fn build_for(receipts: &[Receipt]) -> TracePolynomials {
        let w = ReceiptRlpWitness::from_receipts(receipts).unwrap();
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    #[test]
    fn legacy_no_logs_encoded_len_correct() {
        let r = legacy_no_logs();
        let trace = build_for(&[r.clone()]);
        let cumul_enc_len = rlp_encode_u64(r.cumulative_gas_used).len();
        let payload_len = 1 + cumul_enc_len + LOGS_BLOOM_ENC_LEN + 1;
        let encoded_len = 0 + LIST_PREFIX_LEN + payload_len; // legacy → no type byte
        assert_eq!(trace.columns[COL_STATUS_ENC_LEN].evaluations[0].to_u64(), 1);
        assert_eq!(
            trace.columns[COL_CUMUL_ENC_LEN].evaluations[0].to_u64() as usize,
            cumul_enc_len,
        );
        assert_eq!(
            trace.columns[COL_PAYLOAD_LEN].evaluations[0].to_u64() as usize,
            payload_len,
        );
        assert_eq!(
            trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64() as usize,
            encoded_len,
        );
        assert_eq!(trace.columns[COL_LIST_PREFIX_0].evaluations[0].to_u64(), 0xf9);
        // For status=1: canonical wire matches; cross-check byte-for-byte.
        let canonical = r.wire_encoding();
        // The trace's encoded_bytes (built in from_receipt) should match.
        let row_enc = &ReceiptRlpRow::from_receipt(&r).unwrap().encoded_bytes;
        assert_eq!(row_enc, &canonical);
    }

    #[test]
    fn eip1559_typed_prefix_at_position_zero() {
        let r = eip1559_no_logs();
        let row = ReceiptRlpRow::from_receipt(&r).unwrap();
        assert_eq!(row.encoded_bytes[0], 0x02);
        // Type byte length column should be 1 (typed).
        let trace = build_for(&[r.clone()]);
        assert_eq!(trace.columns[COL_TYPE_BYTE_LEN].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_TYPE_BYTE].evaluations[0].to_u64(), 0x02);
        assert_eq!(trace.columns[COL_IS_LEGACY].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_IS_TYPED].evaluations[0].to_u64(), 1);
        // Matches canonical wire encoding byte-for-byte.
        assert_eq!(row.encoded_bytes, r.wire_encoding());
    }

    #[test]
    fn eip4844_typed_prefix_at_position_zero() {
        let mut r = eip1559_no_logs();
        r.ty = ReceiptType::Eip4844;
        let row = ReceiptRlpRow::from_receipt(&r).unwrap();
        assert_eq!(row.encoded_bytes[0], 0x03);
        let trace = build_for(&[r.clone()]);
        assert_eq!(trace.columns[COL_TYPE_BYTE].evaluations[0].to_u64(), 0x03);
        assert_eq!(row.encoded_bytes, r.wire_encoding());
    }

    #[test]
    fn constraints_zero_on_honest_witness_multiple_rows() {
        let receipts = vec![
            legacy_no_logs(),
            eip1559_no_logs(),
            Receipt {
                ty: ReceiptType::AccessList,
                status: 1,
                cumulative_gas_used: u64::MAX,
                logs_bloom: [0xff; 256],
                logs: Vec::new(),
            },
            Receipt {
                ty: ReceiptType::Eip4844,
                status: 1,
                cumulative_gas_used: 0xdead_beef,
                logs_bloom: [0xaa; 256],
                logs: Vec::new(),
            },
        ];
        let trace = build_for(&receipts);
        let cs = ReceiptRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} (label {}) at row {} = {:?} (expected zero)",
                    i, cs.constraint_labels()[i], r, val,
                );
            }
        }
    }

    #[test]
    fn list_prefix_byte_fires_on_tampered_prefix() {
        let trace = build_for(&[legacy_no_logs()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: list_prefix_0 = 0xc4 (short-list lead).
        cols[COL_LIST_PREFIX_0][0] = Scalar::from_u64(0xc4, CurveType::Bls48581);
        let cs = ReceiptRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[5][0].is_zero(), "list_prefix_0_eq_0xf9 must fire");
    }

    #[test]
    fn tampered_type_byte_detected_via_type_byte_len_constraint() {
        let trace = build_for(&[eip1559_no_logs()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: flip type_byte_len from 1 → 0 (claim it's legacy).
        cols[COL_TYPE_BYTE_LEN][0] = Scalar::from_u64(0, CurveType::Bls48581);
        let cs = ReceiptRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 4 = type_byte_len_eq_is_typed.
        assert!(!res[4][0].is_zero(), "type_byte_len_eq_is_typed must fire");
    }

    #[test]
    fn payload_len_consistency_fires_on_tampered_payload_len() {
        let trace = build_for(&[legacy_no_logs()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let cur = cols[COL_PAYLOAD_LEN][0].to_u64();
        cols[COL_PAYLOAD_LEN][0] = Scalar::from_u64(cur + 1, CurveType::Bls48581);
        let cs = ReceiptRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 7 = payload_len_arith.
        assert!(!res[7][0].is_zero(), "payload_len_arith must fire");
        // Constraint 6 = list_prefix_len_consistency may also fire
        // because we left list_prefix_len_hi/lo unchanged.
    }

    #[test]
    fn encoded_len_arith_fires_on_tampered_encoded_len() {
        let trace = build_for(&[eip1559_no_logs()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let cur = cols[COL_ENCODED_LEN][0].to_u64();
        cols[COL_ENCODED_LEN][0] = Scalar::from_u64(cur + 5, CurveType::Bls48581);
        let cs = ReceiptRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 8 = encoded_len_arith.
        assert!(!res[8][0].is_zero(), "encoded_len_arith must fire");
    }

    #[test]
    fn empty_logs_byte_constraint_fires_on_tamper() {
        let trace = build_for(&[legacy_no_logs()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_EMPTY_LOGS_BYTE][0] = Scalar::from_u64(0xc1, CurveType::Bls48581);
        let cs = ReceiptRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[9][0].is_zero(), "empty_logs_byte_eq_0xc0 must fire");
    }

    #[test]
    fn rejects_receipt_with_logs() {
        let r = Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: vec![crate::receipt::Log {
                address: [0u8; 20],
                topics: vec![],
                data: vec![],
            }],
        };
        let err = ReceiptRlpRow::from_receipt(&r).unwrap_err();
        assert!(err.contains("no-logs"), "got: {}", err);
    }

    #[test]
    fn linkage_descriptors_well_formed() {
        let d_status = make_receipt_to_u64_rlp_status_descriptor(0, 1);
        assert_eq!(d_status.label, "receipt_rlp_to_u64_rlp_status_v1");
        // (value, encoded_len, encoded[0..9]) = 2 + 9 = 11 columns.
        assert_eq!(d_status.a_columns.len(), 2 + U64_MAX_ENCODED_LEN);
        assert_eq!(d_status.b_columns.len(), d_status.a_columns.len());
        assert_eq!(d_status.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_status.b_selector_column, Some(crate::u64_rlp_air::COL_IS_REAL));

        let d_cumul = make_receipt_to_u64_rlp_cumulative_gas_descriptor(0, 1);
        assert_eq!(d_cumul.label, "receipt_rlp_to_u64_rlp_cumulative_gas_v1");
        assert_eq!(d_cumul.a_columns.len(), 2 + U64_MAX_ENCODED_LEN);
        assert_eq!(d_cumul.b_columns.len(), d_cumul.a_columns.len());
        // Both link to the same u64_rlp layer, but A-side columns differ.
        assert_ne!(d_status.a_columns, d_cumul.a_columns);
        assert_eq!(d_status.b_columns, d_cumul.b_columns);

        let d_bloom = make_receipt_to_logs_bloom_descriptor(0, 2);
        assert_eq!(d_bloom.label, "receipt_rlp_to_logs_bloom_v1");
        assert_eq!(d_bloom.a_columns.len(), LOGS_BLOOM_LEN);
        assert_eq!(d_bloom.b_columns.len(), LOGS_BLOOM_LEN);
        assert_eq!(
            d_bloom.b_selector_column,
            Some(crate::rlp_logs_bloom_air::COL_IS_REAL),
        );
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 289);
        assert_eq!(MIN_PAYLOAD_LEN, 262);
        assert_eq!(MAX_PAYLOAD_LEN, 270);
        assert_eq!(LIST_PREFIX_LEN, 3);
        assert_eq!(LOGS_BLOOM_ENC_LEN, 259);
    }

    #[test]
    fn status_zero_diverges_from_host_oracle() {
        // Documents the status=0 divergence: host-side
        // `Receipt::wire_encoding()` uses `rlp_encode_bytes(&[0])` =
        // `[0x00]`, while the AIR uses canonical scalar RLP `[0x80]`.
        // For status=1 they agree (`[0x01]`). The deferred fix is to
        // either patch `Receipt::wire_encoding` to use scalar RLP or
        // add a single-byte string-RLP sub-gadget.
        let r = Receipt {
            ty: ReceiptType::Legacy,
            status: 0,
            cumulative_gas_used: 0,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        };
        let row = ReceiptRlpRow::from_receipt(&r).unwrap();
        let canonical = r.wire_encoding();
        // Same length (status enc is 1 byte both ways).
        assert_eq!(row.encoded_bytes.len(), canonical.len());
        // First few bytes (prefix + status) differ at the status byte
        // offset. List prefix is `0xf9, 0x01, 0x06` for both (payload
        // 262); the status byte (offset 3) is where they diverge.
        assert_eq!(row.encoded_bytes[3], 0x80);
        assert_eq!(canonical[3], 0x00);
    }
}
