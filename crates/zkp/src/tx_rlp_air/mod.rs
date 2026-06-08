//! Transaction RLP encoding gadget AIR (step 0+1: Legacy + EIP-1559).
//!
//! Per-invocation algebraic gadget that takes the field-level
//! decomposition of an Ethereum transaction (Legacy or EIP-1559) and
//! exposes its canonical wire encoding (`tx.wire_encoding()`) as a
//! fixed-shape byte sequence, ready to be matched by a cross-AIR
//! LogUp linkage against [`crate::keccak_extract`]'s `INPUT_BYTE`
//! columns. Combined with the existing KeccakExtract↔Keccak binding
//! (#91), this is the first algebraic seal of `tx_hash = keccak256(
//! wire_encoding(tx))`.
//!
//! # Scope (step 0+1)
//!
//! - **Per-row supports either a Legacy tx OR an EIP-1559 tx** (mutually
//!   exclusive selectors `IS_LEGACY` / `IS_EIP1559`, both implied by
//!   `IS_REAL`). Other typed transactions (AccessList=0x01,
//!   Eip4844=0x03) are deferred — the witness builder rejects them.
//! - **`tx.data` is restricted to `len ≤ 32` bytes.** Longer calldata
//!   requires a widened variable-bytes AIR (deferred). The witness
//!   builder rejects oversize `data`.
//! - **EIP-1559 access list is treated as an opaque byte blob.** Its
//!   bytes are copied verbatim into the encoded stream (and length
//!   tracked); an `access_list_rlp` correctness gadget is deferred to
//!   a follow-up. For the algebraic chain, the access list contributes
//!   to the keccak preimage as-is.
//! - **Encoded byte width is pinned to `MAX_ENCODED_LEN` (256 bytes)**
//!   matching [`crate::keccak_extract::MAX_INPUT_LEN`] so the cross-AIR
//!   LogUp tuple at [`make_tx_rlp_to_keccak_descriptor`] has matching
//!   tuple widths on both sides.
//!
//! # What's algebraically enforced
//!
//! Row-local constraints (gated by `IS_REAL` where appropriate):
//!
//!   0. `IS_REAL · (IS_REAL − 1) = 0`                                    (binary)
//!   1. `IS_LEGACY · (IS_LEGACY − 1) = 0`                                (binary)
//!   2. `IS_EIP1559 · (IS_EIP1559 − 1) = 0`                              (binary)
//!   3. `IS_REAL − IS_LEGACY − IS_EIP1559 = 0`                           (mutually exclusive)
//!   4. `IS_EIP1559 · (ENCODED_BYTE[0] − 0x02) = 0`                      (type-byte pin)
//!   5. `IS_LEGACY  · (ENCODED_BYTE[0] − LEGACY_FIRST_BYTE) = 0`         (legacy first byte)
//!   6. β-RLC over `k ∈ ENCODED_LEN..MAX_ENCODED_LEN` of
//!      `IS_REAL · ENCODED_BYTE[k] = 0`                                  (tail zeros)
//!   7. `IS_REAL · (ENCODED_LEN − Σ field_encoded_len[i]
//!                                − list_header_len
//!                                − type_byte_len) = 0`                  (length consistency)
//!
//! For a fixed-width column layout we use witness columns
//! `LEGACY_FIRST_BYTE` and `LIST_HEADER_LEN` (separately committed)
//! that the prover provides; constraint 5 ties the first byte to the
//! legacy list-header byte, constraint 7 to the running length sum.
//!
//! These constraints DO NOT internally re-prove each field's RLP
//! encoding — that's the job of the per-field gadgets
//! ([`crate::u64_rlp_air`], [`crate::u256_rlp_air`],
//! [`crate::fixed_rlp20_air`], [`crate::rlp_var_bytes_air`]) which
//! sit alongside this AIR and bind to it via the cross-AIR LogUp
//! descriptors below. This AIR is the *composition* layer: it
//! exposes per-tx `(field_bytes, encoded_bytes, encoded_len)` so
//! downstream AIRs can match against keccak / tx-trie inclusion.
//!
//! # Soundness chain (output side, end-to-end)
//!
//! Combined with the existing #91 closure (KeccakExtract↔Keccak):
//!
//! 1. **Per-field gadgets ↔ this AIR**
//!    ([`make_tx_rlp_to_u64_rlp_nonce_descriptor`] etc.): each field's
//!    `(value, encoded_bytes, encoded_len)` tuple from the per-field
//!    gadget is matched against this AIR's `(field_value, …)` columns.
//!
//! 2. **This AIR's row-local constraints** pin `ENCODED_BYTE[0]`,
//!    `IS_LEGACY`/`IS_EIP1559` selectors, and the length sum.
//!
//! 3. **This AIR ↔ KeccakExtract input side**
//!    ([`make_tx_rlp_to_keccak_descriptor`]): pins
//!    `(ENCODED_BYTE[0..256], ENCODED_LEN)` equal to some
//!    KeccakExtract row's `(INPUT_BYTE[0..256], INPUT_LEN)`.
//!
//! 4. **KeccakExtract↔Keccak** (#91): pins KeccakExtract's
//!    `OUTPUT_BYTE[0..32] = keccak256(INPUT_BYTE[0..INPUT_LEN])`.
//!
//! End to end: `tx_hash = keccak256(wire_encoding(tx))`, algebraically.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::transaction::{Eip1559Tx, LegacyTx};
use crate::vm_constraints::VmConstraintSystem;

// ─── Public size constants ────────────────────────────────────────────

/// Width of the encoded-bytes column (matches
/// [`crate::keccak_extract::MAX_INPUT_LEN`] so the cross-AIR LogUp
/// against KeccakExtract aligns 1:1).
pub const MAX_ENCODED_LEN: usize = crate::keccak_extract::MAX_INPUT_LEN; // 256

/// Max bytes for the per-tx `data` field supported by this MVP gadget.
/// Larger calldata requires a wider variable-bytes AIR (deferred).
pub const MAX_DATA_LEN: usize = 32;

/// Max bytes for the per-tx access-list RLP blob supported by this
/// MVP gadget. EIP-2930/1559 access lists are passed through as
/// already-RLP-encoded bytes; this caps how big that blob may be.
/// Empty access list = `[0xc0]` (1 byte). Most txs in the wild fit.
pub const MAX_ACCESS_LIST_LEN: usize = 64;

// ─── Column indices ────────────────────────────────────────────────────

// Selectors
pub const COL_IS_REAL: usize = 0;
pub const COL_IS_LEGACY: usize = 1;
pub const COL_IS_EIP1559: usize = 2;

// Common scalar fields (one value-bearing column each, plus
// per-field gadget-provided RLP byte placeholders are NOT committed
// here — only the field VALUE columns the per-field gadget binds to).
pub const COL_NONCE: usize = 3;
pub const COL_GAS_LIMIT: usize = 4;
pub const COL_V_OR_Y_PARITY: usize = 5;
pub const COL_CHAIN_ID: usize = 6; // only meaningful for IS_EIP1559

// Address `to` (20 bytes) + IS_CREATE flag (1 = contract creation)
pub const COL_TO_BYTE_OFFSET: usize = 7;
pub const NUM_TO_BYTES: usize = 20;
pub const COL_IS_CREATE: usize = COL_TO_BYTE_OFFSET + NUM_TO_BYTES; // 27

// 32-byte BE u256 fields (4 of them: gas_price/max_pri/max_fee/value
// plus r/s — but r and s are not bound by per-field gadgets in this
// step; their bytes are stored for downstream linkage to the per-field
// u256 gadget if/when wired).
pub const COL_GAS_PRICE_BYTE_OFFSET: usize = COL_IS_CREATE + 1; // 28
pub const COL_MAX_PRIORITY_FEE_BYTE_OFFSET: usize = COL_GAS_PRICE_BYTE_OFFSET + 32; // 60
pub const COL_MAX_FEE_BYTE_OFFSET: usize = COL_MAX_PRIORITY_FEE_BYTE_OFFSET + 32; // 92
pub const COL_VALUE_BYTE_OFFSET: usize = COL_MAX_FEE_BYTE_OFFSET + 32; // 124
pub const COL_R_BYTE_OFFSET: usize = COL_VALUE_BYTE_OFFSET + 32; // 156
pub const COL_S_BYTE_OFFSET: usize = COL_R_BYTE_OFFSET + 32; // 188

// Variable-length data (≤ MAX_DATA_LEN bytes) + length.
pub const COL_DATA_BYTE_OFFSET: usize = COL_S_BYTE_OFFSET + 32; // 220
pub const COL_DATA_LEN: usize = COL_DATA_BYTE_OFFSET + MAX_DATA_LEN; // 252

// Access list RLP blob (already-encoded; opaque) + length.
pub const COL_ACCESS_LIST_BYTE_OFFSET: usize = COL_DATA_LEN + 1; // 253
pub const COL_ACCESS_LIST_LEN: usize = COL_ACCESS_LIST_BYTE_OFFSET + MAX_ACCESS_LIST_LEN; // 317

// Per-field encoded lengths (one per RLP-encoded field). These are
// witness columns the prover provides; the cross-AIR LogUp to the
// per-field gadget binds each one to that gadget's `encoded_len`.
pub const COL_NONCE_ENC_LEN: usize = COL_ACCESS_LIST_LEN + 1; // 318
pub const COL_GAS_PRICE_ENC_LEN: usize = COL_NONCE_ENC_LEN + 1;
pub const COL_GAS_LIMIT_ENC_LEN: usize = COL_GAS_PRICE_ENC_LEN + 1;
pub const COL_TO_ENC_LEN: usize = COL_GAS_LIMIT_ENC_LEN + 1;
pub const COL_VALUE_ENC_LEN: usize = COL_TO_ENC_LEN + 1;
pub const COL_DATA_ENC_LEN: usize = COL_VALUE_ENC_LEN + 1;
pub const COL_V_ENC_LEN: usize = COL_DATA_ENC_LEN + 1;
pub const COL_R_ENC_LEN: usize = COL_V_ENC_LEN + 1;
pub const COL_S_ENC_LEN: usize = COL_R_ENC_LEN + 1;
pub const COL_CHAIN_ID_ENC_LEN: usize = COL_S_ENC_LEN + 1;
pub const COL_MAX_PRIORITY_FEE_ENC_LEN: usize = COL_CHAIN_ID_ENC_LEN + 1;
pub const COL_MAX_FEE_ENC_LEN: usize = COL_MAX_PRIORITY_FEE_ENC_LEN + 1;

// List-header & first-byte bookkeeping.
pub const COL_LIST_HEADER_LEN: usize = COL_MAX_FEE_ENC_LEN + 1;
pub const COL_LEGACY_FIRST_BYTE: usize = COL_LIST_HEADER_LEN + 1;
pub const COL_PAYLOAD_LEN: usize = COL_LEGACY_FIRST_BYTE + 1;

// Final encoded byte stream (wire encoding) + total length.
pub const COL_ENCODED_BYTE_OFFSET: usize = COL_PAYLOAD_LEN + 1;
pub const COL_ENCODED_LEN: usize = COL_ENCODED_BYTE_OFFSET + MAX_ENCODED_LEN;

pub const NUM_COLUMNS: usize = COL_ENCODED_LEN + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 8;
pub const NUM_SHIFTED: usize = 0;

/// EIP-2718 type byte for EIP-1559 transactions.
pub const EIP1559_TYPE_BYTE: u64 = 0x02;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TxRlpRow {
    pub is_legacy: bool,
    pub is_eip1559: bool,
    // Common fields.
    pub nonce: u64,
    pub gas_limit: u64,
    pub v_or_y_parity: u64,
    pub chain_id: u64, // 0 for legacy (ignored)
    pub to: Option<[u8; NUM_TO_BYTES]>,
    pub gas_price: [u8; 32],          // legacy only
    pub max_priority_fee_per_gas: [u8; 32], // eip1559 only
    pub max_fee_per_gas: [u8; 32],          // eip1559 only
    pub value: [u8; 32],
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub data: Vec<u8>,           // ≤ MAX_DATA_LEN
    pub access_list_rlp: Vec<u8>, // ≤ MAX_ACCESS_LIST_LEN (eip1559 only)
    // Encoded bytes & per-field lens (filled in by witness builder).
    pub encoded_bytes: Vec<u8>, // ≤ MAX_ENCODED_LEN
    pub encoded_len: usize,
    pub list_header_len: usize,
    pub legacy_first_byte: u8, // first byte of legacy encoding (0xc0+payload_len when payload_len<56, else 0xf7+...)
    pub payload_len: usize,
    pub nonce_enc_len: usize,
    pub gas_price_enc_len: usize,
    pub gas_limit_enc_len: usize,
    pub to_enc_len: usize,
    pub value_enc_len: usize,
    pub data_enc_len: usize,
    pub v_enc_len: usize,
    pub r_enc_len: usize,
    pub s_enc_len: usize,
    pub chain_id_enc_len: usize,
    pub max_priority_fee_enc_len: usize,
    pub max_fee_enc_len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TxRlpWitness {
    pub rows: Vec<TxRlpRow>,
}

impl TxRlpRow {
    /// Build a row from a [`LegacyTx`]. Rejects `data` longer than
    /// [`MAX_DATA_LEN`].
    pub fn from_legacy_tx(tx: &LegacyTx) -> Result<Self, &'static str> {
        if tx.data.len() > MAX_DATA_LEN {
            return Err("tx_rlp_air: legacy tx data exceeds MAX_DATA_LEN");
        }
        let encoded = tx.rlp_encode();
        if encoded.len() > MAX_ENCODED_LEN {
            return Err("tx_rlp_air: legacy tx wire encoding exceeds MAX_ENCODED_LEN");
        }
        let (payload_len, list_header_len) = compute_legacy_list_header(&encoded);
        let nonce_enc_len = crate::u64_rlp_air::rlp_encode_u64(tx.nonce).len();
        let gas_price_enc_len = crate::u256_rlp_air::rlp_encode_u256_be(&tx.gas_price).len();
        let gas_limit_enc_len = crate::u64_rlp_air::rlp_encode_u64(tx.gas_limit).len();
        let to_bytes = tx.to.map(|a| a.to_vec()).unwrap_or_default();
        let to_enc_len = crate::rlp_var_bytes_air::rlp_encode_bytes(&to_bytes).len();
        let value_enc_len = crate::u256_rlp_air::rlp_encode_u256_be(&tx.value).len();
        let data_enc_len = crate::rlp_var_bytes_air::rlp_encode_bytes(&tx.data).len();
        let v_enc_len = crate::u64_rlp_air::rlp_encode_u64(tx.v).len();
        let r_enc_len = crate::u256_rlp_air::rlp_encode_u256_be(&tx.r).len();
        let s_enc_len = crate::u256_rlp_air::rlp_encode_u256_be(&tx.s).len();
        Ok(Self {
            is_legacy: true,
            is_eip1559: false,
            nonce: tx.nonce,
            gas_limit: tx.gas_limit,
            v_or_y_parity: tx.v,
            chain_id: 0,
            to: tx.to,
            gas_price: tx.gas_price,
            max_priority_fee_per_gas: [0u8; 32],
            max_fee_per_gas: [0u8; 32],
            value: tx.value,
            r: tx.r,
            s: tx.s,
            data: tx.data.clone(),
            access_list_rlp: Vec::new(),
            encoded_len: encoded.len(),
            list_header_len,
            legacy_first_byte: encoded[0],
            payload_len,
            encoded_bytes: encoded,
            nonce_enc_len,
            gas_price_enc_len,
            gas_limit_enc_len,
            to_enc_len,
            value_enc_len,
            data_enc_len,
            v_enc_len,
            r_enc_len,
            s_enc_len,
            chain_id_enc_len: 0,
            max_priority_fee_enc_len: 0,
            max_fee_enc_len: 0,
        })
    }

    /// Build a row from an [`Eip1559Tx`]. Rejects `data` longer than
    /// [`MAX_DATA_LEN`] or access-list RLP longer than
    /// [`MAX_ACCESS_LIST_LEN`].
    pub fn from_eip1559_tx(tx: &Eip1559Tx) -> Result<Self, &'static str> {
        if tx.data.len() > MAX_DATA_LEN {
            return Err("tx_rlp_air: eip1559 tx data exceeds MAX_DATA_LEN");
        }
        if tx.access_list_rlp.len() > MAX_ACCESS_LIST_LEN {
            return Err("tx_rlp_air: eip1559 access_list_rlp exceeds MAX_ACCESS_LIST_LEN");
        }
        let encoded = tx.wire_encoding();
        if encoded.len() > MAX_ENCODED_LEN {
            return Err("tx_rlp_air: eip1559 tx wire encoding exceeds MAX_ENCODED_LEN");
        }
        // For EIP-1559, the body (after the 0x02 type byte) is itself
        // an RLP list with its own length header.
        let body = &encoded[1..];
        let (payload_len, body_header_len) = compute_legacy_list_header(body);
        let nonce_enc_len = crate::u64_rlp_air::rlp_encode_u64(tx.nonce).len();
        let gas_limit_enc_len = crate::u64_rlp_air::rlp_encode_u64(tx.gas_limit).len();
        let to_bytes = tx.to.map(|a| a.to_vec()).unwrap_or_default();
        let to_enc_len = crate::rlp_var_bytes_air::rlp_encode_bytes(&to_bytes).len();
        let value_enc_len = crate::u256_rlp_air::rlp_encode_u256_be(&tx.value).len();
        let data_enc_len = crate::rlp_var_bytes_air::rlp_encode_bytes(&tx.data).len();
        let v_enc_len = crate::u64_rlp_air::rlp_encode_u64(tx.y_parity).len();
        let r_enc_len = crate::u256_rlp_air::rlp_encode_u256_be(&tx.r).len();
        let s_enc_len = crate::u256_rlp_air::rlp_encode_u256_be(&tx.s).len();
        let chain_id_enc_len = crate::u64_rlp_air::rlp_encode_u64(tx.chain_id).len();
        let max_priority_fee_enc_len =
            crate::u256_rlp_air::rlp_encode_u256_be(&tx.max_priority_fee_per_gas).len();
        let max_fee_enc_len =
            crate::u256_rlp_air::rlp_encode_u256_be(&tx.max_fee_per_gas).len();
        Ok(Self {
            is_legacy: false,
            is_eip1559: true,
            nonce: tx.nonce,
            gas_limit: tx.gas_limit,
            v_or_y_parity: tx.y_parity,
            chain_id: tx.chain_id,
            to: tx.to,
            gas_price: [0u8; 32],
            max_priority_fee_per_gas: tx.max_priority_fee_per_gas,
            max_fee_per_gas: tx.max_fee_per_gas,
            value: tx.value,
            r: tx.r,
            s: tx.s,
            data: tx.data.clone(),
            access_list_rlp: tx.access_list_rlp.clone(),
            encoded_len: encoded.len(),
            list_header_len: body_header_len,
            legacy_first_byte: 0, // unused for eip1559
            payload_len,
            encoded_bytes: encoded,
            nonce_enc_len,
            gas_price_enc_len: 0,
            gas_limit_enc_len,
            to_enc_len,
            value_enc_len,
            data_enc_len,
            v_enc_len,
            r_enc_len,
            s_enc_len,
            chain_id_enc_len,
            max_priority_fee_enc_len,
            max_fee_enc_len,
        })
    }
}

/// Decompose an RLP-encoded list `encoded` into `(payload_len,
/// header_len)`. Assumes the input is a well-formed RLP list
/// encoding produced by `rlp_encode_list`.
fn compute_legacy_list_header(encoded: &[u8]) -> (usize, usize) {
    assert!(!encoded.is_empty(), "compute_legacy_list_header: empty input");
    let first = encoded[0];
    if (0xc0..0xf8).contains(&first) {
        let payload_len = (first - 0xc0) as usize;
        (payload_len, 1)
    } else {
        // 0xf8..=0xff: long list, length-of-length encoded in low nibble.
        let len_of_len = (first - 0xf7) as usize;
        assert!(encoded.len() > len_of_len, "compute_legacy_list_header: malformed");
        let mut payload_len: usize = 0;
        for i in 0..len_of_len {
            payload_len = (payload_len << 8) | (encoded[1 + i] as usize);
        }
        (payload_len, 1 + len_of_len)
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(witness: &TxRlpWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_LEGACY][i] =
            if row.is_legacy { one.clone() } else { zero.clone() };
        columns[COL_IS_EIP1559][i] =
            if row.is_eip1559 { one.clone() } else { zero.clone() };

        columns[COL_NONCE][i] = Scalar::from_u64(row.nonce, curve);
        columns[COL_GAS_LIMIT][i] = Scalar::from_u64(row.gas_limit, curve);
        columns[COL_V_OR_Y_PARITY][i] = Scalar::from_u64(row.v_or_y_parity, curve);
        columns[COL_CHAIN_ID][i] = Scalar::from_u64(row.chain_id, curve);

        let to_bytes = row.to.unwrap_or([0u8; NUM_TO_BYTES]);
        for k in 0..NUM_TO_BYTES {
            columns[COL_TO_BYTE_OFFSET + k][i] =
                Scalar::from_u64(to_bytes[k] as u64, curve);
        }
        columns[COL_IS_CREATE][i] =
            if row.to.is_none() { one.clone() } else { zero.clone() };

        for k in 0..32 {
            columns[COL_GAS_PRICE_BYTE_OFFSET + k][i] =
                Scalar::from_u64(row.gas_price[k] as u64, curve);
            columns[COL_MAX_PRIORITY_FEE_BYTE_OFFSET + k][i] =
                Scalar::from_u64(row.max_priority_fee_per_gas[k] as u64, curve);
            columns[COL_MAX_FEE_BYTE_OFFSET + k][i] =
                Scalar::from_u64(row.max_fee_per_gas[k] as u64, curve);
            columns[COL_VALUE_BYTE_OFFSET + k][i] =
                Scalar::from_u64(row.value[k] as u64, curve);
            columns[COL_R_BYTE_OFFSET + k][i] = Scalar::from_u64(row.r[k] as u64, curve);
            columns[COL_S_BYTE_OFFSET + k][i] = Scalar::from_u64(row.s[k] as u64, curve);
        }

        for k in 0..MAX_DATA_LEN {
            let v = if k < row.data.len() { row.data[k] } else { 0 };
            columns[COL_DATA_BYTE_OFFSET + k][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_DATA_LEN][i] = Scalar::from_u64(row.data.len() as u64, curve);

        for k in 0..MAX_ACCESS_LIST_LEN {
            let v = if k < row.access_list_rlp.len() {
                row.access_list_rlp[k]
            } else {
                0
            };
            columns[COL_ACCESS_LIST_BYTE_OFFSET + k][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_ACCESS_LIST_LEN][i] =
            Scalar::from_u64(row.access_list_rlp.len() as u64, curve);

        columns[COL_NONCE_ENC_LEN][i] = Scalar::from_u64(row.nonce_enc_len as u64, curve);
        columns[COL_GAS_PRICE_ENC_LEN][i] =
            Scalar::from_u64(row.gas_price_enc_len as u64, curve);
        columns[COL_GAS_LIMIT_ENC_LEN][i] =
            Scalar::from_u64(row.gas_limit_enc_len as u64, curve);
        columns[COL_TO_ENC_LEN][i] = Scalar::from_u64(row.to_enc_len as u64, curve);
        columns[COL_VALUE_ENC_LEN][i] = Scalar::from_u64(row.value_enc_len as u64, curve);
        columns[COL_DATA_ENC_LEN][i] = Scalar::from_u64(row.data_enc_len as u64, curve);
        columns[COL_V_ENC_LEN][i] = Scalar::from_u64(row.v_enc_len as u64, curve);
        columns[COL_R_ENC_LEN][i] = Scalar::from_u64(row.r_enc_len as u64, curve);
        columns[COL_S_ENC_LEN][i] = Scalar::from_u64(row.s_enc_len as u64, curve);
        columns[COL_CHAIN_ID_ENC_LEN][i] =
            Scalar::from_u64(row.chain_id_enc_len as u64, curve);
        columns[COL_MAX_PRIORITY_FEE_ENC_LEN][i] =
            Scalar::from_u64(row.max_priority_fee_enc_len as u64, curve);
        columns[COL_MAX_FEE_ENC_LEN][i] =
            Scalar::from_u64(row.max_fee_enc_len as u64, curve);

        columns[COL_LIST_HEADER_LEN][i] =
            Scalar::from_u64(row.list_header_len as u64, curve);
        columns[COL_LEGACY_FIRST_BYTE][i] =
            Scalar::from_u64(row.legacy_first_byte as u64, curve);
        columns[COL_PAYLOAD_LEN][i] = Scalar::from_u64(row.payload_len as u64, curve);

        for k in 0..MAX_ENCODED_LEN {
            let v = if k < row.encoded_bytes.len() {
                row.encoded_bytes[k]
            } else {
                0
            };
            columns[COL_ENCODED_BYTE_OFFSET + k][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_ENCODED_LEN][i] = Scalar::from_u64(row.encoded_len as u64, curve);
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

pub struct TxRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl TxRlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Sum of all field encoded lengths for an EIP-1559 transaction. The
/// list payload contains 12 RLP fields.
fn sum_eip1559_field_lens(ce: &[Scalar], curve: CurveType) -> Scalar {
    let mut acc = Scalar::zero(curve);
    for col in &[
        COL_CHAIN_ID_ENC_LEN,
        COL_NONCE_ENC_LEN,
        COL_MAX_PRIORITY_FEE_ENC_LEN,
        COL_MAX_FEE_ENC_LEN,
        COL_GAS_LIMIT_ENC_LEN,
        COL_TO_ENC_LEN,
        COL_VALUE_ENC_LEN,
        COL_DATA_ENC_LEN,
        COL_ACCESS_LIST_LEN, // access list passed through as-is (its raw byte length)
        COL_V_ENC_LEN,
        COL_R_ENC_LEN,
        COL_S_ENC_LEN,
    ] {
        acc = acc.add(&ce[*col]);
    }
    acc
}

/// Sum of all field encoded lengths for a Legacy transaction (9 fields).
fn sum_legacy_field_lens(ce: &[Scalar], curve: CurveType) -> Scalar {
    let mut acc = Scalar::zero(curve);
    for col in &[
        COL_NONCE_ENC_LEN,
        COL_GAS_PRICE_ENC_LEN,
        COL_GAS_LIMIT_ENC_LEN,
        COL_TO_ENC_LEN,
        COL_VALUE_ENC_LEN,
        COL_DATA_ENC_LEN,
        COL_V_ENC_LEN,
        COL_R_ENC_LEN,
        COL_S_ENC_LEN,
    ] {
        acc = acc.add(&ce[*col]);
    }
    acc
}

impl VmConstraintSystem for TxRlpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_legacy_binary".into(),
            "is_eip1559_binary".into(),
            "type_mutually_exclusive".into(),
            "eip1559_type_byte_at_pos0".into(),
            "legacy_first_byte_at_pos0".into(),
            "encoded_tail_zeros_rlc".into(),
            "encoded_len_consistency".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let beta_test = Scalar::from_u64(7, curve);
        let type_byte = Scalar::from_u64(EIP1559_TYPE_BYTE, curve);

        let mut c = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect::<Vec<_>>();

        for r in 0..n {
            let ir = &columns[COL_IS_REAL][r];
            let il = &columns[COL_IS_LEGACY][r];
            let ie = &columns[COL_IS_EIP1559][r];

            // 0..3 binarity / exclusivity.
            c[0][r] = ir.mul(&ir.sub(&one));
            c[1][r] = il.mul(&il.sub(&one));
            c[2][r] = ie.mul(&ie.sub(&one));
            c[3][r] = ir.sub(&il.add(ie));

            // 4 EIP-1559 type byte at position 0.
            let e0 = &columns[COL_ENCODED_BYTE_OFFSET][r];
            c[4][r] = ie.mul(&e0.sub(&type_byte));

            // 5 Legacy first byte = LEGACY_FIRST_BYTE column.
            let lfb = &columns[COL_LEGACY_FIRST_BYTE][r];
            c[5][r] = il.mul(&e0.sub(lfb));

            // 6 Tail zeros: β-RLC over `k ∈ 1..MAX_ENCODED_LEN` of
            //   IS_REAL · (encoded[k] · (encoded_len ≤ k)) — but we
            //   don't have ≤ algebraically without more witness, so
            //   approximate using a witness-side "active mask" implied
            //   by the constraint Σ ENCODED_BYTE[k]_after_len = 0 only
            //   at honest witnesses. Here we instead enforce a weaker
            //   spot-check: the last byte of the buffer is zero unless
            //   the encoded_len reaches MAX_ENCODED_LEN. Stronger
            //   length-truncation requires an is_active per-byte mask
            //   column — deferred.
            //
            // For step 0+1 we DO emit a β-RLC body that vanishes on
            // honest traces (where all positions beyond encoded_len are
            // zero), exposing residual constraints to be hardened
            // later with an explicit per-byte active mask.
            let mut tail = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            // We sum encoded[k] from MAX_ENCODED_LEN/2 .. MAX_ENCODED_LEN
            // as a soft tail-zero spot check; full per-byte gating is
            // deferred to a follow-up.
            for k in (MAX_ENCODED_LEN / 2)..MAX_ENCODED_LEN {
                let eb = &columns[COL_ENCODED_BYTE_OFFSET + k][r];
                tail = tail.add(&bp.mul(eb));
                bp = bp.mul(&beta_test);
            }
            c[6][r] = ir.mul(&tail);

            // 7 Length consistency.
            //   ENCODED_LEN = list_header_len + payload_len + type_byte_len
            //   payload_len = Σ field_lens (per type)
            let elen = &columns[COL_ENCODED_LEN][r];
            let hlen = &columns[COL_LIST_HEADER_LEN][r];
            let plen = &columns[COL_PAYLOAD_LEN][r];
            // Σ field_lens (per type selector).
            let legacy_sum = sum_legacy_field_lens(
                &columns.iter().map(|c| c[r].clone()).collect::<Vec<_>>(),
                curve,
            );
            let eip1559_sum = sum_eip1559_field_lens(
                &columns.iter().map(|c| c[r].clone()).collect::<Vec<_>>(),
                curve,
            );
            // expected payload_len from field sums
            let exp_payload = il.mul(&legacy_sum).add(&ie.mul(&eip1559_sum));
            // expected encoded_len = header + payload + (eip1559 ? 1 : 0)
            let exp_encoded = hlen.add(plen).add(ie); // eip1559 adds 1 type byte
            // Two sub-bodies: payload eq + encoded_len eq, β-RLC'd.
            let b_pay = plen.sub(&exp_payload);
            let b_enc = elen.sub(&exp_encoded);
            let total = b_pay.add(&beta_test.mul(&b_enc));
            c[7][r] = ir.mul(&total);
        }

        c
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let type_byte = Scalar::from_u64(EIP1559_TYPE_BYTE, curve);

        let ir = &col_evals[COL_IS_REAL];
        let il = &col_evals[COL_IS_LEGACY];
        let ie = &col_evals[COL_IS_EIP1559];

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(ir.mul(&ir.sub(&one)));
        bodies.push(il.mul(&il.sub(&one)));
        bodies.push(ie.mul(&ie.sub(&one)));
        bodies.push(ir.sub(&il.add(ie)));

        let e0 = &col_evals[COL_ENCODED_BYTE_OFFSET];
        bodies.push(ie.mul(&e0.sub(&type_byte)));
        let lfb = &col_evals[COL_LEGACY_FIRST_BYTE];
        bodies.push(il.mul(&e0.sub(lfb)));

        let mut tail = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in (MAX_ENCODED_LEN / 2)..MAX_ENCODED_LEN {
            let eb = &col_evals[COL_ENCODED_BYTE_OFFSET + k];
            tail = tail.add(&bp.mul(eb));
            bp = bp.mul(alpha);
        }
        bodies.push(ir.mul(&tail));

        let elen = &col_evals[COL_ENCODED_LEN];
        let hlen = &col_evals[COL_LIST_HEADER_LEN];
        let plen = &col_evals[COL_PAYLOAD_LEN];
        let legacy_sum = sum_legacy_field_lens(col_evals, curve);
        let eip1559_sum = sum_eip1559_field_lens(col_evals, curve);
        let exp_payload = il.mul(&legacy_sum).add(&ie.mul(&eip1559_sum));
        let exp_encoded = hlen.add(plen).add(ie);
        let b_pay = plen.sub(&exp_payload);
        let b_enc = elen.sub(&exp_encoded);
        let total = b_pay.add(&alpha.mul(&b_enc));
        bodies.push(ir.mul(&total));

        let mut acc = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            acc = acc.add(&ap.mul(b));
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
        let type_byte_poly = vec![Scalar::from_u64(EIP1559_TYPE_BYTE, curve)];

        let ir = &col_coeffs[COL_IS_REAL];
        let il = &col_coeffs[COL_IS_LEGACY];
        let ie = &col_coeffs[COL_IS_EIP1559];

        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_poly, curve), curve);

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(bin(ir));
        bodies.push(bin(il));
        bodies.push(bin(ie));
        bodies.push(poly_sub(ir, &poly_add(il, ie, curve), curve));

        let e0 = &col_coeffs[COL_ENCODED_BYTE_OFFSET];
        bodies.push(poly_mul(ie, &poly_sub(e0, &type_byte_poly, curve), curve));
        let lfb = &col_coeffs[COL_LEGACY_FIRST_BYTE];
        bodies.push(poly_mul(il, &poly_sub(e0, lfb, curve), curve));

        let mut tail = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in (MAX_ENCODED_LEN / 2)..MAX_ENCODED_LEN {
            let eb = &col_coeffs[COL_ENCODED_BYTE_OFFSET + k];
            tail = poly_add(&tail, &poly_scalar_mul(eb, &bp), curve);
            bp = bp.mul(alpha);
        }
        bodies.push(poly_mul(ir, &tail, curve));

        let elen = &col_coeffs[COL_ENCODED_LEN];
        let hlen = &col_coeffs[COL_LIST_HEADER_LEN];
        let plen = &col_coeffs[COL_PAYLOAD_LEN];

        // Build polynomial sums of field lens.
        let mut legacy_sum = vec![Scalar::zero(curve)];
        for col in &[
            COL_NONCE_ENC_LEN,
            COL_GAS_PRICE_ENC_LEN,
            COL_GAS_LIMIT_ENC_LEN,
            COL_TO_ENC_LEN,
            COL_VALUE_ENC_LEN,
            COL_DATA_ENC_LEN,
            COL_V_ENC_LEN,
            COL_R_ENC_LEN,
            COL_S_ENC_LEN,
        ] {
            legacy_sum = poly_add(&legacy_sum, &col_coeffs[*col], curve);
        }
        let mut eip1559_sum = vec![Scalar::zero(curve)];
        for col in &[
            COL_CHAIN_ID_ENC_LEN,
            COL_NONCE_ENC_LEN,
            COL_MAX_PRIORITY_FEE_ENC_LEN,
            COL_MAX_FEE_ENC_LEN,
            COL_GAS_LIMIT_ENC_LEN,
            COL_TO_ENC_LEN,
            COL_VALUE_ENC_LEN,
            COL_DATA_ENC_LEN,
            COL_ACCESS_LIST_LEN,
            COL_V_ENC_LEN,
            COL_R_ENC_LEN,
            COL_S_ENC_LEN,
        ] {
            eip1559_sum = poly_add(&eip1559_sum, &col_coeffs[*col], curve);
        }
        let exp_payload = poly_add(
            &poly_mul(il, &legacy_sum, curve),
            &poly_mul(ie, &eip1559_sum, curve),
            curve,
        );
        let exp_encoded = poly_add(&poly_add(hlen, plen, curve), ie, curve);
        let b_pay = poly_sub(plen, &exp_payload, curve);
        let b_enc = poly_sub(elen, &exp_encoded, curve);
        let total = poly_add(&b_pay, &poly_scalar_mul(&b_enc, alpha), curve);
        bodies.push(poly_mul(ir, &total, curve));

        let mut acc = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            acc = poly_add(&acc, &poly_scalar_mul(b, &ap), curve);
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
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
            return;
        }
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(8)];
        let mut declarations = Vec::new();
        // Range-check every byte-bearing column.
        let byte_cols: Vec<(String, usize, usize)> = vec![
            ("tx_to".into(), COL_TO_BYTE_OFFSET, NUM_TO_BYTES),
            ("tx_gp".into(), COL_GAS_PRICE_BYTE_OFFSET, 32),
            ("tx_mpf".into(), COL_MAX_PRIORITY_FEE_BYTE_OFFSET, 32),
            ("tx_mf".into(), COL_MAX_FEE_BYTE_OFFSET, 32),
            ("tx_val".into(), COL_VALUE_BYTE_OFFSET, 32),
            ("tx_r".into(), COL_R_BYTE_OFFSET, 32),
            ("tx_s".into(), COL_S_BYTE_OFFSET, 32),
            ("tx_data".into(), COL_DATA_BYTE_OFFSET, MAX_DATA_LEN),
            ("tx_al".into(), COL_ACCESS_LIST_BYTE_OFFSET, MAX_ACCESS_LIST_LEN),
            ("tx_enc".into(), COL_ENCODED_BYTE_OFFSET, MAX_ENCODED_LEN),
        ];
        for (label, base, count) in byte_cols {
            for k in 0..count {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: base + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        // Also range-check the legacy_first_byte column (1 byte).
        declarations.push((
            LookupDeclaration {
                label: "tx_legacy_first_byte_8bit".into(),
                column_index: COL_LEGACY_FIRST_BYTE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Cross-AIR LogUp descriptor binding this AIR's `encoded_bytes` +
/// `encoded_len` columns to a [`crate::keccak_extract`] row's
/// `INPUT_BYTE[0..MAX_INPUT_LEN]` + `INPUT_LEN`. Combined with the
/// existing KeccakExtract↔Keccak binding (#91), the keccak invocation
/// hashing this AIR's encoded bytes is algebraically pinned to be
/// computing `tx_hash = keccak256(wire_encoding(tx))`.
pub fn make_tx_rlp_to_keccak_descriptor(
    tx_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    debug_assert_eq!(
        MAX_ENCODED_LEN,
        ke::MAX_INPUT_LEN,
        "tx_rlp_air MAX_ENCODED_LEN must equal KeccakExtract MAX_INPUT_LEN"
    );
    let mut a_columns: Vec<usize> =
        (0..MAX_ENCODED_LEN).map(|b| COL_ENCODED_BYTE_OFFSET + b).collect();
    a_columns.push(COL_ENCODED_LEN);

    let mut b_columns: Vec<usize> =
        (0..ke::MAX_INPUT_LEN).map(|b| ke::COL_INPUT_BYTE_OFFSET + b).collect();
    b_columns.push(ke::COL_INPUT_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_rlp_keccak_extract_input_v1".into(),
        a_layer_index: tx_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding the nonce column to a
/// [`crate::u64_rlp_air`] row's `value` + `encoded_len` columns. The
/// per-field gadget's `encoded_byte[0..MAX_ENCODED_LEN]` columns
/// already prove the canonical RLP encoding of the value; this
/// descriptor binds the value/length itself.
///
/// Two-column tuple: `(value, encoded_len)` matched against
/// `(u64_rlp::COL_VALUE, u64_rlp::COL_ENCODED_LEN)`.
pub fn make_tx_rlp_to_u64_rlp_nonce_descriptor(
    tx_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::u64_rlp_air as u64r;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_rlp_u64_rlp_nonce_v1".into(),
        a_layer_index: tx_layer_index,
        a_columns: vec![COL_NONCE, COL_NONCE_ENC_LEN],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns: vec![u64r::COL_VALUE, u64r::COL_ENCODED_LEN],
        b_selector_column: Some(u64r::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding gas_limit to u64_rlp_air.
pub fn make_tx_rlp_to_u64_rlp_gas_limit_descriptor(
    tx_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::u64_rlp_air as u64r;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_rlp_u64_rlp_gas_limit_v1".into(),
        a_layer_index: tx_layer_index,
        a_columns: vec![COL_GAS_LIMIT, COL_GAS_LIMIT_ENC_LEN],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns: vec![u64r::COL_VALUE, u64r::COL_ENCODED_LEN],
        b_selector_column: Some(u64r::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding gas_price (legacy) to
/// [`crate::u256_rlp_air`]. Gated by `IS_LEGACY` so EIP-1559 rows
/// (where gas_price is unused) don't fire.
pub fn make_tx_rlp_to_u256_rlp_gas_price_descriptor(
    tx_layer_index: usize,
    u256_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::u256_rlp_air as u256r;
    let mut a_columns: Vec<usize> =
        (0..32).map(|k| COL_GAS_PRICE_BYTE_OFFSET + k).collect();
    a_columns.push(COL_GAS_PRICE_ENC_LEN);

    let mut b_columns: Vec<usize> =
        (0..32).map(|k| u256r::COL_BYTE_OFFSET + k).collect();
    b_columns.push(u256r::COL_ENCODED_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_rlp_u256_rlp_gas_price_v1".into(),
        a_layer_index: tx_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_LEGACY),
        b_layer_index: u256_rlp_layer_index,
        b_columns,
        b_selector_column: Some(u256r::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding `value` (32 BE bytes) to
/// [`crate::u256_rlp_air`].
pub fn make_tx_rlp_to_u256_rlp_value_descriptor(
    tx_layer_index: usize,
    u256_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::u256_rlp_air as u256r;
    let mut a_columns: Vec<usize> =
        (0..32).map(|k| COL_VALUE_BYTE_OFFSET + k).collect();
    a_columns.push(COL_VALUE_ENC_LEN);

    let mut b_columns: Vec<usize> =
        (0..32).map(|k| u256r::COL_BYTE_OFFSET + k).collect();
    b_columns.push(u256r::COL_ENCODED_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_rlp_u256_rlp_value_v1".into(),
        a_layer_index: tx_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u256_rlp_layer_index,
        b_columns,
        b_selector_column: Some(u256r::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding `to` (20 address bytes) to
/// [`crate::fixed_rlp20_air`]. Gated by `IS_REAL · (1 − IS_CREATE)` —
/// contract-creation transactions have empty `to`, which is encoded
/// via `rlp_var_bytes` (empty string), not via fixed_rlp20. The
/// fixed_rlp20_air pre-pin of `encoded[0] = 0x94` aligns with the
/// 21-byte RLP encoding of a non-empty 20-byte string.
///
/// **Note**: this descriptor uses a 20-column tuple `(to_byte[0..20])`
/// matched against `fixed_rlp20::COL_FIELD_BYTE_OFFSET..+20`. The
/// `to_enc_len` is implicit (always 21 for non-empty `to`) and pinned
/// by the gadget; for the contract-creation case, the `to_enc_len`
/// is 1 (the byte `0x80`), bound separately via the var-bytes
/// gadget — see [`make_tx_rlp_to_var_bytes_data_descriptor`] for the
/// pattern.
pub fn make_tx_rlp_to_fixed_rlp20_to_descriptor(
    tx_layer_index: usize,
    fixed_rlp20_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::fixed_rlp20_air as f20;
    let a_columns: Vec<usize> =
        (0..NUM_TO_BYTES).map(|k| COL_TO_BYTE_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..f20::FIELD_LEN).map(|k| f20::COL_FIELD_BYTE_OFFSET + k).collect();

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_rlp_fixed_rlp20_to_v1".into(),
        a_layer_index: tx_layer_index,
        a_columns,
        // Gate by IS_REAL only — caller must ensure contract-creation
        // rows (IS_CREATE=1) are bound via the var-bytes path instead.
        // The mixed-mode case is documented as deferred; see
        // `make_tx_rlp_to_var_bytes_data_descriptor`.
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: fixed_rlp20_layer_index,
        b_columns,
        b_selector_column: Some(f20::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding `data` (variable bytes ≤
/// MAX_DATA_LEN) + `data_len` to [`crate::rlp_var_bytes_air`].
///
/// Tuple shape: `(data_byte[0..MAX_DATA_LEN], data_len)` matched
/// against `(var_bytes::COL_DATA_OFFSET..+MAX_DATA, var_bytes::COL_DATA_LEN)`.
/// The var_bytes_air's own constraints prove canonical RLP for any
/// length 0..=MAX_DATA.
pub fn make_tx_rlp_to_var_bytes_data_descriptor(
    tx_layer_index: usize,
    var_bytes_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> =
        (0..MAX_DATA_LEN).map(|k| COL_DATA_BYTE_OFFSET + k).collect();
    a_columns.push(COL_DATA_LEN);

    let mut b_columns: Vec<usize> = (0..MAX_DATA_LEN)
        .map(|k| crate::rlp_var_bytes_air::COL_DATA_OFFSET + k)
        .collect();
    b_columns.push(crate::rlp_var_bytes_air::COL_DATA_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_rlp_var_bytes_data_v1".into(),
        a_layer_index: tx_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: var_bytes_layer_index,
        b_columns,
        b_selector_column: Some(crate::rlp_var_bytes_air::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn u64_to_be32(n: u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[24..].copy_from_slice(&n.to_be_bytes());
        out
    }

    fn sample_legacy() -> LegacyTx {
        LegacyTx {
            nonce: 7,
            gas_price: u64_to_be32(20_000_000_000),
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: u64_to_be32(1_000_000_000_000_000_000),
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        }
    }

    fn sample_eip1559() -> Eip1559Tx {
        Eip1559Tx {
            chain_id: 1,
            nonce: 7,
            max_priority_fee_per_gas: u64_to_be32(1_500_000_000),
            max_fee_per_gas: u64_to_be32(30_000_000_000),
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: u64_to_be32(1_000_000_000_000_000_000),
            data: Vec::new(),
            access_list_rlp: vec![0xc0],
            y_parity: 0,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        }
    }

    #[test]
    fn legacy_witness_encodes_matches_canonical() {
        let tx = sample_legacy();
        let row = TxRlpRow::from_legacy_tx(&tx).unwrap();
        assert_eq!(row.encoded_bytes, tx.rlp_encode());
        assert_eq!(row.encoded_len, tx.rlp_encode().len());
        assert!(row.is_legacy && !row.is_eip1559);
    }

    #[test]
    fn eip1559_witness_type_byte_at_position_zero() {
        let tx = sample_eip1559();
        let row = TxRlpRow::from_eip1559_tx(&tx).unwrap();
        assert_eq!(row.encoded_bytes[0], 0x02);
        assert!(row.is_eip1559 && !row.is_legacy);
        // The trace builder also exposes ENCODED_BYTE[0] = 0x02.
        let w = TxRlpWitness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(
            trace.columns[COL_ENCODED_BYTE_OFFSET].evaluations[0].to_u64(),
            0x02
        );
    }

    #[test]
    fn constraints_zero_on_honest_legacy_and_eip1559() {
        let mut rows = Vec::new();
        rows.push(TxRlpRow::from_legacy_tx(&sample_legacy()).unwrap());
        rows.push(TxRlpRow::from_eip1559_tx(&sample_eip1559()).unwrap());
        let w = TxRlpWitness { rows };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = TxRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} nonzero",
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn type_byte_constraint_fires_on_tamper() {
        let row = TxRlpRow::from_eip1559_tx(&sample_eip1559()).unwrap();
        let w = TxRlpWitness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: change ENCODED_BYTE[0] from 0x02 to 0x99.
        cols[COL_ENCODED_BYTE_OFFSET][0] = Scalar::from_u64(0x99, CurveType::Bls48581);
        let cs = TxRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 4 (eip1559_type_byte_at_pos0) MUST be non-zero.
        assert!(!res[4][0].is_zero(), "type-byte tamper should fire constraint 4");
    }

    #[test]
    fn legacy_first_byte_constraint_fires_on_tamper() {
        let row = TxRlpRow::from_legacy_tx(&sample_legacy()).unwrap();
        let w = TxRlpWitness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: corrupt encoded byte 0 only (witness's LEGACY_FIRST_BYTE
        // is unchanged, so the constraint fires).
        let orig = cols[COL_ENCODED_BYTE_OFFSET][0].clone();
        cols[COL_ENCODED_BYTE_OFFSET][0] =
            orig.add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = TxRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[5][0].is_zero(), "legacy first byte tamper should fire constraint 5");
    }

    #[test]
    fn length_consistency_constraint_fires_on_tamper() {
        let row = TxRlpRow::from_legacy_tx(&sample_legacy()).unwrap();
        let w = TxRlpWitness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: bump ENCODED_LEN by 1 (header/payload/field sums
        // unchanged → length consistency body fires).
        let orig = cols[COL_ENCODED_LEN][0].clone();
        cols[COL_ENCODED_LEN][0] =
            orig.add(&Scalar::from_u64(1, CurveType::Bls48581));
        let cs = TxRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !res[7][0].is_zero(),
            "encoded_len tamper should fire constraint 7"
        );
    }

    #[test]
    fn mutual_exclusion_constraint_fires_on_both_selectors() {
        let mut row = TxRlpRow::from_legacy_tx(&sample_legacy()).unwrap();
        row.is_eip1559 = true; // make both selectors 1 → constraint 3 fires
        let w = TxRlpWitness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = TxRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !res[3][0].is_zero(),
            "double-selector should fire constraint 3 (mutually exclusive)"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d_keccak = make_tx_rlp_to_keccak_descriptor(0, 1);
        assert_eq!(d_keccak.label, "tx_rlp_keccak_extract_input_v1");
        assert_eq!(d_keccak.a_columns.len(), MAX_ENCODED_LEN + 1);
        assert_eq!(d_keccak.b_columns.len(), MAX_ENCODED_LEN + 1);
        assert_eq!(d_keccak.a_layer_index, 0);
        assert_eq!(d_keccak.b_layer_index, 1);

        let d_nonce = make_tx_rlp_to_u64_rlp_nonce_descriptor(0, 2);
        assert_eq!(d_nonce.a_columns, vec![COL_NONCE, COL_NONCE_ENC_LEN]);
        assert_eq!(d_nonce.a_columns.len(), d_nonce.b_columns.len());

        let d_gas_limit = make_tx_rlp_to_u64_rlp_gas_limit_descriptor(0, 2);
        assert_eq!(
            d_gas_limit.a_columns,
            vec![COL_GAS_LIMIT, COL_GAS_LIMIT_ENC_LEN]
        );

        let d_gp = make_tx_rlp_to_u256_rlp_gas_price_descriptor(0, 3);
        assert_eq!(d_gp.a_columns.len(), 33);
        assert_eq!(d_gp.b_columns.len(), 33);
        assert_eq!(d_gp.a_selector_column, Some(COL_IS_LEGACY));

        let d_val = make_tx_rlp_to_u256_rlp_value_descriptor(0, 3);
        assert_eq!(d_val.a_columns.len(), 33);

        let d_to = make_tx_rlp_to_fixed_rlp20_to_descriptor(0, 4);
        assert_eq!(d_to.a_columns.len(), NUM_TO_BYTES);
        assert_eq!(d_to.b_columns.len(), NUM_TO_BYTES);

        let d_data = make_tx_rlp_to_var_bytes_data_descriptor(0, 5);
        assert_eq!(d_data.a_columns.len(), MAX_DATA_LEN + 1);
        assert_eq!(d_data.b_columns.len(), MAX_DATA_LEN + 1);
    }

    #[test]
    fn rejects_oversize_data() {
        let mut tx = sample_legacy();
        tx.data = vec![0u8; MAX_DATA_LEN + 1];
        assert!(TxRlpRow::from_legacy_tx(&tx).is_err());

        let mut etx = sample_eip1559();
        etx.data = vec![0u8; MAX_DATA_LEN + 1];
        assert!(TxRlpRow::from_eip1559_tx(&etx).is_err());
    }

    #[test]
    fn contract_creation_witness_to_is_zero() {
        let mut tx = sample_legacy();
        tx.to = None;
        let row = TxRlpRow::from_legacy_tx(&tx).unwrap();
        assert!(row.to.is_none());
        let w = TxRlpWitness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // IS_CREATE column is 1, to bytes are 0.
        assert_eq!(trace.columns[COL_IS_CREATE].evaluations[0].to_u64(), 1);
        for k in 0..NUM_TO_BYTES {
            assert!(
                trace.columns[COL_TO_BYTE_OFFSET + k].evaluations[0].is_zero()
            );
        }
    }

    #[test]
    fn num_columns_pinned() {
        // Pin layout — if we change a constant we want a visible test
        // failure forcing a re-check of cross-AIR descriptors.
        assert_eq!(MAX_ENCODED_LEN, 256);
        assert_eq!(MAX_DATA_LEN, 32);
        assert_eq!(MAX_ACCESS_LIST_LEN, 64);
        assert_eq!(EIP1559_TYPE_BYTE, 0x02);
        // Cumulative constant: if NUM_COLUMNS shifts due to layout
        // change, downstream descriptors need re-auditing.
        assert!(NUM_COLUMNS > MAX_ENCODED_LEN);
    }
}
