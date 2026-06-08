//! Algebraic account-state RLP encoding AIR.
//!
//! Phase A2 step 3+: a dedicated gadget AIR that algebraically commits
//! to the canonical RLP encoding of an Ethereum [`Account`]:
//!
//! ```text
//! account_rlp = RLP([nonce, balance, storage_root, code_hash])
//! ```
//!
//! # Structure of the encoding
//!
//! Each field encodes as:
//!   - `nonce` (u64) → variable-length, 1..=9 bytes via
//!     [`crate::u64_rlp_air::rlp_encode_u64`].
//!   - `balance` (u256 BE) → variable-length, 1..=33 bytes via
//!     [`crate::u256_rlp_air::rlp_encode_u256_be`].
//!   - `storage_root` (32 bytes) → fixed 33 bytes `[0xa0 || bytes]`.
//!   - `code_hash` (32 bytes) → fixed 33 bytes `[0xa0 || bytes]`.
//!
//! Total payload length therefore ranges from
//!   `1 + 1 + 33 + 33 = 68` (zero-nonce, zero-balance EOA)
//! up to
//!   `9 + 33 + 33 + 33 = 108` (saturated u64 nonce + saturated u256
//!   balance).
//!
//! Since `68 ≤ payload_len ≤ 108` and `56 ≤ payload_len ≤ 255`, the
//! account-RLP list prefix is **always** exactly the two bytes
//!   `[0xf8, payload_len]`
//! (long-list prefix with a single length byte). This invariant
//! collapses what would otherwise be a case split on the list-prefix
//! shape into a pair of trivial byte equalities.
//!
//! # What this AIR enforces algebraically
//!
//! Per row, gated by `IS_REAL`:
//!   1. `IS_REAL` is binary.
//!   2. `LIST_PREFIX_0 = 0xf8`.
//!   3. `LIST_PREFIX_1 = PAYLOAD_LEN`.
//!   4. `PAYLOAD_LEN = NONCE_ENC_LEN + BALANCE_ENC_LEN + 66`.
//!   5. `ENCODED_LEN  = PAYLOAD_LEN + 2`.
//!   6. `STORAGE_ROOT_ENC[0] = 0xa0`.
//!   7. β-RLC over k=0..32: `STORAGE_ROOT_ENC[k+1] = STORAGE_ROOT[k]`.
//!   8. `CODE_HASH_ENC[0] = 0xa0`.
//!   9. β-RLC over k=0..32: `CODE_HASH_ENC[k+1] = CODE_HASH[k]`.
//!
//! All 8-bit columns get range checks via [`LookupTable::range(256)`].
//!
//! # What is bound via cross-AIR LogUp (not row-local)
//!
//! The per-field canonical encoded buffers are exposed as columns so
//! they can be looked up against the dedicated per-field encoder AIRs:
//!
//!   - `make_account_rlp_to_u64_rlp_linkage_descriptor` — binds
//!     `(NONCE, NONCE_ENC_LEN, NONCE_ENC[0..9])` to
//!     `(u64_rlp::VALUE, ENCODED_LEN, ENCODED[0..9])`.
//!   - `make_account_rlp_to_u256_rlp_linkage_descriptor` — binds
//!     `(BALANCE_BE[0..32], BALANCE_ENC_LEN, BALANCE_ENC[0..33])` to
//!     `(u256_rlp::BYTE[0..32], ENCODED_LEN, ENCODED[0..33])`.
//!   - `make_account_rlp_storage_root_to_fixed_rlp_linkage_descriptor`
//!     — binds `(STORAGE_ROOT[0..32], STORAGE_ROOT_ENC[0..33])` to
//!     `(fixed_rlp32::FIELD_BYTES, ENCODED_BYTES)`.
//!   - `make_account_rlp_code_hash_to_fixed_rlp_linkage_descriptor`
//!     — analogous for the code_hash field.
//!
//! Combined with the existing per-field AIRs (which themselves
//! enforce the canonical RLP encoding of their input), the full
//! account RLP is algebraically determined.
//!
//! # What is NOT in scope (deferred)
//!
//! - A single contiguous `encoded_bytes[0..110]` column with
//!   alignment of variable-length fields into specific byte offsets.
//!   That is the job of a downstream byte-concat gadget (see
//!   [`crate::rlp_byte_concat_air`] / [`crate::rlp_list_concat_air`])
//!   which can compose with this AIR's per-field canonical buffers.
//! - SSTORE/SLOAD state transitions (the account's nonce/balance/
//!   storage_root changing across EVM rows).
//! - MPT inclusion of the resulting RLP at `keccak256(address)` in
//!   the state root — covered by [`crate::storage_access_air`] +
//!   [`crate::account_state_air`] descriptors.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

use crate::account::Account;
use crate::fixed_rlp_air::RLP32_PREFIX;
use crate::u256_rlp_air::{rlp_encode_u256_be, MAX_ENCODED_LEN as U256_MAX_ENCODED_LEN};
use crate::u64_rlp_air::{rlp_encode_u64, MAX_ENCODED_LEN as U64_MAX_ENCODED_LEN};

// ─── Encoding constants ──────────────────────────────────────────────

/// The fixed list-prefix length byte index (always 0xf8 for an
/// Account: payload_len ∈ [68, 108] so we always use the 1-length-byte
/// long-list prefix `0xf7 + 1 = 0xf8`).
pub const LIST_PREFIX_BYTE_0: u8 = 0xf8;

/// List prefix is always exactly 2 bytes for an Account RLP.
pub const LIST_PREFIX_LEN: usize = 2;

/// Storage root + code hash each encode to a fixed 33 bytes
/// (`0xa0 || bytes`). Two such fields contribute 66 bytes of payload.
pub const FIXED_PAIR_PAYLOAD: usize = 2 * 33;

/// Min/max payload length bounds.
pub const MIN_PAYLOAD_LEN: usize = 1 + 1 + FIXED_PAIR_PAYLOAD; // 68
pub const MAX_PAYLOAD_LEN: usize = 9 + 33 + FIXED_PAIR_PAYLOAD; // 108

/// Max total encoded length (list prefix + payload).
pub const MAX_ENCODED_LEN: usize = LIST_PREFIX_LEN + MAX_PAYLOAD_LEN; // 110

/// 32-byte field width (storage_root, code_hash, balance backing).
const N32: usize = 32;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_NONCE: usize = 0;
pub const COL_BALANCE_OFFSET: usize = 1;                 // 1..33
pub const COL_STORAGE_ROOT_OFFSET: usize = 33;           // 33..65
pub const COL_CODE_HASH_OFFSET: usize = 65;              // 65..97

pub const COL_NONCE_ENC_LEN: usize = 97;
pub const COL_BALANCE_ENC_LEN: usize = 98;
pub const COL_PAYLOAD_LEN: usize = 99;
pub const COL_ENCODED_LEN: usize = 100;

pub const COL_LIST_PREFIX_0: usize = 101;
pub const COL_LIST_PREFIX_1: usize = 102;

pub const COL_NONCE_ENC_OFFSET: usize = 103;             // 103..112 (9 bytes max)
pub const COL_BALANCE_ENC_OFFSET: usize = COL_NONCE_ENC_OFFSET + U64_MAX_ENCODED_LEN; // 112..145 (33 bytes max)
pub const COL_STORAGE_ROOT_ENC_OFFSET: usize = COL_BALANCE_ENC_OFFSET + U256_MAX_ENCODED_LEN; // 145..178 (33 bytes)
pub const COL_CODE_HASH_ENC_OFFSET: usize = COL_STORAGE_ROOT_ENC_OFFSET + 33; // 178..211 (33 bytes)

pub const COL_IS_REAL: usize = COL_CODE_HASH_ENC_OFFSET + 33; // 211

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 212

/// Row-local constraint count.
///
///   0: is_real binary
///   1: list_prefix[0] = 0xf8
///   2: list_prefix[1] = payload_len
///   3: payload_len = nonce_enc_len + balance_enc_len + 66
///   4: encoded_len = payload_len + 2
///   5: storage_root_enc[0] = 0xa0
///   6: β-RLC storage_root_enc[k+1] = storage_root[k] for k=0..32
///   7: code_hash_enc[0] = 0xa0
///   8: β-RLC code_hash_enc[k+1] = code_hash[k] for k=0..32
pub const NUM_ROW_CONSTRAINTS: usize = 9;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AccountRlpRow {
    pub account: Account,
}

#[derive(Clone, Debug, Default)]
pub struct AccountRlpWitness {
    pub rows: Vec<AccountRlpRow>,
}

impl AccountRlpWitness {
    pub fn from_accounts(accounts: Vec<Account>) -> Self {
        Self {
            rows: accounts.into_iter().map(|account| AccountRlpRow { account }).collect(),
        }
    }

    pub fn from_account(account: Account) -> Self {
        Self::from_accounts(vec![account])
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &AccountRlpWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        let a = &row.account;

        // ── Field columns ────────────────────────────────────────────
        columns[COL_NONCE][r] = Scalar::from_u64(a.nonce, curve);
        for k in 0..N32 {
            columns[COL_BALANCE_OFFSET + k][r] = Scalar::from_u64(a.balance[k] as u64, curve);
            columns[COL_STORAGE_ROOT_OFFSET + k][r] =
                Scalar::from_u64(a.storage_root[k] as u64, curve);
            columns[COL_CODE_HASH_OFFSET + k][r] = Scalar::from_u64(a.code_hash[k] as u64, curve);
        }

        // ── Per-field canonical encodings ─────────────────────────────
        let nonce_enc = rlp_encode_u64(a.nonce);
        let nonce_enc_len = nonce_enc.len();
        debug_assert!(nonce_enc_len <= U64_MAX_ENCODED_LEN);
        columns[COL_NONCE_ENC_LEN][r] = Scalar::from_u64(nonce_enc_len as u64, curve);
        for (k, &b) in nonce_enc.iter().enumerate() {
            columns[COL_NONCE_ENC_OFFSET + k][r] = Scalar::from_u64(b as u64, curve);
        }

        let balance_enc = rlp_encode_u256_be(&a.balance);
        let balance_enc_len = balance_enc.len();
        debug_assert!(balance_enc_len <= U256_MAX_ENCODED_LEN);
        columns[COL_BALANCE_ENC_LEN][r] = Scalar::from_u64(balance_enc_len as u64, curve);
        for (k, &b) in balance_enc.iter().enumerate() {
            columns[COL_BALANCE_ENC_OFFSET + k][r] = Scalar::from_u64(b as u64, curve);
        }

        // Storage root / code hash are always 33 bytes: 0xa0 || bytes.
        columns[COL_STORAGE_ROOT_ENC_OFFSET][r] = Scalar::from_u64(RLP32_PREFIX as u64, curve);
        for k in 0..N32 {
            columns[COL_STORAGE_ROOT_ENC_OFFSET + 1 + k][r] =
                Scalar::from_u64(a.storage_root[k] as u64, curve);
        }
        columns[COL_CODE_HASH_ENC_OFFSET][r] = Scalar::from_u64(RLP32_PREFIX as u64, curve);
        for k in 0..N32 {
            columns[COL_CODE_HASH_ENC_OFFSET + 1 + k][r] =
                Scalar::from_u64(a.code_hash[k] as u64, curve);
        }

        // ── List-prefix + lengths ────────────────────────────────────
        let payload_len = nonce_enc_len + balance_enc_len + FIXED_PAIR_PAYLOAD;
        let encoded_len = payload_len + LIST_PREFIX_LEN;
        debug_assert!(
            (MIN_PAYLOAD_LEN..=MAX_PAYLOAD_LEN).contains(&payload_len),
            "payload_len {} out of bounds for Account RLP",
            payload_len,
        );
        columns[COL_PAYLOAD_LEN][r] = Scalar::from_u64(payload_len as u64, curve);
        columns[COL_ENCODED_LEN][r] = Scalar::from_u64(encoded_len as u64, curve);
        columns[COL_LIST_PREFIX_0][r] = Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve);
        columns[COL_LIST_PREFIX_1][r] = Scalar::from_u64(payload_len as u64, curve);

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

pub struct AccountRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AccountRlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for AccountRlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "list_prefix_0_eq_0xf8".into(),
            "list_prefix_1_eq_payload_len".into(),
            "payload_len_arith".into(),
            "encoded_len_arith".into(),
            "storage_root_enc_0_eq_0xa0".into(),
            "storage_root_enc_body_rlc".into(),
            "code_hash_enc_0_eq_0xa0".into(),
            "code_hash_enc_body_rlc".into(),
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
        let prefix_byte = Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve);
        let rlp32_byte = Scalar::from_u64(RLP32_PREFIX as u64, curve);
        let two_const = Scalar::from_u64(LIST_PREFIX_LEN as u64, curve);
        let fixed_const = Scalar::from_u64(FIXED_PAIR_PAYLOAD as u64, curve);
        let beta_test = Scalar::from_u64(7, curve);

        let mk = || vec![Scalar::zero(curve); n];
        let mut c_is_real_bin = mk();
        let mut c_prefix0 = mk();
        let mut c_prefix1 = mk();
        let mut c_payload = mk();
        let mut c_enclen = mk();
        let mut c_sr_enc0 = mk();
        let mut c_sr_body = mk();
        let mut c_ch_enc0 = mk();
        let mut c_ch_body = mk();

        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            c_is_real_bin[r] = v.mul(&v.sub(&one));

            let p0 = &columns[COL_LIST_PREFIX_0][r];
            c_prefix0[r] = v.mul(&p0.sub(&prefix_byte));

            let p1 = &columns[COL_LIST_PREFIX_1][r];
            let pl = &columns[COL_PAYLOAD_LEN][r];
            c_prefix1[r] = v.mul(&p1.sub(pl));

            // payload_len = nonce_enc_len + balance_enc_len + 66
            let nl = &columns[COL_NONCE_ENC_LEN][r];
            let bl = &columns[COL_BALANCE_ENC_LEN][r];
            let expected_pl = nl.add(bl).add(&fixed_const);
            c_payload[r] = v.mul(&pl.sub(&expected_pl));

            // encoded_len = payload_len + 2
            let el = &columns[COL_ENCODED_LEN][r];
            let expected_el = pl.add(&two_const);
            c_enclen[r] = v.mul(&el.sub(&expected_el));

            // storage_root_enc[0] = 0xa0
            let sr0 = &columns[COL_STORAGE_ROOT_ENC_OFFSET][r];
            c_sr_enc0[r] = v.mul(&sr0.sub(&rlp32_byte));

            // β-RLC: storage_root_enc[k+1] - storage_root[k]
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..N32 {
                let e = &columns[COL_STORAGE_ROOT_ENC_OFFSET + 1 + k][r];
                let f = &columns[COL_STORAGE_ROOT_OFFSET + k][r];
                acc = acc.add(&bp.mul(&e.sub(f)));
                bp = bp.mul(&beta_test);
            }
            c_sr_body[r] = v.mul(&acc);

            // code_hash_enc[0] = 0xa0
            let ch0 = &columns[COL_CODE_HASH_ENC_OFFSET][r];
            c_ch_enc0[r] = v.mul(&ch0.sub(&rlp32_byte));

            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..N32 {
                let e = &columns[COL_CODE_HASH_ENC_OFFSET + 1 + k][r];
                let f = &columns[COL_CODE_HASH_OFFSET + k][r];
                acc = acc.add(&bp.mul(&e.sub(f)));
                bp = bp.mul(&beta_test);
            }
            c_ch_body[r] = v.mul(&acc);
        }

        vec![
            c_is_real_bin,
            c_prefix0,
            c_prefix1,
            c_payload,
            c_enclen,
            c_sr_enc0,
            c_sr_body,
            c_ch_enc0,
            c_ch_body,
        ]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let prefix_byte = Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve);
        let rlp32_byte = Scalar::from_u64(RLP32_PREFIX as u64, curve);
        let two_const = Scalar::from_u64(LIST_PREFIX_LEN as u64, curve);
        let fixed_const = Scalar::from_u64(FIXED_PAIR_PAYLOAD as u64, curve);

        let v = &col_evals[COL_IS_REAL];
        let c0 = v.mul(&v.sub(&one));

        let p0 = &col_evals[COL_LIST_PREFIX_0];
        let c1 = v.mul(&p0.sub(&prefix_byte));

        let p1 = &col_evals[COL_LIST_PREFIX_1];
        let pl = &col_evals[COL_PAYLOAD_LEN];
        let c2 = v.mul(&p1.sub(pl));

        let nl = &col_evals[COL_NONCE_ENC_LEN];
        let bl = &col_evals[COL_BALANCE_ENC_LEN];
        let expected_pl = nl.add(bl).add(&fixed_const);
        let c3 = v.mul(&pl.sub(&expected_pl));

        let el = &col_evals[COL_ENCODED_LEN];
        let expected_el = pl.add(&two_const);
        let c4 = v.mul(&el.sub(&expected_el));

        let sr0 = &col_evals[COL_STORAGE_ROOT_ENC_OFFSET];
        let c5 = v.mul(&sr0.sub(&rlp32_byte));

        let mut sr_acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..N32 {
            let e = &col_evals[COL_STORAGE_ROOT_ENC_OFFSET + 1 + k];
            let f = &col_evals[COL_STORAGE_ROOT_OFFSET + k];
            sr_acc = sr_acc.add(&bp.mul(&e.sub(f)));
            bp = bp.mul(alpha);
        }
        let c6 = v.mul(&sr_acc);

        let ch0 = &col_evals[COL_CODE_HASH_ENC_OFFSET];
        let c7 = v.mul(&ch0.sub(&rlp32_byte));

        let mut ch_acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..N32 {
            let e = &col_evals[COL_CODE_HASH_ENC_OFFSET + 1 + k];
            let f = &col_evals[COL_CODE_HASH_OFFSET + k];
            ch_acc = ch_acc.add(&bp.mul(&e.sub(f)));
            bp = bp.mul(alpha);
        }
        let c8 = v.mul(&ch_acc);

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7, c8];
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
        let prefix_poly = vec![Scalar::from_u64(LIST_PREFIX_BYTE_0 as u64, curve)];
        let rlp32_poly = vec![Scalar::from_u64(RLP32_PREFIX as u64, curve)];
        let two_poly = vec![Scalar::from_u64(LIST_PREFIX_LEN as u64, curve)];
        let fixed_poly = vec![Scalar::from_u64(FIXED_PAIR_PAYLOAD as u64, curve)];

        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let c0 = poly_mul(v, &v_m1, curve);

        let p0 = &col_coeffs[COL_LIST_PREFIX_0];
        let p0_m = poly_sub(p0, &prefix_poly, curve);
        let c1 = poly_mul(v, &p0_m, curve);

        let p1 = &col_coeffs[COL_LIST_PREFIX_1];
        let pl = &col_coeffs[COL_PAYLOAD_LEN];
        let diff_p1 = poly_sub(p1, pl, curve);
        let c2 = poly_mul(v, &diff_p1, curve);

        let nl = &col_coeffs[COL_NONCE_ENC_LEN];
        let bl = &col_coeffs[COL_BALANCE_ENC_LEN];
        let nl_plus_bl = poly_add(nl, bl, curve);
        let expected_pl = poly_add(&nl_plus_bl, &fixed_poly, curve);
        let diff_pl = poly_sub(pl, &expected_pl, curve);
        let c3 = poly_mul(v, &diff_pl, curve);

        let el = &col_coeffs[COL_ENCODED_LEN];
        let expected_el = poly_add(pl, &two_poly, curve);
        let diff_el = poly_sub(el, &expected_el, curve);
        let c4 = poly_mul(v, &diff_el, curve);

        let sr0 = &col_coeffs[COL_STORAGE_ROOT_ENC_OFFSET];
        let sr0_m = poly_sub(sr0, &rlp32_poly, curve);
        let c5 = poly_mul(v, &sr0_m, curve);

        let mut sr_acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..N32 {
            let e = &col_coeffs[COL_STORAGE_ROOT_ENC_OFFSET + 1 + k];
            let f = &col_coeffs[COL_STORAGE_ROOT_OFFSET + k];
            let d = poly_sub(e, f, curve);
            sr_acc = poly_add(&sr_acc, &poly_scalar_mul(&d, &bp), curve);
            bp = bp.mul(alpha);
        }
        let c6 = poly_mul(v, &sr_acc, curve);

        let ch0 = &col_coeffs[COL_CODE_HASH_ENC_OFFSET];
        let ch0_m = poly_sub(ch0, &rlp32_poly, curve);
        let c7 = poly_mul(v, &ch0_m, curve);

        let mut ch_acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..N32 {
            let e = &col_coeffs[COL_CODE_HASH_ENC_OFFSET + 1 + k];
            let f = &col_coeffs[COL_CODE_HASH_OFFSET + k];
            let d = poly_sub(e, f, curve);
            ch_acc = poly_add(&ch_acc, &poly_scalar_mul(&d, &bp), curve);
            bp = bp.mul(alpha);
        }
        let c8 = poly_mul(v, &ch_acc, curve);

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7, c8];
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

        // 32-byte field columns get byte range checks.
        for k in 0..N32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("account_rlp_balance_{}_8bit", k),
                    column_index: COL_BALANCE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("account_rlp_storage_root_{}_8bit", k),
                    column_index: COL_STORAGE_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("account_rlp_code_hash_{}_8bit", k),
                    column_index: COL_CODE_HASH_OFFSET + k,
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
                    label: format!("account_rlp_nonce_enc_{}_8bit", k),
                    column_index: COL_NONCE_ENC_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..U256_MAX_ENCODED_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("account_rlp_balance_enc_{}_8bit", k),
                    column_index: COL_BALANCE_ENC_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..33 {
            declarations.push((
                LookupDeclaration {
                    label: format!("account_rlp_storage_root_enc_{}_8bit", k),
                    column_index: COL_STORAGE_ROOT_ENC_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("account_rlp_code_hash_enc_{}_8bit", k),
                    column_index: COL_CODE_HASH_ENC_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // List-prefix bytes also range-checked.
        declarations.push((
            LookupDeclaration {
                label: "account_rlp_list_prefix_0_8bit".into(),
                column_index: COL_LIST_PREFIX_0,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        declarations.push((
            LookupDeclaration {
                label: "account_rlp_list_prefix_1_8bit".into(),
                column_index: COL_LIST_PREFIX_1,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Linkage descriptor binding this AIR's `(nonce, nonce_enc_len,
/// nonce_enc[0..9])` tuple to the dedicated [`crate::u64_rlp_air`]'s
/// `(value, encoded_len, encoded[0..9])`. Combined with the u64 RLP
/// AIR's own constraints, this fixes `nonce_enc[0..9]` to the unique
/// canonical RLP encoding of the committed `nonce`.
///
/// Both sides gate on the AIR's `is_real` column.
pub fn make_account_rlp_to_u64_rlp_linkage_descriptor(
    account_rlp_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(2 + U64_MAX_ENCODED_LEN);
    a_columns.push(COL_NONCE);
    a_columns.push(COL_NONCE_ENC_LEN);
    for k in 0..U64_MAX_ENCODED_LEN {
        a_columns.push(COL_NONCE_ENC_OFFSET + k);
    }

    let mut b_columns = Vec::with_capacity(2 + U64_MAX_ENCODED_LEN);
    b_columns.push(crate::u64_rlp_air::COL_VALUE);
    b_columns.push(crate::u64_rlp_air::COL_ENCODED_LEN);
    for k in 0..U64_MAX_ENCODED_LEN {
        b_columns.push(crate::u64_rlp_air::COL_ENCODED_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "account_rlp_to_u64_rlp_v1".into(),
        a_layer_index: account_rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns,
        b_selector_column: Some(crate::u64_rlp_air::COL_IS_REAL),
    }
}

/// Linkage descriptor binding this AIR's `(balance_be[0..32],
/// balance_enc_len, balance_enc[0..33])` tuple to the dedicated
/// [`crate::u256_rlp_air`]'s `(byte[0..32], encoded_len,
/// encoded[0..33])`.
pub fn make_account_rlp_to_u256_rlp_linkage_descriptor(
    account_rlp_layer_index: usize,
    u256_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(N32 + 1 + U256_MAX_ENCODED_LEN);
    for k in 0..N32 {
        a_columns.push(COL_BALANCE_OFFSET + k);
    }
    a_columns.push(COL_BALANCE_ENC_LEN);
    for k in 0..U256_MAX_ENCODED_LEN {
        a_columns.push(COL_BALANCE_ENC_OFFSET + k);
    }

    let mut b_columns = Vec::with_capacity(N32 + 1 + U256_MAX_ENCODED_LEN);
    for k in 0..N32 {
        b_columns.push(crate::u256_rlp_air::COL_BYTE_OFFSET + k);
    }
    b_columns.push(crate::u256_rlp_air::COL_ENCODED_LEN);
    for k in 0..U256_MAX_ENCODED_LEN {
        b_columns.push(crate::u256_rlp_air::COL_ENCODED_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "account_rlp_to_u256_rlp_v1".into(),
        a_layer_index: account_rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u256_rlp_layer_index,
        b_columns,
        b_selector_column: Some(crate::u256_rlp_air::COL_IS_REAL),
    }
}

/// Linkage descriptor binding this AIR's `(storage_root[0..32],
/// storage_root_enc[0..33])` tuple to the dedicated
/// [`crate::fixed_rlp_air`]'s `(field_bytes, encoded_bytes)`.
pub fn make_account_rlp_storage_root_to_fixed_rlp_linkage_descriptor(
    account_rlp_layer_index: usize,
    fixed_rlp32_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(N32 + 33);
    for k in 0..N32 {
        a_columns.push(COL_STORAGE_ROOT_OFFSET + k);
    }
    for k in 0..33 {
        a_columns.push(COL_STORAGE_ROOT_ENC_OFFSET + k);
    }

    let mut b_columns = Vec::with_capacity(N32 + 33);
    for k in 0..N32 {
        b_columns.push(crate::fixed_rlp_air::COL_FIELD_BYTE_OFFSET + k);
    }
    for k in 0..33 {
        b_columns.push(crate::fixed_rlp_air::COL_ENCODED_BYTE_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "account_rlp_storage_root_to_fixed_rlp32_v1".into(),
        a_layer_index: account_rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: fixed_rlp32_layer_index,
        b_columns,
        b_selector_column: Some(crate::fixed_rlp_air::COL_IS_REAL),
    }
}

/// Linkage descriptor binding this AIR's `(code_hash[0..32],
/// code_hash_enc[0..33])` tuple to the dedicated
/// [`crate::fixed_rlp_air`]'s `(field_bytes, encoded_bytes)`.
pub fn make_account_rlp_code_hash_to_fixed_rlp_linkage_descriptor(
    account_rlp_layer_index: usize,
    fixed_rlp32_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(N32 + 33);
    for k in 0..N32 {
        a_columns.push(COL_CODE_HASH_OFFSET + k);
    }
    for k in 0..33 {
        a_columns.push(COL_CODE_HASH_ENC_OFFSET + k);
    }

    let mut b_columns = Vec::with_capacity(N32 + 33);
    for k in 0..N32 {
        b_columns.push(crate::fixed_rlp_air::COL_FIELD_BYTE_OFFSET + k);
    }
    for k in 0..33 {
        b_columns.push(crate::fixed_rlp_air::COL_ENCODED_BYTE_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "account_rlp_code_hash_to_fixed_rlp32_v1".into(),
        a_layer_index: account_rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: fixed_rlp32_layer_index,
        b_columns,
        b_selector_column: Some(crate::fixed_rlp_air::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::{empty_code_hash, empty_storage_root};

    fn funded_account() -> Account {
        let mut balance = [0u8; 32];
        // 1 ether == 10^18 wei
        balance[24..32].copy_from_slice(&1_000_000_000_000_000_000u64.to_be_bytes());
        Account {
            nonce: 0x42,
            balance,
            storage_root: [0xaa; 32],
            code_hash: [0xbb; 32],
        }
    }

    fn build_for(accounts: Vec<Account>) -> TracePolynomials {
        let w = AccountRlpWitness::from_accounts(accounts);
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    #[test]
    fn empty_account_encoded_len_correct() {
        let a = Account::default();
        let trace = build_for(vec![a.clone()]);
        // Default account: nonce=0 → [0x80] (len 1); balance=0 → [0x80] (len 1).
        // payload = 1 + 1 + 33 + 33 = 68; encoded = 70.
        assert_eq!(trace.columns[COL_NONCE_ENC_LEN].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_BALANCE_ENC_LEN].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_PAYLOAD_LEN].evaluations[0].to_u64(), 68);
        assert_eq!(trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64(), 70);
        assert_eq!(trace.columns[COL_LIST_PREFIX_0].evaluations[0].to_u64(), 0xf8);
        assert_eq!(trace.columns[COL_LIST_PREFIX_1].evaluations[0].to_u64(), 68);

        // Cross-check against the canonical encoder.
        let canonical = crate::account::account_rlp(&a);
        assert_eq!(canonical.len(), 70);
        assert_eq!(canonical[0], 0xf8);
        assert_eq!(canonical[1], 68);
    }

    #[test]
    fn funded_account_encoded_len_correct() {
        let a = funded_account();
        let trace = build_for(vec![a.clone()]);
        let canonical = crate::account::account_rlp(&a);
        let nonce_enc_len = rlp_encode_u64(a.nonce).len();
        let balance_enc_len = rlp_encode_u256_be(&a.balance).len();
        let payload_len = nonce_enc_len + balance_enc_len + 66;
        assert_eq!(
            trace.columns[COL_NONCE_ENC_LEN].evaluations[0].to_u64() as usize,
            nonce_enc_len,
        );
        assert_eq!(
            trace.columns[COL_BALANCE_ENC_LEN].evaluations[0].to_u64() as usize,
            balance_enc_len,
        );
        assert_eq!(
            trace.columns[COL_PAYLOAD_LEN].evaluations[0].to_u64() as usize,
            payload_len,
        );
        assert_eq!(
            trace.columns[COL_ENCODED_LEN].evaluations[0].to_u64() as usize,
            payload_len + 2,
        );
        assert_eq!(canonical.len(), payload_len + 2);
        assert_eq!(canonical[0], 0xf8);
        assert_eq!(canonical[1] as usize, payload_len);
    }

    #[test]
    fn per_field_encoded_buffers_match_canonical_byte_for_byte() {
        let a = funded_account();
        let trace = build_for(vec![a.clone()]);

        let nonce_enc = rlp_encode_u64(a.nonce);
        for (k, &b) in nonce_enc.iter().enumerate() {
            assert_eq!(
                trace.columns[COL_NONCE_ENC_OFFSET + k].evaluations[0].to_u64(),
                b as u64,
                "nonce_enc[{}]", k,
            );
        }

        let balance_enc = rlp_encode_u256_be(&a.balance);
        for (k, &b) in balance_enc.iter().enumerate() {
            assert_eq!(
                trace.columns[COL_BALANCE_ENC_OFFSET + k].evaluations[0].to_u64(),
                b as u64,
                "balance_enc[{}]", k,
            );
        }

        // Storage root + code hash: 0xa0 || 32 bytes.
        assert_eq!(
            trace.columns[COL_STORAGE_ROOT_ENC_OFFSET].evaluations[0].to_u64(),
            0xa0,
        );
        for k in 0..N32 {
            assert_eq!(
                trace.columns[COL_STORAGE_ROOT_ENC_OFFSET + 1 + k].evaluations[0].to_u64(),
                a.storage_root[k] as u64,
            );
        }
        assert_eq!(
            trace.columns[COL_CODE_HASH_ENC_OFFSET].evaluations[0].to_u64(),
            0xa0,
        );
        for k in 0..N32 {
            assert_eq!(
                trace.columns[COL_CODE_HASH_ENC_OFFSET + 1 + k].evaluations[0].to_u64(),
                a.code_hash[k] as u64,
            );
        }
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let accounts = vec![
            Account::default(),
            funded_account(),
            Account {
                nonce: u64::MAX,
                balance: [0xff; 32],
                storage_root: [0x11; 32],
                code_hash: [0x22; 32],
            },
            Account {
                nonce: 1,
                balance: [0u8; 32],
                storage_root: empty_storage_root(),
                code_hash: empty_code_hash(),
            },
        ];
        let trace = build_for(accounts);
        let cs = AccountRlpConstraintSystem::new(trace.num_rows);
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
    fn list_prefix_byte_check_fires_on_tampered_prefix() {
        let trace = build_for(vec![Account::default()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: list_prefix[0] = 0xc4 instead of 0xf8.
        cols[COL_LIST_PREFIX_0][0] = Scalar::from_u64(0xc4, CurveType::Bls48581);
        let cs = AccountRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[1][0].is_zero(), "list_prefix_0_eq_0xf8 must fire");
    }

    #[test]
    fn payload_len_arith_fires_on_tampered_payload_len() {
        let trace = build_for(vec![funded_account()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: payload_len bumped by 1.
        let cur = cols[COL_PAYLOAD_LEN][0].to_u64();
        cols[COL_PAYLOAD_LEN][0] = Scalar::from_u64(cur + 1, CurveType::Bls48581);
        let cs = AccountRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 3 = payload_len_arith; constraint 2 (list_prefix_1
        // = payload_len) may also fire since list_prefix_1 is unchanged.
        assert!(!res[3][0].is_zero(), "payload_len_arith must fire");
    }

    #[test]
    fn storage_root_body_fires_on_tampered_encoded_byte() {
        let trace = build_for(vec![funded_account()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: storage_root_enc[5] flipped.
        cols[COL_STORAGE_ROOT_ENC_OFFSET + 5][0] =
            Scalar::from_u64(0x77, CurveType::Bls48581);
        let cs = AccountRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 6 = storage_root_enc_body_rlc.
        assert!(!res[6][0].is_zero(), "storage_root body match must fire");
    }

    #[test]
    fn code_hash_prefix_byte_fires_on_tamper() {
        let trace = build_for(vec![funded_account()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_CODE_HASH_ENC_OFFSET][0] =
            Scalar::from_u64(0xb8, CurveType::Bls48581);
        let cs = AccountRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[7][0].is_zero(), "code_hash_enc[0] = 0xa0 must fire");
    }

    #[test]
    fn linkage_descriptors_well_formed() {
        let d_nonce = make_account_rlp_to_u64_rlp_linkage_descriptor(0, 1);
        assert_eq!(d_nonce.label, "account_rlp_to_u64_rlp_v1");
        // (value, encoded_len, encoded[0..9]) = 2 + 9 = 11 columns.
        assert_eq!(d_nonce.a_columns.len(), 2 + U64_MAX_ENCODED_LEN);
        assert_eq!(d_nonce.b_columns.len(), d_nonce.a_columns.len());
        assert_eq!(d_nonce.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_nonce.b_selector_column, Some(crate::u64_rlp_air::COL_IS_REAL));
        assert_eq!(d_nonce.a_layer_index, 0);
        assert_eq!(d_nonce.b_layer_index, 1);

        let d_balance = make_account_rlp_to_u256_rlp_linkage_descriptor(0, 2);
        // (balance_be[0..32], encoded_len, encoded[0..33]) = 32+1+33 = 66
        assert_eq!(d_balance.a_columns.len(), N32 + 1 + U256_MAX_ENCODED_LEN);
        assert_eq!(d_balance.b_columns.len(), d_balance.a_columns.len());
        assert_eq!(d_balance.b_selector_column, Some(crate::u256_rlp_air::COL_IS_REAL));

        let d_sr = make_account_rlp_storage_root_to_fixed_rlp_linkage_descriptor(0, 3);
        // (field_bytes[0..32], encoded_bytes[0..33]) = 32+33 = 65
        assert_eq!(d_sr.a_columns.len(), N32 + 33);
        assert_eq!(d_sr.b_columns.len(), d_sr.a_columns.len());
        assert_eq!(d_sr.b_selector_column, Some(crate::fixed_rlp_air::COL_IS_REAL));

        let d_ch = make_account_rlp_code_hash_to_fixed_rlp_linkage_descriptor(0, 3);
        assert_eq!(d_ch.a_columns.len(), N32 + 33);
        assert_eq!(d_ch.b_columns.len(), d_ch.a_columns.len());

        // Storage-root and code-hash descriptors both target the same
        // fixed_rlp32 layer but differ in their A-side column choice.
        assert_ne!(d_sr.a_columns, d_ch.a_columns);
        assert_eq!(d_sr.b_columns, d_ch.b_columns);
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 212);
        assert_eq!(MAX_PAYLOAD_LEN, 108);
        assert_eq!(MAX_ENCODED_LEN, 110);
    }

    #[test]
    fn tampered_balance_byte_breaks_canonical_match() {
        // Documents the tampering surface: changing a balance byte
        // without re-deriving balance_enc no longer matches the
        // canonical encoder, but the LogUp linkage to u256_rlp_air is
        // what catches it cryptographically.
        let mut a = funded_account();
        let original_canonical = crate::account::account_rlp(&a);
        a.balance[31] ^= 0xff;
        let tampered_canonical = crate::account::account_rlp(&a);
        assert_ne!(original_canonical, tampered_canonical);
    }
}
