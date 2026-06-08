//! Algebraic EIP-2930 / EIP-1559 access-list RLP encoding AIR.
//!
//! Phase A2 #59 follow-on: lifts the per-access-list-entry RLP
//! encoding into row-local constraints + cross-AIR LogUp descriptors
//! that bind the per-entry encoded windows to the dedicated
//! [`crate::fixed_rlp20_air`] (address) and
//! [`crate::fixed_rlp_air`] (each 32-byte storage key) gadgets, and
//! the per-entry encoded blob region to the deferred access-list slot
//! in [`crate::tx_rlp_air`].
//!
//! # Encoding shape
//!
//! A canonical EIP-2930 access-list entry RLP is `RLP([address, [keys...]])`:
//!
//! ```text
//! Entry_RLP =
//!     ENTRY_OUTER_PREFIX                 (1 or 2 bytes)
//!  || ADDRESS_RLP                        (21 bytes,    0x94 || addr)
//!  || KEYS_LIST_PREFIX                   (1 or 2 bytes)
//!  || KEY_RLP_0 || ... || KEY_RLP_{n-1}  (n × 33 bytes; n = num_keys ∈ {0..=8})
//! ```
//!
//! Keys-inner length = `33 * num_keys`. For num_keys ∈ {0, 1} inner is
//! < 56 → 1-byte prefix `0xc0 + inner`; for num_keys ∈ {2..=8} inner
//! is ≥ 66 → long form `[0xf8, inner]` (2 bytes).
//!
//! Entry payload-length ranges from `21 + 1 = 22` (no keys) up to
//! `21 + 2 + 264 = 287` (8 keys). For payload < 56 the outer prefix
//! is one byte; otherwise long form 2 bytes. Concretely:
//!   - num_keys = 0 → payload = 22 → outer = 1 byte
//!   - num_keys = 1 → payload = 21 + 1 + 33 = 55 → outer = 1 byte
//!   - num_keys ≥ 2 → payload ≥ 21 + 2 + 66 = 89 → outer = 2 bytes
//!
//! # What this AIR enforces algebraically (per row, gated by IS_REAL)
//!
//! 1. `IS_REAL` binary.
//! 2. `IS_KEYS_LONG` binary; `IS_OUTER_LONG` binary.
//! 3. Key-active flags: each binary, monotonically non-increasing,
//!    sum = `NUM_KEYS`. (`NUM_KEYS ≤ 8` follows from the 8-slot sum +
//!    binary.)
//! 4. Case splits for `num_keys`:
//!    - `(1 - IS_KEYS_LONG) * NUM_KEYS * (NUM_KEYS - 1) = 0`
//!      [short form ⇒ NUM_KEYS ∈ {0, 1}]
//!    - `IS_KEYS_LONG * Π_{k=2..=8} (NUM_KEYS - k) = 0`
//!      [long form ⇒ NUM_KEYS ∈ {2..=8}]
//! 5. `KEYS_INNER_LEN = 33 * NUM_KEYS`.
//! 6. `KEYS_PREFIX_LEN = 1 + IS_KEYS_LONG`.
//! 7. `KEYS_TOTAL_LEN = KEYS_PREFIX_LEN + KEYS_INNER_LEN`.
//! 8. `PAYLOAD_LEN = 21 + KEYS_TOTAL_LEN`.
//! 9. `OUTER_PREFIX_LEN = 1 + IS_OUTER_LONG`.
//! 10. `ENCODED_LEN = OUTER_PREFIX_LEN + PAYLOAD_LEN`.
//!
//! All exposed 8-bit byte columns get range checks via `LookupTable::range(256)`.
//!
//! Soft selector binding (not enforced algebraically): the host-side
//! witness builder pins `IS_KEYS_LONG` / `IS_OUTER_LONG` consistent
//! with `NUM_KEYS` / `PAYLOAD_LEN`. Strict range arguments tying these
//! flags to numeric thresholds are deferred.
//!
//! # Cross-AIR LogUp linkages (descriptors below)
//!
//! - [`make_access_list_to_address_descriptor`] — binds entry address
//!   bytes to [`crate::fixed_rlp20_air`].
//! - [`make_access_list_to_storage_key_descriptor(k)`] — binds storage
//!   key slot `k` (32 raw bytes) to [`crate::fixed_rlp_air`], gated by
//!   per-key `KEY_ACTIVE[k]`.
//! - [`make_access_list_to_tx_rlp_descriptor`] — for single-entry MVP
//!   access lists that fit within tx_rlp_air's `MAX_ACCESS_LIST_LEN`
//!   slot, binds entry 0's encoded bytes window + length to
//!   [`crate::tx_rlp_air`]'s deferred access-list region. Gated by
//!   `IS_FIRST_ENTRY` on the A side.
//!
//! # Deferred
//!
//! - Multi-entry-list ↔ concatenated tx_rlp blob binding (would
//!   follow the [`crate::logs_list_concat_air`] pattern).
//! - Strict bound `IS_KEYS_LONG ↔ NUM_KEYS ≥ 2` via range argument.
//! - Strict bound `IS_OUTER_LONG ↔ PAYLOAD_LEN ≥ 56`.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const MAX_KEYS_PER_ENTRY: usize = 8;
pub const KEY_BYTES: usize = 32;
pub const ADDRESS_BYTES: usize = 20;
pub const ADDRESS_ENC_LEN: usize = 21; // 0x94 || 20 bytes
pub const KEY_ENC_LEN: usize = 33; // 0xa0 || 32 bytes

/// Worst-case inner concatenation length of the storage-keys sub-list.
pub const KEYS_INNER_MAX: usize = MAX_KEYS_PER_ENTRY * KEY_ENC_LEN; // 264

/// Worst-case per-entry payload (`address_rlp || keys_list_rlp`).
pub const MAX_PAYLOAD_LEN: usize = ADDRESS_ENC_LEN + 2 + KEYS_INNER_MAX; // 287

/// Worst-case per-entry total encoded length.
pub const MAX_ENTRY_ENCODED_LEN: usize = 2 + MAX_PAYLOAD_LEN; // 289

/// Scaffold cap on the number of entries per access list. Real lists
/// can be larger; this is sufficient for typical txs. Beyond this
/// the witness builder errors.
pub const MAX_ENTRIES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

// Bookkeeping selectors.
pub const COL_IS_REAL: usize = 0;
pub const COL_ENTRY_INDEX: usize = 1;
pub const COL_IS_FIRST_ENTRY: usize = 2;
pub const COL_NUM_KEYS: usize = 3;
pub const COL_KEYS_INNER_LEN: usize = 4;
pub const COL_KEYS_PREFIX_LEN: usize = 5;
pub const COL_KEYS_TOTAL_LEN: usize = 6;
pub const COL_PAYLOAD_LEN: usize = 7;
pub const COL_OUTER_PREFIX_LEN: usize = 8;
pub const COL_ENCODED_LEN: usize = 9;
pub const COL_IS_KEYS_LONG: usize = 10;
pub const COL_IS_OUTER_LONG: usize = 11;

// Address bytes (20).
pub const COL_ADDRESS_OFFSET: usize = 12;
pub const COL_ADDRESS_END: usize = COL_ADDRESS_OFFSET + ADDRESS_BYTES; // 32

// Key-active flags (8 cols) + per-key raw 32 bytes (8 × 32 = 256).
pub const COL_KEY_ACTIVE_OFFSET: usize = COL_ADDRESS_END;                       // 32
pub const COL_KEY_ACTIVE_END: usize = COL_KEY_ACTIVE_OFFSET + MAX_KEYS_PER_ENTRY; // 40
pub const COL_KEY_BYTES_OFFSET: usize = COL_KEY_ACTIVE_END;                     // 40
pub const COL_KEY_BYTES_END: usize =
    COL_KEY_BYTES_OFFSET + MAX_KEYS_PER_ENTRY * KEY_BYTES; // 296

// Per-entry encoded byte window (mirrors the canonical RLP encoding).
pub const COL_ENCODED_OFFSET: usize = COL_KEY_BYTES_END;                          // 296
pub const COL_ENCODED_END: usize = COL_ENCODED_OFFSET + MAX_ENTRY_ENCODED_LEN;    // 585

pub const NUM_COLUMNS: usize = COL_ENCODED_END;                                   // 585

/// Row-local constraint count.
///
///   0: is_real binary
///   1: is_keys_long binary
///   2: is_outer_long binary
///   3: is_first_entry binary
///   4: is_first_entry * entry_index = 0
///   5: (1 - is_keys_long) * num_keys * (num_keys - 1) = 0
///   6: is_keys_long * Π_{k=2..=8} (num_keys - k) = 0
///   7: key_active[k] binary, β-RLC over k ∈ 0..8
///   8: key_active monotonic, β-RLC
///   9: Σ key_active[k] = num_keys
///  10: keys_inner_len = 33 * num_keys
///  11: keys_prefix_len = 1 + is_keys_long
///  12: keys_total_len = keys_prefix_len + keys_inner_len
///  13: payload_len = 21 + keys_total_len
///  14: outer_prefix_len = 1 + is_outer_long
///  15: encoded_len = outer_prefix_len + payload_len
pub const NUM_ROW_CONSTRAINTS: usize = 16;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AccessListRow {
    pub entry_index: usize,
    pub address: [u8; ADDRESS_BYTES],
    pub num_keys: usize,
    pub storage_keys: Vec<[u8; KEY_BYTES]>, // ≤ MAX_KEYS_PER_ENTRY
    pub encoded_bytes: Vec<u8>,
    pub encoded_len: usize,
    pub keys_inner_len: usize,
    pub keys_prefix_len: usize,
    pub keys_total_len: usize,
    pub payload_len: usize,
    pub outer_prefix_len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AccessListWitness {
    pub rows: Vec<AccessListRow>,
}

/// Build a per-entry row.
pub fn from_entry(
    entry_index: usize,
    address: [u8; ADDRESS_BYTES],
    storage_keys: &[[u8; KEY_BYTES]],
) -> Result<AccessListRow, &'static str> {
    if storage_keys.len() > MAX_KEYS_PER_ENTRY {
        return Err("access_list_air: too many storage keys for one entry (max 8)");
    }
    let num_keys = storage_keys.len();
    let keys_inner_len = num_keys * KEY_ENC_LEN;
    let keys_prefix_len = if keys_inner_len < 56 { 1 } else { 2 };
    let keys_total_len = keys_prefix_len + keys_inner_len;
    let payload_len = ADDRESS_ENC_LEN + keys_total_len;
    let outer_prefix_len = if payload_len < 56 { 1 } else { 2 };
    let encoded_len = outer_prefix_len + payload_len;

    if encoded_len > MAX_ENTRY_ENCODED_LEN {
        return Err("access_list_air: entry encoded length exceeds MAX_ENTRY_ENCODED_LEN");
    }

    let mut encoded_bytes = Vec::with_capacity(encoded_len);
    // Outer prefix.
    if outer_prefix_len == 1 {
        encoded_bytes.push(0xc0u8 + payload_len as u8);
    } else {
        encoded_bytes.push(0xf8u8);
        encoded_bytes.push(payload_len as u8);
    }
    // Address RLP: 0x94 || addr[0..20].
    encoded_bytes.push(crate::fixed_rlp20_air::RLP20_PREFIX);
    encoded_bytes.extend_from_slice(&address);
    // Keys list prefix.
    if keys_prefix_len == 1 {
        encoded_bytes.push(0xc0u8 + keys_inner_len as u8);
    } else {
        encoded_bytes.push(0xf8u8);
        encoded_bytes.push(keys_inner_len as u8);
    }
    // Keys inner: 33-byte chunks.
    for k in storage_keys {
        encoded_bytes.push(crate::fixed_rlp_air::RLP32_PREFIX);
        encoded_bytes.extend_from_slice(k);
    }

    debug_assert_eq!(encoded_bytes.len(), encoded_len);

    Ok(AccessListRow {
        entry_index,
        address,
        num_keys,
        storage_keys: storage_keys.to_vec(),
        encoded_bytes,
        encoded_len,
        keys_inner_len,
        keys_prefix_len,
        keys_total_len,
        payload_len,
        outer_prefix_len,
    })
}

impl AccessListWitness {
    /// Build a witness from a full access list:
    /// `entries[i] = (address, storage_keys)`.
    pub fn from_access_list(
        entries: &[([u8; ADDRESS_BYTES], Vec<[u8; KEY_BYTES]>)],
    ) -> Result<Self, &'static str> {
        if entries.len() > MAX_ENTRIES {
            return Err("access_list_air: too many access-list entries (max 8)");
        }
        let mut rows = Vec::with_capacity(entries.len());
        for (i, (addr, keys)) in entries.iter().enumerate() {
            rows.push(from_entry(i, *addr, keys)?);
        }
        Ok(Self { rows })
    }
}

/// Public top-level builder matching the task description signature.
pub fn from_access_list(
    entries: &[([u8; ADDRESS_BYTES], Vec<[u8; KEY_BYTES]>)],
) -> AccessListWitness {
    AccessListWitness::from_access_list(entries)
        .expect("from_access_list: input violated AIR bounds")
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &AccessListWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        columns[COL_IS_REAL][r] = one.clone();
        columns[COL_ENTRY_INDEX][r] = Scalar::from_u64(row.entry_index as u64, curve);
        columns[COL_IS_FIRST_ENTRY][r] =
            Scalar::from_u64(if row.entry_index == 0 { 1 } else { 0 }, curve);
        columns[COL_NUM_KEYS][r] = Scalar::from_u64(row.num_keys as u64, curve);
        columns[COL_KEYS_INNER_LEN][r] = Scalar::from_u64(row.keys_inner_len as u64, curve);
        columns[COL_KEYS_PREFIX_LEN][r] = Scalar::from_u64(row.keys_prefix_len as u64, curve);
        columns[COL_KEYS_TOTAL_LEN][r] = Scalar::from_u64(row.keys_total_len as u64, curve);
        columns[COL_PAYLOAD_LEN][r] = Scalar::from_u64(row.payload_len as u64, curve);
        columns[COL_OUTER_PREFIX_LEN][r] = Scalar::from_u64(row.outer_prefix_len as u64, curve);
        columns[COL_ENCODED_LEN][r] = Scalar::from_u64(row.encoded_len as u64, curve);
        columns[COL_IS_KEYS_LONG][r] =
            Scalar::from_u64(if row.keys_prefix_len == 2 { 1 } else { 0 }, curve);
        columns[COL_IS_OUTER_LONG][r] =
            Scalar::from_u64(if row.outer_prefix_len == 2 { 1 } else { 0 }, curve);

        // Address bytes.
        for k in 0..ADDRESS_BYTES {
            columns[COL_ADDRESS_OFFSET + k][r] =
                Scalar::from_u64(row.address[k] as u64, curve);
        }

        // Key active flags + per-key raw 32-byte payloads.
        for k in 0..MAX_KEYS_PER_ENTRY {
            let active = if k < row.num_keys { 1u64 } else { 0 };
            columns[COL_KEY_ACTIVE_OFFSET + k][r] = Scalar::from_u64(active, curve);
            let key = if k < row.num_keys { row.storage_keys[k] } else { [0u8; KEY_BYTES] };
            for b in 0..KEY_BYTES {
                columns[COL_KEY_BYTES_OFFSET + k * KEY_BYTES + b][r] =
                    Scalar::from_u64(key[b] as u64, curve);
            }
        }

        // Encoded window (zero-padded out to MAX_ENTRY_ENCODED_LEN).
        for k in 0..MAX_ENTRY_ENCODED_LEN {
            let byte = if k < row.encoded_bytes.len() { row.encoded_bytes[k] } else { 0 };
            columns[COL_ENCODED_OFFSET + k][r] = Scalar::from_u64(byte as u64, curve);
        }
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct AccessListConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AccessListConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for AccessListConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_keys_long_binary".into(),
            "is_outer_long_binary".into(),
            "is_first_entry_binary".into(),
            "is_first_entry_implies_index_zero".into(),
            "keys_short_implies_num_keys_le_1".into(),
            "keys_long_implies_num_keys_ge_2".into(),
            "key_active_binary_rlc".into(),
            "key_active_monotonic_rlc".into(),
            "key_active_sum_eq_num_keys".into(),
            "keys_inner_len_eq_33_times_num_keys".into(),
            "keys_prefix_len_formula".into(),
            "keys_total_len_formula".into(),
            "payload_len_formula".into(),
            "outer_prefix_len_formula".into(),
            "encoded_len_formula".into(),
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
        let c21 = Scalar::from_u64(ADDRESS_ENC_LEN as u64, curve);
        let c33 = Scalar::from_u64(KEY_ENC_LEN as u64, curve);
        let beta = Scalar::from_u64(7, curve);

        let mk = || vec![Scalar::zero(curve); n];
        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS).map(|_| mk()).collect();

        for r in 0..n {
            let ir = &columns[COL_IS_REAL][r];
            out[0][r] = ir.mul(&ir.sub(&one));

            let ikl = &columns[COL_IS_KEYS_LONG][r];
            out[1][r] = ikl.mul(&ikl.sub(&one));
            let iol = &columns[COL_IS_OUTER_LONG][r];
            out[2][r] = iol.mul(&iol.sub(&one));

            let ife = &columns[COL_IS_FIRST_ENTRY][r];
            out[3][r] = ife.mul(&ife.sub(&one));

            let ei = &columns[COL_ENTRY_INDEX][r];
            out[4][r] = ife.mul(ei);

            let nk = &columns[COL_NUM_KEYS][r];
            let not_ikl = one.sub(ikl);
            let nk_m1 = nk.sub(&one);
            // (1 - ikl) * num_keys * (num_keys - 1), gated by is_real
            out[5][r] = ir.mul(&not_ikl.mul(&nk.mul(&nk_m1)));

            // ikl * Π_{k=2..=8} (num_keys - k), gated by is_real
            let mut prod = nk.sub(&Scalar::from_u64(2, curve));
            for k in 3..=8u64 {
                prod = prod.mul(&nk.sub(&Scalar::from_u64(k, curve)));
            }
            out[6][r] = ir.mul(&ikl.mul(&prod));

            // key_active binary / monotonic / sum
            let mut acc_bin = Scalar::zero(curve);
            let mut acc_mono = Scalar::zero(curve);
            let mut acc_sum = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..MAX_KEYS_PER_ENTRY {
                let a = &columns[COL_KEY_ACTIVE_OFFSET + k][r];
                acc_bin = acc_bin.add(&bp.mul(&a.mul(&a.sub(&one))));
                acc_sum = acc_sum.add(a);
                if k < MAX_KEYS_PER_ENTRY - 1 {
                    let a1 = &columns[COL_KEY_ACTIVE_OFFSET + k + 1][r];
                    acc_mono = acc_mono.add(&bp.mul(&a1.mul(&one.sub(a))));
                }
                bp = bp.mul(&beta);
            }
            out[7][r] = ir.mul(&acc_bin);
            out[8][r] = ir.mul(&acc_mono);
            out[9][r] = ir.mul(&acc_sum.sub(nk));

            // keys_inner_len = 33 * num_keys
            let kil = &columns[COL_KEYS_INNER_LEN][r];
            out[10][r] = ir.mul(&kil.sub(&c33.mul(nk)));

            // keys_prefix_len = 1 + is_keys_long
            let kpl = &columns[COL_KEYS_PREFIX_LEN][r];
            out[11][r] = ir.mul(&kpl.sub(&one.add(ikl)));

            // keys_total_len = keys_prefix_len + keys_inner_len
            let ktl = &columns[COL_KEYS_TOTAL_LEN][r];
            out[12][r] = ir.mul(&ktl.sub(&kpl.add(kil)));

            // payload_len = 21 + keys_total_len
            let pl = &columns[COL_PAYLOAD_LEN][r];
            out[13][r] = ir.mul(&pl.sub(&c21.add(ktl)));

            // outer_prefix_len = 1 + is_outer_long
            let opl = &columns[COL_OUTER_PREFIX_LEN][r];
            out[14][r] = ir.mul(&opl.sub(&one.add(iol)));

            // encoded_len = outer_prefix_len + payload_len
            let el = &columns[COL_ENCODED_LEN][r];
            out[15][r] = ir.mul(&el.sub(&opl.add(pl)));
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let c21 = Scalar::from_u64(ADDRESS_ENC_LEN as u64, curve);
        let c33 = Scalar::from_u64(KEY_ENC_LEN as u64, curve);

        let ir = &col_evals[COL_IS_REAL];
        let ikl = &col_evals[COL_IS_KEYS_LONG];
        let iol = &col_evals[COL_IS_OUTER_LONG];
        let ife = &col_evals[COL_IS_FIRST_ENTRY];
        let ei = &col_evals[COL_ENTRY_INDEX];
        let nk = &col_evals[COL_NUM_KEYS];
        let kil = &col_evals[COL_KEYS_INNER_LEN];
        let kpl = &col_evals[COL_KEYS_PREFIX_LEN];
        let ktl = &col_evals[COL_KEYS_TOTAL_LEN];
        let pl = &col_evals[COL_PAYLOAD_LEN];
        let opl = &col_evals[COL_OUTER_PREFIX_LEN];
        let el = &col_evals[COL_ENCODED_LEN];

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(ir.mul(&ir.sub(&one)));
        bodies.push(ikl.mul(&ikl.sub(&one)));
        bodies.push(iol.mul(&iol.sub(&one)));
        bodies.push(ife.mul(&ife.sub(&one)));
        bodies.push(ife.mul(ei));

        let not_ikl = one.sub(ikl);
        bodies.push(ir.mul(&not_ikl.mul(&nk.mul(&nk.sub(&one)))));

        let mut prod = nk.sub(&Scalar::from_u64(2, curve));
        for k in 3..=8u64 {
            prod = prod.mul(&nk.sub(&Scalar::from_u64(k, curve)));
        }
        bodies.push(ir.mul(&ikl.mul(&prod)));

        let mut acc_bin = Scalar::zero(curve);
        let mut acc_mono = Scalar::zero(curve);
        let mut acc_sum = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_KEYS_PER_ENTRY {
            let a = &col_evals[COL_KEY_ACTIVE_OFFSET + k];
            acc_bin = acc_bin.add(&bp.mul(&a.mul(&a.sub(&one))));
            acc_sum = acc_sum.add(a);
            if k < MAX_KEYS_PER_ENTRY - 1 {
                let a1 = &col_evals[COL_KEY_ACTIVE_OFFSET + k + 1];
                acc_mono = acc_mono.add(&bp.mul(&a1.mul(&one.sub(a))));
            }
            bp = bp.mul(alpha);
        }
        bodies.push(ir.mul(&acc_bin));
        bodies.push(ir.mul(&acc_mono));
        bodies.push(ir.mul(&acc_sum.sub(nk)));

        bodies.push(ir.mul(&kil.sub(&c33.mul(nk))));
        bodies.push(ir.mul(&kpl.sub(&one.add(ikl))));
        bodies.push(ir.mul(&ktl.sub(&kpl.add(kil))));
        bodies.push(ir.mul(&pl.sub(&c21.add(ktl))));
        bodies.push(ir.mul(&opl.sub(&one.add(iol))));
        bodies.push(ir.mul(&el.sub(&opl.add(pl))));

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
        let c21_poly = vec![Scalar::from_u64(ADDRESS_ENC_LEN as u64, curve)];
        let c33_poly = vec![Scalar::from_u64(KEY_ENC_LEN as u64, curve)];

        let ir = &col_coeffs[COL_IS_REAL];
        let ikl = &col_coeffs[COL_IS_KEYS_LONG];
        let iol = &col_coeffs[COL_IS_OUTER_LONG];
        let ife = &col_coeffs[COL_IS_FIRST_ENTRY];
        let ei = &col_coeffs[COL_ENTRY_INDEX];
        let nk = &col_coeffs[COL_NUM_KEYS];
        let kil = &col_coeffs[COL_KEYS_INNER_LEN];
        let kpl = &col_coeffs[COL_KEYS_PREFIX_LEN];
        let ktl = &col_coeffs[COL_KEYS_TOTAL_LEN];
        let pl = &col_coeffs[COL_PAYLOAD_LEN];
        let opl = &col_coeffs[COL_OUTER_PREFIX_LEN];
        let el = &col_coeffs[COL_ENCODED_LEN];

        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_poly, curve), curve);

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(bin(ir));
        bodies.push(bin(ikl));
        bodies.push(bin(iol));
        bodies.push(bin(ife));
        bodies.push(poly_mul(ife, ei, curve));

        // (1 - ikl) * nk * (nk - 1), times ir.
        let not_ikl = poly_sub(&one_poly, ikl, curve);
        let nk_m1 = poly_sub(nk, &one_poly, curve);
        let nk_nk_m1 = poly_mul(nk, &nk_m1, curve);
        let body5 = poly_mul(&not_ikl, &nk_nk_m1, curve);
        bodies.push(poly_mul(ir, &body5, curve));

        // ikl * Π_{k=2..=8} (nk - k), times ir.
        let mut prod = poly_sub(nk, &vec![Scalar::from_u64(2, curve)], curve);
        for k in 3..=8u64 {
            let term = poly_sub(nk, &vec![Scalar::from_u64(k, curve)], curve);
            prod = poly_mul(&prod, &term, curve);
        }
        let body6 = poly_mul(ikl, &prod, curve);
        bodies.push(poly_mul(ir, &body6, curve));

        // key_active binary / monotonic / sum
        let mut acc_bin = vec![Scalar::zero(curve)];
        let mut acc_mono = vec![Scalar::zero(curve)];
        let mut acc_sum = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_KEYS_PER_ENTRY {
            let a = &col_coeffs[COL_KEY_ACTIVE_OFFSET + k];
            acc_bin = poly_add(&acc_bin, &poly_scalar_mul(&bin(a), &bp), curve);
            acc_sum = poly_add(&acc_sum, a, curve);
            if k < MAX_KEYS_PER_ENTRY - 1 {
                let a1 = &col_coeffs[COL_KEY_ACTIVE_OFFSET + k + 1];
                let term = poly_mul(a1, &poly_sub(&one_poly, a, curve), curve);
                acc_mono = poly_add(&acc_mono, &poly_scalar_mul(&term, &bp), curve);
            }
            bp = bp.mul(alpha);
        }
        bodies.push(poly_mul(ir, &acc_bin, curve));
        bodies.push(poly_mul(ir, &acc_mono, curve));
        bodies.push(poly_mul(ir, &poly_sub(&acc_sum, nk, curve), curve));

        // keys_inner_len = 33 * num_keys
        let c33_nk = poly_mul(&c33_poly, nk, curve);
        bodies.push(poly_mul(ir, &poly_sub(kil, &c33_nk, curve), curve));

        // keys_prefix_len = 1 + is_keys_long
        let one_plus_ikl = poly_add(&one_poly, ikl, curve);
        bodies.push(poly_mul(ir, &poly_sub(kpl, &one_plus_ikl, curve), curve));

        // keys_total_len = keys_prefix_len + keys_inner_len
        let sum_t = poly_add(kpl, kil, curve);
        bodies.push(poly_mul(ir, &poly_sub(ktl, &sum_t, curve), curve));

        // payload_len = 21 + keys_total_len
        let expected_pl = poly_add(&c21_poly, ktl, curve);
        bodies.push(poly_mul(ir, &poly_sub(pl, &expected_pl, curve), curve));

        // outer_prefix_len = 1 + is_outer_long
        let one_plus_iol = poly_add(&one_poly, iol, curve);
        bodies.push(poly_mul(ir, &poly_sub(opl, &one_plus_iol, curve), curve));

        // encoded_len = outer_prefix_len + payload_len
        let sum_e = poly_add(opl, pl, curve);
        bodies.push(poly_mul(ir, &poly_sub(el, &sum_e, curve), curve));

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

        for k in 0..ADDRESS_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("access_list_address_{}_8bit", k),
                    column_index: COL_ADDRESS_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..(MAX_KEYS_PER_ENTRY * KEY_BYTES) {
            declarations.push((
                LookupDeclaration {
                    label: format!("access_list_key_byte_{}_8bit", k),
                    column_index: COL_KEY_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MAX_ENTRY_ENCODED_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("access_list_enc_{}_8bit", k),
                    column_index: COL_ENCODED_OFFSET + k,
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

/// Binds the 20 address bytes on this AIR to
/// [`crate::fixed_rlp20_air`]'s field byte columns. Combined with the
/// fixed_rlp20 RLP constraints, this pins the address RLP encoding
/// window `[0x94, addr[0], ..., addr[19]]` inside the entry's
/// encoded byte stream.
pub fn make_access_list_to_address_descriptor(
    access_list_layer_index: usize,
    fixed_rlp20_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(ADDRESS_BYTES);
    let mut b_columns = Vec::with_capacity(ADDRESS_BYTES);
    for k in 0..ADDRESS_BYTES {
        a_columns.push(COL_ADDRESS_OFFSET + k);
        b_columns.push(crate::fixed_rlp20_air::COL_FIELD_BYTE_OFFSET + k);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "access_list_to_fixed_rlp20_address_v1".into(),
        a_layer_index: access_list_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: fixed_rlp20_layer_index,
        b_columns,
        b_selector_column: Some(crate::fixed_rlp20_air::COL_IS_REAL),
    }
}

/// Binds the 32 raw bytes of storage-key slot `key_index ∈ 0..8` on
/// this AIR to [`crate::fixed_rlp_air`]'s field byte columns, gated by
/// the per-key `KEY_ACTIVE[key_index]` flag on the A side. Combined
/// with the fixed_rlp32 RLP constraints this pins the storage-key
/// RLP encoding `[0xa0, k[0], ..., k[31]]`.
pub fn make_access_list_to_storage_key_descriptor(
    key_index: usize,
    access_list_layer_index: usize,
    fixed_rlp32_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert!(key_index < MAX_KEYS_PER_ENTRY, "key_index out of range");
    let mut a_columns = Vec::with_capacity(KEY_BYTES);
    let mut b_columns = Vec::with_capacity(KEY_BYTES);
    for b in 0..KEY_BYTES {
        a_columns.push(COL_KEY_BYTES_OFFSET + key_index * KEY_BYTES + b);
        b_columns.push(crate::fixed_rlp_air::COL_FIELD_BYTE_OFFSET + b);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("access_list_to_fixed_rlp32_key_{}_v1", key_index),
        a_layer_index: access_list_layer_index,
        a_columns,
        a_selector_column: Some(COL_KEY_ACTIVE_OFFSET + key_index),
        b_layer_index: fixed_rlp32_layer_index,
        b_columns,
        b_selector_column: Some(crate::fixed_rlp_air::COL_IS_REAL),
    }
}

/// Binds the per-entry encoded byte window on this AIR (entry 0 only,
/// gated by `IS_FIRST_ENTRY`) to [`crate::tx_rlp_air`]'s deferred
/// access-list region. Limited by the smaller of the two windows
/// (`MAX_ACCESS_LIST_LEN`), so only single-entry access lists whose
/// canonical encoding fits within tx_rlp_air's slot are bound here.
/// Larger multi-entry lists require a list-concatenation gadget
/// (deferred).
pub fn make_access_list_to_tx_rlp_descriptor(
    access_list_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let n = crate::tx_rlp_air::MAX_ACCESS_LIST_LEN
        .min(MAX_ENTRY_ENCODED_LEN);
    let mut a_columns = Vec::with_capacity(n + 1);
    let mut b_columns = Vec::with_capacity(n + 1);
    for k in 0..n {
        a_columns.push(COL_ENCODED_OFFSET + k);
        b_columns.push(crate::tx_rlp_air::COL_ACCESS_LIST_BYTE_OFFSET + k);
    }
    a_columns.push(COL_ENCODED_LEN);
    b_columns.push(crate::tx_rlp_air::COL_ACCESS_LIST_LEN);
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "access_list_to_tx_rlp_single_entry_v1".into(),
        a_layer_index: access_list_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_FIRST_ENTRY),
        b_layer_index: tx_rlp_layer_index,
        b_columns,
        b_selector_column: Some(crate::tx_rlp_air::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_list() -> Vec<([u8; 20], Vec<[u8; 32]>)> {
        vec![]
    }

    fn single_entry_no_keys() -> Vec<([u8; 20], Vec<[u8; 32]>)> {
        vec![([0x11u8; 20], vec![])]
    }

    fn single_entry_one_key() -> Vec<([u8; 20], Vec<[u8; 32]>)> {
        vec![([0x22u8; 20], vec![[0xaau8; 32]])]
    }

    fn multi_entry_multi_keys() -> Vec<([u8; 20], Vec<[u8; 32]>)> {
        vec![
            ([0x33u8; 20], vec![[0xbbu8; 32], [0xccu8; 32]]),
            ([0x44u8; 20], vec![[0xddu8; 32], [0xeeu8; 32], [0xffu8; 32]]),
            ([0x55u8; 20], vec![]),
        ]
    }

    fn build_for(
        entries: &[([u8; ADDRESS_BYTES], Vec<[u8; KEY_BYTES]>)],
    ) -> (AccessListWitness, TracePolynomials) {
        let w = AccessListWitness::from_access_list(entries).unwrap();
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        (w, t)
    }

    fn assert_constraints_zero(trace: &TracePolynomials) {
        let cs = AccessListConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} (label {}) at row {} = {:?}",
                    i,
                    cs.constraint_labels()[i],
                    r,
                    val,
                );
            }
        }
    }

    #[test]
    fn empty_access_list_witness() {
        // Empty list = 0 rows. The witness is empty but should still
        // be valid (degenerate case). `nearest_power_of_two(0.max(1))
        // = 1` so the trace has one row of zeros — IS_REAL = 0 there
        // and all gated constraints reduce to 0.
        let entries = empty_list();
        let (w, trace) = build_for(&entries);
        assert_eq!(w.rows.len(), 0);
        assert_eq!(trace.num_rows, 0);
        assert_constraints_zero(&trace);
    }

    #[test]
    fn single_entry_no_keys_encodes_canonically() {
        let entries = single_entry_no_keys();
        let row = from_entry(0, entries[0].0, &entries[0].1).unwrap();
        // payload = 21 (addr) + 1 (0xc0 empty keys list) = 22
        // outer prefix = 0xc0 + 22 = 0xd6 (1 byte)
        // encoded_len = 1 + 22 = 23
        assert_eq!(row.num_keys, 0);
        assert_eq!(row.keys_inner_len, 0);
        assert_eq!(row.keys_prefix_len, 1);
        assert_eq!(row.keys_total_len, 1);
        assert_eq!(row.payload_len, 22);
        assert_eq!(row.outer_prefix_len, 1);
        assert_eq!(row.encoded_len, 23);
        assert_eq!(row.encoded_bytes.len(), 23);
        assert_eq!(row.encoded_bytes[0], 0xd6);
        assert_eq!(row.encoded_bytes[1], 0x94);
        // address bytes
        for k in 0..20 {
            assert_eq!(row.encoded_bytes[2 + k], 0x11);
        }
        // empty keys list prefix
        assert_eq!(row.encoded_bytes[22], 0xc0);

        let (_w, trace) = build_for(&entries);
        assert_constraints_zero(&trace);
    }

    #[test]
    fn single_entry_one_key_encodes_canonically() {
        let entries = single_entry_one_key();
        let row = from_entry(0, entries[0].0, &entries[0].1).unwrap();
        // payload = 21 + 1 + 33 = 55; outer = 1 byte
        // encoded_len = 1 + 55 = 56
        assert_eq!(row.num_keys, 1);
        assert_eq!(row.keys_inner_len, 33);
        assert_eq!(row.keys_prefix_len, 1);
        assert_eq!(row.keys_total_len, 34);
        assert_eq!(row.payload_len, 55);
        assert_eq!(row.outer_prefix_len, 1);
        assert_eq!(row.encoded_len, 56);
        assert_eq!(row.encoded_bytes[0], 0xc0 + 55);

        let (_w, trace) = build_for(&entries);
        assert_constraints_zero(&trace);
    }

    #[test]
    fn multi_entry_multi_keys_pass_constraints() {
        let entries = multi_entry_multi_keys();
        // Sanity per-row shapes:
        let row0 = from_entry(0, entries[0].0, &entries[0].1).unwrap();
        // 2 keys: inner = 66, prefix = 2 (long), payload = 21+2+66=89, outer=2, enc=91
        assert_eq!(row0.encoded_len, 91);
        assert_eq!(row0.outer_prefix_len, 2);
        assert_eq!(row0.keys_prefix_len, 2);
        let row1 = from_entry(1, entries[1].0, &entries[1].1).unwrap();
        // 3 keys: inner = 99, prefix = 2, payload = 21+2+99=122, outer=2, enc=124
        assert_eq!(row1.encoded_len, 124);
        let row2 = from_entry(2, entries[2].0, &entries[2].1).unwrap();
        assert_eq!(row2.encoded_len, 23);

        let (_w, trace) = build_for(&entries);
        assert_constraints_zero(&trace);
    }

    #[test]
    fn tampered_encoded_len_detected() {
        let entries = single_entry_one_key();
        let (_w, trace) = build_for(&entries);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let cur = cols[COL_ENCODED_LEN][0].to_u64();
        cols[COL_ENCODED_LEN][0] = Scalar::from_u64(cur + 1, CurveType::Bls48581);
        let cs = AccessListConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 15 = encoded_len_formula
        assert!(!res[15][0].is_zero(), "encoded_len_formula must fire");
    }

    #[test]
    fn tampered_key_active_monotonic_detected() {
        // 2 keys → active = [1, 1, 0, 0, 0, 0, 0, 0]. Force a 0→1
        // transition at slot 3 → constraint 8 (monotonic) must fire.
        let entries = vec![(
            [0x77u8; 20],
            vec![[0x88u8; 32], [0x99u8; 32]],
        )];
        let (_w, trace) = build_for(&entries);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_KEY_ACTIVE_OFFSET + 3][0] = Scalar::from_u64(1, CurveType::Bls48581);
        let cs = AccessListConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 8 = key_active_monotonic_rlc
        assert!(!res[8][0].is_zero(), "key_active_monotonic must fire");
    }

    #[test]
    fn tampered_keys_inner_len_detected() {
        let entries = single_entry_one_key();
        let (_w, trace) = build_for(&entries);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let cur = cols[COL_KEYS_INNER_LEN][0].to_u64();
        cols[COL_KEYS_INNER_LEN][0] = Scalar::from_u64(cur + 33, CurveType::Bls48581);
        let cs = AccessListConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 10 = keys_inner_len_eq_33_times_num_keys
        assert!(!res[10][0].is_zero(), "keys_inner_len formula must fire");
    }

    #[test]
    fn descriptors_well_formed() {
        let d_addr = make_access_list_to_address_descriptor(0, 1);
        assert_eq!(d_addr.label, "access_list_to_fixed_rlp20_address_v1");
        assert_eq!(d_addr.a_columns.len(), ADDRESS_BYTES);
        assert_eq!(d_addr.b_columns.len(), ADDRESS_BYTES);
        assert_eq!(d_addr.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_addr.b_selector_column,
            Some(crate::fixed_rlp20_air::COL_IS_REAL),
        );

        for k in 0..MAX_KEYS_PER_ENTRY {
            let d_k = make_access_list_to_storage_key_descriptor(k, 0, 2);
            assert_eq!(
                d_k.label,
                format!("access_list_to_fixed_rlp32_key_{}_v1", k),
            );
            assert_eq!(d_k.a_columns.len(), KEY_BYTES);
            assert_eq!(d_k.b_columns.len(), KEY_BYTES);
            assert_eq!(
                d_k.a_selector_column,
                Some(COL_KEY_ACTIVE_OFFSET + k),
            );
            assert_eq!(
                d_k.b_selector_column,
                Some(crate::fixed_rlp_air::COL_IS_REAL),
            );
        }

        let d_tx = make_access_list_to_tx_rlp_descriptor(0, 3);
        assert_eq!(d_tx.label, "access_list_to_tx_rlp_single_entry_v1");
        let expected_tuple_len = crate::tx_rlp_air::MAX_ACCESS_LIST_LEN
            .min(MAX_ENTRY_ENCODED_LEN) + 1;
        assert_eq!(d_tx.a_columns.len(), expected_tuple_len);
        assert_eq!(d_tx.b_columns.len(), expected_tuple_len);
        assert_eq!(d_tx.a_selector_column, Some(COL_IS_FIRST_ENTRY));
        assert_eq!(
            d_tx.b_selector_column,
            Some(crate::tx_rlp_air::COL_IS_REAL),
        );
    }

    #[test]
    fn rejects_too_many_keys() {
        let entries = vec![([0u8; 20], vec![[0u8; 32]; 9])];
        let err = AccessListWitness::from_access_list(&entries).unwrap_err();
        assert!(err.contains("too many storage keys"), "got: {}", err);
    }

    #[test]
    fn rejects_too_many_entries() {
        let entries: Vec<([u8; 20], Vec<[u8; 32]>)> =
            (0..9).map(|i| ([i as u8; 20], vec![])).collect();
        let err = AccessListWitness::from_access_list(&entries).unwrap_err();
        assert!(err.contains("too many access-list entries"), "got: {}", err);
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 585);
        assert_eq!(MAX_ENTRY_ENCODED_LEN, 289);
        assert_eq!(MAX_PAYLOAD_LEN, 287);
        assert_eq!(KEYS_INNER_MAX, 264);
        assert_eq!(ADDRESS_ENC_LEN, 21);
        assert_eq!(KEY_ENC_LEN, 33);
    }

    #[test]
    fn from_access_list_top_level_builder() {
        let entries = multi_entry_multi_keys();
        let w = from_access_list(&entries);
        assert_eq!(w.rows.len(), entries.len());
        for (r, (addr, keys)) in entries.iter().enumerate() {
            assert_eq!(w.rows[r].address, *addr);
            assert_eq!(w.rows[r].num_keys, keys.len());
            assert_eq!(w.rows[r].entry_index, r);
        }
    }
}
