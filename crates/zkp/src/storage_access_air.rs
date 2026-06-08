//! Storage access gadget AIR — Phase A2 step 1.
//!
//! Bridges EVM `SLOAD`/`SSTORE` rows to the MPT inclusion chain. One
//! row per storage access. Per row, the gadget commits:
//!   - `address[L0..L3]` — 20-byte contract address in low 160 bits of
//!     4-limb U256 (top 96 bits = 0; constrained via byte decomp).
//!   - `slot[L0..L3]` — U256 slot as 4 LE u64 limbs.
//!   - `slot_be[0..32]` — 32 BE bytes of the slot (Keccak input).
//!   - `value[L0..L3]` — U256 value as 4 LE u64 limbs.
//!   - `value_be[0..32]` — 32 BE bytes of the value (MPT value encoding
//!     after RLP wrap, deferred to step 1c).
//!   - `storage_root[0..32]` — 32-byte root of the contract's storage
//!     trie at the time of access.
//!   - `slot_trie_key[0..32]` — `keccak256(slot_be)`, to be bound to
//!     KeccakExtract via cross-AIR LogUp in step 1b.
//!   - `is_write` — 1 for SSTORE, 0 for SLOAD.
//!   - `is_real` — 1 on real rows, 0 on padding.
//!
//! Row-local constraints (step 1a, this commit):
//!   - `is_write * (is_write - 1) = 0` (binary)
//!   - `is_real * (is_real - 1) = 0` (binary)
//!   - 4 slot limb-decomp equations: `slot_limb_j = Σ slot_be[…] * 2^…`
//!     (BE byte decomp into 4 LE u64 limbs)
//!   - 4 value limb-decomp equations (same pattern)
//!
//! 8-bit range checks (LogUp) on every byte column:
//!   `slot_be[0..32]`, `value_be[0..32]`, `storage_root[0..32]`,
//!   `slot_trie_key[0..32]` → 128 byte range checks.
//!
//! Linkages (descriptors built in step 1b — EVM ↔ gadget — and step 1c —
//! gadget ↔ Keccak chain, gadget ↔ MPT inclusion chain — are NOT in
//! this commit). The gadget AIR is standalone-provable today.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_ADDR_L0: usize = 0;
pub const COL_ADDR_L1: usize = 1;
pub const COL_ADDR_L2: usize = 2;
pub const COL_ADDR_L3: usize = 3;

pub const COL_SLOT_L0: usize = 4;
pub const COL_SLOT_L1: usize = 5;
pub const COL_SLOT_L2: usize = 6;
pub const COL_SLOT_L3: usize = 7;

pub const COL_SLOT_BE_OFFSET: usize = 8; // slot_be[0..32] at 8..40

pub const COL_VALUE_L0: usize = 40;
pub const COL_VALUE_L1: usize = 41;
pub const COL_VALUE_L2: usize = 42;
pub const COL_VALUE_L3: usize = 43;

pub const COL_VALUE_BE_OFFSET: usize = 44; // value_be[0..32] at 44..76

pub const COL_STORAGE_ROOT_OFFSET: usize = 76; // 76..108

pub const COL_SLOT_TRIE_KEY_OFFSET: usize = 108; // 108..140

pub const COL_IS_WRITE: usize = 140;
pub const COL_IS_REAL: usize = 141;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 142

/// Row-local constraints:
///   0: is_write binary
///   1: is_real binary
///   2..6: slot limb-decomp (4 equations)
///   6..10: value limb-decomp (4 equations)
pub const NUM_ROW_CONSTRAINTS: usize = 2 + 4 + 4;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness type ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct StorageAccessRow {
    /// 20-byte contract address, BE.
    pub address: [u8; 20],
    /// U256 slot as 4 LE u64 limbs.
    pub slot: [u64; 4],
    /// U256 value as 4 LE u64 limbs.
    pub value: [u64; 4],
    /// 32-byte storage root.
    pub storage_root: [u8; 32],
    /// `is_write == 1` for SSTORE, 0 for SLOAD.
    pub is_write: bool,
}

#[derive(Clone, Debug, Default)]
pub struct StorageAccessWitness {
    pub invocations: Vec<StorageAccessRow>,
}

impl StorageAccessWitness {
    pub fn from_rows(rows: Vec<StorageAccessRow>) -> Self {
        Self { invocations: rows }
    }
}

/// Extract the 32 BE bytes of a U256 held as 4 LE u64 limbs.
fn be_bytes_of_u256(limbs: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        let limb_idx = 3 - (i / 8);
        let byte_in_limb = 7 - (i % 8);
        bytes[i] = ((limbs[limb_idx] >> (byte_in_limb * 8)) & 0xff) as u8;
    }
    bytes
}

/// Convert a 20-byte BE address into 4 LE u64 limbs (top 96 bits = 0).
fn address_to_limbs(address: [u8; 20]) -> [u64; 4] {
    let mut full = [0u8; 32];
    full[12..32].copy_from_slice(&address);
    // Build LE limbs: limb 0 = low 64 bits = full[24..32] BE.
    [
        u64::from_be_bytes([
            full[24], full[25], full[26], full[27],
            full[28], full[29], full[30], full[31],
        ]),
        u64::from_be_bytes([
            full[16], full[17], full[18], full[19],
            full[20], full[21], full[22], full[23],
        ]),
        u64::from_be_bytes([
            full[8],  full[9],  full[10], full[11],
            full[12], full[13], full[14], full[15],
        ]),
        u64::from_be_bytes([
            full[0], full[1], full[2], full[3],
            full[4], full[5], full[6], full[7],
        ]),
    ]
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &StorageAccessWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.invocations.iter().enumerate() {
        // Address.
        let addr_limbs = address_to_limbs(row.address);
        for j in 0..4 {
            columns[COL_ADDR_L0 + j][i] = Scalar::from_u64(addr_limbs[j], curve);
        }
        // Slot.
        for j in 0..4 {
            columns[COL_SLOT_L0 + j][i] = Scalar::from_u64(row.slot[j], curve);
        }
        let slot_be = be_bytes_of_u256(row.slot);
        for k in 0..32 {
            columns[COL_SLOT_BE_OFFSET + k][i] = Scalar::from_u64(slot_be[k] as u64, curve);
        }
        // Value.
        for j in 0..4 {
            columns[COL_VALUE_L0 + j][i] = Scalar::from_u64(row.value[j], curve);
        }
        let value_be = be_bytes_of_u256(row.value);
        for k in 0..32 {
            columns[COL_VALUE_BE_OFFSET + k][i] = Scalar::from_u64(value_be[k] as u64, curve);
        }
        // Storage root.
        for k in 0..32 {
            columns[COL_STORAGE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.storage_root[k] as u64, curve);
        }
        // Slot trie key = keccak256(slot_be).
        let slot_trie_key = crate::keccak::keccak256(&slot_be);
        for k in 0..32 {
            columns[COL_SLOT_TRIE_KEY_OFFSET + k][i] =
                Scalar::from_u64(slot_trie_key[k] as u64, curve);
        }
        // Flags.
        columns[COL_IS_WRITE][i] =
            if row.is_write { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = one.clone();
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

pub struct StorageAccessConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl StorageAccessConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// `(byte_index_in_be, weight)` for the `limb_idx`-th LE u64 limb of a
/// U256 stored as 32 BE bytes (byte 0 = MSB).
fn limb_decomp_targets(limb_idx: usize) -> [(usize, u64); 8] {
    let lo = match limb_idx {
        3 => 0,
        2 => 8,
        1 => 16,
        0 => 24,
        _ => unreachable!(),
    };
    let mut out = [(0usize, 0u64); 8];
    for k in 0..8 {
        out[k] = (lo + k, 1u64 << (8 * (7 - k)));
    }
    out
}

impl VmConstraintSystem for StorageAccessConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "is_write_binary".into(),
            "is_real_binary".into(),
        ];
        for j in 0..4 { labels.push(format!("slot_limb_{}_decomp", j)); }
        for j in 0..4 { labels.push(format!("value_limb_{}_decomp", j)); }
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

        // 0: is_write binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_WRITE][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 1: is_real binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 2..6: slot_limb_j_decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in targets {
                    let b = &columns[COL_SLOT_BE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(pow, curve)));
                }
                c[r] = columns[COL_SLOT_L0 + limb_idx][r].sub(&sum);
            }
            out.push(c);
        }
        // 6..10: value_limb_j_decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in targets {
                    let b = &columns[COL_VALUE_BE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(pow, curve)));
                }
                c[r] = columns[COL_VALUE_L0 + limb_idx][r].sub(&sum);
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

        // 0: is_write binary.
        {
            let v = &col_evals[COL_IS_WRITE];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 2..6: slot limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in targets {
                sum = sum.add(
                    &col_evals[COL_SLOT_BE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(pow, curve)),
                );
            }
            let body = col_evals[COL_SLOT_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6..10: value limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in targets {
                sum = sum.add(
                    &col_evals[COL_VALUE_BE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(pow, curve)),
                );
            }
            let body = col_evals[COL_VALUE_L0 + limb_idx].sub(&sum);
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

        // 0: is_write binary.
        {
            let v = &col_coeffs[COL_IS_WRITE];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 2..6: slot limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in targets {
                let b = &col_coeffs[COL_SLOT_BE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_SLOT_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6..10: value limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in targets {
                let b = &col_coeffs[COL_VALUE_BE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_VALUE_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        acc
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
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 8-bit range check on every byte column:
        //   slot_be[0..32], value_be[0..32], storage_root[0..32],
        //   slot_trie_key[0..32] = 128 byte cols.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("storage_slot_be_{}_8bit", k),
                    column_index: COL_SLOT_BE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("storage_value_be_{}_8bit", k),
                    column_index: COL_VALUE_BE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("storage_root_{}_8bit", k),
                    column_index: COL_STORAGE_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("storage_slot_trie_key_{}_8bit", k),
                    column_index: COL_SLOT_TRIE_KEY_OFFSET + k,
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

/// **Phase A2 step 1c — Keccak chain link**: storage access gadget ↔
/// KeccakExtract on the slot-hash binding.
///
/// A side (this gadget): tuple `(slot_be[0..32], slot_trie_key[0..32])`
/// — 64 cols — gated by `is_real`.
/// B side (KeccakExtract): tuple `(INPUT_BYTE[0..32], OUTPUT_BYTE[0..32])`
/// — 64 cols — gated by `is_real`.
///
/// Algebraically binds `slot_trie_key = keccak256(slot_be)` per access
/// (KeccakExtract's own constraints already enforce the
/// keccak-correctness of `OUTPUT_BYTE` against `INPUT_BYTE`).
///
/// **Caveat**: KeccakExtract's `INPUT_BYTE` is 256 cols (max input
/// length 256). For a 32-byte slot hash, the witness must populate
/// `INPUT_BYTE[0..32]` with `slot_be` and `INPUT_BYTE[32..256]` with
/// zeros + set `INPUT_LEN = 32`. The linkage only projects the first
/// 32 input bytes, leaving the rest unconstrained by this linkage
/// (constrained by KeccakExtract's INPUT_LEN-truncation logic).
pub fn make_storage_to_keccak_extract_linkage_descriptor(
    storage_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let a_columns: Vec<usize> = (0..32)
        .map(|k| COL_SLOT_BE_OFFSET + k)
        .chain((0..32).map(|k| COL_SLOT_TRIE_KEY_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..32)
        .map(|k| ke::COL_INPUT_BYTE_OFFSET + k)
        .chain((0..32).map(|k| ke::COL_OUTPUT_BYTE_OFFSET + k))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "storage_slot_keccak_v1".into(),
        a_layer_index: storage_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// **Phase A2 step 1d (partial)** — storage_access_air's storage_root
/// ↔ mpt_air's root row (gated by `IS_ROOT`).
///
/// Tuple: 32-col `(storage_root_bytes[0..32])` ↔ `(parent_hash[0..32])`
/// of mpt_air row 0 (where parent_hash equals the trie root).
///
/// **Caveat (documented step 1d limitation)**: this linkage binds the
/// ROOT only — every storage gadget row's storage_root must appear as
/// some MPT chain's row-0 parent_hash. The leaf's `(slot_trie_key,
/// value)` is NOT bound by this single linkage; that requires a
/// separate cross-row "leaf summary" constraint propagating leaf data
/// back to row 0 (deferred follow-up). Without leaf binding, a
/// malicious prover could pair a correct storage_root with arbitrary
/// `(slot_trie_key, value)` claims in storage_access_air.
///
/// Until the leaf summary lands, this linkage gives **partial
/// soundness**: storage_root is bound to a real MPT proof's root,
/// but the proof's leaf data isn't required to match what the
/// storage gadget claims. Combined with phase 1c (slot_trie_key =
/// keccak256(slot_be) via KeccakExtract), the slot_trie_key derivation
/// IS bound; only the value remains floating.
pub fn make_storage_root_to_mpt_root_linkage_descriptor(
    storage_layer_index: usize,
    mpt_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;
    let a_columns: Vec<usize> = (0..32).map(|k| COL_STORAGE_ROOT_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..32).map(|k| mpt_col::PARENT_HASH_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "storage_root_to_mpt_root_v1".into(),
        a_layer_index: storage_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_ROOT),
    }
}

/// **Phase A2 step 1d (root + value)** — storage_access_air's
/// `(storage_root, value_be)` ↔ mpt_air's `(parent_hash,
/// claimed_leaf_value_bytes)` gated by `IS_ROOT`.
///
/// 64-col tuple. Binds BOTH the trie root AND the leaf value
/// algebraically (modulo the deferred cross-row constancy constraint
/// on `claimed_leaf_value_bytes` and the leaf-row equality with
/// `value_byte`). With those constraints in place, the prover cannot:
///   - Fabricate a storage_root (binds via parent_hash on row 0)
///   - Fabricate a slot value (binds via claimed_leaf_value_bytes →
///     leaf row's value)
///
/// **Caveat (still deferred)**: the SLOT TRIE KEY is NOT bound by
/// this linkage — the MPT proof's actual walked key isn't exposed
/// as a column on row 0. The slot_trie_key is independently bound
/// via the keccak chain (storage_access_air → KeccakExtract on
/// `slot_trie_key = keccak256(slot_be)`), so the soundness gap is
/// only "the MPT proof being for a DIFFERENT key than what storage
/// claims" — i.e. prover could verify a proof for key K' and pair it
/// with claimed slot_trie_key K via the value match alone. Closing
/// this requires a `claimed_key_bytes` column on mpt_air with a
/// nibble-walk constraint ensuring the walked path equals the
/// claimed key.
pub fn make_storage_root_value_to_mpt_root_linkage_descriptor(
    storage_layer_index: usize,
    mpt_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;
    let a_columns: Vec<usize> = (0..32)
        .map(|k| COL_STORAGE_ROOT_OFFSET + k)
        .chain((0..32).map(|k| COL_VALUE_BE_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..32)
        .map(|k| mpt_col::PARENT_HASH_OFFSET + k)
        .chain((0..32).map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "storage_root_value_to_mpt_v1".into(),
        a_layer_index: storage_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_ROOT),
    }
}

/// **Phase A2 step 1d-full** — storage_access_air's full
/// `(storage_root, value_be, slot_trie_key)` ↔ mpt_air's
/// `(parent_hash, claimed_leaf_value_bytes, claimed_leaf_key_bytes)`
/// gated by `IS_ROOT`.
///
/// 96-col tuple. Combined with mpt_constraints' algebraic constraints
/// (cross-row constancy + leaf-row equality for both
/// claimed_leaf_value_bytes AND claimed_leaf_key_bytes), this binds
/// the storage gadget's full `(root, key, value)` triple to the MPT
/// proof's actual root + leaf row.
///
/// **For single-leaf tries** (the common test case): the leaf's
/// `key_path_bytes` IS the full storage trie key (= keccak256(slot_be)),
/// so the slot_trie_key binding is complete via this linkage.
///
/// **For multi-row tries**: `claimed_leaf_key_bytes` only carries the
/// leaf's REMAINING suffix after branch nibble consumption. The full
/// trie key would need a `claimed_consumed_nibbles` accumulator
/// column with a per-branch-row accumulation constraint to be fully
/// bound. Until that's added, this linkage works for SLOAD/SSTORE
/// proofs against single-leaf storage tries (the common case for
/// freshly-deployed contracts with one storage slot).
pub fn make_storage_full_to_mpt_root_linkage_descriptor(
    storage_layer_index: usize,
    mpt_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;
    let a_columns: Vec<usize> = (0..32)
        .map(|k| COL_STORAGE_ROOT_OFFSET + k)
        .chain((0..32).map(|k| COL_VALUE_BE_OFFSET + k))
        .chain((0..32).map(|k| COL_SLOT_TRIE_KEY_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..32)
        .map(|k| mpt_col::PARENT_HASH_OFFSET + k)
        .chain((0..32).map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k))
        .chain((0..32).map(|k| mpt_col::CLAIMED_LEAF_KEY_BYTE_OFFSET + k))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "storage_full_to_mpt_v1".into(),
        a_layer_index: storage_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_ROOT),
    }
}

/// **Phase A2 step 1d-multirow** — storage_access_air's full
/// `(storage_root, value_be, slot_trie_key)` ↔ mpt_air's
/// `(parent_hash, claimed_leaf_value_bytes, claimed_full_key_bytes)`
/// gated by `IS_ROOT`.
///
/// Same shape as `make_storage_full_to_mpt_root_linkage_descriptor`
/// (96 cols) but uses the new `claimed_full_key_bytes` column on the
/// MPT side instead of `claimed_leaf_key_bytes`. This is the
/// **multi-row-aware** version that binds the FULL trie key (not
/// just the leaf suffix).
///
/// **Soundness state**: cross-row constancy of
/// `claimed_full_key_bytes` is constrained (shifted body 7), so the
/// full key column is constant across the chain. The per-branch
/// nibble-equality binding (which would tie consumed `path_nibble`
/// values back to the corresponding nibble of `claimed_full_key`) is
/// the FINAL deferred piece. Until that lands, the multi-row variant
/// trusts the prover to populate `claimed_full_key_bytes` correctly
/// — but the LINKAGE pins the gadget's slot_trie_key against
/// whatever value the MPT side commits (i.e., the prover must
/// commit a SPECIFIC full key on row 0 of each chain, and that key
/// is what the linkage closure compares against).
///
/// The remaining trust gap is "is the committed full_key actually
/// the key the MPT proof walks?". For single-leaf tries the answer
/// is yes (claimed_full_key == claimed_leaf_key == leaf's key_path).
/// For multi-row tries the answer requires per-branch nibble
/// equality (the "nibble accumulator" final piece).
pub fn make_storage_multirow_to_mpt_root_linkage_descriptor(
    storage_layer_index: usize,
    mpt_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;
    let a_columns: Vec<usize> = (0..32)
        .map(|k| COL_STORAGE_ROOT_OFFSET + k)
        .chain((0..32).map(|k| COL_VALUE_BE_OFFSET + k))
        .chain((0..32).map(|k| COL_SLOT_TRIE_KEY_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..32)
        .map(|k| mpt_col::PARENT_HASH_OFFSET + k)
        .chain((0..32).map(|k| mpt_col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k))
        .chain((0..32).map(|k| mpt_col::CLAIMED_FULL_KEY_BYTE_OFFSET + k))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "storage_multirow_to_mpt_v1".into(),
        a_layer_index: storage_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: mpt_layer_index,
        b_columns,
        b_selector_column: Some(mpt_col::IS_ROOT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> StorageAccessRow {
        StorageAccessRow {
            address: [
                0xab, 0xcd, 0xef, 0x01, 0x02, 0x03, 0x04, 0x05,
                0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x10, 0x11,
            ],
            slot: [0x42, 0, 0, 0],
            value: [0x99, 0, 0, 0],
            storage_root: [0u8; 32], // placeholder; real one populated by trace builder
            is_write: false,
        }
    }

    #[test]
    fn address_to_limbs_known_value() {
        let addr = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
            0x10, 0x11, 0x12, 0x13,
        ];
        let limbs = address_to_limbs(addr);
        // Full BE = 0x...(12 zeros)...00010203040506070809...13
        // limb 0 (low 64 bits) = BE bytes 24..32 = 0x0c0d0e0f10111213
        assert_eq!(limbs[0], 0x0c0d_0e0f_1011_1213);
        // limb 1 = BE bytes 16..24 = 0x0405060708090a0b
        assert_eq!(limbs[1], 0x0405_0607_0809_0a0b);
        // limb 2 = BE bytes 8..16 = 0x00000000_00010203
        assert_eq!(limbs[2], 0x0000_0000_0001_0203);
        // limb 3 = BE bytes 0..8 = all zero (top 96 bits of 160-bit addr)
        assert_eq!(limbs[3], 0);
    }

    #[test]
    fn trace_builder_populates_columns_and_computes_slot_trie_key() {
        let w = StorageAccessWitness::from_rows(vec![StorageAccessRow {
            address: [0xab; 20],
            slot: [0, 0, 0, 0],   // slot 0
            value: [0x42, 0, 0, 0],
            storage_root: [0xcc; 32],
            is_write: false,
        }]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // slot 0 → keccak256(0x00 ×32) = 0x290decd9...e3e563
        let known_first_byte = 0x29u64;
        let known_last_byte = 0x63u64;
        assert_eq!(
            trace.columns[COL_SLOT_TRIE_KEY_OFFSET].evaluations[0].to_u64(),
            known_first_byte,
        );
        assert_eq!(
            trace.columns[COL_SLOT_TRIE_KEY_OFFSET + 31].evaluations[0].to_u64(),
            known_last_byte,
        );
        // storage_root[0] = 0xcc.
        assert_eq!(
            trace.columns[COL_STORAGE_ROOT_OFFSET].evaluations[0].to_u64(),
            0xcc,
        );
        // is_write = 0, is_real = 1.
        assert_eq!(trace.columns[COL_IS_WRITE].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = StorageAccessWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = StorageAccessConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
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
    fn slot_limb_decomp_fires_on_tampered_byte() {
        let w = StorageAccessWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper byte 31 of slot (LSB of slot_limb_0).
        cols[COL_SLOT_BE_OFFSET + 31][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = StorageAccessConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // slot_limb_0 decomp is at constraint index 2 + 0 = 2.
        let slot_limb_0_idx = 2;
        assert!(!results[slot_limb_0_idx][0].is_zero(),
                "slot_limb_0 decomp should fire on tampered byte");
    }

    #[test]
    fn value_limb_decomp_fires_on_tampered_byte() {
        let w = StorageAccessWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_VALUE_BE_OFFSET + 31][0] = Scalar::from_u64(0x55, CurveType::Bls48581);
        let cs = StorageAccessConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // value_limb_0 is at constraint index 2 + 4 + 0 = 6.
        let value_limb_0_idx = 6;
        assert!(!results[value_limb_0_idx][0].is_zero(),
                "value_limb_0 decomp should fire on tampered byte");
    }

    #[test]
    fn is_write_binary_fires_on_nonbinary() {
        let w = StorageAccessWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_WRITE][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = StorageAccessConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "is_write binary should fire");
    }

    #[test]
    fn storage_to_keccak_descriptor_well_formed() {
        let desc = make_storage_to_keccak_extract_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "storage_slot_keccak_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 64);
        assert_eq!(desc.b_columns.len(), 64);
        // First 32 cols = slot_be / INPUT_BYTE.
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], COL_SLOT_BE_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                crate::keccak_extract::COL_INPUT_BYTE_OFFSET + k,
            );
        }
        // Next 32 cols = slot_trie_key / OUTPUT_BYTE.
        for k in 0..32 {
            assert_eq!(desc.a_columns[32 + k], COL_SLOT_TRIE_KEY_OFFSET + k);
            assert_eq!(
                desc.b_columns[32 + k],
                crate::keccak_extract::COL_OUTPUT_BYTE_OFFSET + k,
            );
        }
    }

    #[test]
    fn storage_to_keccak_tuples_align_on_honest_witness() {
        // Build storage witness with a known slot, build matching
        // KeccakExtract witness with slot_be as input. Verify the
        // 64-col tuples (slot_be ++ trie_key) and (INPUT_BYTE[0..32] ++
        // OUTPUT_BYTE[0..32]) match byte-for-byte.
        use crate::keccak::keccak256;
        use crate::keccak_extract::KeccakExtractWitness;

        let slot_limbs = [0x42u64, 0, 0, 0];
        let row = StorageAccessRow {
            address: [0xab; 20],
            slot: slot_limbs,
            value: [0x99, 0, 0, 0],
            storage_root: [0xcc; 32],
            is_write: false,
        };
        let storage_w = StorageAccessWitness::from_rows(vec![row]);
        let storage_trace =
            build_trace_polynomials(&storage_w, CurveType::Bls48581);

        let slot_be = be_bytes_of_u256(slot_limbs);
        let keccak_w = KeccakExtractWitness::from_inputs(&[slot_be.to_vec()]).unwrap();
        let keccak_trace = crate::keccak_extract::build_trace_polynomials(
            &keccak_w,
            CurveType::Bls48581,
        );

        let expected_hash = keccak256(&slot_be);

        // A-side tuple bytes.
        for k in 0..32 {
            let storage_slot_be_k = storage_trace
                .columns[COL_SLOT_BE_OFFSET + k]
                .evaluations[0]
                .to_u64() as u8;
            let storage_trie_key_k = storage_trace
                .columns[COL_SLOT_TRIE_KEY_OFFSET + k]
                .evaluations[0]
                .to_u64() as u8;
            assert_eq!(storage_slot_be_k, slot_be[k]);
            assert_eq!(storage_trie_key_k, expected_hash[k]);
        }
        // B-side tuple bytes.
        for k in 0..32 {
            let ke_input_k = keccak_trace
                .columns[crate::keccak_extract::COL_INPUT_BYTE_OFFSET + k]
                .evaluations[0]
                .to_u64() as u8;
            let ke_output_k = keccak_trace
                .columns[crate::keccak_extract::COL_OUTPUT_BYTE_OFFSET + k]
                .evaluations[0]
                .to_u64() as u8;
            assert_eq!(ke_input_k, slot_be[k],
                       "INPUT_BYTE[{}] mismatch", k);
            assert_eq!(ke_output_k, expected_hash[k],
                       "OUTPUT_BYTE[{}] mismatch", k);
        }
    }

    #[test]
    fn storage_root_to_mpt_root_linkage_descriptor_well_formed() {
        let desc = make_storage_root_to_mpt_root_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "storage_root_to_mpt_root_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 32);
        assert_eq!(desc.b_columns.len(), 32);
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], COL_STORAGE_ROOT_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                crate::mpt_air::col::PARENT_HASH_OFFSET + k,
            );
        }
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::mpt_air::col::IS_ROOT),
        );
    }

    #[test]
    fn storage_root_value_to_mpt_descriptor_well_formed() {
        let desc = make_storage_root_value_to_mpt_root_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "storage_root_value_to_mpt_v1");
        assert_eq!(desc.a_columns.len(), 64);
        assert_eq!(desc.b_columns.len(), 64);
        // First 32 = storage_root / parent_hash.
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], COL_STORAGE_ROOT_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                crate::mpt_air::col::PARENT_HASH_OFFSET + k,
            );
        }
        // Next 32 = value_be / claimed_leaf_value_bytes.
        for k in 0..32 {
            assert_eq!(desc.a_columns[32 + k], COL_VALUE_BE_OFFSET + k);
            assert_eq!(
                desc.b_columns[32 + k],
                crate::mpt_air::col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k,
            );
        }
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(crate::mpt_air::col::IS_ROOT));
    }

    #[test]
    fn storage_full_to_mpt_descriptor_well_formed() {
        let desc = make_storage_full_to_mpt_root_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "storage_full_to_mpt_v1");
        // 32 storage_root + 32 value + 32 slot_trie_key = 96 cols.
        assert_eq!(desc.a_columns.len(), 96);
        assert_eq!(desc.b_columns.len(), 96);
        // First 32 = storage_root / parent_hash.
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], COL_STORAGE_ROOT_OFFSET + k);
            assert_eq!(desc.b_columns[k], crate::mpt_air::col::PARENT_HASH_OFFSET + k);
        }
        // Next 32 = value / claimed_leaf_value.
        for k in 0..32 {
            assert_eq!(desc.a_columns[32 + k], COL_VALUE_BE_OFFSET + k);
            assert_eq!(
                desc.b_columns[32 + k],
                crate::mpt_air::col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k,
            );
        }
        // Last 32 = slot_trie_key / claimed_leaf_key.
        for k in 0..32 {
            assert_eq!(desc.a_columns[64 + k], COL_SLOT_TRIE_KEY_OFFSET + k);
            assert_eq!(
                desc.b_columns[64 + k],
                crate::mpt_air::col::CLAIMED_LEAF_KEY_BYTE_OFFSET + k,
            );
        }
    }

    #[test]
    fn storage_multirow_to_mpt_descriptor_well_formed() {
        let desc = make_storage_multirow_to_mpt_root_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "storage_multirow_to_mpt_v1");
        assert_eq!(desc.a_columns.len(), 96);
        assert_eq!(desc.b_columns.len(), 96);
        // First 32 = storage_root.
        for k in 0..32 {
            assert_eq!(desc.a_columns[k], COL_STORAGE_ROOT_OFFSET + k);
        }
        // Last 32 = full_key (vs. leaf_key in the non-multirow variant).
        for k in 0..32 {
            assert_eq!(
                desc.b_columns[64 + k],
                crate::mpt_air::col::CLAIMED_FULL_KEY_BYTE_OFFSET + k,
            );
        }
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let w = StorageAccessWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = StorageAccessConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone storage_access_air proof must verify",
        );
    }
}
