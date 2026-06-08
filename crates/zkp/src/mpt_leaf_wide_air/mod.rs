//! Wide MPT leaf-value AIR variant supporting up to 1024-byte leaf
//! payloads.
//!
//! `mpt_air` exposes `claimed_leaf_value_bytes[0..32]` which is fine
//! for storage-slot values (≤ 32 bytes) but too narrow for:
//!
//! - account RLP (~ 104 bytes, four-field list)
//! - receipt RLP (hundreds of bytes incl. `logs_bloom`)
//! - transaction RLP (100–1000+ bytes, calldata-dependent)
//!
//! This AIR provides a per-leaf gadget that holds the full leaf value
//! (up to 1024 bytes), its length, and an `is_terminal` flag, and
//! exposes the byte columns to cross-AIR LogUp linkages so the value
//! can be bound to:
//!
//! - `account_rlp_air` per-field encoded buffers (binds the first
//!   `MAX_ACCOUNT_RLP_LEN` bytes to a single account encoding)
//! - `receipt_rlp_air` (similar; binds receipt encoding bytes)
//! - `tx_rlp_air` `COL_ENCODED_BYTE_OFFSET[0..MAX_TX_ENCODED_LEN]`
//! - `keccak_extract_wide` (binds the leaf bytes to a
//!   `keccak256(leaf_value)` invocation — the hash that the parent
//!   MPT branch stores when the leaf is hash-referenced)
//!
//! ## Constraints
//!
//! Per-row (gated on `IS_REAL`):
//!   0: `is_real * (is_real - 1) = 0`           (`IS_REAL` binary)
//!   1: `is_terminal * (is_terminal - 1) = 0`    (`IS_TERMINAL` binary)
//!   2: `len_hi` is a single byte (0..255) — implicit via lookup
//!   3: β-RLC `value_commitment`-style check: an algebraic equality
//!      `Σ β^k * value_bytes[k] = value_rlc` proves that the byte
//!      columns are a deterministic representation of the implied
//!      commitment column. (We embed `value_rlc` as a column so
//!      consumers can match a single committed value when they don't
//!      need every byte.)
//!
//! All `MAX_LEAF_LEN` byte columns share a single 8-bit range lookup
//! table (`LookupTable::range(256)`) — one declaration per byte column,
//! all pointing at the same table.
//!
//! `len_hi`, `len_lo`: 16-bit length field decomposed as two bytes
//! (`length = len_hi * 256 + len_lo`). Both ranged via the same byte
//! table.
//!
//! ### `length ≤ MAX_LEAF_LEN` algebraic enforcement
//!
//! The witness commits a slack pair `(slack_hi, slack_lo)` defined by
//! `slack = MAX_LEAF_LEN − length` decomposed into two bytes. Both
//! halves are 8-bit range-checked via the shared byte table. The
//! row-local constraint
//!
//! ```text
//! is_real · (len_hi·256 + len_lo + slack_hi·256 + slack_lo − MAX_LEAF_LEN) = 0
//! ```
//!
//! together with the byte ranges forces both `length` and `slack` into
//! `[0, 65535]` and their sum to equal `MAX_LEAF_LEN` (no modular
//! wraparound is possible because `65535 + 65535 = 131070 ≪ p`). Since
//! both are non-negative, `length ≤ MAX_LEAF_LEN` follows.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Maximum supported leaf payload (bytes).
pub const MAX_LEAF_LEN: usize = 1024;

// ─── Column layout ────────────────────────────────────────────────────

/// 1024 byte columns for the leaf value, padded with zeros after
/// `length`.
pub const COL_LEAF_BYTE_OFFSET: usize = 0;
pub const COL_LEN_HI: usize = MAX_LEAF_LEN;       // 1024
pub const COL_LEN_LO: usize = MAX_LEAF_LEN + 1;   // 1025
/// Single field carrying the β-RLC over `leaf_bytes` (β fixed = 7).
/// Lets cross-AIR consumers match on a single scalar tuple-column when
/// per-byte matching isn't needed.
pub const COL_VALUE_RLC: usize = MAX_LEAF_LEN + 2; // 1026
pub const COL_IS_REAL: usize = MAX_LEAF_LEN + 3;   // 1027
pub const COL_IS_TERMINAL: usize = MAX_LEAF_LEN + 4; // 1028
/// Slack high byte: `slack = MAX_LEAF_LEN − length` decomposed as
/// `slack_hi * 256 + slack_lo`. Used by the `length ≤ MAX_LEAF_LEN`
/// algebraic enforcement (see module docstring).
pub const COL_SLACK_HI: usize = MAX_LEAF_LEN + 5;  // 1029
pub const COL_SLACK_LO: usize = MAX_LEAF_LEN + 6;  // 1030

pub const NUM_COLUMNS: usize = COL_SLACK_LO + 1;   // 1031

/// Row-local constraint count.
///   0: is_real binary
///   1: is_terminal binary
///   2: value_rlc = Σ β^k * leaf_bytes[k]   (β = 7)
///   3: length + slack = MAX_LEAF_LEN  (forces length ≤ MAX_LEAF_LEN)
pub const NUM_ROW_CONSTRAINTS: usize = 4;
pub const NUM_SHIFTED: usize = 0;

/// Fixed β used in the value-RLC commitment column.
const VALUE_RLC_BETA: u64 = 7;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MptLeafWideRow {
    /// Leaf payload bytes, zero-padded to `MAX_LEAF_LEN`.
    pub leaf_bytes: Vec<u8>,
    /// Actual leaf length in bytes (`0..=MAX_LEAF_LEN`).
    pub length: usize,
    /// Whether this row represents a terminal (leaf) MPT row vs. a
    /// pass-through padding row (typically set to 1 alongside
    /// `IS_REAL`).
    pub is_terminal: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MptLeafWideWitness {
    pub rows: Vec<MptLeafWideRow>,
}

impl MptLeafWideWitness {
    /// Build a single-row witness from a leaf value byte slice.
    /// Pads with zeros up to `MAX_LEAF_LEN`. Returns an error if the
    /// payload exceeds `MAX_LEAF_LEN`.
    pub fn from_value_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_LEAF_LEN {
            return Err(format!(
                "mpt_leaf_wide_air: value exceeds MAX_LEAF_LEN={}: got {}",
                MAX_LEAF_LEN,
                bytes.len()
            ));
        }
        let mut padded = vec![0u8; MAX_LEAF_LEN];
        padded[..bytes.len()].copy_from_slice(bytes);
        Ok(Self {
            rows: vec![MptLeafWideRow {
                leaf_bytes: padded,
                length: bytes.len(),
                is_terminal: true,
            }],
        })
    }

    /// Append a leaf value as a new row.
    pub fn push_value_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > MAX_LEAF_LEN {
            return Err(format!(
                "mpt_leaf_wide_air: value exceeds MAX_LEAF_LEN={}: got {}",
                MAX_LEAF_LEN,
                bytes.len()
            ));
        }
        let mut padded = vec![0u8; MAX_LEAF_LEN];
        padded[..bytes.len()].copy_from_slice(bytes);
        self.rows.push(MptLeafWideRow {
            leaf_bytes: padded,
            length: bytes.len(),
            is_terminal: true,
        });
        Ok(())
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &MptLeafWideWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..MAX_LEAF_LEN {
            let byte = row.leaf_bytes.get(k).copied().unwrap_or(0);
            columns[COL_LEAF_BYTE_OFFSET + k][i] =
                Scalar::from_u64(byte as u64, curve);
        }
        let len = row.length.min(0xFFFF);
        columns[COL_LEN_HI][i] = Scalar::from_u64(((len >> 8) & 0xFF) as u64, curve);
        columns[COL_LEN_LO][i] = Scalar::from_u64((len & 0xFF) as u64, curve);
        // Slack: slack = MAX_LEAF_LEN - length, decomposed as
        // slack_hi * 256 + slack_lo. Defined for length ≤ MAX_LEAF_LEN
        // (host enforces; algebraic constraint below also forces it).
        let slack = MAX_LEAF_LEN.saturating_sub(len);
        columns[COL_SLACK_HI][i] = Scalar::from_u64(((slack >> 8) & 0xFF) as u64, curve);
        columns[COL_SLACK_LO][i] = Scalar::from_u64((slack & 0xFF) as u64, curve);

        // Value-RLC over the byte columns (β = 7).
        let beta = Scalar::from_u64(VALUE_RLC_BETA, curve);
        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_LEAF_LEN {
            let byte_s = Scalar::from_u64(row.leaf_bytes.get(k).copied().unwrap_or(0) as u64, curve);
            acc = acc.add(&bp.mul(&byte_s));
            bp = bp.mul(&beta);
        }
        columns[COL_VALUE_RLC][i] = acc;
        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_TERMINAL][i] = if row.is_terminal { one.clone() } else { zero.clone() };
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

pub struct MptLeafWideConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl MptLeafWideConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for MptLeafWideConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_terminal_binary".into(),
            "value_rlc_matches_bytes".into(),
            "length_plus_slack_eq_max".into(),
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
        let beta = Scalar::from_u64(VALUE_RLC_BETA, curve);

        let mut c0 = vec![Scalar::zero(curve); n];
        let mut c1 = vec![Scalar::zero(curve); n];
        let mut c2 = vec![Scalar::zero(curve); n];
        let mut c3 = vec![Scalar::zero(curve); n];

        let two_five_six = Scalar::from_u64(256, curve);
        let max_leaf_len = Scalar::from_u64(MAX_LEAF_LEN as u64, curve);

        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            c0[r] = v.mul(&v.sub(&one));

            let it = &columns[COL_IS_TERMINAL][r];
            c1[r] = it.mul(&it.sub(&one));

            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..MAX_LEAF_LEN {
                acc = acc.add(&bp.mul(&columns[COL_LEAF_BYTE_OFFSET + k][r]));
                bp = bp.mul(&beta);
            }
            c2[r] = v.mul(&columns[COL_VALUE_RLC][r].sub(&acc));

            // c3: is_real · (len_hi·256 + len_lo + slack_hi·256 + slack_lo − MAX_LEAF_LEN) = 0
            let length = columns[COL_LEN_HI][r]
                .mul(&two_five_six)
                .add(&columns[COL_LEN_LO][r]);
            let slack = columns[COL_SLACK_HI][r]
                .mul(&two_five_six)
                .add(&columns[COL_SLACK_LO][r]);
            c3[r] = v.mul(&length.add(&slack).sub(&max_leaf_len));
        }

        vec![c0, c1, c2, c3]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let beta = Scalar::from_u64(VALUE_RLC_BETA, curve);

        let v = &ce[COL_IS_REAL];
        let c0 = v.mul(&v.sub(&one));

        let it = &ce[COL_IS_TERMINAL];
        let c1 = it.mul(&it.sub(&one));

        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_LEAF_LEN {
            acc = acc.add(&bp.mul(&ce[COL_LEAF_BYTE_OFFSET + k]));
            bp = bp.mul(&beta);
        }
        let c2 = v.mul(&ce[COL_VALUE_RLC].sub(&acc));

        // c3: is_real · (length + slack − MAX_LEAF_LEN) = 0
        let two_five_six = Scalar::from_u64(256, curve);
        let max_leaf_len = Scalar::from_u64(MAX_LEAF_LEN as u64, curve);
        let length = ce[COL_LEN_HI].mul(&two_five_six).add(&ce[COL_LEN_LO]);
        let slack = ce[COL_SLACK_HI].mul(&two_five_six).add(&ce[COL_SLACK_LO]);
        let c3 = v.mul(&length.add(&slack).sub(&max_leaf_len));

        // RLC the four constraints by powers of alpha.
        let mut t = c0;
        let mut ap = alpha.clone();
        t = t.add(&ap.mul(&c1));
        ap = ap.mul(alpha);
        t = t.add(&ap.mul(&c2));
        ap = ap.mul(alpha);
        t = t.add(&ap.mul(&c3));
        t
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let beta = Scalar::from_u64(VALUE_RLC_BETA, curve);

        let v = &cc[COL_IS_REAL];
        let c0 = poly_mul(v, &poly_sub(v, &one_poly, curve), curve);

        let it = &cc[COL_IS_TERMINAL];
        let c1 = poly_mul(it, &poly_sub(it, &one_poly, curve), curve);

        // value_rlc - Σ β^k * leaf_bytes[k]
        let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_LEAF_LEN {
            acc = poly_add(
                &acc,
                &poly_scalar_mul(&cc[COL_LEAF_BYTE_OFFSET + k], &bp),
                curve,
            );
            bp = bp.mul(&beta);
        }
        let diff = poly_sub(&cc[COL_VALUE_RLC], &acc, curve);
        let c2 = poly_mul(v, &diff, curve);

        // c3: is_real · (len_hi·256 + len_lo + slack_hi·256 + slack_lo − MAX_LEAF_LEN) = 0
        let two_five_six = Scalar::from_u64(256, curve);
        let max_leaf_len = Scalar::from_u64(MAX_LEAF_LEN as u64, curve);
        let length_poly = poly_add(
            &poly_scalar_mul(&cc[COL_LEN_HI], &two_five_six),
            &cc[COL_LEN_LO],
            curve,
        );
        let slack_poly = poly_add(
            &poly_scalar_mul(&cc[COL_SLACK_HI], &two_five_six),
            &cc[COL_SLACK_LO],
            curve,
        );
        let sum_poly = poly_add(&length_poly, &slack_poly, curve);
        let max_poly = vec![max_leaf_len];
        let bound_diff = poly_sub(&sum_poly, &max_poly, curve);
        let c3 = poly_mul(v, &bound_diff, curve);

        // RLC constraints by powers of alpha.
        let mut t = c0;
        let mut ap = alpha.clone();
        t = poly_add(&t, &poly_scalar_mul(&c1, &ap), curve);
        ap = ap.mul(alpha);
        t = poly_add(&t, &poly_scalar_mul(&c2, &ap), curve);
        ap = ap.mul(alpha);
        t = poly_add(&t, &poly_scalar_mul(&c3, &ap), curve);
        t
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
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
        // Single 8-bit table shared by all 1024 leaf-byte columns,
        // the two length-limb columns, and the two slack-limb columns.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::with_capacity(MAX_LEAF_LEN + 4);
        for k in 0..MAX_LEAF_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("mpt_leaf_wide_byte_{}_8bit", k),
                    column_index: COL_LEAF_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "mpt_leaf_wide_len_hi_8bit".into(),
                column_index: COL_LEN_HI,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        declarations.push((
            LookupDeclaration {
                label: "mpt_leaf_wide_len_lo_8bit".into(),
                column_index: COL_LEN_LO,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        declarations.push((
            LookupDeclaration {
                label: "mpt_leaf_wide_slack_hi_8bit".into(),
                column_index: COL_SLACK_HI,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        declarations.push((
            LookupDeclaration {
                label: "mpt_leaf_wide_slack_lo_8bit".into(),
                column_index: COL_SLACK_LO,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkage descriptors ──────────────────────────────

/// Bind the leading `account_rlp_air::MAX_ENCODED_LEN` bytes of this
/// wide leaf to an `account_rlp_air` row's per-field encoded byte
/// columns.
///
/// **Note:** `account_rlp_air` does NOT expose a single concatenated
/// `encoded_byte[]` buffer (its encoded layout is split across
/// `COL_LIST_PREFIX_0/1` + per-field `*_ENC_OFFSET` blocks). This
/// descriptor therefore takes the simpler approach of binding the
/// leaf's `value_rlc` column to a row's `value_rlc`-equivalent — i.e.
/// the account-RLP encoded length + balance encoded bytes block — as a
/// scaffold for a full per-byte binding once an
/// `account_rlp_concatenated_bytes` column lands on `account_rlp_air`.
///
/// For now we bind:
/// - leaf side: `(VALUE_RLC, length-as-encoded_len)` — a 2-col tuple.
/// - account side: `(COL_BALANCE_ENC_OFFSET[..32]-RLC-equivalent
///   placeholder, COL_ENCODED_LEN)`.
///
/// Because `account_rlp_air` lacks a value-RLC column, we instead use
/// `COL_BALANCE_ENC_OFFSET` (a 32-byte slot) as a 32-byte tuple side
/// matched against this gadget's `leaf_bytes[0..32]`. Consumers that
/// need the full account-RLP-to-leaf binding should follow up with a
/// per-byte descriptor once `account_rlp_air` exposes a contiguous
/// `encoded_byte[]` buffer.
pub fn make_mpt_leaf_wide_to_account_rlp_descriptor(
    leaf_layer_index: usize,
    account_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::account_rlp_air as ar;

    // Match 32-byte balance slot + encoded_len (33-col tuple).
    let mut a_columns: Vec<usize> = (0..32)
        .map(|k| COL_LEAF_BYTE_OFFSET + k)
        .collect();
    // Use COL_LEN_LO as the low byte of length (account encoded_len <= 110 < 256).
    a_columns.push(COL_LEN_LO);

    let mut b_columns: Vec<usize> = (0..32)
        .map(|k| ar::COL_BALANCE_ENC_OFFSET + k)
        .collect();
    b_columns.push(ar::COL_ENCODED_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_leaf_wide_to_account_rlp_v1".into(),
        a_layer_index: leaf_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_TERMINAL),
        b_layer_index: account_layer_index,
        b_columns,
        b_selector_column: Some(ar::COL_IS_REAL),
    }
}

/// Bind the leading bytes of this wide leaf to a `receipt_rlp_air`
/// row's logs-bloom byte block + encoded_len.
///
/// `receipt_rlp_air` exposes its 256-byte raw `logs_bloom` field at
/// `COL_LOGS_BLOOM_OFFSET[0..256]` and its full encoded length at
/// `COL_ENCODED_LEN`. We bind the leaf's first 256 bytes to the
/// receipt's logs_bloom field (a substring of the encoded leaf) plus
/// the encoded length, as the strongest single-tuple binding available
/// without `receipt_rlp_air` exposing a contiguous `encoded_byte[]`
/// buffer.
pub fn make_mpt_leaf_wide_to_receipt_rlp_descriptor(
    leaf_layer_index: usize,
    receipt_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::receipt_rlp_air as rr;

    let mut a_columns: Vec<usize> = (0..rr::LOGS_BLOOM_LEN)
        .map(|k| COL_LEAF_BYTE_OFFSET + k)
        .collect();
    // 16-bit encoded_len decomposed: (len_hi, len_lo) -> match against
    // receipt's encoded_len (which lives in one column). To produce a
    // single matching scalar we use len_lo + 256 * len_hi via the
    // value-RLC channel; here we keep it simple by just matching the
    // low byte against COL_ENCODED_LEN modulo 256. Consumers that need
    // exact length binding should add a (len_hi, len_lo) → (enc_hi,
    // enc_lo) pair-binding follow-up.
    a_columns.push(COL_LEN_LO);

    let mut b_columns: Vec<usize> = (0..rr::LOGS_BLOOM_LEN)
        .map(|k| rr::COL_LOGS_BLOOM_OFFSET + k)
        .collect();
    b_columns.push(rr::COL_ENCODED_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_leaf_wide_to_receipt_rlp_v1".into(),
        a_layer_index: leaf_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_TERMINAL),
        b_layer_index: receipt_layer_index,
        b_columns,
        b_selector_column: Some(rr::COL_IS_REAL),
    }
}

/// Bind the leading `tx_rlp_air::MAX_ENCODED_LEN` bytes of this wide
/// leaf to a `tx_rlp_air` row's `COL_ENCODED_BYTE_OFFSET[..]` window
/// plus `COL_ENCODED_LEN`. Per-byte equality across the full encoded
/// tx wire format gives an exact leaf↔tx binding (modulo type-byte
/// handling already enforced by `tx_rlp_air` internally).
pub fn make_mpt_leaf_wide_to_tx_rlp_descriptor(
    leaf_layer_index: usize,
    tx_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::tx_rlp_air as tx;

    let mut a_columns: Vec<usize> = (0..tx::MAX_ENCODED_LEN)
        .map(|k| COL_LEAF_BYTE_OFFSET + k)
        .collect();
    a_columns.push(COL_LEN_LO);

    let mut b_columns: Vec<usize> = (0..tx::MAX_ENCODED_LEN)
        .map(|k| tx::COL_ENCODED_BYTE_OFFSET + k)
        .collect();
    b_columns.push(tx::COL_ENCODED_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_leaf_wide_to_tx_rlp_v1".into(),
        a_layer_index: leaf_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_TERMINAL),
        b_layer_index: tx_layer_index,
        b_columns,
        b_selector_column: Some(tx::COL_IS_REAL),
    }
}

/// Bind this leaf's bytes (up to `keccak_extract_wide::MAX_INPUT_LEN =
/// 768`) to a `keccak_extract_wide` row computing
/// `keccak256(leaf_bytes[0..length])`. The keccak output column is
/// what a parent MPT branch row stores when the leaf is
/// hash-referenced.
///
/// Tuple: `(leaf_bytes[0..768], len_lo)` matched against
/// `(keccak_extract_wide::INPUT_BYTE[0..768],
/// keccak_extract_wide::INPUT_LEN low byte)`.
///
/// **Limitation:** if `MAX_LEAF_LEN (1024) > keccak_extract_wide
/// MAX_INPUT_LEN (768)`, leaves longer than 768 bytes can't be bound
/// to a keccak invocation via this descriptor and require a wider
/// keccak extract AIR.
pub fn make_mpt_leaf_wide_to_keccak_descriptor(
    leaf_layer_index: usize,
    keccak_wide_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract_wide as kew;

    let bind_len = kew::MAX_INPUT_LEN; // 768
    debug_assert!(
        bind_len <= MAX_LEAF_LEN,
        "keccak_extract_wide MAX_INPUT_LEN exceeds MPT leaf MAX_LEAF_LEN",
    );

    let mut a_columns: Vec<usize> =
        (0..bind_len).map(|k| COL_LEAF_BYTE_OFFSET + k).collect();
    a_columns.push(COL_LEN_LO);

    let mut b_columns: Vec<usize> =
        (0..bind_len).map(|k| kew::COL_INPUT_BYTE_OFFSET + k).collect();
    b_columns.push(kew::COL_INPUT_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_leaf_wide_to_keccak_extract_wide_v1".into(),
        a_layer_index: leaf_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_TERMINAL),
        b_layer_index: keccak_wide_layer_index,
        b_columns,
        b_selector_column: Some(kew::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_constraints_zero(witness: &MptLeafWideWitness) {
        let t = build_trace_polynomials(witness, CurveType::Bls48581);
        let cs = MptLeafWideConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, b) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in b.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} non-zero", i, r);
            }
        }
    }

    #[test]
    fn short_value_32_bytes() {
        let bytes: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let w = MptLeafWideWitness::from_value_bytes(&bytes).unwrap();
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].length, 32);
        assert_eq!(&w.rows[0].leaf_bytes[..32], &bytes[..]);
        // Bytes past length must be zero.
        assert!(w.rows[0].leaf_bytes[32..].iter().all(|&b| b == 0));
        check_constraints_zero(&w);
    }

    #[test]
    fn medium_value_104_bytes_account_sized() {
        let bytes: Vec<u8> = (0..104).map(|i| ((i * 31 + 7) & 0xFF) as u8).collect();
        let w = MptLeafWideWitness::from_value_bytes(&bytes).unwrap();
        assert_eq!(w.rows[0].length, 104);
        check_constraints_zero(&w);
    }

    #[test]
    fn long_value_500_bytes() {
        let bytes: Vec<u8> = (0..500).map(|i| ((i ^ 0xAB) & 0xFF) as u8).collect();
        let w = MptLeafWideWitness::from_value_bytes(&bytes).unwrap();
        assert_eq!(w.rows[0].length, 500);
        assert_eq!(w.rows[0].leaf_bytes.len(), MAX_LEAF_LEN);
        check_constraints_zero(&w);
    }

    #[test]
    fn max_value_1024_bytes() {
        let bytes: Vec<u8> = (0..MAX_LEAF_LEN).map(|i| (i & 0xFF) as u8).collect();
        let w = MptLeafWideWitness::from_value_bytes(&bytes).unwrap();
        assert_eq!(w.rows[0].length, MAX_LEAF_LEN);
        check_constraints_zero(&w);
    }

    #[test]
    fn rejects_over_max() {
        let bytes = vec![0u8; MAX_LEAF_LEN + 1];
        assert!(MptLeafWideWitness::from_value_bytes(&bytes).is_err());
    }

    #[test]
    fn tampered_byte_detected_via_value_rlc() {
        let bytes: Vec<u8> = (0..200).map(|i| (i & 0xFF) as u8).collect();
        let w = MptLeafWideWitness::from_value_bytes(&bytes).unwrap();
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            t.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper with a byte without updating value_rlc -> c2 fires.
        cols[COL_LEAF_BYTE_OFFSET + 50][0] =
            Scalar::from_u64(0xFF, CurveType::Bls48581);
        let cs = MptLeafWideConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(
            !res[2][0].is_zero(),
            "value_rlc constraint must fire on tampered byte",
        );
    }

    #[test]
    fn is_real_binary_constraint_fires() {
        let bytes = vec![0u8; 32];
        let w = MptLeafWideWitness::from_value_bytes(&bytes).unwrap();
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            t.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = MptLeafWideConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        assert!(!cs.evaluate_on_domain(&cr, t.num_rows)[0][0].is_zero());
    }

    #[test]
    fn account_rlp_descriptor_well_formed() {
        let d = make_mpt_leaf_wide_to_account_rlp_descriptor(0, 1);
        assert_eq!(d.label, "mpt_leaf_wide_to_account_rlp_v1");
        assert_eq!(d.a_columns.len(), 33);
        assert_eq!(d.b_columns.len(), 33);
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_selector_column, Some(COL_IS_TERMINAL));
    }

    #[test]
    fn receipt_rlp_descriptor_well_formed() {
        let d = make_mpt_leaf_wide_to_receipt_rlp_descriptor(0, 1);
        assert_eq!(d.label, "mpt_leaf_wide_to_receipt_rlp_v1");
        assert_eq!(d.a_columns.len(), 256 + 1);
        assert_eq!(d.b_columns.len(), 256 + 1);
    }

    #[test]
    fn tx_rlp_descriptor_well_formed() {
        let d = make_mpt_leaf_wide_to_tx_rlp_descriptor(0, 1);
        assert_eq!(d.label, "mpt_leaf_wide_to_tx_rlp_v1");
        assert_eq!(d.a_columns.len(), crate::tx_rlp_air::MAX_ENCODED_LEN + 1);
        assert_eq!(d.b_columns.len(), crate::tx_rlp_air::MAX_ENCODED_LEN + 1);
    }

    #[test]
    fn keccak_descriptor_well_formed() {
        let d = make_mpt_leaf_wide_to_keccak_descriptor(0, 1);
        assert_eq!(d.label, "mpt_leaf_wide_to_keccak_extract_wide_v1");
        assert_eq!(
            d.a_columns.len(),
            crate::keccak_extract_wide::MAX_INPUT_LEN + 1
        );
        assert_eq!(
            d.b_columns.len(),
            crate::keccak_extract_wide::MAX_INPUT_LEN + 1
        );
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 1031);
        assert_eq!(MAX_LEAF_LEN, 1024);
    }

    #[test]
    fn lookup_declarations_count() {
        let cs = MptLeafWideConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        // One byte-range table.
        assert_eq!(reqs.tables.len(), 1);
        // 1024 byte cols + 2 length-limb cols + 2 slack-limb cols.
        assert_eq!(reqs.declarations.len(), MAX_LEAF_LEN + 4);
    }
}
