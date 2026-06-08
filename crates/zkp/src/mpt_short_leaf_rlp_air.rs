//! MPT short-leaf RLP-decoding consistency gadget (Phase 3 of #92).
//!
//! Companion to [`crate::mpt_rlp_air`] (the Phase 1 64-nibble-even-leaf
//! gadget). This module handles the **4-nibble even-length leaf with
//! 32-byte value** shape — a 38-byte short-form-list RLP encoding that
//! corresponds to mid-depth Ethereum trie leaves where 60 of 64 key
//! nibbles have been consumed by parent branches/extensions.
//!
//! # Phase 3 scope
//!
//! Single fixed RLP shape: leaf, **4-nibble even-length path** + 32-byte
//! value. Differences from Phase 1:
//!   - path is 2 packed bytes instead of 32 — KEY_PATH_BYTE bytes 2..32
//!     are zero-pinned by a new row-local body.
//!   - payload (37 bytes) ≤ 55 → **short-form list** header (no
//!     length-of-length byte), so the structural prefix shrinks from
//!     2 bytes (`0xf8 0x43`) to 1 byte (`0xe5`).
//!   - HP RLP header is `0x83` (vs `0xa1`) since HP_LEN drops from 33
//!     to 3 bytes.
//!
//! # Canonical 38-byte RLP layout
//!
//! ```text
//!   [0]      0xe5   short-list header (0xc0 + payload_len = 0xc0 + 37)
//!   [1]      0x83   HP-string header (0x80 + 3)
//!   [2]      0x20   HP prefix (leaf-flag + even-length, no leading nibble)
//!   [3..5]   ─      2 path bytes (each packs 2 path nibbles)
//!   [5]      0xa0   value-string header (0x80 + 32)
//!   [6..38]  ─      32 value bytes
//!   [38..256]       zero-pad to MAX_RLP_LEN (= mpt_air::MAX_RLP_LEN)
//! ```
//!
//! Total = 38 bytes, zero-padded to 256 to match
//! [`crate::mpt_air::MAX_RLP_LEN`] for the cross-AIR LogUp.
//!
//! # Shared MPT-side columns
//!
//! Phase 3 reuses the SAME MPT-side decoded columns introduced in
//! Phase 2: `KEY_PATH_BYTE[0..32]` and `VALUE_BYTE[0..32]`. The witness
//! populates `KEY_PATH_BYTE[0..2]` from the leaf's path bytes and zeros
//! `KEY_PATH_BYTE[2..32]`. The selector `col::IS_PHASE3_LEAF_SHAPE`
//! gates which MPT rows participate in the Phase 3 cross-AIR LogUp;
//! Phase 1 and Phase 3 selectors are mutually exclusive (a row matches
//! at most one shape), so each shape's multiset stays clean.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::mpt_rlp_gadget_helpers::{
    build_byte_pinned_poly, build_byte_slice_binding_poly,
    build_decoded_zero_tail_poly, build_zero_tail_poly, eval_byte_pinned,
    eval_byte_slice_binding, eval_decoded_zero_tail, eval_zero_tail,
};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of path nibbles in the Phase-3 leaf shape (even-length).
pub const NUM_PATH_NIBBLES: usize = 4;

/// Number of HP-packed path bytes (2 nibbles per byte).
pub const NUM_PATH_BYTES: usize = NUM_PATH_NIBBLES / 2; // 2

/// Number of value bytes (= 32, same as Phase 1 / shared MPT VALUE_BYTE).
pub const NUM_VALUE_BYTES: usize = 32;

/// Width of KEY_PATH_BYTE in the SHARED MPT-side schema (32 cols). The
/// gadget commits all 32 columns — only the first 2 carry path data,
/// the rest are pinned to zero.
pub const KEY_PATH_BYTE_WIDTH: usize = 32;

/// HP-encoded path length in bytes (1 prefix + 2 packed nibble bytes).
pub const HP_LEN: usize = 1 + NUM_PATH_BYTES; // 3

/// HP-string RLP header byte: 0x80 + HP_LEN = 0x83.
pub const HP_STRING_HEADER: u64 = 0x80 + HP_LEN as u64;

/// HP prefix byte for an even-length leaf (any length, any first nibble
/// = 0): 0x20 (= leaf-flag bit 5 + even-length bit 4 = 0).
pub const HP_PREFIX_BYTE: u64 = 0x20;

/// Value-string RLP header byte: 0x80 + 32 = 0xa0.
pub const VALUE_STRING_HEADER: u64 = 0x80 + NUM_VALUE_BYTES as u64;

/// RLP list payload length: HP-string body (1 + 3) + value-string body
/// (1 + 32) = 4 + 33 = 37. ≤ 55 → short-form list.
pub const RLP_PAYLOAD_LEN: u64 = (1 + HP_LEN + 1 + NUM_VALUE_BYTES) as u64; // 37

/// Short-list header byte: 0xc0 + payload_len = 0xc0 + 37 = 0xe5.
pub const RLP_LIST_HEADER: u64 = 0xc0 + RLP_PAYLOAD_LEN; // 0xe5

/// Total RLP length: 1 (list header) + 37 (payload) = 38.
pub const RLP_OUTPUT_LEN: usize = 1 + RLP_PAYLOAD_LEN as usize; // 38

/// Decoded `node_kind` for this shape (= 2 = leaf).
pub const NODE_KIND_LEAF: u64 = 2;

/// Width of the keccak-input-shaped RLP byte column. Must match
/// [`crate::mpt_air::MAX_RLP_LEN`] so the cross-AIR LogUp tuple has
/// equal widths on both sides.
pub const RLP_BYTE_WIDTH: usize = crate::mpt_air::MAX_RLP_LEN;

// Byte offsets of structural fields within the canonical RLP.
const RLP_OFFSET_HP_STR_HEADER: usize = 1;
const RLP_OFFSET_HP_PREFIX: usize = 2;
const RLP_OFFSET_PATH_BYTES: usize = 3;
const RLP_OFFSET_VALUE_HEADER: usize = RLP_OFFSET_PATH_BYTES + NUM_PATH_BYTES; // 5
const RLP_OFFSET_VALUE_BYTES: usize = RLP_OFFSET_VALUE_HEADER + 1; // 6

const _: () = assert!(RLP_OFFSET_VALUE_BYTES + NUM_VALUE_BYTES == RLP_OUTPUT_LEN);

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_RLP_BYTE_OFFSET: usize = 0;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_NODE_KIND: usize = COL_RLP_LEN + 1;
/// Mirrors the MPT-side `KEY_PATH_BYTE` column width (32 bytes). Only
/// indices 0..NUM_PATH_BYTES carry path data; the remaining
/// `NUM_PATH_BYTES..32` columns are pinned to zero by `key_path_zero_tail`.
pub const COL_KEY_PATH_BYTE_OFFSET: usize = COL_NODE_KIND + 1;
pub const COL_VALUE_BYTE_OFFSET: usize = COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH;
pub const COL_IS_REAL: usize = COL_VALUE_BYTE_OFFSET + NUM_VALUE_BYTES;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 12;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShortLeafRow {
    /// 2 path bytes — each packs 2 path nibbles (hi-nibble high).
    pub path_bytes: [u8; NUM_PATH_BYTES],
    /// 32-byte leaf value.
    pub value: [u8; NUM_VALUE_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct ShortLeafRlpWitness {
    pub leaves: Vec<ShortLeafRow>,
}

impl ShortLeafRlpWitness {
    pub fn from_leaves(leaves: &[ShortLeafRow]) -> Self {
        Self {
            leaves: leaves.to_vec(),
        }
    }
}

/// Construct the canonical 38-byte RLP encoding of the Phase-3 leaf
/// shape from `(path_bytes, value)`.
pub fn canonical_rlp(
    path_bytes: &[u8; NUM_PATH_BYTES],
    value: &[u8; NUM_VALUE_BYTES],
) -> [u8; RLP_OUTPUT_LEN] {
    let mut out = [0u8; RLP_OUTPUT_LEN];
    out[0] = RLP_LIST_HEADER as u8;
    out[RLP_OFFSET_HP_STR_HEADER] = HP_STRING_HEADER as u8;
    out[RLP_OFFSET_HP_PREFIX] = HP_PREFIX_BYTE as u8;
    out[RLP_OFFSET_PATH_BYTES..RLP_OFFSET_PATH_BYTES + NUM_PATH_BYTES]
        .copy_from_slice(path_bytes);
    out[RLP_OFFSET_VALUE_HEADER] = VALUE_STRING_HEADER as u8;
    out[RLP_OFFSET_VALUE_BYTES..RLP_OFFSET_VALUE_BYTES + NUM_VALUE_BYTES]
        .copy_from_slice(value);
    out
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ShortLeafRlpWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.leaves.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (i, leaf) in witness.leaves.iter().enumerate() {
        let rlp = canonical_rlp(&leaf.path_bytes, &leaf.value);
        for (b, &v) in rlp.iter().enumerate() {
            columns[COL_RLP_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_RLP_LEN][i] = Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve);
        columns[COL_NODE_KIND][i] = Scalar::from_u64(NODE_KIND_LEAF, curve);
        for (k, &b) in leaf.path_bytes.iter().enumerate() {
            columns[COL_KEY_PATH_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
        // KEY_PATH_BYTE[NUM_PATH_BYTES..32] pinned to zero — already
        // zero by initial allocation; zero-tail constraint enforces.
        for (k, &b) in leaf.value.iter().enumerate() {
            columns[COL_VALUE_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial {
            evaluations: evals,
            degree: num_rows,
        })
        .collect();

    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct ShortLeafRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ShortLeafRlpConstraintSystem {
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

// (Helpers extracted to crate::mpt_rlp_gadget_helpers — see imports.)

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for ShortLeafRlpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "node_kind_pinned".into(),
            "rlp_len_pinned".into(),
            "rlp_list_header_pinned".into(),
            "rlp_hp_string_header_pinned".into(),
            "rlp_hp_prefix_pinned".into(),
            "rlp_value_string_header_pinned".into(),
            "path_byte_binding".into(),
            "key_path_byte_zero_tail".into(),
            "value_byte_binding".into(),
            "rlp_zero_tail".into(),
            "value_byte_count_padding".into(),
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
        let alpha_for_rlc = Scalar::from_u64(7, curve);

        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let v = &row_evals[COL_IS_REAL];
            bodies[0][row] = v.mul(&v.sub(&one));
            bodies[1][row] = eval_byte_pinned(&row_evals, COL_NODE_KIND, NODE_KIND_LEAF, COL_IS_REAL);
            bodies[2][row] = eval_byte_pinned(&row_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL);
            bodies[3][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL);
            bodies[4][row] = eval_byte_pinned(
                &row_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_STR_HEADER,
                HP_STRING_HEADER,
                COL_IS_REAL,
            );
            bodies[5][row] = eval_byte_pinned(
                &row_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX,
                HP_PREFIX_BYTE,
                COL_IS_REAL,
            );
            bodies[6][row] = eval_byte_pinned(
                &row_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER,
                VALUE_STRING_HEADER,
                COL_IS_REAL,
            );
            bodies[7][row] = eval_byte_slice_binding(
                &row_evals,
                &alpha_for_rlc,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_PATH_BYTES,
                COL_KEY_PATH_BYTE_OFFSET,
                NUM_PATH_BYTES,
                COL_IS_REAL,
            );
            // KEY_PATH_BYTE[NUM_PATH_BYTES..32] pinned to zero so the
            // shared cross-AIR LogUp tuple agrees with the MPT side
            // (which also zeros these positions on Phase-3 rows).
            bodies[8][row] = eval_decoded_zero_tail(
                &row_evals,
                &alpha_for_rlc,
                COL_KEY_PATH_BYTE_OFFSET + NUM_PATH_BYTES,
                COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH,
                COL_IS_REAL,
            );
            bodies[9][row] = eval_byte_slice_binding(
                &row_evals,
                &alpha_for_rlc,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_VALUE_BYTES,
                COL_VALUE_BYTE_OFFSET,
                NUM_VALUE_BYTES,
                COL_IS_REAL,
            );
            bodies[10][row] = eval_zero_tail(
                &row_evals,
                &alpha_for_rlc,
                COL_RLP_BYTE_OFFSET,
                RLP_OUTPUT_LEN,
                RLP_BYTE_WIDTH,
                COL_IS_REAL,
            );
            // Defensive: the gadget commits VALUE_BYTE for all 32
            // positions (the leaf value) — there's no zero-tail on
            // VALUE_BYTE (Phase 3 still uses 32 value bytes). This
            // body is reserved for future shape variants and currently
            // vanishes (= 0).
            bodies[11][row] = Scalar::zero(curve);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            v.mul(&v.sub(&one)),
            eval_byte_pinned(col_evals, COL_NODE_KIND, NODE_KIND_LEAF, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL),
            eval_byte_pinned(
                col_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_STR_HEADER,
                HP_STRING_HEADER,
                COL_IS_REAL,
            ),
            eval_byte_pinned(
                col_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX,
                HP_PREFIX_BYTE,
                COL_IS_REAL,
            ),
            eval_byte_pinned(
                col_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER,
                VALUE_STRING_HEADER,
                COL_IS_REAL,
            ),
            eval_byte_slice_binding(
                col_evals,
                alpha,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_PATH_BYTES,
                COL_KEY_PATH_BYTE_OFFSET,
                NUM_PATH_BYTES,
                COL_IS_REAL,
            ),
            eval_decoded_zero_tail(
                col_evals,
                alpha,
                COL_KEY_PATH_BYTE_OFFSET + NUM_PATH_BYTES,
                COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH,
                COL_IS_REAL,
            ),
            eval_byte_slice_binding(
                col_evals,
                alpha,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_VALUE_BYTES,
                COL_VALUE_BYTE_OFFSET,
                NUM_VALUE_BYTES,
                COL_IS_REAL,
            ),
            eval_zero_tail(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OUTPUT_LEN, RLP_BYTE_WIDTH, COL_IS_REAL),
            Scalar::zero(curve),
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
        let v = &col_coeffs[COL_IS_REAL];
        let v_minus_1 = poly_sub(v, &one_poly, curve);
        let zero_body = vec![Scalar::zero(curve)];
        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(v, &v_minus_1, curve),
            build_byte_pinned_poly(col_coeffs, COL_NODE_KIND, NODE_KIND_LEAF, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL, curve),
            build_byte_pinned_poly(
                col_coeffs,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_STR_HEADER,
                HP_STRING_HEADER,
                COL_IS_REAL,
                curve,
            ),
            build_byte_pinned_poly(
                col_coeffs,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX,
                HP_PREFIX_BYTE,
                COL_IS_REAL,
                curve,
            ),
            build_byte_pinned_poly(
                col_coeffs,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER,
                VALUE_STRING_HEADER,
                COL_IS_REAL,
                curve,
            ),
            build_byte_slice_binding_poly(
                col_coeffs,
                alpha,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_PATH_BYTES,
                COL_KEY_PATH_BYTE_OFFSET,
                NUM_PATH_BYTES,
                COL_IS_REAL,
                curve,
            ),
            build_decoded_zero_tail_poly(
                col_coeffs,
                alpha,
                COL_KEY_PATH_BYTE_OFFSET + NUM_PATH_BYTES,
                COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH,
                COL_IS_REAL,
                curve,
            ),
            build_byte_slice_binding_poly(
                col_coeffs,
                alpha,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_VALUE_BYTES,
                COL_VALUE_BYTE_OFFSET,
                NUM_VALUE_BYTES,
                COL_IS_REAL,
                curve,
            ),
            build_zero_tail_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OUTPUT_LEN, RLP_BYTE_WIDTH, COL_IS_REAL, curve),
            zero_body,
        ];
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
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp linkage descriptor ───────────────────────────────

/// Cross-AIR LogUp descriptor binding the **MPT inclusion AIR's
/// Phase-3-shape leaf rows** to **this gadget's algebraically decoded
/// rows**: matches the same 322-element tuple as the Phase 1 linkage,
/// but gated by [`crate::mpt_air::col::IS_PHASE3_LEAF_SHAPE`] on the
/// MPT side.
///
/// The shared MPT-side `KEY_PATH_BYTE[0..32]` and `VALUE_BYTE[0..32]`
/// columns participate in BOTH Phase 1 and Phase 3 linkages: each
/// linkage's selector keeps the multisets disjoint by selector, and
/// the MPT witness builder ensures `IS_PHASE1_LEAF_SHAPE` and
/// `IS_PHASE3_LEAF_SHAPE` are mutually exclusive (a row matches at
/// most one Phase shape).
pub fn make_mpt_short_leaf_rlp_linkage_descriptor(
    mpt_layer_index: usize,
    rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;

    let mut a_columns: Vec<usize> = (0..RLP_BYTE_WIDTH)
        .map(|b| mpt_col::NODE_RLP_OFFSET + b)
        .collect();
    a_columns.push(mpt_col::NODE_RLP_LEN);
    a_columns.push(mpt_col::NODE_KIND);
    for b in 0..KEY_PATH_BYTE_WIDTH {
        a_columns.push(mpt_col::KEY_PATH_BYTE_OFFSET + b);
    }
    for b in 0..NUM_VALUE_BYTES {
        a_columns.push(mpt_col::VALUE_BYTE_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = (0..RLP_BYTE_WIDTH)
        .map(|b| COL_RLP_BYTE_OFFSET + b)
        .collect();
    b_columns.push(COL_RLP_LEN);
    b_columns.push(COL_NODE_KIND);
    for b in 0..KEY_PATH_BYTE_WIDTH {
        b_columns.push(COL_KEY_PATH_BYTE_OFFSET + b);
    }
    for b in 0..NUM_VALUE_BYTES {
        b_columns.push(COL_VALUE_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_short_leaf_rlp_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_PHASE3_LEAF_SHAPE),
        b_layer_index: rlp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::{mpt_node_rlp, MptNode, Nibbles};

    fn sample_path_bytes() -> [u8; NUM_PATH_BYTES] {
        [0x12, 0xab]
    }

    fn sample_value() -> [u8; NUM_VALUE_BYTES] {
        let mut v = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            v[i] = (i as u8).wrapping_mul(11).wrapping_add(17);
        }
        v
    }

    fn sample_witness() -> ShortLeafRlpWitness {
        ShortLeafRlpWitness::from_leaves(&[
            ShortLeafRow {
                path_bytes: [0u8; NUM_PATH_BYTES],
                value: [0u8; NUM_VALUE_BYTES],
            },
            ShortLeafRow {
                path_bytes: sample_path_bytes(),
                value: sample_value(),
            },
            ShortLeafRow {
                path_bytes: [0xffu8; NUM_PATH_BYTES],
                value: [0xffu8; NUM_VALUE_BYTES],
            },
        ])
    }

    /// Cross-check our hand-rolled `canonical_rlp` against the reference
    /// `mpt_node_rlp` encoder on a 4-nibble even leaf with 32-byte value.
    #[test]
    fn canonical_rlp_matches_reference_encoder() {
        let path_bytes = sample_path_bytes();
        let value = sample_value();
        let ours = canonical_rlp(&path_bytes, &value);

        // 4 nibbles from 2 path bytes (hi, lo).
        let mut nibbles = Vec::with_capacity(NUM_PATH_NIBBLES);
        for &b in &path_bytes {
            nibbles.push(b >> 4);
            nibbles.push(b & 0x0f);
        }
        let node = MptNode::Leaf {
            path: Nibbles::from_nibbles(nibbles),
            value: value.to_vec(),
        };
        let reference = mpt_node_rlp(&node);

        assert_eq!(reference.len(), RLP_OUTPUT_LEN);
        assert_eq!(&ours[..], &reference[..]);
    }

    #[test]
    fn canonical_rlp_structural_bytes_match_constants() {
        let path_bytes = sample_path_bytes();
        let value = sample_value();
        let rlp = canonical_rlp(&path_bytes, &value);
        assert_eq!(rlp[0] as u64, RLP_LIST_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_STR_HEADER] as u64, HP_STRING_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_PREFIX] as u64, HP_PREFIX_BYTE);
        assert_eq!(rlp[RLP_OFFSET_VALUE_HEADER] as u64, VALUE_STRING_HEADER);
        for k in 0..NUM_PATH_BYTES {
            assert_eq!(rlp[RLP_OFFSET_PATH_BYTES + k], path_bytes[k]);
        }
        for k in 0..NUM_VALUE_BYTES {
            assert_eq!(rlp[RLP_OFFSET_VALUE_BYTES + k], value[k]);
        }
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 3);
        // KEY_PATH_BYTE[0..2] = path bytes; KEY_PATH_BYTE[2..32] = 0.
        let path = sample_path_bytes();
        for k in 0..NUM_PATH_BYTES {
            assert_eq!(
                trace.columns[COL_KEY_PATH_BYTE_OFFSET + k].evaluations[1].to_u64(),
                path[k] as u64
            );
        }
        for k in NUM_PATH_BYTES..KEY_PATH_BYTE_WIDTH {
            assert!(
                trace.columns[COL_KEY_PATH_BYTE_OFFSET + k].evaluations[1].is_zero(),
                "KEY_PATH_BYTE[{}] must be zero on Phase 3 rows", k
            );
        }
        // VALUE_BYTE row 1 matches sample_value.
        let value = sample_value();
        for k in 0..NUM_VALUE_BYTES {
            assert_eq!(
                trace.columns[COL_VALUE_BYTE_OFFSET + k].evaluations[1].to_u64(),
                value[k] as u64
            );
        }
        // Padding rows: IS_REAL = 0.
        for row in 3..trace.padded_size as usize {
            assert!(trace.columns[COL_IS_REAL].evaluations[row].is_zero());
        }
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = ShortLeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(13, curve);
        for row in 0..trace.padded_size as usize {
            let col_vals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            let body = cs.evaluate_at_point(&col_vals, &alpha);
            assert!(body.is_zero(), "row {} body must vanish", row);
        }
    }

    /// Tampering: a non-zero KEY_PATH_BYTE in the zero-tail region must
    /// fire the `key_path_byte_zero_tail` body.
    #[test]
    fn tampered_key_path_zero_tail_fires() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = ShortLeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        // Set KEY_PATH_BYTE[5] to 1 (should be 0 — outside path region).
        trace.columns[COL_KEY_PATH_BYTE_OFFSET + 5].evaluations[1] =
            Scalar::from_u64(1, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "key_path_byte_zero_tail must fire on non-zero pad byte"
        );
    }

    /// Tampering: flipping the short-list header byte must fire body 3.
    #[test]
    fn tampered_list_header_byte_fires() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = ShortLeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(19, curve);
        let one = Scalar::one(curve);
        trace.columns[COL_RLP_BYTE_OFFSET].evaluations[0] =
            trace.columns[COL_RLP_BYTE_OFFSET].evaluations[0].add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "rlp_list_header_pinned must fire on tampered RLP[0]"
        );
    }

    /// Tampering: a path byte mismatch between RLP region and decoded
    /// KEY_PATH_BYTE must fire body 7.
    #[test]
    fn tampered_path_byte_breaks_binding() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = ShortLeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(23, curve);
        let one = Scalar::one(curve);
        // Tamper RLP[3] (path byte 0); leave KEY_PATH_BYTE[0] honest.
        trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_PATH_BYTES].evaluations[1] =
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_PATH_BYTES].evaluations[1]
                .add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "path_byte_binding must fire on RLP/decoded mismatch"
        );
    }

    // ── Phase 3 cross-AIR LogUp linkage tests ──────────────────────

    #[test]
    fn mpt_short_leaf_rlp_descriptor_well_formed() {
        use crate::mpt_air::col as mpt_col;
        let desc = make_mpt_short_leaf_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "mpt_short_leaf_rlp_v1");
        let expected_width =
            RLP_BYTE_WIDTH + 1 + 1 + KEY_PATH_BYTE_WIDTH + NUM_VALUE_BYTES;
        assert_eq!(desc.a_columns.len(), expected_width);
        assert_eq!(desc.b_columns.len(), expected_width);
        assert_eq!(desc.a_columns[0], mpt_col::NODE_RLP_OFFSET);
        assert_eq!(desc.a_columns[RLP_BYTE_WIDTH + 1], mpt_col::NODE_KIND);
        assert_eq!(
            desc.a_columns[RLP_BYTE_WIDTH + 2],
            mpt_col::KEY_PATH_BYTE_OFFSET
        );
        assert_eq!(desc.b_columns[0], COL_RLP_BYTE_OFFSET);
        assert_eq!(desc.b_columns[RLP_BYTE_WIDTH], COL_RLP_LEN);
        assert_eq!(desc.a_selector_column, Some(mpt_col::IS_PHASE3_LEAF_SHAPE));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    /// Build a real single-leaf MPT inclusion witness with a 2-byte
    /// (= 4-nibble) key + 32-byte value, then build the matching
    /// gadget invocation. Assert the 322-element cross-AIR tuples
    /// agree element-by-element.
    #[test]
    fn mpt_short_leaf_gadget_byte_tuples_match_real_mpt_witness() {
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::build_trace_polynomials_from_rows;

        let curve = CurveType::Bls48581;
        let key = vec![0x12u8, 0xabu8]; // 4 nibbles: 1, 2, a, b
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(11).wrapping_add(17);
        }

        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.is_phase3_leaf_shape, 1);
        assert_eq!(row.is_phase1_leaf_shape, 0);
        assert_eq!(row.node_rlp_len, RLP_OUTPUT_LEN);

        // KEY_PATH_BYTE[0..2] should mirror the 2-byte key bytes
        // (which is exactly what the HP encoding packs).
        assert_eq!(row.key_path_bytes[0], key[0]);
        assert_eq!(row.key_path_bytes[1], key[1]);
        for k in 2..32 {
            assert_eq!(row.key_path_bytes[k], 0);
        }
        assert_eq!(row.value, value);

        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let gadget_w = ShortLeafRlpWitness::from_leaves(&[ShortLeafRow {
            path_bytes: [key[0], key[1]],
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        let descriptor = make_mpt_short_leaf_rlp_linkage_descriptor(0, 1);
        for (k, (&a_col, &b_col)) in descriptor
            .a_columns
            .iter()
            .zip(descriptor.b_columns.iter())
            .enumerate()
        {
            let a = mpt_trace.columns[a_col].evaluations[0].to_u64();
            let b = gadget_trace.columns[b_col].evaluations[0].to_u64();
            assert_eq!(
                a, b,
                "tuple element {} mismatch: MPT col {} = {}, gadget col {} = {}",
                k, a_col, a, b_col, b
            );
        }

        assert_eq!(
            mpt_trace.columns[mpt_col::IS_PHASE3_LEAF_SHAPE].evaluations[0]
                .to_u64(),
            1
        );
        assert!(
            mpt_trace.columns[mpt_col::IS_PHASE1_LEAF_SHAPE].evaluations[0]
                .is_zero(),
            "Phase 1 selector must NOT fire on a Phase 3 row (mutual exclusion)"
        );
    }

    /// Confirm Phase 1 and Phase 3 shapes are mutually exclusive on a
    /// real witness: a 32-byte-key trie hits Phase 1 (not Phase 3),
    /// and a 2-byte-key trie hits Phase 3 (not Phase 1).
    #[test]
    fn mpt_phase_selectors_mutually_exclusive() {
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::inclusion_witness;

        // Phase 1 case: 32-byte key + 32-byte value.
        let key1 = vec![0x55u8; 32];
        let val1 = vec![0xaau8; 32];
        let (_, proof1) = single_leaf_trie(&key1, &val1);
        let rows1 = inclusion_witness(&key1, &proof1);
        assert_eq!(rows1[0].is_phase1_leaf_shape, 1);
        assert_eq!(rows1[0].is_phase3_leaf_shape, 0);

        // Phase 3 case: 2-byte key + 32-byte value.
        let key3 = vec![0x12u8, 0xabu8];
        let val3 = vec![0x77u8; 32];
        let (_, proof3) = single_leaf_trie(&key3, &val3);
        let rows3 = inclusion_witness(&key3, &proof3);
        assert_eq!(rows3[0].is_phase1_leaf_shape, 0);
        assert_eq!(rows3[0].is_phase3_leaf_shape, 1);

        // Off-shape case: 2-byte key + 5-byte value (NEITHER Phase 1
        // NOR Phase 3 shape).
        let val_off = b"hello".to_vec();
        let (_, proof_off) = single_leaf_trie(&key3, &val_off);
        let rows_off = inclusion_witness(&key3, &proof_off);
        assert_eq!(rows_off[0].is_phase1_leaf_shape, 0);
        assert_eq!(rows_off[0].is_phase3_leaf_shape, 0);
    }

    /// Slow regression: full prove + verify on a single-row honest
    /// witness validates the gadget end-to-end through the existing
    /// scheme-generic prover/verifier.
    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn mpt_short_leaf_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = ShortLeafRlpWitness::from_leaves(&[ShortLeafRow {
            path_bytes: sample_path_bytes(),
            value: sample_value(),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = ShortLeafRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "mpt_short_leaf_rlp gadget proof must verify");
    }

    /// End-to-end joint_prove + joint_verify across MPT + Phase 3
    /// gadget for a 2-byte-key single-leaf trie.
    #[test]
    #[ignore = "slow: full joint_prove + joint_verify across 2 AIRs; \
                run with --release --ignored"]
    fn joint_prove_mpt_short_leaf_rlp_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::inclusion_witness;
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x42u8, 0x73u8];
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = ShortLeafRlpWitness::from_leaves(&[ShortLeafRow {
            path_bytes: [key[0], key[1]],
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs =
            ShortLeafRlpConstraintSystem::new(gadget_trace.num_rows)
                .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_short_leaf_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched MPT + short-leaf gadget");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mpt_cs, &gadget_cs];
        let valid = joint_verify(
            &proofs, &cs_refs, &linkages, &extension, &scheme, curve,
        );
        assert!(valid, "joint verifier must accept honest short-leaf proof");
    }

    /// Phase 3 cross-AIR LogUp tampering regression: corrupt the
    /// MPT-side `KEY_PATH_BYTE[0]` on the Phase 3 leaf row while
    /// leaving the gadget side honest. The 322-element cross-AIR
    /// LogUp tuple includes KEY_PATH_BYTE; the multiset diverges
    /// and `joint_prove` rejects with "multiset equality cannot
    /// hold" during witness construction.
    ///
    /// Companion to the honest `joint_prove_mpt_short_leaf_rlp_linkage`
    /// regression and parallel to Phase 1+2's
    /// `joint_prove_mpt_leaf_rlp_rejects_tampered_key_path` and Phase
    /// 6's `joint_prove_mpt_odd_leaf_rejects_tampered_leading_nibble`.
    /// Closes the cross-AIR tampering coverage gap for Phase 3.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_short_leaf_rlp_rejects_tampered_key_path() {
        use crate::cross_air_logup::joint_prove;
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x42u8, 0x73u8];
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows[0].is_phase3_leaf_shape, 1);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper KEY_PATH_BYTE[0] on the Phase 3 leaf row: honest
        // value is `key[0] = 0x42`. Set to 0xff. The cross-AIR
        // LogUp tuple now diverges from the gadget's KEY_PATH_BYTE[0]
        // = 0x42.
        mpt_trace.columns[mpt_col::KEY_PATH_BYTE_OFFSET].evaluations[0] =
            Scalar::from_u64(0xff, curve);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = ShortLeafRlpWitness::from_leaves(&[ShortLeafRow {
            path_bytes: [key[0], key[1]],
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs =
            ShortLeafRlpConstraintSystem::new(gadget_trace.num_rows)
                .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_short_leaf_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered KEY_PATH_BYTE on Phase 3 leaf row"
        );
    }

    /// Phase 5 integration test: 3-AIR `joint_prove` exercising TWO
    /// simultaneous cross-AIR LogUp linkages from the MPT inclusion
    /// AIR — first to KeccakExtract (existing #91 hash chain) and
    /// second to the Phase 3 short-leaf gadget (new #92 RLP-decoding).
    ///
    /// This is the FIRST test where the MPT AIR participates in
    /// multiple cross-AIR LogUp linkages simultaneously, validating
    /// that the multi-linkage architecture composes cleanly. Combined
    /// with the existing single-linkage tests, it confirms that:
    ///   - the MPT AIR's `NODE_RLP` columns can be the A-side of TWO
    ///     different linkages (hash chain + RLP decoding) at once;
    ///   - the cross-AIR LogUp permutation argument correctly handles
    ///     N>2 AIR setups via a single `joint_prove` call;
    ///   - the linkages' selectors are independently honored so each
    ///     linkage's multiset stays consistent with its own gating.
    ///
    /// Setup:
    ///   - MPT: 2-byte-key single-leaf trie, 1 real row.
    ///   - KeccakExtract: 1 real row hashing the leaf's 38-byte RLP.
    ///   - Phase 3 gadget: 1 real row decoding the same RLP.
    ///   - L1: MPT ↔ KeccakExtract (no MPT-side selector, padding-row
    ///         zero tuples cancel naturally).
    ///   - L2: MPT ↔ Phase 3 gadget (gated by IS_PHASE3_LEAF_SHAPE on
    ///         MPT, IS_REAL on gadget).
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify; \
                run with --release --ignored"]
    fn joint_prove_mpt_short_leaf_with_keccak_extract_chain() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build, make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness, MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::inclusion_witness;
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x42u8, 0x73u8];
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        let (_root, proof) = single_leaf_trie(&key, &value);

        // ── MPT side (layer 0) ────────────────────────────────────
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].is_phase3_leaf_shape, 1);
        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // ── Phase 3 gadget side (layer 1) ─────────────────────────
        let gadget_w = ShortLeafRlpWitness::from_leaves(&[ShortLeafRow {
            path_bytes: [key[0], key[1]],
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs =
            ShortLeafRlpConstraintSystem::new(gadget_trace.num_rows)
                .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        // ── KeccakExtract side (layer 2) ──────────────────────────
        let leaf_rlp = &proof[0];
        let extract_w = KeccakExtractWitness::from_inputs(&[leaf_rlp.clone()])
            .expect("rlp output fits in MAX_INPUT_LEN");
        let extract_trace = ke_build(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        // ── Linkages ──────────────────────────────────────────────
        // L1 uses NODE_RLP_LEN as the MPT-side selector — non-zero on
        // real rows, zero on padding. Cross-AIR LogUp treats selectors
        // as binary indicators (`!is_zero()`), so this gates padding
        // rows out of the multiset and matches KeccakExtract's
        // IS_REAL gating. The default `_default` helper passes None
        // for the selector and only works when the MPT trace has no
        // padding — auto-inflation to a common domain breaks that.
        let mpt_rlp_bytes: Vec<usize> = (0..MAX_INPUT_LEN)
            .map(|b| crate::mpt_air::col::NODE_RLP_OFFSET + b)
            .collect();
        let mpt_node_hash: Vec<usize> = (0..OUTPUT_LEN)
            .map(|b| crate::mpt_air::col::NODE_HASH_OFFSET + b)
            .collect();
        let l1_mpt_keccak = make_mpt_keccak_extract_linkage_descriptor(
            /* mpt layer */ 0,
            /* keccak_extract layer */ 2,
            mpt_rlp_bytes,
            crate::mpt_air::col::NODE_RLP_LEN,
            mpt_node_hash,
            Some(crate::mpt_air::col::NODE_RLP_LEN),
        );
        let l2_mpt_gadget = make_mpt_short_leaf_rlp_linkage_descriptor(
            /* mpt layer */ 0,
            /* gadget layer */ 1,
        );

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&mpt_trace, &mpt_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1_mpt_keccak, l2_mpt_gadget];

        let (proofs, ext_proof) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for honest 3-AIR setup");
        assert_eq!(proofs.len(), 3);
        assert_eq!(ext_proof.linkage_proofs.len(), 2);
        for (i, lp) in ext_proof.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closure scalars must match for honest witness",
                i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mpt_cs, &gadget_cs, &extract_cs];
        let valid = joint_verify(
            &proofs, &cs_refs, &linkages, &ext_proof, &scheme, curve,
        );
        assert!(
            valid,
            "joint verifier must accept honest 3-AIR proof with two MPT-side linkages"
        );
    }

    /// Phase 5 3-AIR tampering regression: corrupt MPT-side
    /// `NODE_HASH` on the Phase 3 leaf row. `NODE_HASH` is in L1's
    /// tuple `(NODE_RLP, NODE_RLP_LEN, NODE_HASH)` but NOT in L2's
    /// tuple, so L1 (MPT ↔ KeccakExtract) catches the tampering
    /// independently of L2 (MPT ↔ Phase 3 gadget). `joint_prove`
    /// rejects with multiset mismatch.
    ///
    /// Validates that BOTH linkages in the 3-AIR setup are
    /// independently functional gates — Phase 5's honest test alone
    /// doesn't prove this. Confirms the L1 hash-chain integrity
    /// holds even when the L2 RLP-decoding linkage is satisfied.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_short_leaf_3air_rejects_tampered_node_hash() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness, MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x42u8, 0x73u8];
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper MPT-side NODE_HASH[3] on row 0. NODE_HASH is in L1's
        // (MPT↔KeccakExtract) tuple, so the multiset on the MPT side
        // diverges from KeccakExtract's honest OUTPUT.
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::NODE_HASH_OFFSET + 3].evaluations[0] =
            mpt_trace.columns[mpt_col::NODE_HASH_OFFSET + 3].evaluations[0]
                .add(&one);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // Phase 3 gadget side (honest).
        let gadget_w = ShortLeafRlpWitness::from_leaves(&[ShortLeafRow {
            path_bytes: [key[0], key[1]],
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs =
            ShortLeafRlpConstraintSystem::new(gadget_trace.num_rows)
                .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        // KeccakExtract side (honest, hashes the real leaf RLP).
        let leaf_rlp = &proof[0];
        let extract_w = KeccakExtractWitness::from_inputs(&[leaf_rlp.clone()])
            .expect("rlp output fits");
        let extract_trace = ke_build(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        let mpt_rlp_bytes: Vec<usize> = (0..MAX_INPUT_LEN)
            .map(|b| mpt_col::NODE_RLP_OFFSET + b)
            .collect();
        let mpt_node_hash: Vec<usize> = (0..OUTPUT_LEN)
            .map(|b| mpt_col::NODE_HASH_OFFSET + b)
            .collect();
        let l1 = make_mpt_keccak_extract_linkage_descriptor(
            0, 2,
            mpt_rlp_bytes,
            mpt_col::NODE_RLP_LEN,
            mpt_node_hash,
            Some(mpt_col::NODE_RLP_LEN),
        );
        let l2 = make_mpt_short_leaf_rlp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&mpt_trace, &mpt_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1, l2];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered MPT NODE_HASH (caught by L1)"
        );
    }

    /// Phase 5 L2-targeting tampering regression: corrupt MPT-side
    /// `KEY_PATH_BYTE[0]` on the Phase 3 leaf row. KEY_PATH_BYTE is
    /// in L2's 322-tuple but NOT in L1's hash-chain tuple. L2 catches
    /// the tampering independently of L1. Complement to
    /// `joint_prove_mpt_short_leaf_3air_rejects_tampered_node_hash`
    /// (which targets L1). Together both validate L1 and L2 are
    /// independent soundness gates in Phase 5's 3-AIR setup.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_short_leaf_3air_rejects_tampered_l2_key_path() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness, MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x42u8, 0x73u8];
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper KEY_PATH_BYTE[0] (in L2 only).
        mpt_trace.columns[mpt_col::KEY_PATH_BYTE_OFFSET].evaluations[0] =
            Scalar::from_u64(0xff, curve);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = ShortLeafRlpWitness::from_leaves(&[ShortLeafRow {
            path_bytes: [key[0], key[1]],
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs =
            ShortLeafRlpConstraintSystem::new(gadget_trace.num_rows)
                .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let leaf_rlp = &proof[0];
        let extract_w = KeccakExtractWitness::from_inputs(&[leaf_rlp.clone()])
            .expect("rlp output fits");
        let extract_trace = ke_build(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        let mpt_rlp_bytes: Vec<usize> = (0..MAX_INPUT_LEN)
            .map(|b| mpt_col::NODE_RLP_OFFSET + b)
            .collect();
        let mpt_node_hash: Vec<usize> = (0..OUTPUT_LEN)
            .map(|b| mpt_col::NODE_HASH_OFFSET + b)
            .collect();
        let l1 = make_mpt_keccak_extract_linkage_descriptor(
            0, 2,
            mpt_rlp_bytes,
            mpt_col::NODE_RLP_LEN,
            mpt_node_hash,
            Some(mpt_col::NODE_RLP_LEN),
        );
        let l2 = make_mpt_short_leaf_rlp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&mpt_trace, &mpt_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1, l2];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered MPT KEY_PATH_BYTE (caught by L2)"
        );
    }
}
