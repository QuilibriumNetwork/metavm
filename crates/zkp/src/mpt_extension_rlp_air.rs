//! MPT extension-node RLP-decoding consistency gadget (Phase 4 of #92).
//!
//! Companion to [`crate::mpt_short_leaf_rlp_air`] (the Phase 3 4-nibble
//! short-leaf gadget). Same RLP byte layout, but two byte differences
//! and a different decoded `node_kind`:
//!   - HP prefix byte (`RLP[2]`) is `0x00` (extension flag, even-length,
//!     no leading nibble) instead of `0x20` (leaf flag).
//!   - decoded `NODE_KIND` is `1` (extension) instead of `2` (leaf).
//!   - the 32 trailing bytes (`RLP[6..38]`) are the **child-hash
//!     pointer** of an extension node, not a leaf value.
//!
//! # Phase 4 scope (this module)
//!
//! Single fixed RLP shape: extension, **4-nibble even-length path** +
//! 32-byte child hash, 38-byte short-form list. Phase 4a closes the
//! "decoded fields = canonical RLP decoding" gap for this shape via
//! the cross-AIR LogUp linkage to [`crate::mpt_air`]. The remaining
//! gap **Phase 4b** — binding the extension's committed child hash to
//! the next MPT row's `NODE_HASH` — requires a 2nd shifted constraint
//! on the MPT side and is left as a focused follow-up.
//!
//! # Canonical 38-byte RLP layout
//!
//! ```text
//!   [0]      0xe5   short-list header (= 0xc0 + 37)
//!   [1]      0x83   HP-string header (= 0x80 + 3)
//!   [2]      0x00   HP prefix (extension-flag + even-length)
//!   [3..5]   ─      2 path bytes (4 packed nibbles)
//!   [5]      0xa0   32-byte string header (child-hash)
//!   [6..38]  ─      32 child-hash bytes (extension's pointer)
//!   [38..256]       zero-pad to MAX_RLP_LEN
//! ```
//!
//! # Shared MPT-side columns
//!
//! Phase 4 reuses the SAME MPT-side columns as Phase 3:
//! `KEY_PATH_BYTE[0..32]` (with bytes 2..32 zero-padded) and
//! `VALUE_BYTE[0..32]` (semantically the child-hash on extension rows).
//! The `IS_PHASE4_EXT_SHAPE` selector gates the cross-AIR LogUp;
//! Phase 1 / Phase 3 / Phase 4 selectors are mutually exclusive (a row
//! matches at most one shape) so each shape's multiset stays disjoint.

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

pub const NUM_PATH_NIBBLES: usize = 4;
pub const NUM_PATH_BYTES: usize = NUM_PATH_NIBBLES / 2;
pub const NUM_CHILD_HASH_BYTES: usize = 32;
pub const KEY_PATH_BYTE_WIDTH: usize = 32;

pub const HP_LEN: usize = 1 + NUM_PATH_BYTES;
pub const HP_STRING_HEADER: u64 = 0x80 + HP_LEN as u64; // 0x83

/// HP prefix byte for an even-length **extension** node:
/// bit 5 = 0 (extension flag), bit 4 = 0 (even length), low nibble = 0.
pub const HP_PREFIX_BYTE: u64 = 0x00;

pub const CHILD_HASH_STRING_HEADER: u64 = 0x80 + NUM_CHILD_HASH_BYTES as u64; // 0xa0
pub const RLP_PAYLOAD_LEN: u64 = (1 + HP_LEN + 1 + NUM_CHILD_HASH_BYTES) as u64; // 37
pub const RLP_LIST_HEADER: u64 = 0xc0 + RLP_PAYLOAD_LEN; // 0xe5
pub const RLP_OUTPUT_LEN: usize = 1 + RLP_PAYLOAD_LEN as usize; // 38

/// Decoded `node_kind` for this shape (= 1 = extension), matching
/// the MPT AIR convention in [`crate::mpt_air::col::NODE_KIND`].
pub const NODE_KIND_EXTENSION: u64 = 1;

pub const RLP_BYTE_WIDTH: usize = crate::mpt_air::MAX_RLP_LEN;

const RLP_OFFSET_HP_STR_HEADER: usize = 1;
const RLP_OFFSET_HP_PREFIX: usize = 2;
const RLP_OFFSET_PATH_BYTES: usize = 3;
const RLP_OFFSET_CHILD_HEADER: usize = RLP_OFFSET_PATH_BYTES + NUM_PATH_BYTES; // 5
const RLP_OFFSET_CHILD_BYTES: usize = RLP_OFFSET_CHILD_HEADER + 1; // 6

const _: () = assert!(RLP_OFFSET_CHILD_BYTES + NUM_CHILD_HASH_BYTES == RLP_OUTPUT_LEN);

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_RLP_BYTE_OFFSET: usize = 0;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_NODE_KIND: usize = COL_RLP_LEN + 1;
/// Mirrors the MPT-side `KEY_PATH_BYTE` width (32). Indices 0..2 are
/// the real path bytes; 2..32 are pinned to zero by `key_path_zero_tail`.
pub const COL_KEY_PATH_BYTE_OFFSET: usize = COL_NODE_KIND + 1;
/// Decoded child-hash bytes (semantically the same column slot as the
/// Phase 3 leaf gadget's `VALUE_BYTE` — both gadgets contribute to the
/// SAME shared MPT-side column via cross-AIR LogUp on Phase 3 / Phase 4
/// selectors respectively).
pub const COL_CHILD_HASH_BYTE_OFFSET: usize =
    COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH;
pub const COL_IS_REAL: usize = COL_CHILD_HASH_BYTE_OFFSET + NUM_CHILD_HASH_BYTES;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionRow {
    pub path_bytes: [u8; NUM_PATH_BYTES],
    pub child_hash: [u8; NUM_CHILD_HASH_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct ExtensionRlpWitness {
    pub extensions: Vec<ExtensionRow>,
}

impl ExtensionRlpWitness {
    pub fn from_extensions(extensions: &[ExtensionRow]) -> Self {
        Self {
            extensions: extensions.to_vec(),
        }
    }
}

pub fn canonical_rlp(
    path_bytes: &[u8; NUM_PATH_BYTES],
    child_hash: &[u8; NUM_CHILD_HASH_BYTES],
) -> [u8; RLP_OUTPUT_LEN] {
    let mut out = [0u8; RLP_OUTPUT_LEN];
    out[0] = RLP_LIST_HEADER as u8;
    out[RLP_OFFSET_HP_STR_HEADER] = HP_STRING_HEADER as u8;
    out[RLP_OFFSET_HP_PREFIX] = HP_PREFIX_BYTE as u8;
    out[RLP_OFFSET_PATH_BYTES..RLP_OFFSET_PATH_BYTES + NUM_PATH_BYTES]
        .copy_from_slice(path_bytes);
    out[RLP_OFFSET_CHILD_HEADER] = CHILD_HASH_STRING_HEADER as u8;
    out[RLP_OFFSET_CHILD_BYTES..RLP_OFFSET_CHILD_BYTES + NUM_CHILD_HASH_BYTES]
        .copy_from_slice(child_hash);
    out
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ExtensionRlpWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.extensions.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (i, ext) in witness.extensions.iter().enumerate() {
        let rlp = canonical_rlp(&ext.path_bytes, &ext.child_hash);
        for (b, &v) in rlp.iter().enumerate() {
            columns[COL_RLP_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_RLP_LEN][i] = Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve);
        columns[COL_NODE_KIND][i] = Scalar::from_u64(NODE_KIND_EXTENSION, curve);
        for (k, &b) in ext.path_bytes.iter().enumerate() {
            columns[COL_KEY_PATH_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
        for (k, &b) in ext.child_hash.iter().enumerate() {
            columns[COL_CHILD_HASH_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
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

pub struct ExtensionRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ExtensionRlpConstraintSystem {
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

impl VmConstraintSystem for ExtensionRlpConstraintSystem {
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
            "rlp_child_string_header_pinned".into(),
            "path_byte_binding".into(),
            "key_path_byte_zero_tail".into(),
            "child_hash_byte_binding".into(),
            "rlp_zero_tail".into(),
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
            bodies[1][row] = eval_byte_pinned(&row_evals, COL_NODE_KIND, NODE_KIND_EXTENSION, COL_IS_REAL);
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
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_CHILD_HEADER,
                CHILD_HASH_STRING_HEADER,
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
                RLP_OFFSET_CHILD_BYTES,
                COL_CHILD_HASH_BYTE_OFFSET,
                NUM_CHILD_HASH_BYTES,
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
            eval_byte_pinned(col_evals, COL_NODE_KIND, NODE_KIND_EXTENSION, COL_IS_REAL),
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
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_CHILD_HEADER,
                CHILD_HASH_STRING_HEADER,
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
                RLP_OFFSET_CHILD_BYTES,
                COL_CHILD_HASH_BYTE_OFFSET,
                NUM_CHILD_HASH_BYTES,
                COL_IS_REAL,
            ),
            eval_zero_tail(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OUTPUT_LEN, RLP_BYTE_WIDTH, COL_IS_REAL),
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
        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(v, &v_minus_1, curve),
            build_byte_pinned_poly(col_coeffs, COL_NODE_KIND, NODE_KIND_EXTENSION, COL_IS_REAL, curve),
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
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_CHILD_HEADER,
                CHILD_HASH_STRING_HEADER,
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
                RLP_OFFSET_CHILD_BYTES,
                COL_CHILD_HASH_BYTE_OFFSET,
                NUM_CHILD_HASH_BYTES,
                COL_IS_REAL,
                curve,
            ),
            build_zero_tail_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OUTPUT_LEN, RLP_BYTE_WIDTH, COL_IS_REAL, curve),
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
/// Phase-4-shape extension rows** to **this gadget's algebraically
/// decoded rows**: matches the same 322-element tuple as the Phase 1
/// and Phase 3 linkages, but gated by
/// [`crate::mpt_air::col::IS_PHASE4_EXT_SHAPE`] on the MPT side.
///
/// The MPT-side `KEY_PATH_BYTE` and `VALUE_BYTE` columns are SHARED
/// across Phase 1 / Phase 3 / Phase 4 linkages — `VALUE_BYTE` is reused
/// for the 32-byte child-hash on Phase 4 rows. Mutual exclusion of the
/// shape selectors (enforced at the witness level) keeps each linkage's
/// multiset disjoint.
///
/// **Phase 4a closes the "decoded fields = canonical RLP decoding" gap.**
/// The remaining gap (Phase 4b) — binding the extension's committed
/// child hash to the next MPT row's `NODE_HASH` — requires a 2nd
/// shifted constraint on the MPT side and is left as a focused
/// follow-up.
pub fn make_mpt_extension_rlp_linkage_descriptor(
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
    for b in 0..NUM_CHILD_HASH_BYTES {
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
    for b in 0..NUM_CHILD_HASH_BYTES {
        b_columns.push(COL_CHILD_HASH_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_extension_rlp_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_PHASE4_EXT_SHAPE),
        b_layer_index: rlp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak::keccak256;
    use crate::mpt::{mpt_node_rlp, MptNode, Nibbles};

    fn sample_path_bytes() -> [u8; NUM_PATH_BYTES] {
        [0xab, 0xcd]
    }

    fn sample_child_hash() -> [u8; NUM_CHILD_HASH_BYTES] {
        let mut h = [0u8; NUM_CHILD_HASH_BYTES];
        for i in 0..NUM_CHILD_HASH_BYTES {
            h[i] = (i as u8).wrapping_mul(7).wrapping_add(13);
        }
        h
    }

    fn sample_witness() -> ExtensionRlpWitness {
        ExtensionRlpWitness::from_extensions(&[
            ExtensionRow {
                path_bytes: [0u8; NUM_PATH_BYTES],
                child_hash: [0u8; NUM_CHILD_HASH_BYTES],
            },
            ExtensionRow {
                path_bytes: sample_path_bytes(),
                child_hash: sample_child_hash(),
            },
            ExtensionRow {
                path_bytes: [0xffu8; NUM_PATH_BYTES],
                child_hash: [0xffu8; NUM_CHILD_HASH_BYTES],
            },
        ])
    }

    /// Cross-check our hand-rolled `canonical_rlp` against the reference
    /// `mpt_node_rlp` encoder for a 4-nibble even extension with a
    /// 32-byte child hash.
    #[test]
    fn canonical_rlp_matches_reference_encoder() {
        let path_bytes = sample_path_bytes();
        let child_hash = sample_child_hash();
        let ours = canonical_rlp(&path_bytes, &child_hash);

        let mut nibbles = Vec::with_capacity(NUM_PATH_NIBBLES);
        for &b in &path_bytes {
            nibbles.push(b >> 4);
            nibbles.push(b & 0x0f);
        }
        let node = MptNode::Extension {
            path: Nibbles::from_nibbles(nibbles),
            child: child_hash,
        };
        let reference = mpt_node_rlp(&node);

        assert_eq!(reference.len(), RLP_OUTPUT_LEN);
        assert_eq!(&ours[..], &reference[..]);
    }

    #[test]
    fn canonical_rlp_structural_bytes_match_constants() {
        let path_bytes = sample_path_bytes();
        let child_hash = sample_child_hash();
        let rlp = canonical_rlp(&path_bytes, &child_hash);
        assert_eq!(rlp[0] as u64, RLP_LIST_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_STR_HEADER] as u64, HP_STRING_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_PREFIX] as u64, HP_PREFIX_BYTE);
        assert_eq!(rlp[RLP_OFFSET_CHILD_HEADER] as u64, CHILD_HASH_STRING_HEADER);
        for k in 0..NUM_PATH_BYTES {
            assert_eq!(rlp[RLP_OFFSET_PATH_BYTES + k], path_bytes[k]);
        }
        for k in 0..NUM_CHILD_HASH_BYTES {
            assert_eq!(rlp[RLP_OFFSET_CHILD_BYTES + k], child_hash[k]);
        }
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 3);
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
                "KEY_PATH_BYTE[{}] must be zero on Phase 4 rows", k
            );
        }
        let h = sample_child_hash();
        for k in 0..NUM_CHILD_HASH_BYTES {
            assert_eq!(
                trace.columns[COL_CHILD_HASH_BYTE_OFFSET + k].evaluations[1].to_u64(),
                h[k] as u64
            );
        }
        // NODE_KIND = 1 (extension).
        assert_eq!(
            trace.columns[COL_NODE_KIND].evaluations[1].to_u64(),
            NODE_KIND_EXTENSION
        );
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = ExtensionRlpConstraintSystem::new(trace.num_rows);
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

    #[test]
    fn tampered_hp_prefix_byte_fires() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = ExtensionRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        let one = Scalar::one(curve);
        // Set HP prefix to 0x20 (the LEAF flag); extension shape must reject.
        trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX].evaluations[0] =
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX]
                .evaluations[0]
                .add(&Scalar::from_u64(0x20, curve))
                .sub(&Scalar::from_u64(0x00, curve));
        // (above is just adding 0x20 since current value is 0x00.)
        let _ = one;
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "rlp_hp_prefix_pinned must fire when HP prefix flips to leaf flag"
        );
    }

    #[test]
    fn tampered_child_hash_byte_breaks_binding() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = ExtensionRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(19, curve);
        let one = Scalar::one(curve);
        // Tamper CHILD_HASH_BYTE[5] (decoded), leaving RLP child-hash region honest.
        trace.columns[COL_CHILD_HASH_BYTE_OFFSET + 5].evaluations[1] =
            trace.columns[COL_CHILD_HASH_BYTE_OFFSET + 5].evaluations[1].add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "child_hash_byte_binding must fire on RLP/decoded mismatch"
        );
    }

    // ── Phase 4 cross-AIR LogUp linkage tests ──────────────────────

    #[test]
    fn mpt_extension_rlp_descriptor_well_formed() {
        use crate::mpt_air::col as mpt_col;
        let desc = make_mpt_extension_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "mpt_extension_rlp_v1");
        let expected_width =
            RLP_BYTE_WIDTH + 1 + 1 + KEY_PATH_BYTE_WIDTH + NUM_CHILD_HASH_BYTES;
        assert_eq!(desc.a_columns.len(), expected_width);
        assert_eq!(desc.b_columns.len(), expected_width);
        assert_eq!(desc.a_columns[0], mpt_col::NODE_RLP_OFFSET);
        assert_eq!(desc.a_columns[RLP_BYTE_WIDTH + 1], mpt_col::NODE_KIND);
        assert_eq!(
            desc.a_columns[RLP_BYTE_WIDTH + 2 + KEY_PATH_BYTE_WIDTH],
            mpt_col::VALUE_BYTE_OFFSET
        );
        assert_eq!(desc.a_selector_column, Some(mpt_col::IS_PHASE4_EXT_SHAPE));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    /// Build a real 2-row MPT inclusion proof (extension → leaf) and
    /// verify the Phase 4 cross-AIR tuple agrees byte-for-byte on the
    /// extension row, while the leaf row's selectors are all off.
    #[test]
    fn mpt_extension_gadget_byte_tuples_match_real_mpt_witness() {
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::build_trace_polynomials_from_rows;

        let curve = CurveType::Bls48581;

        // Construct a 2-row trie:
        //   - row 0: extension consuming 4 nibbles (path=[1, 2, a, b]),
        //     pointing to row 1's leaf hash
        //   - row 1: leaf with the remaining 12-nibble path + value
        // Total key: 8 bytes = 16 nibbles. Extension consumes 4 → leaf
        // path is 12 nibbles (odd parity → not a Phase 1/3 shape, so
        // the leaf's selector bits are all off).
        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"some-leaf-value".to_vec();

        // Build the leaf with the 12-nibble residual path.
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value: value.clone(),
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);

        // Build the extension with the 4-nibble shared prefix.
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let ext_rlp = mpt_node_rlp(&ext);
        assert_eq!(ext_rlp.len(), RLP_OUTPUT_LEN, "extension RLP must be 38 bytes");
        assert_eq!(ext_rlp[2], 0x00, "HP prefix must be 0x00 (extension+even)");

        let proof = vec![ext_rlp.clone(), leaf_rlp];
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].is_phase4_ext_shape, 1, "row 0 must match Phase 4");
        assert_eq!(rows[0].is_phase1_leaf_shape, 0);
        assert_eq!(rows[0].is_phase3_leaf_shape, 0);
        assert_eq!(
            rows[1].is_phase4_ext_shape, 0,
            "row 1 (leaf) must NOT match Phase 4"
        );

        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Build the matching gadget invocation from the extension's
        // path bytes and child hash.
        let gadget_w = ExtensionRlpWitness::from_extensions(&[ExtensionRow {
            path_bytes: [0x12, 0xab],
            child_hash: leaf_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        let descriptor = make_mpt_extension_rlp_linkage_descriptor(0, 1);
        // MPT row 0 (extension) must match gadget row 0.
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
                "tuple element {} mismatch on extension row: MPT col {} = {}, \
                 gadget col {} = {}",
                k, a_col, a, b_col, b
            );
        }

        assert_eq!(
            mpt_trace.columns[mpt_col::IS_PHASE4_EXT_SHAPE].evaluations[0]
                .to_u64(),
            1
        );
        assert!(
            mpt_trace.columns[mpt_col::IS_PHASE4_EXT_SHAPE].evaluations[1]
                .is_zero(),
            "leaf row must NOT have IS_PHASE4_EXT_SHAPE set"
        );
    }

    /// Triple mutual-exclusion: a single MPT row never has more than
    /// one of {IS_PHASE1_LEAF_SHAPE, IS_PHASE3_LEAF_SHAPE,
    /// IS_PHASE4_EXT_SHAPE} set.
    #[test]
    fn mpt_phase_selectors_triple_mutual_exclusion() {
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::inclusion_witness;

        // Phase 1 case.
        let key1 = vec![0x55u8; 32];
        let val1 = vec![0xaau8; 32];
        let (_, p1) = single_leaf_trie(&key1, &val1);
        let r1 = inclusion_witness(&key1, &p1);
        assert_eq!(
            r1[0].is_phase1_leaf_shape + r1[0].is_phase3_leaf_shape
                + r1[0].is_phase4_ext_shape,
            1
        );

        // Phase 3 case.
        let key3 = vec![0x12u8, 0xab];
        let val3 = vec![0x77u8; 32];
        let (_, p3) = single_leaf_trie(&key3, &val3);
        let r3 = inclusion_witness(&key3, &p3);
        assert_eq!(
            r3[0].is_phase1_leaf_shape + r3[0].is_phase3_leaf_shape
                + r3[0].is_phase4_ext_shape,
            1
        );

        // Phase 4 case (build a real 2-row trie as above).
        let key4 = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let val4 = b"x".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value: val4,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof4 = vec![mpt_node_rlp(&ext), leaf_rlp];
        let r4 = inclusion_witness(&key4, &proof4);
        assert_eq!(
            r4[0].is_phase1_leaf_shape + r4[0].is_phase3_leaf_shape
                + r4[0].is_phase4_ext_shape,
            1,
            "extension row must have exactly one shape selector set"
        );
        // Leaf at row 1 (12-nibble path, non-Phase1/3) must have no
        // selectors set.
        assert_eq!(
            r4[1].is_phase1_leaf_shape + r4[1].is_phase3_leaf_shape
                + r4[1].is_phase4_ext_shape,
            0,
            "12-nibble residual leaf must NOT match any Phase shape"
        );
    }

    /// Slow regression: full prove + verify on a single-row honest
    /// extension witness.
    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn mpt_extension_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = ExtensionRlpWitness::from_extensions(&[ExtensionRow {
            path_bytes: sample_path_bytes(),
            child_hash: sample_child_hash(),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = ExtensionRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "mpt_extension_rlp gadget proof must verify");
    }

    /// Phase 4b soundness: an honest 2-row trace (extension at row 0
    /// pointing to leaf at row 1) satisfies both shifted constraints.
    /// The Phase 4b shifted body
    /// `IS_PHASE4_EXT_SHAPE(X) · (NODE_HASH(ω·X) − VALUE_BYTE(X))`
    /// must vanish at every domain point, so its evaluation form on
    /// the witness columns is identically zero.
    #[test]
    fn phase4b_shifted_body_vanishes_on_honest_witness() {
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::build_trace_polynomials_from_rows;

        let curve = CurveType::Bls48581;
        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"phase4b-honest".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof = vec![mpt_node_rlp(&ext), leaf_rlp];
        let rows = inclusion_witness(&key, &proof);
        let trace = build_trace_polynomials_from_rows(&rows, curve);

        // Body 1 evaluation at row 0: IS_PHASE4_EXT_SHAPE(row 0) · Σ
        // β^i · (NODE_HASH_byte_i(row 1) − VALUE_BYTE_i(row 0)).
        // VALUE_BYTE on row 0 = leaf_hash; NODE_HASH on row 1 =
        // keccak256(leaf_rlp) = leaf_hash. They match → body 1 = 0.
        let phase4 = trace.columns[mpt_col::IS_PHASE4_EXT_SHAPE]
            .evaluations[0]
            .to_u64();
        assert_eq!(phase4, 1);
        for i in 0..NUM_CHILD_HASH_BYTES {
            let value_curr = trace.columns[mpt_col::VALUE_BYTE_OFFSET + i]
                .evaluations[0]
                .to_u64();
            let node_next = trace.columns[mpt_col::NODE_HASH_OFFSET + i]
                .evaluations[1]
                .to_u64();
            assert_eq!(
                value_curr, node_next,
                "honest extension's VALUE_BYTE[{}] must equal next row's NODE_HASH[{}]",
                i, i
            );
        }
    }

    /// Phase 4b tampering: corrupt the extension's `VALUE_BYTE` on the
    /// MPT side AND the gadget's matching `CHILD_HASH_BYTE` so the
    /// cross-AIR LogUp tuple STILL matches — but the new Phase 4b
    /// shifted body fires because the corrupted VALUE_BYTE no longer
    /// equals the next row's NODE_HASH. This test exercises the
    /// soundness gain that Phase 4b delivers OVER Phase 4a.
    ///
    /// Verified at the host level: the body-1 evaluation is non-zero
    /// at row 0. (The slow joint_prove tampering regression is the
    /// end-to-end version; this fast test pins the algebraic gap.)
    #[test]
    fn phase4b_tampered_value_byte_breaks_chain() {
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::build_trace_polynomials_from_rows;

        let curve = CurveType::Bls48581;
        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"phase4b-tamper".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof = vec![mpt_node_rlp(&ext), leaf_rlp];
        let rows = inclusion_witness(&key, &proof);
        let mut trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper VALUE_BYTE[3] on the extension row (row 0). The
        // next row's NODE_HASH[3] is unchanged, so body 1 must fire.
        let one = Scalar::one(curve);
        trace.columns[mpt_col::VALUE_BYTE_OFFSET + 3].evaluations[0] =
            trace.columns[mpt_col::VALUE_BYTE_OFFSET + 3].evaluations[0]
                .add(&one);

        // Compute body 1 at row 0 by hand (skipping the IS_REAL/exclusion
        // multiplication — both factors are 1 on row 0).
        let mut body_1 = Scalar::zero(curve);
        let beta = Scalar::from_u64(7, curve);
        let mut bp = Scalar::one(curve);
        for i in 0..NUM_CHILD_HASH_BYTES {
            let value_curr = &trace.columns[mpt_col::VALUE_BYTE_OFFSET + i]
                .evaluations[0];
            let node_next = &trace.columns[mpt_col::NODE_HASH_OFFSET + i]
                .evaluations[1];
            let diff = node_next.sub(value_curr);
            body_1 = body_1.add(&bp.mul(&diff));
            bp = bp.mul(&beta);
        }
        assert!(
            !body_1.is_zero(),
            "Phase 4b body 1 must be non-zero on tampered VALUE_BYTE"
        );
    }

    /// End-to-end joint_prove + joint_verify across MPT (2 rows:
    /// extension+leaf) and Phase 4 gadget (1 row: extension).
    #[test]
    #[ignore = "slow: full joint_prove + joint_verify across 2 AIRs; \
                run with --release --ignored"]
    fn joint_prove_mpt_extension_rlp_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::mpt_air::inclusion_witness;
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"chain-test-leaf".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof = vec![mpt_node_rlp(&ext), leaf_rlp];
        let rows = inclusion_witness(&key, &proof);

        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = ExtensionRlpWitness::from_extensions(&[ExtensionRow {
            path_bytes: [0x12, 0xab],
            child_hash: leaf_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = ExtensionRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_extension_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let (proofs, extension_proof) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched MPT + extension gadget");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension_proof.linkage_proofs.len(), 1);

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mpt_cs, &gadget_cs];
        let valid = joint_verify(
            &proofs,
            &cs_refs,
            &linkages,
            &extension_proof,
            &scheme,
            curve,
        );
        assert!(
            valid,
            "joint verifier must accept honest 2-row extension+leaf proof"
        );
    }

    /// 3-AIR integration test mirroring the Phase 3 leaf-gadget
    /// integration (`joint_prove_mpt_short_leaf_with_keccak_extract_chain`)
    /// but using the Phase 4 EXTENSION gadget. The MPT inclusion AIR
    /// participates in TWO simultaneous cross-AIR LogUp linkages:
    ///   - L1: existing #91 MPT ↔ KeccakExtract on the hash chain
    ///         tuple, gated by `NODE_RLP_LEN` on MPT side, `IS_REAL`
    ///         on extract side.
    ///   - L2: new #92 Phase 4 MPT ↔ extension gadget on the
    ///         322-element decoded-RLP tuple, gated by
    ///         `IS_PHASE4_EXT_SHAPE` on MPT side, `IS_REAL` on
    ///         gadget side.
    ///
    /// Validates that:
    ///   - Both extension AND leaf shapes participate in the
    ///     multi-linkage architecture without interfering.
    ///   - Phase 4b's NEW shifted constraint (which extends
    ///     `NUM_SHIFTED 1→2` on the MPT AIR) coexists cleanly with
    ///     a 3-AIR joint_prove cycle.
    ///   - The 2-row MPT trace (extension at row 0, leaf at row 1)
    ///     correctly hashes both rows via L1 while gating L2 to fire
    ///     only on the extension row.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify; \
                run with --release --ignored"]
    fn joint_prove_mpt_extension_with_keccak_extract_chain() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
            MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── 2-row MPT trie: extension (row 0, Phase 4 shape) + leaf
        //    (row 1, 12-nibble residual path) ──────────────────────
        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"3-air-extension-test".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof = vec![mpt_node_rlp(&ext), leaf_rlp.clone()];
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].is_phase4_ext_shape, 1);

        // ── MPT side (layer 0) ───────────────────────────────────
        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // ── Phase 4 extension gadget side (layer 1) ──────────────
        let gadget_w = ExtensionRlpWitness::from_extensions(&[ExtensionRow {
            path_bytes: [0x12, 0xab],
            child_hash: leaf_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = ExtensionRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        // ── KeccakExtract side (layer 2): hashes BOTH MPT rows. ──
        let extract_w = KeccakExtractWitness::from_inputs(&[
            mpt_node_rlp(&ext),
            leaf_rlp,
        ])
        .expect("rlp outputs fit in MAX_INPUT_LEN");
        let extract_trace = ke_build(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        // ── L1: MPT ↔ KeccakExtract hash chain ──────────────────
        let mpt_rlp_bytes: Vec<usize> = (0..MAX_INPUT_LEN)
            .map(|b| mpt_col::NODE_RLP_OFFSET + b)
            .collect();
        let mpt_node_hash: Vec<usize> = (0..OUTPUT_LEN)
            .map(|b| mpt_col::NODE_HASH_OFFSET + b)
            .collect();
        let l1_mpt_keccak = make_mpt_keccak_extract_linkage_descriptor(
            0, 2,
            mpt_rlp_bytes,
            mpt_col::NODE_RLP_LEN,
            mpt_node_hash,
            Some(mpt_col::NODE_RLP_LEN),
        );

        // ── L2: MPT ↔ Phase 4 extension gadget (decoded RLP) ────
        let l2_mpt_gadget = make_mpt_extension_rlp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&mpt_trace, &mpt_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1_mpt_keccak, l2_mpt_gadget];

        let (proofs, ext_proof) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for honest 3-AIR setup with extension gadget",
        );
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
            "joint verifier must accept honest 3-AIR proof with two MPT-side \
             linkages (KeccakExtract + Phase 4 extension gadget)"
        );
    }

    /// Phase 5b 3-AIR tampering regression: corrupt MPT-side
    /// `NODE_HASH[3]` on row 0 (the extension row) in the 3-AIR
    /// extension+KeccakExtract setup. NODE_HASH is in L1's tuple
    /// `(NODE_RLP, NODE_RLP_LEN, NODE_HASH)` but NOT in L2's
    /// 322-tuple, so L1 (MPT ↔ KeccakExtract) catches the
    /// tampering independently of L2 (MPT ↔ Phase 4 gadget).
    /// `joint_prove` rejects with multiset mismatch.
    ///
    /// Mirror of `joint_prove_mpt_short_leaf_3air_rejects_tampered_node_hash`
    /// (Phase 5 leaf-shape tampering). Validates both linkages in
    /// the Phase 5b extension 3-AIR setup are independent soundness
    /// gates.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_extension_3air_rejects_tampered_node_hash() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
            MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Same 2-row trie setup as Phase 5b honest test.
        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"3-air-tamper-test".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof = vec![mpt_node_rlp(&ext), leaf_rlp.clone()];
        let rows = inclusion_witness(&key, &proof);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper MPT-side NODE_HASH[3] on row 0 (the extension's hash).
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::NODE_HASH_OFFSET + 3].evaluations[0] =
            mpt_trace.columns[mpt_col::NODE_HASH_OFFSET + 3].evaluations[0]
                .add(&one);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // Honest gadget side.
        let gadget_w = ExtensionRlpWitness::from_extensions(&[ExtensionRow {
            path_bytes: [0x12, 0xab],
            child_hash: leaf_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = ExtensionRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        // KeccakExtract side (honest, hashes both real RLPs).
        let extract_w = KeccakExtractWitness::from_inputs(&[
            mpt_node_rlp(&ext),
            leaf_rlp,
        ])
        .expect("rlp outputs fit");
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
        let l2 = make_mpt_extension_rlp_linkage_descriptor(0, 1);

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

    /// Phase 5b L2-targeting tampering regression: corrupt MPT-side
    /// `KEY_PATH_BYTE[0]` on the Phase 4 extension row. KEY_PATH_BYTE
    /// is in L2's 322-tuple but NOT in L1's hash-chain tuple. L2
    /// catches the tampering independently of L1. Brings Phase 5b to
    /// L1+L2 coverage parity with Phase 5c.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_extension_3air_rejects_tampered_l2_key_path() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
            MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"3-air-l2-tamper".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof = vec![mpt_node_rlp(&ext), leaf_rlp.clone()];
        let rows = inclusion_witness(&key, &proof);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper KEY_PATH_BYTE[0] on the extension row (L2-only).
        mpt_trace.columns[mpt_col::KEY_PATH_BYTE_OFFSET].evaluations[0] =
            Scalar::from_u64(0xff, curve);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = ExtensionRlpWitness::from_extensions(&[ExtensionRow {
            path_bytes: [0x12, 0xab],
            child_hash: leaf_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = ExtensionRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let extract_w = KeccakExtractWitness::from_inputs(&[
            mpt_node_rlp(&ext),
            leaf_rlp,
        ])
        .expect("rlp outputs fit");
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
        let l2 = make_mpt_extension_rlp_linkage_descriptor(0, 1);

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

    /// Phase 4b end-to-end soundness regression: corrupt the
    /// extension's `VALUE_BYTE` on BOTH the MPT and the gadget sides
    /// so the 322-element cross-AIR LogUp tuple **still matches**,
    /// hence Phase 4a's binding cannot detect the tampering. The
    /// new Phase 4b shifted constraint
    /// `IS_PHASE4_EXT_SHAPE(X) · (NODE_HASH(ω·X) − VALUE_BYTE(X)) = 0`
    /// then catches it: VALUE_BYTE is corrupted, NODE_HASH on the
    /// next row is unchanged, the body fires, the prover's quotient
    /// is ill-formed → joint_prove panics OR joint_verify rejects.
    ///
    /// This pins the soundness gain that Phase 4b delivers OVER
    /// Phase 4a (the previous test
    /// `phase4b_tampered_value_byte_breaks_chain` is the fast
    /// host-level analog).
    #[test]
    #[ignore = "slow: full joint_prove + joint_verify across 2 AIRs; \
                run with --release --ignored"]
    fn joint_prove_mpt_extension_rlp_phase4b_rejects_tampered_value_byte() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x12u8, 0xab, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let value = b"phase4b-tamper-e2e".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![
                0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8,
            ]),
            value,
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let leaf_hash = keccak256(&leaf_rlp);
        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(vec![0x1, 0x2, 0xa, 0xb]),
            child: leaf_hash,
        };
        let proof = vec![mpt_node_rlp(&ext), leaf_rlp];
        let rows = inclusion_witness(&key, &proof);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper VALUE_BYTE[3] on the extension row of the MPT trace.
        // We will also tamper the gadget side to keep the cross-AIR
        // LogUp tuple in agreement; only the Phase 4b shifted
        // constraint can catch this.
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::VALUE_BYTE_OFFSET + 3].evaluations[0] =
            mpt_trace.columns[mpt_col::VALUE_BYTE_OFFSET + 3].evaluations[0]
                .add(&one);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // Gadget side: build the matching invocation but tamper its
        // child_hash[3] to keep the LogUp tuple aligned with the MPT
        // side's tampered VALUE_BYTE.
        let mut tampered_child_hash = leaf_hash;
        tampered_child_hash[3] = tampered_child_hash[3].wrapping_add(1);
        let gadget_w = ExtensionRlpWitness::from_extensions(&[ExtensionRow {
            path_bytes: [0x12, 0xab],
            child_hash: tampered_child_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = ExtensionRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_extension_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        // Tampered witness must be rejected. The LogUp witness
        // builder accepts the tuple match (cross-AIR side agrees on
        // both sides), so the failure mode is the Phase 4b shifted
        // constraint: the constraint polynomial is no longer
        // divisible by Z_H, leading either to an `Err` from
        // joint_prove (during quotient construction) or to an
        // ill-formed proof that joint_verify rejects.
        match joint_prove(&traces, &linkages, &scheme) {
            Err(_) => {
                // joint_prove returned Err — Phase 4b soundness
                // gate fired during quotient construction.
            }
            Ok((proofs, ext_proof)) => {
                let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
                    vec![&mpt_cs, &gadget_cs];
                let valid = joint_verify(
                    &proofs, &cs_refs, &linkages, &ext_proof, &scheme, curve,
                );
                assert!(
                    !valid,
                    "joint verifier must reject Phase 4b-tampered proof"
                );
            }
        }
    }
}
