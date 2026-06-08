//! MPT-leaf RLP-decoding consistency gadget AIR (closes the Phase 1 slice
//! of #92).
//!
//! Per-row algebraic gadget that takes the raw RLP bytes of an MPT *leaf*
//! node and exposes its decoded fields (`node_kind`, `key_path`, `value`)
//! as separate columns, with row-local constraints pinning the decoded
//! fields to the canonical RLP shape. The future cross-AIR LogUp linkage
//! to [`crate::mpt_air`] (Phase 2) will match the gadget's `(node_rlp,
//! node_rlp_len, node_kind, key_path, value)` tuple against the MPT AIR's
//! corresponding columns, transferring the algebraic decode guarantee
//! into the MPT AIR's currently-oracle decoded fields.
//!
//! # Phase 1 scope (this module)
//!
//! Single fixed RLP shape: **leaf node, 64-nibble even-length path,
//! 32-byte value**. This is the realistic Ethereum storage / account
//! leaf shape (every key is a 32-byte keccak256 digest = 64 nibbles, and
//! 32-byte values cover the bulk of storage slots — anything wider /
//! account records / branch + extension nodes are explicit follow-ups
//! left to Phase 2+).
//!
//! Within that shape **every structural byte is a known constant**, so
//! the gadget decodes by pinning the 5 header bytes, copying the path
//! and value byte slices into separate decoded columns, and asserting
//! the zero-pad tail. No nibble unpacking, no variable-length headers,
//! no branches. Phase 2+ generalises (extension / branch / odd-length /
//! larger values + cross-AIR LogUp wiring).
//!
//! # Canonical 69-byte RLP layout for the Phase 1 shape
//!
//! ```text
//!   [0]      0xf8   long-list header (0xf7 + 1 byte of payload length)
//!   [1]      0x43   payload length = 67 (0x43)
//!   [2]      0xa1   HP-encoded path string header (0x80 + 33)
//!   [3]      0x20   HP prefix byte (leaf flag + even-length, first low nibble = 0)
//!   [4..36]  ─      32 path bytes (each packs 2 path nibbles, hi-nibble high)
//!   [36]     0xa0   value string header (0x80 + 32)
//!   [37..69] ─      32 value bytes
//!   [69..256]       zero-pad to MAX_RLP_LEN
//! ```
//!
//! Total = 69 bytes, zero-padded to 256 to match
//! [`crate::mpt_air::MAX_RLP_LEN`] for the Phase 2 cross-AIR LogUp.
//!
//! # Soundness chain (target end-state, after Phase 2)
//!
//! 1. **MPT AIR ↔ this gadget** (cross-AIR LogUp on the
//!    `(node_rlp[..256], node_rlp_len, node_kind, key_path, value)`
//!    tuple, gated by some "leaf-with-this-shape" selector on the MPT
//!    side and `IS_REAL` here): pins the gadget's RLP bytes equal to
//!    the MPT AIR's committed `node_rlp` AND pins the MPT AIR's
//!    decoded fields equal to this gadget's algebraically-decoded
//!    fields.
//!
//! 2. **This gadget's row-locals** (below) pin the decoded fields to
//!    canonical RLP decoding of the bytes.
//!
//! End state: a malicious prover cannot supply RLP bytes that hash to
//! the right `node_hash` (already pinned by the existing #91 KeccakExtract
//! linkage) but commit to fake `node_kind` / `key_path` / `value`.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::mpt_rlp_gadget_helpers::{
    build_byte_pinned_poly, build_byte_slice_binding_poly, build_zero_tail_poly,
    eval_byte_pinned, eval_byte_slice_binding, eval_zero_tail,
};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Number of nibbles in the path of the Phase-1 leaf shape. Fixed at
/// 64 — the full keccak256 hashed-key depth used by every account /
/// storage trie leaf at the canonical "deepest" position.
pub const NUM_PATH_NIBBLES: usize = 64;

/// Number of path bytes (HP-packed, 2 nibbles per byte). 64 / 2 = 32.
pub const NUM_PATH_BYTES: usize = NUM_PATH_NIBBLES / 2;

/// Number of value bytes in the Phase-1 leaf shape. Fixed at 32.
pub const NUM_VALUE_BYTES: usize = 32;

/// HP-encoded path length (1 prefix byte + 32 packed nibble bytes).
pub const HP_LEN: usize = 1 + NUM_PATH_BYTES;

/// HP-string RLP header byte: 0x80 + HP_LEN = 0x80 + 33 = 0xa1.
pub const HP_STRING_HEADER: u64 = 0x80 + HP_LEN as u64;

/// HP prefix byte for a 64-nibble (even-length) leaf:
///   - bit 5 (0x20) = leaf flag,
///   - bit 4 (0x10) = 0 because path length is even,
///   - low nibble = 0 because even leaves carry no isolated leading nibble.
pub const HP_PREFIX_BYTE: u64 = 0x20;

/// Value-string RLP header byte: 0x80 + NUM_VALUE_BYTES = 0x80 + 32 = 0xa0.
pub const VALUE_STRING_HEADER: u64 = 0x80 + NUM_VALUE_BYTES as u64;

/// RLP list payload length: HP-string body (1+33) + value-string body
/// (1+32) = 34 + 33 = 67. Always > 55, so the list goes long-form.
pub const RLP_PAYLOAD_LEN: u64 = (1 + HP_LEN + 1 + NUM_VALUE_BYTES) as u64;

/// List header byte (long form): 0xf7 + length-of-length (= 1) = 0xf8.
pub const RLP_LIST_HEADER: u64 = 0xf8;

/// Total RLP length for the Phase-1 leaf shape: 2 (long-list header +
/// payload-length byte) + 67 (payload) = 69 bytes.
pub const RLP_OUTPUT_LEN: usize = 2 + RLP_PAYLOAD_LEN as usize;

/// Decoded `node_kind` value for this shape (= 2 = leaf), matching the
/// MPT AIR convention in [`crate::mpt_air::col::NODE_KIND`].
pub const NODE_KIND_LEAF: u64 = 2;

/// Width of the keccak-input-shaped RLP byte column. Must match
/// [`crate::mpt_air::MAX_RLP_LEN`] so the Phase-2 cross-AIR LogUp tuple
/// has equal widths on both sides.
pub const RLP_BYTE_WIDTH: usize = crate::mpt_air::MAX_RLP_LEN;

// Byte offsets of the structural fields within the canonical RLP.
const RLP_OFFSET_HP_PREFIX: usize = 3;
const RLP_OFFSET_PATH_BYTES: usize = 4;
const RLP_OFFSET_VALUE_HEADER: usize = RLP_OFFSET_PATH_BYTES + NUM_PATH_BYTES; // 36
const RLP_OFFSET_VALUE_BYTES: usize = RLP_OFFSET_VALUE_HEADER + 1; // 37

const _: () = assert!(RLP_OFFSET_VALUE_BYTES + NUM_VALUE_BYTES == RLP_OUTPUT_LEN);

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_RLP_BYTE_OFFSET: usize = 0;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_NODE_KIND: usize = COL_RLP_LEN + 1;
pub const COL_KEY_PATH_BYTE_OFFSET: usize = COL_NODE_KIND + 1;
pub const COL_VALUE_BYTE_OFFSET: usize = COL_KEY_PATH_BYTE_OFFSET + NUM_PATH_BYTES;
pub const COL_IS_REAL: usize = COL_VALUE_BYTE_OFFSET + NUM_VALUE_BYTES;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafRow {
    /// 32 path bytes — each byte packs 2 path nibbles (hi-nibble high).
    /// Caller is responsible for HP-packing the leaf's nibble path; the
    /// gadget treats this as an opaque byte sequence.
    pub path_bytes: [u8; NUM_PATH_BYTES],
    /// 32-byte leaf value.
    pub value: [u8; NUM_VALUE_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct LeafRlpWitness {
    pub leaves: Vec<LeafRow>,
}

impl LeafRlpWitness {
    pub fn from_leaves(leaves: &[LeafRow]) -> Self {
        Self {
            leaves: leaves.to_vec(),
        }
    }
}

/// Construct the canonical 69-byte RLP encoding of the Phase-1 leaf
/// shape from `(path_bytes, value)`. Used by the witness builder and by
/// tests cross-checking against [`crate::mpt::mpt_node_rlp`].
pub fn canonical_rlp(
    path_bytes: &[u8; NUM_PATH_BYTES],
    value: &[u8; NUM_VALUE_BYTES],
) -> [u8; RLP_OUTPUT_LEN] {
    let mut out = [0u8; RLP_OUTPUT_LEN];
    out[0] = RLP_LIST_HEADER as u8;
    out[1] = RLP_PAYLOAD_LEN as u8;
    out[2] = HP_STRING_HEADER as u8;
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
    witness: &LeafRlpWitness,
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
        // bytes [RLP_OUTPUT_LEN..RLP_BYTE_WIDTH] are zero-padded by the
        // initial column allocation.
        columns[COL_RLP_LEN][i] = Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve);
        columns[COL_NODE_KIND][i] = Scalar::from_u64(NODE_KIND_LEAF, curve);
        for (k, &b) in leaf.path_bytes.iter().enumerate() {
            columns[COL_KEY_PATH_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
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

pub struct LeafRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl LeafRlpConstraintSystem {
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

// ─── Body helpers (eval form) ──────────────────────────────────────────

// (Helpers extracted to crate::mpt_rlp_gadget_helpers — see imports.)

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for LeafRlpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "node_kind_pinned".into(),
            "rlp_len_pinned".into(),
            "rlp_list_header_pinned".into(),
            "rlp_payload_len_pinned".into(),
            "rlp_hp_string_header_pinned".into(),
            "rlp_hp_prefix_pinned".into(),
            "rlp_value_string_header_pinned".into(),
            "path_byte_binding".into(),
            "value_byte_binding".into(),
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
            bodies[1][row] = eval_byte_pinned(&row_evals, COL_NODE_KIND, NODE_KIND_LEAF, COL_IS_REAL);
            bodies[2][row] = eval_byte_pinned(&row_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL);
            bodies[3][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL);
            bodies[4][row] =
                eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + 1, RLP_PAYLOAD_LEN, COL_IS_REAL);
            bodies[5][row] =
                eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + 2, HP_STRING_HEADER, COL_IS_REAL);
            bodies[6][row] = eval_byte_pinned(
                &row_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX,
                HP_PREFIX_BYTE,
                COL_IS_REAL,
            );
            bodies[7][row] = eval_byte_pinned(
                &row_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER,
                VALUE_STRING_HEADER,
                COL_IS_REAL,
            );
            bodies[8][row] = eval_byte_slice_binding(
                &row_evals,
                &alpha_for_rlc,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_PATH_BYTES,
                COL_KEY_PATH_BYTE_OFFSET,
                NUM_PATH_BYTES,
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
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + 1, RLP_PAYLOAD_LEN, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + 2, HP_STRING_HEADER, COL_IS_REAL),
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
            build_byte_pinned_poly(col_coeffs, COL_NODE_KIND, NODE_KIND_LEAF, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL, curve),
            build_byte_pinned_poly(
                col_coeffs,
                COL_RLP_BYTE_OFFSET + 1,
                RLP_PAYLOAD_LEN,
                COL_IS_REAL,
                curve,
            ),
            build_byte_pinned_poly(
                col_coeffs,
                COL_RLP_BYTE_OFFSET + 2,
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
/// Phase-1-shape leaf rows** to **this gadget's algebraically decoded
/// rows**: matches the 322-element tuple
/// `(NODE_RLP[0..256], NODE_RLP_LEN, NODE_KIND, KEY_PATH_BYTE[0..32],
/// VALUE_BYTE[0..32])` against the gadget's
/// `(RLP_BYTE[0..256], RLP_LEN, NODE_KIND, KEY_PATH_BYTE[0..32],
/// VALUE_BYTE[0..32])`.
///
/// MPT side is gated by [`crate::mpt_air::col::IS_PHASE1_LEAF_SHAPE`]
/// (1 only on terminal leaf rows whose RLP is the 69-byte canonical
/// Phase-1 shape; the witness builder
/// [`crate::mpt_air::inclusion_witness`] sets this automatically).
/// Gadget side is gated by [`COL_IS_REAL`].
///
/// Combined with the gadget's row-locals (which pin
/// `RLP_BYTE = canonical_rlp(KEY_PATH_BYTE, VALUE_BYTE)` and
/// `NODE_KIND = 2`), this linkage forces the MPT AIR's
/// `(NODE_KIND, KEY_PATH_BYTE, VALUE_BYTE)` columns to be the
/// canonical decoding of `NODE_RLP` for every selected MPT row —
/// closing the algebraic gap that otherwise lets a malicious prover
/// supply genuine RLP bytes (still hashing correctly via the existing
/// #91 KeccakExtract linkage) while committing fake decoded fields.
pub fn make_mpt_leaf_rlp_linkage_descriptor(
    mpt_layer_index: usize,
    rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;

    let mut a_columns: Vec<usize> = (0..RLP_BYTE_WIDTH)
        .map(|b| mpt_col::NODE_RLP_OFFSET + b)
        .collect();
    a_columns.push(mpt_col::NODE_RLP_LEN);
    a_columns.push(mpt_col::NODE_KIND);
    for b in 0..NUM_PATH_BYTES {
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
    for b in 0..NUM_PATH_BYTES {
        b_columns.push(COL_KEY_PATH_BYTE_OFFSET + b);
    }
    for b in 0..NUM_VALUE_BYTES {
        b_columns.push(COL_VALUE_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_leaf_rlp_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_PHASE1_LEAF_SHAPE),
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
        let mut p = [0u8; NUM_PATH_BYTES];
        for i in 0..NUM_PATH_BYTES {
            p[i] = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        p
    }

    fn sample_value() -> [u8; NUM_VALUE_BYTES] {
        let mut v = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            v[i] = (i as u8).wrapping_mul(11).wrapping_add(17);
        }
        v
    }

    fn sample_witness() -> LeafRlpWitness {
        let leaves = vec![
            LeafRow {
                path_bytes: [0u8; NUM_PATH_BYTES],
                value: [0u8; NUM_VALUE_BYTES],
            },
            LeafRow {
                path_bytes: sample_path_bytes(),
                value: sample_value(),
            },
            LeafRow {
                path_bytes: [0xffu8; NUM_PATH_BYTES],
                value: [0xffu8; NUM_VALUE_BYTES],
            },
        ];
        LeafRlpWitness::from_leaves(&leaves)
    }

    /// Cross-check our hand-rolled `canonical_rlp` against the reference
    /// `mpt_node_rlp` encoder on the same `(path, value)` pair.
    #[test]
    fn canonical_rlp_matches_reference_encoder() {
        let path_bytes = sample_path_bytes();
        let value = sample_value();
        let ours = canonical_rlp(&path_bytes, &value);

        // Reconstruct the equivalent MptNode::Leaf and RLP-encode it via
        // the reference encoder. Path = 64 nibbles (32 bytes × 2 nibbles).
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
        assert_eq!(rlp[1] as u64, RLP_PAYLOAD_LEN);
        assert_eq!(rlp[2] as u64, HP_STRING_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_PREFIX] as u64, HP_PREFIX_BYTE);
        assert_eq!(rlp[RLP_OFFSET_VALUE_HEADER] as u64, VALUE_STRING_HEADER);
        // Path bytes copied verbatim.
        for k in 0..NUM_PATH_BYTES {
            assert_eq!(rlp[RLP_OFFSET_PATH_BYTES + k], path_bytes[k]);
        }
        // Value bytes copied verbatim.
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
        // `nearest_power_of_two` floors at 16.
        assert_eq!(trace.padded_size, 16);
        // Row 1: structural bytes pinned to constants.
        assert_eq!(
            trace.columns[COL_RLP_BYTE_OFFSET].evaluations[1].to_u64(),
            RLP_LIST_HEADER
        );
        assert_eq!(
            trace.columns[COL_RLP_BYTE_OFFSET + 1].evaluations[1].to_u64(),
            RLP_PAYLOAD_LEN
        );
        assert_eq!(
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX].evaluations[1]
                .to_u64(),
            HP_PREFIX_BYTE
        );
        assert_eq!(
            trace.columns[COL_RLP_LEN].evaluations[1].to_u64(),
            RLP_OUTPUT_LEN as u64
        );
        assert_eq!(
            trace.columns[COL_NODE_KIND].evaluations[1].to_u64(),
            NODE_KIND_LEAF
        );
        // Decoded path / value match the witness leaf.
        let path = sample_path_bytes();
        let value = sample_value();
        for k in 0..NUM_PATH_BYTES {
            assert_eq!(
                trace.columns[COL_KEY_PATH_BYTE_OFFSET + k].evaluations[1].to_u64(),
                path[k] as u64
            );
        }
        for k in 0..NUM_VALUE_BYTES {
            assert_eq!(
                trace.columns[COL_VALUE_BYTE_OFFSET + k].evaluations[1].to_u64(),
                value[k] as u64
            );
        }
        // Zero-pad tail.
        assert!(
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_OUTPUT_LEN].evaluations[1]
                .is_zero()
        );
        assert!(
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH - 1].evaluations[1]
                .is_zero()
        );
        // Padding rows (indices 3..16): IS_REAL = 0.
        for row in 3..trace.padded_size as usize {
            assert!(
                trace.columns[COL_IS_REAL].evaluations[row].is_zero(),
                "padding row {} IS_REAL must be zero", row
            );
        }
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows);
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

    /// Tampering test: flipping a structural header byte must make the
    /// corresponding pinning body fire.
    #[test]
    fn tampered_list_header_byte_fires() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        let one = Scalar::one(curve);
        // Flip RLP[0] (= 0xf8 → 0xf9).
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

    /// Tampering test: a path byte in the RLP region not matching the
    /// decoded KEY_PATH_BYTE must make body 8 fire.
    #[test]
    fn tampered_path_byte_breaks_binding() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(19, curve);
        let one = Scalar::one(curve);
        // Tamper RLP[4] (path byte 0) but leave KEY_PATH_BYTE[0] honest.
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
            "path_byte_binding must fire when RLP path byte disagrees with KEY_PATH_BYTE"
        );
    }

    /// Tampering test: a value byte in the RLP region not matching the
    /// decoded VALUE_BYTE must make body 9 fire.
    #[test]
    fn tampered_value_byte_breaks_binding() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(23, curve);
        let one = Scalar::one(curve);
        // Tamper VALUE_BYTE[5] (decoded value), leaving RLP value bytes honest.
        trace.columns[COL_VALUE_BYTE_OFFSET + 5].evaluations[1] =
            trace.columns[COL_VALUE_BYTE_OFFSET + 5].evaluations[1].add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "value_byte_binding must fire when decoded VALUE_BYTE disagrees with RLP"
        );
    }

    /// Tampering test: a non-zero byte in the keccak-input zero-pad
    /// region (offset ≥ RLP_OUTPUT_LEN) must make body 10 fire.
    #[test]
    fn tampered_zero_tail_byte_fires() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(29, curve);
        // Set RLP[RLP_OUTPUT_LEN] to 1 (should be 0).
        trace.columns[COL_RLP_BYTE_OFFSET + RLP_OUTPUT_LEN].evaluations[0] =
            Scalar::from_u64(1, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "rlp_zero_tail must fire on non-zero pad byte"
        );
    }

    /// Tampering test: non-leaf NODE_KIND must fire body 1.
    #[test]
    fn tampered_node_kind_fires() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(31, curve);
        trace.columns[COL_NODE_KIND].evaluations[0] = Scalar::from_u64(0, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "node_kind_pinned must fire when NODE_KIND ≠ 2"
        );
    }

    // ── Phase 2 cross-AIR LogUp linkage tests ──────────────────────

    #[test]
    fn mpt_leaf_rlp_descriptor_well_formed() {
        use crate::mpt_air::col as mpt_col;
        let desc = make_mpt_leaf_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "mpt_leaf_rlp_v1");
        // Tuple width: 256 RLP bytes + 1 RLP_LEN + 1 NODE_KIND +
        // 32 KEY_PATH_BYTE + 32 VALUE_BYTE = 322.
        let expected_width = RLP_BYTE_WIDTH + 1 + 1 + NUM_PATH_BYTES + NUM_VALUE_BYTES;
        assert_eq!(desc.a_columns.len(), expected_width);
        assert_eq!(desc.b_columns.len(), expected_width);

        // A side starts at MPT NODE_RLP_OFFSET.
        assert_eq!(desc.a_columns[0], mpt_col::NODE_RLP_OFFSET);
        assert_eq!(
            desc.a_columns[RLP_BYTE_WIDTH - 1],
            mpt_col::NODE_RLP_OFFSET + RLP_BYTE_WIDTH - 1
        );
        assert_eq!(desc.a_columns[RLP_BYTE_WIDTH], mpt_col::NODE_RLP_LEN);
        assert_eq!(desc.a_columns[RLP_BYTE_WIDTH + 1], mpt_col::NODE_KIND);
        assert_eq!(
            desc.a_columns[RLP_BYTE_WIDTH + 2],
            mpt_col::KEY_PATH_BYTE_OFFSET
        );
        assert_eq!(
            desc.a_columns[RLP_BYTE_WIDTH + 2 + NUM_PATH_BYTES],
            mpt_col::VALUE_BYTE_OFFSET
        );

        // B side starts at gadget RLP_BYTE_OFFSET.
        assert_eq!(desc.b_columns[0], COL_RLP_BYTE_OFFSET);
        assert_eq!(desc.b_columns[RLP_BYTE_WIDTH], COL_RLP_LEN);
        assert_eq!(desc.b_columns[RLP_BYTE_WIDTH + 1], COL_NODE_KIND);
        assert_eq!(
            desc.b_columns[RLP_BYTE_WIDTH + 2],
            COL_KEY_PATH_BYTE_OFFSET
        );
        assert_eq!(
            desc.b_columns[RLP_BYTE_WIDTH + 2 + NUM_PATH_BYTES],
            COL_VALUE_BYTE_OFFSET
        );

        assert_eq!(desc.a_selector_column, Some(mpt_col::IS_PHASE1_LEAF_SHAPE));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    /// Build a real single-leaf MPT inclusion witness whose leaf is the
    /// canonical Phase 1 shape (32-byte hashed key → 64-nibble even
    /// path; 32-byte value), then build the matching gadget invocation
    /// from the same `(path_bytes, value)`. Assert the 322-element
    /// cross-AIR tuples agree element-by-element. This is the host-side
    /// foundation of the linkage's soundness.
    #[test]
    fn mpt_leaf_gadget_byte_tuples_match_real_mpt_witness() {
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::build_trace_polynomials_from_rows;

        let curve = CurveType::Bls48581;

        // Pick a representative 32-byte key and 32-byte value.
        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = (i as u8).wrapping_mul(13).wrapping_add(5);
        }
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(11).wrapping_add(17);
        }

        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.is_phase1_leaf_shape, 1);
        assert_eq!(row.node_rlp_len, RLP_OUTPUT_LEN);

        // The MPT witness' `key_path_bytes` should match the bytes from
        // the leaf's HP-encoded path region. Since the key is 32 bytes
        // (= 64 nibbles, even), the HP prefix is `0x20` (in RLP byte 3)
        // and bytes 4..36 are the packed nibbles, which equal `key`
        // verbatim (each path byte is `(hi_nibble << 4) | lo_nibble` =
        // the original key byte).
        assert_eq!(row.key_path_bytes, key);
        assert_eq!(row.value, value);

        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Build the matching gadget invocation.
        let gadget_w = LeafRlpWitness::from_leaves(&[LeafRow {
            path_bytes: key,
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        // Tuple-by-tuple equality on row 0 of each side.
        let descriptor = make_mpt_leaf_rlp_linkage_descriptor(0, 1);
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

        // Sanity: the MPT side selector fires on row 0.
        assert_eq!(
            mpt_trace.columns[mpt_col::IS_PHASE1_LEAF_SHAPE].evaluations[0]
                .to_u64(),
            1
        );
        // Padding rows: selector must be 0.
        for row_idx in 1..mpt_trace.padded_size as usize {
            assert!(
                mpt_trace.columns[mpt_col::IS_PHASE1_LEAF_SHAPE].evaluations[row_idx]
                    .is_zero(),
                "MPT padding row {} IS_PHASE1_LEAF_SHAPE must be zero",
                row_idx
            );
        }
    }

    /// Confirm that an MPT inclusion proof whose leaf does NOT match
    /// the Phase 1 shape (e.g. a short leaf with non-32-byte value)
    /// has `IS_PHASE1_LEAF_SHAPE = 0` on every row. The cross-AIR LogUp
    /// gates these rows out so they do not pollute the multiset.
    #[test]
    fn mpt_leaf_phase1_selector_off_for_non_matching_shape() {
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::inclusion_witness;

        // Short key (2 bytes) + short value: leaf shape mismatches the
        // Phase 1 64-nibble path requirement.
        let key = vec![0xab, 0xcd];
        let value = b"too-short-to-be-32-bytes".to_vec();
        assert!(value.len() != NUM_VALUE_BYTES);
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].is_phase1_leaf_shape, 0,
            "non-Phase1 shape leaf must NOT have the selector set"
        );
    }

    /// Tampering: an honest MPT proof whose decoded VALUE_BYTE column
    /// is corrupted (NODE_RLP left honest) breaks the cross-AIR
    /// tuple match. Detected here at host level by direct comparison;
    /// the `joint_prove` witness builder rejects the same condition
    /// with "multiset equality cannot hold".
    #[test]
    fn mpt_leaf_tampered_decoded_value_breaks_tuple_match() {
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::build_trace_polynomials_from_rows;

        let curve = CurveType::Bls48581;
        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = i as u8;
        }
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_add(100);
        }
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Corrupt VALUE_BYTE[3] on the MPT side, leaving NODE_RLP
        // honest.
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::VALUE_BYTE_OFFSET + 3].evaluations[0] =
            mpt_trace.columns[mpt_col::VALUE_BYTE_OFFSET + 3].evaluations[0]
                .add(&one);

        let gadget_w = LeafRlpWitness::from_leaves(&[LeafRow {
            path_bytes: key,
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        let descriptor = make_mpt_leaf_rlp_linkage_descriptor(0, 1);
        let mut found_mismatch = false;
        for (&a_col, &b_col) in
            descriptor.a_columns.iter().zip(descriptor.b_columns.iter())
        {
            let a = mpt_trace.columns[a_col].evaluations[0].to_u64();
            let b = gadget_trace.columns[b_col].evaluations[0].to_u64();
            if a != b {
                found_mismatch = true;
                break;
            }
        }
        assert!(
            found_mismatch,
            "tampered MPT VALUE_BYTE must produce a tuple element mismatch"
        );
    }

    /// Slow regression: full prove + verify on a single-row honest
    /// witness validates the gadget end-to-end through the existing
    /// scheme-generic prover/verifier.
    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn mpt_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = LeafRlpWitness::from_leaves(&[LeafRow {
            path_bytes: sample_path_bytes(),
            value: sample_value(),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "mpt_rlp leaf gadget proof must verify");
    }

    /// End-to-end cross-AIR `joint_prove`/`joint_verify` test for the
    /// MPT-leaf↔gadget linkage. Mirrors the closed regression
    /// `joint_prove_create_rlp_keccak_extract_input_linkage` (#95).
    ///
    /// Setup:
    ///   - MPT inclusion AIR with a single-leaf trie proof (32-byte
    ///     key, 32-byte value).
    ///   - RLP gadget AIR with the matching invocation.
    ///   - Linkage: 322-tuple
    ///     `(NODE_RLP[0..256], NODE_RLP_LEN, NODE_KIND,
    ///       KEY_PATH_BYTE[0..32], VALUE_BYTE[0..32])` ↔
    ///     `(RLP_BYTE[0..256], RLP_LEN, NODE_KIND,
    ///       KEY_PATH_BYTE[0..32], VALUE_BYTE[0..32])`,
    ///     gated by `IS_PHASE1_LEAF_SHAPE` on MPT side, `IS_REAL` on
    ///     gadget side.
    ///
    /// Validates that `joint_prove` accepts the matched tuple and
    /// `joint_verify` confirms the joint proof. Closes the algebraic
    /// gap between the MPT AIR's decoded-field oracles and the raw
    /// `NODE_RLP` bytes for the Phase 1 shape.
    #[test]
    #[ignore = "slow: full joint_prove + joint_verify across 2 AIRs; \
                run with --release --ignored"]
    fn joint_prove_mpt_leaf_rlp_linkage() {
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

        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_mul(13).wrapping_add(11);
        }

        // MPT side.
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // Gadget side.
        let gadget_w = LeafRlpWitness::from_leaves(&[LeafRow {
            path_bytes: key,
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = LeafRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_leaf_rlp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched MPT + RLP gadget");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b,
            "honest closure scalars must match"
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mpt_cs, &gadget_cs];
        let valid =
            joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest MPT leaf ↔ RLP gadget proof"
        );
    }

    /// Tampering: corrupt MPT-side `KEY_PATH_BYTE` while leaving
    /// `NODE_RLP` honest → `joint_prove` witness builder rejects with
    /// "multiset equality cannot hold".
    #[test]
    #[ignore = "slow: full joint_prove; run with --release --ignored"]
    fn joint_prove_mpt_leaf_rlp_rejects_tampered_key_path() {
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

        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = i as u8;
        }
        let mut value = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value[i] = (i as u8).wrapping_add(50);
        }
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper KEY_PATH_BYTE[0] — leaves NODE_RLP byte 4 untouched,
        // so the cross-AIR tuple disagrees on element 257 (RLP_BYTE +
        // RLP_LEN + NODE_KIND + KEY_PATH_BYTE[0]).
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::KEY_PATH_BYTE_OFFSET].evaluations[0] =
            mpt_trace.columns[mpt_col::KEY_PATH_BYTE_OFFSET].evaluations[0]
                .add(&one);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = LeafRlpWitness::from_leaves(&[LeafRow {
            path_bytes: key,
            value,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = LeafRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_leaf_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered KEY_PATH_BYTE on MPT side"
        );
    }
}
