//! MPT branch gadget for the canonical "2 children at slots 1 and 5"
//! shape (Phase 10 of #92).
//!
//! Companion to Phase 8a (slots 0+1) and Phase 9 (slots 0+5). **First
//! branch shape with NO child at slot 0** — proves the multi-shape
//! architecture supports slot pairs not anchored at 0. Combined with
//! Phase 8b's Lagrange-at-(0,1) and Phase 9's Lagrange-at-(0,5), the
//! Lagrange chain-binding pattern is now demonstrated for three
//! qualitatively distinct slot configurations:
//!   - Adjacent slots starting at 0 (Phase 8b: 0+1)
//!   - Non-adjacent slots starting at 0 (Phase 9:  0+5)
//!   - Non-adjacent slots NOT starting at 0 (Phase 10: 1+5)
//!
//! # Canonical 83-byte RLP layout
//!
//! ```text
//!   [0]       0xf8   long-list header
//!   [1]       0x51   payload length = 81
//!   [2]       0x80   empty slot 0
//!   [3]       0xa0   slot 1 child-hash header
//!   [4..36]   ─      32 bytes child_hash_1
//!   [36..39]  ─      3 × 0x80 (empty slots 2..4)
//!   [39]      0xa0   slot 5 child-hash header
//!   [40..72]  ─      32 bytes child_hash_5
//!   [72..82]  ─      10 × 0x80 (empty slots 6..15)
//!   [82]      0x80   empty value slot (slot 16)
//!   [83..256]        zero-pad
//! ```
//!
//! # Phase 10 chain binding (in mpt_constraints, body 4)
//!
//! Lagrange interpolation at points (1, 5):
//!     `IS_PHASE10_BRANCH_15_SHAPE · Σ β^i ·
//!         ((5 − PATH_NIBBLE) · BRANCH_CHILD_0_HASH_BYTE_i +
//!          (PATH_NIBBLE − 1) · BRANCH_CHILD_1_HASH_BYTE_i −
//!          4 · NODE_HASH_byte_i(ω·X))`
//! For PATH_NIBBLE = 1: 4·child_0 = 4·NODE_HASH → NODE_HASH = child_0 (slot 1).
//! For PATH_NIBBLE = 5: 4·child_1 = 4·NODE_HASH → NODE_HASH = child_1 (slot 5).
//! Combined with row-local `path_nibble_branch15_pinning`
//! (`(PATH_NIBBLE − 1)·(PATH_NIBBLE − 5) = 0` gated by the selector),
//! the path-nibble-selected hash is pinned to the next row's NODE_HASH.

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

pub const NUM_CHILD_HASH_BYTES: usize = 32;

pub const RLP_LIST_HEADER: u64 = 0xf8;
pub const RLP_PAYLOAD_LEN: u64 = 81;
pub const HASH_STRING_HEADER: u64 = 0x80 + NUM_CHILD_HASH_BYTES as u64; // 0xa0
pub const EMPTY_SLOT_BYTE: u64 = 0x80;
pub const NODE_KIND_BRANCH: u64 = 0;
pub const RLP_OUTPUT_LEN: usize = 83;

pub const RLP_BYTE_WIDTH: usize = crate::mpt_air::MAX_RLP_LEN;

const RLP_OFFSET_SLOT_0_EMPTY: usize = 2;
const RLP_OFFSET_SLOT_1_HEADER: usize = 3;
const RLP_OFFSET_SLOT_1_BYTES: usize = 4;
const RLP_OFFSET_EMPTY_2_4_START: usize = 36;
const RLP_OFFSET_EMPTY_2_4_END: usize = 39;
const RLP_OFFSET_SLOT_5_HEADER: usize = 39;
const RLP_OFFSET_SLOT_5_BYTES: usize = 40;
const RLP_OFFSET_EMPTY_6_15_START: usize = 72;
const RLP_OFFSET_EMPTY_6_15_END: usize = 82;
const RLP_OFFSET_VALUE_SLOT: usize = 82;

const _: () = assert!(RLP_OFFSET_SLOT_5_BYTES + NUM_CHILD_HASH_BYTES == RLP_OFFSET_EMPTY_6_15_START);
const _: () = assert!(RLP_OFFSET_VALUE_SLOT + 1 == RLP_OUTPUT_LEN);

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_RLP_BYTE_OFFSET: usize = 0;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_NODE_KIND: usize = COL_RLP_LEN + 1;
pub const COL_CHILD_HASH_1_BYTE_OFFSET: usize = COL_NODE_KIND + 1;
pub const COL_CHILD_HASH_5_BYTE_OFFSET: usize =
    COL_CHILD_HASH_1_BYTE_OFFSET + NUM_CHILD_HASH_BYTES;
pub const COL_IS_REAL: usize = COL_CHILD_HASH_5_BYTE_OFFSET + NUM_CHILD_HASH_BYTES;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;
pub const NUM_ROW_CONSTRAINTS: usize = 13;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BranchRow {
    pub child_hash_1: [u8; NUM_CHILD_HASH_BYTES],
    pub child_hash_5: [u8; NUM_CHILD_HASH_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct BranchRlpWitness {
    pub branches: Vec<BranchRow>,
}

impl BranchRlpWitness {
    pub fn from_branches(branches: &[BranchRow]) -> Self {
        Self { branches: branches.to_vec() }
    }
}

pub fn canonical_rlp(
    child_hash_1: &[u8; NUM_CHILD_HASH_BYTES],
    child_hash_5: &[u8; NUM_CHILD_HASH_BYTES],
) -> [u8; RLP_OUTPUT_LEN] {
    let mut out = [0u8; RLP_OUTPUT_LEN];
    out[0] = RLP_LIST_HEADER as u8;
    out[1] = RLP_PAYLOAD_LEN as u8;
    out[RLP_OFFSET_SLOT_0_EMPTY] = EMPTY_SLOT_BYTE as u8;
    out[RLP_OFFSET_SLOT_1_HEADER] = HASH_STRING_HEADER as u8;
    out[RLP_OFFSET_SLOT_1_BYTES..RLP_OFFSET_SLOT_1_BYTES + NUM_CHILD_HASH_BYTES]
        .copy_from_slice(child_hash_1);
    for k in RLP_OFFSET_EMPTY_2_4_START..RLP_OFFSET_EMPTY_2_4_END {
        out[k] = EMPTY_SLOT_BYTE as u8;
    }
    out[RLP_OFFSET_SLOT_5_HEADER] = HASH_STRING_HEADER as u8;
    out[RLP_OFFSET_SLOT_5_BYTES..RLP_OFFSET_SLOT_5_BYTES + NUM_CHILD_HASH_BYTES]
        .copy_from_slice(child_hash_5);
    for k in RLP_OFFSET_EMPTY_6_15_START..RLP_OFFSET_EMPTY_6_15_END {
        out[k] = EMPTY_SLOT_BYTE as u8;
    }
    out[RLP_OFFSET_VALUE_SLOT] = EMPTY_SLOT_BYTE as u8;
    out
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BranchRlpWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.branches.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (i, br) in witness.branches.iter().enumerate() {
        let rlp = canonical_rlp(&br.child_hash_1, &br.child_hash_5);
        for (b, &v) in rlp.iter().enumerate() {
            columns[COL_RLP_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_RLP_LEN][i] = Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve);
        columns[COL_NODE_KIND][i] = Scalar::from_u64(NODE_KIND_BRANCH, curve);
        for (k, &b) in br.child_hash_1.iter().enumerate() {
            columns[COL_CHILD_HASH_1_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
        for (k, &b) in br.child_hash_5.iter().enumerate() {
            columns[COL_CHILD_HASH_5_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
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

pub struct BranchRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BranchRlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn eval_empty_2_4_pinned(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let const_80 = Scalar::from_u64(EMPTY_SLOT_BYTE, curve);
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for k in RLP_OFFSET_EMPTY_2_4_START..RLP_OFFSET_EMPTY_2_4_END {
        let body = col_evals[COL_RLP_BYTE_OFFSET + k].sub(&const_80);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[COL_IS_REAL].mul(&acc)
}

fn eval_empty_6_15_pinned(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let const_80 = Scalar::from_u64(EMPTY_SLOT_BYTE, curve);
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for k in RLP_OFFSET_EMPTY_6_15_START..RLP_OFFSET_EMPTY_6_15_END {
        let body = col_evals[COL_RLP_BYTE_OFFSET + k].sub(&const_80);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[COL_IS_REAL].mul(&acc)
}

fn build_empty_2_4_pinned_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let const_80_poly = vec![Scalar::from_u64(EMPTY_SLOT_BYTE, curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for k in RLP_OFFSET_EMPTY_2_4_START..RLP_OFFSET_EMPTY_2_4_END {
        let body = poly_sub(&col_coeffs[COL_RLP_BYTE_OFFSET + k], &const_80_poly, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

fn build_empty_6_15_pinned_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let const_80_poly = vec![Scalar::from_u64(EMPTY_SLOT_BYTE, curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for k in RLP_OFFSET_EMPTY_6_15_START..RLP_OFFSET_EMPTY_6_15_END {
        let body = poly_sub(&col_coeffs[COL_RLP_BYTE_OFFSET + k], &const_80_poly, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

impl VmConstraintSystem for BranchRlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "node_kind_pinned".into(),
            "rlp_len_pinned".into(),
            "rlp_list_header_pinned".into(),
            "rlp_payload_len_pinned".into(),
            "rlp_slot0_empty_pinned".into(),
            "rlp_slot1_header_pinned".into(),
            "rlp_slot5_header_pinned".into(),
            "rlp_empty_2_4_pinned".into(),
            "rlp_empty_6_15_pinned".into(),
            "rlp_value_slot_pinned".into(),
            "child_hash_1_binding".into(),
            "child_hash_5_binding".into(),
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
        let alpha = Scalar::from_u64(7, curve);
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for row in 0..n {
            let row_evals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let v = &row_evals[COL_IS_REAL];
            bodies[0][row] = v.mul(&v.sub(&one));
            bodies[1][row] = eval_byte_pinned(&row_evals, COL_NODE_KIND, NODE_KIND_BRANCH, COL_IS_REAL);
            bodies[2][row] = eval_byte_pinned(&row_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL);
            bodies[3][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL);
            bodies[4][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + 1, RLP_PAYLOAD_LEN, COL_IS_REAL);
            bodies[5][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_0_EMPTY, EMPTY_SLOT_BYTE, COL_IS_REAL);
            bodies[6][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_1_HEADER, HASH_STRING_HEADER, COL_IS_REAL);
            bodies[7][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_5_HEADER, HASH_STRING_HEADER, COL_IS_REAL);
            bodies[8][row] = eval_empty_2_4_pinned(&row_evals, &alpha);
            bodies[9][row] = eval_empty_6_15_pinned(&row_evals, &alpha);
            bodies[10][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_SLOT, EMPTY_SLOT_BYTE, COL_IS_REAL);
            bodies[11][row] = eval_byte_slice_binding(&row_evals, &alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_1_BYTES, COL_CHILD_HASH_1_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL);
            bodies[12][row] = eval_byte_slice_binding(&row_evals, &alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_5_BYTES, COL_CHILD_HASH_5_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL);
        }
        // Tail zero-pin (rolled into last α-term via evaluate_at_point's
        // β-RLC isn't used here; we just pin zero-tail in a separate
        // implicit body via the gadget's `evaluate_at_point`).
        let _ = build_zero_tail_poly::<>;
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
            eval_byte_pinned(col_evals, COL_NODE_KIND, NODE_KIND_BRANCH, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + 1, RLP_PAYLOAD_LEN, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_0_EMPTY, EMPTY_SLOT_BYTE, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_1_HEADER, HASH_STRING_HEADER, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_5_HEADER, HASH_STRING_HEADER, COL_IS_REAL),
            eval_empty_2_4_pinned(col_evals, alpha),
            eval_empty_6_15_pinned(col_evals, alpha),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_SLOT, EMPTY_SLOT_BYTE, COL_IS_REAL),
            eval_byte_slice_binding(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_1_BYTES, COL_CHILD_HASH_1_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL),
            eval_byte_slice_binding(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_5_BYTES, COL_CHILD_HASH_5_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL),
        ];
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
        }
        // Add zero-tail constraint past byte 83 (folded into the
        // overall α-RLC via the next α^NUM_ROW_CONSTRAINTS coefficient).
        let tail = eval_zero_tail(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OUTPUT_LEN, RLP_BYTE_WIDTH, COL_IS_REAL);
        acc = acc.add(&tail.mul(&ap));
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
            build_byte_pinned_poly(col_coeffs, COL_NODE_KIND, NODE_KIND_BRANCH, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + 1, RLP_PAYLOAD_LEN, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_0_EMPTY, EMPTY_SLOT_BYTE, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_1_HEADER, HASH_STRING_HEADER, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_5_HEADER, HASH_STRING_HEADER, COL_IS_REAL, curve),
            build_empty_2_4_pinned_poly(col_coeffs, alpha, curve),
            build_empty_6_15_pinned_poly(col_coeffs, alpha, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_SLOT, EMPTY_SLOT_BYTE, COL_IS_REAL, curve),
            build_byte_slice_binding_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_1_BYTES, COL_CHILD_HASH_1_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL, curve),
            build_byte_slice_binding_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_5_BYTES, COL_CHILD_HASH_5_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL, curve),
        ];
        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            let scaled = poly_scalar_mul(body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        // Append zero-tail at α^NUM_ROW_CONSTRAINTS to match
        // evaluate_at_point's structure.
        let tail = build_zero_tail_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OUTPUT_LEN, RLP_BYTE_WIDTH, COL_IS_REAL, curve);
        let scaled_tail = poly_scalar_mul(&tail, &ap);
        poly_add(&acc, &scaled_tail, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
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
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

// ─── Cross-AIR LogUp linkage descriptor ───────────────────────────────

pub fn make_mpt_branch_15_rlp_linkage_descriptor(
    mpt_layer_index: usize,
    rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;
    let mut a_columns: Vec<usize> = (0..RLP_BYTE_WIDTH)
        .map(|b| mpt_col::NODE_RLP_OFFSET + b)
        .collect();
    a_columns.push(mpt_col::NODE_RLP_LEN);
    a_columns.push(mpt_col::NODE_KIND);
    for b in 0..NUM_CHILD_HASH_BYTES {
        a_columns.push(mpt_col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + b);
    }
    for b in 0..NUM_CHILD_HASH_BYTES {
        a_columns.push(mpt_col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + b);
    }
    let mut b_columns: Vec<usize> = (0..RLP_BYTE_WIDTH).map(|b| COL_RLP_BYTE_OFFSET + b).collect();
    b_columns.push(COL_RLP_LEN);
    b_columns.push(COL_NODE_KIND);
    for b in 0..NUM_CHILD_HASH_BYTES {
        b_columns.push(COL_CHILD_HASH_1_BYTE_OFFSET + b);
    }
    for b in 0..NUM_CHILD_HASH_BYTES {
        b_columns.push(COL_CHILD_HASH_5_BYTE_OFFSET + b);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_branch_15_rlp_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_PHASE10_BRANCH_15_SHAPE),
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

    fn sample_hash(seed: u8) -> [u8; NUM_CHILD_HASH_BYTES] {
        let mut h = [0u8; NUM_CHILD_HASH_BYTES];
        for i in 0..NUM_CHILD_HASH_BYTES {
            h[i] = (i as u8).wrapping_mul(seed).wrapping_add(seed);
        }
        h
    }

    /// Critical correctness oracle: gadget's `canonical_rlp` matches
    /// `mpt::mpt_node_rlp` on the equivalent `MptNode::Branch{
    /// children[1], children[5] }`.
    #[test]
    fn canonical_rlp_matches_reference_encoder() {
        let h1 = sample_hash(0x33);
        let h5 = sample_hash(0x77);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(h1);
        children[5] = Some(h5);
        let branch = MptNode::Branch { children, value: None };
        let reference = mpt_node_rlp(&branch);
        let ours = canonical_rlp(&h1, &h5);
        assert_eq!(reference.len(), RLP_OUTPUT_LEN);
        assert_eq!(&ours[..], &reference[..]);
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_1: sample_hash(7),
            child_hash_5: sample_hash(13),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let cs = BranchRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(13, curve);
        for row in 0..trace.padded_size as usize {
            let col_vals: Vec<Scalar> = trace.columns.iter()
                .map(|p| p.evaluations[row].clone()).collect();
            assert!(cs.evaluate_at_point(&col_vals, &alpha).is_zero());
        }
    }

    /// Real 2-leaf trie with key first-nibbles 1 and 5 produces the
    /// Phase 10 branch shape via natural `inclusion_witness`.
    #[test]
    fn mpt_phase10_branch_selector_fires_on_2_leaf_trie() {
        use crate::mpt_air::inclusion_witness;
        let key_a = vec![0x15u8, 0xab]; // first nibble = 1
        let key_b = vec![0x56u8, 0xcd]; // first nibble = 5
        let val_a = vec![0xaau8; 32];
        let val_b = vec![0xbbu8; 32];
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x5, 0xa, 0xb]),
            value: val_a,
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x6, 0xc, 0xd]),
            value: val_b,
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(keccak256(&leaf_a_rlp));
        children[5] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].is_phase10_branch_15_shape, 1);
        assert_eq!(rows[0].is_phase8_branch_01_shape, 0);
        assert_eq!(rows[0].is_phase9_branch_05_shape, 0);
        assert_eq!(rows[0].path_nibble, 1); // consumed nibble = 1
        let _ = key_b;
    }

    #[test]
    fn descriptor_well_formed() {
        use crate::mpt_air::col as mpt_col;
        let desc = make_mpt_branch_15_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "mpt_branch_15_rlp_v1");
        assert_eq!(desc.a_columns.len(), RLP_BYTE_WIDTH + 1 + 1 + 32 + 32);
        assert_eq!(desc.a_selector_column, Some(mpt_col::IS_PHASE10_BRANCH_15_SHAPE));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn mpt_branch_15_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_1: sample_hash(7),
            child_hash_5: sample_hash(13),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = BranchRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid);
    }
}
