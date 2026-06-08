//! MPT branch gadget for the canonical "2 children at slots 0 and 1"
//! shape (Phase 8a of #92).
//!
//! Demonstrates the multi-shape pattern's reach beyond leaves and
//! extensions to **17-list branch nodes**. This gadget covers a
//! single specific shape: branch with EXACTLY children at slots 0 and
//! 1, all other 14 child slots empty (`0x80`), empty value slot
//! (`0x80`), 83-byte long-form-list RLP. Other branch shapes (more
//! children, different positions, branches with values, etc.) are
//! deferred — each variant needs its own per-shape gadget OR a
//! generalised gadget with per-row "occupied-slot positions" columns
//! (~variable structural bytes pattern, like Phase 6's
//! LEADING_PATH_NIBBLE).
//!
//! # Canonical 83-byte RLP layout
//!
//! ```text
//!   [0]       0xf8   long-list header (0xf7 + 1 length-byte)
//!   [1]       0x51   payload length = 81 (0x51)
//!   [2]       0xa0   slot 0 child-hash header (0x80 + 32)
//!   [3..35]   ─      32 bytes child_hash_0
//!   [35]      0xa0   slot 1 child-hash header
//!   [36..68]  ─      32 bytes child_hash_1
//!   [68..82]  ─      14 × 0x80 (empty child slots 2..15)
//!   [82]      0x80   empty value slot (slot 16)
//!   [83..256]        zero-pad to MAX_RLP_LEN
//! ```
//!
//! # Cross-AIR LogUp tuple
//!
//! 258 elements (no decoded fields exposed to MPT):
//!   - 256 RLP bytes
//!   - 1 RLP_LEN (= 83)
//!   - 1 NODE_KIND (= 0 for branch)
//!
//! The gadget commits the two child hashes internally
//! (`CHILD_HASH_0_BYTE`, `CHILD_HASH_1_BYTE`) and pins the RLP byte
//! sequence to the canonical encoding. The cross-AIR LogUp transfers
//! "MPT's NODE_RLP for this branch row IS the canonical encoding of
//! some pair of child hashes" to the MPT side — i.e. structural
//! validity of the branch's RLP encoding is algebraically pinned.
//! Future Phase 8b work would expose the child hashes as MPT-side
//! columns (analogous to Phase 4 extensions exposing VALUE_BYTE)
//! and add a chain-binding shifted constraint pinning each child
//! hash to the next row's `NODE_HASH` along the inclusion path.

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

pub const RLP_LIST_HEADER: u64 = 0xf8; // long-list, len-of-len = 1
pub const RLP_PAYLOAD_LEN: u64 = 81;
pub const HASH_STRING_HEADER: u64 = 0x80 + NUM_CHILD_HASH_BYTES as u64; // 0xa0
pub const EMPTY_SLOT_BYTE: u64 = 0x80;
pub const NODE_KIND_BRANCH: u64 = 0;
pub const RLP_OUTPUT_LEN: usize = 83;

pub const RLP_BYTE_WIDTH: usize = crate::mpt_air::MAX_RLP_LEN;

const RLP_OFFSET_SLOT_0_HEADER: usize = 2;
const RLP_OFFSET_SLOT_0_BYTES: usize = 3;
const RLP_OFFSET_SLOT_1_HEADER: usize = 35;
const RLP_OFFSET_SLOT_1_BYTES: usize = 36;
const RLP_OFFSET_EMPTY_SLOTS_START: usize = 68; // slots 2..15 + slot 16 (value)
const RLP_OFFSET_EMPTY_SLOTS_END: usize = 83; // exclusive
const NUM_EMPTY_SLOT_BYTES: usize = RLP_OFFSET_EMPTY_SLOTS_END - RLP_OFFSET_EMPTY_SLOTS_START; // 15

const _: () = assert!(NUM_EMPTY_SLOT_BYTES == 15);

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_RLP_BYTE_OFFSET: usize = 0;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_NODE_KIND: usize = COL_RLP_LEN + 1;
pub const COL_CHILD_HASH_0_BYTE_OFFSET: usize = COL_NODE_KIND + 1;
pub const COL_CHILD_HASH_1_BYTE_OFFSET: usize =
    COL_CHILD_HASH_0_BYTE_OFFSET + NUM_CHILD_HASH_BYTES;
pub const COL_IS_REAL: usize = COL_CHILD_HASH_1_BYTE_OFFSET + NUM_CHILD_HASH_BYTES;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;
pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BranchRow {
    pub child_hash_0: [u8; NUM_CHILD_HASH_BYTES],
    pub child_hash_1: [u8; NUM_CHILD_HASH_BYTES],
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
    child_hash_0: &[u8; NUM_CHILD_HASH_BYTES],
    child_hash_1: &[u8; NUM_CHILD_HASH_BYTES],
) -> [u8; RLP_OUTPUT_LEN] {
    let mut out = [0u8; RLP_OUTPUT_LEN];
    out[0] = RLP_LIST_HEADER as u8;
    out[1] = RLP_PAYLOAD_LEN as u8;
    out[RLP_OFFSET_SLOT_0_HEADER] = HASH_STRING_HEADER as u8;
    out[RLP_OFFSET_SLOT_0_BYTES..RLP_OFFSET_SLOT_0_BYTES + NUM_CHILD_HASH_BYTES]
        .copy_from_slice(child_hash_0);
    out[RLP_OFFSET_SLOT_1_HEADER] = HASH_STRING_HEADER as u8;
    out[RLP_OFFSET_SLOT_1_BYTES..RLP_OFFSET_SLOT_1_BYTES + NUM_CHILD_HASH_BYTES]
        .copy_from_slice(child_hash_1);
    for k in RLP_OFFSET_EMPTY_SLOTS_START..RLP_OFFSET_EMPTY_SLOTS_END {
        out[k] = EMPTY_SLOT_BYTE as u8;
    }
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
        let rlp = canonical_rlp(&br.child_hash_0, &br.child_hash_1);
        for (b, &v) in rlp.iter().enumerate() {
            columns[COL_RLP_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_RLP_LEN][i] = Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve);
        columns[COL_NODE_KIND][i] = Scalar::from_u64(NODE_KIND_BRANCH, curve);
        for (k, &b) in br.child_hash_0.iter().enumerate() {
            columns[COL_CHILD_HASH_0_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
        for (k, &b) in br.child_hash_1.iter().enumerate() {
            columns[COL_CHILD_HASH_1_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
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

/// Branch-specific helper: pin all 15 empty-slot bytes to `0x80` via
/// β-RLC. (Different from `eval_zero_tail` which pins to 0.)
fn eval_empty_slots_pinned(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let const_80 = Scalar::from_u64(EMPTY_SLOT_BYTE, curve);
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for k in RLP_OFFSET_EMPTY_SLOTS_START..RLP_OFFSET_EMPTY_SLOTS_END {
        let body = col_evals[COL_RLP_BYTE_OFFSET + k].sub(&const_80);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[COL_IS_REAL].mul(&acc)
}

fn build_empty_slots_pinned_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let const_80_poly = vec![Scalar::from_u64(EMPTY_SLOT_BYTE, curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for k in RLP_OFFSET_EMPTY_SLOTS_START..RLP_OFFSET_EMPTY_SLOTS_END {
        let body = poly_sub(
            &col_coeffs[COL_RLP_BYTE_OFFSET + k],
            &const_80_poly,
            curve,
        );
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
            "rlp_slot0_header_pinned".into(),
            "rlp_slot1_header_pinned".into(),
            "rlp_empty_slots_pinned".into(),
            "child_hash_0_binding".into(),
            "child_hash_1_binding".into(),
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
            bodies[5][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_0_HEADER, HASH_STRING_HEADER, COL_IS_REAL);
            bodies[6][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_1_HEADER, HASH_STRING_HEADER, COL_IS_REAL);
            bodies[7][row] = eval_empty_slots_pinned(&row_evals, &alpha);
            bodies[8][row] = eval_byte_slice_binding(&row_evals, &alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_0_BYTES, COL_CHILD_HASH_0_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL);
            bodies[9][row] = eval_byte_slice_binding(&row_evals, &alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_1_BYTES, COL_CHILD_HASH_1_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL);
            bodies[10][row] = eval_zero_tail(&row_evals, &alpha, COL_RLP_BYTE_OFFSET, RLP_OUTPUT_LEN, RLP_BYTE_WIDTH, COL_IS_REAL);
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
            eval_byte_pinned(col_evals, COL_NODE_KIND, NODE_KIND_BRANCH, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + 1, RLP_PAYLOAD_LEN, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_0_HEADER, HASH_STRING_HEADER, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_1_HEADER, HASH_STRING_HEADER, COL_IS_REAL),
            eval_empty_slots_pinned(col_evals, alpha),
            eval_byte_slice_binding(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_0_BYTES, COL_CHILD_HASH_0_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL),
            eval_byte_slice_binding(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_1_BYTES, COL_CHILD_HASH_1_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL),
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
            build_byte_pinned_poly(col_coeffs, COL_NODE_KIND, NODE_KIND_BRANCH, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + 1, RLP_PAYLOAD_LEN, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_0_HEADER, HASH_STRING_HEADER, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_SLOT_1_HEADER, HASH_STRING_HEADER, COL_IS_REAL, curve),
            build_empty_slots_pinned_poly(col_coeffs, alpha, curve),
            build_byte_slice_binding_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_0_BYTES, COL_CHILD_HASH_0_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL, curve),
            build_byte_slice_binding_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_SLOT_1_BYTES, COL_CHILD_HASH_1_BYTE_OFFSET, NUM_CHILD_HASH_BYTES, COL_IS_REAL, curve),
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

pub fn make_mpt_branch_2child_rlp_linkage_descriptor(
    mpt_layer_index: usize,
    rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::mpt_air::col as mpt_col;
    // Phase 8b expanded tuple: 256 RLP bytes + 1 RLP_LEN + 1 NODE_KIND
    // + 32 BRANCH_CHILD_0_HASH_BYTE + 32 BRANCH_CHILD_1_HASH_BYTE = 322.
    // The new child-hash bytes ensure the MPT-side commitments equal
    // the gadget-internal CHILD_HASH_*_BYTE values, which are
    // algebraically pinned to the canonical RLP decoding by the
    // gadget's row-locals. Phase 8b's MPT shifted constraint then
    // pins the path-nibble-selected MPT child hash to the next row's
    // NODE_HASH.
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
        b_columns.push(COL_CHILD_HASH_0_BYTE_OFFSET + b);
    }
    for b in 0..NUM_CHILD_HASH_BYTES {
        b_columns.push(COL_CHILD_HASH_1_BYTE_OFFSET + b);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_branch_2child_rlp_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_PHASE8_BRANCH_01_SHAPE),
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

    /// Critical correctness oracle: gadget's canonical_rlp byte-identical
    /// to `mpt_node_rlp` on the equivalent `MptNode::Branch`.
    #[test]
    fn canonical_rlp_matches_reference_encoder() {
        let h0 = sample_hash(0x11);
        let h1 = sample_hash(0x22);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(h0);
        children[1] = Some(h1);
        let branch = MptNode::Branch { children, value: None };
        let reference = mpt_node_rlp(&branch);
        let ours = canonical_rlp(&h0, &h1);
        assert_eq!(reference.len(), RLP_OUTPUT_LEN);
        assert_eq!(&ours[..], &reference[..]);
    }

    #[test]
    fn canonical_rlp_structural_bytes() {
        let h0 = [0u8; 32];
        let h1 = [0u8; 32];
        let rlp = canonical_rlp(&h0, &h1);
        assert_eq!(rlp[0] as u64, RLP_LIST_HEADER);
        assert_eq!(rlp[1] as u64, RLP_PAYLOAD_LEN);
        assert_eq!(rlp[RLP_OFFSET_SLOT_0_HEADER] as u64, HASH_STRING_HEADER);
        assert_eq!(rlp[RLP_OFFSET_SLOT_1_HEADER] as u64, HASH_STRING_HEADER);
        for k in RLP_OFFSET_EMPTY_SLOTS_START..RLP_OFFSET_EMPTY_SLOTS_END {
            assert_eq!(rlp[k] as u64, EMPTY_SLOT_BYTE);
        }
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: sample_hash(7),
            child_hash_1: sample_hash(13),
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

    #[test]
    fn tampered_empty_slot_byte_fires() {
        let curve = CurveType::Bls48581;
        let w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: sample_hash(7),
            child_hash_1: sample_hash(13),
        }]);
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = BranchRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        // Set RLP[68] (first empty child slot) to 0xff (should be 0x80).
        trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_EMPTY_SLOTS_START].evaluations[0] =
            Scalar::from_u64(0xff, curve);
        let row_evals: Vec<Scalar> = trace.columns.iter()
            .map(|p| p.evaluations[0].clone()).collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "rlp_empty_slots_pinned must fire on non-0x80 empty slot byte"
        );
    }

    #[test]
    fn descriptor_well_formed() {
        use crate::mpt_air::col as mpt_col;
        let desc = make_mpt_branch_2child_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "mpt_branch_2child_rlp_v1");
        // Phase 8b: 256 RLP + 1 RLP_LEN + 1 NODE_KIND + 32 + 32 = 322.
        let expected = RLP_BYTE_WIDTH + 1 + 1 + NUM_CHILD_HASH_BYTES + NUM_CHILD_HASH_BYTES;
        assert_eq!(desc.a_columns.len(), expected);
        assert_eq!(desc.b_columns.len(), expected);
        assert_eq!(desc.a_selector_column, Some(mpt_col::IS_PHASE8_BRANCH_01_SHAPE));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
        // Last 64 a-columns are the new BRANCH_CHILD_*_HASH_BYTE ranges.
        assert_eq!(
            desc.a_columns[RLP_BYTE_WIDTH + 2],
            mpt_col::BRANCH_CHILD_0_HASH_BYTE_OFFSET
        );
        assert_eq!(
            desc.a_columns[RLP_BYTE_WIDTH + 2 + NUM_CHILD_HASH_BYTES],
            mpt_col::BRANCH_CHILD_1_HASH_BYTE_OFFSET
        );
    }

    /// Real 2-leaf trie with key first-nibbles 0 and 1 produces the
    /// Phase 8a branch shape via natural `inclusion_witness`.
    #[test]
    fn mpt_phase8_branch_selector_fires_on_2_leaf_trie() {
        use crate::mpt_air::inclusion_witness;
        let key_a = vec![0x05u8, 0xab];   // first nibble = 0
        let _key_b = vec![0x16u8, 0xcd];   // first nibble = 1
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
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(leaf_a_hash);
        children[1] = Some(leaf_b_hash);
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].is_phase8_branch_01_shape, 1, "branch row must match Phase 8a");
        assert_eq!(rows[0].node_kind, 0); // branch
        // Row 1 is a 3-nibble odd leaf (key residual after consuming
        // nibble 0 = nibbles 5, a, b) → matches Phase 6 shape.
        assert_eq!(rows[1].is_phase6_odd_leaf_shape, 1);
        // Phase 8a and Phase 6 selectors mutually exclusive (rows differ).
        assert_eq!(rows[0].is_phase6_odd_leaf_shape, 0);
        assert_eq!(rows[1].is_phase8_branch_01_shape, 0);
    }

    /// Honest joint_prove + joint_verify across MPT inclusion AIR +
    /// Phase 8a branch gadget for a real 2-leaf trie (key first-nibbles
    /// 0 and 1). Mirrors `joint_prove_mpt_short_leaf_rlp_linkage`
    /// (Phase 3) and brings Phase 8a to coverage parity with other
    /// phases.
    #[test]
    #[ignore = "slow: full joint_prove + joint_verify across 2 AIRs; \
                run with --release --ignored"]
    fn joint_prove_mpt_branch_2child_rlp_linkage() {
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

        // Same 2-leaf trie as the selector-fires test.
        let key_a = vec![0x05u8, 0xab];
        let _key_b = vec![0x16u8, 0xcd];
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
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(leaf_a_hash);
        children[1] = Some(leaf_b_hash);
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].is_phase8_branch_01_shape, 1);

        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: leaf_a_hash,
            child_hash_1: leaf_b_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = BranchRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_branch_2child_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let (proofs, ext_proof) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for honest MPT + Phase 8a branch gadget",
        );
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext_proof.linkage_proofs.len(), 1);
        assert_eq!(
            ext_proof.linkage_proofs[0].closure_a,
            ext_proof.linkage_proofs[0].closure_b,
            "honest closure scalars must match"
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mpt_cs, &gadget_cs];
        let valid = joint_verify(
            &proofs, &cs_refs, &linkages, &ext_proof, &scheme, curve,
        );
        assert!(
            valid,
            "joint verifier must accept honest 2-row branch+leaf proof"
        );
    }

    /// Phase 8a cross-AIR LogUp tampering regression: corrupt MPT-side
    /// `NODE_RLP[3]` (first byte of slot 0's child hash) on the
    /// branch row while leaving the gadget honest. Multiset diverges
    /// at element 3 of the 258-tuple → joint_prove rejects fast.
    /// Parallel to all other phases' tampering regressions.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_branch_2child_rlp_rejects_tampered_node_rlp() {
        use crate::cross_air_logup::joint_prove;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key_a = vec![0x05u8, 0xab];
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
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(leaf_a_hash);
        children[1] = Some(leaf_b_hash);
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper NODE_RLP[3] on the branch row (first byte of slot 0's
        // hash, originally part of leaf_a_hash). The 258-tuple
        // includes NODE_RLP, so the multiset diverges from the gadget
        // side's (honest) RLP_BYTE[3].
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::NODE_RLP_OFFSET + 3].evaluations[0] =
            mpt_trace.columns[mpt_col::NODE_RLP_OFFSET + 3].evaluations[0]
                .add(&one);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: leaf_a_hash,
            child_hash_1: leaf_b_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = BranchRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_branch_2child_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered MPT NODE_RLP byte"
        );
    }

    /// Phase 8b end-to-end soundness regression: corrupt slot-0 child
    /// hash on BOTH MPT side and gadget side (and correspondingly the
    /// 32 RLP_BYTE positions where it lives) so Phase 8a's cross-AIR
    /// LogUp tuple still matches AND the gadget's row-local
    /// `child_hash_0_binding` is still satisfied. Leave row 0's
    /// `NODE_HASH` and row 1 untouched so the parent-chain holds.
    /// The Phase 8b shifted constraint is the ONLY thing that catches
    /// this: `selected_child_hash` (fake) ≠ row 1's `NODE_HASH` (real).
    /// Mirror of Phase 4b's tampering regression
    /// (`joint_prove_mpt_extension_rlp_phase4b_rejects_tampered_value_byte`).
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_branch_2child_rlp_phase8b_rejects_tampered_child_0() {
        use crate::cross_air_logup::joint_prove;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Same 2-leaf trie as the honest test.
        let key_a = vec![0x05u8, 0xab];
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
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(leaf_a_hash);
        children[1] = Some(leaf_b_hash);
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper slot-0 child hash to a FAKE value on both the
        // MPT-side BRANCH_CHILD_0_HASH_BYTE columns AND the
        // corresponding NODE_RLP[3..35] bytes. We leave NODE_HASH on
        // row 0 unchanged (the parent chain `parent_hash[1] = node_hash[0]`
        // stays satisfied), and leave row 1 unchanged (the real leaf).
        let fake_hash: [u8; 32] = [0xdeu8; 32];
        for i in 0..32 {
            let fake_byte = Scalar::from_u64(fake_hash[i] as u64, curve);
            mpt_trace.columns[mpt_col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + i]
                .evaluations[0] = fake_byte.clone();
            // RLP slot-0 child bytes live at NODE_RLP[3..35].
            mpt_trace.columns[mpt_col::NODE_RLP_OFFSET + 3 + i].evaluations[0] =
                fake_byte;
        }

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // Gadget side: build an invocation with the SAME fake child_hash_0
        // (and the real leaf_b_hash for child_hash_1, since slot 1 isn't
        // on the inclusion path). Cross-AIR LogUp tuple now agrees on
        // the fake bytes.
        let gadget_w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: fake_hash,
            child_hash_1: leaf_b_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = BranchRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_branch_2child_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        // Tampered witness must be rejected. Cross-AIR LogUp matches
        // (both sides have same fake bytes). Gadget row-locals
        // satisfied. MPT shifted body 0 (parent chain) satisfied
        // (NODE_HASH on row 0 unchanged). MPT shifted body 2 (Phase 8b
        // chain binding) is the gate: PATH_NIBBLE = 0 on row 0 →
        // selected = BRANCH_CHILD_0 = fake; NODE_HASH(ω·z) = leaf_a's
        // real hash → fake ≠ real → body 2 non-zero on row 0.
        //
        // Two failure modes (mirrors Phase 4b's tampering pattern):
        //   - joint_prove returns Err during quotient construction
        //     when C(X) is detectably not divisible by Z_H(X), OR
        //   - joint_prove succeeds with an ill-formed quotient and
        //     joint_verify rejects.
        use crate::cross_air_logup::joint_verify;
        match joint_prove(&traces, &linkages, &scheme) {
            Err(_) => {
                // Phase 8b soundness gate fired during quotient
                // construction.
            }
            Ok((proofs, ext_proof)) => {
                let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
                    vec![&mpt_cs, &gadget_cs];
                let valid = joint_verify(
                    &proofs, &cs_refs, &linkages, &ext_proof, &scheme, curve,
                );
                assert!(
                    !valid,
                    "joint verifier must reject Phase 8b-tampered \
                     child_0_hash (selected fake child ≠ real next NODE_HASH)"
                );
            }
        }
    }

    /// Phase 5c integration test: 3-AIR `joint_prove` with the MPT
    /// AIR participating in TWO simultaneous cross-AIR LogUp
    /// linkages — first to KeccakExtract (existing #91 hash chain)
    /// and second to the Phase 8a branch gadget (Phase 8b's
    /// 322-element decoded-fields tuple).
    ///
    /// Mirror of Phase 5 (leaf 3-AIR) and Phase 5b (extension 3-AIR)
    /// but for the branch shape. Validates:
    ///   - Phase 8b's NEW MPT shifted constraint (body 2) coexists
    ///     cleanly with multi-AIR joint_prove.
    ///   - KeccakExtract linkage hashes BOTH rows (branch + leaf)
    ///     while Phase 8a gadget linkage fires only on the branch
    ///     row (gated by IS_PHASE8_BRANCH_01_SHAPE).
    ///   - The MPT AIR is correctly the A-side of two linkages.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify; \
                run with --release --ignored"]
    fn joint_prove_mpt_branch_2child_with_keccak_extract_chain() {
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

        let key_a = vec![0x05u8, 0xab];
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
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(leaf_a_hash);
        children[1] = Some(leaf_b_hash);
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp.clone()];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].is_phase8_branch_01_shape, 1);

        // ── MPT side (layer 0) ────────────────────────────────────
        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // ── Phase 8a branch gadget (layer 1) ──────────────────────
        let gadget_w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: leaf_a_hash,
            child_hash_1: leaf_b_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = BranchRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        // ── KeccakExtract side (layer 2): hashes BOTH MPT rows. ──
        let extract_w = KeccakExtractWitness::from_inputs(&[
            mpt_node_rlp(&branch),
            leaf_a_rlp,
        ])
        .expect("rlp outputs fit");
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

        // ── L2: MPT ↔ Phase 8a branch gadget (322-tuple) ───────
        let l2_mpt_branch = make_mpt_branch_2child_rlp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&mpt_trace, &mpt_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1_mpt_keccak, l2_mpt_branch];

        let (proofs, ext_proof) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for honest 3-AIR with branch gadget",
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
            "joint verifier must accept honest 3-AIR proof with branch gadget"
        );
    }

    /// Phase 5c 3-AIR tampering regression: corrupt MPT-side
    /// `NODE_HASH[3]` on row 0 (the branch row). `NODE_HASH` is in
    /// L1's tuple but NOT in L2's — L1 (MPT ↔ KeccakExtract) catches
    /// the tampering independently of L2 (MPT ↔ branch gadget).
    /// joint_prove rejects with multiset mismatch. Mirror of Phase
    /// 5 / 5b tampering tests.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_branch_3air_rejects_tampered_node_hash() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness, MAX_INPUT_LEN, OUTPUT_LEN,
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

        let key_a = vec![0x05u8, 0xab];
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
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(leaf_a_hash);
        children[1] = Some(leaf_b_hash);
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp.clone()];
        let rows = inclusion_witness(&key_a, &proof);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper NODE_HASH[3] on the branch row (row 0). L1's tuple
        // contains NODE_HASH; KeccakExtract's OUTPUT stays honest →
        // multiset diverges → joint_prove rejects.
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::NODE_HASH_OFFSET + 3].evaluations[0] =
            mpt_trace.columns[mpt_col::NODE_HASH_OFFSET + 3].evaluations[0]
                .add(&one);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: leaf_a_hash,
            child_hash_1: leaf_b_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = BranchRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let extract_w = KeccakExtractWitness::from_inputs(&[
            mpt_node_rlp(&branch),
            leaf_a_rlp,
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
        let l2 = make_mpt_branch_2child_rlp_linkage_descriptor(0, 1);

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

    /// Phase 5c L2-targeting tampering regression: corrupt MPT-side
    /// `BRANCH_CHILD_0_HASH_BYTE[3]`. This column is in L2's
    /// (MPT ↔ branch gadget) 322-tuple but NOT in L1's (MPT ↔
    /// KeccakExtract) tuple, so L2 catches the tampering
    /// independently of L1. Complement to
    /// `joint_prove_mpt_branch_3air_rejects_tampered_node_hash`
    /// (which targets L1). Together, both validate that L1 and L2
    /// are independent soundness gates in the 3-AIR setup.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_branch_3air_rejects_tampered_l2_branch_child() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness, MAX_INPUT_LEN, OUTPUT_LEN,
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

        let key_a = vec![0x05u8, 0xab];
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
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(leaf_a_hash);
        children[1] = Some(leaf_b_hash);
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp.clone()];
        let rows = inclusion_witness(&key_a, &proof);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper BRANCH_CHILD_0_HASH_BYTE[3] on the branch row. This
        // column is in L2's 322-tuple but NOT in L1's hash-chain
        // tuple. NODE_RLP stays honest (so L1 multiset still
        // matches KeccakExtract). The L2 multiset diverges because
        // the gadget side keeps the honest CHILD_HASH_0_BYTE.
        let one = Scalar::one(curve);
        mpt_trace.columns[mpt_col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + 3]
            .evaluations[0] = mpt_trace.columns
            [mpt_col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + 3]
            .evaluations[0]
            .add(&one);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        let gadget_w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: leaf_a_hash,
            child_hash_1: leaf_b_hash,
        }]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = BranchRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let extract_w = KeccakExtractWitness::from_inputs(&[
            mpt_node_rlp(&branch),
            leaf_a_rlp,
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
        let l2 = make_mpt_branch_2child_rlp_linkage_descriptor(0, 1);

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
            "joint_prove must reject tampered BRANCH_CHILD_0_HASH_BYTE \
             (caught by L2)"
        );
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn mpt_branch_2child_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = BranchRlpWitness::from_branches(&[BranchRow {
            child_hash_0: sample_hash(7),
            child_hash_1: sample_hash(13),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = BranchRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "branch gadget proof must verify");
    }
}
