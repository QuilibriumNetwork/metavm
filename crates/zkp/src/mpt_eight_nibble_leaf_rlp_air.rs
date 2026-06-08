//! MPT 8-nibble even-length-path leaf gadget (Phase 11 of #92).
//!
//! Companion to the Phase 1 (64-nibble), Phase 3 (4-nibble), Phase 6
//! (3-nibble odd), and Phase 7 (6-nibble) leaf gadgets. Same structural
//! pattern as Phase 7 but with 8 path nibbles → 4 path bytes → 40-byte
//! total RLP. Covers a common residual leaf shape in real Ethereum
//! storage proofs at intermediate depths.
//!
//! # Canonical 40-byte RLP layout
//!
//! ```text
//!   [0]      0xe7   short-list header (= 0xc0 + 39)
//!   [1]      0x85   HP-string header  (= 0x80 + 5)
//!   [2]      0x20   HP prefix (leaf-flag + even-length, no leading nibble)
//!   [3..7]   ─      4 path bytes (8 packed nibbles)
//!   [7]      0xa0   value-string header (= 0x80 + 32)
//!   [8..40]  ─      32 value bytes
//!   [40..256]       zero-pad to MAX_RLP_LEN
//! ```

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

pub const NUM_PATH_NIBBLES: usize = 8;
pub const NUM_PATH_BYTES: usize = NUM_PATH_NIBBLES / 2; // 4
pub const NUM_VALUE_BYTES: usize = 32;
pub const KEY_PATH_BYTE_WIDTH: usize = 32;

pub const HP_LEN: usize = 1 + NUM_PATH_BYTES; // 5
pub const HP_STRING_HEADER: u64 = 0x80 + HP_LEN as u64; // 0x85
pub const HP_PREFIX_BYTE: u64 = 0x20;
pub const VALUE_STRING_HEADER: u64 = 0x80 + NUM_VALUE_BYTES as u64; // 0xa0
pub const RLP_PAYLOAD_LEN: u64 = (1 + HP_LEN + 1 + NUM_VALUE_BYTES) as u64; // 39
pub const RLP_LIST_HEADER: u64 = 0xc0 + RLP_PAYLOAD_LEN; // 0xe7
pub const RLP_OUTPUT_LEN: usize = 1 + RLP_PAYLOAD_LEN as usize; // 40
pub const NODE_KIND_LEAF: u64 = 2;
pub const RLP_BYTE_WIDTH: usize = crate::mpt_air::MAX_RLP_LEN;

const RLP_OFFSET_HP_STR_HEADER: usize = 1;
const RLP_OFFSET_HP_PREFIX: usize = 2;
const RLP_OFFSET_PATH_BYTES: usize = 3;
const RLP_OFFSET_VALUE_HEADER: usize = RLP_OFFSET_PATH_BYTES + NUM_PATH_BYTES; // 7
const RLP_OFFSET_VALUE_BYTES: usize = RLP_OFFSET_VALUE_HEADER + 1; // 8

const _: () = assert!(RLP_OFFSET_VALUE_BYTES + NUM_VALUE_BYTES == RLP_OUTPUT_LEN);

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_RLP_BYTE_OFFSET: usize = 0;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_NODE_KIND: usize = COL_RLP_LEN + 1;
pub const COL_KEY_PATH_BYTE_OFFSET: usize = COL_NODE_KIND + 1;
pub const COL_VALUE_BYTE_OFFSET: usize = COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH;
pub const COL_IS_REAL: usize = COL_VALUE_BYTE_OFFSET + NUM_VALUE_BYTES;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;
pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafRow {
    pub path_bytes: [u8; NUM_PATH_BYTES],
    pub value: [u8; NUM_VALUE_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct LeafRlpWitness {
    pub leaves: Vec<LeafRow>,
}

impl LeafRlpWitness {
    pub fn from_leaves(leaves: &[LeafRow]) -> Self {
        Self { leaves: leaves.to_vec() }
    }
}

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

pub struct LeafRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl LeafRlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for LeafRlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

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
            bodies[1][row] = eval_byte_pinned(&row_evals, COL_NODE_KIND, NODE_KIND_LEAF, COL_IS_REAL);
            bodies[2][row] = eval_byte_pinned(&row_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL);
            bodies[3][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL);
            bodies[4][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_STR_HEADER, HP_STRING_HEADER, COL_IS_REAL);
            bodies[5][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX, HP_PREFIX_BYTE, COL_IS_REAL);
            bodies[6][row] = eval_byte_pinned(&row_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER, VALUE_STRING_HEADER, COL_IS_REAL);
            bodies[7][row] = eval_byte_slice_binding(&row_evals, &alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_PATH_BYTES, COL_KEY_PATH_BYTE_OFFSET, NUM_PATH_BYTES, COL_IS_REAL);
            bodies[8][row] = eval_decoded_zero_tail(&row_evals, &alpha, COL_KEY_PATH_BYTE_OFFSET + NUM_PATH_BYTES, COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH, COL_IS_REAL);
            bodies[9][row] = eval_byte_slice_binding(&row_evals, &alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_VALUE_BYTES, COL_VALUE_BYTE_OFFSET, NUM_VALUE_BYTES, COL_IS_REAL);
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
            eval_byte_pinned(col_evals, COL_NODE_KIND, NODE_KIND_LEAF, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_LEN, RLP_OUTPUT_LEN as u64, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET, RLP_LIST_HEADER, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_STR_HEADER, HP_STRING_HEADER, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX, HP_PREFIX_BYTE, COL_IS_REAL),
            eval_byte_pinned(col_evals, COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER, VALUE_STRING_HEADER, COL_IS_REAL),
            eval_byte_slice_binding(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_PATH_BYTES, COL_KEY_PATH_BYTE_OFFSET, NUM_PATH_BYTES, COL_IS_REAL),
            eval_decoded_zero_tail(col_evals, alpha, COL_KEY_PATH_BYTE_OFFSET + NUM_PATH_BYTES, COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH, COL_IS_REAL),
            eval_byte_slice_binding(col_evals, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_VALUE_BYTES, COL_VALUE_BYTE_OFFSET, NUM_VALUE_BYTES, COL_IS_REAL),
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
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_STR_HEADER, HP_STRING_HEADER, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_PREFIX, HP_PREFIX_BYTE, COL_IS_REAL, curve),
            build_byte_pinned_poly(col_coeffs, COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER, VALUE_STRING_HEADER, COL_IS_REAL, curve),
            build_byte_slice_binding_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_PATH_BYTES, COL_KEY_PATH_BYTE_OFFSET, NUM_PATH_BYTES, COL_IS_REAL, curve),
            build_decoded_zero_tail_poly(col_coeffs, alpha, COL_KEY_PATH_BYTE_OFFSET + NUM_PATH_BYTES, COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH, COL_IS_REAL, curve),
            build_byte_slice_binding_poly(col_coeffs, alpha, COL_RLP_BYTE_OFFSET, RLP_OFFSET_VALUE_BYTES, COL_VALUE_BYTE_OFFSET, NUM_VALUE_BYTES, COL_IS_REAL, curve),
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

pub fn make_mpt_eight_nibble_leaf_rlp_linkage_descriptor(
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
    let mut b_columns: Vec<usize> = (0..RLP_BYTE_WIDTH).map(|b| COL_RLP_BYTE_OFFSET + b).collect();
    b_columns.push(COL_RLP_LEN);
    b_columns.push(COL_NODE_KIND);
    for b in 0..KEY_PATH_BYTE_WIDTH {
        b_columns.push(COL_KEY_PATH_BYTE_OFFSET + b);
    }
    for b in 0..NUM_VALUE_BYTES {
        b_columns.push(COL_VALUE_BYTE_OFFSET + b);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_eight_nibble_leaf_rlp_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_PHASE11_LEAF_8N_SHAPE),
        b_layer_index: rlp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::{mpt_node_rlp, MptNode, Nibbles};

    fn sample_path() -> [u8; NUM_PATH_BYTES] { [0x12, 0xab, 0x34, 0xcd] }
    fn sample_value() -> [u8; NUM_VALUE_BYTES] {
        let mut v = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            v[i] = (i as u8).wrapping_mul(11).wrapping_add(17);
        }
        v
    }

    /// Critical correctness oracle: gadget's `canonical_rlp` matches
    /// `mpt::mpt_node_rlp` on the equivalent `MptNode::Leaf`.
    #[test]
    fn canonical_rlp_matches_reference_encoder() {
        let path = sample_path();
        let value = sample_value();
        let ours = canonical_rlp(&path, &value);
        let mut nibbles = Vec::with_capacity(NUM_PATH_NIBBLES);
        for &b in &path {
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
    fn canonical_rlp_structural_bytes() {
        let path = sample_path();
        let value = sample_value();
        let rlp = canonical_rlp(&path, &value);
        assert_eq!(rlp[0] as u64, RLP_LIST_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_STR_HEADER] as u64, HP_STRING_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_PREFIX] as u64, HP_PREFIX_BYTE);
        assert_eq!(rlp[RLP_OFFSET_VALUE_HEADER] as u64, VALUE_STRING_HEADER);
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = LeafRlpWitness::from_leaves(&[LeafRow {
            path_bytes: sample_path(),
            value: sample_value(),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(13, curve);
        for row in 0..trace.padded_size as usize {
            let col_vals: Vec<Scalar> = trace
                .columns.iter().map(|p| p.evaluations[row].clone()).collect();
            assert!(cs.evaluate_at_point(&col_vals, &alpha).is_zero());
        }
    }

    #[test]
    fn descriptor_well_formed() {
        use crate::mpt_air::col as mpt_col;
        let desc = make_mpt_eight_nibble_leaf_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "mpt_eight_nibble_leaf_rlp_v1");
        let expected = RLP_BYTE_WIDTH + 1 + 1 + KEY_PATH_BYTE_WIDTH + NUM_VALUE_BYTES;
        assert_eq!(desc.a_columns.len(), expected);
        assert_eq!(desc.b_columns.len(), expected);
        assert_eq!(desc.a_selector_column, Some(mpt_col::IS_PHASE11_LEAF_8N_SHAPE));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    /// Real 4-byte-key single-leaf trie produces the Phase 11 shape via
    /// natural `inclusion_witness`.
    #[test]
    fn mpt_phase11_selector_fires_on_4_byte_key_single_leaf() {
        use crate::mpt::single_leaf_trie;
        use crate::mpt_air::inclusion_witness;
        let key = vec![0x12, 0xab, 0x34, 0xcd];
        let value = vec![0x77u8; NUM_VALUE_BYTES];
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].is_phase11_leaf_8n_shape, 1);
        assert_eq!(rows[0].is_phase1_leaf_shape, 0);
        assert_eq!(rows[0].is_phase3_leaf_shape, 0);
        assert_eq!(rows[0].is_phase6_odd_leaf_shape, 0);
        assert_eq!(rows[0].is_phase7_leaf_6n_shape, 0);
        assert_eq!(rows[0].key_path_bytes[..4], key[..]);
        for k in 4..32 {
            assert_eq!(rows[0].key_path_bytes[k], 0);
        }
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn mpt_eight_nibble_leaf_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = LeafRlpWitness::from_leaves(&[LeafRow {
            path_bytes: sample_path(),
            value: sample_value(),
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = LeafRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid);
    }
}
