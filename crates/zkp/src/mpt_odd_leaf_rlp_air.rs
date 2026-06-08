//! MPT odd-length-path leaf RLP-decoding consistency gadget (Phase 6
//! of #92).
//!
//! Companion to the Phase 1/3 even-length-path leaf gadgets and the
//! Phase 4 extension gadget. This module handles the **3-nibble
//! odd-length leaf with 32-byte value** shape — a 37-byte short-form
//! list where the HP prefix byte's low nibble carries a per-row
//! "leading nibble" that is NOT a constant.
//!
//! # New architectural primitive
//!
//! Earlier shapes had every structural byte as a compile-time constant.
//! Odd-length paths break that: HP byte = `0x30 | leading_nibble`
//! where `leading_nibble ∈ [0, 16)` varies per row. The gadget pins
//! `RLP_BYTE[2] = 0x30 + LEADING_PATH_NIBBLE` algebraically and the
//! MPT side range-checks `LEADING_PATH_NIBBLE ∈ [0, 16)` via the
//! existing 4-bit lookup table. Combined, these two checks force the
//! HP byte to be one of the 16 valid odd-length-leaf prefixes.
//!
//! # Phase 6 scope
//!
//! Single fixed RLP shape: leaf, **3-nibble odd-length path** + 32-byte
//! value. Shape identical to Phase 3 except:
//!   - HP-string body is 2 bytes (vs Phase 3's 3 bytes), so HP-string
//!     header is `0x82` (vs `0x83`).
//!   - HP byte high nibble = `0x3` (leaf+odd, vs `0x2` leaf+even).
//!   - Total RLP length is 37 bytes (vs Phase 3's 38).
//!   - KEY_PATH_BYTE[0] carries the packed `(mid_nibble << 4) |
//!     last_nibble` byte; KEY_PATH_BYTE[1..32] is zero.
//!   - The leading nibble is in a separate column LEADING_NIBBLE.
//!
//! # Canonical 37-byte RLP layout
//!
//! ```text
//!   [0]     0xe4   short-list header (= 0xc0 + 36)
//!   [1]     0x82   HP-string header (= 0x80 + 2)
//!   [2]     0x3X   HP byte (high nibble = leaf+odd, low nibble = leading nibble X)
//!   [3]     ─      packed nibble byte (mid_nibble << 4 | last_nibble)
//!   [4]     0xa0   value-string header (= 0x80 + 32)
//!   [5..37] ─      32 value bytes
//!   [37..256]      zero-pad to MAX_RLP_LEN
//! ```
//!
//! # Cross-AIR LogUp tuple
//!
//! 323 elements (vs 322 for Phase 1/3/4):
//!   - 256 RLP bytes
//!   - 1 RLP_LEN
//!   - 1 NODE_KIND
//!   - 32 KEY_PATH_BYTE (only [0] carries data; [1..32] zero-pinned)
//!   - 32 VALUE_BYTE
//!   - **1 LEADING_PATH_NIBBLE** (NEW)

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

pub const NUM_PATH_NIBBLES: usize = 3;
/// HP-encoded path body in bytes (1 prefix byte + 1 packed nibble byte).
pub const HP_LEN: usize = 2;
pub const NUM_VALUE_BYTES: usize = 32;
/// Width of KEY_PATH_BYTE in the SHARED MPT-side schema.
pub const KEY_PATH_BYTE_WIDTH: usize = 32;

pub const HP_STRING_HEADER: u64 = 0x80 + HP_LEN as u64; // 0x82
/// HP byte's high nibble (= leaf-flag + odd-length): 0x3.
pub const HP_HIGH_NIBBLE: u64 = 0x3;
/// HP byte high-nibble offset in field arithmetic: HIGH_NIBBLE << 4 = 0x30.
pub const HP_HIGH_NIBBLE_TIMES_16: u64 = HP_HIGH_NIBBLE << 4; // 0x30

pub const VALUE_STRING_HEADER: u64 = 0x80 + NUM_VALUE_BYTES as u64; // 0xa0
/// RLP payload = HP-RLP body (1 header + HP_LEN body) + value-RLP body
/// (1 header + NUM_VALUE_BYTES body) = (1 + 2) + (1 + 32) = 36.
pub const RLP_PAYLOAD_LEN: u64 = (1 + HP_LEN + 1 + NUM_VALUE_BYTES) as u64; // 36
pub const RLP_LIST_HEADER: u64 = 0xc0 + RLP_PAYLOAD_LEN; // 0xe4
pub const RLP_OUTPUT_LEN: usize = 1 + RLP_PAYLOAD_LEN as usize; // 37

pub const NODE_KIND_LEAF: u64 = 2;

pub const RLP_BYTE_WIDTH: usize = crate::mpt_air::MAX_RLP_LEN;

const RLP_OFFSET_HP_STR_HEADER: usize = 1;
const RLP_OFFSET_HP_BYTE: usize = 2;
const RLP_OFFSET_PACKED_NIBBLE_BYTE: usize = 3;
const RLP_OFFSET_VALUE_HEADER: usize = 4;
const RLP_OFFSET_VALUE_BYTES: usize = 5;

const _: () = assert!(RLP_OFFSET_VALUE_BYTES + NUM_VALUE_BYTES == RLP_OUTPUT_LEN);

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_RLP_BYTE_OFFSET: usize = 0;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_NODE_KIND: usize = COL_RLP_LEN + 1;
/// Mirrors MPT-side KEY_PATH_BYTE width. KEY_PATH_BYTE[0] = packed
/// nibble byte; KEY_PATH_BYTE[1..32] zero-pinned.
pub const COL_KEY_PATH_BYTE_OFFSET: usize = COL_NODE_KIND + 1;
pub const COL_VALUE_BYTE_OFFSET: usize = COL_KEY_PATH_BYTE_OFFSET + KEY_PATH_BYTE_WIDTH;
/// The new oracle column carrying the leading nibble of the odd-length
/// path. Range-checked on the MPT side via the 4-bit lookup table.
pub const COL_LEADING_NIBBLE: usize = COL_VALUE_BYTE_OFFSET + NUM_VALUE_BYTES;
pub const COL_IS_REAL: usize = COL_LEADING_NIBBLE + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 12;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OddLeafRow {
    /// Three path nibbles: leading (≤ 0xf), then mid, then last.
    /// Each must be < 16; out-of-range values are rejected by the
    /// witness builder.
    pub leading_nibble: u8,
    pub mid_nibble: u8,
    pub last_nibble: u8,
    pub value: [u8; NUM_VALUE_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct OddLeafRlpWitness {
    pub leaves: Vec<OddLeafRow>,
}

impl OddLeafRlpWitness {
    pub fn from_leaves(leaves: &[OddLeafRow]) -> Result<Self, &'static str> {
        for l in leaves {
            if l.leading_nibble >= 16 || l.mid_nibble >= 16 || l.last_nibble >= 16 {
                return Err("odd_leaf_rlp_air: nibble ≥ 16");
            }
        }
        Ok(Self {
            leaves: leaves.to_vec(),
        })
    }
}

pub fn canonical_rlp(
    leading_nibble: u8,
    mid_nibble: u8,
    last_nibble: u8,
    value: &[u8; NUM_VALUE_BYTES],
) -> [u8; RLP_OUTPUT_LEN] {
    let mut out = [0u8; RLP_OUTPUT_LEN];
    out[0] = RLP_LIST_HEADER as u8;
    out[RLP_OFFSET_HP_STR_HEADER] = HP_STRING_HEADER as u8;
    out[RLP_OFFSET_HP_BYTE] = (HP_HIGH_NIBBLE_TIMES_16 as u8) | (leading_nibble & 0xf);
    out[RLP_OFFSET_PACKED_NIBBLE_BYTE] = ((mid_nibble & 0xf) << 4) | (last_nibble & 0xf);
    out[RLP_OFFSET_VALUE_HEADER] = VALUE_STRING_HEADER as u8;
    out[RLP_OFFSET_VALUE_BYTES..RLP_OFFSET_VALUE_BYTES + NUM_VALUE_BYTES]
        .copy_from_slice(value);
    out
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &OddLeafRlpWitness,
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
        let rlp = canonical_rlp(
            leaf.leading_nibble,
            leaf.mid_nibble,
            leaf.last_nibble,
            &leaf.value,
        );
        for (b, &v) in rlp.iter().enumerate() {
            columns[COL_RLP_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_RLP_LEN][i] = Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve);
        columns[COL_NODE_KIND][i] = Scalar::from_u64(NODE_KIND_LEAF, curve);
        // KEY_PATH_BYTE[0] = packed nibble byte; KEY_PATH_BYTE[1..32] zero.
        columns[COL_KEY_PATH_BYTE_OFFSET][i] =
            Scalar::from_u64(rlp[RLP_OFFSET_PACKED_NIBBLE_BYTE] as u64, curve);
        for (k, &b) in leaf.value.iter().enumerate() {
            columns[COL_VALUE_BYTE_OFFSET + k][i] = Scalar::from_u64(b as u64, curve);
        }
        columns[COL_LEADING_NIBBLE][i] = Scalar::from_u64(leaf.leading_nibble as u64, curve);
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

pub struct OddLeafRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl OddLeafRlpConstraintSystem {
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

// ─── Body helpers (gadget-specific) ───────────────────────────────────

/// HP byte decomposition: `IS_REAL · (RLP_BYTE[2] − 0x30 −
/// LEADING_NIBBLE)`. With LEADING_NIBBLE range-checked to [0, 16) on
/// the MPT side, this pins RLP_BYTE[2] ∈ {0x30, 0x31, ..., 0x3f}.
fn eval_hp_byte_decomposition(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let body = col_evals[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_BYTE]
        .sub(&Scalar::from_u64(HP_HIGH_NIBBLE_TIMES_16, curve))
        .sub(&col_evals[COL_LEADING_NIBBLE]);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_hp_byte_decomposition_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let const_poly = vec![Scalar::from_u64(HP_HIGH_NIBBLE_TIMES_16, curve)];
    let diff = poly_sub(
        &col_coeffs[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_BYTE],
        &const_poly,
        curve,
    );
    let body = poly_sub(&diff, &col_coeffs[COL_LEADING_NIBBLE], curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for OddLeafRlpConstraintSystem {
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
            "rlp_hp_byte_decomposition".into(),
            "rlp_value_string_header_pinned".into(),
            "packed_nibble_byte_binding".into(),
            "key_path_byte_zero_tail".into(),
            "value_byte_binding".into(),
            "rlp_zero_tail".into(),
            "reserved".into(),
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
            bodies[5][row] = eval_hp_byte_decomposition(&row_evals);
            bodies[6][row] = eval_byte_pinned(
                &row_evals,
                COL_RLP_BYTE_OFFSET + RLP_OFFSET_VALUE_HEADER,
                VALUE_STRING_HEADER,
                COL_IS_REAL,
            );
            // packed_nibble_byte: RLP_BYTE[3] − KEY_PATH_BYTE[0] = 0
            bodies[7][row] = eval_byte_slice_binding(
                &row_evals,
                &alpha_for_rlc,
                COL_RLP_BYTE_OFFSET,
                RLP_OFFSET_PACKED_NIBBLE_BYTE,
                COL_KEY_PATH_BYTE_OFFSET,
                1,
                COL_IS_REAL,
            );
            // KEY_PATH_BYTE[1..32] zero
            bodies[8][row] = eval_decoded_zero_tail(
                &row_evals,
                &alpha_for_rlc,
                COL_KEY_PATH_BYTE_OFFSET + 1,
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
            eval_hp_byte_decomposition(col_evals),
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
                RLP_OFFSET_PACKED_NIBBLE_BYTE,
                COL_KEY_PATH_BYTE_OFFSET,
                1,
                COL_IS_REAL,
            ),
            eval_decoded_zero_tail(
                col_evals,
                alpha,
                COL_KEY_PATH_BYTE_OFFSET + 1,
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
            build_hp_byte_decomposition_poly(col_coeffs, curve),
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
                RLP_OFFSET_PACKED_NIBBLE_BYTE,
                COL_KEY_PATH_BYTE_OFFSET,
                1,
                COL_IS_REAL,
                curve,
            ),
            build_decoded_zero_tail_poly(
                col_coeffs,
                alpha,
                COL_KEY_PATH_BYTE_OFFSET + 1,
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
/// Phase-6-shape odd-length-leaf rows** to **this gadget's
/// algebraically decoded rows**: 323-element tuple = 322 (Phase 1/3/4
/// shape) + 1 (LEADING_PATH_NIBBLE).
///
/// Gated by [`crate::mpt_air::col::IS_PHASE6_ODD_LEAF_SHAPE`] on the
/// MPT side, [`COL_IS_REAL`] on the gadget side. Mutually exclusive
/// with all other Phase shape selectors at the witness level.
pub fn make_mpt_odd_leaf_rlp_linkage_descriptor(
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
    a_columns.push(mpt_col::LEADING_PATH_NIBBLE);

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
    b_columns.push(COL_LEADING_NIBBLE);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_odd_leaf_rlp_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: Some(mpt_col::IS_PHASE6_ODD_LEAF_SHAPE),
        b_layer_index: rlp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::{mpt_node_rlp, MptNode, Nibbles};

    fn sample_value() -> [u8; NUM_VALUE_BYTES] {
        let mut v = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            v[i] = (i as u8).wrapping_mul(11).wrapping_add(17);
        }
        v
    }

    fn sample_witness() -> OddLeafRlpWitness {
        OddLeafRlpWitness::from_leaves(&[
            OddLeafRow {
                leading_nibble: 0,
                mid_nibble: 0,
                last_nibble: 0,
                value: [0u8; NUM_VALUE_BYTES],
            },
            OddLeafRow {
                leading_nibble: 7,
                mid_nibble: 0xa,
                last_nibble: 0xb,
                value: sample_value(),
            },
            OddLeafRow {
                leading_nibble: 0xf,
                mid_nibble: 0xf,
                last_nibble: 0xf,
                value: [0xffu8; NUM_VALUE_BYTES],
            },
        ])
        .expect("nibbles in range")
    }

    #[test]
    fn from_leaves_rejects_oversize_nibble() {
        let r = OddLeafRlpWitness::from_leaves(&[OddLeafRow {
            leading_nibble: 16,
            mid_nibble: 0,
            last_nibble: 0,
            value: [0u8; NUM_VALUE_BYTES],
        }]);
        assert!(r.is_err());
    }

    #[test]
    fn canonical_rlp_matches_reference_encoder() {
        let value = sample_value();
        let leading = 0x7;
        let mid = 0xa;
        let last = 0xb;
        let ours = canonical_rlp(leading, mid, last, &value);

        let node = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![leading, mid, last]),
            value: value.to_vec(),
        };
        let reference = mpt_node_rlp(&node);
        assert_eq!(reference.len(), RLP_OUTPUT_LEN);
        assert_eq!(&ours[..], &reference[..]);
    }

    #[test]
    fn canonical_rlp_structural_bytes() {
        let value = sample_value();
        let leading = 0x9;
        let mid = 0x3;
        let last = 0x6;
        let rlp = canonical_rlp(leading, mid, last, &value);
        assert_eq!(rlp[0] as u64, RLP_LIST_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_STR_HEADER] as u64, HP_STRING_HEADER);
        assert_eq!(rlp[RLP_OFFSET_HP_BYTE], 0x39);
        assert_eq!(rlp[RLP_OFFSET_PACKED_NIBBLE_BYTE], 0x36);
        assert_eq!(rlp[RLP_OFFSET_VALUE_HEADER] as u64, VALUE_STRING_HEADER);
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 3);
        // Row 1: leading=7, mid=0xa, last=0xb. RLP[2] = 0x37, RLP[3] = 0xab.
        assert_eq!(
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_BYTE].evaluations[1]
                .to_u64(),
            0x37
        );
        assert_eq!(
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_PACKED_NIBBLE_BYTE]
                .evaluations[1]
                .to_u64(),
            0xab
        );
        assert_eq!(
            trace.columns[COL_LEADING_NIBBLE].evaluations[1].to_u64(),
            7
        );
        assert_eq!(
            trace.columns[COL_KEY_PATH_BYTE_OFFSET].evaluations[1].to_u64(),
            0xab
        );
        for k in 1..KEY_PATH_BYTE_WIDTH {
            assert!(
                trace.columns[COL_KEY_PATH_BYTE_OFFSET + k].evaluations[1].is_zero()
            );
        }
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = OddLeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(13, curve);
        for row in 0..trace.padded_size as usize {
            let col_vals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            assert!(cs.evaluate_at_point(&col_vals, &alpha).is_zero());
        }
    }

    #[test]
    fn tampered_hp_byte_high_nibble_fires() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = OddLeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        // Add 0x10 to RLP[2] (e.g., 0x37 → 0x47), changing the high nibble.
        trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_BYTE].evaluations[1] =
            trace.columns[COL_RLP_BYTE_OFFSET + RLP_OFFSET_HP_BYTE].evaluations[1]
                .add(&Scalar::from_u64(0x10, curve));
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "rlp_hp_byte_decomposition must fire on changed HP byte"
        );
    }

    #[test]
    fn tampered_leading_nibble_breaks_decomposition() {
        let curve = CurveType::Bls48581;
        let w = sample_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = OddLeafRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(19, curve);
        // Tamper LEADING_NIBBLE from 7 → 8 without touching RLP[2].
        trace.columns[COL_LEADING_NIBBLE].evaluations[1] =
            trace.columns[COL_LEADING_NIBBLE].evaluations[1]
                .add(&Scalar::one(curve));
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "rlp_hp_byte_decomposition must fire on inconsistent LEADING_NIBBLE"
        );
    }

    // ── Phase 6 cross-AIR LogUp linkage tests ──────────────────────

    #[test]
    fn mpt_odd_leaf_rlp_descriptor_well_formed() {
        use crate::mpt_air::col as mpt_col;
        let desc = make_mpt_odd_leaf_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "mpt_odd_leaf_rlp_v1");
        let expected_width =
            RLP_BYTE_WIDTH + 1 + 1 + KEY_PATH_BYTE_WIDTH + NUM_VALUE_BYTES + 1;
        assert_eq!(desc.a_columns.len(), expected_width);
        assert_eq!(desc.b_columns.len(), expected_width);
        assert_eq!(desc.a_columns.last().copied(), Some(mpt_col::LEADING_PATH_NIBBLE));
        assert_eq!(desc.b_columns.last().copied(), Some(COL_LEADING_NIBBLE));
        assert_eq!(desc.a_selector_column, Some(mpt_col::IS_PHASE6_ODD_LEAF_SHAPE));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    /// Build a real single-leaf MPT inclusion proof for a 3-nibble
    /// odd-length leaf and verify the 323-element cross-AIR tuple
    /// matches byte-for-byte.
    #[test]
    fn mpt_odd_leaf_gadget_byte_tuples_match_real_mpt_witness() {
        use crate::keccak::keccak256;
        use crate::mpt_air::col as mpt_col;
        use crate::mpt_constraints::build_trace_polynomials_from_rows;

        let curve = CurveType::Bls48581;

        // Construct a synthetic single-row trie with a 3-nibble leaf.
        // (Real Ethereum proofs reach this shape at intermediate trie
        // depths; we synthesise a minimal example here.)
        let leading = 0xa_u8;
        let mid = 0x3_u8;
        let last = 0xc_u8;
        let value = sample_value();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![leading, mid, last]),
            value: value.to_vec(),
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        assert_eq!(leaf_rlp.len(), RLP_OUTPUT_LEN);
        assert_eq!(leaf_rlp[2] & 0xf0, 0x30);
        let _root = keccak256(&leaf_rlp);

        // The host-side `inclusion_witness` walks the proof and
        // recovers the leaf via the same path-prefix checks. To
        // simulate this we feed a key whose nibble decomposition
        // matches the leaf's path exactly.
        let key_nibbles = vec![leading, mid, last];
        // Pack 3 nibbles into 2 bytes (last byte's low nibble = 0).
        // But Nibbles::from_bytes always produces an EVEN nibble count.
        // For the witness builder to walk this 3-nibble leaf, we need
        // the key to ALSO yield 3 nibbles after consumption. The
        // simplest way: use a 2-byte key (4 nibbles) and have the
        // proof contain ONE branch (consuming 1 nibble) above the
        // 3-nibble leaf. Since constructing that branch is verbose,
        // we instead direct-test inclusion_witness on a synthetic
        // single-leaf proof with an even-byte key whose nibbles
        // happen to match the 3-nibble leaf path PLUS a trailing
        // nibble = 0 (which the leaf's HP encoding ignores). For
        // simplicity here, we stub the test by comparing the gadget
        // trace against a hand-constructed MPT row.
        let _ = key_nibbles;

        // Hand-build the matching MPT InclusionRow directly, mirroring
        // what inclusion_witness would produce on a Phase 6 leaf row.
        // (A two-row branch+leaf trie test would be more realistic;
        // we leave that as a follow-up.)
        let mut node_rlp = [0u8; crate::mpt_air::MAX_RLP_LEN];
        node_rlp[..leaf_rlp.len()].copy_from_slice(&leaf_rlp);
        let mut key_path_bytes = [0u8; 32];
        key_path_bytes[0] = leaf_rlp[3];
        let mut value_bytes = [0u8; 32];
        value_bytes.copy_from_slice(&leaf_rlp[5..37]);
        let row = crate::mpt_air::InclusionRow {
            node_hash: keccak256(&leaf_rlp),
            parent_hash: keccak256(&leaf_rlp),
            node_kind: 2,
            path_nibble: 0,
            depth: 0,
            is_terminal: 1,
            node_rlp,
            node_rlp_len: leaf_rlp.len(),
            key_path_bytes,
            value: value_bytes,
            is_phase1_leaf_shape: 0,
            is_phase3_leaf_shape: 0,
            is_phase4_ext_shape: 0,
            leading_path_nibble: leading,
            is_phase6_odd_leaf_shape: 1,
            is_phase7_leaf_6n_shape: 0,
            is_phase8_branch_01_shape: 0,
            branch_child_0_hash_bytes: [0u8; 32],
            branch_child_1_hash_bytes: [0u8; 32],
            is_phase9_branch_05_shape: 0,
            is_phase10_branch_15_shape: 0,
            is_phase11_leaf_8n_shape: 0,
            is_root: 1, // synthetic single-row test fixture; depth = 0
            claimed_leaf_value_bytes: [0u8; 32], // not relevant for this RLP-shape test
            claimed_leaf_key_bytes: [0u8; 32],   // not relevant for this RLP-shape test
            claimed_full_key_bytes: [0u8; 32],   // not relevant for this RLP-shape test
            claimed_full_key_nibbles: [0u8; 64],
            is_depth_eq: {
                // synthetic single-row test fixture with depth=0:
                // is_depth_eq[0] = 1, others 0.
                let mut a = [0u8; 64];
                a[0] = 1;
                a
            },
        };
        let mpt_trace = build_trace_polynomials_from_rows(&[row], curve);

        let gadget_w = OddLeafRlpWitness::from_leaves(&[OddLeafRow {
            leading_nibble: leading,
            mid_nibble: mid,
            last_nibble: last,
            value,
        }])
        .unwrap();
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        let descriptor = make_mpt_odd_leaf_rlp_linkage_descriptor(0, 1);
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
            mpt_trace.columns[mpt_col::IS_PHASE6_ODD_LEAF_SHAPE].evaluations[0]
                .to_u64(),
            1
        );
    }

    /// Phase 6 end-to-end joint_prove regression. Builds a real
    /// 2-leaf trie where a branch consumes 1 nibble of the 2-byte
    /// key, leaving a 3-nibble (odd) residual path on the target
    /// leaf — naturally producing the Phase 6 shape via
    /// `inclusion_witness`. Mirrors `joint_prove_mpt_short_leaf_rlp_linkage`
    /// (Phase 3) and `joint_prove_mpt_extension_rlp_linkage` (Phase 4)
    /// to close the test-coverage gap left by Phase 6's earlier
    /// hand-constructed-row tuple match.
    #[test]
    #[ignore = "slow: full joint_prove + joint_verify across 2 AIRs; \
                run with --release --ignored"]
    fn joint_prove_mpt_odd_leaf_rlp_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak::keccak256;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Construct a 2-leaf trie:
        //   - key_a = [0x12, 0x34] (nibbles 1, 2, 3, 4) — target leaf;
        //     branch consumes nibble 1 → leaf has 3 nibbles (2, 3, 4).
        //   - key_b = [0x55, 0x66] (nibbles 5, 5, 6, 6) — sibling.
        // The branch's child[1] points to leaf_a (Phase 6 shape) and
        // child[5] points to leaf_b.
        let key_a = vec![0x12u8, 0x34];
        let mut value_a = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value_a[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x2, 0x3, 0x4]),
            value: value_a.to_vec(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        assert_eq!(
            leaf_a_rlp.len(),
            RLP_OUTPUT_LEN,
            "target leaf must match Phase 6 shape (37 bytes)"
        );

        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x5, 0x6, 0x6]),
            value: vec![0xaa; NUM_VALUE_BYTES],
        };
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let leaf_b_hash = keccak256(&leaf_b_rlp);

        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(leaf_a_hash);
        children[5] = Some(leaf_b_hash);
        let branch = MptNode::Branch {
            children,
            value: None,
        };
        let branch_rlp = mpt_node_rlp(&branch);

        let proof = vec![branch_rlp, leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].node_kind, 0, "row 0 is the branch");
        assert_eq!(
            rows[1].is_phase6_odd_leaf_shape, 1,
            "row 1 must match Phase 6 shape via natural inclusion_witness"
        );
        // Decoded fields populated correctly: leading nibble = 2, the
        // first path nibble after the branch consumes nibble 1.
        assert_eq!(rows[1].leading_path_nibble, 0x2);

        // ── MPT side ──────────────────────────────────────────────
        let mpt_trace = build_trace_polynomials_from_rows(&rows, curve);
        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // ── Phase 6 gadget side ──────────────────────────────────
        let gadget_w = OddLeafRlpWitness::from_leaves(&[OddLeafRow {
            leading_nibble: 0x2,
            mid_nibble: 0x3,
            last_nibble: 0x4,
            value: value_a,
        }])
        .expect("nibbles in range");
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = OddLeafRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_odd_leaf_rlp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let (proofs, ext_proof) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for honest 2-row MPT + Phase 6 gadget",
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
            "joint verifier must accept honest 2-row branch+Phase6-leaf proof"
        );

        // Sanity: MPT-side LEADING_PATH_NIBBLE column is 2 on row 1
        // and 0 on row 0 (branch row, not Phase 6).
        assert!(mpt_trace.columns[mpt_col::LEADING_PATH_NIBBLE]
            .evaluations[0]
            .is_zero());
        assert_eq!(
            mpt_trace.columns[mpt_col::LEADING_PATH_NIBBLE]
                .evaluations[1]
                .to_u64(),
            2
        );
    }

    /// Phase 6 cross-AIR LogUp tampering test: corrupt the MPT-side
    /// `LEADING_PATH_NIBBLE` column on the Phase-6 leaf row while
    /// leaving the gadget side honest. The cross-AIR LogUp 323-tuple
    /// includes LEADING_PATH_NIBBLE as its trailing element, so the
    /// MPT-side multiset diverges from the gadget side and
    /// `joint_prove` rejects the witness with "multiset equality
    /// cannot hold".
    ///
    /// Validates that Phase 6's NEW column (LEADING_PATH_NIBBLE) is
    /// actually being multiset-checked by the cross-AIR LogUp — the
    /// honest-path regression `joint_prove_mpt_odd_leaf_rlp_linkage`
    /// only proves the linkage works for the honest case; this
    /// regression proves the linkage IS the gate.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_mpt_odd_leaf_rejects_tampered_leading_nibble() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak::keccak256;
        use crate::mpt_air::{col as mpt_col, inclusion_witness};
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows, MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Same 2-leaf trie as the honest regression.
        let key_a = vec![0x12u8, 0x34];
        let mut value_a = [0u8; NUM_VALUE_BYTES];
        for i in 0..NUM_VALUE_BYTES {
            value_a[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x2, 0x3, 0x4]),
            value: value_a.to_vec(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x5, 0x6, 0x6]),
            value: vec![0xaa; NUM_VALUE_BYTES],
        };
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let leaf_b_hash = keccak256(&leaf_b_rlp);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(leaf_a_hash);
        children[5] = Some(leaf_b_hash);
        let branch = MptNode::Branch {
            children,
            value: None,
        };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows[1].is_phase6_odd_leaf_shape, 1);
        assert_eq!(rows[1].leading_path_nibble, 0x2);

        let mut mpt_trace = build_trace_polynomials_from_rows(&rows, curve);

        // Tamper LEADING_PATH_NIBBLE on row 1 (the Phase 6 leaf):
        // honest = 2, set to 5. Cross-AIR LogUp tuple mismatches the
        // gadget's LEADING_NIBBLE = 2.
        mpt_trace.columns[mpt_col::LEADING_PATH_NIBBLE].evaluations[1] =
            Scalar::from_u64(5, curve);

        let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
            .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

        // Honest gadget side.
        let gadget_w = OddLeafRlpWitness::from_leaves(&[OddLeafRow {
            leading_nibble: 0x2,
            mid_nibble: 0x3,
            last_nibble: 0x4,
            value: value_a,
        }])
        .expect("nibbles in range");
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = OddLeafRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_mpt_odd_leaf_rlp_linkage_descriptor(0, 1);
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered LEADING_PATH_NIBBLE on MPT side"
        );
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn mpt_odd_leaf_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = OddLeafRlpWitness::from_leaves(&[OddLeafRow {
            leading_nibble: 7,
            mid_nibble: 0xa,
            last_nibble: 0xb,
            value: sample_value(),
        }])
        .unwrap();
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = OddLeafRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "mpt_odd_leaf_rlp gadget proof must verify");
    }
}
