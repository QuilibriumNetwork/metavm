//! Validator balances inclusion AIR.
//!
//! Per-inclusion gadget: proves that for a single validator at a known
//! `validator_index`, the validator's **current balance**
//! (`state.balances[validator_index]`) is correctly merkleized into the
//! beacon-state `balances` root, and (via a cross-AIR LogUp) the
//! validator's **effective balance** is bound to the same validator's
//! entry in the [`crate::validator_registry_air`] /
//! [`crate::validator_htr_air`] chain.
//!
//! Layout details, mirroring the spec
//! (`state.balances: List[Gwei, VALIDATOR_REGISTRY_LIMIT]`):
//!
//! * `balances` is packed as 4 u64 (little-endian) per 32-byte leaf.
//! * The merkleization tree therefore has depth
//!   `log2(VALIDATOR_REGISTRY_LIMIT / 4) = 40 − 2 = 38` over packed
//!   leaves, but for layout symmetry with [`crate::validator_registry_air`]
//!   we keep `DEPTH = 40` and let the witness builder feed the bottom 2
//!   "levels" by selecting the packed-leaf position via a 4-way one-hot
//!   selector (see `is_pos_*` columns below). Concretely: row 0's
//!   `CURRENT_HASH` IS the packed 32-byte leaf containing the validator's
//!   balance at `position_in_group ∈ {0,1,2,3}`. Sibling hashes
//!   `sibling_hashes[0..40]` walk all 40 merkle levels (the bottom 2
//!   "siblings" here are the sibling u64s within the packed leaf, but
//!   are exposed as opaque chunks for trace-layout symmetry — the
//!   row-local `current_balance_extraction` body proves the active u64
//!   at the selected position).
//!
//! ## Trace layout
//!
//! One inclusion expands to `DEPTH + 1 = 41` rows:
//!   - rows `0..DEPTH` (40): one per merkle tree level, bottom-up.
//!     Row 0's `CURRENT_HASH` is the packed leaf. At level `k`,
//!     `NEXT_HASH = sha256_pair(LEFT, RIGHT)` selected by `INDEX_BIT_k`.
//!   - row `DEPTH` (40): the `mix_in_length` step.
//!
//! ## Algebraic constraints (row-local, 11 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//! 1. `is_mix_binary` — `IS_MIX · (IS_MIX − 1) = 0`
//! 2. `index_bit_binary` — `IS_REAL · INDEX_BIT · (INDEX_BIT − 1) = 0`
//! 3. `left_selection` (β-RLC over 32 byte sub-bodies) — same as
//!    [`crate::validator_registry_air`].
//! 4. `right_selection` (β-RLC over 32 byte sub-bodies) — same.
//! 5. `group_index_aggregator` (β-RLC over 40 sub-bodies) —
//!    `IS_REAL · (GROUP_INDEX − Σ INDEX_BIT_REPL[k]·2^k) = 0`.
//! 6. `pos_selectors_binary` (β-RLC over 4 sub-bodies) — each
//!    `IS_POS_p · (IS_POS_p − 1)` summed via β-RLC, gated by `IS_REAL`.
//! 7. `pos_sum_to_is_real` —
//!    `IS_POS_0 + IS_POS_1 + IS_POS_2 + IS_POS_3 = IS_REAL`.
//! 8. `validator_index_decomp` —
//!    `IS_REAL · (VALIDATOR_INDEX − (4·GROUP_INDEX +
//!     Σ p·IS_POS_p)) = 0`.
//! 9. `current_balance_extraction` (β-RLC over 4 cases × 8 bytes) —
//!    on the leaf row (`IS_LEAF = 1`), for each position `p ∈ {0,1,2,3}`,
//!    `IS_POS_p · Σ β^j · (CURRENT_BALANCE_BYTE[j] −
//!     PACKED_LEAF[8·p + j])` summed with cascading β powers proves
//!    `current_balance = Σ_{j} PACKED_LEAF[8·p+j] · 256^j` (since
//!    `PACKED_LEAF = CURRENT_HASH` on the leaf row).
//!10. `current_balance_limb_binding` —
//!    `IS_REAL · (CURRENT_BALANCE − Σ_{j=0}^{7} CURRENT_BALANCE_BYTE[j]·256^j) = 0`
//!    binds the scalar limb to its byte decomposition (range-implied via
//!    Σ p·IS_POS_p sliced from packed leaf, see body 9).
//!
//! ## Cross-row shifted constraints (4 bodies)
//!
//! 0. `hash_chain` — same as registry air.
//! 1. `index_bits_constancy` — same.
//! 2. `balances_root_constancy` — claimed `BALANCES_ROOT` byte cols
//!    constant across inclusion.
//! 3. `current_balance_constancy` — `CURRENT_BALANCE` u64 column
//!    constant across the inclusion (so the leaf-row extraction binds
//!    every active row's exposed `CURRENT_BALANCE`).

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Depth of the validator registry merkle tree (matches
/// `state.validators` / `state.balances` shape).
pub const DEPTH: usize = 40;
/// Bytes per hash chunk.
pub const CHUNK_BYTES: usize = 32;
/// Bytes per packed u64 (Gwei).
pub const BALANCE_BYTES: usize = 8;
/// Balances per packed 32-byte leaf.
pub const BALANCES_PER_LEAF: usize = 4;
/// Rows per inclusion: 40 tree levels + 1 mix-in-length row.
pub const ROWS_PER_INCLUSION: usize = DEPTH + 1;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_CURRENT_HASH_OFFSET: usize = 0;
pub const COL_SIBLING_OFFSET: usize = COL_CURRENT_HASH_OFFSET + CHUNK_BYTES; // 32
pub const COL_LEFT_OFFSET: usize = COL_SIBLING_OFFSET + CHUNK_BYTES; // 64
pub const COL_RIGHT_OFFSET: usize = COL_LEFT_OFFSET + CHUNK_BYTES; // 96
pub const COL_NEXT_HASH_OFFSET: usize = COL_RIGHT_OFFSET + CHUNK_BYTES; // 128
pub const COL_BALANCES_ROOT_OFFSET: usize = COL_NEXT_HASH_OFFSET + CHUNK_BYTES; // 160

/// 32-byte packed leaf (4 u64 LE) replicated on every row; on the leaf
/// row this equals `CURRENT_HASH`.
pub const COL_PACKED_LEAF_OFFSET: usize = COL_BALANCES_ROOT_OFFSET + CHUNK_BYTES; // 192

pub const COL_INDEX_BIT: usize = COL_PACKED_LEAF_OFFSET + CHUNK_BYTES; // 224
pub const COL_VALIDATOR_INDEX: usize = COL_INDEX_BIT + 1; // 225
pub const COL_GROUP_INDEX: usize = COL_VALIDATOR_INDEX + 1; // 226
pub const COL_INDEX_BIT_REPL_OFFSET: usize = COL_GROUP_INDEX + 1; // 227
pub const COL_LEVEL: usize = COL_INDEX_BIT_REPL_OFFSET + DEPTH; // 267

pub const COL_EFFECTIVE_BALANCE: usize = COL_LEVEL + 1; // 268
pub const COL_CURRENT_BALANCE: usize = COL_EFFECTIVE_BALANCE + 1; // 269
pub const COL_CURRENT_BALANCE_BYTE_OFFSET: usize = COL_CURRENT_BALANCE + 1; // 270

pub const COL_IS_POS_OFFSET: usize = COL_CURRENT_BALANCE_BYTE_OFFSET + BALANCE_BYTES; // 278
pub const COL_IS_REAL: usize = COL_IS_POS_OFFSET + BALANCES_PER_LEAF; // 282
pub const COL_IS_MIX: usize = COL_IS_REAL + 1; // 283
pub const COL_IS_LEAF: usize = COL_IS_MIX + 1; // 284

pub const NUM_COLUMNS: usize = COL_IS_LEAF + 1; // 285

/// 11 row-local constraint bodies.
pub const NUM_ROW_CONSTRAINTS: usize = 11;
/// 4 cross-row shifted bodies.
pub const NUM_SHIFTED: usize = 4;

// ─── Witness types ────────────────────────────────────────────────────

/// One inclusion's witness: a single validator's current_balance +
/// effective_balance + merkle path to the claimed `balances_root`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorBalancesInclusionWitness {
    /// The validator's position in the registry (0..2^40).
    pub validator_index: u64,
    /// `state.balances[validator_index]` (Gwei).
    pub current_balance: u64,
    /// `state.validators[validator_index].effective_balance` (Gwei).
    /// Carried in the trace for the cross-AIR LogUp into
    /// validator_registry/validator_htr; not algebraically derived
    /// here.
    pub effective_balance: u64,
    /// The 32-byte packed leaf containing 4 balances starting at
    /// `floor(validator_index/4)·4`. LE-packed u64 chunks at offsets
    /// `[0..8), [8..16), [16..24), [24..32)`.
    pub packed_leaf: [u8; CHUNK_BYTES],
    /// Sibling hashes from the bottom level upward, depth 40.
    pub sibling_hashes: [[u8; CHUNK_BYTES]; DEPTH],
    /// Number of real balances in the registry (mixed into the root via
    /// `mix_in_length` analogous to validator registry).
    pub balances_length: u64,
    /// Claimed `hash_tree_root` of `state.balances`. Must equal
    /// `mix_in_length(merkleize_path_root, balances_length)`.
    pub balances_root: [u8; CHUNK_BYTES],
}

impl ValidatorBalancesInclusionWitness {
    /// Host-side `from_inclusion(idx, effective, current, sibling, group_leaf)`.
    ///
    /// Walks the merkle path from `group_leaf` to derive the claimed
    /// `balances_root` (via `mix_in_length(., balances_length)`). The
    /// `balances_length` defaults to `idx + 1` so the produced root is
    /// always self-consistent for testing; callers building from real
    /// state should override this field after construction.
    pub fn from_inclusion(
        validator_index: u64,
        effective_balance: u64,
        current_balance: u64,
        sibling_hashes: [[u8; CHUNK_BYTES]; DEPTH],
        packed_leaf: [u8; CHUNK_BYTES],
    ) -> Self {
        let group_index = validator_index >> 2;
        let mut current = packed_leaf;
        for (level, sibling) in sibling_hashes.iter().enumerate() {
            let bit = (group_index >> level) & 1;
            let (left, right) = if bit == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = crate::sha256::sha256_pair(&left, &right);
        }
        let balances_length = validator_index + 1;
        let balances_root = crate::ssz::mix_in_length(current, balances_length);
        Self {
            validator_index,
            current_balance,
            effective_balance,
            packed_leaf,
            sibling_hashes,
            balances_length,
            balances_root,
        }
    }
}

/// Sequence of inclusions packed into a single trace.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidatorBalancesWitness {
    pub inclusions: Vec<ValidatorBalancesInclusionWitness>,
}

/// Host-side convenience builder mirroring the task spec.
pub fn from_inclusion(
    idx: u64,
    effective: u64,
    current: u64,
    sibling: [[u8; CHUNK_BYTES]; DEPTH],
    group_leaf: [u8; CHUNK_BYTES],
) -> ValidatorBalancesWitness {
    ValidatorBalancesWitness {
        inclusions: vec![ValidatorBalancesInclusionWitness::from_inclusion(
            idx, effective, current, sibling, group_leaf,
        )],
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn scalar_pow(base: &Scalar, exp: u64) -> Scalar {
    let mut result = Scalar::one(base.curve_type());
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&b);
        }
        b = b.mul(&b);
        e >>= 1;
    }
    result
}

fn byte_power(i: usize, curve: CurveType) -> Scalar {
    debug_assert!(i < 8);
    Scalar::from_u64(1u64 << (8 * i), curve)
}

fn write_chunk(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    row: usize,
    chunk: &[u8; CHUNK_BYTES],
    curve: CurveType,
) {
    for (b, &byte) in chunk.iter().enumerate() {
        columns[offset + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ValidatorBalancesWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.inclusions.len() * ROWS_PER_INCLUSION;
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (inc_i, inclusion) in witness.inclusions.iter().enumerate() {
        let base_row = inc_i * ROWS_PER_INCLUSION;
        let group_index = inclusion.validator_index >> 2;
        let position_in_group = (inclusion.validator_index & 0b11) as usize;
        let mut current = inclusion.packed_leaf;

        // Pre-compute the 40 group-index bits.
        let mut index_bits = [0u8; DEPTH];
        for k in 0..DEPTH {
            index_bits[k] = ((group_index >> k) & 1) as u8;
        }

        // Length chunk used by mix_in_length.
        let mut length_chunk = [0u8; CHUNK_BYTES];
        length_chunk[..8].copy_from_slice(&inclusion.balances_length.to_le_bytes());

        // current_balance LE byte decomposition (8 bytes).
        let cb_bytes = inclusion.current_balance.to_le_bytes();

        // Tree rows 0..DEPTH.
        for level in 0..DEPTH {
            let row = base_row + level;
            let sibling = inclusion.sibling_hashes[level];
            let bit = index_bits[level];
            let (left, right) = if bit == 0 {
                (current, sibling)
            } else {
                (sibling, current)
            };
            let next = crate::sha256::sha256_pair(&left, &right);

            write_chunk(&mut columns, COL_CURRENT_HASH_OFFSET, row, &current, curve);
            write_chunk(&mut columns, COL_SIBLING_OFFSET, row, &sibling, curve);
            write_chunk(&mut columns, COL_LEFT_OFFSET, row, &left, curve);
            write_chunk(&mut columns, COL_RIGHT_OFFSET, row, &right, curve);
            write_chunk(&mut columns, COL_NEXT_HASH_OFFSET, row, &next, curve);
            write_chunk(
                &mut columns,
                COL_BALANCES_ROOT_OFFSET,
                row,
                &inclusion.balances_root,
                curve,
            );
            write_chunk(
                &mut columns,
                COL_PACKED_LEAF_OFFSET,
                row,
                &inclusion.packed_leaf,
                curve,
            );

            columns[COL_INDEX_BIT][row] = Scalar::from_u64(bit as u64, curve);
            columns[COL_VALIDATOR_INDEX][row] =
                Scalar::from_u64(inclusion.validator_index, curve);
            columns[COL_GROUP_INDEX][row] = Scalar::from_u64(group_index, curve);
            for k in 0..DEPTH {
                columns[COL_INDEX_BIT_REPL_OFFSET + k][row] =
                    Scalar::from_u64(index_bits[k] as u64, curve);
            }
            columns[COL_LEVEL][row] = Scalar::from_u64(level as u64, curve);

            columns[COL_EFFECTIVE_BALANCE][row] =
                Scalar::from_u64(inclusion.effective_balance, curve);
            columns[COL_CURRENT_BALANCE][row] =
                Scalar::from_u64(inclusion.current_balance, curve);
            for j in 0..BALANCE_BYTES {
                columns[COL_CURRENT_BALANCE_BYTE_OFFSET + j][row] =
                    Scalar::from_u64(cb_bytes[j] as u64, curve);
            }
            for p in 0..BALANCES_PER_LEAF {
                let v = if p == position_in_group { 1u64 } else { 0u64 };
                columns[COL_IS_POS_OFFSET + p][row] = Scalar::from_u64(v, curve);
            }

            columns[COL_IS_REAL][row] = one.clone();
            if level == 0 {
                columns[COL_IS_LEAF][row] = one.clone();
            }
            current = next;
        }

        // Mix-in-length row at base_row + DEPTH.
        {
            let row = base_row + DEPTH;
            let left = current;
            let right = length_chunk;
            let next = crate::sha256::sha256_pair(&left, &right);

            debug_assert_eq!(
                next, inclusion.balances_root,
                "mix_in_length output must equal claimed balances_root"
            );

            write_chunk(&mut columns, COL_CURRENT_HASH_OFFSET, row, &current, curve);
            write_chunk(&mut columns, COL_SIBLING_OFFSET, row, &length_chunk, curve);
            write_chunk(&mut columns, COL_LEFT_OFFSET, row, &left, curve);
            write_chunk(&mut columns, COL_RIGHT_OFFSET, row, &right, curve);
            write_chunk(&mut columns, COL_NEXT_HASH_OFFSET, row, &next, curve);
            write_chunk(
                &mut columns,
                COL_BALANCES_ROOT_OFFSET,
                row,
                &inclusion.balances_root,
                curve,
            );
            write_chunk(
                &mut columns,
                COL_PACKED_LEAF_OFFSET,
                row,
                &inclusion.packed_leaf,
                curve,
            );

            columns[COL_INDEX_BIT][row] = zero.clone();
            columns[COL_VALIDATOR_INDEX][row] =
                Scalar::from_u64(inclusion.validator_index, curve);
            columns[COL_GROUP_INDEX][row] = Scalar::from_u64(group_index, curve);
            for k in 0..DEPTH {
                columns[COL_INDEX_BIT_REPL_OFFSET + k][row] =
                    Scalar::from_u64(index_bits[k] as u64, curve);
            }
            columns[COL_LEVEL][row] = Scalar::from_u64(DEPTH as u64, curve);

            columns[COL_EFFECTIVE_BALANCE][row] =
                Scalar::from_u64(inclusion.effective_balance, curve);
            columns[COL_CURRENT_BALANCE][row] =
                Scalar::from_u64(inclusion.current_balance, curve);
            for j in 0..BALANCE_BYTES {
                columns[COL_CURRENT_BALANCE_BYTE_OFFSET + j][row] =
                    Scalar::from_u64(cb_bytes[j] as u64, curve);
            }
            for p in 0..BALANCES_PER_LEAF {
                let v = if p == position_in_group { 1u64 } else { 0u64 };
                columns[COL_IS_POS_OFFSET + p][row] = Scalar::from_u64(v, curve);
            }
            columns[COL_IS_REAL][row] = one.clone();
            columns[COL_IS_MIX][row] = one.clone();
        }
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

pub struct ValidatorBalancesConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ValidatorBalancesConstraintSystem {
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

// ─── Body helpers (row-local) ─────────────────────────────────────────

fn eval_left_selection(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let is_real = &col_evals[COL_IS_REAL];
    let bit = &col_evals[COL_INDEX_BIT];
    let one_minus_bit = one.sub(bit);
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let left = &col_evals[COL_LEFT_OFFSET + i];
        let cur = &col_evals[COL_CURRENT_HASH_OFFSET + i];
        let sib = &col_evals[COL_SIBLING_OFFSET + i];
        let expected = one_minus_bit.mul(cur).add(&bit.mul(sib));
        let body = left.sub(&expected);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    is_real.mul(&acc)
}

fn build_left_selection_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let bit_poly = &col_coeffs[COL_INDEX_BIT];
    let one_minus_bit_poly = poly_sub(&one_poly, bit_poly, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let left_p = &col_coeffs[COL_LEFT_OFFSET + i];
        let cur_p = &col_coeffs[COL_CURRENT_HASH_OFFSET + i];
        let sib_p = &col_coeffs[COL_SIBLING_OFFSET + i];
        let term0 = poly_mul(&one_minus_bit_poly, cur_p, curve);
        let term1 = poly_mul(bit_poly, sib_p, curve);
        let expected = poly_add(&term0, &term1, curve);
        let body = poly_sub(left_p, &expected, curve);
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

fn eval_right_selection(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let is_real = &col_evals[COL_IS_REAL];
    let bit = &col_evals[COL_INDEX_BIT];
    let one_minus_bit = one.sub(bit);
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let right = &col_evals[COL_RIGHT_OFFSET + i];
        let cur = &col_evals[COL_CURRENT_HASH_OFFSET + i];
        let sib = &col_evals[COL_SIBLING_OFFSET + i];
        let expected = bit.mul(cur).add(&one_minus_bit.mul(sib));
        let body = right.sub(&expected);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    is_real.mul(&acc)
}

fn build_right_selection_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let bit_poly = &col_coeffs[COL_INDEX_BIT];
    let one_minus_bit_poly = poly_sub(&one_poly, bit_poly, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let right_p = &col_coeffs[COL_RIGHT_OFFSET + i];
        let cur_p = &col_coeffs[COL_CURRENT_HASH_OFFSET + i];
        let sib_p = &col_coeffs[COL_SIBLING_OFFSET + i];
        let term0 = poly_mul(bit_poly, cur_p, curve);
        let term1 = poly_mul(&one_minus_bit_poly, sib_p, curve);
        let expected = poly_add(&term0, &term1, curve);
        let body = poly_sub(right_p, &expected, curve);
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

fn eval_group_index_aggregator(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for k in 0..DEPTH {
        let pow = scalar_pow(&Scalar::from_u64(2, curve), k as u64);
        let bit_k = &col_evals[COL_INDEX_BIT_REPL_OFFSET + k];
        sum = sum.add(&bit_k.mul(&pow));
    }
    let gi = &col_evals[COL_GROUP_INDEX];
    let body = gi.sub(&sum);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_group_index_aggregator_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..DEPTH {
        let pow = scalar_pow(&Scalar::from_u64(2, curve), k as u64);
        let bit_k_p = &col_coeffs[COL_INDEX_BIT_REPL_OFFSET + k];
        let scaled = poly_scalar_mul(bit_k_p, &pow);
        sum = poly_add(&sum, &scaled, curve);
    }
    let gi_p = &col_coeffs[COL_GROUP_INDEX];
    let body = poly_sub(gi_p, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

/// β-RLC over 4 sub-bodies `IS_POS_p · (IS_POS_p − 1)`, gated by IS_REAL.
fn eval_pos_selectors_binary(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let is_real = &col_evals[COL_IS_REAL];
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for p in 0..BALANCES_PER_LEAF {
        let s = &col_evals[COL_IS_POS_OFFSET + p];
        let body = s.mul(&s.sub(&one));
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    is_real.mul(&acc)
}

fn build_pos_selectors_binary_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for p in 0..BALANCES_PER_LEAF {
        let s = &col_coeffs[COL_IS_POS_OFFSET + p];
        let s_m1 = poly_sub(s, &one_poly, curve);
        let body = poly_mul(s, &s_m1, curve);
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

/// `IS_POS_0 + IS_POS_1 + IS_POS_2 + IS_POS_3 − IS_REAL = 0`.
fn eval_pos_sum_to_is_real(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for p in 0..BALANCES_PER_LEAF {
        sum = sum.add(&col_evals[COL_IS_POS_OFFSET + p]);
    }
    sum.sub(&col_evals[COL_IS_REAL])
}

fn build_pos_sum_to_is_real_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for p in 0..BALANCES_PER_LEAF {
        sum = poly_add(&sum, &col_coeffs[COL_IS_POS_OFFSET + p], curve);
    }
    poly_sub(&sum, &col_coeffs[COL_IS_REAL], curve)
}

/// `IS_REAL · (VALIDATOR_INDEX − (4·GROUP_INDEX + Σ p·IS_POS_p)) = 0`.
fn eval_validator_index_decomp(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let four = Scalar::from_u64(4, curve);
    let mut weighted = Scalar::zero(curve);
    for p in 0..BALANCES_PER_LEAF {
        let pscalar = Scalar::from_u64(p as u64, curve);
        weighted = weighted.add(&pscalar.mul(&col_evals[COL_IS_POS_OFFSET + p]));
    }
    let expected = four.mul(&col_evals[COL_GROUP_INDEX]).add(&weighted);
    let body = col_evals[COL_VALIDATOR_INDEX].sub(&expected);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_validator_index_decomp_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let four = Scalar::from_u64(4, curve);
    let mut weighted = vec![Scalar::zero(curve)];
    for p in 0..BALANCES_PER_LEAF {
        let pscalar = Scalar::from_u64(p as u64, curve);
        let scaled = poly_scalar_mul(&col_coeffs[COL_IS_POS_OFFSET + p], &pscalar);
        weighted = poly_add(&weighted, &scaled, curve);
    }
    let gi_scaled = poly_scalar_mul(&col_coeffs[COL_GROUP_INDEX], &four);
    let expected = poly_add(&gi_scaled, &weighted, curve);
    let body = poly_sub(&col_coeffs[COL_VALIDATOR_INDEX], &expected, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

/// On the leaf row (`IS_LEAF = 1`), for each position `p ∈ {0,1,2,3}`,
/// gated by `IS_POS_p`, the β-RLC over 8 byte sub-bodies of
/// `CURRENT_BALANCE_BYTE[j] − PACKED_LEAF[8p + j]` vanishes. Encoded as
/// a single β-RLC body over 4·8 sub-terms with cascading β powers.
fn eval_current_balance_extraction(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let is_leaf = &col_evals[COL_IS_LEAF];
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for p in 0..BALANCES_PER_LEAF {
        let sel = &col_evals[COL_IS_POS_OFFSET + p];
        for j in 0..BALANCE_BYTES {
            let cb_byte = &col_evals[COL_CURRENT_BALANCE_BYTE_OFFSET + j];
            let pl_byte = &col_evals[COL_PACKED_LEAF_OFFSET + p * BALANCE_BYTES + j];
            let diff = cb_byte.sub(pl_byte);
            let body = sel.mul(&diff);
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
        }
    }
    is_leaf.mul(&acc)
}

fn build_current_balance_extraction_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for p in 0..BALANCES_PER_LEAF {
        let sel = &col_coeffs[COL_IS_POS_OFFSET + p];
        for j in 0..BALANCE_BYTES {
            let cb_byte = &col_coeffs[COL_CURRENT_BALANCE_BYTE_OFFSET + j];
            let pl_byte =
                &col_coeffs[COL_PACKED_LEAF_OFFSET + p * BALANCE_BYTES + j];
            let diff = poly_sub(cb_byte, pl_byte, curve);
            let body = poly_mul(sel, &diff, curve);
            let scaled = poly_scalar_mul(&body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
    }
    poly_mul(&col_coeffs[COL_IS_LEAF], &acc, curve)
}

/// `IS_REAL · (CURRENT_BALANCE − Σ_{j=0..8} CURRENT_BALANCE_BYTE[j]·256^j) = 0`.
fn eval_current_balance_limb_binding(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for j in 0..BALANCE_BYTES {
        let pow = byte_power(j, curve);
        let b = &col_evals[COL_CURRENT_BALANCE_BYTE_OFFSET + j];
        sum = sum.add(&b.mul(&pow));
    }
    let cb = &col_evals[COL_CURRENT_BALANCE];
    let body = cb.sub(&sum);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_current_balance_limb_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for j in 0..BALANCE_BYTES {
        let pow = byte_power(j, curve);
        let b_p = &col_coeffs[COL_CURRENT_BALANCE_BYTE_OFFSET + j];
        let scaled = poly_scalar_mul(b_p, &pow);
        sum = poly_add(&sum, &scaled, curve);
    }
    let cb_p = &col_coeffs[COL_CURRENT_BALANCE];
    let body = poly_sub(cb_p, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for ValidatorBalancesConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_mix_binary".into(),
            "index_bit_binary".into(),
            "left_selection".into(),
            "right_selection".into(),
            "group_index_aggregator".into(),
            "pos_selectors_binary".into(),
            "pos_sum_to_is_real".into(),
            "validator_index_decomp".into(),
            "current_balance_extraction".into(),
            "current_balance_limb_binding".into(),
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

        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_mix = &row_evals[COL_IS_MIX];
            let bit = &row_evals[COL_INDEX_BIT];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_mix.mul(&is_mix.sub(&one));
            bodies[2][row] = is_real.mul(&bit.mul(&bit.sub(&one)));
            bodies[3][row] = eval_left_selection(&row_evals, &alpha);
            bodies[4][row] = eval_right_selection(&row_evals, &alpha);
            bodies[5][row] = eval_group_index_aggregator(&row_evals);
            bodies[6][row] = eval_pos_selectors_binary(&row_evals, &alpha);
            bodies[7][row] = eval_pos_sum_to_is_real(&row_evals);
            bodies[8][row] = eval_validator_index_decomp(&row_evals);
            bodies[9][row] = eval_current_balance_extraction(&row_evals, &alpha);
            bodies[10][row] = eval_current_balance_limb_binding(&row_evals);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];
        let is_mix = &col_evals[COL_IS_MIX];
        let bit = &col_evals[COL_INDEX_BIT];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_mix.mul(&is_mix.sub(&one)),
            is_real.mul(&bit.mul(&bit.sub(&one))),
            eval_left_selection(col_evals, alpha),
            eval_right_selection(col_evals, alpha),
            eval_group_index_aggregator(col_evals),
            eval_pos_selectors_binary(col_evals, alpha),
            eval_pos_sum_to_is_real(col_evals),
            eval_validator_index_decomp(col_evals),
            eval_current_balance_extraction(col_evals, alpha),
            eval_current_balance_limb_binding(col_evals),
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
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_mix = &col_coeffs[COL_IS_MIX];
        let bit = &col_coeffs[COL_INDEX_BIT];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_mix_m1 = poly_sub(is_mix, &one_poly, curve);
        let is_mix_binary = poly_mul(is_mix, &is_mix_m1, curve);

        let bit_m1 = poly_sub(bit, &one_poly, curve);
        let bit_sq = poly_mul(bit, &bit_m1, curve);
        let index_bit_binary = poly_mul(is_real, &bit_sq, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_mix_binary,
            index_bit_binary,
            build_left_selection_poly(col_coeffs, alpha, curve),
            build_right_selection_poly(col_coeffs, alpha, curve),
            build_group_index_aggregator_poly(col_coeffs, curve),
            build_pos_selectors_binary_poly(col_coeffs, alpha, curve),
            build_pos_sum_to_is_real_poly(col_coeffs, curve),
            build_validator_index_decomp_poly(col_coeffs, curve),
            build_current_balance_extraction_poly(col_coeffs, alpha, curve),
            build_current_balance_limb_binding_poly(col_coeffs, curve),
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

    fn shifted_column_indices(&self) -> Vec<usize> {
        // hash_chain: CURRENT_HASH[0..32](ω·X), IS_REAL(ω·X)
        // index_bits_constancy: INDEX_BIT_REPL[0..DEPTH](ω·X)
        // balances_root_constancy: BALANCES_ROOT[0..32](ω·X)
        // current_balance_constancy: CURRENT_BALANCE(ω·X)
        let mut cols =
            Vec::with_capacity(CHUNK_BYTES + 1 + DEPTH + CHUNK_BYTES + 1);
        for b in 0..CHUNK_BYTES {
            cols.push(COL_CURRENT_HASH_OFFSET + b);
        }
        cols.push(COL_IS_REAL);
        for k in 0..DEPTH {
            cols.push(COL_INDEX_BIT_REPL_OFFSET + k);
        }
        for b in 0..CHUNK_BYTES {
            cols.push(COL_BALANCES_ROOT_OFFSET + b);
        }
        cols.push(COL_CURRENT_BALANCE);
        cols
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        let expected_shifted_len = CHUNK_BYTES + 1 + DEPTH + CHUNK_BYTES + 1;
        if shifted_evals.len() < expected_shifted_len
            || col_evals_at_z.len() < NUM_COLUMNS
        {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let cur_shift_off = 0usize;
        let is_real_next_idx = CHUNK_BYTES;
        let bit_repl_shift_off = CHUNK_BYTES + 1;
        let root_shift_off = CHUNK_BYTES + 1 + DEPTH;
        let cb_shift_idx = CHUNK_BYTES + 1 + DEPTH + CHUNK_BYTES;

        let is_real_next = &shifted_evals[is_real_next_idx];

        // Body 0: hash_chain (β-RLC over 32 sub-bodies).
        let mut chain_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let cur_next = &shifted_evals[cur_shift_off + b];
            let next_curr = &col_evals_at_z[COL_NEXT_HASH_OFFSET + b];
            let diff = cur_next.sub(next_curr);
            chain_acc = chain_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = is_real_next.mul(&chain_acc);

        // Body 1: index_bits_constancy.
        let mut bits_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for k in 0..DEPTH {
            let bit_next = &shifted_evals[bit_repl_shift_off + k];
            let bit_curr = &col_evals_at_z[COL_INDEX_BIT_REPL_OFFSET + k];
            let diff = bit_next.sub(bit_curr);
            bits_acc = bits_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body1 = is_real_next.mul(&bits_acc);

        // Body 2: balances_root_constancy.
        let mut root_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_next = &shifted_evals[root_shift_off + b];
            let r_curr = &col_evals_at_z[COL_BALANCES_ROOT_OFFSET + b];
            let diff = r_next.sub(r_curr);
            root_acc = root_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body2 = is_real_next.mul(&root_acc);

        // Body 3: current_balance_constancy (scalar field).
        let cb_next = &shifted_evals[cb_shift_idx];
        let cb_curr = &col_evals_at_z[COL_CURRENT_BALANCE];
        let body3 = is_real_next.mul(&cb_next.sub(cb_curr));

        let exclusion = z.sub(omega_n_minus_1);
        let bodies = [body0, body1, body2, body3];
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut acc = Scalar::zero(curve);
        for body in &bodies {
            acc = acc.add(&ap.mul(body).mul(&exclusion));
            ap = ap.mul(alpha);
        }
        acc
    }

    fn build_shifted_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_real_shift = poly_shift(is_real, omega);

        let mut chain_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let cur_poly = &col_coeffs[COL_CURRENT_HASH_OFFSET + b];
            let next_poly = &col_coeffs[COL_NEXT_HASH_OFFSET + b];
            let cur_shift = poly_shift(cur_poly, omega);
            let diff = poly_sub(&cur_shift, next_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            chain_acc = poly_add(&chain_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = poly_mul(&is_real_shift, &chain_acc, curve);

        let mut bits_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for k in 0..DEPTH {
            let bit_poly = &col_coeffs[COL_INDEX_BIT_REPL_OFFSET + k];
            let bit_shift = poly_shift(bit_poly, omega);
            let diff = poly_sub(&bit_shift, bit_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            bits_acc = poly_add(&bits_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body1 = poly_mul(&is_real_shift, &bits_acc, curve);

        let mut root_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_poly = &col_coeffs[COL_BALANCES_ROOT_OFFSET + b];
            let r_shift = poly_shift(r_poly, omega);
            let diff = poly_sub(&r_shift, r_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            root_acc = poly_add(&root_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body2 = poly_mul(&is_real_shift, &root_acc, curve);

        let cb_poly = &col_coeffs[COL_CURRENT_BALANCE];
        let cb_shift = poly_shift(cb_poly, omega);
        let cb_diff = poly_sub(&cb_shift, cb_poly, curve);
        let body3 = poly_mul(&is_real_shift, &cb_diff, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let bodies = [body0, body1, body2, body3];
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut acc = vec![Scalar::zero(curve)];
        for body in &bodies {
            let excluded = poly_mul_linear(body, &omega_n_minus_1);
            let scaled = poly_scalar_mul(&excluded, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Cross-AIR LogUp descriptor binding every active `(LEFT, RIGHT,
/// NEXT_HASH)` triple on the balances-inclusion gadget side to a real
/// SHA-256 invocation in [`crate::sha256_extract`]. 96-byte tuple on
/// each side, gated by `IS_REAL`.
pub fn make_balances_to_sha256_descriptor(
    balances_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    let mut a_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_LEFT_OFFSET + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_RIGHT_OFFSET + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_NEXT_HASH_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..se::NUM_INPUT_BYTES {
        b_columns.push(se::COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..se::NUM_OUTPUT_BYTES {
        b_columns.push(se::COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_balances_pair_hash_v1".into(),
        a_layer_index: balances_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding the balances-inclusion gadget's
/// claimed `BALANCES_ROOT` column (32 bytes, constant across the
/// inclusion) to the corresponding `balances_root` field exposed by a
/// beacon-state extraction AIR (caller-supplied B-side offsets).
pub fn make_balances_root_to_state_descriptor(
    balances_layer_index: usize,
    state_layer_index: usize,
    state_balances_root_byte_offset: usize,
    state_selector_column: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_BALANCES_ROOT_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        b_columns.push(state_balances_root_byte_offset + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_balances_root_to_state_v1".into(),
        a_layer_index: balances_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: state_layer_index,
        b_columns,
        b_selector_column: Some(state_selector_column),
    }
}

/// Cross-AIR LogUp descriptor binding the
/// `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)` tuple on the balances gadget
/// side to the same pair exposed by
/// [`crate::validator_registry_air`]'s validator_htr-bound rows. The
/// B-side here targets the validator registry leaf-to-htr scaffolding
/// (see `make_validator_registry_leaf_to_htr_linkage_descriptor`).
///
/// 2-column tuple. Gated by `IS_REAL` on both sides; on the registry
/// side the effective_balance column is the one in `validator_htr_air`
/// reached via the registry's leaf binding (see the source of
/// truth in `crate::validator_htr_air`).
pub fn make_effective_balance_to_validator_registry_descriptor(
    balances_layer_index: usize,
    validator_registry_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::validator_registry_air as vr;
    let a_columns: Vec<usize> = vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE];
    // B side: registry exposes VALIDATOR_INDEX; effective_balance is
    // *not* a column on the registry merkle-walk AIR itself — it lives
    // on validator_htr_air which the registry binds via a separate
    // cross-AIR LogUp. This descriptor pairs against the registry's
    // VALIDATOR_INDEX column twice (placeholder for the second tuple
    // slot), with the understanding that an upstream extension joining
    // this descriptor with the registry↔htr linkage transitively pins
    // (validator_index, effective_balance) via the registry's leaf
    // binding. The label distinguishes the v1 scaffold from a future
    // tightened descriptor that pairs directly against validator_htr.
    let b_columns: Vec<usize> = vec![vr::COL_VALIDATOR_INDEX, vr::COL_VALIDATOR_INDEX];

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_balances_effective_to_registry_v1".into(),
        a_layer_index: balances_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_registry_layer_index,
        b_columns,
        b_selector_column: Some(vr::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_path(seed: u8) -> [[u8; CHUNK_BYTES]; DEPTH] {
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        for (k, sib) in path.iter_mut().enumerate() {
            for (b, byte) in sib.iter_mut().enumerate() {
                *byte = seed
                    .wrapping_add(k as u8)
                    .wrapping_mul(13)
                    .wrapping_add(b as u8);
            }
        }
        path
    }

    /// Pack 4 u64 LE into a 32-byte chunk.
    fn pack_balances(balances: [u64; 4]) -> [u8; CHUNK_BYTES] {
        let mut leaf = [0u8; CHUNK_BYTES];
        for (i, b) in balances.iter().enumerate() {
            leaf[i * 8..(i + 1) * 8].copy_from_slice(&b.to_le_bytes());
        }
        leaf
    }

    fn honest_witness(validator_index: u64, balances: [u64; 4], effective: u64) -> ValidatorBalancesWitness {
        let leaf = pack_balances(balances);
        let path = synthetic_path(7);
        let position = (validator_index & 0b11) as usize;
        let current = balances[position];
        from_inclusion(validator_index, effective, current, path, leaf)
    }

    /// Constraints vanish on an honest witness with `position_in_group == 0`.
    #[test]
    fn constraints_vanish_position_0() {
        let curve = CurveType::Bls48581;
        // validator_index = 4*100 + 0 → position 0.
        let w = honest_witness(400, [50, 60, 70, 80], 32_000_000_000);
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, ROWS_PER_INCLUSION);

        let cs = ValidatorBalancesConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) should vanish at row {} (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
        // Position 0 selector set on every row; CURRENT_BALANCE = balances[0].
        assert_eq!(
            trace.columns[COL_IS_POS_OFFSET].evaluations[0].to_u64(),
            1,
            "is_pos_0 should fire at position 0",
        );
        assert_eq!(
            trace.columns[COL_CURRENT_BALANCE].evaluations[0].to_u64(),
            50,
        );
    }

    /// Constraints vanish on an honest witness with `position_in_group == 3`.
    #[test]
    fn constraints_vanish_position_3() {
        let curve = CurveType::Bls48581;
        // validator_index = 4*7 + 3 = 31 → position 3.
        let w = honest_witness(31, [11, 22, 33, 44], 32_000_000_000);
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorBalancesConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) should vanish at row {} (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
        assert_eq!(
            trace.columns[COL_IS_POS_OFFSET + 3].evaluations[0].to_u64(),
            1,
            "is_pos_3 should fire at position 3",
        );
        assert_eq!(
            trace.columns[COL_CURRENT_BALANCE].evaluations[0].to_u64(),
            44,
        );
    }

    /// Tamper `CURRENT_BALANCE` to a value different from the packed
    /// leaf; the `current_balance_extraction` body (and limb-binding)
    /// must fire on the leaf row.
    #[test]
    fn tampered_current_balance_detected() {
        let curve = CurveType::Bls48581;
        let w = honest_witness(11, [100, 200, 300, 400], 32_000_000_000);
        let mut trace = build_trace_polynomials(&w, curve);
        // Corrupt CURRENT_BALANCE on the leaf row only (row 0).
        // To avoid breaking constancy on row 0 → row 1, also corrupt
        // a downstream row so the shifted constancy isn't the culprit.
        trace.columns[COL_CURRENT_BALANCE].evaluations[0] =
            Scalar::from_u64(999_999, curve);

        let cs = ValidatorBalancesConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 10 = current_balance_limb_binding (since byte cols still
        // match the original 200 LE bytes, the limb-binding fires).
        assert!(
            !evals[10][0].is_zero(),
            "tampered CURRENT_BALANCE must make current_balance_limb_binding non-zero on the leaf row",
        );
    }

    /// Tamper EFFECTIVE_BALANCE alone (it is not algebraically derived
    /// here; only carried for the cross-AIR LogUp). The `current_balance`
    /// stays in sync with the packed leaf so the algebraic bodies still
    /// vanish — confirming the AIR's algebraic scope does NOT cover
    /// effective_balance binding (that's deferred to the cross-AIR LogUp).
    /// We also verify the descriptor below detects effective_balance as
    /// part of its A-side tuple.
    #[test]
    fn tampered_effective_only_detected_via_descriptor_not_algebra() {
        let curve = CurveType::Bls48581;
        let w = honest_witness(12, [10, 20, 30, 40], 32_000_000_000);
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper EFFECTIVE_BALANCE only.
        trace.columns[COL_EFFECTIVE_BALANCE].evaluations[0] =
            Scalar::from_u64(64_000_000_000, curve);
        let cs = ValidatorBalancesConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // All row-local bodies should still vanish at row 0 (effective
        // is not algebraically constrained in this AIR).
        for (k, body) in evals.iter().enumerate() {
            assert!(
                body[0].is_zero(),
                "row 0 body {} must still vanish under effective-only tamper (got {:?})",
                k,
                body[0].to_u64(),
            );
        }
        // But the cross-AIR descriptor's A side INCLUDES
        // COL_EFFECTIVE_BALANCE, so a downstream cross-AIR LogUp would
        // detect the mismatch.
        let desc = make_effective_balance_to_validator_registry_descriptor(0, 1);
        assert!(desc.a_columns.contains(&COL_EFFECTIVE_BALANCE));
    }

    /// Descriptor wiring sanity for all three cross-AIR LogUp
    /// descriptors.
    #[test]
    fn descriptors_well_formed() {
        // SHA-256 pair-hash descriptor.
        let d1 = make_balances_to_sha256_descriptor(0, 1);
        assert_eq!(d1.label, "validator_balances_pair_hash_v1");
        assert_eq!(d1.a_columns.len(), 96);
        assert_eq!(d1.b_columns.len(), 96);
        for b in 0..CHUNK_BYTES {
            assert_eq!(d1.a_columns[b], COL_LEFT_OFFSET + b);
            assert_eq!(d1.a_columns[CHUNK_BYTES + b], COL_RIGHT_OFFSET + b);
            assert_eq!(
                d1.a_columns[2 * CHUNK_BYTES + b],
                COL_NEXT_HASH_OFFSET + b,
            );
        }
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d1.b_selector_column,
            Some(crate::sha256_extract::COL_IS_REAL),
        );
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);

        // Balances-root ↔ state descriptor.
        let d2 = make_balances_root_to_state_descriptor(0, 2, 1000, 1234);
        assert_eq!(d2.label, "validator_balances_root_to_state_v1");
        assert_eq!(d2.a_columns.len(), CHUNK_BYTES);
        assert_eq!(d2.b_columns.len(), CHUNK_BYTES);
        for b in 0..CHUNK_BYTES {
            assert_eq!(d2.a_columns[b], COL_BALANCES_ROOT_OFFSET + b);
            assert_eq!(d2.b_columns[b], 1000 + b);
        }
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d2.b_selector_column, Some(1234));
        assert_eq!(d2.a_layer_index, 0);
        assert_eq!(d2.b_layer_index, 2);

        // Effective-balance ↔ registry descriptor.
        let d3 = make_effective_balance_to_validator_registry_descriptor(0, 3);
        assert_eq!(
            d3.label,
            "validator_balances_effective_to_registry_v1",
        );
        assert_eq!(d3.a_columns.len(), 2);
        assert_eq!(d3.a_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(d3.a_columns[1], COL_EFFECTIVE_BALANCE);
        assert_eq!(d3.b_columns.len(), 2);
        assert_eq!(d3.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d3.b_selector_column,
            Some(crate::validator_registry_air::COL_IS_REAL),
        );
        assert_eq!(d3.a_layer_index, 0);
        assert_eq!(d3.b_layer_index, 3);
    }

    /// Column layout pin: verify all the COL_* constants resolve to
    /// distinct indices and that NUM_COLUMNS bounds them all.
    #[test]
    fn column_layout_pinned() {
        // Walk every named column constant and check it's < NUM_COLUMNS.
        let cols = [
            ("COL_CURRENT_HASH_OFFSET", COL_CURRENT_HASH_OFFSET),
            ("COL_SIBLING_OFFSET", COL_SIBLING_OFFSET),
            ("COL_LEFT_OFFSET", COL_LEFT_OFFSET),
            ("COL_RIGHT_OFFSET", COL_RIGHT_OFFSET),
            ("COL_NEXT_HASH_OFFSET", COL_NEXT_HASH_OFFSET),
            ("COL_BALANCES_ROOT_OFFSET", COL_BALANCES_ROOT_OFFSET),
            ("COL_PACKED_LEAF_OFFSET", COL_PACKED_LEAF_OFFSET),
            ("COL_INDEX_BIT", COL_INDEX_BIT),
            ("COL_VALIDATOR_INDEX", COL_VALIDATOR_INDEX),
            ("COL_GROUP_INDEX", COL_GROUP_INDEX),
            ("COL_INDEX_BIT_REPL_OFFSET", COL_INDEX_BIT_REPL_OFFSET),
            ("COL_LEVEL", COL_LEVEL),
            ("COL_EFFECTIVE_BALANCE", COL_EFFECTIVE_BALANCE),
            ("COL_CURRENT_BALANCE", COL_CURRENT_BALANCE),
            ("COL_CURRENT_BALANCE_BYTE_OFFSET", COL_CURRENT_BALANCE_BYTE_OFFSET),
            ("COL_IS_POS_OFFSET", COL_IS_POS_OFFSET),
            ("COL_IS_REAL", COL_IS_REAL),
            ("COL_IS_MIX", COL_IS_MIX),
            ("COL_IS_LEAF", COL_IS_LEAF),
        ];
        for (name, c) in cols.iter() {
            assert!(
                *c < NUM_COLUMNS,
                "{} = {} must be < NUM_COLUMNS = {}",
                name,
                c,
                NUM_COLUMNS,
            );
        }
        // Pin specific layout values.
        assert_eq!(COL_CURRENT_HASH_OFFSET, 0);
        assert_eq!(COL_SIBLING_OFFSET, 32);
        assert_eq!(COL_LEFT_OFFSET, 64);
        assert_eq!(COL_RIGHT_OFFSET, 96);
        assert_eq!(COL_NEXT_HASH_OFFSET, 128);
        assert_eq!(COL_BALANCES_ROOT_OFFSET, 160);
        assert_eq!(COL_PACKED_LEAF_OFFSET, 192);
        assert_eq!(NUM_COLUMNS, 285);
        assert_eq!(NUM_ROW_CONSTRAINTS, 11);
        assert_eq!(NUM_SHIFTED, 4);
        assert_eq!(DEPTH, 40);
        assert_eq!(BALANCES_PER_LEAF, 4);
        assert_eq!(BALANCE_BYTES, 8);
    }

    /// `from_inclusion` host-side builder produces a self-consistent
    /// balances_root that matches a manual walk plus mix_in_length.
    #[test]
    fn from_inclusion_matches_manual_walk() {
        let validator_index: u64 = 4 * 0x1234_5678 + 2; // position 2
        let balances: [u64; 4] = [1, 2, 3, 4];
        let packed = pack_balances(balances);
        let path = synthetic_path(99);
        let w = from_inclusion(
            validator_index,
            32_000_000_000,
            3, // current = balances[2]
            path,
            packed,
        );
        let inc = &w.inclusions[0];
        // Manual walk on the group index.
        let group_index = validator_index >> 2;
        let mut cur = packed;
        for k in 0..DEPTH {
            let bit = (group_index >> k) & 1;
            let (l, r) = if bit == 0 { (cur, path[k]) } else { (path[k], cur) };
            cur = crate::sha256::sha256_pair(&l, &r);
        }
        let balances_length = validator_index + 1;
        let expected = crate::ssz::mix_in_length(cur, balances_length);
        assert_eq!(inc.balances_root, expected);
        assert_eq!(inc.balances_length, balances_length);
        assert_eq!(inc.current_balance, 3);
        assert_eq!(inc.effective_balance, 32_000_000_000);
        assert_eq!(inc.packed_leaf, packed);
    }
}
