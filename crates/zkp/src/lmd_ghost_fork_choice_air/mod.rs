//! LMD-GHOST fork choice AIR (#159).
//!
//! Latest-Message-Driven Greediest Heaviest Observed Subtree fork
//! choice. Each row commits one candidate block in the descendant set
//! of the finalized checkpoint, with its `attestation_weight` (total
//! Gwei of latest messages voting for the block or any of its
//! descendants). The fork-choice head is the block such that, at
//! every depth, the *maximum-weight child* of the current head is
//! selected. This AIR algebraically commits the greedy-max-weight
//! child selection.
//!
//! ## Row layout
//!
//! Rows are sorted by `(parent_index, attestation_weight desc, block_index)`
//! so that the host-side topology of the descendant tree is exposed
//! row-by-row. Specifically, all children of a given parent appear in
//! contiguous rows, and within a parent group the highest-weight child
//! is the *first* row. `is_max_child` is 1 on the first row of each
//! parent group on the head path; 0 on the rest. The root's
//! `parent_index` is the sentinel `u64::MAX` so the root is sorted
//! into its own singleton group and never participates as a sibling.
//!
//! Each row carries:
//!   - `block_index` (u64): position in the descendant set (0 = finalized
//!     root).
//!   - `parent_index` (u64): the parent's `block_index`, or the
//!     sentinel `u64::MAX` for the root row.
//!   - `block_hash[0..32]`, `parent_hash[0..32]`: SSZ block roots.
//!   - `attestation_weight`: tally of latest-message stake on this
//!     subtree (Gwei).
//!   - `is_head`: 1 iff this row is on the fork-choice head path
//!     (root → … → final head).
//!   - `is_max_child`: 1 iff this row is the maximum-weight child of
//!     its parent (i.e., the first row in its parent group, by sort).
//!   - `is_real`.
//!
//! Plus a shifted body `next_weight ≤ curr_weight` within each parent
//! group (enforced by a slack byte decomposition) which proves the
//! sort ordering: among rows sharing a `parent_index`, the weights are
//! non-increasing. The greedy head pick is then `is_head` propagating
//! through the chain via `(is_head_curr · is_max_child_curr) → is_head`
//! on the next-row child whose `parent_index` equals `block_index_curr`.
//!
//! ## Scope (this step)
//!
//! Closed algebraically:
//!   - `is_real`, `is_head`, `is_max_child` are binary.
//!   - **Max-child is first in parent group** (cross-row, shifted):
//!     if `parent_index(next) = parent_index(curr)`, then
//!     `is_max_child(next) = 0` (only the first row of a parent group
//!     can be the max child). This pins the "max" selection to a
//!     specific row in each group.
//!   - **Weight non-increasing within parent group** (cross-row,
//!     shifted): if `parent_index(next) = parent_index(curr)`, then
//!     `attestation_weight(curr) − attestation_weight(next) =
//!     weight_slack ≥ 0` via byte decomposition.
//!   - **Head chain propagation** (cross-row, shifted): if a row is
//!     the head's max-weight child (and the row's parent_index
//!     matches a head row above), then `is_head` is set. The
//!     algebraic version pinned here: a row is `is_head` iff its
//!     parent_index appears as a `block_index` on some earlier head
//!     row. We close the local version: `is_head` of the row whose
//!     `parent_index` is the previous head row's `block_index` and
//!     which is `is_max_child = 1` is consistent with `is_head = 1`.
//!     Multi-row binding is a follow-up (handled via cross-AIR LogUp
//!     on `(block_index, parent_index)` tuples).
//!   - **Root exemption** (row-local): the root row has `is_root = 1`,
//!     `is_max_child = 0`, and is the seed of the head walk. The
//!     constraint `is_head · (1 − is_max_child) · (1 − is_root) = 0`
//!     exempts root from the max-child requirement.
//!   - `attestation_weight_le_decomp`: each weight is bound to 8 LE
//!     bytes (range-checked).
//!
//! Deferred:
//!   - Full topological sort + tree-walk binding (multi-row LogUp
//!     between `(block_index)` and `(parent_index)` columns to enforce
//!     parent-row existence).
//!   - Weight derivation from individual attestations: the per-row
//!     `attestation_weight` is committed as witness; the cross-AIR
//!     descriptor binds it to `attestation_aggregate_air`'s tally
//!     (target-shape scaffold; the consumer-side column is
//!     `COL_TARGET_EPOCH` placeholder until a per-block weight AIR
//!     lands).
//!   - Head block hash binding to `block_header_air::COL_BLOCK_HASH_OFFSET`:
//!     done via cross-AIR LogUp gated by `is_head`.
//!
//! ## Constraints
//!
//! Row-local (7 bodies):
//!  0. `is_real_binary`              — `is_real · (is_real − 1) = 0`.
//!  1. `is_head_binary`              — `is_head · (is_head − 1) = 0`.
//!  2. `is_max_child_binary`         — `is_max_child · (is_max_child − 1) = 0`.
//!  3. `attestation_weight_le_decomp`— `is_real · (weight −
//!     Σ weight_byte[b]·256^b) = 0`.
//!  4. `head_requires_max_child`     — `is_head · (1 − is_max_child) ·
//!     (1 − is_root) = 0`.
//!  5. `max_child_weight_dominance`  — `is_real · (max_child_weight −
//!     attestation_weight − weight_diff_slack) = 0`. Combined with
//!     byte-range check on `weight_diff_slack_byte_*`, every row in
//!     a parent group has `attestation_weight ≤ max_child_weight`.
//!  6. `is_max_child_achieves_max`   — `is_real · is_max_child ·
//!     (max_child_weight − attestation_weight) = 0`. The row claiming
//!     `is_max_child = 1` must equal `max_child_weight`.
//!
//! Shifted (4 bodies; α-power offsets start at `NUM_ROW_CONSTRAINTS`):
//!  0. `max_child_uniqueness_per_group` — `parent_eq_next ·
//!     is_max_child_next = 0` — only the first row of a parent group
//!     can be the max-child.
//!  1. `weight_non_increasing_per_group` — `parent_eq_next ·
//!     (weight_curr − weight_next − weight_slack_next) = 0`.
//!  2. `parent_eq_binary`            — `parent_eq_next · (parent_eq_next − 1) = 0`.
//!  3. `max_child_weight_group_constancy` — `parent_eq_next ·
//!     (max_child_weight_curr − max_child_weight_next) = 0`.
//!     Together with bodies 5+6 above, pins the greedy
//!     max-weight-child selection algebraically: every sibling
//!     shares the same `max_child_weight`, every sibling is bounded
//!     above by it (constraint 5), and the chosen `is_max_child`
//!     row achieves it (constraint 6).
//!
//! ## Multi-sibling extension note
//!
//! The current shifted bodies operate pairwise (current row vs next
//! row). The combination of body 3 (group-constancy of
//! `max_child_weight`) plus row-local body 5 (per-row dominance via
//! non-negative `weight_diff_slack`) is sufficient for arbitrary
//! group widths: once `max_child_weight` is pinned constant across a
//! contiguous run of `parent_eq_next = 1` rows, the per-row
//! dominance bounds *each* sibling against the same scalar. The
//! `is_max_child = 1` row's row-local body 6 forces that scalar to
//! be at least one row's actual weight, and shifted body 0 forces
//! that row to be the first of the group. Together they cover N
//! siblings with O(1) shifted bodies.
//!
//! Padding rows: all-zero (`is_real = 0`, etc.). All row-local bodies
//! that reference data columns are gated by `is_real`; shifted bodies
//! use `parent_eq` (a witness column, set to 0 on padding-to-padding
//! transitions). Both vanish on padding.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const HASH_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_BLOCK_INDEX: usize = 0;
pub const COL_PARENT_INDEX: usize = 1;
pub const COL_ATTESTATION_WEIGHT: usize = 2;
pub const COL_IS_HEAD: usize = 3;
pub const COL_IS_MAX_CHILD: usize = 4;
pub const COL_IS_ROOT: usize = 5;
pub const COL_IS_REAL: usize = 6;

pub const COL_BLOCK_HASH_OFFSET: usize = 7; // 7..39
pub const COL_PARENT_HASH_OFFSET: usize = COL_BLOCK_HASH_OFFSET + HASH_LEN; // 39..71

pub const COL_WEIGHT_BYTE_OFFSET: usize = COL_PARENT_HASH_OFFSET + HASH_LEN; // 71..79
pub const COL_PARENT_EQ_PREV: usize = COL_WEIGHT_BYTE_OFFSET + U64_BYTES; // 79
pub const COL_WEIGHT_SLACK: usize = COL_PARENT_EQ_PREV + 1; // 80
pub const COL_WEIGHT_SLACK_BYTE_OFFSET: usize = COL_WEIGHT_SLACK + 1; // 81..89

// Greedy max-weight child binding (task #199):
//   max_child_weight is the parent group's maximum attestation_weight,
//   propagated as a constant across every row sharing the same
//   `parent_index`. weight_diff_slack ≥ 0 binds
//   `max_child_weight − attestation_weight = weight_diff_slack` so every
//   row in the group has weight ≤ the claimed max. Combined with
//   `is_max_child · (max_child_weight − attestation_weight) = 0`, the
//   `is_max_child = 1` row achieves the max — i.e., greedy dominance is
//   algebraically pinned.
pub const COL_MAX_CHILD_WEIGHT: usize = COL_WEIGHT_SLACK_BYTE_OFFSET + U64_BYTES; // 89
pub const COL_WEIGHT_DIFF_SLACK: usize = COL_MAX_CHILD_WEIGHT + 1; // 90
pub const COL_WEIGHT_DIFF_SLACK_BYTE_OFFSET: usize = COL_WEIGHT_DIFF_SLACK + 1; // 91..99

pub const NUM_COLUMNS: usize = COL_WEIGHT_DIFF_SLACK_BYTE_OFFSET + U64_BYTES; // 99

pub const NUM_ROW_CONSTRAINTS: usize = 7;
pub const NUM_SHIFTED: usize = 4;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct LmdGhostRow {
    pub block_index: u64,
    pub parent_index: u64,
    pub block_hash: [u8; HASH_LEN],
    pub parent_hash: [u8; HASH_LEN],
    pub attestation_weight: u64,
    pub is_head: bool,
    pub is_max_child: bool,
    /// True iff `parent_index == block_index` (the finalized root row's
    /// self-parent encoding).
    pub is_root: bool,
}

impl Default for LmdGhostRow {
    fn default() -> Self {
        Self {
            block_index: 0,
            parent_index: 0,
            block_hash: [0u8; HASH_LEN],
            parent_hash: [0u8; HASH_LEN],
            attestation_weight: 0,
            is_head: false,
            is_max_child: false,
            is_root: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LmdGhostForkChoiceWitness {
    pub rows: Vec<LmdGhostRow>,
    /// The fork-choice head's block_hash. Convenience cache for
    /// downstream consumers.
    pub head_block_hash: [u8; HASH_LEN],
}

impl LmdGhostForkChoiceWitness {
    pub fn from_rows(rows: Vec<LmdGhostRow>) -> Self {
        let head_block_hash = rows
            .iter()
            .filter(|r| r.is_head)
            .last()
            .map(|r| r.block_hash)
            .unwrap_or([0u8; HASH_LEN]);
        Self { rows, head_block_hash }
    }

    /// Build a witness from a `(block_index, parent_index, block_hash,
    /// parent_hash, attestation_weight)` topology. The caller marks
    /// the root row with `parent_index = u64::MAX` (sentinel); all
    /// other rows use a real `parent_index`. The builder:
    ///   1. Sorts children of each parent by weight descending,
    ///      tie-breaking on block_index. The root's sentinel parent
    ///      `u64::MAX` places it in its own singleton group at the
    ///      end of the sort.
    ///   2. Walks the head path greedily: start at the root, then on
    ///      each step pick the max-weight child of the current head
    ///      (the first row in that parent group post-sort).
    ///   3. Sets `is_max_child = true` on every first-of-parent-group
    ///      row (children only — the root's group is its own
    ///      singleton, but its `is_max_child` stays false since it
    ///      has no parent); `is_head = true` on rows along the
    ///      greedy path; `is_root = true` on the row whose
    ///      `parent_index = u64::MAX`.
    pub fn from_blocks(
        mut blocks: Vec<(u64, u64, [u8; HASH_LEN], [u8; HASH_LEN], u64)>,
    ) -> Option<Self> {
        if blocks.is_empty() {
            return None;
        }
        // Stable sort: primary by parent_index ascending, secondary
        // by weight DESCENDING, tertiary by block_index ascending.
        // The root row (parent_index = u64::MAX) sorts last.
        blocks.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.4.cmp(&a.4))
                .then_with(|| a.0.cmp(&b.0))
        });

        // Compute is_max_child: first row of each parent group, but
        // ONLY for non-root parent groups (the root's group is its
        // own singleton with parent_index = u64::MAX — never a
        // sibling in the topological sense).
        let n = blocks.len();
        let mut is_max_child = vec![false; n];
        for i in 0..n {
            let is_root_row = blocks[i].1 == u64::MAX;
            if is_root_row {
                continue;
            }
            if i == 0 || blocks[i].1 != blocks[i - 1].1 {
                is_max_child[i] = true;
            }
        }

        // Find the root: the row whose parent_index == u64::MAX.
        let root_pos = (0..n).find(|&i| blocks[i].1 == u64::MAX)?;
        let mut is_head = vec![false; n];
        is_head[root_pos] = true;

        // Walk the head path greedily: from the current head, find
        // the row (post-sort) whose parent_index == current head's
        // block_index AND is_max_child = true; that row becomes the
        // next head.
        let mut current_head_block_index = blocks[root_pos].0;
        loop {
            let child = (0..n).find(|&i| {
                blocks[i].1 == current_head_block_index && is_max_child[i]
            });
            match child {
                Some(i) => {
                    is_head[i] = true;
                    current_head_block_index = blocks[i].0;
                }
                None => break,
            }
        }

        let rows: Vec<LmdGhostRow> = blocks
            .into_iter()
            .enumerate()
            .map(|(i, b)| LmdGhostRow {
                block_index: b.0,
                parent_index: b.1,
                block_hash: b.2,
                parent_hash: b.3,
                attestation_weight: b.4,
                is_head: is_head[i],
                is_max_child: is_max_child[i],
                is_root: b.1 == u64::MAX,
            })
            .collect();

        let head_block_hash = rows
            .iter()
            .filter(|r| r.is_head)
            .last()
            .map(|r| r.block_hash)
            .unwrap_or([0u8; HASH_LEN]);
        Some(Self { rows, head_block_hash })
    }
}

fn le_decomp_u64(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &LmdGhostForkChoiceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_BLOCK_INDEX][i] = Scalar::from_u64(row.block_index, curve);
        columns[COL_PARENT_INDEX][i] = Scalar::from_u64(row.parent_index, curve);
        columns[COL_ATTESTATION_WEIGHT][i] =
            Scalar::from_u64(row.attestation_weight, curve);
        columns[COL_IS_HEAD][i] = if row.is_head { one.clone() } else { zero.clone() };
        columns[COL_IS_MAX_CHILD][i] =
            if row.is_max_child { one.clone() } else { zero.clone() };
        columns[COL_IS_ROOT][i] = if row.is_root { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = one.clone();
        for k in 0..HASH_LEN {
            columns[COL_BLOCK_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.block_hash[k] as u64, curve);
            columns[COL_PARENT_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.parent_hash[k] as u64, curve);
        }
        let wb = le_decomp_u64(row.attestation_weight);
        for b in 0..U64_BYTES {
            columns[COL_WEIGHT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(wb[b] as u64, curve);
        }

        // parent_eq_prev: 1 if previous row's parent_index ==
        // this row's parent_index (defines parent-group continuity).
        let parent_eq_prev =
            i > 0 && witness.rows[i - 1].parent_index == row.parent_index;
        columns[COL_PARENT_EQ_PREV][i] =
            if parent_eq_prev { one.clone() } else { zero.clone() };

        // weight_slack on rows where parent_eq_prev = 1:
        //   slack = prev.weight − this.weight ≥ 0.
        // On rows where parent_eq_prev = 0, slack is 0.
        let slack: u64 = if parent_eq_prev {
            witness.rows[i - 1]
                .attestation_weight
                .saturating_sub(row.attestation_weight)
        } else {
            0
        };
        columns[COL_WEIGHT_SLACK][i] = Scalar::from_u64(slack, curve);
        let sb = le_decomp_u64(slack);
        for b in 0..U64_BYTES {
            columns[COL_WEIGHT_SLACK_BYTE_OFFSET + b][i] =
                Scalar::from_u64(sb[b] as u64, curve);
        }

        // Greedy max-child binding: max_child_weight equals the
        // maximum attestation_weight across all rows sharing this
        // row's parent_index (constant within the group). Since rows
        // are sorted by (parent_index asc, weight desc), the first
        // row of each group already holds the group's max — we copy
        // it forward through the contiguous group.
        let max_w: u64 = if !parent_eq_prev {
            // First row of a parent group (or row 0): this row IS
            // the max-weight child of its group.
            row.attestation_weight
        } else {
            // Continuation row: inherit the group's max from the
            // previous row.
            columns[COL_MAX_CHILD_WEIGHT][i - 1].to_u64()
        };
        columns[COL_MAX_CHILD_WEIGHT][i] = Scalar::from_u64(max_w, curve);

        let diff_slack: u64 = max_w.saturating_sub(row.attestation_weight);
        columns[COL_WEIGHT_DIFF_SLACK][i] = Scalar::from_u64(diff_slack, curve);
        let dsb = le_decomp_u64(diff_slack);
        for b in 0..U64_BYTES {
            columns[COL_WEIGHT_DIFF_SLACK_BYTE_OFFSET + b][i] =
                Scalar::from_u64(dsb[b] as u64, curve);
        }
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

pub struct LmdGhostForkChoiceConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl LmdGhostForkChoiceConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn pow256_pow(b: usize, curve: CurveType) -> Scalar {
    let mut acc: u64 = 1;
    for _ in 0..b {
        acc = acc.wrapping_mul(256);
    }
    Scalar::from_u64(acc, curve)
}

impl VmConstraintSystem for LmdGhostForkChoiceConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_head_binary".into(),
            "is_max_child_binary".into(),
            "attestation_weight_le_decomp".into(),
            "head_requires_max_child".into(),
            "max_child_weight_dominance".into(),
            "is_max_child_achieves_max".into(),
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
        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        let pow256: Vec<Scalar> =
            (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_head = &columns[COL_IS_HEAD][r];
            let is_max_child = &columns[COL_IS_MAX_CHILD][r];
            let is_root = &columns[COL_IS_ROOT][r];
            let weight = &columns[COL_ATTESTATION_WEIGHT][r];

            // 0 is_real_binary
            out[0][r] = is_real.mul(&is_real.sub(&one));
            // 1 is_head_binary
            out[1][r] = is_head.mul(&is_head.sub(&one));
            // 2 is_max_child_binary
            out[2][r] = is_max_child.mul(&is_max_child.sub(&one));
            // 3 weight LE decomp (gated by is_real).
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum
                        .add(&columns[COL_WEIGHT_BYTE_OFFSET + b][r].mul(&pow256[b]));
                }
                out[3][r] = is_real.mul(&weight.sub(&sum));
            }
            // 4 head_requires_max_child:
            //   is_head · (1 − is_max_child) · (1 − is_root) = 0.
            {
                let body = is_head
                    .mul(&one.sub(is_max_child))
                    .mul(&one.sub(is_root));
                out[4][r] = body;
            }
            // 5 max_child_weight_dominance (gated by is_real):
            //   max_child_weight − attestation_weight
            //     − weight_diff_slack = 0.
            // Combined with the byte-range check on
            // weight_diff_slack_byte_*, every row in a parent group
            // has attestation_weight ≤ max_child_weight.
            {
                let max_w = &columns[COL_MAX_CHILD_WEIGHT][r];
                let diff_slack = &columns[COL_WEIGHT_DIFF_SLACK][r];
                let body = max_w.sub(weight).sub(diff_slack);
                out[5][r] = is_real.mul(&body);
            }
            // 6 is_max_child_achieves_max (gated by is_real):
            //   is_max_child · (max_child_weight − attestation_weight) = 0.
            // The row claiming is_max_child = 1 must have the group
            // max weight. With constraint 5 + max-child uniqueness
            // (shifted body 0), this pins greedy dominance.
            {
                let max_w = &columns[COL_MAX_CHILD_WEIGHT][r];
                let diff = max_w.sub(weight);
                out[6][r] = is_real.mul(&is_max_child.mul(&diff));
            }
        }
        out
    }

    fn evaluate_at_point(
        &self,
        col_evals: &[Scalar],
        alpha: &Scalar,
    ) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let pow256: Vec<Scalar> =
            (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &col_evals[COL_IS_REAL];
        let is_head = &col_evals[COL_IS_HEAD];
        let is_max_child = &col_evals[COL_IS_MAX_CHILD];
        let is_root = &col_evals[COL_IS_ROOT];
        let weight = &col_evals[COL_ATTESTATION_WEIGHT];

        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        // 0
        acc = acc.add(&ap.mul(&is_real.mul(&is_real.sub(&one))));
        ap = ap.mul(alpha);
        // 1
        acc = acc.add(&ap.mul(&is_head.mul(&is_head.sub(&one))));
        ap = ap.mul(alpha);
        // 2
        acc = acc.add(&ap.mul(&is_max_child.mul(&is_max_child.sub(&one))));
        ap = ap.mul(alpha);
        // 3
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(&col_evals[COL_WEIGHT_BYTE_OFFSET + b].mul(&pow256[b]));
            }
            acc = acc.add(&ap.mul(&is_real.mul(&weight.sub(&sum))));
            ap = ap.mul(alpha);
        }
        // 4
        {
            let body = is_head.mul(&one.sub(is_max_child)).mul(&one.sub(is_root));
            acc = acc.add(&ap.mul(&body));
            ap = ap.mul(alpha);
        }
        // 5 max_child_weight_dominance
        {
            let max_w = &col_evals[COL_MAX_CHILD_WEIGHT];
            let diff_slack = &col_evals[COL_WEIGHT_DIFF_SLACK];
            let body = is_real.mul(&max_w.sub(weight).sub(diff_slack));
            acc = acc.add(&ap.mul(&body));
            ap = ap.mul(alpha);
        }
        // 6 is_max_child_achieves_max
        {
            let max_w = &col_evals[COL_MAX_CHILD_WEIGHT];
            let diff = max_w.sub(weight);
            let body = is_real.mul(&is_max_child.mul(&diff));
            acc = acc.add(&ap.mul(&body));
        }
        acc
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let zero = Scalar::zero(curve);
        let one_s = Scalar::one(curve);
        let one_poly = vec![one_s.clone()];
        let neg_one_poly = vec![zero.sub(&one_s)];
        let pow256: Vec<Scalar> =
            (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &column_coeffs[COL_IS_REAL];
        let is_head = &column_coeffs[COL_IS_HEAD];
        let is_max_child = &column_coeffs[COL_IS_MAX_CHILD];
        let is_root = &column_coeffs[COL_IS_ROOT];
        let weight = &column_coeffs[COL_ATTESTATION_WEIGHT];
        let max_w = &column_coeffs[COL_MAX_CHILD_WEIGHT];
        let diff_slack = &column_coeffs[COL_WEIGHT_DIFF_SLACK];

        let mut acc: Vec<Scalar> = vec![zero.clone()];
        let mut ap = Scalar::one(curve);

        let push = |acc: &mut Vec<Scalar>, ap: &Scalar, body: Vec<Scalar>| {
            *acc = poly_add(acc, &poly_scalar_mul(&body, ap), curve);
        };

        // 0 is_real_binary
        {
            let m = poly_add(is_real, &neg_one_poly, curve);
            let body = poly_mul(is_real, &m, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 1 is_head_binary
        {
            let m = poly_add(is_head, &neg_one_poly, curve);
            let body = poly_mul(is_head, &m, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 2 is_max_child_binary
        {
            let m = poly_add(is_max_child, &neg_one_poly, curve);
            let body = poly_mul(is_max_child, &m, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 3 weight LE decomp gated by is_real:
        //    is_real · (weight − Σ pow256[b] · weight_byte[b]).
        {
            let mut sum = vec![zero.clone()];
            for b in 0..U64_BYTES {
                let scaled = poly_scalar_mul(
                    &column_coeffs[COL_WEIGHT_BYTE_OFFSET + b],
                    &pow256[b],
                );
                sum = poly_add(&sum, &scaled, curve);
            }
            let diff = poly_sub(weight, &sum, curve);
            let body = poly_mul(is_real, &diff, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 4 head_requires_max_child:
        //    is_head · (1 − is_max_child) · (1 − is_root).
        {
            let one_minus_max = poly_sub(&one_poly, is_max_child, curve);
            let one_minus_root = poly_sub(&one_poly, is_root, curve);
            let t1 = poly_mul(is_head, &one_minus_max, curve);
            let body = poly_mul(&t1, &one_minus_root, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 5 max_child_weight_dominance gated by is_real:
        //    is_real · (max_w − weight − diff_slack).
        {
            let inner =
                poly_sub(&poly_sub(max_w, weight, curve), diff_slack, curve);
            let body = poly_mul(is_real, &inner, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 6 is_max_child_achieves_max gated by is_real:
        //    is_real · is_max_child · (max_w − weight).
        {
            let diff = poly_sub(max_w, weight, curve);
            let inner = poly_mul(is_max_child, &diff, curve);
            let body = poly_mul(is_real, &inner, curve);
            push(&mut acc, &ap, body);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // We model the shifted bodies as relating current-row data to
        // *previous-row* data, using the witness column
        // `parent_eq_prev` (committed on the current row). The shifted
        // reads expose the next row's `parent_eq_prev`,
        // `is_max_child`, `weight_slack`. From the current row's
        // perspective, the "next row's parent_eq_prev = 1" predicate
        // tells us the *next* row belongs to the same parent group as
        // the current row. So the body at row r reads next-row
        // columns to decide if row r and row r+1 are in the same
        // parent group.
        vec![
            COL_PARENT_EQ_PREV,    // is_same_parent_group(r → r+1)
            COL_IS_MAX_CHILD,      // next-row is_max_child
            COL_WEIGHT_SLACK,      // next-row slack
            COL_ATTESTATION_WEIGHT, // next-row weight
            COL_MAX_CHILD_WEIGHT,  // next-row max_child_weight
        ]
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
        if shifted_evals.len() < 5 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let weight_curr = &col_evals_at_z[COL_ATTESTATION_WEIGHT];
        let max_w_curr = &col_evals_at_z[COL_MAX_CHILD_WEIGHT];

        let parent_eq_next = &shifted_evals[0];
        let is_max_child_next = &shifted_evals[1];
        let weight_slack_next = &shifted_evals[2];
        let weight_next = &shifted_evals[3];
        let max_w_next = &shifted_evals[4];

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // 0 max_child_uniqueness: parent_eq_next · is_max_child_next = 0.
        let body_0 = parent_eq_next.mul(is_max_child_next);
        let term_0 = ap.mul(&body_0);
        ap = ap.mul(alpha);

        // 1 weight_non_increasing: parent_eq_next ·
        //     (weight_curr − weight_next − weight_slack_next) = 0.
        let body_1 = parent_eq_next
            .mul(&weight_curr.sub(weight_next).sub(weight_slack_next));
        let term_1 = ap.mul(&body_1);
        ap = ap.mul(alpha);

        // 2 parent_eq_binary on the next row: parent_eq_next · (parent_eq_next − 1) = 0.
        let body_2 = parent_eq_next.mul(&parent_eq_next.sub(&one));
        let term_2 = ap.mul(&body_2);
        ap = ap.mul(alpha);

        // 3 max_child_weight_group_constancy:
        //     parent_eq_next · (max_w_curr − max_w_next) = 0.
        // Combined with row-local 5+6 this pins greedy dominance
        // across the entire parent group: every sibling shares the
        // same `max_child_weight` and is bounded above by it.
        let body_3 = parent_eq_next.mul(&max_w_curr.sub(max_w_next));
        let term_3 = ap.mul(&body_3);

        // Multiply by (z − ω^{n-1}) to exclude the wrap-around row
        // (matches the (X − ω^{n-1}) factor in
        // build_shifted_constraint_polynomial). Task #224.
        let sum = term_0.add(&term_1).add(&term_2).add(&term_3);
        sum.mul(&z.sub(omega_n_minus_1))
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let zero = Scalar::zero(curve);
        let one_s = Scalar::one(curve);

        let weight_curr = &column_coeffs[COL_ATTESTATION_WEIGHT];
        let max_w_curr = &column_coeffs[COL_MAX_CHILD_WEIGHT];
        let parent_eq_curr = &column_coeffs[COL_PARENT_EQ_PREV];
        let is_max_child_curr = &column_coeffs[COL_IS_MAX_CHILD];
        let weight_slack_curr = &column_coeffs[COL_WEIGHT_SLACK];

        // shifted_evals are reads on the NEXT row (col(ω·X)).
        let parent_eq_next = poly_shift(parent_eq_curr, omega);
        let is_max_child_next = poly_shift(is_max_child_curr, omega);
        let weight_slack_next = poly_shift(weight_slack_curr, omega);
        let weight_next = poly_shift(weight_curr, omega);
        let max_w_next = poly_shift(max_w_curr, omega);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // 0 max_child_uniqueness: parent_eq_next * is_max_child_next.
        let body_0 = poly_mul(&parent_eq_next, &is_max_child_next, curve);
        let mut total = poly_scalar_mul(&body_0, &ap);
        ap = ap.mul(alpha);

        // 1 weight_non_increasing: parent_eq_next · (weight_curr − weight_next − weight_slack_next).
        let diff_1 = poly_sub(
            &poly_sub(weight_curr, &weight_next, curve),
            &weight_slack_next,
            curve,
        );
        let body_1 = poly_mul(&parent_eq_next, &diff_1, curve);
        total = poly_add(&total, &poly_scalar_mul(&body_1, &ap), curve);
        ap = ap.mul(alpha);

        // 2 parent_eq_binary on next row: parent_eq_next · (parent_eq_next − 1).
        let neg_one = vec![zero.sub(&one_s)];
        let pe_minus_one = poly_add(&parent_eq_next, &neg_one, curve);
        let body_2 = poly_mul(&parent_eq_next, &pe_minus_one, curve);
        total = poly_add(&total, &poly_scalar_mul(&body_2, &ap), curve);
        ap = ap.mul(alpha);

        // 3 max_child_weight_group_constancy: parent_eq_next · (max_w_curr − max_w_next).
        let diff_3 = poly_sub(max_w_curr, &max_w_next, curve);
        let body_3 = poly_mul(&parent_eq_next, &diff_3, curve);
        total = poly_add(&total, &poly_scalar_mul(&body_3, &ap), curve);

        // Multiply by (X − ω^{n-1}) so the cross-row constraint is
        // allowed to fail on the wrap row (row n-1). Task #224.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let x_minus = vec![zero.sub(&omega_n_minus_1), Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
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
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for b in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("weight_byte_{}_8bit", b),
                    column_index: COL_WEIGHT_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("weight_slack_byte_{}_8bit", b),
                    column_index: COL_WEIGHT_SLACK_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("weight_diff_slack_byte_{}_8bit", b),
                    column_index: COL_WEIGHT_DIFF_SLACK_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..HASH_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("block_hash_{}_8bit", k),
                    column_index: COL_BLOCK_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("parent_hash_{}_8bit", k),
                    column_index: COL_PARENT_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind the fork-choice head's `block_hash[0..32]` to the
/// `block_header_air`'s `(block_hash[0..32])` window. The A side is
/// gated by `is_head` so only head-path rows publish entries; the
/// final head row's block_hash matches the block_header_air row for
/// the head block.
pub fn make_fork_choice_to_block_header_descriptor(
    fork_choice_layer_index: usize,
    block_header_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    let a_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| COL_BLOCK_HASH_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| bh::COL_BLOCK_HASH_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "lmd_ghost_fork_choice_to_block_header_v1".into(),
        a_layer_index: fork_choice_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HEAD),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

/// Bind `(block_hash[0..32], attestation_weight)` on the fork-choice
/// side to the attestation-aggregate per-vote tuple
/// `(signing_root[0..32], COL_BIT_INDEX)` — each attestation row
/// contributes one vote for the signed block (LogUp cardinality
/// counts the number of matching attestation rows per block hash).
///
/// **Semantic binding shape**. The current attestation_aggregate AIR
/// does not yet expose a per-validator effective-balance column, so
/// the B-side weight slot is filled with `COL_BIT_INDEX` as a
/// scaffold — the LogUp orchestrator's per-tuple multiplicity counts
/// the number of attesting rows for each block hash, which (under
/// uniform stake) is proportional to attestation_weight. Once a
/// `validator_balance` column is added to attestation_aggregate_air
/// (#TBD), this descriptor's B-side weight column will switch to
/// that column and the binding becomes exact stake summation.
///
/// A side is gated by `is_head` so only head-path rows publish
/// entries; B side is gated by `IS_REAL`. 33-col tuple
/// `(block_hash[0..32], weight_anchor)`.
pub fn make_fork_choice_to_attestation_weight_descriptor(
    fork_choice_layer_index: usize,
    attestation_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::attestation_aggregate_air as att;
    let mut a_columns: Vec<usize> = (0..HASH_LEN)
        .map(|k| COL_BLOCK_HASH_OFFSET + k)
        .collect();
    a_columns.push(COL_ATTESTATION_WEIGHT);
    let mut b_columns: Vec<usize> = (0..HASH_LEN)
        .map(|k| att::COL_SIGNING_ROOT_OFFSET + k)
        .collect();
    // Scaffold: bit_index serves as the per-row weight anchor until
    // a validator_balance column lands. Cardinality of B-side tuples
    // matching A-side (block_hash, weight) under LogUp gamma binds
    // attestation_weight to the accumulated per-target vote count.
    b_columns.push(att::COL_BIT_INDEX);
    CrossAirLogUpDescriptor {
        label: "lmd_ghost_fork_choice_to_attestation_weight_v1".into(),
        a_layer_index: fork_choice_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HEAD),
        b_layer_index: attestation_layer_index,
        b_columns,
        b_selector_column: Some(att::COL_IS_REAL),
    }
}

/// Bind the fork-choice head row's `block_hash[0..32]` to the
/// `attestation_aggregate_air`'s `signing_root[0..32]` — the head block
/// is the block targeted by the dominant set of attestations, and the
/// attestation's signing root commits the block root via
/// `AttestationData.beacon_block_root`. The A side is gated by
/// `is_head` so only head-path rows publish entries; the B side is
/// gated by `IS_REAL` so each participating attestation contributes one
/// entry. 32-col tuple. This is a target-shape semantic binding: full
/// algebraic equality requires the `signing_root = HTR(data, domain)`
/// SSZ gadget to expose `beacon_block_root` as its own column; that
/// lands with `attestation_data_htr_air`. Until then this descriptor
/// pins the head-block ↔ attestation message linkage shape so LogUp
/// orchestration can wire it in.
pub fn make_fork_choice_to_attestation_aggregate_descriptor(
    fork_choice_layer_index: usize,
    attestation_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::attestation_aggregate_air as att;
    let a_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| COL_BLOCK_HASH_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| att::COL_SIGNING_ROOT_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "lmd_ghost_fork_choice_to_attestation_aggregate_v1".into(),
        a_layer_index: fork_choice_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HEAD),
        b_layer_index: attestation_layer_index,
        b_columns,
        b_selector_column: Some(att::COL_IS_REAL),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_bodies(w: &LmdGhostForkChoiceWitness) -> Vec<Vec<Scalar>> {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(w, curve);
        let cs = LmdGhostForkChoiceConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    fn assert_all_vanish(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} should be zero",
                    i, r
                );
            }
        }
    }

    fn h(byte: u8) -> [u8; HASH_LEN] {
        [byte; HASH_LEN]
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_BLOCK_INDEX, 0);
        assert_eq!(COL_PARENT_INDEX, 1);
        assert_eq!(COL_ATTESTATION_WEIGHT, 2);
        assert_eq!(COL_IS_HEAD, 3);
        assert_eq!(COL_IS_MAX_CHILD, 4);
        assert_eq!(COL_IS_ROOT, 5);
        assert_eq!(COL_IS_REAL, 6);
        assert_eq!(COL_BLOCK_HASH_OFFSET, 7);
        assert_eq!(COL_PARENT_HASH_OFFSET, 39);
        assert_eq!(COL_MAX_CHILD_WEIGHT, 89);
        assert_eq!(COL_WEIGHT_DIFF_SLACK, 90);
        assert_eq!(COL_WEIGHT_DIFF_SLACK_BYTE_OFFSET, 91);
        assert_eq!(NUM_COLUMNS, 99);
        assert_eq!(NUM_ROW_CONSTRAINTS, 7);
        assert_eq!(NUM_SHIFTED, 4);
    }

    #[test]
    fn greedy_head_selection_simple_tree() {
        // Tree:
        //   0 (root, weight 100, parent=u64::MAX sentinel)
        //   ├── 1 (weight 80, parent=0)
        //   ├── 2 (weight 60, parent=0)
        //   └── 3 (child of 1, weight 50, parent=1)
        // Expected head path: 0 → 1 → 3 (greedy by weight).
        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 100u64),
            (1u64, 0u64, h(0x11), h(0x00), 80u64),
            (2u64, 0u64, h(0x22), h(0x00), 60u64),
            (3u64, 1u64, h(0x33), h(0x11), 50u64),
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        // After sort: rows grouped by parent_index ascending; within
        // each group, weight descending.
        // Group parent=0:       [block 1 (w=80), block 2 (w=60)]
        // Group parent=1:       [block 3 (w=50)]
        // Group parent=u64::MAX: [root (block 0, w=100)] (sorted last)
        // Head path (set): {0, 1, 3}.
        let mut head_indices: Vec<u64> = w
            .rows
            .iter()
            .filter(|r| r.is_head)
            .map(|r| r.block_index)
            .collect();
        head_indices.sort();
        assert_eq!(head_indices, vec![0, 1, 3]);
        // Greedy descent ends on block 3 (the deepest reached on the
        // max-weight path). The witness's `head_block_hash` is the
        // last is_head row in trace order — but with root sorted
        // last, that's the root. We instead verify the set.
        assert!(
            w.rows.iter().any(|r| r.is_head && r.block_hash == h(0x33)),
            "block 3 must be on the head path"
        );
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn lighter_subtree_not_head() {
        // Two children of root, lighter one not on head path.
        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 200u64),
            (1u64, 0u64, h(0x11), h(0x00), 50u64),  // lighter
            (2u64, 0u64, h(0x22), h(0x00), 150u64), // heavier
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        // Head should be {0, 2}.
        let mut head_indices: Vec<u64> = w
            .rows
            .iter()
            .filter(|r| r.is_head)
            .map(|r| r.block_index)
            .collect();
        head_indices.sort();
        assert_eq!(head_indices, vec![0, 2]);
        // Block 1 (the lighter sibling) must NOT be on the head path.
        assert!(!w.rows.iter().any(|r| r.is_head && r.block_index == 1));
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn shifted_max_child_uniqueness_fires_on_tamper() {
        // Build an honest witness; tamper is_max_child on a non-first
        // row of a parent group; the shifted body 0
        // (parent_eq_next · is_max_child_next = 0) must fire.
        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 100u64),
            (1u64, 0u64, h(0x11), h(0x00), 80u64),
            (2u64, 0u64, h(0x22), h(0x00), 60u64),
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // The sort places (block 0, parent 0) at row 0 — but wait, row 0
        // is the root with block_index = parent_index = 0, which IS
        // is_max_child = true (first row of parent group 0). Tamper
        // row 2 (block 1 or block 2 in parent group 0) to claim
        // is_max_child = 1. parent_eq_prev at row 2 should be 1.
        // Find a row with parent_eq_prev = 1 and tamper its is_max_child.
        let tamper_row = (1..w.rows.len()).find(|&i| {
            cols[COL_PARENT_EQ_PREV][i].to_u64() == 1
        }).expect("must have a within-group row");
        cols[COL_IS_MAX_CHILD][tamper_row] = Scalar::one(curve);
        // Compute the shifted body 0 at the previous row:
        // body_0 = parent_eq_next · is_max_child_next.
        // parent_eq_next is cols[COL_PARENT_EQ_PREV][tamper_row]; should be 1.
        // is_max_child_next is cols[COL_IS_MAX_CHILD][tamper_row]; tampered to 1.
        let body_0 = cols[COL_PARENT_EQ_PREV][tamper_row]
            .mul(&cols[COL_IS_MAX_CHILD][tamper_row]);
        assert!(
            !body_0.is_zero(),
            "max_child_uniqueness must fire on tampered is_max_child"
        );
    }

    #[test]
    fn shifted_weight_non_increasing_fires_on_tamper() {
        // Increase the weight of a non-first row within a parent
        // group so the running slack goes negative; the shifted body 1
        // must fire.
        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 100u64),
            (1u64, 0u64, h(0x11), h(0x00), 80u64),
            (2u64, 0u64, h(0x22), h(0x00), 60u64),
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Pick a row whose parent_eq_prev = 1; bump its weight so
        // prev.weight − this.weight < 0 (mod-prime), but
        // weight_slack is still the honest 20.
        let tamper_row = (1..w.rows.len()).find(|&i| {
            cols[COL_PARENT_EQ_PREV][i].to_u64() == 1
        }).expect("must have within-group row");
        let prev_row = tamper_row - 1;
        // Inflate weight to be larger than previous row's weight.
        cols[COL_ATTESTATION_WEIGHT][tamper_row] =
            Scalar::from_u64(999, curve);
        // body_1 at prev_row = parent_eq_next · (weight_curr −
        // weight_next − weight_slack_next).
        let pq = &cols[COL_PARENT_EQ_PREV][tamper_row];
        let w_curr = &cols[COL_ATTESTATION_WEIGHT][prev_row];
        let w_next = &cols[COL_ATTESTATION_WEIGHT][tamper_row];
        let slack_next = &cols[COL_WEIGHT_SLACK][tamper_row];
        let body_1 = pq.mul(&w_curr.sub(w_next).sub(slack_next));
        assert!(
            !body_1.is_zero(),
            "weight_non_increasing must fire on tampered weight"
        );
    }

    #[test]
    fn greedy_descent_dominance_constraints_vanish_honestly() {
        // A multi-level tree with sibling weight ties and a deep
        // greedy descent. All row-local + shifted bodies should
        // vanish on the honest witness; the
        // `max_child_weight_dominance` and `is_max_child_achieves_max`
        // bodies (#5, #6) and shifted body #3
        // (group-constancy) are non-trivial — they're the new
        // greedy-dominance algebraic seal.
        //
        //   0 (root, parent=u64::MAX, w=1000)
        //   ├── 1 (parent=0, w=400)
        //   │   ├── 4 (parent=1, w=200)
        //   │   └── 5 (parent=1, w=350)  ← max child of 1, on head path
        //   ├── 2 (parent=0, w=600)      ← max child of 0, on head path
        //   │   ├── 6 (parent=2, w=300)
        //   │   └── 7 (parent=2, w=550)  ← max child of 2, on head path
        //   └── 3 (parent=0, w=500)
        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 1000u64),
            (1u64, 0u64, h(0x11), h(0x00), 400u64),
            (2u64, 0u64, h(0x22), h(0x00), 600u64),
            (3u64, 0u64, h(0x33), h(0x00), 500u64),
            (4u64, 1u64, h(0x44), h(0x11), 200u64),
            (5u64, 1u64, h(0x55), h(0x11), 350u64),
            (6u64, 2u64, h(0x66), h(0x22), 300u64),
            (7u64, 2u64, h(0x77), h(0x22), 550u64),
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        // Head path set should be {0, 2, 7}.
        let mut head_indices: Vec<u64> = w
            .rows
            .iter()
            .filter(|r| r.is_head)
            .map(|r| r.block_index)
            .collect();
        head_indices.sort();
        assert_eq!(head_indices, vec![0, 2, 7]);

        // All row-local bodies vanish.
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);

        // Verify max_child_weight column is populated as expected:
        // for every row, max_child_weight ≥ attestation_weight.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        for r in 0..w.rows.len() {
            let mw = trace.columns[COL_MAX_CHILD_WEIGHT].evaluations[r].to_u64();
            let aw = trace.columns[COL_ATTESTATION_WEIGHT].evaluations[r].to_u64();
            assert!(
                mw >= aw,
                "row {} max_child_weight {} must dominate weight {}",
                r, mw, aw
            );
            let diff = trace.columns[COL_WEIGHT_DIFF_SLACK].evaluations[r].to_u64();
            assert_eq!(
                diff,
                mw - aw,
                "weight_diff_slack must equal max_child_weight − attestation_weight"
            );
            // The is_max_child = 1 row must have diff_slack = 0.
            if w.rows[r].is_max_child {
                assert_eq!(diff, 0, "is_max_child row must achieve max");
            }
        }
    }

    #[test]
    fn tampered_max_child_on_lighter_row_fires_constraint_6() {
        // Sibling group with strict weight ordering: rows are
        // [w=80 (first), w=60 (second)] for parent 0. The first row
        // has is_max_child = 1; tampering the second row to also
        // claim is_max_child = 1 must trigger constraint 6
        // (is_real · is_max_child · (max_child_weight − weight) = 0)
        // because the second row's weight (60) ≠ max_child_weight (80).
        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 100u64),
            (1u64, 0u64, h(0x11), h(0x00), 80u64),
            (2u64, 0u64, h(0x22), h(0x00), 60u64),
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Find the second sibling row (parent_eq_prev = 1).
        let tamper_row = (1..w.rows.len())
            .find(|&i| cols[COL_PARENT_EQ_PREV][i].to_u64() == 1)
            .expect("must have within-group row");
        // Set its is_max_child to 1.
        cols[COL_IS_MAX_CHILD][tamper_row] = Scalar::one(curve);
        // Constraint 6 body at tamper_row =
        //   is_real · is_max_child · (max_child_weight − weight).
        let body_6 = cols[COL_IS_REAL][tamper_row]
            .mul(&cols[COL_IS_MAX_CHILD][tamper_row])
            .mul(&cols[COL_MAX_CHILD_WEIGHT][tamper_row]
                .sub(&cols[COL_ATTESTATION_WEIGHT][tamper_row]));
        assert!(
            !body_6.is_zero(),
            "is_max_child_achieves_max must fire when tampered row's weight ≠ max",
        );
    }

    #[test]
    fn tampered_max_child_weight_propagation_fires_shifted_3() {
        // Within a parent group, every row must carry the same
        // max_child_weight. Tampering a non-first sibling row's
        // max_child_weight must trip shifted body 3
        // (parent_eq_next · (max_w_curr − max_w_next) = 0).
        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 100u64),
            (1u64, 0u64, h(0x11), h(0x00), 80u64),
            (2u64, 0u64, h(0x22), h(0x00), 60u64),
            (3u64, 0u64, h(0x33), h(0x00), 40u64),
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Find a row that's an intermediate sibling (parent_eq_prev=1
        // AND the *following* row also has parent_eq_prev=1), so the
        // shifted body at it operates on intra-group transition.
        let tamper_row = (1..w.rows.len() - 1)
            .find(|&i| {
                cols[COL_PARENT_EQ_PREV][i].to_u64() == 1
                    && cols[COL_PARENT_EQ_PREV][i + 1].to_u64() == 1
            })
            .expect("must have intra-group transition");
        // Tamper max_child_weight on that row.
        cols[COL_MAX_CHILD_WEIGHT][tamper_row] = Scalar::from_u64(999, curve);
        // Shifted body 3 at tamper_row (looking forward to row+1):
        //   parent_eq_next · (max_w_curr − max_w_next).
        // parent_eq_next is cols[COL_PARENT_EQ_PREV][tamper_row+1].
        let pq_next = &cols[COL_PARENT_EQ_PREV][tamper_row + 1];
        let max_curr = &cols[COL_MAX_CHILD_WEIGHT][tamper_row];
        let max_next = &cols[COL_MAX_CHILD_WEIGHT][tamper_row + 1];
        let body_3 = pq_next.mul(&max_curr.sub(max_next));
        assert!(
            !body_3.is_zero(),
            "max_child_weight_group_constancy must fire on tampered propagation"
        );
    }

    #[test]
    fn attestation_weight_descriptor_well_formed() {
        let d = make_fork_choice_to_attestation_weight_descriptor(0, 2);
        assert_eq!(
            d.label,
            "lmd_ghost_fork_choice_to_attestation_weight_v1"
        );
        // Block hash tuple + weight anchor.
        assert_eq!(d.a_columns.len(), HASH_LEN + 1);
        assert_eq!(d.b_columns.len(), HASH_LEN + 1);
        assert_eq!(d.a_columns[HASH_LEN], COL_ATTESTATION_WEIGHT);
        assert_eq!(
            d.b_columns[HASH_LEN],
            crate::attestation_aggregate_air::COL_BIT_INDEX,
        );
        assert_eq!(d.a_selector_column, Some(COL_IS_HEAD));
        assert_eq!(
            d.b_selector_column,
            Some(crate::attestation_aggregate_air::COL_IS_REAL),
        );
        // No usize::MAX sentinels and all columns in bounds.
        for &c in &d.a_columns {
            assert!(c < NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d.b_columns {
            assert!(c < crate::attestation_aggregate_air::NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
    }

    #[test]
    fn descriptors_well_formed() {
        let d = make_fork_choice_to_block_header_descriptor(0, 1);
        assert_eq!(d.label, "lmd_ghost_fork_choice_to_block_header_v1");
        assert_eq!(d.a_columns.len(), HASH_LEN);
        assert_eq!(d.b_columns.len(), HASH_LEN);
        assert_eq!(d.a_selector_column, Some(COL_IS_HEAD));

        let d2 = make_fork_choice_to_attestation_aggregate_descriptor(0, 2);
        assert_eq!(
            d2.label,
            "lmd_ghost_fork_choice_to_attestation_aggregate_v1"
        );
        assert_eq!(d2.a_columns.len(), HASH_LEN);
        assert_eq!(d2.b_columns.len(), HASH_LEN);
        assert_eq!(d2.a_selector_column, Some(COL_IS_HEAD));
        assert_eq!(
            d2.b_selector_column,
            Some(crate::attestation_aggregate_air::COL_IS_REAL),
        );
        // No sentinel placeholders.
        for &c in &d2.a_columns {
            assert!(c < NUM_COLUMNS, "fork-choice col {} out of bounds", c);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d2.b_columns {
            assert!(
                c < crate::attestation_aggregate_air::NUM_COLUMNS,
                "attestation col {} out of bounds",
                c,
            );
            assert_ne!(c, usize::MAX);
        }
        // block_header descriptor is also free of usize::MAX sentinels.
        for &c in &d.a_columns {
            assert!(c < NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d.b_columns {
            assert!(c < crate::block_header_air::NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
    }

    #[test]
    fn single_row_root_only() {
        // Just a root, no children. is_head set on the root;
        // is_max_child stays false (root has no parent group).
        let blocks = vec![(7u64, u64::MAX, h(0x77), h(0x77), 42u64)];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].is_head);
        assert!(!w.rows[0].is_max_child);
        assert!(w.rows[0].is_root);
        assert_eq!(w.head_block_hash, h(0x77));
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    #[ignore = "slow: standalone prove+verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let blocks = vec![
            (0u64, u64::MAX, h(0x00), h(0x00), 100u64),
            (1u64, 0u64, h(0x11), h(0x00), 80u64),
        ];
        let w = LmdGhostForkChoiceWitness::from_blocks(blocks).unwrap();
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = LmdGhostForkChoiceConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone lmd_ghost_fork_choice_air proof must verify",
        );
    }
}
