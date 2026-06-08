//! [`VmConstraintSystem`] wiring for the MPT inclusion-proof AIR.
//!
//! This module adapts the MPT inclusion AIR defined in [`crate::mpt_air`]
//! into the shape required by the generic [`prove_with_scheme`] /
//! [`verify_with_scheme`] pipeline. It provides:
//!
//! - `MptInclusionConstraintSystem`: the trait implementer. Construct with
//!   `MptInclusionConstraintSystem::new(num_rows)` where `num_rows` is the
//!   real, pre-padding trace height (= `inclusion_witness(...).len()`).
//! - `build_trace_polynomials_from_rows`: helper that lifts a
//!   `Vec<InclusionRow>` into the `TracePolynomials` the pipeline expects.
//!
//! # Constraint layout
//!
//! 4 row-local consolidated categories (label → index):
//!
//!   0. `is_terminal_binary`            — `is_terminal · (is_terminal − 1) = 0`
//!   1. `terminal_is_leaf`              — `is_terminal · (node_kind − 2) = 0`
//!   2. `node_kind_set`                 — `node_kind · (node_kind − 1) · (node_kind − 2) = 0`
//!   3. `path_nibble_zero_on_non_branch` — `node_kind · path_nibble = 0`
//!
//! 2 shifted (cross-row) constraint categories, β-RLC'd over their
//! component lanes:
//!
//!   0. `parent_chain_per_byte`  — for every byte i ∈ 0..32:
//!      `parent_hash_byte_i(ω·X) − node_hash_byte_i(X) = 0`. The
//!      verifier sums the bodies via β-powers (β = α).
//!   1. `depth_increment`        — `depth(ω·X) − depth(X) − 1 = 0`,
//!      attached to the same shifted batch (β-power follows the 32
//!      parent-chain bodies).
//!
//! Both shifted bodies are excluded on the last real row and on the
//! domain wrap-around (boundary-row exclusion product).
//!
//! # Padding strategy
//!
//! [`VmConstraintSystem::padding_selector_column`] returns `None`. Padding
//! rows carry all-zero columns: every row-local body trivially vanishes
//! (`0 · (0 − 1) = 0` for the binary check; `0 · (0 − 2) = 0` for the
//! terminal-is-leaf check). The cross-row body uses boundary exclusion
//! to skip the last real row (where the next row's `parent_hash = 0`
//! and `depth = 0` would yield non-zero δ) and the domain wrap.
//!
//! # Soundness gaps (deferred)
//!
//! This is the **structural** AIR; several semantic constraints are
//! stubbed out and listed here so reviewers see them at a glance:
//!
//! 1. **Per-node hash check** `node_hash == keccak256(node_rlp)` — out
//!    of scope; delegated to a separate `keccak_constraints` AIR via
//!    cross-AIR linkage (future task).
//! 2. **RLP-decoding consistency**: the AIR does not enforce that the
//!    declared `node_kind` matches the RLP-decoded shape, that
//!    `path_nibble` matches the nibble consumed by a branch row's
//!    proof index, nor that the *child slot* picked from a branch row
//!    references the next row's `node_hash`. Host-side
//!    [`crate::mpt_air::inclusion_witness`] derives the witness rows
//!    correctly; the AIR currently trusts that derivation.
//! 3. **Root binding**: row 0's `parent_hash` is the trie root and must
//!    be bound externally (e.g. block-header `state_root`).
//! 4. **`node_kind ∈ {0,1,2}`**: now enforced algebraically via the
//!    degree-3 `node_kind_set` row constraint. The 2-bit lookup remains
//!    as a defensive belt-and-suspenders bound.
//! 5. **`path_nibble = 0` on non-branch rows**: now enforced
//!    algebraically via the degree-2 `path_nibble_zero_on_non_branch`
//!    row constraint (`node_kind · path_nibble = 0`). On branch rows
//!    (`node_kind = 0`) `path_nibble` ranges over 0..16 freely; on
//!    extension or leaf rows it is pinned to zero, matching the witness
//!    builder's "0 otherwise" invariant.
//!
//! # Lookup declarations
//!
//! - Each of the 64 node-hash + parent-hash byte columns is declared as
//!   an 8-bit range check.
//! - `NODE_KIND` is declared as a 2-bit range check (slack documented).
//! - `PATH_NIBBLE` is declared as a 4-bit range check.
//! - `IS_TERMINAL` is declared as a 1-bit range check (binary; the
//!   binary constraint above is a redundant algebraic check for cheap
//!   verifier rejection).
//! - `DEPTH` has no explicit range check (the depth-increment cross-row
//!   constraint pins it to a contiguous run [0, num_rows)).

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::mpt_air::{col, inclusion_witness, InclusionRow, HASH_BYTES, MAX_RLP_LEN};
use crate::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Number of consolidated row-local constraint categories.
///
/// Constraint 16: leaf-row equality between `claimed_leaf_value_bytes`
/// and `value_byte`, β-RLC over 32 bytes, gated by `is_terminal`.
/// Constraint 17: same pattern for `claimed_leaf_key_bytes` ↔
/// `key_path_bytes`.
///
/// Constraint 18 (full multi-row nibble binding): β-RLC over 32 byte
/// decompositions `byte_i = 16 * nib_{2i} + nib_{2i+1}` for i ∈ 0..32.
///
/// Constraint 19: β-RLC over 64 binary checks for `is_depth_eq[k]`.
///
/// Constraint 20: `Σ is_depth_eq[k] - 1 = 0` (sum-to-one).
///
/// Constraint 21: `Σ k * is_depth_eq[k] - depth = 0` (linear-combination
/// pins the one-hot indicator to the actual depth value).
///
/// Constraint 22 (PER-DEPTH NIBBLE EQUALITY — the final algebraic
/// piece): for every branch row at depth d, the consumed `path_nibble`
/// must equal nibble d of `claimed_full_key`. β-RLC'd over 64
/// positions, with the per-depth selectors picking the right nibble:
///   `is_branch_doubled * (path_nibble - Σ is_depth_eq[k] * claimed_full_key_nibbles[k]) = 0`
/// where `is_branch_doubled = (1 - node_kind) * (2 - node_kind)` avoids
/// the field inverse. Combined with constraints 19-21 (forcing
/// is_depth_eq to be the correct one-hot indicator of depth) and
/// constraint 18 (forcing nibbles to be the correct decomposition of
/// claimed_full_key_bytes), this algebraically binds:
///
///   "at every branch row in the inclusion chain, the consumed path
///    nibble equals nibble_at_that_depth of the claimed full key"
///
/// — completing the multi-row storage chain soundness.
pub const NUM_ROW_CONSTRAINTS: usize = 23;

/// Number of consolidated cross-row (shifted) constraint categories.
///
/// Body 0: per-byte parent-chain check (32 sub-bodies) + depth-increment
/// check (1 sub-body), β-RLC'd into one body. Excludes transitions
/// from the last real row through the domain wrap.
///
/// Body 1: Phase 4b extension chain binding — pins the extension row's
/// committed child hash (in `VALUE_BYTE`) to the next MPT row's
/// `NODE_HASH`. β-RLC over 32 byte sub-bodies, gated by
/// `IS_PHASE4_EXT_SHAPE` so non-extension rows vanish naturally.
/// Same boundary exclusion as body 0.
///
/// Body 2: Phase 8b branch chain binding — pins the path-nibble-selected
/// child hash on a Phase 8a branch row to the next MPT row's NODE_HASH:
///     `IS_PHASE8_BRANCH_01_SHAPE(X) · Σ β^i ·
///         ((1 − PATH_NIBBLE) · BRANCH_CHILD_0_HASH_BYTE_i +
///          PATH_NIBBLE       · BRANCH_CHILD_1_HASH_BYTE_i
///          − NODE_HASH_byte_i(ω·X))`
/// Combined with the row-local `path_nibble_branch01_binarity`
/// (which pins PATH_NIBBLE ∈ {0, 1} on Phase 8a rows) and the
/// existing #91 NODE_HASH = keccak256(NODE_RLP) binding, the
/// branch-row inclusion chain is algebraically sound for the Phase
/// 8a shape.
///
/// Body 3: Phase 9 branch chain binding (slots 0 and 5) — Lagrange
/// interpolation at points (0, 5):
///     `IS_PHASE9_BRANCH_05_SHAPE(X) · Σ β^i ·
///         ((5 − PATH_NIBBLE) · BRANCH_CHILD_0_HASH_BYTE_i +
///          (PATH_NIBBLE − 0) · BRANCH_CHILD_1_HASH_BYTE_i −
///          5 · NODE_HASH_byte_i(ω·X))`
/// Combined with the row-local `path_nibble_branch05_pinning`
/// (PATH_NIBBLE ∈ {0, 5}), the path-nibble-selected slot's child
/// hash is pinned to the next row's NODE_HASH.
///
/// Body 4: Phase 10 branch chain binding (slots 1 and 5) — Lagrange
/// interpolation at points (1, 5), demonstrating the pattern
/// generalizes to slot pairs not anchored at 0:
///     `IS_PHASE10_BRANCH_15_SHAPE(X) · Σ β^i ·
///         ((5 − PATH_NIBBLE) · BRANCH_CHILD_0_HASH_BYTE_i +
///          (PATH_NIBBLE − 1) · BRANCH_CHILD_1_HASH_BYTE_i −
///          4 · NODE_HASH_byte_i(ω·X))`
/// (Multiplier `4` is `j − i = 5 − 1`. Combined with the row-local
/// `path_nibble_branch15_pinning` pinning PATH_NIBBLE ∈ {1, 5}.)
///
/// Body 5 (added 2026-05-13): cross-row constancy of
/// `claimed_leaf_value_bytes` within an MPT chain. β-RLC over 32 byte
/// sub-bodies. Body 6: same pattern for `claimed_leaf_key_bytes`.
/// Body 7: same pattern for `claimed_full_key_bytes` (multi-row
/// prefix accumulator scaffold — propagates the claimed full trie
/// key constant across the chain. Per-branch nibble-equality binding
/// is a focused follow-up requiring per-depth selectors).
pub const NUM_SHIFTED: usize = 8;

// ──── Constraint system ────────────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the MPT inclusion AIR.
pub struct MptInclusionConstraintSystem {
    /// Number of real trace rows (before padding). The padded domain
    /// size is determined by [`TracePolynomials`] to the next power of
    /// two ≥ `num_rows`.
    pub num_rows: usize,
    /// The domain generator ω for the trace's padded domain. When
    /// `Some`, `evaluate_shifted_at_point` excludes the last real row
    /// (`X − ω^{num_rows − 1}`) in addition to the wrap row
    /// (`X − ω^{n−1}`). When `None`, only the wrap-around factor is
    /// excluded — sound only when `num_rows == domain_size`.
    pub omega: Option<Scalar>,
    /// The padded domain size (power of two ≥ `num_rows`).
    pub domain_size: Option<u64>,
}

impl MptInclusionConstraintSystem {
    /// Construct an MPT inclusion constraint system for a trace of
    /// `num_rows` real rows (= `inclusion_witness(...).len()`).
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    /// Attach the domain generator `omega` and `domain_size` for the
    /// scheme/trace this constraint system is paired with.
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

// ──── Trace construction helpers ───────────────────────────────────────

/// Allocate a zeroed scalar trace with `padded_size` rows and
/// `col::NUM_COLUMNS` columns, targeting `curve`.
fn alloc_trace(padded_size: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..col::NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded_size])
        .collect()
}

/// Populate a scalar trace from an `InclusionRow` slice. Mirrors
/// [`populate_trace`] but writes directly into the field rather than
/// into a `u64` staging buffer.
fn populate_scalar_trace(rows: &[InclusionRow], columns: &mut [Vec<Scalar>], curve: CurveType) {
    debug_assert!(columns.len() >= col::NUM_COLUMNS);
    for (row_idx, row) in rows.iter().enumerate() {
        for i in 0..HASH_BYTES {
            columns[col::NODE_HASH_OFFSET + i][row_idx] =
                Scalar::from_u64(row.node_hash[i] as u64, curve);
            columns[col::PARENT_HASH_OFFSET + i][row_idx] =
                Scalar::from_u64(row.parent_hash[i] as u64, curve);
        }
        columns[col::NODE_KIND][row_idx] = Scalar::from_u64(row.node_kind as u64, curve);
        columns[col::PATH_NIBBLE][row_idx] = Scalar::from_u64(row.path_nibble as u64, curve);
        columns[col::DEPTH][row_idx] = Scalar::from_u64(row.depth as u64, curve);
        columns[col::IS_TERMINAL][row_idx] = Scalar::from_u64(row.is_terminal as u64, curve);
        for b in 0..MAX_RLP_LEN {
            columns[col::NODE_RLP_OFFSET + b][row_idx] =
                Scalar::from_u64(row.node_rlp[b] as u64, curve);
        }
        columns[col::NODE_RLP_LEN][row_idx] = Scalar::from_u64(row.node_rlp_len as u64, curve);
        for b in 0..32 {
            columns[col::KEY_PATH_BYTE_OFFSET + b][row_idx] =
                Scalar::from_u64(row.key_path_bytes[b] as u64, curve);
            columns[col::VALUE_BYTE_OFFSET + b][row_idx] =
                Scalar::from_u64(row.value[b] as u64, curve);
        }
        columns[col::IS_PHASE1_LEAF_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase1_leaf_shape as u64, curve);
        columns[col::IS_PHASE3_LEAF_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase3_leaf_shape as u64, curve);
        columns[col::IS_PHASE4_EXT_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase4_ext_shape as u64, curve);
        columns[col::LEADING_PATH_NIBBLE][row_idx] =
            Scalar::from_u64(row.leading_path_nibble as u64, curve);
        columns[col::IS_PHASE6_ODD_LEAF_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase6_odd_leaf_shape as u64, curve);
        columns[col::IS_PHASE7_LEAF_6N_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase7_leaf_6n_shape as u64, curve);
        columns[col::IS_PHASE8_BRANCH_01_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase8_branch_01_shape as u64, curve);
        for b in 0..32 {
            columns[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + b][row_idx] =
                Scalar::from_u64(row.branch_child_0_hash_bytes[b] as u64, curve);
            columns[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + b][row_idx] =
                Scalar::from_u64(row.branch_child_1_hash_bytes[b] as u64, curve);
        }
        columns[col::IS_PHASE9_BRANCH_05_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase9_branch_05_shape as u64, curve);
        columns[col::IS_PHASE10_BRANCH_15_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase10_branch_15_shape as u64, curve);
        columns[col::IS_PHASE11_LEAF_8N_SHAPE][row_idx] =
            Scalar::from_u64(row.is_phase11_leaf_8n_shape as u64, curve);
        columns[col::IS_ROOT][row_idx] = Scalar::from_u64(row.is_root as u64, curve);
        for b in 0..32 {
            columns[col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + b][row_idx] =
                Scalar::from_u64(row.claimed_leaf_value_bytes[b] as u64, curve);
            columns[col::CLAIMED_LEAF_KEY_BYTE_OFFSET + b][row_idx] =
                Scalar::from_u64(row.claimed_leaf_key_bytes[b] as u64, curve);
            columns[col::CLAIMED_FULL_KEY_BYTE_OFFSET + b][row_idx] =
                Scalar::from_u64(row.claimed_full_key_bytes[b] as u64, curve);
        }
        for k in 0..64 {
            columns[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + k][row_idx] =
                Scalar::from_u64(row.claimed_full_key_nibbles[k] as u64, curve);
            columns[col::IS_DEPTH_EQ_OFFSET + k][row_idx] =
                Scalar::from_u64(row.is_depth_eq[k] as u64, curve);
        }
    }
    // A2 step 1d-multirow padding: on rows past the witness, set
    // is_depth_eq[0] = 1 so the sum-to-one constraint vanishes.
    let one = Scalar::one(curve);
    let padded_size = columns[col::IS_DEPTH_EQ_OFFSET].len();
    for cell in columns[col::IS_DEPTH_EQ_OFFSET]
        .iter_mut()
        .skip(rows.len())
        .take(padded_size.saturating_sub(rows.len()))
    {
        *cell = one.clone();
    }
}

/// Build a [`TracePolynomials`] wrapping the MPT inclusion trace
/// produced by populating `rows` in order.
pub fn build_trace_polynomials_from_rows(
    rows: &[InclusionRow],
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let mut columns = alloc_trace(padded, curve);
    populate_scalar_trace(rows, &mut columns, curve);
    into_trace_polynomials(columns, num_rows, padded, curve)
}

/// Variant for direct (key, proof) inputs: builds the witness then
/// lifts it.
pub fn build_trace_polynomials_from_proof(
    key: &[u8],
    proof: &[Vec<u8>],
    curve: CurveType,
) -> TracePolynomials {
    let rows = inclusion_witness(key, proof);
    build_trace_polynomials_from_rows(&rows, curve)
}

fn into_trace_polynomials(
    columns: Vec<Vec<Scalar>>,
    num_rows: usize,
    padded: usize,
    curve: CurveType,
) -> TracePolynomials {
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

// ──── Helpers for scalar-point evaluation ──────────────────────────────

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

// ──── Scalar-point evaluation of each row-local body ───────────────────

/// 0. `is_terminal_binary`: `is_terminal · (is_terminal − 1) = 0`.
fn eval_is_terminal_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_TERMINAL];
    v.mul(&v.sub(&one))
}

/// 1. `terminal_is_leaf`: `is_terminal · (node_kind − 2) = 0`.
fn eval_terminal_is_leaf_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let two = Scalar::from_u64(2, curve);
    let it = &cols[col::IS_TERMINAL];
    let kind = &cols[col::NODE_KIND];
    it.mul(&kind.sub(&two))
}

/// 2. `node_kind_set`: `node_kind · (node_kind − 1) · (node_kind − 2) = 0`.
/// Constrains `node_kind ∈ {0, 1, 2}` algebraically — tighter than the
/// 2-bit range check that allowed `node_kind = 3` as slack.
fn eval_node_kind_set_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);
    let kind = &cols[col::NODE_KIND];
    kind.mul(&kind.sub(&one)).mul(&kind.sub(&two))
}

/// 3. `path_nibble_zero_on_non_branch`: `node_kind · path_nibble = 0`.
/// On branch rows (`node_kind = 0`) this is trivially satisfied for any
/// `path_nibble`. On extension or leaf rows (`node_kind ∈ {1, 2}`) it
/// forces `path_nibble = 0`, matching the witness builder's invariant
/// that `path_nibble` is "0 otherwise" (only meaningful for branches).
fn eval_path_nibble_zero_on_non_branch_at_point(cols: &[Scalar]) -> Scalar {
    let kind = &cols[col::NODE_KIND];
    let nibble = &cols[col::PATH_NIBBLE];
    kind.mul(nibble)
}

/// 4. `is_phase1_leaf_shape_binary`:
/// `IS_PHASE1_LEAF_SHAPE · (IS_PHASE1_LEAF_SHAPE − 1) = 0`. Required
/// because this column is the A-side selector for the cross-AIR LogUp
/// to `crate::mpt_rlp_air` — non-binary values would let a malicious
/// prover smuggle non-unit multiplicities through the multiset balance.
fn eval_is_phase1_leaf_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE1_LEAF_SHAPE];
    v.mul(&v.sub(&one))
}

/// 5. `is_phase3_leaf_shape_binary`:
/// `IS_PHASE3_LEAF_SHAPE · (IS_PHASE3_LEAF_SHAPE − 1) = 0`. Same
/// motivation as #4: this column is the A-side selector for the
/// cross-AIR LogUp to `crate::mpt_short_leaf_rlp_air`.
fn eval_is_phase3_leaf_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE3_LEAF_SHAPE];
    v.mul(&v.sub(&one))
}

/// 6. `is_phase4_ext_shape_binary`:
/// `IS_PHASE4_EXT_SHAPE · (IS_PHASE4_EXT_SHAPE − 1) = 0`. A-side
/// selector for the cross-AIR LogUp to `crate::mpt_extension_rlp_air`.
fn eval_is_phase4_ext_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE4_EXT_SHAPE];
    v.mul(&v.sub(&one))
}

/// 7. `is_phase6_odd_leaf_shape_binary`:
/// `IS_PHASE6_ODD_LEAF_SHAPE · (IS_PHASE6_ODD_LEAF_SHAPE − 1) = 0`.
/// A-side selector for the cross-AIR LogUp to
/// `crate::mpt_odd_leaf_rlp_air`.
fn eval_is_phase6_odd_leaf_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE6_ODD_LEAF_SHAPE];
    v.mul(&v.sub(&one))
}

/// 8. `is_phase7_leaf_6n_shape_binary`:
/// `IS_PHASE7_LEAF_6N_SHAPE · (IS_PHASE7_LEAF_6N_SHAPE − 1) = 0`.
/// A-side selector for the cross-AIR LogUp to
/// `crate::mpt_six_nibble_leaf_rlp_air`.
fn eval_is_phase7_leaf_6n_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE7_LEAF_6N_SHAPE];
    v.mul(&v.sub(&one))
}

/// 9. `is_phase8_branch_01_shape_binary`:
/// `IS_PHASE8_BRANCH_01_SHAPE · (IS_PHASE8_BRANCH_01_SHAPE − 1) = 0`.
/// A-side selector for the cross-AIR LogUp to
/// `crate::mpt_branch_2child_rlp_air`.
fn eval_is_phase8_branch_01_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE8_BRANCH_01_SHAPE];
    v.mul(&v.sub(&one))
}

/// 10. `path_nibble_branch01_binarity` (Phase 8b prerequisite):
/// `IS_PHASE8_BRANCH_01_SHAPE · PATH_NIBBLE · (PATH_NIBBLE − 1) = 0`.
/// On Phase 8a branch rows (where children only at slots 0 and 1),
/// the inclusion path's `path_nibble` MUST be 0 or 1 — the Phase 8b
/// shifted constraint's selection arithmetic
/// `(1 − PATH_NIBBLE) · child_0 + PATH_NIBBLE · child_1` is only
/// well-defined for those values.
fn eval_path_nibble_branch01_binarity_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let phase8 = &cols[col::IS_PHASE8_BRANCH_01_SHAPE];
    let nib = &cols[col::PATH_NIBBLE];
    phase8.mul(&nib.mul(&nib.sub(&one)))
}

/// 11. `is_phase9_branch_05_shape_binary`:
/// `IS_PHASE9_BRANCH_05_SHAPE · (IS_PHASE9_BRANCH_05_SHAPE − 1) = 0`.
fn eval_is_phase9_branch_05_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE9_BRANCH_05_SHAPE];
    v.mul(&v.sub(&one))
}

/// 12. `path_nibble_branch05_pinning` (Phase 9 prerequisite):
/// `IS_PHASE9_BRANCH_05_SHAPE · PATH_NIBBLE · (PATH_NIBBLE − 5) = 0`.
fn eval_path_nibble_branch05_pinning_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let five = Scalar::from_u64(5, curve);
    let phase9 = &cols[col::IS_PHASE9_BRANCH_05_SHAPE];
    let nib = &cols[col::PATH_NIBBLE];
    phase9.mul(&nib.mul(&nib.sub(&five)))
}

/// 13. `is_phase10_branch_15_shape_binary`:
/// `IS_PHASE10_BRANCH_15_SHAPE · (IS_PHASE10_BRANCH_15_SHAPE − 1) = 0`.
fn eval_is_phase10_branch_15_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE10_BRANCH_15_SHAPE];
    v.mul(&v.sub(&one))
}

/// 14. `path_nibble_branch15_pinning` (Phase 10 prerequisite):
/// `IS_PHASE10_BRANCH_15_SHAPE · (PATH_NIBBLE − 1) · (PATH_NIBBLE − 5) = 0`.
/// On Phase 10 branch rows (children at slots 1 and 5), PATH_NIBBLE
/// must be 1 or 5.
fn eval_path_nibble_branch15_pinning_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let five = Scalar::from_u64(5, curve);
    let phase10 = &cols[col::IS_PHASE10_BRANCH_15_SHAPE];
    let nib = &cols[col::PATH_NIBBLE];
    let nib_m1 = nib.sub(&one);
    let nib_m5 = nib.sub(&five);
    phase10.mul(&nib_m1.mul(&nib_m5))
}

/// 15. `is_phase11_leaf_8n_shape_binary`:
/// `IS_PHASE11_LEAF_8N_SHAPE · (IS_PHASE11_LEAF_8N_SHAPE − 1) = 0`.
fn eval_is_phase11_leaf_8n_shape_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[col::IS_PHASE11_LEAF_8N_SHAPE];
    v.mul(&v.sub(&one))
}

// ──── Polynomial-form builders for each row-local body ─────────────────

fn build_is_terminal_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_TERMINAL];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_terminal_is_leaf_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let two_poly = vec![Scalar::from_u64(2, curve)];
    let it = &cols[col::IS_TERMINAL];
    let kind = &cols[col::NODE_KIND];
    let kind_m2 = poly_sub(kind, &two_poly, curve);
    poly_mul(it, &kind_m2, curve)
}

fn build_node_kind_set_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let two_poly = vec![Scalar::from_u64(2, curve)];
    let kind = &cols[col::NODE_KIND];
    let kind_m1 = poly_sub(kind, &one_poly, curve);
    let kind_m2 = poly_sub(kind, &two_poly, curve);
    let kind_kind_m1 = poly_mul(kind, &kind_m1, curve);
    poly_mul(&kind_kind_m1, &kind_m2, curve)
}

fn build_path_nibble_zero_on_non_branch_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let kind = &cols[col::NODE_KIND];
    let nibble = &cols[col::PATH_NIBBLE];
    poly_mul(kind, nibble, curve)
}

fn build_is_phase1_leaf_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE1_LEAF_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_phase3_leaf_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE3_LEAF_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_phase4_ext_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE4_EXT_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_phase6_odd_leaf_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE6_ODD_LEAF_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_phase7_leaf_6n_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE7_LEAF_6N_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_phase8_branch_01_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE8_BRANCH_01_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_path_nibble_branch01_binarity_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let phase8 = &cols[col::IS_PHASE8_BRANCH_01_SHAPE];
    let nib = &cols[col::PATH_NIBBLE];
    let nib_m1 = poly_sub(nib, &one_poly, curve);
    let nib_times_nibm1 = poly_mul(nib, &nib_m1, curve);
    poly_mul(phase8, &nib_times_nibm1, curve)
}

fn build_is_phase9_branch_05_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE9_BRANCH_05_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_path_nibble_branch05_pinning_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let five_poly = vec![Scalar::from_u64(5, curve)];
    let phase9 = &cols[col::IS_PHASE9_BRANCH_05_SHAPE];
    let nib = &cols[col::PATH_NIBBLE];
    let nib_m5 = poly_sub(nib, &five_poly, curve);
    let nib_times_nibm5 = poly_mul(nib, &nib_m5, curve);
    poly_mul(phase9, &nib_times_nibm5, curve)
}

fn build_is_phase10_branch_15_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE10_BRANCH_15_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_path_nibble_branch15_pinning_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let five_poly = vec![Scalar::from_u64(5, curve)];
    let phase10 = &cols[col::IS_PHASE10_BRANCH_15_SHAPE];
    let nib = &cols[col::PATH_NIBBLE];
    let nib_m1 = poly_sub(nib, &one_poly, curve);
    let nib_m5 = poly_sub(nib, &five_poly, curve);
    let nib_m1_times_nib_m5 = poly_mul(&nib_m1, &nib_m5, curve);
    poly_mul(phase10, &nib_m1_times_nib_m5, curve)
}

fn build_is_phase11_leaf_8n_shape_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[col::IS_PHASE11_LEAF_8N_SHAPE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

// ──── Cross-row helpers ────────────────────────────────────────────────

/// Boundary rows whose cross-row transition must be excluded from
/// vanishing. We exclude every transition starting from
/// `num_rows − 1` onwards:
///   - `num_rows − 1` (last real row → first padding row, where
///     `parent_hash = 0` and `depth = 0` yield non-zero δ on every byte
///     and depth).
///   - `num_rows .. domain_size − 2` (every padding-to-padding
///     transition, where the depth-increment body `0 − 0 − 1 = −1`
///     would otherwise fire).
///   - `domain_size − 1` (domain wrap-around).
///
/// The shifted constraint is therefore enforced only on real-to-real
/// transitions (rows `0..num_rows − 2`). Excluding all padding-row
/// transitions is required for `C(X)` to be divisible by `Z_H(X)`.
fn boundary_rows(num_rows: usize, domain_size: usize) -> Vec<usize> {
    if num_rows == 0 || domain_size == 0 {
        return Vec::new();
    }
    (num_rows - 1..domain_size).collect()
}

/// Derive the proof domain size from `omega_n_minus_1 = ω^(n−1)` by
/// repeated squaring. Domain sizes are always powers of two in this
/// codebase, so the order of `ω = omega_n_minus_1^(−1)` is the smallest
/// `2^k` such that `ω^(2^k) = 1`.
///
/// Used by the verifier-side `evaluate_shifted_at_point` to recover
/// the actual proof domain size (which the prover may have inflated
/// past the trace's natural padded size when LogUp byte tables require
/// `domain_size ≥ 256`). Reading `self.domain_size` from the constraint
/// system is incorrect — that field reflects the trace's natural
/// `padded_size`, not the inflated proof domain.
fn domain_size_from_omega_n_minus_1(omega_n_minus_1: &Scalar) -> usize {
    let curve = omega_n_minus_1.curve_type();
    let one = Scalar::one(curve);
    let omega = omega_n_minus_1.inverse();
    let mut p = omega;
    let mut size: usize = 1;
    // Cap at 2^30 as a defensive bound; production proofs are < 2^25.
    while size <= (1 << 30) {
        if p.sub(&one).is_zero() {
            return size;
        }
        p = p.mul(&p);
        size *= 2;
    }
    size
}

// ──── VmConstraintSystem implementation ────────────────────────────────

impl VmConstraintSystem for MptInclusionConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_terminal_binary".into(),
            "terminal_is_leaf".into(),
            "node_kind_set".into(),
            "path_nibble_zero_on_non_branch".into(),
            "is_phase1_leaf_shape_binary".into(),
            "is_phase3_leaf_shape_binary".into(),
            "is_phase4_ext_shape_binary".into(),
            "is_phase6_odd_leaf_shape_binary".into(),
            "is_phase7_leaf_6n_shape_binary".into(),
            "is_phase8_branch_01_shape_binary".into(),
            "path_nibble_branch01_binarity".into(),
            "is_phase9_branch_05_shape_binary".into(),
            "path_nibble_branch05_pinning".into(),
            "is_phase10_branch_15_shape_binary".into(),
            "path_nibble_branch15_pinning".into(),
            "is_phase11_leaf_8n_shape_binary".into(),
            "leaf_value_summary_eq_at_terminal".into(),
            "leaf_key_summary_eq_at_terminal".into(),
            "full_key_byte_decomp_all_32".into(),
            "is_depth_eq_binary_all_64".into(),
            "is_depth_eq_sum_to_one".into(),
            "is_depth_eq_linear_combo_eq_depth".into(),
            "per_branch_nibble_eq_full_key_at_depth".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(
            columns.len() >= col::NUM_COLUMNS,
            "mpt AIR expects at least {} columns",
            col::NUM_COLUMNS
        );
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(2, curve);
        let n = columns[0].len();

        let mut term_bin_evals = vec![zero.clone(); n];
        let mut term_leaf_evals = vec![zero.clone(); n];
        let mut kind_set_evals = vec![zero.clone(); n];
        let mut path_nibble_evals = vec![zero.clone(); n];
        let mut phase1_bin_evals = vec![zero.clone(); n];
        let mut phase3_bin_evals = vec![zero.clone(); n];
        let mut phase4_bin_evals = vec![zero.clone(); n];
        let mut phase6_bin_evals = vec![zero.clone(); n];
        let mut phase7_bin_evals = vec![zero.clone(); n];
        let mut phase8_bin_evals = vec![zero.clone(); n];
        let mut path_nibble_b01_evals = vec![zero.clone(); n];
        let mut phase9_bin_evals = vec![zero.clone(); n];
        let mut path_nibble_b05_evals = vec![zero.clone(); n];
        let mut phase10_bin_evals = vec![zero.clone(); n];
        let mut path_nibble_b15_evals = vec![zero.clone(); n];
        let mut phase11_bin_evals = vec![zero.clone(); n];
        for row in 0..n {
            let it = &columns[col::IS_TERMINAL][row];
            term_bin_evals[row] = it.mul(&it.sub(&one));
            let kind = &columns[col::NODE_KIND][row];
            term_leaf_evals[row] = it.mul(&kind.sub(&two));
            // node_kind · (node_kind − 1) · (node_kind − 2) = 0
            kind_set_evals[row] =
                kind.mul(&kind.sub(&one)).mul(&kind.sub(&two));
            // node_kind · path_nibble = 0 — pins path_nibble to 0 on
            // non-branch rows (kind ∈ {1, 2}).
            let nibble = &columns[col::PATH_NIBBLE][row];
            path_nibble_evals[row] = kind.mul(nibble);
            // is_phase1_leaf_shape ∈ {0, 1}.
            let p1 = &columns[col::IS_PHASE1_LEAF_SHAPE][row];
            phase1_bin_evals[row] = p1.mul(&p1.sub(&one));
            // is_phase3_leaf_shape ∈ {0, 1}.
            let p3 = &columns[col::IS_PHASE3_LEAF_SHAPE][row];
            phase3_bin_evals[row] = p3.mul(&p3.sub(&one));
            // is_phase4_ext_shape ∈ {0, 1}.
            let p4 = &columns[col::IS_PHASE4_EXT_SHAPE][row];
            phase4_bin_evals[row] = p4.mul(&p4.sub(&one));
            // is_phase6_odd_leaf_shape ∈ {0, 1}.
            let p6 = &columns[col::IS_PHASE6_ODD_LEAF_SHAPE][row];
            phase6_bin_evals[row] = p6.mul(&p6.sub(&one));
            // is_phase7_leaf_6n_shape ∈ {0, 1}.
            let p7 = &columns[col::IS_PHASE7_LEAF_6N_SHAPE][row];
            phase7_bin_evals[row] = p7.mul(&p7.sub(&one));
            // is_phase8_branch_01_shape ∈ {0, 1}.
            let p8 = &columns[col::IS_PHASE8_BRANCH_01_SHAPE][row];
            phase8_bin_evals[row] = p8.mul(&p8.sub(&one));
            // PHASE 8a · PATH_NIBBLE · (PATH_NIBBLE - 1) = 0.
            let nib = &columns[col::PATH_NIBBLE][row];
            path_nibble_b01_evals[row] = p8.mul(&nib.mul(&nib.sub(&one)));
            // is_phase9_branch_05_shape ∈ {0, 1}.
            let p9 = &columns[col::IS_PHASE9_BRANCH_05_SHAPE][row];
            phase9_bin_evals[row] = p9.mul(&p9.sub(&one));
            // PHASE 9 · PATH_NIBBLE · (PATH_NIBBLE - 5) = 0.
            let five = Scalar::from_u64(5, curve);
            path_nibble_b05_evals[row] = p9.mul(&nib.mul(&nib.sub(&five)));
            // is_phase10_branch_15_shape ∈ {0, 1}.
            let p10 = &columns[col::IS_PHASE10_BRANCH_15_SHAPE][row];
            phase10_bin_evals[row] = p10.mul(&p10.sub(&one));
            // PHASE 10 · (PATH_NIBBLE - 1) · (PATH_NIBBLE - 5) = 0.
            let nib_m1 = nib.sub(&one);
            let nib_m5 = nib.sub(&five);
            path_nibble_b15_evals[row] = p10.mul(&nib_m1.mul(&nib_m5));
            // is_phase11_leaf_8n_shape ∈ {0, 1}.
            let p11 = &columns[col::IS_PHASE11_LEAF_8N_SHAPE][row];
            phase11_bin_evals[row] = p11.mul(&p11.sub(&one));
        }
        // Constraint 16: leaf-row equality between claimed_leaf_value_bytes
        // and value_byte, β-RLC over 32 bytes, gated by is_terminal.
        // β = α in evaluate_at_point + build_constraint_polynomial.
        // Here in evaluate_on_domain (host-side test path with no α),
        // use β = 7 — nonzero, gives strong probabilistic catch on
        // tampering (a coincidental collision would require the
        // tampered byte differences to cancel under 7^k weights).
        let beta_test = Scalar::from_u64(7, curve);
        let mut leaf_value_summary_evals = vec![zero.clone(); n];
        let mut leaf_key_summary_evals = vec![zero.clone(); n];
        for row in 0..n {
            let it = &columns[col::IS_TERMINAL][row];
            // Constraint 16: leaf-row equality for value bytes.
            let mut acc_val = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let claimed = &columns[col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k][row];
                let value = &columns[col::VALUE_BYTE_OFFSET + k][row];
                acc_val = acc_val.add(&bp.mul(&claimed.sub(value)));
                bp = bp.mul(&beta_test);
            }
            leaf_value_summary_evals[row] = it.mul(&acc_val);
            // Constraint 17: leaf-row equality for key path bytes.
            let mut acc_key = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let claimed = &columns[col::CLAIMED_LEAF_KEY_BYTE_OFFSET + k][row];
                let key_path = &columns[col::KEY_PATH_BYTE_OFFSET + k][row];
                acc_key = acc_key.add(&bp.mul(&claimed.sub(key_path)));
                bp = bp.mul(&beta_test);
            }
            leaf_key_summary_evals[row] = it.mul(&acc_key);
        }
        // Constraints 18-22: full multi-row nibble binding via per-depth
        // selectors. β = 7 (fixed) for on-domain testing — see
        // evaluate_at_point / build_constraint_polynomial for the
        // α-based version used in the proof pipeline.
        let beta_test = Scalar::from_u64(7, curve);
        let sixteen = Scalar::from_u64(16, curve);
        let mut full_key_byte_decomp_evals = vec![zero.clone(); n];
        let mut is_depth_eq_binary_evals = vec![zero.clone(); n];
        let mut is_depth_eq_sum_evals = vec![zero.clone(); n];
        let mut is_depth_eq_linear_evals = vec![zero.clone(); n];
        let mut per_branch_nibble_eq_evals = vec![zero.clone(); n];
        for row in 0..n {
            // Constraint 18: β-RLC of 32 byte-decomp sub-bodies.
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for i in 0..32 {
                let byte_i = &columns[col::CLAIMED_FULL_KEY_BYTE_OFFSET + i][row];
                let nib_hi = &columns[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + 2 * i][row];
                let nib_lo = &columns[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + 2 * i + 1][row];
                let recombined = sixteen.mul(nib_hi).add(nib_lo);
                acc = acc.add(&bp.mul(&byte_i.sub(&recombined)));
                bp = bp.mul(&beta_test);
            }
            full_key_byte_decomp_evals[row] = acc;

            // Constraint 19: β-RLC of 64 binary checks for is_depth_eq.
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..64 {
                let s = &columns[col::IS_DEPTH_EQ_OFFSET + k][row];
                acc = acc.add(&bp.mul(&s.mul(&s.sub(&one))));
                bp = bp.mul(&beta_test);
            }
            is_depth_eq_binary_evals[row] = acc;

            // Constraint 20: Σ is_depth_eq - 1 = 0
            let mut sum = Scalar::zero(curve);
            for k in 0..64 {
                let s = &columns[col::IS_DEPTH_EQ_OFFSET + k][row];
                sum = sum.add(s);
            }
            is_depth_eq_sum_evals[row] = sum.sub(&one);

            // Constraint 21: Σ k * is_depth_eq - depth = 0
            let mut weighted = Scalar::zero(curve);
            for k in 0..64 {
                let s = &columns[col::IS_DEPTH_EQ_OFFSET + k][row];
                weighted = weighted.add(&Scalar::from_u64(k as u64, curve).mul(s));
            }
            let depth = &columns[col::DEPTH][row];
            is_depth_eq_linear_evals[row] = weighted.sub(depth);

            // Constraint 22: is_branch_doubled * (path_nibble - Σ is_depth_eq[k] * nib_k) = 0
            let kind = &columns[col::NODE_KIND][row];
            let is_branch_doubled = one.sub(kind).mul(&two.sub(kind));
            let nib = &columns[col::PATH_NIBBLE][row];
            let mut nibble_at_depth = Scalar::zero(curve);
            for k in 0..64 {
                let s = &columns[col::IS_DEPTH_EQ_OFFSET + k][row];
                let n = &columns[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + k][row];
                nibble_at_depth = nibble_at_depth.add(&s.mul(n));
            }
            let diff = nib.sub(&nibble_at_depth);
            per_branch_nibble_eq_evals[row] = is_branch_doubled.mul(&diff);
        }
        vec![
            term_bin_evals,
            term_leaf_evals,
            kind_set_evals,
            path_nibble_evals,
            phase1_bin_evals,
            phase3_bin_evals,
            phase4_bin_evals,
            phase6_bin_evals,
            phase7_bin_evals,
            phase8_bin_evals,
            path_nibble_b01_evals,
            phase9_bin_evals,
            path_nibble_b05_evals,
            phase10_bin_evals,
            path_nibble_b15_evals,
            phase11_bin_evals,
            leaf_value_summary_evals,
            leaf_key_summary_evals,
            full_key_byte_decomp_evals,
            is_depth_eq_binary_evals,
            is_depth_eq_sum_evals,
            is_depth_eq_linear_evals,
            per_branch_nibble_eq_evals,
        ]
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < col::NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        // Constraints 16 and 17: leaf-row equality with β-RLC over 32 bytes.
        // β = α (matches existing β-RLC convention).
        let curve = alpha.curve_type();
        let it = &col_evals_at_z[col::IS_TERMINAL];
        let mut leaf_value_summary = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let claimed = &col_evals_at_z[col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k];
            let value = &col_evals_at_z[col::VALUE_BYTE_OFFSET + k];
            leaf_value_summary = leaf_value_summary.add(&bp.mul(&claimed.sub(value)));
            bp = bp.mul(alpha);
        }
        let leaf_value_summary = it.mul(&leaf_value_summary);

        let mut leaf_key_summary = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let claimed = &col_evals_at_z[col::CLAIMED_LEAF_KEY_BYTE_OFFSET + k];
            let key_path = &col_evals_at_z[col::KEY_PATH_BYTE_OFFSET + k];
            leaf_key_summary = leaf_key_summary.add(&bp.mul(&claimed.sub(key_path)));
            bp = bp.mul(alpha);
        }
        let leaf_key_summary = it.mul(&leaf_key_summary);

        // Constraints 18-22: full multi-row nibble binding via per-depth
        // selectors. β = α (matches existing β-RLC convention).
        let sixteen = Scalar::from_u64(16, curve);
        let two_const = Scalar::from_u64(2, curve);
        let one_local = Scalar::one(curve);

        // Constraint 18: β-RLC of 32 byte-decomps.
        let mut full_key_byte_decomp = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for i in 0..32 {
            let byte_i = &col_evals_at_z[col::CLAIMED_FULL_KEY_BYTE_OFFSET + i];
            let nib_hi = &col_evals_at_z[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + 2 * i];
            let nib_lo = &col_evals_at_z[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + 2 * i + 1];
            let recombined = sixteen.mul(nib_hi).add(nib_lo);
            full_key_byte_decomp = full_key_byte_decomp.add(&bp.mul(&byte_i.sub(&recombined)));
            bp = bp.mul(alpha);
        }

        // Constraint 19: β-RLC of 64 binary checks for is_depth_eq.
        let mut is_depth_eq_binary = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..64 {
            let s = &col_evals_at_z[col::IS_DEPTH_EQ_OFFSET + k];
            is_depth_eq_binary = is_depth_eq_binary.add(&bp.mul(&s.mul(&s.sub(&one_local))));
            bp = bp.mul(alpha);
        }

        // Constraint 20: Σ is_depth_eq - 1 = 0
        let mut is_depth_eq_sum_acc = Scalar::zero(curve);
        for k in 0..64 {
            is_depth_eq_sum_acc = is_depth_eq_sum_acc.add(&col_evals_at_z[col::IS_DEPTH_EQ_OFFSET + k]);
        }
        let is_depth_eq_sum = is_depth_eq_sum_acc.sub(&one_local);

        // Constraint 21: Σ k * is_depth_eq - depth = 0
        let mut is_depth_eq_linear_acc = Scalar::zero(curve);
        for k in 0..64 {
            let s = &col_evals_at_z[col::IS_DEPTH_EQ_OFFSET + k];
            is_depth_eq_linear_acc = is_depth_eq_linear_acc.add(&Scalar::from_u64(k as u64, curve).mul(s));
        }
        let is_depth_eq_linear = is_depth_eq_linear_acc.sub(&col_evals_at_z[col::DEPTH]);

        // Constraint 22: per-branch nibble equality.
        let kind = &col_evals_at_z[col::NODE_KIND];
        let one_minus_kind = one_local.sub(kind);
        let two_minus_kind = two_const.sub(kind);
        let is_branch_doubled = one_minus_kind.mul(&two_minus_kind);
        let nib = &col_evals_at_z[col::PATH_NIBBLE];
        let mut nibble_at_depth = Scalar::zero(curve);
        for k in 0..64 {
            let s = &col_evals_at_z[col::IS_DEPTH_EQ_OFFSET + k];
            let n_at_k = &col_evals_at_z[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + k];
            nibble_at_depth = nibble_at_depth.add(&s.mul(n_at_k));
        }
        let per_branch_nibble_eq = is_branch_doubled.mul(&nib.sub(&nibble_at_depth));

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            eval_is_terminal_binary_at_point(col_evals_at_z),
            eval_terminal_is_leaf_at_point(col_evals_at_z),
            eval_node_kind_set_at_point(col_evals_at_z),
            eval_path_nibble_zero_on_non_branch_at_point(col_evals_at_z),
            eval_is_phase1_leaf_shape_binary_at_point(col_evals_at_z),
            eval_is_phase3_leaf_shape_binary_at_point(col_evals_at_z),
            eval_is_phase4_ext_shape_binary_at_point(col_evals_at_z),
            eval_is_phase6_odd_leaf_shape_binary_at_point(col_evals_at_z),
            eval_is_phase7_leaf_6n_shape_binary_at_point(col_evals_at_z),
            eval_is_phase8_branch_01_shape_binary_at_point(col_evals_at_z),
            eval_path_nibble_branch01_binarity_at_point(col_evals_at_z),
            eval_is_phase9_branch_05_shape_binary_at_point(col_evals_at_z),
            eval_path_nibble_branch05_pinning_at_point(col_evals_at_z),
            eval_is_phase10_branch_15_shape_binary_at_point(col_evals_at_z),
            eval_path_nibble_branch15_pinning_at_point(col_evals_at_z),
            eval_is_phase11_leaf_8n_shape_binary_at_point(col_evals_at_z),
            leaf_value_summary,
            leaf_key_summary,
            full_key_byte_decomp,
            is_depth_eq_binary,
            is_depth_eq_sum,
            is_depth_eq_linear,
            per_branch_nibble_eq,
        ];
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        // No selectors — MPT structural constraints are ungated row-local
        // checks (and one β-RLC'd cross-row).
        Vec::new()
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
        if columns.len() < col::NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col_v in columns.iter_mut().take(col::NUM_COLUMNS) {
            for cell in col_v.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
        // A2 step 1d-multirow padding handling: set is_depth_eq[0] = 1
        // on every padding row so the sum-to-one constraint (#20)
        // and the linear-combination depth constraint (#21, with
        // depth=0 on padding) both vanish naturally. The per-branch
        // nibble-equality constraint (#22) also vanishes on padding
        // because node_kind=0 → is_branch_doubled=2, path_nibble=0,
        // nibble_at_depth_0=claimed_full_key_nibble[0]=0 → diff=0.
        let one = Scalar::one(curve);
        for cell in columns[col::IS_DEPTH_EQ_OFFSET]
            .iter_mut()
            .skip(num_rows)
            .take(padded_size - num_rows)
        {
            *cell = one.clone();
        }
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let bodies: Vec<Vec<Scalar>> = vec![
            build_is_terminal_binary_poly(column_coeffs, curve),
            build_terminal_is_leaf_poly(column_coeffs, curve),
            build_node_kind_set_poly(column_coeffs, curve),
            build_path_nibble_zero_on_non_branch_poly(column_coeffs, curve),
            build_is_phase1_leaf_shape_binary_poly(column_coeffs, curve),
            build_is_phase3_leaf_shape_binary_poly(column_coeffs, curve),
            build_is_phase4_ext_shape_binary_poly(column_coeffs, curve),
            build_is_phase6_odd_leaf_shape_binary_poly(column_coeffs, curve),
            build_is_phase7_leaf_6n_shape_binary_poly(column_coeffs, curve),
            build_is_phase8_branch_01_shape_binary_poly(column_coeffs, curve),
            build_path_nibble_branch01_binarity_poly(column_coeffs, curve),
            build_is_phase9_branch_05_shape_binary_poly(column_coeffs, curve),
            build_path_nibble_branch05_pinning_poly(column_coeffs, curve),
            build_is_phase10_branch_15_shape_binary_poly(column_coeffs, curve),
            build_path_nibble_branch15_pinning_poly(column_coeffs, curve),
            build_is_phase11_leaf_8n_shape_binary_poly(column_coeffs, curve),
            // Constraint 16: leaf-row equality for value with β-RLC (β = α).
            {
                let it = &column_coeffs[col::IS_TERMINAL];
                let mut summary = vec![Scalar::zero(curve)];
                let mut bp = Scalar::one(curve);
                for k in 0..32 {
                    let claimed = &column_coeffs[col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k];
                    let value = &column_coeffs[col::VALUE_BYTE_OFFSET + k];
                    let diff = poly_sub(claimed, value, curve);
                    summary = poly_add(&summary, &poly_scalar_mul(&diff, &bp), curve);
                    bp = bp.mul(alpha);
                }
                poly_mul(it, &summary, curve)
            },
            // Constraint 17: leaf-row equality for key with β-RLC.
            {
                let it = &column_coeffs[col::IS_TERMINAL];
                let mut summary = vec![Scalar::zero(curve)];
                let mut bp = Scalar::one(curve);
                for k in 0..32 {
                    let claimed = &column_coeffs[col::CLAIMED_LEAF_KEY_BYTE_OFFSET + k];
                    let key_path = &column_coeffs[col::KEY_PATH_BYTE_OFFSET + k];
                    let diff = poly_sub(claimed, key_path, curve);
                    summary = poly_add(&summary, &poly_scalar_mul(&diff, &bp), curve);
                    bp = bp.mul(alpha);
                }
                poly_mul(it, &summary, curve)
            },
            // Constraint 18: β-RLC of 32 byte-decomps (β = α).
            {
                let sixteen = Scalar::from_u64(16, curve);
                let mut acc = vec![Scalar::zero(curve)];
                let mut bp = Scalar::one(curve);
                for i in 0..32 {
                    let byte_i = &column_coeffs[col::CLAIMED_FULL_KEY_BYTE_OFFSET + i];
                    let nib_hi = &column_coeffs[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + 2 * i];
                    let nib_lo = &column_coeffs[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + 2 * i + 1];
                    let scaled_hi = poly_scalar_mul(nib_hi, &sixteen);
                    let recombined = poly_add(&scaled_hi, nib_lo, curve);
                    let diff = poly_sub(byte_i, &recombined, curve);
                    acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
                    bp = bp.mul(alpha);
                }
                acc
            },
            // Constraint 19: β-RLC of 64 binary checks for is_depth_eq.
            {
                let one_poly = vec![Scalar::one(curve)];
                let mut acc = vec![Scalar::zero(curve)];
                let mut bp = Scalar::one(curve);
                for k in 0..64 {
                    let s = &column_coeffs[col::IS_DEPTH_EQ_OFFSET + k];
                    let s_m1 = poly_sub(s, &one_poly, curve);
                    let body = poly_mul(s, &s_m1, curve);
                    acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                    bp = bp.mul(alpha);
                }
                acc
            },
            // Constraint 20: Σ is_depth_eq - 1
            {
                let one_poly = vec![Scalar::one(curve)];
                let mut sum = vec![Scalar::zero(curve)];
                for k in 0..64 {
                    let s = &column_coeffs[col::IS_DEPTH_EQ_OFFSET + k];
                    sum = poly_add(&sum, s, curve);
                }
                poly_sub(&sum, &one_poly, curve)
            },
            // Constraint 21: Σ k * is_depth_eq - depth
            {
                let depth = &column_coeffs[col::DEPTH];
                let mut weighted = vec![Scalar::zero(curve)];
                for k in 0..64 {
                    let s = &column_coeffs[col::IS_DEPTH_EQ_OFFSET + k];
                    let scaled = poly_scalar_mul(s, &Scalar::from_u64(k as u64, curve));
                    weighted = poly_add(&weighted, &scaled, curve);
                }
                poly_sub(&weighted, depth, curve)
            },
            // Constraint 22: is_branch_doubled * (path_nibble - Σ is_depth_eq[k] * nibble[k])
            {
                let kind = &column_coeffs[col::NODE_KIND];
                let one_poly = vec![Scalar::one(curve)];
                let two_poly = vec![Scalar::from_u64(2, curve)];
                let one_minus_kind = poly_sub(&one_poly, kind, curve);
                let two_minus_kind = poly_sub(&two_poly, kind, curve);
                let is_branch_doubled = poly_mul(&one_minus_kind, &two_minus_kind, curve);
                let nib = &column_coeffs[col::PATH_NIBBLE];
                let mut nibble_at_depth = vec![Scalar::zero(curve)];
                for k in 0..64 {
                    let s = &column_coeffs[col::IS_DEPTH_EQ_OFFSET + k];
                    let n_at_k = &column_coeffs[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + k];
                    let product = poly_mul(s, n_at_k, curve);
                    nibble_at_depth = poly_add(&nibble_at_depth, &product, curve);
                }
                let diff = poly_sub(nib, &nibble_at_depth, curve);
                poly_mul(&is_branch_doubled, &diff, curve)
            },
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

    // ── Cross-row support ──────────────────────────────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Order matters: the verifier reads `shifted_evals` in this
        // exact order. Layout:
        //   [0..32]     parent_hash bytes (body 0)
        //   [32]        depth (body 0)
        //   [33..65]    node_hash bytes (body 1: Phase 4b extension chain)
        //   [65..97]    claimed_leaf_value_bytes (body 5: cross-row constancy)
        //   [97..129]   claimed_leaf_key_bytes   (body 6: cross-row constancy)
        //   [129..161]  claimed_full_key_bytes   (body 7: cross-row constancy)
        let mut idxs: Vec<usize> = (0..HASH_BYTES)
            .map(|i| col::PARENT_HASH_OFFSET + i)
            .collect();
        idxs.push(col::DEPTH);
        for i in 0..HASH_BYTES {
            idxs.push(col::NODE_HASH_OFFSET + i);
        }
        for i in 0..32 {
            idxs.push(col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + i);
        }
        for i in 0..32 {
            idxs.push(col::CLAIMED_LEAF_KEY_BYTE_OFFSET + i);
        }
        for i in 0..32 {
            idxs.push(col::CLAIMED_FULL_KEY_BYTE_OFFSET + i);
        }
        idxs
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
        // shifted_evals layout: 32 parent_hash + 1 depth + 32 node_hash + 32 claimed_value + 32 claimed_key + 32 claimed_full_key = 161.
        if shifted_evals.len() != 2 * HASH_BYTES + 1 + 32 + 32 + 32
            || col_evals_at_z.len() < col::NUM_COLUMNS
        {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let beta = alpha; // β = α (matches build-side).
        let one = Scalar::one(curve);

        // ── Body 0: parent-chain + depth-increment ─────────────────
        // First HASH_BYTES sub-bodies: `parent_hash_byte_i(ω·z) −
        // node_hash_byte_i(z) = 0`. Trailing sub-body: `depth(ω·z) −
        // depth(z) − 1 = 0`. β-RLC'd.
        let mut body_0 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let parent_next = &shifted_evals[i];
            let node_curr = &col_evals_at_z[col::NODE_HASH_OFFSET + i];
            let diff = parent_next.sub(node_curr);
            body_0 = body_0.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        let depth_next = &shifted_evals[HASH_BYTES];
        let depth_curr = &col_evals_at_z[col::DEPTH];
        let depth_delta = depth_next.sub(depth_curr).sub(&one);
        body_0 = body_0.add(&bp.mul(&depth_delta));

        // ── Body 1: Phase 4b extension chain binding ───────────────
        // β-RLC over `IS_PHASE4_EXT_SHAPE(z) · (NODE_HASH_byte_i(ω·z)
        //   − VALUE_BYTE_i(z))` for i ∈ 0..32.
        // On non-extension rows IS_PHASE4_EXT_SHAPE=0 so body 1
        // naturally vanishes; we still apply the boundary exclusion
        // defensively so a malicious prover cannot place a Phase 4
        // selector on the last real row to escape the chain.
        let phase4 = &col_evals_at_z[col::IS_PHASE4_EXT_SHAPE];
        let mut body_1 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node_next = &shifted_evals[HASH_BYTES + 1 + i];
            let value_curr = &col_evals_at_z[col::VALUE_BYTE_OFFSET + i];
            let diff = node_next.sub(value_curr);
            body_1 = body_1.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        body_1 = phase4.mul(&body_1);

        // ── Body 2: Phase 8b branch chain binding ──────────────────
        let phase8 = &col_evals_at_z[col::IS_PHASE8_BRANCH_01_SHAPE];
        let nib = &col_evals_at_z[col::PATH_NIBBLE];
        let one_minus_nib = one.sub(nib);
        let mut body_2 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node_next = &shifted_evals[HASH_BYTES + 1 + i];
            let child_0 = &col_evals_at_z[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + i];
            let child_1 = &col_evals_at_z[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + i];
            let selected = one_minus_nib.mul(child_0).add(&nib.mul(child_1));
            let diff = selected.sub(node_next);
            body_2 = body_2.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        body_2 = phase8.mul(&body_2);

        // ── Body 3: Phase 9 branch chain binding (slots 0, 5) ──────
        let phase9 = &col_evals_at_z[col::IS_PHASE9_BRANCH_05_SHAPE];
        let five = Scalar::from_u64(5, curve);
        let five_minus_nib = five.sub(nib);
        let mut body_3 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node_next = &shifted_evals[HASH_BYTES + 1 + i];
            let child_0 = &col_evals_at_z[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + i];
            let child_1 = &col_evals_at_z[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + i];
            let selected = five_minus_nib.mul(child_0).add(&nib.mul(child_1));
            let five_node = five.mul(node_next);
            let diff = selected.sub(&five_node);
            body_3 = body_3.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        body_3 = phase9.mul(&body_3);

        // ── Body 4: Phase 10 branch chain binding (slots 1, 5) ─────
        // Lagrange at (1, 5):
        //   `IS_PHASE10 · Σ β^i · ((5 − PATH_NIBBLE) · child_0_i +
        //    (PATH_NIBBLE − 1) · child_1_i − 4 · NODE_HASH(ω·X)_i)`
        // For PATH_NIBBLE = 1: selected = 4·child_0 → child_0 = NODE_HASH.
        // For PATH_NIBBLE = 5: selected = 4·child_1 → child_1 = NODE_HASH.
        let phase10 = &col_evals_at_z[col::IS_PHASE10_BRANCH_15_SHAPE];
        let four = Scalar::from_u64(4, curve);
        let nib_minus_one = nib.sub(&one);
        let mut body_4 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node_next = &shifted_evals[HASH_BYTES + 1 + i];
            let child_0 = &col_evals_at_z[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + i];
            let child_1 = &col_evals_at_z[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + i];
            let selected = five_minus_nib.mul(child_0).add(&nib_minus_one.mul(child_1));
            let four_node = four.mul(node_next);
            let diff = selected.sub(&four_node);
            body_4 = body_4.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        body_4 = phase10.mul(&body_4);

        // α^alpha_offset for body 0; α^(alpha_offset+1) for body 1;
        // α^(alpha_offset+2) for body 2; α^(alpha_offset+3) for body 3;
        // α^(alpha_offset+4) for body 4.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // Boundary-row exclusion product Π_{r ∈ boundary_rows} (z − ω^r).
        // Excludes every transition from `num_rows − 1` through
        // `domain_size − 1` so both bodies vanish on every padding
        // and wrap transition.
        //
        // `domain_size` is recovered from `omega_n_minus_1` (the proof's
        // actual domain order) rather than `self.domain_size` because
        // the latter is set from the trace's natural padded_size and
        // does not reflect LogUp-driven domain inflation.
        let domain_size = domain_size_from_omega_n_minus_1(omega_n_minus_1);
        let rows = boundary_rows(self.num_rows, domain_size);
        let exclusion = if rows.is_empty() {
            Scalar::one(curve)
        } else {
            let omega = omega_n_minus_1.inverse();
            let mut prod = Scalar::one(curve);
            for &r in &rows {
                let omega_r = scalar_pow(&omega, r as u64);
                prod = prod.mul(&z.sub(&omega_r));
            }
            prod
        };

        let term_0 = ap.mul(&body_0).mul(&exclusion);
        ap = ap.mul(alpha);
        let term_1 = ap.mul(&body_1).mul(&exclusion);
        ap = ap.mul(alpha);
        let term_2 = ap.mul(&body_2).mul(&exclusion);
        ap = ap.mul(alpha);
        let term_3 = ap.mul(&body_3).mul(&exclusion);
        ap = ap.mul(alpha);
        let term_4 = ap.mul(&body_4).mul(&exclusion);
        ap = ap.mul(alpha);

        // ── Body 5: leaf_value cross-row constancy ─────────────────
        // (1 − is_terminal(z)) · Σ β^k · (claimed_leaf_value_bytes_k(ω·z)
        //                                   − claimed_leaf_value_bytes_k(z))
        // β = α (matches existing β-RLC convention). When current row
        // is terminal (leaf), gate is 0 → no constraint (this is
        // expected: leaf is the last row of its chain, so the next
        // row is either padding or the start of a new chain, where
        // the constancy doesn't apply).
        let it_curr = &col_evals_at_z[col::IS_TERMINAL];
        let one_minus_it = one.sub(it_curr);
        let mut body_5 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            // Layout: shifted_evals[2 * HASH_BYTES + 1 + k]
            //       = claimed_leaf_value_bytes_k(ω·z)
            let claimed_next = &shifted_evals[2 * HASH_BYTES + 1 + k];
            let claimed_curr = &col_evals_at_z[col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k];
            let diff = claimed_next.sub(claimed_curr);
            body_5 = body_5.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        body_5 = one_minus_it.mul(&body_5);
        let term_5 = ap.mul(&body_5).mul(&exclusion);
        ap = ap.mul(alpha);

        // ── Body 6: claimed_leaf_key_bytes cross-row constancy ─────
        // Same pattern as body 5 but for the key column.
        let mut body_6 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            // Layout: shifted_evals[2 * HASH_BYTES + 1 + 32 + k]
            //       = claimed_leaf_key_bytes_k(ω·z)
            let claimed_next = &shifted_evals[2 * HASH_BYTES + 1 + 32 + k];
            let claimed_curr = &col_evals_at_z[col::CLAIMED_LEAF_KEY_BYTE_OFFSET + k];
            let diff = claimed_next.sub(claimed_curr);
            body_6 = body_6.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        body_6 = one_minus_it.mul(&body_6);
        let term_6 = ap.mul(&body_6).mul(&exclusion);
        ap = ap.mul(alpha);

        // ── Body 7: claimed_full_key_bytes cross-row constancy ─────
        // Same pattern as bodies 5 and 6.
        let mut body_7 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            // Layout: shifted_evals[2 * HASH_BYTES + 1 + 32 + 32 + k]
            //       = claimed_full_key_bytes_k(ω·z)
            let claimed_next = &shifted_evals[2 * HASH_BYTES + 1 + 32 + 32 + k];
            let claimed_curr = &col_evals_at_z[col::CLAIMED_FULL_KEY_BYTE_OFFSET + k];
            let diff = claimed_next.sub(claimed_curr);
            body_7 = body_7.add(&bp.mul(&diff));
            bp = bp.mul(beta);
        }
        body_7 = one_minus_it.mul(&body_7);
        let term_7 = ap.mul(&body_7).mul(&exclusion);

        term_0
            .add(&term_1)
            .add(&term_2)
            .add(&term_3)
            .add(&term_4)
            .add(&term_5)
            .add(&term_6)
            .add(&term_7)
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
        let beta = alpha.clone();
        let one_poly = vec![Scalar::one(curve)];

        // ── Body 0(X) = Σ β^i · (parent_hash_byte_i(ω·X) − node_hash_byte_i(X))
        //              + β^HASH_BYTES · (depth(ω·X) − depth(X) − 1).
        let mut body_0 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let parent = &column_coeffs[col::PARENT_HASH_OFFSET + i];
            let node = &column_coeffs[col::NODE_HASH_OFFSET + i];
            let parent_shift = poly_shift(parent, omega);
            let diff = poly_sub(&parent_shift, node, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            body_0 = poly_add(&body_0, &scaled, curve);
            bp = bp.mul(&beta);
        }
        let depth = &column_coeffs[col::DEPTH];
        let depth_shift = poly_shift(depth, omega);
        let depth_delta_pre = poly_sub(&depth_shift, depth, curve);
        let depth_delta = poly_sub(&depth_delta_pre, &one_poly, curve);
        let scaled_depth = poly_scalar_mul(&depth_delta, &bp);
        body_0 = poly_add(&body_0, &scaled_depth, curve);

        // ── Body 1(X) = IS_PHASE4_EXT_SHAPE(X) · Σ β^i ·
        //               (node_hash_byte_i(ω·X) − value_byte_i(X)).
        let mut sum_b1 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node = &column_coeffs[col::NODE_HASH_OFFSET + i];
            let value = &column_coeffs[col::VALUE_BYTE_OFFSET + i];
            let node_shift = poly_shift(node, omega);
            let diff = poly_sub(&node_shift, value, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            sum_b1 = poly_add(&sum_b1, &scaled, curve);
            bp = bp.mul(&beta);
        }
        let phase4 = &column_coeffs[col::IS_PHASE4_EXT_SHAPE];
        let body_1 = poly_mul(phase4, &sum_b1, curve);

        // ── Body 2(X) = IS_PHASE8_BRANCH_01_SHAPE(X) · Σ β^i ·
        //   ((1 − PATH_NIBBLE) · BRANCH_CHILD_0_HASH_BYTE_i(X) +
        //    PATH_NIBBLE       · BRANCH_CHILD_1_HASH_BYTE_i(X) −
        //    node_hash_byte_i(ω·X)).
        let nib = &column_coeffs[col::PATH_NIBBLE];
        let one_minus_nib = poly_sub(&one_poly, nib, curve);
        let mut sum_b2 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node = &column_coeffs[col::NODE_HASH_OFFSET + i];
            let node_shift = poly_shift(node, omega);
            let child_0 = &column_coeffs[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + i];
            let child_1 = &column_coeffs[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + i];
            let term0 = poly_mul(&one_minus_nib, child_0, curve);
            let term1 = poly_mul(nib, child_1, curve);
            let selected = poly_add(&term0, &term1, curve);
            let diff = poly_sub(&selected, &node_shift, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            sum_b2 = poly_add(&sum_b2, &scaled, curve);
            bp = bp.mul(&beta);
        }
        let phase8 = &column_coeffs[col::IS_PHASE8_BRANCH_01_SHAPE];
        let body_2 = poly_mul(phase8, &sum_b2, curve);

        // ── Body 3(X) = IS_PHASE9_BRANCH_05_SHAPE(X) · Σ β^i ·
        //   ((5 − PATH_NIBBLE) · child_0_i + PATH_NIBBLE · child_1_i
        //    − 5 · node_hash_byte_i(ω·X))
        let five_poly = vec![Scalar::from_u64(5, curve)];
        let five_minus_nib = poly_sub(&five_poly, nib, curve);
        let mut sum_b3 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node = &column_coeffs[col::NODE_HASH_OFFSET + i];
            let node_shift = poly_shift(node, omega);
            let child_0 = &column_coeffs[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + i];
            let child_1 = &column_coeffs[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + i];
            let term0 = poly_mul(&five_minus_nib, child_0, curve);
            let term1 = poly_mul(nib, child_1, curve);
            let selected = poly_add(&term0, &term1, curve);
            let five_node = poly_scalar_mul(&node_shift, &Scalar::from_u64(5, curve));
            let diff = poly_sub(&selected, &five_node, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            sum_b3 = poly_add(&sum_b3, &scaled, curve);
            bp = bp.mul(&beta);
        }
        let phase9 = &column_coeffs[col::IS_PHASE9_BRANCH_05_SHAPE];
        let body_3 = poly_mul(phase9, &sum_b3, curve);

        // ── Body 4(X) = IS_PHASE10_BRANCH_15_SHAPE(X) · Σ β^i ·
        //   ((5 − PATH_NIBBLE) · child_0_i + (PATH_NIBBLE − 1) · child_1_i
        //    − 4 · node_hash_byte_i(ω·X))
        let nib_minus_one = poly_sub(nib, &one_poly, curve);
        let mut sum_b4 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..HASH_BYTES {
            let node = &column_coeffs[col::NODE_HASH_OFFSET + i];
            let node_shift = poly_shift(node, omega);
            let child_0 = &column_coeffs[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + i];
            let child_1 = &column_coeffs[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + i];
            let term0 = poly_mul(&five_minus_nib, child_0, curve);
            let term1 = poly_mul(&nib_minus_one, child_1, curve);
            let selected = poly_add(&term0, &term1, curve);
            let four_node = poly_scalar_mul(&node_shift, &Scalar::from_u64(4, curve));
            let diff = poly_sub(&selected, &four_node, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            sum_b4 = poly_add(&sum_b4, &scaled, curve);
            bp = bp.mul(&beta);
        }
        let phase10 = &column_coeffs[col::IS_PHASE10_BRANCH_15_SHAPE];
        let body_4 = poly_mul(phase10, &sum_b4, curve);

        // Multiply all five bodies by (X − ω^r) for every boundary row r.
        let rows = boundary_rows(self.num_rows, domain_size as usize);
        // ── Body 5 poly: leaf_value cross-row constancy ────────────
        let it = &column_coeffs[col::IS_TERMINAL];
        let one_poly = vec![Scalar::one(curve)];
        let one_minus_it = poly_sub(&one_poly, it, curve);
        let mut summary_5 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let claimed = &column_coeffs[col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + k];
            let claimed_shift = poly_shift(claimed, omega);
            let diff = poly_sub(&claimed_shift, claimed, curve);
            summary_5 = poly_add(&summary_5, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body_5 = poly_mul(&one_minus_it, &summary_5, curve);

        // ── Body 6 poly: leaf_key cross-row constancy ──────────────
        let mut summary_6 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let claimed = &column_coeffs[col::CLAIMED_LEAF_KEY_BYTE_OFFSET + k];
            let claimed_shift = poly_shift(claimed, omega);
            let diff = poly_sub(&claimed_shift, claimed, curve);
            summary_6 = poly_add(&summary_6, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body_6 = poly_mul(&one_minus_it, &summary_6, curve);

        // ── Body 7 poly: full_key cross-row constancy ──────────────
        let mut summary_7 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let claimed = &column_coeffs[col::CLAIMED_FULL_KEY_BYTE_OFFSET + k];
            let claimed_shift = poly_shift(claimed, omega);
            let diff = poly_sub(&claimed_shift, claimed, curve);
            summary_7 = poly_add(&summary_7, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body_7 = poly_mul(&one_minus_it, &summary_7, curve);

        let mut excluded_0 = body_0;
        let mut excluded_1 = body_1;
        let mut excluded_2 = body_2;
        let mut excluded_3 = body_3;
        let mut excluded_4 = body_4;
        let mut excluded_5 = body_5;
        let mut excluded_6 = body_6;
        let mut excluded_7 = body_7;
        for r in &rows {
            let omega_r = scalar_pow(omega, *r as u64);
            excluded_0 = poly_mul_linear(&excluded_0, &omega_r);
            excluded_1 = poly_mul_linear(&excluded_1, &omega_r);
            excluded_2 = poly_mul_linear(&excluded_2, &omega_r);
            excluded_3 = poly_mul_linear(&excluded_3, &omega_r);
            excluded_4 = poly_mul_linear(&excluded_4, &omega_r);
            excluded_5 = poly_mul_linear(&excluded_5, &omega_r);
            excluded_6 = poly_mul_linear(&excluded_6, &omega_r);
            excluded_7 = poly_mul_linear(&excluded_7, &omega_r);
        }

        // α^alpha_offset · excluded_0 + ... + α^(alpha_offset+7) · excluded_7
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term_0 = poly_scalar_mul(&excluded_0, &ap);
        ap = ap.mul(alpha);
        let term_1 = poly_scalar_mul(&excluded_1, &ap);
        ap = ap.mul(alpha);
        let term_2 = poly_scalar_mul(&excluded_2, &ap);
        ap = ap.mul(alpha);
        let term_3 = poly_scalar_mul(&excluded_3, &ap);
        ap = ap.mul(alpha);
        let term_4 = poly_scalar_mul(&excluded_4, &ap);
        ap = ap.mul(alpha);
        let term_5 = poly_scalar_mul(&excluded_5, &ap);
        ap = ap.mul(alpha);
        let term_6 = poly_scalar_mul(&excluded_6, &ap);
        ap = ap.mul(alpha);
        let term_7 = poly_scalar_mul(&excluded_7, &ap);
        let sum01 = poly_add(&term_0, &term_1, curve);
        let sum012 = poly_add(&sum01, &term_2, curve);
        let sum0123 = poly_add(&sum012, &term_3, curve);
        let sum01234 = poly_add(&sum0123, &term_4, curve);
        let sum012345 = poly_add(&sum01234, &term_5, curve);
        let sum0123456 = poly_add(&sum012345, &term_6, curve);
        poly_add(&sum0123456, &term_7, curve)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // Tables: 8-bit, 4-bit, 2-bit, 1-bit range tables.
        let table8 = LookupTable::range(8);
        let table4 = LookupTable::range(4);
        let table2 = LookupTable::range(2);
        let table1 = LookupTable::range(1);
        let tables = vec![table8, table4, table2, table1];
        const TBL_8: usize = 0;
        const TBL_4: usize = 1;
        const TBL_2: usize = 2;
        const TBL_1: usize = 3;

        let mut declarations: Vec<(LookupDeclaration, usize)> = Vec::new();

        // 32 node_hash byte columns → 8-bit range.
        for i in 0..HASH_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("node_hash_byte_{}_range_8", i),
                    column_index: col::NODE_HASH_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }
        // 32 parent_hash byte columns → 8-bit range.
        for i in 0..HASH_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("parent_hash_byte_{}_range_8", i),
                    column_index: col::PARENT_HASH_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }

        // node_kind ∈ {0,1,2}: the algebraic `node_kind_set` row constraint
        // (node_kind · (node_kind − 1) · (node_kind − 2) = 0) is the
        // primary check; this 2-bit lookup is kept as a defensive bound.
        declarations.push((
            LookupDeclaration {
                label: "node_kind_range_2".into(),
                column_index: col::NODE_KIND,
                max_bits: 2,
                selector_column: None,
            },
            TBL_2,
        ));

        // path_nibble ∈ [0, 16) → 4-bit range.
        declarations.push((
            LookupDeclaration {
                label: "path_nibble_range_4".into(),
                column_index: col::PATH_NIBBLE,
                max_bits: 4,
                selector_column: None,
            },
            TBL_4,
        ));

        // is_terminal ∈ {0, 1} → 1-bit range (redundant with the
        // algebraic binary check; defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_terminal_range_1".into(),
                column_index: col::IS_TERMINAL,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // 256 RLP byte columns → 8-bit range. Bounds the RLP encoding
        // committed by each row to a byte string; the cross-AIR LogUp
        // linkage to KeccakExtract uses these columns as the input
        // tuple, so without an 8-bit bound a malicious prover could
        // commit field-element values that don't represent real bytes.
        for b in 0..MAX_RLP_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("node_rlp_byte_{}_range_8", b),
                    column_index: col::NODE_RLP_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }

        // NODE_RLP_LEN ∈ [0, 256] → 9-bit range (≤ 256 = 2⁸ requires 9
        // bits to admit the boundary). The KeccakExtract tuple includes
        // this column, so bounding it prevents the prover from forging
        // a length value that would shift which RLP bytes participate
        // in the tuple match.
        let table9 = LookupTable::range(9);
        let tbl_9 = tables.len();
        let mut tables = tables;
        tables.push(table9);
        declarations.push((
            LookupDeclaration {
                label: "node_rlp_len_range_9".into(),
                column_index: col::NODE_RLP_LEN,
                max_bits: 9,
                selector_column: None,
            },
            tbl_9,
        ));

        // 32 KEY_PATH_BYTE columns + 32 VALUE_BYTE columns → 8-bit range.
        // These are committed columns participating in the cross-AIR
        // LogUp tuple to `mpt_rlp_air`; without an 8-bit bound a
        // malicious prover could commit field values outside [0, 256)
        // that would silently disagree with the gadget's HP-decoded
        // bytes after canonical RLP-byte arithmetic.
        for b in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("key_path_byte_{}_range_8", b),
                    column_index: col::KEY_PATH_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("value_byte_{}_range_8", b),
                    column_index: col::VALUE_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }

        // IS_PHASE1_LEAF_SHAPE ∈ {0, 1} → 1-bit range (defensive,
        // redundant with the algebraic binary constraint above).
        declarations.push((
            LookupDeclaration {
                label: "is_phase1_leaf_shape_range_1".into(),
                column_index: col::IS_PHASE1_LEAF_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // IS_PHASE3_LEAF_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase3_leaf_shape_range_1".into(),
                column_index: col::IS_PHASE3_LEAF_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // IS_PHASE4_EXT_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase4_ext_shape_range_1".into(),
                column_index: col::IS_PHASE4_EXT_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // LEADING_PATH_NIBBLE ∈ [0, 16) → 4-bit range. Required so the
        // gadget's HP-byte decomposition `RLP_BYTE[2] = 0x30 +
        // LEADING_PATH_NIBBLE` doesn't admit out-of-range field values
        // that would shift the high nibble (e.g., LEADING_PATH_NIBBLE
        // = 16 would make RLP_BYTE[2] = 0x40, satisfying the algebraic
        // constraint while breaking the canonical decoding).
        declarations.push((
            LookupDeclaration {
                label: "leading_path_nibble_range_4".into(),
                column_index: col::LEADING_PATH_NIBBLE,
                max_bits: 4,
                selector_column: None,
            },
            TBL_4,
        ));

        // IS_PHASE6_ODD_LEAF_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase6_odd_leaf_shape_range_1".into(),
                column_index: col::IS_PHASE6_ODD_LEAF_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // IS_PHASE7_LEAF_6N_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase7_leaf_6n_shape_range_1".into(),
                column_index: col::IS_PHASE7_LEAF_6N_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // IS_PHASE8_BRANCH_01_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase8_branch_01_shape_range_1".into(),
                column_index: col::IS_PHASE8_BRANCH_01_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // 32 BRANCH_CHILD_0_HASH_BYTE + 32 BRANCH_CHILD_1_HASH_BYTE
        // → 8-bit range each. Required because Phase 8b/9b's shifted
        // constraint reads these and forms an arithmetic combination
        // with PATH_NIBBLE; without an 8-bit bound a malicious prover
        // could supply field-element values out of byte range.
        for b in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("branch_child_0_hash_byte_{}_range_8", b),
                    column_index: col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("branch_child_1_hash_byte_{}_range_8", b),
                    column_index: col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }

        // IS_PHASE9_BRANCH_05_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase9_branch_05_shape_range_1".into(),
                column_index: col::IS_PHASE9_BRANCH_05_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // IS_PHASE10_BRANCH_15_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase10_branch_15_shape_range_1".into(),
                column_index: col::IS_PHASE10_BRANCH_15_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        // IS_PHASE11_LEAF_8N_SHAPE ∈ {0, 1} → 1-bit range (defensive).
        declarations.push((
            LookupDeclaration {
                label: "is_phase11_leaf_8n_shape_range_1".into(),
                column_index: col::IS_PHASE11_LEAF_8N_SHAPE,
                max_bits: 1,
                selector_column: None,
            },
            TBL_1,
        ));

        LookupRequirements { tables, declarations }
    }
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;
    use crate::keccak::keccak256;
    use crate::mpt::{mpt_node_rlp, single_leaf_trie, MptNode, Nibbles};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    /// Build a two-row inclusion witness (branch → leaf) plus the
    /// padded scalar columns.
    fn two_row_columns() -> (Vec<InclusionRow>, Vec<Vec<Scalar>>) {
        let key_a = vec![0x1a, 0xbc];
        let val_a = b"value-a".to_vec();
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: val_a.clone(),
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: b"vb".to_vec(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(keccak256(&leaf_a_rlp));
        children[2] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch { children, value: None };
        let branch_rlp = mpt_node_rlp(&branch);
        let proof = vec![branch_rlp, leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows.len(), 2);

        let padded = crate::trace::nearest_power_of_two(rows.len().max(1));
        let mut columns = alloc_trace(padded, CurveType::Bls48581);
        populate_scalar_trace(&rows, &mut columns, CurveType::Bls48581);
        (rows, columns)
    }

    /// Build a single-row inclusion witness (single leaf trie) plus
    /// the padded scalar columns.
    fn single_row_columns() -> (Vec<InclusionRow>, Vec<Vec<Scalar>>) {
        let key = vec![0xab, 0xcd];
        let value = b"payload".to_vec();
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        let padded = crate::trace::nearest_power_of_two(rows.len().max(1));
        let mut columns = alloc_trace(padded, CurveType::Bls48581);
        populate_scalar_trace(&rows, &mut columns, CurveType::Bls48581);
        (rows, columns)
    }

    #[test]
    fn mpt_cs_labels_and_counts() {
        let (rows, _cols) = two_row_columns();
        let cs = MptInclusionConstraintSystem::new(rows.len());
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_constraints(), 23);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert!(cs.selector_column_indices().is_empty());
        assert!(cs.padding_selector_column().is_none());
        // A2 step 1d-leaf-summary added 32 claimed_leaf_value_bytes +
        // 32 claimed_leaf_key_bytes + 32 claimed_full_key_bytes shifted
        // entries (bodies 5, 6, 7):
        //   32 parent_hash + 1 depth + 32 node_hash + 32 leaf_value
        //   + 32 leaf_key + 32 full_key = 161
        assert_eq!(
            cs.shifted_column_indices().len(),
            2 * HASH_BYTES + 1 + 32 + 32 + 32
        );
    }

    #[test]
    fn mpt_cs_evaluate_on_domain_matches_witness() {
        let (rows, columns) = two_row_columns();
        let cs = MptInclusionConstraintSystem::new(rows.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, vec_) in evals.iter().enumerate() {
            for (row, v) in vec_.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {} on valid witness",
                    k,
                    cs.constraint_labels()[k],
                    row
                );
            }
        }
    }

    #[test]
    fn mpt_cs_evaluate_at_point_zero_on_real_rows() {
        let (rows, columns) = two_row_columns();
        let cs = MptInclusionConstraintSystem::new(rows.len());
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);
        for row in 0..rows.len() {
            let col_vals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let c_at_row = cs.evaluate_at_point(&col_vals, &alpha);
            assert!(
                c_at_row.is_zero(),
                "combined constraint C(row {}) nonzero on valid witness",
                row
            );
        }
    }

    #[test]
    fn mpt_cs_evaluate_at_point_zero_on_all_zero_row() {
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let mut col_vals = vec![zero.clone(); col::NUM_COLUMNS];
        // A2 step 1d-multirow padding convention: set is_depth_eq[0] = 1
        // on padding rows so the sum-to-one constraint (#20) vanishes.
        // This matches what fix_trace_padding / populate_scalar_trace
        // do on padding rows.
        col_vals[col::IS_DEPTH_EQ_OFFSET] = Scalar::one(curve);
        let cs = MptInclusionConstraintSystem::new(8);
        let alpha = Scalar::from_u64(23, curve);
        let c_at = cs.evaluate_at_point(&col_vals, &alpha);
        assert!(
            c_at.is_zero(),
            "padding row (all-zero except is_depth_eq[0]=1) must evaluate to zero"
        );
    }

    #[test]
    fn mpt_cs_constraints_reject_tampered_node_kind() {
        // Set node_kind = 5 on a terminal row → both the (loose) algebraic
        // is_terminal-implies-leaf check fires (5 − 2 = 3, times 1 = 3 ≠ 0).
        let (rows, mut columns) = single_row_columns();
        let curve = CurveType::Bls48581;
        // Row 0 of single_row_columns is the leaf, is_terminal = 1.
        columns[col::NODE_KIND][0] = Scalar::from_u64(5, curve);

        let cs = MptInclusionConstraintSystem::new(rows.len());
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered NODE_KIND on terminal row should make C(row 0) non-zero"
        );

        // Domain-form check: terminal_is_leaf (index 1) fires on row 0.
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            !evals[1][0].is_zero(),
            "terminal_is_leaf must fire on row 0"
        );
    }

    #[test]
    fn mpt_cs_constraints_reject_nonzero_path_nibble_on_leaf() {
        // The leaf row has node_kind = 2 and (correctly) path_nibble = 0.
        // Forge path_nibble = 7. The new `path_nibble_zero_on_non_branch`
        // constraint (`node_kind · path_nibble = 0`) must fire: 2·7 = 14 ≠ 0.
        let (rows, mut columns) = single_row_columns();
        let curve = CurveType::Bls48581;
        columns[col::PATH_NIBBLE][0] = Scalar::from_u64(7, curve);

        let cs = MptInclusionConstraintSystem::new(rows.len());

        // Domain-form check: index 3 (path_nibble_zero_on_non_branch).
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            !evals[3][0].is_zero(),
            "path_nibble_zero_on_non_branch must fire on a non-branch row \
             with nonzero path_nibble"
        );

        // Point-form check: combined C(row 0) is non-zero.
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered path_nibble on leaf row must make C(row 0) non-zero"
        );
    }

    #[test]
    fn mpt_cs_constraints_allow_path_nibble_on_branch() {
        // Branch rows (node_kind = 0) have node_kind · path_nibble = 0
        // regardless of path_nibble — the constraint must NOT fire on a
        // branch row even if path_nibble is 15 (max nibble).
        let (rows, columns) = two_row_columns();
        let cs = MptInclusionConstraintSystem::new(rows.len());

        // Find a branch row (node_kind = 0) and confirm the constraint
        // is zero there even if path_nibble varies. two_row_columns gives
        // a [branch, leaf] pair; row 0 should be the branch.
        let curve = CurveType::Bls48581;
        let kind_row0 = &columns[col::NODE_KIND][0];
        if !kind_row0.is_zero() {
            // If the fixture isn't a branch on row 0, just assert there's
            // no path_nibble constraint violation in the unmodified trace.
            let refs = col_refs(&columns);
            let evals = cs.evaluate_on_domain(&refs, rows.len());
            for row in 0..rows.len() {
                assert!(
                    evals[3][row].is_zero(),
                    "path_nibble constraint must hold on row {} of valid witness",
                    row
                );
            }
            return;
        }

        // Branch row: tamper path_nibble to a non-zero value and confirm
        // the constraint still vanishes (because node_kind = 0 absorbs).
        let mut tampered = columns.clone();
        tampered[col::PATH_NIBBLE][0] = Scalar::from_u64(15, curve);
        let refs: Vec<&Vec<Scalar>> = tampered.iter().collect();
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            evals[3][0].is_zero(),
            "branch row (node_kind = 0) must satisfy the constraint for any \
             path_nibble value"
        );
    }

    /// Phase 8b row-local tampering: pick a Phase 8a branch row and
    /// set `PATH_NIBBLE` to a value outside `{0, 1}` (e.g. 5). The
    /// new `path_nibble_branch01_binarity` body must fire because
    /// `IS_PHASE8_BRANCH_01_SHAPE · PATH_NIBBLE · (PATH_NIBBLE − 1)`
    /// = 1 · 5 · 4 = 20 ≠ 0. (Also note: the existing row-local
    /// `path_nibble_zero_on_non_branch` is `node_kind · path_nibble`;
    /// on a branch row `node_kind = 0` so it doesn't fire — meaning
    /// without Phase 8b's new body, malicious branch-row PATH_NIBBLE
    /// values would slip through.)
    #[test]
    fn mpt_cs_phase8b_path_nibble_binarity_fires_on_invalid_nibble() {
        use crate::keccak::keccak256;
        use crate::mpt::{MptNode, Nibbles};

        // Build the same 2-leaf Phase 8a branch trie as the gadget tests.
        let key_a = vec![0x05u8, 0xab];
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x5, 0xa, 0xb]),
            value: vec![0xaau8; 32],
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x6, 0xc, 0xd]),
            value: vec![0xbbu8; 32],
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(keccak256(&leaf_a_rlp));
        children[1] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows[0].is_phase8_branch_01_shape, 1);
        assert_eq!(rows[0].path_nibble, 0); // honest

        let curve = CurveType::Bls48581;
        let padded = crate::trace::nearest_power_of_two(rows.len().max(1));
        let mut columns = alloc_trace(padded, curve);
        populate_scalar_trace(&rows, &mut columns, curve);

        // Honest case: body 10 (path_nibble_branch01_binarity) is zero
        // because path_nibble = 0.
        let cs = MptInclusionConstraintSystem::new(rows.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            evals[10][0].is_zero(),
            "path_nibble_branch01_binarity must vanish on honest Phase 8a row"
        );

        // Tamper PATH_NIBBLE to 5 on the branch row. Body 10 must fire.
        columns[col::PATH_NIBBLE][0] = Scalar::from_u64(5, curve);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            !evals[10][0].is_zero(),
            "path_nibble_branch01_binarity must fire when PATH_NIBBLE ∉ \
             {{0, 1}} on a Phase 8a branch row"
        );

        // The existing `path_nibble_zero_on_non_branch` (body 3) must
        // STILL not fire (node_kind = 0 absorbs it) — confirming Phase
        // 8b's body 10 is a NEW gate not redundant with body 3.
        assert!(
            evals[3][0].is_zero(),
            "path_nibble_zero_on_non_branch must NOT fire on branch row"
        );
    }

    /// Phase 9 row-local tampering: pick a Phase 9 branch row (real
    /// 2-leaf trie with first-nibbles 0 and 5) and tamper PATH_NIBBLE
    /// to a value outside `{0, 5}` (e.g. 7). Body 12
    /// (`path_nibble_branch05_pinning`) must fire because
    /// `IS_PHASE9_BRANCH_05_SHAPE · PATH_NIBBLE · (PATH_NIBBLE - 5)`
    /// = 1 · 7 · 2 = 14 ≠ 0. (The existing body 3
    /// `path_nibble_zero_on_non_branch` doesn't fire — branch row
    /// has node_kind = 0 which absorbs PATH_NIBBLE.)
    #[test]
    fn mpt_cs_phase9_path_nibble_pinning_fires_on_invalid_nibble() {
        use crate::keccak::keccak256;
        use crate::mpt::{MptNode, Nibbles};

        let key_a = vec![0x05u8, 0xab];
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x5, 0xa, 0xb]),
            value: vec![0xaau8; 32],
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0x6, 0xc, 0xd]),
            value: vec![0xbbu8; 32],
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[0] = Some(keccak256(&leaf_a_rlp));
        children[5] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch { children, value: None };
        let proof = vec![mpt_node_rlp(&branch), leaf_a_rlp];
        let rows = inclusion_witness(&key_a, &proof);
        assert_eq!(rows[0].is_phase9_branch_05_shape, 1);
        assert_eq!(rows[0].path_nibble, 0); // honest: PATH_NIBBLE = 0

        let curve = CurveType::Bls48581;
        let padded = crate::trace::nearest_power_of_two(rows.len().max(1));
        let mut columns = alloc_trace(padded, curve);
        populate_scalar_trace(&rows, &mut columns, curve);

        // Honest case: body 12 (`path_nibble_branch05_pinning`) is
        // zero because path_nibble = 0 satisfies 0 · (0 - 5) = 0.
        let cs = MptInclusionConstraintSystem::new(rows.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            evals[12][0].is_zero(),
            "path_nibble_branch05_pinning must vanish on honest Phase 9 row"
        );

        // Tamper PATH_NIBBLE to 7 on the branch row. Body 12 must fire.
        columns[col::PATH_NIBBLE][0] = Scalar::from_u64(7, curve);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            !evals[12][0].is_zero(),
            "path_nibble_branch05_pinning must fire when PATH_NIBBLE ∉ \
             {{0, 5}} on a Phase 9 branch row"
        );

        // Body 3 (path_nibble_zero_on_non_branch) must STILL not fire
        // (node_kind = 0 absorbs) — confirms body 12 is a NEW gate.
        assert!(
            evals[3][0].is_zero(),
            "path_nibble_zero_on_non_branch must NOT fire on branch row"
        );

        // Sanity: PATH_NIBBLE = 5 also satisfies the constraint.
        columns[col::PATH_NIBBLE][0] = Scalar::from_u64(5, curve);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(
            evals[12][0].is_zero(),
            "path_nibble_branch05_pinning must vanish for PATH_NIBBLE = 5"
        );
    }

    #[test]
    fn mpt_cs_constraints_reject_tampered_is_terminal() {
        // Set is_terminal = 2 on row 0 → binary check fires (2·1 = 2 ≠ 0).
        let (rows, mut columns) = two_row_columns();
        let curve = CurveType::Bls48581;
        columns[col::IS_TERMINAL][0] = Scalar::from_u64(2, curve);

        let cs = MptInclusionConstraintSystem::new(rows.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, rows.len());
        assert!(!evals[0][0].is_zero(), "is_terminal_binary must fire on row 0");
    }

    #[test]
    fn mpt_cs_constraints_reject_broken_chain() {
        // Tamper row 1's parent_hash so it doesn't match row 0's
        // node_hash. The cross-row body must be non-zero on row 0.
        let (rows, mut columns) = two_row_columns();
        let curve = CurveType::Bls48581;
        columns[col::PARENT_HASH_OFFSET][1] =
            columns[col::PARENT_HASH_OFFSET][1].add(&Scalar::from_u64(1, curve));

        let cs = MptInclusionConstraintSystem::new(rows.len());
        let one = Scalar::one(curve);
        // Manually compute the per-row cross-row body at row 0
        // (β-RLC matching evaluate_shifted_at_point), without the
        // boundary-exclusion factor.
        let beta = Scalar::from_u64(11, curve);
        let mut bp = Scalar::one(curve);
        let mut body = Scalar::zero(curve);
        for i in 0..HASH_BYTES {
            let p_next = &columns[col::PARENT_HASH_OFFSET + i][1];
            let n_curr = &columns[col::NODE_HASH_OFFSET + i][0];
            body = body.add(&bp.mul(&p_next.sub(n_curr)));
            bp = bp.mul(&beta);
        }
        let d_next = &columns[col::DEPTH][1];
        let d_curr = &columns[col::DEPTH][0];
        body = body.add(&bp.mul(&d_next.sub(d_curr).sub(&one)));
        assert!(
            !body.is_zero(),
            "cross-row body must be non-zero on row 0 with a broken chain"
        );
        // Sanity: with an unmodified pair the body is zero.
        let _ = cs;
    }

    #[test]
    fn mpt_cs_constraints_reject_non_monotone_depth() {
        // Force depth on row 1 to 5 (instead of 1). Even with the chain
        // intact, the depth-increment body fires at row 0 in the β-RLC.
        let (_rows, mut columns) = two_row_columns();
        let curve = CurveType::Bls48581;
        columns[col::DEPTH][1] = Scalar::from_u64(5, curve);

        let one = Scalar::one(curve);
        let beta = Scalar::from_u64(11, curve);
        // Walk the same β powers as evaluate_shifted_at_point — first
        // skip past the 32 parent-hash byte bodies (which are zero).
        let mut bp = Scalar::one(curve);
        let mut body = Scalar::zero(curve);
        for i in 0..HASH_BYTES {
            let p_next = &columns[col::PARENT_HASH_OFFSET + i][1];
            let n_curr = &columns[col::NODE_HASH_OFFSET + i][0];
            body = body.add(&bp.mul(&p_next.sub(n_curr)));
            bp = bp.mul(&beta);
        }
        let d_next = &columns[col::DEPTH][1];
        let d_curr = &columns[col::DEPTH][0];
        body = body.add(&bp.mul(&d_next.sub(d_curr).sub(&one)));
        assert!(
            !body.is_zero(),
            "cross-row body must be non-zero when depth increment ≠ 1"
        );
    }

    #[test]
    fn mpt_cs_lookup_declarations_are_well_formed() {
        let cs = MptInclusionConstraintSystem::new(2);
        let reqs = cs.lookup_declarations();
        // Five tables: 8-bit, 4-bit, 2-bit, 1-bit range, plus 9-bit for
        // the NODE_RLP_LEN bound.
        assert_eq!(reqs.tables.len(), 5);
        assert_eq!(reqs.tables[0].bits, 8);
        assert_eq!(reqs.tables[1].bits, 4);
        assert_eq!(reqs.tables[2].bits, 2);
        assert_eq!(reqs.tables[3].bits, 1);
        assert_eq!(reqs.tables[4].bits, 9);

        // Declarations: 32 node_hash + 32 parent_hash + 1 node_kind +
        // 1 path_nibble + 1 is_terminal + 256 node_rlp bytes + 1
        // node_rlp_len + 32 key_path_byte + 32 value_byte +
        // 1 is_phase1_leaf_shape + 1 is_phase3_leaf_shape +
        // 1 is_phase4_ext_shape + 1 leading_path_nibble +
        // 1 is_phase6_odd_leaf_shape + 1 is_phase7_leaf_6n_shape +
        // 1 is_phase8_branch_01_shape +
        // 64 (branch_child_0/1_hash_byte) +
        // 1 is_phase9_branch_05_shape +
        // 1 is_phase10_branch_15_shape +
        // 1 is_phase11_leaf_8n_shape = 462.
        assert_eq!(
            reqs.declarations.len(),
            64 + 1 + 1 + 1 + MAX_RLP_LEN + 1 + 32 + 32 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 64 + 1 + 1 + 1
        );

        for (decl, table_idx) in &reqs.declarations {
            assert!(
                decl.column_index < col::NUM_COLUMNS,
                "declaration {} references out-of-range column {}",
                decl.label,
                decl.column_index
            );
            assert!(*table_idx < reqs.tables.len());
        }

        // node_kind uses the 2-bit table.
        let nk = reqs
            .declarations
            .iter()
            .find(|(d, _)| d.column_index == col::NODE_KIND)
            .expect("node_kind must have a declaration");
        assert_eq!(nk.0.max_bits, 2);
        assert_eq!(nk.1, 2);

        // path_nibble uses the 4-bit table.
        let pn = reqs
            .declarations
            .iter()
            .find(|(d, _)| d.column_index == col::PATH_NIBBLE)
            .expect("path_nibble must have a declaration");
        assert_eq!(pn.0.max_bits, 4);
        assert_eq!(pn.1, 1);

        // NODE_RLP_LEN uses the 9-bit table (table index 4).
        let rl = reqs
            .declarations
            .iter()
            .find(|(d, _)| d.column_index == col::NODE_RLP_LEN)
            .expect("node_rlp_len must have a declaration");
        assert_eq!(rl.0.max_bits, 9);
        assert_eq!(rl.1, 4);

        // First NODE_RLP byte uses the 8-bit table.
        let rb = reqs
            .declarations
            .iter()
            .find(|(d, _)| d.column_index == col::NODE_RLP_OFFSET)
            .expect("first node_rlp byte must have a declaration");
        assert_eq!(rb.0.max_bits, 8);
        assert_eq!(rb.1, 0);
    }

    #[test]
    fn mpt_cs_boundary_rows_basic() {
        // boundary_rows now excludes every transition from `num_rows-1`
        // through `domain_size-1` (covers last-real-row → first-padding,
        // every padding-to-padding, and the wrap), so the shifted
        // constraint is enforced only on real-to-real transitions.
        let br = boundary_rows(2, 4);
        assert_eq!(br, vec![1, 2, 3]);
    }

    /// Marked `#[ignore]`: full prove/verify roundtrip is expensive
    /// (KZG commitments + schoolbook builds across 68 columns under
    /// BLS48-581). Run manually with:
    ///     cargo test --release -p metavm-zkp --lib \
    ///         mpt_cs_prove_verify_small -- --ignored --nocapture
    #[test]
    #[ignore = "slow: full prover roundtrip; run with --release --ignored"]
    fn mpt_cs_prove_verify_small() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (rows, _cols) = two_row_columns();
        let trace = build_trace_polynomials_from_rows(&rows, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = MptInclusionConstraintSystem::new(rows.len())
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "mpt two-row inclusion proof must verify");
    }

    /// Soundness regression for the closed verifier-skips-constraints
    /// bug. Before the fix, the verifier's `Q(z)·Z(z) == C(z)` identity
    /// check was gated by `has_selectors =
    /// !selector_column_indices().is_empty()` and the prover's fallback
    /// path (also selector-gated) omitted shifted, LogUp, and
    /// permutation contributions from C(X). MPT, SSZ, and any future
    /// selector-less AIR therefore had **none of their constraints
    /// enforced** — every malicious witness verified.
    ///
    /// Three coordinated changes closed the bug:
    /// 1. `prover::prove_inner` and `prove_phase2_from_main_commit`:
    ///    `use_build_poly = true` unconditionally.
    /// 2. `verifier::verify_inner` and `verify_inner_scheme`:
    ///    `has_selectors = true` unconditionally.
    /// 3. `mpt_constraints::boundary_rows`: expanded to exclude every
    ///    padding-row transition (`num_rows − 1 .. domain_size`) so the
    ///    depth-increment body's `0 − 0 − 1 = −1` value on
    ///    padding-to-padding rows is voided by the exclusion product.
    ///    The verifier's `evaluate_shifted_at_point` derives the proof
    ///    domain size from `omega_n_minus_1` rather than reading the
    ///    stale `self.domain_size` (which reflects the pre-LogUp-inflation
    ///    natural padded size).
    ///
    /// This test forges `node_kind = 5` on a real row, violating the
    /// `node_kind_set` row-local constraint. After the fix the verifier
    /// correctly rejects.
    #[test]
    #[ignore = "slow: forged MPT trace, full prover roundtrip; run with --release --ignored"]
    fn mpt_verifier_rejects_violated_node_kind_set_via_full_proof() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (rows, mut columns) = two_row_columns();
        // Forge: set node_kind = 5 on row 0. Violates `node_kind_set`
        // (`kind · (kind − 1) · (kind − 2) = 0`) since 5 ∉ {0, 1, 2}.
        columns[col::NODE_KIND][0] = Scalar::from_u64(5, curve);

        let polys: Vec<Polynomial> = columns
            .into_iter()
            .map(|evals| Polynomial { evaluations: evals.clone(), degree: evals.len() })
            .collect();
        let trace = TracePolynomials::from_polynomials(polys, rows.len(), curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = MptInclusionConstraintSystem::new(rows.len())
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(
            !valid,
            "verifier must reject a proof whose trace violates node_kind_set"
        );
    }
}
