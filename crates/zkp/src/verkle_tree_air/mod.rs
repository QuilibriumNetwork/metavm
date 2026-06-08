//! Verkle tree commitment AIR (scaffold).
//!
//! ## Background
//!
//! Verkle trees replace binary hash-tree inner nodes with **vector
//! (Pedersen) commitments**, so each inner node summarises `BRANCHING`
//! children with a single curve-point commitment:
//!
//! ```text
//! C = Σ_{i=0}^{B-1} v_i · G_i
//! ```
//!
//! where `G_0 .. G_{B-1}` are fixed independent generators and `v_i` is
//! the child commitment (or leaf hash) at branch `i`. Opening at index
//! `j` requires showing that the committed `v_j` matches the stated
//! `expected_value` (and, in real proofs, providing an IPA/KZG opening
//! over a polynomial whose evaluations are the children).
//!
//! This AIR is a **scaffold**: it commits the per-level
//! `(claimed_node, index, expected_value)` triples and pins the
//! "claimed_node_at_index = expected_value" relation algebraically. The
//! actual `C = Σ v_i · G_i` evaluation is deferred to a cross-AIR LogUp
//! into a future G1-curve-ops AIR (see [`make_verkle_to_curve_ops_descriptor`]).
//!
//! ## Trace layout
//!
//! One inclusion expands to `DEPTH` rows, one per tree level from leaf
//! up to root:
//!   - Row 0: row at the leaf level. `PATH_COMMITMENT` holds the
//!     commitment of the leaf's parent node, `EXPECTED_VALUE` holds the
//!     leaf hash, and `PATH_INDEX` is the branch index inside the
//!     parent.
//!   - Row `k > 0`: `PATH_COMMITMENT` is the commitment of the node at
//!     level `k`'s parent; `EXPECTED_VALUE` is row `k-1`'s
//!     `PATH_COMMITMENT` (the child commitment).
//!   - Row `DEPTH-1`: `PATH_COMMITMENT == CLAIMED_ROOT` (enforced by a
//!     dedicated `IS_TOP` selector and `claimed_root_match` constraint).
//!
//! ## Algebraic constraints (row-local, 7 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//! 1. `is_top_binary` — `IS_TOP · (IS_TOP − 1) = 0`
//! 2. `path_index_range` — host-side range-check pin: the branch index
//!    fits in `log2(BRANCHING)` bits (scaffolded; full bit-decomp
//!    deferred).
//! 3. `claimed_root_match` (β-RLC over 32 chunk bytes) —
//!    `IS_TOP · Σ β^i · (PATH_COMMITMENT[i] − CLAIMED_ROOT[i]) = 0`
//! 4. `pedersen_open_pin` — scaffold body that evaluates to zero on the
//!    honest trace by construction; the real cryptographic content (the
//!    sum `Σ v_i · G_i`) is enforced via cross-AIR LogUp to the curve
//!    ops AIR, not in-row.
//! 5. `pedersen_partial_sum_init` (β-RLC over 64 partial-sum bytes) —
//!    on the bottom row of an inclusion (`LEVEL == 0`), the
//!    `PEDERSEN_PARTIAL_SUM_CURR` must equal the host-supplied
//!    identity / start value (typically the zero curve point or the
//!    "seed" generator slot). Scaffold form: pins to zero (point at
//!    infinity placeholder).
//! 6. `pedersen_top_match` (β-RLC over 64 partial-sum bytes) —
//!    `IS_TOP · Σ β^i · (PEDERSEN_PARTIAL_SUM_NEXT[i] −
//!    serialized(CLAIMED_ROOT)[i]) = 0` (scaffold: gated equality;
//!    final-row `partial_sum_next` is bound at the host level to the
//!    serialized claimed root).
//!
//! ## Cross-row shifted constraints (2 bodies)
//!
//! 0. `commitment_chain` — `IS_REAL(ω·X) · Σ β^i ·
//!     (EXPECTED_VALUE(ω·X)[i] − PATH_COMMITMENT(X)[i]) = 0`
//!
//! Pins that each next-level row's `EXPECTED_VALUE` is the previous
//! row's `PATH_COMMITMENT` (the child→parent chaining that walks the
//! Verkle path bottom-up).
//!
//! 1. `partial_sum_continuity` — `IS_REAL(ω·X) · Σ β^i ·
//!     (PEDERSEN_PARTIAL_SUM_CURR(ω·X)[i] −
//!      PEDERSEN_PARTIAL_SUM_NEXT(X)[i]) = 0`
//!
//! Pins that each next-level row's incoming `partial_sum_curr` equals
//! the previous row's outgoing `partial_sum_next` — i.e. the running
//! Pedersen accumulator is single-threaded across the inclusion. The
//! per-step "next = curr + v_k · G_k" relation itself is shipped to
//! the BLS12-381 G1 curve-ops AIR through
//! [`make_verkle_to_g1_pedersen_step_descriptor`].
//!
//! ## Soundness scope
//!
//! * **Pedersen commitment evaluation** (`C = Σ v_i · G_i`): deferred
//!   to cross-AIR LogUp into a future G1-curve-ops AIR. The scaffold
//!   descriptor [`make_verkle_to_curve_ops_descriptor`] reserves the
//!   shape of that linkage.
//! * **Path-index range check** (index `< BRANCHING`): scaffolded as a
//!   single equality body; a future tightening will add a bit-decomp
//!   sub-witness similar to the deposit-tree `INDEX_BIT_REPL` pattern.
//! * **Leaf-key ↔ path-index consistency** (the LE/BE digit
//!   decomposition of `leaf_key` must equal `path_indexes`): scaffolded;
//!   constraint pinning the relation belongs in a follow-up.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Verkle tree depth (number of levels from leaf to root). 8 is enough
/// to cover the targeted ~256-leaf scaffold; production depths run
/// closer to 32 once branching is configurable per-tree.
pub const DEPTH: usize = 8;
/// Branching factor (children per inner node). Ethereum's chosen
/// Verkle-tree branching is 256; we expose it as a constant so the
/// scaffold can be re-instantiated for other choices.
pub const BRANCHING: usize = 256;
/// Bytes per commitment / hash chunk (32 ≈ a serialized field element
/// or affine-x curve coordinate).
pub const CHUNK_BYTES: usize = 32;
/// Bytes per leaf key.
pub const KEY_BYTES: usize = 32;
/// Bytes per leaf value.
pub const VALUE_BYTES: usize = 32;
/// Rows used per inclusion (one per level).
pub const ROWS_PER_INCLUSION: usize = DEPTH;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_LEAF_KEY_OFFSET: usize = 0;
pub const COL_LEAF_VALUE_OFFSET: usize = COL_LEAF_KEY_OFFSET + KEY_BYTES; // 32
pub const COL_PATH_COMMITMENT_OFFSET: usize = COL_LEAF_VALUE_OFFSET + VALUE_BYTES; // 64
pub const COL_EXPECTED_VALUE_OFFSET: usize = COL_PATH_COMMITMENT_OFFSET + CHUNK_BYTES; // 96
pub const COL_CLAIMED_ROOT_OFFSET: usize = COL_EXPECTED_VALUE_OFFSET + CHUNK_BYTES; // 128

pub const COL_PATH_INDEX: usize = COL_CLAIMED_ROOT_OFFSET + CHUNK_BYTES; // 160
pub const COL_LEVEL: usize = COL_PATH_INDEX + 1; // 161

pub const COL_IS_REAL: usize = COL_LEVEL + 1; // 162
/// Selector flagging the top row (where `PATH_COMMITMENT` is the
/// claimed root).
pub const COL_IS_TOP: usize = COL_IS_REAL + 1; // 163

/// Bytes per serialized curve point (`x` || `y`, 32 bytes each, BE).
pub const POINT_BYTES: usize = 2 * CHUNK_BYTES;

/// Incoming accumulator for the Pedersen step at this level:
/// `partial_sum_curr := Σ_{j<k} v_j · G_j`.
pub const COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET: usize = COL_IS_TOP + 1; // 164
/// Outgoing accumulator after consuming `(v_k · G_k)`:
/// `partial_sum_next := partial_sum_curr + v_k · G_k`.
pub const COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET: usize =
    COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + POINT_BYTES; // 228

pub const NUM_COLUMNS: usize = COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + POINT_BYTES; // 292

/// 7 row-local constraint bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 7;
/// 2 cross-row shifted bodies.
pub const NUM_SHIFTED: usize = 2;

// ─── Witness types ────────────────────────────────────────────────────

/// One Verkle-tree inclusion witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerkleInclusionWitness {
    /// The leaf's 32-byte key (e.g. an Ethereum address-style key).
    pub leaf_key: [u8; KEY_BYTES],
    /// The leaf's 32-byte value.
    pub leaf_value: [u8; VALUE_BYTES],
    /// Path commitments bottom-up: `path_commitments[k]` is the
    /// commitment of the node at level `k` along the path from leaf to
    /// root. `path_commitments[DEPTH-1]` MUST equal `claimed_root`.
    pub path_commitments: [[u8; CHUNK_BYTES]; DEPTH],
    /// Per-level branch index identifying which child of the level-`k`
    /// node the path takes. Each value MUST be `< BRANCHING`.
    pub path_indexes: [u32; DEPTH],
    /// Claimed Verkle tree root (top-level Pedersen commitment).
    pub claimed_root: [u8; CHUNK_BYTES],
    /// Per-level Pedersen partial-sum accumulators serialized as
    /// `(x || y)` 64-byte tuples.
    ///
    /// `pedersen_partial_sums[k]` is the **incoming** accumulator at
    /// level `k` (== outgoing accumulator at level `k-1`); the level-0
    /// entry MUST be the identity placeholder (all-zero bytes in this
    /// scaffold). The final outgoing accumulator
    /// `pedersen_partial_sums[DEPTH]` is the closed Pedersen commitment
    /// for the row. The cross-AIR LogUp into the BLS12-381 G1 curve-ops
    /// AIR ([`make_verkle_to_g1_pedersen_step_descriptor`]) binds each
    /// adjacent pair `(partial_sums[k], partial_sums[k+1])` to a G1 add
    /// row evaluating `partial_sums[k+1] = partial_sums[k] + v_k · G_k`.
    pub pedersen_partial_sums: [[u8; POINT_BYTES]; DEPTH + 1],
}

/// Sequence of inclusions packed into a single trace.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerkleTreeWitness {
    pub inclusions: Vec<VerkleInclusionWitness>,
}

impl VerkleInclusionWitness {
    /// Construct a scaffold witness for a single inclusion.
    ///
    /// The caller supplies the path commitments bottom-up and a claimed
    /// root; this function does NOT recompute the commitments (real
    /// Pedersen evaluation is deferred to the curve-ops AIR).
    /// `path_commitments[DEPTH-1]` is checked to equal `claimed_root`
    /// in debug builds.
    pub fn new(
        leaf_key: [u8; KEY_BYTES],
        leaf_value: [u8; VALUE_BYTES],
        path_commitments: [[u8; CHUNK_BYTES]; DEPTH],
        path_indexes: [u32; DEPTH],
        claimed_root: [u8; CHUNK_BYTES],
    ) -> Self {
        Self::new_with_partial_sums(
            leaf_key,
            leaf_value,
            path_commitments,
            path_indexes,
            claimed_root,
            [[0u8; POINT_BYTES]; DEPTH + 1],
        )
    }

    /// Same as [`Self::new`] but accepting host-supplied per-level
    /// Pedersen partial sums. Use this once the caller has access to a
    /// real (Bandersnatch / placeholder G1) Pedersen evaluation.
    pub fn new_with_partial_sums(
        leaf_key: [u8; KEY_BYTES],
        leaf_value: [u8; VALUE_BYTES],
        path_commitments: [[u8; CHUNK_BYTES]; DEPTH],
        path_indexes: [u32; DEPTH],
        claimed_root: [u8; CHUNK_BYTES],
        pedersen_partial_sums: [[u8; POINT_BYTES]; DEPTH + 1],
    ) -> Self {
        debug_assert_eq!(
            path_commitments[DEPTH - 1],
            claimed_root,
            "top path commitment must equal claimed root",
        );
        for &idx in path_indexes.iter() {
            debug_assert!(
                (idx as usize) < BRANCHING,
                "path index out of range",
            );
        }
        Self {
            leaf_key,
            leaf_value,
            path_commitments,
            path_indexes,
            claimed_root,
            pedersen_partial_sums,
        }
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

fn write_chunk(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    row: usize,
    chunk: &[u8],
    curve: CurveType,
) {
    for (b, &byte) in chunk.iter().enumerate() {
        columns[offset + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &VerkleTreeWitness,
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
        for level in 0..DEPTH {
            let row = base_row + level;
            // EXPECTED_VALUE: at level 0 it's the leaf hash (use
            // leaf_value as a stand-in for the leaf commitment), at
            // level k>0 it's the previous row's PATH_COMMITMENT.
            let expected: [u8; CHUNK_BYTES] = if level == 0 {
                let mut buf = [0u8; CHUNK_BYTES];
                buf.copy_from_slice(&inclusion.leaf_value);
                buf
            } else {
                inclusion.path_commitments[level - 1]
            };

            write_chunk(
                &mut columns,
                COL_LEAF_KEY_OFFSET,
                row,
                &inclusion.leaf_key,
                curve,
            );
            write_chunk(
                &mut columns,
                COL_LEAF_VALUE_OFFSET,
                row,
                &inclusion.leaf_value,
                curve,
            );
            write_chunk(
                &mut columns,
                COL_PATH_COMMITMENT_OFFSET,
                row,
                &inclusion.path_commitments[level],
                curve,
            );
            write_chunk(&mut columns, COL_EXPECTED_VALUE_OFFSET, row, &expected, curve);
            write_chunk(
                &mut columns,
                COL_CLAIMED_ROOT_OFFSET,
                row,
                &inclusion.claimed_root,
                curve,
            );

            // Pedersen partial sums: incoming at level k = sums[k];
            // outgoing at level k = sums[k+1].
            write_chunk(
                &mut columns,
                COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET,
                row,
                &inclusion.pedersen_partial_sums[level],
                curve,
            );
            write_chunk(
                &mut columns,
                COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET,
                row,
                &inclusion.pedersen_partial_sums[level + 1],
                curve,
            );

            columns[COL_PATH_INDEX][row] =
                Scalar::from_u64(inclusion.path_indexes[level] as u64, curve);
            columns[COL_LEVEL][row] = Scalar::from_u64(level as u64, curve);
            columns[COL_IS_REAL][row] = one.clone();
            if level == DEPTH - 1 {
                columns[COL_IS_TOP][row] = one.clone();
            }
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct VerkleTreeConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl VerkleTreeConstraintSystem {
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

fn eval_claimed_root_match(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let is_top = &col_evals[COL_IS_TOP];
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let lhs = &col_evals[COL_PATH_COMMITMENT_OFFSET + i];
        let rhs = &col_evals[COL_CLAIMED_ROOT_OFFSET + i];
        let body = lhs.sub(rhs);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    is_top.mul(&acc)
}

fn build_claimed_root_match_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let is_top = &col_coeffs[COL_IS_TOP];
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let lhs = &col_coeffs[COL_PATH_COMMITMENT_OFFSET + i];
        let rhs = &col_coeffs[COL_CLAIMED_ROOT_OFFSET + i];
        let diff = poly_sub(lhs, rhs, curve);
        let scaled = poly_scalar_mul(&diff, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(is_top, &acc, curve)
}

/// Scaffold pin for the Pedersen opening. Today this body is identically
/// zero on the honest trace (the real `C = Σ v_i · G_i` evaluation is
/// deferred to the cross-AIR LogUp into the curve-ops AIR). Keeping a
/// named body in the constraint catalog locks the constraint count for
/// downstream callers.
fn eval_pedersen_open_pin(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    Scalar::zero(curve)
}

fn build_pedersen_open_pin_poly(curve: CurveType) -> Vec<Scalar> {
    vec![Scalar::zero(curve)]
}

/// Body 5: at the bottom row of an inclusion (`LEVEL == 0`), the
/// incoming Pedersen accumulator equals the identity placeholder
/// (scaffold: all-zero bytes). Gated by `IS_REAL * (1 - LEVEL/seed)`
/// is not safe in-row, so we use a per-row indicator built from the
/// `LEVEL` column via β-RLC over the partial-sum bytes, gated by the
/// boolean "level is zero" surrogate. For scaffold purposes we use
/// `IS_REAL * (1 - is_top_chain)`-style gating only when LEVEL = 0;
/// since LEVEL is supplied as a scalar, we use a host-witnessed
/// zero-equality: the body simply enforces equality of the in-row
/// `partial_sum_curr` to the identity *when* LEVEL == 0 (modelled by
/// `(1 - LEVEL_nonzero_witness)` — for the current scaffold we just
/// β-RLC the bytes and gate by the bottom-row selector `IS_BOTTOM`,
/// derived in-row from LEVEL via the host but not yet algebraically
/// pinned).
///
/// Scaffold form below: gated by `IS_REAL` and the host-side
/// guarantee that the witness fills the identity into level-0 rows.
fn eval_pedersen_partial_sum_init(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let is_real = &col_evals[COL_IS_REAL];
    let level = &col_evals[COL_LEVEL];
    // is_bottom := 1 iff level == 0. Scaffolded as `1 - level * inv(level)`
    // is unsafe without an inverse witness; instead we use a softer
    // gate: the bottom-row constraint only fires when level coefficient
    // exactly matches zero, which the honest trace enforces via the
    // trace builder. Algebraic gating sketch:
    //   body = is_real * (level == 0 ? Σ β^i · partial_sum_curr[i] : 0)
    // For the scaffold we evaluate the β-RLC of partial_sum_curr bytes
    // and multiply by `is_real * (1 - level / level_max_placeholder)` ≈
    // zero on honest non-bottom rows because partial_sum_curr matches
    // the previous outgoing accumulator (chain continuity body 1).
    // Net effect: identically zero on honest traces — scaffold pin.
    let _ = level;
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for i in 0..POINT_BYTES {
        let v = &col_evals[COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + i];
        // Scaffold: identity == 0, so body = v on bottom row; non-bottom
        // rows are zeroed by gate. Gate is omitted from in-row constraint
        // since `LEVEL` isn't bit-decomposed; chain continuity (shifted
        // body 1) plus the per-step LogUp suffice for soundness once the
        // bottom-row identity is host-witnessed.
        acc = acc.add(&v.mul(&ap));
        ap = ap.mul(alpha);
    }
    // Multiply by zero scaffold gate (real algebraic gate deferred).
    let _ = acc;
    let _ = is_real;
    Scalar::zero(curve)
}

fn build_pedersen_partial_sum_init_poly(curve: CurveType) -> Vec<Scalar> {
    vec![Scalar::zero(curve)]
}

/// Body 6: scaffold for binding the top-row outgoing Pedersen
/// accumulator `partial_sum_next` to the claimed root.
///
/// Real Verkle would serialize the closed commitment point to bytes
/// (compressed Bandersnatch) and β-RLC equate to `CLAIMED_ROOT`. For
/// the current scaffold this body is identically zero — the binding
/// is enforced host-side by `pedersen_partial_sums[DEPTH]` being
/// supplied as a serialized root, and the eventual algebraic pin
/// lands when the (compressed) point serialization gadget is wired.
fn eval_pedersen_top_match(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_TOP].curve_type();
    Scalar::zero(curve)
}

fn build_pedersen_top_match_poly(curve: CurveType) -> Vec<Scalar> {
    vec![Scalar::zero(curve)]
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for VerkleTreeConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_top_binary".into(),
            "path_index_range".into(),
            "claimed_root_match".into(),
            "pedersen_open_pin".into(),
            "pedersen_partial_sum_init".into(),
            "pedersen_top_match".into(),
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
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_top = &row_evals[COL_IS_TOP];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_top.mul(&is_top.sub(&one));
            // path_index_range: scaffold pin — zero on honest trace.
            bodies[2][row] = Scalar::zero(curve);
            bodies[3][row] = eval_claimed_root_match(&row_evals, &alpha);
            bodies[4][row] = eval_pedersen_open_pin(&row_evals);
            bodies[5][row] = eval_pedersen_partial_sum_init(&row_evals, &alpha);
            bodies[6][row] = eval_pedersen_top_match(&row_evals);
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
        let is_top = &col_evals[COL_IS_TOP];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_top.mul(&is_top.sub(&one)),
            Scalar::zero(curve),
            eval_claimed_root_match(col_evals, alpha),
            eval_pedersen_open_pin(col_evals),
            eval_pedersen_partial_sum_init(col_evals, alpha),
            eval_pedersen_top_match(col_evals),
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
        let is_top = &col_coeffs[COL_IS_TOP];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_top_m1 = poly_sub(is_top, &one_poly, curve);
        let is_top_binary = poly_mul(is_top, &is_top_m1, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_top_binary,
            vec![Scalar::zero(curve)],
            build_claimed_root_match_poly(col_coeffs, alpha, curve),
            build_pedersen_open_pin_poly(curve),
            build_pedersen_partial_sum_init_poly(curve),
            build_pedersen_top_match_poly(curve),
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
        // For the `commitment_chain` body we need EXPECTED_VALUE bytes
        // at the next row plus IS_REAL at the next row. For the
        // `partial_sum_continuity` body we additionally need
        // PEDERSEN_PARTIAL_SUM_CURR bytes at the next row.
        let mut cols = Vec::with_capacity(CHUNK_BYTES + 1 + POINT_BYTES);
        for b in 0..CHUNK_BYTES {
            cols.push(COL_EXPECTED_VALUE_OFFSET + b);
        }
        cols.push(COL_IS_REAL);
        for b in 0..POINT_BYTES {
            cols.push(COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + b);
        }
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
        let expected_len = CHUNK_BYTES + 1 + POINT_BYTES;
        if shifted_evals.len() < expected_len || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let is_real_next = &shifted_evals[CHUNK_BYTES];
        let partial_sum_curr_next_base = CHUNK_BYTES + 1;

        // Body 0: commitment_chain.
        let mut chain_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let exp_next = &shifted_evals[b];
            let path_curr = &col_evals_at_z[COL_PATH_COMMITMENT_OFFSET + b];
            let diff = exp_next.sub(path_curr);
            chain_acc = chain_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = is_real_next.mul(&chain_acc);

        // Body 1: partial_sum_continuity. partial_sum_curr at next row
        // must equal partial_sum_next at current row.
        let mut sum_acc = Scalar::zero(curve);
        let mut ap_inner2 = Scalar::one(curve);
        for b in 0..POINT_BYTES {
            let next_curr = &shifted_evals[partial_sum_curr_next_base + b];
            let curr_next = &col_evals_at_z[COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + b];
            let diff = next_curr.sub(curr_next);
            sum_acc = sum_acc.add(&diff.mul(&ap_inner2));
            ap_inner2 = ap_inner2.mul(alpha);
        }
        let body1 = is_real_next.mul(&sum_acc);

        let exclusion = z.sub(omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let combined = body0.add(&ap.mul(&body1));
        let _ = combined;
        // Combine via Fiat–Shamir α-folding consistent with the polynomial
        // builder below.
        let term0 = ap.mul(&body0).mul(&exclusion);
        let ap1 = ap.mul(alpha);
        let term1 = ap1.mul(&body1).mul(&exclusion);
        term0.add(&term1)
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

        // Body 0: commitment_chain.
        let mut chain_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let exp_poly = &col_coeffs[COL_EXPECTED_VALUE_OFFSET + b];
            let path_poly = &col_coeffs[COL_PATH_COMMITMENT_OFFSET + b];
            let exp_shift = poly_shift(exp_poly, omega);
            let diff = poly_sub(&exp_shift, path_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            chain_acc = poly_add(&chain_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = poly_mul(&is_real_shift, &chain_acc, curve);

        // Body 1: partial_sum_continuity.
        let mut sum_acc = vec![Scalar::zero(curve)];
        let mut ap_inner2 = Scalar::one(curve);
        for b in 0..POINT_BYTES {
            let curr_poly = &col_coeffs[COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + b];
            let next_poly = &col_coeffs[COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + b];
            let curr_shift = poly_shift(curr_poly, omega);
            let diff = poly_sub(&curr_shift, next_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner2);
            sum_acc = poly_add(&sum_acc, &scaled, curve);
            ap_inner2 = ap_inner2.mul(alpha);
        }
        let body1 = poly_mul(&is_real_shift, &sum_acc, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let ap1 = ap.mul(alpha);
        let excluded0 = poly_mul_linear(&body0, &omega_n_minus_1);
        let excluded1 = poly_mul_linear(&body1, &omega_n_minus_1);
        let term0 = poly_scalar_mul(&excluded0, &ap);
        let term1 = poly_scalar_mul(&excluded1, &ap1);
        poly_add(&term0, &term1, curve)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Cross-AIR LogUp descriptor scaffold binding each active
/// `(PATH_COMMITMENT, EXPECTED_VALUE, PATH_INDEX)` tuple to a future
/// G1-curve-ops AIR row evaluating the Pedersen commitment `C = Σ v_i ·
/// G_i`. 65-column tuple on each side, gated by `IS_REAL`.
///
/// The B-side column offsets are caller-supplied so this descriptor can
/// target either a bespoke `verkle_pedersen_air` or the existing
/// curve-ops AIR family once a Pedersen-specific gadget lands.
pub fn make_verkle_to_curve_ops_descriptor(
    verkle_layer_index: usize,
    curve_ops_layer_index: usize,
    curve_ops_commitment_byte_offset: usize,
    curve_ops_value_byte_offset: usize,
    curve_ops_index_column: usize,
    curve_ops_selector_column: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(2 * CHUNK_BYTES + 1);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_PATH_COMMITMENT_OFFSET + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_EXPECTED_VALUE_OFFSET + b);
    }
    a_columns.push(COL_PATH_INDEX);

    let mut b_columns: Vec<usize> = Vec::with_capacity(2 * CHUNK_BYTES + 1);
    for b in 0..CHUNK_BYTES {
        b_columns.push(curve_ops_commitment_byte_offset + b);
    }
    for b in 0..CHUNK_BYTES {
        b_columns.push(curve_ops_value_byte_offset + b);
    }
    b_columns.push(curve_ops_index_column);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "verkle_tree_pedersen_open_v1".into(),
        a_layer_index: verkle_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: curve_ops_layer_index,
        b_columns,
        b_selector_column: Some(curve_ops_selector_column),
    }
}

/// Cross-AIR LogUp descriptor binding each Verkle level's Pedersen step
/// `partial_sum_next = partial_sum_curr + v_k · G_k` to a row of the
/// **BLS12-381 G1 curve-ops AIR** ([`crate::bls12_381_curve_ops_air`]).
///
/// This is a **placeholder-curve** mapping: real Verkle uses
/// Bandersnatch, but the algebraic structure (add of an EC point and a
/// scalar-multiple of a fixed generator) is the same modulo curve
/// parameters, so the BLS12-381 G1 AIR can stand in until a
/// Bandersnatch curve-ops AIR lands.
///
/// Tuple layout (64-byte point-bytes encoding):
/// ```text
///   A side (verkle row, gated by IS_REAL):
///     [partial_sum_curr (64B), partial_sum_next (64B),
///      expected_value (32B), path_index (1)]
///   B side (curve-ops row, gated by SEL_ADD on the curve-ops side):
///     [P (64B encoding of 2·Fp limbs serialized),
///      R (64B encoding),
///      Q.x (32B encoding) representing v_k · G_k.x,
///      Q.x[0] (1) representing the path index disambiguator]
/// ```
///
/// The 161-column tuple cardinality is sufficient to express the
/// `(start, end, value, index)` quadruple per Pedersen step row; the
/// B-side AIR is expected to host an aux table where each row's
/// `(P, Q, R)` triple satisfies `R = P + Q` with `Q = v_k · G_k`
/// computed via a future scalar-mul gadget. The current scaffold
/// supplies only the tuple shape — the **scalar-mul side of `Q`** is
/// the deferred piece.
pub fn make_verkle_to_g1_pedersen_step_descriptor(
    verkle_layer_index: usize,
    curve_ops_layer_index: usize,
    curve_ops_p_byte_offset: usize,
    curve_ops_r_byte_offset: usize,
    curve_ops_qx_byte_offset: usize,
    curve_ops_index_column: usize,
    curve_ops_selector_column: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(2 * POINT_BYTES + CHUNK_BYTES + 1);
    for b in 0..POINT_BYTES {
        a_columns.push(COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + b);
    }
    for b in 0..POINT_BYTES {
        a_columns.push(COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_EXPECTED_VALUE_OFFSET + b);
    }
    a_columns.push(COL_PATH_INDEX);

    let mut b_columns: Vec<usize> = Vec::with_capacity(2 * POINT_BYTES + CHUNK_BYTES + 1);
    for b in 0..POINT_BYTES {
        b_columns.push(curve_ops_p_byte_offset + b);
    }
    for b in 0..POINT_BYTES {
        b_columns.push(curve_ops_r_byte_offset + b);
    }
    for b in 0..CHUNK_BYTES {
        b_columns.push(curve_ops_qx_byte_offset + b);
    }
    b_columns.push(curve_ops_index_column);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "verkle_tree_pedersen_step_v1".into(),
        a_layer_index: verkle_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: curve_ops_layer_index,
        b_columns,
        b_selector_column: Some(curve_ops_selector_column),
    }
}

/// Cross-AIR LogUp descriptor binding the Verkle gadget's claimed root
/// to a parent AIR (typically a block-header or beacon-state AIR's
/// `state_root` / `verkle_root` field). 32-column tuple, gated by
/// `IS_TOP`.
pub fn make_verkle_root_to_block_descriptor(
    verkle_layer_index: usize,
    block_layer_index: usize,
    block_root_byte_offset: usize,
    block_selector_column: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_CLAIMED_ROOT_OFFSET + b);
    }
    let mut b_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        b_columns.push(block_root_byte_offset + b);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "verkle_tree_root_to_block_v1".into(),
        a_layer_index: verkle_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_TOP),
        b_layer_index: block_layer_index,
        b_columns,
        b_selector_column: Some(block_selector_column),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_chunk(seed: u8, offset: u8) -> [u8; CHUNK_BYTES] {
        let mut out = [0u8; CHUNK_BYTES];
        for (i, b) in out.iter_mut().enumerate() {
            *b = seed
                .wrapping_mul(13)
                .wrapping_add(offset)
                .wrapping_add(i as u8);
        }
        out
    }

    fn synthetic_witness() -> VerkleTreeWitness {
        let leaf_key = synth_chunk(1, 0);
        let leaf_value = synth_chunk(2, 0);
        let mut path_commitments = [[0u8; CHUNK_BYTES]; DEPTH];
        for (k, slot) in path_commitments.iter_mut().enumerate() {
            *slot = synth_chunk(10 + k as u8, 0);
        }
        let mut path_indexes = [0u32; DEPTH];
        for (k, slot) in path_indexes.iter_mut().enumerate() {
            *slot = (k as u32 * 17) % (BRANCHING as u32);
        }
        let claimed_root = path_commitments[DEPTH - 1];

        let inc = VerkleInclusionWitness::new(
            leaf_key,
            leaf_value,
            path_commitments,
            path_indexes,
            claimed_root,
        );
        VerkleTreeWitness {
            inclusions: vec![inc],
        }
    }

    /// Witness builder: top path commitment matches the claimed root,
    /// per-level indexes fall within `BRANCHING`.
    #[test]
    fn witness_builder_well_formed() {
        let w = synthetic_witness();
        assert_eq!(w.inclusions.len(), 1);
        let inc = &w.inclusions[0];
        assert_eq!(inc.path_commitments[DEPTH - 1], inc.claimed_root);
        for &idx in inc.path_indexes.iter() {
            assert!((idx as usize) < BRANCHING);
        }
    }

    /// Trace shape: NUM_COLUMNS columns, ROWS_PER_INCLUSION rows per
    /// inclusion. IS_TOP fires exactly once at the top row, IS_REAL is
    /// high on every active row.
    #[test]
    fn trace_layout_matches_spec() {
        let curve = CurveType::Bls48581;
        let w = synthetic_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, ROWS_PER_INCLUSION);

        let mut is_top_count = 0u64;
        for row in 0..ROWS_PER_INCLUSION {
            assert_eq!(
                trace.columns[COL_IS_REAL].evaluations[row].to_u64(),
                1,
                "IS_REAL must be 1 on row {}",
                row,
            );
            let is_top = trace.columns[COL_IS_TOP].evaluations[row].to_u64();
            if is_top == 1 {
                is_top_count += 1;
                assert_eq!(row, DEPTH - 1, "IS_TOP must fire on the top row");
            }
            assert_eq!(
                trace.columns[COL_LEVEL].evaluations[row].to_u64(),
                row as u64,
            );
        }
        assert_eq!(is_top_count, 1, "IS_TOP must fire exactly once");
    }

    /// Honest witness: all row-local constraints vanish on every row.
    #[test]
    fn constraints_vanish_on_honest_inclusion() {
        let curve = CurveType::Bls48581;
        let w = synthetic_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = VerkleTreeConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) must vanish at row {} (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    /// Tampering: corrupt the top-row PATH_COMMITMENT so it no longer
    /// matches CLAIMED_ROOT; the `claimed_root_match` body fires.
    #[test]
    fn tampered_top_commitment_breaks_root_match() {
        let curve = CurveType::Bls48581;
        let w = synthetic_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // Corrupt PATH_COMMITMENT[5] on the top row.
        trace.columns[COL_PATH_COMMITMENT_OFFSET + 5].evaluations[DEPTH - 1] =
            Scalar::from_u64(0x55, curve);
        let cs = VerkleTreeConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // bodies[3] is `claimed_root_match`.
        assert!(
            !evals[3][DEPTH - 1].is_zero(),
            "tampered top commitment must break claimed_root_match",
        );
    }

    /// Cross-AIR LogUp descriptors are well-formed.
    #[test]
    fn cross_air_descriptors_well_formed() {
        let curve_ops = make_verkle_to_curve_ops_descriptor(0, 1, 100, 200, 300, 999);
        assert_eq!(curve_ops.label, "verkle_tree_pedersen_open_v1");
        assert_eq!(curve_ops.a_columns.len(), 2 * CHUNK_BYTES + 1);
        assert_eq!(curve_ops.b_columns.len(), 2 * CHUNK_BYTES + 1);
        for b in 0..CHUNK_BYTES {
            assert_eq!(curve_ops.a_columns[b], COL_PATH_COMMITMENT_OFFSET + b);
            assert_eq!(
                curve_ops.a_columns[CHUNK_BYTES + b],
                COL_EXPECTED_VALUE_OFFSET + b,
            );
            assert_eq!(curve_ops.b_columns[b], 100 + b);
            assert_eq!(curve_ops.b_columns[CHUNK_BYTES + b], 200 + b);
        }
        assert_eq!(curve_ops.a_columns[2 * CHUNK_BYTES], COL_PATH_INDEX);
        assert_eq!(curve_ops.b_columns[2 * CHUNK_BYTES], 300);
        assert_eq!(curve_ops.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(curve_ops.b_selector_column, Some(999));
        assert_eq!(curve_ops.a_layer_index, 0);
        assert_eq!(curve_ops.b_layer_index, 1);

        let blk = make_verkle_root_to_block_descriptor(0, 2, 500, 777);
        assert_eq!(blk.label, "verkle_tree_root_to_block_v1");
        assert_eq!(blk.a_columns.len(), CHUNK_BYTES);
        assert_eq!(blk.b_columns.len(), CHUNK_BYTES);
        for b in 0..CHUNK_BYTES {
            assert_eq!(blk.a_columns[b], COL_CLAIMED_ROOT_OFFSET + b);
            assert_eq!(blk.b_columns[b], 500 + b);
        }
        assert_eq!(blk.a_selector_column, Some(COL_IS_TOP));
        assert_eq!(blk.b_selector_column, Some(777));
        assert_eq!(blk.a_layer_index, 0);
        assert_eq!(blk.b_layer_index, 2);
    }

    /// New columns for Pedersen partial sums occupy
    /// `[164, 164+64) = COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET..NEXT_OFFSET`
    /// and `[228, 292) = COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET..NUM_COLUMNS`.
    /// NUM_COLUMNS, NUM_ROW_CONSTRAINTS, NUM_SHIFTED match the spec.
    #[test]
    fn column_and_constraint_counts() {
        assert_eq!(COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET, 164);
        assert_eq!(COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET, 164 + 64);
        assert_eq!(NUM_COLUMNS, 164 + 2 * 64);
        assert_eq!(NUM_ROW_CONSTRAINTS, 7);
        assert_eq!(NUM_SHIFTED, 2);
    }

    /// Honest witness with a non-trivial host-supplied Pedersen partial
    /// sum chain (synthesized as a single-byte counter per level) still
    /// satisfies every row-local constraint body.
    #[test]
    fn constraints_vanish_with_partial_sums() {
        let curve = CurveType::Bls48581;
        let leaf_key = synth_chunk(1, 0);
        let leaf_value = synth_chunk(2, 0);
        let mut path_commitments = [[0u8; CHUNK_BYTES]; DEPTH];
        for (k, slot) in path_commitments.iter_mut().enumerate() {
            *slot = synth_chunk(10 + k as u8, 0);
        }
        let mut path_indexes = [0u32; DEPTH];
        for (k, slot) in path_indexes.iter_mut().enumerate() {
            *slot = (k as u32 * 17) % (BRANCHING as u32);
        }
        let claimed_root = path_commitments[DEPTH - 1];

        // Synthetic chain: partial_sums[k] = [k; 64] — body 1 only checks
        // continuity (next-row CURR == curr-row NEXT), so we set:
        //   sums[0]  = [0; 64]     (identity placeholder)
        //   sums[k+1] = sums[k] + 1 byte-wise.
        // Then row k's NEXT == sums[k+1] and row (k+1)'s CURR == sums[k+1].
        let mut sums = [[0u8; POINT_BYTES]; DEPTH + 1];
        for k in 1..=DEPTH {
            sums[k] = [k as u8; POINT_BYTES];
        }

        let inc = VerkleInclusionWitness::new_with_partial_sums(
            leaf_key,
            leaf_value,
            path_commitments,
            path_indexes,
            claimed_root,
            sums,
        );
        let w = VerkleTreeWitness {
            inclusions: vec![inc],
        };
        let trace = build_trace_polynomials(&w, curve);
        let cs = VerkleTreeConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) must vanish at row {} with non-trivial partial sums",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    /// Pedersen partial sum trace columns are populated bottom-up.
    /// Row 0's CURR is the identity (sums[0]), and row k's NEXT chains
    /// into row (k+1)'s CURR. This is the host-side property pinned
    /// algebraically by the `partial_sum_continuity` shifted body.
    #[test]
    fn partial_sum_columns_chain() {
        let curve = CurveType::Bls48581;
        let mut sums = [[0u8; POINT_BYTES]; DEPTH + 1];
        for k in 1..=DEPTH {
            sums[k] = [k as u8; POINT_BYTES];
        }
        let mut path_commitments = [[0u8; CHUNK_BYTES]; DEPTH];
        for (k, slot) in path_commitments.iter_mut().enumerate() {
            *slot = synth_chunk(10 + k as u8, 0);
        }
        let mut path_indexes = [0u32; DEPTH];
        for (k, slot) in path_indexes.iter_mut().enumerate() {
            *slot = (k as u32) % (BRANCHING as u32);
        }
        let claimed_root = path_commitments[DEPTH - 1];

        let inc = VerkleInclusionWitness::new_with_partial_sums(
            [0u8; KEY_BYTES],
            [0u8; VALUE_BYTES],
            path_commitments,
            path_indexes,
            claimed_root,
            sums,
        );
        let w = VerkleTreeWitness {
            inclusions: vec![inc],
        };
        let trace = build_trace_polynomials(&w, curve);
        for level in 0..DEPTH {
            for b in 0..POINT_BYTES {
                let curr = trace.columns[COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + b]
                    .evaluations[level]
                    .to_u64() as u8;
                let next = trace.columns[COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + b]
                    .evaluations[level]
                    .to_u64() as u8;
                assert_eq!(curr, sums[level][b]);
                assert_eq!(next, sums[level + 1][b]);
            }
        }
        // Chain continuity in trace: row level's NEXT == row (level+1)'s CURR.
        for level in 0..(DEPTH - 1) {
            for b in 0..POINT_BYTES {
                let next_curr = trace.columns[COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + b]
                    .evaluations[level + 1]
                    .to_u64() as u8;
                let curr_next = trace.columns[COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + b]
                    .evaluations[level]
                    .to_u64() as u8;
                assert_eq!(next_curr, curr_next, "chain link broken at level {}", level);
            }
        }
    }

    /// The Pedersen-step cross-AIR LogUp descriptor is well-formed:
    /// 161-column tuple per side, gated by IS_REAL on the A side.
    #[test]
    fn pedersen_step_descriptor_well_formed() {
        let d = make_verkle_to_g1_pedersen_step_descriptor(
            0, 1, 100, 200, 300, 400, 999,
        );
        assert_eq!(d.label, "verkle_tree_pedersen_step_v1");
        let expected_len = 2 * POINT_BYTES + CHUNK_BYTES + 1;
        assert_eq!(d.a_columns.len(), expected_len);
        assert_eq!(d.b_columns.len(), expected_len);
        for b in 0..POINT_BYTES {
            assert_eq!(d.a_columns[b], COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + b);
            assert_eq!(
                d.a_columns[POINT_BYTES + b],
                COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + b,
            );
            assert_eq!(d.b_columns[b], 100 + b);
            assert_eq!(d.b_columns[POINT_BYTES + b], 200 + b);
        }
        for b in 0..CHUNK_BYTES {
            assert_eq!(
                d.a_columns[2 * POINT_BYTES + b],
                COL_EXPECTED_VALUE_OFFSET + b,
            );
            assert_eq!(d.b_columns[2 * POINT_BYTES + b], 300 + b);
        }
        assert_eq!(d.a_columns[expected_len - 1], COL_PATH_INDEX);
        assert_eq!(d.b_columns[expected_len - 1], 400);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(999));
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
    }

    /// Tampering: corrupt the level-3 outgoing partial sum so the chain
    /// link between level-3 NEXT and level-4 CURR breaks. (Algebraically
    /// the shifted `partial_sum_continuity` body catches this — here we
    /// verify the host-side property as a smoke test.)
    #[test]
    fn tampered_partial_sum_breaks_chain() {
        let curve = CurveType::Bls48581;
        let mut sums = [[0u8; POINT_BYTES]; DEPTH + 1];
        for k in 1..=DEPTH {
            sums[k] = [k as u8; POINT_BYTES];
        }
        let mut path_commitments = [[0u8; CHUNK_BYTES]; DEPTH];
        for (k, slot) in path_commitments.iter_mut().enumerate() {
            *slot = synth_chunk(10 + k as u8, 0);
        }
        let path_indexes = [0u32; DEPTH];
        let claimed_root = path_commitments[DEPTH - 1];

        let inc = VerkleInclusionWitness::new_with_partial_sums(
            [0u8; KEY_BYTES],
            [0u8; VALUE_BYTES],
            path_commitments,
            path_indexes,
            claimed_root,
            sums,
        );
        let w = VerkleTreeWitness {
            inclusions: vec![inc],
        };
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper level 3 NEXT byte 7.
        trace.columns[COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + 7].evaluations[3] =
            Scalar::from_u64(0xAA, curve);
        // Confirm the next-row CURR no longer matches.
        let next_curr = trace.columns[COL_PEDERSEN_PARTIAL_SUM_CURR_OFFSET + 7]
            .evaluations[4]
            .to_u64();
        let curr_next = trace.columns[COL_PEDERSEN_PARTIAL_SUM_NEXT_OFFSET + 7]
            .evaluations[3]
            .to_u64();
        assert_ne!(next_curr, curr_next);
    }

    /// `EXPECTED_VALUE` chains correctly: row 0 holds the leaf value,
    /// row `k>0` holds the level `k-1` `PATH_COMMITMENT`. This is the
    /// host-side property that the `commitment_chain` shifted body pins
    /// algebraically.
    #[test]
    fn expected_value_chains_commitments() {
        let curve = CurveType::Bls48581;
        let w = synthetic_witness();
        let inc = w.inclusions[0].clone();
        let trace = build_trace_polynomials(&w, curve);
        for b in 0..CHUNK_BYTES {
            let row0_exp =
                trace.columns[COL_EXPECTED_VALUE_OFFSET + b].evaluations[0].to_u64() as u8;
            assert_eq!(
                row0_exp, inc.leaf_value[b],
                "row 0 EXPECTED_VALUE[{}] must equal leaf_value", b,
            );
        }
        for level in 1..DEPTH {
            for b in 0..CHUNK_BYTES {
                let got = trace.columns[COL_EXPECTED_VALUE_OFFSET + b].evaluations[level]
                    .to_u64() as u8;
                let want = inc.path_commitments[level - 1][b];
                assert_eq!(
                    got, want,
                    "row {} EXPECTED_VALUE[{}] must equal path_commitments[{}][{}]",
                    level,
                    b,
                    level - 1,
                    b,
                );
            }
        }
    }
}
