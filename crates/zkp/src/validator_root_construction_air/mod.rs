//! Validator-registry batch-root construction AIR (small batches).
//!
//! Sibling to [`crate::validator_registry_air`] which proves *inclusion*
//! of one validator at a known index within a depth-40 registry tree.
//!
//! This AIR proves the **complete merkleization** of a small validator
//! subset of size `N ∈ [1, MAX_BATCH]` (with `MAX_BATCH = 256`). For a
//! testnet / devnet with a few hundred validators the full registry root
//! can be constructed in-circuit using one SHA-256 invocation per
//! internal-tree node + the standard `mix_in_length` step.
//!
//! ## Witness shape
//!
//! Caller supplies the per-validator `hash_tree_root` chunks
//! (already proven correct by [`crate::validator_htr_air`] in the
//! companion linkage). The witness builder pads the leaf vector to the
//! next power of two with [`ZERO_CHUNK`] and walks the tree bottom-up,
//! emitting one row per `(LEFT, RIGHT, PARENT)` sha256 invocation. The
//! final row is the `mix_in_length` step that produces the registry
//! root.
//!
//! Note: this AIR only models the **populated subtree** rooted at depth
//! `ceil(log2(N))`. The full `2^40` depth is **not** materialized
//! algebraically; instead it is intended to be combined with a
//! single-value `mix_in_length(zero_extend_to_depth_40(subtree_root))`
//! step host-side, or with a small wrapper AIR that re-uses
//! `validator_registry_air`'s `mix_in_length` row to bridge the gap
//! when the registry is exactly `2^k` for `k ≤ ceil(log2(MAX_BATCH))`.
//! For testnets that pin the type-level limit at the batch size itself
//! (e.g. `VALIDATOR_REGISTRY_LIMIT = 256`) the produced root after the
//! `mix_in_length` row equals the SSZ canonical registry hash_tree_root
//! directly.
//!
//! ## Row layout
//!
//! Maximum row count is `MAX_BATCH = 256`: at most `MAX_BATCH - 1` pair
//! hashes for a 256-leaf tree + 1 `mix_in_length` row = `MAX_BATCH`
//! rows. The trace builder pads to the next power of two for FFT
//! sizing.
//!
//! Per-row columns:
//!   * `LEFT[0..32]`, `RIGHT[0..32]`, `PARENT[0..32]` — the sha256
//!     pair invocation matched against [`crate::sha256_extract`] via
//!     the pair-hash cross-AIR LogUp.
//!   * `LEAF_INDEX` — for leaf-pickup rows, the validator index of
//!     `LEFT` (the index of `RIGHT` is `LEAF_INDEX + 1`). Zero on
//!     non-leaf rows.
//!   * `IS_REAL` — selector gating both row-local bodies and the
//!     pair-hash cross-AIR LogUp.
//!   * `IS_LEAF` — selector flagging rows that consume two leaf
//!     hashes (i.e. layer-0 pair rows). Used by the
//!     validator-htr leaf cross-AIR LogUp descriptor.
//!   * `IS_MIX` — selector flagging the final `mix_in_length` row.
//!   * `IS_ROOT` — selector flagging the row that outputs the
//!     pre-mix subtree root. The downstream registry-root descriptor
//!     binds `PARENT` of that row to a downstream consumer.
//!   * `IS_FINAL` — selector flagging the row whose `PARENT` is the
//!     final registry root (= the mix-in-length output). Equal to
//!     `IS_MIX` (kept as an explicit alias for descriptor clarity).
//!   * `CLAIMED_ROOT[0..32]` — the final registry root, replicated
//!     identically across all real rows.
//!
//! ## Algebraic constraints
//!
//! 6 row-local bodies, all gated by `IS_REAL`:
//!   0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `is_leaf_binary` — `IS_REAL · IS_LEAF · (IS_LEAF − 1) = 0`
//!   2. `is_mix_binary`  — `IS_REAL · IS_MIX  · (IS_MIX  − 1) = 0`
//!   3. `is_root_binary` — `IS_REAL · IS_ROOT · (IS_ROOT − 1) = 0`
//!   4. `is_final_eq_is_mix` — `IS_REAL · (IS_FINAL − IS_MIX) = 0`
//!   5. `final_output_eq_claimed_root` — β-RLC over 32 sub-bodies of
//!      `IS_MIX · (PARENT[i] − CLAIMED_ROOT[i]) = 0` (gated by IS_MIX
//!      so it only fires on the mix row).
//!
//! 1 shifted body:
//!   0. `claimed_root_constancy` — `IS_REAL(ω·X) · Σ β^i ·
//!      (CLAIMED_ROOT_i(ω·X) − CLAIMED_ROOT_i(X)) = 0`
//!      (excluded at wrap-around).
//!
//! ## What this AIR does NOT prove (algebraically; bound via cross-AIR LogUps)
//!
//! * `PARENT = sha256(LEFT || RIGHT)` for every active row — bound by
//!   the pair-hash linkage descriptor (cross-AIR LogUp to
//!   `sha256_extract`).
//! * `LEFT = leaf[i]` and `RIGHT = leaf[i+1]` for leaf rows — bound
//!   by the leaf-to-htr descriptor (cross-AIR LogUp to
//!   `validator_htr_air`). Note: this descriptor is over-published on
//!   the A side — it requires the prover to publish a tuple per leaf
//!   row, which the host-side ordering convention pins to the
//!   per-validator HTR rows in index order. The internal tree-level
//!   chaining (parent[k+1].left = parent[k].hash for the right
//!   pairing) is **NOT** algebraically pinned here (deferred); a
//!   subsequent revision will add relay-style cross-row constraints
//!   analogous to [`crate::beacon_block_header_pair_air`].
//! * `CLAIMED_ROOT = mix_in_length(subtree_root, registry_length)` —
//!   the `mix_in_length` row's pair-hash cross-AIR LogUp pins the
//!   sha256 invocation; the row-local `final_output_eq_claimed_root`
//!   body pins the output to the published CLAIMED_ROOT column. The
//!   `RIGHT` chunk on the mix row is expected to be the length chunk
//!   (`registry_length_le[0..8] || 0[8..32]`); this is bound to the
//!   public input host-side (no on-trace algebraic length-decomposition
//!   in this scaffold).

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Bytes per merkle chunk (SHA-256 output).
pub const CHUNK_BYTES: usize = 32;

/// Maximum batch size supported by this scaffold. Each batch occupies
/// at most `MAX_BATCH` rows: up to `MAX_BATCH - 1` pair-hash rows + 1
/// `mix_in_length` row.
pub const MAX_BATCH: usize = 256;

/// Maximum tree depth (= log2(MAX_BATCH)).
pub const MAX_DEPTH: usize = 8;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_LEFT_OFFSET: usize = 0;                              // 0..32
pub const COL_RIGHT_OFFSET: usize = COL_LEFT_OFFSET + CHUNK_BYTES; // 32..64
pub const COL_PARENT_OFFSET: usize = COL_RIGHT_OFFSET + CHUNK_BYTES; // 64..96
pub const COL_CLAIMED_ROOT_OFFSET: usize = COL_PARENT_OFFSET + CHUNK_BYTES; // 96..128

pub const COL_LEAF_INDEX: usize = COL_CLAIMED_ROOT_OFFSET + CHUNK_BYTES; // 128
pub const COL_IS_REAL: usize = COL_LEAF_INDEX + 1;                       // 129
pub const COL_IS_LEAF: usize = COL_IS_REAL + 1;                          // 130
pub const COL_IS_MIX: usize = COL_IS_LEAF + 1;                           // 131
pub const COL_IS_ROOT: usize = COL_IS_MIX + 1;                           // 132
pub const COL_IS_FINAL: usize = COL_IS_ROOT + 1;                         // 133

pub const NUM_COLUMNS: usize = COL_IS_FINAL + 1;                         // 134

/// 6 row-local constraints (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 6;
/// 1 shifted constraint (claimed_root_constancy).
pub const NUM_SHIFTED: usize = 1;

// ─── Witness types ────────────────────────────────────────────────────

/// One pair-hash row emitted during merkleization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairHashRow {
    pub left: [u8; CHUNK_BYTES],
    pub right: [u8; CHUNK_BYTES],
    pub parent: [u8; CHUNK_BYTES],
    /// Validator index of `LEFT` when this row consumes two leaves
    /// (i.e. `is_leaf = true`). `RIGHT` then corresponds to index
    /// `leaf_index + 1`. Zero on non-leaf and mix rows.
    pub leaf_index: u64,
    pub is_leaf: bool,
    pub is_mix: bool,
    pub is_root: bool,
}

/// Per-batch witness. The `validator_htrs` field is the input set; the
/// constructor pads it to the next power of two with [`ZERO_CHUNK`].
/// The trace builder consumes the populated `rows` list directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorRootConstructionWitness {
    /// Original (un-padded) per-validator HTRs. Cap: `MAX_BATCH`.
    pub validator_htrs: Vec<[u8; CHUNK_BYTES]>,
    /// Padded leaf vector — length = `next_pow_of_two(validator_htrs.len())`.
    pub padded_leaves: Vec<[u8; CHUNK_BYTES]>,
    /// Pair-hash rows: `padded_leaves.len() − 1` internal hashes followed
    /// by 1 `mix_in_length` row.
    pub rows: Vec<PairHashRow>,
    /// Mixed-in registry length (typically the un-padded batch size).
    pub registry_length: u64,
    /// Computed registry root = mix_in_length(subtree_root, registry_length).
    pub registry_root: [u8; CHUNK_BYTES],
}

/// Pad to next power of 2, lower bound 1.
fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

impl ValidatorRootConstructionWitness {
    /// Build a witness from a batch of validator HTRs. Pads to the
    /// next power of two with [`crate::ssz::ZERO_CHUNK`]. Computes
    /// every internal hash + the `mix_in_length` row whose `parent`
    /// is the registry root.
    ///
    /// Panics if `htrs.len() > MAX_BATCH`.
    pub fn from_validator_htrs(htrs: &[[u8; CHUNK_BYTES]]) -> Self {
        Self::from_validator_htrs_with_length(htrs, htrs.len() as u64)
    }

    /// Same as `from_validator_htrs` but lets the caller pin a custom
    /// registry length for `mix_in_length` (useful for SSZ List shapes
    /// where the type-level cap exceeds the batch size).
    pub fn from_validator_htrs_with_length(
        htrs: &[[u8; CHUNK_BYTES]],
        registry_length: u64,
    ) -> Self {
        assert!(
            htrs.len() <= MAX_BATCH,
            "batch size {} exceeds MAX_BATCH={}",
            htrs.len(),
            MAX_BATCH,
        );

        // For zero-length batches treat as length 0 → root = zero_hash(0).
        let n_unpadded = htrs.len().max(1);
        let padded_len = next_pow2(n_unpadded);

        let mut padded_leaves: Vec<[u8; CHUNK_BYTES]> =
            htrs.iter().copied().collect();
        // Pad with ZERO_CHUNK.
        padded_leaves.resize(padded_len, crate::ssz::ZERO_CHUNK);

        // Bottom-up tree walk. At layer k (0-indexed from leaves), pair
        // adjacent nodes and emit one row each.
        let mut rows: Vec<PairHashRow> = Vec::with_capacity(padded_len);
        let mut layer: Vec<[u8; CHUNK_BYTES]> = padded_leaves.clone();
        let mut current_depth: usize = 0;
        while layer.len() > 1 {
            let mut next_layer: Vec<[u8; CHUNK_BYTES]> =
                Vec::with_capacity(layer.len() / 2);
            for i in 0..(layer.len() / 2) {
                let left = layer[2 * i];
                let right = layer[2 * i + 1];
                let parent = crate::sha256::sha256_pair(&left, &right);
                next_layer.push(parent);

                let is_leaf = current_depth == 0;
                rows.push(PairHashRow {
                    left,
                    right,
                    parent,
                    leaf_index: if is_leaf { 2 * i as u64 } else { 0 },
                    is_leaf,
                    is_mix: false,
                    // Mark the root-producing internal row (top layer
                    // before mix). When the tree is just one leaf
                    // (padded_len == 1) there are no internal rows;
                    // the mix row consumes the single leaf directly.
                    is_root: layer.len() == 2,
                });
            }
            layer = next_layer;
            current_depth += 1;
        }

        // For padded_len == 1 (single leaf, possibly zero-leaf-batch),
        // the subtree root is the leaf itself; no internal rows were
        // emitted above. We still emit a mix-in-length row whose LEFT
        // is the single leaf.
        let subtree_root: [u8; CHUNK_BYTES] = if rows.is_empty() {
            padded_leaves[0]
        } else {
            rows.last().unwrap().parent
        };

        // mix_in_length row.
        let mut length_chunk = [0u8; CHUNK_BYTES];
        length_chunk[..8].copy_from_slice(&registry_length.to_le_bytes());
        let registry_root = crate::sha256::sha256_pair(&subtree_root, &length_chunk);
        rows.push(PairHashRow {
            left: subtree_root,
            right: length_chunk,
            parent: registry_root,
            leaf_index: 0,
            is_leaf: false,
            is_mix: true,
            is_root: false,
        });

        Self {
            validator_htrs: htrs.to_vec(),
            padded_leaves,
            rows,
            registry_length,
            registry_root,
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

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

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ValidatorRootConstructionWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (i, row) in witness.rows.iter().enumerate() {
        write_chunk(&mut columns, COL_LEFT_OFFSET, i, &row.left, curve);
        write_chunk(&mut columns, COL_RIGHT_OFFSET, i, &row.right, curve);
        write_chunk(&mut columns, COL_PARENT_OFFSET, i, &row.parent, curve);
        write_chunk(
            &mut columns,
            COL_CLAIMED_ROOT_OFFSET,
            i,
            &witness.registry_root,
            curve,
        );
        columns[COL_LEAF_INDEX][i] = Scalar::from_u64(row.leaf_index, curve);
        columns[COL_IS_REAL][i] = one.clone();
        if row.is_leaf {
            columns[COL_IS_LEAF][i] = one.clone();
        }
        if row.is_mix {
            columns[COL_IS_MIX][i] = one.clone();
            columns[COL_IS_FINAL][i] = one.clone();
        }
        if row.is_root {
            columns[COL_IS_ROOT][i] = one.clone();
        }
    }

    // Replicate CLAIMED_ROOT across padding rows too (constancy body
    // would otherwise fire at the boundary). Padding rows have IS_REAL
    // = 0, but shifted constraints multiply by IS_REAL(ω·X), so only
    // the next-row-is-real case matters; replicating CLAIMED_ROOT
    // everywhere keeps the trace coherent and avoids spurious diffs.
    for r in num_rows..padded {
        write_chunk(
            &mut columns,
            COL_CLAIMED_ROOT_OFFSET,
            r,
            &witness.registry_root,
            curve,
        );
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

pub struct ValidatorRootConstructionConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ValidatorRootConstructionConstraintSystem {
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

// ─── Body helpers ─────────────────────────────────────────────────────

/// β-RLC body for `IS_MIX · Σ β^i · (PARENT[i] − CLAIMED_ROOT[i])`.
fn eval_final_output_eq_claimed_root(
    col_evals: &[Scalar],
    alpha: &Scalar,
) -> Scalar {
    let curve = alpha.curve_type();
    let is_mix = &col_evals[COL_IS_MIX];
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let parent = &col_evals[COL_PARENT_OFFSET + i];
        let claimed = &col_evals[COL_CLAIMED_ROOT_OFFSET + i];
        let body = parent.sub(claimed);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    is_mix.mul(&acc)
}

fn build_final_output_eq_claimed_root_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let parent_p = &col_coeffs[COL_PARENT_OFFSET + i];
        let claimed_p = &col_coeffs[COL_CLAIMED_ROOT_OFFSET + i];
        let body = poly_sub(parent_p, claimed_p, curve);
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_MIX], &acc, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for ValidatorRootConstructionConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_leaf_binary".into(),
            "is_mix_binary".into(),
            "is_root_binary".into(),
            "is_final_eq_is_mix".into(),
            "final_output_eq_claimed_root".into(),
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
            let is_leaf = &row_evals[COL_IS_LEAF];
            let is_mix = &row_evals[COL_IS_MIX];
            let is_root = &row_evals[COL_IS_ROOT];
            let is_final = &row_evals[COL_IS_FINAL];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_real.mul(&is_leaf.mul(&is_leaf.sub(&one)));
            bodies[2][row] = is_real.mul(&is_mix.mul(&is_mix.sub(&one)));
            bodies[3][row] = is_real.mul(&is_root.mul(&is_root.sub(&one)));
            bodies[4][row] = is_real.mul(&is_final.sub(is_mix));
            bodies[5][row] = eval_final_output_eq_claimed_root(&row_evals, &alpha);
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
        let is_leaf = &col_evals[COL_IS_LEAF];
        let is_mix = &col_evals[COL_IS_MIX];
        let is_root = &col_evals[COL_IS_ROOT];
        let is_final = &col_evals[COL_IS_FINAL];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_real.mul(&is_leaf.mul(&is_leaf.sub(&one))),
            is_real.mul(&is_mix.mul(&is_mix.sub(&one))),
            is_real.mul(&is_root.mul(&is_root.sub(&one))),
            is_real.mul(&is_final.sub(is_mix)),
            eval_final_output_eq_claimed_root(col_evals, alpha),
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
        let is_leaf = &col_coeffs[COL_IS_LEAF];
        let is_mix = &col_coeffs[COL_IS_MIX];
        let is_root = &col_coeffs[COL_IS_ROOT];
        let is_final = &col_coeffs[COL_IS_FINAL];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_leaf_m1 = poly_sub(is_leaf, &one_poly, curve);
        let leaf_sq = poly_mul(is_leaf, &is_leaf_m1, curve);
        let is_leaf_binary = poly_mul(is_real, &leaf_sq, curve);

        let is_mix_m1 = poly_sub(is_mix, &one_poly, curve);
        let mix_sq = poly_mul(is_mix, &is_mix_m1, curve);
        let is_mix_binary = poly_mul(is_real, &mix_sq, curve);

        let is_root_m1 = poly_sub(is_root, &one_poly, curve);
        let root_sq = poly_mul(is_root, &is_root_m1, curve);
        let is_root_binary = poly_mul(is_real, &root_sq, curve);

        let final_minus_mix = poly_sub(is_final, is_mix, curve);
        let is_final_eq_is_mix = poly_mul(is_real, &final_minus_mix, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_leaf_binary,
            is_mix_binary,
            is_root_binary,
            is_final_eq_is_mix,
            build_final_output_eq_claimed_root_poly(col_coeffs, alpha, curve),
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
        // Zero everything except the CLAIMED_ROOT byte columns, which
        // we deliberately replicate across padding rows so that the
        // shifted constancy body has no spurious diffs at the trace
        // boundary.
        for (idx, col) in columns.iter_mut().enumerate().take(NUM_COLUMNS) {
            if (COL_CLAIMED_ROOT_OFFSET..COL_CLAIMED_ROOT_OFFSET + CHUNK_BYTES)
                .contains(&idx)
            {
                continue;
            }
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // CLAIMED_ROOT(ω·z)[0..32] + IS_REAL(ω·z).
        let mut cols = Vec::with_capacity(CHUNK_BYTES + 1);
        for b in 0..CHUNK_BYTES {
            cols.push(COL_CLAIMED_ROOT_OFFSET + b);
        }
        cols.push(COL_IS_REAL);
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
        let expected_shifted_len = CHUNK_BYTES + 1;
        if shifted_evals.len() < expected_shifted_len
            || col_evals_at_z.len() < NUM_COLUMNS
        {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let is_real_next = &shifted_evals[CHUNK_BYTES];

        let mut root_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_next = &shifted_evals[b];
            let r_curr = &col_evals_at_z[COL_CLAIMED_ROOT_OFFSET + b];
            let diff = r_next.sub(r_curr);
            root_acc = root_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = is_real_next.mul(&root_acc);

        let exclusion = z.sub(omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&body0).mul(&exclusion)
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

        let mut root_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_poly = &col_coeffs[COL_CLAIMED_ROOT_OFFSET + b];
            let r_shift = poly_shift(r_poly, omega);
            let diff = poly_sub(&r_shift, r_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            root_acc = poly_add(&root_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = poly_mul(&is_real_shift, &root_acc, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let excluded = poly_mul_linear(&body0, &omega_n_minus_1);
        poly_scalar_mul(&excluded, &ap)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Cross-AIR LogUp descriptor binding every active `(LEFT, RIGHT,
/// PARENT)` triple to a real SHA-256 invocation in
/// [`crate::sha256_extract`]. 96-byte tuple, gated by `IS_REAL`.
///
/// Combined with the bit-level SHA-256 binding from `sha256_extract`,
/// this algebraically pins `PARENT = sha256(LEFT || RIGHT)` for every
/// internal-tree row and for the `mix_in_length` row.
pub fn make_validator_root_to_sha256_descriptor(
    root_layer_index: usize,
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
        a_columns.push(COL_PARENT_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..se::NUM_INPUT_BYTES {
        b_columns.push(se::COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..se::NUM_OUTPUT_BYTES {
        b_columns.push(se::COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_root_to_sha256_v1".into(),
        a_layer_index: root_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding one leaf-row's `LEFT` chunk to
/// the validator at index `LEAF_INDEX` exposed by
/// [`crate::validator_htr_air`]. 33-column tuple
/// `(LEAF_INDEX, LEFT[0..32])`, gated on the A side by `IS_LEAF`.
///
/// The B side uses `validator_htr_air`'s `(VALIDATOR_INDEX,
/// VALIDATOR_ROOT_BYTE[0..32])` tuple gated by `IS_REAL`.
///
/// Note: this only binds the `LEFT` half of each leaf row. A
/// companion descriptor `make_validator_root_to_htr_right_descriptor`
/// binds `RIGHT` similarly with `LEAF_INDEX + 1` — but since
/// validator_htr_air does not expose a "leaf_index+1" column we
/// instead rely on the host-side ordering convention that leaf rows
/// appear in pairs of consecutive validators. A future tightening can
/// add a per-row `leaf_index_plus_1` aggregator column.
pub fn make_validator_root_to_htr_descriptor(
    root_layer_index: usize,
    htr_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::validator_htr_air as vh;
    let mut a_columns: Vec<usize> = Vec::with_capacity(33);
    a_columns.push(COL_LEAF_INDEX);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_LEFT_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(33);
    b_columns.push(vh::COL_VALIDATOR_INDEX);
    for b in 0..CHUNK_BYTES {
        b_columns.push(vh::COL_VALIDATOR_ROOT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_root_to_htr_v1".into(),
        a_layer_index: root_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_LEAF),
        b_layer_index: htr_layer_index,
        b_columns,
        b_selector_column: Some(vh::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding the computed registry root
/// (`CLAIMED_ROOT` on the mix row) to the `claimed_registry_root`
/// column (= `COL_REGISTRY_ROOT_OFFSET`) exposed by
/// [`crate::validator_registry_air`]. 32-column tuple gated by
/// `IS_MIX` on the A side and `IS_REAL` on the B side.
///
/// Caveat: validator_registry_air publishes the same `REGISTRY_ROOT`
/// on every active inclusion row; this descriptor is over-published
/// on the B side relative to the single A-side mix-row tuple. The
/// LogUp argument still pins set-equality up to multiplicity, so the
/// computed root must appear among the registry inclusions' claimed
/// roots — sufficient as the integrity binding for single-inclusion
/// uses. For multi-inclusion or strict-multiplicity bindings, a
/// future revision can add a dedicated `IS_ROOT_PUBLISH` selector
/// column to the registry AIR.
pub fn make_validator_root_to_registry_inclusion_descriptor(
    root_layer_index: usize,
    registry_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::validator_registry_air as vr;
    let mut a_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_CLAIMED_ROOT_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        b_columns.push(vr::COL_REGISTRY_ROOT_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_root_to_registry_inclusion_v1".into(),
        a_layer_index: root_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_MIX),
        b_layer_index: registry_layer_index,
        b_columns,
        b_selector_column: Some(vr::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leaf(seed: u8) -> [u8; CHUNK_BYTES] {
        let mut out = [0u8; CHUNK_BYTES];
        for i in 0..CHUNK_BYTES {
            out[i] = seed.wrapping_add(i as u8).wrapping_mul(13);
        }
        out
    }

    /// 1-validator batch: padded_len = 1; no internal rows; just the
    /// mix-in-length row. Subtree root = the single leaf; registry
    /// root = sha256_pair(leaf, length_chunk(1)).
    #[test]
    fn validator_root_construction_air_one_validator() {
        let leaf = make_leaf(0x11);
        let w = ValidatorRootConstructionWitness::from_validator_htrs(&[leaf]);
        assert_eq!(w.padded_leaves.len(), 1);
        assert_eq!(w.rows.len(), 1);
        let mix = &w.rows[0];
        assert!(mix.is_mix);
        assert!(!mix.is_leaf);
        assert!(!mix.is_root);
        assert_eq!(mix.left, leaf);
        // length chunk = LE bytes of 1.
        let mut expected_right = [0u8; CHUNK_BYTES];
        expected_right[0] = 1;
        assert_eq!(mix.right, expected_right);
        let expected_root = crate::sha256::sha256_pair(&leaf, &expected_right);
        assert_eq!(mix.parent, expected_root);
        assert_eq!(w.registry_root, expected_root);

        // Trace.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 1);
        // CLAIMED_ROOT[0] on row 0.
        assert_eq!(
            trace.columns[COL_CLAIMED_ROOT_OFFSET].evaluations[0].to_u64(),
            expected_root[0] as u64,
        );
        // IS_MIX = 1 on row 0.
        assert_eq!(trace.columns[COL_IS_MIX].evaluations[0].to_u64(), 1);
    }

    /// 4-validator batch: padded_len = 4; 3 internal rows (2 layer-0 +
    /// 1 layer-1 = root) + 1 mix row = 4 rows. IS_LEAF on first 2
    /// rows; IS_ROOT on the third row (the layer-1 pair producing the
    /// subtree root); IS_MIX on the fourth row.
    #[test]
    fn validator_root_construction_air_four_validators() {
        let leaves: Vec<[u8; CHUNK_BYTES]> =
            (0..4u8).map(make_leaf).collect();
        let w = ValidatorRootConstructionWitness::from_validator_htrs(&leaves);
        assert_eq!(w.padded_leaves.len(), 4);
        assert_eq!(w.rows.len(), 4); // 3 internal + 1 mix
        assert!(w.rows[0].is_leaf && w.rows[0].leaf_index == 0);
        assert!(w.rows[1].is_leaf && w.rows[1].leaf_index == 2);
        assert!(!w.rows[2].is_leaf && w.rows[2].is_root);
        assert!(w.rows[3].is_mix);

        // Subtree root = root row's parent.
        let l0 = crate::sha256::sha256_pair(&leaves[0], &leaves[1]);
        let l1 = crate::sha256::sha256_pair(&leaves[2], &leaves[3]);
        let subtree = crate::sha256::sha256_pair(&l0, &l1);
        assert_eq!(w.rows[2].parent, subtree);

        // Constraint vanishing.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorRootConstructionConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) at row {} must vanish (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    /// 8-validator batch: padded_len = 8 → 7 internal rows + 1 mix = 8
    /// rows; matches `ssz::merkleize_chunks` reference for the
    /// pre-mix subtree root.
    #[test]
    fn validator_root_construction_air_eight_validators_matches_ssz() {
        let leaves: Vec<[u8; CHUNK_BYTES]> =
            (0..8u8).map(make_leaf).collect();
        let w = ValidatorRootConstructionWitness::from_validator_htrs(&leaves);
        assert_eq!(w.padded_leaves.len(), 8);
        assert_eq!(w.rows.len(), 8); // 4 layer-0 + 2 layer-1 + 1 layer-2 + 1 mix

        // SSZ reference: merkleize_chunks(&leaves, Some(8)).
        let ssz_subtree = crate::ssz::merkleize_chunks(&leaves, Some(8));
        // The last internal row's parent equals the subtree root.
        let internal_root_row = &w.rows[w.rows.len() - 2];
        assert!(internal_root_row.is_root);
        assert_eq!(internal_root_row.parent, ssz_subtree);

        // mix_in_length(subtree, 8) == registry_root.
        let expected_root = crate::ssz::mix_in_length(ssz_subtree, 8);
        assert_eq!(w.registry_root, expected_root);

        // Constraints vanish.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorRootConstructionConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) at row {} must vanish",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    /// Non-power-of-2 batch (5 validators): padded_len = 8 (next pow
    /// of 2); 3 zero-padded leaves; subtree root matches
    /// `merkleize_chunks(&leaves, Some(8))`.
    #[test]
    fn validator_root_construction_air_non_pow2_pads_to_next_pow2() {
        let leaves: Vec<[u8; CHUNK_BYTES]> =
            (0..5u8).map(make_leaf).collect();
        let w = ValidatorRootConstructionWitness::from_validator_htrs(&leaves);
        assert_eq!(w.padded_leaves.len(), 8);
        // Padded slots are ZERO_CHUNK.
        for slot in 5..8 {
            assert_eq!(w.padded_leaves[slot], crate::ssz::ZERO_CHUNK);
        }
        // 4 layer-0 + 2 layer-1 + 1 layer-2 + 1 mix = 8 rows.
        assert_eq!(w.rows.len(), 8);

        // SSZ reference (limit = 8 = next pow of 2 of 5 means
        // merkleize_chunks pads with ZERO_CHUNK to 8 leaves).
        let ssz_subtree = crate::ssz::merkleize_chunks(&leaves, Some(8));
        assert_eq!(w.rows[w.rows.len() - 2].parent, ssz_subtree);
        // Mix-in-length uses the un-padded length (5).
        let expected_root = crate::ssz::mix_in_length(ssz_subtree, 5);
        assert_eq!(w.registry_root, expected_root);

        // Constraints vanish.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorRootConstructionConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) at row {} must vanish",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    /// Cross-AIR LogUp descriptors are well-formed: correct tuple
    /// widths, labels, selectors.
    #[test]
    fn validator_root_construction_air_descriptors_well_formed() {
        // sha256 pair-hash descriptor.
        let sha = make_validator_root_to_sha256_descriptor(0, 1);
        assert_eq!(sha.label, "validator_root_to_sha256_v1");
        assert_eq!(sha.a_columns.len(), 96);
        assert_eq!(sha.b_columns.len(), 96);
        for b in 0..CHUNK_BYTES {
            assert_eq!(sha.a_columns[b], COL_LEFT_OFFSET + b);
            assert_eq!(sha.a_columns[CHUNK_BYTES + b], COL_RIGHT_OFFSET + b);
            assert_eq!(sha.a_columns[2 * CHUNK_BYTES + b], COL_PARENT_OFFSET + b);
        }
        assert_eq!(sha.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            sha.b_selector_column,
            Some(crate::sha256_extract::COL_IS_REAL)
        );
        assert_eq!(sha.a_layer_index, 0);
        assert_eq!(sha.b_layer_index, 1);

        // leaf↔validator_htr descriptor.
        let htr = make_validator_root_to_htr_descriptor(0, 2);
        assert_eq!(htr.label, "validator_root_to_htr_v1");
        assert_eq!(htr.a_columns.len(), 33);
        assert_eq!(htr.b_columns.len(), 33);
        assert_eq!(htr.a_columns[0], COL_LEAF_INDEX);
        for b in 0..CHUNK_BYTES {
            assert_eq!(htr.a_columns[1 + b], COL_LEFT_OFFSET + b);
        }
        assert_eq!(htr.a_selector_column, Some(COL_IS_LEAF));
        assert_eq!(
            htr.b_columns[0],
            crate::validator_htr_air::COL_VALIDATOR_INDEX,
        );
        for b in 0..CHUNK_BYTES {
            assert_eq!(
                htr.b_columns[1 + b],
                crate::validator_htr_air::COL_VALIDATOR_ROOT_BYTE_OFFSET + b,
            );
        }
        assert_eq!(
            htr.b_selector_column,
            Some(crate::validator_htr_air::COL_IS_REAL),
        );

        // registry inclusion descriptor.
        let reg = make_validator_root_to_registry_inclusion_descriptor(0, 3);
        assert_eq!(reg.label, "validator_root_to_registry_inclusion_v1");
        assert_eq!(reg.a_columns.len(), CHUNK_BYTES);
        assert_eq!(reg.b_columns.len(), CHUNK_BYTES);
        for b in 0..CHUNK_BYTES {
            assert_eq!(reg.a_columns[b], COL_CLAIMED_ROOT_OFFSET + b);
            assert_eq!(
                reg.b_columns[b],
                crate::validator_registry_air::COL_REGISTRY_ROOT_OFFSET + b,
            );
        }
        assert_eq!(reg.a_selector_column, Some(COL_IS_MIX));
        assert_eq!(
            reg.b_selector_column,
            Some(crate::validator_registry_air::COL_IS_REAL),
        );
    }

    /// Tampering: corrupt CLAIMED_ROOT on the mix row; the
    /// `final_output_eq_claimed_root` body must fire.
    #[test]
    fn validator_root_construction_air_tampered_claimed_root_rejected() {
        let leaves: Vec<[u8; CHUNK_BYTES]> =
            (0..4u8).map(make_leaf).collect();
        let w = ValidatorRootConstructionWitness::from_validator_htrs(&leaves);
        let curve = CurveType::Bls48581;
        let mut trace = build_trace_polynomials(&w, curve);
        // Find the mix row (last real row).
        let mix_row = w.rows.len() - 1;
        // Tamper CLAIMED_ROOT[5] on the mix row.
        trace.columns[COL_CLAIMED_ROOT_OFFSET + 5].evaluations[mix_row] =
            Scalar::from_u64(0xEE, curve);
        let cs = ValidatorRootConstructionConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // Body 5 is final_output_eq_claimed_root.
        assert!(
            !evals[5][mix_row].is_zero(),
            "tampered CLAIMED_ROOT[5] on mix row must fire final_output_eq_claimed_root"
        );
    }
}
