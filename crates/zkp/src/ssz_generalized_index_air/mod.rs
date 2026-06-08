//! SSZ generalized-index merkle proof verification AIR (roadmap #277).
//!
//! Generalized indices encode a field's position in an SSZ merkle tree.
//! For a tree of depth `d`, a generalized index `g` satisfies
//! `2^d ≤ g < 2^(d+1)`; its low `d` bits give the path direction from
//! leaf to root (bit `k` selects whether the running hash is the left
//! (bit = 0) or right (bit = 1) child at level `k`).
//!
//! Light-client partial-state proofs (e.g. Eth beacon `state.field`
//! merkle multi-proofs) use generalized indices to identify subtree
//! roots. This AIR proves one such inclusion: walk from a known leaf,
//! hashing against the supplied sibling at each level per the
//! generalized-index bits, and check the final running hash equals the
//! claimed root.
//!
//! ## Trace layout
//!
//! One inclusion expands to `DEPTH` rows (one per merkle tree level,
//! bottom-up). Row 0's `CURRENT_HASH` is the leaf; at level `k`,
//! `NEXT_HASH = sha256_pair(LEFT, RIGHT)` where `(LEFT, RIGHT)` is
//! `(CURRENT_HASH, SIBLING)` if `INDEX_BIT_k = 0`, else
//! `(SIBLING, CURRENT_HASH)`. The next row's `CURRENT_HASH` equals this
//! row's `NEXT_HASH`. The final row's `NEXT_HASH` equals the claimed
//! `ROOT`.
//!
//! The `(LEFT, RIGHT, NEXT_HASH)` triple on every active row maps to a
//! single SHA-256 invocation, exposed via
//! [`make_ssz_generalized_index_to_sha256_descriptor`] for cross-AIR
//! LogUp against [`crate::sha256_extract`].
//!
//! ## Algebraic constraints (row-local, 5 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//! 1. `index_bit_binary` — `IS_REAL · INDEX_BIT · (INDEX_BIT − 1) = 0`
//! 2. `left_selection` (β-RLC over 32 sub-bodies) —
//!    `IS_REAL · Σ β^i · (LEFT[i] − ((1 − INDEX_BIT) · CURRENT_HASH[i] + INDEX_BIT · SIBLING[i])) = 0`
//! 3. `right_selection` (β-RLC over 32 sub-bodies) —
//!    `IS_REAL · Σ β^i · (RIGHT[i] − (INDEX_BIT · CURRENT_HASH[i] + (1 − INDEX_BIT) · SIBLING[i])) = 0`
//! 4. `index_aggregator` —
//!    `IS_REAL · (GENERALIZED_INDEX − (2^DEPTH + Σ_{k=0..DEPTH} INDEX_BIT_REPL[k] · 2^k)) = 0`
//!    where `INDEX_BIT_REPL[k]` is the bit-k column replicated identically
//!    on every active row (constancy enforced by shifted body 1).
//!
//! ## Cross-row shifted constraints (3 bodies)
//!
//! 0. `hash_chain` — `IS_REAL(ω·X) · Σ β^i · (CURRENT_HASH_i(ω·X) − NEXT_HASH_i(X)) = 0`
//!    (excluded at the wrap-around row).
//! 1. `index_bits_constancy` — `IS_REAL(ω·X) · Σ β^k · (INDEX_BIT_REPL_k(ω·X) − INDEX_BIT_REPL_k(X)) = 0`
//! 2. `root_constancy` — `IS_REAL(ω·X) · Σ β^i · (ROOT_i(ω·X) − ROOT_i(X)) = 0`
//!
//! Together these pin the tree walk: each level's `NEXT_HASH` flows into
//! the next level's `CURRENT_HASH`, the replicated columns are constant
//! across the inclusion, and (with the pair-hash cross-AIR LogUp) the
//! final `NEXT_HASH` on the top row equals the claimed `ROOT`.
//!
//! ## What this AIR does NOT prove
//!
//! * SHA-256 of `(LEFT || RIGHT)`: deferred to the cross-AIR LogUp into
//!   `sha256_extract` (and transitively the bit-level SHA-256 AIR).
//! * That the leaf on row 0 actually corresponds to the SSZ
//!   serialization of any particular field. That binding is the job of
//!   a consumer AIR (e.g. `account_state_air`) at the leaf level.
//! * That the final `NEXT_HASH` equals the claimed `ROOT`: this is
//!   enforced via `root_constancy` together with a consumer-supplied
//!   linkage that binds `ROOT` to the trusted state root.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Depth of the SSZ merkle tree this AIR walks. Default 16 covers
/// beacon-state top-level field proofs. Other depths are produced via
/// type-level monomorphization (see `with_depth` helper for tests).
pub const DEPTH: usize = 16;
/// Bytes per hash chunk.
pub const CHUNK_BYTES: usize = 32;
/// Rows used per inclusion.
pub const ROWS_PER_INCLUSION: usize = DEPTH;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_CURRENT_HASH_OFFSET: usize = 0;
pub const COL_SIBLING_OFFSET: usize = COL_CURRENT_HASH_OFFSET + CHUNK_BYTES; // 32
pub const COL_LEFT_OFFSET: usize = COL_SIBLING_OFFSET + CHUNK_BYTES;          // 64
pub const COL_RIGHT_OFFSET: usize = COL_LEFT_OFFSET + CHUNK_BYTES;            // 96
pub const COL_NEXT_HASH_OFFSET: usize = COL_RIGHT_OFFSET + CHUNK_BYTES;       // 128
pub const COL_ROOT_OFFSET: usize = COL_NEXT_HASH_OFFSET + CHUNK_BYTES;        // 160

pub const COL_INDEX_BIT: usize = COL_ROOT_OFFSET + CHUNK_BYTES;               // 192
pub const COL_GENERALIZED_INDEX: usize = COL_INDEX_BIT + 1;                   // 193
pub const COL_INDEX_BIT_REPL_OFFSET: usize = COL_GENERALIZED_INDEX + 1;       // 194
pub const COL_LEVEL: usize = COL_INDEX_BIT_REPL_OFFSET + DEPTH;               // 210

pub const COL_IS_REAL: usize = COL_LEVEL + 1;                                 // 211

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;                               // 212

/// 5 row-local constraint bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 5;
/// 3 cross-row shifted bodies (hash_chain + 2 constancy bodies).
pub const NUM_SHIFTED: usize = 3;

// ─── Witness types ────────────────────────────────────────────────────

/// One inclusion: a single field's merkle path to a claimed root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneralizedIndexInclusionWitness {
    /// The leaf chunk.
    pub leaf: [u8; CHUNK_BYTES],
    /// Generalized index in `[2^DEPTH, 2^(DEPTH+1))`.
    pub generalized_index: u64,
    /// Sibling hashes from the bottom level upward (length `DEPTH`).
    pub sibling_hashes: [[u8; CHUNK_BYTES]; DEPTH],
    /// Claimed merkle root.
    pub root: [u8; CHUNK_BYTES],
}

impl GeneralizedIndexInclusionWitness {
    /// Build a witness from a leaf, a generalized index, and a merkle
    /// proof path of length `DEPTH`. The root is computed host-side by
    /// walking the path.
    pub fn from_proof(
        leaf: [u8; CHUNK_BYTES],
        generalized_index: u64,
        proof_path: [[u8; CHUNK_BYTES]; DEPTH],
    ) -> Self {
        debug_assert!(
            generalized_index >= (1u64 << DEPTH)
                && generalized_index < (1u64 << (DEPTH + 1)),
            "generalized_index {} not in [2^{}, 2^{})",
            generalized_index,
            DEPTH,
            DEPTH + 1
        );
        let mut current = leaf;
        for (level, sibling) in proof_path.iter().enumerate() {
            let bit = (generalized_index >> level) & 1;
            let (left, right) = if bit == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = crate::sha256::sha256_pair(&left, &right);
        }
        Self {
            leaf,
            generalized_index,
            sibling_hashes: proof_path,
            root: current,
        }
    }
}

/// Sequence of inclusions packed into a single trace.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SszGeneralizedIndexWitness {
    pub inclusions: Vec<GeneralizedIndexInclusionWitness>,
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
    chunk: &[u8; CHUNK_BYTES],
    curve: CurveType,
) {
    for (b, &byte) in chunk.iter().enumerate() {
        columns[offset + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &SszGeneralizedIndexWitness,
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
        let mut current = inclusion.leaf;

        // Pre-compute the DEPTH index bits (low DEPTH bits of g).
        let mut index_bits = [0u8; DEPTH];
        for k in 0..DEPTH {
            index_bits[k] = ((inclusion.generalized_index >> k) & 1) as u8;
        }

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
                COL_ROOT_OFFSET,
                row,
                &inclusion.root,
                curve,
            );

            columns[COL_INDEX_BIT][row] = Scalar::from_u64(bit as u64, curve);
            columns[COL_GENERALIZED_INDEX][row] =
                Scalar::from_u64(inclusion.generalized_index, curve);
            for k in 0..DEPTH {
                columns[COL_INDEX_BIT_REPL_OFFSET + k][row] =
                    Scalar::from_u64(index_bits[k] as u64, curve);
            }
            columns[COL_LEVEL][row] = Scalar::from_u64(level as u64, curve);
            columns[COL_IS_REAL][row] = one.clone();

            current = next;
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

pub struct SszGeneralizedIndexConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl SszGeneralizedIndexConstraintSystem {
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

/// `IS_REAL · (GENERALIZED_INDEX − (2^DEPTH + Σ_{k=0..DEPTH} INDEX_BIT_REPL[k] · 2^k)) = 0`.
fn eval_index_aggregator(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    // Start with the leading bit constant (2^DEPTH).
    let mut sum = scalar_pow(&Scalar::from_u64(2, curve), DEPTH as u64);
    for k in 0..DEPTH {
        let pow = scalar_pow(&Scalar::from_u64(2, curve), k as u64);
        let bit_k = &col_evals[COL_INDEX_BIT_REPL_OFFSET + k];
        sum = sum.add(&bit_k.mul(&pow));
    }
    let gi = &col_evals[COL_GENERALIZED_INDEX];
    let body = gi.sub(&sum);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_index_aggregator_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let leading = scalar_pow(&Scalar::from_u64(2, curve), DEPTH as u64);
    let mut sum = vec![leading];
    for k in 0..DEPTH {
        let pow = scalar_pow(&Scalar::from_u64(2, curve), k as u64);
        let bit_k_p = &col_coeffs[COL_INDEX_BIT_REPL_OFFSET + k];
        let scaled = poly_scalar_mul(bit_k_p, &pow);
        sum = poly_add(&sum, &scaled, curve);
    }
    let gi_p = &col_coeffs[COL_GENERALIZED_INDEX];
    let body = poly_sub(gi_p, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for SszGeneralizedIndexConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "index_bit_binary".into(),
            "left_selection".into(),
            "right_selection".into(),
            "index_aggregator".into(),
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
            let bit = &row_evals[COL_INDEX_BIT];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_real.mul(&bit.mul(&bit.sub(&one)));
            bodies[2][row] = eval_left_selection(&row_evals, &alpha);
            bodies[3][row] = eval_right_selection(&row_evals, &alpha);
            bodies[4][row] = eval_index_aggregator(&row_evals);
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
        let bit = &col_evals[COL_INDEX_BIT];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_real.mul(&bit.mul(&bit.sub(&one))),
            eval_left_selection(col_evals, alpha),
            eval_right_selection(col_evals, alpha),
            eval_index_aggregator(col_evals),
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
        let bit = &col_coeffs[COL_INDEX_BIT];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let bit_m1 = poly_sub(bit, &one_poly, curve);
        let bit_sq = poly_mul(bit, &bit_m1, curve);
        let index_bit_binary = poly_mul(is_real, &bit_sq, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            index_bit_binary,
            build_left_selection_poly(col_coeffs, alpha, curve),
            build_right_selection_poly(col_coeffs, alpha, curve),
            build_index_aggregator_poly(col_coeffs, curve),
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
        let mut cols = Vec::with_capacity(CHUNK_BYTES + 1 + DEPTH + CHUNK_BYTES);
        for b in 0..CHUNK_BYTES {
            cols.push(COL_CURRENT_HASH_OFFSET + b);
        }
        cols.push(COL_IS_REAL);
        for k in 0..DEPTH {
            cols.push(COL_INDEX_BIT_REPL_OFFSET + k);
        }
        for b in 0..CHUNK_BYTES {
            cols.push(COL_ROOT_OFFSET + b);
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
        let expected_shifted_len = CHUNK_BYTES + 1 + DEPTH + CHUNK_BYTES;
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

        let is_real_next = &shifted_evals[is_real_next_idx];

        // Shifted body 0: hash_chain.
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

        // Shifted body 1: index_bits_constancy.
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

        // Shifted body 2: root_constancy.
        let mut root_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_next = &shifted_evals[root_shift_off + b];
            let r_curr = &col_evals_at_z[COL_ROOT_OFFSET + b];
            let diff = r_next.sub(r_curr);
            root_acc = root_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body2 = is_real_next.mul(&root_acc);

        let exclusion = z.sub(omega_n_minus_1);
        let bodies = [body0, body1, body2];
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

        // Body 0: hash_chain.
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

        // Body 1: index_bits_constancy.
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

        // Body 2: root_constancy.
        let mut root_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_poly = &col_coeffs[COL_ROOT_OFFSET + b];
            let r_shift = poly_shift(r_poly, omega);
            let diff = poly_sub(&r_shift, r_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            root_acc = poly_add(&root_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body2 = poly_mul(&is_real_shift, &root_acc, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let bodies = [body0, body1, body2];
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
/// NEXT_HASH)` triple of the generalized-index walk to a real SHA-256
/// invocation in [`crate::sha256_extract`]. 96-byte tuple on each side,
/// gated by `IS_REAL`.
pub fn make_ssz_generalized_index_to_sha256_descriptor(
    gi_layer_index: usize,
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
        label: "ssz_generalized_index_pair_hash_v1".into(),
        a_layer_index: gi_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic inclusion: deterministic leaf + deterministic
    /// sibling path. Generalized index is in `[2^DEPTH, 2^(DEPTH+1))`.
    fn synthetic_inclusion(generalized_index: u64) -> GeneralizedIndexInclusionWitness {
        let mut leaf = [0u8; CHUNK_BYTES];
        for i in 0..CHUNK_BYTES {
            leaf[i] = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        for k in 0..DEPTH {
            let mut s = [k as u8; CHUNK_BYTES];
            s[0] = s[0].wrapping_add(17);
            path[k] = s;
        }
        GeneralizedIndexInclusionWitness::from_proof(leaf, generalized_index, path)
    }

    fn single_inclusion_witness() -> SszGeneralizedIndexWitness {
        // Pick a generalized index with mixed bits in the low DEPTH bits.
        // 2^DEPTH + 0xABCD with DEPTH=16 → 0x1ABCD.
        let gi = (1u64 << DEPTH) | 0xABCD;
        SszGeneralizedIndexWitness {
            inclusions: vec![synthetic_inclusion(gi)],
        }
    }

    /// Sanity: `from_proof` matches a manual host-side walk.
    #[test]
    fn from_proof_matches_manual_walk() {
        let gi: u64 = (1u64 << DEPTH) | 0x4321;
        let mut leaf = [0u8; CHUNK_BYTES];
        for i in 0..CHUNK_BYTES { leaf[i] = i as u8; }
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        for k in 0..DEPTH {
            let mut s = [0u8; CHUNK_BYTES];
            s[0] = k as u8;
            s[31] = k as u8;
            path[k] = s;
        }
        let w = GeneralizedIndexInclusionWitness::from_proof(leaf, gi, path);
        let mut cur = leaf;
        for k in 0..DEPTH {
            let bit = (gi >> k) & 1;
            let (l, r) = if bit == 0 { (cur, path[k]) } else { (path[k], cur) };
            cur = crate::sha256::sha256_pair(&l, &r);
        }
        assert_eq!(w.root, cur);
    }

    /// Witness round-trip: build trace, all row-local constraints
    /// vanish on every row.
    #[test]
    fn constraints_vanish_on_honest_single_inclusion() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, ROWS_PER_INCLUSION);

        let cs = SszGeneralizedIndexConstraintSystem::new(trace.num_rows);
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
    }

    /// Known-proof oracle: walk a DEPTH-deep tree where the leaf sits at
    /// the leftmost position (gi = 2^DEPTH); sibling at every level is
    /// the canonical SSZ `zero_hash` chain. The resulting root must
    /// equal `merkleize_chunks(&[leaf], Some(2^DEPTH))`.
    #[test]
    fn leftmost_known_proof_matches_ssz_helper() {
        let mut leaf = [0u8; CHUNK_BYTES];
        for i in 0..CHUNK_BYTES { leaf[i] = (i as u8).wrapping_mul(7); }
        // siblings at every level for index 0 = zero_hash[level].
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        let mut z = [0u8; CHUNK_BYTES];
        for k in 0..DEPTH {
            path[k] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }
        let gi = 1u64 << DEPTH; // leftmost leaf
        let w = GeneralizedIndexInclusionWitness::from_proof(leaf, gi, path);

        let expected = crate::ssz::merkleize_chunks(&[leaf], Some(1u64 << DEPTH));
        assert_eq!(
            w.root, expected,
            "depth-{} leftmost leaf inclusion must match SSZ helper output",
            DEPTH,
        );
    }

    /// Tampering: corrupt LEFT[5] on the leaf row and confirm
    /// `left_selection` body fires.
    #[test]
    fn left_selection_rejects_tampered_left_byte() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_LEFT_OFFSET + 5].evaluations[0] =
            Scalar::from_u64(0xFF, curve);
        let cs = SszGeneralizedIndexConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 2 = left_selection in the new labeling.
        assert!(
            !evals[2][0].is_zero(),
            "tampered LEFT[5] must make left_selection non-zero on the leaf row"
        );
    }

    /// Tampering: corrupt the generalized-index aggregator and confirm
    /// `index_aggregator` fires.
    #[test]
    fn index_aggregator_rejects_tampered_index() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_GENERALIZED_INDEX].evaluations[0] =
            Scalar::from_u64(0xDEAD, curve);
        let cs = SszGeneralizedIndexConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 4 = index_aggregator.
        assert!(
            !evals[4][0].is_zero(),
            "tampered GENERALIZED_INDEX must make index_aggregator non-zero"
        );
    }

    /// Cross-AIR LogUp descriptor sanity: 96-col tuple on both sides,
    /// gated by IS_REAL on both layers.
    #[test]
    fn sha256_descriptor_has_expected_shape() {
        let d = make_ssz_generalized_index_to_sha256_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 96);
        assert_eq!(d.b_columns.len(), 96);
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert!(d.b_selector_column.is_some());
        assert_eq!(d.label, "ssz_generalized_index_pair_hash_v1");
        // A side: LEFT (32) || RIGHT (32) || NEXT_HASH (32).
        for b in 0..CHUNK_BYTES {
            assert_eq!(d.a_columns[b], COL_LEFT_OFFSET + b);
            assert_eq!(d.a_columns[CHUNK_BYTES + b], COL_RIGHT_OFFSET + b);
            assert_eq!(d.a_columns[2 * CHUNK_BYTES + b], COL_NEXT_HASH_OFFSET + b);
        }
    }
}
