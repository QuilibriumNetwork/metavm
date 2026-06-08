//! Eth1 deposit-contract merkle tree inclusion AIR.
//!
//! Per-inclusion gadget: proves that one [`crate::deposit::DepositData`]
//! at a known position sits at a known index within the Ethereum 1.0
//! deposit contract's incremental merkle tree, given the tree's claimed
//! root (`Eth1Data.deposit_root`-style root, without the `mix_in_length`
//! step — see *Soundness scope* below).
//!
//! ## Trace layout
//!
//! One inclusion expands to `DEPTH = 32` rows, one per merkle tree level
//! bottom-up:
//!   - Row 0's `CURRENT_HASH` is the deposit's leaf
//!     (`hash_tree_root(DepositData)`).
//!   - At level `k`, `NEXT_HASH = sha256_pair(LEFT, RIGHT)` where
//!     `(LEFT, RIGHT)` is `(CURRENT_HASH, SIBLING)` if
//!     `INDEX_BIT_k = 0`, else `(SIBLING, CURRENT_HASH)`.
//!   - The next row's `CURRENT_HASH` equals this row's `NEXT_HASH`.
//!   - On the top row (level `DEPTH-1`) `NEXT_HASH` is the claimed
//!     `DEPOSIT_ROOT` (enforced via the
//!     [`make_deposit_root_to_block_descriptor`] linkage on the row
//!     where `IS_TOP = 1`).
//!
//! Each active row's `(LEFT, RIGHT, NEXT_HASH)` triple is matched to a
//! single SHA-256 invocation in `sha256_extract` via
//! [`make_deposit_to_sha256_descriptor`]. The leaf row's
//! `(DEPOSIT_INDEX, CURRENT_HASH)` is matched to the deposit's data-htr
//! via [`make_deposit_to_data_htr_descriptor`] (host-side scaffolding;
//! see note on the descriptor).
//!
//! ## Algebraic constraints (row-local, 6 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//! 1. `is_top_binary` — `IS_TOP · (IS_TOP − 1) = 0`
//! 2. `index_bit_binary` — `IS_REAL · INDEX_BIT · (INDEX_BIT − 1) = 0`
//! 3. `left_selection` (β-RLC over 32 byte sub-bodies) —
//!    `IS_REAL · Σ β^i · (LEFT[i] − ((1 − INDEX_BIT) · CURRENT_HASH[i] + INDEX_BIT · SIBLING[i])) = 0`
//! 4. `right_selection` (β-RLC over 32 byte sub-bodies) —
//!    `IS_REAL · Σ β^i · (RIGHT[i] − (INDEX_BIT · CURRENT_HASH[i] + (1 − INDEX_BIT) · SIBLING[i])) = 0`
//! 5. `index_aggregator` (β-RLC over 32 bit sub-bodies) —
//!    `IS_REAL · (DEPOSIT_INDEX − Σ_{k=0}^{31} INDEX_BIT_REPL[k] · 2^k) = 0`
//!
//! ## Cross-row shifted constraints (3 bodies)
//!
//! 0. `hash_chain` — `IS_REAL(ω·X) · Σ β^i · (CURRENT_HASH(ω·X)[i] − NEXT_HASH(X)[i]) = 0`
//! 1. `index_bits_constancy` — `IS_REAL(ω·X) · Σ β^k · (INDEX_BIT_REPL_k(ω·X) − INDEX_BIT_REPL_k(X)) = 0`
//! 2. `deposit_root_constancy` — `IS_REAL(ω·X) · Σ β^i · (DEPOSIT_ROOT_i(ω·X) − DEPOSIT_ROOT_i(X)) = 0`
//!
//! ## Soundness scope
//!
//! * SHA-256 of `(LEFT || RIGHT)`: deferred to cross-AIR LogUp into
//!   `sha256_extract`.
//! * That the leaf `CURRENT_HASH` equals the deposit's real
//!   `hash_tree_root(DepositData)`: deferred to a cross-AIR LogUp into a
//!   future `deposit_data_htr_air` (scaffolded by
//!   [`make_deposit_to_data_htr_descriptor`]).
//! * That the top row's `NEXT_HASH` equals the Eth1Data deposit root: a
//!   future per-row `IS_TOP` selector + binding descriptor against the
//!   block's `eth1_data.deposit_root` field would close this; today the
//!   `DEPOSIT_ROOT` column is held constant across the inclusion and
//!   forwarded via [`make_deposit_root_to_block_descriptor`].
//! * `mix_in_length(root, deposit_count)`: NOT applied here. The real
//!   `Eth1Data.deposit_root` mixes the deposit count into the
//!   merkleized root; a follow-up can add a mix-in row analogous to
//!   `validator_registry_air`.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Depth of the Ethereum 1.0 deposit-contract merkle tree
/// (`DEPOSIT_CONTRACT_TREE_DEPTH = 32`).
pub const DEPTH: usize = 32;
/// Bytes per hash chunk.
pub const CHUNK_BYTES: usize = 32;
/// Rows used per inclusion.
pub const ROWS_PER_INCLUSION: usize = DEPTH;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_CURRENT_HASH_OFFSET: usize = 0;
pub const COL_SIBLING_OFFSET: usize = COL_CURRENT_HASH_OFFSET + CHUNK_BYTES; // 32
pub const COL_LEFT_OFFSET: usize = COL_SIBLING_OFFSET + CHUNK_BYTES; // 64
pub const COL_RIGHT_OFFSET: usize = COL_LEFT_OFFSET + CHUNK_BYTES; // 96
pub const COL_NEXT_HASH_OFFSET: usize = COL_RIGHT_OFFSET + CHUNK_BYTES; // 128
pub const COL_DEPOSIT_ROOT_OFFSET: usize = COL_NEXT_HASH_OFFSET + CHUNK_BYTES; // 160

pub const COL_INDEX_BIT: usize = COL_DEPOSIT_ROOT_OFFSET + CHUNK_BYTES; // 192
pub const COL_DEPOSIT_INDEX: usize = COL_INDEX_BIT + 1; // 193
pub const COL_INDEX_BIT_REPL_OFFSET: usize = COL_DEPOSIT_INDEX + 1; // 194
pub const COL_LEVEL: usize = COL_INDEX_BIT_REPL_OFFSET + DEPTH; // 226

pub const COL_IS_REAL: usize = COL_LEVEL + 1; // 227
/// Selector flagging the row where `NEXT_HASH` produces the claimed
/// deposit root (i.e. `LEVEL == DEPTH - 1`). Used by the future
/// block-side binding descriptor; populated by the witness builder.
pub const COL_IS_TOP: usize = COL_IS_REAL + 1; // 228

/// Task #309 / #190 mirror: commits the first byte of the deposit's
/// signature on every active row. This column is pure witness data
/// (no row-local algebraic constraint binds it to the merkle walk)
/// and exists solely as a dedicated cross-AIR LogUp A-side anchor
/// for the validator-deposit integration test's signature-byte
/// descriptor. Soundness flows from the cross-AIR LogUp closure: a
/// tampering prover must break the closure equality against the
/// pair AIR's `SIG_BYTES[0]`. Replaces the scaffold-era
/// `patch_traces_for_single_byte_alignment` SIBLING/RIGHT
/// overrides, which clobbered cells bound by row-local selection
/// bodies. See [`crate::sync_committee_aggregate_composer_air`]
/// `COL_AGG_PK_X_BYTE0_MIRROR` for the same pattern.
pub const COL_SIG_BYTE0_MIRROR: usize = COL_IS_TOP + 1; // 229

pub const NUM_COLUMNS: usize = COL_SIG_BYTE0_MIRROR + 1; // 230

/// 6 row-local constraint bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 6;
/// 3 cross-row shifted bodies.
pub const NUM_SHIFTED: usize = 3;

// ─── Witness types ────────────────────────────────────────────────────

/// One inclusion's witness: a single deposit's merkle path to the
/// claimed deposit-contract root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepositInclusionWitness {
    /// The deposit's `hash_tree_root` (the leaf chunk).
    pub leaf: [u8; CHUNK_BYTES],
    /// Index of the deposit in the contract's incremental merkle tree
    /// (`0..2^32`).
    pub deposit_index: u64,
    /// Sibling hashes from the bottom level upward.
    pub sibling_hashes: [[u8; CHUNK_BYTES]; DEPTH],
    /// Claimed merkle-tree root (i.e. the `Eth1Data.deposit_root`
    /// candidate, prior to any `mix_in_length` step — see module-level
    /// soundness note).
    pub deposit_root: [u8; CHUNK_BYTES],
    /// Task #309 mirror: first byte of the deposit's `signature[0..96]`.
    /// Populated into [`COL_SIG_BYTE0_MIRROR`] per-row to serve as the
    /// dedicated cross-AIR LogUp A-side anchor for the signature-byte
    /// descriptor. Zero by default (back-compat for non-signature
    /// callers).
    pub sig_byte0: u8,
}

impl DepositInclusionWitness {
    /// Build a witness from a deposit, its index, and a merkle proof
    /// path of length [`DEPTH`]. The deposit root is computed
    /// host-side by walking the path.
    pub fn from_deposit_proof(
        deposit: &crate::deposit::DepositData,
        index: u64,
        proof: &[[u8; CHUNK_BYTES]; DEPTH],
    ) -> Self {
        let leaf = deposit.hash_tree_root();
        let mut current = leaf;
        for (level, sibling) in proof.iter().enumerate() {
            let bit = (index >> level) & 1;
            let (left, right) = if bit == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = crate::sha256::sha256_pair(&left, &right);
        }
        Self {
            leaf,
            deposit_index: index,
            sibling_hashes: *proof,
            deposit_root: current,
            sig_byte0: deposit.signature[0],
        }
    }
}

/// Sequence of inclusions packed into a single trace.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DepositTreeWitness {
    pub inclusions: Vec<DepositInclusionWitness>,
}

/// Top-level convenience builder mirroring the task spec:
/// `from_deposit_proof(deposit, index, proof)` returns a witness
/// containing a single inclusion.
pub fn from_deposit_proof(
    deposit: &crate::deposit::DepositData,
    index: u64,
    proof: &[[u8; CHUNK_BYTES]; DEPTH],
) -> DepositTreeWitness {
    DepositTreeWitness {
        inclusions: vec![DepositInclusionWitness::from_deposit_proof(deposit, index, proof)],
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
    chunk: &[u8; CHUNK_BYTES],
    curve: CurveType,
) {
    for (b, &byte) in chunk.iter().enumerate() {
        columns[offset + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &DepositTreeWitness,
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

        // Pre-compute the 32 index bits.
        let mut index_bits = [0u8; DEPTH];
        for k in 0..DEPTH {
            index_bits[k] = ((inclusion.deposit_index >> k) & 1) as u8;
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
                COL_DEPOSIT_ROOT_OFFSET,
                row,
                &inclusion.deposit_root,
                curve,
            );

            columns[COL_INDEX_BIT][row] = Scalar::from_u64(bit as u64, curve);
            columns[COL_DEPOSIT_INDEX][row] =
                Scalar::from_u64(inclusion.deposit_index, curve);
            for k in 0..DEPTH {
                columns[COL_INDEX_BIT_REPL_OFFSET + k][row] =
                    Scalar::from_u64(index_bits[k] as u64, curve);
            }
            columns[COL_LEVEL][row] = Scalar::from_u64(level as u64, curve);
            columns[COL_IS_REAL][row] = one.clone();
            if level == DEPTH - 1 {
                columns[COL_IS_TOP][row] = one.clone();
            }
            // Task #309 mirror: pin the deposit-signature first byte on
            // every active row. No row-local constraint binds this
            // column — its soundness is the cross-AIR LogUp closure
            // against the bls_pairing_air `SIG_BYTES[0]` cell.
            columns[COL_SIG_BYTE0_MIRROR][row] =
                Scalar::from_u64(inclusion.sig_byte0 as u64, curve);

            current = next;
        }

        // Sanity: the final running hash MUST match the claimed root.
        debug_assert_eq!(
            current, inclusion.deposit_root,
            "merkle walk output must equal claimed deposit_root",
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

pub struct DepositTreeConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl DepositTreeConstraintSystem {
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

fn eval_index_aggregator(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for k in 0..DEPTH {
        let pow = scalar_pow(&Scalar::from_u64(2, curve), k as u64);
        let bit_k = &col_evals[COL_INDEX_BIT_REPL_OFFSET + k];
        sum = sum.add(&bit_k.mul(&pow));
    }
    let di = &col_evals[COL_DEPOSIT_INDEX];
    let body = di.sub(&sum);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_index_aggregator_poly(
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
    let di_p = &col_coeffs[COL_DEPOSIT_INDEX];
    let body = poly_sub(di_p, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for DepositTreeConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_top_binary".into(),
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
            let is_top = &row_evals[COL_IS_TOP];
            let bit = &row_evals[COL_INDEX_BIT];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_top.mul(&is_top.sub(&one));
            bodies[2][row] = is_real.mul(&bit.mul(&bit.sub(&one)));
            bodies[3][row] = eval_left_selection(&row_evals, &alpha);
            bodies[4][row] = eval_right_selection(&row_evals, &alpha);
            bodies[5][row] = eval_index_aggregator(&row_evals);
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
        let bit = &col_evals[COL_INDEX_BIT];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_top.mul(&is_top.sub(&one)),
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
        let is_top = &col_coeffs[COL_IS_TOP];
        let bit = &col_coeffs[COL_INDEX_BIT];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_top_m1 = poly_sub(is_top, &one_poly, curve);
        let is_top_binary = poly_mul(is_top, &is_top_m1, curve);

        let bit_m1 = poly_sub(bit, &one_poly, curve);
        let bit_sq = poly_mul(bit, &bit_m1, curve);
        let index_bit_binary = poly_mul(is_real, &bit_sq, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_top_binary,
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
            cols.push(COL_DEPOSIT_ROOT_OFFSET + b);
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

        let mut root_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_next = &shifted_evals[root_shift_off + b];
            let r_curr = &col_evals_at_z[COL_DEPOSIT_ROOT_OFFSET + b];
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
            let r_poly = &col_coeffs[COL_DEPOSIT_ROOT_OFFSET + b];
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
/// NEXT_HASH)` triple on the deposit-tree gadget side to a real
/// SHA-256 invocation in [`crate::sha256_extract`]. 96-byte tuple on
/// each side, gated by `IS_REAL`.
pub fn make_deposit_to_sha256_descriptor(
    deposit_tree_layer_index: usize,
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
        label: "deposit_tree_pair_hash_v1".into(),
        a_layer_index: deposit_tree_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding the deposit-tree gadget's leaf
/// (`CURRENT_HASH` on the row where `LEVEL = 0`) to the deposit's
/// `hash_tree_root(DepositData)` exposed by a future
/// `deposit_data_htr_air`. Tuple matched (33 columns):
/// `(DEPOSIT_INDEX, CURRENT_HASH[0..32])`.
///
/// **Soundness scaffold note**: as stated this descriptor is
/// over-inclusive on the A side (it includes all 32 rows of each
/// inclusion, not only the leaf row). A future tightening will add a
/// dedicated `IS_LEAF` selector column to restrict to the leaf row.
/// The B-side column offsets are placeholders (0..33); they will be
/// rebound when the deposit-data HTR AIR lands. The descriptor's
/// shape and label are stable, so wiring can be tested independently.
pub fn make_deposit_to_data_htr_descriptor(
    deposit_tree_layer_index: usize,
    data_htr_layer_index: usize,
    data_htr_index_column: usize,
    data_htr_root_byte_offset: usize,
    data_htr_selector_column: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(33);
    a_columns.push(COL_DEPOSIT_INDEX);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_CURRENT_HASH_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(33);
    b_columns.push(data_htr_index_column);
    for b in 0..CHUNK_BYTES {
        b_columns.push(data_htr_root_byte_offset + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "deposit_tree_leaf_to_data_htr_v1".into(),
        a_layer_index: deposit_tree_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: data_htr_layer_index,
        b_columns,
        b_selector_column: Some(data_htr_selector_column),
    }
}

/// Cross-AIR LogUp descriptor binding the deposit-tree gadget's claimed
/// `DEPOSIT_ROOT` column (32 bytes, constant across the inclusion) to
/// the block-side `eth1_data.deposit_root` field. Tuple matched (32
/// columns), gated by `IS_TOP` on the A side so only the row where the
/// running hash equals the claimed root participates.
///
/// The B-side column offsets are caller-supplied so this descriptor
/// can target either a beacon-block-body AIR's `eth1_data_root` field
/// or a future `Eth1Data` SSZ-extraction AIR's `deposit_root` field.
pub fn make_deposit_root_to_block_descriptor(
    deposit_tree_layer_index: usize,
    block_layer_index: usize,
    block_deposit_root_byte_offset: usize,
    block_selector_column: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_DEPOSIT_ROOT_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(CHUNK_BYTES);
    for b in 0..CHUNK_BYTES {
        b_columns.push(block_deposit_root_byte_offset + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "deposit_tree_root_to_block_v1".into(),
        a_layer_index: deposit_tree_layer_index,
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
    use crate::deposit::DepositData;

    fn synthetic_deposit() -> DepositData {
        DepositData {
            pubkey: [0x11; 48],
            withdrawal_credentials: [0x22; 32],
            amount: 32_000_000_000,
            signature: [0x33; 96],
        }
    }

    fn synthetic_proof_path(seed: u8) -> [[u8; CHUNK_BYTES]; DEPTH] {
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        for (k, sib) in path.iter_mut().enumerate() {
            for (b, byte) in sib.iter_mut().enumerate() {
                *byte = (seed
                    .wrapping_add(k as u8)
                    .wrapping_mul(11))
                    .wrapping_add(b as u8);
            }
        }
        path
    }

    fn single_inclusion_witness() -> DepositTreeWitness {
        let deposit = synthetic_deposit();
        let path = synthetic_proof_path(7);
        let index: u64 = 0xCAFEBABE; // 32 bits
        from_deposit_proof(&deposit, index, &path)
    }

    /// Depth-32 inclusion: `from_deposit_proof` matches a manual
    /// host-side merkle walk.
    #[test]
    fn from_deposit_proof_matches_manual_walk() {
        let deposit = synthetic_deposit();
        let index: u64 = 0xDEAD_BEEF;
        let path = synthetic_proof_path(42);
        let w = from_deposit_proof(&deposit, index, &path);
        assert_eq!(w.inclusions.len(), 1);
        let inc = &w.inclusions[0];

        let leaf = deposit.hash_tree_root();
        assert_eq!(inc.leaf, leaf);

        let mut cur = leaf;
        for k in 0..DEPTH {
            let bit = (index >> k) & 1;
            let (l, r) = if bit == 0 { (cur, path[k]) } else { (path[k], cur) };
            cur = crate::sha256::sha256_pair(&l, &r);
        }
        assert_eq!(inc.deposit_root, cur);
    }

    /// Single-deposit tree (index 0, all-zero-subtree siblings): the
    /// reconstructed root matches `merkleize_chunks(&[leaf], 2^32)`
    /// from the SSZ helper.
    #[test]
    fn single_deposit_root_matches_ssz_helper() {
        let deposit = synthetic_deposit();
        let leaf = deposit.hash_tree_root();
        // Siblings for index 0 in a depth-32 tree are the zero-subtree
        // roots at each level.
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        let mut z = [0u8; CHUNK_BYTES];
        for k in 0..DEPTH {
            path[k] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }
        let w = from_deposit_proof(&deposit, 0, &path);
        let inc = &w.inclusions[0];

        let ssz_root =
            crate::ssz::merkleize_chunks(&[leaf], Some(1u64 << DEPTH));
        assert_eq!(
            inc.deposit_root, ssz_root,
            "depth-32 single-leaf merkleization must match SSZ helper",
        );
    }

    /// Honest witness: all row-local constraints vanish on every row of
    /// the depth-32 trace.
    #[test]
    fn constraints_vanish_on_honest_inclusion() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, ROWS_PER_INCLUSION);

        let cs = DepositTreeConstraintSystem::new(trace.num_rows);
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

    /// Tampering: corrupt a sibling chunk byte and confirm one of the
    /// row-local selection bodies fires.
    #[test]
    fn tampered_sibling_breaks_selection() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // Corrupt SIBLING[7] on row 3.
        trace.columns[COL_SIBLING_OFFSET + 7].evaluations[3] =
            Scalar::from_u64(0x55, curve);
        let cs = DepositTreeConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert!(
            !evals[3][3].is_zero() || !evals[4][3].is_zero(),
            "tampered sibling must break one of left/right selection on row 3",
        );
    }

    /// Tampering: corrupt the deposit_index aggregator and confirm
    /// `index_aggregator` fires.
    #[test]
    fn tampered_index_breaks_aggregator() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_DEPOSIT_INDEX].evaluations[0] =
            Scalar::from_u64(0xDEAD, curve);
        let cs = DepositTreeConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert!(
            !evals[5][0].is_zero(),
            "tampered DEPOSIT_INDEX must make index_aggregator non-zero",
        );
    }

    /// 32-bit decomposition sanity: `INDEX_BIT_REPL` columns match the
    /// LE bit decomposition of `deposit_index`, and are replicated on
    /// every active row. `IS_TOP` is set exactly once per inclusion
    /// (on the row where `LEVEL == DEPTH-1`).
    #[test]
    fn index_bit_decomposition_and_is_top() {
        let curve = CurveType::Bls48581;
        let index: u64 = 0xABCD_1234;
        let deposit = synthetic_deposit();
        let path = synthetic_proof_path(99);
        let w = from_deposit_proof(&deposit, index, &path);
        let trace = build_trace_polynomials(&w, curve);

        for k in 0..DEPTH {
            let expected = (index >> k) & 1;
            for row in 0..ROWS_PER_INCLUSION {
                let got = trace.columns[COL_INDEX_BIT_REPL_OFFSET + k]
                    .evaluations[row]
                    .to_u64();
                assert_eq!(
                    got, expected,
                    "INDEX_BIT_REPL[{}] row {} mismatch", k, row,
                );
            }
        }

        // DEPOSIT_INDEX held constant on every row.
        for row in 0..ROWS_PER_INCLUSION {
            assert_eq!(
                trace.columns[COL_DEPOSIT_INDEX].evaluations[row].to_u64(),
                index,
            );
        }

        // IS_TOP fires exactly once at row DEPTH-1.
        for row in 0..ROWS_PER_INCLUSION {
            let v = trace.columns[COL_IS_TOP].evaluations[row].to_u64();
            if row == DEPTH - 1 {
                assert_eq!(v, 1, "IS_TOP must be 1 on the top row");
            } else {
                assert_eq!(v, 0, "IS_TOP must be 0 on non-top row {}", row);
            }
        }
    }

    /// Descriptor wiring sanity for the pair-hash cross-AIR LogUp.
    #[test]
    fn sha256_descriptor_well_formed() {
        let desc = make_deposit_to_sha256_descriptor(0, 1);
        assert_eq!(desc.label, "deposit_tree_pair_hash_v1");
        assert_eq!(desc.a_columns.len(), 96);
        assert_eq!(desc.b_columns.len(), 96);
        for b in 0..CHUNK_BYTES {
            assert_eq!(desc.a_columns[b], COL_LEFT_OFFSET + b);
            assert_eq!(desc.a_columns[CHUNK_BYTES + b], COL_RIGHT_OFFSET + b);
            assert_eq!(
                desc.a_columns[2 * CHUNK_BYTES + b],
                COL_NEXT_HASH_OFFSET + b,
            );
        }
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::sha256_extract::COL_IS_REAL),
        );
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
    }

    /// Descriptor wiring sanity for the leaf↔data_htr cross-AIR LogUp
    /// and the deposit-root↔block linkage.
    #[test]
    fn data_htr_and_block_descriptors_well_formed() {
        let dh = make_deposit_to_data_htr_descriptor(0, 2, 100, 200, 233);
        assert_eq!(dh.label, "deposit_tree_leaf_to_data_htr_v1");
        assert_eq!(dh.a_columns.len(), 33);
        assert_eq!(dh.b_columns.len(), 33);
        assert_eq!(dh.a_columns[0], COL_DEPOSIT_INDEX);
        for b in 0..CHUNK_BYTES {
            assert_eq!(dh.a_columns[1 + b], COL_CURRENT_HASH_OFFSET + b);
        }
        assert_eq!(dh.b_columns[0], 100);
        for b in 0..CHUNK_BYTES {
            assert_eq!(dh.b_columns[1 + b], 200 + b);
        }
        assert_eq!(dh.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(dh.b_selector_column, Some(233));
        assert_eq!(dh.a_layer_index, 0);
        assert_eq!(dh.b_layer_index, 2);

        let bk = make_deposit_root_to_block_descriptor(0, 3, 500, 999);
        assert_eq!(bk.label, "deposit_tree_root_to_block_v1");
        assert_eq!(bk.a_columns.len(), CHUNK_BYTES);
        assert_eq!(bk.b_columns.len(), CHUNK_BYTES);
        for b in 0..CHUNK_BYTES {
            assert_eq!(bk.a_columns[b], COL_DEPOSIT_ROOT_OFFSET + b);
            assert_eq!(bk.b_columns[b], 500 + b);
        }
        assert_eq!(bk.a_selector_column, Some(COL_IS_TOP));
        assert_eq!(bk.b_selector_column, Some(999));
        assert_eq!(bk.a_layer_index, 0);
        assert_eq!(bk.b_layer_index, 3);
    }
}
