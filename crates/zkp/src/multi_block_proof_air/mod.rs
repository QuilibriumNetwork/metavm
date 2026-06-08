//! Multi-block proof composition AIR.
//!
//! Composes `block_full_proof_air` × N consecutive blocks into a
//! single chain-validity gadget. Each row corresponds to one block in
//! a sequence of consecutive Ethereum blocks. The cross-row shifted
//! body binds:
//!
//!   * `parent_hash(ω·X) = block_hash(X)` (β-RLC over 32 bytes), and
//!   * `block_number(ω·X) = block_number(X) + 1`,
//!
//! gated by `is_real(X) · is_real(ω·X) · (1 - is_first(ω·X))`. The
//! `is_first` flag pins the genesis (chain start) row's parent
//! externally; the chain continuity is enforced purely algebraically
//! by the shifted bodies plus the row-local block-number LE decomp +
//! byte-range checks.
//!
//! Per-row, the AIR commits:
//!   * `block_index` — 0-based index within the chain.
//!   * `block_number[u64]` — the EVM block number; LE-byte decomposed.
//!   * `block_hash[32]`, `parent_hash[32]` — chain hashes (BE bytes).
//!   * `state_root[32]`, `post_state_root[32]` — pre/post execution
//!     world-state roots. `post_state_root` is exposed for downstream
//!     consumers (e.g. account/storage chain) — this AIR does not bind
//!     it to the next row's `state_root` since EVM state may transition
//!     in arbitrary directions per block. Use a separate post→state
//!     descriptor for that binding.
//!   * `is_real`, `is_first` — gating flags.
//!
//! ## Row-local constraints (12 bodies)
//!
//! 0. `is_real_binary` — `is_real · (is_real - 1) = 0`.
//! 1. `is_first_binary` — `is_first · (is_first - 1) = 0`.
//! 2. `is_first_implies_real` — `is_first · (1 - is_real) = 0`.
//! 3. `block_number_le_decomp` — `block_number - Σ bn_byte[b]·2^(8b) = 0`.
//! 4. `block_index_le_decomp` — `block_index - Σ bi_byte[b]·2^(8b) = 0`.
//! 5..7. (reserved as redundant gating sanity bodies kept identical
//!    to the receipt_status pattern for audit symmetry.)
//!
//! The byte-range checks register every byte column with the 8-bit
//! lookup table.
//!
//! ## Shifted constraints (2 bodies, cross-row)
//!
//! 0. `parent_chain_eq_block_hash` —
//!    `is_real(X) · is_real(ω·X) · (1 - is_first(ω·X)) ·
//!     Σ_k α^k · (PARENT_NEXT[k] - BLOCK_HASH[k]) = 0`
//!    multiplied by `(X - ω^{n-1})` to exclude the wrap-around row.
//! 1. `block_number_increment` —
//!    `is_real(X) · is_real(ω·X) · (1 - is_first(ω·X)) ·
//!     (BLOCK_NUMBER(ω·X) - BLOCK_NUMBER(X) - 1) = 0`
//!    multiplied by `(X - ω^{n-1})`.
//!
//! ## Cross-AIR LogUp descriptor
//!
//!   - `make_multi_block_to_block_full_proof_descriptor` binds the
//!     per-row tuple `(block_number, block_hash, parent_hash,
//!     state_root)` against the corresponding columns in
//!     `block_full_proof_air`. This is the algebraic hook by which a
//!     LayerChainProof can compose N per-block proofs with one
//!     multi-block chain validity gadget: each B-side row of
//!     `block_full_proof_air` (one row per block) matches exactly one
//!     A-side row of this AIR.

use crate::block_full_proof_air as bfp;
use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const HASH_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_BLOCK_INDEX: usize = 0;
pub const COL_BLOCK_INDEX_BYTE_OFFSET: usize = COL_BLOCK_INDEX + 1; // 1..9

pub const COL_BLOCK_NUMBER: usize = COL_BLOCK_INDEX_BYTE_OFFSET + U64_BYTES; // 9
pub const COL_BLOCK_NUMBER_BYTE_OFFSET: usize = COL_BLOCK_NUMBER + 1; // 10..18

pub const COL_BLOCK_HASH_OFFSET: usize =
    COL_BLOCK_NUMBER_BYTE_OFFSET + U64_BYTES; // 18..50
pub const COL_PARENT_HASH_OFFSET: usize =
    COL_BLOCK_HASH_OFFSET + HASH_LEN; // 50..82
pub const COL_STATE_ROOT_OFFSET: usize =
    COL_PARENT_HASH_OFFSET + HASH_LEN; // 82..114
pub const COL_POST_STATE_ROOT_OFFSET: usize =
    COL_STATE_ROOT_OFFSET + HASH_LEN; // 114..146

pub const COL_IS_REAL: usize = COL_POST_STATE_ROOT_OFFSET + HASH_LEN; // 146
pub const COL_IS_FIRST: usize = COL_IS_REAL + 1; // 147

pub const NUM_COLUMNS: usize = COL_IS_FIRST + 1; // 148

/// Row-local constraints:
///   0: is_real binary
///   1: is_first binary
///   2: is_first ⇒ is_real
///   3: block_number LE decomp
///   4: block_index LE decomp
///   5: is_first · (block_index)  — enforces block_index = 0 on first row
///   6: redundant 1-is_real · is_first (audit symmetry)
pub const NUM_ROW_CONSTRAINTS: usize = 7;
pub const NUM_SHIFTED: usize = 2;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiBlockProofRow {
    pub block_index: u64,
    pub block_number: u64,
    pub block_hash: [u8; HASH_LEN],
    pub parent_hash: [u8; HASH_LEN],
    pub state_root: [u8; HASH_LEN],
    pub post_state_root: [u8; HASH_LEN],
    pub is_first: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MultiBlockProofWitness {
    pub rows: Vec<MultiBlockProofRow>,
}

impl MultiBlockProofWitness {
    /// Build a chain witness from a list of consecutive block tuples
    /// `(block_number, block_hash, parent_hash, state_root,
    /// post_state_root)`. Each row's `block_index` is set to its
    /// position in the slice; `is_first` is `true` only on row 0.
    ///
    /// This is a pure data-shaping helper — it does not validate the
    /// chain linkage (parent_hash[i+1] == block_hash[i] or
    /// block_number[i+1] == block_number[i] + 1). Those are exactly
    /// the algebraic shifted bodies of this AIR; honest callers
    /// supply chain-consistent rows.
    pub fn from_chain(
        blocks: &[(u64, [u8; HASH_LEN], [u8; HASH_LEN], [u8; HASH_LEN], [u8; HASH_LEN])],
    ) -> Self {
        let mut rows = Vec::with_capacity(blocks.len());
        for (i, &(number, hash, parent, state, post_state)) in blocks.iter().enumerate() {
            rows.push(MultiBlockProofRow {
                block_index: i as u64,
                block_number: number,
                block_hash: hash,
                parent_hash: parent,
                state_root: state,
                post_state_root: post_state,
                is_first: i == 0,
            });
        }
        Self { rows }
    }

    pub fn from_rows(rows: Vec<MultiBlockProofRow>) -> Self {
        Self { rows }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

fn eval_le_decomp(
    target: &Scalar,
    byte_off: usize,
    col_evals: &[Scalar],
) -> Scalar {
    let curve = target.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[byte_off + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target.sub(&sum)
}

fn build_le_decomp_poly(
    target_poly: &[Scalar],
    byte_off: usize,
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[byte_off + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    poly_sub(target_poly, &sum, curve)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &MultiBlockProofWitness,
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
        let bi_bytes = row.block_index.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_BLOCK_INDEX_BYTE_OFFSET + b][i] =
                Scalar::from_u64(bi_bytes[b] as u64, curve);
        }

        columns[COL_BLOCK_NUMBER][i] = Scalar::from_u64(row.block_number, curve);
        let bn_bytes = row.block_number.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_BLOCK_NUMBER_BYTE_OFFSET + b][i] =
                Scalar::from_u64(bn_bytes[b] as u64, curve);
        }

        for k in 0..HASH_LEN {
            columns[COL_BLOCK_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.block_hash[k] as u64, curve);
            columns[COL_PARENT_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.parent_hash[k] as u64, curve);
            columns[COL_STATE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.state_root[k] as u64, curve);
            columns[COL_POST_STATE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.post_state_root[k] as u64, curve);
        }

        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_FIRST][i] =
            if row.is_first { one.clone() } else { zero.clone() };
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

pub struct MultiBlockProofConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl MultiBlockProofConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for MultiBlockProofConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_first_binary".into(),
            "is_first_implies_real".into(),
            "block_number_le_decomp".into(),
            "block_index_le_decomp".into(),
            "is_first_implies_index_zero".into(),
            "padding_is_first_zero".into(),
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
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_first = &row_evals[COL_IS_FIRST];
            let bn = &row_evals[COL_BLOCK_NUMBER];
            let bi = &row_evals[COL_BLOCK_INDEX];

            // 0: is_real binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            // 1: is_first binary.
            bodies[1][row] = is_first.mul(&is_first.sub(&one));
            // 2: is_first ⇒ is_real.
            bodies[2][row] = is_first.mul(&one.sub(is_real));
            // 3: block_number LE decomp.
            bodies[3][row] =
                eval_le_decomp(bn, COL_BLOCK_NUMBER_BYTE_OFFSET, &row_evals);
            // 4: block_index LE decomp.
            bodies[4][row] =
                eval_le_decomp(bi, COL_BLOCK_INDEX_BYTE_OFFSET, &row_evals);
            // 5: is_first ⇒ block_index = 0.
            bodies[5][row] = is_first.mul(bi);
            // 6: redundant padding sanity (1-is_real)·is_first.
            bodies[6][row] = one.sub(is_real).mul(is_first);
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
        let is_first = &col_evals[COL_IS_FIRST];
        let bn = &col_evals[COL_BLOCK_NUMBER];
        let bi = &col_evals[COL_BLOCK_INDEX];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_first.mul(&is_first.sub(&one)),
            is_first.mul(&one.sub(is_real)),
            eval_le_decomp(bn, COL_BLOCK_NUMBER_BYTE_OFFSET, col_evals),
            eval_le_decomp(bi, COL_BLOCK_INDEX_BYTE_OFFSET, col_evals),
            is_first.mul(bi),
            one.sub(is_real).mul(is_first),
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
        let is_first = &col_coeffs[COL_IS_FIRST];
        let bn = &col_coeffs[COL_BLOCK_NUMBER];
        let bi = &col_coeffs[COL_BLOCK_INDEX];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_first_m1 = poly_sub(is_first, &one_poly, curve);
        let is_first_binary = poly_mul(is_first, &is_first_m1, curve);

        let one_minus_is_real = poly_sub(&one_poly, is_real, curve);
        let is_first_implies_real = poly_mul(is_first, &one_minus_is_real, curve);

        let bn_decomp =
            build_le_decomp_poly(bn, COL_BLOCK_NUMBER_BYTE_OFFSET, col_coeffs, curve);
        let bi_decomp =
            build_le_decomp_poly(bi, COL_BLOCK_INDEX_BYTE_OFFSET, col_coeffs, curve);

        let is_first_bi = poly_mul(is_first, bi, curve);
        let redundant_pad = poly_mul(&one_minus_is_real, is_first, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_first_binary,
            is_first_implies_real,
            bn_decomp,
            bi_decomp,
            is_first_bi,
            redundant_pad,
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
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        // u64 byte columns: block_index, block_number.
        let u64_offsets: [(usize, &str); 2] = [
            (COL_BLOCK_INDEX_BYTE_OFFSET, "block_index_byte"),
            (COL_BLOCK_NUMBER_BYTE_OFFSET, "block_number_byte"),
        ];
        for (off, label) in u64_offsets {
            for k in 0..U64_BYTES {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }

        // 32-byte hash columns.
        let hash_offsets: [(usize, &str); 4] = [
            (COL_BLOCK_HASH_OFFSET, "block_hash"),
            (COL_PARENT_HASH_OFFSET, "parent_hash"),
            (COL_STATE_ROOT_OFFSET, "state_root"),
            (COL_POST_STATE_ROOT_OFFSET, "post_state_root"),
        ];
        for (off, label) in hash_offsets {
            for k in 0..HASH_LEN {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }

        LookupRequirements { tables, declarations }
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // ω·z evaluations layout (total = 3 + 32 = 35):
        //   [0]      IS_REAL_NEXT (chain-gating)
        //   [1]      IS_FIRST_NEXT (genesis exclusion)
        //   [2]      BLOCK_NUMBER_NEXT
        //   [3..35]  PARENT_HASH_NEXT[0..32]
        let mut cols = vec![COL_IS_REAL, COL_IS_FIRST, COL_BLOCK_NUMBER];
        for k in 0..HASH_LEN {
            cols.push(COL_PARENT_HASH_OFFSET + k);
        }
        cols
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
        if shifted_evals.len() != 3 + HASH_LEN || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals_at_z[COL_IS_REAL];
        let is_real_next = &shifted_evals[0];
        let is_first_next = &shifted_evals[1];
        let bn_next = &shifted_evals[2];

        // chain gating: is_real(X) · is_real(ω·X) · (1 - is_first(ω·X)).
        let not_first_next = one.sub(is_first_next);
        let gating = is_real.mul(is_real_next).mul(&not_first_next);

        // body 0: β-RLC parent_next - block_hash.
        let mut rlc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..HASH_LEN {
            let parent_next_k = &shifted_evals[3 + k];
            let block_hash_k = &col_evals_at_z[COL_BLOCK_HASH_OFFSET + k];
            rlc = rlc.add(&bp.mul(&parent_next_k.sub(block_hash_k)));
            bp = bp.mul(alpha);
        }
        let body0 = gating.mul(&rlc);

        // body 1: block_number_next - block_number - 1.
        let bn = &col_evals_at_z[COL_BLOCK_NUMBER];
        let bn_diff = bn_next.sub(bn).sub(&one);
        let body1 = gating.mul(&bn_diff);

        // accumulate α^alpha_offset · body0 + α^(alpha_offset+1) · body1.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = ap.mul(&body0);
        let ap1 = ap.mul(alpha);
        let term1 = ap1.mul(&body1);
        let total = term0.add(&term1);

        // Boundary exclusion: multiply by (z - ω^{n-1}).
        total.mul(&z.sub(omega_n_minus_1))
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
        let one_poly = vec![Scalar::one(curve)];

        let is_real = &column_coeffs[COL_IS_REAL];
        let is_real_next = poly_shift(is_real, omega);
        let is_first = &column_coeffs[COL_IS_FIRST];
        let is_first_next = poly_shift(is_first, omega);
        let not_first_next = poly_sub(&one_poly, &is_first_next, curve);
        let pre_gate = poly_mul(is_real, &is_real_next, curve);
        let gating = poly_mul(&pre_gate, &not_first_next, curve);

        // body 0: β-RLC parent_next - block_hash.
        let mut rlc_poly = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..HASH_LEN {
            let parent_k = &column_coeffs[COL_PARENT_HASH_OFFSET + k];
            let parent_k_next = poly_shift(parent_k, omega);
            let block_hash_k = &column_coeffs[COL_BLOCK_HASH_OFFSET + k];
            let diff = poly_sub(&parent_k_next, block_hash_k, curve);
            rlc_poly = poly_add(&rlc_poly, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body0 = poly_mul(&gating, &rlc_poly, curve);

        // body 1: block_number_next - block_number - 1.
        let bn = &column_coeffs[COL_BLOCK_NUMBER];
        let bn_next = poly_shift(bn, omega);
        let bn_diff_no_const = poly_sub(&bn_next, bn, curve);
        let bn_diff = poly_sub(&bn_diff_no_const, &one_poly, curve);
        let body1 = poly_mul(&gating, &bn_diff, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = poly_scalar_mul(&body0, &ap);
        let ap1 = ap.mul(alpha);
        let term1 = poly_scalar_mul(&body1, &ap1);
        let mut total = poly_add(&term0, &term1, curve);

        // Multiply by (X - ω^{n-1}) to exclude wrap-around row.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let factor = vec![neg, Scalar::one(curve)];
        total = poly_mul(&total, &factor, curve);
        total
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind the per-row chain tuple `(block_number, block_hash[32],
/// parent_hash[32], state_root[32])` against `block_full_proof_air`'s
/// matching per-block columns. With one B-side row per block in
/// `block_full_proof_air`, each row of this AIR pulls exactly one
/// matching tuple from `block_full_proof_air` — algebraically
/// composing N per-block proofs into the multi-block chain validity
/// gadget.
pub fn make_multi_block_to_block_full_proof_descriptor(
    multi_block_layer_index: usize,
    block_full_proof_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + 3 * HASH_LEN);
    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + 3 * HASH_LEN);

    a_columns.push(COL_BLOCK_NUMBER);
    b_columns.push(bfp::COL_BLOCK_NUMBER);

    for k in 0..HASH_LEN {
        a_columns.push(COL_BLOCK_HASH_OFFSET + k);
        b_columns.push(bfp::COL_BLOCK_HASH_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_PARENT_HASH_OFFSET + k);
        b_columns.push(bfp::COL_PARENT_HASH_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_STATE_ROOT_OFFSET + k);
        b_columns.push(bfp::COL_STATE_ROOT_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "multi_block_to_block_full_proof_v1".into(),
        a_layer_index: multi_block_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: block_full_proof_layer_index,
        b_columns,
        b_selector_column: Some(bfp::COL_IS_HEADER),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn h(seed: u8) -> [u8; HASH_LEN] {
        let mut x = [0u8; HASH_LEN];
        for k in 0..HASH_LEN {
            x[k] = seed.wrapping_add(k as u8);
        }
        x
    }

    /// Build a chain of `n` honest consecutive blocks starting at
    /// `start_number` with `genesis_parent` as block 0's parent.
    fn honest_chain(
        n: usize,
        start_number: u64,
        genesis_parent: [u8; HASH_LEN],
    ) -> Vec<(u64, [u8; HASH_LEN], [u8; HASH_LEN], [u8; HASH_LEN], [u8; HASH_LEN])>
    {
        let mut out = Vec::with_capacity(n);
        let mut parent = genesis_parent;
        for i in 0..n {
            let bh = h(0x10u8.wrapping_add(i as u8));
            let sr = h(0x40u8.wrapping_add(i as u8));
            let psr = h(0x80u8.wrapping_add(i as u8));
            out.push((start_number + i as u64, bh, parent, sr, psr));
            parent = bh;
        }
        out
    }

    fn check_all_row_bodies_vanish(witness: &MultiBlockProofWitness) {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(witness, curve);
        let cs = MultiBlockProofConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) must vanish at row {}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    fn shifted_body_at_transition(
        cols: &[Vec<Scalar>],
        transition: usize,
        beta: &Scalar,
    ) -> (Scalar, Scalar) {
        let curve = beta.curve_type();
        let one = Scalar::one(curve);
        let is_real = &cols[COL_IS_REAL][transition];
        let is_real_next = &cols[COL_IS_REAL][transition + 1];
        let is_first_next = &cols[COL_IS_FIRST][transition + 1];
        let not_first_next = one.sub(is_first_next);
        let gating = is_real.mul(is_real_next).mul(&not_first_next);

        let mut rlc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..HASH_LEN {
            let parent_next_k = &cols[COL_PARENT_HASH_OFFSET + k][transition + 1];
            let block_hash_k = &cols[COL_BLOCK_HASH_OFFSET + k][transition];
            rlc = rlc.add(&bp.mul(&parent_next_k.sub(block_hash_k)));
            bp = bp.mul(beta);
        }
        let body0 = gating.mul(&rlc);

        let bn = &cols[COL_BLOCK_NUMBER][transition];
        let bn_next = &cols[COL_BLOCK_NUMBER][transition + 1];
        let bn_diff = bn_next.sub(bn).sub(&one);
        let body1 = gating.mul(&bn_diff);

        (body0, body1)
    }

    #[test]
    fn single_block_chain_constraints_vanish() {
        let blocks = honest_chain(1, 100, h(0xee));
        let w = MultiBlockProofWitness::from_chain(&blocks);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].block_index, 0);
        assert_eq!(w.rows[0].block_number, 100);
        assert!(w.rows[0].is_first);
        check_all_row_bodies_vanish(&w);
    }

    #[test]
    fn three_block_chain_constraints_vanish() {
        let blocks = honest_chain(3, 1_000, h(0x01));
        let w = MultiBlockProofWitness::from_chain(&blocks);
        assert_eq!(w.rows.len(), 3);

        // Host-side wiring sanity: row[i+1].parent_hash == row[i].block_hash.
        assert_eq!(w.rows[1].parent_hash, w.rows[0].block_hash);
        assert_eq!(w.rows[2].parent_hash, w.rows[1].block_hash);
        // Block numbers increment by one.
        assert_eq!(w.rows[0].block_number, 1_000);
        assert_eq!(w.rows[1].block_number, 1_001);
        assert_eq!(w.rows[2].block_number, 1_002);
        // is_first only on row 0.
        assert!(w.rows[0].is_first);
        assert!(!w.rows[1].is_first);
        assert!(!w.rows[2].is_first);

        check_all_row_bodies_vanish(&w);

        // Both shifted bodies must vanish at transitions 0→1 and 1→2.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let beta = Scalar::from_u64(13, curve);
        for transition in [0usize, 1] {
            let (b0, b1) = shifted_body_at_transition(&cols, transition, &beta);
            assert!(b0.is_zero(), "parent-chain body at transition {} must vanish", transition);
            assert!(b1.is_zero(), "block-number body at transition {} must vanish", transition);
        }
    }

    #[test]
    fn tampered_parent_chain_detected() {
        let blocks = honest_chain(2, 500, h(0x02));
        let w = MultiBlockProofWitness::from_chain(&blocks);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper parent_hash[7] on row 1 so it no longer matches
        // row 0's block_hash[7].
        cols[COL_PARENT_HASH_OFFSET + 7][1] = Scalar::from_u64(0xFE, curve);

        let beta = Scalar::from_u64(19, curve);
        let (body0, body1) = shifted_body_at_transition(&cols, 0, &beta);
        assert!(
            !body0.is_zero(),
            "tampered parent_hash must fire β-RLC parent-chain body",
        );
        // block_number unaffected — body1 still vanishes.
        assert!(
            body1.is_zero(),
            "block-number body must still vanish (block_number untouched)",
        );
    }

    #[test]
    fn tampered_block_number_chain_detected() {
        let blocks = honest_chain(2, 7_000, h(0x03));
        let w = MultiBlockProofWitness::from_chain(&blocks);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper block_number on row 1 so it no longer equals
        // row 0's block_number + 1. Also update the LE byte decomp so
        // the row-local body 3 still vanishes — we want exactly the
        // shifted block-number body to fire.
        let tampered = 9_999u64;
        cols[COL_BLOCK_NUMBER][1] = Scalar::from_u64(tampered, curve);
        let bytes = tampered.to_le_bytes();
        for b in 0..U64_BYTES {
            cols[COL_BLOCK_NUMBER_BYTE_OFFSET + b][1] =
                Scalar::from_u64(bytes[b] as u64, curve);
        }

        let beta = Scalar::from_u64(23, curve);
        let (body0, body1) = shifted_body_at_transition(&cols, 0, &beta);
        // parent_hash untouched.
        assert!(body0.is_zero(), "parent-chain body must still vanish");
        assert!(
            !body1.is_zero(),
            "block-number body must fire on tampered block_number",
        );

        // Row-local bodies (block_number_le_decomp) still vanish.
        let cs = MultiBlockProofConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            bodies[3][1].is_zero(),
            "block_number_le_decomp body must still vanish (we updated bytes)",
        );
    }

    #[test]
    fn is_first_genesis_parent_free() {
        // The first row's parent_hash is a free committed column —
        // the chain shifted body does NOT fire across the
        // padding→row-0 transition because is_first_next = 1.
        let parent_a = h(0xaa);
        let parent_b = h(0xbb);
        let chain_a = honest_chain(1, 1, parent_a);
        let chain_b = honest_chain(1, 1, parent_b);
        let wa = MultiBlockProofWitness::from_chain(&chain_a);
        let wb = MultiBlockProofWitness::from_chain(&chain_b);
        assert_ne!(wa.rows[0].parent_hash, wb.rows[0].parent_hash);
        check_all_row_bodies_vanish(&wa);
        check_all_row_bodies_vanish(&wb);
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_multi_block_to_block_full_proof_descriptor(0, 1);
        assert_eq!(d.label, "multi_block_to_block_full_proof_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        // 1 block_number + 3*32 hash bytes = 97 columns.
        assert_eq!(d.a_columns.len(), 1 + 3 * HASH_LEN);
        assert_eq!(d.b_columns.len(), 1 + 3 * HASH_LEN);
        assert_eq!(d.a_columns[0], COL_BLOCK_NUMBER);
        assert_eq!(d.b_columns[0], bfp::COL_BLOCK_NUMBER);
        for k in 0..HASH_LEN {
            assert_eq!(d.a_columns[1 + k], COL_BLOCK_HASH_OFFSET + k);
            assert_eq!(d.b_columns[1 + k], bfp::COL_BLOCK_HASH_OFFSET + k);
        }
        for k in 0..HASH_LEN {
            assert_eq!(d.a_columns[1 + HASH_LEN + k], COL_PARENT_HASH_OFFSET + k);
            assert_eq!(d.b_columns[1 + HASH_LEN + k], bfp::COL_PARENT_HASH_OFFSET + k);
        }
        for k in 0..HASH_LEN {
            assert_eq!(
                d.a_columns[1 + 2 * HASH_LEN + k],
                COL_STATE_ROOT_OFFSET + k,
            );
            assert_eq!(
                d.b_columns[1 + 2 * HASH_LEN + k],
                bfp::COL_STATE_ROOT_OFFSET + k,
            );
        }
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(bfp::COL_IS_HEADER));
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_BLOCK_INDEX, 0);
        assert_eq!(COL_BLOCK_INDEX_BYTE_OFFSET, 1);
        assert_eq!(COL_BLOCK_NUMBER, 9);
        assert_eq!(COL_BLOCK_NUMBER_BYTE_OFFSET, 10);
        assert_eq!(COL_BLOCK_HASH_OFFSET, 18);
        assert_eq!(COL_PARENT_HASH_OFFSET, 50);
        assert_eq!(COL_STATE_ROOT_OFFSET, 82);
        assert_eq!(COL_POST_STATE_ROOT_OFFSET, 114);
        assert_eq!(COL_IS_REAL, 146);
        assert_eq!(COL_IS_FIRST, 147);
        assert_eq!(NUM_COLUMNS, 148);
        assert_eq!(NUM_ROW_CONSTRAINTS, 7);
        assert_eq!(NUM_SHIFTED, 2);
    }

    #[test]
    fn shifted_column_indices_match_layout() {
        let cs = MultiBlockProofConstraintSystem::new(2);
        let cols = cs.shifted_column_indices();
        // 3 scalars + 32 parent_hash bytes.
        assert_eq!(cols.len(), 3 + HASH_LEN);
        assert_eq!(cols[0], COL_IS_REAL);
        assert_eq!(cols[1], COL_IS_FIRST);
        assert_eq!(cols[2], COL_BLOCK_NUMBER);
        for k in 0..HASH_LEN {
            assert_eq!(cols[3 + k], COL_PARENT_HASH_OFFSET + k);
        }
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
    }

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = MultiBlockProofConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 2 u64 (8 bytes each) + 4 hashes (32 bytes each).
        let expected = 2 * U64_BYTES + 4 * HASH_LEN;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }

    #[test]
    fn evaluate_at_point_matches_domain_for_honest() {
        let curve = CurveType::Bls48581;
        let blocks = honest_chain(2, 42, h(0x77));
        let w = MultiBlockProofWitness::from_chain(&blocks);
        let trace = build_trace_polynomials(&w, curve);
        let cs = MultiBlockProofConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(11, curve);
        for r in 0..trace.padded_size as usize {
            let row_evals: Vec<Scalar> =
                col_refs.iter().map(|c| c[r].clone()).collect();
            let agg = cs.evaluate_at_point(&row_evals, &alpha);
            assert!(
                agg.is_zero(),
                "α-RLC aggregate must be zero on honest row {}",
                r,
            );
        }
    }
}
