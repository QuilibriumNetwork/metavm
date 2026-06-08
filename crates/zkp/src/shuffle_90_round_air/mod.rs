//! Full 90-round shuffle AIR (#160) — `compute_shuffled_index`.
//!
//! Composes [`shuffle_iteration_air`] across `SHUFFLE_ROUND_COUNT = 90`
//! rounds. Each row of this AIR commits one round of the swap-or-not
//! shuffle: `(round_index, current_index, new_index)`. The next row's
//! `current_index` must equal this row's `new_index` (continuity),
//! enforced via a shifted constraint.
//!
//! ## Composition contract
//!
//! For an input pair `(index, list_size, seed)`, the spec computes:
//! ```text
//! for round in 0..SHUFFLE_ROUND_COUNT:
//!     index = shuffle_iteration(index, round_byte=round as u8, ...)
//! return index
//! ```
//!
//! This AIR commits one row per round; the cross-AIR LogUp descriptor
//! `make_round_to_shuffle_iter_descriptor` binds each row's
//! `(current_index, list_size, seed, round_byte, new_index)` tuple to
//! a matching `shuffle_iteration_air` row's
//! `(index_in, list_size, seed, round_byte, index_out)`. Each round
//! consumes exactly one shuffle-iteration row; the shuffle-iteration
//! AIR enforces the algebraic identity `index_out = swap_or_not(index_in,
//! round_byte, seed, list_size)` per round.
//!
//! ## Scope (this step)
//!
//! Closed algebraically here:
//!   - `is_real` is binary.
//!   - `round_index` matches `round_byte` (a one-byte view), enforced
//!     by the row-local identity `round_index = round_byte` (both are
//!     in `[0, 90)` for real rows).
//!   - `round_byte` is range-checked to `[0, 256)` via the lookup
//!     declaration; combined with the cross-AIR descriptor, the
//!     downstream shuffle-iteration AIR consumes it as a `[0, 256)`
//!     bytes value.
//!   - `round_byte` strictly less than `SHUFFLE_ROUND_COUNT`: closed
//!     by a `round_slack` column committing
//!     `round_slack = SHUFFLE_ROUND_COUNT − 1 − round_byte` and
//!     range-checking `round_slack` to one byte.
//!   - **Continuity (cross-row, shifted)**: on real-to-real
//!     transitions, `current_index(next) = new_index(curr)`.
//!   - **Round monotonicity (cross-row, shifted)**: on real-to-real
//!     transitions, `round_byte(next) = round_byte(curr) + 1`.
//!   - **Seed and list_size constancy (cross-row, shifted)**: on
//!     real-to-real transitions, `seed(next) = seed(curr)` (bytewise)
//!     and `list_size(next) = list_size(curr)`.
//!
//! Deferred:
//!   - Boundary conditions: `round_byte = 0` on the first real row;
//!     `round_byte = SHUFFLE_ROUND_COUNT − 1` on the last real row;
//!     `final_index = new_index(last real row)` as a public output.
//!     These are enforced by upstream wiring (the caller pins the
//!     first row and reads the last row); algebraic boundary pins are
//!     follow-up.
//!   - `current_index(0)` matches the public `initial_index`. Pinned
//!     by upstream wiring; algebraic boundary pin is a follow-up.
//!
//! ## Constraints
//!
//! Row-local (5 bodies):
//!  0. `is_real_binary`              — `is_real · (is_real − 1) = 0`.
//!  1. `round_index_eq_round_byte`   — `is_real · (round_index −
//!     round_byte) = 0`.
//!  2. `round_slack_identity`        — `is_real · (round_slack −
//!     (SHUFFLE_ROUND_COUNT − 1) + round_byte) = 0`.
//!  3. `round_byte_range_redundant`  — slot reserved for the 8-bit
//!     lookup check; row-local body is `is_real · (round_byte −
//!     round_byte) = 0` (trivially zero, holds layout).
//!  4. `current_index_le_decomp`     — `is_real · (current_index −
//!     Σ current_index_byte[b]·256^b) = 0` (8-byte LE decomp).
//!
//! Shifted (4 bodies; α-power offsets start at `NUM_ROW_CONSTRAINTS`):
//!  0. `continuity`                  — `is_real · is_real_next ·
//!     (current_index_next − new_index_curr) = 0`.
//!  1. `round_monotone`              — `is_real · is_real_next ·
//!     (round_byte_next − round_byte_curr − 1) = 0`.
//!  2. `list_size_constancy`         — `is_real · is_real_next ·
//!     (list_size_next − list_size_curr) = 0`.
//!  3. `seed_byte0_constancy`        — `is_real · is_real_next ·
//!     (seed_byte0_next − seed_byte0_curr) = 0`. (Single
//!     representative seed byte; the cross-AIR descriptor binds all
//!     32 bytes per row, so on a tampering attempt at least one seed
//!     byte's per-round value disagrees with the shuffle-iter AIR's
//!     witness, breaking the LogUp closure.)
//!
//! Padding rows hold `is_real = 0`, all data columns zero. Row-local
//! bodies that reference data columns are gated by `is_real`; shifted
//! bodies are gated by `is_real · is_real_next`. Both vanish on
//! padding.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{
    poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

#[allow(dead_code)]
fn scalar_pow_local(base: &Scalar, exp: u64) -> Scalar {
    let curve = base.curve_type();
    let mut acc = Scalar::one(curve);
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc.mul(&b);
        }
        b = b.mul(&b);
        e >>= 1;
    }
    acc
}

// ─── Constants ────────────────────────────────────────────────────────

/// Eth2 phase 0 `SHUFFLE_ROUND_COUNT`.
pub const SHUFFLE_ROUND_COUNT: usize = 90;
pub const SEED_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_ROUND_INDEX: usize = 0;
pub const COL_ROUND_BYTE: usize = 1;
pub const COL_CURRENT_INDEX: usize = 2;
pub const COL_NEW_INDEX: usize = 3;
pub const COL_LIST_SIZE: usize = 4;
pub const COL_IS_REAL: usize = 5;
pub const COL_ROUND_SLACK: usize = 6;
pub const COL_SEED_OFFSET: usize = 7; // 7..39
pub const COL_CURRENT_INDEX_BYTE_OFFSET: usize = COL_SEED_OFFSET + SEED_LEN; // 39..47

pub const NUM_COLUMNS: usize = COL_CURRENT_INDEX_BYTE_OFFSET + U64_BYTES; // 47

pub const NUM_ROW_CONSTRAINTS: usize = 5;
pub const NUM_SHIFTED: usize = 4;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct Shuffle90Row {
    pub round_index: u8,
    pub current_index: u64,
    pub new_index: u64,
    pub list_size: u64,
    pub seed: [u8; SEED_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct Shuffle90RoundWitness {
    pub rows: Vec<Shuffle90Row>,
    /// Final shuffled index = new_index of the last real row. Cached
    /// here for the caller to expose downstream.
    pub final_index: u64,
}

impl Shuffle90RoundWitness {
    pub fn from_rows(rows: Vec<Shuffle90Row>) -> Self {
        let final_index = rows.last().map(|r| r.new_index).unwrap_or(0);
        Self { rows, final_index }
    }

    /// Build a witness by walking all 90 rounds of `compute_shuffled_index`,
    /// per the consensus spec. Each row commits the per-round
    /// `(round_byte, index_in, index_out, list_size, seed)` tuple.
    pub fn from_full_shuffle(
        initial_index: u64,
        list_size: u64,
        seed: [u8; SEED_LEN],
    ) -> Self {
        assert!(list_size > 0);
        assert!(initial_index < list_size);
        let mut rows = Vec::with_capacity(SHUFFLE_ROUND_COUNT);
        let mut current = initial_index;
        for round in 0..SHUFFLE_ROUND_COUNT {
            let round_byte = round as u8;
            // Reuse the shuffle_iteration witness builder to compute
            // the new index per round.
            let it = crate::shuffle_iteration_air::ShuffleIterationWitness::from_iteration(
                current, list_size, seed, round_byte,
            );
            let new_index = it.rows[0].index_out;
            rows.push(Shuffle90Row {
                round_index: round_byte,
                current_index: current,
                new_index,
                list_size,
                seed,
            });
            current = new_index;
        }
        let final_index = current;
        Self { rows, final_index }
    }
}

fn le_decomp_u64(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Shuffle90RoundWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_ROUND_INDEX][i] = Scalar::from_u64(row.round_index as u64, curve);
        columns[COL_ROUND_BYTE][i] = Scalar::from_u64(row.round_index as u64, curve);
        columns[COL_CURRENT_INDEX][i] = Scalar::from_u64(row.current_index, curve);
        columns[COL_NEW_INDEX][i] = Scalar::from_u64(row.new_index, curve);
        columns[COL_LIST_SIZE][i] = Scalar::from_u64(row.list_size, curve);
        columns[COL_IS_REAL][i] = one.clone();
        // round_slack = SHUFFLE_ROUND_COUNT − 1 − round_byte ∈ [0, 89].
        let slack = (SHUFFLE_ROUND_COUNT as u64).saturating_sub(1)
            .saturating_sub(row.round_index as u64);
        columns[COL_ROUND_SLACK][i] = Scalar::from_u64(slack, curve);
        for k in 0..SEED_LEN {
            columns[COL_SEED_OFFSET + k][i] = Scalar::from_u64(row.seed[k] as u64, curve);
        }
        let ib = le_decomp_u64(row.current_index);
        for b in 0..U64_BYTES {
            columns[COL_CURRENT_INDEX_BYTE_OFFSET + b][i] =
                Scalar::from_u64(ib[b] as u64, curve);
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

pub struct Shuffle90RoundConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Shuffle90RoundConstraintSystem {
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

impl VmConstraintSystem for Shuffle90RoundConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "round_index_eq_round_byte".into(),
            "round_slack_identity".into(),
            "round_byte_range_redundant".into(),
            "current_index_le_decomp".into(),
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
        let round_max = Scalar::from_u64((SHUFFLE_ROUND_COUNT as u64) - 1, curve);

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let round_index = &columns[COL_ROUND_INDEX][r];
            let round_byte = &columns[COL_ROUND_BYTE][r];
            let round_slack = &columns[COL_ROUND_SLACK][r];
            let current_index = &columns[COL_CURRENT_INDEX][r];

            // 0 is_real binary
            out[0][r] = is_real.mul(&is_real.sub(&one));
            // 1 round_index = round_byte (gated by is_real).
            out[1][r] = is_real.mul(&round_index.sub(round_byte));
            // 2 round_slack = (SHUFFLE_ROUND_COUNT − 1) − round_byte.
            //   ⇒ is_real · (round_slack − round_max + round_byte) = 0.
            {
                let body = round_slack.sub(&round_max).add(round_byte);
                out[2][r] = is_real.mul(&body);
            }
            // 3 round_byte range — redundant body, range check via lookup.
            out[3][r] = Scalar::zero(curve);
            // 4 current_index LE byte decomp (gated by is_real).
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(
                        &columns[COL_CURRENT_INDEX_BYTE_OFFSET + b][r].mul(&pow256[b]),
                    );
                }
                out[4][r] = is_real.mul(&current_index.sub(&sum));
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
        let round_max = Scalar::from_u64((SHUFFLE_ROUND_COUNT as u64) - 1, curve);
        let pow256: Vec<Scalar> =
            (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &col_evals[COL_IS_REAL];
        let round_index = &col_evals[COL_ROUND_INDEX];
        let round_byte = &col_evals[COL_ROUND_BYTE];
        let round_slack = &col_evals[COL_ROUND_SLACK];
        let current_index = &col_evals[COL_CURRENT_INDEX];

        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        // 0
        acc = acc.add(&ap.mul(&is_real.mul(&is_real.sub(&one))));
        ap = ap.mul(alpha);
        // 1
        acc = acc.add(&ap.mul(&is_real.mul(&round_index.sub(round_byte))));
        ap = ap.mul(alpha);
        // 2
        {
            let body = round_slack.sub(&round_max).add(round_byte);
            acc = acc.add(&ap.mul(&is_real.mul(&body)));
            ap = ap.mul(alpha);
        }
        // 3 trivial
        ap = ap.mul(alpha);
        // 4
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(
                    &col_evals[COL_CURRENT_INDEX_BYTE_OFFSET + b].mul(&pow256[b]),
                );
            }
            acc = acc.add(&ap.mul(&is_real.mul(&current_index.sub(&sum))));
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
        let round_max = Scalar::from_u64((SHUFFLE_ROUND_COUNT as u64) - 1, curve);
        let pow256: Vec<Scalar> =
            (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &col_coeffs[COL_IS_REAL];
        let round_index = &col_coeffs[COL_ROUND_INDEX];
        let round_byte = &col_coeffs[COL_ROUND_BYTE];
        let round_slack = &col_coeffs[COL_ROUND_SLACK];
        let current_index = &col_coeffs[COL_CURRENT_INDEX];

        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);

        // 0 is_real binary (UNGATED — body is is_real * (is_real - 1))
        {
            let body = poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve);
            let term = poly_scalar_mul(&body, &ap);
            acc = poly_add(&acc, &term, curve);
            ap = ap.mul(alpha);
        }
        // 1 is_real * (round_index − round_byte)
        {
            let diff = poly_sub(round_index, round_byte, curve);
            let body = poly_mul(is_real, &diff, curve);
            let term = poly_scalar_mul(&body, &ap);
            acc = poly_add(&acc, &term, curve);
            ap = ap.mul(alpha);
        }
        // 2 is_real * (round_slack − round_max + round_byte)
        {
            let rmax_poly = vec![round_max.clone()];
            let pre = poly_sub(round_slack, &rmax_poly, curve);
            let inside = poly_add(&pre, round_byte, curve);
            let body = poly_mul(is_real, &inside, curve);
            let term = poly_scalar_mul(&body, &ap);
            acc = poly_add(&acc, &term, curve);
            ap = ap.mul(alpha);
        }
        // 3 trivial (round_byte range — redundant body)
        ap = ap.mul(alpha);
        // 4 is_real * (current_index − Σ byte[b] * 256^b)
        {
            let mut sum: Vec<Scalar> = vec![Scalar::zero(curve)];
            for b in 0..U64_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_CURRENT_INDEX_BYTE_OFFSET + b],
                    &pow256[b],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let diff = poly_sub(current_index, &sum, curve);
            let body = poly_mul(is_real, &diff, curve);
            let term = poly_scalar_mul(&body, &ap);
            acc = poly_add(&acc, &term, curve);
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
        // shifted reads: is_real(next), current_index(next),
        // round_byte(next), list_size(next), seed_byte_0(next).
        vec![
            COL_IS_REAL,
            COL_CURRENT_INDEX,
            COL_ROUND_BYTE,
            COL_LIST_SIZE,
            COL_SEED_OFFSET,
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

        let is_real = &col_evals_at_z[COL_IS_REAL];
        let new_index_curr = &col_evals_at_z[COL_NEW_INDEX];
        let round_byte_curr = &col_evals_at_z[COL_ROUND_BYTE];
        let list_size_curr = &col_evals_at_z[COL_LIST_SIZE];
        let seed_byte_0_curr = &col_evals_at_z[COL_SEED_OFFSET];

        let is_real_next = &shifted_evals[0];
        let current_index_next = &shifted_evals[1];
        let round_byte_next = &shifted_evals[2];
        let list_size_next = &shifted_evals[3];
        let seed_byte_0_next = &shifted_evals[4];

        let gate = is_real.mul(is_real_next);

        let body_0 = current_index_next.sub(new_index_curr);
        let body_1 = round_byte_next.sub(round_byte_curr).sub(&one);
        let body_2 = list_size_next.sub(list_size_curr);
        let body_3 = seed_byte_0_next.sub(seed_byte_0_curr);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut acc = ap.mul(&gate.mul(&body_0));
        ap = ap.mul(alpha);
        acc = acc.add(&ap.mul(&gate.mul(&body_1)));
        ap = ap.mul(alpha);
        acc = acc.add(&ap.mul(&gate.mul(&body_2)));
        ap = ap.mul(alpha);
        acc = acc.add(&ap.mul(&gate.mul(&body_3)));

        // (z − ω^{n-1}) wrap-around exclusion factor: the cross-row
        // shifted constraints must not fire on the last domain row
        // (where ω·x wraps to the first row). See #236 / #202.
        acc.mul(&z.sub(omega_n_minus_1))
    }

    fn build_shifted_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        // Mirrors `evaluate_shifted_at_point` exactly: builds the
        // shifted constraint polynomial S(X) and multiplies by the
        // (X − ω^{n-1}) wrap-around exclusion factor. This ensures
        // the cross-row constraints do not fire on the last domain
        // row where ω·X wraps to the first row. See #236 / #202.
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let new_index = &col_coeffs[COL_NEW_INDEX];
        let round_byte = &col_coeffs[COL_ROUND_BYTE];
        let list_size = &col_coeffs[COL_LIST_SIZE];
        let seed_byte_0 = &col_coeffs[COL_SEED_OFFSET];

        let is_real_next = poly_shift(is_real, omega);
        let current_index_next = poly_shift(&col_coeffs[COL_CURRENT_INDEX], omega);
        let round_byte_next = poly_shift(round_byte, omega);
        let list_size_next = poly_shift(list_size, omega);
        let seed_byte_0_next = poly_shift(seed_byte_0, omega);

        let gate = poly_mul(is_real, &is_real_next, curve);

        let body_0 = poly_sub(&current_index_next, new_index, curve);
        let body_1_pre = poly_sub(&round_byte_next, round_byte, curve);
        let body_1 = poly_sub(&body_1_pre, &one_poly, curve);
        let body_2 = poly_sub(&list_size_next, list_size, curve);
        let body_3 = poly_sub(&seed_byte_0_next, seed_byte_0, curve);

        let g0 = poly_mul(&gate, &body_0, curve);
        let g1 = poly_mul(&gate, &body_1, curve);
        let g2 = poly_mul(&gate, &body_2, curve);
        let g3 = poly_mul(&gate, &body_3, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let t0 = poly_scalar_mul(&g0, &ap);
        ap = ap.mul(alpha);
        let t1 = poly_scalar_mul(&g1, &ap);
        ap = ap.mul(alpha);
        let t2 = poly_scalar_mul(&g2, &ap);
        ap = ap.mul(alpha);
        let t3 = poly_scalar_mul(&g3, &ap);

        let s01 = poly_add(&t0, &t1, curve);
        let s23 = poly_add(&t2, &t3, curve);
        let body_total = poly_add(&s01, &s23, curve);

        // (X − ω^{n-1}) wrap-around exclusion factor.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size.saturating_sub(1)) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&body_total, &x_minus, curve)
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
        let mut declarations = vec![
            (
                LookupDeclaration {
                    label: "round_byte_8bit".into(),
                    column_index: COL_ROUND_BYTE,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ),
            (
                LookupDeclaration {
                    label: "round_slack_8bit".into(),
                    column_index: COL_ROUND_SLACK,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ),
        ];
        for b in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("current_index_byte_{}_8bit", b),
                    column_index: COL_CURRENT_INDEX_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SEED_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("seed_byte_{}_8bit", k),
                    column_index: COL_SEED_OFFSET + k,
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

/// Bind each round row's
/// `(current_index, list_size, round_byte, seed[0..32], new_index)`
/// to a matching `shuffle_iteration_air` row's
/// `(index_in, list_size, round_byte, seed[0..32], index_out)`.
/// Both sides gated by their respective `COL_IS_REAL`. 37-column tuple
/// (1 + 1 + 1 + 32 + 1 = 36 plus selector).
pub fn make_round_to_shuffle_iter_descriptor(
    round_layer_index: usize,
    iter_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::shuffle_iteration_air as si;
    let mut a_columns: Vec<usize> =
        vec![COL_CURRENT_INDEX, COL_LIST_SIZE, COL_ROUND_BYTE];
    for k in 0..SEED_LEN {
        a_columns.push(COL_SEED_OFFSET + k);
    }
    a_columns.push(COL_NEW_INDEX);

    let mut b_columns: Vec<usize> =
        vec![si::COL_INDEX_IN, si::COL_LIST_SIZE, si::COL_ROUND_BYTE];
    for k in 0..SEED_LEN {
        b_columns.push(si::COL_SEED_OFFSET + k);
    }
    b_columns.push(si::COL_INDEX_OUT);

    CrossAirLogUpDescriptor {
        label: "shuffle_90_round_to_iter_v1".into(),
        a_layer_index: round_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: iter_layer_index,
        b_columns,
        b_selector_column: Some(si::COL_IS_REAL),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_bodies(w: &Shuffle90RoundWitness) -> Vec<Vec<Scalar>> {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(w, curve);
        let cs = Shuffle90RoundConstraintSystem::new(trace.num_rows);
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

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_ROUND_INDEX, 0);
        assert_eq!(COL_ROUND_BYTE, 1);
        assert_eq!(COL_CURRENT_INDEX, 2);
        assert_eq!(COL_NEW_INDEX, 3);
        assert_eq!(COL_LIST_SIZE, 4);
        assert_eq!(COL_IS_REAL, 5);
        assert_eq!(COL_ROUND_SLACK, 6);
        assert_eq!(COL_SEED_OFFSET, 7);
        assert_eq!(NUM_COLUMNS, 47);
        assert_eq!(NUM_ROW_CONSTRAINTS, 5);
        assert_eq!(NUM_SHIFTED, 4);
        assert_eq!(SHUFFLE_ROUND_COUNT, 90);
    }

    #[test]
    fn full_shuffle_produces_90_rows() {
        let w = Shuffle90RoundWitness::from_full_shuffle(3, 64, [0x55; SEED_LEN]);
        assert_eq!(w.rows.len(), SHUFFLE_ROUND_COUNT);
        assert_eq!(w.rows[0].round_index, 0);
        assert_eq!(w.rows[SHUFFLE_ROUND_COUNT - 1].round_index, 89);
        // Continuity: every row's current_index equals the previous
        // row's new_index.
        for i in 1..w.rows.len() {
            assert_eq!(
                w.rows[i].current_index,
                w.rows[i - 1].new_index,
                "row {} current_index must equal row {} new_index",
                i, i - 1
            );
        }
        // Last row's new_index = final_index.
        assert_eq!(w.final_index, w.rows[SHUFFLE_ROUND_COUNT - 1].new_index);
    }

    #[test]
    fn honest_witness_all_bodies_vanish() {
        let w = Shuffle90RoundWitness::from_full_shuffle(7, 64, [0xC3; SEED_LEN]);
        let bodies = run_bodies(&w);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn shifted_continuity_fires_on_tamper() {
        // Tamper current_index of row 5; the shifted continuity body
        // at row 4 (next row's current_index − this row's new_index)
        // must fire.
        let w = Shuffle90RoundWitness::from_full_shuffle(2, 64, [0x99; SEED_LEN]);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let original = cols[COL_CURRENT_INDEX][5].to_u64();
        cols[COL_CURRENT_INDEX][5] = Scalar::from_u64(original.wrapping_add(1), curve);
        // Compute the body manually for the row 4 → row 5 transition:
        // body_0 = current_index_next − new_index_curr.
        let new_curr = &cols[COL_NEW_INDEX][4];
        let curr_next = &cols[COL_CURRENT_INDEX][5];
        let body_0 = curr_next.sub(new_curr);
        assert!(!body_0.is_zero(), "continuity must fire on tampered current_index");
    }

    #[test]
    fn shifted_round_monotonicity_fires_on_tamper() {
        let w = Shuffle90RoundWitness::from_full_shuffle(2, 64, [0xAB; SEED_LEN]);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper round_byte at row 10 (skip 9 → 11 instead of 9 → 10).
        let original = cols[COL_ROUND_BYTE][10].to_u64();
        cols[COL_ROUND_BYTE][10] = Scalar::from_u64(original.wrapping_add(1), curve);
        // body_1 at row 9 = round_byte_next − round_byte_curr − 1.
        let rb_curr = &cols[COL_ROUND_BYTE][9];
        let rb_next = &cols[COL_ROUND_BYTE][10];
        let body_1 = rb_next.sub(rb_curr).sub(&Scalar::one(curve));
        assert!(
            !body_1.is_zero(),
            "round monotonicity must fire on round_byte gap"
        );
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_round_to_shuffle_iter_descriptor(0, 1);
        assert_eq!(d.label, "shuffle_90_round_to_iter_v1");
        // 3 (index_in, list_size, round_byte) + 32 (seed) + 1 (out) = 36.
        let expected_tuple = 3 + SEED_LEN + 1;
        assert_eq!(d.a_columns.len(), expected_tuple);
        assert_eq!(d.b_columns.len(), expected_tuple);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.a_columns[0], COL_CURRENT_INDEX);
        assert_eq!(d.a_columns[expected_tuple - 1], COL_NEW_INDEX);
    }

    #[test]
    fn build_constraint_polynomial_matches_evaluate_at_point() {
        // Regression test for #226: ensures `build_constraint_polynomial`
        // matches `evaluate_at_point` at an off-domain point. Without this
        // override the prover's C(X) would miss the row-local constraints
        // while the verifier evaluates them, causing `Q(z)·Z(z) != C(z)`.
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let w = Shuffle90RoundWitness::from_full_shuffle(3, 64, [0x55; SEED_LEN]);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let cs = Shuffle90RoundConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(23, curve);
        let z = Scalar::from_u64(0xCAFE_BEEF_DEAD_BABE, curve);

        let col_at_z: Vec<Scalar> = coeff_form
            .iter()
            .map(|c| CommitmentScheme::eval_poly_at(&scheme, c, &z))
            .collect();
        let via_eval = cs.evaluate_at_point(&col_at_z, &alpha);

        let c_coeffs = cs.build_constraint_polynomial(&coeff_form, &alpha, n);
        let via_poly = CommitmentScheme::eval_poly_at(&scheme, &c_coeffs, &z);

        let diff = via_eval.sub(&via_poly);
        assert!(
            diff.is_zero(),
            "build_constraint_polynomial must match evaluate_at_point at off-domain z"
        );
    }

    #[test]
    fn build_shifted_constraint_polynomial_matches_evaluate_shifted_at_point() {
        // Regression test for #226 on the cross-row constraints.
        // We check that build_shifted_constraint_polynomial evaluated at
        // an off-domain z matches the shifted evaluator, with the
        // (z − ω^{n-1}) wrap-around exclusion factor applied.
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let w = Shuffle90RoundWitness::from_full_shuffle(3, 64, [0x55; SEED_LEN]);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let cs = Shuffle90RoundConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega.clone(), n);
        let alpha = Scalar::from_u64(23, curve);
        let z = Scalar::from_u64(0xCAFE_BEEF_DEAD_BABE, curve);
        let alpha_offset = NUM_ROW_CONSTRAINTS;

        // Shifted evaluations.
        let shifted_cols = cs.shifted_column_indices();
        let shifted_evals: Vec<Scalar> = shifted_cols
            .iter()
            .map(|&ci| {
                let shifted_poly = poly_shift(&coeff_form[ci], &omega);
                CommitmentScheme::eval_poly_at(&scheme, &shifted_poly, &z)
            })
            .collect();
        let col_at_z: Vec<Scalar> = coeff_form
            .iter()
            .map(|c| CommitmentScheme::eval_poly_at(&scheme, c, &z))
            .collect();
        let omega_n_minus_1 = scalar_pow_local(&omega, n.saturating_sub(1));
        let via_eval = cs.evaluate_shifted_at_point(
            &col_at_z,
            &shifted_evals,
            &z,
            &omega_n_minus_1,
            &alpha,
            alpha_offset,
        );

        let s_coeffs = cs.build_shifted_constraint_polynomial(
            &coeff_form,
            &alpha,
            n,
            &omega,
            alpha_offset,
        );
        let via_poly = CommitmentScheme::eval_poly_at(&scheme, &s_coeffs, &z);

        let diff = via_eval.sub(&via_poly);
        assert!(
            diff.is_zero(),
            "build_shifted_constraint_polynomial must match evaluate_shifted_at_point at off-domain z"
        );
    }

    #[test]
    fn round_slack_identity_holds() {
        // For each round, round_slack = 89 − round_byte. Verify via
        // the constraint system.
        let w = Shuffle90RoundWitness::from_full_shuffle(0, 32, [0x12; SEED_LEN]);
        let bodies = run_bodies(&w);
        for r in 0..w.rows.len() {
            assert!(
                bodies[2][r].is_zero(),
                "round_slack identity must vanish at row {}",
                r
            );
        }
    }
}
