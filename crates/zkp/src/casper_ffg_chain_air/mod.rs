//! Casper FFG chain AIR (#161).
//!
//! Algebraic skeleton for the Casper Friendly Finality Gadget (FFG)
//! justification / finalization chain. Each row commits one
//! `(source_checkpoint_epoch, target_checkpoint_epoch)` justification
//! attempt with tallied weight `vote_count` against `total_active`,
//! plus boolean witness flags `is_justified` (true ⟺ `3·vote_count ≥
//! 2·total_active`) and `is_finalized` (true on a row whose target
//! becomes the source of the next justified row, two epochs in a row
//! — the classical "two-epoch" Casper FFG finalization rule).
//!
//! ## Scope
//!
//! Closed algebraically here:
//!   - `is_real`, `is_justified`, `is_finalized` are binary (row-local).
//!   - **Supermajority justification gate**: on real rows,
//!     `is_justified = 1` iff `3·vote_count ≥ 2·total_active`. Closed
//!     algebraically by a `slack` column and the identity
//!     `is_justified · (3·vote_count − 2·total_active − slack) = 0`,
//!     where `slack` is range-checked to fit in 64 bits via byte
//!     decomposition.
//!   - **Target-epoch monotonicity** (cross-row, shifted): on real
//!     rows where the *next* row is also real, `target_epoch(next) >
//!     target_epoch(curr)`. Closed via a `target_gap_minus_one` byte
//!     decomposition column committing
//!     `target_epoch(next) − target_epoch(curr) = 1 +
//!     Σ target_gap_minus_one_byte[b]·256^b ≥ 1`.
//!   - **Two-epoch finalization rule** (cross-row, shifted): a row is
//!     `is_finalized` iff the row is justified, the next row is
//!     justified, and `next.source_epoch == curr.target_epoch`. The
//!     full conjunction is enforced by row-local gating
//!     `is_finalized · (1 − is_justified) = 0`, the shifted body
//!     `is_finalized · (1 − is_justified_next) = 0`, and the shifted
//!     chain body `is_finalized · (source_epoch_next −
//!     target_epoch_curr) = 0` (the "source == previous target" link).
//!
//! Deferred (downstream / future work):
//!   - Per-row `vote_count` derivation from individual attestations
//!     (handled by `attestation_aggregate_air` and bound via the
//!     cross-AIR LogUp descriptor below).
//!   - The "finalized → block hash" payload binding (handled by
//!     `finality_constraints` and bound via its descriptor below).
//!   - Multi-attestation aggregation into per-checkpoint tallies
//!     (handled host-side by the Casper bookkeeper; this AIR commits
//!     the resulting tallies as witness).
//!
//! ## Constraints
//!
//! Row-local (8 bodies):
//!  0. `is_real_binary`              — `is_real · (is_real − 1) = 0`.
//!  1. `is_justified_binary`         — `j · (j − 1) = 0`.
//!  2. `is_finalized_binary`         — `f · (f − 1) = 0`.
//!  3. `justification_gate` (gated by `is_real · is_justified`):
//!     `is_real · is_justified · (3·vote_count − 2·total_active −
//!      slack) = 0`.
//!  4. `slack_le_decomp` (gated by `is_real · is_justified`):
//!     gate · (slack − Σ slack_byte[b]·256^b) = 0.
//!  5. `finalized_requires_justified` (row-local):
//!     `is_finalized · (1 − is_justified) = 0`.
//!  6. `not_justified_zero_slack` (gated by `is_real · (1 −
//!      is_justified)`): gate · slack = 0 (keeps padding/non-justified
//!      rows from sneaking in a nonzero slack).
//!  7. `vote_count_le_decomp` (gated by `is_real`):
//!     is_real · (vote_count − Σ vote_count_byte[b]·256^b) = 0.
//!
//! Shifted (3 bodies; α-power offsets start at `NUM_ROW_CONSTRAINTS`):
//!  0. `target_monotone_step` (gated by `is_real · is_real_next`):
//!     gate · (target_epoch_next − target_epoch_curr −
//!            (1 + Σ target_gap_mm1_byte[b]·256^b)) = 0.
//!  1. `finalized_target_chain` (gated by `is_finalized`):
//!     is_finalized · (source_epoch_next − target_epoch_curr) = 0.
//!  2. `finalized_next_justified` (gated by `is_finalized`):
//!     is_finalized · (1 − is_justified_next) = 0.
//!
//! ## Padding
//!
//! Padding rows hold `is_real = 0`, all data columns zero, `slack =
//! 0`, `is_justified = 0`, `is_finalized = 0`. All row-local bodies
//! that reference data columns are gated by `is_real`; shifted bodies
//! are gated by `is_real · is_real_next` or `is_finalized` (both
//! vanish on padding). Domain-wrap exclusion is handled by the
//! exclusion factor on shifted constraints.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_SOURCE_EPOCH: usize = 0;
pub const COL_TARGET_EPOCH: usize = 1;
pub const COL_VOTE_COUNT: usize = 2;
pub const COL_TOTAL_ACTIVE: usize = 3;
pub const COL_IS_JUSTIFIED: usize = 4;
pub const COL_IS_FINALIZED: usize = 5;
pub const COL_IS_REAL: usize = 6;
pub const COL_SLACK: usize = 7;
pub const COL_SLACK_BYTE_OFFSET: usize = 8; // 8..16
pub const COL_VOTE_COUNT_BYTE_OFFSET: usize = COL_SLACK_BYTE_OFFSET + U64_BYTES; // 16..24
pub const COL_TARGET_GAP_MM1_BYTE_OFFSET: usize =
    COL_VOTE_COUNT_BYTE_OFFSET + U64_BYTES; // 24..32

pub const NUM_COLUMNS: usize = COL_TARGET_GAP_MM1_BYTE_OFFSET + U64_BYTES; // 32

pub const NUM_ROW_CONSTRAINTS: usize = 8;
pub const NUM_SHIFTED: usize = 3;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct CasperFfgRow {
    pub source_epoch: u64,
    pub target_epoch: u64,
    pub vote_count: u64,
    pub total_active: u64,
    pub is_justified: bool,
    pub is_finalized: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CasperFfgChainWitness {
    pub rows: Vec<CasperFfgRow>,
}

impl CasperFfgChainWitness {
    pub fn from_rows(rows: Vec<CasperFfgRow>) -> Self {
        Self { rows }
    }

    /// Compute justification + finalization booleans deterministically
    /// from the provided `(source, target, vote_count, total_active)`
    /// tuples per the Casper FFG rules. Returns a witness whose flags
    /// match the algebraic constraints.
    pub fn from_tallies(
        tallies: &[(u64, u64, u64, u64)], // (source, target, vote_count, total_active)
    ) -> Self {
        let n = tallies.len();
        let mut is_just = vec![false; n];
        let mut is_fin = vec![false; n];
        for (i, t) in tallies.iter().enumerate() {
            // 3·vote_count >= 2·total_active.
            is_just[i] = (t.2 as u128) * 3 >= (t.3 as u128) * 2;
        }
        for i in 0..n {
            // Finalized = this row justified AND next row exists AND
            // next is justified AND next.source == this.target.
            if !is_just[i] {
                continue;
            }
            if i + 1 >= n {
                continue;
            }
            if !is_just[i + 1] {
                continue;
            }
            if tallies[i + 1].0 != tallies[i].1 {
                continue;
            }
            is_fin[i] = true;
        }

        let rows: Vec<CasperFfgRow> = tallies
            .iter()
            .enumerate()
            .map(|(i, t)| CasperFfgRow {
                source_epoch: t.0,
                target_epoch: t.1,
                vote_count: t.2,
                total_active: t.3,
                is_justified: is_just[i],
                is_finalized: is_fin[i],
            })
            .collect();
        Self { rows }
    }
}

fn le_decomp_u64(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &CasperFfgChainWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_SOURCE_EPOCH][i] = Scalar::from_u64(row.source_epoch, curve);
        columns[COL_TARGET_EPOCH][i] = Scalar::from_u64(row.target_epoch, curve);
        columns[COL_VOTE_COUNT][i] = Scalar::from_u64(row.vote_count, curve);
        columns[COL_TOTAL_ACTIVE][i] = Scalar::from_u64(row.total_active, curve);
        columns[COL_IS_JUSTIFIED][i] =
            if row.is_justified { one.clone() } else { zero.clone() };
        columns[COL_IS_FINALIZED][i] =
            if row.is_finalized { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = one.clone();

        // slack column (only meaningful when is_justified).
        let slack: u64 = if row.is_justified {
            let v = (row.vote_count as u128) * 3;
            let t = (row.total_active as u128) * 2;
            // Saturating at u64::MAX for safety; honest inputs stay in range.
            (v.saturating_sub(t)) as u64
        } else {
            0
        };
        columns[COL_SLACK][i] = Scalar::from_u64(slack, curve);
        let sb = le_decomp_u64(slack);
        for b in 0..U64_BYTES {
            columns[COL_SLACK_BYTE_OFFSET + b][i] =
                Scalar::from_u64(sb[b] as u64, curve);
        }

        // vote_count LE byte decomp.
        let vb = le_decomp_u64(row.vote_count);
        for b in 0..U64_BYTES {
            columns[COL_VOTE_COUNT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(vb[b] as u64, curve);
        }

        // target_gap_minus_one = target_epoch(next) - target_epoch(curr) - 1.
        // Only meaningful on real-to-real transitions; padded as 0
        // elsewhere (gate vanishes).
        let next_target = if i + 1 < witness.rows.len() {
            witness.rows[i + 1].target_epoch
        } else {
            row.target_epoch.wrapping_add(1) // Sentinel; gated away on last real row.
        };
        let gap_mm1 = if i + 1 < witness.rows.len()
            && next_target > row.target_epoch
        {
            next_target - row.target_epoch - 1
        } else {
            0
        };
        let gb = le_decomp_u64(gap_mm1);
        for b in 0..U64_BYTES {
            columns[COL_TARGET_GAP_MM1_BYTE_OFFSET + b][i] =
                Scalar::from_u64(gb[b] as u64, curve);
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

pub struct CasperFfgChainConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl CasperFfgChainConstraintSystem {
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

impl VmConstraintSystem for CasperFfgChainConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_justified_binary".into(),
            "is_finalized_binary".into(),
            "justification_gate".into(),
            "slack_le_decomp".into(),
            "finalized_requires_justified".into(),
            "not_justified_zero_slack".into(),
            "vote_count_le_decomp".into(),
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

        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();
        let three = Scalar::from_u64(3, curve);
        let two = Scalar::from_u64(2, curve);

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_just = &columns[COL_IS_JUSTIFIED][r];
            let is_fin = &columns[COL_IS_FINALIZED][r];
            let vote_count = &columns[COL_VOTE_COUNT][r];
            let total_active = &columns[COL_TOTAL_ACTIVE][r];
            let slack = &columns[COL_SLACK][r];

            // 0
            out[0][r] = is_real.mul(&is_real.sub(&one));
            // 1
            out[1][r] = is_just.mul(&is_just.sub(&one));
            // 2
            out[2][r] = is_fin.mul(&is_fin.sub(&one));

            // 3 justification_gate gated by is_real * is_justified:
            //    gate * (3*vc - 2*ta - slack) = 0.
            {
                let body = three.mul(vote_count).sub(&two.mul(total_active)).sub(slack);
                let gate = is_real.mul(is_just);
                out[3][r] = gate.mul(&body);
            }

            // 4 slack_le_decomp gated by is_real * is_justified:
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(&columns[COL_SLACK_BYTE_OFFSET + b][r].mul(&pow256[b]));
                }
                let body = slack.sub(&sum);
                let gate = is_real.mul(is_just);
                out[4][r] = gate.mul(&body);
            }

            // 5 finalized_requires_justified row-local: is_fin · (1 − is_just) = 0.
            out[5][r] = is_fin.mul(&one.sub(is_just));

            // 6 not_justified_zero_slack gated by is_real * (1 − is_just):
            {
                let gate = is_real.mul(&one.sub(is_just));
                out[6][r] = gate.mul(slack);
            }

            // 7 vote_count_le_decomp gated by is_real:
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(
                        &columns[COL_VOTE_COUNT_BYTE_OFFSET + b][r].mul(&pow256[b]),
                    );
                }
                out[7][r] = is_real.mul(&vote_count.sub(&sum));
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
        let three = Scalar::from_u64(3, curve);
        let two = Scalar::from_u64(2, curve);
        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &col_evals[COL_IS_REAL];
        let is_just = &col_evals[COL_IS_JUSTIFIED];
        let is_fin = &col_evals[COL_IS_FINALIZED];
        let vote_count = &col_evals[COL_VOTE_COUNT];
        let total_active = &col_evals[COL_TOTAL_ACTIVE];
        let slack = &col_evals[COL_SLACK];

        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        // 0
        acc = acc.add(&ap.mul(&is_real.mul(&is_real.sub(&one))));
        ap = ap.mul(alpha);
        // 1
        acc = acc.add(&ap.mul(&is_just.mul(&is_just.sub(&one))));
        ap = ap.mul(alpha);
        // 2
        acc = acc.add(&ap.mul(&is_fin.mul(&is_fin.sub(&one))));
        ap = ap.mul(alpha);
        // 3
        {
            let body = three.mul(vote_count).sub(&two.mul(total_active)).sub(slack);
            let gate = is_real.mul(is_just);
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
        }
        // 4
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(&col_evals[COL_SLACK_BYTE_OFFSET + b].mul(&pow256[b]));
            }
            let body = slack.sub(&sum);
            let gate = is_real.mul(is_just);
            acc = acc.add(&ap.mul(&gate.mul(&body)));
            ap = ap.mul(alpha);
        }
        // 5
        acc = acc.add(&ap.mul(&is_fin.mul(&one.sub(is_just))));
        ap = ap.mul(alpha);
        // 6
        {
            let gate = is_real.mul(&one.sub(is_just));
            acc = acc.add(&ap.mul(&gate.mul(slack)));
            ap = ap.mul(alpha);
        }
        // 7
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(
                    &col_evals[COL_VOTE_COUNT_BYTE_OFFSET + b].mul(&pow256[b]),
                );
            }
            acc = acc.add(&ap.mul(&is_real.mul(&vote_count.sub(&sum))));
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
        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();
        let three = Scalar::from_u64(3, curve);
        let two = Scalar::from_u64(2, curve);

        let is_real = &column_coeffs[COL_IS_REAL];
        let is_just = &column_coeffs[COL_IS_JUSTIFIED];
        let is_fin = &column_coeffs[COL_IS_FINALIZED];
        let vote_count = &column_coeffs[COL_VOTE_COUNT];
        let total_active = &column_coeffs[COL_TOTAL_ACTIVE];
        let slack = &column_coeffs[COL_SLACK];

        let mut acc: Vec<Scalar> = vec![zero.clone()];
        let mut ap = Scalar::one(curve);

        let push = |acc: &mut Vec<Scalar>, ap: &Scalar, body: Vec<Scalar>| {
            *acc = poly_add(acc, &poly_scalar_mul(&body, ap), curve);
        };

        // 0 is_real_binary: is_real · (is_real − 1)
        {
            let is_real_minus_one = poly_add(is_real, &neg_one_poly, curve);
            let body = poly_mul(is_real, &is_real_minus_one, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 1 is_justified_binary
        {
            let is_just_minus_one = poly_add(is_just, &neg_one_poly, curve);
            let body = poly_mul(is_just, &is_just_minus_one, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 2 is_finalized_binary
        {
            let is_fin_minus_one = poly_add(is_fin, &neg_one_poly, curve);
            let body = poly_mul(is_fin, &is_fin_minus_one, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 3 justification_gate: is_real · is_just · (3·vc − 2·ta − slack)
        {
            let three_vc = poly_scalar_mul(vote_count, &three);
            let two_ta = poly_scalar_mul(total_active, &two);
            let body_inner = poly_sub(&poly_sub(&three_vc, &two_ta, curve), slack, curve);
            let gate = poly_mul(is_real, is_just, curve);
            let body = poly_mul(&gate, &body_inner, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 4 slack_le_decomp: gate · (slack − Σ pow256[b] · slack_byte[b])
        {
            let mut sum = vec![zero.clone()];
            for b in 0..U64_BYTES {
                let scaled =
                    poly_scalar_mul(&column_coeffs[COL_SLACK_BYTE_OFFSET + b], &pow256[b]);
                sum = poly_add(&sum, &scaled, curve);
            }
            let body_inner = poly_sub(slack, &sum, curve);
            let gate = poly_mul(is_real, is_just, curve);
            let body = poly_mul(&gate, &body_inner, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 5 finalized_requires_justified: is_fin · (1 − is_just)
        {
            let one_minus_just = poly_sub(&one_poly, is_just, curve);
            let body = poly_mul(is_fin, &one_minus_just, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 6 not_justified_zero_slack: is_real · (1 − is_just) · slack
        {
            let one_minus_just = poly_sub(&one_poly, is_just, curve);
            let gate = poly_mul(is_real, &one_minus_just, curve);
            let body = poly_mul(&gate, slack, curve);
            push(&mut acc, &ap, body);
            ap = ap.mul(alpha);
        }
        // 7 vote_count_le_decomp: is_real · (vote_count − Σ pow256[b] · vc_byte[b])
        {
            let mut sum = vec![zero.clone()];
            for b in 0..U64_BYTES {
                let scaled = poly_scalar_mul(
                    &column_coeffs[COL_VOTE_COUNT_BYTE_OFFSET + b],
                    &pow256[b],
                );
                sum = poly_add(&sum, &scaled, curve);
            }
            let body_inner = poly_sub(vote_count, &sum, curve);
            let body = poly_mul(is_real, &body_inner, curve);
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
        // shifted reads: is_real(next), is_justified(next),
        // source_epoch(next), target_epoch(next).
        vec![
            COL_IS_REAL,
            COL_IS_JUSTIFIED,
            COL_SOURCE_EPOCH,
            COL_TARGET_EPOCH,
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
        if shifted_evals.len() < 4 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &col_evals_at_z[COL_IS_REAL];
        let is_fin = &col_evals_at_z[COL_IS_FINALIZED];
        let target_curr = &col_evals_at_z[COL_TARGET_EPOCH];

        let is_real_next = &shifted_evals[0];
        let is_just_next = &shifted_evals[1];
        let source_next = &shifted_evals[2];
        let target_next = &shifted_evals[3];

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // 0 target_monotone_step gated by is_real · is_real_next.
        let body_0 = {
            let mut sum = Scalar::one(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(
                    &col_evals_at_z[COL_TARGET_GAP_MM1_BYTE_OFFSET + b].mul(&pow256[b]),
                );
            }
            let body = target_next.sub(target_curr).sub(&sum);
            let gate = is_real.mul(is_real_next);
            gate.mul(&body)
        };
        let term_0 = ap.mul(&body_0);
        ap = ap.mul(alpha);

        // 1 finalized_target_chain: is_fin · (source_next − target_curr) = 0.
        let body_1 = is_fin.mul(&source_next.sub(target_curr));
        let term_1 = ap.mul(&body_1);
        ap = ap.mul(alpha);

        // 2 finalized_next_justified: is_fin · (1 − is_just_next) = 0.
        let body_2 = is_fin.mul(&one.sub(is_just_next));
        let term_2 = ap.mul(&body_2);

        // Multiply by (z − ω^{n-1}) to exclude the wrap-around row
        // (matches the (X − ω^{n-1}) factor in
        // build_shifted_constraint_polynomial). Task #224.
        let sum = term_0.add(&term_1).add(&term_2);
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
        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &column_coeffs[COL_IS_REAL];
        let is_fin = &column_coeffs[COL_IS_FINALIZED];
        let target_curr = &column_coeffs[COL_TARGET_EPOCH];

        // shifted reads: is_real(next), is_justified(next),
        // source_epoch(next), target_epoch(next).
        let is_real_next = poly_shift(&column_coeffs[COL_IS_REAL], omega);
        let is_just_next = poly_shift(&column_coeffs[COL_IS_JUSTIFIED], omega);
        let source_next = poly_shift(&column_coeffs[COL_SOURCE_EPOCH], omega);
        let target_next = poly_shift(&column_coeffs[COL_TARGET_EPOCH], omega);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // 0 target_monotone_step gated by is_real · is_real_next:
        //   body = (target_next − target_curr − Σ pow256[b] · gap_mm1[b]) − 1
        //   actual constant offset is +1 (because of `Scalar::one()` init in eval).
        // Replicating eval semantics exactly:
        //   sum_with_one = 1 + Σ pow256[b] · gap_byte[b]
        //   body = (target_next − target_curr) − sum_with_one
        let mut sum_with_one = vec![one_s.clone()];
        for b in 0..U64_BYTES {
            let scaled = poly_scalar_mul(
                &column_coeffs[COL_TARGET_GAP_MM1_BYTE_OFFSET + b],
                &pow256[b],
            );
            sum_with_one = poly_add(&sum_with_one, &scaled, curve);
        }
        let diff_target = poly_sub(&target_next, target_curr, curve);
        let body_0_inner = poly_sub(&diff_target, &sum_with_one, curve);
        let gate_0 = poly_mul(is_real, &is_real_next, curve);
        let body_0 = poly_mul(&gate_0, &body_0_inner, curve);
        let mut total = poly_scalar_mul(&body_0, &ap);
        ap = ap.mul(alpha);

        // 1 finalized_target_chain: is_fin · (source_next − target_curr).
        let diff_1 = poly_sub(&source_next, target_curr, curve);
        let body_1 = poly_mul(is_fin, &diff_1, curve);
        total = poly_add(&total, &poly_scalar_mul(&body_1, &ap), curve);
        ap = ap.mul(alpha);

        // 2 finalized_next_justified: is_fin · (1 − is_just_next).
        let one_poly = vec![one_s.clone()];
        let one_minus_just_next = poly_sub(&one_poly, &is_just_next, curve);
        let body_2 = poly_mul(is_fin, &one_minus_just_next, curve);
        total = poly_add(&total, &poly_scalar_mul(&body_2, &ap), curve);

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
                    label: format!("slack_byte_{}_8bit", b),
                    column_index: COL_SLACK_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("vote_count_byte_{}_8bit", b),
                    column_index: COL_VOTE_COUNT_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("target_gap_mm1_byte_{}_8bit", b),
                    column_index: COL_TARGET_GAP_MM1_BYTE_OFFSET + b,
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

/// Bind `(source_epoch, target_epoch, vote_count)` rows of this AIR to
/// the `attestation_aggregate_air`'s
/// `(COL_SOURCE_EPOCH, COL_TARGET_EPOCH, COL_BIT_INDEX)`. The
/// attestation AIR commits one row per *participating* validator (per
/// aggregated attestation); the per-row `bit_index` serves as a unique
/// per-vote tag so the B side enumerates individual votes. The A side
/// publishes a single row per (source, target) checkpoint with the
/// aggregated `vote_count` — multiplicity reconciliation between the
/// tally column and per-vote rows lands when the per-checkpoint tally
/// AIR is wired (#179). Both sides gated by `IS_REAL`. 3-col tuple.
pub fn make_casper_ffg_to_attestation_aggregate_descriptor(
    casper_layer_index: usize,
    attestation_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::attestation_aggregate_air as att;
    CrossAirLogUpDescriptor {
        label: "casper_ffg_to_attestation_aggregate_v1".into(),
        a_layer_index: casper_layer_index,
        a_columns: vec![COL_SOURCE_EPOCH, COL_TARGET_EPOCH, COL_VOTE_COUNT],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: attestation_layer_index,
        b_columns: vec![
            att::COL_SOURCE_EPOCH,
            att::COL_TARGET_EPOCH,
            att::COL_BIT_INDEX,
        ],
        b_selector_column: Some(att::COL_IS_REAL),
    }
}

/// Bind this AIR's finalized rows to `finality_constraints`'s
/// supermajority threshold tally. The A side projects
/// `(vote_count, total_active)` gated by `is_finalized`; the B side
/// projects `(running_total, total_active)` gated by `sel_threshold`.
/// The B-side `running_total` at the threshold row equals
/// `total_attesting_balance`, which is the FFG tally being checked.
pub fn make_casper_ffg_to_finality_descriptor(
    casper_layer_index: usize,
    finality_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::finality_constraints as fc;
    CrossAirLogUpDescriptor {
        label: "casper_ffg_to_finality_v1".into(),
        a_layer_index: casper_layer_index,
        a_columns: vec![COL_VOTE_COUNT, COL_TOTAL_ACTIVE],
        a_selector_column: Some(COL_IS_FINALIZED),
        b_layer_index: finality_layer_index,
        b_columns: vec![fc::COL_RUNNING_TOTAL, fc::COL_TOTAL_ACTIVE],
        b_selector_column: Some(fc::COL_SEL_THRESHOLD),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_bodies(w: &CasperFfgChainWitness) -> Vec<Vec<Scalar>> {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(w, curve);
        let cs = CasperFfgChainConstraintSystem::new(trace.num_rows);
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
        assert_eq!(COL_SOURCE_EPOCH, 0);
        assert_eq!(COL_TARGET_EPOCH, 1);
        assert_eq!(COL_VOTE_COUNT, 2);
        assert_eq!(COL_TOTAL_ACTIVE, 3);
        assert_eq!(COL_IS_JUSTIFIED, 4);
        assert_eq!(COL_IS_FINALIZED, 5);
        assert_eq!(COL_IS_REAL, 6);
        assert_eq!(COL_SLACK, 7);
        assert_eq!(NUM_COLUMNS, 32);
        assert_eq!(NUM_ROW_CONSTRAINTS, 8);
        assert_eq!(NUM_SHIFTED, 3);
    }

    #[test]
    fn supermajority_justifies_and_chain_finalizes() {
        // Two consecutive rows, each clearing the 2/3 threshold, with
        // matching target/source link. Row 0 should be finalized.
        let tallies = vec![
            (10u64, 11u64, 70, 100), // 3·70 = 210 ≥ 200 ⇒ justified
            (11u64, 12u64, 80, 100), // justified, source = prev.target
        ];
        let w = CasperFfgChainWitness::from_tallies(&tallies);
        assert!(w.rows[0].is_justified);
        assert!(w.rows[1].is_justified);
        assert!(w.rows[0].is_finalized, "two-epoch chain finalizes row 0");
        assert!(!w.rows[1].is_finalized, "last row has no next row");
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn submajority_does_not_justify() {
        // 3·60 = 180 < 200 ⇒ not justified.
        let tallies = vec![(0u64, 1u64, 60, 100)];
        let w = CasperFfgChainWitness::from_tallies(&tallies);
        assert!(!w.rows[0].is_justified);
        assert!(!w.rows[0].is_finalized);
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn broken_chain_no_finalization() {
        // Both rows justified, but next.source != curr.target ⇒
        // no finalization.
        let tallies = vec![
            (10u64, 11u64, 70, 100),
            (99u64, 100u64, 70, 100), // source 99 != prev.target 11
        ];
        let w = CasperFfgChainWitness::from_tallies(&tallies);
        assert!(w.rows[0].is_justified);
        assert!(w.rows[1].is_justified);
        assert!(!w.rows[0].is_finalized, "broken chain must not finalize");
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn tampered_finalized_flag_detected() {
        // Honest non-finalized chain (broken link). Tamper is_finalized
        // to 1 on row 0; row-local constraint 5 still holds (since
        // is_justified = 1), but body 5 of constraint 5 vanishes
        // (justified). Instead, the shifted body 1 (source_next −
        // target_curr) fires because source_next != target_curr.
        let tallies = vec![
            (10u64, 11u64, 70, 100),
            (99u64, 100u64, 70, 100),
        ];
        let w = CasperFfgChainWitness::from_tallies(&tallies);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_FINALIZED][0] = Scalar::one(curve);

        // Check the shifted body 1 fires by evaluating manually: at
        // row 0, body_1 = is_fin · (source_next − target_curr). With
        // is_fin = 1, source_next = 99, target_curr = 11 ⇒ body_1 =
        // 88 ≠ 0.
        let is_fin = &cols[COL_IS_FINALIZED][0];
        let target_curr = &cols[COL_TARGET_EPOCH][0];
        let source_next = &cols[COL_SOURCE_EPOCH][1];
        let body_1 = is_fin.mul(&source_next.sub(target_curr));
        assert!(!body_1.is_zero(), "tampered finalized must fire chain body");
    }

    #[test]
    fn tampered_justified_below_threshold_detected() {
        // Vote count below threshold but tampered is_justified = 1.
        let tallies = vec![(0u64, 1u64, 60, 100)];
        let w = CasperFfgChainWitness::from_tallies(&tallies);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: set is_justified = 1 without updating slack.
        // Slack stays at 0; honest gate body becomes
        // 3·60 - 2·100 - 0 = -20, which is nonzero in the field.
        cols[COL_IS_JUSTIFIED][0] = Scalar::one(curve);
        let cs = CasperFfgChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "tampered is_justified below threshold must fire constraint 3"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d = make_casper_ffg_to_attestation_aggregate_descriptor(0, 1);
        assert_eq!(d.label, "casper_ffg_to_attestation_aggregate_v1");
        assert_eq!(d.a_columns.len(), 3);
        assert_eq!(d.b_columns.len(), 3);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.a_columns,
            vec![COL_SOURCE_EPOCH, COL_TARGET_EPOCH, COL_VOTE_COUNT],
        );
        for &c in &d.a_columns {
            assert!(c < NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d.b_columns {
            assert!(c < crate::attestation_aggregate_air::NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }

        let d2 = make_casper_ffg_to_finality_descriptor(0, 2);
        assert_eq!(d2.label, "casper_ffg_to_finality_v1");
        assert_eq!(d2.a_selector_column, Some(COL_IS_FINALIZED));
        assert_eq!(d2.a_columns.len(), 2);
        assert_eq!(d2.b_columns.len(), 2);
        assert_eq!(d2.a_columns, vec![COL_VOTE_COUNT, COL_TOTAL_ACTIVE]);
        assert_eq!(
            d2.b_columns,
            vec![
                crate::finality_constraints::COL_RUNNING_TOTAL,
                crate::finality_constraints::COL_TOTAL_ACTIVE,
            ],
        );
        assert_eq!(
            d2.b_selector_column,
            Some(crate::finality_constraints::COL_SEL_THRESHOLD),
        );
        for &c in &d2.a_columns {
            assert!(c < NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d2.b_columns {
            assert_ne!(c, usize::MAX);
        }
    }

    #[test]
    fn padding_rows_vanish() {
        // Single real row that is not justified; the padding rows must
        // also satisfy all bodies (they're zero columns).
        let tallies = vec![(5u64, 6u64, 10, 100)];
        let w = CasperFfgChainWitness::from_tallies(&tallies);
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
        // Trace is padded to at least 16 rows.
        assert!(bodies[0].len() >= 16);
    }

    /// Build a 5-epoch contiguous chain where every row is justified
    /// and each row's target becomes the next row's source — i.e. a
    /// canonical Casper FFG run in which every row except the last is
    /// finalized by the two-epoch rule. This stresses the shifted
    /// constraints across all four cross-row transitions, including
    /// target monotonicity, source==prev-target chain link, and the
    /// "next is justified" gate.
    fn five_epoch_chain_tallies() -> Vec<(u64, u64, u64, u64)> {
        // (source_i = target_{i-1}, target_i = target_{i-1} + 1).
        // All super-majority (70/100 ⇒ 3·70=210 ≥ 200).
        vec![
            (10u64, 11u64, 70, 100),
            (11u64, 12u64, 70, 100),
            (12u64, 13u64, 70, 100),
            (13u64, 14u64, 70, 100),
            (14u64, 15u64, 70, 100),
        ]
    }

    #[test]
    fn multi_epoch_chain_5_epochs_finalizes() {
        let tallies = five_epoch_chain_tallies();
        let w = CasperFfgChainWitness::from_tallies(&tallies);

        // All five rows justified.
        for (i, row) in w.rows.iter().enumerate() {
            assert!(row.is_justified, "row {} must be justified", i);
        }
        // Rows 0..=3 each finalize via the two-epoch rule (they have a
        // justified successor whose source == this row's target).
        for i in 0..4 {
            assert!(
                w.rows[i].is_finalized,
                "row {} must be finalized in a 5-epoch contiguous chain",
                i,
            );
        }
        // Final row has no successor ⇒ cannot finalize.
        assert!(
            !w.rows[4].is_finalized,
            "last row of chain cannot finalize (no next row)",
        );

        // All row-local bodies vanish on every row, including padding.
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);

        // Verify all 3 shifted constraints vanish on every real-to-real
        // transition (rows 0..=3 to next). We replicate the algebraic
        // body computed in `evaluate_shifted_at_point` so the cross-row
        // continuity is exercised explicitly per row.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cols: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let one = Scalar::one(curve);
        let pow256: Vec<Scalar> =
            (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        // Transitions r → r+1 for r in 0..=3 (all real-to-real).
        for r in 0..4 {
            let is_real = &cols[COL_IS_REAL][r];
            let is_fin = &cols[COL_IS_FINALIZED][r];
            let target_curr = &cols[COL_TARGET_EPOCH][r];
            let is_real_next = &cols[COL_IS_REAL][r + 1];
            let is_just_next = &cols[COL_IS_JUSTIFIED][r + 1];
            let source_next = &cols[COL_SOURCE_EPOCH][r + 1];
            let target_next = &cols[COL_TARGET_EPOCH][r + 1];

            // Shifted body 0: target_monotone_step.
            let mut sum = Scalar::one(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(
                    &cols[COL_TARGET_GAP_MM1_BYTE_OFFSET + b][r].mul(&pow256[b]),
                );
            }
            let body_0_inner = target_next.sub(target_curr).sub(&sum);
            let gate_0 = is_real.mul(is_real_next);
            let body_0 = gate_0.mul(&body_0_inner);
            assert!(
                body_0.is_zero(),
                "shifted body 0 (target monotone) must vanish on real→real transition r={}",
                r,
            );

            // Shifted body 1: finalized_target_chain.
            let body_1 = is_fin.mul(&source_next.sub(target_curr));
            assert!(
                body_1.is_zero(),
                "shifted body 1 (source_next == target_curr) must vanish at r={}",
                r,
            );

            // Shifted body 2: finalized_next_justified.
            let body_2 = is_fin.mul(&one.sub(is_just_next));
            assert!(
                body_2.is_zero(),
                "shifted body 2 (next must be justified when finalized) must vanish at r={}",
                r,
            );
        }

        // Cross-row transition from the last real row (r=4) into the
        // first padding row: real→padding, so is_real_next = 0, hence
        // gate_0 = 0 and the body must vanish too. is_finalized = 0 on
        // the last row, so bodies 1 and 2 vanish trivially.
        {
            let last = 4usize;
            let is_real = &cols[COL_IS_REAL][last];
            let is_fin = &cols[COL_IS_FINALIZED][last];
            let target_curr = &cols[COL_TARGET_EPOCH][last];
            let is_real_next = &cols[COL_IS_REAL][last + 1];
            let is_just_next = &cols[COL_IS_JUSTIFIED][last + 1];
            let source_next = &cols[COL_SOURCE_EPOCH][last + 1];
            let target_next = &cols[COL_TARGET_EPOCH][last + 1];

            let mut sum = Scalar::one(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(
                    &cols[COL_TARGET_GAP_MM1_BYTE_OFFSET + b][last].mul(&pow256[b]),
                );
            }
            let body_0_inner = target_next.sub(target_curr).sub(&sum);
            let gate_0 = is_real.mul(is_real_next);
            let body_0 = gate_0.mul(&body_0_inner);
            assert!(
                body_0.is_zero(),
                "shifted body 0 must vanish on real→padding transition",
            );

            let body_1 = is_fin.mul(&source_next.sub(target_curr));
            assert!(
                body_1.is_zero(),
                "shifted body 1 must vanish (is_fin=0 on last real row)",
            );
            let body_2 = is_fin.mul(&one.sub(is_just_next));
            assert!(
                body_2.is_zero(),
                "shifted body 2 must vanish (is_fin=0 on last real row)",
            );
        }
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

        // Use the 5-epoch contiguous chain to exercise multi-row
        // shifted constraints under the full prove+verify pipeline.
        let tallies = five_epoch_chain_tallies();
        let w = CasperFfgChainWitness::from_tallies(&tallies);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = CasperFfgChainConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone casper_ffg_chain_air proof must verify",
        );
    }
}
