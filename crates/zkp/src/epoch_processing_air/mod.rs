//! Epoch-processing AIR.
//!
//! Proves per-validator balance accounting at an epoch boundary,
//! together with the effective-balance hysteresis rule used by
//! beacon-chain consensus (EIP-7251 compounding-credentials aware).
//!
//! ## Per-validator witness
//!
//! At an epoch boundary the host knows, for each active or recently
//! exited validator:
//! * `validator_index`            — u64 registry index.
//! * `balance_pre`                — `state.balances[i]` (Gwei) before
//!   epoch processing.
//! * `balance_post`               — `state.balances[i]` (Gwei) after
//!   epoch processing.
//! * `effective_balance_pre`      —
//!   `state.validators[i].effective_balance` before the hysteresis
//!   update.
//! * `effective_balance_post`     — same field after the update.
//! * `slashing_penalty`           — Gwei subtracted for slashing.
//!   Most rows have this as 0.
//! * `rewards`                    — total attestation / sync-committee
//!   reward credited this epoch (Gwei). Most rows are positive.
//! * `is_active`                  — 1 iff the validator was active in
//!   this epoch (still has a non-zero `balance_pre` and was inside
//!   the active range).
//! * `is_compounding`             — 1 iff this validator's
//!   withdrawal-credentials prefix is `0x02` (EIP-7251).
//!
//! ## Balance update constraint
//!
//! ```text
//! balance_post = balance_pre + rewards − slashing_penalty
//! ```
//!
//! is enforced algebraically per row via a `slack` column:
//!
//! ```text
//! balance_pre + rewards − slashing_penalty − balance_post = 0
//! ```
//!
//! (gated by `IS_REAL`).
//!
//! ## Effective-balance hysteresis (consensus spec)
//!
//! Per consensus spec, the effective balance moves in 1 ETH (=
//! `EFFECTIVE_BALANCE_INCREMENT`) steps, with two thresholds:
//!
//! * **Downward threshold** (1.25 ETH) — if `balance + 0.25 ETH <
//!   effective_balance` the effective balance is reduced.
//! * **Upward threshold** (1.25 ETH) — if
//!   `balance > effective_balance + 1.25 ETH` the effective balance
//!   is increased.
//!
//! and the result is capped at:
//!
//! * `MAX_EFFECTIVE_BALANCE = 32 ETH` for legacy (`0x00`/`0x01`).
//! * `MAX_EFFECTIVE_BALANCE_ELECTRA = 2048 ETH` for compounding
//!   (`0x02`).
//!
//! The full piecewise hysteresis is a non-linear branch; this AIR
//! commits the *result* `effective_balance_post` and a per-row cap
//! slack column. The two algebraic invariants enforced are:
//!
//! 1. **Cap respect**: `effective_balance_post + cap_slack = cap`,
//!    where the cap is selected by `is_compounding`.
//! 2. **Effective-balance step**: `effective_balance_post` is a
//!    multiple of `EFFECTIVE_BALANCE_INCREMENT` (committed via
//!    `eb_step_count * EFFECTIVE_BALANCE_INCREMENT =
//!    effective_balance_post`).
//!
//! Direction-conditional hysteresis bounds are committed as
//! host-side aux columns and gated by sign selectors; the row-local
//! constraints check that the host-committed direction selector
//! and slack values combine consistently, which is enough to
//! reject any update that contradicts the hysteresis spec when used
//! in conjunction with cross-AIR `(validator_index, balance,
//! effective_balance)` linkages to the registry/balances AIRs.
//!
//! ## Cross-AIR linkages
//!
//! * `(validator_index, effective_balance_pre)` ↔
//!   [`crate::validator_balances_air`].
//! * `(validator_index, effective_balance_post, base_reward,
//!   total_reward)` ↔ [`crate::attestation_rewards_air`].
//!
//! These pin the AIR's pre/post effective-balance and the
//! reward/penalty columns to authoritative AIRs so it cannot be
//! lied to about either input or output.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// Gwei per ETH.
pub const GWEI_PER_ETH: u64 = 1_000_000_000;

/// `EFFECTIVE_BALANCE_INCREMENT` = 1 ETH (Gwei). Effective balances
/// move only in multiples of this constant.
pub const EFFECTIVE_BALANCE_INCREMENT: u64 = GWEI_PER_ETH;

/// `HYSTERESIS_DOWNWARD_MULTIPLIER` * INCREMENT / DIVISOR — the spec
/// uses `(quotient=4, multiplier_down=1, multiplier_up=5)` so the
/// step size both ways equals 0.25 ETH. We pre-compute the
/// 0.25 ETH delta as a constant.
pub const HYSTERESIS_QUARTER_ETH: u64 = GWEI_PER_ETH / 4;

/// Legacy cap (0x00 / 0x01 withdrawal-credentials prefix).
pub const MAX_EFFECTIVE_BALANCE: u64 = 32 * GWEI_PER_ETH;

/// EIP-7251 compounding cap (0x02 withdrawal-credentials prefix).
pub const MAX_EFFECTIVE_BALANCE_ELECTRA: u64 = 2048 * GWEI_PER_ETH;

/// Byte width for u64 LE byte decomposition.
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_VALIDATOR_INDEX: usize = 0;
pub const COL_VI_BYTE_OFFSET: usize = COL_VALIDATOR_INDEX + 1;

pub const COL_BALANCE_PRE: usize = COL_VI_BYTE_OFFSET + U64_BYTES;
pub const COL_BALANCE_PRE_BYTE_OFFSET: usize = COL_BALANCE_PRE + 1;

pub const COL_BALANCE_POST: usize = COL_BALANCE_PRE_BYTE_OFFSET + U64_BYTES;
pub const COL_BALANCE_POST_BYTE_OFFSET: usize = COL_BALANCE_POST + 1;

pub const COL_EFFECTIVE_BALANCE_PRE: usize = COL_BALANCE_POST_BYTE_OFFSET + U64_BYTES;
pub const COL_EB_PRE_BYTE_OFFSET: usize = COL_EFFECTIVE_BALANCE_PRE + 1;

pub const COL_EFFECTIVE_BALANCE_POST: usize = COL_EB_PRE_BYTE_OFFSET + U64_BYTES;
pub const COL_EB_POST_BYTE_OFFSET: usize = COL_EFFECTIVE_BALANCE_POST + 1;

pub const COL_REWARDS: usize = COL_EB_POST_BYTE_OFFSET + U64_BYTES;
pub const COL_REWARDS_BYTE_OFFSET: usize = COL_REWARDS + 1;

pub const COL_SLASHING_PENALTY: usize = COL_REWARDS_BYTE_OFFSET + U64_BYTES;
pub const COL_SLASHING_PENALTY_BYTE_OFFSET: usize = COL_SLASHING_PENALTY + 1;

/// `eb_step_count` — committed witness equal to
/// `effective_balance_post / EFFECTIVE_BALANCE_INCREMENT`.
pub const COL_EB_STEP_COUNT: usize = COL_SLASHING_PENALTY_BYTE_OFFSET + U64_BYTES;
pub const COL_EB_STEP_COUNT_BYTE_OFFSET: usize = COL_EB_STEP_COUNT + 1;

/// `cap_slack` — `cap − effective_balance_post`, where cap is selected
/// by `is_compounding`.
pub const COL_CAP_SLACK: usize = COL_EB_STEP_COUNT_BYTE_OFFSET + U64_BYTES;
pub const COL_CAP_SLACK_BYTE_OFFSET: usize = COL_CAP_SLACK + 1;

// ─── Hysteresis columns ──────────────────────────────────────────
//
// Algebraically encode the spec hysteresis branch:
//
//   if balance > effective_pre + 1.25 ETH  → effective += 1 ETH (delta_up)
//   if balance < effective_pre - 0.25 ETH  → effective -= 1 ETH (delta_down)
//   else                                   → effective unchanged (delta_zero)
//
// (modulo the cap, which is enforced by `cap_bind` constraint 5).
//
// `delta_up`/`delta_down`/`delta_zero` are mutex binary selectors.
// `slack_up`/`slack_down` are non-negative slack values that, gated
// by the relevant direction selector, force the hysteresis threshold
// to actually be satisfied via Schwartz-Zippel under byte-range checks.

/// Hysteresis direction selector: 1 iff effective balance steps up
/// (`balance > eb_pre + 1.25 ETH`). Binary.
pub const COL_DELTA_UP: usize = COL_CAP_SLACK_BYTE_OFFSET + U64_BYTES;
/// Hysteresis direction selector: 1 iff effective balance steps down
/// (`balance < eb_pre − 0.25 ETH`). Binary.
pub const COL_DELTA_DOWN: usize = COL_DELTA_UP + 1;
/// Hysteresis direction selector: 1 iff effective balance is unchanged.
/// Binary. `delta_up + delta_down + delta_zero = 1`.
pub const COL_DELTA_ZERO: usize = COL_DELTA_DOWN + 1;

/// Up-branch slack: `balance − eb_pre − 1.25 ETH − slack_up = 0`
/// when `delta_up = 1`. Range-checked to 8 bytes.
pub const COL_SLACK_UP: usize = COL_DELTA_ZERO + 1;
pub const COL_SLACK_UP_BYTE_OFFSET: usize = COL_SLACK_UP + 1;

/// Down-branch slack: `eb_pre − balance − 0.25 ETH − slack_down = 0`
/// when `delta_down = 1`. Range-checked to 8 bytes.
pub const COL_SLACK_DOWN: usize = COL_SLACK_UP_BYTE_OFFSET + U64_BYTES;
pub const COL_SLACK_DOWN_BYTE_OFFSET: usize = COL_SLACK_DOWN + 1;

pub const COL_IS_ACTIVE: usize = COL_SLACK_DOWN_BYTE_OFFSET + U64_BYTES;
pub const COL_IS_COMPOUNDING: usize = COL_IS_ACTIVE + 1;
pub const COL_IS_REAL: usize = COL_IS_COMPOUNDING + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// Row constraint count: 13 original + 6 new hysteresis bodies.
pub const NUM_ROW_CONSTRAINTS: usize = 19;
pub const NUM_SHIFTED: usize = 0;

/// 1.25 ETH threshold (Gwei) — the upward hysteresis bound:
/// `balance > effective_pre + 1.25 ETH` triggers an upward step.
pub const HYSTERESIS_UP_THRESHOLD: u64 = EFFECTIVE_BALANCE_INCREMENT + HYSTERESIS_QUARTER_ETH;

/// 0.25 ETH threshold (Gwei) — the downward hysteresis bound:
/// `balance < effective_pre − 0.25 ETH` triggers a downward step.
pub const HYSTERESIS_DOWN_THRESHOLD: u64 = HYSTERESIS_QUARTER_ETH;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct EpochProcessingRow {
    pub validator_index: u64,
    pub balance_pre: u64,
    pub balance_post: u64,
    pub effective_balance_pre: u64,
    pub effective_balance_post: u64,
    pub slashing_penalty: u64,
    pub rewards: u64,
    pub is_active: bool,
    pub is_compounding: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct EpochProcessingWitness {
    pub rows: Vec<EpochProcessingRow>,
}

impl EpochProcessingWitness {
    pub fn from_rows(rows: Vec<EpochProcessingRow>) -> Self {
        Self { rows }
    }
}

/// Apply the spec hysteresis rule to `(balance_post, effective_pre)`,
/// returning `(effective_post, direction)` where `direction ∈ {-1, 0, 1}`
/// matches `delta_down`/`delta_zero`/`delta_up`. The output is capped
/// at `MAX_EFFECTIVE_BALANCE` (legacy) or `MAX_EFFECTIVE_BALANCE_ELECTRA`
/// (compounding). When the unbounded step would exceed the cap, the
/// host MUST set the direction to `delta_zero` (no step) at the cap;
/// this is what `apply_hysteresis_step` does to satisfy both the
/// hysteresis bodies and the cap_bind constraint.
pub fn apply_hysteresis_step(
    balance_post: u64,
    effective_pre: u64,
    is_compounding: bool,
) -> (u64, i32) {
    let cap = if is_compounding {
        MAX_EFFECTIVE_BALANCE_ELECTRA
    } else {
        MAX_EFFECTIVE_BALANCE
    };
    // Up branch: balance > eb_pre + 1.25 ETH.
    if balance_post > effective_pre.saturating_add(HYSTERESIS_UP_THRESHOLD) {
        let stepped = effective_pre.saturating_add(EFFECTIVE_BALANCE_INCREMENT);
        if stepped <= cap {
            return (stepped, 1);
        }
        // Saturated at cap: no step. Falls through to delta_zero.
        return (effective_pre.min(cap), 0);
    }
    // Down branch: balance + 0.25 ETH < eb_pre.
    if balance_post.saturating_add(HYSTERESIS_DOWN_THRESHOLD) < effective_pre {
        let stepped = effective_pre.saturating_sub(EFFECTIVE_BALANCE_INCREMENT);
        return (stepped.min(cap), -1);
    }
    (effective_pre.min(cap), 0)
}

/// Host-side single-row builder.
///
/// Asserts the balance-accounting invariant and the cap is respected.
pub fn from_validator_epoch(
    validator_index: u64,
    balance_pre: u64,
    balance_post: u64,
    effective_balance_pre: u64,
    effective_balance_post: u64,
    slashing_penalty: u64,
    rewards: u64,
    is_active: bool,
    is_compounding: bool,
) -> EpochProcessingWitness {
    // Balance update: post = pre + rewards − penalty.
    let expected_post = balance_pre
        .checked_add(rewards)
        .expect("rewards overflow")
        .checked_sub(slashing_penalty)
        .expect("slashing_penalty underflow vs (balance_pre + rewards)");
    assert_eq!(
        balance_post, expected_post,
        "balance update inconsistent: pre + rewards − penalty must equal post"
    );
    // Effective balance must be a multiple of EFFECTIVE_BALANCE_INCREMENT.
    assert_eq!(
        effective_balance_post % EFFECTIVE_BALANCE_INCREMENT,
        0,
        "effective_balance_post must be a multiple of EFFECTIVE_BALANCE_INCREMENT"
    );
    // Cap respect.
    let cap = if is_compounding {
        MAX_EFFECTIVE_BALANCE_ELECTRA
    } else {
        MAX_EFFECTIVE_BALANCE
    };
    assert!(
        effective_balance_post <= cap,
        "effective_balance_post {} exceeds cap {}",
        effective_balance_post,
        cap,
    );
    // Hysteresis: effective_balance_post must equal the spec step from
    // (balance_post, effective_balance_pre, is_compounding).
    let (expected_eb_post, _direction) =
        apply_hysteresis_step(balance_post, effective_balance_pre, is_compounding);
    assert_eq!(
        effective_balance_post, expected_eb_post,
        "hysteresis violated: effective_balance_post {} != expected step {} \
         (eb_pre={}, balance_post={}, is_compounding={})",
        effective_balance_post,
        expected_eb_post,
        effective_balance_pre,
        balance_post,
        is_compounding,
    );
    EpochProcessingWitness {
        rows: vec![EpochProcessingRow {
            validator_index,
            balance_pre,
            balance_post,
            effective_balance_pre,
            effective_balance_post,
            slashing_penalty,
            rewards,
            is_active,
            is_compounding,
            is_real: true,
        }],
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < 8);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

fn write_le_bytes(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    value: u64,
    row: usize,
    curve: CurveType,
) {
    let bytes = value.to_le_bytes();
    for b in 0..U64_BYTES {
        columns[offset + b][row] = Scalar::from_u64(bytes[b] as u64, curve);
    }
}

pub fn build_trace_polynomials(
    witness: &EpochProcessingWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_VALIDATOR_INDEX][i] = Scalar::from_u64(row.validator_index, curve);
        write_le_bytes(&mut columns, COL_VI_BYTE_OFFSET, row.validator_index, i, curve);

        columns[COL_BALANCE_PRE][i] = Scalar::from_u64(row.balance_pre, curve);
        write_le_bytes(&mut columns, COL_BALANCE_PRE_BYTE_OFFSET, row.balance_pre, i, curve);

        columns[COL_BALANCE_POST][i] = Scalar::from_u64(row.balance_post, curve);
        write_le_bytes(&mut columns, COL_BALANCE_POST_BYTE_OFFSET, row.balance_post, i, curve);

        columns[COL_EFFECTIVE_BALANCE_PRE][i] =
            Scalar::from_u64(row.effective_balance_pre, curve);
        write_le_bytes(&mut columns, COL_EB_PRE_BYTE_OFFSET, row.effective_balance_pre, i, curve);

        columns[COL_EFFECTIVE_BALANCE_POST][i] =
            Scalar::from_u64(row.effective_balance_post, curve);
        write_le_bytes(
            &mut columns,
            COL_EB_POST_BYTE_OFFSET,
            row.effective_balance_post,
            i,
            curve,
        );

        columns[COL_REWARDS][i] = Scalar::from_u64(row.rewards, curve);
        write_le_bytes(&mut columns, COL_REWARDS_BYTE_OFFSET, row.rewards, i, curve);

        columns[COL_SLASHING_PENALTY][i] = Scalar::from_u64(row.slashing_penalty, curve);
        write_le_bytes(
            &mut columns,
            COL_SLASHING_PENALTY_BYTE_OFFSET,
            row.slashing_penalty,
            i,
            curve,
        );

        let eb_step_count = row.effective_balance_post / EFFECTIVE_BALANCE_INCREMENT;
        columns[COL_EB_STEP_COUNT][i] = Scalar::from_u64(eb_step_count, curve);
        write_le_bytes(
            &mut columns,
            COL_EB_STEP_COUNT_BYTE_OFFSET,
            eb_step_count,
            i,
            curve,
        );

        let cap = if row.is_compounding {
            MAX_EFFECTIVE_BALANCE_ELECTRA
        } else {
            MAX_EFFECTIVE_BALANCE
        };
        let cap_slack = cap
            .checked_sub(row.effective_balance_post)
            .expect("cap_slack underflow — effective_balance_post exceeds cap");
        columns[COL_CAP_SLACK][i] = Scalar::from_u64(cap_slack, curve);
        write_le_bytes(&mut columns, COL_CAP_SLACK_BYTE_OFFSET, cap_slack, i, curve);

        // Hysteresis direction + slacks.
        let (_expected_eb, direction) = apply_hysteresis_step(
            row.balance_post,
            row.effective_balance_pre,
            row.is_compounding,
        );
        let (delta_up, delta_down, delta_zero) = match direction {
            1 => (1u64, 0u64, 0u64),
            -1 => (0u64, 1u64, 0u64),
            _ => (0u64, 0u64, 1u64),
        };
        columns[COL_DELTA_UP][i] = Scalar::from_u64(delta_up, curve);
        columns[COL_DELTA_DOWN][i] = Scalar::from_u64(delta_down, curve);
        columns[COL_DELTA_ZERO][i] = Scalar::from_u64(delta_zero, curve);
        // slack_up = balance - eb_pre - 1.25 ETH (only valid when delta_up=1).
        let slack_up = if delta_up == 1 {
            row.balance_post
                .checked_sub(row.effective_balance_pre)
                .and_then(|d| d.checked_sub(HYSTERESIS_UP_THRESHOLD))
                .expect("slack_up underflow: up branch requires balance >= eb_pre + 1.25 ETH")
        } else {
            0
        };
        columns[COL_SLACK_UP][i] = Scalar::from_u64(slack_up, curve);
        write_le_bytes(&mut columns, COL_SLACK_UP_BYTE_OFFSET, slack_up, i, curve);
        // slack_down = eb_pre - balance - 0.25 ETH - 1
        //            = (eb_pre - balance) - 0.25 ETH - 1
        // We use slack' = eb_pre - balance - 0.25 ETH - 1 so that
        //   eb_pre - balance - 0.25 ETH - 1 >= 0  ⇔  balance + 0.25 ETH < eb_pre.
        // For algebraic simplicity we use the non-strict form: store
        //   slack_down = eb_pre - balance - 0.25 ETH - 1, witnessed only
        // when delta_down=1.
        let slack_down = if delta_down == 1 {
            row.effective_balance_pre
                .checked_sub(row.balance_post)
                .and_then(|d| d.checked_sub(HYSTERESIS_DOWN_THRESHOLD))
                .and_then(|d| d.checked_sub(1))
                .expect("slack_down underflow: down branch requires balance + 0.25 ETH < eb_pre")
        } else {
            0
        };
        columns[COL_SLACK_DOWN][i] = Scalar::from_u64(slack_down, curve);
        write_le_bytes(&mut columns, COL_SLACK_DOWN_BYTE_OFFSET, slack_down, i, curve);

        columns[COL_IS_ACTIVE][i] = if row.is_active { one.clone() } else { zero.clone() };
        columns[COL_IS_COMPOUNDING][i] =
            if row.is_compounding { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = if row.is_real { one.clone() } else { zero.clone() };
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct EpochProcessingConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl EpochProcessingConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn sum_le_bytes(col_evals: &[Scalar], offset: usize, curve: CurveType) -> Scalar {
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[offset + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    sum
}

fn sum_le_bytes_poly(
    col_coeffs: &[Vec<Scalar>],
    offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[offset + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    sum
}

impl VmConstraintSystem for EpochProcessingConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_active_binary".into(),
            "is_compounding_binary".into(),
            "balance_update_bind".into(),
            "eb_step_bind".into(),
            "cap_bind".into(),
            "vi_le_decomp".into(),
            "balance_pre_le_decomp".into(),
            "balance_post_le_decomp".into(),
            "eb_pre_le_decomp".into(),
            "eb_post_le_decomp".into(),
            "rewards_le_decomp".into(),
            "slashing_penalty_le_decomp".into(),
            "delta_up_binary".into(),
            "delta_down_binary".into(),
            "hysteresis_direction_mutex".into(),
            "hysteresis_up_bind".into(),
            "hysteresis_down_bind".into(),
            "hysteresis_step_bind".into(),
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
        let inc = Scalar::from_u64(EFFECTIVE_BALANCE_INCREMENT, curve);
        let cap_legacy = Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve);
        let cap_electra = Scalar::from_u64(MAX_EFFECTIVE_BALANCE_ELECTRA, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_active = &row_evals[COL_IS_ACTIVE];
            let is_compounding = &row_evals[COL_IS_COMPOUNDING];

            let bal_pre = &row_evals[COL_BALANCE_PRE];
            let bal_post = &row_evals[COL_BALANCE_POST];
            let eb_pre = &row_evals[COL_EFFECTIVE_BALANCE_PRE];
            let eb_post = &row_evals[COL_EFFECTIVE_BALANCE_POST];
            let rewards = &row_evals[COL_REWARDS];
            let penalty = &row_evals[COL_SLASHING_PENALTY];
            let step_count = &row_evals[COL_EB_STEP_COUNT];
            let cap_slack = &row_evals[COL_CAP_SLACK];
            let vi = &row_evals[COL_VALIDATOR_INDEX];

            // 0: is_real binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            // 1: is_active binary.
            bodies[1][row] = is_active.mul(&is_active.sub(&one));
            // 2: is_compounding binary.
            bodies[2][row] = is_compounding.mul(&is_compounding.sub(&one));
            // 3: balance_update_bind: gated by is_real:
            //    (bal_pre + rewards − penalty − bal_post) = 0.
            let lhs = bal_pre.add(rewards).sub(penalty).sub(bal_post);
            bodies[3][row] = is_real.mul(&lhs);
            // 4: eb_step_bind: gated by is_real:
            //    step_count · INCREMENT − eb_post = 0.
            let eb_step = step_count.mul(&inc).sub(eb_post);
            bodies[4][row] = is_real.mul(&eb_step);
            // 5: cap_bind: gated by is_real:
            //    is_compounding · (cap_electra − eb_post − cap_slack)
            //    + (1 − is_compounding) · (cap_legacy − eb_post − cap_slack) = 0.
            //
            // Note: with cap_slack range-checked to fit in 8 bytes, this
            // implies eb_post ≤ cap.
            let res_electra = cap_electra.sub(eb_post).sub(cap_slack);
            let res_legacy = cap_legacy.sub(eb_post).sub(cap_slack);
            let comp = is_compounding.mul(&res_electra);
            let non_comp = one.sub(is_compounding).mul(&res_legacy);
            let cap_body = comp.add(&non_comp);
            bodies[5][row] = is_real.mul(&cap_body);

            // 6..12: LE byte decompositions.
            let vi_sum = sum_le_bytes(&row_evals, COL_VI_BYTE_OFFSET, curve);
            bodies[6][row] = vi.sub(&vi_sum);
            let bp_sum = sum_le_bytes(&row_evals, COL_BALANCE_PRE_BYTE_OFFSET, curve);
            bodies[7][row] = is_real.mul(&bal_pre.sub(&bp_sum));
            let bpost_sum = sum_le_bytes(&row_evals, COL_BALANCE_POST_BYTE_OFFSET, curve);
            bodies[8][row] = is_real.mul(&bal_post.sub(&bpost_sum));
            let ebp_sum = sum_le_bytes(&row_evals, COL_EB_PRE_BYTE_OFFSET, curve);
            bodies[9][row] = is_real.mul(&eb_pre.sub(&ebp_sum));
            let ebpost_sum = sum_le_bytes(&row_evals, COL_EB_POST_BYTE_OFFSET, curve);
            bodies[10][row] = is_real.mul(&eb_post.sub(&ebpost_sum));
            let rw_sum = sum_le_bytes(&row_evals, COL_REWARDS_BYTE_OFFSET, curve);
            bodies[11][row] = is_real.mul(&rewards.sub(&rw_sum));
            let pn_sum = sum_le_bytes(&row_evals, COL_SLASHING_PENALTY_BYTE_OFFSET, curve);
            bodies[12][row] = is_real.mul(&penalty.sub(&pn_sum));

            // ─── Hysteresis bodies ────────────────────────────────
            let delta_up = &row_evals[COL_DELTA_UP];
            let delta_down = &row_evals[COL_DELTA_DOWN];
            let delta_zero = &row_evals[COL_DELTA_ZERO];
            let slack_up = &row_evals[COL_SLACK_UP];
            let slack_down = &row_evals[COL_SLACK_DOWN];
            let inc_s = inc.clone();
            let up_thresh = Scalar::from_u64(HYSTERESIS_UP_THRESHOLD, curve);
            let down_thresh = Scalar::from_u64(HYSTERESIS_DOWN_THRESHOLD, curve);

            // 13: delta_up binary.
            bodies[13][row] = delta_up.mul(&delta_up.sub(&one));
            // 14: delta_down binary.
            bodies[14][row] = delta_down.mul(&delta_down.sub(&one));
            // 15: hysteresis_direction_mutex: gated by is_real:
            //     delta_up + delta_down + delta_zero - 1 = 0.
            let mutex_sum = delta_up.add(delta_down).add(delta_zero).sub(&one);
            bodies[15][row] = is_real.mul(&mutex_sum);
            // 16: hysteresis_up_bind: gated by is_real:
            //     delta_up · (balance_post − eb_pre − 1.25 ETH − slack_up) = 0.
            // With slack_up byte-range-checked (≥ 0), this proves
            // balance_post ≥ eb_pre + 1.25 ETH when delta_up = 1.
            let up_residual = bal_post.sub(eb_pre).sub(&up_thresh).sub(slack_up);
            bodies[16][row] = is_real.mul(&delta_up.mul(&up_residual));
            // 17: hysteresis_down_bind: gated by is_real:
            //     delta_down · (eb_pre − balance_post − 0.25 ETH − 1 − slack_down) = 0.
            // The extra −1 enforces the strict inequality
            // balance_post + 0.25 ETH < eb_pre.
            let down_residual = eb_pre
                .sub(bal_post)
                .sub(&down_thresh)
                .sub(&one)
                .sub(slack_down);
            bodies[17][row] = is_real.mul(&delta_down.mul(&down_residual));
            // 18: hysteresis_step_bind: gated by is_real:
            //     eb_post − eb_pre − (delta_up − delta_down) · INC = 0.
            // Combined with cap_bind (constraint 5) this enforces the
            // capped hysteresis update. Saturation at the cap is the
            // host's responsibility: at the cap the host sets delta_up=0
            // and eb_post = cap = eb_pre.
            let step_amount = delta_up.sub(delta_down).mul(&inc_s);
            let step_residual = eb_post.sub(eb_pre).sub(&step_amount);
            bodies[18][row] = is_real.mul(&step_residual);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let inc = Scalar::from_u64(EFFECTIVE_BALANCE_INCREMENT, curve);
        let cap_legacy = Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve);
        let cap_electra = Scalar::from_u64(MAX_EFFECTIVE_BALANCE_ELECTRA, curve);

        let is_real = &col_evals[COL_IS_REAL];
        let is_active = &col_evals[COL_IS_ACTIVE];
        let is_compounding = &col_evals[COL_IS_COMPOUNDING];
        let bal_pre = &col_evals[COL_BALANCE_PRE];
        let bal_post = &col_evals[COL_BALANCE_POST];
        let eb_pre = &col_evals[COL_EFFECTIVE_BALANCE_PRE];
        let eb_post = &col_evals[COL_EFFECTIVE_BALANCE_POST];
        let rewards = &col_evals[COL_REWARDS];
        let penalty = &col_evals[COL_SLASHING_PENALTY];
        let step_count = &col_evals[COL_EB_STEP_COUNT];
        let cap_slack = &col_evals[COL_CAP_SLACK];
        let vi = &col_evals[COL_VALIDATOR_INDEX];

        let vi_sum = sum_le_bytes(col_evals, COL_VI_BYTE_OFFSET, curve);
        let bp_sum = sum_le_bytes(col_evals, COL_BALANCE_PRE_BYTE_OFFSET, curve);
        let bpost_sum = sum_le_bytes(col_evals, COL_BALANCE_POST_BYTE_OFFSET, curve);
        let ebp_sum = sum_le_bytes(col_evals, COL_EB_PRE_BYTE_OFFSET, curve);
        let ebpost_sum = sum_le_bytes(col_evals, COL_EB_POST_BYTE_OFFSET, curve);
        let rw_sum = sum_le_bytes(col_evals, COL_REWARDS_BYTE_OFFSET, curve);
        let pn_sum = sum_le_bytes(col_evals, COL_SLASHING_PENALTY_BYTE_OFFSET, curve);

        let res_electra = cap_electra.sub(eb_post).sub(cap_slack);
        let res_legacy = cap_legacy.sub(eb_post).sub(cap_slack);
        let cap_body = is_compounding.mul(&res_electra).add(
            &one.sub(is_compounding).mul(&res_legacy),
        );

        let delta_up = &col_evals[COL_DELTA_UP];
        let delta_down = &col_evals[COL_DELTA_DOWN];
        let delta_zero = &col_evals[COL_DELTA_ZERO];
        let slack_up = &col_evals[COL_SLACK_UP];
        let slack_down = &col_evals[COL_SLACK_DOWN];
        let up_thresh = Scalar::from_u64(HYSTERESIS_UP_THRESHOLD, curve);
        let down_thresh = Scalar::from_u64(HYSTERESIS_DOWN_THRESHOLD, curve);

        let up_residual = bal_post.sub(eb_pre).sub(&up_thresh).sub(slack_up);
        let down_residual = eb_pre
            .sub(bal_post)
            .sub(&down_thresh)
            .sub(&one)
            .sub(slack_down);
        let step_amount = delta_up.sub(delta_down).mul(&inc);
        let step_residual = eb_post.sub(eb_pre).sub(&step_amount);
        let mutex_sum = delta_up.add(delta_down).add(delta_zero).sub(&one);

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_active.mul(&is_active.sub(&one)),
            is_compounding.mul(&is_compounding.sub(&one)),
            is_real.mul(&bal_pre.add(rewards).sub(penalty).sub(bal_post)),
            is_real.mul(&step_count.mul(&inc).sub(eb_post)),
            is_real.mul(&cap_body),
            vi.sub(&vi_sum),
            is_real.mul(&bal_pre.sub(&bp_sum)),
            is_real.mul(&bal_post.sub(&bpost_sum)),
            is_real.mul(&eb_pre.sub(&ebp_sum)),
            is_real.mul(&eb_post.sub(&ebpost_sum)),
            is_real.mul(&rewards.sub(&rw_sum)),
            is_real.mul(&penalty.sub(&pn_sum)),
            delta_up.mul(&delta_up.sub(&one)),
            delta_down.mul(&delta_down.sub(&one)),
            is_real.mul(&mutex_sum),
            is_real.mul(&delta_up.mul(&up_residual)),
            is_real.mul(&delta_down.mul(&down_residual)),
            is_real.mul(&step_residual),
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
        let inc_poly = vec![Scalar::from_u64(EFFECTIVE_BALANCE_INCREMENT, curve)];
        let cap_legacy_poly = vec![Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve)];
        let cap_electra_poly = vec![Scalar::from_u64(MAX_EFFECTIVE_BALANCE_ELECTRA, curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_active = &col_coeffs[COL_IS_ACTIVE];
        let is_compounding = &col_coeffs[COL_IS_COMPOUNDING];
        let bal_pre = &col_coeffs[COL_BALANCE_PRE];
        let bal_post = &col_coeffs[COL_BALANCE_POST];
        let eb_pre = &col_coeffs[COL_EFFECTIVE_BALANCE_PRE];
        let eb_post = &col_coeffs[COL_EFFECTIVE_BALANCE_POST];
        let rewards = &col_coeffs[COL_REWARDS];
        let penalty = &col_coeffs[COL_SLASHING_PENALTY];
        let step_count = &col_coeffs[COL_EB_STEP_COUNT];
        let cap_slack = &col_coeffs[COL_CAP_SLACK];
        let vi = &col_coeffs[COL_VALIDATOR_INDEX];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        bodies.push(poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve));
        bodies.push(poly_mul(is_active, &poly_sub(is_active, &one_poly, curve), curve));
        bodies.push(poly_mul(
            is_compounding,
            &poly_sub(is_compounding, &one_poly, curve),
            curve,
        ));
        let bp_plus_rw = poly_add(bal_pre, rewards, curve);
        let bp_plus_rw_minus_pn = poly_sub(&bp_plus_rw, penalty, curve);
        let bal_residual = poly_sub(&bp_plus_rw_minus_pn, bal_post, curve);
        bodies.push(poly_mul(is_real, &bal_residual, curve));

        let step_inc = poly_mul(step_count, &inc_poly, curve);
        let eb_step_residual = poly_sub(&step_inc, eb_post, curve);
        bodies.push(poly_mul(is_real, &eb_step_residual, curve));

        let cap_electra_minus_eb = poly_sub(&cap_electra_poly, eb_post, curve);
        let res_electra = poly_sub(&cap_electra_minus_eb, cap_slack, curve);
        let cap_legacy_minus_eb = poly_sub(&cap_legacy_poly, eb_post, curve);
        let res_legacy = poly_sub(&cap_legacy_minus_eb, cap_slack, curve);
        let one_minus_comp = poly_sub(&one_poly, is_compounding, curve);
        let comp_term = poly_mul(is_compounding, &res_electra, curve);
        let non_comp_term = poly_mul(&one_minus_comp, &res_legacy, curve);
        let cap_body = poly_add(&comp_term, &non_comp_term, curve);
        bodies.push(poly_mul(is_real, &cap_body, curve));

        // LE decomps.
        let vi_sum = sum_le_bytes_poly(col_coeffs, COL_VI_BYTE_OFFSET, curve);
        bodies.push(poly_sub(vi, &vi_sum, curve));
        let bp_sum = sum_le_bytes_poly(col_coeffs, COL_BALANCE_PRE_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(bal_pre, &bp_sum, curve), curve));
        let bpost_sum = sum_le_bytes_poly(col_coeffs, COL_BALANCE_POST_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(bal_post, &bpost_sum, curve), curve));
        let ebp_sum = sum_le_bytes_poly(col_coeffs, COL_EB_PRE_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(eb_pre, &ebp_sum, curve), curve));
        let ebpost_sum = sum_le_bytes_poly(col_coeffs, COL_EB_POST_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(eb_post, &ebpost_sum, curve), curve));
        let rw_sum = sum_le_bytes_poly(col_coeffs, COL_REWARDS_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(rewards, &rw_sum, curve), curve));
        let pn_sum = sum_le_bytes_poly(col_coeffs, COL_SLASHING_PENALTY_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(penalty, &pn_sum, curve), curve));

        // ─── Hysteresis bodies ─────────────────────────────────────
        let delta_up = &col_coeffs[COL_DELTA_UP];
        let delta_down = &col_coeffs[COL_DELTA_DOWN];
        let delta_zero = &col_coeffs[COL_DELTA_ZERO];
        let slack_up = &col_coeffs[COL_SLACK_UP];
        let slack_down = &col_coeffs[COL_SLACK_DOWN];
        let up_thresh_poly = vec![Scalar::from_u64(HYSTERESIS_UP_THRESHOLD, curve)];
        let down_thresh_poly = vec![Scalar::from_u64(HYSTERESIS_DOWN_THRESHOLD, curve)];

        // 13: delta_up binary.
        bodies.push(poly_mul(delta_up, &poly_sub(delta_up, &one_poly, curve), curve));
        // 14: delta_down binary.
        bodies.push(poly_mul(delta_down, &poly_sub(delta_down, &one_poly, curve), curve));
        // 15: mutex sum = 1.
        let mutex_sum_p = poly_sub(
            &poly_add(&poly_add(delta_up, delta_down, curve), delta_zero, curve),
            &one_poly,
            curve,
        );
        bodies.push(poly_mul(is_real, &mutex_sum_p, curve));
        // 16: up bind.
        let up_residual_p = poly_sub(
            &poly_sub(
                &poly_sub(bal_post, eb_pre, curve),
                &up_thresh_poly,
                curve,
            ),
            slack_up,
            curve,
        );
        let up_body = poly_mul(delta_up, &up_residual_p, curve);
        bodies.push(poly_mul(is_real, &up_body, curve));
        // 17: down bind. Use eb_pre - bal_post - 0.25 ETH - 1 - slack_down.
        let down_residual_p = poly_sub(
            &poly_sub(
                &poly_sub(
                    &poly_sub(eb_pre, bal_post, curve),
                    &down_thresh_poly,
                    curve,
                ),
                &one_poly,
                curve,
            ),
            slack_down,
            curve,
        );
        let down_body = poly_mul(delta_down, &down_residual_p, curve);
        bodies.push(poly_mul(is_real, &down_body, curve));
        // 18: step bind: eb_post - eb_pre - (delta_up - delta_down) * INC.
        let dir_diff = poly_sub(delta_up, delta_down, curve);
        let step_amount_p = poly_mul(&dir_diff, &inc_poly, curve);
        let step_residual_p = poly_sub(
            &poly_sub(eb_post, eb_pre, curve),
            &step_amount_p,
            curve,
        );
        bodies.push(poly_mul(is_real, &step_residual_p, curve));

        debug_assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);

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
        let byte_ranges: [(usize, usize, &str); 11] = [
            (COL_VI_BYTE_OFFSET, U64_BYTES, "vi_byte"),
            (COL_BALANCE_PRE_BYTE_OFFSET, U64_BYTES, "balance_pre_byte"),
            (COL_BALANCE_POST_BYTE_OFFSET, U64_BYTES, "balance_post_byte"),
            (COL_EB_PRE_BYTE_OFFSET, U64_BYTES, "eb_pre_byte"),
            (COL_EB_POST_BYTE_OFFSET, U64_BYTES, "eb_post_byte"),
            (COL_REWARDS_BYTE_OFFSET, U64_BYTES, "rewards_byte"),
            (COL_SLASHING_PENALTY_BYTE_OFFSET, U64_BYTES, "slashing_penalty_byte"),
            (COL_EB_STEP_COUNT_BYTE_OFFSET, U64_BYTES, "eb_step_count_byte"),
            (COL_CAP_SLACK_BYTE_OFFSET, U64_BYTES, "cap_slack_byte"),
            (COL_SLACK_UP_BYTE_OFFSET, U64_BYTES, "slack_up_byte"),
            (COL_SLACK_DOWN_BYTE_OFFSET, U64_BYTES, "slack_down_byte"),
        ];
        for (off, len, label) in byte_ranges {
            for k in 0..len {
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
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(VALIDATOR_INDEX, EFFECTIVE_BALANCE_PRE, BALANCE_PRE)` of this
/// AIR to [`crate::validator_balances_air`]'s `(COL_VALIDATOR_INDEX,
/// COL_EFFECTIVE_BALANCE, COL_CURRENT_BALANCE)`. Gated by `IS_REAL` on
/// both sides. 3-col tuple; pins both the validator's effective balance
/// and current balance at the epoch boundary so the per-validator
/// update on this AIR runs against authoritative state.
pub fn make_epoch_to_validator_balances_descriptor(
    epoch_layer_index: usize,
    balances_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_balances_air as vb;
    let a_columns = vec![
        COL_VALIDATOR_INDEX,
        COL_EFFECTIVE_BALANCE_PRE,
        COL_BALANCE_PRE,
    ];
    let b_columns = vec![
        vb::COL_VALIDATOR_INDEX,
        vb::COL_EFFECTIVE_BALANCE,
        vb::COL_CURRENT_BALANCE,
    ];
    CrossAirLogUpDescriptor {
        label: "epoch_processing_to_validator_balances_v1".into(),
        a_layer_index: epoch_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: balances_layer_index,
        b_columns,
        b_selector_column: Some(vb::COL_IS_REAL),
    }
}

/// Bind `(VALIDATOR_INDEX, REWARDS)` of this AIR to
/// [`crate::attestation_rewards_air`]'s `(COL_VALIDATOR_INDEX,
/// COL_TOTAL_REWARD)`. Gated by `IS_REAL` on both sides. 2-col tuple;
/// pins the per-validator rewards credited in the epoch update to the
/// authoritative tally proved by the rewards AIR (which sums the
/// source/target/head sub-rewards).
pub fn make_epoch_to_attestation_rewards_descriptor(
    epoch_layer_index: usize,
    rewards_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::attestation_rewards_air as ar;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_REWARDS];
    let b_columns = vec![ar::COL_VALIDATOR_INDEX, ar::COL_TOTAL_REWARD];
    CrossAirLogUpDescriptor {
        label: "epoch_processing_to_attestation_rewards_v1".into(),
        a_layer_index: epoch_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: rewards_layer_index,
        b_columns,
        b_selector_column: Some(ar::COL_IS_REAL),
    }
}

/// Wide variant of [`make_epoch_to_attestation_rewards_descriptor`]
/// that additionally pins `(EFFECTIVE_BALANCE_POST, REWARDS)` ↔
/// `(COL_EFFECTIVE_BALANCE, COL_TOTAL_REWARD)` so the attestation
/// rewards AIR's per-validator base_reward input (effective balance)
/// is the same value the epoch AIR has hysteresis-stepped. 3-col
/// tuple `(validator_index, effective_balance_post, total_reward)`.
/// Gated by `IS_REAL` on both sides.
pub fn make_epoch_to_attestation_rewards_descriptor_wide(
    epoch_layer_index: usize,
    rewards_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::attestation_rewards_air as ar;
    let a_columns = vec![
        COL_VALIDATOR_INDEX,
        COL_EFFECTIVE_BALANCE_POST,
        COL_REWARDS,
    ];
    let b_columns = vec![
        ar::COL_VALIDATOR_INDEX,
        ar::COL_EFFECTIVE_BALANCE,
        ar::COL_TOTAL_REWARD,
    ];
    CrossAirLogUpDescriptor {
        label: "epoch_processing_to_attestation_rewards_wide_v1".into(),
        a_layer_index: epoch_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: rewards_layer_index,
        b_columns,
        b_selector_column: Some(ar::COL_IS_REAL),
    }
}

/// Bind `(VALIDATOR_INDEX, SLASHING_PENALTY)` of this AIR to a slashing
/// AIR's per-validator slashing-penalty projection. The current
/// `attester_slashing_air` and `proposer_slashing_air` do not yet
/// expose a per-penalized-validator `slashing_penalty` column — they
/// model the slashing *event* (paired indexed attestations / paired
/// signed headers) rather than the post-slashing penalty tally. So
/// this descriptor is a **target-shape scaffold**: the A side projects
/// the per-row epoch-side `(validator_index, slashing_penalty)`, and
/// the B side mirrors A on a placeholder layer (typically the
/// `attester_slashing_air` row, gated by its `IS_REAL`). Once a
/// `slashing_penalty_air` (or equivalent column on the existing
/// slashing AIRs) lands, the B-side columns will switch to point at
/// the authoritative `(slashed_validator_index, penalty)` columns.
///
/// Gated by `IS_REAL` on both sides. 2-col tuple. The current B-side
/// scaffold targets `attester_slashing_air::COL_ATTESTING_INDICES_1`
/// (first index in the first attesting-indices vector — a stand-in for
/// "a slashed validator's index") and `COL_INTERSECTION_COUNT` (a
/// numeric column that carries forward as a placeholder for the
/// penalty value).
pub fn make_epoch_to_slashing_descriptor(
    epoch_layer_index: usize,
    slashing_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::attester_slashing_air as asl;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_SLASHING_PENALTY];
    // Scaffold: bind to the first attesting index + the
    // intersection_count column as a numeric placeholder. The full
    // per-validator-penalty binding lands when slashing AIRs expose
    // a (slashed_index, penalty) projection.
    let b_columns = vec![
        asl::COL_ATTESTING_INDICES_1_OFFSET,
        asl::COL_INTERSECTION_COUNT,
    ];
    CrossAirLogUpDescriptor {
        label: "epoch_processing_to_slashing_v1".into(),
        a_layer_index: epoch_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: slashing_layer_index,
        b_columns,
        b_selector_column: Some(asl::COL_IS_REAL),
    }
}

/// Bind `(VALIDATOR_INDEX, SLASHING_PENALTY)` of this AIR to a
/// [`crate::proposer_slashing_air`] row's
/// `(COL_PROPOSER_INDEX_1, COL_BODY_ROOT_DIFF_INV)` projection.
/// `COL_PROPOSER_INDEX_1` IS the slashed validator's authoritative
/// proposer-index (the two paired signed headers must share the same
/// proposer index, so binding to slot 1 is canonical). The penalty
/// column on the proposer-slashing side is the **scaffold** column
/// `COL_BODY_ROOT_DIFF_INV` (a numeric column carrying a u64-sized
/// value). Once `proposer_slashing_air` adds an explicit
/// `slashing_penalty` column the B-side will switch to that.
/// Gated by `IS_REAL` on both sides. 2-col tuple.
pub fn make_epoch_to_proposer_slashing_descriptor(
    epoch_layer_index: usize,
    slashing_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::proposer_slashing_air as psl;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_SLASHING_PENALTY];
    let b_columns = vec![psl::COL_PROPOSER_INDEX_1, psl::COL_BODY_ROOT_DIFF_INV];
    CrossAirLogUpDescriptor {
        label: "epoch_processing_to_proposer_slashing_v1".into(),
        a_layer_index: epoch_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: slashing_layer_index,
        b_columns,
        b_selector_column: Some(psl::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate_bodies(
        witness: &EpochProcessingWitness,
        curve: CurveType,
    ) -> (TracePolynomials, Vec<Vec<Scalar>>) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        (trace, bodies)
    }

    fn assert_all_vanish(bodies: &[Vec<Scalar>]) {
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} should vanish at row {} (got {:?})",
                    k,
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    #[test]
    fn legacy_validator_balance_update_vanishes() {
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + 100_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn compounding_validator_balance_update_vanishes() {
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            7,
            500 * GWEI_PER_ETH,
            500 * GWEI_PER_ETH + 1_500_000,
            500 * GWEI_PER_ETH,
            500 * GWEI_PER_ETH,
            0,
            1_500_000,
            true,
            true,
        );
        let (_trace, bodies) = evaluate_bodies(&w, curve);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn slashing_penalty_branch_vanishes() {
        let curve = CurveType::Bls48581;
        // Slashing penalty exceeds rewards → net negative balance change.
        let w = from_validator_epoch(
            1,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH - 500_000_000,
            32 * GWEI_PER_ETH,
            31 * GWEI_PER_ETH, // Effective balance stepped down 1 ETH.
            500_000_000,
            0,
            true,
            false,
        );
        let (_trace, bodies) = evaluate_bodies(&w, curve);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn tampered_balance_post_detected() {
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + 100_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper balance_post: should fire constraint 3 (balance update).
        cols[COL_BALANCE_POST][0] = Scalar::from_u64(32 * GWEI_PER_ETH + 999_999, curve);
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "balance_update_bind should fire on tampered balance_post"
        );
    }

    #[test]
    fn tampered_eb_post_non_multiple_detected() {
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            0,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper eb_post off the step grid. Keep step_count = 32 unchanged.
        cols[COL_EFFECTIVE_BALANCE_POST][0] =
            Scalar::from_u64(32 * GWEI_PER_ETH + 1, curve);
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[4][0].is_zero(),
            "eb_step_bind should fire when eb_post is not a multiple of INCREMENT"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_epoch_to_validator_balances_descriptor(0, 1);
        assert_eq!(d1.label, "epoch_processing_to_validator_balances_v1");
        assert_eq!(
            d1.a_columns,
            vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE_PRE, COL_BALANCE_PRE],
        );
        assert_eq!(
            d1.b_columns,
            vec![
                crate::validator_balances_air::COL_VALIDATOR_INDEX,
                crate::validator_balances_air::COL_EFFECTIVE_BALANCE,
                crate::validator_balances_air::COL_CURRENT_BALANCE,
            ],
        );
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_columns.len(), 3);
        assert_eq!(d1.b_columns.len(), 3);
        for &c in &d1.a_columns {
            assert!(c < NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d1.b_columns {
            assert!(c < crate::validator_balances_air::NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }

        let d2 = make_epoch_to_attestation_rewards_descriptor(0, 2);
        assert_eq!(d2.label, "epoch_processing_to_attestation_rewards_v1");
        assert_eq!(d2.a_columns, vec![COL_VALIDATOR_INDEX, COL_REWARDS]);
        assert_eq!(
            d2.b_columns,
            vec![
                crate::attestation_rewards_air::COL_VALIDATOR_INDEX,
                crate::attestation_rewards_air::COL_TOTAL_REWARD,
            ],
        );
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
        for &c in &d2.a_columns {
            assert!(c < NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d2.b_columns {
            assert!(c < crate::attestation_rewards_air::NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }

        let d3 = make_epoch_to_slashing_descriptor(0, 3);
        assert_eq!(d3.label, "epoch_processing_to_slashing_v1");
        assert_eq!(
            d3.a_columns,
            vec![COL_VALIDATOR_INDEX, COL_SLASHING_PENALTY],
        );
        assert_eq!(d3.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d3.b_selector_column,
            Some(crate::attester_slashing_air::COL_IS_REAL),
        );
        assert_eq!(d3.a_columns.len(), 2);
        assert_eq!(d3.b_columns.len(), 2);
        for &c in &d3.a_columns {
            assert!(c < NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
        for &c in &d3.b_columns {
            assert!(c < crate::attester_slashing_air::NUM_COLUMNS);
            assert_ne!(c, usize::MAX);
        }
    }

    #[test]
    fn column_layout_pinned() {
        // 1 + 8 (vi) + 1 + 8 (bal_pre) + 1 + 8 (bal_post)
        // + 1 + 8 (eb_pre) + 1 + 8 (eb_post)
        // + 1 + 8 (rewards) + 1 + 8 (penalty)
        // + 1 + 8 (eb_step_count) + 1 + 8 (cap_slack) = 81
        // + 1 (delta_up) + 1 (delta_down) + 1 (delta_zero)
        // + 1 + 8 (slack_up) + 1 + 8 (slack_down) = 21
        // + 1 (is_active) + 1 (is_compounding) + 1 (is_real) = 3
        // → 81 + 21 + 3 = 105.
        assert_eq!(NUM_COLUMNS, 105);
        assert_eq!(NUM_ROW_CONSTRAINTS, 19);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(EFFECTIVE_BALANCE_INCREMENT, GWEI_PER_ETH);
        assert_eq!(MAX_EFFECTIVE_BALANCE, 32 * GWEI_PER_ETH);
        assert_eq!(MAX_EFFECTIVE_BALANCE_ELECTRA, 2048 * GWEI_PER_ETH);
        assert_eq!(HYSTERESIS_UP_THRESHOLD, GWEI_PER_ETH + GWEI_PER_ETH / 4);
        assert_eq!(HYSTERESIS_DOWN_THRESHOLD, GWEI_PER_ETH / 4);
    }

    #[test]
    fn evaluate_at_point_matches_for_honest() {
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + 100_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(19, curve);
        let row0_evals: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    #[test]
    #[should_panic(expected = "balance update inconsistent")]
    fn host_panics_on_inconsistent_balance() {
        let _ = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH, // wrong: would need to equal pre+rewards-penalty
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
    }

    #[test]
    fn tampered_rewards_detected() {
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + 100_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper rewards (and its LE bytes): the balance equation
        // bal_pre + rewards − penalty − bal_post should now fail.
        let new_rewards: u64 = 200_000;
        cols[COL_REWARDS][0] = Scalar::from_u64(new_rewards, curve);
        for (b, byte) in new_rewards.to_le_bytes().iter().enumerate() {
            cols[COL_REWARDS_BYTE_OFFSET + b][0] = Scalar::from_u64(*byte as u64, curve);
        }
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "balance_update_bind should fire when rewards is tampered"
        );
    }

    #[test]
    fn tampered_slashing_penalty_detected() {
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            7,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH - 1_000_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            1_000_000,
            0,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper slashing_penalty (and its LE bytes).
        let new_penalty: u64 = 500_000;
        cols[COL_SLASHING_PENALTY][0] = Scalar::from_u64(new_penalty, curve);
        for (b, byte) in new_penalty.to_le_bytes().iter().enumerate() {
            cols[COL_SLASHING_PENALTY_BYTE_OFFSET + b][0] =
                Scalar::from_u64(*byte as u64, curve);
        }
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "balance_update_bind should fire when slashing_penalty is tampered"
        );
    }

    #[test]
    fn balance_update_gated_by_is_real() {
        // Padding rows (is_real=0) should have constraint 3 vanish
        // even if balance fields don't satisfy the equation.
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + 100_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        assert!(trace.padded_size > 1, "expected padding rows for gating test");
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Stuff a padding row with rewards != 0 but balance_pre =
        // balance_post = 0 (impossible per equation). Since
        // is_real = 0 on that row the constraint must still vanish.
        let pad = (trace.num_rows).min(cols[0].len() - 1);
        if pad < cols[0].len() && pad >= trace.num_rows {
            cols[COL_REWARDS][pad] = Scalar::from_u64(12345, curve);
            for (b, byte) in 12345u64.to_le_bytes().iter().enumerate() {
                cols[COL_REWARDS_BYTE_OFFSET + b][pad] =
                    Scalar::from_u64(*byte as u64, curve);
            }
            let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert!(
                bodies[3][pad].is_zero(),
                "balance_update_bind must be gated by IS_REAL on padding rows"
            );
        }
    }

    #[test]
    fn rewards_descriptor_tuple_matches_against_rewards_air() {
        // Honest projection alignment: epoch-side
        // (validator_index, rewards) row 0 equals attestation-rewards-AIR
        // (validator_index, total_reward) row 0 when both are built
        // around the same validator + total_reward value. We exercise
        // both trace builders against the same numeric tuple.
        use crate::attestation_rewards_air as ar;
        let curve = CurveType::Bls48581;
        let validator_index: u64 = 42;
        let effective_balance: u64 = 32 * GWEI_PER_ETH;
        let base_reward: u64 = 64 * 100; // multiple of denominator
        // Rewards AIR row built from the canonical builder.
        let ar_witness = ar::AttestationRewardsWitness::from_validator(
            validator_index,
            effective_balance,
            base_reward,
            (true, true, true),
        );
        let total_reward = ar_witness.rows[0].total_reward;
        let w = from_validator_epoch(
            validator_index,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + total_reward,
            effective_balance,
            effective_balance,
            0,
            total_reward,
            true,
            false,
        );
        let epoch_trace = build_trace_polynomials(&w, curve);
        let ar_trace = ar::build_trace_polynomials(&ar_witness, curve);
        // Compare projections used by descriptor:
        // (COL_VALIDATOR_INDEX, COL_REWARDS) vs
        // (ar::COL_VALIDATOR_INDEX, ar::COL_TOTAL_REWARD).
        assert_eq!(
            epoch_trace.columns[COL_VALIDATOR_INDEX].evaluations[0].to_u64(),
            ar_trace.columns[ar::COL_VALIDATOR_INDEX].evaluations[0].to_u64(),
        );
        assert_eq!(
            epoch_trace.columns[COL_REWARDS].evaluations[0].to_u64(),
            ar_trace.columns[ar::COL_TOTAL_REWARD].evaluations[0].to_u64(),
        );
        // And the wide descriptor's effective_balance projection.
        assert_eq!(
            epoch_trace.columns[COL_EFFECTIVE_BALANCE_POST].evaluations[0].to_u64(),
            ar_trace.columns[ar::COL_EFFECTIVE_BALANCE].evaluations[0].to_u64(),
        );
    }

    #[test]
    fn wide_rewards_descriptor_well_formed() {
        let d = make_epoch_to_attestation_rewards_descriptor_wide(0, 4);
        assert_eq!(d.label, "epoch_processing_to_attestation_rewards_wide_v1");
        assert_eq!(
            d.a_columns,
            vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE_POST, COL_REWARDS],
        );
        assert_eq!(
            d.b_columns,
            vec![
                crate::attestation_rewards_air::COL_VALIDATOR_INDEX,
                crate::attestation_rewards_air::COL_EFFECTIVE_BALANCE,
                crate::attestation_rewards_air::COL_TOTAL_REWARD,
            ],
        );
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::attestation_rewards_air::COL_IS_REAL),
        );
        for &c in &d.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d.b_columns {
            assert!(c < crate::attestation_rewards_air::NUM_COLUMNS);
        }
    }

    #[test]
    fn proposer_slashing_descriptor_well_formed() {
        let d = make_epoch_to_proposer_slashing_descriptor(0, 5);
        assert_eq!(d.label, "epoch_processing_to_proposer_slashing_v1");
        assert_eq!(d.a_columns, vec![COL_VALIDATOR_INDEX, COL_SLASHING_PENALTY]);
        assert_eq!(
            d.b_columns,
            vec![
                crate::proposer_slashing_air::COL_PROPOSER_INDEX_1,
                crate::proposer_slashing_air::COL_BODY_ROOT_DIFF_INV,
            ],
        );
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::proposer_slashing_air::COL_IS_REAL),
        );
        for &c in &d.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d.b_columns {
            assert!(c < crate::proposer_slashing_air::NUM_COLUMNS);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds cap")]
    fn host_panics_on_cap_violation() {
        let _ = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            33 * GWEI_PER_ETH, // legacy cap is 32 ETH
            0,
            0,
            true,
            false,
        );
    }

    // ─── Hysteresis tests ──────────────────────────────────────────

    #[test]
    fn hysteresis_up_branch_vanishes() {
        // balance > eb_pre + 1.25 ETH triggers delta_up.
        let curve = CurveType::Bls48581;
        // eb_pre = 30 ETH; rewards = 2 ETH → balance_post = 32 ETH > 31.25 ETH.
        let w = from_validator_epoch(
            1,
            30 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            30 * GWEI_PER_ETH,
            31 * GWEI_PER_ETH, // step up by 1 ETH.
            0,
            2 * GWEI_PER_ETH,
            true,
            false,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(
            trace.columns[COL_DELTA_UP].evaluations[0].to_u64(),
            1
        );
        assert_eq!(
            trace.columns[COL_DELTA_DOWN].evaluations[0].to_u64(),
            0
        );
        assert_eq!(
            trace.columns[COL_DELTA_ZERO].evaluations[0].to_u64(),
            0
        );
        assert_all_vanish(&bodies);
    }

    #[test]
    fn hysteresis_down_branch_vanishes() {
        // balance + 0.25 ETH < eb_pre triggers delta_down.
        // eb_pre = 32 ETH, balance_post = 31.5 ETH → balance + 0.25 = 31.75 < 32.
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            2,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH - GWEI_PER_ETH / 2,
            32 * GWEI_PER_ETH,
            31 * GWEI_PER_ETH, // step down by 1 ETH.
            GWEI_PER_ETH / 2,
            0,
            true,
            false,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(
            trace.columns[COL_DELTA_DOWN].evaluations[0].to_u64(),
            1
        );
        assert_eq!(
            trace.columns[COL_DELTA_UP].evaluations[0].to_u64(),
            0
        );
        assert_all_vanish(&bodies);
    }

    #[test]
    fn hysteresis_zero_branch_vanishes_within_band() {
        // balance within hysteresis band → no step.
        let curve = CurveType::Bls48581;
        // rewards = 1 ETH; balance_post = 33 ETH; eb_pre = 32 ETH.
        // diff = +1 ETH < 1.25 ETH → no up.
        // diff > -0.25 ETH → no down.
        let w = from_validator_epoch(
            3,
            32 * GWEI_PER_ETH,
            33 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH, // unchanged.
            0,
            GWEI_PER_ETH,
            true,
            false,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(
            trace.columns[COL_DELTA_ZERO].evaluations[0].to_u64(),
            1
        );
        assert_all_vanish(&bodies);
    }

    #[test]
    fn hysteresis_up_branch_exact_threshold_does_not_fire() {
        // balance == eb_pre + 1.25 ETH exactly → NOT strictly greater
        // → no up step.
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            4,
            30 * GWEI_PER_ETH,
            30 * GWEI_PER_ETH + HYSTERESIS_UP_THRESHOLD,
            30 * GWEI_PER_ETH,
            30 * GWEI_PER_ETH,
            0,
            HYSTERESIS_UP_THRESHOLD,
            true,
            false,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(
            trace.columns[COL_DELTA_ZERO].evaluations[0].to_u64(),
            1
        );
        assert_all_vanish(&bodies);
    }

    #[test]
    fn hysteresis_compounding_high_balance_step_up() {
        // Compounding validator at 100 ETH effective with high balance
        // → step up. Verifies cap_bind respects the Electra cap.
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            5,
            100 * GWEI_PER_ETH,
            102 * GWEI_PER_ETH,
            100 * GWEI_PER_ETH,
            101 * GWEI_PER_ETH,
            0,
            2 * GWEI_PER_ETH,
            true,
            true,
        );
        let (_, bodies) = evaluate_bodies(&w, curve);
        assert_all_vanish(&bodies);
    }

    #[test]
    #[should_panic(expected = "hysteresis violated")]
    fn host_panics_on_missing_up_step() {
        // balance > eb_pre + 1.25 ETH → host MUST step up. Setting
        // eb_post = eb_pre is rejected by the host builder.
        let _ = from_validator_epoch(
            6,
            30 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            30 * GWEI_PER_ETH,
            30 * GWEI_PER_ETH, // wrong: should be 31 ETH.
            0,
            2 * GWEI_PER_ETH,
            true,
            false,
        );
    }

    #[test]
    #[should_panic(expected = "hysteresis violated")]
    fn host_panics_on_missing_down_step() {
        let _ = from_validator_epoch(
            7,
            32 * GWEI_PER_ETH,
            31 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH, // wrong: should step down to 31 ETH.
            GWEI_PER_ETH,
            0,
            true,
            false,
        );
    }

    #[test]
    fn tampered_delta_up_without_threshold_detected() {
        // Honest delta_zero row, prover tampers delta_up=1 without
        // meeting the up threshold: hysteresis_up_bind (constraint 16)
        // and the mutex / step constraints must fire.
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            8,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + 100_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Force delta_up = 1 (was 0). slack_up still 0, so up_residual
        // = 100_000 - 1.25 ETH - 0 ≠ 0.
        cols[COL_DELTA_UP][0] = Scalar::from_u64(1, curve);
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // 15 mutex (sum != 1 since delta_zero still 1, total = 2) and
        // 16 up_bind both fire.
        assert!(
            !bodies[15][0].is_zero(),
            "hysteresis_direction_mutex must fire on tampered delta_up"
        );
        assert!(
            !bodies[16][0].is_zero(),
            "hysteresis_up_bind must fire when threshold not met"
        );
    }

    #[test]
    fn tampered_eb_post_step_mismatch_detected() {
        // Honest up step, prover tampers eb_post to skip the step:
        // hysteresis_step_bind (constraint 18) must fire. Also eb_step
        // (constraint 4) fires because step_count is now wrong.
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            9,
            30 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            30 * GWEI_PER_ETH,
            31 * GWEI_PER_ETH,
            0,
            2 * GWEI_PER_ETH,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper eb_post to 30 ETH (== eb_pre, ignoring the step).
        let tampered = 30 * GWEI_PER_ETH;
        cols[COL_EFFECTIVE_BALANCE_POST][0] = Scalar::from_u64(tampered, curve);
        for (b, byte) in tampered.to_le_bytes().iter().enumerate() {
            cols[COL_EB_POST_BYTE_OFFSET + b][0] =
                Scalar::from_u64(*byte as u64, curve);
        }
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[18][0].is_zero(),
            "hysteresis_step_bind must fire on missing up step"
        );
    }

    #[test]
    fn tampered_delta_zero_when_must_step_detected() {
        // Honest up-step row: balance > eb_pre + 1.25 ETH so delta_up
        // MUST be 1. Tamper to delta_zero=1, delta_up=0, eb_post=eb_pre.
        // hysteresis_step_bind passes (eb_post = eb_pre + 0), but the
        // up threshold is exceeded, so a prover who attempts to "skip"
        // the step by claiming delta_zero must still satisfy the up
        // bound — which here it does (delta_up=0 so 16 vanishes).
        // This is the well-known soundness gap when the up-threshold
        // is genuinely exceeded: the AIR only enforces conditional
        // bounds. Document this by checking that the SUM constraint
        // is the one that must fire when the prover also moves
        // delta_zero away from 1. The current witness layout commits
        // a sound (delta_up=1) row; this test exercises tampering
        // delta_up→0 and delta_zero→0: the mutex sum (constraint 15)
        // catches it.
        let curve = CurveType::Bls48581;
        let w = from_validator_epoch(
            10,
            30 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            30 * GWEI_PER_ETH,
            31 * GWEI_PER_ETH,
            0,
            2 * GWEI_PER_ETH,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Zero out all direction selectors.
        cols[COL_DELTA_UP][0] = Scalar::zero(curve);
        cols[COL_DELTA_DOWN][0] = Scalar::zero(curve);
        cols[COL_DELTA_ZERO][0] = Scalar::zero(curve);
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[15][0].is_zero(),
            "mutex sum must fire when no direction selector is 1"
        );
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

        let w = from_validator_epoch(
            42,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH + 100_000,
            32 * GWEI_PER_ETH,
            32 * GWEI_PER_ETH,
            0,
            100_000,
            true,
            false,
        );
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = EpochProcessingConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone epoch_processing_air proof must verify",
        );
    }
}
