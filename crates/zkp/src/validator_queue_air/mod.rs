//! Validator activation/exit queue AIR.
//!
//! Proves a validator's queue epoch transitions follow the beacon-chain
//! spec rules:
//!
//!   * `activation_epoch == activation_eligibility_epoch + waiting_activation`
//!     where `waiting_activation` is a non-negative witnessed delay
//!     (the spec's per-epoch churn delay, simplified here to a single
//!     witnessed integer; the actual churn limit depends on
//!     `len(active_validators)` and is deferred).
//!   * `withdrawable_epoch == exit_epoch + MIN_VALIDATOR_WITHDRAWABILITY_DELAY`
//!     (constant `256` epochs post-Cancun).
//!   * `is_activated · (activation_epoch − current_epoch − waiting_activation) = 0`
//!     — when the validator has been activated this row, the witnessed
//!     waiting_activation absorbs the slack so the algebra closes; this
//!     enforces ordering consistency between `activation_epoch` and
//!     `current_epoch` for activated rows.
//!
//! ## Column layout
//!
//! Per-row witness:
//!   * `VALIDATOR_INDEX`              u64
//!   * `ACTIVATION_ELIGIBILITY_EPOCH` u64
//!   * `ACTIVATION_EPOCH`             u64
//!   * `EXIT_EPOCH`                   u64
//!   * `WITHDRAWABLE_EPOCH`           u64
//!   * `CURRENT_EPOCH`                u64
//!   * `WAITING_ACTIVATION`           u64 (= activation_epoch − activation_eligibility_epoch)
//!   * `WAITING_EXIT`                 u64 — for activated rows the
//!     host-side value is `current_epoch − activation_epoch` (epochs
//!     elapsed since activation, used in the `activated_ordering`
//!     constraint). For not-yet-activated rows it is 0. The name
//!     reflects its role as the "queue-wait" slack absorbed by the
//!     ordering constraint; it is range-checked non-negative.
//!   * `WITHDRAWABILITY_DELAY`        u64 (= 256, witnessed and pinned)
//!   * 6× 8-byte LE decompositions (range-checked) for the six primary
//!     u64 fields above (validator_index, eligibility, activation,
//!     exit, withdrawable, current).
//!   * 3× 8-byte LE decompositions for the three witnessed differences
//!     (range-checked only; their algebraic decomp identities are NOT
//!     part of the 6 advertised algebraic decomp constraints — they are
//!     kept as range-checked witness columns plus the linkage equations).
//!   * `IS_ELIGIBLE`, `IS_ACTIVATED`, `IS_EXITING`, `IS_REAL` (all binary).
//!
//! ## Algebraic constraints (13 row-local bodies)
//!
//!   0. `is_real_binary`     — `IS_REAL · (IS_REAL − 1) = 0`.
//!   1. `is_eligible_binary` — `IS_ELIGIBLE · (IS_ELIGIBLE − 1) = 0`.
//!   2. `is_activated_binary`— `IS_ACTIVATED · (IS_ACTIVATED − 1) = 0`.
//!   3. `is_exiting_binary`  — `IS_EXITING · (IS_EXITING − 1) = 0`.
//!   4. `validator_index_le_decomp`
//!   5. `eligibility_le_decomp`
//!   6. `activation_le_decomp`
//!   7. `exit_le_decomp`
//!   8. `withdrawable_le_decomp`
//!   9. `current_le_decomp`
//!   10. `activation_delta`  —
//!       `IS_REAL · (ACTIVATION_EPOCH − ACTIVATION_ELIGIBILITY_EPOCH
//!                   − WAITING_ACTIVATION) = 0`.
//!   11. `withdrawability_delay` —
//!       `IS_REAL · (WITHDRAWABLE_EPOCH − EXIT_EPOCH −
//!                   MIN_VALIDATOR_WITHDRAWABILITY_DELAY) = 0`.
//!   12. `activated_ordering` —
//!       `IS_ACTIVATED · (CURRENT_EPOCH − ACTIVATION_EPOCH −
//!                       WAITING_EXIT) = 0`.
//!       Combined with the per-byte range check on WAITING_EXIT this
//!       proves `current_epoch ≥ activation_epoch` on every activated
//!       row.
//!
//! ## Range checks
//!
//! Per-byte 8-bit range checks via `lookup_declarations` for every byte
//! of the 9 LE byte-decomposition columns (6 primary + 3 difference).
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   * **Churn limit.** The per-epoch activation/exit churn limit
//!     depends on the count of active validators; this AIR treats
//!     `waiting_activation` as a witnessed non-negative delay without
//!     binding it to any churn budget.
//!   * **Queue position.** The AIR does not prove the validator's
//!     queue position relative to other queued validators.
//!   * **`MIN_PER_EPOCH_CHURN_LIMIT`** vs. dynamic churn — the constant
//!     `MIN_VALIDATOR_WITHDRAWABILITY_DELAY` is pinned to `256`.
//!
//! ## Cross-AIR LogUp descriptors
//!
//!   * [`make_validator_queue_to_validator_htr_descriptor`] binds
//!     `(VALIDATOR_INDEX, AE_EPOCH, ACT_EPOCH, EXIT_EPOCH, WD_EPOCH)`
//!     against the per-validator HTR AIR's identical 5-tuple. This
//!     transitively pins the queue fields to the registry record.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// Minimum number of epochs between a validator's `exit_epoch` and the
/// point at which its balance becomes withdrawable. Ethereum mainnet
/// (post-Cancun) value is `256`.
pub const MIN_VALIDATOR_WITHDRAWABILITY_DELAY: u64 = 256;

pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_VALIDATOR_INDEX: usize = 0;
pub const COL_ACTIVATION_ELIGIBILITY_EPOCH: usize = COL_VALIDATOR_INDEX + 1;
pub const COL_ACTIVATION_EPOCH: usize = COL_ACTIVATION_ELIGIBILITY_EPOCH + 1;
pub const COL_EXIT_EPOCH: usize = COL_ACTIVATION_EPOCH + 1;
pub const COL_WITHDRAWABLE_EPOCH: usize = COL_EXIT_EPOCH + 1;
pub const COL_CURRENT_EPOCH: usize = COL_WITHDRAWABLE_EPOCH + 1;

pub const COL_WAITING_ACTIVATION: usize = COL_CURRENT_EPOCH + 1;
pub const COL_WAITING_EXIT: usize = COL_WAITING_ACTIVATION + 1;
pub const COL_WITHDRAWABILITY_DELAY: usize = COL_WAITING_EXIT + 1;

// Six primary 8-byte LE decompositions.
pub const COL_VI_BYTE_OFFSET: usize = COL_WITHDRAWABILITY_DELAY + 1;
pub const COL_AE_EPOCH_BYTE_OFFSET: usize = COL_VI_BYTE_OFFSET + U64_BYTES;
pub const COL_ACT_EPOCH_BYTE_OFFSET: usize = COL_AE_EPOCH_BYTE_OFFSET + U64_BYTES;
pub const COL_EXIT_EPOCH_BYTE_OFFSET: usize = COL_ACT_EPOCH_BYTE_OFFSET + U64_BYTES;
pub const COL_WD_EPOCH_BYTE_OFFSET: usize = COL_EXIT_EPOCH_BYTE_OFFSET + U64_BYTES;
pub const COL_CE_BYTE_OFFSET: usize = COL_WD_EPOCH_BYTE_OFFSET + U64_BYTES;

// Three difference-column LE decompositions (range-checked only).
pub const COL_WAITING_ACT_BYTE_OFFSET: usize = COL_CE_BYTE_OFFSET + U64_BYTES;
pub const COL_WAITING_EXIT_BYTE_OFFSET: usize = COL_WAITING_ACT_BYTE_OFFSET + U64_BYTES;
pub const COL_WD_DELAY_BYTE_OFFSET: usize = COL_WAITING_EXIT_BYTE_OFFSET + U64_BYTES;

// Selectors.
pub const COL_IS_ELIGIBLE: usize = COL_WD_DELAY_BYTE_OFFSET + U64_BYTES;
pub const COL_IS_ACTIVATED: usize = COL_IS_ELIGIBLE + 1;
pub const COL_IS_EXITING: usize = COL_IS_ACTIVATED + 1;
pub const COL_IS_REAL: usize = COL_IS_EXITING + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 13;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct ValidatorQueueRow {
    pub validator_index: u64,
    pub activation_eligibility_epoch: u64,
    pub activation_epoch: u64,
    pub exit_epoch: u64,
    pub withdrawable_epoch: u64,
    pub current_epoch: u64,
    pub waiting_activation: u64,
    pub waiting_exit: u64,
    pub withdrawability_delay: u64,
    pub is_eligible: bool,
    pub is_activated: bool,
    pub is_exiting: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ValidatorQueueWitness {
    pub rows: Vec<ValidatorQueueRow>,
}

impl ValidatorQueueWitness {
    /// Build a witness row for one validator's queue state.
    ///
    /// Parameters:
    ///   * `validator_index`        — registry index.
    ///   * `activation_eligibility` — `activation_eligibility_epoch`.
    ///   * `activation`             — `activation_epoch`.
    ///   * `exit`                   — `exit_epoch`.
    ///   * `current_epoch`          — beacon head epoch.
    ///
    /// The withdrawable epoch is derived as
    /// `exit + MIN_VALIDATOR_WITHDRAWABILITY_DELAY`. Selector flags are
    /// inferred from the supplied epochs vs. `current_epoch`. Panics if
    /// the supplied `activation < activation_eligibility` or if
    /// `exit + 256` overflows `u64`.
    pub fn from_validator(
        validator_index: u64,
        activation_eligibility: u64,
        activation: u64,
        exit: u64,
        current_epoch: u64,
    ) -> Self {
        let waiting_activation = activation
            .checked_sub(activation_eligibility)
            .expect("activation_epoch < activation_eligibility_epoch");
        let withdrawable = exit
            .checked_add(MIN_VALIDATOR_WITHDRAWABILITY_DELAY)
            .expect("exit_epoch + 256 overflowed u64");
        let is_eligible = activation_eligibility <= current_epoch;
        let is_activated = activation <= current_epoch;
        let is_exiting = exit != u64::MAX && exit <= current_epoch;

        // waiting_exit is reused for the `activated_ordering`
        // constraint: on activated rows it equals
        // `current_epoch − activation_epoch` (the epochs elapsed
        // since activation). On not-yet-activated rows it is 0 so the
        // selector-gated body vanishes trivially.
        let waiting_exit = if is_activated {
            current_epoch
                .checked_sub(activation)
                .expect("activated row: current < activation")
        } else {
            0
        };

        Self {
            rows: vec![ValidatorQueueRow {
                validator_index,
                activation_eligibility_epoch: activation_eligibility,
                activation_epoch: activation,
                exit_epoch: exit,
                withdrawable_epoch: withdrawable,
                current_epoch,
                waiting_activation,
                waiting_exit,
                withdrawability_delay: MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
                is_eligible,
                is_activated,
                is_exiting,
            }],
        }
    }

    pub fn from_rows(rows: Vec<ValidatorQueueRow>) -> Self {
        Self { rows }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

/// `target_value − Σ_b col[byte_off + b] · 2^(8b)`.
fn eval_le_decomp(
    target_value: &Scalar,
    byte_off: usize,
    col_evals: &[Scalar],
) -> Scalar {
    let curve = target_value.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[byte_off + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target_value.sub(&sum)
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
    witness: &ValidatorQueueWitness,
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
        columns[COL_ACTIVATION_ELIGIBILITY_EPOCH][i] =
            Scalar::from_u64(row.activation_eligibility_epoch, curve);
        columns[COL_ACTIVATION_EPOCH][i] = Scalar::from_u64(row.activation_epoch, curve);
        columns[COL_EXIT_EPOCH][i] = Scalar::from_u64(row.exit_epoch, curve);
        columns[COL_WITHDRAWABLE_EPOCH][i] =
            Scalar::from_u64(row.withdrawable_epoch, curve);
        columns[COL_CURRENT_EPOCH][i] = Scalar::from_u64(row.current_epoch, curve);

        columns[COL_WAITING_ACTIVATION][i] =
            Scalar::from_u64(row.waiting_activation, curve);
        columns[COL_WAITING_EXIT][i] = Scalar::from_u64(row.waiting_exit, curve);
        columns[COL_WITHDRAWABILITY_DELAY][i] =
            Scalar::from_u64(row.withdrawability_delay, curve);

        let byte_targets: [(u64, usize); 9] = [
            (row.validator_index, COL_VI_BYTE_OFFSET),
            (row.activation_eligibility_epoch, COL_AE_EPOCH_BYTE_OFFSET),
            (row.activation_epoch, COL_ACT_EPOCH_BYTE_OFFSET),
            (row.exit_epoch, COL_EXIT_EPOCH_BYTE_OFFSET),
            (row.withdrawable_epoch, COL_WD_EPOCH_BYTE_OFFSET),
            (row.current_epoch, COL_CE_BYTE_OFFSET),
            (row.waiting_activation, COL_WAITING_ACT_BYTE_OFFSET),
            (row.waiting_exit, COL_WAITING_EXIT_BYTE_OFFSET),
            (row.withdrawability_delay, COL_WD_DELAY_BYTE_OFFSET),
        ];
        for (val, off) in byte_targets {
            let bytes = val.to_le_bytes();
            for b in 0..U64_BYTES {
                columns[off + b][i] = Scalar::from_u64(bytes[b] as u64, curve);
            }
        }

        columns[COL_IS_ELIGIBLE][i] =
            if row.is_eligible { one.clone() } else { zero.clone() };
        columns[COL_IS_ACTIVATED][i] =
            if row.is_activated { one.clone() } else { zero.clone() };
        columns[COL_IS_EXITING][i] =
            if row.is_exiting { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = one.clone();
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

pub struct ValidatorQueueConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ValidatorQueueConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for ValidatorQueueConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_eligible_binary".into(),
            "is_activated_binary".into(),
            "is_exiting_binary".into(),
            "validator_index_le_decomp".into(),
            "eligibility_le_decomp".into(),
            "activation_le_decomp".into(),
            "exit_le_decomp".into(),
            "withdrawable_le_decomp".into(),
            "current_le_decomp".into(),
            "activation_delta".into(),
            "withdrawability_delay".into(),
            "activated_ordering".into(),
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
        let withdrawability_delay = Scalar::from_u64(
            MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
            curve,
        );
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_eligible = &row_evals[COL_IS_ELIGIBLE];
            let is_activated = &row_evals[COL_IS_ACTIVATED];
            let is_exiting = &row_evals[COL_IS_EXITING];

            // 0..3: binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_eligible.mul(&is_eligible.sub(&one));
            bodies[2][row] = is_activated.mul(&is_activated.sub(&one));
            bodies[3][row] = is_exiting.mul(&is_exiting.sub(&one));

            // 4..9: LE decomps.
            bodies[4][row] = eval_le_decomp(
                &row_evals[COL_VALIDATOR_INDEX],
                COL_VI_BYTE_OFFSET,
                &row_evals,
            );
            bodies[5][row] = eval_le_decomp(
                &row_evals[COL_ACTIVATION_ELIGIBILITY_EPOCH],
                COL_AE_EPOCH_BYTE_OFFSET,
                &row_evals,
            );
            bodies[6][row] = eval_le_decomp(
                &row_evals[COL_ACTIVATION_EPOCH],
                COL_ACT_EPOCH_BYTE_OFFSET,
                &row_evals,
            );
            bodies[7][row] = eval_le_decomp(
                &row_evals[COL_EXIT_EPOCH],
                COL_EXIT_EPOCH_BYTE_OFFSET,
                &row_evals,
            );
            bodies[8][row] = eval_le_decomp(
                &row_evals[COL_WITHDRAWABLE_EPOCH],
                COL_WD_EPOCH_BYTE_OFFSET,
                &row_evals,
            );
            bodies[9][row] = eval_le_decomp(
                &row_evals[COL_CURRENT_EPOCH],
                COL_CE_BYTE_OFFSET,
                &row_evals,
            );

            // 10: activation_delta
            //   is_real · (activation − eligibility − waiting_activation).
            let act = &row_evals[COL_ACTIVATION_EPOCH];
            let elig = &row_evals[COL_ACTIVATION_ELIGIBILITY_EPOCH];
            let waiting_act = &row_evals[COL_WAITING_ACTIVATION];
            bodies[10][row] = is_real.mul(&act.sub(elig).sub(waiting_act));

            // 11: withdrawability_delay
            //   is_real · (withdrawable − exit − 256).
            let wd = &row_evals[COL_WITHDRAWABLE_EPOCH];
            let exit = &row_evals[COL_EXIT_EPOCH];
            bodies[11][row] = is_real.mul(&wd.sub(exit).sub(&withdrawability_delay));

            // 12: activated_ordering
            //   is_activated · (current − activation − waiting_exit).
            let ce = &row_evals[COL_CURRENT_EPOCH];
            let waiting_exit = &row_evals[COL_WAITING_EXIT];
            bodies[12][row] = is_activated.mul(&ce.sub(act).sub(waiting_exit));
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let withdrawability_delay = Scalar::from_u64(
            MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
            curve,
        );
        let is_real = &col_evals[COL_IS_REAL];
        let is_eligible = &col_evals[COL_IS_ELIGIBLE];
        let is_activated = &col_evals[COL_IS_ACTIVATED];
        let is_exiting = &col_evals[COL_IS_EXITING];

        let act = &col_evals[COL_ACTIVATION_EPOCH];
        let elig = &col_evals[COL_ACTIVATION_ELIGIBILITY_EPOCH];
        let wd = &col_evals[COL_WITHDRAWABLE_EPOCH];
        let exit = &col_evals[COL_EXIT_EPOCH];
        let ce = &col_evals[COL_CURRENT_EPOCH];
        let waiting_act = &col_evals[COL_WAITING_ACTIVATION];
        let waiting_exit_v = &col_evals[COL_WAITING_EXIT];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_eligible.mul(&is_eligible.sub(&one)),
            is_activated.mul(&is_activated.sub(&one)),
            is_exiting.mul(&is_exiting.sub(&one)),
            eval_le_decomp(&col_evals[COL_VALIDATOR_INDEX], COL_VI_BYTE_OFFSET, col_evals),
            eval_le_decomp(
                &col_evals[COL_ACTIVATION_ELIGIBILITY_EPOCH],
                COL_AE_EPOCH_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_ACTIVATION_EPOCH],
                COL_ACT_EPOCH_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_EXIT_EPOCH],
                COL_EXIT_EPOCH_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_WITHDRAWABLE_EPOCH],
                COL_WD_EPOCH_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_CURRENT_EPOCH],
                COL_CE_BYTE_OFFSET,
                col_evals,
            ),
            is_real.mul(&act.sub(elig).sub(waiting_act)),
            is_real.mul(&wd.sub(exit).sub(&withdrawability_delay)),
            is_activated.mul(&ce.sub(act).sub(waiting_exit_v)),
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
        let neg_delay = Scalar::from_u64(MIN_VALIDATOR_WITHDRAWABILITY_DELAY, curve);

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_eligible = &col_coeffs[COL_IS_ELIGIBLE];
        let is_activated = &col_coeffs[COL_IS_ACTIVATED];
        let is_exiting = &col_coeffs[COL_IS_EXITING];

        let binary = |sel: &Vec<Scalar>| -> Vec<Scalar> {
            let m1 = poly_sub(sel, &one_poly, curve);
            poly_mul(sel, &m1, curve)
        };

        let is_real_binary = binary(is_real);
        let is_eligible_binary = binary(is_eligible);
        let is_activated_binary = binary(is_activated);
        let is_exiting_binary = binary(is_exiting);

        let vi_decomp = build_le_decomp_poly(
            &col_coeffs[COL_VALIDATOR_INDEX],
            COL_VI_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let elig_decomp = build_le_decomp_poly(
            &col_coeffs[COL_ACTIVATION_ELIGIBILITY_EPOCH],
            COL_AE_EPOCH_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let act_decomp = build_le_decomp_poly(
            &col_coeffs[COL_ACTIVATION_EPOCH],
            COL_ACT_EPOCH_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let exit_decomp = build_le_decomp_poly(
            &col_coeffs[COL_EXIT_EPOCH],
            COL_EXIT_EPOCH_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let wd_decomp = build_le_decomp_poly(
            &col_coeffs[COL_WITHDRAWABLE_EPOCH],
            COL_WD_EPOCH_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let ce_decomp = build_le_decomp_poly(
            &col_coeffs[COL_CURRENT_EPOCH],
            COL_CE_BYTE_OFFSET,
            col_coeffs,
            curve,
        );

        // activation_delta: is_real · (act − elig − waiting_act).
        let act_p = &col_coeffs[COL_ACTIVATION_EPOCH];
        let elig_p = &col_coeffs[COL_ACTIVATION_ELIGIBILITY_EPOCH];
        let waiting_act_p = &col_coeffs[COL_WAITING_ACTIVATION];
        let diff_act = poly_sub(
            &poly_sub(act_p, elig_p, curve),
            waiting_act_p,
            curve,
        );
        let activation_delta_body = poly_mul(is_real, &diff_act, curve);

        // withdrawability_delay: is_real · (wd − exit − 256).
        let wd_p = &col_coeffs[COL_WITHDRAWABLE_EPOCH];
        let exit_p = &col_coeffs[COL_EXIT_EPOCH];
        let delay_poly = vec![neg_delay.clone()];
        let diff_wd = poly_sub(
            &poly_sub(wd_p, exit_p, curve),
            &delay_poly,
            curve,
        );
        let withdrawability_body = poly_mul(is_real, &diff_wd, curve);

        // activated_ordering: is_activated · (ce − act − waiting_exit).
        let ce_p = &col_coeffs[COL_CURRENT_EPOCH];
        let waiting_exit_p = &col_coeffs[COL_WAITING_EXIT];
        let diff_ord = poly_sub(
            &poly_sub(ce_p, act_p, curve),
            waiting_exit_p,
            curve,
        );
        let activated_ordering_body = poly_mul(is_activated, &diff_ord, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_eligible_binary,
            is_activated_binary,
            is_exiting_binary,
            vi_decomp,
            elig_decomp,
            act_decomp,
            exit_decomp,
            wd_decomp,
            ce_decomp,
            activation_delta_body,
            withdrawability_body,
            activated_ordering_body,
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

        let u64_ranges: [(usize, &str); 9] = [
            (COL_VI_BYTE_OFFSET, "vi_byte"),
            (COL_AE_EPOCH_BYTE_OFFSET, "ae_epoch_byte"),
            (COL_ACT_EPOCH_BYTE_OFFSET, "act_epoch_byte"),
            (COL_EXIT_EPOCH_BYTE_OFFSET, "exit_epoch_byte"),
            (COL_WD_EPOCH_BYTE_OFFSET, "wd_epoch_byte"),
            (COL_CE_BYTE_OFFSET, "ce_byte"),
            (COL_WAITING_ACT_BYTE_OFFSET, "waiting_act_byte"),
            (COL_WAITING_EXIT_BYTE_OFFSET, "waiting_exit_byte"),
            (COL_WD_DELAY_BYTE_OFFSET, "wd_delay_byte"),
        ];
        for (off, label) in u64_ranges {
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

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(VALIDATOR_INDEX, ACTIVATION_ELIGIBILITY_EPOCH,
/// ACTIVATION_EPOCH, EXIT_EPOCH, WITHDRAWABLE_EPOCH)` of this AIR
/// against the per-validator HTR AIR ([`crate::validator_htr_air`]),
/// which carries the same 5-tuple in its registry record.
///
/// Combined with the validator-registry inclusion AIR, this
/// transitively pins the queue's epoch fields to the registry root, so
/// any algebraic property proven here (activation delta,
/// withdrawability delay, ordering) is proven against the canonical
/// beacon-chain record.
pub fn make_validator_queue_to_validator_htr_descriptor(
    queue_layer_index: usize,
    validator_htr_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_htr_air as vh;

    let a_columns: Vec<usize> = vec![
        COL_VALIDATOR_INDEX,
        COL_ACTIVATION_ELIGIBILITY_EPOCH,
        COL_ACTIVATION_EPOCH,
        COL_EXIT_EPOCH,
        COL_WITHDRAWABLE_EPOCH,
    ];

    let b_columns: Vec<usize> = vec![
        vh::COL_VALIDATOR_INDEX,
        vh::COL_AE_EPOCH,
        vh::COL_ACT_EPOCH,
        vh::COL_EXIT_EPOCH,
        vh::COL_WD_EPOCH,
    ];

    CrossAirLogUpDescriptor {
        label: "validator_queue_to_validator_htr_v1".into(),
        a_layer_index: queue_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_htr_layer_index,
        b_columns,
        b_selector_column: Some(vh::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard activated validator: eligibility=100, activation=105
    /// (waiting=5), exit=u64::MAX (not yet exiting), current=200.
    fn activated_witness() -> ValidatorQueueWitness {
        // Use a non-FAR-FUTURE exit so the withdrawable arithmetic
        // doesn't overflow. For a not-yet-exiting validator we pick a
        // sentinel like current_epoch + 1_000_000 as a placeholder.
        ValidatorQueueWitness::from_validator(
            1234,           // validator_index
            100,            // activation_eligibility_epoch
            105,            // activation_epoch (waiting = 5)
            1_000_000,      // exit_epoch (placeholder; not yet exiting)
            200,            // current_epoch
        )
    }

    /// Exiting validator: eligibility=100, activation=200,
    /// exit=10_000, withdrawable=10_256, current=10_500.
    fn exiting_witness() -> ValidatorQueueWitness {
        ValidatorQueueWitness::from_validator(
            5678,
            100,
            200,
            10_000,
            10_500,
        )
    }

    #[test]
    fn standard_activated_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let w = activated_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);

        let cs = ValidatorQueueConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);

        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "activated constraint {} should vanish at row {} (got {:?})",
                    k,
                    row,
                    v,
                );
            }
        }
    }

    #[test]
    fn exiting_validator_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let w = exiting_witness();
        let trace = build_trace_polynomials(&w, curve);

        let cs = ValidatorQueueConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "exiting constraint {} should vanish at row {} (got {:?})",
                    k,
                    row,
                    v,
                );
            }
        }
        // Sanity: the row should be flagged exiting AND activated.
        let row = &w.rows[0];
        assert!(row.is_activated);
        assert!(row.is_exiting);
        assert_eq!(row.withdrawable_epoch, row.exit_epoch + 256);
    }

    #[test]
    fn withdrawability_delay_enforced() {
        let curve = CurveType::Bls48581;
        // Tamper: withdrawable = exit + 100 instead of 256.
        let row = ValidatorQueueRow {
            validator_index: 1,
            activation_eligibility_epoch: 100,
            activation_epoch: 110,
            exit_epoch: 5_000,
            withdrawable_epoch: 5_100, // WRONG: should be 5_256.
            current_epoch: 5_500,
            waiting_activation: 10,
            waiting_exit: 0,
            withdrawability_delay: MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
            is_eligible: true,
            is_activated: true,
            is_exiting: true,
        };
        let w = ValidatorQueueWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorQueueConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 11 = withdrawability_delay; should be non-zero.
        assert!(
            !bodies[11][0].is_zero(),
            "withdrawability_delay should fire on bad delta",
        );
        // Constraint 8 = withdrawable_le_decomp; trace builder
        // populated bytes from withdrawable_epoch=5100, so still 0.
        assert!(bodies[8][0].is_zero());
    }

    #[test]
    fn tampered_activation_delta_detected() {
        let curve = CurveType::Bls48581;
        // Tamper waiting_activation to claim a different delta from
        // (activation - eligibility).
        let row = ValidatorQueueRow {
            validator_index: 7,
            activation_eligibility_epoch: 100,
            activation_epoch: 110,
            exit_epoch: 1_000_000,
            withdrawable_epoch: 1_000_000 + 256,
            current_epoch: 500,
            waiting_activation: 7, // WRONG: real delta is 10.
            waiting_exit: 999_500,
            withdrawability_delay: MIN_VALIDATOR_WITHDRAWABILITY_DELAY,
            is_eligible: true,
            is_activated: true,
            is_exiting: false,
        };
        let w = ValidatorQueueWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorQueueConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 10 = activation_delta; must fire.
        assert!(
            !bodies[10][0].is_zero(),
            "activation_delta should fire on tampered waiting_activation",
        );
        // Constraint 12 = activated_ordering; also keyed on
        // waiting_activation, must fire too.
        assert!(
            !bodies[12][0].is_zero(),
            "activated_ordering should fire on tampered waiting_activation",
        );
    }

    #[test]
    fn validator_htr_descriptor_well_formed() {
        let desc = make_validator_queue_to_validator_htr_descriptor(0, 1);
        assert_eq!(desc.label, "validator_queue_to_validator_htr_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 5);
        assert_eq!(desc.b_columns.len(), 5);
        assert_eq!(desc.a_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(desc.a_columns[1], COL_ACTIVATION_ELIGIBILITY_EPOCH);
        assert_eq!(desc.a_columns[2], COL_ACTIVATION_EPOCH);
        assert_eq!(desc.a_columns[3], COL_EXIT_EPOCH);
        assert_eq!(desc.a_columns[4], COL_WITHDRAWABLE_EPOCH);
        assert_eq!(
            desc.b_columns[0],
            crate::validator_htr_air::COL_VALIDATOR_INDEX,
        );
        assert_eq!(desc.b_columns[1], crate::validator_htr_air::COL_AE_EPOCH);
        assert_eq!(desc.b_columns[2], crate::validator_htr_air::COL_ACT_EPOCH);
        assert_eq!(desc.b_columns[3], crate::validator_htr_air::COL_EXIT_EPOCH);
        assert_eq!(desc.b_columns[4], crate::validator_htr_air::COL_WD_EPOCH);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::validator_htr_air::COL_IS_REAL),
        );
    }

    #[test]
    fn byte_range_coverage_complete() {
        let cs = ValidatorQueueConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 9 u64 fields × 8 bytes = 72 declarations, all 8-bit.
        assert_eq!(req.declarations.len(), 9 * U64_BYTES);
        for (decl, _) in &req.declarations {
            assert_eq!(decl.max_bits, 8);
            assert!(decl.column_index >= COL_VI_BYTE_OFFSET);
            assert!(decl.column_index < COL_IS_ELIGIBLE);
        }
        assert_eq!(req.tables.len(), 1);
    }

    #[test]
    fn evaluate_at_point_matches_domain() {
        // Verify the at-point evaluator agrees with the on-domain
        // evaluator at row 0: both should produce 0 for an honest row
        // under any random α.
        let curve = CurveType::Bls48581;
        let w = activated_witness();
        let trace = build_trace_polynomials(&w, curve);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let cs = ValidatorQueueConstraintSystem::new(trace.num_rows);

        let alpha = Scalar::from_u64(1_234_567, curve);
        let row_evals: Vec<Scalar> =
            col_refs.iter().map(|c| c[0].clone()).collect();
        let at_point = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(
            at_point.is_zero(),
            "honest row should give zero at-point body acc",
        );
    }

    #[test]
    fn column_layout_sanity() {
        // Sanity-check the column offsets line up with the witness
        // builder.
        assert_eq!(COL_VALIDATOR_INDEX, 0);
        assert_eq!(COL_IS_REAL + 1, NUM_COLUMNS);
        assert_eq!(COL_AE_EPOCH_BYTE_OFFSET, COL_VI_BYTE_OFFSET + U64_BYTES);
        assert_eq!(COL_WD_DELAY_BYTE_OFFSET + U64_BYTES, COL_IS_ELIGIBLE);
        // 6 primary u64 + 3 difference u64 + 9*8 bytes + 4 selectors.
        assert_eq!(NUM_COLUMNS, 6 + 3 + 9 * U64_BYTES + 4);
    }
}
