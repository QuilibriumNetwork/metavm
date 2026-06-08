//! Stake-weighted finality AIR — `VmConstraintSystem` wiring.
//!
//! Proves the two arithmetic checks at the heart of the Gasper /
//! Casper-FFG decision rule:
//!
//! 1. **Stake summation** — for `committee_size` validators, each row
//!    carries `(effective_balance, attesting_bit)` and contributes
//!    `effective_balance · attesting_bit` to a running total.
//!    A cross-row argument forces the running total at the threshold row
//!    to equal `Σ_i effective_balance_i · attesting_bit_i`.
//! 2. **≥ 2/3 threshold** — on a single threshold row,
//!    `3 · running_total = 2 · total_active_balance + slack` with a
//!    64-bit range check on `slack`. Range-checked non-negativity of the
//!    slack column is the algebraic encoding of the inequality.
//!
//! The committed `attestation_data_root` and `finalized_root` are 32-byte
//! public hints carried by the [`FinalityWitness`]. The cross-AIR linkage
//! at the [`crate::layer_chain::LayerChainProof`] level is what binds them
//! to adjacent layers; this AIR exposes them only via the witness so the
//! caller can pin them in the `Finality` claim's public values.
//!
//! # Soundness scope (documented gaps)
//!
//! This AIR proves: "given the committed `effective_balance` and
//! `attesting_bit` columns, the running sum is correct and clears the
//! 2/3 threshold." It **does not** prove:
//!
//! - That those `effective_balance` values were drawn from the canonical
//!   beacon-chain validator registry (delegated to the SSZ + cross-AIR
//!   linkage stack).
//! - That `attesting_bit_i = 1` corresponds to a valid BLS signature from
//!   validator `i` (delegated to the BLS aggregate-signature AIR /
//!   `LayerProofKind::BlsSig` layer).
//! - That `total_active_balance_gwei` matches the actual active set
//!   (delegated to the SSZ merkleization of the registry).
//! - The full Casper-FFG state machine (justification bits, finalization
//!   rules 2a/2b/3a/3b). The Layer 4 envelope's `attestation_data_root` →
//!   `finalized_root` hint encodes the result; checking the FFG transition
//!   rules in-circuit is a strictly larger AIR and out of scope.
//!
//! # Column layout
//!
//! 8 columns. Validator rows live at indices `0..committee_size`; the
//! single threshold row is at index `committee_size`; rows beyond that
//! are padding (power-of-two extension).
//!
//!   0. `EFFECTIVE_BALANCE` — `Gwei` (u64), only meaningful on validator
//!      rows.
//!   1. `ATTESTING_BIT` — `{0, 1}`, only meaningful on validator rows.
//!   2. `ROW_CONTRIBUTION` — `effective_balance · attesting_bit`. Zero on
//!      non-validator rows.
//!   3. `RUNNING_TOTAL` — sum BEFORE the row's contribution, by the
//!      EXCLUSIVE convention. So `RUNNING_TOTAL[0] = 0` and
//!      `RUNNING_TOTAL[committee_size] = Σ_{i=0..committee_size-1}
//!      ROW_CONTRIBUTION[i]`. Zero on padding rows beyond the threshold
//!      row (so the cross-row chain can re-bind `RUNNING_TOTAL[0] = 0`
//!      via wrap-around).
//!   4. `SEL_VALIDATOR` — `1` on rows `0..committee_size`, `0` elsewhere.
//!   5. `SEL_THRESHOLD` — `1` on row `committee_size`, `0` elsewhere.
//!   6. `TOTAL_ACTIVE` — total active balance; constant on the threshold
//!      row, zero elsewhere.
//!   7. `SLACK` — `3 · RUNNING_TOTAL - 2 · TOTAL_ACTIVE` on the threshold
//!      row (a 64-bit non-negative value), zero elsewhere.
//!
//! # Constraints (6 row-local + 2 shifted = 8 total)
//!
//! Row-local (`alpha^0..alpha^5`):
//!   0. `bit_binary`       — `SEL_VALIDATOR · ATTESTING_BIT · (ATTESTING_BIT − 1) = 0`.
//!   1. `contribution_def` — `SEL_VALIDATOR · (ROW_CONTRIBUTION − EFFECTIVE_BALANCE · ATTESTING_BIT) = 0`.
//!   2. `sel_validator_binary` — `SEL_VALIDATOR · (SEL_VALIDATOR − 1) = 0`.
//!   3. `sel_threshold_binary` — `SEL_THRESHOLD · (SEL_THRESHOLD − 1) = 0`.
//!   4. `threshold_check`  — `SEL_THRESHOLD · (3·RUNNING_TOTAL − 2·TOTAL_ACTIVE − SLACK) = 0`.
//!   5. `row_contribution_zero_on_non_validator` —
//!      `(1 − SEL_VALIDATOR) · ROW_CONTRIBUTION = 0`. Pins `ROW_CONTRIBUTION = 0`
//!      on threshold + padding rows so the running-sum chain through padding
//!      can't accumulate ghost values.
//!
//! Shifted (cross-row, `alpha^6..alpha^7`):
//!   6. `running_sum_step` —
//!      `RUNNING_TOTAL(ω·X) − RUNNING_TOTAL(X) − ROW_CONTRIBUTION(X) = 0`,
//!      excluded only at row `committee_size` (threshold → first padding,
//!      where `RUNNING_TOTAL` drops from the total back to zero).
//!   7. `running_total_zero_post_threshold` —
//!      `SEL_THRESHOLD(X) · RUNNING_TOTAL(ω·X) = 0`. On row `committee_size`
//!      (where `SEL_THRESHOLD = 1`) forces `RUNNING_TOTAL[committee_size+1] = 0`;
//!      on every other row the body is trivially zero. Combined with
//!      constraint 5 above (RC = 0 on non-validator rows) this propagates
//!      `RUNNING_TOTAL = 0` through padding, and the wrap-around chain
//!      transition then forces `RUNNING_TOTAL[0] = 0` as the AIR-internal
//!      boundary check.
//!
//! Together, constraints 5 + 7 close a soundness vulnerability the
//! original 5+1 set had: without them, `RUNNING_TOTAL[0]` was unconstrained
//! (the doc claim that the wrap-around alone re-binds `RUNNING_TOTAL[0] = 0`
//! relied on `RUNNING_TOTAL[n−1] = 0` and `ROW_CONTRIBUTION[n−1] = 0`,
//! which the prover can violate). A malicious prover could inflate
//! `RUNNING_TOTAL[committee_size]` by setting `RUNNING_TOTAL[0] ≠ 0` and
//! pass the threshold check with sub-2/3 actual stake.
//!
//! # Lookup declarations
//!
//! The `SLACK` column is declared as a 64-bit range check (no selector;
//! padding rows hold 0 which is trivially in range).
//!
//! # u64 fits in scalar — assumption
//!
//! Even at the upper bound (~32 M validators × 32 ETH × 10⁹ gwei ≈ 10¹⁸
//! gwei) the running total fits comfortably in `2^63`. The range check
//! is on `slack`, which is `3·running_total − 2·total_active`, bounded
//! above by `3·(2^63 - 1) ≈ 2.7·10¹⁹`, fitting in `2^65` — but we use a
//! 64-bit range check, which constrains `total_active` and
//! `running_total` to be small enough that `slack` stays in range. In
//! practice the prover pre-validates witness inputs to satisfy this.

use crate::beacon::Gwei;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_EFFECTIVE_BALANCE: usize = 0;
pub const COL_ATTESTING_BIT: usize = 1;
pub const COL_ROW_CONTRIBUTION: usize = 2;
pub const COL_RUNNING_TOTAL: usize = 3;
pub const COL_SEL_VALIDATOR: usize = 4;
pub const COL_SEL_THRESHOLD: usize = 5;
pub const COL_TOTAL_ACTIVE: usize = 6;
pub const COL_SLACK: usize = 7;
/// Validator index, monotonically increasing on validator rows
/// (`SEL_VALIDATOR = 1`). Set to `i` on the i-th validator row, `0`
/// elsewhere. Used for cross-AIR LogUp linkage to the SSZ validator
/// registry (#84): the `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)` tuple
/// must appear in the SSZ side's per-validator table.
pub const COL_VALIDATOR_INDEX: usize = 8;

/// 32 bytes of the claimed finalized block root, populated at the
/// threshold row (where `SEL_THRESHOLD = 1`). Other rows hold zero.
/// Exposed as cross-AIR LogUp output for binding to the BBH-pair
/// AIR's `CLAIMED_ROOT` (which is "the BeaconBlockHeader we're
/// finalizing"). Soundness: gating by `SEL_THRESHOLD` means the
/// LogUp argument consumes exactly one entry per AIR per side, so
/// other-row values are irrelevant.
pub const COL_CLAIMED_FINALIZED_ROOT_OFFSET: usize = 9;  // 9..41

pub const NUM_COLUMNS: usize = COL_CLAIMED_FINALIZED_ROOT_OFFSET + 32; // 41

/// Number of row-local consolidated constraint categories.
pub const NUM_ROW_CONSTRAINTS: usize = 6;

/// Number of cross-row (shifted) constraints.
pub const NUM_SHIFTED: usize = 2;

// ─── Witness type ──────────────────────────────────────────────────────

/// Witness for the stake-weighted finality AIR.
#[derive(Debug, Clone)]
pub struct FinalityWitness {
    /// `(effective_balance_gwei, attesting_bit)` per validator in the
    /// committee. `attesting_bit` must be `0` or `1`.
    pub validators: Vec<(Gwei, u8)>,
    /// Sum of effective balances of all active validators (committee or
    /// not). The 2/3 supermajority is measured against this.
    pub total_active_balance_gwei: Gwei,
    /// Public hint: the SSZ root of the attestation data this finality
    /// argument consumes.
    pub attestation_data_root: [u8; 32],
    /// Public hint: the resulting finalized block root.
    pub finalized_root: [u8; 32],
}

impl FinalityWitness {
    /// Convenience constructor that copies the validator slice.
    pub fn new(
        validators: Vec<(Gwei, u8)>,
        total_active_balance_gwei: Gwei,
        attestation_data_root: [u8; 32],
        finalized_root: [u8; 32],
    ) -> Self {
        Self {
            validators,
            total_active_balance_gwei,
            attestation_data_root,
            finalized_root,
        }
    }

    /// Σ effective_balance · attesting_bit (host-side reference, used by
    /// the trace builder and tests).
    pub fn total_attesting_balance(&self) -> Gwei {
        self.validators
            .iter()
            .map(|(eb, bit)| if *bit != 0 { *eb } else { 0 })
            .fold(0u64, u64::saturating_add)
    }

    /// Algebraic slack on the threshold row:
    ///   `slack = 3 · total_attesting_balance − 2 · total_active_balance_gwei`.
    /// Returns `None` when the supermajority does NOT clear (so the
    /// caller can short-circuit witness construction with a clear error).
    pub fn slack(&self) -> Option<Gwei> {
        let three_a = self.total_attesting_balance().saturating_mul(3);
        let two_t = self.total_active_balance_gwei.saturating_mul(2);
        if three_a >= two_t {
            Some(three_a - two_t)
        } else {
            None
        }
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the stake-weighted finality
/// AIR.
pub struct FinalityConstraintSystem {
    /// Number of validator rows (one per committee member).
    pub committee_size: usize,
    /// Padded domain generator ω. When `Some`, the boundary-row
    /// exclusion product `(z − ω^{committee_size})` is built explicitly.
    pub omega: Option<Scalar>,
    /// Padded domain size (power of two ≥ `num_rows`). Used together
    /// with `omega` to build the boundary-row exclusion.
    pub domain_size: Option<u64>,
}

impl FinalityConstraintSystem {
    /// Construct a finality constraint system for a committee of
    /// `committee_size` validators. The total trace height is
    /// `committee_size + 1` (validators + threshold row), padded to the
    /// next power of two.
    pub fn new(committee_size: usize) -> Self {
        Self {
            committee_size,
            omega: None,
            domain_size: None,
        }
    }

    /// Attach the domain generator and padded size for the trace this
    /// constraint system is paired with. Required for sound shifted
    /// constraint evaluation.
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }

    /// The number of real (pre-padding) trace rows: `committee_size + 1`.
    pub fn num_rows(&self) -> usize {
        self.committee_size + 1
    }

    /// Index of the threshold row in the trace.
    pub fn threshold_row(&self) -> usize {
        self.committee_size
    }
}

// ─── Trace construction ────────────────────────────────────────────────

/// Build a [`TracePolynomials`] from a [`FinalityWitness`]. The trace is
/// padded to the next power of two; padding rows hold all zeros (so the
/// cross-row chain re-binds `RUNNING_TOTAL[0] = 0` through wrap-around).
pub fn build_finality_trace_polynomials(
    witness: &FinalityWitness,
    curve: CurveType,
) -> TracePolynomials {
    let committee_size = witness.validators.len();
    let num_rows = committee_size + 1;
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));

    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    let mut running_total: u64 = 0;
    for (i, &(eb, bit)) in witness.validators.iter().enumerate() {
        // Validator rows: 0..committee_size.
        let bit_u = if bit != 0 { 1u64 } else { 0u64 };
        let contribution = if bit_u == 1 { eb } else { 0 };

        columns[COL_EFFECTIVE_BALANCE][i] = Scalar::from_u64(eb, curve);
        columns[COL_ATTESTING_BIT][i] = Scalar::from_u64(bit_u, curve);
        columns[COL_ROW_CONTRIBUTION][i] = Scalar::from_u64(contribution, curve);
        columns[COL_RUNNING_TOTAL][i] = Scalar::from_u64(running_total, curve);
        columns[COL_SEL_VALIDATOR][i] = one.clone();
        // VALIDATOR_INDEX = i on the i-th validator row. Cross-AIR
        // LogUp linkage to SSZ uses (VALIDATOR_INDEX, EFFECTIVE_BALANCE)
        // as the tuple shape.
        columns[COL_VALIDATOR_INDEX][i] = Scalar::from_u64(i as u64, curve);
        // SEL_THRESHOLD, TOTAL_ACTIVE, SLACK remain zero on validator rows.

        running_total = running_total.saturating_add(contribution);
    }

    // Threshold row at index = committee_size.
    let t_row = committee_size;
    let total_attesting = running_total;
    let total_active = witness.total_active_balance_gwei;
    let three_a = total_attesting.saturating_mul(3);
    let two_t = total_active.saturating_mul(2);
    // Note: a malformed witness (supermajority not met) yields a slack
    // that does not satisfy the 64-bit range check (it's the saturating
    // mathematical 0, but the equation 3·rt − 2·ta = 0 fails to hold).
    // We compute the natural u64 slack for valid witnesses; for
    // unsoundness tests the prover will set whatever value it likes.
    let slack = if three_a >= two_t { three_a - two_t } else { 0 };

    columns[COL_RUNNING_TOTAL][t_row] = Scalar::from_u64(total_attesting, curve);
    columns[COL_SEL_THRESHOLD][t_row] = one.clone();
    columns[COL_TOTAL_ACTIVE][t_row] = Scalar::from_u64(total_active, curve);
    columns[COL_SLACK][t_row] = Scalar::from_u64(slack, curve);
    // SEL_VALIDATOR, EFFECTIVE_BALANCE, ATTESTING_BIT, ROW_CONTRIBUTION
    // remain zero on the threshold row.

    // Populate the claimed finalized root bytes at the threshold row.
    // Cross-AIR LogUp gating by SEL_THRESHOLD picks exactly this row.
    for k in 0..32 {
        columns[COL_CLAIMED_FINALIZED_ROOT_OFFSET + k][t_row] =
            Scalar::from_u64(witness.finalized_root[k] as u64, curve);
    }

    // Padding rows (t_row + 1 .. padded) are all-zero. RUNNING_TOTAL = 0
    // on padding is exactly what the cross-row wrap-around re-binding
    // depends on.

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

// ─── Cross-AIR LogUp linkage helpers (#84) ─────────────────────────────

/// Construct the [`crate::cross_air_logup::CrossAirLogUpDescriptor`]
/// for the Finality ↔ SSZ-validator-registry linkage.
///
/// **A side (Finality)**: the validator rows, gated by `SEL_VALIDATOR`.
/// Tuple columns: `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)` at indices
/// `[COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE]`.
///
/// **B side (SSZ validator extraction)**: not yet implemented. The SSZ
/// AIR currently merkleizes byte chunks; binding to per-validator
/// `(index, effective_balance)` requires an extraction layer that
/// either:
///   - aggregates the 8 bytes of the `effective_balance` field's chunk
///     into a u64 column (gated by a "validator extraction" selector),
///     and exposes the validator index as another column, or
///   - introduces a dedicated `ValidatorExtractConstraintSystem` that
///     consumes the validator registry deserialization and exposes the
///     per-validator tuple directly.
///
/// `b_*` parameters are the SSZ-side column indices (or extraction-AIR
/// indices) the caller has already set up. Today no SSZ side exposes
/// these; this helper documents the descriptor shape so the future
/// extraction work can plug in.
///
/// Tested only at the descriptor-shape level
/// (`finality_ssz_linkage_descriptor_well_formed`); end-to-end
/// validation requires the SSZ extraction work.
pub fn make_finality_ssz_linkage_descriptor(
    a_layer_index: usize,
    b_layer_index: usize,
    b_validator_index_column: usize,
    b_effective_balance_column: usize,
    b_selector_column: Option<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "finality_ssz_validator_v1".into(),
        a_layer_index,
        a_columns: vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE],
        a_selector_column: Some(COL_SEL_VALIDATOR),
        b_layer_index,
        b_columns: vec![b_validator_index_column, b_effective_balance_column],
        b_selector_column,
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Fast scalar exponentiation by a `u64`.
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

// ─── Row-local body evaluations (scalar-point form) ────────────────────

/// 0. `SEL_VALIDATOR · ATTESTING_BIT · (ATTESTING_BIT − 1) = 0`.
fn eval_bit_binary(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let sv = &cols[COL_SEL_VALIDATOR];
    let bit = &cols[COL_ATTESTING_BIT];
    sv.mul(&bit.mul(&bit.sub(&one)))
}

/// 1. `SEL_VALIDATOR · (ROW_CONTRIBUTION − EFFECTIVE_BALANCE · ATTESTING_BIT) = 0`.
fn eval_contribution_def(cols: &[Scalar]) -> Scalar {
    let sv = &cols[COL_SEL_VALIDATOR];
    let contribution = &cols[COL_ROW_CONTRIBUTION];
    let eb = &cols[COL_EFFECTIVE_BALANCE];
    let bit = &cols[COL_ATTESTING_BIT];
    let prod = eb.mul(bit);
    sv.mul(&contribution.sub(&prod))
}

/// 2. `SEL_VALIDATOR · (SEL_VALIDATOR − 1) = 0`.
fn eval_sel_validator_binary(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let sv = &cols[COL_SEL_VALIDATOR];
    sv.mul(&sv.sub(&one))
}

/// 3. `SEL_THRESHOLD · (SEL_THRESHOLD − 1) = 0`.
fn eval_sel_threshold_binary(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let st = &cols[COL_SEL_THRESHOLD];
    st.mul(&st.sub(&one))
}

/// 4. `SEL_THRESHOLD · (3 · RUNNING_TOTAL − 2 · TOTAL_ACTIVE − SLACK) = 0`.
fn eval_threshold_check(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let three = Scalar::from_u64(3, curve);
    let two = Scalar::from_u64(2, curve);
    let st = &cols[COL_SEL_THRESHOLD];
    let rt = &cols[COL_RUNNING_TOTAL];
    let ta = &cols[COL_TOTAL_ACTIVE];
    let slack = &cols[COL_SLACK];
    let body = three.mul(rt).sub(&two.mul(ta)).sub(slack);
    st.mul(&body)
}

/// 5. `(1 − SEL_VALIDATOR) · ROW_CONTRIBUTION = 0`.
/// Pins `ROW_CONTRIBUTION = 0` on every non-validator row (the threshold
/// row and all padding rows). Without this, a malicious prover could put
/// arbitrary values into `ROW_CONTRIBUTION` on non-validator rows; the
/// shifted running-sum constraint would then accumulate those bogus
/// values into `RUNNING_TOTAL` through the padding-row chain and affect
/// `RUNNING_TOTAL[0]` via wrap-around — defeating the threshold check.
fn eval_row_contribution_zero_on_non_validator(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let sv = &cols[COL_SEL_VALIDATOR];
    let rc = &cols[COL_ROW_CONTRIBUTION];
    one.sub(sv).mul(rc)
}

// ─── Row-local body builders (polynomial form) ─────────────────────────

fn build_bit_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let sv = &cols[COL_SEL_VALIDATOR];
    let bit = &cols[COL_ATTESTING_BIT];
    let bit_m1 = poly_sub(bit, &one_poly, curve);
    let bit_body = poly_mul(bit, &bit_m1, curve);
    poly_mul(sv, &bit_body, curve)
}

fn build_contribution_def_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let sv = &cols[COL_SEL_VALIDATOR];
    let contribution = &cols[COL_ROW_CONTRIBUTION];
    let eb = &cols[COL_EFFECTIVE_BALANCE];
    let bit = &cols[COL_ATTESTING_BIT];
    let prod = poly_mul(eb, bit, curve);
    let diff = poly_sub(contribution, &prod, curve);
    poly_mul(sv, &diff, curve)
}

fn build_sel_validator_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let sv = &cols[COL_SEL_VALIDATOR];
    let sv_m1 = poly_sub(sv, &one_poly, curve);
    poly_mul(sv, &sv_m1, curve)
}

fn build_sel_threshold_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let st = &cols[COL_SEL_THRESHOLD];
    let st_m1 = poly_sub(st, &one_poly, curve);
    poly_mul(st, &st_m1, curve)
}

fn build_threshold_check_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let three = Scalar::from_u64(3, curve);
    let two = Scalar::from_u64(2, curve);
    let st = &cols[COL_SEL_THRESHOLD];
    let rt = &cols[COL_RUNNING_TOTAL];
    let ta = &cols[COL_TOTAL_ACTIVE];
    let slack = &cols[COL_SLACK];

    // 3·rt − 2·ta − slack
    let three_rt = poly_scalar_mul(rt, &three);
    let two_ta = poly_scalar_mul(ta, &two);
    let mut body = poly_sub(&three_rt, &two_ta, curve);
    body = poly_sub(&body, slack, curve);
    poly_mul(st, &body, curve)
}

fn build_row_contribution_zero_on_non_validator_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let sv = &cols[COL_SEL_VALIDATOR];
    let rc = &cols[COL_ROW_CONTRIBUTION];
    let one_minus_sv = poly_sub(&one_poly, sv, curve);
    poly_mul(&one_minus_sv, rc, curve)
}

// ─── Cross-row boundary helpers ────────────────────────────────────────

/// Boundary rows whose cross-row transition is excluded from vanishing.
///
/// We exclude **only** the threshold row → first-padding transition
/// (row index `committee_size`). The wrap-around row `domain_size − 1`
/// is intentionally NOT excluded — the wrap binds
/// `RUNNING_TOTAL[0] = 0` (see top-of-module description).
fn boundary_rows(committee_size: usize) -> Vec<usize> {
    vec![committee_size]
}

// ─── VmConstraintSystem implementation ─────────────────────────────────

impl VmConstraintSystem for FinalityConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "bit_binary".into(),
            "contribution_def".into(),
            "sel_validator_binary".into(),
            "sel_threshold_binary".into(),
            "threshold_check".into(),
            "row_contribution_zero_on_non_validator".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(
            columns.len() >= NUM_COLUMNS,
            "finality AIR expects at least {} columns",
            NUM_COLUMNS
        );
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let three = Scalar::from_u64(3, curve);
        let two = Scalar::from_u64(2, curve);
        let n = columns[0].len();

        let mut bit_evals = vec![Scalar::zero(curve); n];
        let mut contrib_evals = vec![Scalar::zero(curve); n];
        let mut sv_bin_evals = vec![Scalar::zero(curve); n];
        let mut st_bin_evals = vec![Scalar::zero(curve); n];
        let mut threshold_evals = vec![Scalar::zero(curve); n];
        let mut rc_zero_non_val_evals = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let sv = &columns[COL_SEL_VALIDATOR][row];
            let bit = &columns[COL_ATTESTING_BIT][row];
            let eb = &columns[COL_EFFECTIVE_BALANCE][row];
            let contribution = &columns[COL_ROW_CONTRIBUTION][row];
            let st = &columns[COL_SEL_THRESHOLD][row];
            let rt = &columns[COL_RUNNING_TOTAL][row];
            let ta = &columns[COL_TOTAL_ACTIVE][row];
            let slack = &columns[COL_SLACK][row];

            bit_evals[row] = sv.mul(&bit.mul(&bit.sub(&one)));
            contrib_evals[row] = sv.mul(&contribution.sub(&eb.mul(bit)));
            sv_bin_evals[row] = sv.mul(&sv.sub(&one));
            st_bin_evals[row] = st.mul(&st.sub(&one));
            let body = three.mul(rt).sub(&two.mul(ta)).sub(slack);
            threshold_evals[row] = st.mul(&body);
            // (1 − SEL_VALIDATOR) · ROW_CONTRIBUTION = 0
            rc_zero_non_val_evals[row] = one.sub(sv).mul(contribution);
        }
        vec![
            bit_evals,
            contrib_evals,
            sv_bin_evals,
            st_bin_evals,
            threshold_evals,
            rc_zero_non_val_evals,
        ]
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            eval_bit_binary(col_evals_at_z),
            eval_contribution_def(col_evals_at_z),
            eval_sel_validator_binary(col_evals_at_z),
            eval_sel_threshold_binary(col_evals_at_z),
            eval_threshold_check(col_evals_at_z),
            eval_row_contribution_zero_on_non_validator(col_evals_at_z),
        ];
        let curve = alpha.curve_type();
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_SEL_VALIDATOR, COL_SEL_THRESHOLD]
    }

    /// Padding rows are all-zero. All row-local bodies are gated by a
    /// selector that is zero on padding (so they vanish trivially), and
    /// the cross-row chain is compatible with `RUNNING_TOTAL = 0` on
    /// padding. No "no-op" selector is needed.
    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        // Defensive: zero every column on padding rows. The trace
        // constructor already does this; we re-zero in case an upstream
        // stage mutated padding cells.
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
        for col_v in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col_v.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
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
            build_bit_binary_poly(column_coeffs, curve),
            build_contribution_def_poly(column_coeffs, curve),
            build_sel_validator_binary_poly(column_coeffs, curve),
            build_sel_threshold_binary_poly(column_coeffs, curve),
            build_threshold_check_poly(column_coeffs, curve),
            build_row_contribution_zero_on_non_validator_poly(column_coeffs, curve),
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
        vec![COL_RUNNING_TOTAL]
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
        if shifted_evals.is_empty() || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();

        let rt_curr = &col_evals_at_z[COL_RUNNING_TOTAL];
        let rt_next = &shifted_evals[0];
        let rc = &col_evals_at_z[COL_ROW_CONTRIBUTION];
        let st = &col_evals_at_z[COL_SEL_THRESHOLD];

        // ── Body 0: running-sum step ────────────────────────────────
        // body_0 = RT(ω·z) − RT(z) − RC(z), excluded only at the
        // threshold-row → first-padding transition (row committee_size).
        // The domain wrap is intentionally NOT excluded — combined with
        // body_1 below, the wrap re-binds `RUNNING_TOTAL[0] = 0`.
        //
        // Important: derive ω from `omega_n_minus_1` (not `self.omega`)
        // because the actual proof domain may be larger than the trace's
        // natural padded size when downstream LogUp byte tables inflate
        // the domain. Using a stale `self.omega` would compute exclusion
        // at the wrong root of unity. ω · ω^(n-1) = 1 ⟹ ω = inverse.
        let body_0 = rt_next.sub(rt_curr).sub(rc);
        let omega = omega_n_minus_1.inverse();
        let rows = boundary_rows(self.committee_size);
        let mut exclusion_0 = Scalar::one(curve);
        for r in &rows {
            let omega_r = scalar_pow(&omega, *r as u64);
            exclusion_0 = exclusion_0.mul(&z.sub(&omega_r));
        }

        // ── Body 1: post-threshold zero ─────────────────────────────
        // body_1 = SEL_THRESHOLD(z) · RT(ω·z). On row committee_size
        // (where SEL_THRESHOLD = 1) this forces RT[committee_size+1] = 0.
        // On all other rows SEL_THRESHOLD = 0 so the body is trivially
        // zero. Combined with body_0's chain enforcement on padding
        // rows and the new row-local pin of RC = 0 on non-validator
        // rows, the wrap-around re-binds RT[0] = 0 and the threshold
        // check becomes sound.
        let body_1 = st.mul(rt_next);

        // α^alpha_offset for body_0; α^(alpha_offset+1) for body_1.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term_0 = ap.mul(&body_0).mul(&exclusion_0);
        ap = ap.mul(alpha);
        let term_1 = ap.mul(&body_1);
        term_0.add(&term_1)
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();

        let rt = &column_coeffs[COL_RUNNING_TOTAL];
        let rc = &column_coeffs[COL_ROW_CONTRIBUTION];
        let st = &column_coeffs[COL_SEL_THRESHOLD];
        let rt_shift = poly_shift(rt, omega);

        // body_0(X) = RT(ω·X) − RT(X) − RC(X), excluded at row committee_size
        let body_0 = poly_sub(&poly_sub(&rt_shift, rt, curve), rc, curve);
        let rows = boundary_rows(self.committee_size);
        let mut excluded_0 = body_0;
        for r in &rows {
            let omega_r = scalar_pow(omega, *r as u64);
            excluded_0 = poly_mul_linear(&excluded_0, &omega_r);
        }

        // body_1(X) = SEL_THRESHOLD(X) · RT(ω·X), no exclusion
        let body_1 = poly_mul(st, &rt_shift, curve);

        // α^alpha_offset · excluded_0 + α^(alpha_offset+1) · body_1
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term_0 = poly_scalar_mul(&excluded_0, &ap);
        ap = ap.mul(alpha);
        let term_1 = poly_scalar_mul(&body_1, &ap);
        poly_add(&term_0, &term_1, curve)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 64-bit range check on the slack column, gated by sel_threshold
        // so only the threshold row is range-checked (validator rows hold
        // slack=0, which is trivially in range). This is what makes the
        // ≥2/3 threshold check sound in-circuit: a malicious prover
        // claiming `3·effective < 2·active` would have to commit a
        // negative-mod-prime slack which can't byte-decompose into a
        // valid 64-bit range.
        let table64 = LookupTable::range(64);
        let table8 = LookupTable::range(8);
        const TBL_64: usize = 0;
        const TBL_8: usize = 1;
        let mut declarations = vec![(
            LookupDeclaration {
                label: "slack_range_64".into(),
                column_index: COL_SLACK,
                max_bits: 64,
                selector_column: None,
            },
            TBL_64,
        )];
        for k in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("claimed_finalized_root_{}_8bit", k),
                    column_index: COL_CLAIMED_FINALIZED_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }
        LookupRequirements {
            tables: vec![table64, table8],
            declarations,
        }
    }
}

/// Cross-AIR LogUp linkage: BBH-pair AIR's `CLAIMED_ROOT` ↔ Finality
/// AIR's `CLAIMED_FINALIZED_ROOT`.
///
/// **A side (BBH-pair)**: 32-byte CLAIMED_ROOT tuple gated by
/// `IS_PAIR0` (=1 only at BBH invocation row 0 → exactly 1 entry =
/// BBH's computed `hash_tree_root`).
///
/// **B side (Finality)**: 32-byte CLAIMED_FINALIZED_ROOT tuple gated
/// by `COL_SEL_THRESHOLD` (=1 only at the threshold row, the one row
/// where stake-weighted finalization is checked → exactly 1 entry).
///
/// Multiset equality on 1-vs-1 algebraically pins
/// `bbh.hash_tree_root() == finality_witness.finalized_root`. This is
/// the algebraic version of the `claimed_finalized.root == bbh.hash_tree_root()`
/// check in the host-side `beacon_world_proof::verify_ffg_finality_oracle`.
///
/// Composed with the existing C↔B chain, this closes
/// **"transaction's successful execution, mutation of world state,
/// inclusion in the beacon chain, AND economic finality"** as a
/// single algebraic proof.
pub fn make_finality_to_bbh_pair_linkage_descriptor(
    finality_layer_index: usize,
    bbh_pair_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::beacon_block_header_pair_air as bbh;

    let mut a_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { a_columns.push(COL_CLAIMED_FINALIZED_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { b_columns.push(bbh::COL_CLAIMED_ROOT_OFFSET + k); }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "finality_finalized_root_to_bbh_pair_claimed_root_v1".into(),
        a_layer_index: finality_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_THRESHOLD),
        b_layer_index: bbh_pair_layer_index,
        b_columns,
        // BBH-pair AIR exposes CLAIMED_ROOT constant across rows; pick
        // any one row via IS_PAIR0 (= 1 only at row 0).
        b_selector_column: Some(bbh::COL_IS_PAIR0),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    /// Minimal "supermajority clears" witness:
    ///   - 4 validators, each with 32 ETH effective balance,
    ///   - 3 of 4 attesting (75% > 66.6…%),
    ///   - total active balance = 4 · 32 ETH.
    fn good_witness() -> FinalityWitness {
        let eb = 32_000_000_000u64;
        FinalityWitness {
            validators: vec![(eb, 1), (eb, 1), (eb, 1), (eb, 0)],
            total_active_balance_gwei: 4 * eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        }
    }

    fn extract_columns(trace: &TracePolynomials) -> Vec<Vec<Scalar>> {
        trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect()
    }

    // ── Witness sanity ────────────────────────────────────────────────

    #[test]
    fn finality_witness_total_attesting_balance() {
        let w = good_witness();
        assert_eq!(w.total_attesting_balance(), 3 * 32_000_000_000);
    }

    #[test]
    fn finality_witness_slack_when_supermajority_clears() {
        let w = good_witness();
        // 3·(3·32) = 288, 2·(4·32) = 256, slack = 32.
        let expected = 3u64 * 3 * 32_000_000_000 - 2 * 4 * 32_000_000_000;
        assert_eq!(w.slack(), Some(expected));
    }

    #[test]
    fn finality_witness_slack_none_when_supermajority_fails() {
        // 1 of 4 attesting → 25% → fails.
        let eb = 32_000_000_000u64;
        let w = FinalityWitness {
            validators: vec![(eb, 1), (eb, 0), (eb, 0), (eb, 0)],
            total_active_balance_gwei: 4 * eb,
            attestation_data_root: [0; 32],
            finalized_root: [0; 32],
        };
        assert!(w.slack().is_none());
    }

    // ── AIR shape ─────────────────────────────────────────────────────

    #[test]
    fn finality_cs_labels_and_counts() {
        let cs = FinalityConstraintSystem::new(4);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_constraints(), 6);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert_eq!(cs.num_shifted_constraints(), 2);
        assert_eq!(
            cs.selector_column_indices(),
            vec![COL_SEL_VALIDATOR, COL_SEL_THRESHOLD]
        );
        assert_eq!(cs.shifted_column_indices(), vec![COL_RUNNING_TOTAL]);
        assert!(cs.padding_selector_column().is_none());
        assert_eq!(cs.num_rows(), 5);
        assert_eq!(cs.threshold_row(), 4);
    }

    #[test]
    fn finality_cs_lookup_declarations_slack_and_root_bytes() {
        let cs = FinalityConstraintSystem::new(4);
        let reqs = cs.lookup_declarations();
        // 2 tables now: 64-bit range for slack, 8-bit range for the 32
        // claimed_finalized_root bytes.
        assert_eq!(reqs.tables.len(), 2);
        assert_eq!(reqs.tables[0].bits, 64);
        assert_eq!(reqs.tables[1].bits, 8);
        // 1 slack decl + 32 root-byte decls.
        assert_eq!(reqs.declarations.len(), 1 + 32);
        let (slack_decl, slack_tbl) = &reqs.declarations[0];
        assert_eq!(slack_decl.column_index, COL_SLACK);
        assert_eq!(slack_decl.max_bits, 64);
        assert_eq!(*slack_tbl, 0);
        // First root-byte decl indexes COL_CLAIMED_FINALIZED_ROOT_OFFSET.
        let (root_decl, root_tbl) = &reqs.declarations[1];
        assert_eq!(root_decl.column_index, COL_CLAIMED_FINALIZED_ROOT_OFFSET);
        assert_eq!(root_decl.max_bits, 8);
        assert_eq!(*root_tbl, 1);
    }

    #[test]
    fn finality_trace_populates_claimed_finalized_root_at_threshold_row() {
        let curve = CurveType::Bls48581;
        let mut w = good_witness();
        w.finalized_root = [0x42; 32];
        let trace = build_finality_trace_polynomials(&w, curve);
        let columns = extract_columns(&trace);
        let t_row = trace.num_rows - 1;
        for k in 0..32 {
            assert_eq!(
                columns[COL_CLAIMED_FINALIZED_ROOT_OFFSET + k][t_row].to_u64(),
                0x42,
                "claimed_finalized_root[{}] at threshold row", k,
            );
        }
        // Validator rows are zero on the new column.
        for r in 0..t_row {
            for k in 0..32 {
                assert!(
                    columns[COL_CLAIMED_FINALIZED_ROOT_OFFSET + k][r].is_zero(),
                    "claimed_finalized_root[{}] on validator row {}", k, r,
                );
            }
        }
    }

    #[test]
    fn finality_to_bbh_linkage_descriptor_shape() {
        let d = make_finality_to_bbh_pair_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        assert_eq!(d.label, "finality_finalized_root_to_bbh_pair_claimed_root_v1");
        assert_eq!(d.a_selector_column, Some(COL_SEL_THRESHOLD));
        use crate::beacon_block_header_pair_air as bbh;
        assert_eq!(d.b_selector_column, Some(bbh::COL_IS_PAIR0));
        assert_eq!(d.a_columns[0], COL_CLAIMED_FINALIZED_ROOT_OFFSET);
        assert_eq!(d.b_columns[0], bbh::COL_CLAIMED_ROOT_OFFSET);
    }

    // ── Witness → trace ───────────────────────────────────────────────

    #[test]
    fn finality_trace_lays_out_validators_and_threshold() {
        let curve = CurveType::Bls48581;
        let w = good_witness();
        let trace = build_finality_trace_polynomials(&w, curve);
        assert_eq!(trace.num_rows, 5); // 4 validators + 1 threshold
        let columns = extract_columns(&trace);
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);

        let scalar_eq = |a: &Scalar, b: &Scalar| a.sub(b).is_zero();

        // Validator rows (0..4): SEL_VALIDATOR = 1.
        for r in 0..4 {
            assert!(scalar_eq(&columns[COL_SEL_VALIDATOR][r], &one));
            assert!(scalar_eq(&columns[COL_SEL_THRESHOLD][r], &zero));
        }
        // Threshold row (4): SEL_THRESHOLD = 1.
        assert!(scalar_eq(&columns[COL_SEL_VALIDATOR][4], &zero));
        assert!(scalar_eq(&columns[COL_SEL_THRESHOLD][4], &one));

        // Running total trajectory: 0, 32e9, 64e9, 96e9, 96e9.
        let eb = 32_000_000_000u64;
        let expected = [0, eb, 2 * eb, 3 * eb, 3 * eb];
        for (r, exp) in expected.iter().enumerate() {
            assert!(
                scalar_eq(
                    &columns[COL_RUNNING_TOTAL][r],
                    &Scalar::from_u64(*exp, curve),
                ),
                "running total mismatch at row {}",
                r,
            );
        }

        // Padding rows beyond threshold: all zero RUNNING_TOTAL (so the
        // wrap re-binds RUNNING_TOTAL[0] = 0).
        for r in 5..(trace.padded_size as usize) {
            assert!(scalar_eq(&columns[COL_RUNNING_TOTAL][r], &zero));
            assert!(scalar_eq(&columns[COL_ROW_CONTRIBUTION][r], &zero));
        }
    }

    // ── Constraint evaluation on valid witness ────────────────────────

    #[test]
    fn finality_witness_evaluates_zero_on_valid_input() {
        let curve = CurveType::Bls48581;
        let w = good_witness();
        let trace = build_finality_trace_polynomials(&w, curve);
        let columns = extract_columns(&trace);
        let cs = FinalityConstraintSystem::new(w.validators.len());

        // evaluate_on_domain: every constraint zero on every row.
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, trace.num_rows);
        for (k, vec_) in evals.iter().enumerate() {
            for (row, v) in vec_.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) fired at row {} on valid witness",
                    k,
                    cs.constraint_labels()[k],
                    row
                );
            }
        }

        // evaluate_at_point: also zero per real row.
        let alpha = Scalar::from_u64(7, curve);
        for row in 0..trace.padded_size as usize {
            let row_vals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let c_at = cs.evaluate_at_point(&row_vals, &alpha);
            assert!(
                c_at.is_zero(),
                "combined row-local constraint nonzero on valid witness at row {}",
                row
            );
        }
    }

    /// Cross-row (shifted) chain holds at every row except the
    /// threshold-row → first-padding transition (the only excluded
    /// boundary). The wrap row `n − 1` is intentionally NOT excluded
    /// and DOES vanish on a valid witness, since
    /// `RUNNING_TOTAL[0] − RUNNING_TOTAL[n−1] − ROW_CONTRIBUTION[n−1] =
    /// 0 − 0 − 0 = 0`.
    #[test]
    fn finality_cross_row_chain_holds_on_valid_witness() {
        let curve = CurveType::Bls48581;
        let w = good_witness();
        let trace = build_finality_trace_polynomials(&w, curve);
        let columns = extract_columns(&trace);
        let n = trace.padded_size as usize;
        let committee_size = w.validators.len();

        for r in 0..n {
            let r_next = (r + 1) % n;
            let rt_curr = &columns[COL_RUNNING_TOTAL][r];
            let rt_next = &columns[COL_RUNNING_TOTAL][r_next];
            let rc = &columns[COL_ROW_CONTRIBUTION][r];
            let body = rt_next.sub(rt_curr).sub(rc);
            if r == committee_size {
                // Threshold row → first padding: RUNNING_TOTAL drops
                // from total back to zero. Body is nonzero, but
                // excluded by `(z − ω^{committee_size})`.
                assert!(
                    !body.is_zero(),
                    "expected threshold→padding boundary to have nonzero body"
                );
            } else {
                assert!(
                    body.is_zero(),
                    "cross-row chain failed at non-boundary row {} (r_next={})",
                    r,
                    r_next
                );
            }
        }
    }

    // ── Soundness: tamper detection ───────────────────────────────────

    /// Flipping an attesting bit from 0 → 1 (free supermajority for the
    /// adversary) makes the row's contribution disagree with
    /// `EFFECTIVE_BALANCE · ATTESTING_BIT`, AND propagates a wrong
    /// running-total that breaks the cross-row chain.
    #[test]
    fn finality_constraints_reject_flipped_attesting_bit() {
        let curve = CurveType::Bls48581;
        let w = good_witness();
        let trace = build_finality_trace_polynomials(&w, curve);
        let mut columns = extract_columns(&trace);

        // Flip ATTESTING_BIT at row 3 (currently 0 → 1) without updating
        // ROW_CONTRIBUTION or RUNNING_TOTAL. The contribution_def
        // constraint must fire.
        columns[COL_ATTESTING_BIT][3] = Scalar::one(curve);

        let cs = FinalityConstraintSystem::new(w.validators.len());
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, trace.num_rows);
        // Constraint 1 = contribution_def.
        assert!(
            !evals[1][3].is_zero(),
            "contribution_def must fire at tampered row 3"
        );
    }

    /// If the prover updates ROW_CONTRIBUTION but lies about
    /// RUNNING_TOTAL — leaving the threshold row's RUNNING_TOTAL
    /// inflated — the threshold check still depends on TOTAL_ACTIVE.
    /// Lowering TOTAL_ACTIVE so a non-supermajority "looks like" one
    /// makes the slack negative, which fails the 64-bit range check
    /// (and the threshold equation, depending on what slack the prover
    /// commits).
    #[test]
    fn finality_constraints_reject_tampered_total_active() {
        // A non-supermajority witness: 1 of 4 attesters (25%).
        let curve = CurveType::Bls48581;
        let eb = 32_000_000_000u64;
        let w = FinalityWitness {
            validators: vec![(eb, 1), (eb, 0), (eb, 0), (eb, 0)],
            total_active_balance_gwei: 4 * eb,
            attestation_data_root: [0; 32],
            finalized_root: [0; 32],
        };
        let trace = build_finality_trace_polynomials(&w, curve);
        let columns = extract_columns(&trace);
        let cs = FinalityConstraintSystem::new(w.validators.len());

        // The threshold check `3·rt − 2·ta − slack = 0` uses
        // ROW_CONTRIBUTION-derived rt. The `build_finality_trace_polynomials`
        // helper sets slack to 0 when supermajority fails (saturating
        // sub), but 3·rt = 3·32e9 = 96e9 and 2·ta = 256e9, so
        // 3·rt − 2·ta = -160e9 (mod p). With slack = 0, the equation
        // 3·rt − 2·ta − slack = -160e9 ≠ 0. The threshold_check
        // constraint must fire on row 4.
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, trace.num_rows);
        // Constraint 4 = threshold_check.
        assert!(
            !evals[4][4].is_zero(),
            "threshold_check must fire at row 4 when supermajority is not met"
        );
    }

    #[test]
    fn finality_ssz_linkage_descriptor_well_formed() {
        // Synthetic SSZ-side column indices — placeholder until the
        // SSZ extraction AIR is wired. The descriptor shape is what
        // we validate.
        let desc = make_finality_ssz_linkage_descriptor(
            0,    // Finality layer
            1,    // SSZ extraction layer
            42,   // SSZ-side validator_index column (placeholder)
            43,   // SSZ-side effective_balance column (placeholder)
            Some(44),
        );
        assert_eq!(desc.label, "finality_ssz_validator_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        // Tuple shape: (validator_index, effective_balance) on each side.
        assert_eq!(desc.a_columns.len(), 2);
        assert_eq!(desc.b_columns.len(), 2);
        assert_eq!(desc.a_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(desc.a_columns[1], COL_EFFECTIVE_BALANCE);
        assert_eq!(desc.b_columns[0], 42);
        assert_eq!(desc.b_columns[1], 43);
        // A side gated by SEL_VALIDATOR (only validator rows expose
        // tuples); B side gated by caller-supplied selector.
        assert_eq!(desc.a_selector_column, Some(COL_SEL_VALIDATOR));
        assert_eq!(desc.b_selector_column, Some(44));
    }

    /// Soundness regression for the closed `RUNNING_TOTAL[0]` gap.
    ///
    /// Pre-fix: a malicious prover could keep the chain transition and
    /// threshold-equation constraints satisfied while inflating
    /// `RUNNING_TOTAL[committee_size]` by setting `RUNNING_TOTAL[0] ≠ 0`
    /// and rebalancing through padding. The wrap-around constraint
    /// alone did NOT pin `RUNNING_TOTAL[0] = 0` — the doc relied on
    /// `RUNNING_TOTAL[n−1] = 0` and `ROW_CONTRIBUTION[n−1] = 0`, which
    /// are unconstrained. With insufficient committee stake the prover
    /// could still pass the threshold check by inflating
    /// `RUNNING_TOTAL[committee_size]` via this trick.
    ///
    /// The fix added two constraints:
    /// - `(1 − SEL_VALIDATOR) · ROW_CONTRIBUTION = 0` (row-local 5)
    /// - `SEL_THRESHOLD(X) · RUNNING_TOTAL(ω·X) = 0` (shifted 7)
    ///
    /// This test forges a trace with `RUNNING_TOTAL[committee_size+1] ≠ 0`
    /// (the post-threshold padding row) and confirms the new shifted
    /// constraint catches it via the polynomial body at row
    /// committee_size.
    #[test]
    fn finality_constraints_reject_post_threshold_running_total_inflation() {
        let curve = CurveType::Bls48581;
        let w = good_witness();
        let trace = build_finality_trace_polynomials(&w, curve);
        let mut columns = extract_columns(&trace);

        let cs = FinalityConstraintSystem::new(w.validators.len());
        let committee_size = w.validators.len();
        // Forge: set RUNNING_TOTAL on the row AFTER the threshold to
        // a non-zero value. In a malicious flow this would propagate
        // through padding to inflate RUNNING_TOTAL[0] via wrap.
        let post_threshold_row = committee_size + 1;
        let bogus = Scalar::from_u64(123_456, curve);
        columns[COL_RUNNING_TOTAL][post_threshold_row] = bogus.clone();

        // The row-local checks are unaffected (they don't reference
        // RUNNING_TOTAL on the post-threshold row).
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, trace.num_rows);
        for k in 0..NUM_ROW_CONSTRAINTS {
            for r in 0..trace.num_rows {
                assert!(
                    evals[k][r].is_zero(),
                    "row-local constraint {} at row {} must remain satisfied",
                    cs.constraint_labels()[k],
                    r
                );
            }
        }

        // The new shifted body at row = committee_size:
        // SEL_THRESHOLD[committee_size] · RUNNING_TOTAL[committee_size+1]
        //   = 1 · bogus = bogus ≠ 0.
        // Manually compute the body the verifier evaluates.
        let st = &columns[COL_SEL_THRESHOLD][committee_size];
        let rt_next = &columns[COL_RUNNING_TOTAL][post_threshold_row];
        let body_1 = st.mul(rt_next);
        assert!(
            !body_1.is_zero(),
            "running_total_zero_post_threshold body must fire on the forged trace"
        );
    }

    /// Tampering with RUNNING_TOTAL on the threshold row breaks the
    /// cross-row chain at the validator → threshold transition (row
    /// committee_size − 1).
    #[test]
    fn finality_cross_row_rejects_tampered_running_total() {
        let curve = CurveType::Bls48581;
        let w = good_witness();
        let trace = build_finality_trace_polynomials(&w, curve);
        let mut columns = extract_columns(&trace);

        // Inflate RUNNING_TOTAL on the threshold row.
        let cheated = Scalar::from_u64(999_999_999_999, curve);
        columns[COL_RUNNING_TOTAL][4] = cheated;

        // Cross-row body at r = 3 (last validator → threshold):
        //   rt[4] − rt[3] − rc[3] = cheated − rt[3] − rc[3] ≠ 0.
        let r = 3;
        let r_next = 4;
        let body = columns[COL_RUNNING_TOTAL][r_next]
            .sub(&columns[COL_RUNNING_TOTAL][r])
            .sub(&columns[COL_ROW_CONTRIBUTION][r]);
        assert!(
            !body.is_zero(),
            "tampered RUNNING_TOTAL must break the validator→threshold transition"
        );
    }

    // ── End-to-end prove/verify ──────────────────────────────────────

    /// Prove + verify the Finality AIR end-to-end via the scheme-generic
    /// pipeline. Validates the full path: trace → IFFT → commitment →
    /// constraint poly → quotient → opening → pairing check.
    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn finality_proof_round_trips_through_scheme() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = good_witness();
        let trace = build_finality_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = FinalityConstraintSystem::new(w.validators.len())
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "Finality proof must verify end-to-end");
    }

    /// Wrap a real Finality `ExecutionProof`'s bytes inside a
    /// [`crate::layer_chain::LayerChainProof`] tagged
    /// [`crate::layer_chain::LayerProofKind::Finality`] and verify via
    /// the chain envelope's per-layer verifier closure. This is the
    /// integration point for the 4-layer unified Ethereum proof.
    #[test]
    #[ignore = "slow: prove + chain-envelope verify; run with --release --ignored"]
    fn finality_proof_flows_through_layer_chain_envelope() {
        use crate::layer_chain::{
            ChainBoundaries, LayerChain, LayerChainProof, LayerProof, LayerProofKind,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = good_witness();
        let num_validators = w.validators.len();
        let trace = build_finality_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = FinalityConstraintSystem::new(num_validators)
            .with_omega_and_domain(omega.clone(), domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let proof_bytes = proof.to_bytes();

        // Build a 4-layer chain. The Finality layer (chain index 3)
        // carries our real proof; the others are reference-only.
        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: w.attestation_data_root,
            num_attesters: w
                .validators
                .iter()
                .filter(|(_, b)| *b == 1)
                .count() as u64,
            finalized_root: w.finalized_root,
            total_effective_balance_gwei: w.total_attesting_balance(),
        };
        let chain = LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 3 {
                    LayerProof::with_proof(
                        claim,
                        LayerProofKind::Finality,
                        proof_bytes.clone(),
                    )
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        let result = chain_proof.verify_with_layer_verifier(|layer| match layer.kind {
            LayerProofKind::Finality => {
                let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                    .map_err(|e| format!("finality decode: {:?}", e))?;
                let inner_cs = FinalityConstraintSystem::new(num_validators)
                    .with_omega_and_domain(omega.clone(), domain_size);
                if crate::verifier::verify_with_scheme(&p, &inner_cs, &scheme, curve) {
                    Ok(())
                } else {
                    Err("finality proof did not verify".to_string())
                }
            }
            LayerProofKind::ReferenceOnly => Ok(()),
            other => Err(format!("unsupported kind {}", other.as_str())),
        });
        assert_eq!(result, Ok(()), "Finality chain must verify end-to-end");
    }

    /// 2-AIR `joint_prove`/`joint_verify` regression for the
    /// Finality ↔ ValidatorExtract cross-AIR LogUp linkage. Closes
    /// the cryptographic side of #84: the descriptor was already
    /// wired (`make_finality_validator_extract_linkage_descriptor`),
    /// this exercises both AIRs together with the linkage active.
    ///
    /// Setup: 4 validators, all with `effective_balance = 32 ETH`.
    ///   - Finality validators: 3 attesting + 1 non-attesting (gives
    ///     75% participation, exceeds 2/3 threshold).
    ///   - ValidatorExtract: same 4 validators with the same effective
    ///     balances and validator_root values.
    ///
    /// L1 multiset: each Finality validator row's `(idx, eff_bal)`
    /// pair must appear in some ValidatorExtract row's
    /// `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)` tuple. With identical
    /// per-row data on both sides, multiset equality holds trivially.
    ///
    /// Combined with ValidatorExtract's `validator_index_chain`
    /// constraint (which pins `VALIDATOR_INDEX[i] = i` canonically),
    /// the linkage cryptographically forces Finality's effective
    /// balances to match the canonical validator ordering.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove (~1 min); run with --release --ignored"]
    fn joint_prove_finality_validator_extract_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Consistent per-validator effective balances on both sides.
        let eb = 32_000_000_000u64;

        // ── Finality side ──
        let finality_w = FinalityWitness {
            validators: vec![(eb, 1), (eb, 1), (eb, 1), (eb, 0)],
            total_active_balance_gwei: 4 * eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // ── ValidatorExtract side ──
        // Validator roots are arbitrary distinct 32-byte arrays; the
        // L1 linkage only matches (idx, eff_bal), not the root.
        let mk_root = |seed: u8| {
            let mut r = [0u8; 32];
            for (i, b) in r.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            r
        };
        let ve_w = ValidatorExtractWitness {
            validators: vec![
                ValidatorExtractRow {
                    effective_balance: eb,
                    validator_root: mk_root(0x10),
                },
                ValidatorExtractRow {
                    effective_balance: eb,
                    validator_root: mk_root(0x20),
                },
                ValidatorExtractRow {
                    effective_balance: eb,
                    validator_root: mk_root(0x30),
                },
                ValidatorExtractRow {
                    effective_balance: eb,
                    validator_root: mk_root(0x40),
                },
            ],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        // ── Linkage L1: Finality ↔ ValidatorExtract on (idx, eff_bal) ──
        let l1 = make_finality_validator_extract_linkage_descriptor(
            /* finality */ 0, /* ve */ 1,
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&finality_trace, &finality_cs), (&ve_trace, &ve_cs)];
        let linkages = vec![l1];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched Finality + ValidatorExtract");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b,
            "honest closure scalars must match"
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&finality_cs, &ve_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the Finality ↔ ValidatorExtract honest joint proof"
        );
    }

    /// 3-AIR `joint_prove`/`joint_verify` regression closing more of #84:
    /// Finality + ValidatorExtract + SSZ. Two cross-AIR LogUp linkages
    /// active simultaneously:
    ///   - L1: Finality↔ValidatorExtract on `(VALIDATOR_INDEX,
    ///     EFFECTIVE_BALANCE)` (already validated by
    ///     `joint_prove_finality_validator_extract_linkage`).
    ///   - L2: ValidatorExtract↔SSZ Left on the 32-byte `validator_root`
    ///     tuple. Pins each ValidatorExtract row's `validator_root` to a
    ///     real SSZ leaf chunk that's part of the registry merkle tree.
    ///
    /// Setup: SINGLE validator. SSZ merkleizes a 1-leaf chunk slice
    /// (the validator_root), padded by the merkleization algorithm
    /// to 2 leaves: LEFT=val_root (real, IS_LEFT_REAL=1) +
    /// RIGHT=zero_chunk(0) (padded, IS_RIGHT_REAL=0). The L2 multiset
    /// balances on the Left side (1 entry on each side); Right side
    /// is intentionally not added as a third linkage (it would have
    /// A={val_root}, B={} since IS_RIGHT_REAL=0 — multi-validator
    /// setups would need both linkages active with parity-aware A-side
    /// selectors, see `make_validator_extract_ssz_linkage_descriptor`
    /// docs).
    ///
    /// Combined with ValidatorExtract's `validator_index_chain`
    /// constraint and SSZ's pair-hashing chain, this proves: the
    /// single validator's effective_balance (claimed by Finality) is
    /// the canonical balance of the validator at index 0 of the
    /// registry, whose validator_root participates in the SSZ merkle
    /// tree.
    #[test]
    #[ignore = "slow: 3-AIR + 2-linkage joint_prove (~2 min); run with --release --ignored"]
    fn joint_prove_finality_validator_extract_ssz_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as build_ssz_trace, SszConstraintSystem,
        };
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            make_validator_extract_ssz_linkage_descriptor,
            SszLeafSide, ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Single validator with a distinct root.
        let eb = 32_000_000_000u64;
        let mut val_root = [0u8; 32];
        for i in 0..32 {
            val_root[i] = 0x10u8.wrapping_add(i as u8);
        }

        // ── Finality (1 validator, attesting; will fail threshold but
        //    constraint system doesn't reject — threshold is informational
        //    via SLACK + 64-bit lookup). ──
        let finality_w = FinalityWitness {
            validators: vec![(eb, 1)],
            total_active_balance_gwei: eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // ── ValidatorExtract (1 validator, same eff_bal, with val_root) ──
        let ve_w = ValidatorExtractWitness {
            validators: vec![ValidatorExtractRow {
                effective_balance: eb,
                validator_root: val_root,
            }],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        // ── SSZ (merkleize 1 chunk with limit=2 to force a 2-leaf
        //    tree). Produces 1 row at depth=0: LEFT=val_root (real,
        //    IS_LEFT_REAL=1) + RIGHT=zero_chunk(0) (padded,
        //    IS_RIGHT_REAL=0). Without `limit`, 1-chunk merkleization
        //    pads to 1 leaf (no hash needed) and emits zero rows. ──
        let ssz_rows = merkleize_witness(&[val_root], Some(2));
        assert_eq!(ssz_rows.len(), 1, "1-leaf merkleize with limit=2 must produce 1 hash row");
        let ssz_trace = build_ssz_trace(&ssz_rows, curve);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len());

        // ── Linkages ──
        let l1 = make_finality_validator_extract_linkage_descriptor(
            /* finality */ 0, /* ve */ 1,
        );
        let l2 = make_validator_extract_ssz_linkage_descriptor(
            /* ve */ 1, /* ssz */ 2, SszLeafSide::Left,
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&finality_trace, &finality_cs),
            (&ve_trace, &ve_cs),
            (&ssz_trace, &ssz_cs),
        ];
        let linkages = vec![l1, l2];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the 3-AIR Finality+VE+SSZ chain");
        assert_eq!(proofs.len(), 3);
        assert_eq!(extension.linkage_proofs.len(), 2);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&finality_cs, &ve_cs, &ssz_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(valid, "joint verifier must accept the 3-AIR Finality+VE+SSZ chain");
    }

    /// Tampering test: modify ValidatorExtract's effective_balance for
    /// the single validator while leaving Finality's claim honest. The
    /// L1 multiset (Finality (idx, eff_bal) ↔ VE (idx, eff_bal)) MUST
    /// fail — Finality's tuple doesn't appear in VE's table — so
    /// joint_prove returns an error from witness building (the
    /// multiset-equality check in `compute_cross_air_logup_witness`).
    ///
    /// Validates the soundness of the L1 binding: a malicious prover
    /// cannot fake an alternate balance for a validator without
    /// breaking the cross-AIR LogUp.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove tampering test (~15s); run with --release --ignored"]
    fn joint_prove_finality_validator_extract_rejects_tampered_eb() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eb = 32_000_000_000u64;
        let eb_tampered = 16_000_000_000u64; // half — different multiset entry

        let finality_w = FinalityWitness {
            validators: vec![(eb, 1), (eb, 1), (eb, 1), (eb, 0)],
            total_active_balance_gwei: 4 * eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // Tamper: ValidatorExtract row 0's effective_balance is
        // half of what Finality claims. L1 will fail multiset equality.
        let mk_root = |seed: u8| {
            let mut r = [0u8; 32];
            for (i, b) in r.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            r
        };
        let ve_w = ValidatorExtractWitness {
            validators: vec![
                ValidatorExtractRow {
                    effective_balance: eb_tampered, // ← TAMPERED
                    validator_root: mk_root(0x10),
                },
                ValidatorExtractRow {
                    effective_balance: eb,
                    validator_root: mk_root(0x20),
                },
                ValidatorExtractRow {
                    effective_balance: eb,
                    validator_root: mk_root(0x30),
                },
                ValidatorExtractRow {
                    effective_balance: eb,
                    validator_root: mk_root(0x40),
                },
            ],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        let l1 = make_finality_validator_extract_linkage_descriptor(0, 1);

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&finality_trace, &finality_cs), (&ve_trace, &ve_cs)];
        let linkages = vec![l1];

        // joint_prove must FAIL because the witness builder detects the
        // multiset mismatch. The error message is from
        // `compute_cross_air_logup_witness`.
        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered Finality+VE witness"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}", err
        );
    }

    /// **N=4 validators** Finality+VE+SSZ chain test exercising the
    /// new leaf-depth gating (#121). With 4 validators in a 4-leaf
    /// SSZ tree, the merkleization produces 3 hash rows:
    ///   - row 0 at depth=0: left=val_root_0, right=val_root_1
    ///   - row 1 at depth=0: left=val_root_2, right=val_root_3
    ///   - row 2 at depth=1: left=parent_of_pair_0, right=parent_of_pair_1
    ///
    /// Without the new `IS_LEFT_VALIDATOR_LEAF` / `IS_RIGHT_VALIDATOR_LEAF`
    /// gating (which fires only at `LAYER_DEPTH = 0`), the LEFT linkage's
    /// B-side multiset would include `parent_of_pair_0` (the depth=1
    /// row's LEFT chunk) — but A-side has no such tuple. Multiset
    /// equality would fail. The leaf-only gating excludes the depth=1
    /// row's LEFT/RIGHT chunks, restoring multiset balance.
    #[test]
    #[ignore = "slow: 3-AIR + 3-linkage joint_prove with 4 validators (~3 min); run with --release --ignored"]
    fn joint_prove_finality_ve_ssz_four_validators_leaf_depth_gating() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as build_ssz_trace, SszConstraintSystem,
        };
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            make_validator_extract_ssz_linkage_descriptor,
            SszLeafSide, ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Four validators with distinct effective_balances and roots.
        let ebs = [32_000_000_000u64, 16_000_000_000, 8_000_000_000, 4_000_000_000];
        let mk_root = |seed: u8| {
            let mut r = [0u8; 32];
            for (i, b) in r.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            r
        };
        let val_roots: [[u8; 32]; 4] =
            [mk_root(0x10), mk_root(0x20), mk_root(0x30), mk_root(0x40)];

        // ── Finality (4 validators, all attesting) ──
        let finality_w = FinalityWitness {
            validators: ebs.iter().map(|&eb| (eb, 1)).collect(),
            total_active_balance_gwei: ebs.iter().sum(),
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // ── ValidatorExtract (4 validators) ──
        let ve_w = ValidatorExtractWitness {
            validators: (0..4)
                .map(|i| ValidatorExtractRow {
                    effective_balance: ebs[i],
                    validator_root: val_roots[i],
                })
                .collect(),
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        // ── SSZ (4-leaf merkleize → 3 hash rows: 2 at depth=0, 1 at depth=1) ──
        let ssz_rows = merkleize_witness(&val_roots, None);
        assert_eq!(ssz_rows.len(), 3, "4-leaf merkleize must produce 3 hash rows");
        let ssz_trace = build_ssz_trace(&ssz_rows, curve);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len());

        // ── Linkages: L1 + L2(LEFT) + L3(RIGHT) ──
        // - L2 (LEFT): A side = {val 0, val 2} (parity-even validators).
        //              B side = SSZ rows {0,1} LEFT chunks = {val_0, val_2}.
        //              Multiset balances; depth=1 row's LEFT (parent hash)
        //              is excluded by the new IS_LEFT_VALIDATOR_LEAF gating.
        // - L3 (RIGHT): symmetric for {val 1, val 3} ↔ {val_1, val_3}.
        let l1 = make_finality_validator_extract_linkage_descriptor(0, 1);
        let l2_left = make_validator_extract_ssz_linkage_descriptor(
            1, 2, SszLeafSide::Left,
        );
        let l3_right = make_validator_extract_ssz_linkage_descriptor(
            1, 2, SszLeafSide::Right,
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&finality_trace, &finality_cs),
            (&ve_trace, &ve_cs),
            (&ssz_trace, &ssz_cs),
        ];
        let linkages = vec![l1, l2_left, l3_right];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for the 4-validator Finality+VE+SSZ chain with leaf-depth gating",
        );
        assert_eq!(proofs.len(), 3);
        assert_eq!(extension.linkage_proofs.len(), 3);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&finality_cs, &ve_cs, &ssz_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the 4-validator Finality+VE+SSZ chain"
        );
    }

    /// Multi-validator parity-aware tampering test: confirms a
    /// malicious prover that swaps the parity assignment (e.g., claims
    /// validator 0 is on the RIGHT side when it's at index 0/even) is
    /// caught by the cross-AIR LogUp witness builder.
    ///
    /// The honest assignment: validator 0 → LEFT, validator 1 → RIGHT.
    /// Tamper: swap the validator_root values in ValidatorExtract so
    /// the (effective_balance, validator_root) pair binds the SAME
    /// effective_balance to a different root than the honest one. The
    /// VE↔SSZ LEFT linkage will fail because A's val_root_0_at_left_position
    /// no longer matches SSZ's actual LEFT chunk (val_root_0).
    #[test]
    #[ignore = "slow: 3-AIR + 3-linkage joint_prove tampering test (~10s); run with --release --ignored"]
    fn joint_prove_finality_ve_ssz_two_validators_rejects_swapped_roots() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as build_ssz_trace, SszConstraintSystem,
        };
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_validator_extract_ssz_linkage_descriptor,
            SszLeafSide, ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eb0 = 32_000_000_000u64;
        let eb1 = 16_000_000_000u64;
        let mk_root = |seed: u8| {
            let mut r = [0u8; 32];
            for (i, b) in r.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            r
        };
        let val_root_0 = mk_root(0x10);
        let val_root_1 = mk_root(0x20);

        // SSZ honestly merkleizes (val_root_0, val_root_1).
        let ssz_rows = merkleize_witness(&[val_root_0, val_root_1], None);
        let ssz_trace = build_ssz_trace(&ssz_rows, curve);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len());

        // VE TAMPERED: validator 0 stores val_root_1, validator 1 stores val_root_0.
        // The LEFT linkage A side expects val_root_0 at the IS_LEFT_VALIDATOR=1 row
        // (validator 0), but it now reads val_root_1 — multiset mismatch.
        let ve_w = ValidatorExtractWitness {
            validators: vec![
                ValidatorExtractRow {
                    effective_balance: eb0,
                    validator_root: val_root_1, // ← TAMPERED (should be val_root_0)
                },
                ValidatorExtractRow {
                    effective_balance: eb1,
                    validator_root: val_root_0, // ← TAMPERED (should be val_root_1)
                },
            ],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        let l_left = make_validator_extract_ssz_linkage_descriptor(
            0, 1, SszLeafSide::Left,
        );
        let l_right = make_validator_extract_ssz_linkage_descriptor(
            0, 1, SszLeafSide::Right,
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&ve_trace, &ve_cs), (&ssz_trace, &ssz_cs)];
        let linkages = vec![l_left, l_right];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject swapped-root multi-validator chain"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}", err
        );
    }

    /// Three-AIR Finality+VE+SSZ joint_prove for **2 validators**,
    /// exercising BOTH the LEFT and RIGHT VE↔SSZ linkages
    /// simultaneously. The new parity-aware A-side selectors
    /// (`COL_IS_LEFT_VALIDATOR` / `COL_IS_RIGHT_VALIDATOR`) ensure each
    /// validator participates in exactly ONE of the two multisets:
    ///   - Validator 0 (parity=0): contributes to LEFT
    ///   - Validator 1 (parity=1): contributes to RIGHT
    /// SSZ row 0 has both chunks real (IS_LEFT_REAL=IS_RIGHT_REAL=1)
    /// since the 2-leaf merkleize hashes (val_root_0, val_root_1).
    ///
    /// Closes #119: validates that the parity-aware selectors enable
    /// multi-validator chains. Without them, the previous IS_REAL
    /// selector counted EACH validator on BOTH sides, breaking
    /// multiset equality the moment N > 1.
    #[test]
    #[ignore = "slow: 3-AIR + 3-linkage joint_prove with 2 validators (~3 min); run with --release --ignored"]
    fn joint_prove_finality_ve_ssz_two_validators_parity_aware() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as build_ssz_trace, SszConstraintSystem,
        };
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            make_validator_extract_ssz_linkage_descriptor,
            SszLeafSide, ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Two validators with distinct effective_balances and roots.
        let eb0 = 32_000_000_000u64;
        let eb1 = 16_000_000_000u64;
        let mk_root = |seed: u8| {
            let mut r = [0u8; 32];
            for (i, b) in r.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            r
        };
        let val_root_0 = mk_root(0x10);
        let val_root_1 = mk_root(0x20);

        // ── Finality (2 validators, both attesting) ──
        let finality_w = FinalityWitness {
            validators: vec![(eb0, 1), (eb1, 1)],
            total_active_balance_gwei: eb0 + eb1,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // ── ValidatorExtract (2 validators) ──
        let ve_w = ValidatorExtractWitness {
            validators: vec![
                ValidatorExtractRow {
                    effective_balance: eb0,
                    validator_root: val_root_0,
                },
                ValidatorExtractRow {
                    effective_balance: eb1,
                    validator_root: val_root_1,
                },
            ],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        // ── SSZ (merkleize 2 chunks → 1 hash row, both leaves real) ──
        let ssz_rows = merkleize_witness(&[val_root_0, val_root_1], None);
        assert_eq!(ssz_rows.len(), 1, "2-leaf merkleize must produce 1 hash row");
        let ssz_trace = build_ssz_trace(&ssz_rows, curve);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len());

        // ── Linkages: L1 + L2(LEFT) + L3(RIGHT) ──
        let l1 = make_finality_validator_extract_linkage_descriptor(0, 1);
        let l2_left = make_validator_extract_ssz_linkage_descriptor(
            1, 2, SszLeafSide::Left,
        );
        let l3_right = make_validator_extract_ssz_linkage_descriptor(
            1, 2, SszLeafSide::Right,
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&finality_trace, &finality_cs),
            (&ve_trace, &ve_cs),
            (&ssz_trace, &ssz_cs),
        ];
        let linkages = vec![l1, l2_left, l3_right];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for the honest 2-validator Finality+VE+SSZ chain",
        );
        assert_eq!(proofs.len(), 3);
        assert_eq!(extension.linkage_proofs.len(), 3);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&finality_cs, &ve_cs, &ssz_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest 2-validator Finality+VE+SSZ chain"
        );
    }

    /// Tampering test for L2 (VE↔SSZ Left): modify SSZ's LEFT chunk
    /// for the (single) hash row while leaving ValidatorExtract's
    /// claimed `validator_root` honest. The L2 multiset (VE roots ↔
    /// SSZ LEFT chunks where IS_LEFT_REAL=1) MUST fail since the
    /// committed root no longer appears in SSZ's leaf chunks.
    ///
    /// Validates the soundness of the L2 binding: a malicious prover
    /// cannot fake an SSZ tree where a validator's leaf differs from
    /// what ValidatorExtract committed, without breaking the
    /// cross-AIR LogUp.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove tampering test (~5s); run with --release --ignored"]
    fn joint_prove_finality_ve_ssz_rejects_tampered_ssz_left_chunk() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::ssz::Chunk;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as build_ssz_trace, SszConstraintSystem,
        };
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            make_validator_extract_ssz_linkage_descriptor,
            SszLeafSide, ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eb = 32_000_000_000u64;
        let mut val_root: Chunk = [0u8; 32];
        for i in 0..32 {
            val_root[i] = 0x10u8.wrapping_add(i as u8);
        }
        // A different chunk that we'll inject into SSZ's LEFT slot.
        let mut tampered: Chunk = [0u8; 32];
        for i in 0..32 {
            tampered[i] = 0xAAu8.wrapping_add(i as u8);
        }

        let finality_w = FinalityWitness {
            validators: vec![(eb, 1)],
            total_active_balance_gwei: eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // ValidatorExtract claims val_root.
        let ve_w = ValidatorExtractWitness {
            validators: vec![ValidatorExtractRow {
                effective_balance: eb,
                validator_root: val_root,
            }],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        // SSZ trace built from `tampered` instead of val_root. The
        // single hash row's LEFT bytes will be `tampered`, not val_root.
        let ssz_rows = merkleize_witness(&[tampered], Some(2));
        let ssz_trace = build_ssz_trace(&ssz_rows, curve);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len());

        let l1 = make_finality_validator_extract_linkage_descriptor(0, 1);
        let l2 = make_validator_extract_ssz_linkage_descriptor(1, 2, SszLeafSide::Left);

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&finality_trace, &finality_cs),
            (&ve_trace, &ve_cs),
            (&ssz_trace, &ssz_cs),
        ];
        let linkages = vec![l1, l2];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered SSZ trace"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}", err
        );
    }

    /// 5-AIR `joint_prove`/`joint_verify` regression — the largest
    /// multi-AIR configuration to date. Extends the 3-AIR Finality
    /// chain (Finality+VE+SSZ) by adding Sha256Extract + bit-level
    /// SHA-256, plus their two linkages, validating the **complete**
    /// Finality-side soundness chain end-to-end through bit-level
    /// SHA-256 algebraic constraints.
    ///
    /// Five AIRs:
    ///   0. Finality (9 cols + 20 LogUp)
    ///   1. ValidatorExtract (35 cols)
    ///   2. SSZ (100 cols + 214 LogUp)
    ///   3. Sha256Extract (97 cols)
    ///   4. bit-level SHA-256 (1154 cols, 128 native rows for sha256_pair)
    ///
    /// Four cross-AIR LogUp linkages active simultaneously:
    ///   - L1: Finality↔VE on `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)`
    ///   - L2: VE↔SSZ Left on validator_root
    ///   - L3: SSZ↔Sha256Extract on `(LEFT||RIGHT, PARENT)` (96-byte tuple)
    ///   - L4: Sha256Extract↔bit-level SHA-256 on byte aggregator
    ///     (96-byte tuple)
    ///
    /// Soundness chain end-to-end:
    /// 1. L1 pins Finality's `(idx, eff_bal)` to a real ValidatorExtract row.
    /// 2. L2 pins ValidatorExtract's `validator_root` to a real SSZ leaf chunk.
    /// 3. L3 pins SSZ's `parent` column to be `sha256(LEFT || RIGHT)`
    ///    according to Sha256Extract's claimed digest.
    /// 4. L4 + bit-level SHA-256's algebraic input/output bindings (#87)
    ///    pin Sha256Extract's claimed digest to be the actual
    ///    `sha256(LEFT || RIGHT)` computed by the bit-level prover.
    ///
    /// **Net result**: the validator's `effective_balance` claimed by
    /// Finality is algebraically pinned all the way down to the
    /// validator's hash being part of the SSZ merkle tree, with the
    /// SSZ pair-hashing computed by real bit-level SHA-256 constraints.
    /// (The remaining residual gap — proving `validator_root =
    /// hash_tree_root(validator_data)` so eff_bal can't be decoupled
    /// from val_root — would require an additional gadget AIR for
    /// validator container hashing, mirroring `EvmCreateRlpAir` for
    /// CREATE.)
    #[test]
    #[ignore = "very slow: 5-AIR + 4-linkage joint_prove with bit-level \
                SHA-256 (~15 min); run with --release --ignored"]
    fn joint_prove_finality_ssz_chain_e2e_with_bit_level_sha256() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::sha256::{sha256_witness, NUM_ROUNDS};
        use crate::sha256_air;
        use crate::sha256_constraints::Sha256ConstraintSystem;
        use crate::sha256_extract::{
            build_trace_polynomials as build_sha_extract_trace,
            make_sha256_extract_sha256_linkage_descriptor,
            make_ssz_sha256_extract_linkage_descriptor,
            Sha256ExtractConstraintSystem, Sha256ExtractWitness,
        };
        use crate::ssz::ZERO_CHUNK;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as build_ssz_trace, SszConstraintSystem,
        };
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            make_validator_extract_ssz_linkage_descriptor,
            SszLeafSide, ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eb = 32_000_000_000u64;
        let mut val_root = [0u8; 32];
        for i in 0..32 {
            val_root[i] = 0x10u8.wrapping_add(i as u8);
        }

        // ── Finality (1 attesting validator) ──
        let finality_w = FinalityWitness {
            validators: vec![(eb, 1)],
            total_active_balance_gwei: eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // ── ValidatorExtract (same eff_bal + val_root) ──
        let ve_w = ValidatorExtractWitness {
            validators: vec![ValidatorExtractRow {
                effective_balance: eb,
                validator_root: val_root,
            }],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        // ── SSZ (1 chunk forced to 2-leaf tree) ──
        let ssz_rows = merkleize_witness(&[val_root], Some(2));
        let ssz_trace = build_ssz_trace(&ssz_rows, curve);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len());

        // The SSZ row's hash invocation: sha256_pair(val_root, ZERO_CHUNK).
        let zero_chunk_0 = ZERO_CHUNK;

        // ── Sha256Extract (matches the SSZ row's hash invocation) ──
        let extract_w =
            Sha256ExtractWitness::from_pair_inputs(&[(val_root, zero_chunk_0)]);
        let extract_trace = build_sha_extract_trace(&extract_w, curve);
        let extract_cs = Sha256ExtractConstraintSystem::new(extract_trace.num_rows);

        // ── Bit-level SHA-256 (one sha256_pair invocation = 2 blocks) ──
        let mut input64 = [0u8; 64];
        input64[..32].copy_from_slice(&val_root);
        input64[32..].copy_from_slice(&zero_chunk_0);
        let ht = sha256_witness(&input64);
        let num_sha_rows = ht.blocks.len() * NUM_ROUNDS;
        let sha_padded = crate::trace::nearest_power_of_two(num_sha_rows.max(1));
        let mut sha_columns =
            sha256_air::populate_trace_from_hash_with_invocation_bytes(&ht, &input64, curve);
        for col in sha_columns.iter_mut() {
            if col.len() < sha_padded {
                col.resize(sha_padded, Scalar::zero(curve));
            }
        }
        let sha_polys: Vec<crate::trace::Polynomial> = sha_columns
            .into_iter()
            .map(|evals| crate::trace::Polynomial {
                evaluations: evals,
                degree: num_sha_rows,
            })
            .collect();
        let sha_trace = TracePolynomials {
            columns: sha_polys,
            num_rows: num_sha_rows,
            padded_size: sha_padded as u64,
            curve,
        };
        let sha_cs = Sha256ConstraintSystem::new(num_sha_rows);

        // ── Linkages ──
        let l1 = make_finality_validator_extract_linkage_descriptor(
            /* finality */ 0, /* ve */ 1,
        );
        let l2 = make_validator_extract_ssz_linkage_descriptor(
            /* ve */ 1, /* ssz */ 2, SszLeafSide::Left,
        );
        let l3 = make_ssz_sha256_extract_linkage_descriptor(
            /* ssz */ 2, /* extract */ 3,
        );
        let l4 = make_sha256_extract_sha256_linkage_descriptor(
            /* extract */ 3, /* sha256 */ 4,
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&finality_trace, &finality_cs),
            (&ve_trace, &ve_cs),
            (&ssz_trace, &ssz_cs),
            (&extract_trace, &extract_cs),
            (&sha_trace, &sha_cs),
        ];
        let linkages = vec![l1, l2, l3, l4];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the 5-AIR Finality chain");
        assert_eq!(proofs.len(), 5);
        assert_eq!(extension.linkage_proofs.len(), 4);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&finality_cs, &ve_cs, &ssz_cs, &extract_cs, &sha_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the 5-AIR Finality chain through bit-level SHA-256"
        );
    }

    /// Multi-validator (N=2) version of
    /// `joint_prove_finality_ssz_chain_e2e_with_bit_level_sha256`.
    ///
    /// With 2 validators the SSZ tree has both LEFT and RIGHT chunks
    /// real (val_root_0 and val_root_1, both contributed by validators).
    /// The new parity-aware A-side selectors (#119) ensure each
    /// validator participates in exactly ONE of LEFT/RIGHT — without
    /// them, the multiset would be unbalanced because each validator
    /// would be counted on BOTH the LEFT and RIGHT linkages.
    ///
    /// Validates the FULL 5-AIR multi-validator soundness chain through
    /// bit-level SHA-256 with three VE↔SSZ linkages (LEFT + RIGHT) plus
    /// the SSZ↔Sha256Extract↔SHA-256 binding chain.
    #[test]
    #[ignore = "very slow: 5-AIR + 5-linkage joint_prove with bit-level \
                SHA-256 and 2 validators (~16 min); run with --release --ignored"]
    fn joint_prove_finality_ssz_chain_e2e_with_bit_level_sha256_two_validators() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::sha256::{sha256_witness, NUM_ROUNDS};
        use crate::sha256_air;
        use crate::sha256_constraints::Sha256ConstraintSystem;
        use crate::sha256_extract::{
            build_trace_polynomials as build_sha_extract_trace,
            make_sha256_extract_sha256_linkage_descriptor,
            make_ssz_sha256_extract_linkage_descriptor,
            Sha256ExtractConstraintSystem, Sha256ExtractWitness,
        };
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as build_ssz_trace, SszConstraintSystem,
        };
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            make_validator_extract_ssz_linkage_descriptor,
            SszLeafSide, ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eb0 = 32_000_000_000u64;
        let eb1 = 16_000_000_000u64;
        let mk_root = |seed: u8| {
            let mut r = [0u8; 32];
            for (i, b) in r.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            r
        };
        let val_root_0 = mk_root(0x10);
        let val_root_1 = mk_root(0x20);

        // ── Finality (2 validators, both attesting) ──
        let finality_w = FinalityWitness {
            validators: vec![(eb0, 1), (eb1, 1)],
            total_active_balance_gwei: eb0 + eb1,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        // ── ValidatorExtract (2 validators) ──
        let ve_w = ValidatorExtractWitness {
            validators: vec![
                ValidatorExtractRow {
                    effective_balance: eb0,
                    validator_root: val_root_0,
                },
                ValidatorExtractRow {
                    effective_balance: eb1,
                    validator_root: val_root_1,
                },
            ],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        // ── SSZ (2-leaf tree, both leaves real) ──
        let ssz_rows = merkleize_witness(&[val_root_0, val_root_1], None);
        assert_eq!(ssz_rows.len(), 1, "2-leaf merkleize must produce 1 hash row");
        let ssz_trace = build_ssz_trace(&ssz_rows, curve);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len());

        // ── Sha256Extract (1 invocation: sha256_pair(val_root_0, val_root_1)) ──
        let extract_w =
            Sha256ExtractWitness::from_pair_inputs(&[(val_root_0, val_root_1)]);
        let extract_trace = build_sha_extract_trace(&extract_w, curve);
        let extract_cs = Sha256ExtractConstraintSystem::new(extract_trace.num_rows);

        // ── Bit-level SHA-256 (1 sha256_pair invocation = 2 blocks) ──
        let mut input64 = [0u8; 64];
        input64[..32].copy_from_slice(&val_root_0);
        input64[32..].copy_from_slice(&val_root_1);
        let ht = sha256_witness(&input64);
        let num_sha_rows = ht.blocks.len() * NUM_ROUNDS;
        let sha_padded = crate::trace::nearest_power_of_two(num_sha_rows.max(1));
        let mut sha_columns =
            sha256_air::populate_trace_from_hash_with_invocation_bytes(&ht, &input64, curve);
        for col in sha_columns.iter_mut() {
            if col.len() < sha_padded {
                col.resize(sha_padded, Scalar::zero(curve));
            }
        }
        let sha_polys: Vec<crate::trace::Polynomial> = sha_columns
            .into_iter()
            .map(|evals| crate::trace::Polynomial {
                evaluations: evals,
                degree: num_sha_rows,
            })
            .collect();
        let sha_trace = TracePolynomials {
            columns: sha_polys,
            num_rows: num_sha_rows,
            padded_size: sha_padded as u64,
            curve,
        };
        let sha_cs = Sha256ConstraintSystem::new(num_sha_rows);

        // ── Linkages (5: L1 + L2_LEFT + L2_RIGHT + L3 + L4) ──
        let l1 = make_finality_validator_extract_linkage_descriptor(0, 1);
        let l2_left = make_validator_extract_ssz_linkage_descriptor(
            1, 2, SszLeafSide::Left,
        );
        let l2_right = make_validator_extract_ssz_linkage_descriptor(
            1, 2, SszLeafSide::Right,
        );
        let l3 = make_ssz_sha256_extract_linkage_descriptor(2, 3);
        let l4 = make_sha256_extract_sha256_linkage_descriptor(3, 4);

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&finality_trace, &finality_cs),
            (&ve_trace, &ve_cs),
            (&ssz_trace, &ssz_cs),
            (&extract_trace, &extract_cs),
            (&sha_trace, &sha_cs),
        ];
        let linkages = vec![l1, l2_left, l2_right, l3, l4];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for the 5-AIR multi-validator Finality chain",
        );
        assert_eq!(proofs.len(), 5);
        assert_eq!(extension.linkage_proofs.len(), 5);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&finality_cs, &ve_cs, &ssz_cs, &extract_cs, &sha_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the 5-AIR multi-validator Finality chain through bit-level SHA-256"
        );
    }
}
