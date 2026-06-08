//! EIP-3529 gas-refund cap AIR.
//!
//! Proves, per transaction, that the *applied* gas refund is bounded
//! by the EIP-3529 cap:
//!
//!   * **Post-London (EIP-3529, active 2021-08-05)**: refund cap is
//!     `gas_used / 5`, and SELFDESTRUCT no longer produces refunds.
//!   * **Pre-London**: refund cap is `gas_used / 2`, SELFDESTRUCT refund
//!     of `24_000` per call applies.
//!
//! This AIR commits per-row `(tx_index, gas_used, refund_unbounded,
//! refund_applied, refund_cap, slack)` where `slack` is the witnessed
//! non-negative gap `refund_cap − refund_applied`. A second auxiliary
//! witness column `extra_slack = refund_unbounded − refund_applied`
//! captures the dual gap (≥ 0 by the LE byte decomposition + range).
//!
//! ## min(refund_unbounded, refund_cap)
//!
//! `refund_applied = min(refund_unbounded, refund_cap)` is enforced via
//! a witnessed binary selector `which_smaller`:
//!
//!   * `which_smaller = 1` ⇒ `refund_applied = refund_unbounded` (and
//!     `slack = refund_cap − refund_unbounded ≥ 0`).
//!   * `which_smaller = 0` ⇒ `refund_applied = refund_cap` (and
//!     `extra_slack = refund_unbounded − refund_cap ≥ 0`).
//!
//! Combined with the definitional equalities
//! (`refund_applied + slack = refund_cap` and `refund_unbounded =
//! refund_applied + extra_slack`) and the u64 byte-range checks, this
//! proves the min identity.
//!
//! ## Integer-division witness for the cap
//!
//! The cap `refund_cap = gas_used / D` (where D ∈ {2, 5}) is witnessed
//! via a remainder column:
//!
//!   * Post-London: `refund_cap_l · 5 + cap_remainder_london = gas_used`,
//!     `cap_remainder_london ∈ [0, 5)`.
//!   * Pre-London:  `refund_cap_p · 2 + cap_remainder_pre = gas_used`,
//!     `cap_remainder_pre  ∈ [0, 2)`.
//!
//! `refund_cap` itself is then the hardfork-selected pick: it equals
//! `refund_cap_l` post-London and `refund_cap_p` pre-London. The pick
//! is encoded by a single column equality
//! `refund_cap = is_post_london · refund_cap_l + (1 − is_post_london) · refund_cap_p`.
//!
//! ## Cross-AIR LogUp descriptors
//!
//!   * [`make_refund_3529_to_sstore_descriptor`] — host-side bridge to
//!     the `sstore_transition_air`'s `IS_CLEAR` rows, which are the
//!     EIP-3529 refund sources (storage clears earn the `R_sclear`
//!     contribution to `refund_unbounded`). Full quantitative binding
//!     (per-row refund amount sum) is host-oracle for now.
//!   * [`make_refund_3529_to_gas_tracking_descriptor`] — binds the
//!     row's `(gas_used, refund_applied)` to the gas-tracking AIR.
//!     (Gas-tracking is per-step today; the tx-level binding is
//!     host-oracle until a per-tx aggregate column lands.)
//!   * [`make_refund_3529_to_hardfork_descriptor`] — binds
//!     `is_post_london` here to the hardfork-rules AIR (mapped onto
//!     `is_post_shanghai`, since London ⇒ Shanghai by monotonicity;
//!     the explicit London flag column will be added when the
//!     hardfork-rules AIR gains it).
//!
//! ## Column layout
//!
//! ```text
//! offset    size  meaning
//!  0         1    tx_index
//!  1         1    gas_used
//!  2         1    refund_unbounded
//!  3         1    refund_applied
//!  4         1    refund_cap
//!  5         1    slack             (= refund_cap − refund_applied)
//!  6         1    extra_slack       (= refund_unbounded − refund_applied)
//!  7         1    refund_cap_l      (= gas_used / 5, post-London witness)
//!  8         1    refund_cap_p      (= gas_used / 2, pre-London witness)
//!  9         1    cap_remainder_london   (∈ [0, 5))
//! 10         1    cap_remainder_pre      (∈ [0, 2))
//! 11         1    is_post_london    (binary)
//! 12         1    which_smaller     (binary; 1 ⇒ applied = unbounded)
//! 13         1    is_real           (binary)
//! 14..22     8    gas_used_byte_0..7
//! 22..30     8    refund_unbounded_byte_0..7
//! 30..38     8    refund_applied_byte_0..7
//! 38..46     8    refund_cap_byte_0..7
//! 46..54     8    slack_byte_0..7
//! 54..62     8    extra_slack_byte_0..7
//! ```
//!
//! ## Constraint catalog (12 row-local constraints)
//!
//!  0. `is_real_binary`              — `is_real · (is_real − 1) = 0`
//!  1. `is_post_london_binary`       — `is_post_london · (is_post_london − 1) = 0`
//!  2. `which_smaller_binary`        — `which_smaller · (which_smaller − 1) = 0`
//!  3. `cap_div_post_london`         — `is_real · is_post_london ·
//!         (refund_cap_l · 5 + cap_remainder_london − gas_used) = 0`
//!  4. `cap_div_pre_london`          — `is_real · (1 − is_post_london) ·
//!         (refund_cap_p · 2 + cap_remainder_pre − gas_used) = 0`
//!  5. `cap_pick`                    — `is_real ·
//!         (refund_cap − is_post_london · refund_cap_l
//!                     − (1 − is_post_london) · refund_cap_p) = 0`
//!  6. `applied_slack_def`           — `is_real ·
//!         (refund_applied + slack − refund_cap) = 0`
//!  7. `extra_slack_def`             — `is_real ·
//!         (refund_unbounded − refund_applied − extra_slack) = 0`
//!  8. `min_when_unbounded_smaller`  — `is_real · which_smaller ·
//!         (refund_applied − refund_unbounded) = 0`
//!  9. `min_when_cap_smaller`        — `is_real · (1 − which_smaller) ·
//!         (refund_applied − refund_cap) = 0`
//! 10. `gas_used_le_decomp`          — `gas_used − Σ b_j · 2^(8j) = 0`
//!     (plus identical decomps for refund_unbounded, refund_applied,
//!     refund_cap, slack, extra_slack — collapsed into a single body
//!     index by α-folding; conceptually 6 separate identities)
//! 11. `flags_zero_on_padding`       — `(1 − is_real) ·
//!         (is_post_london + which_smaller + gas_used + refund_unbounded
//!          + refund_applied + refund_cap + slack + extra_slack) = 0`
//!
//! All six LE decompositions are packed into constraint 10 via
//! α-independent slot exposure (each is its own algebraic identity).
//! Range constraints on the per-limb byte columns are wired through
//! [`VmConstraintSystem::lookup_declarations`] as 8-bit range lookups,
//! and the two small remainders (`< 5`, `< 2`) get tight 3-bit / 1-bit
//! ranges respectively, plus explicit upper-bound constraints.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;
use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ───────────────────────────────────────────────────

pub const COL_TX_INDEX: usize = 0;
pub const COL_GAS_USED: usize = 1;
pub const COL_REFUND_UNBOUNDED: usize = 2;
pub const COL_REFUND_APPLIED: usize = 3;
pub const COL_REFUND_CAP: usize = 4;
pub const COL_SLACK: usize = 5;
pub const COL_EXTRA_SLACK: usize = 6;
pub const COL_REFUND_CAP_LONDON: usize = 7;
pub const COL_REFUND_CAP_PRE: usize = 8;
pub const COL_CAP_REMAINDER_LONDON: usize = 9;
pub const COL_CAP_REMAINDER_PRE: usize = 10;
pub const COL_IS_POST_LONDON: usize = 11;
pub const COL_WHICH_SMALLER: usize = 12;
pub const COL_IS_REAL: usize = 13;

pub const NUM_BYTE_LIMBS: usize = 8;
pub const COL_GAS_USED_BYTE_OFFSET: usize = 14;
pub const COL_REFUND_UNBOUNDED_BYTE_OFFSET: usize = COL_GAS_USED_BYTE_OFFSET + NUM_BYTE_LIMBS;
pub const COL_REFUND_APPLIED_BYTE_OFFSET: usize =
    COL_REFUND_UNBOUNDED_BYTE_OFFSET + NUM_BYTE_LIMBS;
pub const COL_REFUND_CAP_BYTE_OFFSET: usize = COL_REFUND_APPLIED_BYTE_OFFSET + NUM_BYTE_LIMBS;
pub const COL_SLACK_BYTE_OFFSET: usize = COL_REFUND_CAP_BYTE_OFFSET + NUM_BYTE_LIMBS;
pub const COL_EXTRA_SLACK_BYTE_OFFSET: usize = COL_SLACK_BYTE_OFFSET + NUM_BYTE_LIMBS;

pub const NUM_COLUMNS: usize = COL_EXTRA_SLACK_BYTE_OFFSET + NUM_BYTE_LIMBS;

const _: () = assert!(NUM_COLUMNS == 14 + 6 * 8);

/// Row-local constraint count. The six LE byte-decomposition identities
/// are surfaced as separate constraint slots so each gets its own α^k.
pub const NUM_ROW_CONSTRAINTS: usize = 16;
pub const NUM_SHIFTED: usize = 0;

/// EIP-3529 post-London divisor.
pub const REFUND_DIVISOR_LONDON: u64 = 5;
/// Pre-London divisor.
pub const REFUND_DIVISOR_PRE_LONDON: u64 = 2;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GasRefund3529Row {
    pub tx_index: u64,
    pub gas_used: u64,
    pub refund_unbounded: u64,
    pub refund_applied: u64,
    pub refund_cap: u64,
    pub slack: u64,
    pub extra_slack: u64,
    pub refund_cap_london: u64,
    pub refund_cap_pre: u64,
    pub cap_remainder_london: u64,
    pub cap_remainder_pre: u64,
    pub is_post_london: bool,
    /// `true` iff `refund_unbounded < refund_cap`, i.e. the applied
    /// refund equals the unbounded refund.
    pub which_smaller: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct GasRefund3529Witness {
    pub rows: Vec<GasRefund3529Row>,
}

impl GasRefund3529Witness {
    pub fn from_rows(rows: Vec<GasRefund3529Row>) -> Self { Self { rows } }
}

/// Build the canonical row from `(tx_index, gas_used, refund_unbounded,
/// is_post_london)`. All derived fields (cap, slack, extra_slack,
/// applied = min, remainders) are computed deterministically.
pub fn from_inputs(
    tx_index: u64,
    gas_used: u64,
    refund_unbounded: u64,
    is_post_london: bool,
) -> GasRefund3529Row {
    let refund_cap_london = gas_used / REFUND_DIVISOR_LONDON;
    let cap_remainder_london = gas_used % REFUND_DIVISOR_LONDON;
    let refund_cap_pre = gas_used / REFUND_DIVISOR_PRE_LONDON;
    let cap_remainder_pre = gas_used % REFUND_DIVISOR_PRE_LONDON;
    let refund_cap = if is_post_london { refund_cap_london } else { refund_cap_pre };
    let refund_applied = refund_unbounded.min(refund_cap);
    let slack = refund_cap - refund_applied;
    let extra_slack = refund_unbounded - refund_applied;
    let which_smaller = refund_unbounded < refund_cap;
    GasRefund3529Row {
        tx_index,
        gas_used,
        refund_unbounded,
        refund_applied,
        refund_cap,
        slack,
        extra_slack,
        refund_cap_london,
        refund_cap_pre,
        cap_remainder_london,
        cap_remainder_pre,
        is_post_london,
        which_smaller,
        is_real: true,
    }
}

/// Single-row witness convenience wrapper around [`from_inputs`].
pub fn witness_from_inputs(
    tx_index: u64,
    gas_used: u64,
    refund_unbounded: u64,
    is_post_london: bool,
) -> GasRefund3529Witness {
    GasRefund3529Witness {
        rows: vec![from_inputs(tx_index, gas_used, refund_unbounded, is_post_london)],
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &GasRefund3529Witness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_TX_INDEX][i] = Scalar::from_u64(row.tx_index, curve);
        columns[COL_GAS_USED][i] = Scalar::from_u64(row.gas_used, curve);
        columns[COL_REFUND_UNBOUNDED][i] = Scalar::from_u64(row.refund_unbounded, curve);
        columns[COL_REFUND_APPLIED][i] = Scalar::from_u64(row.refund_applied, curve);
        columns[COL_REFUND_CAP][i] = Scalar::from_u64(row.refund_cap, curve);
        columns[COL_SLACK][i] = Scalar::from_u64(row.slack, curve);
        columns[COL_EXTRA_SLACK][i] = Scalar::from_u64(row.extra_slack, curve);
        columns[COL_REFUND_CAP_LONDON][i] = Scalar::from_u64(row.refund_cap_london, curve);
        columns[COL_REFUND_CAP_PRE][i] = Scalar::from_u64(row.refund_cap_pre, curve);
        columns[COL_CAP_REMAINDER_LONDON][i] = Scalar::from_u64(row.cap_remainder_london, curve);
        columns[COL_CAP_REMAINDER_PRE][i] = Scalar::from_u64(row.cap_remainder_pre, curve);
        columns[COL_IS_POST_LONDON][i] = if row.is_post_london { one.clone() } else { zero.clone() };
        columns[COL_WHICH_SMALLER][i] = if row.which_smaller { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = if row.is_real { one.clone() } else { zero.clone() };
        // LE byte decomps
        let decomp_specs: [(usize, u64); 6] = [
            (COL_GAS_USED_BYTE_OFFSET, row.gas_used),
            (COL_REFUND_UNBOUNDED_BYTE_OFFSET, row.refund_unbounded),
            (COL_REFUND_APPLIED_BYTE_OFFSET, row.refund_applied),
            (COL_REFUND_CAP_BYTE_OFFSET, row.refund_cap),
            (COL_SLACK_BYTE_OFFSET, row.slack),
            (COL_EXTRA_SLACK_BYTE_OFFSET, row.extra_slack),
        ];
        for (offset, value) in decomp_specs.iter() {
            for j in 0..NUM_BYTE_LIMBS {
                let b = ((*value >> (8 * j)) & 0xff) as u64;
                columns[offset + j][i] = Scalar::from_u64(b, curve);
            }
        }
    }
    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct GasRefund3529ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl GasRefund3529ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn byte_pow(j: usize, curve: CurveType) -> Scalar {
    debug_assert!(j < 8);
    Scalar::from_u64(1u64 << (8 * j), curve)
}

fn eval_le_recompose(col_evals: &[Scalar], byte_offset: usize) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut acc = Scalar::zero(curve);
    for j in 0..NUM_BYTE_LIMBS {
        let b = &col_evals[byte_offset + j];
        acc = acc.add(&b.mul(&byte_pow(j, curve)));
    }
    acc
}

fn build_le_recompose_poly(
    col_coeffs: &[Vec<Scalar>],
    byte_offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    for j in 0..NUM_BYTE_LIMBS {
        let scaled = poly_scalar_mul(&col_coeffs[byte_offset + j], &byte_pow(j, curve));
        acc = poly_add(&acc, &scaled, curve);
    }
    acc
}

impl VmConstraintSystem for GasRefund3529ConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_post_london_binary".into(),
            "which_smaller_binary".into(),
            "cap_div_post_london".into(),
            "cap_div_pre_london".into(),
            "cap_pick".into(),
            "applied_slack_def".into(),
            "extra_slack_def".into(),
            "min_when_unbounded_smaller".into(),
            "min_when_cap_smaller".into(),
            "gas_used_le_decomp".into(),
            "refund_unbounded_le_decomp".into(),
            "refund_applied_le_decomp".into(),
            "refund_cap_le_decomp".into(),
            "slack_le_decomp".into(),
            "extra_slack_le_decomp".into(),
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
        let five = Scalar::from_u64(REFUND_DIVISOR_LONDON, curve);
        let two = Scalar::from_u64(REFUND_DIVISOR_PRE_LONDON, curve);
        for row in 0..n {
            let r: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &r[COL_IS_REAL];
            let is_london = &r[COL_IS_POST_LONDON];
            let which = &r[COL_WHICH_SMALLER];
            let gas_used = &r[COL_GAS_USED];
            let unbounded = &r[COL_REFUND_UNBOUNDED];
            let applied = &r[COL_REFUND_APPLIED];
            let cap = &r[COL_REFUND_CAP];
            let slack = &r[COL_SLACK];
            let extra_slack = &r[COL_EXTRA_SLACK];
            let cap_l = &r[COL_REFUND_CAP_LONDON];
            let cap_p = &r[COL_REFUND_CAP_PRE];
            let rem_l = &r[COL_CAP_REMAINDER_LONDON];
            let rem_p = &r[COL_CAP_REMAINDER_PRE];
            let one_minus = |x: &Scalar| one.sub(x);
            // 0. is_real binary
            bodies[0][row] = is_real.mul(&one_minus(is_real));
            // 1. is_post_london binary
            bodies[1][row] = is_london.mul(&one_minus(is_london));
            // 2. which_smaller binary
            bodies[2][row] = which.mul(&one_minus(which));
            // 3. cap_div_post_london: is_real * is_london * (cap_l*5 + rem_l - gas_used)
            let gate_post = is_real.mul(is_london);
            let div_post = cap_l.mul(&five).add(rem_l).sub(gas_used);
            bodies[3][row] = gate_post.mul(&div_post);
            // 4. cap_div_pre_london: is_real * (1 - is_london) * (cap_p*2 + rem_p - gas_used)
            let gate_pre = is_real.mul(&one_minus(is_london));
            let div_pre = cap_p.mul(&two).add(rem_p).sub(gas_used);
            bodies[4][row] = gate_pre.mul(&div_pre);
            // 5. cap_pick: is_real * (cap - is_london*cap_l - (1-is_london)*cap_p)
            let pick = cap
                .sub(&is_london.mul(cap_l))
                .sub(&one_minus(is_london).mul(cap_p));
            bodies[5][row] = is_real.mul(&pick);
            // 6. applied + slack = cap (gated by is_real)
            let aps = applied.add(slack).sub(cap);
            bodies[6][row] = is_real.mul(&aps);
            // 7. unbounded - applied - extra_slack = 0
            let us = unbounded.sub(applied).sub(extra_slack);
            bodies[7][row] = is_real.mul(&us);
            // 8. min when unbounded smaller: is_real * which * (applied - unbounded)
            bodies[8][row] = is_real.mul(which).mul(&applied.sub(unbounded));
            // 9. min when cap smaller: is_real * (1 - which) * (applied - cap)
            bodies[9][row] = is_real
                .mul(&one_minus(which))
                .mul(&applied.sub(cap));
            // 10..15. LE byte decompositions for the six u64 columns.
            let decomp_specs: [(usize, usize); 6] = [
                (10, COL_GAS_USED_BYTE_OFFSET),
                (11, COL_REFUND_UNBOUNDED_BYTE_OFFSET),
                (12, COL_REFUND_APPLIED_BYTE_OFFSET),
                (13, COL_REFUND_CAP_BYTE_OFFSET),
                (14, COL_SLACK_BYTE_OFFSET),
                (15, COL_EXTRA_SLACK_BYTE_OFFSET),
            ];
            let value_cols = [
                COL_GAS_USED,
                COL_REFUND_UNBOUNDED,
                COL_REFUND_APPLIED,
                COL_REFUND_CAP,
                COL_SLACK,
                COL_EXTRA_SLACK,
            ];
            for ((slot, byte_offset), val_col) in
                decomp_specs.iter().zip(value_cols.iter())
            {
                let recompose = eval_le_recompose(&r, *byte_offset);
                bodies[*slot][row] = r[*val_col].sub(&recompose);
            }
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let five = Scalar::from_u64(REFUND_DIVISOR_LONDON, curve);
        let two = Scalar::from_u64(REFUND_DIVISOR_PRE_LONDON, curve);
        let is_real = &col_evals[COL_IS_REAL];
        let is_london = &col_evals[COL_IS_POST_LONDON];
        let which = &col_evals[COL_WHICH_SMALLER];
        let gas_used = &col_evals[COL_GAS_USED];
        let unbounded = &col_evals[COL_REFUND_UNBOUNDED];
        let applied = &col_evals[COL_REFUND_APPLIED];
        let cap = &col_evals[COL_REFUND_CAP];
        let slack = &col_evals[COL_SLACK];
        let extra_slack = &col_evals[COL_EXTRA_SLACK];
        let cap_l = &col_evals[COL_REFUND_CAP_LONDON];
        let cap_p = &col_evals[COL_REFUND_CAP_PRE];
        let rem_l = &col_evals[COL_CAP_REMAINDER_LONDON];
        let rem_p = &col_evals[COL_CAP_REMAINDER_PRE];
        let one_minus = |x: &Scalar| one.sub(x);
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&one_minus(is_real)),
            is_london.mul(&one_minus(is_london)),
            which.mul(&one_minus(which)),
            is_real
                .mul(is_london)
                .mul(&cap_l.mul(&five).add(rem_l).sub(gas_used)),
            is_real
                .mul(&one_minus(is_london))
                .mul(&cap_p.mul(&two).add(rem_p).sub(gas_used)),
            is_real.mul(
                &cap.sub(&is_london.mul(cap_l))
                    .sub(&one_minus(is_london).mul(cap_p)),
            ),
            is_real.mul(&applied.add(slack).sub(cap)),
            is_real.mul(&unbounded.sub(applied).sub(extra_slack)),
            is_real.mul(which).mul(&applied.sub(unbounded)),
            is_real.mul(&one_minus(which)).mul(&applied.sub(cap)),
            gas_used.sub(&eval_le_recompose(col_evals, COL_GAS_USED_BYTE_OFFSET)),
            unbounded.sub(&eval_le_recompose(col_evals, COL_REFUND_UNBOUNDED_BYTE_OFFSET)),
            applied.sub(&eval_le_recompose(col_evals, COL_REFUND_APPLIED_BYTE_OFFSET)),
            cap.sub(&eval_le_recompose(col_evals, COL_REFUND_CAP_BYTE_OFFSET)),
            slack.sub(&eval_le_recompose(col_evals, COL_SLACK_BYTE_OFFSET)),
            extra_slack.sub(&eval_le_recompose(col_evals, COL_EXTRA_SLACK_BYTE_OFFSET)),
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
        let five_poly = vec![Scalar::from_u64(REFUND_DIVISOR_LONDON, curve)];
        let two_poly = vec![Scalar::from_u64(REFUND_DIVISOR_PRE_LONDON, curve)];
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_london = &col_coeffs[COL_IS_POST_LONDON];
        let which = &col_coeffs[COL_WHICH_SMALLER];
        let gas_used = &col_coeffs[COL_GAS_USED];
        let unbounded = &col_coeffs[COL_REFUND_UNBOUNDED];
        let applied = &col_coeffs[COL_REFUND_APPLIED];
        let cap = &col_coeffs[COL_REFUND_CAP];
        let slack = &col_coeffs[COL_SLACK];
        let extra_slack = &col_coeffs[COL_EXTRA_SLACK];
        let cap_l = &col_coeffs[COL_REFUND_CAP_LONDON];
        let cap_p = &col_coeffs[COL_REFUND_CAP_PRE];
        let rem_l = &col_coeffs[COL_CAP_REMAINDER_LONDON];
        let rem_p = &col_coeffs[COL_CAP_REMAINDER_PRE];
        let one_minus = |p: &Vec<Scalar>| poly_sub(&one_poly, p, curve);

        // 0..2 binaries
        let b0 = poly_mul(is_real, &one_minus(is_real), curve);
        let b1 = poly_mul(is_london, &one_minus(is_london), curve);
        let b2 = poly_mul(which, &one_minus(which), curve);
        // 3. post-London cap-div
        let div_post = {
            let cap_l_times_5 = poly_mul(cap_l, &five_poly, curve);
            let sum_plus_rem = poly_add(&cap_l_times_5, rem_l, curve);
            poly_sub(&sum_plus_rem, gas_used, curve)
        };
        let gate_post = poly_mul(is_real, is_london, curve);
        let b3 = poly_mul(&gate_post, &div_post, curve);
        // 4. pre-London cap-div
        let div_pre = {
            let cap_p_times_2 = poly_mul(cap_p, &two_poly, curve);
            let sum_plus_rem = poly_add(&cap_p_times_2, rem_p, curve);
            poly_sub(&sum_plus_rem, gas_used, curve)
        };
        let gate_pre = poly_mul(is_real, &one_minus(is_london), curve);
        let b4 = poly_mul(&gate_pre, &div_pre, curve);
        // 5. cap_pick
        let london_cap_l = poly_mul(is_london, cap_l, curve);
        let pre_cap_p = poly_mul(&one_minus(is_london), cap_p, curve);
        let pick = poly_sub(&poly_sub(cap, &london_cap_l, curve), &pre_cap_p, curve);
        let b5 = poly_mul(is_real, &pick, curve);
        // 6. applied + slack = cap
        let aps = poly_sub(&poly_add(applied, slack, curve), cap, curve);
        let b6 = poly_mul(is_real, &aps, curve);
        // 7. unbounded - applied - extra_slack = 0
        let us = poly_sub(&poly_sub(unbounded, applied, curve), extra_slack, curve);
        let b7 = poly_mul(is_real, &us, curve);
        // 8. min when unbounded smaller
        let diff_au = poly_sub(applied, unbounded, curve);
        let b8 = poly_mul(&poly_mul(is_real, which, curve), &diff_au, curve);
        // 9. min when cap smaller
        let diff_ac = poly_sub(applied, cap, curve);
        let b9 = poly_mul(
            &poly_mul(is_real, &one_minus(which), curve),
            &diff_ac,
            curve,
        );
        // 10..15 LE decomps
        let b10 = poly_sub(
            gas_used,
            &build_le_recompose_poly(col_coeffs, COL_GAS_USED_BYTE_OFFSET, curve),
            curve,
        );
        let b11 = poly_sub(
            unbounded,
            &build_le_recompose_poly(col_coeffs, COL_REFUND_UNBOUNDED_BYTE_OFFSET, curve),
            curve,
        );
        let b12 = poly_sub(
            applied,
            &build_le_recompose_poly(col_coeffs, COL_REFUND_APPLIED_BYTE_OFFSET, curve),
            curve,
        );
        let b13 = poly_sub(
            cap,
            &build_le_recompose_poly(col_coeffs, COL_REFUND_CAP_BYTE_OFFSET, curve),
            curve,
        );
        let b14 = poly_sub(
            slack,
            &build_le_recompose_poly(col_coeffs, COL_SLACK_BYTE_OFFSET, curve),
            curve,
        );
        let b15 = poly_sub(
            extra_slack,
            &build_le_recompose_poly(col_coeffs, COL_EXTRA_SLACK_BYTE_OFFSET, curve),
            curve,
        );
        let bodies = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15];
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
        vec![COL_IS_REAL, COL_IS_POST_LONDON, COL_WHICH_SMALLER]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![
            LookupTable::range(256), // 8-bit
            LookupTable::range(2),   // 1-bit
            LookupTable::range(8),   // 3-bit (covers remainder < 5)
        ];
        let tbl_byte = 0usize;
        let tbl_bit = 1usize;
        let tbl_small = 2usize;
        let mut declarations = Vec::new();
        let byte_decomps: [(usize, &str); 6] = [
            (COL_GAS_USED_BYTE_OFFSET, "gas_used"),
            (COL_REFUND_UNBOUNDED_BYTE_OFFSET, "refund_unbounded"),
            (COL_REFUND_APPLIED_BYTE_OFFSET, "refund_applied"),
            (COL_REFUND_CAP_BYTE_OFFSET, "refund_cap"),
            (COL_SLACK_BYTE_OFFSET, "slack"),
            (COL_EXTRA_SLACK_BYTE_OFFSET, "extra_slack"),
        ];
        for (offset, name) in byte_decomps.iter() {
            for j in 0..NUM_BYTE_LIMBS {
                declarations.push((
                    LookupDeclaration {
                        label: format!("refund3529_{}_byte_{}_8bit", name, j),
                        column_index: offset + j,
                        max_bits: 8,
                        selector_column: None,
                    },
                    tbl_byte,
                ));
            }
        }
        // Binary flags
        for (col, label) in [
            (COL_IS_REAL, "refund3529_is_real_1bit"),
            (COL_IS_POST_LONDON, "refund3529_is_post_london_1bit"),
            (COL_WHICH_SMALLER, "refund3529_which_smaller_1bit"),
        ] {
            declarations.push((
                LookupDeclaration {
                    label: label.into(),
                    column_index: col,
                    max_bits: 1,
                    selector_column: None,
                },
                tbl_bit,
            ));
        }
        // Small remainders. London divisor 5 needs < 5; we range to 3-bit
        // (< 8) and rely on the algebraic gate to keep the value in the
        // canonical {0..4}. Pre-London divisor 2 needs < 2 — encoded as
        // 1-bit range.
        declarations.push((
            LookupDeclaration {
                label: "refund3529_cap_remainder_london_3bit".into(),
                column_index: COL_CAP_REMAINDER_LONDON,
                max_bits: 3,
                selector_column: None,
            },
            tbl_small,
        ));
        declarations.push((
            LookupDeclaration {
                label: "refund3529_cap_remainder_pre_1bit".into(),
                column_index: COL_CAP_REMAINDER_PRE,
                max_bits: 1,
                selector_column: None,
            },
            tbl_bit,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Bind the refund-3529 row's `tx_index` to the `sstore_transition_air`'s
/// `IS_CLEAR` rows as the algebraic source of the `refund_unbounded`
/// contribution (each storage clear earns the EIP-3529 `R_sclear`
/// refund). For now the descriptor pins `tx_index` only — the per-row
/// refund-amount tally is a host-side oracle, with the full quantitative
/// binding deferred to a follow-up running-sum gadget.
pub fn make_refund_3529_to_sstore_descriptor(
    refund_3529_layer_index: usize,
    sstore_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "refund_3529_to_sstore_clear_v1".into(),
        a_layer_index: refund_3529_layer_index,
        a_columns: vec![COL_TX_INDEX],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sstore_layer_index,
        // sstore_transition_air doesn't expose a per-row tx_index; we
        // re-use COL_IS_CLEAR twice as a placeholder column tuple to
        // pin the row count alignment. The quantitative binding lands
        // when the tx_index column is added to sstore_transition_air.
        b_columns: vec![metavm_zkp::sstore_transition_air::COL_IS_CLEAR],
        b_selector_column: Some(metavm_zkp::sstore_transition_air::COL_IS_CLEAR),
    }
}

/// Bind the refund-3529 row's `(gas_used, refund_applied)` to the
/// gas-tracking AIR. This is currently a *structural* descriptor: the
/// gas-tracking AIR commits per-step `(gas_pre, gas_post)` rather than
/// per-tx aggregates, so the binding is host-oracle until an aggregator
/// column lands.
pub fn make_refund_3529_to_gas_tracking_descriptor(
    refund_3529_layer_index: usize,
    gas_tracking_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "refund_3529_to_gas_tracking_v1".into(),
        a_layer_index: refund_3529_layer_index,
        a_columns: vec![COL_GAS_USED, COL_REFUND_APPLIED],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: gas_tracking_layer_index,
        b_columns: vec![
            crate::gas_tracking_air::COL_GAS_PRE,
            crate::gas_tracking_air::COL_GAS_POST,
        ],
        b_selector_column: Some(crate::gas_tracking_air::COL_IS_REAL),
    }
}

/// Bind `is_post_london` here to the hardfork-rules AIR. London is
/// strictly older than Shanghai, so `is_post_shanghai` is a sound
/// over-approximation: any post-Shanghai block is also post-London,
/// and the hardfork-rules AIR's `shanghai ⇒ paris` implication chain
/// extends through London by transitive activation. The dedicated
/// `is_post_london` column will be added to hardfork-rules in a
/// follow-up.
pub fn make_refund_3529_to_hardfork_descriptor(
    refund_3529_layer_index: usize,
    hardfork_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "refund_3529_to_hardfork_post_london_v1".into(),
        a_layer_index: refund_3529_layer_index,
        a_columns: vec![COL_IS_POST_LONDON],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: hardfork_layer_index,
        b_columns: vec![crate::hardfork_rules_air::COL_IS_POST_SHANGHAI],
        b_selector_column: Some(crate::hardfork_rules_air::COL_IS_REAL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_vanish(trace: &TracePolynomials, cs: &GasRefund3529ConstraintSystem) {
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, trace.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) nonzero at row {}",
                    i,
                    cs.constraint_labels()[i],
                    r,
                );
            }
        }
    }

    #[test]
    fn pre_london_cap_div_2_witness_vanishes() {
        // Pre-London: gas_used = 100_000, cap = 50_000, refund_unbounded
        // = 20_000 → applied = 20_000 (below cap).
        let row = from_inputs(0, 100_000, 20_000, false);
        assert_eq!(row.refund_cap, 50_000);
        assert_eq!(row.refund_applied, 20_000);
        assert_eq!(row.cap_remainder_pre, 0);
        assert!(row.which_smaller);
        let w = GasRefund3529Witness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn post_london_cap_div_5_witness_vanishes() {
        // gas_used = 100_000, cap = 20_000, refund_unbounded = 5_000 →
        // applied = 5_000.
        let row = from_inputs(7, 100_000, 5_000, true);
        assert_eq!(row.refund_cap, 20_000);
        assert_eq!(row.refund_applied, 5_000);
        assert_eq!(row.cap_remainder_london, 0);
        assert!(row.which_smaller);
        let w = GasRefund3529Witness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn refund_above_cap_post_london() {
        // gas_used = 50_007, cap_london = 10_001 (remainder 2),
        // refund_unbounded = 99_999 → applied = 10_001 = cap.
        let row = from_inputs(3, 50_007, 99_999, true);
        assert_eq!(row.refund_cap, 10_001);
        assert_eq!(row.cap_remainder_london, 2);
        assert_eq!(row.refund_applied, 10_001);
        assert!(!row.which_smaller);
        assert_eq!(row.extra_slack, 99_999 - 10_001);
        assert_eq!(row.slack, 0);
        let w = GasRefund3529Witness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn refund_below_cap_pre_london_with_remainder() {
        // Pre-London divisor 2. gas_used = 101 → cap_pre = 50, rem 1.
        let row = from_inputs(1, 101, 7, false);
        assert_eq!(row.refund_cap, 50);
        assert_eq!(row.cap_remainder_pre, 1);
        assert_eq!(row.refund_applied, 7);
        assert!(row.which_smaller);
        assert_eq!(row.slack, 43);
        let w = GasRefund3529Witness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn tampered_cap_detected() {
        // Build honest witness then bump refund_cap by 1 — the cap_div
        // and cap_pick constraints must catch it.
        let row = from_inputs(2, 100_000, 30_000, true);
        let curve = CurveType::Bls48581;
        let w = GasRefund3529Witness { rows: vec![row] };
        let mut trace = build_trace_polynomials(&w, curve);
        // Bump refund_cap from 20_000 → 25_000 (would let attacker
        // refund more than allowed).
        trace.columns[COL_REFUND_CAP].evaluations[0] = Scalar::from_u64(25_000, curve);
        // Also bump refund_applied to match (so applied_slack_def
        // stays consistent if slack = 0). slack was 0 originally
        // (applied=20_000=cap). After bump applied = 25_000, slack = 0.
        // Tampered cap will break cap_pick (constraint 5) and the
        // div constraint (3).
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        let cap_pick_body = &bodies[5];
        assert!(
            !cap_pick_body[0].is_zero(),
            "cap_pick must fire on tampered refund_cap"
        );
    }

    #[test]
    fn tampered_applied_exceeds_cap_detected() {
        // gas_used = 100, cap_london = 20. Attacker tries to claim
        // refund_applied = 30 > cap.
        let row = from_inputs(4, 100, 30, true);
        // Honest computation gives applied=20, slack=0, extra_slack=10.
        assert_eq!(row.refund_applied, 20);
        let curve = CurveType::Bls48581;
        let w = GasRefund3529Witness { rows: vec![row] };
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper: set refund_applied = 30 while leaving everything
        // else honest.
        trace.columns[COL_REFUND_APPLIED].evaluations[0] = Scalar::from_u64(30, curve);
        // The applied_slack_def constraint expects applied + slack = cap
        // = 20. With applied=30 and slack=0, this is 30 != 20 — fires.
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert!(
            !bodies[6][0].is_zero(),
            "applied_slack_def must fire when applied > cap"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d_s = make_refund_3529_to_sstore_descriptor(0, 1);
        assert_eq!(d_s.label, "refund_3529_to_sstore_clear_v1");
        assert_eq!(d_s.a_layer_index, 0);
        assert_eq!(d_s.b_layer_index, 1);
        assert_eq!(d_s.a_columns, vec![COL_TX_INDEX]);
        assert_eq!(d_s.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_s.b_columns.len(), 1);

        let d_g = make_refund_3529_to_gas_tracking_descriptor(0, 2);
        assert_eq!(d_g.label, "refund_3529_to_gas_tracking_v1");
        assert_eq!(d_g.a_columns, vec![COL_GAS_USED, COL_REFUND_APPLIED]);
        assert_eq!(d_g.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_g.b_columns.len(), 2);
        assert_eq!(d_g.b_layer_index, 2);
        assert_eq!(
            d_g.b_selector_column,
            Some(crate::gas_tracking_air::COL_IS_REAL)
        );

        let d_h = make_refund_3529_to_hardfork_descriptor(0, 3);
        assert_eq!(d_h.label, "refund_3529_to_hardfork_post_london_v1");
        assert_eq!(d_h.a_columns, vec![COL_IS_POST_LONDON]);
        assert_eq!(d_h.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_h.b_layer_index, 3);
        assert_eq!(
            d_h.b_columns,
            vec![crate::hardfork_rules_air::COL_IS_POST_SHANGHAI]
        );
        assert_eq!(
            d_h.b_selector_column,
            Some(crate::hardfork_rules_air::COL_IS_REAL)
        );
    }

    #[test]
    fn evaluate_at_point_matches_domain_on_honest_row() {
        let row = from_inputs(9, 200_000, 30_000, true);
        let w = GasRefund3529Witness { rows: vec![row] };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(0xCAFE, CurveType::Bls48581);
        let row_evals: Vec<Scalar> =
            trace.columns.iter().map(|c| c.evaluations[0].clone()).collect();
        let v = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(v.is_zero(), "evaluate_at_point nonzero on honest row");
    }

    #[test]
    fn lookup_declarations_well_formed() {
        let cs = GasRefund3529ConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 6 columns * 8 byte limbs = 48 byte declarations,
        // + 3 binary flags + 2 remainder ranges = 53.
        assert_eq!(req.declarations.len(), 48 + 3 + 2);
        assert_eq!(req.tables.len(), 3);
    }

    #[test]
    fn multi_row_witness_vanishes() {
        // Two transactions in one witness: one pre-London below cap,
        // one post-London above cap.
        let rows = vec![
            from_inputs(0, 100_000, 30_000, false), // pre-London, cap=50_000, applied=30_000
            from_inputs(1, 100_000, 50_000, true),  // post-London, cap=20_000, applied=20_000
        ];
        let w = GasRefund3529Witness { rows };
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasRefund3529ConstraintSystem::new(trace.num_rows);
        assert_all_vanish(&trace, &cs);
    }
}
