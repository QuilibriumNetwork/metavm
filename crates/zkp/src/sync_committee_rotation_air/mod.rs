//! Sync committee period transition (rotation) AIR.
//!
//! Proves that the sync committee rotates correctly at sync-committee
//! period boundaries. Per the Altair beacon spec,
//! `EPOCHS_PER_SYNC_COMMITTEE_PERIOD = 256`; whenever the period changes
//! (i.e. `new_epoch / 256 > prev_epoch / 256`), the committee rotates:
//!
//!   post_current_sync_committee = next_sync_committee
//!   post_next_sync_committee    = <freshly derived from validator state>
//!
//! On non-boundary transitions the committees pass through unchanged:
//!
//!   post_current = current
//!   post_next    = next
//!
//! This AIR algebraically pins:
//!   1. `prev_period = prev_epoch / 256` and `new_period = new_epoch / 256`
//!      via byte-decomposition + range-checked `epoch_mod_256 < 256`.
//!   2. `is_period_boundary == 1` implies `new_period - prev_period == 1`
//!      (single-period step at boundary; multi-period jumps are out of scope).
//!   3. On boundary rows, `post_current_committee_root = next_committee_root`
//!      (byte-wise, β-RLC over 32 bytes into a single constraint).
//!   4. On non-boundary rows, `post_current = current` and `post_next = next`
//!      (β-RLC × 2 → two constraints).
//!   5. `is_period_boundary` and `is_real` are binary.
//!   6. `is_real * (is_period_boundary * (new_period - prev_period) -
//!      is_period_boundary) = 0` — same as (2) but algebraically forced.
//!
//! The provenance of the new `next_committee_root` from validator state is
//! NOT checked here; it is the responsibility of a cross-AIR LogUp linkage
//! into the validator-derivation AIR (deferred).
//!
//! # Column layout (168 cols)
//!
//! Per row:
//!   COL_PREV_EPOCH                     0          u64
//!   COL_NEW_EPOCH                      1          u64
//!   COL_PREV_PERIOD                    2          u64
//!   COL_NEW_PERIOD                     3          u64
//!   COL_PREV_EPOCH_BYTE_OFFSET         4..12      8 LE bytes of prev_epoch
//!   COL_NEW_EPOCH_BYTE_OFFSET          12..20     8 LE bytes of new_epoch
//!   COL_PREV_PERIOD_BYTE_OFFSET        20..28     8 LE bytes of prev_period
//!   COL_NEW_PERIOD_BYTE_OFFSET         28..36     8 LE bytes of new_period
//!   COL_PREV_EPOCH_MOD_256             36         prev_epoch % 256
//!   COL_NEW_EPOCH_MOD_256              37         new_epoch  % 256
//!   COL_CURRENT_COMMITTEE_ROOT         38..70     32 bytes
//!   COL_NEXT_COMMITTEE_ROOT            70..102    32 bytes
//!   COL_POST_CURRENT_COMMITTEE_ROOT    102..134   32 bytes
//!   COL_POST_NEXT_COMMITTEE_ROOT       134..166   32 bytes
//!   COL_IS_PERIOD_BOUNDARY             166        binary
//!   COL_IS_REAL                        167        binary
//!
//! # Row-local constraints (10)
//!
//! 0. `is_real * (is_real - 1) = 0`
//! 1. `is_period_boundary * (is_period_boundary - 1) = 0`
//! 2. `prev_epoch  - Σ prev_epoch_byte_k · 256^k = 0`
//! 3. `new_epoch   - Σ new_epoch_byte_k · 256^k = 0`
//! 4. `prev_period - Σ prev_period_byte_k · 256^k = 0`
//! 5. `new_period  - Σ new_period_byte_k · 256^k = 0`
//! 6. `is_real * (prev_period * 256 + prev_epoch_mod_256 - prev_epoch) = 0`
//! 7. `is_real * (new_period  * 256 + new_epoch_mod_256  - new_epoch ) = 0`
//! 8. `is_period_boundary * (new_period - prev_period - 1) = 0`
//! 9. Combined rotation constraint, β-RLC over 96 byte equalities:
//!     - 32 byte equalities for boundary: `is_period_boundary *
//!       (post_current[k] - next[k])` k∈[0,32)
//!     - 32 byte equalities for non-boundary pass-through current:
//!       `is_real * (1 - is_period_boundary) * (post_current[k] - current[k])`
//!     - 32 byte equalities for non-boundary pass-through next:
//!       `is_real * (1 - is_period_boundary) * (post_next[k] - next[k])`
//!    all folded via β-RLC into one row-local constraint (β = α).
//!
//! # Byte range checks (LogUp)
//! - 32 bytes (4 u64 LE decomps) → 8-bit
//! - 128 bytes (4 × 32-byte committee roots) → 8-bit
//! - 2 epoch_mod_256 cells → 8-bit
//!
//! # Caveats
//! - `post_next_committee_root` provenance from validator-state derivation
//!   is NOT checked here. A cross-AIR linkage to a future
//!   validator-derivation AIR closes it.
//! - Multi-period jumps (rare, would only occur in non-finalising scenarios)
//!   are out of scope.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PREV_EPOCH: usize = 0;
pub const COL_NEW_EPOCH: usize = 1;
pub const COL_PREV_PERIOD: usize = 2;
pub const COL_NEW_PERIOD: usize = 3;

pub const COL_PREV_EPOCH_BYTE_OFFSET: usize = 4; // 4..12
pub const COL_NEW_EPOCH_BYTE_OFFSET: usize = 12; // 12..20
pub const COL_PREV_PERIOD_BYTE_OFFSET: usize = 20; // 20..28
pub const COL_NEW_PERIOD_BYTE_OFFSET: usize = 28; // 28..36

pub const COL_PREV_EPOCH_MOD_256: usize = 36;
pub const COL_NEW_EPOCH_MOD_256: usize = 37;

pub const COL_CURRENT_COMMITTEE_ROOT_OFFSET: usize = 38; // 38..70
pub const COL_NEXT_COMMITTEE_ROOT_OFFSET: usize = 70; // 70..102
pub const COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET: usize = 102; // 102..134
pub const COL_POST_NEXT_COMMITTEE_ROOT_OFFSET: usize = 134; // 134..166

pub const COL_IS_PERIOD_BOUNDARY: usize = 166;
pub const COL_IS_REAL: usize = 167;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 168

pub const NUM_ROW_CONSTRAINTS: usize = 10;

/// Epochs per sync committee period (Altair spec).
pub const EPOCHS_PER_SYNC_COMMITTEE_PERIOD: u64 = 256;

// ─── Witness type ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncCommitteeRotationRow {
    pub prev_epoch: u64,
    pub new_epoch: u64,
    pub prev_period: u64,
    pub new_period: u64,
    pub current_committee_root: [u8; 32],
    pub next_committee_root: [u8; 32],
    pub post_current_committee_root: [u8; 32],
    pub post_next_committee_root: [u8; 32],
    pub is_period_boundary: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SyncCommitteeRotationWitness {
    pub rows: Vec<SyncCommitteeRotationRow>,
}

impl SyncCommitteeRotationWitness {
    pub fn from_rows(rows: Vec<SyncCommitteeRotationRow>) -> Self {
        Self { rows }
    }

    /// Build a single-row rotation witness from a `(prev_epoch, current_root,
    /// next_root)` triple. `new_epoch = prev_epoch + 1`. On a period boundary
    /// the post-current committee root is set to `next_root` (rotation);
    /// otherwise both pass through unchanged. The new `post_next` is set to
    /// `next_root` (placeholder; cross-AIR linkage to validator-derivation
    /// AIR replaces the algebraic check).
    pub fn from_rotation(
        prev_epoch: u64,
        current_root: [u8; 32],
        next_root: [u8; 32],
    ) -> Self {
        let new_epoch = prev_epoch.saturating_add(1);
        let prev_period = prev_epoch / EPOCHS_PER_SYNC_COMMITTEE_PERIOD;
        let new_period = new_epoch / EPOCHS_PER_SYNC_COMMITTEE_PERIOD;
        let is_boundary = new_period > prev_period;
        let (post_current, post_next) = if is_boundary {
            // Rotation: current ← next, next ← <new committee derived>
            // Placeholder for new committee: use next_root again; a real
            // witness would supply the new committee root here.
            (next_root, next_root)
        } else {
            (current_root, next_root)
        };
        let row = SyncCommitteeRotationRow {
            prev_epoch,
            new_epoch,
            prev_period,
            new_period,
            current_committee_root: current_root,
            next_committee_root: next_root,
            post_current_committee_root: post_current,
            post_next_committee_root: post_next,
            is_period_boundary: is_boundary,
        };
        Self { rows: vec![row] }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &SyncCommitteeRotationWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_PREV_EPOCH][i] = Scalar::from_u64(row.prev_epoch, curve);
        columns[COL_NEW_EPOCH][i] = Scalar::from_u64(row.new_epoch, curve);
        columns[COL_PREV_PERIOD][i] = Scalar::from_u64(row.prev_period, curve);
        columns[COL_NEW_PERIOD][i] = Scalar::from_u64(row.new_period, curve);

        let prev_epoch_bytes = row.prev_epoch.to_le_bytes();
        let new_epoch_bytes = row.new_epoch.to_le_bytes();
        let prev_period_bytes = row.prev_period.to_le_bytes();
        let new_period_bytes = row.new_period.to_le_bytes();
        for k in 0..8 {
            columns[COL_PREV_EPOCH_BYTE_OFFSET + k][i] =
                Scalar::from_u64(prev_epoch_bytes[k] as u64, curve);
            columns[COL_NEW_EPOCH_BYTE_OFFSET + k][i] =
                Scalar::from_u64(new_epoch_bytes[k] as u64, curve);
            columns[COL_PREV_PERIOD_BYTE_OFFSET + k][i] =
                Scalar::from_u64(prev_period_bytes[k] as u64, curve);
            columns[COL_NEW_PERIOD_BYTE_OFFSET + k][i] =
                Scalar::from_u64(new_period_bytes[k] as u64, curve);
        }

        let prev_mod = row.prev_epoch % EPOCHS_PER_SYNC_COMMITTEE_PERIOD;
        let new_mod = row.new_epoch % EPOCHS_PER_SYNC_COMMITTEE_PERIOD;
        columns[COL_PREV_EPOCH_MOD_256][i] = Scalar::from_u64(prev_mod, curve);
        columns[COL_NEW_EPOCH_MOD_256][i] = Scalar::from_u64(new_mod, curve);

        for k in 0..32 {
            columns[COL_CURRENT_COMMITTEE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.current_committee_root[k] as u64, curve);
            columns[COL_NEXT_COMMITTEE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.next_committee_root[k] as u64, curve);
            columns[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.post_current_committee_root[k] as u64, curve);
            columns[COL_POST_NEXT_COMMITTEE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.post_next_committee_root[k] as u64, curve);
        }

        columns[COL_IS_PERIOD_BOUNDARY][i] =
            if row.is_period_boundary { one.clone() } else { zero.clone() };
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

pub struct SyncCommitteeRotationConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl SyncCommitteeRotationConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn le_decomp_sum_evals(
    columns: &[&Vec<Scalar>],
    base: usize,
    row: usize,
    curve: CurveType,
) -> Scalar {
    let mut sum = Scalar::zero(curve);
    for k in 0..8 {
        let pow = 1u64 << (8 * k as u32);
        sum = sum.add(&columns[base + k][row].mul(&Scalar::from_u64(pow, curve)));
    }
    sum
}

fn le_decomp_sum_point(
    col_evals_at_z: &[Scalar],
    base: usize,
    curve: CurveType,
) -> Scalar {
    let mut sum = Scalar::zero(curve);
    for k in 0..8 {
        let pow = 1u64 << (8 * k as u32);
        sum = sum.add(&col_evals_at_z[base + k].mul(&Scalar::from_u64(pow, curve)));
    }
    sum
}

fn le_decomp_sum_poly(
    col_coeffs: &[Vec<Scalar>],
    base: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..8 {
        let pow = 1u64 << (8 * k as u32);
        let term = poly_scalar_mul(&col_coeffs[base + k], &Scalar::from_u64(pow, curve));
        sum = poly_add(&sum, &term, curve);
    }
    sum
}

impl VmConstraintSystem for SyncCommitteeRotationConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_period_boundary_binary".into(),
            "prev_epoch_le_decomp".into(),
            "new_epoch_le_decomp".into(),
            "prev_period_le_decomp".into(),
            "new_period_le_decomp".into(),
            "prev_period_compose".into(),
            "new_period_compose".into(),
            "boundary_period_step".into(),
            "rotation_or_passthrough_rlc".into(),
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
        let two_pow_8 = Scalar::from_u64(256, curve);
        // β for RLC over the 96 byte equalities (3 × 32). We reuse a small
        // fixed scalar here for evaluation on-domain; the algebraic
        // soundness comes from `evaluate_at_point` / `build_constraint_polynomial`
        // where β is the Fiat-Shamir α (per project convention β = α).
        let beta = Scalar::from_u64(7919, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_pb = &columns[COL_IS_PERIOD_BOUNDARY][r];
            let prev_epoch = &columns[COL_PREV_EPOCH][r];
            let new_epoch = &columns[COL_NEW_EPOCH][r];
            let prev_period = &columns[COL_PREV_PERIOD][r];
            let new_period = &columns[COL_NEW_PERIOD][r];
            let prev_mod = &columns[COL_PREV_EPOCH_MOD_256][r];
            let new_mod = &columns[COL_NEW_EPOCH_MOD_256][r];

            out[0][r] = is_real.mul(&is_real.sub(&one));
            out[1][r] = is_pb.mul(&is_pb.sub(&one));
            out[2][r] = prev_epoch.sub(&le_decomp_sum_evals(
                columns,
                COL_PREV_EPOCH_BYTE_OFFSET,
                r,
                curve,
            ));
            out[3][r] = new_epoch.sub(&le_decomp_sum_evals(
                columns,
                COL_NEW_EPOCH_BYTE_OFFSET,
                r,
                curve,
            ));
            out[4][r] = prev_period.sub(&le_decomp_sum_evals(
                columns,
                COL_PREV_PERIOD_BYTE_OFFSET,
                r,
                curve,
            ));
            out[5][r] = new_period.sub(&le_decomp_sum_evals(
                columns,
                COL_NEW_PERIOD_BYTE_OFFSET,
                r,
                curve,
            ));
            // 6: prev_period * 256 + prev_mod - prev_epoch (gated by is_real)
            let prev_composed = prev_period.mul(&two_pow_8).add(prev_mod).sub(prev_epoch);
            out[6][r] = is_real.mul(&prev_composed);
            // 7: new_period * 256 + new_mod - new_epoch (gated by is_real)
            let new_composed = new_period.mul(&two_pow_8).add(new_mod).sub(new_epoch);
            out[7][r] = is_real.mul(&new_composed);
            // 8: is_period_boundary * (new_period - prev_period - 1)
            out[8][r] = is_pb.mul(&new_period.sub(prev_period).sub(&one));
            // 9: β-RLC over 96 byte equalities.
            let one_minus_pb = one.sub(is_pb);
            let real_and_not_pb = is_real.mul(&one_minus_pb);
            let mut rlc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            // First 32 terms: boundary rotation post_current = next.
            for k in 0..32 {
                let pc = &columns[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k][r];
                let nx = &columns[COL_NEXT_COMMITTEE_ROOT_OFFSET + k][r];
                let body = is_pb.mul(&pc.sub(nx));
                rlc = rlc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            // Next 32 terms: non-boundary post_current = current.
            for k in 0..32 {
                let pc = &columns[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k][r];
                let cu = &columns[COL_CURRENT_COMMITTEE_ROOT_OFFSET + k][r];
                let body = real_and_not_pb.mul(&pc.sub(cu));
                rlc = rlc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            // Final 32 terms: non-boundary post_next = next.
            for k in 0..32 {
                let pn = &columns[COL_POST_NEXT_COMMITTEE_ROOT_OFFSET + k][r];
                let nx = &columns[COL_NEXT_COMMITTEE_ROOT_OFFSET + k][r];
                let body = real_and_not_pb.mul(&pn.sub(nx));
                rlc = rlc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            out[9][r] = rlc;
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
        let two_pow_8 = Scalar::from_u64(256, curve);
        // β = α (project convention; both come from the same Fiat-Shamir
        // transcript challenge).
        let beta = alpha.clone();

        let is_real = &col_evals[COL_IS_REAL];
        let is_pb = &col_evals[COL_IS_PERIOD_BOUNDARY];
        let prev_epoch = &col_evals[COL_PREV_EPOCH];
        let new_epoch = &col_evals[COL_NEW_EPOCH];
        let prev_period = &col_evals[COL_PREV_PERIOD];
        let new_period = &col_evals[COL_NEW_PERIOD];
        let prev_mod = &col_evals[COL_PREV_EPOCH_MOD_256];
        let new_mod = &col_evals[COL_NEW_EPOCH_MOD_256];

        let body0 = is_real.mul(&is_real.sub(&one));
        let body1 = is_pb.mul(&is_pb.sub(&one));
        let body2 = prev_epoch.sub(&le_decomp_sum_point(col_evals, COL_PREV_EPOCH_BYTE_OFFSET, curve));
        let body3 = new_epoch.sub(&le_decomp_sum_point(col_evals, COL_NEW_EPOCH_BYTE_OFFSET, curve));
        let body4 = prev_period.sub(&le_decomp_sum_point(col_evals, COL_PREV_PERIOD_BYTE_OFFSET, curve));
        let body5 = new_period.sub(&le_decomp_sum_point(col_evals, COL_NEW_PERIOD_BYTE_OFFSET, curve));
        let body6 = is_real.mul(&prev_period.mul(&two_pow_8).add(prev_mod).sub(prev_epoch));
        let body7 = is_real.mul(&new_period.mul(&two_pow_8).add(new_mod).sub(new_epoch));
        let body8 = is_pb.mul(&new_period.sub(prev_period).sub(&one));

        // body9: β-RLC over 96 byte equalities.
        let one_minus_pb = one.sub(is_pb);
        let real_and_not_pb = is_real.mul(&one_minus_pb);
        let mut rlc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let pc = &col_evals[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k];
            let nx = &col_evals[COL_NEXT_COMMITTEE_ROOT_OFFSET + k];
            rlc = rlc.add(&bp.mul(&is_pb.mul(&pc.sub(nx))));
            bp = bp.mul(&beta);
        }
        for k in 0..32 {
            let pc = &col_evals[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k];
            let cu = &col_evals[COL_CURRENT_COMMITTEE_ROOT_OFFSET + k];
            rlc = rlc.add(&bp.mul(&real_and_not_pb.mul(&pc.sub(cu))));
            bp = bp.mul(&beta);
        }
        for k in 0..32 {
            let pn = &col_evals[COL_POST_NEXT_COMMITTEE_ROOT_OFFSET + k];
            let nx = &col_evals[COL_NEXT_COMMITTEE_ROOT_OFFSET + k];
            rlc = rlc.add(&bp.mul(&real_and_not_pb.mul(&pn.sub(nx))));
            bp = bp.mul(&beta);
        }
        let body9 = rlc;

        let bodies = [body0, body1, body2, body3, body4, body5, body6, body7, body8, body9];
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&ap.mul(body));
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
        let beta = alpha.clone();

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_pb = &col_coeffs[COL_IS_PERIOD_BOUNDARY];
        let prev_epoch = &col_coeffs[COL_PREV_EPOCH];
        let new_epoch = &col_coeffs[COL_NEW_EPOCH];
        let prev_period = &col_coeffs[COL_PREV_PERIOD];
        let new_period = &col_coeffs[COL_NEW_PERIOD];
        let prev_mod = &col_coeffs[COL_PREV_EPOCH_MOD_256];
        let new_mod = &col_coeffs[COL_NEW_EPOCH_MOD_256];

        let body0 = poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve);
        let body1 = poly_mul(is_pb, &poly_sub(is_pb, &one_poly, curve), curve);

        let prev_epoch_sum = le_decomp_sum_poly(col_coeffs, COL_PREV_EPOCH_BYTE_OFFSET, curve);
        let body2 = poly_sub(prev_epoch, &prev_epoch_sum, curve);
        let new_epoch_sum = le_decomp_sum_poly(col_coeffs, COL_NEW_EPOCH_BYTE_OFFSET, curve);
        let body3 = poly_sub(new_epoch, &new_epoch_sum, curve);
        let prev_period_sum = le_decomp_sum_poly(col_coeffs, COL_PREV_PERIOD_BYTE_OFFSET, curve);
        let body4 = poly_sub(prev_period, &prev_period_sum, curve);
        let new_period_sum = le_decomp_sum_poly(col_coeffs, COL_NEW_PERIOD_BYTE_OFFSET, curve);
        let body5 = poly_sub(new_period, &new_period_sum, curve);

        // body6: is_real * (prev_period * 256 + prev_mod - prev_epoch)
        let prev_scaled = poly_scalar_mul(prev_period, &Scalar::from_u64(256, curve));
        let prev_composed = poly_sub(&poly_add(&prev_scaled, prev_mod, curve), prev_epoch, curve);
        let body6 = poly_mul(is_real, &prev_composed, curve);
        // body7: is_real * (new_period * 256 + new_mod - new_epoch)
        let new_scaled = poly_scalar_mul(new_period, &Scalar::from_u64(256, curve));
        let new_composed = poly_sub(&poly_add(&new_scaled, new_mod, curve), new_epoch, curve);
        let body7 = poly_mul(is_real, &new_composed, curve);
        // body8: is_pb * (new_period - prev_period - 1)
        let np_diff = poly_sub(&poly_sub(new_period, prev_period, curve), &one_poly, curve);
        let body8 = poly_mul(is_pb, &np_diff, curve);

        // body9: β-RLC.
        let one_minus_pb = poly_sub(&one_poly, is_pb, curve);
        let real_and_not_pb = poly_mul(is_real, &one_minus_pb, curve);
        let mut rlc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let pc = &col_coeffs[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k];
            let nx = &col_coeffs[COL_NEXT_COMMITTEE_ROOT_OFFSET + k];
            let diff = poly_sub(pc, nx, curve);
            let body = poly_mul(is_pb, &diff, curve);
            rlc = poly_add(&rlc, &poly_scalar_mul(&body, &bp), curve);
            bp = bp.mul(&beta);
        }
        for k in 0..32 {
            let pc = &col_coeffs[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k];
            let cu = &col_coeffs[COL_CURRENT_COMMITTEE_ROOT_OFFSET + k];
            let diff = poly_sub(pc, cu, curve);
            let body = poly_mul(&real_and_not_pb, &diff, curve);
            rlc = poly_add(&rlc, &poly_scalar_mul(&body, &bp), curve);
            bp = bp.mul(&beta);
        }
        for k in 0..32 {
            let pn = &col_coeffs[COL_POST_NEXT_COMMITTEE_ROOT_OFFSET + k];
            let nx = &col_coeffs[COL_NEXT_COMMITTEE_ROOT_OFFSET + k];
            let diff = poly_sub(pn, nx, curve);
            let body = poly_mul(&real_and_not_pb, &diff, curve);
            rlc = poly_add(&rlc, &poly_scalar_mul(&body, &bp), curve);
            bp = bp.mul(&beta);
        }
        let body9 = rlc;

        let bodies = [body0, body1, body2, body3, body4, body5, body6, body7, body8, body9];
        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = poly_add(&acc, &poly_scalar_mul(body, &ap), curve);
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
        let tables = vec![LookupTable::range(8)];
        const TBL_8: usize = 0;
        let mut declarations = Vec::new();

        // u64 LE byte cols (4 × 8 = 32) → 8-bit range.
        for (label, base) in [
            ("scr_prev_epoch_byte", COL_PREV_EPOCH_BYTE_OFFSET),
            ("scr_new_epoch_byte", COL_NEW_EPOCH_BYTE_OFFSET),
            ("scr_prev_period_byte", COL_PREV_PERIOD_BYTE_OFFSET),
            ("scr_new_period_byte", COL_NEW_PERIOD_BYTE_OFFSET),
        ] {
            for k in 0..8 {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: base + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    TBL_8,
                ));
            }
        }

        // epoch_mod_256 cells (2) → 8-bit range (which implies < 256).
        for (label, col) in [
            ("scr_prev_epoch_mod_256", COL_PREV_EPOCH_MOD_256),
            ("scr_new_epoch_mod_256", COL_NEW_EPOCH_MOD_256),
        ] {
            declarations.push((
                LookupDeclaration {
                    label: label.into(),
                    column_index: col,
                    max_bits: 8,
                    selector_column: None,
                },
                TBL_8,
            ));
        }

        // Committee root bytes (4 × 32 = 128) → 8-bit range.
        for (label, base) in [
            ("scr_current_root_byte", COL_CURRENT_COMMITTEE_ROOT_OFFSET),
            ("scr_next_root_byte", COL_NEXT_COMMITTEE_ROOT_OFFSET),
            ("scr_post_current_root_byte", COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET),
            ("scr_post_next_root_byte", COL_POST_NEXT_COMMITTEE_ROOT_OFFSET),
        ] {
            for k in 0..32 {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: base + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    TBL_8,
                ));
            }
        }

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Binds the rotation AIR's `new_epoch` (via its 8 LE bytes) to the
/// `beacon_state_transition_air`'s `EPOCH` column (via its 8 LE bytes).
/// Gated by `IS_REAL` on both sides.
///
/// **Caveat**: this is a multiset binding; the rotation AIR exposes one
/// row per period transition while the state-transition AIR exposes one
/// row per slot. The consumer must wire the appropriate side-shape
/// (typically by replicating the rotation row across all 32 slots of
/// the new epoch, or by adding a strict gate so only the
/// `is_epoch_boundary` rows feed the linkage).
pub fn make_rotation_to_state_transition_descriptor(
    rotation_layer_index: usize,
    state_transition_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::beacon_state_transition_air as bst;
    let a_columns: Vec<usize> =
        (0..8).map(|k| COL_NEW_EPOCH_BYTE_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..8).map(|k| bst::COL_EPOCH_BYTE_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_committee_rotation_to_state_transition_v1".into(),
        a_layer_index: rotation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: state_transition_layer_index,
        b_columns,
        b_selector_column: Some(bst::COL_IS_REAL),
    }
}

/// Binds the post-rotation `post_current_committee_root` (32 bytes) to
/// the sync_committee_filter_air's pubkey-hash registry. The filter AIR
/// exposes per-member pubkey bytes; a downstream consumer would weave
/// the committee-root HTR composition through a hash-extract AIR. As a
/// minimal scaffold, this descriptor binds the 32 committee-root bytes
/// 1:1 to the filter AIR's first 32 pubkey-bytes columns, gated by
/// `IS_REAL` on the A side. The consumer is expected to upgrade the
/// shape once an HTR composition AIR for sync committees lands.
pub fn make_rotation_to_sync_filter_descriptor(
    rotation_layer_index: usize,
    sync_filter_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sync_committee_filter_air as scf;
    let a_columns: Vec<usize> =
        (0..32).map(|k| COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..32).map(|k| scf::COL_PUBKEY_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_committee_rotation_to_sync_filter_v1".into(),
        a_layer_index: rotation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sync_filter_layer_index,
        b_columns,
        b_selector_column: Some(scf::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn root(tag: u8) -> [u8; 32] {
        let mut r = [0u8; 32];
        r[0] = tag;
        r[31] = tag.wrapping_mul(13);
        r
    }

    #[test]
    fn column_layout_pinned() {
        // Pin column layout & counts so any reshuffle is a compile-time
        // failure for downstream consumers.
        assert_eq!(COL_PREV_EPOCH, 0);
        assert_eq!(COL_NEW_EPOCH, 1);
        assert_eq!(COL_PREV_PERIOD, 2);
        assert_eq!(COL_NEW_PERIOD, 3);
        assert_eq!(COL_PREV_EPOCH_BYTE_OFFSET, 4);
        assert_eq!(COL_NEW_EPOCH_BYTE_OFFSET, 12);
        assert_eq!(COL_PREV_PERIOD_BYTE_OFFSET, 20);
        assert_eq!(COL_NEW_PERIOD_BYTE_OFFSET, 28);
        assert_eq!(COL_PREV_EPOCH_MOD_256, 36);
        assert_eq!(COL_NEW_EPOCH_MOD_256, 37);
        assert_eq!(COL_CURRENT_COMMITTEE_ROOT_OFFSET, 38);
        assert_eq!(COL_NEXT_COMMITTEE_ROOT_OFFSET, 70);
        assert_eq!(COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET, 102);
        assert_eq!(COL_POST_NEXT_COMMITTEE_ROOT_OFFSET, 134);
        assert_eq!(COL_IS_PERIOD_BOUNDARY, 166);
        assert_eq!(COL_IS_REAL, 167);
        assert_eq!(NUM_COLUMNS, 168);
        assert_eq!(NUM_ROW_CONSTRAINTS, 10);
        assert_eq!(EPOCHS_PER_SYNC_COMMITTEE_PERIOD, 256);
    }

    #[test]
    fn non_boundary_constraints_zero() {
        // prev_epoch = 100, new_epoch = 101 — both in period 0; no rotation.
        let w = SyncCommitteeRotationWitness::from_rotation(100, root(0x11), root(0x22));
        assert_eq!(w.rows.len(), 1);
        assert!(!w.rows[0].is_period_boundary);
        assert_eq!(w.rows[0].prev_period, 0);
        assert_eq!(w.rows[0].new_period, 0);
        assert_eq!(w.rows[0].post_current_committee_root, w.rows[0].current_committee_root);
        assert_eq!(w.rows[0].post_next_committee_root, w.rows[0].next_committee_root);

        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SyncCommitteeRotationConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "non-boundary constraint {} row {} should be zero", i, r);
            }
        }
    }

    #[test]
    fn boundary_at_epoch_256_constraints_zero_and_rotation_correct() {
        // prev_epoch = 255 (period 0), new_epoch = 256 (period 1) — boundary.
        let cur = root(0xAA);
        let nxt = root(0xBB);
        let w = SyncCommitteeRotationWitness::from_rotation(255, cur, nxt);
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].is_period_boundary, "epoch 255 → 256 must trigger period boundary");
        assert_eq!(w.rows[0].prev_period, 0);
        assert_eq!(w.rows[0].new_period, 1);
        assert_eq!(
            w.rows[0].post_current_committee_root, nxt,
            "post_current must equal next on boundary (rotation)"
        );

        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SyncCommitteeRotationConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "boundary constraint {} row {} should be zero", i, r);
            }
        }
    }

    #[test]
    fn tampered_rotation_detected() {
        // Honest boundary witness, then tamper post_current to NOT match next.
        let cur = root(0x33);
        let nxt = root(0x44);
        let w = SyncCommitteeRotationWitness::from_rotation(255, cur, nxt);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);

        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: set post_current_committee_root[0] back to the OLD current
        // value, so rotation is broken.
        cols[COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET][0] =
            Scalar::from_u64(cur[0] as u64, CurveType::Bls48581);

        let cs = SyncCommitteeRotationConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let evals = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 9 (RLC rotation) must fire on row 0.
        assert!(
            !evals[9][0].is_zero(),
            "tampered rotation must trigger rotation-RLC constraint"
        );
    }

    #[test]
    fn tampered_passthrough_detected() {
        // Honest non-boundary witness; tamper post_next_committee_root.
        let w = SyncCommitteeRotationWitness::from_rotation(10, root(0x55), root(0x66));
        assert!(!w.rows[0].is_period_boundary);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);

        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: change post_next byte 5.
        cols[COL_POST_NEXT_COMMITTEE_ROOT_OFFSET + 5][0] =
            Scalar::from_u64(0xFF, CurveType::Bls48581);

        let cs = SyncCommitteeRotationConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let evals = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 9 (RLC) must fire.
        assert!(
            !evals[9][0].is_zero(),
            "tampered passthrough must trigger rotation-RLC constraint"
        );
    }

    #[test]
    fn tampered_period_step_detected() {
        // Honest boundary witness, then tamper is_period_boundary off on a
        // boundary row — composition (compose constraints) and boundary
        // step constraint should detect the mismatch.
        let w = SyncCommitteeRotationWitness::from_rotation(255, root(0x77), root(0x88));
        let t = build_trace_polynomials(&w, CurveType::Bls48581);

        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: claim new_period is 0 instead of 1 (i.e. no rotation).
        cols[COL_NEW_PERIOD][0] = Scalar::from_u64(0, CurveType::Bls48581);

        let cs = SyncCommitteeRotationConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let evals = cs.evaluate_on_domain(&cr, t.num_rows);
        // The new_period LE decomp (constraint 5) or compose (constraint 7)
        // or boundary step (constraint 8) must fire — at least one of them.
        let any_fires = !evals[5][0].is_zero()
            || !evals[7][0].is_zero()
            || !evals[8][0].is_zero();
        assert!(
            any_fires,
            "tampering new_period must fire decomp/compose/boundary-step constraint"
        );
    }

    #[test]
    fn from_rotation_non_boundary_pass_through() {
        let cur = root(0x10);
        let nxt = root(0x20);
        let w = SyncCommitteeRotationWitness::from_rotation(0, cur, nxt);
        assert_eq!(w.rows[0].prev_epoch, 0);
        assert_eq!(w.rows[0].new_epoch, 1);
        assert!(!w.rows[0].is_period_boundary);
        assert_eq!(w.rows[0].post_current_committee_root, cur);
        assert_eq!(w.rows[0].post_next_committee_root, nxt);
    }

    #[test]
    fn from_rotation_boundary_at_512() {
        // Another boundary: epoch 511 → 512 (period 1 → 2).
        let cur = root(0xC0);
        let nxt = root(0xD0);
        let w = SyncCommitteeRotationWitness::from_rotation(511, cur, nxt);
        assert!(w.rows[0].is_period_boundary);
        assert_eq!(w.rows[0].prev_period, 1);
        assert_eq!(w.rows[0].new_period, 2);
        assert_eq!(w.rows[0].post_current_committee_root, nxt);
    }

    #[test]
    fn rotation_to_state_transition_descriptor_well_formed() {
        use crate::beacon_state_transition_air as bst;
        let d = make_rotation_to_state_transition_descriptor(0, 1);
        assert_eq!(d.label, "sync_committee_rotation_to_state_transition_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), 8);
        assert_eq!(d.b_columns.len(), 8);
        assert_eq!(d.a_columns[0], COL_NEW_EPOCH_BYTE_OFFSET);
        assert_eq!(d.a_columns[7], COL_NEW_EPOCH_BYTE_OFFSET + 7);
        assert_eq!(d.b_columns[0], bst::COL_EPOCH_BYTE_OFFSET);
        assert_eq!(d.b_columns[7], bst::COL_EPOCH_BYTE_OFFSET + 7);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(bst::COL_IS_REAL));
    }

    #[test]
    fn rotation_to_sync_filter_descriptor_well_formed() {
        use crate::sync_committee_filter_air as scf;
        let d = make_rotation_to_sync_filter_descriptor(0, 2);
        assert_eq!(d.label, "sync_committee_rotation_to_sync_filter_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 2);
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        assert_eq!(d.a_columns[0], COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET);
        assert_eq!(d.a_columns[31], COL_POST_CURRENT_COMMITTEE_ROOT_OFFSET + 31);
        assert_eq!(d.b_columns[0], scf::COL_PUBKEY_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(scf::COL_IS_REAL));
    }
}
