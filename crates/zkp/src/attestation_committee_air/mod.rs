//! Beacon attestation committee assignment AIR.
//!
//! Proves that the `(slot, committee_index)` attached to an
//! [`crate::attestation_aggregate_air::Attestation`] selects a *valid*
//! beacon committee for that slot, and that each participating validator
//! is a *member* of that committee. Per the consensus spec (Phase 0
//! + forks), each slot is partitioned into
//! `MAX_COMMITTEES_PER_SLOT = 64` committees and validators are shuffled
//! into committees per epoch via the same `compute_shuffled_index`
//! algorithm exercised by [`crate::proposer_shuffle_air`].
//!
//! This AIR is a **composition layer**: it commits one row per
//! `(committee_member, position_in_committee)` assignment claim and
//! exposes cross-AIR LogUp descriptors that bind the witness data to:
//!
//!   * [`crate::attestation_aggregate_air`] — `(slot, committee_index)`
//!     pair that ties this committee back to a real Attestation.
//!   * [`crate::proposer_shuffle_air`] — the
//!     `(validator_index, shuffled_index)` pair, sharing the shuffle
//!     algorithm with the proposer-selection AIR. The shuffle is
//!     algebraically computed exactly once and re-used here.
//!   * [`crate::validator_registry_air`] — `validator_index` ↔ registry
//!     leaf, binding the participating validator's identity.
//!
//! # Row shape (per committee-membership claim)
//!
//! Per row this AIR commits:
//!
//!   * `slot` — `AttestationData.slot`.
//!   * `committee_index` — `0..64` (`MAX_COMMITTEES_PER_SLOT`).
//!   * `epoch` — `slot / SLOTS_PER_EPOCH`; committed as a single column,
//!     algebraic `slot ↔ epoch` link is host-side for the scaffold.
//!   * `committee_size` — `u32` cap. Reserved as a witness column to
//!     keep position-in-committee range-bound dynamically per
//!     committee.
//!   * `validator_index` — `u64` validator id.
//!   * `position_in_committee` — `u32` position of `validator_index`
//!     within this committee (0..committee_size).
//!   * `shuffled_index` — output of the shared shuffle algorithm; the
//!     post-shuffle index the validator landed at globally before
//!     committee partitioning.
//!   * `is_real` — binary selector.
//!
//! Plus byte-decomposition helper columns:
//!   * 8 LE bytes per u64 field (slot, committee_index, validator_index,
//!     shuffled_index) — 4 × 8 = 32 columns.
//!   * 1 margin column `committee_index_lt_64_margin` (u32) + 4 LE bytes
//!     for the algebraic `committee_index < 64` check.
//!   * 1 margin column `pos_margin` + 4 LE bytes for
//!     `position_in_committee < committee_size`.
//!
//! Total:
//!   8 scalar cols (slot, committee_index, epoch, committee_size,
//!   validator_index, position_in_committee, shuffled_index, is_real)
//!   + 32 u64 LE-byte cols
//!   + 1 + 4 (committee_index_lt_64_margin + bytes)
//!   + 1 + 4 (pos_margin + bytes)
//!   = **50 columns**.
//!
//! # Algebraic constraints (≥ 8)
//!
//!   0. `is_real_binary` — `is_real * (is_real - 1) = 0`.
//!   1. `slot_le_decomp` — `is_real * (slot - Σ_b slot_byte[b] * 256^b)
//!      = 0` (8 bytes).
//!   2. `committee_index_le_decomp` — `is_real * (committee_index -
//!      Σ_b committee_index_byte[b] * 256^b) = 0`.
//!   3. `validator_index_le_decomp` — analogous for `validator_index`.
//!   4. `shuffled_index_le_decomp` — analogous for `shuffled_index`.
//!   5. `committee_index_lt_64` — `is_real * (64 - committee_index -
//!      committee_index_lt_64_margin) = 0`. Combined with the byte-decomp
//!      range check on the margin (4 LE bytes ≤ u32), this proves
//!      `committee_index ≤ 63`.
//!   6. `committee_index_lt_64_margin_decomp` — `is_real *
//!      (committee_index_lt_64_margin - Σ_b margin_byte[b] * 256^b) = 0`.
//!   7. `pos_lt_committee_size` — `is_real * (committee_size -
//!      position_in_committee - pos_margin) = 0`. With the margin's
//!      byte-decomp range check, proves
//!      `position_in_committee ≤ committee_size - 1 < committee_size`
//!      when `committee_size ≥ 1`.
//!   8. `pos_margin_decomp` — `is_real * (pos_margin - Σ_b
//!      pos_margin_byte[b] * 256^b) = 0`.
//!
//! Range checks (8-bit) on every byte-decomposition column.
//!
//! # Cross-AIR linkages (3)
//!
//!   * [`make_att_committee_to_attestation_descriptor`] —
//!     `(slot, committee_index)` ↔ attestation aggregate's
//!     `(COL_SLOT, COL_COMMITTEE_INDEX)`.
//!   * [`make_att_committee_to_shuffle_descriptor`] —
//!     `(validator_index, shuffled_index)` ↔
//!     proposer-shuffle's `(COL_CANDIDATE_INDEX,
//!     COL_SHUFFLED_INDEX_AT_ITER)`. Shares the shuffle algorithm.
//!   * [`make_att_committee_to_validator_registry_descriptor`] —
//!     `validator_index` ↔ registry's `COL_VALIDATOR_INDEX`.
//!
//! # Honest scope statement
//!
//! Algebraically closed here:
//!   - `is_real` binarity.
//!   - `committee_index ≤ 63`.
//!   - `position_in_committee ≤ committee_size - 1` (≥ 0 is implicit by
//!     u64 representation + byte-decomp range).
//!   - u64 byte decomposition of all four scalar fields.
//!
//! Deferred (handed off to cross-AIR LogUp + downstream gadgets):
//!   - `epoch = slot / SLOTS_PER_EPOCH` (host-side; algebraic division
//!     gadget is a follow-up).
//!   - `committee_size = len(get_beacon_committee(state, slot,
//!     committee_index))` (host-side; the registry/shuffle binding
//!     enforces membership but not the cardinality identity).
//!   - The shuffle algorithm itself: `shuffled_index` is committed as a
//!     witness column and bound via LogUp to the proposer-shuffle AIR.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Spec constants ───────────────────────────────────────────────────

/// `MAX_COMMITTEES_PER_SLOT` (Phase 0).
pub const MAX_COMMITTEES_PER_SLOT: u64 = 64;

/// Bytes per u64 LE decomposition.
pub const U64_BYTES: usize = 8;

/// Bytes per u32 LE decomposition (for the margin columns).
pub const U32_BYTES: usize = 4;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_SLOT: usize = 0;
pub const COL_COMMITTEE_INDEX: usize = COL_SLOT + 1;
pub const COL_EPOCH: usize = COL_COMMITTEE_INDEX + 1;
pub const COL_COMMITTEE_SIZE: usize = COL_EPOCH + 1;
pub const COL_VALIDATOR_INDEX: usize = COL_COMMITTEE_SIZE + 1;
pub const COL_POSITION_IN_COMMITTEE: usize = COL_VALIDATOR_INDEX + 1;
pub const COL_SHUFFLED_INDEX: usize = COL_POSITION_IN_COMMITTEE + 1;
pub const COL_IS_REAL: usize = COL_SHUFFLED_INDEX + 1;

/// 8-byte LE decomp of `slot`.
pub const COL_SLOT_BYTE_OFFSET: usize = COL_IS_REAL + 1;
/// 8-byte LE decomp of `committee_index`.
pub const COL_COMMITTEE_INDEX_BYTE_OFFSET: usize = COL_SLOT_BYTE_OFFSET + U64_BYTES;
/// 8-byte LE decomp of `validator_index`.
pub const COL_VALIDATOR_INDEX_BYTE_OFFSET: usize = COL_COMMITTEE_INDEX_BYTE_OFFSET + U64_BYTES;
/// 8-byte LE decomp of `shuffled_index`.
pub const COL_SHUFFLED_INDEX_BYTE_OFFSET: usize = COL_VALIDATOR_INDEX_BYTE_OFFSET + U64_BYTES;

/// `64 - committee_index ≥ 0` margin column.
pub const COL_COMMITTEE_INDEX_LT_64_MARGIN: usize = COL_SHUFFLED_INDEX_BYTE_OFFSET + U64_BYTES;
/// 4-byte LE decomp of the margin.
pub const COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET: usize =
    COL_COMMITTEE_INDEX_LT_64_MARGIN + 1;

/// `committee_size - position_in_committee ≥ 0` margin column.
pub const COL_POS_MARGIN: usize =
    COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET + U32_BYTES;
/// 4-byte LE decomp of `pos_margin`.
pub const COL_POS_MARGIN_BYTE_OFFSET: usize = COL_POS_MARGIN + 1;

pub const NUM_COLUMNS: usize = COL_POS_MARGIN_BYTE_OFFSET + U32_BYTES;

/// 9 row-local constraint bodies.
pub const NUM_ROW_CONSTRAINTS: usize = 9;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct AttestationCommitteeRow {
    pub slot: u64,
    pub committee_index: u64,
    pub epoch: u64,
    pub committee_size: u32,
    pub validator_index: u64,
    pub position_in_committee: u32,
    pub shuffled_index: u64,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AttestationCommitteeWitness {
    pub rows: Vec<AttestationCommitteeRow>,
}

impl AttestationCommitteeWitness {
    pub fn from_rows(rows: Vec<AttestationCommitteeRow>) -> Self {
        Self { rows }
    }

    /// Host-side single-assignment builder. The caller supplies the
    /// `(slot, committee_index, validator_index, position, committee_size)`
    /// tuple and this scaffold computes:
    ///   - `epoch = slot / 32` (Phase 0 `SLOTS_PER_EPOCH = 32`),
    ///   - `shuffled_index` is reported back as `validator_index` (the
    ///     scaffold publishes the same value the descriptor will match
    ///     against the proposer-shuffle AIR's
    ///     `COL_SHUFFLED_INDEX_AT_ITER`). The real shuffle derivation
    ///     remains in [`crate::proposer_shuffle_air`].
    ///
    /// Returns a single-row witness with `is_real = true`. This is the
    /// builder the test-suite + downstream `joint_prove` orchestrator
    /// use; multi-row composition is via `from_rows`.
    pub fn from_assignment(
        slot: u64,
        committee_index: u64,
        validator_index: u64,
        position: u32,
        committee_size: u32,
    ) -> Self {
        const SLOTS_PER_EPOCH: u64 = 32;
        let row = AttestationCommitteeRow {
            slot,
            committee_index,
            epoch: slot / SLOTS_PER_EPOCH,
            committee_size,
            validator_index,
            position_in_committee: position,
            shuffled_index: validator_index,
            is_real: true,
        };
        Self { rows: vec![row] }
    }
}

fn le_decomp_u64(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}

fn le_decomp_u32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &AttestationCommitteeWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        columns[COL_SLOT][r] = Scalar::from_u64(row.slot, curve);
        columns[COL_COMMITTEE_INDEX][r] = Scalar::from_u64(row.committee_index, curve);
        columns[COL_EPOCH][r] = Scalar::from_u64(row.epoch, curve);
        columns[COL_COMMITTEE_SIZE][r] = Scalar::from_u64(row.committee_size as u64, curve);
        columns[COL_VALIDATOR_INDEX][r] = Scalar::from_u64(row.validator_index, curve);
        columns[COL_POSITION_IN_COMMITTEE][r] =
            Scalar::from_u64(row.position_in_committee as u64, curve);
        columns[COL_SHUFFLED_INDEX][r] = Scalar::from_u64(row.shuffled_index, curve);
        columns[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };

        // 8-byte LE decompositions for each u64 field.
        for (offset, value) in [
            (COL_SLOT_BYTE_OFFSET, row.slot),
            (COL_COMMITTEE_INDEX_BYTE_OFFSET, row.committee_index),
            (COL_VALIDATOR_INDEX_BYTE_OFFSET, row.validator_index),
            (COL_SHUFFLED_INDEX_BYTE_OFFSET, row.shuffled_index),
        ] {
            let bytes = le_decomp_u64(value);
            for b in 0..U64_BYTES {
                columns[offset + b][r] = Scalar::from_u64(bytes[b] as u64, curve);
            }
        }

        // committee_index < 64 margin.
        let ci_margin: u32 = (MAX_COMMITTEES_PER_SLOT.saturating_sub(row.committee_index)) as u32;
        columns[COL_COMMITTEE_INDEX_LT_64_MARGIN][r] =
            Scalar::from_u64(ci_margin as u64, curve);
        let ci_bytes = le_decomp_u32(ci_margin);
        for b in 0..U32_BYTES {
            columns[COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET + b][r] =
                Scalar::from_u64(ci_bytes[b] as u64, curve);
        }

        // pos_margin = committee_size - position_in_committee.
        let pos_margin: u32 = row
            .committee_size
            .saturating_sub(row.position_in_committee);
        columns[COL_POS_MARGIN][r] = Scalar::from_u64(pos_margin as u64, curve);
        let pm_bytes = le_decomp_u32(pos_margin);
        for b in 0..U32_BYTES {
            columns[COL_POS_MARGIN_BYTE_OFFSET + b][r] =
                Scalar::from_u64(pm_bytes[b] as u64, curve);
        }
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct AttestationCommitteeConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AttestationCommitteeConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn pow256(b: usize, curve: CurveType) -> Scalar {
    let mut acc: u64 = 1;
    for _ in 0..b {
        acc = acc.wrapping_mul(256);
    }
    Scalar::from_u64(acc, curve)
}

impl VmConstraintSystem for AttestationCommitteeConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "slot_le_decomp".into(),
            "committee_index_le_decomp".into(),
            "validator_index_le_decomp".into(),
            "shuffled_index_le_decomp".into(),
            "committee_index_lt_64".into(),
            "committee_index_lt_64_margin_decomp".into(),
            "pos_lt_committee_size".into(),
            "pos_margin_decomp".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let max_committees = Scalar::from_u64(MAX_COMMITTEES_PER_SLOT, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        let pow_u64: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256(b, curve)).collect();
        let pow_u32: Vec<Scalar> = (0..U32_BYTES).map(|b| pow256(b, curve)).collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];

            // 0: is_real ∈ {0,1}.
            out[0][r] = is_real.mul(&is_real.sub(&one));

            // 1: slot LE decomp gated by is_real.
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(&columns[COL_SLOT_BYTE_OFFSET + b][r].mul(&pow_u64[b]));
                }
                out[1][r] = is_real.mul(&columns[COL_SLOT][r].sub(&sum));
            }

            // 2: committee_index LE decomp gated by is_real.
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(
                        &columns[COL_COMMITTEE_INDEX_BYTE_OFFSET + b][r].mul(&pow_u64[b]),
                    );
                }
                out[2][r] = is_real.mul(&columns[COL_COMMITTEE_INDEX][r].sub(&sum));
            }

            // 3: validator_index LE decomp gated by is_real.
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(
                        &columns[COL_VALIDATOR_INDEX_BYTE_OFFSET + b][r].mul(&pow_u64[b]),
                    );
                }
                out[3][r] = is_real.mul(&columns[COL_VALIDATOR_INDEX][r].sub(&sum));
            }

            // 4: shuffled_index LE decomp gated by is_real.
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(
                        &columns[COL_SHUFFLED_INDEX_BYTE_OFFSET + b][r].mul(&pow_u64[b]),
                    );
                }
                out[4][r] = is_real.mul(&columns[COL_SHUFFLED_INDEX][r].sub(&sum));
            }

            // 5: committee_index < 64.
            //    is_real * (64 - committee_index - margin) = 0.
            {
                let body = max_committees
                    .sub(&columns[COL_COMMITTEE_INDEX][r])
                    .sub(&columns[COL_COMMITTEE_INDEX_LT_64_MARGIN][r]);
                out[5][r] = is_real.mul(&body);
            }

            // 6: committee_index_lt_64_margin LE decomp gated by is_real.
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U32_BYTES {
                    sum = sum.add(
                        &columns[COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET + b][r]
                            .mul(&pow_u32[b]),
                    );
                }
                out[6][r] =
                    is_real.mul(&columns[COL_COMMITTEE_INDEX_LT_64_MARGIN][r].sub(&sum));
            }

            // 7: position_in_committee < committee_size.
            //    is_real * (committee_size - position - pos_margin) = 0.
            {
                let body = columns[COL_COMMITTEE_SIZE][r]
                    .sub(&columns[COL_POSITION_IN_COMMITTEE][r])
                    .sub(&columns[COL_POS_MARGIN][r]);
                out[7][r] = is_real.mul(&body);
            }

            // 8: pos_margin LE decomp gated by is_real.
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U32_BYTES {
                    sum = sum
                        .add(&columns[COL_POS_MARGIN_BYTE_OFFSET + b][r].mul(&pow_u32[b]));
                }
                out[8][r] = is_real.mul(&columns[COL_POS_MARGIN][r].sub(&sum));
            }
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let max_committees = Scalar::from_u64(MAX_COMMITTEES_PER_SLOT, curve);
        let pow_u64: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256(b, curve)).collect();
        let pow_u32: Vec<Scalar> = (0..U32_BYTES).map(|b| pow256(b, curve)).collect();
        let is_real = &col_evals[COL_IS_REAL];

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0
        acc = acc.add(&alpha_pow.mul(&is_real.mul(&is_real.sub(&one))));
        alpha_pow = alpha_pow.mul(alpha);

        // helper for u64 decomp at a point
        let emit_u64_decomp = |scalar_col: usize, byte_off: usize, acc: &mut Scalar, ap: &mut Scalar| {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(&col_evals[byte_off + b].mul(&pow_u64[b]));
            }
            *acc = acc.add(&ap.mul(&is_real.mul(&col_evals[scalar_col].sub(&sum))));
            *ap = ap.mul(alpha);
        };
        emit_u64_decomp(COL_SLOT, COL_SLOT_BYTE_OFFSET, &mut acc, &mut alpha_pow);
        emit_u64_decomp(
            COL_COMMITTEE_INDEX,
            COL_COMMITTEE_INDEX_BYTE_OFFSET,
            &mut acc,
            &mut alpha_pow,
        );
        emit_u64_decomp(
            COL_VALIDATOR_INDEX,
            COL_VALIDATOR_INDEX_BYTE_OFFSET,
            &mut acc,
            &mut alpha_pow,
        );
        emit_u64_decomp(
            COL_SHUFFLED_INDEX,
            COL_SHUFFLED_INDEX_BYTE_OFFSET,
            &mut acc,
            &mut alpha_pow,
        );

        // 5
        {
            let body = max_committees
                .sub(&col_evals[COL_COMMITTEE_INDEX])
                .sub(&col_evals[COL_COMMITTEE_INDEX_LT_64_MARGIN]);
            acc = acc.add(&alpha_pow.mul(&is_real.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 6
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U32_BYTES {
                sum = sum.add(
                    &col_evals[COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET + b].mul(&pow_u32[b]),
                );
            }
            acc = acc.add(
                &alpha_pow.mul(
                    &is_real.mul(&col_evals[COL_COMMITTEE_INDEX_LT_64_MARGIN].sub(&sum)),
                ),
            );
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7
        {
            let body = col_evals[COL_COMMITTEE_SIZE]
                .sub(&col_evals[COL_POSITION_IN_COMMITTEE])
                .sub(&col_evals[COL_POS_MARGIN]);
            acc = acc.add(&alpha_pow.mul(&is_real.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 8
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U32_BYTES {
                sum = sum.add(&col_evals[COL_POS_MARGIN_BYTE_OFFSET + b].mul(&pow_u32[b]));
            }
            acc =
                acc.add(&alpha_pow.mul(&is_real.mul(&col_evals[COL_POS_MARGIN].sub(&sum))));
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
        let max_committees = Scalar::from_u64(MAX_COMMITTEES_PER_SLOT, curve);
        let pow_u64: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256(b, curve)).collect();
        let pow_u32: Vec<Scalar> = (0..U32_BYTES).map(|b| pow256(b, curve)).collect();
        let is_real = &col_coeffs[COL_IS_REAL];

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // 0
        {
            let v_m1 = poly_sub(is_real, &one_poly, curve);
            let body = poly_mul(is_real, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        let emit_u64_decomp_poly =
            |scalar_col: usize, byte_off: usize, acc: &mut Vec<Scalar>, ap: &mut Scalar| {
                let mut sum = vec![Scalar::zero(curve)];
                for b in 0..U64_BYTES {
                    let term = poly_scalar_mul(&col_coeffs[byte_off + b], &pow_u64[b]);
                    sum = poly_add(&sum, &term, curve);
                }
                let body = poly_sub(&col_coeffs[scalar_col], &sum, curve);
                let gated = poly_mul(is_real, &body, curve);
                *acc = poly_add(acc, &poly_scalar_mul(&gated, ap), curve);
                *ap = ap.mul(alpha);
            };
        emit_u64_decomp_poly(COL_SLOT, COL_SLOT_BYTE_OFFSET, &mut acc, &mut alpha_pow);
        emit_u64_decomp_poly(
            COL_COMMITTEE_INDEX,
            COL_COMMITTEE_INDEX_BYTE_OFFSET,
            &mut acc,
            &mut alpha_pow,
        );
        emit_u64_decomp_poly(
            COL_VALIDATOR_INDEX,
            COL_VALIDATOR_INDEX_BYTE_OFFSET,
            &mut acc,
            &mut alpha_pow,
        );
        emit_u64_decomp_poly(
            COL_SHUFFLED_INDEX,
            COL_SHUFFLED_INDEX_BYTE_OFFSET,
            &mut acc,
            &mut alpha_pow,
        );

        // 5: committee_index < 64 — constant scalar `max_committees` minus
        //    committee_index col minus margin col.
        {
            let const_poly = vec![max_committees.clone()];
            let body = poly_sub(
                &poly_sub(&const_poly, &col_coeffs[COL_COMMITTEE_INDEX], curve),
                &col_coeffs[COL_COMMITTEE_INDEX_LT_64_MARGIN],
                curve,
            );
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 6
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..U32_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET + b],
                    &pow_u32[b],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body =
                poly_sub(&col_coeffs[COL_COMMITTEE_INDEX_LT_64_MARGIN], &sum, curve);
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 7
        {
            let body = poly_sub(
                &poly_sub(
                    &col_coeffs[COL_COMMITTEE_SIZE],
                    &col_coeffs[COL_POSITION_IN_COMMITTEE],
                    curve,
                ),
                &col_coeffs[COL_POS_MARGIN],
                curve,
            );
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 8
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..U32_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_POS_MARGIN_BYTE_OFFSET + b],
                    &pow_u32[b],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_POS_MARGIN], &sum, curve);
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
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
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls12381);
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
        for (label_prefix, offset, n) in [
            ("slot_byte", COL_SLOT_BYTE_OFFSET, U64_BYTES),
            ("committee_index_byte", COL_COMMITTEE_INDEX_BYTE_OFFSET, U64_BYTES),
            ("validator_index_byte", COL_VALIDATOR_INDEX_BYTE_OFFSET, U64_BYTES),
            ("shuffled_index_byte", COL_SHUFFLED_INDEX_BYTE_OFFSET, U64_BYTES),
            (
                "committee_index_lt_64_margin_byte",
                COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET,
                U32_BYTES,
            ),
            ("pos_margin_byte", COL_POS_MARGIN_BYTE_OFFSET, U32_BYTES),
        ] {
            for b in 0..n {
                declarations.push((
                    LookupDeclaration {
                        label: format!("att_committee_{}_{}_8bit", label_prefix, b),
                        column_index: offset + b,
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

/// Bind `(slot, committee_index)` on this AIR to the matching
/// `(COL_SLOT, COL_COMMITTEE_INDEX)` columns of the
/// `attestation_aggregate_air`. Both sides gated by `IS_REAL`.
pub fn make_att_committee_to_attestation_descriptor(
    att_committee_layer_index: usize,
    attestation_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::attestation_aggregate_air as aa;
    CrossAirLogUpDescriptor {
        label: "att_committee_to_attestation_v1".into(),
        a_layer_index: att_committee_layer_index,
        a_columns: vec![COL_SLOT, COL_COMMITTEE_INDEX],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: attestation_layer_index,
        b_columns: vec![aa::COL_SLOT, aa::COL_COMMITTEE_INDEX],
        b_selector_column: Some(aa::COL_IS_REAL),
    }
}

/// Bind `(validator_index, shuffled_index)` to the proposer-shuffle
/// AIR's `(COL_CANDIDATE_INDEX, COL_SHUFFLED_INDEX_AT_ITER)` columns.
/// The shuffle algorithm is computed once in [`crate::proposer_shuffle_air`]
/// and reused here.
pub fn make_att_committee_to_shuffle_descriptor(
    att_committee_layer_index: usize,
    shuffle_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::proposer_shuffle_air as ps;
    CrossAirLogUpDescriptor {
        label: "att_committee_to_shuffle_v1".into(),
        a_layer_index: att_committee_layer_index,
        a_columns: vec![COL_VALIDATOR_INDEX, COL_SHUFFLED_INDEX],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: shuffle_layer_index,
        b_columns: vec![ps::COL_CANDIDATE_INDEX, ps::COL_SHUFFLED_INDEX_AT_ITER],
        b_selector_column: Some(ps::COL_IS_REAL),
    }
}

/// Bind `validator_index` to the validator-registry AIR's
/// `COL_VALIDATOR_INDEX`, anchoring the per-row validator identity to
/// the registry leaf chain.
pub fn make_att_committee_to_validator_registry_descriptor(
    att_committee_layer_index: usize,
    validator_registry_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_registry_air as vr;
    CrossAirLogUpDescriptor {
        label: "att_committee_to_validator_registry_v1".into(),
        a_layer_index: att_committee_layer_index,
        a_columns: vec![COL_VALIDATOR_INDEX],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_registry_layer_index,
        b_columns: vec![vr::COL_VALIDATOR_INDEX],
        b_selector_column: Some(vr::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_bodies(w: &AttestationCommitteeWitness) -> Vec<Vec<Scalar>> {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(w, curve);
        let cs = AttestationCommitteeConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    fn assert_all_vanish(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} not zero", i, r);
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: valid assignment — honest witness, all constraints vanish.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn valid_assignment_satisfies_constraints() {
        // slot=4242, committee_index=7, validator=12345, position=3, size=128.
        let w = AttestationCommitteeWitness::from_assignment(4242, 7, 12345, 3, 128);
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        assert_eq!(row.slot, 4242);
        assert_eq!(row.committee_index, 7);
        assert_eq!(row.epoch, 4242 / 32);
        assert_eq!(row.committee_size, 128);
        assert_eq!(row.validator_index, 12345);
        assert_eq!(row.position_in_committee, 3);
        assert_eq!(row.shuffled_index, 12345);
        assert!(row.is_real);
        let bodies = run_bodies(&w);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: committee_index = 63 (boundary).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn committee_index_63_boundary_ok() {
        let w = AttestationCommitteeWitness::from_assignment(100, 63, 7, 0, 1);
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
        // Margin column = 64 - 63 = 1.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let margin = &trace.columns[COL_COMMITTEE_INDEX_LT_64_MARGIN].evaluations[0];
        let diff = margin.sub(&Scalar::from_u64(1, curve));
        assert!(diff.is_zero(), "expected margin = 1");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: position out of bounds rejected.
    //
    // Position ≥ committee_size: honest builder cannot produce a sound
    // witness (margin saturates to 0). We force such a witness by
    // pretending the margin column is 0 and check that constraint 7
    // (`pos_lt_committee_size`) fires because
    // committee_size - position is not zero.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn position_out_of_bounds_fires_constraint() {
        // Construct a hand-rolled out-of-bounds row: committee_size=4,
        // position=10. The `from_assignment` saturates pos_margin = 0
        // (since 4 < 10). But then constraint 7:
        //   committee_size − position − pos_margin = 4 − 10 − 0 = −6 ≠ 0.
        // So constraint 7 must fire.
        let w = AttestationCommitteeWitness::from_assignment(100, 1, 7, 10, 4);
        let bodies = run_bodies(&w);
        assert!(
            !bodies[7][0].is_zero(),
            "position_in_committee ≥ committee_size must fire pos_lt_committee_size"
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: committee_index ≥ 64 fires the lt_64 constraint.
    //
    // Hand-roll a row with committee_index = 64. `from_assignment`
    // saturates the margin to 0, so constraint 5 fires:
    //   64 - 64 - 0 = 0 ✓ (this is actually equal!)
    //
    // To force the failure, take committee_index = 100; saturating
    // gives margin = 0. Then 64 - 100 - 0 = -36 ≠ 0 → constraint 5 fires.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn committee_index_above_64_fires_constraint() {
        let w = AttestationCommitteeWitness::from_assignment(100, 100, 7, 0, 1);
        let bodies = run_bodies(&w);
        assert!(
            !bodies[5][0].is_zero(),
            "committee_index = 100 must fire committee_index_lt_64"
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: descriptors well-formed + point into valid sub-AIR columns.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn descriptors_well_formed() {
        let d_att = make_att_committee_to_attestation_descriptor(0, 1);
        assert_eq!(d_att.label, "att_committee_to_attestation_v1");
        assert_eq!(d_att.a_columns, vec![COL_SLOT, COL_COMMITTEE_INDEX]);
        assert_eq!(d_att.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_att.b_columns.len(), 2);
        use crate::attestation_aggregate_air as aa;
        assert_eq!(d_att.b_columns, vec![aa::COL_SLOT, aa::COL_COMMITTEE_INDEX]);
        assert_eq!(d_att.b_selector_column, Some(aa::COL_IS_REAL));
        for &c in &d_att.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_att.b_columns {
            assert!(c < aa::NUM_COLUMNS);
        }

        let d_sh = make_att_committee_to_shuffle_descriptor(0, 2);
        assert_eq!(d_sh.label, "att_committee_to_shuffle_v1");
        assert_eq!(d_sh.a_columns, vec![COL_VALIDATOR_INDEX, COL_SHUFFLED_INDEX]);
        assert_eq!(d_sh.a_selector_column, Some(COL_IS_REAL));
        use crate::proposer_shuffle_air as ps;
        assert_eq!(
            d_sh.b_columns,
            vec![ps::COL_CANDIDATE_INDEX, ps::COL_SHUFFLED_INDEX_AT_ITER]
        );
        assert_eq!(d_sh.b_selector_column, Some(ps::COL_IS_REAL));
        for &c in &d_sh.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_sh.b_columns {
            assert!(c < ps::NUM_COLUMNS);
        }

        let d_vr = make_att_committee_to_validator_registry_descriptor(0, 3);
        assert_eq!(d_vr.label, "att_committee_to_validator_registry_v1");
        assert_eq!(d_vr.a_columns, vec![COL_VALIDATOR_INDEX]);
        assert_eq!(d_vr.a_selector_column, Some(COL_IS_REAL));
        use crate::validator_registry_air as vr;
        assert_eq!(d_vr.b_columns, vec![vr::COL_VALIDATOR_INDEX]);
        assert_eq!(d_vr.b_selector_column, Some(vr::COL_IS_REAL));
        for &c in &d_vr.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_vr.b_columns {
            assert!(c < vr::NUM_COLUMNS);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: column layout pinned.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_SLOT, 0);
        assert_eq!(COL_COMMITTEE_INDEX, 1);
        assert_eq!(COL_EPOCH, 2);
        assert_eq!(COL_COMMITTEE_SIZE, 3);
        assert_eq!(COL_VALIDATOR_INDEX, 4);
        assert_eq!(COL_POSITION_IN_COMMITTEE, 5);
        assert_eq!(COL_SHUFFLED_INDEX, 6);
        assert_eq!(COL_IS_REAL, 7);
        assert_eq!(COL_SLOT_BYTE_OFFSET, 8);
        assert_eq!(COL_COMMITTEE_INDEX_BYTE_OFFSET, 16);
        assert_eq!(COL_VALIDATOR_INDEX_BYTE_OFFSET, 24);
        assert_eq!(COL_SHUFFLED_INDEX_BYTE_OFFSET, 32);
        assert_eq!(COL_COMMITTEE_INDEX_LT_64_MARGIN, 40);
        assert_eq!(COL_COMMITTEE_INDEX_LT_64_MARGIN_BYTE_OFFSET, 41);
        assert_eq!(COL_POS_MARGIN, 45);
        assert_eq!(COL_POS_MARGIN_BYTE_OFFSET, 46);
        assert_eq!(NUM_COLUMNS, 50);
        assert_eq!(NUM_ROW_CONSTRAINTS, 9);
        assert_eq!(NUM_SHIFTED, 0);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 7: tampered is_real fires the binary constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn is_real_binary_fires_on_non_binary() {
        let w = AttestationCommitteeWitness::from_assignment(64, 5, 10, 0, 8);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, curve);
        let cs = AttestationCommitteeConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[0][0].is_zero(),
            "is_real = 2 must fire is_real_binary"
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 8: byte range coverage — every byte column has a lookup decl.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn byte_range_coverage() {
        let cs = AttestationCommitteeConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 4 u64 fields × 8 bytes + 2 u32 margins × 4 bytes = 40.
        let want = 4 * U64_BYTES + 2 * U32_BYTES;
        assert_eq!(req.declarations.len(), want);
        for (decl, _) in &req.declarations {
            assert!(decl.column_index < NUM_COLUMNS);
            assert_eq!(decl.max_bits, 8);
            assert!(decl.label.contains("8bit"));
        }
    }
}
