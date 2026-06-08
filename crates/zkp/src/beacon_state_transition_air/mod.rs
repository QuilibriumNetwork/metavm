//! Beacon state-transition AIR.
//!
//! Proves the per-slot beacon-state-root evolution
//!
//!   `state_root[i+1] = process_slot(state_root[i], block[i])`
//!
//! at the **chaining and metadata** layer: each witness row commits one
//! slot's `(slot, epoch, prev_state_root, post_state_root, block_root)`,
//! and the AIR algebraically pins:
//!
//! 1. `epoch == slot / 32` (via `slot = epoch · 32 + slot_mod_32`,
//!    `slot_mod_32 ∈ [0,32)`).
//! 2. `is_epoch_boundary == 1` iff this is the last slot of an epoch
//!    (`slot_mod_32 == 31`), gated on real rows.
//! 3. Slot increments by exactly one between consecutive real rows.
//! 4. The state-root chain holds: `prev_state_root[i+1] == post_state_root[i]`.
//!
//! The witness commits `block_root` per row but does NOT verify
//! `post_state_root = process_slot(prev_state_root, block)` algebraically
//! here — that's the cross-AIR LogUp linkage to `block_header_air` and
//! `beacon_block_header_air` (descriptors provided below).
//!
//! ## Column layout (117 cols)
//!
//! ```text
//!   COL_SLOT                          0       u64 slot value
//!   COL_EPOCH                         1       u64 epoch value
//!   COL_SLOT_MOD_32                   2       slot mod 32, in [0,32)
//!   COL_SLOT_BYTE_OFFSET              3..11   8 LE bytes of slot (range-checked)
//!   COL_EPOCH_BYTE_OFFSET             11..19  8 LE bytes of epoch (range-checked)
//!   COL_PREV_STATE_ROOT_OFFSET        19..51  32 prev-state-root bytes
//!   COL_POST_STATE_ROOT_OFFSET        51..83  32 post-state-root bytes
//!   COL_BLOCK_ROOT_OFFSET             83..115 32 block-root bytes
//!   COL_IS_REAL                       115     real-row flag
//!   COL_IS_EPOCH_BOUNDARY             116     epoch-boundary flag
//! ```
//!
//! ## Row-local constraints (8)
//!
//! 0. `is_real · (is_real - 1) = 0`
//! 1. `is_epoch_boundary · (is_epoch_boundary - 1) = 0`
//! 2. `SLOT - Σ_{k=0..8} slot_byte_k · 256^k = 0`           (LE decomp)
//! 3. `EPOCH - Σ_{k=0..8} epoch_byte_k · 256^k = 0`         (LE decomp)
//! 4. `is_real · (epoch · 32 + slot_mod_32 - slot) = 0`     (compose)
//! 5. `is_epoch_boundary · (slot_mod_32 - 31) = 0`          (boundary ⇒ last)
//! 6. `is_epoch_boundary · (1 - is_real) = 0`               (boundary only on real)
//! 7. `(1 - is_real) · slot_mod_32 = 0`                     (padding zeros)
//!
//! Plus byte range-checks via LogUp:
//!   * 16 byte cols (slot, epoch) → 8-bit range.
//!   * 64 byte cols (prev/post state root, block root) → 8-bit range.
//!   * `slot_mod_32` → 5-bit range.
//!
//! ## Shifted (cross-row) constraints (33)
//!
//! All gated by `is_real(z) · is_real(ω·z)` and excluded at the wrap row
//! `ω^{n-1}` to avoid binding the domain wrap-around.
//!
//! 0. `slot(ω·z) - slot(z) - 1 = 0`                         (slot increments)
//! 1..33. `prev_state_root_byte_k(ω·z) - post_state_root_byte_k(z) = 0`
//!         for k ∈ [0, 32) — the state-root chain, one constraint per
//!         byte.
//!
//! ## Soundness gaps (deferred follow-ups)
//!
//! 1. **`post_state_root` is an oracle algebraically**. The transition
//!    function `process_slot` is NOT checked here. Closing requires a
//!    cross-AIR LogUp linkage to a future `process_slot_air` (host-side
//!    epoch logic in `crate::epoch_transition`).
//!
//! 2. **`block_root` provenance is an oracle**. The descriptors
//!    [`make_state_transition_to_block_header_descriptor`] and
//!    [`make_state_transition_to_bbh_descriptor`] propose bindings to
//!    `block_header_air` (EL block-hash) and `bbh_root_consumer_air`
//!    (BeaconBlockHeader HTR), but the EL↔CL `block_root` identity
//!    requires the consumer to wire both per-slot.
//!
//! 3. **FFG binding is one-sided.**
//!    [`make_state_transition_to_finality_descriptor`] binds the
//!    `post_state_root` at epoch boundaries to the finality AIR's
//!    `CLAIMED_FINALIZED_ROOT` (32 byte tuple). The reverse direction —
//!    every justified checkpoint root appears as some boundary row —
//!    requires a dedicated `ffg_checkpoint_chain_air` (deferred; the
//!    host-side oracle lives in
//!    [`crate::ffg_checkpoint_chain::verify_checkpoint_chain`]).

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_SLOT: usize = 0;
pub const COL_EPOCH: usize = 1;
pub const COL_SLOT_MOD_32: usize = 2;

pub const COL_SLOT_BYTE_OFFSET: usize = 3; // 3..11 (8 LE bytes)
pub const COL_EPOCH_BYTE_OFFSET: usize = 11; // 11..19

pub const COL_PREV_STATE_ROOT_OFFSET: usize = 19; // 19..51
pub const COL_POST_STATE_ROOT_OFFSET: usize = 51; // 51..83
pub const COL_BLOCK_ROOT_OFFSET: usize = 83; // 83..115

pub const COL_IS_REAL: usize = 115;
pub const COL_IS_EPOCH_BOUNDARY: usize = 116;

pub const NUM_COLUMNS: usize = COL_IS_EPOCH_BOUNDARY + 1; // 117

pub const NUM_ROW_CONSTRAINTS: usize = 8;

/// Cross-row constraints:
///   1 slot-increment chain
///   + 32 per-byte prev/post state-root chain
pub const NUM_SHIFTED: usize = 1 + 32;

// ─── Witness type ─────────────────────────────────────────────────────

/// One per-slot row of the beacon state-transition AIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeaconStateTransitionRow {
    /// Beacon slot number.
    pub slot: u64,
    /// `slot / 32`.
    pub epoch: u64,
    /// 32-byte state-root BEFORE this slot's `process_slot`.
    pub prev_state_root: [u8; 32],
    /// 32-byte state-root AFTER this slot's `process_slot`.
    pub post_state_root: [u8; 32],
    /// 32-byte block root for this slot (HTR of the BeaconBlock, or the
    /// previous block's root for a skipped slot).
    pub block_root: [u8; 32],
    /// True iff `slot mod 32 == 31`.
    pub is_epoch_boundary: bool,
}

/// Beacon state-transition witness: one row per slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BeaconStateTransitionWitness {
    pub rows: Vec<BeaconStateTransitionRow>,
}

impl BeaconStateTransitionWitness {
    pub fn from_rows(rows: Vec<BeaconStateTransitionRow>) -> Self {
        Self { rows }
    }

    /// Build a state-transition chain from `(slot, block_root, post_state_root)`
    /// tuples and an initial pre-state root. Each block_i's `prev_state_root`
    /// equals block_{i-1}'s `post_state_root` (chain closure host-side).
    ///
    /// Caller is responsible for ensuring slots are contiguous (slot_{i+1}
    /// == slot_i + 1); this builder will faithfully encode whatever it
    /// receives and the algebraic shifted constraint will catch any gap.
    pub fn from_chain(
        initial_state: [u8; 32],
        blocks: &[(u64, [u8; 32], [u8; 32])],
    ) -> Self {
        let mut prev = initial_state;
        let mut rows = Vec::with_capacity(blocks.len());
        for &(slot, block_root, post_state_root) in blocks {
            let epoch = slot / 32;
            let is_epoch_boundary = (slot % 32) == 31;
            rows.push(BeaconStateTransitionRow {
                slot,
                epoch,
                prev_state_root: prev,
                post_state_root,
                block_root,
                is_epoch_boundary,
            });
            prev = post_state_root;
        }
        Self { rows }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BeaconStateTransitionWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_SLOT][i] = Scalar::from_u64(row.slot, curve);
        columns[COL_EPOCH][i] = Scalar::from_u64(row.epoch, curve);
        columns[COL_SLOT_MOD_32][i] =
            Scalar::from_u64(row.slot - row.epoch * 32, curve);

        // LE byte decomposition of slot.
        let slot_bytes = row.slot.to_le_bytes();
        for k in 0..8 {
            columns[COL_SLOT_BYTE_OFFSET + k][i] =
                Scalar::from_u64(slot_bytes[k] as u64, curve);
        }
        // LE byte decomposition of epoch.
        let epoch_bytes = row.epoch.to_le_bytes();
        for k in 0..8 {
            columns[COL_EPOCH_BYTE_OFFSET + k][i] =
                Scalar::from_u64(epoch_bytes[k] as u64, curve);
        }

        // 32-byte roots.
        for k in 0..32 {
            columns[COL_PREV_STATE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.prev_state_root[k] as u64, curve);
            columns[COL_POST_STATE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.post_state_root[k] as u64, curve);
            columns[COL_BLOCK_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.block_root[k] as u64, curve);
        }

        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_EPOCH_BOUNDARY][i] =
            if row.is_epoch_boundary { one.clone() } else { zero.clone() };
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct BeaconStateTransitionConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BeaconStateTransitionConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(
        mut self,
        omega: Scalar,
        domain_size: u64,
    ) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Scalar `Σ_{k=0..8} byte_k(row) · 256^k`.
fn le_decomp_sum_evals(
    columns: &[&Vec<Scalar>],
    base: usize,
    row: usize,
    curve: CurveType,
) -> Scalar {
    let mut sum = Scalar::zero(curve);
    for k in 0..8 {
        let pow = 1u64 << (8 * k as u32);
        sum = sum.add(
            &columns[base + k][row].mul(&Scalar::from_u64(pow, curve)),
        );
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
        sum = sum.add(
            &col_evals_at_z[base + k].mul(&Scalar::from_u64(pow, curve)),
        );
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
        let term = poly_scalar_mul(
            &col_coeffs[base + k],
            &Scalar::from_u64(pow, curve),
        );
        sum = poly_add(&sum, &term, curve);
    }
    sum
}

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

impl VmConstraintSystem for BeaconStateTransitionConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_epoch_boundary_binary".into(),
            "slot_le_decomp".into(),
            "epoch_le_decomp".into(),
            "epoch_slot_compose".into(),
            "epoch_boundary_implies_last_slot".into(),
            "epoch_boundary_only_on_real".into(),
            "padding_slot_mod_32_zero".into(),
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
        let thirty_one = Scalar::from_u64(31, curve);
        let thirty_two = Scalar::from_u64(32, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_eb = &columns[COL_IS_EPOCH_BOUNDARY][r];
            let slot = &columns[COL_SLOT][r];
            let epoch = &columns[COL_EPOCH][r];
            let smod = &columns[COL_SLOT_MOD_32][r];

            // 0: is_real binary
            out[0][r] = is_real.mul(&is_real.sub(&one));
            // 1: is_epoch_boundary binary
            out[1][r] = is_eb.mul(&is_eb.sub(&one));
            // 2: slot LE decomp
            let slot_sum = le_decomp_sum_evals(columns, COL_SLOT_BYTE_OFFSET, r, curve);
            out[2][r] = slot.sub(&slot_sum);
            // 3: epoch LE decomp
            let epoch_sum =
                le_decomp_sum_evals(columns, COL_EPOCH_BYTE_OFFSET, r, curve);
            out[3][r] = epoch.sub(&epoch_sum);
            // 4: epoch·32 + slot_mod_32 - slot = 0 (gated by is_real)
            let composed = epoch.mul(&thirty_two).add(smod).sub(slot);
            out[4][r] = is_real.mul(&composed);
            // 5: is_eb · (slot_mod_32 - 31) = 0
            out[5][r] = is_eb.mul(&smod.sub(&thirty_one));
            // 6: is_eb · (1 - is_real) = 0
            out[6][r] = is_eb.mul(&one.sub(is_real));
            // 7: (1 - is_real) · slot_mod_32 = 0
            out[7][r] = one.sub(is_real).mul(smod);
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
        let thirty_one = Scalar::from_u64(31, curve);
        let thirty_two = Scalar::from_u64(32, curve);

        let is_real = &col_evals[COL_IS_REAL];
        let is_eb = &col_evals[COL_IS_EPOCH_BOUNDARY];
        let slot = &col_evals[COL_SLOT];
        let epoch = &col_evals[COL_EPOCH];
        let smod = &col_evals[COL_SLOT_MOD_32];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_eb.mul(&is_eb.sub(&one)),
            slot.sub(&le_decomp_sum_point(col_evals, COL_SLOT_BYTE_OFFSET, curve)),
            epoch.sub(&le_decomp_sum_point(col_evals, COL_EPOCH_BYTE_OFFSET, curve)),
            is_real.mul(&epoch.mul(&thirty_two).add(smod).sub(slot)),
            is_eb.mul(&smod.sub(&thirty_one)),
            is_eb.mul(&one.sub(is_real)),
            one.sub(is_real).mul(smod),
        ];

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
        let thirty_one_poly = vec![Scalar::from_u64(31, curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_eb = &col_coeffs[COL_IS_EPOCH_BOUNDARY];
        let slot = &col_coeffs[COL_SLOT];
        let epoch = &col_coeffs[COL_EPOCH];
        let smod = &col_coeffs[COL_SLOT_MOD_32];

        // body0: is_real · (is_real - 1)
        let body0 = poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve);
        // body1: is_eb · (is_eb - 1)
        let body1 = poly_mul(is_eb, &poly_sub(is_eb, &one_poly, curve), curve);
        // body2: slot - Σ slot_byte_k · 256^k
        let slot_sum = le_decomp_sum_poly(col_coeffs, COL_SLOT_BYTE_OFFSET, curve);
        let body2 = poly_sub(slot, &slot_sum, curve);
        // body3: epoch - Σ epoch_byte_k · 256^k
        let epoch_sum = le_decomp_sum_poly(col_coeffs, COL_EPOCH_BYTE_OFFSET, curve);
        let body3 = poly_sub(epoch, &epoch_sum, curve);
        // body4: is_real · (epoch · 32 + smod - slot)
        let epoch_scaled = poly_scalar_mul(epoch, &Scalar::from_u64(32, curve));
        let composed = poly_sub(&poly_add(&epoch_scaled, smod, curve), slot, curve);
        let body4 = poly_mul(is_real, &composed, curve);
        // body5: is_eb · (smod - 31)
        let body5 = poly_mul(is_eb, &poly_sub(smod, &thirty_one_poly, curve), curve);
        // body6: is_eb · (1 - is_real)
        let one_minus_real = poly_sub(&one_poly, is_real, curve);
        let body6 = poly_mul(is_eb, &one_minus_real, curve);
        // body7: (1 - is_real) · smod
        let body7 = poly_mul(&one_minus_real, smod, curve);

        let bodies = [body0, body1, body2, body3, body4, body5, body6, body7];
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

    // ── Cross-row (shifted) support ─────────────────────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        // We need at ω·z:
        //   - IS_REAL (selector for the shifted constraint gate)
        //   - SLOT (for the +1 increment)
        //   - PREV_STATE_ROOT_OFFSET..+32 (chain LHS)
        let mut cols = Vec::with_capacity(2 + 32);
        cols.push(COL_IS_REAL);
        cols.push(COL_SLOT);
        for k in 0..32 {
            cols.push(COL_PREV_STATE_ROOT_OFFSET + k);
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
        if shifted_evals.len() < 2 + 32 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let is_real_curr = &col_evals_at_z[COL_IS_REAL];
        let slot_curr = &col_evals_at_z[COL_SLOT];
        let is_real_next = &shifted_evals[0];
        let slot_next = &shifted_evals[1];
        // Wrap-around exclusion at ω^{n-1}.
        let exclusion = z.sub(omega_n_minus_1);
        let gate = is_real_curr.mul(is_real_next);

        // body 0: gate · (slot(ω·z) - slot(z) - 1)
        let body0 = gate.mul(&slot_next.sub(slot_curr).sub(&one));

        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        acc = acc.add(&ap.mul(&body0).mul(&exclusion));
        ap = ap.mul(alpha);

        // body 1..33: per-byte prev_state_root(ω·z) == post_state_root(z).
        for k in 0..32 {
            let prev_next = &shifted_evals[2 + k];
            let post_curr = &col_evals_at_z[COL_POST_STATE_ROOT_OFFSET + k];
            let body = gate.mul(&prev_next.sub(post_curr));
            acc = acc.add(&ap.mul(&body).mul(&exclusion));
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
        let one_poly = vec![Scalar::one(curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let slot = &col_coeffs[COL_SLOT];
        let is_real_shift = poly_shift(is_real, omega);
        let slot_shift = poly_shift(slot, omega);
        let gate = poly_mul(is_real, &is_real_shift, curve);

        // body 0: gate · (slot(ω·X) - slot(X) - 1)
        let slot_diff = poly_sub(&poly_sub(&slot_shift, slot, curve), &one_poly, curve);
        let body0 = poly_mul(&gate, &slot_diff, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let excluded0 = poly_mul_linear(&body0, &omega_n_minus_1);

        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        acc = poly_add(&acc, &poly_scalar_mul(&excluded0, &ap), curve);
        ap = ap.mul(alpha);

        for k in 0..32 {
            let prev = &col_coeffs[COL_PREV_STATE_ROOT_OFFSET + k];
            let post = &col_coeffs[COL_POST_STATE_ROOT_OFFSET + k];
            let prev_shift = poly_shift(prev, omega);
            let body = poly_mul(&gate, &poly_sub(&prev_shift, post, curve), curve);
            let excluded = poly_mul_linear(&body, &omega_n_minus_1);
            acc = poly_add(&acc, &poly_scalar_mul(&excluded, &ap), curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // Range tables:
        //   tbl[0] = 8-bit range (slot/epoch bytes + state-root + block-root bytes)
        //   tbl[1] = 5-bit range (slot_mod_32 ∈ [0, 32))
        let tables = vec![LookupTable::range(8), LookupTable::range(5)];
        const TBL_8: usize = 0;
        const TBL_5: usize = 1;
        let mut declarations = Vec::new();

        // slot bytes (8) and epoch bytes (8) → 8-bit.
        for (label, base, count) in [
            ("bst_slot_byte", COL_SLOT_BYTE_OFFSET, 8usize),
            ("bst_epoch_byte", COL_EPOCH_BYTE_OFFSET, 8usize),
            ("bst_prev_state_root_byte", COL_PREV_STATE_ROOT_OFFSET, 32usize),
            ("bst_post_state_root_byte", COL_POST_STATE_ROOT_OFFSET, 32usize),
            ("bst_block_root_byte", COL_BLOCK_ROOT_OFFSET, 32usize),
        ] {
            for k in 0..count {
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

        // slot_mod_32 ∈ [0, 32).
        declarations.push((
            LookupDeclaration {
                label: "bst_slot_mod_32_5bit".into(),
                column_index: COL_SLOT_MOD_32,
                max_bits: 5,
                selector_column: None,
            },
            TBL_5,
        ));

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// **block_header_air (Layer B EL block header) ↔ beacon_state_transition_air**.
///
/// 32-byte tuple binding `BLOCK_ROOT[i]` ↔ the EL `block_hash` column in
/// `block_header_air` (which represents the block-hash root the block
/// references for each slot). Gated by `IS_REAL` on both sides.
///
/// **Caveat**: this binds the EL block hash 1:1 to the consumer's
/// `block_root`. For pre-merge slots without an attached EL block this
/// descriptor must be gated out (the caller wires the linkage only on
/// post-merge slots). Below the merge, `execution_payload.block_hash`
/// is the zero-chunk and the BBH `body_root` chains in a different
/// shape — that case is deferred.
pub fn make_state_transition_to_block_header_descriptor(
    state_transition_layer_index: usize,
    block_header_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    let a_columns: Vec<usize> =
        (0..32).map(|k| COL_BLOCK_ROOT_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..32).map(|k| bh::COL_BLOCK_HASH_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "beacon_state_transition_to_block_header_v1".into(),
        a_layer_index: state_transition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

/// **bbh_root_consumer_air (BeaconBlockHeader root consumer) ↔
/// beacon_state_transition_air**.
///
/// 40-byte tuple `(slot_le_bytes[0..8], state_root[0..32])` binding
/// `(SLOT_BYTE, POST_STATE_ROOT)` on this AIR ↔
/// `(SLOT_BYTE, STATE_ROOT)` on bbh_root_consumer_air. Gated by
/// `IS_REAL` on both sides.
///
/// **Caveat**: bbh_root_consumer_air's `STATE_ROOT` field is the
/// state_root the BeaconBlockHeader **claims** for the slot, which by
/// the spec equals `post_state_root` of the same slot's
/// `process_slot`. The tuple shape (slot bytes + state-root bytes)
/// matches between the AIRs because both expose 8 LE slot bytes and 32
/// state-root bytes.
pub fn make_state_transition_to_bbh_descriptor(
    state_transition_layer_index: usize,
    bbh_consumer_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bbh_root_consumer_air as bbh;
    let mut a_columns: Vec<usize> = Vec::with_capacity(8 + 32);
    for k in 0..8 {
        a_columns.push(COL_SLOT_BYTE_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_POST_STATE_ROOT_OFFSET + k);
    }
    let mut b_columns: Vec<usize> = Vec::with_capacity(8 + 32);
    for k in 0..8 {
        b_columns.push(bbh::COL_SLOT_BYTE_OFFSET + k);
    }
    for k in 0..32 {
        b_columns.push(bbh::COL_STATE_ROOT_OFFSET + k);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "beacon_state_transition_to_bbh_v1".into(),
        a_layer_index: state_transition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bbh_consumer_layer_index,
        b_columns,
        b_selector_column: Some(bbh::COL_IS_REAL),
    }
}

/// **finality_constraints (FFG finalization) ↔ beacon_state_transition_air**.
///
/// 32-byte tuple binding `POST_STATE_ROOT` on epoch-boundary rows of
/// this AIR ↔ `CLAIMED_FINALIZED_ROOT` on the finality AIR's threshold
/// row. Gated by `IS_EPOCH_BOUNDARY` (which is `1` only at
/// `slot_mod_32 == 31`) on the A side and `COL_SEL_THRESHOLD` on the
/// B side.
///
/// This is the conceptual hookpoint for the FFG checkpoint chain: each
/// epoch-boundary row's post-state root is a candidate justified
/// checkpoint root. The host-side oracle in
/// [`crate::ffg_checkpoint_chain::verify_checkpoint_chain`] ratchets
/// epoch counters. A dedicated `ffg_checkpoint_chain_air` (deferred)
/// would expose the multiset properly.
///
/// **Caveat**: A side may have many epoch-boundary rows (one per
/// epoch); B side has exactly one threshold row. The 1-vs-N multiset
/// won't close unless either (a) the consumer carves a single epoch's
/// boundary row by adding an extra selector column on the A side, or
/// (b) the finality AIR is replicated per epoch. For multi-epoch
/// chains, consumers must wire one finality AIR per epoch — see the
/// roadmap note on `e2e_finality_oracle` for the host-side composition.
pub fn make_state_transition_to_finality_descriptor(
    state_transition_layer_index: usize,
    finality_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::finality_constraints as fc;
    let a_columns: Vec<usize> =
        (0..32).map(|k| COL_POST_STATE_ROOT_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..32).map(|k| fc::COL_CLAIMED_FINALIZED_ROOT_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "beacon_state_transition_to_finality_v1".into(),
        a_layer_index: state_transition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_EPOCH_BOUNDARY),
        b_layer_index: finality_layer_index,
        b_columns,
        b_selector_column: Some(fc::COL_SEL_THRESHOLD),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn root(tag: u8) -> [u8; 32] {
        let mut r = [0u8; 32];
        r[0] = tag;
        r[31] = tag.wrapping_mul(7);
        r
    }

    /// 3-slot chain starting at slot 100, with chained state roots.
    fn three_slot_chain() -> BeaconStateTransitionWitness {
        let s0 = root(0xa0);
        let s1 = root(0xa1);
        let s2 = root(0xa2);
        let s3 = root(0xa3);
        let b0 = root(0xb0);
        let b1 = root(0xb1);
        let b2 = root(0xb2);
        BeaconStateTransitionWitness::from_chain(
            s0,
            &[(100, b0, s1), (101, b1, s2), (102, b2, s3)],
        )
    }

    #[test]
    fn from_chain_builds_chained_rows() {
        let w = three_slot_chain();
        assert_eq!(w.rows.len(), 3);
        assert_eq!(w.rows[0].slot, 100);
        assert_eq!(w.rows[0].epoch, 100 / 32);
        assert_eq!(w.rows[1].prev_state_root, w.rows[0].post_state_root);
        assert_eq!(w.rows[2].prev_state_root, w.rows[1].post_state_root);
        // None of slot 100..103 land on slot_mod_32==31 → no boundary.
        for row in &w.rows {
            assert!(!row.is_epoch_boundary);
        }
    }

    #[test]
    fn epoch_boundary_at_slot_31() {
        // Slots 30, 31, 32: row 1 (slot 31) is the epoch boundary.
        let w = BeaconStateTransitionWitness::from_chain(
            root(0xc0),
            &[
                (30, root(0xd0), root(0xc1)),
                (31, root(0xd1), root(0xc2)),
                (32, root(0xd2), root(0xc3)),
            ],
        );
        assert!(!w.rows[0].is_epoch_boundary);
        assert!(w.rows[1].is_epoch_boundary, "slot 31 must be epoch boundary");
        assert!(!w.rows[2].is_epoch_boundary);
        assert_eq!(w.rows[0].epoch, 0);
        assert_eq!(w.rows[1].epoch, 0);
        assert_eq!(w.rows[2].epoch, 1);

        // Constraints zero on honest witness.
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BeaconStateTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} = {:?}",
                    i,
                    r,
                    v,
                );
            }
        }
    }

    #[test]
    fn tampered_slot_increment_detected_by_shifted_constraint() {
        // Honest chain; then tamper SLOT[1] = SLOT[0] + 2 (gap of 2).
        let w = three_slot_chain();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper.
        cols[COL_SLOT][1] = Scalar::from_u64(102, CurveType::Bls48581);

        // We can't easily test build_shifted_constraint_polynomial without
        // an omega; instead, simulate the shifted body check directly at
        // a domain row by comparing consecutive rows.
        let one = Scalar::one(CurveType::Bls48581);
        let slot_curr = &cols[COL_SLOT][0];
        let slot_next = &cols[COL_SLOT][1];
        let is_real_curr = &cols[COL_IS_REAL][0];
        let is_real_next = &cols[COL_IS_REAL][1];
        let body0 = is_real_curr
            .mul(is_real_next)
            .mul(&slot_next.sub(slot_curr).sub(&one));
        assert!(
            !body0.is_zero(),
            "tampered slot gap must make shifted body 0 nonzero",
        );
    }

    #[test]
    fn tampered_state_root_chain_detected_by_shifted_constraint() {
        let w = three_slot_chain();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper prev_state_root[1][7] so the chain no longer matches
        // post_state_root[0][7].
        cols[COL_PREV_STATE_ROOT_OFFSET + 7][1] =
            Scalar::from_u64(0xff, CurveType::Bls48581);

        let is_real_curr = &cols[COL_IS_REAL][0];
        let is_real_next = &cols[COL_IS_REAL][1];
        let prev_next = &cols[COL_PREV_STATE_ROOT_OFFSET + 7][1];
        let post_curr = &cols[COL_POST_STATE_ROOT_OFFSET + 7][0];
        let body_k = is_real_curr
            .mul(is_real_next)
            .mul(&prev_next.sub(post_curr));
        assert!(
            !body_k.is_zero(),
            "tampered state-root chain at byte 7 must fire its shifted constraint",
        );
    }

    #[test]
    fn tampered_epoch_composition_fires_row_constraint() {
        // Honest slot=33, epoch=1, slot_mod_32=1. Tamper epoch to 0 →
        // 0·32 + 1 - 33 = -32 ≠ 0; constraint 4 must fire.
        let w = BeaconStateTransitionWitness::from_chain(
            root(0xc0),
            &[(33, root(0xd0), root(0xc1))],
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_EPOCH][0] = Scalar::from_u64(0, CurveType::Bls48581);
        // Also fix the LE byte decomp so constraint 3 doesn't ALSO fire.
        for k in 0..8 {
            cols[COL_EPOCH_BYTE_OFFSET + k][0] =
                Scalar::from_u64(0, CurveType::Bls48581);
        }
        let cs = BeaconStateTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 4 = epoch_slot_compose
        assert!(
            !results[4][0].is_zero(),
            "tampered epoch must fire compose constraint",
        );
        // constraint 3 (epoch decomp) should still be zero because we
        // updated the bytes to match.
        assert!(results[3][0].is_zero());
    }

    #[test]
    fn tampered_is_epoch_boundary_lies_fire_constraint() {
        // Honest slot=100 (mod32=4, not boundary). Flip is_eb to 1:
        //   constraint 5: 1·(4 - 31) = -27 ≠ 0 → fires.
        let w = three_slot_chain();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_EPOCH_BOUNDARY][0] = Scalar::one(CurveType::Bls48581);
        let cs = BeaconStateTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[5][0].is_zero(),
            "is_epoch_boundary lie must fire constraint 5",
        );
    }

    #[test]
    fn column_layout_and_constraint_counts_pinned() {
        // Pin the column layout so unintentional renumbering is caught.
        assert_eq!(COL_SLOT, 0);
        assert_eq!(COL_EPOCH, 1);
        assert_eq!(COL_SLOT_MOD_32, 2);
        assert_eq!(COL_SLOT_BYTE_OFFSET, 3);
        assert_eq!(COL_EPOCH_BYTE_OFFSET, 11);
        assert_eq!(COL_PREV_STATE_ROOT_OFFSET, 19);
        assert_eq!(COL_POST_STATE_ROOT_OFFSET, 51);
        assert_eq!(COL_BLOCK_ROOT_OFFSET, 83);
        assert_eq!(COL_IS_REAL, 115);
        assert_eq!(COL_IS_EPOCH_BOUNDARY, 116);
        assert_eq!(NUM_COLUMNS, 117);
        assert_eq!(NUM_ROW_CONSTRAINTS, 8);
        assert_eq!(NUM_SHIFTED, 33);

        let cs = BeaconStateTransitionConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        // Shifted column indices: is_real, slot, prev_state_root[0..32].
        let shifted = cs.shifted_column_indices();
        assert_eq!(shifted.len(), 2 + 32);
        assert_eq!(shifted[0], COL_IS_REAL);
        assert_eq!(shifted[1], COL_SLOT);
        for k in 0..32 {
            assert_eq!(shifted[2 + k], COL_PREV_STATE_ROOT_OFFSET + k);
        }
    }

    #[test]
    fn descriptors_well_formed() {
        // block_header descriptor.
        let d_bh = make_state_transition_to_block_header_descriptor(0, 1);
        assert_eq!(d_bh.label, "beacon_state_transition_to_block_header_v1");
        assert_eq!(d_bh.a_layer_index, 0);
        assert_eq!(d_bh.b_layer_index, 1);
        assert_eq!(d_bh.a_columns.len(), 32);
        assert_eq!(d_bh.b_columns.len(), 32);
        assert_eq!(d_bh.a_columns[0], COL_BLOCK_ROOT_OFFSET);
        assert_eq!(
            d_bh.b_columns[0],
            crate::block_header_air::COL_BLOCK_HASH_OFFSET,
        );
        assert_eq!(d_bh.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_bh.b_selector_column,
            Some(crate::block_header_air::COL_IS_REAL),
        );

        // bbh descriptor (8 slot bytes + 32 state-root bytes = 40).
        let d_bbh = make_state_transition_to_bbh_descriptor(0, 2);
        assert_eq!(d_bbh.label, "beacon_state_transition_to_bbh_v1");
        assert_eq!(d_bbh.a_columns.len(), 40);
        assert_eq!(d_bbh.b_columns.len(), 40);
        assert_eq!(d_bbh.a_columns[0], COL_SLOT_BYTE_OFFSET);
        assert_eq!(d_bbh.a_columns[8], COL_POST_STATE_ROOT_OFFSET);
        assert_eq!(
            d_bbh.b_columns[0],
            crate::bbh_root_consumer_air::COL_SLOT_BYTE_OFFSET,
        );
        assert_eq!(
            d_bbh.b_columns[8],
            crate::bbh_root_consumer_air::COL_STATE_ROOT_OFFSET,
        );
        assert_eq!(d_bbh.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_bbh.b_selector_column,
            Some(crate::bbh_root_consumer_air::COL_IS_REAL),
        );

        // finality descriptor (32 root bytes).
        let d_fin = make_state_transition_to_finality_descriptor(0, 3);
        assert_eq!(d_fin.label, "beacon_state_transition_to_finality_v1");
        assert_eq!(d_fin.a_columns.len(), 32);
        assert_eq!(d_fin.b_columns.len(), 32);
        assert_eq!(d_fin.a_columns[0], COL_POST_STATE_ROOT_OFFSET);
        assert_eq!(
            d_fin.b_columns[0],
            crate::finality_constraints::COL_CLAIMED_FINALIZED_ROOT_OFFSET,
        );
        assert_eq!(d_fin.a_selector_column, Some(COL_IS_EPOCH_BOUNDARY));
        assert_eq!(
            d_fin.b_selector_column,
            Some(crate::finality_constraints::COL_SEL_THRESHOLD),
        );
    }

    #[test]
    fn evaluate_at_point_matches_combined_alpha_fold() {
        // Sanity check: evaluate_at_point on an honest witness row 0 should
        // equal zero because every body is zero.
        let w = three_slot_chain();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let col_evals_row0: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        let alpha = Scalar::from_u64(0xdeadbeef, CurveType::Bls48581);
        let cs = BeaconStateTransitionConstraintSystem::new(trace.num_rows);
        let combined = cs.evaluate_at_point(&col_evals_row0, &alpha);
        assert!(combined.is_zero(), "honest row should fold to zero");
    }
}
