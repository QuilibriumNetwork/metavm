//! Gas-tracking algebraic AIR (roadmap #60).
//!
//! Two-AIR design:
//!
//! 1. [`GasTrackingConstraintSystem`] — the *per-step* gas-accounting AIR.
//!    Each row commits a tuple `(opcode, gas_pre, gas_post, static_cost,
//!    dynamic_cost, is_real)` and enforces the per-row chain:
//!
//!        gas_pre - gas_post = static_cost + dynamic_cost
//!
//!    plus the `is_real` binary constraint. `dynamic_cost` is a free
//!    witness column (≥ 0 by being a `u64`); the full memory-expansion /
//!    SSTORE / CALL dynamic-cost binding is *still oracle* and waits on a
//!    memory-watermark AIR and per-opcode dynamic-cost gadgets.
//!
//! 2. [`StaticGasTableConstraintSystem`] — a small 256-row lookup table
//!    AIR that pairs every opcode byte with its
//!    [`crate::gas_cost::static_gas_cost`] value (encoded as `0` for
//!    dynamic-cost opcodes; the cross-AIR LogUp lookups are gated by
//!    `is_real` only on rows where the cost is statically known, so the
//!    `0` placeholder is safe for unused opcodes).
//!
//! Cross-AIR LogUp:
//!
//! - [`make_evm_to_gas_tracking_descriptor`] — binds an EVM main row's
//!   `(opcode, gas_remaining_now, gas_remaining_next)` triple to the
//!   gas-tracking AIR's `(opcode, gas_pre, gas_post)` columns, gated by an
//!   `is_real` selector on the gas-tracking side. Note: `gas_remaining_next`
//!   on the EVM side is taken from the *current* row's `COL_GAS_REMAINING`
//!   column — for now we use a `(opcode, gas_pre)` 2-column descriptor and
//!   leave the `next` binding to a host-side oracle until a proper shifted
//!   column or cross-row LogUp is wired (see roadmap #60 step 2).
//!
//! - [`make_gas_tracking_to_static_table_descriptor`] — binds gas-tracking
//!   `(opcode, static_cost)` rows to the lookup table. Gated by an
//!   "is_static" selector column on the gas-tracking side that is `1` iff
//!   the opcode has a static cost (i.e. iff `static_gas_cost(opcode)` is
//!   `Some`).

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

use crate::gas_cost::static_gas_cost;

// ─── Gas-tracking AIR columns ────────────────────────────────────────

pub const COL_OPCODE: usize = 0;
pub const COL_GAS_PRE: usize = 1;
pub const COL_GAS_POST: usize = 2;
pub const COL_STATIC_COST: usize = 3;
pub const COL_DYNAMIC_COST: usize = 4;
pub const COL_IS_REAL: usize = 5;
/// `1` iff this row's opcode has a static gas cost (i.e. the static-table
/// lookup should fire). Used as the selector for the gas-tracking ↔
/// static-table cross-AIR LogUp. `0` on dynamic-cost opcodes (SHA3, SLOAD,
/// SSTORE, CALL family, etc.) and on padding rows.
pub const COL_IS_STATIC: usize = 6;
pub const NUM_COLUMNS: usize = 7;

/// Per-row constraints:
/// 0. `is_real * (is_real - 1) = 0`
/// 1. `is_static * (is_static - 1) = 0`
/// 2. `is_real * (gas_pre - gas_post - static_cost - dynamic_cost) = 0`
/// 3. `is_static * (1 - is_real) = 0`  — `is_static` implies `is_real`.
pub const NUM_ROW_CONSTRAINTS: usize = 4;
/// Cross-row (shifted) constraints:
/// 0. `is_real(X) * is_real(ω·X) * (gas_pre(ω·X) - gas_post(X)) = 0`
///    — algebraic chain binding: the *next* row's `gas_pre` equals this
///    row's `gas_post`. Combined with row constraint 2 this gives the
///    canonical `gas_remaining[i+1] = gas_remaining[i] - opcode_gas[i]`
///    chain across all is_real rows.
pub const NUM_SHIFTED: usize = 1;

#[derive(Clone, Debug)]
pub struct GasTrackingRow {
    pub opcode: u8,
    pub gas_pre: u64,
    pub gas_post: u64,
    pub static_cost: u64,
    pub dynamic_cost: u64,
}

#[derive(Clone, Debug, Default)]
pub struct GasTrackingWitness {
    pub rows: Vec<GasTrackingRow>,
}

impl GasTrackingWitness {
    pub fn from_rows(rows: Vec<GasTrackingRow>) -> Self { Self { rows } }
}

pub fn build_trace_polynomials(w: &GasTrackingWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_OPCODE][r] = Scalar::from_u64(row.opcode as u64, curve);
        cols[COL_GAS_PRE][r] = Scalar::from_u64(row.gas_pre, curve);
        cols[COL_GAS_POST][r] = Scalar::from_u64(row.gas_post, curve);
        cols[COL_STATIC_COST][r] = Scalar::from_u64(row.static_cost, curve);
        cols[COL_DYNAMIC_COST][r] = Scalar::from_u64(row.dynamic_cost, curve);
        cols[COL_IS_REAL][r] = one.clone();
        if static_gas_cost(row.opcode).is_some() {
            cols[COL_IS_STATIC][r] = one.clone();
        }
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

pub struct GasTrackingConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl GasTrackingConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for GasTrackingConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_static_binary".into(),
            "gas_chain".into(),
            "is_static_implies_is_real".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut c0 = vec![Scalar::zero(curve); n];
        let mut c1 = vec![Scalar::zero(curve); n];
        let mut c2 = vec![Scalar::zero(curve); n];
        let mut c3 = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let real = &columns[COL_IS_REAL][r];
            let stat = &columns[COL_IS_STATIC][r];
            c0[r] = real.mul(&real.sub(&one));
            c1[r] = stat.mul(&stat.sub(&one));
            let pre = &columns[COL_GAS_PRE][r];
            let post = &columns[COL_GAS_POST][r];
            let sc = &columns[COL_STATIC_COST][r];
            let dc = &columns[COL_DYNAMIC_COST][r];
            let total_cost = sc.add(dc);
            let delta = pre.sub(post).sub(&total_cost);
            c2[r] = real.mul(&delta);
            c3[r] = stat.mul(&one.sub(real));
        }
        vec![c0, c1, c2, c3]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let real = &ce[COL_IS_REAL];
        let stat = &ce[COL_IS_STATIC];
        let c0 = real.mul(&real.sub(&one));
        let c1 = stat.mul(&stat.sub(&one));
        let total_cost = ce[COL_STATIC_COST].add(&ce[COL_DYNAMIC_COST]);
        let delta = ce[COL_GAS_PRE].sub(&ce[COL_GAS_POST]).sub(&total_cost);
        let c2 = real.mul(&delta);
        let c3 = stat.mul(&one.sub(real));
        let bodies = [c0, c1, c2, c3];
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let real = &cc[COL_IS_REAL];
        let stat = &cc[COL_IS_STATIC];
        let real_m1 = poly_sub(real, &one_p, curve);
        let stat_m1 = poly_sub(stat, &one_p, curve);
        let c0 = poly_mul(real, &real_m1, curve);
        let c1 = poly_mul(stat, &stat_m1, curve);
        let total_cost = poly_add(&cc[COL_STATIC_COST], &cc[COL_DYNAMIC_COST], curve);
        let pre_minus_post = poly_sub(&cc[COL_GAS_PRE], &cc[COL_GAS_POST], curve);
        let delta = poly_sub(&pre_minus_post, &total_cost, curve);
        let c2 = poly_mul(real, &delta, curve);
        let one_minus_real = poly_sub(&one_p, real, curve);
        let c3 = poly_mul(stat, &one_minus_real, curve);
        let bodies = [c0, c1, c2, c3];
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL, COL_IS_STATIC] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS { return; }
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }

    // ── Cross-row chain: gas_pre(ω·X) = gas_post(X) under is_real·is_real(ω·X)
    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Order matters; consumers index by position into `shifted_evals`.
        vec![COL_IS_REAL, COL_GAS_PRE]
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
        if shifted_evals.len() != 2 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let real_curr = &col_evals_at_z[COL_IS_REAL];
        let real_next = &shifted_evals[0];
        let pre_next = &shifted_evals[1];
        let post_curr = &col_evals_at_z[COL_GAS_POST];
        // body = is_real(X) * is_real(ω·X) * (gas_pre(ω·X) - gas_post(X))
        let gating = real_curr.mul(real_next);
        let chain = pre_next.sub(post_curr);
        let body = gating.mul(&chain);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        // Multiply by (z - ω^{n-1}) to exclude the last-row wrap-around.
        ap.mul(&body).mul(&z.sub(omega_n_minus_1))
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
        let real = &column_coeffs[COL_IS_REAL];
        let real_next = poly_shift(real, omega);
        let gating = poly_mul(real, &real_next, curve);
        let pre = &column_coeffs[COL_GAS_PRE];
        let pre_next = poly_shift(pre, omega);
        let post = &column_coeffs[COL_GAS_POST];
        let chain = poly_sub(&pre_next, post, curve);
        let body = poly_mul(&gating, &chain, curve);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let total = poly_scalar_mul(&body, &ap);
        // Multiply by (X - ω^{n-1}) so the constraint vanishes on the wrap-around row.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }
}

// ─── Static-gas lookup table AIR ────────────────────────────────────

pub mod table {
    use super::*;

    pub const COL_OPCODE: usize = 0;
    pub const COL_STATIC_COST: usize = 1;
    pub const COL_IS_REAL: usize = 2;
    pub const NUM_COLUMNS: usize = 3;
    pub const NUM_ROW_CONSTRAINTS: usize = 1;

    /// Total number of opcode rows in the static-gas lookup table.
    /// One row per opcode byte (0..=255).
    pub const TABLE_SIZE: usize = 256;

    #[derive(Clone, Debug)]
    pub struct StaticGasTableWitness;

    impl Default for StaticGasTableWitness {
        fn default() -> Self { Self }
    }

    /// Build the canonical 256-row static-gas table polynomials.
    /// For opcodes with no static cost, both `static_cost` and
    /// `is_real` are `0` — these rows are *not* selected by the
    /// cross-AIR LogUp (the gas-tracking side gates by `is_static`).
    pub fn build_trace_polynomials(curve: CurveType) -> TracePolynomials {
        let padded = metavm_zkp::trace::nearest_power_of_two(TABLE_SIZE);
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
            .map(|_| vec![zero.clone(); padded])
            .collect();
        for op in 0u16..=255 {
            let opcode = op as u8;
            let r = op as usize;
            cols[COL_OPCODE][r] = Scalar::from_u64(opcode as u64, curve);
            if let Some(cost) = static_gas_cost(opcode) {
                cols[COL_STATIC_COST][r] = Scalar::from_u64(cost, curve);
                cols[COL_IS_REAL][r] = one.clone();
            }
        }
        let polys: Vec<Polynomial> = cols
            .into_iter()
            .map(|e| Polynomial { evaluations: e, degree: TABLE_SIZE })
            .collect();
        TracePolynomials {
            columns: polys,
            num_rows: TABLE_SIZE,
            padded_size: padded as u64,
            curve,
        }
    }

    pub struct StaticGasTableConstraintSystem {
        pub num_rows: usize,
    }

    impl StaticGasTableConstraintSystem {
        pub fn new(num_rows: usize) -> Self { Self { num_rows } }
    }

    impl VmConstraintSystem for StaticGasTableConstraintSystem {
        fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
        fn constraint_labels(&self) -> Vec<String> { vec!["is_real_binary".into()] }

        fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
            let curve = columns[0][0].curve_type();
            let one = Scalar::one(curve);
            let n = columns[0].len();
            let mut bin = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                bin[r] = v.mul(&v.sub(&one));
            }
            vec![bin]
        }
        fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
            if ce.len() < NUM_COLUMNS {
                return Scalar::zero(alpha.curve_type());
            }
            let one = Scalar::one(alpha.curve_type());
            let v = &ce[COL_IS_REAL];
            v.mul(&v.sub(&one))
        }
        fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
            let curve = alpha.curve_type();
            let one_p = vec![Scalar::one(curve)];
            let v = &cc[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_p, curve);
            poly_mul(v, &v_m1, curve)
        }
        fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
        fn padding_selector_column(&self) -> Option<usize> { None }
        fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
            if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS { return; }
            let zero = Scalar::zero(columns[0][0].curve_type());
            for c in columns.iter_mut().take(NUM_COLUMNS) {
                for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                    *cell = zero.clone();
                }
            }
        }
        fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
    }
}

// ─── Witness extraction from EVM trace ────────────────────────────────

/// Build a `GasTrackingWitness` from an EVM trace. For each step we
/// commit the current opcode and the gas-remaining delta from this row
/// to the next, splitting it into `static_cost` (looked up via
/// [`static_gas_cost`], or `0` for dynamic opcodes) and the residual
/// `dynamic_cost`. The final row's `gas_post` is taken as the same
/// row's `gas_remaining` (zero delta — terminator) and is skipped.
pub fn from_evm_trace(cols: &crate::trace::EvmTraceColumns) -> GasTrackingWitness {
    let n = cols.step.len();
    let mut rows = Vec::with_capacity(n);
    for r in 0..n.saturating_sub(1) {
        let opcode = cols.opcode[r] as u8;
        let gas_pre = cols.gas_remaining[r];
        let gas_post = cols.gas_remaining[r + 1];
        let static_cost = static_gas_cost(opcode).unwrap_or(0);
        let actual_delta = gas_pre.saturating_sub(gas_post);
        let dynamic_cost = actual_delta.saturating_sub(static_cost);
        rows.push(GasTrackingRow {
            opcode,
            gas_pre,
            gas_post,
            static_cost,
            dynamic_cost,
        });
    }
    GasTrackingWitness { rows }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// EVM main row `(opcode, gas_remaining)` → gas-tracking `(opcode, gas_pre)`.
/// Two-column tuple gated by `is_real` on the gas-tracking side. (The
/// `gas_post` ↔ `gas_remaining_next` shifted binding is documented as
/// host-side oracle; a shifted-column variant is roadmap follow-up.)
pub fn make_evm_to_gas_tracking_descriptor(
    evm_layer_index: usize,
    gas_tracking_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_GAS_REMAINING as EVM_COL_GAS_REMAINING, COL_OPCODE as EVM_COL_OPCODE};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_main_to_gas_tracking_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![EVM_COL_OPCODE, EVM_COL_GAS_REMAINING],
        // EVM main has no `is_real` per row; selector left unset so every
        // row contributes. The gas-tracking side's `is_real` gating + row
        // count alignment closes the multiset equality.
        a_selector_column: None,
        b_layer_index: gas_tracking_layer_index,
        b_columns: vec![COL_OPCODE, COL_GAS_PRE],
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Gas-tracking `(opcode, static_cost)` rows where `is_static = 1` →
/// opcode-dispatch AIR `(opcode, base_gas)` rows where `is_real = 1`.
///
/// Binds the per-step `static_cost` witness to the canonical per-opcode
/// `base_gas` exposed by [`crate::opcode_dispatch_air`]. The dispatch AIR's
/// own `opcode_dispatch_to_table` LogUp anchors `base_gas` against the
/// 256-row table, so this descriptor composes into a full
/// `gas_tracking → opcode_dispatch → opcode_table` chain that grounds
/// `static_cost` in the canonical cost table without duplicating the
/// table lookup.
pub fn make_gas_tracking_to_opcode_dispatch_descriptor(
    gas_tracking_layer_index: usize,
    opcode_dispatch_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::opcode_dispatch_air::{COL_BASE_GAS as OD_COL_BASE_GAS, COL_IS_REAL as OD_COL_IS_REAL, COL_OPCODE as OD_COL_OPCODE};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "gas_tracking_to_opcode_dispatch_v1".into(),
        a_layer_index: gas_tracking_layer_index,
        a_columns: vec![COL_OPCODE, COL_STATIC_COST],
        a_selector_column: Some(COL_IS_STATIC),
        b_layer_index: opcode_dispatch_layer_index,
        b_columns: vec![OD_COL_OPCODE, OD_COL_BASE_GAS],
        b_selector_column: Some(OD_COL_IS_REAL),
    }
}

/// Gas-tracking `(opcode, static_cost)` rows where `is_static = 1` →
/// static-gas table `(opcode, static_cost)` rows where `is_real = 1`.
pub fn make_gas_tracking_to_static_table_descriptor(
    gas_tracking_layer_index: usize,
    table_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "gas_tracking_to_static_table_v1".into(),
        a_layer_index: gas_tracking_layer_index,
        a_columns: vec![COL_OPCODE, COL_STATIC_COST],
        a_selector_column: Some(COL_IS_STATIC),
        b_layer_index: table_layer_index,
        b_columns: vec![table::COL_OPCODE, table::COL_STATIC_COST],
        b_selector_column: Some(table::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn honest_witness_constraints_zero() {
        // PUSH1 1; PUSH1 2; ADD; STOP — gas decreases by 3+3+3 = 9
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        assert!(w.rows.len() >= 3);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasTrackingConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn simple_add_gas_extraction() {
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        // First three rows: PUSH1 (3), PUSH1 (3), ADD (3) — each static_cost=3, dynamic=0.
        let push1 = &w.rows[0];
        assert_eq!(push1.opcode, 0x60);
        assert_eq!(push1.static_cost, 3);
        assert_eq!(push1.dynamic_cost, 0);
        let add = &w.rows[2];
        assert_eq!(add.opcode, 0x01);
        assert_eq!(add.static_cost, 3);
        assert_eq!(add.dynamic_cost, 0);
    }

    #[test]
    fn tampered_gas_detected() {
        // Hand-craft a row whose gas_pre - gas_post != static + dynamic.
        let rows = vec![GasTrackingRow {
            opcode: 0x01, // ADD
            gas_pre: 100,
            gas_post: 50, // delta = 50, but static=3 + dynamic=0 != 50
            static_cost: 3,
            dynamic_cost: 0,
        }];
        let w = GasTrackingWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasTrackingConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 2 = gas chain — should fire.
        assert!(!bodies[2][0].is_zero(), "tampered gas not caught");
    }

    #[test]
    fn static_cost_lookup_table_well_formed() {
        let t = table::build_trace_polynomials(CurveType::Bls48581);
        assert_eq!(t.num_rows, table::TABLE_SIZE);
        let cs = table::StaticGasTableConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for body in cs.evaluate_on_domain(&cr, t.num_rows).iter() {
            for v in body.iter() {
                assert!(v.is_zero());
            }
        }
        // Spot-check a few opcodes
        let curve = CurveType::Bls48581;
        // ADD (0x01) -> 3, is_real=1
        let r = 0x01_usize;
        assert!(cr[table::COL_STATIC_COST][r].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(cr[table::COL_IS_REAL][r].sub(&Scalar::one(curve)).is_zero());
        // SHA3 (0x20) -> no static cost, is_real=0
        let r = 0x20_usize;
        assert!(cr[table::COL_IS_REAL][r].is_zero());
        // JUMPDEST (0x5B) -> 1, is_real=1
        let r = 0x5B_usize;
        assert!(cr[table::COL_STATIC_COST][r].sub(&Scalar::from_u64(1, curve)).is_zero());
        assert!(cr[table::COL_IS_REAL][r].sub(&Scalar::one(curve)).is_zero());
    }

    #[test]
    fn dynamic_cost_zero_on_simple_ops() {
        // PUSH1 0x42; POP; STOP — no memory expansion, no SSTORE
        let bc = vec![0x60, 0x42, 0x50, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        for (i, row) in w.rows.iter().enumerate() {
            assert_eq!(row.dynamic_cost, 0, "row {} (opcode {:#x}) has nonzero dynamic cost", i, row.opcode);
        }
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_evm_to_gas_tracking_descriptor(0, 1);
        assert_eq!(d1.label, "evm_main_to_gas_tracking_v1");
        assert_eq!(d1.a_columns.len(), 2);
        assert_eq!(d1.b_columns.len(), 2);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert!(d1.a_selector_column.is_none());
        assert_eq!(d1.b_selector_column, Some(COL_IS_REAL));

        let d2 = make_gas_tracking_to_static_table_descriptor(1, 2);
        assert_eq!(d2.label, "gas_tracking_to_static_table_v1");
        assert_eq!(d2.a_columns.len(), 2);
        assert_eq!(d2.b_columns.len(), 2);
        assert_eq!(d2.a_layer_index, 1);
        assert_eq!(d2.b_layer_index, 2);
        assert_eq!(d2.a_selector_column, Some(COL_IS_STATIC));
        assert_eq!(d2.b_selector_column, Some(table::COL_IS_REAL));
    }

    #[test]
    fn is_static_implies_is_real_constraint_fires() {
        // Construct a manual witness where is_static=1 but is_real=0
        // (skip the trace builder and build columns directly).
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        let n = 4;
        let mut cols_v: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); n]).collect();
        cols_v[COL_OPCODE][0] = Scalar::from_u64(0x01, curve); // ADD
        cols_v[COL_GAS_PRE][0] = Scalar::from_u64(10, curve);
        cols_v[COL_GAS_POST][0] = Scalar::from_u64(10, curve);
        cols_v[COL_STATIC_COST][0] = zero.clone();
        cols_v[COL_DYNAMIC_COST][0] = zero.clone();
        cols_v[COL_IS_REAL][0] = zero.clone(); // pretending not real
        cols_v[COL_IS_STATIC][0] = one.clone(); // but claiming static
        let cs = GasTrackingConstraintSystem::new(n);
        let cr: Vec<&Vec<Scalar>> = cols_v.iter().collect();
        let bodies = cs.evaluate_on_domain(&cr, n);
        // Constraint 3 is `is_static * (1 - is_real)` — fires on row 0.
        assert!(!bodies[3][0].is_zero(), "is_static_implies_is_real not caught");
    }

    #[test]
    fn shifted_chain_constraint_honest_zero() {
        // Honest trace: gas_pre[i+1] = gas_post[i] should hold for all real rows.
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        assert!(w.rows.len() >= 2);
        // Verify chain at the witness level.
        for i in 0..w.rows.len() - 1 {
            assert_eq!(w.rows[i + 1].gas_pre, w.rows[i].gas_post,
                "row {} -> {} chain broken", i, i + 1);
        }
        // And via evaluate_shifted_at_point at a random z, the body itself
        // (sans the (z - ω^{n-1}) factor) should be zero on every real row pair.
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasTrackingConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..w.rows.len().saturating_sub(1) {
            let row_evals: Vec<Scalar> = cr.iter().map(|c| c[r].clone()).collect();
            let shifted = vec![cr[COL_IS_REAL][r + 1].clone(), cr[COL_GAS_PRE][r + 1].clone()];
            // Use z=1, omega_n_minus_1=2 so (z - ω^{n-1}) != 0; alpha=1 so ap=1.
            let z = Scalar::from_u64(1, CurveType::Bls48581);
            let onm1 = Scalar::from_u64(2, CurveType::Bls48581);
            let alpha = Scalar::one(CurveType::Bls48581);
            let v = cs.evaluate_shifted_at_point(&row_evals, &shifted, &z, &onm1, &alpha, 0);
            assert!(v.is_zero(), "shifted row {} -> {} nonzero", r, r + 1);
        }
    }

    #[test]
    fn shifted_chain_constraint_tampered_detected() {
        // Two rows where gas_pre[1] != gas_post[0] -- chain broken.
        let rows = vec![
            GasTrackingRow { opcode: 0x60, gas_pre: 100, gas_post: 97, static_cost: 3, dynamic_cost: 0 },
            // Honest chain would have gas_pre = 97; we cheat to 90.
            GasTrackingRow { opcode: 0x01, gas_pre: 90, gas_post: 87, static_cost: 3, dynamic_cost: 0 },
        ];
        let w = GasTrackingWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasTrackingConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let shifted = vec![cr[COL_IS_REAL][1].clone(), cr[COL_GAS_PRE][1].clone()];
        let z = Scalar::from_u64(1, CurveType::Bls48581);
        let onm1 = Scalar::from_u64(2, CurveType::Bls48581);
        let alpha = Scalar::one(CurveType::Bls48581);
        let v = cs.evaluate_shifted_at_point(&row_evals, &shifted, &z, &onm1, &alpha, 0);
        assert!(!v.is_zero(), "tampered gas chain across rows not caught");
    }

    #[test]
    fn gas_tracking_to_opcode_dispatch_descriptor_well_formed() {
        use crate::opcode_dispatch_air::{COL_BASE_GAS as OD_COL_BASE_GAS, COL_IS_REAL as OD_COL_IS_REAL, COL_OPCODE as OD_COL_OPCODE};
        let d = make_gas_tracking_to_opcode_dispatch_descriptor(3, 7);
        assert_eq!(d.label, "gas_tracking_to_opcode_dispatch_v1");
        assert_eq!(d.a_layer_index, 3);
        assert_eq!(d.b_layer_index, 7);
        assert_eq!(d.a_columns, vec![COL_OPCODE, COL_STATIC_COST]);
        assert_eq!(d.b_columns, vec![OD_COL_OPCODE, OD_COL_BASE_GAS]);
        assert_eq!(d.a_selector_column, Some(COL_IS_STATIC));
        assert_eq!(d.b_selector_column, Some(OD_COL_IS_REAL));
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let bc = vec![0x60, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = GasTrackingConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0x9999, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..w.rows.len() {
            let row_evals: Vec<Scalar> = cr.iter().map(|c| c[r].clone()).collect();
            let v = cs.evaluate_at_point(&row_evals, &alpha);
            assert!(v.is_zero(), "row {} nonzero", r);
        }
    }
}
