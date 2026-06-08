//! CALL/CREATE/RETURN frame transition AIR (standalone gadget).
//!
//! Proves algebraically the frame-stack transitions for CALL family
//! and RETURN/REVERT/STOP-pop opcodes. The current EVM main constraint
//! system treats these as oracle skips and only validates them via the
//! frame stack permutation argument; this AIR binds the per-transition
//! relations between (pre, post) frame state.
//!
//! Per witness row represents ONE transition between consecutive EVM
//! rows where a frame-changing opcode fired. The witness commits:
//!   kind, pre_depth, post_depth,
//!   caller_pre[4], callee_pre[4], caller_post[4], callee_post[4],
//!   value_pre[4], value_post[4], gas_pre, gas_post
//!
//! Kinds (one-hot via 9 selectors):
//!   0 = CALL, 1 = CALLCODE, 2 = DELEGATECALL, 3 = STATICCALL,
//!   4 = CREATE, 5 = CREATE2, 6 = RETURN, 7 = REVERT, 8 = STOP-pop.
//!
//! Algebraic constraints (degree-2):
//!   - is_real binary
//!   - each kind selector binary
//!   - one-hot: sum of selectors == is_real
//!   - CALL-family (kinds 0..5): post_depth = pre_depth + 1
//!   - RETURN/REVERT (kinds 6,7): post_depth = pre_depth - 1
//!     (host pre-condition: pre_depth > 0; STOP-pop similarly handled host-side)
//!   - CALL (kind 0): caller_post[k] = callee_pre[k] for k in 0..4
//!   - DELEGATECALL (kind 2): caller_post[k] = caller_pre[k] for k in 0..4
//!   - STATICCALL (kind 3): value_post[k] = 0 for k in 0..4
//!
//! Still oracle (host-only):
//!   - gas allocation (gas_post for callee, gas_pre - gas_post for caller),
//!   - return data propagation,
//!   - frame depth bound (1024 cap),
//!   - STOP-pop depth transition (one-of-{pop,terminate} ambiguity),
//!   - CALLCODE/CREATE/CREATE2 caller/callee semantics beyond depth.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Kind constants ──────────────────────────────────────────────────
pub const KIND_CALL: u64 = 0;
pub const KIND_CALLCODE: u64 = 1;
pub const KIND_DELEGATECALL: u64 = 2;
pub const KIND_STATICCALL: u64 = 3;
pub const KIND_CREATE: u64 = 4;
pub const KIND_CREATE2: u64 = 5;
pub const KIND_RETURN: u64 = 6;
pub const KIND_REVERT: u64 = 7;
pub const KIND_STOP_POP: u64 = 8;

// ─── Column layout ───────────────────────────────────────────────────
pub const COL_KIND: usize = 0;
pub const COL_PRE_DEPTH: usize = 1;
pub const COL_POST_DEPTH: usize = 2;
pub const COL_CALLER_PRE_L0: usize = 3;
pub const COL_CALLER_PRE_L1: usize = 4;
pub const COL_CALLER_PRE_L2: usize = 5;
pub const COL_CALLER_PRE_L3: usize = 6;
pub const COL_CALLEE_PRE_L0: usize = 7;
pub const COL_CALLEE_PRE_L1: usize = 8;
pub const COL_CALLEE_PRE_L2: usize = 9;
pub const COL_CALLEE_PRE_L3: usize = 10;
pub const COL_CALLER_POST_L0: usize = 11;
pub const COL_CALLER_POST_L1: usize = 12;
pub const COL_CALLER_POST_L2: usize = 13;
pub const COL_CALLER_POST_L3: usize = 14;
pub const COL_CALLEE_POST_L0: usize = 15;
pub const COL_CALLEE_POST_L1: usize = 16;
pub const COL_CALLEE_POST_L2: usize = 17;
pub const COL_CALLEE_POST_L3: usize = 18;
pub const COL_VALUE_PRE_L0: usize = 19;
pub const COL_VALUE_PRE_L1: usize = 20;
pub const COL_VALUE_PRE_L2: usize = 21;
pub const COL_VALUE_PRE_L3: usize = 22;
pub const COL_VALUE_POST_L0: usize = 23;
pub const COL_VALUE_POST_L1: usize = 24;
pub const COL_VALUE_POST_L2: usize = 25;
pub const COL_VALUE_POST_L3: usize = 26;
pub const COL_GAS_PRE: usize = 27;
pub const COL_GAS_POST: usize = 28;
pub const COL_IS_REAL: usize = 29;
// One-hot kind selectors (binary)
pub const COL_SEL_CALL: usize = 30;
pub const COL_SEL_CALLCODE: usize = 31;
pub const COL_SEL_DELEGATECALL: usize = 32;
pub const COL_SEL_STATICCALL: usize = 33;
pub const COL_SEL_CREATE: usize = 34;
pub const COL_SEL_CREATE2: usize = 35;
pub const COL_SEL_RETURN: usize = 36;
pub const COL_SEL_REVERT: usize = 37;
pub const COL_SEL_STOP_POP: usize = 38;
pub const NUM_COLUMNS: usize = 39;

// Constraint count:
//   1 is_real binary
//   9 selector binary
//   1 one-hot sum
//   1 CALL-family depth+1
//   1 RETURN/REVERT depth-1
//   4 CALL caller propagation
//   4 DELEGATECALL caller stay
//   4 STATICCALL value zero
// = 25
pub const NUM_ROW_CONSTRAINTS: usize = 25;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness types ───────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct CallFrameRow {
    pub kind: u64,
    pub pre_depth: u64,
    pub post_depth: u64,
    pub caller_pre: [u64; 4],
    pub callee_pre: [u64; 4],
    pub caller_post: [u64; 4],
    pub callee_post: [u64; 4],
    pub value_pre: [u64; 4],
    pub value_post: [u64; 4],
    pub gas_pre: u64,
    pub gas_post: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CallFrameWitness {
    pub rows: Vec<CallFrameRow>,
}

impl CallFrameWitness {
    pub fn from_rows(rows: Vec<CallFrameRow>) -> Self { Self { rows } }
}

// ─── Trace builder ───────────────────────────────────────────────────
pub fn build_trace_polynomials(w: &CallFrameWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_KIND][r] = Scalar::from_u64(row.kind, curve);
        cols[COL_PRE_DEPTH][r] = Scalar::from_u64(row.pre_depth, curve);
        cols[COL_POST_DEPTH][r] = Scalar::from_u64(row.post_depth, curve);
        for j in 0..4 {
            cols[COL_CALLER_PRE_L0 + j][r] = Scalar::from_u64(row.caller_pre[j], curve);
            cols[COL_CALLEE_PRE_L0 + j][r] = Scalar::from_u64(row.callee_pre[j], curve);
            cols[COL_CALLER_POST_L0 + j][r] = Scalar::from_u64(row.caller_post[j], curve);
            cols[COL_CALLEE_POST_L0 + j][r] = Scalar::from_u64(row.callee_post[j], curve);
            cols[COL_VALUE_PRE_L0 + j][r] = Scalar::from_u64(row.value_pre[j], curve);
            cols[COL_VALUE_POST_L0 + j][r] = Scalar::from_u64(row.value_post[j], curve);
        }
        cols[COL_GAS_PRE][r] = Scalar::from_u64(row.gas_pre, curve);
        cols[COL_GAS_POST][r] = Scalar::from_u64(row.gas_post, curve);
        cols[COL_IS_REAL][r] = one.clone();
        let sel_col = match row.kind {
            KIND_CALL => COL_SEL_CALL,
            KIND_CALLCODE => COL_SEL_CALLCODE,
            KIND_DELEGATECALL => COL_SEL_DELEGATECALL,
            KIND_STATICCALL => COL_SEL_STATICCALL,
            KIND_CREATE => COL_SEL_CREATE,
            KIND_CREATE2 => COL_SEL_CREATE2,
            KIND_RETURN => COL_SEL_RETURN,
            KIND_REVERT => COL_SEL_REVERT,
            KIND_STOP_POP => COL_SEL_STOP_POP,
            _ => continue,
        };
        cols[sel_col][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols.into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ───────────────────────────────────────────────
pub struct CallFrameConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl CallFrameConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega); self.domain_size = Some(domain_size); self
    }
}

const SELECTOR_COLS: [usize; 9] = [
    COL_SEL_CALL, COL_SEL_CALLCODE, COL_SEL_DELEGATECALL, COL_SEL_STATICCALL,
    COL_SEL_CREATE, COL_SEL_CREATE2, COL_SEL_RETURN, COL_SEL_REVERT, COL_SEL_STOP_POP,
];

impl VmConstraintSystem for CallFrameConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".to_string()];
        for s in ["call", "callcode", "delegatecall", "staticcall", "create", "create2", "return", "revert", "stop_pop"] {
            labels.push(format!("sel_{}_binary", s));
        }
        labels.push("one_hot_sum".into());
        labels.push("call_family_depth_plus_one".into());
        labels.push("return_revert_depth_minus_one".into());
        for k in 0..4 { labels.push(format!("call_caller_post_eq_callee_pre_l{}", k)); }
        for k in 0..4 { labels.push(format!("delegatecall_caller_stay_l{}", k)); }
        for k in 0..4 { labels.push(format!("staticcall_value_post_zero_l{}", k)); }
        labels
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            // 0: is_real binary
            bodies[0][r] = v.mul(&v.sub(&one));
            // 1..=9: selector binary
            for (i, sc) in SELECTOR_COLS.iter().enumerate() {
                let s = &columns[*sc][r];
                bodies[1 + i][r] = s.mul(&s.sub(&one));
            }
            // 10: sum of selectors == is_real
            let mut sum = columns[SELECTOR_COLS[0]][r].clone();
            for sc in &SELECTOR_COLS[1..] {
                sum = sum.add(&columns[*sc][r]);
            }
            bodies[10][r] = sum.sub(v);
            // 11: CALL-family depth+1
            let cf = columns[COL_SEL_CALL][r].add(&columns[COL_SEL_CALLCODE][r])
                .add(&columns[COL_SEL_DELEGATECALL][r])
                .add(&columns[COL_SEL_STATICCALL][r])
                .add(&columns[COL_SEL_CREATE][r])
                .add(&columns[COL_SEL_CREATE2][r]);
            let depth_diff = columns[COL_POST_DEPTH][r].sub(&columns[COL_PRE_DEPTH][r]).sub(&one);
            bodies[11][r] = cf.mul(&depth_diff);
            // 12: RETURN/REVERT depth-1
            let rr = columns[COL_SEL_RETURN][r].add(&columns[COL_SEL_REVERT][r]);
            // post = pre - 1 → (post - pre + 1) = 0
            let depth_rr = columns[COL_POST_DEPTH][r].sub(&columns[COL_PRE_DEPTH][r]).add(&one);
            bodies[12][r] = rr.mul(&depth_rr);
            // 13..=16: CALL: caller_post[k] = callee_pre[k]
            let sc_call = &columns[COL_SEL_CALL][r];
            for k in 0..4 {
                let d = columns[COL_CALLER_POST_L0 + k][r].sub(&columns[COL_CALLEE_PRE_L0 + k][r]);
                bodies[13 + k][r] = sc_call.mul(&d);
            }
            // 17..=20: DELEGATECALL: caller_post[k] = caller_pre[k]
            let sc_dc = &columns[COL_SEL_DELEGATECALL][r];
            for k in 0..4 {
                let d = columns[COL_CALLER_POST_L0 + k][r].sub(&columns[COL_CALLER_PRE_L0 + k][r]);
                bodies[17 + k][r] = sc_dc.mul(&d);
            }
            // 21..=24: STATICCALL: value_post[k] = 0
            let sc_sc = &columns[COL_SEL_STATICCALL][r];
            for k in 0..4 {
                let d = &columns[COL_VALUE_POST_L0 + k][r];
                bodies[21 + k][r] = sc_sc.mul(d);
            }
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &ce[COL_IS_REAL];
        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(v.mul(&v.sub(&one)));
        for sc in SELECTOR_COLS.iter() {
            let s = &ce[*sc];
            bodies.push(s.mul(&s.sub(&one)));
        }
        let mut sum = ce[SELECTOR_COLS[0]].clone();
        for sc in &SELECTOR_COLS[1..] { sum = sum.add(&ce[*sc]); }
        bodies.push(sum.sub(v));
        let cf = ce[COL_SEL_CALL].add(&ce[COL_SEL_CALLCODE])
            .add(&ce[COL_SEL_DELEGATECALL]).add(&ce[COL_SEL_STATICCALL])
            .add(&ce[COL_SEL_CREATE]).add(&ce[COL_SEL_CREATE2]);
        let depth_diff = ce[COL_POST_DEPTH].sub(&ce[COL_PRE_DEPTH]).sub(&one);
        bodies.push(cf.mul(&depth_diff));
        let rr = ce[COL_SEL_RETURN].add(&ce[COL_SEL_REVERT]);
        let depth_rr = ce[COL_POST_DEPTH].sub(&ce[COL_PRE_DEPTH]).add(&one);
        bodies.push(rr.mul(&depth_rr));
        let sc_call = &ce[COL_SEL_CALL];
        for k in 0..4 {
            let d = ce[COL_CALLER_POST_L0 + k].sub(&ce[COL_CALLEE_PRE_L0 + k]);
            bodies.push(sc_call.mul(&d));
        }
        let sc_dc = &ce[COL_SEL_DELEGATECALL];
        for k in 0..4 {
            let d = ce[COL_CALLER_POST_L0 + k].sub(&ce[COL_CALLER_PRE_L0 + k]);
            bodies.push(sc_dc.mul(&d));
        }
        let sc_sc = &ce[COL_SEL_STATICCALL];
        for k in 0..4 {
            let d = &ce[COL_VALUE_POST_L0 + k];
            bodies.push(sc_sc.mul(d));
        }
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
        let v = &cc[COL_IS_REAL];
        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        // 0: is_real binary
        let v_m1 = poly_sub(v, &one_p, curve);
        bodies.push(poly_mul(v, &v_m1, curve));
        // 1..=9: selector binary
        for sc in SELECTOR_COLS.iter() {
            let s = &cc[*sc];
            let s_m1 = poly_sub(s, &one_p, curve);
            bodies.push(poly_mul(s, &s_m1, curve));
        }
        // 10: one-hot sum
        let mut sum = cc[SELECTOR_COLS[0]].clone();
        for sc in &SELECTOR_COLS[1..] { sum = poly_add(&sum, &cc[*sc], curve); }
        bodies.push(poly_sub(&sum, v, curve));
        // 11: CALL-family depth+1
        let mut cf = cc[COL_SEL_CALL].clone();
        for sc in [COL_SEL_CALLCODE, COL_SEL_DELEGATECALL, COL_SEL_STATICCALL, COL_SEL_CREATE, COL_SEL_CREATE2] {
            cf = poly_add(&cf, &cc[sc], curve);
        }
        let post_minus_pre = poly_sub(&cc[COL_POST_DEPTH], &cc[COL_PRE_DEPTH], curve);
        let depth_diff = poly_sub(&post_minus_pre, &one_p, curve);
        bodies.push(poly_mul(&cf, &depth_diff, curve));
        // 12: RETURN/REVERT depth-1: (post - pre + 1) = 0
        let rr = poly_add(&cc[COL_SEL_RETURN], &cc[COL_SEL_REVERT], curve);
        let depth_rr = poly_add(&post_minus_pre, &one_p, curve);
        bodies.push(poly_mul(&rr, &depth_rr, curve));
        // 13..=16: CALL caller propagation
        for k in 0..4 {
            let d = poly_sub(&cc[COL_CALLER_POST_L0 + k], &cc[COL_CALLEE_PRE_L0 + k], curve);
            bodies.push(poly_mul(&cc[COL_SEL_CALL], &d, curve));
        }
        // 17..=20: DELEGATECALL caller stay
        for k in 0..4 {
            let d = poly_sub(&cc[COL_CALLER_POST_L0 + k], &cc[COL_CALLER_PRE_L0 + k], curve);
            bodies.push(poly_mul(&cc[COL_SEL_DELEGATECALL], &d, curve));
        }
        // 21..=24: STATICCALL value zero
        for k in 0..4 {
            let d = &cc[COL_VALUE_POST_L0 + k];
            bodies.push(poly_mul(&cc[COL_SEL_STATICCALL], d, curve));
        }
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS { return; }
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) { *cell = zero.clone(); }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

// ─── Witness extraction from EVM trace ──────────────────────────────
/// Walk an EVM trace and emit one CallFrameRow for each frame-changing
/// opcode (CALL family + RETURN/REVERT + STOP). Each row reads (pre, post)
/// frame state from consecutive trace rows.
pub fn from_evm_trace(cols: &crate::trace::EvmTraceColumns) -> CallFrameWitness {
    let n = cols.step.len();
    let mut rows = Vec::new();
    if n < 2 { return CallFrameWitness { rows }; }
    for r in 0..n - 1 {
        let opcode = cols.opcode[r] as u8;
        let kind = match opcode {
            0xF1 => KIND_CALL,
            0xF2 => KIND_CALLCODE,
            0xF4 => KIND_DELEGATECALL,
            0xFA => KIND_STATICCALL,
            0xF0 => KIND_CREATE,
            0xF5 => KIND_CREATE2,
            0xF3 => KIND_RETURN,
            0xFD => KIND_REVERT,
            0x00 => {
                // STOP only counts as a frame-pop if pre_depth > 0.
                if cols.frame_depth[r] == 0 { continue; }
                KIND_STOP_POP
            }
            _ => continue,
        };
        rows.push(CallFrameRow {
            kind,
            pre_depth: cols.frame_depth[r],
            post_depth: cols.frame_depth[r + 1],
            caller_pre: [cols.frame_caller[0][r], cols.frame_caller[1][r], cols.frame_caller[2][r], cols.frame_caller[3][r]],
            callee_pre: [cols.frame_callee[0][r], cols.frame_callee[1][r], cols.frame_callee[2][r], cols.frame_callee[3][r]],
            caller_post: [cols.frame_caller[0][r+1], cols.frame_caller[1][r+1], cols.frame_caller[2][r+1], cols.frame_caller[3][r+1]],
            callee_post: [cols.frame_callee[0][r+1], cols.frame_callee[1][r+1], cols.frame_callee[2][r+1], cols.frame_callee[3][r+1]],
            value_pre: [cols.frame_value[0][r], cols.frame_value[1][r], cols.frame_value[2][r], cols.frame_value[3][r]],
            value_post: [cols.frame_value[0][r+1], cols.frame_value[1][r+1], cols.frame_value[2][r+1], cols.frame_value[3][r+1]],
            gas_pre: cols.frame_gas[r],
            gas_post: cols.frame_gas[r + 1],
        });
    }
    CallFrameWitness { rows }
}

// ─── Cross-AIR LogUp descriptor ──────────────────────────────────────
/// Bind EVM main's CALL row (sel_call gated) to this AIR's CALL row.
/// Tuple: (frame_depth, callee_pre[0..4], caller_post[0..4]) — 9 cols.
/// On the EVM side: (frame_depth_curr, frame_callee_curr[0..4], frame_caller_next[0..4])
/// — but since LogUp tuples must be drawn from a single row, this
/// binding instead exposes the within-row CALL marker plus the gadget's
/// caller_post/callee_pre fields. Algebraic depth+1 and caller propagation
/// constraints live entirely inside the gadget.
///
/// We bind on (pre_depth, callee_pre L0..L3, caller_pre L0..L3) to the
/// EVM row (frame_depth, frame_callee L0..L3, frame_caller L0..L3).
pub fn make_evm_to_call_frame_air_descriptor(
    label: &str,
    evm_layer: usize,
    gadget_layer: usize,
    evm_selector_col: usize,
    gadget_selector_col: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::*;
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: evm_layer,
        a_columns: vec![
            COL_FRAME_DEPTH,
            COL_FRAME_CALLEE_L0, COL_FRAME_CALLEE_L1, COL_FRAME_CALLEE_L2, COL_FRAME_CALLEE_L3,
            COL_FRAME_CALLER_L0, COL_FRAME_CALLER_L1, COL_FRAME_CALLER_L2, COL_FRAME_CALLER_L3,
        ],
        a_selector_column: Some(evm_selector_col),
        b_layer_index: gadget_layer,
        b_columns: vec![
            COL_PRE_DEPTH,
            COL_CALLEE_PRE_L0, COL_CALLEE_PRE_L1, COL_CALLEE_PRE_L2, COL_CALLEE_PRE_L3,
            COL_CALLER_PRE_L0, COL_CALLER_PRE_L1, COL_CALLER_PRE_L2, COL_CALLER_PRE_L3,
        ],
        b_selector_column: Some(gadget_selector_col),
    }
}

/// Convenience: bind the umbrella CALL family selector (COL_SEL_CALL) on
/// the EVM side to the gadget's CALL selector column.
pub fn make_call_descriptor(evm_layer: usize, gadget_layer: usize)
    -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor
{
    make_evm_to_call_frame_air_descriptor(
        "evm_call_to_call_frame_air_v1",
        evm_layer,
        gadget_layer,
        crate::trace::COL_SEL_CALL,
        COL_SEL_CALL,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn honest_call_row() -> CallFrameRow {
        CallFrameRow {
            kind: KIND_CALL,
            pre_depth: 1,
            post_depth: 2,
            caller_pre: [1, 0, 0, 0],
            callee_pre: [2, 0, 0, 0],
            caller_post: [2, 0, 0, 0], // = callee_pre
            callee_post: [3, 0, 0, 0],
            value_pre: [10, 0, 0, 0],
            value_post: [5, 0, 0, 0],
            gas_pre: 1000,
            gas_post: 800,
        }
    }

    fn honest_return_row() -> CallFrameRow {
        CallFrameRow {
            kind: KIND_RETURN,
            pre_depth: 2,
            post_depth: 1,
            caller_pre: [2, 0, 0, 0],
            callee_pre: [3, 0, 0, 0],
            caller_post: [1, 0, 0, 0],
            callee_post: [2, 0, 0, 0],
            value_pre: [0, 0, 0, 0],
            value_post: [0, 0, 0, 0],
            gas_pre: 500,
            gas_post: 600,
        }
    }

    fn honest_delegatecall_row() -> CallFrameRow {
        CallFrameRow {
            kind: KIND_DELEGATECALL,
            pre_depth: 1,
            post_depth: 2,
            caller_pre: [7, 0, 0, 0],
            callee_pre: [8, 0, 0, 0],
            caller_post: [7, 0, 0, 0], // unchanged
            callee_post: [9, 0, 0, 0],
            value_pre: [0, 0, 0, 0],
            value_post: [0, 0, 0, 0],
            gas_pre: 1000,
            gas_post: 800,
        }
    }

    fn honest_staticcall_row() -> CallFrameRow {
        CallFrameRow {
            kind: KIND_STATICCALL,
            pre_depth: 1,
            post_depth: 2,
            caller_pre: [4, 0, 0, 0],
            callee_pre: [5, 0, 0, 0],
            caller_post: [5, 0, 0, 0],
            callee_post: [6, 0, 0, 0],
            value_pre: [0, 0, 0, 0],
            value_post: [0, 0, 0, 0], // STATICCALL → no value
            gas_pre: 1000,
            gas_post: 800,
        }
    }

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn honest_call_passes() {
        let w = CallFrameWitness::from_rows(vec![honest_call_row()]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn honest_return_passes() {
        let w = CallFrameWitness::from_rows(vec![honest_return_row()]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn honest_delegatecall_passes() {
        let w = CallFrameWitness::from_rows(vec![honest_delegatecall_row()]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn honest_staticcall_passes() {
        let w = CallFrameWitness::from_rows(vec![honest_staticcall_row()]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn mixed_honest_witness_passes() {
        let w = CallFrameWitness::from_rows(vec![
            honest_call_row(),
            honest_delegatecall_row(),
            honest_staticcall_row(),
            honest_return_row(),
        ]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn tampered_call_depth_fails() {
        let mut row = honest_call_row();
        row.post_depth = 1; // should be 2
        let w = CallFrameWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // constraint 11 = CALL-family depth+1
        assert!(!bodies[11][0].is_zero(), "expected depth constraint to fire");
    }

    #[test]
    fn tampered_return_depth_fails() {
        let mut row = honest_return_row();
        row.post_depth = 5; // should be 1
        let w = CallFrameWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[12][0].is_zero(), "expected RETURN depth constraint to fire");
    }

    #[test]
    fn tampered_caller_propagation_fails() {
        let mut row = honest_call_row();
        row.caller_post[0] = 99; // should equal callee_pre[0] = 2
        let w = CallFrameWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // constraint 13 = CALL caller_post[0] == callee_pre[0]
        assert!(!bodies[13][0].is_zero(), "expected CALL caller propagation constraint to fire");
    }

    #[test]
    fn tampered_delegatecall_caller_fails() {
        let mut row = honest_delegatecall_row();
        row.caller_post[0] = 99; // should equal caller_pre[0]
        let w = CallFrameWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // constraint 17 = DELEGATECALL caller stay l0
        assert!(!bodies[17][0].is_zero(), "expected DELEGATECALL caller stay constraint to fire");
    }

    #[test]
    fn staticcall_value_zero_enforced() {
        let mut row = honest_staticcall_row();
        row.value_post[1] = 7; // must be zero
        let w = CallFrameWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // constraint 22 = STATICCALL value_post[1] = 0
        assert!(!bodies[22][0].is_zero(), "expected STATICCALL value=0 constraint to fire");
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let w = CallFrameWitness::from_rows(vec![honest_call_row()]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0x1234, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero());
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_call_descriptor(0, 1);
        assert_eq!(d.label, "evm_call_to_call_frame_air_v1");
        assert_eq!(d.a_columns.len(), 9);
        assert_eq!(d.b_columns.len(), 9);
        assert!(d.a_selector_column.is_some());
        assert!(d.b_selector_column.is_some());
    }

    #[test]
    fn from_evm_trace_extracts_stop_at_top_level_ignored() {
        // Bytecode: PUSH1 1; PUSH1 2; ADD; STOP
        // depth stays 0 throughout — STOP at depth 0 is NOT a pop.
        use crate::executor::execute_bytecode;
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        // No CALL/RETURN/REVERT, and STOP at depth 0 is skipped → 0 rows.
        assert_eq!(w.rows.len(), 0);
    }

    #[test]
    fn from_evm_trace_then_constraints_zero() {
        use crate::executor::execute_bytecode;
        // Simple bytecode with no frame transitions.
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFrameConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }
}
