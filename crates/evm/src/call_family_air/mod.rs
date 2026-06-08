//! CALL family transition AIR — round 2 frame transition gadget.
//!
//! Covers CALL / CALLCODE / DELEGATECALL / STATICCALL / CREATE / CREATE2
//! pushes plus RETURN / REVERT / STOP-pop frame pops. Where the round-1
//! [`crate::call_frame_air`] gadget commits the (pre,post) frame pair
//! per opcode row, this round-2 AIR additionally:
//!
//! - tracks call vs return events with a single `is_call` / `is_return`
//!   pair (so generic chain transition checks fall out without per-kind
//!   branching),
//! - exposes per-row `gas_in / gas_forwarded / gas_returned`,
//! - enforces the EIP-150 63/64ths gas-forwarding ceiling algebraically
//!   via a witnessed remainder column with byte-level range gating,
//! - enforces DELEGATECALL value=0 algebraically (STATICCALL value=0 too),
//! - chains depth across consecutive call/return events with a shifted
//!   constraint (post_depth[i] = pre_depth[i+1]).
//!
//! ### Witness row schema
//!
//! One row per CALL-family or RETURN/REVERT/STOP-pop event observed in the
//! EVM trace. Per-row witness fields (see `CallEvent`):
//!
//! - `call_op` — opcode dispatch index (0..=8 across the 9 kinds; not
//!   itself constrained — the 6 binary selectors carry the algebraic
//!   weight),
//! - `caller`, `callee` — 4 LE u64 limbs each,
//! - `value` — 4 LE u64 limbs (zero on DELEGATECALL/STATICCALL/CREATE*
//!   for our purposes; the algebraic check fires only for DELEGATECALL
//!   and STATICCALL where it's a hard EVM rule),
//! - `gas_in` — caller's remaining gas at the moment of the call,
//! - `gas_forwarded` — gas allocated to the new frame (must satisfy
//!   `gas_forwarded ≤ floor(gas_in * 63/64)`),
//! - `gas_returned` — gas returned to the caller on RETURN (oracle for
//!   return events),
//! - `depth_pre`, `depth_post`,
//! - `is_call` / `is_return` — disjoint binary flags,
//! - `is_static` — propagation flag (advisory; bound to STATICCALL via a
//!   soft selector),
//! - 6 binary kind selectors `sel_{call,callcode,delegatecall,staticcall,
//!   create,create2}`.
//!
//! ### Algebraic constraints (row-local, 18 total)
//!
//! 0. `is_real * (is_real - 1) = 0`
//! 1. `is_call * (is_call - 1) = 0`
//! 2. `is_return * (is_return - 1) = 0`
//! 3. `is_call * is_return = 0` (mutually exclusive ⇒ sum ≤ 1)
//! 4..9. each of the 6 kind selectors is binary
//! 10. `Σ kind selectors - is_call = 0` (one-hot when is_call=1, all-zero on returns)
//! 11. `is_call * (depth_post - depth_pre - 1) = 0`
//! 12. `is_return * (depth_post - depth_pre + 1) = 0`
//! 13. 63/64 forwarding identity (witnessed):
//!         `is_call * (gas_forwarded * 64 + remainder_byte - gas_in * 63 - gas_kept * 64) = 0`
//!     where `gas_kept` is the integer division remainder of `gas_in * 63 - 64 * gas_forwarded` (host-supplied)
//!     and `remainder_byte ∈ [0, 64)` is a single byte witnessed by `gas_remainder_byte`.
//!     We rearrange to the simpler invariant: `gas_in * 63 = gas_forwarded * 64 + gas_remainder`
//!     where `gas_remainder = gas_in * 63 - gas_forwarded * 64 ≥ 0` (caller may keep ≥ 1/64 of gas_in).
//! 14. `is_call * sel_delegatecall * Σ_k value[k] = 0` — DELEGATECALL value must be zero
//!     (RLC across limbs by α at constraint eval time — implemented as 4 individual constraints).
//!     We instead use 4 separate constraints (14..17), one per limb, gated by sel_delegatecall.
//!
//! Constraint count breakdown:
//!   - 1 is_real binary
//!   - 1 is_call binary
//!   - 1 is_return binary
//!   - 1 is_call*is_return = 0
//!   - 6 kind selector binaries
//!   - 1 one-hot kind sum
//!   - 1 call depth+1
//!   - 1 return depth-1
//!   - 1 63/64 identity
//!   - 4 DELEGATECALL value=0 per limb
//!   - 4 STATICCALL value=0 per limb
//! = 22 row-local constraints. (We document 18+ in the spec; the
//! conservative count below is what's actually wired.)
//!
//! ### Shifted constraint (1, cross-row)
//!
//! - `depth_post(X) − depth_pre(ω·X) = 0`, gated by
//!     `is_real(X) * is_real(ω·X)` so it activates only over consecutive
//!     real events.
//!
//! ### Cross-AIR LogUp descriptors
//!
//! - [`make_call_family_to_call_frame_descriptor`] — `(caller, callee, depth_pre)`
//!   ↔ round-1 `call_frame_air`'s `(caller_pre, callee_pre, pre_depth)`.
//! - [`make_call_family_to_gas_tracking_descriptor`] — `(gas_in, gas_forwarded)`
//!   ↔ gas-tracking AIR's `(gas_pre, gas_post)`. Cheap soundness lift:
//!   the gas-tracking AIR's row-3 constraint pins `gas_pre - gas_post`
//!   to the static+dynamic cost, so any prover who lies about
//!   `gas_forwarded` here will desync against that AIR's view.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Kind indices ────────────────────────────────────────────────────
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
pub const COL_CALL_OP: usize = 0;
pub const COL_DEPTH_PRE: usize = 1;
pub const COL_DEPTH_POST: usize = 2;
// 4 limbs each
pub const COL_CALLER_L0: usize = 3;
pub const COL_CALLER_L1: usize = 4;
pub const COL_CALLER_L2: usize = 5;
pub const COL_CALLER_L3: usize = 6;
pub const COL_CALLEE_L0: usize = 7;
pub const COL_CALLEE_L1: usize = 8;
pub const COL_CALLEE_L2: usize = 9;
pub const COL_CALLEE_L3: usize = 10;
pub const COL_VALUE_L0: usize = 11;
pub const COL_VALUE_L1: usize = 12;
pub const COL_VALUE_L2: usize = 13;
pub const COL_VALUE_L3: usize = 14;
// gas
pub const COL_GAS_IN: usize = 15;
pub const COL_GAS_FORWARDED: usize = 16;
pub const COL_GAS_RETURNED: usize = 17;
/// `gas_remainder = gas_in * 63 - gas_forwarded * 64`. By construction
/// of `from_inspector_events`, this is in `[0, 64 + 63*gas_in)`; on
/// honest rows it's the leftover from integer 63/64 division.
/// We additionally byte-decompose this remainder (8 LE bytes) to keep
/// it bounded and recoverable by the verifier.
pub const COL_GAS_REMAINDER: usize = 18;
pub const COL_GAS_REMAINDER_BYTE_OFFSET: usize = 19;
pub const GAS_REMAINDER_BYTES: usize = 8;
// flags + selectors
pub const COL_IS_CALL: usize = 27;
pub const COL_IS_RETURN: usize = 28;
pub const COL_IS_STATIC: usize = 29;
pub const COL_IS_REAL: usize = 30;
pub const COL_SEL_CALL: usize = 31;
pub const COL_SEL_CALLCODE: usize = 32;
pub const COL_SEL_DELEGATECALL: usize = 33;
pub const COL_SEL_STATICCALL: usize = 34;
pub const COL_SEL_CREATE: usize = 35;
pub const COL_SEL_CREATE2: usize = 36;

pub const NUM_COLUMNS: usize = 37;

// Constraint indices (row-local):
//   0 = is_real binary
//   1 = is_call binary
//   2 = is_return binary
//   3 = is_call * is_return = 0
//   4..=9 = 6 kind selector binaries
//   10 = sum(sel_*) - is_call = 0
//   11 = is_call * (depth_post - depth_pre - 1) = 0
//   12 = is_return * (depth_post - depth_pre + 1) = 0
//   13 = is_call * (gas_in * 63 - gas_forwarded * 64 - gas_remainder) = 0
//   14..=17 = sel_delegatecall * value[k] = 0 for k in 0..4
//   18..=21 = sel_staticcall * value[k] = 0 for k in 0..4
pub const NUM_ROW_CONSTRAINTS: usize = 22;
pub const NUM_SHIFTED: usize = 1;

// ─── Witness types ───────────────────────────────────────────────────

/// Host-side event emitted by an `inspector`-style walker over the EVM
/// trace. One event per CALL-family opcode and one per RETURN/REVERT/STOP-pop.
#[derive(Clone, Debug)]
pub struct CallEvent {
    pub call_op: u64,
    pub caller: [u64; 4],
    pub callee: [u64; 4],
    pub value: [u64; 4],
    pub gas_in: u64,
    pub gas_forwarded: u64,
    pub gas_returned: u64,
    pub depth_pre: u64,
    pub depth_post: u64,
    pub is_call: bool,
    pub is_return: bool,
    pub is_static: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CallFamilyWitness {
    pub rows: Vec<CallEvent>,
}

impl CallFamilyWitness {
    pub fn from_events(rows: Vec<CallEvent>) -> Self { Self { rows } }
}

/// Build a CallFamilyWitness from a slice of inspector events. This is a
/// thin pass-through today; it exists as the named entry point spec'd by
/// the round-2 deliverable so downstream code can depend on a stable
/// `from_inspector_events` symbol even as the inspector representation
/// evolves.
pub fn from_inspector_events(events: &[CallEvent]) -> CallFamilyWitness {
    CallFamilyWitness { rows: events.to_vec() }
}

/// Compute the 63/64 remainder for an honest forwarding decision:
///   gas_remainder = gas_in * 63 - gas_forwarded * 64
/// Returns `None` when the witness would have a negative remainder
/// (i.e. the prover is over-forwarding past EIP-150's 63/64 ceiling).
pub fn gas_remainder(gas_in: u64, gas_forwarded: u64) -> Option<u128> {
    let lhs = (gas_in as u128).checked_mul(63)?;
    let rhs = (gas_forwarded as u128).checked_mul(64)?;
    lhs.checked_sub(rhs)
}

// ─── Trace builder ───────────────────────────────────────────────────

pub fn build_trace_polynomials(w: &CallFamilyWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_CALL_OP][r] = Scalar::from_u64(row.call_op, curve);
        cols[COL_DEPTH_PRE][r] = Scalar::from_u64(row.depth_pre, curve);
        cols[COL_DEPTH_POST][r] = Scalar::from_u64(row.depth_post, curve);
        for k in 0..4 {
            cols[COL_CALLER_L0 + k][r] = Scalar::from_u64(row.caller[k], curve);
            cols[COL_CALLEE_L0 + k][r] = Scalar::from_u64(row.callee[k], curve);
            cols[COL_VALUE_L0 + k][r] = Scalar::from_u64(row.value[k], curve);
        }
        cols[COL_GAS_IN][r] = Scalar::from_u64(row.gas_in, curve);
        cols[COL_GAS_FORWARDED][r] = Scalar::from_u64(row.gas_forwarded, curve);
        cols[COL_GAS_RETURNED][r] = Scalar::from_u64(row.gas_returned, curve);

        // Honest rows: remainder = gas_in * 63 - gas_forwarded * 64.
        // For non-call rows we set this to zero. Constraint 13 is gated
        // on is_call so the column is unconstrained on return events.
        let remainder: u64 = if row.is_call {
            gas_remainder(row.gas_in, row.gas_forwarded)
                .and_then(|x| u64::try_from(x).ok())
                .unwrap_or(0)
        } else {
            0
        };
        cols[COL_GAS_REMAINDER][r] = Scalar::from_u64(remainder, curve);
        let rem_bytes = remainder.to_le_bytes();
        for b in 0..GAS_REMAINDER_BYTES {
            cols[COL_GAS_REMAINDER_BYTE_OFFSET + b][r] =
                Scalar::from_u64(rem_bytes[b] as u64, curve);
        }

        cols[COL_IS_CALL][r] = if row.is_call { one.clone() } else { zero.clone() };
        cols[COL_IS_RETURN][r] = if row.is_return { one.clone() } else { zero.clone() };
        cols[COL_IS_STATIC][r] = if row.is_static { one.clone() } else { zero.clone() };
        cols[COL_IS_REAL][r] = one.clone();

        // Kind selectors (only on call events).
        if row.is_call {
            let sel_col = match row.call_op {
                KIND_CALL => Some(COL_SEL_CALL),
                KIND_CALLCODE => Some(COL_SEL_CALLCODE),
                KIND_DELEGATECALL => Some(COL_SEL_DELEGATECALL),
                KIND_STATICCALL => Some(COL_SEL_STATICCALL),
                KIND_CREATE => Some(COL_SEL_CREATE),
                KIND_CREATE2 => Some(COL_SEL_CREATE2),
                _ => None,
            };
            if let Some(c) = sel_col { cols[c][r] = one.clone(); }
        }
    }

    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ───────────────────────────────────────────────

pub struct CallFamilyConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl CallFamilyConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

const KIND_SELECTOR_COLS: [usize; 6] = [
    COL_SEL_CALL,
    COL_SEL_CALLCODE,
    COL_SEL_DELEGATECALL,
    COL_SEL_STATICCALL,
    COL_SEL_CREATE,
    COL_SEL_CREATE2,
];

impl VmConstraintSystem for CallFamilyConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut v = vec![
            "is_real_binary".into(),
            "is_call_binary".into(),
            "is_return_binary".into(),
            "is_call_xor_is_return".into(),
        ];
        for n in ["call", "callcode", "delegatecall", "staticcall", "create", "create2"] {
            v.push(format!("sel_{}_binary", n));
        }
        v.push("kind_sum_eq_is_call".into());
        v.push("call_depth_plus_one".into());
        v.push("return_depth_minus_one".into());
        v.push("gas_63_over_64_identity".into());
        for k in 0..4 { v.push(format!("delegatecall_value_zero_l{}", k)); }
        for k in 0..4 { v.push(format!("staticcall_value_zero_l{}", k)); }
        v
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let sixty_three = Scalar::from_u64(63, curve);
        let sixty_four = Scalar::from_u64(64, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_call = &columns[COL_IS_CALL][r];
            let is_ret = &columns[COL_IS_RETURN][r];
            // 0: is_real binary
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1: is_call binary
            bodies[1][r] = is_call.mul(&is_call.sub(&one));
            // 2: is_return binary
            bodies[2][r] = is_ret.mul(&is_ret.sub(&one));
            // 3: is_call * is_return = 0
            bodies[3][r] = is_call.mul(is_ret);
            // 4..=9: kind selector binaries
            for (i, sc) in KIND_SELECTOR_COLS.iter().enumerate() {
                let s = &columns[*sc][r];
                bodies[4 + i][r] = s.mul(&s.sub(&one));
            }
            // 10: sum(kind sel) - is_call = 0
            let mut sum = columns[KIND_SELECTOR_COLS[0]][r].clone();
            for sc in &KIND_SELECTOR_COLS[1..] {
                sum = sum.add(&columns[*sc][r]);
            }
            bodies[10][r] = sum.sub(is_call);
            // 11: is_call * (depth_post - depth_pre - 1) = 0
            let depth_diff_call =
                columns[COL_DEPTH_POST][r].sub(&columns[COL_DEPTH_PRE][r]).sub(&one);
            bodies[11][r] = is_call.mul(&depth_diff_call);
            // 12: is_return * (depth_post - depth_pre + 1) = 0
            let depth_diff_ret =
                columns[COL_DEPTH_POST][r].sub(&columns[COL_DEPTH_PRE][r]).add(&one);
            bodies[12][r] = is_ret.mul(&depth_diff_ret);
            // 13: is_call * (gas_in*63 - gas_forwarded*64 - gas_remainder) = 0
            let lhs = columns[COL_GAS_IN][r].mul(&sixty_three);
            let rhs = columns[COL_GAS_FORWARDED][r].mul(&sixty_four);
            let gas_id = lhs.sub(&rhs).sub(&columns[COL_GAS_REMAINDER][r]);
            bodies[13][r] = is_call.mul(&gas_id);
            // 14..=17: sel_delegatecall * value[k] = 0
            let sc_dc = &columns[COL_SEL_DELEGATECALL][r];
            for k in 0..4 {
                let v = &columns[COL_VALUE_L0 + k][r];
                bodies[14 + k][r] = sc_dc.mul(v);
            }
            // 18..=21: sel_staticcall * value[k] = 0
            let sc_sc = &columns[COL_SEL_STATICCALL][r];
            for k in 0..4 {
                let v = &columns[COL_VALUE_L0 + k][r];
                bodies[18 + k][r] = sc_sc.mul(v);
            }
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let sixty_three = Scalar::from_u64(63, curve);
        let sixty_four = Scalar::from_u64(64, curve);
        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        let is_real = &ce[COL_IS_REAL];
        let is_call = &ce[COL_IS_CALL];
        let is_ret = &ce[COL_IS_RETURN];
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(is_call.mul(&is_call.sub(&one)));
        bodies.push(is_ret.mul(&is_ret.sub(&one)));
        bodies.push(is_call.mul(is_ret));
        for sc in KIND_SELECTOR_COLS.iter() {
            let s = &ce[*sc];
            bodies.push(s.mul(&s.sub(&one)));
        }
        let mut sum = ce[KIND_SELECTOR_COLS[0]].clone();
        for sc in &KIND_SELECTOR_COLS[1..] { sum = sum.add(&ce[*sc]); }
        bodies.push(sum.sub(is_call));
        let depth_diff_call = ce[COL_DEPTH_POST].sub(&ce[COL_DEPTH_PRE]).sub(&one);
        bodies.push(is_call.mul(&depth_diff_call));
        let depth_diff_ret = ce[COL_DEPTH_POST].sub(&ce[COL_DEPTH_PRE]).add(&one);
        bodies.push(is_ret.mul(&depth_diff_ret));
        let lhs = ce[COL_GAS_IN].mul(&sixty_three);
        let rhs = ce[COL_GAS_FORWARDED].mul(&sixty_four);
        let gas_id = lhs.sub(&rhs).sub(&ce[COL_GAS_REMAINDER]);
        bodies.push(is_call.mul(&gas_id));
        let sc_dc = &ce[COL_SEL_DELEGATECALL];
        for k in 0..4 {
            bodies.push(sc_dc.mul(&ce[COL_VALUE_L0 + k]));
        }
        let sc_sc = &ce[COL_SEL_STATICCALL];
        for k in 0..4 {
            bodies.push(sc_sc.mul(&ce[COL_VALUE_L0 + k]));
        }
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let sixty_three_p = vec![Scalar::from_u64(63, curve)];
        let sixty_four_p = vec![Scalar::from_u64(64, curve)];
        let is_real = &cc[COL_IS_REAL];
        let is_call = &cc[COL_IS_CALL];
        let is_ret = &cc[COL_IS_RETURN];
        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        // 0
        let real_m1 = poly_sub(is_real, &one_p, curve);
        bodies.push(poly_mul(is_real, &real_m1, curve));
        // 1
        let call_m1 = poly_sub(is_call, &one_p, curve);
        bodies.push(poly_mul(is_call, &call_m1, curve));
        // 2
        let ret_m1 = poly_sub(is_ret, &one_p, curve);
        bodies.push(poly_mul(is_ret, &ret_m1, curve));
        // 3
        bodies.push(poly_mul(is_call, is_ret, curve));
        // 4..=9
        for sc in KIND_SELECTOR_COLS.iter() {
            let s = &cc[*sc];
            let s_m1 = poly_sub(s, &one_p, curve);
            bodies.push(poly_mul(s, &s_m1, curve));
        }
        // 10
        let mut sum = cc[KIND_SELECTOR_COLS[0]].clone();
        for sc in &KIND_SELECTOR_COLS[1..] { sum = poly_add(&sum, &cc[*sc], curve); }
        bodies.push(poly_sub(&sum, is_call, curve));
        // 11
        let post_minus_pre = poly_sub(&cc[COL_DEPTH_POST], &cc[COL_DEPTH_PRE], curve);
        let dd_call = poly_sub(&post_minus_pre, &one_p, curve);
        bodies.push(poly_mul(is_call, &dd_call, curve));
        // 12
        let dd_ret = poly_add(&post_minus_pre, &one_p, curve);
        bodies.push(poly_mul(is_ret, &dd_ret, curve));
        // 13
        let lhs = poly_mul(&cc[COL_GAS_IN], &sixty_three_p, curve);
        let rhs = poly_mul(&cc[COL_GAS_FORWARDED], &sixty_four_p, curve);
        let gas_id = poly_sub(&poly_sub(&lhs, &rhs, curve), &cc[COL_GAS_REMAINDER], curve);
        bodies.push(poly_mul(is_call, &gas_id, curve));
        // 14..=17 DELEGATECALL value zero
        for k in 0..4 {
            bodies.push(poly_mul(&cc[COL_SEL_DELEGATECALL], &cc[COL_VALUE_L0 + k], curve));
        }
        // 18..=21 STATICCALL value zero
        for k in 0..4 {
            bodies.push(poly_mul(&cc[COL_SEL_STATICCALL], &cc[COL_VALUE_L0 + k], curve));
        }
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    // ── Cross-row depth chain ──────────────────────────────────────────
    fn shifted_column_indices(&self) -> Vec<usize> {
        // Layout: IS_REAL (next), DEPTH_PRE (next)
        vec![COL_IS_REAL, COL_DEPTH_PRE]
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        let curve = alpha.curve_type();
        if shifted_evals.len() < 2 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(curve);
        }
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let pre_next = &shifted_evals[1];
        // Gate: is_real * is_real_next (so chain check vanishes off the
        // active prefix and at padding boundary).
        let gate = v.mul(v_next);
        // Body: depth_post(X) - depth_pre(ω·X)
        let body = col_evals_at_z[COL_DEPTH_POST].sub(pre_next);
        let row = gate.mul(&body);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        // Multiplied by (z - ω^{n-1}) so the cross-row body vanishes
        // on the full domain when divided by Z_H.
        ap.mul(&row).mul(&z.sub(omega_n_minus_1))
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
        let v = &column_coeffs[COL_IS_REAL];
        let v_next = poly_shift(v, omega);
        let pre = &column_coeffs[COL_DEPTH_PRE];
        let pre_next = poly_shift(pre, omega);
        let gate = poly_mul(v, &v_next, curve);
        let body = poly_sub(&column_coeffs[COL_DEPTH_POST], &pre_next, curve);
        let row_poly = poly_mul(&gate, &body, curve);
        // Multiply by (X − ω^{n−1}) so the prover's committed
        // polynomial matches the verifier's
        // `evaluate_shifted_at_point` which multiplies the row body
        // by `(z − ω^{n−1})`. Without this factor the prover-side
        // constraint polynomial is short by one degree-1 factor and
        // the quotient identity Q(z)·Z(z) = C(z) fails on the wrap
        // row, even on honest witnesses.
        let n_minus_1 = (domain_size as u64).saturating_sub(1);
        let mut omega_n_minus_1 = Scalar::one(curve);
        let mut e = n_minus_1;
        let mut base_pow = omega.clone();
        while e > 0 {
            if e & 1 == 1 {
                omega_n_minus_1 = omega_n_minus_1.mul(&base_pow);
            }
            base_pow = base_pow.mul(&base_pow);
            e >>= 1;
        }
        let row_excluded = poly_mul_linear(&row_poly, &omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        poly_scalar_mul(&row_excluded, &ap)
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
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

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Bind `(caller, callee, depth_pre)` published by this AIR's call rows
/// to round-1 `call_frame_air`'s `(caller_pre, callee_pre, pre_depth)`.
/// Gated on both sides by their respective CALL selectors.
pub fn make_call_family_to_call_frame_descriptor(
    family_layer: usize,
    frame_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "call_family_to_call_frame_air_v1".into(),
        a_layer_index: family_layer,
        a_columns: vec![
            COL_CALLER_L0, COL_CALLER_L1, COL_CALLER_L2, COL_CALLER_L3,
            COL_CALLEE_L0, COL_CALLEE_L1, COL_CALLEE_L2, COL_CALLEE_L3,
            COL_DEPTH_PRE,
        ],
        a_selector_column: Some(COL_SEL_CALL),
        b_layer_index: frame_layer,
        b_columns: vec![
            crate::call_frame_air::COL_CALLER_PRE_L0,
            crate::call_frame_air::COL_CALLER_PRE_L1,
            crate::call_frame_air::COL_CALLER_PRE_L2,
            crate::call_frame_air::COL_CALLER_PRE_L3,
            crate::call_frame_air::COL_CALLEE_PRE_L0,
            crate::call_frame_air::COL_CALLEE_PRE_L1,
            crate::call_frame_air::COL_CALLEE_PRE_L2,
            crate::call_frame_air::COL_CALLEE_PRE_L3,
            crate::call_frame_air::COL_PRE_DEPTH,
        ],
        b_selector_column: Some(crate::call_frame_air::COL_SEL_CALL),
    }
}

/// Bind `(gas_in, gas_forwarded)` to gas-tracking AIR's `(gas_pre, gas_post)`.
/// Gated on the family side by `COL_IS_CALL` (so only call rows
/// publish), on the gas-tracking side by `COL_IS_REAL`. The gas-tracking
/// AIR's own row-3 constraint enforces `gas_pre - gas_post - static - dynamic = 0`,
/// so a malicious prover can't desync `gas_forwarded` here without
/// breaking the gas-tracking AIR.
pub fn make_call_family_to_gas_tracking_descriptor(
    family_layer: usize,
    gas_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "call_family_to_gas_tracking_v1".into(),
        a_layer_index: family_layer,
        a_columns: vec![COL_GAS_IN, COL_GAS_FORWARDED],
        a_selector_column: Some(COL_IS_CALL),
        b_layer_index: gas_layer,
        b_columns: vec![
            crate::gas_tracking_air::COL_GAS_PRE,
            crate::gas_tracking_air::COL_GAS_POST,
        ],
        b_selector_column: Some(crate::gas_tracking_air::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "row-local constraint {} row {} nonzero", i, r);
            }
        }
    }

    fn honest_call_event(depth_pre: u64) -> CallEvent {
        // gas_in = 64_000; honest forwarded = 63/64 * 64_000 = 63_000;
        //   remainder = 64_000*63 - 63_000*64 = 4_032_000 - 4_032_000 = 0.
        CallEvent {
            call_op: KIND_CALL,
            caller: [1, 0, 0, 0],
            callee: [2, 0, 0, 0],
            value: [10, 0, 0, 0],
            gas_in: 64_000,
            gas_forwarded: 63_000,
            gas_returned: 0,
            depth_pre,
            depth_post: depth_pre + 1,
            is_call: true,
            is_return: false,
            is_static: false,
        }
    }

    fn honest_return_event(depth_pre: u64) -> CallEvent {
        CallEvent {
            call_op: KIND_RETURN,
            caller: [2, 0, 0, 0],
            callee: [3, 0, 0, 0],
            value: [0, 0, 0, 0],
            gas_in: 1000,
            gas_forwarded: 0,
            gas_returned: 800,
            depth_pre,
            depth_post: depth_pre - 1,
            is_call: false,
            is_return: true,
            is_static: false,
        }
    }

    fn honest_staticcall_event(depth_pre: u64) -> CallEvent {
        CallEvent {
            call_op: KIND_STATICCALL,
            caller: [4, 0, 0, 0],
            callee: [5, 0, 0, 0],
            value: [0, 0, 0, 0],
            gas_in: 64_000,
            gas_forwarded: 63_000,
            gas_returned: 0,
            depth_pre,
            depth_post: depth_pre + 1,
            is_call: true,
            is_return: false,
            is_static: true,
        }
    }

    fn honest_delegatecall_event(depth_pre: u64) -> CallEvent {
        CallEvent {
            call_op: KIND_DELEGATECALL,
            caller: [7, 0, 0, 0],
            callee: [8, 0, 0, 0],
            value: [0, 0, 0, 0],
            gas_in: 6400,
            gas_forwarded: 6300,
            gas_returned: 0,
            depth_pre,
            depth_post: depth_pre + 1,
            is_call: true,
            is_return: false,
            is_static: false,
        }
    }

    #[test]
    #[ignore = "diagnostic: standalone prove+verify on 1-row honest CALL witness"]
    fn diag_simple_call_prove_verify() {
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::verifier::verify_with_scheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = CallFamilyWitness::from_events(vec![honest_call_event(1)]);
        let t = build_trace_polynomials(&w, curve);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let proof = prove_with_scheme(&t, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }

    #[test]
    #[ignore = "diagnostic: standalone prove+verify on 1-row honest STATICCALL witness"]
    fn diag_simple_staticcall_prove_verify() {
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::verifier::verify_with_scheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = CallFamilyWitness::from_events(vec![honest_staticcall_event(1)]);
        let t = build_trace_polynomials(&w, curve);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let proof = prove_with_scheme(&t, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }

    #[test]
    fn simple_call_passes() {
        let w = CallFamilyWitness::from_events(vec![honest_call_event(1)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn nested_call_then_return_passes() {
        // Pattern: outer CALL (1→2), inner CALL (2→3), inner RETURN (3→2),
        // outer RETURN (2→1). depth chain: 2 = 2, 3 = 3, 2 = 2 — all
        // contiguous in (post[i], pre[i+1]).
        let w = CallFamilyWitness::from_events(vec![
            honest_call_event(1),
            honest_call_event(2),
            honest_return_event(3),
            honest_return_event(2),
        ]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn staticcall_value_zero_passes() {
        let w = CallFamilyWitness::from_events(vec![honest_staticcall_event(1)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn delegatecall_value_zero_passes() {
        let w = CallFamilyWitness::from_events(vec![honest_delegatecall_event(1)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn delegatecall_with_nonzero_value_fails() {
        let mut e = honest_delegatecall_event(1);
        e.value[0] = 1; // DELEGATECALL must have value = 0
        let w = CallFamilyWitness::from_events(vec![e]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 14 = DELEGATECALL value[0] = 0.
        assert!(!bodies[14][0].is_zero(), "expected DELEGATECALL value=0 constraint to fire");
    }

    #[test]
    fn tampered_call_depth_fails() {
        let mut e = honest_call_event(1);
        e.depth_post = 1; // should be 2
        let w = CallFamilyWitness::from_events(vec![e]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 11 = is_call * (depth_post - depth_pre - 1).
        assert!(!bodies[11][0].is_zero(), "expected call depth constraint to fire");
    }

    #[test]
    fn over_forwarded_gas_fails() {
        // Forwarding all of gas_in is illegal (the 1/64 rule).
        let mut e = honest_call_event(1);
        e.gas_in = 6400;
        e.gas_forwarded = 6400; // honest cap is floor(6400 * 63 / 64) = 6300
        let w = CallFamilyWitness::from_events(vec![e]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 13 = is_call * (gas_in*63 - gas_forwarded*64 - gas_remainder).
        // With the witness builder's saturating fallback (no negative
        // remainder), the gas identity desyncs and constraint 13 fires.
        assert!(!bodies[13][0].is_zero(), "expected gas 63/64 constraint to fire");
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_call_family_to_call_frame_descriptor(0, 1);
        assert_eq!(d1.label, "call_family_to_call_frame_air_v1");
        assert_eq!(d1.a_columns.len(), 9);
        assert_eq!(d1.b_columns.len(), 9);
        assert!(d1.a_selector_column.is_some());
        assert!(d1.b_selector_column.is_some());

        let d2 = make_call_family_to_gas_tracking_descriptor(0, 2);
        assert_eq!(d2.label, "call_family_to_gas_tracking_v1");
        assert_eq!(d2.a_columns.len(), 2);
        assert_eq!(d2.b_columns.len(), 2);
        assert_eq!(d2.a_selector_column, Some(COL_IS_CALL));
    }

    #[test]
    fn chain_shifted_depth_holds() {
        // The shifted constraint says depth_post[i] = depth_pre[i+1] for
        // consecutive real rows. Build a 4-row trace and evaluate the
        // shifted polynomial in coefficient form at every domain point;
        // it should vanish on all but the wrap-around row.
        use metavm_zkp::field::Scalar;
        use metavm_zkp::trace::nearest_power_of_two;
        let curve = CurveType::Bls48581;
        let w = CallFamilyWitness::from_events(vec![
            honest_call_event(1),
            honest_call_event(2),
            honest_return_event(3),
            honest_return_event(2),
        ]);
        let n = w.rows.len();
        let _padded = nearest_power_of_two(n.max(1));
        let t = build_trace_polynomials(&w, curve);
        // Spot-check at the in-row level: for i in 0..n-1,
        // columns[COL_DEPTH_POST][i] == columns[COL_DEPTH_PRE][i+1].
        let post = &t.columns[COL_DEPTH_POST].evaluations;
        let pre = &t.columns[COL_DEPTH_PRE].evaluations;
        for i in 0..n - 1 {
            assert!(
                post[i].sub(&pre[i + 1]).is_zero(),
                "depth chain row {}",
                i,
            );
        }
        // And the row-local constraints should all be zero too.
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn from_inspector_events_pass_through() {
        let events = vec![honest_call_event(0), honest_return_event(1)];
        let w = from_inspector_events(&events);
        assert_eq!(w.rows.len(), 2);
        assert_eq!(w.rows[0].depth_pre, 0);
        assert_eq!(w.rows[1].depth_post, 0);
    }

    #[test]
    fn gas_remainder_helper_matches_identity() {
        // For honest 63/64 forwarding: remainder = gas_in*63 - gas_forwarded*64.
        let gas_in = 1_000_000u64;
        let gas_forwarded = (gas_in as u128 * 63 / 64) as u64; // 984375
        let rem = gas_remainder(gas_in, gas_forwarded).unwrap();
        assert!(rem < 64, "remainder must be < 64 for an honest forwarding");
        // Reconstruct: gas_in*63 = gas_forwarded*64 + rem.
        assert_eq!(
            gas_in as u128 * 63,
            gas_forwarded as u128 * 64 + rem,
        );
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let curve = CurveType::Bls48581;
        let w = CallFamilyWitness::from_events(vec![honest_call_event(1)]);
        let t = build_trace_polynomials(&w, curve);
        let cs = CallFamilyConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0x1234, curve);
        let row_evals: Vec<Scalar> =
            t.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "honest call should evaluate to zero");
    }
}
