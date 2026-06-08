//! EVM stack depth limit AIR.
//!
//! Proves, per EVM step:
//!
//!   - `depth_post = depth_pre + push_count - pop_count` (signed delta)
//!   - `0 <= depth_post <= 1024` (the EVM 1024-item stack limit)
//!   - `(opcode, push_count, pop_count)` matches the canonical EVM
//!     stack-delta table, via cross-AIR LogUp to a constant opcode
//!     table.
//!
//! ## Why
//!
//! The EVM stack has a hard cap of 1024 items (each 256-bit). Per-
//! opcode push/pop counts are part of the EVM spec. A correct trace
//! must respect both invariants; this AIR commits a per-step witness
//! that algebraically checks the local delta and the depth-bound, and
//! defers the per-opcode count check to a cross-AIR LogUp against a
//! constant table (built inline as [`OPCODE_STACK_DELTA`]).
//!
//! ## Per-row layout
//!
//! ```text
//! offset  meaning
//!  0      pc                  (u64-shaped scalar)
//!  1      opcode              (u8)
//!  2      depth_pre           (u16)
//!  3      depth_post          (u16)
//!  4      push_count          (u8)
//!  5      pop_count           (u8)
//!  6      delta_pos           (u8; max(push-pop, 0))
//!  7      delta_neg           (u8; max(pop-push, 0))
//!  8      depth_post_lo       (low byte of depth_post)
//!  9      depth_post_hi       (high byte of depth_post)
//! 10      depth_slack         (1024 - depth_post, range-checked to u16)
//! 11      depth_slack_lo      (low byte of depth_slack)
//! 12      depth_slack_hi      (high byte of depth_slack)
//! 13      is_real             (binary; 1 on real steps, 0 on padding)
//! ```
//!
//! Total: **14 data columns**.
//!
//! ## Row-local constraints (11)
//!
//!  1. `is_real_binary`            — `v · (v - 1) = 0`
//!  2. `delta_mutual_exclusion`    — `delta_pos · delta_neg = 0`
//!  3. `delta_matches_counts`      — `v · (push_count - pop_count - delta_pos + delta_neg) = 0`
//!  4. `depth_post_eq_pre_plus_delta`
//!                                 — `v · (depth_post - depth_pre - delta_pos + delta_neg) = 0`
//!  5. `depth_post_byte_decomp`    — `depth_post = depth_post_lo + 256·depth_post_hi`
//!  6. `slack_plus_depth_eq_1024`  — `v · (depth_slack + depth_post - 1024) = 0`
//!  7. `slack_byte_decomp`         — `depth_slack = depth_slack_lo + 256·depth_slack_hi`
//!  8. `push_count_byte_decomp`    — `v · (push_count - push_count) = 0`  (range check via lookup)
//!  9. `pop_count_byte_decomp`     — `v · (pop_count  - pop_count)  = 0`  (range check via lookup)
//! 10. `delta_pos_byte_range`      — `v · delta_pos · (delta_pos - delta_pos) = 0`  (range via lookup)
//! 11. `delta_neg_byte_range`      — `v · delta_neg · (delta_neg - delta_neg) = 0`  (range via lookup)
//!
//! Constraints 8-11 are placeholders for the range-check side of the
//! soundness story — the actual byte-range checking is delegated to
//! the LogUp lookup table declared in [`lookup_declarations`]. They
//! are kept as algebraic zeros so the constraint count and label list
//! stay self-documenting.
//!
//! ## Shifted constraint (1)
//!
//!  S1. `depth_post(X) = depth_pre(ω·X)` gated by `is_real(X) · is_real(ω·X)`.
//!     Closes the cross-row chain: the next row's `depth_pre` must equal
//!     this row's `depth_post`.
//!
//! ## Cross-AIR LogUp
//!
//! [`make_stack_depth_to_opcode_table_descriptor`] binds the
//! `(opcode, push_count, pop_count)` tuple on `is_real=1` rows to a
//! constant 256-entry opcode table (column layout
//! `(opcode, push_count, pop_count)` on `is_real=1` rows of the
//! companion table AIR — caller wires the table at the named layer
//! index).

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_PC: usize = 0;
pub const COL_OPCODE: usize = 1;
pub const COL_DEPTH_PRE: usize = 2;
pub const COL_DEPTH_POST: usize = 3;
pub const COL_PUSH_COUNT: usize = 4;
pub const COL_POP_COUNT: usize = 5;
pub const COL_DELTA_POS: usize = 6;
pub const COL_DELTA_NEG: usize = 7;
pub const COL_DEPTH_POST_LO: usize = 8;
pub const COL_DEPTH_POST_HI: usize = 9;
pub const COL_DEPTH_SLACK: usize = 10;
pub const COL_DEPTH_SLACK_LO: usize = 11;
pub const COL_DEPTH_SLACK_HI: usize = 12;
pub const COL_IS_REAL: usize = 13;

pub const NUM_STACK_DEPTH_COLUMNS: usize = 14;

pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 1;

/// Hard EVM stack depth cap.
pub const MAX_STACK_DEPTH: u64 = 1024;

const _: () = assert!(NUM_STACK_DEPTH_COLUMNS == 14);

// ─── Canonical opcode → (push, pop) table ────────────────────────────
//
// Built inline as a const so the AIR is fully self-contained. For
// opcodes outside the spec, the entry is `(0, 0)` and is_real should
// be 0 on those rows (or the cross-AIR LogUp will mismatch).
//
// Convention: tuple is `(push_count, pop_count)` — i.e., items pushed
// then items popped. This is the order used in the witness.

/// `OPCODE_STACK_DELTA[op] = (push_count, pop_count)`.
pub const OPCODE_STACK_DELTA: [(u8, u8); 256] = {
    let mut t = [(0u8, 0u8); 256];
    // 0x00 STOP
    t[0x00] = (0, 0);
    // 0x01..=0x0B  ADD..SIGNEXTEND: pop 2 push 1
    let mut i = 0x01;
    while i <= 0x0B { t[i] = (1, 2); i += 1; }
    // 0x10..=0x1D  LT..SAR
    let mut i = 0x10;
    while i <= 0x1D { t[i] = (1, 2); i += 1; }
    // 0x20 KECCAK256
    t[0x20] = (1, 2);
    // 0x30 ADDRESS
    t[0x30] = (1, 0);
    // 0x31 BALANCE
    t[0x31] = (1, 1);
    // 0x32..=0x34 ORIGIN, CALLER, CALLVALUE
    t[0x32] = (1, 0);
    t[0x33] = (1, 0);
    t[0x34] = (1, 0);
    // 0x35 CALLDATALOAD
    t[0x35] = (1, 1);
    // 0x36 CALLDATASIZE
    t[0x36] = (1, 0);
    // 0x37 CALLDATACOPY
    t[0x37] = (0, 3);
    // 0x38 CODESIZE
    t[0x38] = (1, 0);
    // 0x39 CODECOPY
    t[0x39] = (0, 3);
    // 0x3A GASPRICE
    t[0x3A] = (1, 0);
    // 0x3B EXTCODESIZE
    t[0x3B] = (1, 1);
    // 0x3C EXTCODECOPY
    t[0x3C] = (0, 4);
    // 0x3D RETURNDATASIZE
    t[0x3D] = (1, 0);
    // 0x3E RETURNDATACOPY
    t[0x3E] = (0, 3);
    // 0x3F EXTCODEHASH
    t[0x3F] = (1, 1);
    // 0x40 BLOCKHASH
    t[0x40] = (1, 1);
    // 0x41..=0x48 COINBASE..BASEFEE
    let mut i = 0x41;
    while i <= 0x48 { t[i] = (1, 0); i += 1; }
    // 0x50 POP
    t[0x50] = (0, 1);
    // 0x51 MLOAD
    t[0x51] = (1, 1);
    // 0x52 MSTORE
    t[0x52] = (0, 2);
    // 0x53 MSTORE8
    t[0x53] = (0, 2);
    // 0x54 SLOAD
    t[0x54] = (1, 1);
    // 0x55 SSTORE
    t[0x55] = (0, 2);
    // 0x56 JUMP
    t[0x56] = (0, 1);
    // 0x57 JUMPI
    t[0x57] = (0, 2);
    // 0x58 PC
    t[0x58] = (1, 0);
    // 0x59 MSIZE
    t[0x59] = (1, 0);
    // 0x5A GAS
    t[0x5A] = (1, 0);
    // 0x5B JUMPDEST
    t[0x5B] = (0, 0);
    // 0x5F PUSH0
    t[0x5F] = (1, 0);
    // 0x60..=0x7F PUSH1..PUSH32
    let mut i = 0x60;
    while i <= 0x7F { t[i] = (1, 0); i += 1; }
    // 0x80..=0x8F DUPn — pops n, pushes n+1
    let mut n: u8 = 1;
    while n <= 16 { t[0x80 + (n as usize) - 1] = (n + 1, n); n += 1; }
    // 0x90..=0x9F SWAPn — pops n+1, pushes n+1 (net 0)
    let mut n: u8 = 1;
    while n <= 16 { t[0x90 + (n as usize) - 1] = (n + 1, n + 1); n += 1; }
    // 0xA0..=0xA4 LOG0..LOG4
    t[0xA0] = (0, 2);
    t[0xA1] = (0, 3);
    t[0xA2] = (0, 4);
    t[0xA3] = (0, 5);
    t[0xA4] = (0, 6);
    // 0xF0 CREATE
    t[0xF0] = (1, 3);
    // 0xF1 CALL
    t[0xF1] = (1, 7);
    // 0xF3 RETURN
    t[0xF3] = (0, 2);
    // 0xF5 CREATE2
    t[0xF5] = (1, 4);
    // 0xFA STATICCALL
    t[0xFA] = (1, 6);
    // 0xFD REVERT
    t[0xFD] = (0, 2);
    // 0xFE INVALID
    t[0xFE] = (0, 0);
    // 0xFF SELFDESTRUCT
    t[0xFF] = (0, 1);
    t
};

/// Returns `(push_count, pop_count)` for an opcode from the canonical
/// table. Unknown / unallocated opcodes return `(0, 0)`.
pub const fn opcode_delta(opcode: u8) -> (u8, u8) {
    OPCODE_STACK_DELTA[opcode as usize]
}

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackDepthRow {
    pub pc: u64,
    pub opcode: u8,
    pub depth_pre: u16,
    pub depth_post: u16,
    pub push_count: u8,
    pub pop_count: u8,
}

#[derive(Clone, Debug, Default)]
pub struct StackDepthWitness {
    pub rows: Vec<StackDepthRow>,
}

impl StackDepthWitness {
    /// Build a witness from a list of `(pc, opcode, depth_pre)`
    /// inspector events. The push/pop counts and depth_post are
    /// derived from the canonical [`OPCODE_STACK_DELTA`] table.
    ///
    /// Returns the witness directly; the per-step `depth_post` for
    /// row `i` is computed as
    /// `depth_pre + push_count - pop_count`. Saturating semantics are
    /// used to keep depth in the u16 range even on adversarial
    /// inputs; the AIR's algebraic constraints catch any mismatch.
    pub fn from_inspector_events(events: &[(u64, u8, u16)]) -> Self {
        let mut rows = Vec::with_capacity(events.len());
        for &(pc, opcode, depth_pre) in events {
            let (push, pop) = opcode_delta(opcode);
            let depth_post = (depth_pre as i32 + push as i32 - pop as i32).max(0) as u32;
            let depth_post = depth_post.min(u16::MAX as u32) as u16;
            rows.push(StackDepthRow {
                pc,
                opcode,
                depth_pre,
                depth_post,
                push_count: push,
                pop_count: pop,
            });
        }
        Self { rows }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &StackDepthWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_STACK_DEPTH_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (i, r) in witness.rows.iter().enumerate() {
        columns[COL_PC][i] = Scalar::from_u64(r.pc, curve);
        columns[COL_OPCODE][i] = Scalar::from_u64(r.opcode as u64, curve);
        columns[COL_DEPTH_PRE][i] = Scalar::from_u64(r.depth_pre as u64, curve);
        columns[COL_DEPTH_POST][i] = Scalar::from_u64(r.depth_post as u64, curve);
        columns[COL_PUSH_COUNT][i] = Scalar::from_u64(r.push_count as u64, curve);
        columns[COL_POP_COUNT][i] = Scalar::from_u64(r.pop_count as u64, curve);
        // Signed delta encoded as (delta_pos, delta_neg).
        let push = r.push_count as i32;
        let pop = r.pop_count as i32;
        let delta = push - pop;
        let (dp, dn) = if delta >= 0 { (delta as u64, 0u64) } else { (0u64, (-delta) as u64) };
        columns[COL_DELTA_POS][i] = Scalar::from_u64(dp, curve);
        columns[COL_DELTA_NEG][i] = Scalar::from_u64(dn, curve);
        // Byte decomp of depth_post.
        let dp_lo = (r.depth_post & 0xFF) as u64;
        let dp_hi = ((r.depth_post >> 8) & 0xFF) as u64;
        columns[COL_DEPTH_POST_LO][i] = Scalar::from_u64(dp_lo, curve);
        columns[COL_DEPTH_POST_HI][i] = Scalar::from_u64(dp_hi, curve);
        // Slack and its byte decomp.
        let slack = MAX_STACK_DEPTH.saturating_sub(r.depth_post as u64);
        let s_lo = slack & 0xFF;
        let s_hi = (slack >> 8) & 0xFF;
        columns[COL_DEPTH_SLACK][i] = Scalar::from_u64(slack, curve);
        columns[COL_DEPTH_SLACK_LO][i] = Scalar::from_u64(s_lo, curve);
        columns[COL_DEPTH_SLACK_HI][i] = Scalar::from_u64(s_hi, curve);
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

pub struct StackDepthConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl StackDepthConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn s256(curve: CurveType) -> Scalar { Scalar::from_u64(256, curve) }
fn s1024(curve: CurveType) -> Scalar { Scalar::from_u64(MAX_STACK_DEPTH, curve) }

impl VmConstraintSystem for StackDepthConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "delta_mutual_exclusion".into(),
            "delta_matches_counts".into(),
            "depth_post_eq_pre_plus_delta".into(),
            "depth_post_byte_decomp".into(),
            "slack_plus_depth_eq_1024".into(),
            "slack_byte_decomp".into(),
            "push_count_range_placeholder".into(),
            "pop_count_range_placeholder".into(),
            "delta_pos_range_placeholder".into(),
            "delta_neg_range_placeholder".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_STACK_DEPTH_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![zero.clone(); n])
            .collect();
        for row in 0..n {
            let v = &columns[COL_IS_REAL][row];
            let dpre = &columns[COL_DEPTH_PRE][row];
            let dpost = &columns[COL_DEPTH_POST][row];
            let push = &columns[COL_PUSH_COUNT][row];
            let pop = &columns[COL_POP_COUNT][row];
            let dp = &columns[COL_DELTA_POS][row];
            let dn = &columns[COL_DELTA_NEG][row];
            let dp_lo = &columns[COL_DEPTH_POST_LO][row];
            let dp_hi = &columns[COL_DEPTH_POST_HI][row];
            let slack = &columns[COL_DEPTH_SLACK][row];
            let s_lo = &columns[COL_DEPTH_SLACK_LO][row];
            let s_hi = &columns[COL_DEPTH_SLACK_HI][row];

            // 1. is_real binary
            bodies[0][row] = v.mul(&v.sub(&one));
            // 2. delta_pos · delta_neg = 0
            bodies[1][row] = dp.mul(dn);
            // 3. push - pop - delta_pos + delta_neg = 0  (gated by is_real)
            bodies[2][row] = v.mul(&push.sub(pop).sub(dp).add(dn));
            // 4. depth_post - depth_pre - delta_pos + delta_neg = 0 (gated)
            bodies[3][row] = v.mul(&dpost.sub(dpre).sub(dp).add(dn));
            // 5. depth_post = lo + 256·hi (unconditional; padding rows have all-zero)
            bodies[4][row] = dpost.sub(&dp_lo.add(&dp_hi.mul(&s256(curve))));
            // 6. slack + depth_post = 1024 (gated by is_real)
            bodies[5][row] = v.mul(&slack.add(dpost).sub(&s1024(curve)));
            // 7. slack = lo + 256·hi
            bodies[6][row] = slack.sub(&s_lo.add(&s_hi.mul(&s256(curve))));
            // 8-11. Range-check placeholders (actual range checks via lookup).
            bodies[7][row] = zero.clone();
            bodies[8][row] = zero.clone();
            bodies[9][row] = zero.clone();
            bodies[10][row] = zero.clone();
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_STACK_DEPTH_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        let v = &col_evals[COL_IS_REAL];
        let dpre = &col_evals[COL_DEPTH_PRE];
        let dpost = &col_evals[COL_DEPTH_POST];
        let push = &col_evals[COL_PUSH_COUNT];
        let pop = &col_evals[COL_POP_COUNT];
        let dp = &col_evals[COL_DELTA_POS];
        let dn = &col_evals[COL_DELTA_NEG];
        let dp_lo = &col_evals[COL_DEPTH_POST_LO];
        let dp_hi = &col_evals[COL_DEPTH_POST_HI];
        let slack = &col_evals[COL_DEPTH_SLACK];
        let s_lo = &col_evals[COL_DEPTH_SLACK_LO];
        let s_hi = &col_evals[COL_DEPTH_SLACK_HI];
        let s256v = s256(curve);
        let s1024v = s1024(curve);

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            v.mul(&v.sub(&one)),
            dp.mul(dn),
            v.mul(&push.sub(pop).sub(dp).add(dn)),
            v.mul(&dpost.sub(dpre).sub(dp).add(dn)),
            dpost.sub(&dp_lo.add(&dp_hi.mul(&s256v))),
            v.mul(&slack.add(dpost).sub(&s1024v)),
            slack.sub(&s_lo.add(&s_hi.mul(&s256v))),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
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
        let one_p = vec![Scalar::one(curve)];
        let zero_p = vec![Scalar::zero(curve)];
        let s256_p = vec![s256(curve)];
        let s1024_p = vec![s1024(curve)];

        let v = &col_coeffs[COL_IS_REAL];
        let dpre = &col_coeffs[COL_DEPTH_PRE];
        let dpost = &col_coeffs[COL_DEPTH_POST];
        let push = &col_coeffs[COL_PUSH_COUNT];
        let pop = &col_coeffs[COL_POP_COUNT];
        let dp = &col_coeffs[COL_DELTA_POS];
        let dn = &col_coeffs[COL_DELTA_NEG];
        let dp_lo = &col_coeffs[COL_DEPTH_POST_LO];
        let dp_hi = &col_coeffs[COL_DEPTH_POST_HI];
        let slack = &col_coeffs[COL_DEPTH_SLACK];
        let s_lo = &col_coeffs[COL_DEPTH_SLACK_LO];
        let s_hi = &col_coeffs[COL_DEPTH_SLACK_HI];

        // 1. v · (v - 1)
        let v_m1 = poly_sub(v, &one_p, curve);
        let b1 = poly_mul(v, &v_m1, curve);
        // 2. dp · dn
        let b2 = poly_mul(dp, dn, curve);
        // 3. v · (push - pop - dp + dn)
        let mut diff = poly_sub(push, pop, curve);
        diff = poly_sub(&diff, dp, curve);
        diff = poly_add(&diff, dn, curve);
        let b3 = poly_mul(v, &diff, curve);
        // 4. v · (dpost - dpre - dp + dn)
        let mut diff4 = poly_sub(dpost, dpre, curve);
        diff4 = poly_sub(&diff4, dp, curve);
        diff4 = poly_add(&diff4, dn, curve);
        let b4 = poly_mul(v, &diff4, curve);
        // 5. dpost - (lo + 256·hi)
        let hi_scaled = poly_mul(dp_hi, &s256_p, curve);
        let sum56 = poly_add(dp_lo, &hi_scaled, curve);
        let b5 = poly_sub(dpost, &sum56, curve);
        // 6. v · (slack + dpost - 1024)
        let sum6 = poly_add(slack, dpost, curve);
        let body6 = poly_sub(&sum6, &s1024_p, curve);
        let b6 = poly_mul(v, &body6, curve);
        // 7. slack - (s_lo + 256·s_hi)
        let s_hi_scaled = poly_mul(s_hi, &s256_p, curve);
        let sum78 = poly_add(s_lo, &s_hi_scaled, curve);
        let b7 = poly_sub(slack, &sum78, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            b1, b2, b3, b4, b5, b6, b7,
            zero_p.clone(), zero_p.clone(), zero_p.clone(), zero_p,
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

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_STACK_DEPTH_COLUMNS { return; }
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);
        // On padding rows: set everything to 0, except depth_slack = 1024
        // so that the unconditional slack-byte-decomp constraint (7)
        // would fail unless we also set the byte decomp consistently.
        // The cleanest approach is to set depth_slack=0, depth_post=0,
        // and let the byte decomps be 0 — constraints 5 and 7 still
        // hold (0 = 0 + 256·0); constraint 6 is gated by is_real=0.
        for c in columns.iter_mut().take(NUM_STACK_DEPTH_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let tbl = 0usize;
        let decls = vec![
            (LookupDeclaration {
                label: "stack_depth_opcode_8bit".into(),
                column_index: COL_OPCODE,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_push_count_8bit".into(),
                column_index: COL_PUSH_COUNT,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_pop_count_8bit".into(),
                column_index: COL_POP_COUNT,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_delta_pos_8bit".into(),
                column_index: COL_DELTA_POS,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_delta_neg_8bit".into(),
                column_index: COL_DELTA_NEG,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_post_lo_8bit".into(),
                column_index: COL_DEPTH_POST_LO,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_post_hi_8bit".into(),
                column_index: COL_DEPTH_POST_HI,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_slack_lo_8bit".into(),
                column_index: COL_DEPTH_SLACK_LO,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "stack_depth_slack_hi_8bit".into(),
                column_index: COL_DEPTH_SLACK_HI,
                max_bits: 8,
                selector_column: None,
            }, tbl),
        ];
        LookupRequirements { tables, declarations: decls }
    }

    // ─── Shifted constraint ──────────────────────────────────────────
    // depth_post(X) = depth_pre(ω·X), gated by is_real(X) · is_real(ω·X).

    fn shifted_column_indices(&self) -> Vec<usize> {
        // shifted[0] = IS_REAL_next, shifted[1] = DEPTH_PRE_next
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
        if shifted_evals.len() < 2 || col_evals_at_z.len() < NUM_STACK_DEPTH_COLUMNS {
            return Scalar::zero(curve);
        }
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let dpre_next = &shifted_evals[1];
        let dpost = &col_evals_at_z[COL_DEPTH_POST];
        let gate = v.mul(v_next);
        let body = dpost.sub(dpre_next);
        let row = gate.mul(&body);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
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
        let dpre = &column_coeffs[COL_DEPTH_PRE];
        let dpre_next = poly_shift(dpre, omega);
        let dpost = &column_coeffs[COL_DEPTH_POST];
        let gate = poly_mul(v, &v_next, curve);
        let body = poly_sub(dpost, &dpre_next, curve);
        let row = poly_mul(&gate, &body, curve);
        // Multiply by (X − ω^{n−1}) so the prover's committed polynomial
        // matches the verifier's `evaluate_shifted_at_point`, which
        // multiplies the row body by `(z − ω^{n−1})`. Without this
        // factor the prover-side polynomial is short by one degree-1
        // factor and the quotient identity Q(z)·Z(z) = C(z) fails on
        // the wrap row, even on honest witnesses. (Task #256/#268.)
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
        let row_excluded = poly_mul_linear(&row, &omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        poly_scalar_mul(&row_excluded, &ap)
    }
}

// ─── Cross-AIR LogUp linkage descriptor ───────────────────────────────

/// Binds this AIR's `(opcode, push_count, pop_count)` triple on
/// `is_real=1` rows to a companion opcode-table AIR's
/// `(opcode, push_count, pop_count)` triple on `is_real=1` rows.
///
/// The companion table is expected to lay out columns in the same
/// `(opcode, push_count, pop_count)` order with `is_real` as its
/// selector; the caller (e.g. gas_tracking_air-style table layer)
/// wires it.
pub fn make_stack_depth_to_opcode_table_descriptor(
    stack_depth_layer_index: usize,
    table_layer_index: usize,
    table_opcode_col: usize,
    table_push_col: usize,
    table_pop_col: usize,
    table_is_real_col: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "stack_depth_to_opcode_table_v1".into(),
        a_layer_index: stack_depth_layer_index,
        a_columns: vec![COL_OPCODE, COL_PUSH_COUNT, COL_POP_COUNT],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: table_layer_index,
        b_columns: vec![table_opcode_col, table_push_col, table_pop_col],
        b_selector_column: Some(table_is_real_col),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opcodes_sequence_push_chain() -> Vec<(u64, u8, u16)> {
        // 4× PUSH1: depth 0→1→2→3→4
        vec![
            (0, 0x60, 0),
            (2, 0x60, 1),
            (4, 0x60, 2),
            (6, 0x60, 3),
        ]
    }

    #[test]
    fn push_chain_witness_consistent() {
        let evs = opcodes_sequence_push_chain();
        let w = StackDepthWitness::from_inspector_events(&evs);
        assert_eq!(w.rows.len(), 4);
        assert_eq!(w.rows[0].depth_pre, 0);
        assert_eq!(w.rows[0].depth_post, 1);
        assert_eq!(w.rows[3].depth_pre, 3);
        assert_eq!(w.rows[3].depth_post, 4);
        // delta is purely positive
        for r in &w.rows {
            assert_eq!(r.push_count, 1);
            assert_eq!(r.pop_count, 0);
        }
    }

    #[test]
    fn pop_chain_witness_consistent() {
        // depth_pre starts at 4, sequence of POPs draining to 0
        let evs = vec![
            (0, 0x50u8, 4u16),
            (1, 0x50u8, 3u16),
            (2, 0x50u8, 2u16),
            (3, 0x50u8, 1u16),
        ];
        let w = StackDepthWitness::from_inspector_events(&evs);
        for (i, r) in w.rows.iter().enumerate() {
            assert_eq!(r.depth_pre, (4 - i) as u16);
            assert_eq!(r.depth_post, (3 - i) as u16);
            assert_eq!(r.push_count, 0);
            assert_eq!(r.pop_count, 1);
        }
    }

    #[test]
    fn mixed_program_witness_consistent_and_constraints_vanish() {
        // PUSH1; PUSH1; ADD; POP; STOP
        let evs = vec![
            (0, 0x60u8, 0u16),  // PUSH1: 0 -> 1
            (2, 0x60u8, 1u16),  // PUSH1: 1 -> 2
            (4, 0x01u8, 2u16),  // ADD:   2 -> 1
            (5, 0x50u8, 1u16),  // POP:   1 -> 0
            (6, 0x00u8, 0u16),  // STOP:  0 -> 0
        ];
        let w = StackDepthWitness::from_inspector_events(&evs);
        assert_eq!(w.rows[2].depth_post, 1);
        assert_eq!(w.rows[3].depth_post, 0);
        let curve = CurveType::Bls48581;
        let t = build_trace_polynomials(&w, curve);
        let cs = StackDepthConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        for row in 0..t.padded_size as usize {
            let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let v = cs.evaluate_at_point(&cols, &alpha);
            assert!(v.is_zero(), "row {} body must vanish", row);
        }
    }

    #[test]
    fn depth_1024_boundary_witness_and_constraints() {
        // depth_pre = 1023, PUSH1 -> 1024 (the max).
        let evs = vec![(0u64, 0x60u8, 1023u16)];
        let w = StackDepthWitness::from_inspector_events(&evs);
        assert_eq!(w.rows[0].depth_post, 1024);
        let curve = CurveType::Bls48581;
        let t = build_trace_polynomials(&w, curve);
        let cs = StackDepthConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(5, curve);
        for row in 0..t.padded_size as usize {
            let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let v = cs.evaluate_at_point(&cols, &alpha);
            assert!(v.is_zero(), "row {} body must vanish at boundary", row);
        }
        // And verify slack is 0 at the boundary.
        assert!(t.columns[COL_DEPTH_SLACK].evaluations[0].is_zero());
    }

    #[test]
    fn tampered_delta_detected() {
        // Honest PUSH1 row, then tamper depth_post to be wrong.
        let evs = vec![(0u64, 0x60u8, 5u16)];
        let w = StackDepthWitness::from_inspector_events(&evs);
        let curve = CurveType::Bls48581;
        let mut t = build_trace_polynomials(&w, curve);
        let cs = StackDepthConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(11, curve);
        // Original: depth_post = 6. Tamper to 7 (and keep byte decomp
        // consistent so constraint 5 still passes — that way only the
        // delta constraint #4 catches the lie).
        t.columns[COL_DEPTH_POST].evaluations[0] = Scalar::from_u64(7, curve);
        t.columns[COL_DEPTH_POST_LO].evaluations[0] = Scalar::from_u64(7, curve);
        t.columns[COL_DEPTH_POST_HI].evaluations[0] = Scalar::from_u64(0, curve);
        let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let v = cs.evaluate_at_point(&cols, &alpha);
        assert!(!v.is_zero(), "tampered depth_post must violate body");
    }

    #[test]
    fn tampered_overflow_detected_by_slack() {
        // depth_pre = 1023 with PUSH1 → honest depth_post = 1024. Tamper
        // to claim depth_post=1025 while keeping byte decomp consistent;
        // the slack constraint requires slack = 1024 - depth_post = -1
        // (field), which won't byte-decompose. We force slack consistent
        // with the tamper via a non-byte-decomposable value to show the
        // slack chain catches violations.
        let evs = vec![(0u64, 0x60u8, 1023u16)];
        let w = StackDepthWitness::from_inspector_events(&evs);
        let curve = CurveType::Bls48581;
        let mut t = build_trace_polynomials(&w, curve);
        let cs = StackDepthConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(3, curve);
        // Tamper depth_post 1024 → 1025 and update its byte decomp so
        // constraint 5 still vanishes. Then constraint 6 (slack+post=1024)
        // is broken because we haven't touched slack.
        t.columns[COL_DEPTH_POST].evaluations[0] = Scalar::from_u64(1025, curve);
        t.columns[COL_DEPTH_POST_LO].evaluations[0] = Scalar::from_u64(1, curve);
        t.columns[COL_DEPTH_POST_HI].evaluations[0] = Scalar::from_u64(4, curve);
        let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let v = cs.evaluate_at_point(&cols, &alpha);
        assert!(!v.is_zero(), "depth_post > 1024 must violate slack body");
    }

    #[test]
    fn opcode_delta_table_spot_checks() {
        // Spec-driven spot checks for OPCODE_STACK_DELTA.
        assert_eq!(opcode_delta(0x00), (0, 0));        // STOP
        assert_eq!(opcode_delta(0x01), (1, 2));        // ADD
        assert_eq!(opcode_delta(0x50), (0, 1));        // POP
        assert_eq!(opcode_delta(0x5B), (0, 0));        // JUMPDEST
        assert_eq!(opcode_delta(0x5F), (1, 0));        // PUSH0
        assert_eq!(opcode_delta(0x60), (1, 0));        // PUSH1
        assert_eq!(opcode_delta(0x7F), (1, 0));        // PUSH32
        assert_eq!(opcode_delta(0x80), (2, 1));        // DUP1: pop 1, push 2
        assert_eq!(opcode_delta(0x8F), (17, 16));      // DUP16
        assert_eq!(opcode_delta(0x90), (2, 2));        // SWAP1: pop 2, push 2
        assert_eq!(opcode_delta(0x9F), (17, 17));      // SWAP16
        assert_eq!(opcode_delta(0xA2), (0, 4));        // LOG2: pop 4
        assert_eq!(opcode_delta(0xF1), (1, 7));        // CALL: pop 7 push 1
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_stack_depth_to_opcode_table_descriptor(0, 1, 0, 1, 2, 3);
        assert_eq!(d.label, "stack_depth_to_opcode_table_v1");
        assert_eq!(d.a_columns, vec![COL_OPCODE, COL_PUSH_COUNT, COL_POP_COUNT]);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.b_columns, vec![0, 1, 2]);
        assert_eq!(d.b_selector_column, Some(3));
    }

    #[test]
    fn shifted_constraint_continuity() {
        // Two real rows whose depth_post[r] = depth_pre[r+1] should
        // satisfy the shifted constraint at the boundary z = ω^0
        // (i.e., we approximate by checking the witness values are
        // consistent — the algebraic check is exercised by the full
        // prove pipeline in the slow test).
        let evs = vec![
            (0u64, 0x60u8, 0u16),  // PUSH1: 0->1
            (2u64, 0x60u8, 1u16),  // PUSH1: 1->2
        ];
        let w = StackDepthWitness::from_inspector_events(&evs);
        for i in 0..(w.rows.len() - 1) {
            assert_eq!(w.rows[i].depth_post, w.rows[i + 1].depth_pre);
        }
    }
}
