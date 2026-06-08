//! PUSH0 opcode AIR (EIP-3855).
//!
//! Proves algebraically that on a PUSH0 row (opcode 0x5F):
//!
//!   - The opcode only fires post-Shanghai (gated by `is_post_shanghai`
//!     ↔ `hardfork_rules_air`).
//!   - The value pushed onto the stack is the u256 zero (all 4 limbs are
//!     zero) — bound to `stack_contents_air`'s unsorted view via cross-AIR
//!     LogUp at `(pc, value_limb_0..3)`.
//!   - The static gas cost is `G_BASE = 2`.
//!
//! ## Per-row layout (`NUM_COLUMNS = 9`)
//!
//! ```text
//! offset  meaning
//!  0      pc                  (u64; program counter of the PUSH0 event)
//!  1      value_limb_0        (u64; LE limb 0 of u256 — MUST be 0)
//!  2      value_limb_1        (u64; LE limb 1 — MUST be 0)
//!  3      value_limb_2        (u64; LE limb 2 — MUST be 0)
//!  4      value_limb_3        (u64; LE limb 3 — MUST be 0)
//!  5      gas_cost            (u64; MUST be 2 on real rows)
//!  6      is_post_shanghai    (bin; MUST be 1 on real rows)
//!  7      is_real             (bin)
//!  8      opcode              (u8; pinned to 0x5F for stack-contents binding)
//! ```
//!
//! ## Constraint catalog (10 row-local constraints)
//!
//!  0. `is_real_binary`               — `is_real · (is_real − 1) = 0`
//!  1. `is_post_shanghai_binary`      — `s · (s − 1) = 0`
//!  2. `push0_post_shanghai_only`     — `is_real · (1 − is_post_shanghai) = 0`
//!  3. `gas_cost_eq_2`                — `is_real · (gas_cost − 2) = 0`
//!  4. `value_limb_0_zero`            — `is_real · value_limb_0 = 0`
//!  5. `value_limb_1_zero`            — `is_real · value_limb_1 = 0`
//!  6. `value_limb_2_zero`            — `is_real · value_limb_2 = 0`
//!  7. `value_limb_3_zero`            — `is_real · value_limb_3 = 0`
//!  8. `opcode_eq_push0`              — `is_real · (opcode − 0x5F) = 0`
//!  9. `is_real_zeroes_value`         — (redundant on its own) padding rows
//!                                       are forced zero by `fix_trace_padding`;
//!                                       constraint redundantly checks
//!                                       `(1 − is_real) · opcode = 0` to pin
//!                                       padding rows to a clean zero state.
//!
//! Plus byte range lookups (`max_bits = 1`) on the binary selectors and an
//! 8-bit range check on `opcode`.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PC: usize = 0;
pub const COL_VALUE_LIMB_0: usize = 1;
pub const COL_VALUE_LIMB_1: usize = 2;
pub const COL_VALUE_LIMB_2: usize = 3;
pub const COL_VALUE_LIMB_3: usize = 4;
pub const COL_GAS_COST: usize = 5;
pub const COL_IS_POST_SHANGHAI: usize = 6;
pub const COL_IS_REAL: usize = 7;
pub const COL_OPCODE: usize = 8;

pub const NUM_COLUMNS: usize = 9;
pub const NUM_ROW_CONSTRAINTS: usize = 10;
pub const NUM_SHIFTED: usize = 0;

const _: () = assert!(NUM_COLUMNS == 9);

/// EIP-3855: PUSH0 opcode byte.
pub const PUSH0_OPCODE: u8 = 0x5F;
/// EIP-3855: PUSH0 static gas cost (`G_base`).
pub const PUSH0_GAS_COST: u64 = 2;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Push0Row {
    pub pc: u64,
    /// LE u64 limbs of the pushed u256 — must be `[0, 0, 0, 0]`.
    pub value: [u64; 4],
    pub gas_cost: u64,
    pub is_post_shanghai: bool,
    pub is_real: bool,
    pub opcode: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Push0Witness {
    pub rows: Vec<Push0Row>,
}

impl Push0Witness {
    /// Build a witness from a slice of `(pc, is_post_shanghai)` events.
    /// Each event populates a row with `value = 0`, `gas_cost = 2`,
    /// `opcode = 0x5F`, `is_real = true`.
    pub fn from_events(events: &[(u64, bool)]) -> Self {
        let rows = events
            .iter()
            .map(|&(pc, is_post_shanghai)| Push0Row {
                pc,
                value: [0u64; 4],
                gas_cost: PUSH0_GAS_COST,
                is_post_shanghai,
                is_real: true,
                opcode: PUSH0_OPCODE,
            })
            .collect();
        Self { rows }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(w: &Push0Witness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_PC][r] = Scalar::from_u64(row.pc, curve);
        cols[COL_VALUE_LIMB_0][r] = Scalar::from_u64(row.value[0], curve);
        cols[COL_VALUE_LIMB_1][r] = Scalar::from_u64(row.value[1], curve);
        cols[COL_VALUE_LIMB_2][r] = Scalar::from_u64(row.value[2], curve);
        cols[COL_VALUE_LIMB_3][r] = Scalar::from_u64(row.value[3], curve);
        cols[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        cols[COL_IS_POST_SHANGHAI][r] =
            if row.is_post_shanghai { one.clone() } else { zero.clone() };
        cols[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };
        cols[COL_OPCODE][r] = Scalar::from_u64(row.opcode as u64, curve);
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct Push0ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Push0ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Push0ConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_post_shanghai_binary".into(),
            "push0_post_shanghai_only".into(),
            "gas_cost_eq_2".into(),
            "value_limb_0_zero".into(),
            "value_limb_1_zero".into(),
            "value_limb_2_zero".into(),
            "value_limb_3_zero".into(),
            "opcode_eq_push0".into(),
            "padding_opcode_zero".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(PUSH0_GAS_COST, curve);
        let push0 = Scalar::from_u64(PUSH0_OPCODE as u64, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let s = &columns[COL_IS_POST_SHANGHAI][r];
            let v0 = &columns[COL_VALUE_LIMB_0][r];
            let v1 = &columns[COL_VALUE_LIMB_1][r];
            let v2 = &columns[COL_VALUE_LIMB_2][r];
            let v3 = &columns[COL_VALUE_LIMB_3][r];
            let gc = &columns[COL_GAS_COST][r];
            let op = &columns[COL_OPCODE][r];
            let one_minus_real = one.sub(is_real);
            let one_minus_s = one.sub(s);
            // 0: is_real binary
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1: is_post_shanghai binary
            bodies[1][r] = s.mul(&s.sub(&one));
            // 2: PUSH0 only post-Shanghai
            bodies[2][r] = is_real.mul(&one_minus_s);
            // 3: gas_cost == 2
            bodies[3][r] = is_real.mul(&gc.sub(&two));
            // 4..7: value_limb_k == 0
            bodies[4][r] = is_real.mul(v0);
            bodies[5][r] = is_real.mul(v1);
            bodies[6][r] = is_real.mul(v2);
            bodies[7][r] = is_real.mul(v3);
            // 8: opcode == 0x5F
            bodies[8][r] = is_real.mul(&op.sub(&push0));
            // 9: padding rows have opcode = 0
            bodies[9][r] = one_minus_real.mul(op);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(PUSH0_GAS_COST, curve);
        let push0 = Scalar::from_u64(PUSH0_OPCODE as u64, curve);
        let is_real = &ce[COL_IS_REAL];
        let s = &ce[COL_IS_POST_SHANGHAI];
        let v0 = &ce[COL_VALUE_LIMB_0];
        let v1 = &ce[COL_VALUE_LIMB_1];
        let v2 = &ce[COL_VALUE_LIMB_2];
        let v3 = &ce[COL_VALUE_LIMB_3];
        let gc = &ce[COL_GAS_COST];
        let op = &ce[COL_OPCODE];
        let one_minus_real = one.sub(is_real);
        let one_minus_s = one.sub(s);
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            s.mul(&s.sub(&one)),
            is_real.mul(&one_minus_s),
            is_real.mul(&gc.sub(&two)),
            is_real.mul(v0),
            is_real.mul(v1),
            is_real.mul(v2),
            is_real.mul(v3),
            is_real.mul(&op.sub(&push0)),
            one_minus_real.mul(op),
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
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let two_p = vec![Scalar::from_u64(PUSH0_GAS_COST, curve)];
        let push0_p = vec![Scalar::from_u64(PUSH0_OPCODE as u64, curve)];
        let is_real = &cc[COL_IS_REAL];
        let s = &cc[COL_IS_POST_SHANGHAI];
        let v0 = &cc[COL_VALUE_LIMB_0];
        let v1 = &cc[COL_VALUE_LIMB_1];
        let v2 = &cc[COL_VALUE_LIMB_2];
        let v3 = &cc[COL_VALUE_LIMB_3];
        let gc = &cc[COL_GAS_COST];
        let op = &cc[COL_OPCODE];
        let one_minus_real = poly_sub(&one_p, is_real, curve);
        let one_minus_s = poly_sub(&one_p, s, curve);
        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(is_real, &poly_sub(is_real, &one_p, curve), curve),
            poly_mul(s, &poly_sub(s, &one_p, curve), curve),
            poly_mul(is_real, &one_minus_s, curve),
            poly_mul(is_real, &poly_sub(gc, &two_p, curve), curve),
            poly_mul(is_real, v0, curve),
            poly_mul(is_real, v1, curve),
            poly_mul(is_real, v2, curve),
            poly_mul(is_real, v3, curve),
            poly_mul(is_real, &poly_sub(op, &push0_p, curve), curve),
            poly_mul(&one_minus_real, op, curve),
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
        vec![COL_IS_REAL, COL_IS_POST_SHANGHAI]
    }
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

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(1), LookupTable::range(8)];
        let tbl_bit = 0usize;
        let tbl_byte = 1usize;
        let mut declarations = Vec::new();
        for (col, label) in [
            (COL_IS_REAL, "push0_is_real_1bit"),
            (COL_IS_POST_SHANGHAI, "push0_is_post_shanghai_1bit"),
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
        declarations.push((
            LookupDeclaration {
                label: "push0_opcode_8bit".into(),
                column_index: COL_OPCODE,
                max_bits: 8,
                selector_column: None,
            },
            tbl_byte,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkage descriptors ──────────────────────────────

/// PUSH0 AIR → `hardfork_rules_air`. Binds the gadget's `is_post_shanghai`
/// flag on real rows to the hardfork-rules AIR's `is_post_shanghai`
/// column. Combined with constraint 2 (`push0_post_shanghai_only`), this
/// algebraically forbids PUSH0 from firing pre-Shanghai: the gadget's
/// flag must be 1 on every real row, and the LogUp join forces that
/// value to come from a real hardfork-rules row (where the timestamp
/// gate has already pinned `is_post_shanghai = 1` ⇔ `ts ≥ SHANGHAI_TIMESTAMP`).
pub fn make_push0_to_hardfork_descriptor(
    push0_layer_index: usize,
    hardfork_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "push0_to_hardfork_shanghai_v1".into(),
        a_layer_index: push0_layer_index,
        a_columns: vec![COL_IS_POST_SHANGHAI],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: hardfork_layer_index,
        b_columns: vec![crate::hardfork_rules_air::COL_IS_POST_SHANGHAI],
        b_selector_column: Some(crate::hardfork_rules_air::COL_IS_REAL),
    }
}

/// PUSH0 AIR → `stack_contents_air` (unsorted view). Binds the gadget's
/// `(pc, value_limb_0..3)` tuple on real rows to the stack-contents AIR's
/// unsorted view `(pc, value_limb_0..3)`. Combined with constraints 4..7
/// (each value limb forced to 0), this proves that the stack write
/// committed in stack-contents at this PC has value = u256(0).
pub fn make_push0_to_stack_contents_descriptor(
    push0_layer_index: usize,
    stack_contents_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "push0_to_stack_contents_value_zero_v1".into(),
        a_layer_index: push0_layer_index,
        a_columns: vec![
            COL_PC,
            COL_VALUE_LIMB_0,
            COL_VALUE_LIMB_1,
            COL_VALUE_LIMB_2,
            COL_VALUE_LIMB_3,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: stack_contents_layer_index,
        b_columns: vec![
            crate::stack_contents_air::COL_PC,
            crate::stack_contents_air::COL_VALUE_LIMB_0,
            crate::stack_contents_air::COL_VALUE_LIMB_1,
            crate::stack_contents_air::COL_VALUE_LIMB_2,
            crate::stack_contents_air::COL_VALUE_LIMB_3,
        ],
        b_selector_column: Some(crate::stack_contents_air::COL_IS_REAL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_vanish(trace: &TracePolynomials, cs: &Push0ConstraintSystem) {
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
    fn push0_air_simple_push0_vanishes() {
        // Single PUSH0 event post-Shanghai.
        let w = Push0Witness::from_events(&[(42, true)]);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].pc, 42);
        assert_eq!(w.rows[0].value, [0u64; 4]);
        assert_eq!(w.rows[0].gas_cost, PUSH0_GAS_COST);
        assert_eq!(w.rows[0].opcode, PUSH0_OPCODE);
        assert!(w.rows[0].is_post_shanghai);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Push0ConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn push0_air_pre_shanghai_rejected() {
        // is_post_shanghai = false on a real PUSH0 row should violate
        // constraint 2 (push0_post_shanghai_only).
        let w = Push0Witness::from_events(&[(7, false)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Push0ConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "push0_post_shanghai_only must fire on pre-Shanghai row"
        );
    }

    #[test]
    fn push0_air_tampered_value_detected() {
        // Honest setup, then corrupt value_limb_0 to a nonzero value.
        let curve = CurveType::Bls48581;
        let w = Push0Witness::from_events(&[(0, true)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_VALUE_LIMB_0].evaluations[0] = Scalar::from_u64(1, curve);
        let cs = Push0ConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 4 = value_limb_0_zero should fire.
        assert!(!bodies[4][0].is_zero(), "value_limb_0_zero must fire");
    }

    #[test]
    fn push0_air_tampered_gas_cost_detected() {
        let curve = CurveType::Bls48581;
        let w = Push0Witness::from_events(&[(0, true)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_GAS_COST].evaluations[0] = Scalar::from_u64(3, curve);
        let cs = Push0ConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 3 = gas_cost_eq_2 should fire.
        assert!(!bodies[3][0].is_zero(), "gas_cost_eq_2 must fire");
    }

    #[test]
    fn push0_air_descriptors_well_formed() {
        let d_hf = make_push0_to_hardfork_descriptor(0, 1);
        assert_eq!(d_hf.label, "push0_to_hardfork_shanghai_v1");
        assert_eq!(d_hf.a_layer_index, 0);
        assert_eq!(d_hf.b_layer_index, 1);
        assert_eq!(d_hf.a_columns, vec![COL_IS_POST_SHANGHAI]);
        assert_eq!(d_hf.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_hf.b_columns,
            vec![crate::hardfork_rules_air::COL_IS_POST_SHANGHAI]
        );
        assert_eq!(
            d_hf.b_selector_column,
            Some(crate::hardfork_rules_air::COL_IS_REAL)
        );

        let d_sc = make_push0_to_stack_contents_descriptor(2, 3);
        assert_eq!(d_sc.label, "push0_to_stack_contents_value_zero_v1");
        assert_eq!(d_sc.a_layer_index, 2);
        assert_eq!(d_sc.b_layer_index, 3);
        assert_eq!(d_sc.a_columns.len(), 5);
        assert_eq!(d_sc.b_columns.len(), 5);
        assert_eq!(d_sc.a_columns[0], COL_PC);
        assert_eq!(d_sc.a_columns[1], COL_VALUE_LIMB_0);
        assert_eq!(d_sc.a_columns[4], COL_VALUE_LIMB_3);
        assert_eq!(d_sc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_sc.b_columns,
            vec![
                crate::stack_contents_air::COL_PC,
                crate::stack_contents_air::COL_VALUE_LIMB_0,
                crate::stack_contents_air::COL_VALUE_LIMB_1,
                crate::stack_contents_air::COL_VALUE_LIMB_2,
                crate::stack_contents_air::COL_VALUE_LIMB_3,
            ]
        );
        assert_eq!(
            d_sc.b_selector_column,
            Some(crate::stack_contents_air::COL_IS_REAL)
        );
    }

    #[test]
    fn push0_air_column_layout_pinned() {
        assert_eq!(COL_PC, 0);
        assert_eq!(COL_VALUE_LIMB_0, 1);
        assert_eq!(COL_VALUE_LIMB_1, 2);
        assert_eq!(COL_VALUE_LIMB_2, 3);
        assert_eq!(COL_VALUE_LIMB_3, 4);
        assert_eq!(COL_GAS_COST, 5);
        assert_eq!(COL_IS_POST_SHANGHAI, 6);
        assert_eq!(COL_IS_REAL, 7);
        assert_eq!(COL_OPCODE, 8);
        assert_eq!(NUM_COLUMNS, 9);
        assert_eq!(NUM_ROW_CONSTRAINTS, 10);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(PUSH0_OPCODE, 0x5F);
        assert_eq!(PUSH0_GAS_COST, 2);
    }

    #[test]
    fn push0_air_multi_row_chain_honest() {
        // Three PUSH0 events at different PCs, all post-Shanghai.
        let w = Push0Witness::from_events(&[(0, true), (10, true), (42, true)]);
        assert_eq!(w.rows.len(), 3);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Push0ConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn push0_air_evaluate_at_point_zero_on_honest() {
        let w = Push0Witness::from_events(&[(11, true)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Push0ConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xdeadbeef, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "evaluate_at_point should be zero on honest row");
    }

    #[test]
    fn push0_air_tampered_opcode_detected() {
        let curve = CurveType::Bls48581;
        let w = Push0Witness::from_events(&[(0, true)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_OPCODE].evaluations[0] = Scalar::from_u64(0x60, curve); // PUSH1
        let cs = Push0ConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 8 = opcode_eq_push0 should fire.
        assert!(!bodies[8][0].is_zero(), "opcode_eq_push0 must fire");
    }

    #[test]
    fn push0_air_lookup_declarations_well_formed() {
        let cs = Push0ConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 2 binary + 1 byte = 3 declarations.
        assert_eq!(req.declarations.len(), 3);
        assert_eq!(req.tables.len(), 2);
        assert_eq!(req.tables[0].bits, 1);
        assert_eq!(req.tables[1].bits, 8);
    }
}
