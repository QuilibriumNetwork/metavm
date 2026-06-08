//! ENV opcode constraint AIR (standalone gadget).
//!
//! Proves algebraically that on a row marked as ADDRESS or CALLER,
//! the claimed output equals the claimed frame callee/caller. Composed
//! with the EVM main trace via cross-AIR LogUp:
//!
//!   EVM main: (sel_address, output0, frame_callee) →
//!   ENV AIR: (is_address, claimed_output, claimed_frame_callee) where
//!             constraint enforces claimed_output == claimed_frame_callee.
//!
//! For ADDRESS: output0 = frame_callee
//! For CALLER:  output0 = frame_caller

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// Per row: kind (0=ADDRESS, 1=CALLER, etc.) + 4 output limbs + 4 expected limbs + is_real
pub const COL_KIND: usize = 0;
pub const COL_OUTPUT_L0: usize = 1;
pub const COL_OUTPUT_L1: usize = 2;
pub const COL_OUTPUT_L2: usize = 3;
pub const COL_OUTPUT_L3: usize = 4;
pub const COL_EXPECTED_L0: usize = 5;
pub const COL_EXPECTED_L1: usize = 6;
pub const COL_EXPECTED_L2: usize = 7;
pub const COL_EXPECTED_L3: usize = 8;
pub const COL_IS_REAL: usize = 9;
pub const NUM_COLUMNS: usize = 10;

// Constraints: is_real binary + 4 limb equalities (gated by is_real)
pub const NUM_ROW_CONSTRAINTS: usize = 5;
pub const NUM_SHIFTED: usize = 0;

pub const KIND_ADDRESS: u64 = 0;
pub const KIND_CALLER: u64 = 1;
pub const KIND_CALLVALUE: u64 = 2;
pub const KIND_ORIGIN: u64 = 3;
pub const KIND_GASPRICE: u64 = 4;
pub const KIND_CALLDATASIZE: u64 = 5;
pub const KIND_CODESIZE: u64 = 6;
pub const KIND_RETURNDATASIZE: u64 = 7;
pub const KIND_PC: u64 = 8;
pub const KIND_GAS: u64 = 9;
pub const KIND_MSIZE: u64 = 10;

#[derive(Clone, Debug)]
pub struct EnvRow {
    pub kind: u64,
    pub output: [u64; 4],
    pub expected: [u64; 4],
}

#[derive(Clone, Debug, Default)]
pub struct EnvWitness {
    pub rows: Vec<EnvRow>,
}

impl EnvWitness {
    pub fn from_rows(rows: Vec<EnvRow>) -> Self { Self { rows } }
}

pub fn build_trace_polynomials(w: &EnvWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_KIND][r] = Scalar::from_u64(row.kind, curve);
        for j in 0..4 {
            cols[COL_OUTPUT_L0 + j][r] = Scalar::from_u64(row.output[j], curve);
            cols[COL_EXPECTED_L0 + j][r] = Scalar::from_u64(row.expected[j], curve);
        }
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols.into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

pub struct EnvConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl EnvConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega); self.domain_size = Some(domain_size); self
    }
}

impl VmConstraintSystem for EnvConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "output_eq_expected_l0".into(),
            "output_eq_expected_l1".into(),
            "output_eq_expected_l2".into(),
            "output_eq_expected_l3".into(),
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
        let mut c4 = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            c0[r] = v.mul(&v.sub(&one));
            c1[r] = v.mul(&columns[COL_OUTPUT_L0][r].sub(&columns[COL_EXPECTED_L0][r]));
            c2[r] = v.mul(&columns[COL_OUTPUT_L1][r].sub(&columns[COL_EXPECTED_L1][r]));
            c3[r] = v.mul(&columns[COL_OUTPUT_L2][r].sub(&columns[COL_EXPECTED_L2][r]));
            c4[r] = v.mul(&columns[COL_OUTPUT_L3][r].sub(&columns[COL_EXPECTED_L3][r]));
        }
        vec![c0, c1, c2, c3, c4]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &ce[COL_IS_REAL];
        let c0 = v.mul(&v.sub(&one));
        let c1 = v.mul(&ce[COL_OUTPUT_L0].sub(&ce[COL_EXPECTED_L0]));
        let c2 = v.mul(&ce[COL_OUTPUT_L1].sub(&ce[COL_EXPECTED_L1]));
        let c3 = v.mul(&ce[COL_OUTPUT_L2].sub(&ce[COL_EXPECTED_L2]));
        let c4 = v.mul(&ce[COL_OUTPUT_L3].sub(&ce[COL_EXPECTED_L3]));
        let bodies = [c0, c1, c2, c3, c4];
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
        let v_m1 = poly_sub(v, &one_p, curve);
        let c0 = poly_mul(v, &v_m1, curve);
        let d0 = poly_sub(&cc[COL_OUTPUT_L0], &cc[COL_EXPECTED_L0], curve);
        let d1 = poly_sub(&cc[COL_OUTPUT_L1], &cc[COL_EXPECTED_L1], curve);
        let d2 = poly_sub(&cc[COL_OUTPUT_L2], &cc[COL_EXPECTED_L2], curve);
        let d3 = poly_sub(&cc[COL_OUTPUT_L3], &cc[COL_EXPECTED_L3], curve);
        let c1 = poly_mul(v, &d0, curve);
        let c2 = poly_mul(v, &d1, curve);
        let c3 = poly_mul(v, &d2, curve);
        let c4 = poly_mul(v, &d3, curve);
        let bodies = [c0, c1, c2, c3, c4];
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

// ─── Witness extraction from EVM trace ────────────────────────────────

/// Extract ENV opcode rows from an EVM trace and build an EnvWitness.
/// Each row of the witness corresponds to one ENV opcode invocation,
/// with `expected` populated from the appropriate trace column.
pub fn from_evm_trace(cols: &crate::trace::EvmTraceColumns) -> EnvWitness {
    
    let mut rows = Vec::new();
    let n = cols.step.len();
    for r in 0..n {
        let opcode = cols.opcode[r] as u8;
        let output = [cols.output0[0][r], cols.output0[1][r], cols.output0[2][r], cols.output0[3][r]];
        match opcode {
            0x30 => rows.push(EnvRow { // ADDRESS
                kind: KIND_ADDRESS,
                output,
                expected: [cols.frame_callee[0][r], cols.frame_callee[1][r], cols.frame_callee[2][r], cols.frame_callee[3][r]],
            }),
            0x32 => rows.push(EnvRow { // ORIGIN
                kind: KIND_ORIGIN,
                output,
                expected: [cols.tx_origin[0][r], cols.tx_origin[1][r], cols.tx_origin[2][r], cols.tx_origin[3][r]],
            }),
            0x33 => rows.push(EnvRow { // CALLER
                kind: KIND_CALLER,
                output,
                expected: [cols.frame_caller[0][r], cols.frame_caller[1][r], cols.frame_caller[2][r], cols.frame_caller[3][r]],
            }),
            0x34 => rows.push(EnvRow { // CALLVALUE
                kind: KIND_CALLVALUE,
                output,
                expected: [cols.frame_value[0][r], cols.frame_value[1][r], cols.frame_value[2][r], cols.frame_value[3][r]],
            }),
            0x36 => rows.push(EnvRow { // CALLDATASIZE
                kind: KIND_CALLDATASIZE,
                output,
                expected: [cols.tx_calldata_size[r], 0, 0, 0],
            }),
            0x38 => rows.push(EnvRow { // CODESIZE
                kind: KIND_CODESIZE,
                output,
                expected: [cols.tx_code_size[r], 0, 0, 0],
            }),
            0x3A => rows.push(EnvRow { // GASPRICE
                kind: KIND_GASPRICE,
                output,
                expected: [cols.tx_gas_price[r], 0, 0, 0],
            }),
            0x3D => rows.push(EnvRow { // RETURNDATASIZE
                kind: KIND_RETURNDATASIZE,
                output,
                expected: [cols.returndata_size[r], 0, 0, 0],
            }),
            0x58 => rows.push(EnvRow { // PC
                kind: KIND_PC,
                output,
                expected: [cols.pc[r], 0, 0, 0],
            }),
            0x5A => rows.push(EnvRow { // GAS
                kind: KIND_GAS,
                output,
                expected: [cols.gas_remaining[r], 0, 0, 0],
            }),
            0x59 => rows.push(EnvRow { // MSIZE - approximated as output = output (need watermark column)
                kind: KIND_MSIZE,
                output,
                expected: output,
            }),
            _ => {}
        }
    }
    EnvWitness { rows }
}

// ─── Cross-AIR LogUp descriptor builders ─────────────────────────────

/// Generic descriptor: EVM main row gated by `evm_selector_col` →
/// ENV AIR row enforces output == expected. Tuple: 8 cols (4 output limbs
/// + 4 context limbs from EVM trace).
pub fn make_env_descriptor(
    label: &str,
    evm_layer_index: usize,
    env_air_layer_index: usize,
    evm_selector_col: usize,
    evm_context_cols: [usize; 4],
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![
            COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
            evm_context_cols[0], evm_context_cols[1], evm_context_cols[2], evm_context_cols[3],
        ],
        a_selector_column: Some(evm_selector_col),
        b_layer_index: env_air_layer_index,
        b_columns: vec![
            COL_OUTPUT_L0, COL_OUTPUT_L1, COL_OUTPUT_L2, COL_OUTPUT_L3,
            COL_EXPECTED_L0, COL_EXPECTED_L1, COL_EXPECTED_L2, COL_EXPECTED_L3,
        ],
        b_selector_column: Some(COL_IS_REAL),
    }
}

pub fn make_address_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_FRAME_CALLEE_L0, COL_FRAME_CALLEE_L1, COL_FRAME_CALLEE_L2, COL_FRAME_CALLEE_L3, COL_SEL_ADDRESS};
    make_env_descriptor("evm_address_to_env_air_v1", evm_layer, env_layer, COL_SEL_ADDRESS,
        [COL_FRAME_CALLEE_L0, COL_FRAME_CALLEE_L1, COL_FRAME_CALLEE_L2, COL_FRAME_CALLEE_L3])
}

pub fn make_caller_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_FRAME_CALLER_L0, COL_FRAME_CALLER_L1, COL_FRAME_CALLER_L2, COL_FRAME_CALLER_L3, COL_SEL_CALLER};
    make_env_descriptor("evm_caller_to_env_air_v1", evm_layer, env_layer, COL_SEL_CALLER,
        [COL_FRAME_CALLER_L0, COL_FRAME_CALLER_L1, COL_FRAME_CALLER_L2, COL_FRAME_CALLER_L3])
}

pub fn make_callvalue_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_FRAME_VALUE_L0, COL_FRAME_VALUE_L1, COL_FRAME_VALUE_L2, COL_FRAME_VALUE_L3, COL_SEL_CALLVALUE};
    make_env_descriptor("evm_callvalue_to_env_air_v1", evm_layer, env_layer, COL_SEL_CALLVALUE,
        [COL_FRAME_VALUE_L0, COL_FRAME_VALUE_L1, COL_FRAME_VALUE_L2, COL_FRAME_VALUE_L3])
}

pub fn make_origin_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_TX_ORIGIN_L0, COL_TX_ORIGIN_L1, COL_TX_ORIGIN_L2, COL_TX_ORIGIN_L3, COL_SEL_ORIGIN};
    make_env_descriptor("evm_origin_to_env_air_v1", evm_layer, env_layer, COL_SEL_ORIGIN,
        [COL_TX_ORIGIN_L0, COL_TX_ORIGIN_L1, COL_TX_ORIGIN_L2, COL_TX_ORIGIN_L3])
}

/// Single-limb context descriptors (only limb 0 used; other limbs must be 0 on EVM side).
pub fn make_single_limb_descriptor(
    label: &str,
    evm_layer: usize,
    env_layer: usize,
    evm_selector_col: usize,
    evm_context_col: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::COL_OUTPUT0_L0;
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: evm_layer,
        a_columns: vec![COL_OUTPUT0_L0, evm_context_col],
        a_selector_column: Some(evm_selector_col),
        b_layer_index: env_layer,
        b_columns: vec![COL_OUTPUT_L0, COL_EXPECTED_L0],
        b_selector_column: Some(COL_IS_REAL),
    }
}

pub fn make_gasprice_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_TX_GAS_PRICE, COL_SEL_GASPRICE};
    make_single_limb_descriptor("evm_gasprice_to_env_air_v1", evm_layer, env_layer, COL_SEL_GASPRICE, COL_TX_GAS_PRICE)
}

pub fn make_calldatasize_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_TX_CALLDATA_SIZE, COL_SEL_CALLDATASIZE};
    make_single_limb_descriptor("evm_calldatasize_to_env_air_v1", evm_layer, env_layer, COL_SEL_CALLDATASIZE, COL_TX_CALLDATA_SIZE)
}

pub fn make_codesize_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_TX_CODE_SIZE, COL_SEL_CODESIZE};
    make_single_limb_descriptor("evm_codesize_to_env_air_v1", evm_layer, env_layer, COL_SEL_CODESIZE, COL_TX_CODE_SIZE)
}

pub fn make_returndatasize_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_RETURNDATA_SIZE, COL_SEL_ENV};
    make_single_limb_descriptor("evm_returndatasize_to_env_air_v1", evm_layer, env_layer, COL_SEL_ENV, COL_RETURNDATA_SIZE)
}

pub fn make_pc_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_PC, COL_SEL_PC};
    make_single_limb_descriptor("evm_pc_to_env_air_v1", evm_layer, env_layer, COL_SEL_PC, COL_PC)
}

pub fn make_gas_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_GAS_REMAINING, COL_SEL_GAS};
    make_single_limb_descriptor("evm_gas_to_env_air_v1", evm_layer, env_layer, COL_SEL_GAS, COL_GAS_REMAINING)
}

pub fn make_msize_descriptor(evm_layer: usize, env_layer: usize) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    // MSIZE binds to mem_offset (the current memory size tracker). Approximation —
    // a proper MSIZE AIR would track the max-offset-touched watermark.
    use crate::trace::{COL_SEL_MSIZE_OP};
    // For now use a single-limb descriptor binding output to itself (tautology gated by sel).
    // Real algebraic msize binding requires a memory-watermark column (future work).
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_msize_to_env_air_v1".into(),
        a_layer_index: evm_layer,
        a_columns: vec![crate::trace::COL_OUTPUT0_L0, crate::trace::COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_MSIZE_OP),
        b_layer_index: env_layer,
        b_columns: vec![COL_OUTPUT_L0, COL_EXPECTED_L0],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_witness_constraints_zero() {
        let rows = vec![
            EnvRow { kind: KIND_ADDRESS, output: [1, 2, 3, 4], expected: [1, 2, 3, 4] },
            EnvRow { kind: KIND_CALLER, output: [10, 20, 30, 40], expected: [10, 20, 30, 40] },
        ];
        let w = EnvWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = EnvConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "c{} r{} nonzero", i, r);
            }
        }
    }

    #[test]
    fn tampered_output_detected() {
        let rows = vec![
            EnvRow { kind: KIND_ADDRESS, output: [1, 2, 3, 4], expected: [99, 2, 3, 4] },
        ];
        let w = EnvWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = EnvConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // limb 0 constraint should fire (output 1 != expected 99)
        assert!(!bodies[1][0].is_zero());
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let rows = vec![EnvRow { kind: KIND_ADDRESS, output: [7, 0, 0, 0], expected: [7, 0, 0, 0] }];
        let w = EnvWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = EnvConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0x1234, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero());
    }

    #[test]
    fn address_descriptor_well_formed() {
        let desc = make_address_descriptor(0, 1);
        assert_eq!(desc.label, "evm_address_to_env_air_v1");
        assert_eq!(desc.a_columns.len(), 8);
        assert_eq!(desc.b_columns.len(), 8);
    }

    #[test]
    fn caller_descriptor_well_formed() {
        let desc = make_caller_descriptor(0, 1);
        assert_eq!(desc.label, "evm_caller_to_env_air_v1");
        assert_eq!(desc.a_columns.len(), 8);
    }

    #[test]
    fn callvalue_descriptor_well_formed() {
        let desc = make_callvalue_descriptor(0, 1);
        assert_eq!(desc.label, "evm_callvalue_to_env_air_v1");
        assert_eq!(desc.a_columns.len(), 8);
    }

    #[test]
    fn all_env_descriptors_well_formed() {
        let addr = make_address_descriptor(0, 1);
        let caller = make_caller_descriptor(0, 1);
        let callvalue = make_callvalue_descriptor(0, 1);
        let origin = make_origin_descriptor(0, 1);
        assert_eq!(addr.a_columns.len(), 8);
        assert_eq!(caller.a_columns.len(), 8);
        assert_eq!(callvalue.a_columns.len(), 8);
        assert_eq!(origin.a_columns.len(), 8);

        let gp = make_gasprice_descriptor(0, 1);
        let cds = make_calldatasize_descriptor(0, 1);
        let cs = make_codesize_descriptor(0, 1);
        let rds = make_returndatasize_descriptor(0, 1);
        assert_eq!(gp.a_columns.len(), 2);
        assert_eq!(cds.a_columns.len(), 2);
        assert_eq!(cs.a_columns.len(), 2);
        assert_eq!(rds.a_columns.len(), 2);

        // Verify all 8 ENV opcodes have descriptors
        let descriptors = [&addr, &caller, &callvalue, &origin, &gp, &cds, &cs, &rds];
        for d in descriptors {
            assert!(d.a_selector_column.is_some());
            assert!(d.b_selector_column.is_some());
        }
    }

    #[test]
    fn from_evm_trace_extracts_address() {
        use crate::executor::execute_bytecode;
        // ADDRESS; STOP
        let bc = vec![0x30, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].kind, KIND_ADDRESS);
        // The expected value should equal the output (executor uses [0x42; 20] as contract)
        assert_eq!(w.rows[0].expected, w.rows[0].output);
    }

    #[test]
    fn from_evm_trace_extracts_caller() {
        use crate::executor::execute_bytecode;
        let bc = vec![0x33, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].kind, KIND_CALLER);
        assert_eq!(w.rows[0].expected, w.rows[0].output);
    }

    #[test]
    fn from_evm_trace_extracts_pc_gas_msize() {
        use crate::executor::execute_bytecode;
        // PC; GAS; MSIZE; STOP
        let bc = vec![0x58, 0x5A, 0x59, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        // PC and MSIZE should be extracted (GAS = 0x5A is now classified)
        assert!(w.rows.iter().any(|r| r.kind == KIND_PC));
        assert!(w.rows.iter().any(|r| r.kind == KIND_GAS));
        assert!(w.rows.iter().any(|r| r.kind == KIND_MSIZE));
    }

    #[test]
    fn pc_gas_msize_selectors_fire() {
        use crate::executor::execute_bytecode;
        let bc = vec![0x58, 0x5A, 0x59, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let mut found = [false; 3];
        for r in 0..cols.step.len() {
            match cols.opcode[r] {
                0x58 => { assert_eq!(cols.sel_pc[r], 1); found[0] = true; }
                0x5A => { assert_eq!(cols.sel_gas[r], 1); found[1] = true; }
                0x59 => { assert_eq!(cols.sel_msize_op[r], 1); found[2] = true; }
                _ => {}
            }
        }
        assert!(found.iter().all(|f| *f), "missing selectors: {:?}", found);
    }

    #[test]
    fn pc_gas_msize_descriptors() {
        let pc = make_pc_descriptor(0, 1);
        let gas = make_gas_descriptor(0, 1);
        let msize = make_msize_descriptor(0, 1);
        assert_eq!(pc.label, "evm_pc_to_env_air_v1");
        assert_eq!(gas.label, "evm_gas_to_env_air_v1");
        assert_eq!(msize.label, "evm_msize_to_env_air_v1");
    }

    #[test]
    fn from_evm_trace_multi_opcode() {
        use crate::executor::execute_bytecode;
        // ADDRESS; CALLER; CALLVALUE; STOP
        let bc = vec![0x30, 0x33, 0x34, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        assert_eq!(w.rows.len(), 3);
        // All outputs match expected (honest trace)
        for row in &w.rows {
            assert_eq!(row.expected, row.output);
        }
    }

    #[test]
    fn extracted_witness_constraints_zero() {
        use crate::executor::execute_bytecode;
        let bc = vec![0x30, 0x33, 0x34, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let w = from_evm_trace(&cols);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = EnvConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for body in cs.evaluate_on_domain(&cr, t.num_rows).iter() {
            for v in body.iter() {
                assert!(v.is_zero());
            }
        }
    }

    #[test]
    fn multi_kind_honest_witness() {
        // Mix ADDRESS, CALLER, CALLVALUE rows.
        let rows = vec![
            EnvRow { kind: KIND_ADDRESS, output: [1, 2, 3, 4], expected: [1, 2, 3, 4] },
            EnvRow { kind: KIND_CALLER, output: [10, 20, 30, 40], expected: [10, 20, 30, 40] },
            EnvRow { kind: KIND_CALLVALUE, output: [100, 0, 0, 0], expected: [100, 0, 0, 0] },
        ];
        let w = EnvWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = EnvConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for body in cs.evaluate_on_domain(&cr, t.num_rows).iter() {
            for v in body.iter() {
                assert!(v.is_zero());
            }
        }
    }
}
