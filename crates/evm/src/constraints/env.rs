//! ENV opcode algebraic constraint bodies.
//!
//! These replace oracle skips for ENV opcodes that have deterministic
//! outputs based on trace columns:
//!
//! - ADDRESS (0x30): output0 = frame_callee
//! - CALLER (0x33): output0 = frame_caller
//! - CALLVALUE (0x34): output0 = 0 (for non-payable, deferred for general case)

use crate::trace::*;
use metavm_zkp::field::Scalar;
use metavm_zkp::poly_arith;

pub fn evaluate_address_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_FRAME_CALLEE_L0]);
    let c1 = col_evals[COL_OUTPUT0_L1].sub(&col_evals[COL_FRAME_CALLEE_L1]);
    let c2 = col_evals[COL_OUTPUT0_L2].sub(&col_evals[COL_FRAME_CALLEE_L2]);
    let c3 = col_evals[COL_OUTPUT0_L3].sub(&col_evals[COL_FRAME_CALLEE_L3]);
    c0.add(&c1).add(&c2).add(&c3)
}

pub fn evaluate_caller_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_FRAME_CALLER_L0]);
    let c1 = col_evals[COL_OUTPUT0_L1].sub(&col_evals[COL_FRAME_CALLER_L1]);
    let c2 = col_evals[COL_OUTPUT0_L2].sub(&col_evals[COL_FRAME_CALLER_L2]);
    let c3 = col_evals[COL_OUTPUT0_L3].sub(&col_evals[COL_FRAME_CALLER_L3]);
    c0.add(&c1).add(&c2).add(&c3)
}

pub fn build_address_body(col_coeffs: &[Vec<Scalar>], curve: metavm_zkp::field::CurveType) -> Vec<Scalar> {
    let c0 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L0], &col_coeffs[COL_FRAME_CALLEE_L0], curve);
    let c1 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L1], &col_coeffs[COL_FRAME_CALLEE_L1], curve);
    let c2 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L2], &col_coeffs[COL_FRAME_CALLEE_L2], curve);
    let c3 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L3], &col_coeffs[COL_FRAME_CALLEE_L3], curve);
    poly_arith::poly_add(&poly_arith::poly_add(&c0, &c1, curve), &poly_arith::poly_add(&c2, &c3, curve), curve)
}

pub fn build_caller_body(col_coeffs: &[Vec<Scalar>], curve: metavm_zkp::field::CurveType) -> Vec<Scalar> {
    let c0 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L0], &col_coeffs[COL_FRAME_CALLER_L0], curve);
    let c1 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L1], &col_coeffs[COL_FRAME_CALLER_L1], curve);
    let c2 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L2], &col_coeffs[COL_FRAME_CALLER_L2], curve);
    let c3 = poly_arith::poly_sub(&col_coeffs[COL_OUTPUT0_L3], &col_coeffs[COL_FRAME_CALLER_L3], curve);
    poly_arith::poly_add(&poly_arith::poly_add(&c0, &c1, curve), &poly_arith::poly_add(&c2, &c3, curve), curve)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;
    use metavm_zkp::field::CurveType;

    fn trace_to_scalars(cols: &crate::trace::EvmTraceColumns, row: usize) -> Vec<Scalar> {
        use metavm_core::vm_traits::VmTrace;
        let n_cols = 1 + NUM_EVM_COLUMNS;
        let mut evals = vec![Scalar::zero(CurveType::Bls48581); n_cols];
        for c in 0..n_cols {
            let col_data = cols.column(c);
            if row < col_data.len() {
                evals[c] = Scalar::from_u64(col_data[row], CurveType::Bls48581);
            }
        }
        evals
    }

    #[test]
    fn address_constraint_zero_on_address_row() {
        // ADDRESS; STOP
        let bc = vec![0x30, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let n = cols.step.len();
        for r in 0..n {
            if cols.opcode[r] == 0x30 {
                let evals = trace_to_scalars(&cols, r);
                let body = evaluate_address_raw(&evals);
                assert!(body.is_zero(), "ADDRESS constraint nonzero at row {}", r);
            }
        }
    }

    #[test]
    fn env_selectors_populated() {
        // ADDRESS; ORIGIN; CALLER; CALLVALUE; CALLDATASIZE; CODESIZE; GASPRICE; STOP
        let bc = vec![0x30, 0x32, 0x33, 0x34, 0x36, 0x38, 0x3A, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let n = cols.step.len();
        let mut found = [false; 7];
        for r in 0..n {
            match cols.opcode[r] {
                0x30 => { assert_eq!(cols.sel_address[r], 1); found[0] = true; }
                0x32 => { assert_eq!(cols.sel_origin[r], 1); found[1] = true; }
                0x33 => { assert_eq!(cols.sel_caller[r], 1); found[2] = true; }
                0x34 => { assert_eq!(cols.sel_callvalue[r], 1); found[3] = true; }
                0x36 => { assert_eq!(cols.sel_calldatasize[r], 1); found[4] = true; }
                0x38 => { assert_eq!(cols.sel_codesize[r], 1); found[5] = true; }
                0x3A => { assert_eq!(cols.sel_gasprice[r], 1); found[6] = true; }
                _ => {}
            }
        }
        assert!(found.iter().all(|f| *f), "all 7 ENV selectors: {:?}", found);
    }

    #[test]
    fn caller_constraint_zero_on_caller_row() {
        // CALLER; STOP
        let bc = vec![0x33, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let n = cols.step.len();
        for r in 0..n {
            if cols.opcode[r] == 0x33 {
                let evals = trace_to_scalars(&cols, r);
                let body = evaluate_caller_raw(&evals);
                assert!(body.is_zero(), "CALLER constraint nonzero at row {}", r);
            }
        }
    }
}
