//! Control flow constraints: JUMP/JUMPI/JUMPDEST/PC.

use metavm_zkp::field::Scalar;
use crate::trace::*;

/// Evaluate control flow constraints on the full trace domain.
///
/// - PC opcode: output_l0 = pc, upper limbs = 0
/// - JUMPDEST: no constraint (marker only)
/// - JUMP/JUMPI: deferred to lookup tables (need bytecode validation)
pub fn evaluate_control(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type = columns[COL_INSN_TYPE][i].to_u64() as u8;
        let funct = columns[COL_FUNCT][i].to_u64() as u8;

        let constraint = if insn_type == INSN_JUMP && funct == FUNCT_PC {
            // PC opcode: pushes current PC to stack
            // output_l0 = pc, upper limbs = 0
            let c0 = columns[COL_OUTPUT0_L0][i].sub(&columns[COL_PC][i]);
            let c1 = columns[COL_OUTPUT0_L1][i].clone();
            let c2 = columns[COL_OUTPUT0_L2][i].clone();
            let c3 = columns[COL_OUTPUT0_L3][i].clone();
            c0.add(&c1).add(&c2).add(&c3)
        } else {
            Scalar::zero(curve)
        };

        result.push(constraint);
    }

    result
}

/// Evaluate control flow constraint at a single point.
pub fn evaluate_control_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < NUM_EVM_COLUMNS {
        return Scalar::zero(curve);
    }

    let insn_type = col_evals[COL_INSN_TYPE].to_u64() as u8;
    let funct = col_evals[COL_FUNCT].to_u64() as u8;

    if insn_type == INSN_JUMP && funct == FUNCT_PC {
        let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_PC]);
        let c1 = col_evals[COL_OUTPUT0_L1].clone();
        let c2 = col_evals[COL_OUTPUT0_L2].clone();
        let c3 = col_evals[COL_OUTPUT0_L3].clone();
        c0.add(&c1).add(&c2).add(&c3)
    } else {
        Scalar::zero(curve)
    }
}

/// Evaluate control flow (PC opcode) constraint raw, without instruction type gating.
///
/// Computes output_l0 = pc, upper limbs = 0. The caller gates this by
/// multiplying with `sel_jump`.
pub fn evaluate_control_raw(col_evals: &[Scalar]) -> Scalar {
    let c0 = col_evals[COL_OUTPUT0_L0].sub(&col_evals[COL_PC]);
    let c1 = col_evals[COL_OUTPUT0_L1].clone();
    let c2 = col_evals[COL_OUTPUT0_L2].clone();
    let c3 = col_evals[COL_OUTPUT0_L3].clone();
    c0.add(&c1).add(&c2).add(&c3)
}
