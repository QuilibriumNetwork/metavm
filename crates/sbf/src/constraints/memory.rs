//! Memory constraints for SBF load/store operations.
//!
//! - Load: mem_addr = src_val + immediate, dst_val_after = mem_val
//! - Store: mem_addr = dst_val_before + immediate (for STX variants: mem_val = src_val)

use metavm_zkp::field::Scalar;
use crate::trace::*;

/// Evaluate memory constraints on the full domain.
pub fn evaluate_memory(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type = columns[COL_INSN_TYPE][i].to_u64() as u8;

        let constraint = match insn_type {
            INSN_LOAD => {
                // Address: mem_addr = src_val + immediate
                let expected_addr = columns[COL_SRC_VAL][i].add(&columns[COL_IMMEDIATE][i]);
                let addr_check = columns[COL_MEM_ADDR][i].sub(&expected_addr);

                // Value: dst_val_after = mem_val
                let val_check = columns[COL_DST_VAL_AFTER][i].sub(&columns[COL_MEM_VAL][i]);

                addr_check.add(&val_check)
            }
            INSN_STORE => {
                // For STX: mem_addr = dst_val_before + immediate
                // (dst is the base register for stores in BPF)
                let expected_addr = columns[COL_DST_VAL_BEFORE][i].add(&columns[COL_IMMEDIATE][i]);
                let addr_check = columns[COL_MEM_ADDR][i].sub(&expected_addr);

                let funct = columns[COL_FUNCT][i].to_u64() as u8;
                if funct >= FUNCT_STXB {
                    // STX variants: mem_val = src_val
                    let val_check = columns[COL_MEM_VAL][i].sub(&columns[COL_SRC_VAL][i]);
                    addr_check.add(&val_check)
                } else {
                    // ST variants: mem_val = immediate
                    let val_check = columns[COL_MEM_VAL][i].sub(&columns[COL_IMMEDIATE][i]);
                    addr_check.add(&val_check)
                }
            }
            _ => Scalar::zero(curve),
        };

        result.push(constraint);
    }

    result
}

/// Evaluate memory constraint at a single point (legacy, uses integer branching).
pub fn evaluate_memory_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < NUM_SBF_COLUMNS {
        return Scalar::zero(curve);
    }

    let insn_type = col_evals[COL_INSN_TYPE].to_u64() as u8;

    match insn_type {
        INSN_LOAD => {
            let expected_addr = col_evals[COL_SRC_VAL].add(&col_evals[COL_IMMEDIATE]);
            let addr_check = col_evals[COL_MEM_ADDR].sub(&expected_addr);
            let val_check = col_evals[COL_DST_VAL_AFTER].sub(&col_evals[COL_MEM_VAL]);
            addr_check.add(&val_check)
        }
        INSN_STORE => {
            let expected_addr = col_evals[COL_DST_VAL_BEFORE].add(&col_evals[COL_IMMEDIATE]);
            let addr_check = col_evals[COL_MEM_ADDR].sub(&expected_addr);
            let funct = col_evals[COL_FUNCT].to_u64() as u8;
            if funct >= FUNCT_STXB {
                let val_check = col_evals[COL_MEM_VAL].sub(&col_evals[COL_SRC_VAL]);
                addr_check.add(&val_check)
            } else {
                let val_check = col_evals[COL_MEM_VAL].sub(&col_evals[COL_IMMEDIATE]);
                addr_check.add(&val_check)
            }
        }
        _ => Scalar::zero(curve),
    }
}

/// Evaluate the load constraint body without insn_type gating.
///
/// Computes: addr_check + val_check where
///   addr_check = mem_addr - (src_val + immediate)
///   val_check  = dst_val_after - mem_val
/// The caller is responsible for multiplying by the appropriate selector
/// (sel_load + sel_store).
pub fn evaluate_memory_raw(col_evals: &[Scalar]) -> Scalar {
    // Load: addr = src_val + immediate, val = dst_val_after = mem_val
    let expected_addr = col_evals[COL_SRC_VAL].add(&col_evals[COL_IMMEDIATE]);
    let addr_check = col_evals[COL_MEM_ADDR].sub(&expected_addr);
    let val_check = col_evals[COL_DST_VAL_AFTER].sub(&col_evals[COL_MEM_VAL]);
    addr_check.add(&val_check)
}

/// Evaluate LOAD constraint body without insn_type gating.
///
/// Load: mem_addr = src_val + immediate, dst_val_after = mem_val
pub fn evaluate_memory_load_raw(col_evals: &[Scalar]) -> Scalar {
    let expected_addr = col_evals[COL_SRC_VAL].add(&col_evals[COL_IMMEDIATE]);
    let addr_check = col_evals[COL_MEM_ADDR].sub(&expected_addr);
    let val_check = col_evals[COL_DST_VAL_AFTER].sub(&col_evals[COL_MEM_VAL]);
    addr_check.add(&val_check)
}

/// Evaluate STORE constraint body without insn_type gating.
///
/// Store (STX): mem_addr = dst_val_before + immediate, mem_val = src_val
/// Note: For ST variants (immediate store), the trace must set src_val = immediate.
pub fn evaluate_memory_store_raw(col_evals: &[Scalar]) -> Scalar {
    let expected_addr = col_evals[COL_DST_VAL_BEFORE].add(&col_evals[COL_IMMEDIATE]);
    let addr_check = col_evals[COL_MEM_ADDR].sub(&expected_addr);
    let val_check = col_evals[COL_MEM_VAL].sub(&col_evals[COL_SRC_VAL]);
    addr_check.add(&val_check)
}
