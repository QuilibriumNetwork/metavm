//! Memory operation constraints for load/store address computation.

use metavm_zkp::field::Scalar;
use super::selectors;
use super::{COL_RS1_VAL, COL_RS2_VAL, COL_MEM_ADDR, COL_MEM_VAL, COL_INSN_TYPE, COL_FUNCT, COL_IMMEDIATE, COL_AUX0};

/// Evaluate memory ops constraint at a single point given column evaluations.
pub fn evaluate_memory_ops_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < 16 {
        return Scalar::zero(curve);
    }

    let insn_type_val = col_evals[COL_INSN_TYPE].to_u64() as u8;

    match insn_type_val {
        selectors::INSN_LOAD => {
            let expected_addr = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
            col_evals[COL_MEM_ADDR].sub(&expected_addr)
        }
        selectors::INSN_STORE => {
            let expected_addr = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
            let addr_check = col_evals[COL_MEM_ADDR].sub(&expected_addr);
            let funct_val = col_evals[COL_FUNCT].to_u64() as u8;
            if funct_val == selectors::FUNCT_SD {
                let val_check = col_evals[COL_MEM_VAL].sub(&col_evals[COL_RS2_VAL]);
                addr_check.add(&val_check)
            } else if col_evals.len() >= 18 {
                // Partial stores with aux0: rs2_val = aux0 * mask + mem_val
                let mask = match funct_val {
                    selectors::FUNCT_SB => Scalar::from_u64(256, curve),
                    selectors::FUNCT_SH => Scalar::from_u64(65536, curve),
                    selectors::FUNCT_SW => {
                        // 2^32
                        Scalar::from_u64(1u64 << 32, curve)
                    }
                    _ => return addr_check,
                };
                let quot_times_mask = col_evals[COL_AUX0].mul(&mask);
                let expected = quot_times_mask.add(&col_evals[COL_MEM_VAL]);
                let val_check = col_evals[COL_RS2_VAL].sub(&expected);
                addr_check.add(&val_check)
            } else {
                addr_check
            }
        }
        _ => Scalar::zero(curve),
    }
}

/// Evaluate memory ops constraint without instruction type gating.
/// Returns mem_addr - (rs1_val + immediate).
/// This checks the load/store address computation.
pub fn evaluate_memory_ops_raw(col_evals: &[Scalar]) -> Scalar {
    let expected_addr = col_evals[COL_RS1_VAL].add(&col_evals[COL_IMMEDIATE]);
    col_evals[COL_MEM_ADDR].sub(&expected_addr)
}

/// Evaluate memory operation constraints.
///
/// - Load: mem_addr == rs1_val + immediate
/// - Store: mem_addr == rs1_val + immediate, mem_val == rs2_val (full width)
pub fn evaluate_memory_ops(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type_val = columns[COL_INSN_TYPE][i].to_u64() as u8;

        let constraint = match insn_type_val {
            selectors::INSN_LOAD => {
                // mem_addr == rs1_val + immediate
                let expected_addr = columns[COL_RS1_VAL][i].add(&columns[COL_IMMEDIATE][i]);
                columns[COL_MEM_ADDR][i].sub(&expected_addr)
            }
            selectors::INSN_STORE => {
                // mem_addr == rs1_val + immediate
                let expected_addr = columns[COL_RS1_VAL][i].add(&columns[COL_IMMEDIATE][i]);
                let addr_check = columns[COL_MEM_ADDR][i].sub(&expected_addr);

                let funct_val = columns[COL_FUNCT][i].to_u64() as u8;
                match funct_val {
                    selectors::FUNCT_SD => {
                        // Full 64-bit: mem_val == rs2_val
                        let val_check = columns[COL_MEM_VAL][i].sub(&columns[COL_RS2_VAL][i]);
                        addr_check.add(&val_check)
                    }
                    selectors::FUNCT_SB => {
                        // SB: rs2_val = aux0 * 256 + mem_val
                        let mask = Scalar::from_u64(256, curve);
                        let quot_times_mask = columns[COL_AUX0][i].mul(&mask);
                        let expected = quot_times_mask.add(&columns[COL_MEM_VAL][i]);
                        let val_check = columns[COL_RS2_VAL][i].sub(&expected);
                        addr_check.add(&val_check)
                    }
                    selectors::FUNCT_SH => {
                        // SH: rs2_val = aux0 * 65536 + mem_val
                        let mask = Scalar::from_u64(65536, curve);
                        let quot_times_mask = columns[COL_AUX0][i].mul(&mask);
                        let expected = quot_times_mask.add(&columns[COL_MEM_VAL][i]);
                        let val_check = columns[COL_RS2_VAL][i].sub(&expected);
                        addr_check.add(&val_check)
                    }
                    selectors::FUNCT_SW => {
                        // SW: rs2_val = aux0 * 2^32 + mem_val
                        let mask = Scalar::from_u64(1u64 << 32, curve);
                        let quot_times_mask = columns[COL_AUX0][i].mul(&mask);
                        let expected = quot_times_mask.add(&columns[COL_MEM_VAL][i]);
                        let val_check = columns[COL_RS2_VAL][i].sub(&expected);
                        addr_check.add(&val_check)
                    }
                    _ => addr_check,
                }
            }
            _ => Scalar::zero(curve),
        };

        result.push(constraint);
    }

    result
}
