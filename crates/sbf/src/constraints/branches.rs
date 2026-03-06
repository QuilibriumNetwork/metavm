//! Branch constraints for SBF conditional jumps.
//!
//! All conditional branches: either next_pc = pc + immediate (taken)
//! or next_pc = pc + 8 (not taken, BPF instructions are 8 bytes).
//! Unconditional JA: next_pc = pc + immediate.
//!
//! Condition verification uses aux1:
//! - JEQ/JNE: aux1 = 1 when dst != src, 0 when equal
//! - JGT/JGE/JLT/JLE (unsigned): aux1 = borrow = 1 when dst < src
//! - JSGT/JSGE/JSLT/JSLE (signed): aux1 = 1 when (dst as i64) < (src as i64)

use metavm_zkp::field::Scalar;
use crate::trace::*;

/// BPF instruction size in bytes.
const BPF_INSN_SIZE: u64 = 8;

/// Evaluate branch constraints on the full domain.
///
/// For each branch row:
/// 1. Target validity: (next_pc - taken) * (next_pc - not_taken) = 0
/// 2. Condition verification (per funct):
///    - JEQ: taken requires aux1=0 (equal), not-taken requires aux1=1, plus consistency
///    - JNE: taken requires aux1=1 (not equal), not-taken requires aux1=0
///    - JLT: taken requires aux1=1 (borrow), not-taken requires aux1=0
///    - JGE: taken requires aux1=0, not-taken requires aux1=1
///    - JGT: taken requires aux1=0 AND dst != src (partial)
///    - JLE: taken requires aux1=1 OR dst == src (partial)
pub fn evaluate_branches(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let insn_size = Scalar::from_u64(BPF_INSN_SIZE, curve);
    let one = Scalar::one(curve);
    let mut result = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let insn_type = columns[COL_INSN_TYPE][i].to_u64() as u8;
        let funct = columns[COL_FUNCT][i].to_u64() as u8;

        let constraint = if insn_type == INSN_BRANCH {
            if funct == FUNCT_JA {
                // Unconditional jump: next_pc = pc + immediate
                let expected = columns[COL_PC][i].add(&columns[COL_IMMEDIATE][i]);
                columns[COL_NEXT_PC][i].sub(&expected)
            } else {
                // Conditional branch
                let taken = columns[COL_PC][i].add(&columns[COL_IMMEDIATE][i]);
                let not_taken = columns[COL_PC][i].add(&insn_size);
                let diff_taken = columns[COL_NEXT_PC][i].sub(&taken);
                let diff_not_taken = columns[COL_NEXT_PC][i].sub(&not_taken);

                // Target validity: next_pc is either taken or not-taken
                let target_check = diff_taken.mul(&diff_not_taken);

                let aux1 = &columns[COL_AUX1][i];
                let one_minus_aux1 = one.sub(aux1);
                // aux1 binary: aux1 * (aux1 - 1) = 0
                let aux1_binary = aux1.mul(&aux1.sub(&one));

                let condition_check = match funct {
                    FUNCT_JEQ => {
                        // aux1 = 1 when dst != src, 0 when equal
                        let diff = columns[COL_DST_VAL_BEFORE][i].sub(&columns[COL_SRC_VAL][i]);
                        // taken -> aux1=0 (equal): diff_not_taken * aux1 = 0
                        let c1 = diff_not_taken.mul(aux1);
                        // not-taken -> aux1=1: diff_taken * (1-aux1) = 0
                        let c2 = diff_taken.mul(&one_minus_aux1);
                        // aux1=0 -> diff=0: (1-aux1)*diff = 0
                        let c3 = one_minus_aux1.mul(&diff);
                        c1.add(&c2).add(&c3).add(&aux1_binary)
                    }
                    FUNCT_JNE => {
                        // Same encoding, opposite logic
                        let diff = columns[COL_DST_VAL_BEFORE][i].sub(&columns[COL_SRC_VAL][i]);
                        // taken -> aux1=1 (not equal): diff_not_taken * (1-aux1) = 0
                        let c1 = diff_not_taken.mul(&one_minus_aux1);
                        // not-taken -> aux1=0: diff_taken * aux1 = 0
                        let c2 = diff_taken.mul(aux1);
                        // aux1=0 -> diff=0: (1-aux1)*diff = 0
                        let c3 = one_minus_aux1.mul(&diff);
                        c1.add(&c2).add(&c3).add(&aux1_binary)
                    }
                    FUNCT_JLT => {
                        // aux1 = borrow = 1 iff dst < src (unsigned)
                        // taken -> aux1=1: diff_not_taken * (1-aux1) = 0
                        let c1 = diff_not_taken.mul(&one_minus_aux1);
                        // not-taken -> aux1=0: diff_taken * aux1 = 0
                        let c2 = diff_taken.mul(aux1);
                        c1.add(&c2).add(&aux1_binary)
                    }
                    FUNCT_JGE => {
                        // taken iff dst >= src, i.e. aux1=0 (no borrow)
                        // taken -> aux1=0: diff_not_taken * aux1 = 0
                        let c1 = diff_not_taken.mul(aux1);
                        // not-taken -> aux1=1: diff_taken * (1-aux1) = 0
                        let c2 = diff_taken.mul(&one_minus_aux1);
                        c1.add(&c2).add(&aux1_binary)
                    }
                    FUNCT_JGT => {
                        // taken iff dst > src: aux1=0 (dst >= src) AND dst != src
                        // Partial: check aux1=0 when taken
                        // taken -> aux1=0: diff_not_taken * aux1 = 0
                        let c1 = diff_not_taken.mul(aux1);
                        c1.add(&aux1_binary)
                    }
                    FUNCT_JLE => {
                        // taken iff dst <= src: aux1=1 (dst < src) OR dst == src
                        // Partial: just check binary
                        aux1_binary
                    }
                    FUNCT_JSLT => {
                        // aux1 = 1 iff (dst as i64) < (src as i64)
                        // taken -> aux1=1: diff_not_taken * (1-aux1) = 0
                        let c1 = diff_not_taken.mul(&one_minus_aux1);
                        // not-taken -> aux1=0: diff_taken * aux1 = 0
                        let c2 = diff_taken.mul(aux1);
                        c1.add(&c2).add(&aux1_binary)
                    }
                    FUNCT_JSGE => {
                        // taken iff (dst as i64) >= (src as i64), i.e. aux1=0
                        // taken -> aux1=0: diff_not_taken * aux1 = 0
                        let c1 = diff_not_taken.mul(aux1);
                        // not-taken -> aux1=1: diff_taken * (1-aux1) = 0
                        let c2 = diff_taken.mul(&one_minus_aux1);
                        c1.add(&c2).add(&aux1_binary)
                    }
                    FUNCT_JSGT => {
                        // taken iff (dst as i64) > (src as i64): partial
                        let c1 = diff_not_taken.mul(aux1);
                        c1.add(&aux1_binary)
                    }
                    FUNCT_JSLE => {
                        // Partial: just check binary
                        aux1_binary
                    }
                    _ => {
                        // JSET and unknown: just check binary
                        aux1_binary
                    }
                };

                target_check.add(&condition_check)
            }
        } else {
            Scalar::zero(curve)
        };

        result.push(constraint);
    }

    result
}

/// Evaluate branch constraint at a single point (legacy, uses integer branching).
pub fn evaluate_branches_at_point(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    if col_evals.len() < NUM_SBF_COLUMNS {
        return Scalar::zero(curve);
    }

    let insn_type = col_evals[COL_INSN_TYPE].to_u64() as u8;
    let funct = col_evals[COL_FUNCT].to_u64() as u8;

    if insn_type != INSN_BRANCH {
        return Scalar::zero(curve);
    }

    let insn_size = Scalar::from_u64(BPF_INSN_SIZE, curve);

    if funct == FUNCT_JA {
        let expected = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
        col_evals[COL_NEXT_PC].sub(&expected)
    } else {
        let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
        let not_taken = col_evals[COL_PC].add(&insn_size);
        let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
        let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
        diff_taken.mul(&diff_not_taken)
    }
}

/// Evaluate the branch constraint body without insn_type gating.
///
/// Computes the target validity + condition verification constraints.
/// The target validity part:
///   (next_pc - pc - imm) * (next_pc - pc - 8) = 0
/// Plus condition verification using aux1 (binary check):
///   aux1 * (aux1 - 1) = 0
///
/// Used for sel_branch_other (JA, JGT, JLE, JSGT, JSLE, JSET) which only need
/// the basic target check + aux1 binary constraint.
pub fn evaluate_branches_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let insn_size = Scalar::from_u64(BPF_INSN_SIZE, curve);

    // Target validity: (next_pc - pc - imm) * (next_pc - pc - 8) = 0
    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&insn_size);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let target_check = diff_taken.mul(&diff_not_taken);

    // Condition verification: aux1 must be binary
    let aux1 = &col_evals[COL_AUX1];
    let aux1_binary = aux1.mul(&aux1.sub(&one));

    target_check.add(&aux1_binary)
}

/// JEQ condition raw: target_check + taken->aux1=0, not-taken->aux1=1, aux1=0->equal, aux1 binary.
///
/// Constraints:
///   target_check = (next_pc - taken) * (next_pc - not_taken) = 0
///   c1 = (next_pc - not_taken) * aux1 = 0         (taken -> aux1=0)
///   c2 = (next_pc - taken) * (1 - aux1) = 0       (not-taken -> aux1=1)
///   c3 = (1 - aux1) * (dst - src) = 0             (aux1=0 -> equal)
///   c4 = aux1 * (aux1 - 1) = 0                    (binary)
pub fn evaluate_branch_eq_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let insn_size = Scalar::from_u64(BPF_INSN_SIZE, curve);

    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&insn_size);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let diff = col_evals[COL_DST_VAL_BEFORE].sub(&col_evals[COL_SRC_VAL]);
    let aux1 = &col_evals[COL_AUX1];
    let one_m_aux1 = one.sub(aux1);

    let target = diff_taken.mul(&diff_not_taken);
    let c1 = diff_not_taken.mul(aux1);         // taken -> aux1=0
    let c2 = diff_taken.mul(&one_m_aux1);      // not-taken -> aux1=1
    let c3 = one_m_aux1.mul(&diff);            // aux1=0 -> equal
    let c4 = aux1.mul(&aux1.sub(&one));        // binary

    target.add(&c1).add(&c2).add(&c3).add(&c4)
}

/// JNE condition raw: target_check + taken->aux1=1, not-taken->aux1=0, aux1=0->equal, aux1 binary.
///
/// Constraints:
///   target_check = (next_pc - taken) * (next_pc - not_taken) = 0
///   c1 = (next_pc - not_taken) * (1 - aux1) = 0   (taken -> aux1=1)
///   c2 = (next_pc - taken) * aux1 = 0              (not-taken -> aux1=0)
///   c3 = (1 - aux1) * (dst - src) = 0              (aux1=0 -> equal)
///   c4 = aux1 * (aux1 - 1) = 0                     (binary)
pub fn evaluate_branch_neq_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let insn_size = Scalar::from_u64(BPF_INSN_SIZE, curve);

    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&insn_size);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let diff = col_evals[COL_DST_VAL_BEFORE].sub(&col_evals[COL_SRC_VAL]);
    let aux1 = &col_evals[COL_AUX1];
    let one_m_aux1 = one.sub(aux1);

    let target = diff_taken.mul(&diff_not_taken);
    let c1 = diff_not_taken.mul(&one_m_aux1);  // taken -> aux1=1
    let c2 = diff_taken.mul(aux1);             // not-taken -> aux1=0
    let c3 = one_m_aux1.mul(&diff);            // aux1=0 -> equal
    let c4 = aux1.mul(&aux1.sub(&one));        // binary

    target.add(&c1).add(&c2).add(&c3).add(&c4)
}

/// JLT/JSLT condition raw: target_check + taken->aux1=1, not-taken->aux1=0, aux1 binary.
///
/// aux1 encodes the comparison result: 1 if dst < src (unsigned for JLT, signed for JSLT).
/// Constraints:
///   target_check = (next_pc - taken) * (next_pc - not_taken) = 0
///   c1 = (next_pc - not_taken) * (1 - aux1) = 0   (taken -> aux1=1)
///   c2 = (next_pc - taken) * aux1 = 0              (not-taken -> aux1=0)
///   c3 = aux1 * (aux1 - 1) = 0                     (binary)
pub fn evaluate_branch_lt_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let insn_size = Scalar::from_u64(BPF_INSN_SIZE, curve);

    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&insn_size);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let aux1 = &col_evals[COL_AUX1];
    let one_m_aux1 = one.sub(aux1);

    let target = diff_taken.mul(&diff_not_taken);
    let c1 = diff_not_taken.mul(&one_m_aux1);  // taken -> aux1=1
    let c2 = diff_taken.mul(aux1);             // not-taken -> aux1=0
    let c3 = aux1.mul(&aux1.sub(&one));        // binary

    target.add(&c1).add(&c2).add(&c3)
}

/// JGE/JSGE condition raw: target_check + taken->aux1=0, not-taken->aux1=1, aux1 binary.
///
/// aux1 encodes the comparison result: 1 if dst < src (unsigned for JGE, signed for JSGE).
/// Taken when aux1=0 (dst >= src).
/// Constraints:
///   target_check = (next_pc - taken) * (next_pc - not_taken) = 0
///   c1 = (next_pc - not_taken) * aux1 = 0          (taken -> aux1=0)
///   c2 = (next_pc - taken) * (1 - aux1) = 0        (not-taken -> aux1=1)
///   c3 = aux1 * (aux1 - 1) = 0                     (binary)
pub fn evaluate_branch_ge_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let one = Scalar::one(curve);
    let insn_size = Scalar::from_u64(BPF_INSN_SIZE, curve);

    let taken = col_evals[COL_PC].add(&col_evals[COL_IMMEDIATE]);
    let not_taken = col_evals[COL_PC].add(&insn_size);
    let diff_taken = col_evals[COL_NEXT_PC].sub(&taken);
    let diff_not_taken = col_evals[COL_NEXT_PC].sub(&not_taken);
    let aux1 = &col_evals[COL_AUX1];
    let one_m_aux1 = one.sub(aux1);

    let target = diff_taken.mul(&diff_not_taken);
    let c1 = diff_not_taken.mul(aux1);         // taken -> aux1=0
    let c2 = diff_taken.mul(&one_m_aux1);      // not-taken -> aux1=1
    let c3 = aux1.mul(&aux1.sub(&one));        // binary

    target.add(&c1).add(&c2).add(&c3)
}
